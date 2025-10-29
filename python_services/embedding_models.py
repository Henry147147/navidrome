"""
Embedding model abstractions and concrete implementations for the embedding server.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Generator, List, Optional, Sequence
import librosa

import numpy as np
import torch
import torchaudio
from pymilvus import MilvusClient, DataType

from models import TrackSegment

from muq import MuQMuLan
from music2latent import EncoderDecoder

LOGGER = logging.getLogger("navidrome.embedding_models")


class BaseEmbeddingModel(ABC):
    """
    Abstract base class for embedding models providing lifecycle and Milvus helpers.
    """

    def __init__(
        self,
        *,
        timeout_seconds: int = 360,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._timeout = max(int(timeout_seconds), 30)
        self._lock = Lock()
        self._model: Optional[torch.nn.Module] = None
        self._last_used = datetime.now(UTC)
        self._stop_event = Event()
        self._unloader = Thread(
            target=self._auto_unload_loop,
            name=f"{self.__class__.__name__}UnloadThread",
            daemon=True,
        )
        self._unloader.start()

    def _auto_unload_loop(self) -> None:
        """
        Background loop that releases the model after a period of inactivity.
        """
        check_interval = 5
        while not self._stop_event.wait(check_interval):
            with self._lock:
                if self._model is None:
                    continue
                idle = datetime.now(UTC) - self._last_used
                if idle >= timedelta(seconds=self._timeout):
                    self.logger.info(
                        "Model idle for %s seconds, unloading %s",
                        int(idle.total_seconds()),
                        self.__class__.__name__,
                    )
                    try:
                        self._release_model(self._model)
                    finally:
                        self._model = None
                    self._empty_cuda_cache()

    def shutdown(self) -> None:
        """
        Stop the background thread and release any loaded model.
        """
        self._stop_event.set()
        self._unloader.join(timeout=1)
        with self._lock:
            if self._model is not None:
                self._release_model(self._model)
                self._model = None
        self._empty_cuda_cache()

    def ensure_model_loaded(self) -> torch.nn.Module:
        """
        Load the model into memory if it is not already available.
        """
        with self._lock:
            if self._model is None:
                self.logger.info("Loading embedding model %s", self.__class__.__name__)
                self._model = self._load_model()
            self._last_used = datetime.now(UTC)
            return self._model

    @contextmanager
    def model_session(self) -> Generator[torch.nn.Module, Any, Any]:
        """
        Context manager that ensures the model is loaded and updates usage tracking.
        """
        model = self.ensure_model_loaded()
        try:
            yield model
        finally:
            with self._lock:
                self._last_used = datetime.now(UTC)

    def unload_model(self) -> None:
        """
        Manually unload the model and release GPU/CPU resources.
        """
        with self._lock:
            if self._model is None:
                return
            self.logger.info("Manually unloading model %s", self.__class__.__name__)
            try:
                self._release_model(self._model)
            finally:
                self._model = None
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch.cuda may be unavailable
            pass

    @abstractmethod
    def _load_model(self) -> torch.nn.Module:
        """
        Subclasses must implement model construction/loading.
        """

    def _release_model(self, model: torch.nn.Module) -> None:
        """
        Hook allowing subclasses to customize teardown logic.
        """
        del model

    def prepare_music(
        self,
        music_file: str,
        music_name: str,
        cue_file: Optional[str] = None,
    ) -> List[TrackSegment]:
        """
        Prepare track segments that should be embedded for the provided music file.

        Cue-based segmentation is handled upstream by the uploader; models only
        embed the provided audio file as a single track.
        """
        segment_title = music_name.strip() if music_name else ""
        if not segment_title:
            segment_title = Path(music_file).stem
        return [
            TrackSegment(index=1, title=segment_title, start=0.0, end=None),
        ]

    @abstractmethod
    def embed_music(
        self,
        music_file: str,
        music_name: str,
        cue_file: Optional[str] = None,
    ) -> dict:
        """
        Generate embeddings for the provided music file.
        """

    @abstractmethod
    def embed_string(self, value: str) -> Sequence[float]:
        """
        Generate an embedding for a free-form string input.
        """

    @abstractmethod
    def ensure_milvus_schemas(self, client: MilvusClient) -> None:
        """
        Create Milvus schemas required by this model if they do not already exist.
        """

    @abstractmethod
    def ensure_milvus_index(self, client: MilvusClient) -> None:
        """
        Ensure Milvus indexes are created for the collections used by this model.
        """


@dataclass
class SegmentEmbedding:
    """
    Lightweight container describing the results for a single segment.
    """

    index: int
    title: str
    offset_seconds: float
    duration_seconds: Optional[float]
    embedding: Sequence[float]


class MuQEmbeddingModel(BaseEmbeddingModel):
    """
    Embedding model that wraps the MuQ MuLan model for audio embeddings.
    """

    def __init__(
        self,
        *,
        model_id: str = "OpenMuQ/MuQ-MuLan-large",
        device: str = "cuda",
        storage_dtype: torch.dtype = torch.float32,
        sample_rate: int = 24_000,
        window_seconds: int = 120,
        hop_seconds: int = 15,
        timeout_seconds: int = 360,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, logger=logger)
        self.model_id = model_id
        self.device = device
        self.storage_dtype = storage_dtype
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds

    def _load_model(self) -> torch.nn.Module:
        model = MuQMuLan.from_pretrained(self.model_id)
        model = model.to(self.device).to(self.storage_dtype).eval()
        return model

    def embed_music(
        self,
        music_file: str,
        music_name: str,
        cue_file: Optional[str] = None,
    ) -> dict:
        segments = self.prepare_music(music_file, music_name, cue_file)
        if not segments:
            raise RuntimeError("No segments available for embedding.")

        with self.model_session() as model:
            payload_segments: List[SegmentEmbedding] = []
            for segment in segments:
                prepared = self._embed_single_segment(
                    model=model,
                    music_file=music_file,
                    track_segment=segment,
                )
                if prepared is not None:
                    payload_segments.append(prepared)

        if not payload_segments:
            raise RuntimeError("Unable to generate embeddings for any segments.")

        return {
            "music_file": str(Path(music_file)),
            "cue_file": cue_file if cue_file else None,
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "generated_at": datetime.now(UTC).isoformat(),
            "segments": [segment.__dict__ for segment in payload_segments],
        }

    def embed_string(self, value: str) -> Sequence[float]:
        with self.model_session() as model:
            text_embeds = model(texts=value)
        return text_embeds

    def ensure_milvus_schemas(self, client: MilvusClient) -> None:
        existing = client.list_collections()
        if "embedding" in existing:
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("name", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=512)
        schema.add_field("offset", DataType.FLOAT)
        schema.add_field("model_id", DataType.VARCHAR, max_length=256)
        client.create_collection("embedding", schema=schema)

    def ensure_milvus_index(self, client: MilvusClient) -> None:
        indexes = client.describe_collection("embedding").get("indexes", [])
        index_fields = {index.get("field_name") for index in indexes}
        if "embedding" in index_fields and "name" in index_fields:
            return

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="name", index_type="INVERTED")
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 50, "efConstruction": 250},
        )
        client.create_index("embedding", index_params)

    def _embed_single_segment(
        self,
        *,
        model: torch.nn.Module,
        music_file: str,
        track_segment: TrackSegment,
    ) -> Optional[SegmentEmbedding]:
        offset = float(track_segment.start)
        duration = track_segment.duration
        audio = self._load_audio_segment(
            music_file=music_file,
            offset=offset,
            duration=duration,
        )
        if audio.size == 0:
            self.logger.warning(
                "Segment %s produced empty audio buffer; skipping",
                track_segment.index,
            )
            return None

        chunk_size = int(self.window_seconds * self.sample_rate)
        hop_size = int(self.hop_seconds * self.sample_rate)
        total_samples = audio.shape[0]
        starts = list(range(0, max(total_samples - chunk_size, 0) + 1, hop_size))
        if not starts or (starts[-1] + chunk_size < total_samples):
            starts.append(max(total_samples - chunk_size, 0))

        chunk_arrays = []
        for start_sample in starts:
            end_sample = min(start_sample + chunk_size, total_samples)
            chunk = audio[start_sample:end_sample]
            observed = int(chunk.shape[0])
            if observed == 0:
                continue
            if observed < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - observed))
            chunk_arrays.append(chunk.astype("float32", copy=False))

        if not chunk_arrays:
            self.logger.warning(
                "No audio chunks produced for segment %s",
                track_segment.index,
            )
            return None

        chunk_matrix = np.stack(chunk_arrays, axis=0)
        chunk_tensor = (
            torch.from_numpy(chunk_matrix).to(self.device).to(self.storage_dtype)
        )

        with torch.inference_mode():
            outputs = model(wavs=chunk_tensor)

        outputs = torch.nn.functional.normalize(outputs, dim=1)
        centroid = torch.nn.functional.normalize(outputs.mean(dim=0), dim=0)

        centroid_cpu = centroid.detach().to("cpu", dtype=self.storage_dtype)

        computed_duration: Optional[float]
        if duration is not None:
            computed_duration = float(duration)
        else:
            computed_duration = total_samples / self.sample_rate

        return SegmentEmbedding(
            index=track_segment.index,
            title=track_segment.title,
            offset_seconds=offset,
            duration_seconds=computed_duration,
            embedding=centroid_cpu.tolist(),
        )

    def _load_audio_segment(
        self,
        *,
        music_file: str,
        offset: float,
        duration: Optional[float],
    ) -> np.ndarray:
        self.logger.debug(
            "Loading audio segment from %s (offset=%s, duration=%s)",
            music_file,
            offset,
            duration,
        )
        info_fn = getattr(torchaudio, "info", None)
        if info_fn is None:
            return self._load_audio_with_librosa(
                music_file=music_file,
                offset=offset,
                duration=duration,
            )

        try:
            info = info_fn(str(music_file))
            source_sr = info.sample_rate
        except Exception:
            return self._load_audio_with_librosa(
                music_file=music_file,
                offset=offset,
                duration=duration,
            )

        frame_offset = max(int(offset * source_sr), 0)
        if duration is not None:
            num_frames = max(int(duration * source_sr), 0)
        else:
            num_frames = -1

        try:
            waveform, sr = torchaudio.load(
                str(music_file),
                frame_offset=frame_offset,
                num_frames=num_frames,
            )
        except Exception:
            return self._load_audio_with_librosa(
                music_file=music_file,
                offset=offset,
                duration=duration,
            )

        if waveform.numel() == 0:
            return np.zeros(0, dtype=np.float32)

        waveform = waveform.to(self.storage_dtype)
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        return waveform.contiguous().cpu().numpy()

    def _load_audio_with_librosa(
        self,
        *,
        music_file: str,
        offset: float,
        duration: Optional[float],
    ) -> np.ndarray:
        import librosa  # local import to avoid heavy dependency at module import

        safe_offset = max(float(offset), 0.0)
        safe_duration = None if duration is None else max(float(duration), 0.0)
        audio, _ = librosa.load(
            music_file,
            sr=self.sample_rate,
            mono=True,
            offset=safe_offset,
            duration=safe_duration,
        )
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        return audio.astype(np.float32, copy=False)


class MusicLatentSpaceModel(BaseEmbeddingModel):
    """
    Embedding model that wraps the Music2Latent EncoderDecoder for audio embeddings.
    Produces 64-dimensional latent space embeddings by averaging over time.
    """

    def __init__(
        self,
        *,
        device: str = "cuda",
        sample_rate: int = 44_100,
        max_waveform_length: int = 44100 * 10,
        timeout_seconds: int = 360,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, logger=logger)
        self.device = device
        self.sample_rate = sample_rate
        self.max_waveform_length = max_waveform_length
        self.patched = False

    def _load_model(self) -> EncoderDecoder:  # Type: Ignore
        if not self.patched:
            old = torch.load

            def patched(*args, **kwargs):
                if "weights_only" in kwargs:
                    del kwargs["weights_only"]
                return old(*args, **kwargs, weights_only=False)

            torch.load = patched
            self.patched = True

        model = EncoderDecoder()
        return model

    def embed_music(
        self,
        music_file: str,
        music_name: str,
        cue_file: Optional[str] = None,
    ) -> dict:
        segments = self.prepare_music(music_file, music_name, cue_file)
        if not segments:
            raise RuntimeError("No segments available for embedding.")

        with self.model_session() as model:
            payload_segments: List[SegmentEmbedding] = []
            for segment in segments:
                prepared = self._embed_single_segment(
                    model=model,
                    music_file=music_file,
                    track_segment=segment,
                )
                return prepared
                if prepared is not None:
                    payload_segments.append(prepared)

        if not payload_segments:
            raise RuntimeError("Unable to generate embeddings for any segments.")

        return {
            "music_file": str(Path(music_file)),
            "cue_file": cue_file if cue_file else None,
            "model_id": "Music2Latent_EncoderDecoder",
            "sample_rate": self.sample_rate,
            "generated_at": datetime.now(UTC).isoformat(),
            "segments": [segment.__dict__ for segment in payload_segments],
        }

    def embed_string(self, value: str) -> Sequence[float]:
        # Music2Latent does not support text embedding natively.
        # Return zero vector as placeholder
        self.logger.warning("embed_string not supported by MusicLatentSpaceModel")
        return [0.0] * 64

    def ensure_milvus_schemas(self, client: MilvusClient) -> None:
        existing = client.list_collections()
        if "latent_embedding" in existing:
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("name", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=64)
        schema.add_field("offset", DataType.FLOAT)
        schema.add_field("model_id", DataType.VARCHAR, max_length=256)
        client.create_collection("latent_embedding", schema=schema)

    def ensure_milvus_index(self, client: MilvusClient) -> None:
        indexes = client.describe_collection("latent_embedding").get("indexes", [])
        index_fields = {index.get("field_name") for index in indexes}
        if "embedding" in index_fields and "name" in index_fields:
            return

        index_params = MilvusClient.prepare_index_params()
        index_params.add_index(field_name="name", index_type="INVERTED")
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 50, "efConstruction": 250},
        )
        client.create_index("latent_embedding", index_params)

    def _embed_single_segment(
        self,
        *,
        model: Any,
        music_file: str,
        track_segment: TrackSegment,
    ) -> Optional[SegmentEmbedding]:
        offset = float(track_segment.start)
        duration = track_segment.duration
        audio = self._load_audio_segment(
            music_file=music_file,
            offset=offset,
            duration=duration,
        )
        if audio.size == 0:
            self.logger.warning(
                "Segment %s produced empty audio buffer; skipping",
                track_segment.index,
            )
            return None

        latent = model.encode(music_file, max_waveform_length=self.max_waveform_length)
        latent_with_mag = add_magnitude_channels(latent)
        return latent_with_mag
        
        enriched = flatten_and_enrich_embedding(latent)

        computed_duration: Optional[float]
        if duration is not None:
            computed_duration = float(duration)
        else:
            computed_duration = len(audio) / self.sample_rate

        return SegmentEmbedding(
            index=track_segment.index,
            title=track_segment.title,
            offset_seconds=offset,
            duration_seconds=computed_duration,
            embedding=enriched,
        )

    def _load_audio_segment(
        self,
        *,
        music_file: str,
        offset: float,
        duration: Optional[float],
    ) -> np.ndarray:
        safe_offset = max(float(offset), 0.0)
        safe_duration = None if duration is None else max(float(duration), 0.0)
        audio, _ = librosa.load(
            music_file,
            sr=self.sample_rate,
            mono=True,
            offset=safe_offset,
            duration=safe_duration,
        )
        if audio.size == 0:
            return np.zeros(0, dtype=np.float32)
        return audio.astype(np.float32, copy=False)


_IQR_TO_SIGMA = 1.3489795003921634  # sigma ≈ IQR / 1.34898
import torch.nn.functional as F

def to_R128(x):  # x: [2,64,T] -> [128,T]
    return x.permute(1,0,2).reshape(128, x.size(-1))

def add_magnitude_channels(x):   # x: [2,64,T]
    """The latent space is contstructed of [2, 64, T], 
    where x[0] represents the real and x[1] represents the imaginary, 
    need to have it in 2D, but there are consequences of concatting the im with real"""
    re, im = x[0], x[1]          # [64,T]
    mag = torch.sqrt(re**2 + im**2 + 1e-12)  # [64,T]
    # Return either just [R,I] (128) or concat [R,I,|z|] (192)
    return torch.cat([to_R128(x), mag], dim=0)  # [192,T]


def _block_normalize(v: torch.Tensor, num_blocks: int = 6) -> torch.Tensor:
    """L2 normalize each contiguous block of the 1-D vector."""
    D6 = v.numel()
    if D6 == 0:
        return v
    assert D6 % num_blocks == 0, "vector length must be divisible by num_blocks"
    block_dim = D6 // num_blocks
    blocks = []
    for i in range(num_blocks):
        block = v[i * block_dim : (i + 1) * block_dim]
        blocks.append(F.normalize(block, p=2, dim=0))
    return torch.cat(blocks, dim=0)


def _signed_power(v: torch.Tensor, gamma: float = 0.5) -> torch.Tensor:
    """Apply signed power transform that preserves sign."""
    if v.numel() == 0:
        return v
    return v.sign() * v.abs().clamp_min(1e-12).pow(gamma)


def postprocess_enriched(v: torch.Tensor) -> torch.Tensor:
    """Apply block normalization, signed power, then global L2 normalization."""
    #v = _block_normalize(v, num_blocks=6)
    #v = _signed_power(v, gamma=0.5)
    return F.normalize(v, p=2, dim=0)

def flatten_and_enrich_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """
    embedding: [D, T] float tensor
    returns: [6*D] L2-normalized vector
      [mean, std, robust_sigma_iqr, dmean, dstd, d_robust_sigma_iqr]
    """
    assert embedding.dim() == 2, "Expected [D, T]"
    D, T = embedding.shape
    x = embedding.to("cuda").to(torch.float32)

    # per-dimension mean and std over time
    mean = x.mean(dim=-1)                                  # [D]
    #std = x.var(dim=-1, unbiased=False).clamp_min(1e-12).sqrt()

    # robust spread via IQR
    q25 = torch.quantile(x, 0.25, dim=-1)                  # [D]
    q75 = torch.quantile(x, 0.75, dim=-1)                  # [D]
    robust = (q75 - q25) / _IQR_TO_SIGMA                   # [D]

    # first differences over time
    if T >= 2:
        dx = x[:, 1:] - x[:, :-1]                          # [D, T-1]
        dmean = dx.mean(dim=-1)
        #dstd  = dx.var(dim=-1, unbiased=False).clamp_min(1e-12).sqrt()
        #dq25 = torch.quantile(dx, 0.25, dim=-1)
        #dq75 = torch.quantile(dx, 0.75, dim=-1)
        #drobust = (dq75 - dq25) / _IQR_TO_SIGMA
    else:
        # keep length constant when T < 2
        dmean = torch.zeros(D, device=x.device, dtype=x.dtype)
        #dstd  = torch.zeros(D, device=x.device, dtype=x.dtype)
        #drobust = torch.zeros(D, device=x.device, dtype=x.dtype)

    #vec = torch.cat([mean, std, robust, dmean, dstd, drobust], dim=0)  # [6*D]
    vec = torch.cat([mean, robust, dmean], dim=0)  # [3*D]

    return postprocess_enriched(vec).to("cpu")


__all__ = [
    "BaseEmbeddingModel",
    "MuQEmbeddingModel",
    "MusicLatentSpaceModel",
    "SegmentEmbedding",
]

import pandas as pd

def list_audio():
    files = sorted([*Path(".").glob("*.flac"), *Path(".").glob("*.mp3")])
    return [str(p) for p in files]

def enriched_vec(e):
    v = flatten_and_enrich_embedding(e)
    return F.normalize(v.view(-1), p=2, dim=0)  # [K]

def centroid_vec(e):
    c = e.mean(dim=1)                           # [D]
    return F.normalize(c, p=2, dim=0)

def cosine_table(vectors, names=None, outfile="cosine_similarity.png", annotate=True, dpi=400):
    """
    Build an NxN cosine similarity heatmap from 1-D torch tensors and save to disk.
    Returns the output filepath.
    """
    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # Normalize and keep only valid vectors
    vs, kept_idx = [], []
    for i, v in enumerate(vectors):
        if v is None:
            continue
        v = v.view(-1)
        if v.numel() == 0 or not torch.isfinite(v).all():
            continue
        vs.append(torch.nn.functional.normalize(v, p=2, dim=0))
        kept_idx.append(i)

    N = len(vs)
    if N == 0:
        raise ValueError("No valid vectors to plot.")

    X = torch.stack(vs, dim=0)                 # [N, d]
    S = (X @ X.t()).detach().cpu().numpy()     # [N, N]
    S = np.clip(S, -1.0, 1.0)

    # Align labels to N
    if names is None:
        labels = [f"item_{i}" for i in kept_idx]
    else:
        labels = [names[i] for i in kept_idx if i < len(names)]
        if len(labels) < N:
            labels += [f"item_{i}" for i in range(len(labels), N)]
        labels = labels[:N]

    # Plot
    fig_size = min(0.5 * N + 2, 20)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), dpi=dpi)
    im = ax.imshow(S, vmin=-1, vmax=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("cosine similarity", rotation=270, labelpad=12)

    ax.set_xticks(np.arange(N), labels=labels, rotation=90)
    ax.set_yticks(np.arange(N), labels=labels)
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # minor grid for cell borders
    ax.set_xticks(np.arange(-0.5, N, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.5)
    ax.set_title("Cosine Similarity")

    # optional numeric annotations for small N
    if annotate and N <= 25:
        for i in range(N):
            for j in range(N):
                ax.text(j, i, f"{S[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main():
    paths = list_audio()
    if len(paths) < 2:
        print("need at least two audio files in current directory matching *.flac or *.mp3")
        return

    m = MusicLatentSpaceModel()

    embeds = []
    from tqdm import tqdm
    with torch.no_grad():
        for p in tqdm(list(paths)):
            try:
                e = m.embed_music(p, p)             
                embeds.append(e)
            except:
                print(p, "Failed to parse")

    enriched = [enriched_vec(e) for e in embeds]
    centroids = [centroid_vec(e) for e in embeds]

    names = [Path(p).name for p in paths]
    df_enriched = cosine_table(enriched, names, outfile="enriched.png")
    df_centroid = cosine_table(centroids, names, outfile="centroid.png")

if __name__ == "__main__":
    main()
