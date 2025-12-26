"""
Embedding model abstractions and concrete implementations for the embedding server.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Generator, List, Optional, Sequence
import librosa

import numpy as np
import torch
import torchaudio
from pymilvus import MilvusClient, DataType
from gpu_model_coordinator import GPU_COORDINATOR


def _milvus_uses_lite() -> bool:
    """
    Detect whether we're running Milvus Lite (local file URI).

    Lite mode only supports FLAT/IVF_FLAT/AUTOINDEX indexes.
    """
    uri = os.getenv("NAVIDROME_MILVUS_URI", "")
    return "://" not in uri or uri.startswith("file:")


# Use try-except to handle import from different contexts
try:
    from models import TrackSegment
except ImportError:
    # If 'models' is shadowed by another package, try importing from local file
    from pathlib import Path
    import importlib.util

    _models_path = Path(__file__).parent / "models.py"
    _spec = importlib.util.spec_from_file_location("_ps_models", _models_path)
    _ps_models = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_ps_models)
    TrackSegment = _ps_models.TrackSegment

from muq import MuQ


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
        self._last_used = datetime.now(timezone.utc)
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
                idle = datetime.now(timezone.utc) - self._last_used
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

    def ensure_model_loaded(self) -> Any:
        """
        Load the model into memory if it is not already available.
        """
        with self._lock:
            if self._model is None:
                self.logger.info("Loading embedding model %s", self.__class__.__name__)
                self._model = self._load_model()
            self._last_used = datetime.now(timezone.utc)
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
                self._last_used = datetime.now(timezone.utc)

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

    def offload_to_cpu(self) -> None:
        """
        Move the model to CPU to free VRAM while keeping weights in memory.
        """
        with self._lock:
            if self._model is None:
                return
            try:
                self._model = self._model.to("cpu")  # type: ignore[call-arg]
            except Exception:
                self.logger.exception(
                    "Failed to offload %s to CPU", self.__class__.__name__
                )
        self._empty_cuda_cache()

    def _empty_cuda_cache(self) -> None:
        try:
            torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch.cuda may be unavailable
            pass

    @abstractmethod
    def _load_model(self) -> Any:
        """
        Subclasses must implement model construction/loading.
        """

    def _release_model(self, model: torch.nn.Module) -> None:
        """
        Hook allowing subclasses to customize teardown logic.
        """
        del model

    def prepare_music(self, music_file: str, music_name: str) -> List[TrackSegment]:
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
    def embed_music(self, music_file: str, music_name: str) -> dict:
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
        model_id: str = "OpenMuQ/MuQ-large-msd-iter",
        device: str = "cuda",
        model_dtype: torch.dtype = torch.float16,
        storage_dtype: torch.dtype = torch.float32,
        sample_rate: int = 24_000,
        window_seconds: int = 120,
        hop_seconds: int = 15,
        chunk_batch_size: Optional[int] = None,
        timeout_seconds: int = 360,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, logger=logger)
        self.model_id = model_id
        self.device = device
        self.model_dtype = model_dtype
        self.storage_dtype = storage_dtype
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        env_batch = os.getenv("NAVIDROME_MUQ_CHUNK_BATCH", "1").strip()
        if chunk_batch_size is None:
            try:
                chunk_batch_size = int(env_batch) if env_batch else 1
            except ValueError:
                chunk_batch_size = 1
        self.chunk_batch_size = max(int(chunk_batch_size), 1)
        self._gpu_owner = f"{self.__class__.__name__}"
        GPU_COORDINATOR.register(self._gpu_owner, self.offload_to_cpu)

    def _inference_dtype(self) -> torch.dtype:
        if not str(self.device).startswith("cuda"):
            return torch.float32
        return self.model_dtype

    def _iter_audio_chunks(
        self, audio: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        chunk_size = int(self.window_seconds * self.sample_rate)
        hop_size = int(self.hop_seconds * self.sample_rate)
        if chunk_size <= 0 or hop_size <= 0:
            raise ValueError("window_seconds and hop_seconds must be positive")
        total_samples = int(audio.shape[0])
        if total_samples <= 0:
            return
        last_start = max(total_samples - chunk_size, 0)
        for start_sample in range(0, last_start + 1, hop_size):
            end_sample = min(start_sample + chunk_size, total_samples)
            chunk = audio[start_sample:end_sample]
            observed = int(chunk.shape[0])
            if observed == 0:
                continue
            if observed < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - observed))
            yield chunk.astype("float32", copy=False)
        if last_start % hop_size != 0:
            start_sample = last_start
            end_sample = min(start_sample + chunk_size, total_samples)
            chunk = audio[start_sample:end_sample]
            observed = int(chunk.shape[0])
            if observed == 0:
                return
            if observed < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - observed))
            yield chunk.astype("float32", copy=False)

    def _run_chunk_batch(
        self,
        *,
        model: torch.nn.Module,
        chunk_batch: List[np.ndarray],
    ) -> torch.Tensor:
        chunk_matrix = np.stack(chunk_batch, axis=0)
        chunk_tensor = torch.from_numpy(chunk_matrix).to(
            self.device, dtype=self._inference_dtype()
        )
        model_output: Optional[torch.Tensor] = None
        outputs: Optional[torch.Tensor] = None
        outputs_cpu: Optional[torch.Tensor] = None
        try:
            with torch.inference_mode():
                model_output = model(chunk_tensor)
                if hasattr(model_output, "last_hidden_state"):
                    outputs = model_output.last_hidden_state
                else:
                    outputs = model_output
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
                elif outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)
                outputs_cpu = outputs.detach().to("cpu", dtype=torch.float32)
        finally:
            if self.device.startswith("cuda") and torch.cuda.is_available():
                del chunk_tensor
                if model_output is not None:
                    del model_output
                if outputs is not None:
                    del outputs
                torch.cuda.empty_cache()
        if outputs_cpu is None:
            raise RuntimeError("Model produced no outputs for audio batch")
        return outputs_cpu

    def _load_model(self) -> torch.nn.Module:
        GPU_COORDINATOR.claim(self._gpu_owner, self.logger)
        # Use MuQ class for audio-only embeddings (not MuQMuLan which is for music-text joint embeddings)
        model = MuQ.from_pretrained(self.model_id)
        model = model.to(self.device, dtype=self._inference_dtype()).eval()
        return model

    def ensure_model_loaded(self) -> Any:  # type: ignore[override]
        """
        Override to ensure model is moved back to GPU if it was offloaded.
        """
        with self._lock:
            if self._model is None:
                self.logger.info("Loading embedding model %s", self.__class__.__name__)
                self._model = self._load_model()
            else:
                GPU_COORDINATOR.claim(self._gpu_owner, self.logger)
                try:
                    self._model = self._model.to(  # type: ignore[attr-defined]
                        self.device, dtype=self._inference_dtype()
                    )
                except Exception:
                    self.logger.exception(
                        "Failed to move %s back to GPU", self.__class__.__name__
                    )
                    self._model = self._load_model()
            self._last_used = datetime.now(timezone.utc)
            return self._model

    def embed_music(self, music_file: str, music_name: str) -> dict:
        segments = self.prepare_music(music_file, music_name)
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
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "window_seconds": self.window_seconds,
            "hop_seconds": self.hop_seconds,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "segments": [segment.__dict__ for segment in payload_segments],
        }

    def embed_audio_tensor(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        apply_enrichment: bool = True,
    ) -> torch.Tensor:
        """
        Embed audio from an in-memory waveform tensor.

        Args:
            waveform: Audio tensor of shape [channels, samples] or [samples]
            sample_rate: Sample rate of the waveform
            apply_enrichment: If True, apply enrichment to get [3*D] output.
                            If False, return raw [D, T] embeddings.

        Returns:
            If apply_enrichment=True: [3*D] enriched embedding tensor
            If apply_enrichment=False: [D, T] raw embedding tensor
        """
        # Ensure mono audio
        if waveform.dim() > 1 and waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0) if waveform.dim() > 1 else waveform

        # Resample if necessary
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )

        # Convert to numpy for chunking
        audio = waveform.contiguous().cpu().numpy()

        outputs_cpu: List[torch.Tensor] = []

        with self.model_session() as model:
            batch: List[np.ndarray] = []
            for chunk in self._iter_audio_chunks(audio):
                batch.append(chunk)
                if len(batch) >= self.chunk_batch_size:
                    outputs_cpu.append(
                        self._run_chunk_batch(model=model, chunk_batch=batch)
                    )
                    batch = []
            if batch:
                outputs_cpu.append(
                    self._run_chunk_batch(model=model, chunk_batch=batch)
                )

        if not outputs_cpu:
            raise RuntimeError("No audio chunks produced")

        outputs = torch.cat(outputs_cpu, dim=0)

        # outputs.T is shape [D, T] where T is number of chunks
        raw_embedding = outputs.T

        if apply_enrichment:
            result = enrich_embedding(raw_embedding)
        else:
            result = raw_embedding

        return result

    def embed_audio_tensor_batch(
        self,
        waveforms: List[torch.Tensor],
        sample_rates: List[int],
        apply_enrichment: bool = True,
    ) -> List[torch.Tensor]:
        """
        Embed multiple audio waveforms in batch.

        Args:
            waveforms: List of audio tensors, each of shape [channels, samples] or [samples]
            sample_rates: List of sample rates corresponding to each waveform
            apply_enrichment: If True, apply enrichment to get [3*D] output.

        Returns:
            List of embedding tensors, each either [3*D] or [D, T] depending on apply_enrichment
        """
        num_tracks = len(waveforms)
        track_embeddings: List[List[torch.Tensor]] = [[] for _ in range(num_tracks)]

        with self.model_session() as model:
            batch_chunks: List[np.ndarray] = []
            batch_track_indices: List[int] = []

            for track_idx, (waveform, sample_rate) in enumerate(
                zip(waveforms, sample_rates)
            ):
                # Ensure mono audio
                if waveform.dim() > 1 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0)
                else:
                    waveform = waveform.squeeze(0) if waveform.dim() > 1 else waveform

                # Resample if necessary
                if sample_rate != self.sample_rate:
                    waveform = torchaudio.functional.resample(
                        waveform, sample_rate, self.sample_rate
                    )

                # Convert to numpy for chunking
                audio = waveform.contiguous().cpu().numpy()

                for chunk in self._iter_audio_chunks(audio):
                    batch_chunks.append(chunk)
                    batch_track_indices.append(track_idx)
                    if len(batch_chunks) >= self.chunk_batch_size:
                        outputs = self._run_chunk_batch(
                            model=model, chunk_batch=batch_chunks
                        )
                        for output, mapped_track in zip(outputs, batch_track_indices):
                            track_embeddings[mapped_track].append(output)
                        batch_chunks = []
                        batch_track_indices = []

            if batch_chunks:
                outputs = self._run_chunk_batch(model=model, chunk_batch=batch_chunks)
                for output, mapped_track in zip(outputs, batch_track_indices):
                    track_embeddings[mapped_track].append(output)

        # Process each track's chunks
        results = []
        for track_chunks in track_embeddings:
            if not track_chunks:
                # Empty track - shouldn't happen but handle gracefully
                results.append(
                    torch.zeros(1536 if apply_enrichment else 512, device=self.device)
                )
                continue

            # Stack chunks for this track: [num_chunks, D]
            track_matrix = torch.stack(track_chunks, dim=0)
            # Transpose to [D, num_chunks] for enrichment
            raw_embedding = track_matrix.T

            if apply_enrichment:
                results.append(enrich_embedding(raw_embedding))
            else:
                results.append(raw_embedding)

        return results

    def embed_string(self, value: str) -> Sequence[float]:
        raise NotImplementedError(
            "MuQ text embeddings are disabled; use the description pipeline instead."
        )

    def ensure_milvus_schemas(self, client: MilvusClient) -> None:
        existing = client.list_collections()
        if "embedding" in existing:
            return

        schema = MilvusClient.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
        )
        schema.add_field("name", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=1536)
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
        if _milvus_uses_lite():
            index_params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024},
            )
        else:
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

        total_samples = audio.shape[0]
        if total_samples == 0:
            self.logger.warning(
                "No audio chunks produced for segment %s",
                track_segment.index,
            )
            return None

        outputs_cpu: List[torch.Tensor] = []
        batch: List[np.ndarray] = []
        for chunk in self._iter_audio_chunks(audio):
            batch.append(chunk)
            if len(batch) >= self.chunk_batch_size:
                outputs_cpu.append(
                    self._run_chunk_batch(model=model, chunk_batch=batch)
                )
                batch = []
        if batch:
            outputs_cpu.append(self._run_chunk_batch(model=model, chunk_batch=batch))

        if not outputs_cpu:
            self.logger.warning(
                "No audio chunks produced for segment %s",
                track_segment.index,
            )
            return None

        outputs = torch.cat(outputs_cpu, dim=0)
        enriched = enrich_embedding(outputs.T)
        enriched_cpu = enriched.to("cpu", dtype=self.storage_dtype)

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
            embedding=enriched_cpu.tolist(),
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


def enrich_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """
    embedding: [D, T] float tensor
    returns: [3*D] L2-normalized vector
      [mean, robust_sigma_iqr, dmean]
    """
    # Robustly handle different tensor shapes
    if embedding.dim() == 1:
        # Single vector [D] -> treat as [D, 1]
        embedding = embedding.unsqueeze(1)
    elif embedding.dim() == 3:
        # [batch, seq, D] -> mean over seq -> [batch, D] -> transpose to [D, batch]
        embedding = embedding.mean(dim=1).T
    elif embedding.dim() != 2:
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}, expected 1D, 2D, or 3D"
        )

    D, T = embedding.shape
    # Use the device of the input tensor instead of hardcoding CUDA
    device = embedding.device
    x = embedding.to(device).to(torch.float32)

    # per-dimension mean and std over time
    mean = x.mean(dim=-1)  # [D]

    # robust spread via IQR
    q25 = torch.quantile(x, 0.25, dim=-1)  # [D]
    q75 = torch.quantile(x, 0.75, dim=-1)  # [D]
    robust = (q75 - q25) / _IQR_TO_SIGMA  # [D]

    # first differences over time (descreit d/dx)
    if T >= 2:
        dx = x[:, 1:] - x[:, :-1]  # [D, T-1]
        dmean = dx.mean(dim=-1)
    else:
        dmean = torch.zeros(D, device=x.device, dtype=x.dtype)

    vec = torch.cat([mean, robust, dmean], dim=0)  # [3*D]

    return F.normalize(vec, p=2, dim=0).to("cpu")


__all__ = [
    "BaseEmbeddingModel",
    "MuQEmbeddingModel",
    "SegmentEmbedding",
    "enrich_embedding",
]
