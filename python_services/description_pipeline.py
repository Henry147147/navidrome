"""
Description generation and text embedding pipeline for Navidrome uploads.

This module wires together NVIDIA's Music Flamingo captioning model with
Qwen3-Embedding-8B to produce rich text descriptions and cosine-normalized
embeddings for each uploaded track. The resulting vectors are stored in a
separate Milvus collection so they can be queried alongside the existing
MuQ audio embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from pymilvus import MilvusClient, DataType
from transformers import AutoModel, AutoProcessor, AutoTokenizer

try:
    # Newer Transformers (5.x dev) expose AudioFlamingo3
    from transformers import AudioFlamingo3ForConditionalGeneration
    _FlamingoCls = AudioFlamingo3ForConditionalGeneration
    _HAS_MUSIC_FLAMINGO = True
except Exception:  # pragma: no cover - optional heavy dependency
    try:
        # Older builds used MusicFlamingoForConditionalGeneration
        from transformers import MusicFlamingoForConditionalGeneration

        _FlamingoCls = MusicFlamingoForConditionalGeneration
        _HAS_MUSIC_FLAMINGO = True
    except Exception:
        _FlamingoCls = None
        _HAS_MUSIC_FLAMINGO = False

from gpu_settings import GPUSettings, is_oom_error, load_gpu_settings


DEFAULT_DESCRIPTION_COLLECTION = "description_embedding"


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _normalize(vec: torch.Tensor) -> torch.Tensor:
    vec = vec.detach().float()
    denom = torch.linalg.norm(vec) + 1e-8
    return vec / denom


@dataclass
class DescriptionSegment:
    """Lightweight container for a single caption + embedding."""

    title: str
    description: str
    embedding: List[float]
    offset_seconds: float = 0.0
    duration_seconds: Optional[float] = None


class MusicFlamingoCaptioner:
    """Wrap Music Flamingo caption generation."""

    def __init__(
        self,
        *,
        model_id: str = "nvidia/music-flamingo-hf",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        logger: Optional[logging.Logger] = None,
        gpu_settings: Optional[GPUSettings] = None,
    ) -> None:
        if not _HAS_MUSIC_FLAMINGO:
            raise ImportError(
                "MusicFlamingoForConditionalGeneration not available. "
                "Install the latest transformers (git+https://github.com/huggingface/transformers) "
                "or pin a version that provides AudioFlamingo3/MusicFlamingo."
            )
        self.model_id = model_id
        self.gpu_settings = gpu_settings or load_gpu_settings()
        self.device = device or self.gpu_settings.device_target()
        if torch_dtype is not None:
            self.dtype = torch_dtype
        else:
            self.dtype = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
        self.logger = logger or logging.getLogger("navidrome.description.captioner")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self._build_model()
        self.model.eval()

    def _build_model(self):
        max_memory = self.gpu_settings.max_memory_map()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            model = _FlamingoCls.from_pretrained(
                self.model_id, torch_dtype=self.dtype, max_memory=max_memory
            )
            return model.to(self.device)
        except Exception as exc:
            if is_oom_error(exc):
                self.logger.warning(
                    "MusicFlamingo load hit OOM on %s; retrying on CPU with fp32",
                    self.device,
                )
                self.device = "cpu"
                self.dtype = torch.float32
                model = _FlamingoCls.from_pretrained(
                    self.model_id, torch_dtype=self.dtype
                )
                return model.to(self.device)
            raise

    def unload(self):
        try:
            if self.model:
                self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(self, audio_path: str, prompt: Optional[str] = None) -> str:
        prompt = prompt or (
            "Provide a vivid, detailed description of this song's mood, style, "
            "instrumentation, vocals, tempo, genre influences, and production details."
        )

        if self.model is None:
            self.model = self._build_model()
            self.model.eval()

        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        audio = waveform.squeeze(0).numpy()
        inputs = self.processor(
            audios=audio,
            sampling_rate=sample_rate,
            text=[prompt],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=120)
        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ].strip()
        return caption


class Qwen3Embedder:
    """Embedding helper around Qwen3-Embedding-8B."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.float16,
        logger: Optional[logging.Logger] = None,
        gpu_settings: Optional[GPUSettings] = None,
    ) -> None:
        self.model_id = model_id
        self.gpu_settings = gpu_settings or load_gpu_settings()
        self.device = device or self.gpu_settings.device_target()
        if torch_dtype is not None:
            self.dtype = torch_dtype
        else:
            self.dtype = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
        self.logger = logger or logging.getLogger("navidrome.description.qwen3")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        # Qwen3 embedding models expect right padding; ensure consistent behavior
        try:
            self.tokenizer.padding_side = "right"
        except Exception:
            pass
        self.device_map = "auto" if self.device.startswith("cuda") else None
        self.max_memory = self.gpu_settings.max_memory_map()
        self.model = self._build_model()
        self.model.eval()
        self.gpu_settings.apply_runtime_limits()

    def _build_model(self):
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # If free memory is low, fall back to CPU proactively
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024**3)
                if free_gb < 2.0:
                    self.logger.warning(
                        "Only %.2f GiB free on GPU; loading %s on CPU to avoid OOM",
                        free_gb,
                        self.model_id,
                    )
                    self.device = "cpu"
                    self.device_map = None
                    self.dtype = torch.float32
                    self.max_memory = None
            except Exception:
                pass

        try:
            model = AutoModel.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                device_map=self.device_map,
                max_memory=self.max_memory,
                low_cpu_mem_usage=True,
            )
            if self.device_map is None:
                model = model.to(self.device)
            return model
        except Exception as exc:
            if is_oom_error(exc):
                self.logger.warning(
                    "Qwen3 load hit OOM on %s; retrying on CPU fp32", self.device
                )
                self.device = "cpu"
                self.device_map = None
                self.dtype = torch.float32
                self.max_memory = None
                model = AutoModel.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    torch_dtype=self.dtype,
                    device_map=self.device_map,
                    low_cpu_mem_usage=True,
                ).to(self.device)
                return model
            raise

    def unload(self):
        try:
            if self.model:
                self.model.to("cpu")
        except Exception:
            pass
        self.model = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def embed_text(self, text: str) -> torch.Tensor:
        """
        Embed a single text string and return an L2-normalized CPU tensor.

        The upstream Qwen3 HF implementation no longer exposes an `.encode()`
        helper, so we perform a lightweight forward pass and pool the last token
        per sequence. When the helper exists (older checkpoints), we still use
        it for compatibility.
        """
        if self.model is None:
            self.model = self._build_model()
            self.model.eval()

        encode_fn = getattr(self.model, "encode", None)
        if callable(encode_fn):
            embeddings = encode_fn(text, tokenizer=self.tokenizer)
            if isinstance(embeddings, (list, tuple)):
                embeddings = embeddings[0]
            tensor = torch.as_tensor(embeddings, device="cpu", dtype=torch.float32)
            if tensor.dim() == 2 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            return _normalize(tensor)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(self.tokenizer.model_max_length, 8192),
        )

        target_device = getattr(self.model, "device", torch.device(self.device))
        inputs = {k: v.to(target_device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)
            pooled = self._last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
        if pooled.dim() == 2 and pooled.size(0) == 1:
            pooled = pooled.squeeze(0)
        return _normalize(pooled).to("cpu")

    @staticmethod
    def _last_token_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool by taking the hidden state of the last non-padding token.
        Mirrors the approach recommended in the official Qwen3 embedding docs.
        """
        # attention_mask shape: (batch, seq_len) with 1s for real tokens
        seq_lengths = attention_mask.sum(dim=1) - 1  # last valid token index
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_indices, seq_lengths]


class DescriptionEmbeddingPipeline:
    """Generate descriptions and text embeddings for uploaded tracks."""

    def __init__(
        self,
        *,
        caption_model_id: str = "nvidia/music-flamingo-hf",
        text_model_id: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        gpu_settings: Optional[GPUSettings] = None,
    ) -> None:
        self.gpu_settings = gpu_settings or load_gpu_settings()
        self.device = device or self.gpu_settings.device_target()
        self.logger = logger or logging.getLogger("navidrome.description")
        self.caption_model_id = caption_model_id
        # Avoid name clash with the _get_captioner() accessor below
        self._captioner_instance: Optional[MusicFlamingoCaptioner] = None
        self.embedder = Qwen3Embedder(
            model_id=text_model_id,
            device=self.device,
            logger=self.logger,
            gpu_settings=self.gpu_settings,
        )

    def _get_captioner(self) -> MusicFlamingoCaptioner:
        """
        Lazily construct the Music Flamingo captioner so environments without the
        heavy dependency (or CI) can still run embed-only paths.
        """
        if self._captioner_instance is None:
            if not _HAS_MUSIC_FLAMINGO:
                raise ImportError(
                    "MusicFlamingoForConditionalGeneration not available. "
                    "Install transformers with Music Flamingo support to generate captions."
                )
            self._captioner_instance = MusicFlamingoCaptioner(
                model_id=self.caption_model_id,
                device=self.device,
                logger=self.logger,
                gpu_settings=self.gpu_settings,
            )
        return self._captioner_instance

    def ensure_milvus_schemas(self, client: MilvusClient) -> None:
        existing = set(client.list_collections())
        if DEFAULT_DESCRIPTION_COLLECTION in existing:
            return

        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field("name", DataType.VARCHAR, is_primary=True, max_length=512)
        schema.add_field("description", DataType.VARCHAR, max_length=2048)
        schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=4096)
        schema.add_field("offset", DataType.FLOAT)
        schema.add_field("model_id", DataType.VARCHAR, max_length=256)
        client.create_collection(DEFAULT_DESCRIPTION_COLLECTION, schema=schema)

    def ensure_milvus_index(self, client: MilvusClient) -> None:
        indexes = client.describe_collection(DEFAULT_DESCRIPTION_COLLECTION).get(
            "indexes", []
        )
        index_fields = {index.get("field_name") for index in indexes}
        if "embedding" in index_fields and "name" in index_fields:
            return

        params = MilvusClient.prepare_index_params()
        params.add_index(field_name="name", index_type="INVERTED")
        from embedding_models import _milvus_uses_lite

        if _milvus_uses_lite():
            params.add_index(
                field_name="embedding",
                index_type="IVF_FLAT",
                metric_type="COSINE",
                params={"nlist": 1024},
            )
        else:
            params.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 50, "efConstruction": 250},
            )
        client.create_index(DEFAULT_DESCRIPTION_COLLECTION, params)

    def describe_music(
        self, music_file: str, music_name: str, *, offset_seconds: float = 0.0
    ) -> List[DescriptionSegment]:
        try:
            description = self._get_captioner().generate(music_file)
        except Exception:
            self.logger.exception(
                "Failed to generate caption with Music Flamingo; using fallback text"
            )
            description = (
                f"Audio track titled '{music_name}' with mixed instrumentation."
            )

        embedding = self.embedder.embed_text(description)

        return [
            DescriptionSegment(
                title=music_name,
                description=description,
                embedding=embedding.cpu().tolist(),
                offset_seconds=float(offset_seconds),
                duration_seconds=None,
            )
        ]

    def embed_text(self, text: str) -> List[float]:
        return self.embedder.embed_text(text).cpu().tolist()

    def embed_music(self, music_file: str, music_name: str) -> dict:
        """Compatibility wrapper so batch jobs can treat this like other models."""
        payload = self.prepare_payload(music_file, music_name)
        payload["segments"] = [
            {
                "title": desc.get("title"),
                "embedding": desc.get("embedding"),
                "offset_seconds": desc.get("offset_seconds", 0.0),
                "description": desc.get("description"),
            }
            for desc in payload.get("descriptions", [])
        ]
        return payload

    def ensure_model_loaded(self):
        # Models are loaded during __init__; this mirrors BaseEmbeddingModel API
        return self

    def unload_model(self) -> None:
        try:
            if self._captioner_instance:
                self._captioner_instance.unload()
                self._captioner_instance = None
            if hasattr(self.embedder, "unload"):
                self.embedder.unload()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def prepare_payload(self, music_file: str, music_name: str) -> dict:
        segments = self.describe_music(music_file, music_name)
        return {
            "music_file": str(Path(music_file)),
            "model_id": self.embedder.model_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "descriptions": [
                {
                    **segment.__dict__,
                    "model_id": self.embedder.model_id,
                }
                for segment in segments
            ],
        }
