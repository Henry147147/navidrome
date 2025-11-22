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
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    MusicFlamingoForConditionalGeneration,
)


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
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()
        if torch_dtype is not None:
            self.dtype = torch_dtype
        else:
            self.dtype = (
                torch.float16 if self.device.startswith("cuda") else torch.float32
            )
        self.logger = logger or logging.getLogger("navidrome.description.captioner")

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def generate(self, audio_path: str, prompt: Optional[str] = None) -> str:
        prompt = prompt or (
            "Provide a vivid, detailed description of this song's mood, style, "
            "instrumentation, vocals, tempo, genre influences, and production details."
        )

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
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()
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
        # Use device_map only for CUDA to avoid CPU placement errors
        device_map = "auto" if self.device.startswith("cuda") else None
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map=device_map,
        )
        if device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> torch.Tensor:
        # The model provides an encode helper when trust_remote_code=True
        embeddings = self.model.encode(text, tokenizer=self.tokenizer)
        if isinstance(embeddings, (list, tuple)):
            embeddings = embeddings[0]

        tensor = torch.as_tensor(embeddings, device="cpu", dtype=torch.float32)
        return _normalize(tensor)


class DescriptionEmbeddingPipeline:
    """Generate descriptions and text embeddings for uploaded tracks."""

    def __init__(
        self,
        *,
        caption_model_id: str = "nvidia/music-flamingo-hf",
        text_model_id: str = "Qwen/Qwen3-Embedding-8B",
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.device = device or _default_device()
        self.logger = logger or logging.getLogger("navidrome.description")
        self.captioner = MusicFlamingoCaptioner(
            model_id=caption_model_id, device=self.device, logger=self.logger
        )
        self.embedder = Qwen3Embedder(
            model_id=text_model_id, device=self.device, logger=self.logger
        )

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
            description = self.captioner.generate(music_file)
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
