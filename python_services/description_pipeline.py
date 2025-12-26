"""
Description generation and text embedding pipeline for Navidrome uploads.

This module wires together NVIDIA's Music Flamingo captioning model with
Qwen3-Embedding-4B to produce rich text descriptions and cosine-normalized
embeddings for each uploaded track. Music Flamingo audio features are also
pooled into a standalone audio embedding. Both vectors are stored in
separate Milvus collections so they can be queried alongside the existing
MuQ audio embeddings.
"""

from __future__ import annotations

import logging
import json
from safetensors.torch import load_file
from optimum.quanto import requantize
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from pymilvus import MilvusClient, DataType
from transformers import AutoModel, AutoTokenizer, AutoConfig

from gpu_settings import GPUSettings, is_oom_error, load_gpu_settings
from gpu_model_coordinator import GPU_COORDINATOR

from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor as FlamingoProcessor,
)
DEFAULT_DESCRIPTION_COLLECTION = "description_embedding"
DEFAULT_AUDIO_COLLECTION = "flamingo_audio_embedding"
DESCRIPTION_JSON_PATH = Path("song_descriptions.json")


@dataclass
class DescriptionSegment:
    """Lightweight container for a single caption + embeddings."""

    title: str
    description: str
    description_embedding: List[float]
    audio_embedding: List[float]
    offset_seconds: float = 0.0
    duration_seconds: Optional[float] = None

    def to_payload(self, *, text_model_id: str, audio_model_id: str) -> dict:
        """Serialize segment with stable field names for downstream storage."""
        return {
            "title": self.title,
            "description": self.description,
            "description_embedding": self.description_embedding,
            "audio_embedding": self.audio_embedding,
            # Backwards-compatible alias used by existing Milvus insertion logic.
            "embedding": self.description_embedding,
            "offset_seconds": self.offset_seconds,
            "duration_seconds": self.duration_seconds,
            "model_id": text_model_id,
            "audio_model_id": audio_model_id,
        }


def _pool_audio_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """
    Pool variable-length Flamingo audio embeddings into a single vector.

    Supports 1D, 2D (tokens × dim), or 3D (batch × tokens × dim) tensors.
    Returns an L2-normalized 1D float tensor.
    """
    if embedding.dim() == 1:
        pooled = embedding
    elif embedding.dim() == 2:
        pooled = embedding.mean(dim=0)
    elif embedding.dim() == 3:
        pooled = embedding.mean(dim=1).mean(dim=0)
    else:
        raise ValueError(
            f"Unexpected audio embedding shape {tuple(embedding.shape)}; "
            "expected 1D, 2D, or 3D tensor."
        )

    pooled = pooled.to(torch.float32)
    norm = torch.linalg.norm(pooled)
    if norm > 0:
        pooled = pooled / norm
    return pooled


class MusicFlamingoCaptioner:
    """Wrap Music Flamingo caption generation."""

    def __init__(
        self,
        *,
        model_id: str = "nvidia/music-flamingo-hf",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        logger: Optional[logging.Logger] = None,
        gpu_settings: Optional[GPUSettings] = None,
    ) -> None:
        self.last_input_embeds: Optional[torch.Tensor] = None
        self.model_id = model_id
        self.gpu_settings = gpu_settings or load_gpu_settings()
        self.device = device or self.gpu_settings.device_target()
        if not str(self.device).startswith("cuda"):
            raise RuntimeError("Music Flamingo must run on CUDA; CPU is too slow.")
        self.dtype = torch_dtype or torch.bfloat16
        self.logger = logger or logging.getLogger("navidrome.description.captioner")
        self._gpu_owner = "music_flamingo_captioner"
        GPU_COORDINATOR.register(self._gpu_owner, self._offload_to_cpu)

        self.processor = FlamingoProcessor.from_pretrained(self.model_id)
        self.model = self._build_model()
        self.model.eval()

    def make_quantized(self, model):
        with open("./music_flamingo_fp8_quantization_map.json", "r") as file:
            quant_map = json.load(file)

        quantized_weights = load_file("./music_flamingo_fp8.safetensor")
        requantize(model, quantized_weights, quant_map)

    def _build_model(self) -> AudioFlamingo3ForConditionalGeneration:
        # Prefer PyTorch SDPA attention to avoid optional flash-attn dependency.
        max_memory = self.gpu_settings.max_memory_map()
        device_map = "auto" if max_memory is not None else {"": self.device}

        GPU_COORDINATOR.claim(self._gpu_owner, self.logger)

        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map=device_map,
            max_memory=max_memory,
        )
        self.make_quantized(model)
        if device_map and device_map != "auto":
            # Ensure every module ends up on the requested CUDA device when not auto-sharded.
            model = model.to(self.device)
        self.gpu_settings.apply_runtime_limits()

        # monkeypatch the language model to save the input_embeds so we have access to it
        old_call = model.get_audio_features

        def new_call(
            input_features: torch.FloatTensor, input_features_mask: torch.Tensor
        ):
            return_value = old_call(input_features, input_features_mask)
            if self.last_input_embeds is None:
                self.last_input_embeds = return_value.cpu().detach()
            return return_value

        model.get_audio_features = new_call
        # TODO, test this

        return model

    def _ensure_model_on_device(self):
        GPU_COORDINATOR.claim(self._gpu_owner, self.logger)
        if self.model is None:
            self.model = self._build_model()
        else:
            try:
                self.model = self.model.to(self.device, dtype=self.dtype)
            except Exception:
                self.logger.exception(
                    "Failed to move Music Flamingo back to GPU; rebuilding"
                )
                self.model = self._build_model()
        self.model.eval()

    def _offload_to_cpu(self) -> None:
        try:
            if self.model is not None:
                self.model = self.model.to("cpu")
        except Exception:
            self.logger.exception("Failed to offload Music Flamingo to CPU")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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

    def generate(
        self, audio_path: str, prompt: Optional[str] = None
    ) -> Tuple[str, List[float]]:
        """
        Generate a rich caption for the provided audio file using the official
        Music Flamingo chat template, mirroring the model card example.
        """
        prompt = prompt or (
            """Describe this track in full detail - tell me the genre, tempo, and key, then dive into the instruments, production style, 
            and overall mood it creates. Also describe the language used (or if it is instrumental), any clearly intelligible and 
            important lyrics or recurring phrases and what they are about, the emotions the music and vocals evoke, any cultural or 
            regional influences you can hear, and the era or scene it most strongly resembles based only on the audio. Include any 
            other musically relevant details you can infer directly from the sound, such as song structure, vocal style, and how the energy changes over time."""
        )

        if self.model is None:
            self._ensure_model_on_device()
        else:
            self._ensure_model_on_device()
        assert self.model is not None

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": str(audio_path)},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        # Ensure tensors are on the correct device/dtype
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    inputs[k] = v.to(self.model.device, dtype=self.dtype)
                else:
                    inputs[k] = v.to(self.model.device)

        with torch.inference_mode():
            # Allow long, detailed captions for complex tracks.
            self.last_input_embeds = None
            outputs = self.model.generate(**inputs, max_new_tokens=8192)

        assert self.last_input_embeds is not None
        audio_embedding = _pool_audio_embedding(self.last_input_embeds).tolist()
        self.last_input_embeds = None

        # Strip the prompt tokens from the generated sequence
        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        return decoded[0].strip(), audio_embedding


class Qwen3Embedder:
    """Embedding helper around Qwen3-Embedding-4B."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen3-Embedding-4B",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        logger: Optional[logging.Logger] = None,
        gpu_settings: Optional[GPUSettings] = None,
    ) -> None:
        self.model_id = model_id
        self.gpu_settings = gpu_settings or load_gpu_settings()
        self.device = device or self.gpu_settings.device_target()
        self.dtype = (
            torch_dtype
            if torch_dtype is not None
            else (torch.float16 if self.device.startswith("cuda") else torch.float32)
        )
        self.logger = logger or logging.getLogger("navidrome.description.qwen3")
        self._gpu_owner = "qwen3_embedder"
        GPU_COORDINATOR.register(self._gpu_owner, self._offload_to_cpu)

        # Follow the official model card recommendations: left padding and last-token pooling.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, padding_side="left", trust_remote_code=True
        )
        self.device_map = "auto" if self.device.startswith("cuda") else None
        try:
            self.model = self._build_model()
        except Exception as exc:
            if is_oom_error(exc) and self.device.startswith("cuda"):
                self.logger.warning(
                    "Qwen3 load hit OOM on cuda; retrying on CPU fp32"
                )
                self.device = "cpu"
                self.device_map = None
                self.dtype = torch.float32
                self.model = self._build_model()
            else:
                raise
        self.model.eval()
        self.gpu_settings.apply_runtime_limits()

    def _build_model(self):
        GPU_COORDINATOR.claim(self._gpu_owner, self.logger)
        model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=self.dtype,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            device_map=self.device_map,
        )
        if self.device_map is None:
            model = model.to(self.device)
        return model

    def _ensure_model_on_device(self):
        GPU_COORDINATOR.claim(self._gpu_owner, self.logger)
        if self.model is None:
            self.model = self._build_model()
        else:
            try:
                self.model = self.model.to(self.device)
            except Exception:
                self.logger.exception(
                    "Failed to move Qwen3 embedder back to GPU; rebuilding"
                )
                self.model = self._build_model()
        self.model.eval()

    def _offload_to_cpu(self) -> None:
        try:
            if self.model is not None:
                self.model = self.model.to("cpu")
        except Exception:
            self.logger.exception("Failed to offload Qwen3 embedder to CPU")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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

        Mirrors the official model card recipe: left padding + last token pooling.
        """
        if self.model is None:
            self._ensure_model_on_device()
        else:
            self._ensure_model_on_device()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=min(self.tokenizer.model_max_length, 8192),
        ).to(self.model.device)

        with torch.inference_mode():
            outputs = self.model(**inputs)
            pooled = self._last_token_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

        # Normalize as cosine embedding and move to CPU for downstream storage
        normalized = F.normalize(pooled, p=2, dim=-1)
        if normalized.dim() == 2 and normalized.size(0) == 1:
            normalized = normalized.squeeze(0)
        return normalized.to("cpu")

    @staticmethod
    def _last_token_pool(
        last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool by taking the hidden state of the last non-padding token.
        Mirrors the official Qwen3 embedding docs (handles left padding).
        """
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_state[:, -1]
        seq_lengths = attention_mask.sum(dim=1) - 1  # last valid token index
        batch_indices = torch.arange(
            last_hidden_state.size(0), device=last_hidden_state.device
        )
        return last_hidden_state[batch_indices, seq_lengths]


class DescriptionEmbeddingPipeline:
    """Generate descriptions and text embeddings for uploaded tracks."""

    # Hint for batch jobs to use the two-stage caption + embed flow.
    caption_only = True
    collection_audio = DEFAULT_AUDIO_COLLECTION

    def __init__(
        self,
        *,
        caption_model_id: str = "nvidia/music-flamingo-hf",
        text_model_id: str = "Qwen/Qwen3-Embedding-4B",
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
        self.text_model_id = text_model_id
        self._embedder_instance: Optional[Qwen3Embedder] = None
        self._audio_embedding_dim: Optional[int] = None

    def _get_captioner(self) -> MusicFlamingoCaptioner:
        """
        Lazily construct the Music Flamingo captioner so environments without the
        heavy dependency (or CI) can still run embed-only paths.
        """
        if self._captioner_instance is None:
            self._captioner_instance = MusicFlamingoCaptioner(
                model_id=self.caption_model_id,
                device=self.device,
                logger=self.logger,
                gpu_settings=self.gpu_settings,
            )
        return self._captioner_instance

    def _get_embedder(self) -> Qwen3Embedder:
        """Lazily construct the Qwen3 embedder (never loaded alongside Flamingo)."""
        if self._embedder_instance is None:
            # Make sure captioner is not occupying GPU when embedding
            self.unload_captioner()
            self._embedder_instance = Qwen3Embedder(
                model_id=self.text_model_id,
                device=self.device,
                logger=self.logger,
                gpu_settings=self.gpu_settings,
            )
        return self._embedder_instance

    def _resolve_audio_embedding_dim(self) -> Optional[int]:
        """Resolve Flamingo audio embedding dimension from config or cached value."""
        cached = getattr(self, "_audio_embedding_dim", None)
        if cached:
            return int(cached)
        try:
            config = AutoConfig.from_pretrained(
                self.caption_model_id, trust_remote_code=True
            )
            dim = None
            text_cfg = getattr(config, "text_config", None)
            if text_cfg is not None and hasattr(text_cfg, "hidden_size"):
                dim = int(text_cfg.hidden_size)
            elif hasattr(config, "hidden_size"):
                dim = int(config.hidden_size)
            if dim:
                self._audio_embedding_dim = dim
                return dim
        except Exception:
            # Avoid hard failure on schema creation; will retry after first embedding.
            self.logger.debug(
                "Unable to resolve Flamingo audio embedding dimension yet",
                exc_info=True,
            )
        return None

    def ensure_milvus_schemas(
        self, client: MilvusClient, *, audio_dim: Optional[int] = None
    ) -> None:
        existing = set(client.list_collections())
        if DEFAULT_DESCRIPTION_COLLECTION not in existing:
            schema = MilvusClient.create_schema(
                auto_id=False, enable_dynamic_field=False
            )
            schema.add_field("name", DataType.VARCHAR, is_primary=True, max_length=512)
            schema.add_field("description", DataType.VARCHAR, max_length=2048)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=2560)
            schema.add_field("offset", DataType.FLOAT)
            schema.add_field("model_id", DataType.VARCHAR, max_length=256)
            client.create_collection(DEFAULT_DESCRIPTION_COLLECTION, schema=schema)

        if DEFAULT_AUDIO_COLLECTION not in existing:
            resolved_dim = audio_dim or self._resolve_audio_embedding_dim()
            if resolved_dim:
                self._audio_embedding_dim = int(resolved_dim)
                schema = MilvusClient.create_schema(
                    auto_id=False, enable_dynamic_field=False
                )
                schema.add_field(
                    "name", DataType.VARCHAR, is_primary=True, max_length=512
                )
                schema.add_field(
                    "embedding", DataType.FLOAT_VECTOR, dim=int(resolved_dim)
                )
                schema.add_field("offset", DataType.FLOAT)
                schema.add_field("model_id", DataType.VARCHAR, max_length=256)
                client.create_collection(DEFAULT_AUDIO_COLLECTION, schema=schema)
            else:
                self.logger.warning(
                    "Skipping audio embedding schema creation; dimension unavailable"
                )

    def ensure_milvus_index(self, client: MilvusClient) -> None:
        existing = set(client.list_collections())
        from embedding_models import _milvus_uses_lite

        if DEFAULT_DESCRIPTION_COLLECTION in existing:
            indexes = client.describe_collection(DEFAULT_DESCRIPTION_COLLECTION).get(
                "indexes", []
            )
            index_fields = {index.get("field_name") for index in indexes}
            if "embedding" not in index_fields or "name" not in index_fields:
                params = MilvusClient.prepare_index_params()
                params.add_index(field_name="name", index_type="INVERTED")
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

        if DEFAULT_AUDIO_COLLECTION in existing:
            indexes = client.describe_collection(DEFAULT_AUDIO_COLLECTION).get(
                "indexes", []
            )
            index_fields = {index.get("field_name") for index in indexes}
            if "embedding" not in index_fields or "name" not in index_fields:
                params = MilvusClient.prepare_index_params()
                params.add_index(field_name="name", index_type="INVERTED")
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
                client.create_index(DEFAULT_AUDIO_COLLECTION, params)

    def describe_music(
        self, music_file: str, music_name: str, *, offset_seconds: float = 0.0
    ) -> Optional[List[DescriptionSegment]]:
        try:
            description, audio_embedding = self._get_captioner().generate(music_file)
        except Exception:
            self.logger.exception(
                "Failed to generate caption with Music Flamingo; using fallback text"
            )
            return None

        if audio_embedding:
            self._audio_embedding_dim = len(audio_embedding)

        # Persist raw description immediately
        self._persist_description(music_name, description, music_file)

        # Avoid keeping Flamingo and Qwen on GPU at the same time
        self.unload_captioner()
        embedding = self._get_embedder().embed_text(description)

        return [
            DescriptionSegment(
                title=music_name,
                description=description,
                description_embedding=embedding.cpu().tolist(),
                audio_embedding=audio_embedding,
                offset_seconds=float(offset_seconds),
                duration_seconds=None,
            )
        ]

    def get_caption(self, music_file: str, music_name: str) -> Tuple[str, List[float]]:
        """Generate caption only (no embeddings) and persist to JSON."""
        description, audio_embedding = self._get_captioner().generate(music_file)
        self._persist_description(music_name, description, music_file)
        if audio_embedding:
            self._audio_embedding_dim = len(audio_embedding)
        return description, audio_embedding

    def embed_description(
        self, description: str, music_name: str, *, offset_seconds: float = 0.0
    ) -> DescriptionSegment:
        """Embed a pre-generated description."""
        self.unload_captioner()
        embedding = self._get_embedder().embed_text(description)
        return DescriptionSegment(
            title=music_name,
            description=description,
            description_embedding=embedding.cpu().tolist(),
            audio_embedding=[],
            # TODO
            offset_seconds=float(offset_seconds),
            duration_seconds=None,
        )

    def _persist_description(
        self, music_name: str, description: str, music_file: str
    ) -> None:
        """Append/replace description in song_descriptions.json."""
        try:
            existing: List[dict] = []
            if DESCRIPTION_JSON_PATH.exists():
                existing = json.loads(DESCRIPTION_JSON_PATH.read_text())
            # Replace if name already exists
            existing = [entry for entry in existing if entry.get("name") != music_name]
            existing.append(
                {
                    "name": music_name,
                    "description": description,
                    "music_file": str(music_file),
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            DESCRIPTION_JSON_PATH.write_text(json.dumps(existing, indent=2))
        except Exception:
            self.logger.exception("Failed to persist description for %s", music_name)

    def embed_text(self, text: str) -> List[float]:
        return self._get_embedder().embed_text(text).cpu().tolist()

    def embed_music(self, music_file: str, music_name: str) -> dict:
        """Compatibility wrapper so batch jobs can treat this like other models."""
        payload = self.prepare_payload(music_file, music_name)
        payload["segments"] = [
            {
                "title": desc.get("title"),
                "embedding": desc.get("embedding"),
                "offset_seconds": desc.get("offset_seconds", 0.0),
                "description": desc.get("description"),
                "audio_embedding": desc.get("audio_embedding"),
                "audio_model_id": desc.get("audio_model_id"),
            }
            for desc in payload.get("descriptions", [])
        ]
        return payload

    def ensure_model_loaded(self):
        # Models are loaded during __init__; this mirrors BaseEmbeddingModel API
        return self

    def unload_model(self) -> None:
        self.unload_captioner()
        self.unload_embedder()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    def unload_captioner(self) -> None:
        try:
            if self._captioner_instance:
                self._captioner_instance.unload()
                self._captioner_instance = None
        except Exception:
            pass

    def unload_embedder(self) -> None:
        try:
            if self._embedder_instance and hasattr(self._embedder_instance, "unload"):
                self._embedder_instance.unload()
                self._embedder_instance = None
        except Exception:
            pass

    def prepare_payload(self, music_file: str, music_name: str) -> dict:
        segments = self.describe_music(music_file, music_name) or []
        return {
            "music_file": str(Path(music_file)),
            "model_id": self.text_model_id,
            "caption_model_id": self.caption_model_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "descriptions": [
                segment.to_payload(
                    text_model_id=self.text_model_id,
                    audio_model_id=self.caption_model_id,
                )
                for segment in segments
            ],
        }
