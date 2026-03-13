from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
from typing import Sequence, TypeVar

import librosa
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoProcessor
from transformers.models.audioflamingo3.modeling_audioflamingo3 import AudioFlamingo3MultiModalProjector
from transformers.utils import cached_file

from python_services.muq_shared import (
    AUDIO_SAMPLE_RATE,
    MODEL_MUSIC_FLAMINGO_AUDIO,
    MODEL_MUQ_AUDIO,
    MODEL_MUQ_MULAN,
    default_device,
    default_music_flamingo_ref,
    default_muq_audio_ref,
    default_muq_cache_dir,
    default_muq_max_audio_seconds,
    default_muq_mulan_ref,
    normalize_model_name,
)

T = TypeVar("T")
MUSIC_FLAMINGO_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class MuQProviderConfig:
    flamingo_model_ref: str = default_music_flamingo_ref()
    audio_model_ref: str = default_muq_audio_ref()
    mulan_model_ref: str = default_muq_mulan_ref()
    cache_dir: str = default_muq_cache_dir()
    device: str = default_device()
    audio_batch_size: int = 4
    text_batch_size: int = 16
    max_audio_seconds: float = default_muq_max_audio_seconds()

    @classmethod
    def from_env(cls) -> "MuQProviderConfig":
        return cls(
            flamingo_model_ref=default_music_flamingo_ref(),
            audio_model_ref=default_muq_audio_ref(),
            mulan_model_ref=default_muq_mulan_ref(),
            cache_dir=default_muq_cache_dir(),
            device=default_device(),
            audio_batch_size=max(1, int(float(os.environ.get("MUQ_AUDIO_BATCH_SIZE", "4")))),
            text_batch_size=max(1, int(float(os.environ.get("MUQ_TEXT_BATCH_SIZE", "16")))),
            max_audio_seconds=default_muq_max_audio_seconds(),
        )


class MuQProvider:
    def __init__(self, config: MuQProviderConfig | None = None) -> None:
        self.config = config or MuQProviderConfig.from_env()
        self.device = torch.device(self.config.device)
        self._music_flamingo_model = None
        self._audio_model = None
        self._mulan_model = None

    def load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = _load_audio(path, sample_rate=None, mono=False, max_audio_seconds=self.config.max_audio_seconds)
        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim == 1:
            mono = audio
        elif audio.ndim == 2:
            mono = audio.mean(axis=0)
        else:
            raise ValueError(f"expected audio array with 1 or 2 dims, got {tuple(audio.shape)}")
        if sample_rate != AUDIO_SAMPLE_RATE:
            mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=AUDIO_SAMPLE_RATE)
        return torch.from_numpy(np.ascontiguousarray(mono, dtype=np.float32))

    def embed_audio_files(self, paths: Sequence[str], model: str) -> list[list[float]]:
        canonical = normalize_model_name(model)
        if canonical == MODEL_MUSIC_FLAMINGO_AUDIO:
            return self._embed_with_music_flamingo(paths)
        waveforms = [self.load_audio_file(path) for path in paths]
        return self.embed_audio_waveforms(waveforms, canonical)

    def embed_audio_waveforms(self, waveforms: Sequence[torch.Tensor], model: str) -> list[list[float]]:
        if not waveforms:
            return []

        canonical = normalize_model_name(model)
        if canonical == MODEL_MUSIC_FLAMINGO_AUDIO:
            raise ValueError("music_flamingo_audio embeddings require audio file paths")
        if canonical == MODEL_MUQ_AUDIO:
            return self._embed_with_muq_audio(waveforms)
        if canonical == MODEL_MUQ_MULAN:
            return self._embed_with_muq_mulan_audio(waveforms)
        raise ValueError(f"unsupported audio model: {model}")

    def embed_texts(self, texts: Sequence[str], *, model: str = MODEL_MUQ_MULAN, dimensions: int = 0) -> list[list[float]]:
        if not texts:
            return []

        canonical = normalize_model_name(model)
        if canonical != MODEL_MUQ_MULAN:
            raise ValueError("text embeddings are only available for muq_mulan")

        mulan = self._load_muq_mulan_model()
        batches = []
        for chunk in _chunked(texts, self.config.text_batch_size):
            with torch.inference_mode():
                latents = mulan(texts=list(chunk)).detach().to(torch.float32)
            batches.append(latents.cpu())
        all_latents = torch.cat(batches, dim=0)
        adjusted = _normalize_dimensions(all_latents, dimensions)
        return _tensor_rows_to_unit_vectors(adjusted)

    def unload_model(self, model: str) -> None:
        canonical = normalize_model_name(model)
        if canonical == MODEL_MUSIC_FLAMINGO_AUDIO:
            self._music_flamingo_model = None
        elif canonical == MODEL_MUQ_AUDIO:
            self._audio_model = None
        elif canonical == MODEL_MUQ_MULAN:
            self._mulan_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _embed_with_muq_audio(self, waveforms: Sequence[torch.Tensor]) -> list[list[float]]:
        model = self._load_muq_audio_model()
        model_device, model_dtype = _module_device_and_dtype(model)
        rows: list[list[float]] = []
        for batch in _chunked(waveforms, self.config.audio_batch_size):
            padded, attention_mask = _pad_waveforms(batch)
            with torch.inference_mode():
                output = model(
                    padded.to(model_device, dtype=model_dtype),
                    attention_mask=attention_mask.to(model_device),
                    output_hidden_states=True,
                )
            pooled = _pool_muq_hidden_states(model, output.last_hidden_state, attention_mask.to(model_device))
            rows.extend(_tensor_rows_to_unit_vectors(pooled.cpu()))
        return rows

    def _embed_with_muq_mulan_audio(self, waveforms: Sequence[torch.Tensor]) -> list[list[float]]:
        mulan = self._load_muq_mulan_model()
        mulan_device, mulan_dtype = _module_device_and_dtype(mulan)
        outputs: list[torch.Tensor] = []
        for batch in _chunked(waveforms, self.config.audio_batch_size):
            for waveform in batch:
                with torch.inference_mode():
                    latent = mulan(
                        wavs=waveform.unsqueeze(0).to(mulan_device, dtype=mulan_dtype),
                        parallel_processing=False,
                    )
                outputs.append(latent.squeeze(0).detach().to(torch.float32).cpu())
        if not outputs:
            return []
        return _tensor_rows_to_unit_vectors(torch.stack(outputs, dim=0))

    def _embed_with_music_flamingo(self, paths: Sequence[str]) -> list[list[float]]:
        music_flamingo = self._load_music_flamingo_model()
        outputs: list[torch.Tensor] = []
        for path in paths:
            with torch.inference_mode():
                embedding = music_flamingo.extract_embedding(path)
            if embedding.dim() == 2:
                if embedding.shape[0] <= 0:
                    raise ValueError("music flamingo returned an empty embedding matrix")
                embedding = embedding.mean(dim=0)
            elif embedding.dim() != 1:
                raise ValueError(f"unexpected music flamingo pooled embedding shape: {tuple(embedding.shape)}")
            outputs.append(embedding.detach().to(torch.float32).cpu())
        if not outputs:
            return []
        return _tensor_rows_to_unit_vectors(torch.stack(outputs, dim=0))

    def _load_music_flamingo_model(self):
        if self._music_flamingo_model is None:
            self._music_flamingo_model = MusicFlamingoAudioEmbedder(
                model_ref=self.config.flamingo_model_ref,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
                max_audio_seconds=self.config.max_audio_seconds,
            )
        return self._music_flamingo_model

    def _load_muq_audio_model(self):
        if self._audio_model is None:
            from muq import MuQ

            kwargs = {}
            if self.config.cache_dir:
                kwargs["cache_dir"] = self.config.cache_dir
            self._audio_model = self._load_muq_module(
                lambda: MuQ.from_pretrained(self.config.audio_model_ref, **kwargs)
            )
            self._audio_model.eval()
        return self._audio_model

    def _load_muq_mulan_model(self):
        if self._mulan_model is None:
            from muq import MuQMuLan

            kwargs = {}
            if self.config.cache_dir:
                kwargs["cache_dir"] = self.config.cache_dir
            self._mulan_model = self._load_muq_module(
                lambda: MuQMuLan.from_pretrained(self.config.mulan_model_ref, **kwargs)
            )
            self._mulan_model.eval()
        return self._mulan_model

    def _load_muq_module(self, factory):
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        return factory().to(self.device, dtype=dtype)


def _chunked(values: Sequence[T], size: int) -> list[Sequence[T]]:
    if size <= 0:
        return [values]
    return [values[index : index + size] for index in range(0, len(values), size)]


def _pad_waveforms(waveforms: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(waveform.shape[0] for waveform in waveforms)
    padded = torch.zeros((len(waveforms), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(waveforms), max_len), dtype=torch.long)
    for index, waveform in enumerate(waveforms):
        length = waveform.shape[0]
        padded[index, :length] = waveform.to(torch.float32)
        attention_mask[index, :length] = 1
    return padded, attention_mask


def _pool_muq_hidden_states(model, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    feature_mask = None
    conformer = getattr(getattr(model, "model", None), "conformer", None)
    if conformer is not None and hasattr(conformer, "_get_feature_vector_attention_mask"):
        feature_mask = conformer._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
    if feature_mask is None:
        feature_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
    feature_mask = feature_mask.to(hidden_states.device).bool()
    masked = hidden_states * feature_mask.unsqueeze(-1)
    counts = feature_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return masked.sum(dim=1) / counts


def _normalize_dimensions(latents: torch.Tensor, dimensions: int) -> torch.Tensor:
    if dimensions <= 0:
        return latents
    if latents.shape[-1] == dimensions:
        return latents
    if latents.shape[-1] > dimensions:
        return latents[..., :dimensions]
    raise RuntimeError(f"embedding dimension too small: expected {dimensions} got {latents.shape[-1]}")


def _tensor_rows_to_unit_vectors(rows: torch.Tensor) -> list[list[float]]:
    norms = torch.linalg.vector_norm(rows, dim=1, keepdim=True)
    safe = torch.where(norms > 0, rows / norms, rows)
    return [[float(value) for value in row.tolist()] for row in safe]


def _module_device_and_dtype(module) -> tuple[torch.device, torch.dtype]:
    if hasattr(module, "parameters"):
        for parameter in module.parameters():
            return parameter.device, parameter.dtype
    return torch.device("cpu"), torch.float32


def _pool_music_flamingo_audio_tokens(projected_embeddings: torch.Tensor, input_features_mask: torch.Tensor) -> torch.Tensor:
    if projected_embeddings.dim() != 3:
        raise ValueError(f"unexpected music flamingo embedding shape: {tuple(projected_embeddings.shape)}")
    if input_features_mask.dim() != 2:
        raise ValueError(f"unexpected music flamingo mask shape: {tuple(input_features_mask.shape)}")
    post_lengths = ((input_features_mask.sum(-1) - 2) // 2 + 1).clamp_min(1)
    valid_mask = torch.arange(projected_embeddings.shape[1], device=projected_embeddings.device)[None, :] < post_lengths[
        :, None
    ]
    masked = projected_embeddings * valid_mask.unsqueeze(-1)
    counts = valid_mask.sum(dim=1).clamp_min(1).unsqueeze(-1)
    return masked.sum(dim=1) / counts


def _load_music_flamingo_state_dicts(snapshot_path: Path, *, model_ref: str) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    prefixes = ("audio_tower.", "multi_modal_projector.")
    audio_state: dict[str, torch.Tensor] = {}
    projector_state: dict[str, torch.Tensor] = {}
    try:
        index_path = snapshot_path / "model.safetensors.index.json"
        if not index_path.exists():
            raise OSError(f"missing model.safetensors.index.json in cached snapshot for {model_ref}")
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shard_to_keys: dict[str, list[str]] = {}
        for key, shard_name in weight_map.items():
            if key.startswith(prefixes):
                shard_to_keys.setdefault(shard_name, []).append(key)
        if not shard_to_keys:
            raise RuntimeError(f"no audio tower weights found for music flamingo model {model_ref}")
        for shard_name, keys in shard_to_keys.items():
            shard_path = snapshot_path / shard_name
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in keys:
                    tensor = handle.get_tensor(key)
                    if key.startswith(prefixes[0]):
                        audio_state[key.removeprefix(prefixes[0])] = tensor
                    elif key.startswith(prefixes[1]):
                        projector_state[key.removeprefix(prefixes[1])] = tensor
        return audio_state, projector_state
    except OSError:
        model_path = snapshot_path / "model.safetensors"
        with safe_open(model_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefixes[0]):
                    audio_state[key.removeprefix(prefixes[0])] = handle.get_tensor(key)
                elif key.startswith(prefixes[1]):
                    projector_state[key.removeprefix(prefixes[1])] = handle.get_tensor(key)
    if not audio_state or not projector_state:
        raise RuntimeError(f"incomplete audio-only weights for music flamingo model {model_ref}")
    return audio_state, projector_state


def _resolve_cached_music_flamingo_snapshot(model_ref: str, *, cache_dir: str) -> Path:
    file_kwargs = {}
    if cache_dir:
        file_kwargs["cache_dir"] = cache_dir
    try:
        processor_config_path = cached_file(model_ref, "processor_config.json", local_files_only=True, **file_kwargs)
    except OSError:
        processor_config_path = cached_file(model_ref, "processor_config.json", **file_kwargs)
    return Path(processor_config_path).parent


class MusicFlamingoAudioEmbedder:
    def __init__(self, *, model_ref: str, device: str, cache_dir: str = "", max_audio_seconds: float = 0.0) -> None:
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        local_snapshot = _resolve_cached_music_flamingo_snapshot(model_ref, cache_dir=cache_dir)
        self.processor = AutoProcessor.from_pretrained(local_snapshot, local_files_only=True)
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.max_audio_seconds = max(0.0, float(max_audio_seconds))

        config = AutoConfig.from_pretrained(local_snapshot, local_files_only=True)
        self.audio_tower = AutoModel.from_config(config.audio_config)
        self.multi_modal_projector = AudioFlamingo3MultiModalProjector(config)

        audio_state, projector_state = _load_music_flamingo_state_dicts(local_snapshot, model_ref=model_ref)
        self.audio_tower.load_state_dict(audio_state, strict=True)
        self.multi_modal_projector.load_state_dict(projector_state, strict=True)
        self.audio_tower.to(self.device, dtype=self.dtype)
        self.multi_modal_projector.to(self.device, dtype=self.dtype)
        self.audio_tower.eval()
        self.multi_modal_projector.eval()

    def extract_embedding(self, path: str) -> torch.Tensor:
        audio, _ = _load_audio(
            path,
            sample_rate=MUSIC_FLAMINGO_SAMPLE_RATE,
            mono=True,
            max_audio_seconds=self.max_audio_seconds,
        )
        inputs = self.processor(
            text=getattr(self.processor, "audio_token", "<sound>"),
            audio=np.asarray(audio, dtype=np.float32),
            return_tensors="pt",
        )
        audio_param = next(self.audio_tower.parameters())
        audio_device = audio_param.device
        input_features = inputs["input_features"].to(audio_device, dtype=audio_param.dtype)
        input_features_mask = inputs["input_features_mask"].to(audio_device)
        with torch.inference_mode():
            audio_output = self.audio_tower(
                input_features,
                input_features_mask=input_features_mask,
                return_dict=True,
            )
            projected = self.multi_modal_projector(audio_output.last_hidden_state)
        pooled = _pool_music_flamingo_audio_tokens(projected, input_features_mask)
        if pooled.shape[0] <= 0:
            raise ValueError("music flamingo produced no pooled audio windows")
        track_embedding = pooled.mean(dim=0)
        return track_embedding.detach().to(torch.float32).cpu()


def _load_audio(path: str, *, sample_rate: int | None, mono: bool, max_audio_seconds: float) -> tuple[np.ndarray, int]:
    path_obj = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"audio file not found: {path}")
    load_kwargs = {"sr": sample_rate, "mono": mono}
    if max_audio_seconds > 0:
        load_kwargs["duration"] = max_audio_seconds
    try:
        audio, actual_sample_rate = librosa.load(path, **load_kwargs)
    except Exception as exc:
        raise RuntimeError(f"failed to load audio {path}: {exc!r}") from exc
    return np.asarray(audio, dtype=np.float32), int(actual_sample_rate)
