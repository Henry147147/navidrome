from __future__ import annotations

from dataclasses import dataclass
import gc
import os
from typing import Sequence, TypeVar

import librosa
import numpy as np
import torch

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
        load_kwargs = {"sr": None, "mono": False}
        if self.config.max_audio_seconds > 0:
            load_kwargs["duration"] = self.config.max_audio_seconds
        waveform, sample_rate = librosa.load(path, **load_kwargs)
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
            outputs.append(_flatten_music_flamingo_embedding(embedding))
        if not outputs:
            return []
        return _tensor_rows_to_unit_vectors(torch.stack(outputs, dim=0))

    def _load_music_flamingo_model(self):
        if self._music_flamingo_model is None:
            self._music_flamingo_model = MusicFlamingoAudioEmbedder(
                model_ref=self.config.flamingo_model_ref,
                device=self.config.device,
                cache_dir=self.config.cache_dir,
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


def _flatten_music_flamingo_embedding(embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() == 0:
        raise ValueError("music flamingo embedding must have at least one dimension")
    flattened = embedding.detach().to(torch.float32).reshape(-1).cpu()
    if flattened.numel() == 0:
        raise ValueError("music flamingo embedding must not be empty")
    return flattened


class MusicFlamingoAudioEmbedder:
    def __init__(self, *, model_ref: str, device: str, cache_dir: str = "") -> None:
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir

        model_kwargs = dict(kwargs)
        if device.startswith("cuda"):
            model_kwargs["device_map"] = "auto"
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.processor = AutoProcessor.from_pretrained(model_ref, **kwargs)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_ref, **model_kwargs)
        if "device_map" not in model_kwargs:
            self.model.to(torch.device(device), dtype=torch.float32)
        self.model.eval()

    def extract_embedding(self, path: str) -> torch.Tensor:
        audio, _ = librosa.load(path, sr=MUSIC_FLAMINGO_SAMPLE_RATE, mono=True)
        inputs = self.processor(
            text=getattr(self.processor, "audio_token", "<sound>"),
            audio=np.asarray(audio, dtype=np.float32),
            return_tensors="pt",
        )
        audio_param = next(self.model.audio_tower.parameters())
        audio_device = audio_param.device
        input_features = inputs["input_features"].to(audio_device, dtype=audio_param.dtype)
        input_features_mask = inputs["input_features_mask"].to(audio_device)
        with torch.inference_mode():
            output = self.model.get_audio_features(
                input_features=input_features,
                input_features_mask=input_features_mask,
                return_dict=True,
            )
        pooled = output.pooler_output
        if pooled.dim() != 2:
            raise ValueError(f"unexpected music flamingo pooled shape: {tuple(pooled.shape)}")
        return pooled.mean(dim=0).detach().to(torch.float32).cpu()
