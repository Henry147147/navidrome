from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence, TypeVar

import torch
import torchaudio

from python_services.muq_shared import (
    AUDIO_SAMPLE_RATE,
    MODEL_MUQ_AUDIO,
    MODEL_MUQ_MULAN,
    default_device,
    default_muq_audio_ref,
    default_muq_cache_dir,
    default_muq_mulan_ref,
    normalize_model_name,
)

T = TypeVar("T")


@dataclass(frozen=True)
class MuQProviderConfig:
    audio_model_ref: str = default_muq_audio_ref()
    mulan_model_ref: str = default_muq_mulan_ref()
    cache_dir: str = default_muq_cache_dir()
    device: str = default_device()
    audio_batch_size: int = 4
    text_batch_size: int = 16

    @classmethod
    def from_env(cls) -> "MuQProviderConfig":
        return cls(
            audio_model_ref=default_muq_audio_ref(),
            mulan_model_ref=default_muq_mulan_ref(),
            cache_dir=default_muq_cache_dir(),
            device=default_device(),
            audio_batch_size=max(1, int(float(os.environ.get("MUQ_AUDIO_BATCH_SIZE", "4")))),
            text_batch_size=max(1, int(float(os.environ.get("MUQ_TEXT_BATCH_SIZE", "16")))),
        )


class MuQProvider:
    def __init__(self, config: MuQProviderConfig | None = None) -> None:
        self.config = config or MuQProviderConfig.from_env()
        self.device = torch.device(self.config.device)
        self._audio_model = None
        self._mulan_model = None

    def load_audio_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(path)
        if waveform.dim() != 2:
            raise ValueError(f"expected audio tensor with 2 dims, got {tuple(waveform.shape)}")
        mono = waveform.to(torch.float32).mean(dim=0)
        if sample_rate != AUDIO_SAMPLE_RATE:
            mono = torchaudio.functional.resample(mono, sample_rate, AUDIO_SAMPLE_RATE)
        return mono.contiguous()

    def embed_audio_files(self, paths: Sequence[str], model: str) -> list[list[float]]:
        waveforms = [self.load_audio_file(path) for path in paths]
        return self.embed_audio_waveforms(waveforms, model)

    def embed_audio_waveforms(self, waveforms: Sequence[torch.Tensor], model: str) -> list[list[float]]:
        if not waveforms:
            return []

        canonical = normalize_model_name(model)
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

    def _embed_with_muq_audio(self, waveforms: Sequence[torch.Tensor]) -> list[list[float]]:
        model = self._load_muq_audio_model()
        rows: list[list[float]] = []
        for batch in _chunked(waveforms, self.config.audio_batch_size):
            padded, attention_mask = _pad_waveforms(batch)
            with torch.inference_mode():
                output = model(
                    padded.to(self.device, dtype=torch.float32),
                    attention_mask=attention_mask.to(self.device),
                    output_hidden_states=True,
                )
            pooled = _pool_muq_hidden_states(model, output.last_hidden_state, attention_mask.to(self.device))
            rows.extend(_tensor_rows_to_unit_vectors(pooled.cpu()))
        return rows

    def _embed_with_muq_mulan_audio(self, waveforms: Sequence[torch.Tensor]) -> list[list[float]]:
        mulan = self._load_muq_mulan_model()
        outputs: list[torch.Tensor] = []
        for batch in _chunked(waveforms, self.config.audio_batch_size):
            for waveform in batch:
                with torch.inference_mode():
                    latent = mulan(
                        wavs=waveform.unsqueeze(0).to(self.device, dtype=torch.float32),
                        parallel_processing=False,
                    )
                outputs.append(latent.squeeze(0).detach().to(torch.float32).cpu())
        if not outputs:
            return []
        return _tensor_rows_to_unit_vectors(torch.stack(outputs, dim=0))

    def _load_muq_audio_model(self):
        if self._audio_model is None:
            from muq import MuQ

            kwargs = {}
            if self.config.cache_dir:
                kwargs["cache_dir"] = self.config.cache_dir
            self._audio_model = MuQ.from_pretrained(self.config.audio_model_ref, **kwargs).to(
                self.device,
                dtype=torch.float32,
            )
            self._audio_model.eval()
        return self._audio_model

    def _load_muq_mulan_model(self):
        if self._mulan_model is None:
            from muq import MuQMuLan

            kwargs = {}
            if self.config.cache_dir:
                kwargs["cache_dir"] = self.config.cache_dir
            self._mulan_model = MuQMuLan.from_pretrained(self.config.mulan_model_ref, **kwargs).to(
                self.device,
                dtype=torch.float32,
            )
            self._mulan_model.eval()
        return self._mulan_model


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
