from dataclasses import dataclass
from typing import Any, Optional, Set

import torch

_TRUTHY_STRINGS: Set[str] = {"1", "true", "yes", "on"}
_REASONING_LEVELS: Set[str] = {"none", "low", "medium", "high", "default"}

@dataclass
class SongEmbedding:
    name: str
    embedding: torch.FloatTensor
    window: int
    hop: int
    sample_rate: int
    offset: float
    chunk_ids: list


@dataclass
class ChunkedEmbedding:
    id: int
    parent_id: str
    start_seconds: float
    end_seconds: float
    embedding: torch.FloatTensor
@dataclass
class TrackSegment:
    index: int
    title: str
    start: float
    end: Optional[float]

    @property
    def duration(self) -> Optional[float]:
        if self.end is None:
            return None
        return max(self.end - self.start, 0.0)
    


@dataclass
class UploadSettings:
    rename_enabled: bool = False
    renaming_prompt: str = ""
    openai_endpoint: str = ""
    openai_model: str = ""
    similarity_search_enabled: bool = False
    dedup_threshold: float = 0.85
    reasoning_level: str = "default"

    @classmethod
    def from_payload(cls, payload: Optional[dict]) -> "UploadSettings":
        if not isinstance(payload, dict):
            return cls()

        defaults = cls()

        def _coerce_string(value: Any) -> str:
            if value is None:
                return ""
            return str(value).strip()

        def _coerce_bool(value: Any) -> bool:
            if isinstance(value, str):
                return value.strip().lower() in _TRUTHY_STRINGS
            return bool(value)

        raw_threshold = payload.get("dedupThreshold", defaults.dedup_threshold)
        try:
            dedup_threshold = float(raw_threshold)
        except (TypeError, ValueError):
            dedup_threshold = defaults.dedup_threshold
        else:
            dedup_threshold = min(max(dedup_threshold, 0.0), 1.0)

        raw_reasoning = payload.get("reasoningLevel", defaults.reasoning_level)
        reasoning = cls._normalize_reasoning(raw_reasoning, defaults.reasoning_level)

        return cls(
            rename_enabled=_coerce_bool(payload.get("renameEnabled")),
            renaming_prompt=_coerce_string(payload.get("renamingPrompt")),
            openai_endpoint=_coerce_string(payload.get("openAiEndpoint")),
            openai_model=_coerce_string(payload.get("openAiModel")),
            similarity_search_enabled=_coerce_bool(
                payload.get("similaritySearchEnabled")
            ),
            dedup_threshold=dedup_threshold,
            reasoning_level=reasoning,
        )

    @staticmethod
    def _normalize_reasoning(value: Any, default: str) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if not normalized:
                return default
            if normalized in _REASONING_LEVELS:
                return normalized
            if normalized in {"off", "disabled"}:
                return "none"
            if normalized in {"standard", "normal"}:
                return "default"
        return default