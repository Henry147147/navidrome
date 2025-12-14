from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union

import torch

_TRUTHY_STRINGS = {"1", "true", "yes", "on"}


@dataclass
class SongEmbedding:
    name: str
    embedding: Union[torch.Tensor, Sequence[float]]
    offset: float
    model_id: str = ""
    track_id: str = ""


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
    similarity_search_enabled: bool = False
    dedup_threshold: float = 0.85

    @classmethod
    def from_payload(cls, payload: Optional[dict]) -> "UploadSettings":
        if not isinstance(payload, dict):
            return cls()

        defaults = cls()

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

        return cls(
            similarity_search_enabled=_coerce_bool(
                payload.get("similaritySearchEnabled")
            ),
            dedup_threshold=dedup_threshold,
        )
