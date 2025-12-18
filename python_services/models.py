from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch


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
