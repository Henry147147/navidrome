"""Shared request/response schemas for Navidrome Python services."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class RecommendationSeed(BaseModel):
    """Describes a single track used as input to the recommender."""

    track_id: str = Field(..., description="Unique identifier for the track")
    weight: float = Field(1.0, ge=0.0, description="Relative importance for scoring")
    source: str = Field(
        ..., description="Logical seed source (recent, favorite, playlist)"
    )
    played_at: Optional[datetime] = Field(
        None, description="Last time the user engaged with this track"
    )

    @validator("track_id")
    def _strip_track_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("track_id cannot be empty")
        return cleaned


class RecommendationRequest(BaseModel):
    """Payload accepted by the recommender service."""

    user_id: str = Field(..., description="Persistent Navidrome user identifier")
    user_name: str = Field(..., description="Display username for logging purposes")
    limit: int = Field(25, ge=1, le=500, description="Maximum tracks to return")
    mode: str = Field(..., description="Recommender strategy requested by the caller")
    seeds: List[RecommendationSeed] = Field(default_factory=list)
    diversity: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Degree of exploration vs exploitation (0 => exact matches)",
    )
    exclude_track_ids: List[str] = Field(
        default_factory=list, description="Track ids that must not be returned"
    )
    library_ids: List[int] = Field(
        default_factory=list,
        description="Optional subset of libraries to constrain results",
    )

    @validator("mode")
    def _trim_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("mode cannot be empty")
        return cleaned


class RecommendationItem(BaseModel):
    """Track returned by the recommender with optional explanation."""

    track_id: str
    score: float = Field(..., description="Similarity score for the track")
    reason: Optional[str] = Field(
        None, description="Human readable justification for the recommendation"
    )


class RecommendationResponse(BaseModel):
    """Structured response produced by the recommender service."""

    tracks: List[RecommendationItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def track_ids(self) -> List[str]:
        return [item.track_id for item in self.tracks]
