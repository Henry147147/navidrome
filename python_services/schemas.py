"""Shared request/response schemas for Navidrome Python services."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

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
    embedding: Optional[List[float]] = Field(
        None, description="Direct embedding override (for text embeddings)"
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

    # Multi-model support
    models: List[str] = Field(
        default_factory=lambda: ["muq"],
        description="Embedding models to use (muq, mert, latent)"
    )
    merge_strategy: str = Field(
        default="union",
        description="How to combine multi-model results (union, intersection, priority)"
    )
    model_priorities: Dict[str, int] = Field(
        default_factory=lambda: {"muq": 1, "mert": 2, "latent": 3},
        description="Priority order for models (lower = higher priority)"
    )
    min_model_agreement: int = Field(
        default=1,
        ge=1,
        description="Minimum number of models that must agree (for union bounds)"
    )

    # Negative prompting
    negative_prompts: List[str] = Field(
        default_factory=list,
        description="Text descriptions of music to avoid"
    )
    negative_prompt_penalty: float = Field(
        default=0.85,
        ge=0.3,
        le=1.0,
        description="Penalty multiplier for negative prompt similarity"
    )
    negative_embeddings: Optional[Dict[str, List[List[float]]]] = Field(
        None,
        description="Pre-computed negative embeddings per model"
    )

    @validator("mode")
    def _trim_mode(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("mode cannot be empty")
        return cleaned

    @validator("models")
    def _validate_models(cls, value: List[str]) -> List[str]:
        valid_models = {"muq", "mert", "latent"}
        for model in value:
            if model not in valid_models:
                raise ValueError(f"Invalid model: {model}. Must be one of {valid_models}")
        if not value:
            raise ValueError("At least one model must be specified")
        return value

    @validator("merge_strategy")
    def _validate_merge_strategy(cls, value: str) -> str:
        valid_strategies = {"union", "intersection", "priority"}
        if value not in valid_strategies:
            raise ValueError(f"Invalid merge_strategy: {value}. Must be one of {valid_strategies}")
        return value


class RecommendationItem(BaseModel):
    """Track returned by the recommender with optional explanation."""

    track_id: str
    score: float = Field(..., description="Similarity score for the track")
    reason: Optional[str] = Field(
        None, description="Human readable justification for the recommendation"
    )
    models: Optional[List[str]] = Field(
        None, description="Models that contributed to this recommendation"
    )
    negative_similarity: Optional[float] = Field(
        None, description="Similarity to negative prompts (if applicable)"
    )


class RecommendationResponse(BaseModel):
    """Structured response produced by the recommender service."""

    tracks: List[RecommendationItem] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @property
    def track_ids(self) -> List[str]:
        return [item.track_id for item in self.tracks]
