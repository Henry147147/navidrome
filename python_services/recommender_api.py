"""FastAPI service that exposes playlist recommendation endpoints."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable

from fastapi import FastAPI, HTTPException
import uvicorn
from pymilvus import MilvusClient

from database_query import MilvusSimilaritySearcher
from milvus_schema import MilvusSchemaManager
from schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)


LOGGER = logging.getLogger("navidrome.recommender")


class RecommendationEngine:
    """Lightweight wrapper around Milvus similarity search queries."""

    def __init__(self, searcher: MilvusSimilaritySearcher) -> None:
        self.searcher = searcher
        self.logger = LOGGER

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        if not request.seeds:
            return RecommendationResponse(
                warnings=["No seeds provided; unable to compute recommendations."],
            )

        seed_ids = [seed.track_id for seed in request.seeds]
        seed_embeddings = self.searcher.get_embeddings_by_name(seed_ids)
        if not seed_embeddings:
            return RecommendationResponse(
                warnings=[
                    "No matching embeddings found for supplied seeds; returning empty playlist."
                ]
            )

        candidate_scores: Dict[str, float] = {}
        candidate_reason: Dict[str, str] = {}
        exclude: set[str] = set(request.exclude_track_ids) | set(seed_ids)
        max_hits = max(request.limit * 3, self.searcher.default_top_k)

        for seed in request.seeds:
            embedding = seed_embeddings.get(seed.track_id)
            if embedding is None:
                self.logger.debug("Skipping seed without embedding: %s", seed.track_id)
                continue

            hits = self.searcher.search_similar_embeddings(
                embedding,
                top_k=max_hits,
                exclude_names=exclude,
            )

            for hit in hits:
                candidate_id = str(hit.get("track_id") or hit.get("name") or "").strip()
                if not candidate_id or candidate_id in exclude:
                    continue
                distance = float(hit.get("distance", 0.0))
                score = max(distance, 0.0) * max(seed.weight, 0.0)

                # Apply a mild diversification penalty the more times we revisit the same candidate
                if request.diversity > 0.0:
                    score *= 1.0 - min(request.diversity, 0.9)

                if score <= 0.0:
                    continue

                previous = candidate_scores.get(candidate_id, 0.0)
                if score > previous:
                    candidate_scores[candidate_id] = score
                    candidate_reason[candidate_id] = f"seed:{seed.source}"

        if not candidate_scores:
            self.logger.debug(
                "No similarity hits; falling back to seed ordering for mode %s",
                request.mode,
            )
            fallback = RecommendationResponse()
            seen: set[str] = set()
            for seed in request.seeds:
                if seed.track_id in seen or seed.track_id in exclude:
                    continue
                seen.add(seed.track_id)
                fallback.tracks.append(
                    RecommendationItem(
                        track_id=seed.track_id,
                        score=max(seed.weight, 0.0),
                        reason=f"seed:{seed.source}",
                    )
                )
                if len(fallback.tracks) >= request.limit:
                    break
            if fallback.tracks:
                fallback.warnings.append(
                    "Returned seeds because vector search produced no candidates."
                )
                return fallback
            return RecommendationResponse(
                warnings=[
                    "Similarity search did not yield any candidates; returning empty playlist."
                ]
            )

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        response = RecommendationResponse()
        for track_id, score in ranked[: request.limit]:
            response.tracks.append(
                RecommendationItem(
                    track_id=track_id,
                    score=score,
                    reason=candidate_reason.get(track_id),
                )
            )

        return response


def build_engine() -> RecommendationEngine:
    uri = os.getenv("NAVIDROME_MILVUS_URI", "http://localhost:19530")
    client = MilvusClient(uri=uri)
    schema_manager = MilvusSchemaManager(client=client, logger=LOGGER)
    track_id_supported = schema_manager.ensure_track_id_field("embedding")
    searcher = MilvusSimilaritySearcher(
        client=client,
        logger=LOGGER,
        schema_manager=schema_manager,
    )
    searcher.set_track_id_field_available(track_id_supported)
    return RecommendationEngine(searcher)


def create_app() -> FastAPI:
    app = FastAPI(title="Navidrome Recommender", version="1.0.0")
    engine = build_engine()

    @app.post("/playlist/{mode}", response_model=RecommendationResponse)
    def recommend_playlist(
        mode: str, payload: RecommendationRequest
    ) -> RecommendationResponse:
        normalized_mode = mode.strip().lower()
        if not normalized_mode:
            raise HTTPException(status_code=400, detail="Mode must be supplied")
        request = payload.copy(update={"mode": normalized_mode})
        return engine.recommend(request)

    @app.get("/healthz")
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()


__all__ = ["app", "create_app", "RecommendationEngine"]


if __name__ == "__main__":
    port_env = os.getenv("NAVIDROME_RECOMMENDER_PORT", "9002")
    try:
        port = int(port_env)
    except ValueError:
        port = 9002
    uvicorn.run(app, host="127.0.0.1", port=port)
