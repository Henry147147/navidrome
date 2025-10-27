"""FastAPI service that exposes playlist recommendation endpoints."""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, Sequence

from fastapi import FastAPI, HTTPException
import uvicorn
from pymilvus import MilvusClient

from database_query import MilvusSimilaritySearcher
from schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from track_name_resolver import TrackNameResolver


LOGGER = logging.getLogger("navidrome.recommender")


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on", "debug"}


class RecommendationEngine:
    """Lightweight wrapper around Milvus similarity search queries."""

    def __init__(
        self,
        searcher: MilvusSimilaritySearcher,
        name_resolver: TrackNameResolver,
        *,
        debug_logging: bool = False,
    ) -> None:
        self.searcher = searcher
        self.logger = LOGGER
        self.name_resolver = name_resolver
        self.debug_logging = debug_logging

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        if self.debug_logging:
            self.logger.debug(
                "Processing recommendation request user=%s mode=%s seeds=%d limit=%d exclude=%d",
                request.user_id,
                request.mode,
                len(request.seeds),
                request.limit,
                len(request.exclude_track_ids),
            )
        if not request.seeds:
            return RecommendationResponse(
                warnings=["No seeds provided; unable to compute recommendations."],
            )

        seed_ids = [seed.track_id for seed in request.seeds]
        id_to_name = self.name_resolver.ids_to_names(seed_ids)
        unique_names = list({name for name in id_to_name.values()})
        name_embeddings = self.searcher.get_embeddings_by_name(unique_names)
        seed_embeddings: Dict[str, Sequence[float]] = {}
        for track_id, name in id_to_name.items():
            vector = name_embeddings.get(name)
            if vector is not None:
                seed_embeddings[track_id] = vector

        if self.debug_logging:
            missing_name = [track_id for track_id in seed_ids if track_id not in id_to_name]
            if missing_name:
                self.logger.debug("No canonical name for %d seeds", len(missing_name))
            unresolved = [track_id for track_id in seed_ids if track_id not in seed_embeddings]
            self.logger.debug(
                "Loaded embeddings for %d/%d seeds",
                len(seed_embeddings),
                len(seed_ids),
            )
            if unresolved:
                self.logger.debug("Seeds without embeddings: %s", unresolved[:20])

        if not seed_embeddings:
            return RecommendationResponse(
                warnings=[
                    "No matching embeddings found for supplied seeds; returning empty playlist."
                ]
            )

        candidate_scores: Dict[str, float] = {}
        candidate_reason: Dict[str, str] = {}
        exclude_ids: set[str] = set(request.exclude_track_ids) | set(seed_ids)
        exclude_name_map = self.name_resolver.ids_to_names(exclude_ids)
        exclude_names = set(exclude_name_map.values())
        max_hits = max(request.limit * 3, self.searcher.default_top_k)

        for seed in request.seeds:
            embedding = seed_embeddings.get(seed.track_id)
            if embedding is None:
                self.logger.debug("Skipping seed without embedding: %s", seed.track_id)
                continue

            hits = self.searcher.search_similar_embeddings(
                embedding,
                top_k=max_hits,
                exclude_names=exclude_names,
            )

            if self.debug_logging:
                self.logger.debug(
                    "Seed %s returned %d hits", seed.track_id, len(hits)
                )

            for hit in hits:
                raw_name = str(hit.get("name") or "").strip()
                candidate_id = hit.get("track_id")
                if candidate_id:
                    candidate_id = str(candidate_id).strip()
                else:
                    candidate_id = self.name_resolver.name_to_id(raw_name)
                if not candidate_id or candidate_id in exclude_ids:
                    if self.debug_logging and raw_name and not candidate_id:
                        self.logger.debug("No track id mapping for candidate name: %s", raw_name)
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
                if seed.track_id in seen or seed.track_id in exclude_ids:
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

        if self.debug_logging:
            self.logger.debug(
                "Returning %d recommendations (top score %.4f)",
                len(response.tracks),
                response.tracks[0].score if response.tracks else 0.0,
            )

        return response


def build_engine() -> RecommendationEngine:
    uri = os.getenv("NAVIDROME_MILVUS_URI", "http://localhost:19530")
    client = MilvusClient(uri=uri)
    debug_logging = _env_flag("NAVIDROME_RECOMMENDER_DEBUG")
    # TODO: remove this line
    debug_logging = True
    if debug_logging:
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Debug logging enabled for recommender engine")
    searcher = MilvusSimilaritySearcher(
        client=client,
        logger=LOGGER,
    )
    resolver = TrackNameResolver()
    return RecommendationEngine(
        searcher,
        resolver,
        debug_logging=debug_logging,
    )


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
        request = payload.model_copy(update={"mode": normalized_mode})
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
