"""FastAPI service that exposes playlist recommendation endpoints."""

from __future__ import annotations

import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Sequence

from fastapi import APIRouter, FastAPI, HTTPException
import uvicorn
from pymilvus import MilvusClient
from pydantic import BaseModel

from database_query import MilvusSimilaritySearcher, MultiModelSimilaritySearcher
from gpu_settings import load_gpu_settings
from schemas import (
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
)
from track_name_resolver import TrackNameResolver
import numpy as np


class BatchEmbeddingRequest(BaseModel):
    """Request model for batch embedding job."""

    models: List[str] = ["muq", "qwen3"]
    clearExisting: bool = True


LOGGER = logging.getLogger("navidrome.recommender")


def _configure_logging(verbose: bool) -> str:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return "debug" if verbose else "info"


def _repo_root() -> Path:
    """Return repository root (one level above python_services)."""
    return Path(__file__).resolve().parents[1]


def _resolve_path(env_var: str, default_relative: str) -> Path:
    """Resolve a path from environment or fallback to repo root.

    Using a relative default previously created a brand new SQLite file when the
    service was started from `python_services/`, which resulted in the
    `media_file` table being missing. This helper ensures we always point to the
    real Navidrome assets shipped with the repo unless explicitly overridden via
    environment.
    """

    env_val = os.getenv(env_var)
    if env_val:
        return Path(env_val).expanduser()
    return _repo_root() / default_relative


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
        multi_model_searcher: MultiModelSimilaritySearcher = None,
        debug_logging: bool = False,
    ) -> None:
        self.searcher = searcher
        self.multi_model_searcher = multi_model_searcher
        self.logger = LOGGER
        self.name_resolver = name_resolver
        self.debug_logging = debug_logging

    def _process_hits(
        self,
        hits: list,
        seed,
        exclude_ids: set,
        diversity: float,
        candidate_scores: dict,
        candidate_reason: dict,
        candidate_models: dict,
    ) -> None:
        """Process similarity search hits and update candidate scores."""
        for hit in hits:
            raw_name = str(hit.get("name") or "").strip()
            candidate_id = hit.get("track_id")
            if candidate_id:
                candidate_id = str(candidate_id).strip()
            else:
                candidate_id = self.name_resolver.name_to_id(raw_name)

            if not candidate_id or candidate_id in exclude_ids:
                if self.debug_logging and raw_name and not candidate_id:
                    self.logger.debug(
                        "No track id mapping for candidate name: %s", raw_name
                    )
                continue

            distance = float(hit.get("distance", 0.0))
            score = max(distance, 0.0) * max(seed.weight, 0.0)

            # Apply a mild diversification penalty
            if diversity > 0.0:
                score *= 1.0 - min(diversity, 0.9)

            if score <= 0.0:
                continue

            previous = candidate_scores.get(candidate_id, 0.0)
            if score > previous:
                candidate_scores[candidate_id] = score
                candidate_reason[candidate_id] = f"seed:{seed.source}"

                # Track which models contributed to this candidate
                models = hit.get("models", [seed.source])
                if candidate_id not in candidate_models:
                    candidate_models[candidate_id] = []
                candidate_models[candidate_id].extend(models)

    def _apply_negative_prompt_penalty(
        self,
        candidate_scores: dict,
        candidate_names: dict,
        request: RecommendationRequest,
    ) -> dict:
        """Apply penalty to candidates similar to negative prompts."""
        if not request.negative_embeddings or not request.negative_prompts:
            return {}  # No negative similarities to track

        negative_similarities = {}

        # Get embeddings for all candidates
        candidate_name_list = list(candidate_names.values())

        # Use primary model for negative prompt comparison
        primary_model = request.models[0]

        if self.multi_model_searcher:
            track_embeddings = self.multi_model_searcher.get_embeddings_by_name(
                candidate_name_list, model=primary_model
            )
        else:
            track_embeddings = self.searcher.get_embeddings_by_name(candidate_name_list)

        # For each candidate, compute similarity to negative prompts
        for candidate_id, candidate_name in candidate_names.items():
            track_emb = track_embeddings.get(candidate_name)
            if track_emb is None:
                continue

            track_emb_np = np.array(track_emb)

            # Get negative embeddings for this model
            neg_embeddings_for_model = request.negative_embeddings.get(
                primary_model, []
            )

            if not neg_embeddings_for_model:
                continue

            # Compute max similarity to any negative prompt
            max_negative_sim = 0.0
            for neg_emb_list in neg_embeddings_for_model:
                neg_emb = np.array(neg_emb_list)

                # Normalize both vectors
                track_norm = np.linalg.norm(track_emb_np)
                neg_norm = np.linalg.norm(neg_emb)

                if track_norm > 0 and neg_norm > 0:
                    sim = np.dot(track_emb_np, neg_emb) / (track_norm * neg_norm)
                    max_negative_sim = max(max_negative_sim, float(sim))

            if max_negative_sim > 0:
                # Apply penalty proportional to negative similarity
                penalty_factor = 1.0 - (
                    max_negative_sim * (1.0 - request.negative_prompt_penalty)
                )
                candidate_scores[candidate_id] *= penalty_factor
                negative_similarities[candidate_id] = max_negative_sim

                if self.debug_logging:
                    self.logger.debug(
                        "Applied negative penalty to %s: sim=%.3f, factor=%.3f",
                        candidate_id,
                        max_negative_sim,
                        penalty_factor,
                    )

        return negative_similarities

    def recommend(self, request: RecommendationRequest) -> RecommendationResponse:
        if self.debug_logging:
            self.logger.debug(
                "Processing recommendation request user=%s mode=%s seeds=%d "
                "limit=%d exclude=%d models=%s",
                request.user_id,
                request.mode,
                len(request.seeds),
                request.limit,
                len(request.exclude_track_ids),
                request.models,
            )
        if not request.seeds:
            return RecommendationResponse(
                warnings=["No seeds provided; unable to compute recommendations."],
            )

        # Separate seeds with direct embeddings from those needing lookup
        direct_embedding_seeds = [s for s in request.seeds if s.embedding is not None]
        track_id_seeds = [s for s in request.seeds if s.embedding is None]

        seed_embeddings: Dict[str, Dict[str, Sequence[float]]] = {}

        # Handle direct embeddings (e.g., from text queries)
        for seed in direct_embedding_seeds:
            primary_model = request.models[0] if request.models else "muq"
            seed_embeddings[seed.track_id] = {primary_model: seed.embedding}
            if self.debug_logging:
                self.logger.debug(
                    "Using direct embedding for seed %s (dim=%d model=%s)",
                    seed.track_id,
                    len(seed.embedding),
                    primary_model,
                )

        # Handle track ID seeds (traditional path)
        if track_id_seeds:
            seed_ids = [seed.track_id for seed in track_id_seeds]
            id_to_name = self.name_resolver.ids_to_names(seed_ids)
            unique_names = list({name for name in id_to_name.values()})

            # If multi-model search is unavailable, fall back to primary model.
            if len(request.models) == 1 or self.multi_model_searcher is None:
                primary_model = request.models[0] if request.models else "muq"
                model_searcher = self.searcher
                if (
                    self.multi_model_searcher
                    and primary_model != "muq"
                    and hasattr(self.multi_model_searcher, "searchers")
                ):
                    model_searcher = self.multi_model_searcher.searchers.get(
                        primary_model, model_searcher
                    )

                name_embeddings = (
                    model_searcher.get_embeddings_by_name(unique_names)
                    if model_searcher
                    else {}
                )
                for track_id, name in id_to_name.items():
                    vector = name_embeddings.get(name)
                    if vector is not None:
                        seed_embeddings[track_id] = {primary_model: vector}
            else:
                model_embeddings: Dict[str, Dict[str, Sequence[float]]] = {}
                if self.multi_model_searcher:
                    for model in request.models:
                        model_embeddings[model] = (
                            self.multi_model_searcher.get_embeddings_by_name(
                                unique_names, model=model
                            )
                        )

                for track_id, name in id_to_name.items():
                    vectors_for_models: Dict[str, Sequence[float]] = {}
                    for model in request.models:
                        vector = model_embeddings.get(model, {}).get(name)
                        if vector is not None:
                            vectors_for_models[model] = vector
                    if vectors_for_models:
                        seed_embeddings[track_id] = vectors_for_models

        if self.debug_logging:
            if track_id_seeds:
                missing_name = [
                    s.track_id for s in track_id_seeds if s.track_id not in id_to_name
                ]
                if missing_name:
                    self.logger.debug(
                        "No canonical name for %d seeds", len(missing_name)
                    )
            all_seed_ids = [s.track_id for s in request.seeds]
            unresolved = [
                track_id for track_id in all_seed_ids if track_id not in seed_embeddings
            ]
            self.logger.debug(
                "Loaded embeddings for %d/%d seeds",
                len(seed_embeddings),
                len(all_seed_ids),
            )
            if unresolved:
                self.logger.debug("Seeds without embeddings: %s", unresolved[:20])

        if not seed_embeddings:
            return RecommendationResponse(
                warnings=[
                    "No matching embeddings found for supplied seeds; "
                    "returning empty playlist."
                ]
            )

        candidate_scores: Dict[str, float] = {}
        candidate_reason: Dict[str, str] = {}
        candidate_models: Dict[str, list] = {}  # Track which models contributed
        all_seed_ids = [s.track_id for s in request.seeds]
        exclude_ids: set[str] = set(request.exclude_track_ids) | set(all_seed_ids)
        exclude_name_map = self.name_resolver.ids_to_names(exclude_ids)
        exclude_names = list(set(exclude_name_map.values()))
        max_hits = max(request.limit * 3, self.searcher.default_top_k)

        # Use multi-model search if multiple models requested
        if len(request.models) > 1 and self.multi_model_searcher:
            # For each seed, search across all models
            for seed in request.seeds:
                embedding_map = seed_embeddings.get(seed.track_id, {})
                if not embedding_map:
                    if self.debug_logging:
                        self.logger.debug(
                            "Skipping seed without embedding: %s", seed.track_id
                        )
                    continue

                # Build embeddings dict for multi-model search using only
                # vectors that exist for each model.
                model_embeddings = {
                    model: embedding_map.get(model)
                    for model in request.models
                    if embedding_map.get(model) is not None
                }
                if not model_embeddings:
                    if self.debug_logging:
                        self.logger.debug(
                            "No embeddings available for requested models: %s",
                            request.models,
                        )
                    continue

                try:
                    hits = self.multi_model_searcher.search_multi_model(
                        embeddings=model_embeddings,
                        top_k=max_hits,
                        exclude_names=exclude_names,
                        merge_strategy=request.merge_strategy,
                        model_priorities=request.model_priorities,
                        min_model_agreement=request.min_model_agreement,
                    )

                    if self.debug_logging:
                        self.logger.debug(
                            "Seed %s returned %d hits (multi-model)",
                            seed.track_id,
                            len(hits),
                        )

                    self._process_hits(
                        hits,
                        seed,
                        exclude_ids,
                        request.diversity,
                        candidate_scores,
                        candidate_reason,
                        candidate_models,
                    )
                except Exception as e:
                    self.logger.error(
                        f"Multi-model search failed for seed {seed.track_id}: {e}"
                    )

        else:
            # Single model search (backwards compatible)
            primary_model = request.models[0] if request.models else "muq"
            model_searcher = (
                self.searcher
                if primary_model == "muq"
                else (
                    self.multi_model_searcher.searchers.get(primary_model)
                    if self.multi_model_searcher
                    else None
                )
            )

            for seed in request.seeds:
                embedding = seed_embeddings.get(seed.track_id, {}).get(primary_model)
                if embedding is None:
                    if self.debug_logging:
                        self.logger.debug(
                            "Skipping seed without embedding: %s", seed.track_id
                        )
                    continue

                if model_searcher is None:
                    if self.debug_logging:
                        self.logger.debug(
                            "No searcher configured for model %s", primary_model
                        )
                    continue

                hits = model_searcher.search_similar_embeddings(
                    embedding,
                    top_k=max_hits,
                    exclude_names=exclude_names,
                )

                if self.debug_logging:
                    self.logger.debug(
                        "Seed %s returned %d hits", seed.track_id, len(hits)
                    )

                self._process_hits(
                    hits,
                    seed,
                    exclude_ids,
                    request.diversity,
                    candidate_scores,
                    candidate_reason,
                    candidate_models,
                )

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
                    "Similarity search did not yield any candidates; "
                    "returning empty playlist."
                ]
            )

        # Apply negative prompt penalties if applicable
        negative_similarities = {}
        if request.negative_prompts and request.negative_embeddings:
            # Build candidate name mapping for penalty calculation
            candidate_ids = list(candidate_scores.keys())
            id_to_name_map = self.name_resolver.ids_to_names(candidate_ids)
            candidate_names = {
                cid: name for cid, name in id_to_name_map.items() if name
            }

            negative_similarities = self._apply_negative_prompt_penalty(
                candidate_scores, candidate_names, request
            )

            if self.debug_logging:
                self.logger.debug(
                    "Applied negative prompt penalties to %d candidates",
                    len(negative_similarities),
                )

        # Rank candidates by final score
        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        response = RecommendationResponse()
        for track_id, score in ranked[: request.limit]:
            # Get unique models that contributed to this candidate
            models = list(set(candidate_models.get(track_id, [])))

            response.tracks.append(
                RecommendationItem(
                    track_id=track_id,
                    score=score,
                    reason=candidate_reason.get(track_id),
                    models=models if models else None,
                    negative_similarity=negative_similarities.get(track_id),
                )
            )

        if self.debug_logging:
            self.logger.debug(
                "Returning %d recommendations (top score %.4f)",
                len(response.tracks),
                response.tracks[0].score if response.tracks else 0.0,
            )

        return response


def build_engine(
    *,
    milvus_client: MilvusClient | None = None,
    milvus_uri: str | None = None,
) -> RecommendationEngine:
    """
    Build a recommender engine, allowing callers to inject a Milvus client.

    Injecting the client lets the unified service point at a Milvus Lite
    database file (e.g., file:/path/to/milvus.db) instead of a remote server.
    """

    uri = milvus_uri or os.getenv("NAVIDROME_MILVUS_URI", "http://localhost:19530")
    client = milvus_client or MilvusClient(uri=uri)
    debug_logging = _env_flag("NAVIDROME_RECOMMENDER_DEBUG")
    if debug_logging:
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Debug logging enabled for recommender engine")

    # Create single-model searcher (default, backwards compatible)
    searcher = MilvusSimilaritySearcher(
        client=client,
        logger=LOGGER,
    )

    # Create multi-model searcher
    multi_model_searcher = MultiModelSimilaritySearcher(
        milvus_uri=client, logger=LOGGER
    )

    resolver = TrackNameResolver()
    return RecommendationEngine(
        searcher,
        resolver,
        multi_model_searcher=multi_model_searcher,
        debug_logging=debug_logging,
    )


def build_recommender_router(
    engine: RecommendationEngine | None = None,
) -> APIRouter:
    router = APIRouter()
    service = engine or build_engine()

    @router.post("/playlist/{mode}", response_model=RecommendationResponse)
    def recommend_playlist(
        mode: str, payload: RecommendationRequest
    ) -> RecommendationResponse:
        normalized_mode = mode.strip().lower()
        if not normalized_mode:
            raise HTTPException(status_code=400, detail="Mode must be supplied")
        request = payload.model_copy(update={"mode": normalized_mode})
        return service.recommend(request)

    @router.get("/healthz")
    def healthcheck() -> Dict[str, str]:
        return {"status": "ok"}

    # Batch job endpoints
    from batch_embedding_job import start_batch_job, get_current_job
    import threading

    @router.post("/batch/start")
    def start_batch_embedding(request: BatchEmbeddingRequest) -> Dict:
        """Start batch re-embedding job."""
        current_job = get_current_job()
        if current_job and current_job.progress.status == "running":
            raise HTTPException(400, "A job is already running")

        db_path = _resolve_path("NAVIDROME_DB_PATH", "navidrome.db")
        music_root = _resolve_path("NAVIDROME_MUSIC_ROOT", "music")

        if not db_path.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Navidrome database not found at {db_path}",
            )

        if not music_root.exists():
            raise HTTPException(
                status_code=500,
                detail=f"Music root not found at {music_root}",
            )

        job = start_batch_job(
            db_path=str(db_path),
            music_root=str(music_root),
            milvus_uri=os.getenv("NAVIDROME_MILVUS_URI", "http://localhost:19530"),
            gpu_settings=load_gpu_settings(),
        )

        # Run in background thread
        thread = threading.Thread(
            target=job.run, args=(request.models, request.clearExisting), daemon=True
        )
        thread.start()

        return {"status": "started", "job_id": id(job)}

    @router.get("/batch/progress")
    def get_batch_progress() -> Dict:
        """Get current batch job progress."""
        job = get_current_job()
        if not job:
            return {"status": "no_job"}

        progress = job.get_progress()
        return {
            "total_tracks": progress.total_tracks,
            "total_operations": progress.total_operations,
            "processed_tracks": progress.processed_tracks,
            "processed_operations": progress.processed_operations,
            "failed_tracks": progress.failed_tracks,
            "current_track": progress.current_track,
            "current_model": progress.current_model,
            "status": progress.status,
            "progress_percent": (
                progress.processed_operations / progress.total_operations * 100
                if progress.total_operations > 0
                else 0
            ),
            "estimated_completion": progress.estimated_completion,
            "last_error": progress.last_error,
        }

    @router.post("/batch/cancel")
    def cancel_batch_job() -> Dict:
        """Cancel running batch job."""
        job = get_current_job()
        if not job:
            raise HTTPException(404, "No job running")

        job.cancel()
        return {"status": "cancelling"}

    return router


def create_app() -> FastAPI:
    app = FastAPI(title="Navidrome Recommender", version="1.0.0")
    app.include_router(build_recommender_router())
    return app


# Avoid connecting to Milvus during test collection unless explicitly requested.
if os.getenv("NAVIDROME_SKIP_RECOMMENDER_APP_INIT") or "pytest" in sys.modules:
    app = FastAPI(title="Navidrome Recommender (test)")

    @app.get("/healthz")
    def health_stub():
        return {"status": "ok", "stub": True}

else:
    app = create_app()


__all__ = ["app", "create_app", "RecommendationEngine"]


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Navidrome Recommender Service")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args, _unknown = parser.parse_known_args()
    uvicorn_log_level = _configure_logging(args.verbose)

    port_env = os.getenv("NAVIDROME_RECOMMENDER_PORT", "9002")
    try:
        port = int(port_env)
    except ValueError:
        port = 9002
    uvicorn.run(app, host="127.0.0.1", port=port, log_level=uvicorn_log_level)
