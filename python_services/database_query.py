"""
Helpers for performing Milvus similarity lookups used during uploads.

This module centralizes the reusable database access logic that was previously
embedded inside ad-hoc scripts. It exposes a
small helper that can be wired into the upload pipeline so that the embedding
server and upload features share the same behavior when checking for
duplicates.
"""

import logging
import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
from pymilvus import MilvusClient

from models import SongEmbedding

logger = logging.getLogger("navidrome.database_query")

DEFAULT_COLLECTION = "embedding"
FLAMINGO_AUDIO_COLLECTION = "flamingo_audio_embedding"
DEFAULT_TOP_K = 25
DEFAULT_MIN_EF = 64


def _using_milvus_lite() -> bool:
    """
    Detect whether Milvus Lite is being used (local file path URI).
    Lite mode does not support parameterized filters.
    """
    if os.getenv("NAVIDROME_MILVUS_DB_PATH"):
        return True
    uri = os.getenv("NAVIDROME_MILVUS_URI")
    if not uri:
        return False
    return uri.startswith("file:")


@dataclass
class SimilarityQuery:
    """
    Lightweight payload used when interacting with Milvus.
    """

    name: str
    embedding: Sequence[float]


def _ensure_vector(vector: Any) -> List[float]:
    """
    Convert tensors / numpy arrays / generic iterables to a plain Python list.
    """
    if vector is None:
        raise ValueError("Embedding vector is required for similarity search.")

    if hasattr(vector, "detach"):
        vector = vector.detach()

    if hasattr(vector, "cpu"):
        vector = vector.cpu()

    if hasattr(vector, "numpy"):
        vector = vector.numpy()

    if hasattr(vector, "tolist"):
        raw = vector.tolist()
    else:
        raw = list(vector)

    # Normalize nested structures (e.g. NumPy scalar becomes float).
    if isinstance(raw, (float, int)):
        return [float(raw)]

    return [float(value) for value in raw]


class MilvusSimilaritySearcher:
    """
    Wraps common Milvus search patterns leveraged by the upload pipeline.
    """

    def __init__(
        self,
        client: MilvusClient,
        *,
        collection_name: str = DEFAULT_COLLECTION,
        anns_field: str = "embedding",
        base_search_params: Optional[Dict[str, Any]] = None,
        default_top_k: int = DEFAULT_TOP_K,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.anns_field = anns_field
        self.default_top_k = max(int(default_top_k), 1)
        self.logger = logger or logging.getLogger("navidrome.database_query")
        default_params = {
            "metric_type": "COSINE",
            "params": {},
        }
        if base_search_params:
            merged_params = dict(default_params)
            merged_params.update(base_search_params)
            inner = dict(default_params["params"])
            inner.update(base_search_params.get("params", {}))
            merged_params["params"] = inner
            self.base_search_params = merged_params
        else:
            self.base_search_params = default_params

    def _build_search_params(self, top_k: int) -> Dict[str, Any]:
        params = dict(self.base_search_params)
        inner = dict(params.get("params", {}))
        inner.setdefault("ef", max(DEFAULT_MIN_EF, top_k))
        params["params"] = inner
        return params

    def _load_collection(self) -> None:
        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            self.logger.exception(
                "Failed to load Milvus collection %s", self.collection_name
            )
            raise

    def find_similar(
        self,
        query: SimilarityQuery,
        *,
        top_k: Optional[int] = None,
        exclude_names: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a cosine similarity search against the configured Milvus collection.
        """
        top_limit = max(int(top_k or self.default_top_k), 1)
        vector = _ensure_vector(query.embedding)
        filter_expr: Optional[str] = None
        filter_params: Optional[Dict[str, Any]] = None
        if exclude_names:
            exclusions = sorted({name for name in exclude_names if name})
            if exclusions:
                if _using_milvus_lite():
                    # Lite cannot parse placeholders; embed the list directly.
                    filter_expr = f"name not in {json.dumps(exclusions)}"
                else:
                    filter_expr = "name not in {names}"
                    filter_params = {"names": exclusions}

        self._load_collection()

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                anns_field=self.anns_field,
                data=[vector],
                limit=top_limit,
                output_fields=["name"],
                filter=filter_expr,
                filter_params=filter_params,
                search_params=self._build_search_params(top_limit),
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception(
                "Milvus search failed for %s (top_k=%s)", query.name, top_limit
            )
            raise

        if not results:
            return []
        return results[0]

    def search_similar_embeddings(
        self,
        embedding: Sequence[float],
        *,
        top_k: Optional[int] = None,
        exclude_names: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a similarity search directly from an embedding vector."""

        vector = _ensure_vector(embedding)
        limit = max(int(top_k or self.default_top_k), 1)
        filter_expr: Optional[str] = None
        filter_params: Optional[Dict[str, Any]] = None
        if exclude_names:
            exclusions = sorted({name for name in exclude_names if name})
            if exclusions:
                if _using_milvus_lite():
                    filter_expr = f"name not in {json.dumps(exclusions)}"
                else:
                    filter_expr = "name not in {names}"
                    filter_params = {"names": exclusions}

        self._load_collection()

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                anns_field=self.anns_field,
                data=[vector],
                limit=limit,
                output_fields=["name"],
                filter=filter_expr,
                filter_params=filter_params,
                search_params=self._build_search_params(limit),
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Milvus vector search failed")
            return []

        if not results:
            return []
        return results[0]

    def get_embeddings_by_name(
        self, names: Sequence[str]
    ) -> Dict[str, Sequence[float]]:
        """Fetch stored embeddings for the specified track names."""

        unique_names = sorted({name for name in names if name})
        if not unique_names:
            return {}

        self._load_collection()

        try:
            if _using_milvus_lite():
                filter_expr = f"name in {json.dumps(unique_names)}"
                rows = self.client.query(
                    collection_name=self.collection_name,
                    filter=filter_expr,
                    output_fields=["name", self.anns_field],
                )
            else:
                rows = self.client.query(
                    collection_name=self.collection_name,
                    filter="name in {names}",
                    filter_params={"names": unique_names},
                    output_fields=["name", self.anns_field],
                )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Milvus query failed when loading embeddings")
            return {}

        embeddings: Dict[str, Sequence[float]] = {}
        for row in rows or []:
            identifier = row.get("name")
            vector = row.get(self.anns_field)
            if identifier and vector:
                embeddings[str(identifier)] = vector
        return embeddings

    def identify_duplicates(
        self,
        embedding_payload: SongEmbedding,
        threshold: float,
        *,
        top_k: Optional[int] = None,
    ) -> List[str]:
        """
        Return the names of stored embeddings whose cosine distance exceeds the
        provided threshold for the supplied payload.
        """
        name = str(embedding_payload.name).strip()
        embedding = embedding_payload.embedding
        if not name or embedding is None:
            self.logger.debug(
                "Skipping similarity search due to incomplete payload: "
                "name=%s has_embedding=%s",
                bool(name),
                embedding is not None,
            )
            return []

        try:
            hits = self.find_similar(
                SimilarityQuery(name=name, embedding=embedding),
                top_k=top_k,
                exclude_names=[name],
            )
        except Exception:
            return []

        duplicates: List[str] = []
        for hit in hits:
            distance = hit.get("distance")
            hit_name = hit.get("name")
            if distance is None or hit_name is None:
                continue
            if float(distance) >= float(threshold):
                duplicates.append(str(hit_name))

        if duplicates:
            self.logger.info(
                "Detected %d potential duplicates for %s (threshold=%.3f): %s",
                len(duplicates),
                name,
                threshold,
                ", ".join(duplicates),
            )
        else:
            self.logger.debug(
                "No duplicates detected for %s above threshold %.3f", name, threshold
            )

        return duplicates


class MultiModelSimilaritySearcher:
    """
    Searches across multiple embedding models and merges results.

    Supports three merge strategies:
    - union: Combine all results, rank by maximum similarity across models
    - intersection: Only tracks appearing in ALL model results
    - priority: Try intersection first, fall back to highest priority model
    """

    # Map model names to their Milvus collections
    COLLECTION_MAP = {
        "muq": "embedding",
        "qwen3": "description_embedding",
        "flamingo_audio": FLAMINGO_AUDIO_COLLECTION,
    }

    def __init__(
        self,
        milvus_uri: Union[str, MilvusClient],
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize multi-model searcher.

        Args:
            milvus_uri: URI for Milvus connection or MilvusClient instance
            logger: Optional logger instance
        """
        if isinstance(milvus_uri, str):
            self.client = MilvusClient(uri=milvus_uri)
        else:
            self.client = milvus_uri
        self.logger = logger or logging.getLogger("navidrome.multi_model_search")

        # Create searchers for each model
        self.searchers = {}
        for model, collection in self.COLLECTION_MAP.items():
            try:
                self.searchers[model] = MilvusSimilaritySearcher(
                    client=self.client, collection_name=collection, logger=self.logger
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize searcher for {model}: {e}")

    def search_multi_model(
        self,
        embeddings: Dict[str, Sequence[float]],
        top_k: int = 25,
        exclude_names: Optional[List[str]] = None,
        merge_strategy: str = "union",
        model_priorities: Optional[Dict[str, int]] = None,
        min_model_agreement: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Search across multiple models and merge results.

        Args:
            embeddings: Dict mapping model name to embedding vector
            top_k: Number of results to return
            exclude_names: Track names to exclude
            merge_strategy: How to combine results (union/intersection/priority)
            model_priorities: Priority order for models (lower = higher priority)
            min_model_agreement: Minimum number of models that must agree

        Returns:
            List of dicts with keys: name, distance, models
        """
        if not embeddings:
            self.logger.warning("No embeddings provided for multi-model search")
            return []

        # Run searches in parallel (sequentially for now, could be parallelized)
        all_results = {}
        for model, embedding in embeddings.items():
            if model not in self.searchers:
                self.logger.warning(f"No searcher available for model: {model}")
                continue

            try:
                results = self.searchers[model].search_similar_embeddings(
                    embedding=embedding,
                    top_k=top_k * 2,  # Get more to ensure enough after merging
                    exclude_names=exclude_names,
                )
                # Convert to dict keyed by name for easier merging
                all_results[model] = {r["name"]: r for r in results if "name" in r}
                self.logger.debug(f"Model {model} returned {len(results)} results")
            except Exception as e:
                self.logger.error(f"Search failed for model {model}: {e}")
                all_results[model] = {}

        if not all_results:
            self.logger.warning("All model searches failed")
            return []

        # Merge based on strategy
        if merge_strategy == "union":
            return self._merge_union(all_results, top_k, min_model_agreement)
        elif merge_strategy == "intersection":
            return self._merge_intersection(all_results, top_k)
        elif merge_strategy == "priority":
            return self._merge_priority(all_results, top_k, model_priorities or {})
        else:
            self.logger.error(f"Unknown merge strategy: {merge_strategy}")
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    def _merge_union(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        top_k: int,
        min_model_agreement: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Union: Combine all results, rank by maximum similarity across models.

        Args:
            all_results: Dict mapping model -> track_name -> result
            top_k: Number of results to return
            min_model_agreement: Minimum number of models that must agree

        Returns:
            Merged and ranked results
        """
        track_scores = {}  # track_name -> {scores, models, data}

        for model, results in all_results.items():
            for track_name, data in results.items():
                distance = data.get("distance", 0.0)

                if track_name not in track_scores:
                    track_scores[track_name] = {
                        "scores": [distance],
                        "models": [model],
                        "data": data,
                    }
                else:
                    # Add score for averaging
                    track_scores[track_name]["scores"].append(distance)
                    track_scores[track_name]["models"].append(model)

        # Calculate average scores
        for track_name, info in track_scores.items():
            info["avg_score"] = np.mean(info["scores"])

        # Filter by minimum model agreement
        if min_model_agreement > 1:
            track_scores = {
                name: info
                for name, info in track_scores.items()
                if len(info["models"]) >= min_model_agreement
            }

        # Sort by average score and return top-k
        sorted_tracks = sorted(
            track_scores.items(), key=lambda x: x[1]["avg_score"], reverse=True
        )

        return [
            {
                "track_name": name,
                "score": info["avg_score"],
                "models": info["models"],
                "entity": info["data"].get("entity", {}),
            }
            for name, info in sorted_tracks[:top_k]
        ]

    def _merge_intersection(
        self, all_results: Dict[str, Dict[str, Dict]], top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Intersection: Only tracks appearing in ALL model results.

        Args:
            all_results: Dict mapping model -> track_name -> result
            top_k: Number of results to return

        Returns:
            Merged and ranked results (only common tracks)
        """
        if not all_results:
            return []

        # Find tracks present in all models
        all_tracks = [set(results.keys()) for results in all_results.values()]
        common_tracks = set.intersection(*all_tracks) if all_tracks else set()

        if not common_tracks:
            self.logger.debug("No tracks found in all models (empty intersection)")
            return []

        # Score by average similarity across models
        track_scores = []
        for track_name in common_tracks:
            distances = [
                all_results[model][track_name].get("distance", 0.0)
                for model in all_results.keys()
            ]
            avg_score = np.mean(distances)
            track_scores.append(
                (
                    track_name,
                    avg_score,
                    list(all_results.keys()),
                    all_results[next(iter(all_results))][track_name],
                )
            )

        # Sort by average score
        track_scores.sort(key=lambda x: x[1], reverse=True)

        return [
            {
                "track_name": name,
                "score": score,
                "models": models,
                "entity": data.get("entity", {}),
            }
            for name, score, models, data in track_scores[:top_k]
        ]

    def _merge_priority(
        self,
        all_results: Dict[str, Dict[str, Dict]],
        top_k: int,
        priorities: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        """
        Priority: Try intersection first, fall back to highest priority model.

        Args:
            all_results: Dict mapping model -> track_name -> result
            top_k: Number of results to return
            priorities: Dict mapping model -> priority (lower = higher priority)

        Returns:
            Merged and ranked results
        """
        # Try intersection first
        intersection_results = self._merge_intersection(all_results, top_k)

        if len(intersection_results) >= top_k:
            self.logger.debug(
                f"Using intersection results ({len(intersection_results)} tracks)"
            )
            return intersection_results

        # Fall back to highest priority model
        if not priorities:
            priorities = {"muq": 1, "qwen3": 2}

        sorted_models = sorted(
            [
                (model, priority)
                for model, priority in priorities.items()
                if model in all_results
            ],
            key=lambda x: x[1],
        )

        if not sorted_models:
            self.logger.warning("No priority models available")
            return []

        primary_model = sorted_models[0][0]
        self.logger.debug(f"Falling back to priority model: {primary_model}")

        primary_results = all_results[primary_model]

        # Sort by distance
        sorted_results = sorted(
            primary_results.items(),
            key=lambda x: x[1].get("distance", 0.0),
            reverse=True,
        )

        return [
            {
                "track_name": name,
                "score": data.get("distance", 0.0),
                "models": [primary_model],
                "entity": data.get("entity", {}),
            }
            for name, data in sorted_results[:top_k]
        ]

    def get_embeddings_by_name(
        self, names: Sequence[str], model: str = "muq"
    ) -> Dict[str, Sequence[float]]:
        """
        Fetch stored embeddings for specified track names from a specific model.

        Args:
            names: Track names to fetch
            model: Model to fetch from (muq, qwen3, flamingo_audio)

        Returns:
            Dict mapping track name to embedding vector
        """
        if model not in self.searchers:
            self.logger.warning(f"No searcher available for model: {model}")
            return {}

        return self.searchers[model].get_embeddings_by_name(names)


__all__ = [
    "MilvusSimilaritySearcher",
    "MultiModelSimilaritySearcher",
    "SimilarityQuery",
]
