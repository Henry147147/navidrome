"""
Helpers for performing Milvus similarity lookups used during uploads.

This module centralizes the reusable database access logic that was previously
embedded inside ad-hoc scripts (see load_all.find_closest_vector). It exposes a
small helper that can be wired into the upload pipeline so that the embedding
server and upload features share the same behavior when checking for
duplicates.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pymilvus import MilvusClient

from models import SongEmbedding

logger = logging.getLogger("navidrome.database_query")

DEFAULT_COLLECTION = "embedding"
DEFAULT_TOP_K = 25
DEFAULT_MIN_EF = 64


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
                "Skipping similarity search due to incomplete payload: name=%s has_embedding=%s",
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


__all__ = [
    "MilvusSimilaritySearcher",
    "SimilarityQuery",
]
