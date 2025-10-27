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

from milvus_schema import MilvusSchemaManager
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
        schema_manager: Optional[MilvusSchemaManager] = None,
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
        self.schema_manager = schema_manager or MilvusSchemaManager(
            client=client, logger=self.logger
        )
        self._track_id_field_available: Optional[bool] = None
        self._ensure_track_id_field()

    def _build_search_params(self, top_k: int) -> Dict[str, Any]:
        params = dict(self.base_search_params)
        inner = dict(params.get("params", {}))
        inner.setdefault("ef", max(DEFAULT_MIN_EF, top_k))
        params["params"] = inner
        return params

    def _ensure_track_id_field(self) -> bool:
        if self._track_id_field_available is True:
            return True
        available = self.schema_manager.ensure_track_id_field(self.collection_name)
        self._track_id_field_available = available
        return available

    def set_track_id_field_available(self, available: bool) -> None:
        self._track_id_field_available = available

    def _output_fields(
        self,
        base_fields: Sequence[str],
        *,
        include_track_id: bool = False,
    ) -> List[str]:
        fields = list(base_fields)
        if include_track_id and self._ensure_track_id_field():
            if "track_id" not in fields:
                fields.append("track_id")
        return fields

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
                output_fields=self._output_fields(["name"]),
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
                filter_expr = "name not in {names}"
                filter_params = {"names": exclusions}

        self._load_collection()

        try:
            results = self.client.search(
                collection_name=self.collection_name,
                anns_field=self.anns_field,
                data=[vector],
                limit=limit,
                output_fields=self._output_fields(["name"], include_track_id=True),
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
            rows = self.client.query(
                collection_name=self.collection_name,
                filter="name in {names}",
                filter_params={"names": unique_names},
                output_fields=self._output_fields(
                    ["name", self.anns_field], include_track_id=True
                ),
            )
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Milvus query failed when loading embeddings")
            return {}

        embeddings: Dict[str, Sequence[float]] = {}
        for row in rows or []:
            identifier = row.get("track_id") or row.get("name")
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
