from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from python_services.muq_shared import (
    COLLECTION_MUQ_AUDIO,
    COLLECTION_MUQ_MULAN,
    LEGACY_COLLECTIONS,
    MODEL_MUQ_AUDIO,
    MODEL_MUQ_MULAN,
    default_milvus_uri,
    model_collection_name,
    model_dimension,
    normalize_model_name,
)


@dataclass(frozen=True)
class EmbeddingRow:
    name: str
    embedding: Sequence[float]
    model_id: str
    offset: float = 0.0


class PymilvusBackend:
    def __init__(self, uri: str) -> None:
        self.uri = uri
        self._connected = False

    def has_collection(self, name: str) -> bool:
        utility = self._utility()
        return bool(utility.has_collection(name))

    def drop_collection(self, name: str) -> None:
        utility = self._utility()
        if utility.has_collection(name):
            utility.drop_collection(name)

    def ensure_collection(self, name: str, dimension: int) -> None:
        collection = self._collection(name)
        if collection is not None:
            current_dim = _collection_embedding_dim(collection)
            if current_dim == dimension:
                collection.load()
                return
            self.drop_collection(name)

        pymilvus = self._pymilvus()
        schema = pymilvus.CollectionSchema(
            [
                pymilvus.FieldSchema(
                    name="name",
                    dtype=pymilvus.DataType.VARCHAR,
                    is_primary=True,
                    max_length=512,
                ),
                pymilvus.FieldSchema(name="embedding", dtype=pymilvus.DataType.FLOAT_VECTOR, dim=dimension),
                pymilvus.FieldSchema(name="offset", dtype=pymilvus.DataType.FLOAT),
                pymilvus.FieldSchema(name="model_id", dtype=pymilvus.DataType.VARCHAR, max_length=256),
            ],
            auto_id=False,
        )
        collection = pymilvus.Collection(name, schema)
        collection.create_index(
            "embedding",
            {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 50, "efConstruction": 250},
            },
        )
        collection.load()

    def upsert_rows(self, name: str, rows: Sequence[EmbeddingRow]) -> None:
        collection = self._collection(name)
        if collection is None:
            raise RuntimeError(f"collection does not exist: {name}")
        collection.upsert(
            [
                {
                    "name": row.name,
                    "embedding": list(row.embedding),
                    "offset": row.offset,
                    "model_id": row.model_id,
                }
                for row in rows
            ]
        )

    def flush(self, name: str) -> None:
        collection = self._collection(name)
        if collection is not None:
            collection.flush()

    def _connect(self) -> None:
        if self._connected:
            return
        import pymilvus

        pymilvus.connections.connect(alias="default", uri=self.uri)
        self._connected = True

    def _pymilvus(self):
        import pymilvus

        return pymilvus

    def _utility(self):
        import pymilvus

        self._connect()
        return pymilvus.utility

    def _collection(self, name: str):
        utility = self._utility()
        if not utility.has_collection(name):
            return None
        import pymilvus

        return pymilvus.Collection(name)


class MilvusEmbeddingStore:
    def __init__(
        self,
        uri: str | None = None,
        *,
        backend: Any | None = None,
        dimensions: dict[str, int] | None = None,
    ) -> None:
        self.uri = uri or default_milvus_uri()
        self.backend = backend or PymilvusBackend(self.uri)
        self.dimensions = {
            MODEL_MUQ_AUDIO: model_dimension(MODEL_MUQ_AUDIO),
            MODEL_MUQ_MULAN: model_dimension(MODEL_MUQ_MULAN),
        }
        if dimensions:
            self.dimensions.update(dimensions)

    def reset_for_run(self, models: Sequence[str], *, clear_existing: bool) -> None:
        canonical_models = [normalize_model_name(model) for model in models]
        if clear_existing:
            seen: set[str] = set()
            for model in canonical_models:
                collection_name = model_collection_name(model)
                if collection_name in seen:
                    continue
                seen.add(collection_name)
                self.backend.drop_collection(collection_name)
            for collection_name in LEGACY_COLLECTIONS:
                self.backend.drop_collection(collection_name)
        for model in canonical_models:
            self.backend.ensure_collection(model_collection_name(model), self.dimensions[model])

    def upsert_embeddings(self, model: str, rows: Sequence[EmbeddingRow]) -> None:
        canonical = normalize_model_name(model)
        expected_dim = self.dimensions[canonical]
        for row in rows:
            if len(row.embedding) != expected_dim:
                raise ValueError(
                    f"{canonical} embedding dimension mismatch: expected {expected_dim} got {len(row.embedding)}"
                )
        self.backend.ensure_collection(model_collection_name(canonical), expected_dim)
        self.backend.upsert_rows(model_collection_name(canonical), rows)

    def flush(self, model: str) -> None:
        self.backend.flush(model_collection_name(model))


def _collection_embedding_dim(collection: Any) -> int | None:
    for field in collection.schema.fields:
        if field.name != "embedding":
            continue
        raw_dim = field.params.get("dim")
        if raw_dim is None:
            return None
        return int(raw_dim)
    return None
