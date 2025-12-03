import logging
import types

import pytest

import description_pipeline
import embedding_models


class DummySchema:
    def __init__(self, auto_id=False, enable_dynamic_field=False):
        self.auto_id = auto_id
        self.enable_dynamic_field = enable_dynamic_field
        self.fields = []

    def add_field(self, name, dtype, **kwargs):
        self.fields.append((name, dtype, kwargs))


class DummyIndexParams:
    def __init__(self):
        self.indexes = []

    def add_index(self, field_name, index_type, metric_type=None, params=None):
        self.indexes.append(
            {
                "field_name": field_name,
                "index_type": index_type,
                "metric_type": metric_type,
                "params": params or {},
            }
        )


class DummyMilvusClient:
    def __init__(self, collections=None, indexes=None):
        self.collections = set(collections or [])
        self.indexes = indexes or {}
        self.created_collections = []
        self.created_indexes = []

    def list_collections(self):
        return list(self.collections)

    def create_collection(self, name, schema):
        self.collections.add(name)
        self.created_collections.append((name, schema))

    def describe_collection(self, name):
        return {"indexes": self.indexes.get(name, [])}

    def create_index(self, name, params):
        self.created_indexes.append({"name": name, "params": params})

    # Static helpers mirrored from pymilvus.MilvusClient
    @staticmethod
    def create_schema(auto_id=False, enable_dynamic_field=False):
        return DummySchema(auto_id=auto_id, enable_dynamic_field=enable_dynamic_field)

    @staticmethod
    def prepare_index_params():
        return DummyIndexParams()


@pytest.fixture()
def pipeline(monkeypatch):
    # Avoid loading heavy models by bypassing __init__
    pipe = description_pipeline.DescriptionEmbeddingPipeline.__new__(
        description_pipeline.DescriptionEmbeddingPipeline
    )
    pipe.logger = logging.getLogger("navidrome.tests.description")
    return pipe


def test_ensure_schema_creates_when_missing(monkeypatch, pipeline):
    monkeypatch.setattr(description_pipeline, "MilvusClient", DummyMilvusClient)
    client = description_pipeline.MilvusClient()

    pipeline.ensure_milvus_schemas(client)

    assert "description_embedding" in client.collections
    assert client.created_collections, "Schema should be created when missing"
    schema = client.created_collections[0][1]
    # Expect five fields: name, description, embedding, offset, model_id
    assert len(schema.fields) == 5


def test_ensure_index_respects_lite(monkeypatch, pipeline):
    monkeypatch.setattr(description_pipeline, "MilvusClient", DummyMilvusClient)
    # Force lite path
    monkeypatch.setattr(embedding_models, "_milvus_uses_lite", lambda: True)
    client = description_pipeline.MilvusClient(indexes={})

    pipeline.ensure_milvus_index(client)

    assert client.created_indexes, "Index should be created for empty collection"
    params = client.created_indexes[0]["params"]
    assert any(
        idx["index_type"] == "IVF_FLAT" for idx in params.indexes
    ), "Lite mode should use IVF_FLAT index"


def test_ensure_index_skips_when_present(monkeypatch, pipeline):
    monkeypatch.setattr(description_pipeline, "MilvusClient", DummyMilvusClient)
    # Force non-lite path so HNSW would be chosen if needed
    monkeypatch.setattr(embedding_models, "_milvus_uses_lite", lambda: False)
    existing = {
        "description_embedding": [
            {"field_name": "embedding"},
            {"field_name": "name"},
        ]
    }
    client = description_pipeline.MilvusClient(indexes=existing)

    pipeline.ensure_milvus_index(client)

    assert client.created_indexes == [], "Existing indexes should short-circuit creation"
