import logging
import torch
import numpy as np

import pytest

from embedding_models import (
    BaseEmbeddingModel,
    enrich_embedding,
    add_magnitude_channels,
    MuQEmbeddingModel,
    MusicLatentSpaceModel,
)


class DummyEmbeddingModel(BaseEmbeddingModel):
    def __init__(self) -> None:
        super().__init__(timeout_seconds=60, logger=logging.getLogger("dummy"))
        self.load_count = 0

    def _load_model(self):
        self.load_count += 1
        return object()

    def prepare_music(self, music_file: str, music_name: str, cue_file=None):
        return []

    def embed_music(self, music_file: str, music_name: str, cue_file=None):
        return {}

    def embed_string(self, value: str):
        return []

    def ensure_milvus_schemas(self, client) -> None:
        return None

    def ensure_milvus_index(self, client) -> None:
        return None


@pytest.fixture()
def dummy_model():
    model = DummyEmbeddingModel()
    yield model
    model.shutdown()


def test_model_session_loads_model_once(dummy_model: DummyEmbeddingModel):
    with dummy_model.model_session():
        pass
    with dummy_model.model_session():
        pass
    assert dummy_model.load_count == 1


def test_unload_model_allows_reloading(dummy_model: DummyEmbeddingModel):
    dummy_model.ensure_model_loaded()
    assert dummy_model.load_count == 1
    dummy_model.unload_model()
    dummy_model.ensure_model_loaded()
    assert dummy_model.load_count == 2


def test_enrich_embedding_shape():
    D, T = 64, 10
    embedding = torch.randn(D, T)
    enriched = enrich_embedding(embedding)
    # Should be [3*D] = 192
    assert enriched.shape == (3 * D,)
    # Should be L2 normalized
    assert torch.allclose(enriched.norm(), torch.tensor(1.0))


def test_enrich_embedding_single_time():
    D, T = 10, 1
    embedding = torch.randn(D, T)
    enriched = enrich_embedding(embedding)
    # 3*D = 30
    assert enriched.shape == (3 * D,)


def test_add_magnitude_channels():
    # x: [2,64,T]
    x = torch.randn(2, 64, 5)
    mag = add_magnitude_channels(x)
    # Should be [192, T=5]
    assert mag.shape == (192, 5)
