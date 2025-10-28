import logging

import pytest

from embedding_models import BaseEmbeddingModel


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
