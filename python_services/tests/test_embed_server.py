import logging
from typing import Any, Dict, List, Tuple

import pytest
import torch

import python_embed_server as embed_module
from models import SongEmbedding, UploadSettings
from python_embed_server import EmbedSocketServer


class DummyThread:
    def start(self) -> None:
        return None


class DummyClassifierModel:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.timeout_thread = DummyThread()


class RecordingMilvusClient:
    def __init__(self) -> None:
        self.loaded: List[str] = []
        self.upsert_calls: List[Tuple[str, List[Dict[str, Any]]]] = []
        self.flush_calls: List[str] = []

    def load_collection(self, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def upsert(self, collection_name: str, payload: List[Dict[str, Any]]) -> None:
        # Store a shallow copy so mutations after the call don't affect expectations.
        copied = [dict(item) for item in payload]
        self.upsert_calls.append((collection_name, copied))

    def flush(self, collection_name: str) -> None:
        self.flush_calls.append(collection_name)


class NoOpFeaturePipeline:
    def scan_for_dups(self, embeddings, settings):
        return []

    def rename(self, name: str, settings, *, music_file=None) -> str:
        return name


@pytest.fixture(autouse=True)
def stub_classifier(monkeypatch):
    monkeypatch.setattr(embed_module, "ClassifierModel", DummyClassifierModel)


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def make_song(track_id: str = "abc") -> SongEmbedding:
    return SongEmbedding(
        name="song.flac",
        embedding=torch.tensor([0.1], dtype=torch.float32),
        window=1,
        hop=1,
        sample_rate=1,
        offset=0.0,
        chunk_ids=[],
        track_id=track_id,
    )


def test_add_embedding_drops_track_id_when_schema_missing(logger: logging.Logger):
    client = RecordingMilvusClient()
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=client,
    )
    server.feature_pipeline = NoOpFeaturePipeline()
    server.load_from_json = lambda embedding: ([make_song()], [])

    settings = UploadSettings()
    server.add_embedding_to_db("song.flac", {"music_file": "song.flac"}, settings)

    assert client.upsert_calls
    collection, payload = client.upsert_calls[0]
    # Ensure track_id is never written to Milvus payloads because schema disallows it.
    assert collection == "embedding"
    assert payload[0].get("track_id") is None
