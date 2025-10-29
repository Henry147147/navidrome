import logging
from typing import Any, Dict, List

import pytest

from cue_splitter import SplitTrack
from models import SongEmbedding, UploadSettings
from python_embed_server import EmbedSocketServer


class StubEmbeddingModel:
    def __init__(self, segments: List[Dict[str, Any]] | None = None) -> None:
        self.segments = segments or []
        self.embed_calls: List[Dict[str, Any]] = []

    def ensure_milvus_schemas(self, client) -> None:  # pragma: no cover - simple stub
        return None

    def ensure_milvus_index(self, client) -> None:  # pragma: no cover - simple stub
        return None

    def embed_music(
        self, music_file: str, music_name: str, cue_file: str | None = None
    ) -> Dict[str, Any]:
        payload = {
            "music_file": music_file,
            "cue_file": cue_file,
            "model_id": "stub-model",
            "segments": list(self.segments),
        }
        self.embed_calls.append(payload)
        return payload


class RecordingMilvusClient:
    def __init__(self) -> None:
        self.loaded: List[str] = []
        self.upsert_calls: List[Dict[str, Any]] = []
        self.flush_calls: List[str] = []

    def load_collection(self, collection_name: str) -> None:
        self.loaded.append(collection_name)

    def upsert(self, collection_name: str, payload: List[Dict[str, Any]]) -> None:
        self.upsert_calls.append({"collection": collection_name, "payload": payload})

    def flush(self, collection_name: str) -> None:
        self.flush_calls.append(collection_name)


class NoOpFeaturePipeline:
    def scan_for_dups(self, embeddings, settings):
        return []

    def rename(self, name: str, settings, *, music_file=None) -> str:
        return name


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def test_load_from_json_converts_segments():
    segment = {
        "index": 1,
        "title": "Track 01",
        "offset_seconds": 2.5,
        "duration_seconds": 10.0,
        "embedding": [0.1, 0.2, 0.3],
    }
    payload = {
        "music_file": "song.flac",
        "model_id": "stub-model",
        "segments": [segment],
    }

    songs = EmbedSocketServer.load_from_json(payload)

    assert len(songs) == 1
    song = songs[0]
    assert song.name == "Track 01"
    assert song.embedding == segment["embedding"]
    assert song.offset == pytest.approx(2.5)
    assert song.model_id == "stub-model"
    assert song.track_id  # generated hash identifier


def test_add_embedding_drops_track_id_before_upsert(logger: logging.Logger):
    milvus = RecordingMilvusClient()
    model = StubEmbeddingModel()
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=milvus,
        model=model,
    )
    server.feature_pipeline = NoOpFeaturePipeline()

    sample_song = SongEmbedding(
        name="track.flac",
        embedding=[0.5, 0.5],
        offset=0.0,
        model_id="stub-model",
        track_id="1234",
    )
    server.load_from_json = lambda embedding: [sample_song]

    settings = UploadSettings()
    server.add_embedding_to_db("track.flac", {"music_file": "track.flac"}, settings)

    assert milvus.loaded == ["embedding"]
    assert milvus.flush_calls == ["embedding"]
    assert len(milvus.upsert_calls) == 1
    call = milvus.upsert_calls[0]
    assert call["collection"] == "embedding"
    stored_payload = call["payload"][0]
    assert "track_id" not in stored_payload
    assert stored_payload["model_id"] == "stub-model"


def test_process_embedding_request_with_split(
    monkeypatch, tmp_path, logger: logging.Logger
):
    model = StubEmbeddingModel()

    def embed_music_override(music_file: str, music_name: str, cue_file=None):
        return {
            "music_file": music_file,
            "cue_file": cue_file,
            "model_id": "stub-model",
            "segments": [
                {
                    "index": 1,
                    "title": music_name,
                    "offset_seconds": 0.0,
                    "duration_seconds": 1.0,
                    "embedding": [0.1, 0.2],
                }
            ],
        }

    model.embed_music = embed_music_override  # type: ignore[assignment]

    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=RecordingMilvusClient(),
        model=model,
    )

    track_a_path = tmp_path / "TrackA.flac"
    track_a_path.write_text("", encoding="utf-8")
    track_b_path = tmp_path / "TrackB.flac"
    track_b_path.write_text("", encoding="utf-8")

    split_tracks = [
        SplitTrack(
            index=1,
            title="Track A",
            artist="Artist",
            album="Album",
            album_artist="Album Artist",
            file_path=track_a_path,
            start_seconds=0.0,
            duration_seconds=2.0,
        ),
        SplitTrack(
            index=2,
            title="Track B",
            artist="Artist",
            album="Album",
            album_artist="Album Artist",
            file_path=track_b_path,
            start_seconds=2.0,
            duration_seconds=2.0,
        ),
    ]

    monkeypatch.setattr(
        "python_embed_server.split_flac_with_cue",
        lambda *args, **kwargs: split_tracks,
    )

    payload, returned_tracks = server._process_embedding_request(
        "album.flac",
        "Album",
        str(tmp_path / "album.cue"),
    )

    assert len(payload["segments"]) == 2
    titles = [segment["title"] for segment in payload["segments"]]
    assert titles == ["Artist - Track A", "Artist - Track B"]
    assert payload["music_file"] == "album.flac"
    assert returned_tracks == split_tracks
