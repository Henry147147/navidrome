import json
import logging
import socket
import threading
from typing import Any, Dict, List

import pytest

from cue_splitter import SplitTrack
from models import SongEmbedding
from python_embed_server import EmbedSocketServer

import base64
from pathlib import Path


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
            "track_name": music_name,
            "segments": list(self.segments),
        }
        self.embed_calls.append(payload)
        return payload


class RecordingMilvusClient:
    def __init__(self) -> None:
        self.loaded: List[str] = []
        self.upsert_calls: List[Dict[str, Any]] = []
        self.flush_calls: List[str] = []
        self.query_calls: List[Dict[str, Any]] = []

    def load_collection(self, collection_name: str, *_, **__) -> None:
        self.loaded.append(collection_name)

    def upsert(self, collection_name: str, payload: List[Dict[str, Any]]) -> None:
        self.upsert_calls.append({"collection": collection_name, "payload": payload})

    def flush(self, collection_name: str) -> None:
        self.flush_calls.append(collection_name)

    def query(
        self,
        collection_name: str,
        filter: str | None = None,
        filter_params: Dict[str, Any] | None = None,
        output_fields: List[str] | None = None,
        **__,
    ):
        self.query_calls.append(
            {
                "collection": collection_name,
                "filter": filter,
                "filter_params": filter_params,
                "output_fields": output_fields,
            }
        )
        names: List[str] = []
        if filter_params and "names" in filter_params:
            names = filter_params["names"]
        elif filter:
            try:
                names = json.loads(filter.split("name in ", 1)[1])
            except Exception:
                names = []
        matches = [name for name in names if name.startswith("present")]
        return [{"name": name} for name in matches]


class PresenceMilvusClient(RecordingMilvusClient):
    def __init__(self, present: set[str]) -> None:
        super().__init__()
        self.present = set(present)

    def query(
        self,
        collection_name: str,
        filter: str | None = None,
        filter_params: Dict[str, Any] | None = None,
        output_fields: List[str] | None = None,
        **__,
    ):
        self.query_calls.append(
            {
                "collection": collection_name,
                "filter": filter,
                "filter_params": filter_params,
                "output_fields": output_fields,
            }
        )
        names: List[str] = []
        if filter_params and "names" in filter_params:
            names = filter_params["names"]
        elif filter:
            try:
                names = json.loads(filter.split("name in ", 1)[1])
            except Exception:
                names = []
        matches = [name for name in names if name in self.present]
        return [{"name": name} for name in matches]


@pytest.fixture()
def logger() -> logging.Logger:
    return logging.getLogger("navidrome.tests")


def _socket_roundtrip(server: EmbedSocketServer, payload: dict | str) -> dict:
    client_sock, server_sock = socket.socketpair()
    thread = threading.Thread(
        target=server.handle_connection,
        args=(server_sock,),
        daemon=True,
    )
    thread.start()

    try:
        if isinstance(payload, str):
            message = payload
            if not message.endswith("\n"):
                message += "\n"
        else:
            message = json.dumps(payload) + "\n"
        client_sock.sendall(message.encode("utf-8"))

        data = b""
        while not data.endswith(b"\n"):
            chunk = client_sock.recv(4096)
            if not chunk:
                break
            data += chunk
    finally:
        client_sock.close()
        thread.join(timeout=1)

    if not data:
        return {}
    return json.loads(data.decode("utf-8"))


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


def test_process_payload_uses_canonical_name(tmp_path, logger: logging.Logger):
    audio_path = tmp_path / "Song.flac"
    audio_path.write_text("", encoding="utf-8")
    model = StubEmbeddingModel()
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=RecordingMilvusClient(),
        model=model,
        enable_descriptions=False,
    )
    server.add_embedding_to_db = lambda *_args, **_kwargs: {}

    payload = {
        "music_file": str(audio_path),
        "artist": "Artist One",
        "title": "Track One",
        "name": "original-name.flac",
    }

    server.process_payload(payload)
    assert model.embed_calls, "embed_music should be invoked"
    assert model.embed_calls[0]["track_name"] == "Artist One - Track One"


def test_add_embedding_drops_track_id_before_upsert(logger: logging.Logger):
    milvus = RecordingMilvusClient()
    model = StubEmbeddingModel()
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=milvus,
        model=model,
    )

    sample_song = SongEmbedding(
        name="track.flac",
        embedding=[0.5, 0.5],
        offset=0.0,
        model_id="stub-model",
        track_id="1234",
    )
    server.load_from_json = lambda embedding: [sample_song]

    server.add_embedding_to_db("track.flac", {"music_file": "track.flac"})

    assert milvus.loaded == ["embedding"]
    assert milvus.flush_calls == ["embedding"]
    assert len(milvus.upsert_calls) == 1
    call = milvus.upsert_calls[0]
    assert call["collection"] == "embedding"
    stored_payload = call["payload"][0]
    assert "track_id" not in stored_payload
    assert stored_payload["model_id"] == "stub-model"


def test_check_embedding_status_detects_existing(monkeypatch, logger: logging.Logger):
    monkeypatch.setenv("NAVIDROME_MILVUS_URI", "http://localhost:19530")
    milvus = PresenceMilvusClient({"Artist - Track"})
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=milvus,
        model=StubEmbeddingModel(),
        enable_descriptions=False,
    )

    result = server.check_embedding_status(
        track_id="1",
        artist="Artist",
        title="Track",
        alternate_names=["extra.flac"],
    )

    assert result["embedded"] is True
    assert result["hasDescription"] is True
    assert result["hasAudioEmbedding"] is True
    assert result["name"] == "Artist - Track"
    assert milvus.loaded == [
        "embedding",
        "description_embedding",
        "flamingo_audio_embedding",
    ]


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
        enable_descriptions=False,  # keep tests fast and deterministic
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


def test_process_payload_materializes_base64_when_missing(
    tmp_path, logger: logging.Logger
):
    class PathRecordingModel(StubEmbeddingModel):
        def embed_music(
            self, music_file: str, music_name: str, cue_file: str | None = None
        ):
            assert Path(music_file).exists()
            return super().embed_music(music_file, music_name, cue_file)

    model = PathRecordingModel()
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=RecordingMilvusClient(),
        model=model,
        enable_descriptions=False,
    )
    # Avoid Milvus writes
    server.add_embedding_to_db = lambda *_args, **_kwargs: {}

    original_path = "/tmp/does-not-exist/audio.mp3"
    audio_bytes = b"test-audio"
    payload = {
        "music_file": original_path,
        "name": "audio.mp3",
        "music_data_b64": base64.b64encode(audio_bytes).decode("ascii"),
    }

    response = server.process_payload(payload)

    assert response["status"] == "ok"
    assert len(model.embed_calls) == 1
    materialized_path = model.embed_calls[0]["music_file"]
    assert materialized_path != original_path
    assert not Path(materialized_path).exists()  # cleaned up after processing


def test_socket_status_roundtrip(logger: logging.Logger):
    milvus = PresenceMilvusClient({"Artist - Track"})
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=milvus,
        model=StubEmbeddingModel(),
        enable_descriptions=False,
    )
    payload = {
        "action": "status",
        "track_id": "track-1",
        "artist": "Artist",
        "title": "Track",
        "alternate_names": ["extra.flac"],
        "request_id": "req-1",
    }

    response = _socket_roundtrip(server, payload)

    assert response["embedded"] is True
    assert response["hasDescription"] is True
    assert response["hasAudioEmbedding"] is True
    assert response["name"] == "Artist - Track"
    assert response["request_id"] == "req-1"


def test_socket_embed_roundtrip(tmp_path, logger: logging.Logger):
    audio_path = tmp_path / "song.mp3"
    audio_path.write_text("", encoding="utf-8")

    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=RecordingMilvusClient(),
        model=StubEmbeddingModel(
            segments=[
                {
                    "index": 1,
                    "title": "song.mp3",
                    "offset_seconds": 0.0,
                    "duration_seconds": 1.0,
                    "embedding": [0.1, 0.2],
                }
            ]
        ),
        enable_descriptions=False,
    )
    server.add_embedding_to_db = lambda *_args, **_kwargs: {}

    payload = {
        "action": "embed",
        "music_file": str(audio_path),
        "name": "song.mp3",
        "request_id": "req-2",
    }
    response = _socket_roundtrip(server, payload)

    assert response["status"] == "ok"
    assert response["request_id"] == "req-2"


def test_socket_invalid_json_returns_error(logger: logging.Logger):
    server = EmbedSocketServer(
        socket_path="/tmp/navidrome-test.sock",
        milvus_client=RecordingMilvusClient(),
        model=StubEmbeddingModel(),
        enable_descriptions=False,
    )

    response = _socket_roundtrip(server, "{not-json}")

    assert response["status"] == "error"
    assert "invalid json" in response["message"]
