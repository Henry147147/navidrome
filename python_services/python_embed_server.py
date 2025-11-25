"""
Hosts the server that will create the embedding from the music file and push it to milvus client
"""

import json
import logging
import os
import socket
from dataclasses import asdict
from hashlib import sha224
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from embedding_models import BaseEmbeddingModel, MuQEmbeddingModel
from description_pipeline import (
    DEFAULT_DESCRIPTION_COLLECTION,
    DescriptionEmbeddingPipeline,
)
from gpu_settings import is_oom_error, load_gpu_settings
from cue_splitter import SplitTrack, split_flac_with_cue
from models import SongEmbedding
from upload_features import UploadFeaturePipeline, UploadSettings
from database_query import MilvusSimilaritySearcher


SOCKET_PATH = "/tmp/navidrome_embed.sock"


logger = logging.getLogger("navidrome.embed_server")


class EmbedAudioRequest(BaseModel):
    music_file: str = Field(..., description="Absolute path to the uploaded audio file")
    name: str = Field(..., description="Original filename for metadata")
    cue_file: Optional[str] = Field(None, description="Optional cuesheet path")
    settings: Optional[dict] = Field(None, description="Upload settings payload")


class EmbedAudioResponse(BaseModel):
    status: str
    duplicates: List[str] = Field(default_factory=list)
    renamedFile: Optional[str] = None
    allDuplicates: bool = False
    splitFiles: Optional[List[dict]] = None


class EmbedSocketServer:
    def __init__(
        self,
        socket_path: str = SOCKET_PATH,
        *,
        milvus_client: Optional[MilvusClient] = None,
        model: Optional[BaseEmbeddingModel] = None,
        enable_descriptions: Optional[bool] = None,
    ) -> None:
        self.socket_path = socket_path
        self.logger = logger
        self.milvus_client = milvus_client or MilvusClient("http://localhost:19530")
        self.model = model or MuQEmbeddingModel(logger=self.logger)
        self.gpu_settings = load_gpu_settings()
        self.enable_descriptions = (
            DescriptionEmbeddingPipeline is not None
            if enable_descriptions is None
            else enable_descriptions
        )
        self.description_pipeline: Optional[DescriptionEmbeddingPipeline] = None
        if self.enable_descriptions:
            try:
                self.description_pipeline = DescriptionEmbeddingPipeline(
                    logger=self.logger,
                    gpu_settings=self.gpu_settings,
                )
            except Exception:
                self.logger.warning(
                    "Description pipeline unavailable; continuing without descriptions"
                )
                self.enable_descriptions = False
        self._prepare_milvus()
        self.similarity_searcher = MilvusSimilaritySearcher(
            self.milvus_client,
            logger=self.logger,
        )
        self.feature_pipeline = UploadFeaturePipeline(
            similarity_searcher=self.similarity_searcher,
            logger=self.logger,
        )
        self.logger.debug(
            "EmbedSocketServer initialized for %s",
            self.socket_path,
        )

    def _prepare_milvus(self) -> None:
        try:
            self.model.ensure_milvus_schemas(self.milvus_client)
            self.model.ensure_milvus_index(self.milvus_client)
            if self.description_pipeline:
                self.description_pipeline.ensure_milvus_schemas(self.milvus_client)
                self.description_pipeline.ensure_milvus_index(self.milvus_client)
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to ensure Milvus schema or index")

    def _combine_payloads(
        self,
        payloads: List[dict],
        desc_payloads: List[dict],
        *,
        music_file: str,
    ) -> dict:
        combined: dict = {}
        segments: List[dict] = []
        descriptions: List[dict] = []
        for payload in payloads:
            if not combined:
                combined = dict(payload)
                combined["segments"] = []
                combined.setdefault("descriptions", [])
            segments.extend(payload.get("segments") or [])
        for payload in desc_payloads:
            descriptions.extend(payload.get("descriptions") or [])
        if not combined:
            combined = {"segments": [], "descriptions": []}
        combined["segments"] = segments
        combined["descriptions"] = descriptions
        combined["music_file"] = music_file
        combined["cue_file"] = None
        return combined

    def process_payload(self, payload: dict) -> Dict[str, object]:
        """Handle an embedding request payload (HTTP or socket).

        Args:
            payload: Incoming JSON payload containing music_file, name, optional cue_file and settings.

        Returns:
            Response payload mirroring the legacy socket server structure.
        """

        music_file = payload.get("music_file")
        if not music_file:
            raise ValueError("music_file is required")

        settings = UploadSettings.from_payload(payload.get("settings"))
        cue_file = payload.get("cue_file") or None
        music_name = payload.get("name") or Path(str(music_file)).name

        music_file = str(music_file)
        music_name = str(music_name)
        if cue_file:
            cue_file = str(cue_file)

        self.logger.debug("Processing embed payload music=%s cue=%s", music_file, cue_file)

        summary: Optional[dict] = None
        split_tracks: List[SplitTrack] = []

        result, split_tracks = self._process_embedding_request(
            music_file,
            music_name,
            cue_file,
        )
        summary = self.add_embedding_to_db(music_name, result, settings)

        response_payload: Dict[str, object] = {"status": "ok"}
        if isinstance(summary, dict):
            response_payload.update(summary)
        if split_tracks:
            response_payload["splitFiles"] = [track.to_response() for track in split_tracks]
        return response_payload

    def _process_embedding_request(
        self,
        music_file: str,
        music_name: str,
        cue_file: Optional[str],
    ) -> tuple[dict, List[SplitTrack]]:
        split_tracks: List[SplitTrack] = []
        if cue_file:
            try:
                split_tracks = split_flac_with_cue(
                    music_file,
                    cue_file,
                    output_dir=Path(music_file).parent,
                    logger=self.logger,
                )
            except Exception:  # pragma: no cover - defensive logging
                self.logger.exception(
                    "Failed to split %s using cuesheet %s", music_file, cue_file
                )
                split_tracks = []

        if split_tracks:
            payloads: List[dict] = []
            desc_payloads: List[dict] = []
            for track in split_tracks:
                try:
                    payload = self.model.embed_music(
                        str(track.file_path),
                        track.canonical_name(),
                        cue_file=None,
                    )
                    desc_payload = None
                    if self.description_pipeline:
                        desc_payload = self.description_pipeline.prepare_payload(
                            str(track.file_path), track.canonical_name()
                        )
                except Exception:
                    self.logger.exception(
                        "Embedding failed for split track %s", track.file_path
                    )
                    continue
                payloads.append(payload)
                if desc_payload:
                    desc_payloads.append(desc_payload)
            if payloads:
                combined = self._combine_payloads(
                    payloads, desc_payloads, music_file=music_file
                )
                return combined, split_tracks
            self.logger.warning(
                "All split tracks failed to embed; falling back to original file."
            )
            split_tracks = []

        payload = self.model.embed_music(
            music_file,
            music_name,
            cue_file=None,
        )
        desc_payload = (
            self.description_pipeline.prepare_payload(music_file, music_name)
            if self.description_pipeline
            else None
        )
        payload["descriptions"] = desc_payload.get("descriptions", []) if desc_payload else []
        payload["cue_file"] = None
        return payload, []

    def serve_forever(self) -> None:
        if os.path.exists(self.socket_path):
            self.logger.debug("Removing existing socket at %s", self.socket_path)
            os.unlink(self.socket_path)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server_sock.bind(self.socket_path)
            os.chmod(self.socket_path, 0o600)
            server_sock.listen(1)
            self.logger.info("Embedding server listening on %s", self.socket_path)

            while True:
                self.logger.debug("Waiting for incoming connection")
                conn, _ = server_sock.accept()
                self.logger.debug("Accepted new connection")
                try:
                    self.handle_connection(conn)
                except Exception:  # pragma: no cover - defensive logging
                    self.logger.exception("Unexpected error handling connection")
        finally:
            server_sock.close()
            self.logger.debug("Server socket closed")
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass

    @staticmethod
    def load_from_json(data: dict) -> List[SongEmbedding]:
        segments = data.get("segments") or []
        model_id = str(data.get("model_id") or "")
        songs: List[SongEmbedding] = []
        for segment in segments:
            embedding = segment.get("embedding")
            if not embedding:
                continue
            name = segment.get("title") or segment.get("index")
            if not name:
                continue
            offset_seconds = float(segment.get("offset_seconds") or 0.0)
            duration_seconds = segment.get("duration_seconds")
            hash_payload = f"{name}\n{segment.get('index')}\n{offset_seconds}\n{duration_seconds}".encode(
                "utf-8"
            )
            hash_obj = sha224()
            hash_obj.update(hash_payload)
            hash_bytes = hash_obj.digest()
            track_id = np.frombuffer(hash_bytes[:8], dtype=np.int64)[0]
            songs.append(
                SongEmbedding(
                    name=str(name),
                    embedding=embedding,
                    offset=offset_seconds,
                    model_id=model_id,
                    track_id=str(track_id),
                )
            )
        return songs

    def add_embedding_to_db(
        self, file_name: str, embedding: dict, settings: UploadSettings
    ) -> Dict[str, object]:
        self.logger.info(
            "Uploading embedding for %s to Milvus", embedding.get("music_file")
        )
        self.logger.debug("Received upload settings: %s", asdict(settings))
        songs = self.load_from_json(embedding)
        description_rows = self._load_descriptions(embedding)
        if not songs:
            raise RuntimeError("Embedding payload did not contain any segments.")

        duplicates = self.feature_pipeline.scan_for_dups(songs, settings)
        songs_payload = [self._serialize_song(song) for song in songs]
        new_name = self.feature_pipeline.rename(
            file_name, settings, music_file=embedding.get("music_file")
        )

        should_upsert_audio = len(duplicates) != len(songs_payload)
        if should_upsert_audio:
            try:
                self.milvus_client.load_collection("embedding")
            except Exception:  # pragma: no cover - defensive logging
                self.logger.exception("Failed to load Milvus collection embedding")
            try:
                if songs_payload:
                    self.milvus_client.upsert("embedding", songs_payload)
                    self.milvus_client.flush("embedding")
            except Exception:
                self.logger.exception("Failed to upsert embeddings to Milvus")
        else:
            self.logger.debug("Skipping upsert because all segments are duplicates")

        if description_rows and should_upsert_audio:
            try:
                self.milvus_client.load_collection(DEFAULT_DESCRIPTION_COLLECTION)
            except Exception:
                self.logger.exception(
                    "Failed to load Milvus collection %s",
                    DEFAULT_DESCRIPTION_COLLECTION,
                )
            try:
                self.milvus_client.upsert(
                    DEFAULT_DESCRIPTION_COLLECTION, description_rows
                )
                self.milvus_client.flush(DEFAULT_DESCRIPTION_COLLECTION)
            except Exception:
                self.logger.exception(
                    "Failed to upsert description embeddings to Milvus"
                )

        self.logger.debug(
            "Prepared embedding payload for Milvus. Songs=%d, duplicates=%d",
            len(songs),
            len(duplicates),
        )
        all_duplicates = bool(songs_payload) and len(duplicates) >= len(songs_payload)
        return {
            "duplicates": duplicates,
            "renamedFile": new_name,
            "allDuplicates": all_duplicates,
        }

    def _serialize_song(self, song: SongEmbedding) -> Dict[str, object]:
        payload = asdict(song)
        payload.pop("track_id", None)
        return payload

    def _load_descriptions(self, embedding: dict) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for desc in embedding.get("descriptions") or []:
            vector = desc.get("embedding")
            if not vector:
                continue
            title = desc.get("title") or desc.get("index") or ""
            title = str(title).strip()
            if not title:
                continue
            description_text = str(desc.get("description") or "").strip()
            rows.append(
                {
                    "name": title,
                    "description": description_text,
                    "embedding": vector,
                    "offset": float(desc.get("offset_seconds") or 0.0),
                    "model_id": desc.get("model_id") or embedding.get("model_id") or "",
                }
            )
        return rows

    def handle_connection(self, conn: socket.socket) -> None:
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            writer = conn.makefile("w", encoding="utf-8")
            try:
                line = reader.readline()
                if not line:
                    self.logger.debug("Received empty request; closing connection")
                    return

                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    self.logger.error("Invalid JSON payload: %s", exc)
                    self._write_response(
                        writer, {"status": "error", "message": f"invalid json: {exc}"}
                    )
                    return

                music_file = payload.get("music_file")
                settings = UploadSettings.from_payload(payload.get("settings"))
                cue_file = payload.get("cue_file") or None
                self.logger.debug("Payload: %s", payload)
                self.logger.debug("Normalized upload settings: %s", asdict(settings))
                self.logger.debug(
                    "Received embedding request for %s (cue=%s)", music_file, cue_file
                )
                try:
                    response_payload = self.process_payload(payload)
                except ValueError as exc:
                    self.logger.error("Invalid embed payload: %s", exc)
                    self._write_response(
                        writer, {"status": "error", "message": str(exc)}
                    )
                    return
                except Exception as exc:  # pragma: no cover - propagate to client
                    self.logger.exception("Embedding failed for %s", music_file)
                    if is_oom_error(exc):
                        msg = (
                            "CUDA out of memory while generating embeddings; "
                            "lower the GPU cap or enable CPU offload in settings."
                        )
                    else:
                        msg = str(exc)
                    self._write_response(
                        writer, {"status": "error", "message": msg}
                    )
                    return

                self._write_response(writer, response_payload)
            finally:
                self.logger.debug("Closing connection")
                writer.close()
                reader.close()

    def _write_response(self, writer, payload) -> None:
        self.logger.debug("Sending response: %s", payload)
        writer.write(json.dumps(payload))
        writer.write("\n")
        writer.flush()


def build_embed_router(server: Optional[EmbedSocketServer] = None) -> APIRouter:
    """Create an APIRouter exposing the embedding endpoints."""

    embed_server = server or EmbedSocketServer()
    router = APIRouter()

    @router.post("/embed/audio", response_model=EmbedAudioResponse)
    def embed_audio(request: EmbedAudioRequest) -> Dict[str, object]:
        try:
            payload = request.model_dump(by_alias=True, exclude_none=True)
            return embed_server.process_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime safety
            embed_server.logger.exception(
                "Embedding failed for %s", request.music_file
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("/embed/health")
    def embed_health() -> Dict[str, object]:
        return {
            "status": "ok",
            "descriptions": embed_server.enable_descriptions,
            "socket_mode": False,
        }

    return router


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    app = FastAPI(title="Navidrome Embedding Service", version="2.0.0")
    app.include_router(build_embed_router())

    import uvicorn

    port = int(os.getenv("NAVIDROME_EMBED_PORT", "9004"))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
