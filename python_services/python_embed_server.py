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
from pymilvus import MilvusClient

from embedding_models import BaseEmbeddingModel, MuQEmbeddingModel
from cue_splitter import SplitTrack, split_flac_with_cue
from models import SongEmbedding
from upload_features import UploadFeaturePipeline, UploadSettings
from database_query import MilvusSimilaritySearcher


SOCKET_PATH = "/tmp/navidrome_embed.sock"


logger = logging.getLogger("navidrome.embed_server")


class EmbedSocketServer:
    def __init__(
        self,
        socket_path: str = SOCKET_PATH,
        *,
        milvus_client: Optional[MilvusClient] = None,
        model: Optional[BaseEmbeddingModel] = None,
    ) -> None:
        self.socket_path = socket_path
        self.logger = logger
        self.milvus_client = milvus_client or MilvusClient("http://localhost:19530")
        self.model = model or MuQEmbeddingModel(logger=self.logger)
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
        except Exception:  # pragma: no cover - defensive logging
            self.logger.exception("Failed to ensure Milvus schema or index")

    def _combine_payloads(
        self,
        payloads: List[dict],
        *,
        music_file: str,
    ) -> dict:
        combined: dict = {}
        segments: List[dict] = []
        for payload in payloads:
            if not combined:
                combined = dict(payload)
                combined["segments"] = []
            segments.extend(payload.get("segments") or [])
        if not combined:
            combined = {"segments": []}
        combined["segments"] = segments
        combined["music_file"] = music_file
        combined["cue_file"] = None
        return combined

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
            for track in split_tracks:
                try:
                    payload = self.model.embed_music(
                        str(track.file_path),
                        track.canonical_name(),
                        cue_file=None,
                    )
                except Exception:
                    self.logger.exception(
                        "Embedding failed for split track %s", track.file_path
                    )
                    continue
                payloads.append(payload)
            if payloads:
                combined = self._combine_payloads(payloads, music_file=music_file)
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
        if not songs:
            raise RuntimeError("Embedding payload did not contain any segments.")

        duplicates = self.feature_pipeline.scan_for_dups(songs, settings)
        songs_payload = [self._serialize_song(song) for song in songs]
        new_name = self.feature_pipeline.rename(
            file_name, settings, music_file=embedding.get("music_file")
        )

        if len(duplicates) != len(songs_payload):
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

                if not music_file:
                    self.logger.error("Request missing music_file field")
                    self._write_response(
                        writer,
                        {"status": "error", "message": "music_file is required"},
                    )
                    return

                music_file = str(music_file)
                music_path = Path(music_file)
                music_name = payload.get("name") or music_path.name
                music_name = str(music_name)
                if cue_file:
                    cue_file = str(cue_file)

                summary: Optional[dict] = None
                split_tracks: List[SplitTrack] = []
                try:
                    result, split_tracks = self._process_embedding_request(
                        music_file,
                        music_name,
                        cue_file,
                    )
                    summary = self.add_embedding_to_db(music_name, result, settings)
                except Exception as exc:  # pragma: no cover - propagate to client
                    self.logger.exception("Embedding failed for %s", music_file)
                    self._write_response(
                        writer, {"status": "error", "message": str(exc)}
                    )
                    return

                response_payload = {"status": "ok"}
                if isinstance(summary, dict):
                    response_payload.update(summary)
                if split_tracks:
                    response_payload["splitFiles"] = [
                        track.to_response() for track in split_tracks
                    ]
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


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    server = EmbedSocketServer()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logging.getLogger("navidrome.embed_server").info(
            "Embedding server shutting down"
        )


if __name__ == "__main__":
    main()
