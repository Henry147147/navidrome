"""
Hosts the server that will create the embedding from the music file and push it to milvus client
"""

import argparse
import atexit
import base64
import json
import logging
import os
import signal
import socket
import threading
import time
from dataclasses import asdict
from hashlib import sha224
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from pymilvus import MilvusClient

from embedding_models import BaseEmbeddingModel, MuQEmbeddingModel
from description_pipeline import (
    DEFAULT_AUDIO_COLLECTION,
    DEFAULT_DESCRIPTION_COLLECTION,
    DescriptionEmbeddingPipeline,
)
from gpu_settings import is_oom_error, load_gpu_settings
from cue_splitter import SplitTrack, split_flac_with_cue
from models import SongEmbedding
from database_query import MilvusSimilaritySearcher
from track_name_resolver import TrackNameResolver
from batch_queue_manager import BatchQueueManager, BatchResult
from batch_processor import ModelFirstBatchProcessor


SOCKET_PATH = "/tmp/navidrome_embed.sock"

# Batch processing configuration
BATCH_TIMEOUT_SECONDS = float(os.getenv("NAVIDROME_BATCH_TIMEOUT", "5.0"))
BATCH_SIZE_THRESHOLD = int(os.getenv("NAVIDROME_BATCH_SIZE", "50"))
BATCH_MODE_ENABLED = os.getenv("NAVIDROME_BATCH_MODE", "true").lower() in (
    "true",
    "1",
    "yes",
)


logger = logging.getLogger("navidrome.embed_server")


def _status_timeout_seconds() -> float:
    raw = os.getenv("NAVIDROME_EMBED_STATUS_TIMEOUT", "")
    if not raw:
        return 5.0
    try:
        return max(0.5, float(raw))
    except ValueError:
        return 5.0


def _configure_logging(verbose: bool) -> str:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return "debug" if verbose else "info"


class EmbedAudioRequest(BaseModel):
    music_file: str = Field(..., description="Absolute path to the uploaded audio file")
    name: str = Field(..., description="Original filename for metadata")
    cue_file: Optional[str] = Field(None, description="Optional cuesheet path")
    music_data_b64: Optional[str] = Field(
        None,
        description="Base64-encoded audio contents (used when the file path is not reachable)",
    )
    cue_data_b64: Optional[str] = Field(
        None,
        description="Base64-encoded cuesheet contents (used when the file path is not reachable)",
    )
    track_id: Optional[str] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    album: Optional[str] = None
    alternate_names: List[str] = Field(default_factory=list)


class EmbedAudioResponse(BaseModel):
    status: str
    duplicates: List[str] = Field(default_factory=list)
    allDuplicates: bool = False
    splitFiles: Optional[List[dict]] = None


class EmbedStatusRequest(BaseModel):
    track_id: Optional[str] = None
    artist: Optional[str] = None
    title: Optional[str] = None
    album: Optional[str] = None
    alternate_names: List[str] = Field(default_factory=list)


class EmbedStatusResponse(BaseModel):
    embedded: bool
    hasDescription: bool
    hasAudioEmbedding: bool
    name: str


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
        self.status_timeout = _status_timeout_seconds()
        self._shutdown_event = threading.Event()

        # Initialize batch processing if enabled
        self._batch_mode_enabled = BATCH_MODE_ENABLED
        self._batch_processor: Optional[ModelFirstBatchProcessor] = None
        self._batch_queue: Optional[BatchQueueManager] = None
        if self._batch_mode_enabled:
            self._batch_processor = ModelFirstBatchProcessor(
                muq_model=self.model,
                description_pipeline=self.description_pipeline,
                logger=self.logger,
            )
            self._batch_queue = BatchQueueManager(
                batch_timeout_seconds=BATCH_TIMEOUT_SECONDS,
                batch_size_threshold=BATCH_SIZE_THRESHOLD,
                process_batch_fn=self._process_embedding_batch,
                logger=self.logger,
            )
            self.logger.info(
                "Batch mode enabled: timeout=%.1fs, threshold=%d",
                BATCH_TIMEOUT_SECONDS,
                BATCH_SIZE_THRESHOLD,
            )

        # Register cleanup and signal handlers
        atexit.register(self._cleanup_socket)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        self.logger.debug(
            "EmbedSocketServer initialized for %s",
            self.socket_path,
        )

    @staticmethod
    def _canonical_name_from_meta(artist: str, title: str, fallback: str) -> str:
        artist = (artist or "").strip()
        title = (title or "").strip()
        if artist or title:
            return TrackNameResolver.canonical_name(artist, title)
        return (fallback or "").strip()

    @staticmethod
    def _using_milvus_lite() -> bool:
        if os.getenv("NAVIDROME_MILVUS_DB_PATH"):
            return True
        uri = os.getenv("NAVIDROME_MILVUS_URI")
        if not uri:
            return False
        return uri.startswith("file:")

    def _milvus_name_exists(self, collection: str, names: List[str]) -> bool:
        normalized = sorted({(name or "").strip() for name in names if name})
        if not normalized:
            return False
        start = time.monotonic()
        self.logger.debug(
            "Status check: loading collection=%s names=%d timeout=%.1fs",
            collection,
            len(normalized),
            self.status_timeout,
        )

        try:
            self.milvus_client.load_collection(collection, timeout=self.status_timeout)
        except Exception:
            self.logger.exception("Failed to load collection %s", collection)
            return False

        try:
            if self._using_milvus_lite():
                filter_expr = f"name in {json.dumps(normalized)}"
                rows = self.milvus_client.query(
                    collection_name=collection,
                    filter=filter_expr,
                    output_fields=["name"],
                    timeout=self.status_timeout,
                )
            else:
                rows = self.milvus_client.query(
                    collection_name=collection,
                    filter="name in {names}",
                    filter_params={"names": normalized},
                    output_fields=["name"],
                    timeout=self.status_timeout,
                )
        except Exception:
            self.logger.exception("Milvus query failed for %s", collection)
            return False
        finally:
            elapsed = time.monotonic() - start
            self.logger.debug(
                "Status check: collection=%s elapsed=%.3fs",
                collection,
                elapsed,
            )

        return bool(rows)

    def check_embedding_status(
        self,
        *,
        track_id: Optional[str],
        artist: Optional[str],
        title: Optional[str],
        alternate_names: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        canonical_name = self._canonical_name_from_meta(artist or "", title or "", "")
        names = {canonical_name} if canonical_name else set()
        for alt in alternate_names or []:
            alt_name = str(alt or "").strip()
            if alt_name:
                names.add(alt_name)

        name_list = sorted(names)
        if not name_list:
            return {
                "embedded": False,
                "hasDescription": False,
                "hasAudioEmbedding": False,
                "name": canonical_name,
            }

        self.logger.debug(
            "Checking embedding status track_id=%s names=%d",
            track_id,
            len(name_list),
        )
        embedded = self._milvus_name_exists("embedding", name_list)
        has_description = self._milvus_name_exists(
            DEFAULT_DESCRIPTION_COLLECTION, name_list
        )
        has_audio_embedding = self._milvus_name_exists(
            DEFAULT_AUDIO_COLLECTION, name_list
        )
        self.logger.debug(
            "Status result track_id=%s embedded=%s hasDescription=%s hasAudio=%s",
            track_id,
            embedded,
            has_description,
            has_audio_embedding,
        )
        return {
            "embedded": embedded,
            "hasDescription": has_description,
            "hasAudioEmbedding": has_audio_embedding,
            "name": canonical_name,
        }

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
            payload: Incoming JSON payload containing music_file, name, optional cue_file,
                     optional base64 file contents, and settings.

        Returns:
            Response payload mirroring the legacy socket server structure.
        """

        music_file = payload.get("music_file")
        if not music_file:
            raise ValueError("music_file is required")

        track_id = payload.get("track_id")
        cue_file = payload.get("cue_file") or None
        artist = str(payload.get("artist") or "")
        title = str(payload.get("title") or "")
        base_name = payload.get("name") or Path(str(music_file)).name
        alt_names = payload.get("alternate_names") or []

        music_file = str(music_file)
        base_name = str(base_name)
        if cue_file:
            cue_file = str(cue_file)

        music_name = self._canonical_name_from_meta(artist, title, base_name)

        self.logger.debug(
            "Processing embed payload track_id=%s music=%s cue=%s",
            track_id,
            music_file,
            cue_file,
        )

        summary: Optional[dict] = None
        split_tracks: List[SplitTrack] = []
        temp_paths: List[Path] = []

        try:
            music_file = self._ensure_local_file(
                music_file,
                payload.get("music_data_b64"),
                suggested_name=music_name,
                created_paths=temp_paths,
            )
            if cue_file:
                cue_file = self._ensure_local_file(
                    cue_file,
                    payload.get("cue_data_b64"),
                    suggested_name=cue_file,
                    created_paths=temp_paths,
                )

            result, split_tracks = self._process_embedding_request(
                music_file,
                music_name,
                cue_file,
            )
            summary = self.add_embedding_to_db(music_name, result)
        finally:
            for path in temp_paths:
                try:
                    path.unlink(missing_ok=True)
                    if path.parent.name.startswith("navidrome-embed-"):
                        # Remove the temporary directory if it's empty
                        path.parent.rmdir()
                except Exception:
                    # Best-effort cleanup; don't fail the request because of cleanup issues
                    self.logger.debug(
                        "Failed to clean up temp file %s", path, exc_info=True
                    )

        response_payload: Dict[str, object] = {"status": "ok"}
        if isinstance(summary, dict):
            response_payload.update(summary)
        if split_tracks:
            response_payload["splitFiles"] = [
                track.to_response() for track in split_tracks
            ]
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
        )
        desc_payload = (
            self.description_pipeline.prepare_payload(music_file, music_name)
            if self.description_pipeline
            else None
        )
        payload["descriptions"] = (
            desc_payload.get("descriptions", []) if desc_payload else []
        )
        payload["cue_file"] = None
        return payload, []

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info("Received signal %d, shutting down embedding server", signum)
        self._shutdown_event.set()

    def _cleanup_socket(self):
        """Clean up socket file"""
        try:
            if os.path.exists(self.socket_path):
                os.unlink(self.socket_path)
                self.logger.debug("Cleaned up socket %s", self.socket_path)
        except Exception as e:
            self.logger.debug("Error during socket cleanup: %s", e)

    def _process_embedding_batch(self, payloads: List[dict]) -> Dict[str, BatchResult]:
        """
        Process a batch of embedding requests with model-first ordering.
        Called by BatchQueueManager when a batch is ready.

        Returns results keyed by request_id.
        """
        if not self._batch_processor:
            # Should not happen, but handle gracefully
            return {
                payload.get("request_id", f"unknown-{i}"): BatchResult(
                    request_id=payload.get("request_id", f"unknown-{i}"),
                    success=False,
                    payload={
                        "status": "error",
                        "message": "Batch processor not initialized",
                    },
                    error="Batch processor not initialized",
                )
                for i, payload in enumerate(payloads)
            }

        # Process the batch with model-first ordering
        results = self._batch_processor.process_batch(payloads)

        # Store each successful result in Milvus
        for request_id, result in results.items():
            if result.success:
                try:
                    music_name = result.payload.get("music_name", "unknown")
                    self.add_embedding_to_db(music_name, result.payload)
                except Exception as e:
                    self.logger.error(
                        "Failed to store embedding for %s: %s", request_id, e
                    )
                    results[request_id] = BatchResult(
                        request_id=request_id,
                        success=False,
                        payload={
                            "status": "error",
                            "message": f"Storage failed: {e}",
                        },
                        error=str(e),
                    )

        return results

    def _health_check(self) -> Dict[str, object]:
        """Perform health check on embedding service"""
        milvus_ok = False
        try:
            # Ping Milvus to verify connection
            self.milvus_client.list_collections()
            milvus_ok = True
        except Exception as e:
            self.logger.debug("Milvus connection check failed: %s", e)

        batch_info = {}
        if self._batch_mode_enabled and self._batch_queue:
            batch_info = {
                "batch_mode": True,
                "batch_timeout": BATCH_TIMEOUT_SECONDS,
                "batch_threshold": BATCH_SIZE_THRESHOLD,
                "batch_pending": self._batch_queue.pending_count(),
                "batch_processing": self._batch_queue.is_processing(),
            }
        else:
            batch_info = {"batch_mode": False}

        return {
            "status": "ok" if milvus_ok else "degraded",
            "milvus_connected": milvus_ok,
            "descriptions_enabled": self.enable_descriptions,
            "socket_path": self.socket_path,
            "model_loaded": self.model is not None,
            **batch_info,
        }

    def serve_forever(self) -> None:
        if os.path.exists(self.socket_path):
            self.logger.debug("Removing existing socket at %s", self.socket_path)
            os.unlink(self.socket_path)

        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server_sock.bind(self.socket_path)
            os.chmod(self.socket_path, 0o600)
            server_sock.listen(1)
            server_sock.settimeout(1.0)  # Allow checking shutdown event
            self.logger.info("Embedding server listening on %s", self.socket_path)

            while not self._shutdown_event.is_set():
                try:
                    self.logger.debug("Waiting for incoming connection")
                    conn, _ = server_sock.accept()
                    self.logger.debug("Accepted new connection")
                    try:
                        self.handle_connection(conn)
                    except Exception:  # pragma: no cover - defensive logging
                        self.logger.exception("Unexpected error handling connection")
                except socket.timeout:
                    # Timeout is expected, allows checking shutdown event
                    continue
                except Exception as e:
                    if self._shutdown_event.is_set():
                        break
                    self.logger.error("Error accepting connection: %s", e)
        finally:
            # Shutdown batch queue to process remaining requests
            if self._batch_queue:
                self.logger.info("Shutting down batch queue...")
                self._batch_queue.shutdown()
            server_sock.close()
            self.logger.info("Embedding server shutdown complete")
            self._cleanup_socket()

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

    def add_embedding_to_db(self, file_name: str, embedding: dict) -> Dict[str, object]:
        self.logger.info(
            "Uploading embedding for %s to Milvus", embedding.get("music_file")
        )
        songs = self.load_from_json(embedding)
        description_rows = self._load_descriptions(embedding)
        audio_rows = self._load_audio_embeddings(embedding)
        if not songs:
            raise RuntimeError("Embedding payload did not contain any segments.")

        songs_payload = [self._serialize_song(song) for song in songs]

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

        if description_rows:
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

        if audio_rows:
            if self.description_pipeline:
                try:
                    audio_dim = len(audio_rows[0]["embedding"])
                    self.description_pipeline.ensure_milvus_schemas(
                        self.milvus_client, audio_dim=audio_dim
                    )
                    self.description_pipeline.ensure_milvus_index(self.milvus_client)
                except Exception:
                    self.logger.warning(
                        "Failed to ensure audio embedding schema/index before upsert",
                        exc_info=True,
                    )
            try:
                self.milvus_client.load_collection(DEFAULT_AUDIO_COLLECTION)
            except Exception:
                self.logger.exception(
                    "Failed to load Milvus collection %s",
                    DEFAULT_AUDIO_COLLECTION,
                )
            try:
                self.milvus_client.upsert(DEFAULT_AUDIO_COLLECTION, audio_rows)
                self.milvus_client.flush(DEFAULT_AUDIO_COLLECTION)
            except Exception:
                self.logger.exception("Failed to upsert audio embeddings to Milvus")

        self.logger.debug(
            "Prepared embedding payload for Milvus. Songs=%d",
            len(songs),
        )
        return {
            "duplicates": [],
            "allDuplicates": False,
        }

    def _serialize_song(self, song: SongEmbedding) -> Dict[str, object]:
        payload = asdict(song)
        payload.pop("track_id", None)
        return payload

    def _load_descriptions(self, embedding: dict) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for desc in embedding.get("descriptions") or []:
            vector = (
                desc.get("embedding")
                or desc.get("description_embedding")
                or desc.get("descriptionEmbedding")
            )
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

    def _load_audio_embeddings(self, embedding: dict) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for desc in embedding.get("descriptions") or []:
            vector = desc.get("audio_embedding") or desc.get("audioEmbedding")
            if not vector:
                continue
            title = desc.get("title") or desc.get("index") or ""
            title = str(title).strip()
            if not title:
                continue
            rows.append(
                {
                    "name": title,
                    "embedding": vector,
                    "offset": float(desc.get("offset_seconds") or 0.0),
                    "model_id": desc.get("audio_model_id")
                    or embedding.get("caption_model_id")
                    or "",
                }
            )
        return rows

    def handle_connection(self, conn: socket.socket) -> None:
        with conn:
            reader = conn.makefile("r", encoding="utf-8")
            # Line-buffered writer to flush response promptly.
            writer = conn.makefile("w", encoding="utf-8", buffering=1)
            request_id: Optional[str] = None
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

                # Route based on action field
                action = payload.get("action", "embed")
                request_id = payload.get("request_id")
                track_id = payload.get("track_id")
                payload_bytes = len(line.encode("utf-8"))
                self.logger.debug(
                    "Received %s request via socket request_id=%s track_id=%s bytes=%d",
                    action,
                    request_id,
                    track_id,
                    payload_bytes,
                )

                start = time.monotonic()
                try:
                    if action == "health":
                        response_payload = self._health_check()
                    elif action == "status":
                        response_payload = self.check_embedding_status(
                            track_id=payload.get("track_id"),
                            artist=payload.get("artist"),
                            title=payload.get("title"),
                            alternate_names=payload.get("alternate_names"),
                        )
                    elif action == "flush":
                        # Force immediate processing of pending batch requests
                        if self._batch_queue:
                            pending = self._batch_queue.pending_count()
                            self._batch_queue.flush()
                            response_payload = {
                                "status": "ok",
                                "message": f"Flush triggered for {pending} pending requests",
                            }
                        else:
                            response_payload = {
                                "status": "ok",
                                "message": "Batch mode not enabled",
                            }
                    elif action == "embed":
                        # Use batch mode if enabled, otherwise process immediately
                        if self._batch_mode_enabled and self._batch_queue:
                            # Generate request_id if not provided
                            if not request_id:
                                request_id = f"embed-{time.monotonic_ns()}"
                                payload["request_id"] = request_id

                            # Enqueue and wait for batch result
                            self.logger.debug(
                                "Enqueueing request %s for batch processing", request_id
                            )
                            result_queue = self._batch_queue.enqueue(
                                request_id, payload
                            )

                            # Block until batch completes (connection held open)
                            result: BatchResult = result_queue.get()

                            response_payload = result.payload
                            if request_id:
                                response_payload = dict(response_payload)
                                response_payload["request_id"] = request_id
                        else:
                            # Fallback to immediate processing
                            response_payload = self.process_payload(payload)
                    else:
                        response_payload = {
                            "status": "error",
                            "message": f"Unknown action: {action}",
                        }
                except ValueError as exc:
                    self.logger.error("Invalid payload for %s: %s", action, exc)
                    self._write_response(
                        writer, {"status": "error", "message": str(exc)}
                    )
                    return
                except Exception as exc:  # pragma: no cover - propagate to client
                    self.logger.exception("Request failed for action=%s", action)
                    if is_oom_error(exc):
                        msg = (
                            "CUDA out of memory while generating embeddings; "
                            "lower the GPU cap or enable CPU offload in settings."
                        )
                    else:
                        msg = str(exc)
                    self._write_response(writer, {"status": "error", "message": msg})
                    return

                if request_id:
                    response_payload = dict(response_payload)
                    response_payload["request_id"] = request_id
                self._write_response(writer, response_payload)
                elapsed = time.monotonic() - start
                self.logger.debug(
                    "Completed %s request request_id=%s track_id=%s elapsed=%.3fs",
                    action,
                    request_id,
                    track_id,
                    elapsed,
                )
            finally:
                self.logger.debug("Closing connection")
                try:
                    writer.close()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self.logger.debug(
                        "Socket writer already closed request_id=%s",
                        request_id,
                    )
                try:
                    reader.close()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    self.logger.debug(
                        "Socket reader already closed request_id=%s",
                        request_id,
                    )

    def _write_response(self, writer, payload) -> None:
        response_text = json.dumps(payload)
        request_id = payload.get("request_id") if isinstance(payload, dict) else None
        self.logger.debug(
            "Sending response request_id=%s bytes=%d",
            request_id,
            len(response_text),
        )
        try:
            writer.write(response_text)
            writer.write("\n")
            writer.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self.logger.debug(
                "Client disconnected before response could be sent request_id=%s error=%s",
                request_id,
                exc,
            )

    def _ensure_local_file(
        self,
        path: str,
        b64_data: Optional[str],
        *,
        suggested_name: Optional[str],
        created_paths: List[Path],
    ) -> str:
        """Return a local path to the audio/cue file, materializing from base64 if needed."""

        candidate = Path(path)
        if candidate.exists():
            return str(candidate)

        if not b64_data:
            raise FileNotFoundError(
                f"File not found: {candidate}. Provide music_data_b64 or ensure the path is reachable."
            )

        try:
            data = base64.b64decode(b64_data)
        except Exception as exc:  # pragma: no cover - defensive
            raise ValueError("Invalid base64 payload for uploaded file") from exc

        suffix = Path(suggested_name or candidate.name or "upload").suffix
        temp_dir = Path(tempfile.mkdtemp(prefix="navidrome-embed-"))
        temp_path = temp_dir / f"upload{suffix}"
        temp_path.write_bytes(data)
        created_paths.append(temp_path)
        self.logger.debug(
            "Materialized uploaded file to %s because original path %s was not accessible",
            temp_path,
            path,
        )
        return str(temp_path)


def build_embed_router(server: Optional[EmbedSocketServer] = None) -> APIRouter:
    """Create an APIRouter exposing the embedding endpoints."""

    embed_server = server or EmbedSocketServer()
    router = APIRouter()

    @router.post("/embed/audio", response_model=EmbedAudioResponse)
    def embed_audio(request: EmbedAudioRequest) -> Dict[str, object]:
        try:
            embed_server.logger.debug(
                "HTTP embed audio request name=%s file=%s artist=%s title=%s",
                request.name,
                request.music_file,
                request.artist,
                request.title,
            )
            payload = request.model_dump(by_alias=True, exclude_none=True)
            return embed_server.process_payload(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # pragma: no cover - runtime safety
            embed_server.logger.exception("Embedding failed for %s", request.music_file)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.post("/embed/status", response_model=EmbedStatusResponse)
    def embed_status(request: EmbedStatusRequest) -> Dict[str, object]:
        try:
            embed_server.logger.debug(
                "HTTP embed status request track_id=%s artist=%s title=%s",
                request.track_id,
                request.artist,
                request.title,
            )
            return embed_server.check_embedding_status(
                track_id=request.track_id,
                artist=request.artist,
                title=request.title,
                alternate_names=request.alternate_names,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            embed_server.logger.exception(
                "Embedding status check failed for %s", request.title
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @router.get("/embed/status", response_model=EmbedStatusResponse)
    def embed_status_get(
        track_id: Optional[str] = None,
        artist: Optional[str] = None,
        title: Optional[str] = None,
        album: Optional[str] = None,
        alternate_names: Optional[str] = Query(
            default=None,
            description="Comma-separated alternate names",
        ),
    ) -> Dict[str, object]:
        try:
            embed_server.logger.debug(
                "HTTP embed status GET track_id=%s artist=%s title=%s",
                track_id,
                artist,
                title,
            )
            names: List[str] = []
            if alternate_names:
                names = [
                    name.strip() for name in alternate_names.split(",") if name.strip()
                ]
            return embed_server.check_embedding_status(
                track_id=track_id,
                artist=artist,
                title=title,
                alternate_names=names,
            )
        except Exception as exc:  # pragma: no cover - runtime safety
            embed_server.logger.exception("Embedding status check failed for %s", title)
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
    parser = argparse.ArgumentParser(description="Navidrome Embedding Service")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args, _unknown = parser.parse_known_args()
    uvicorn_log_level = _configure_logging(args.verbose)
    app = FastAPI(title="Navidrome Embedding Service", version="2.0.0")
    app.include_router(build_embed_router())

    import uvicorn

    port = int(os.getenv("NAVIDROME_EMBED_PORT", "9004"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level=uvicorn_log_level)


if __name__ == "__main__":
    main()
