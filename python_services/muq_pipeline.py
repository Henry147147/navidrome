from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch

from python_services.muq_milvus import EmbeddingRow, MilvusEmbeddingStore
from python_services.muq_provider import MuQProvider
from python_services.muq_shared import TrackInfo, canonical_track_name, normalize_model_names, track_storage_key


@dataclass
class BatchProgress:
    status: str = "idle"
    models: list[str] | None = None
    clear_existing: bool = False
    total_tracks: int = 0
    processed_tracks: int = 0
    failed_tracks: int = 0
    total_operations: int = 0
    processed_operations: int = 0
    current_model: str = ""
    current_track: str = ""
    progress_percent: float = 0.0
    estimated_completion: float = 0.0
    last_error: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class BatchJobManager:
    def __init__(
        self,
        *,
        track_loader: Callable[[], list[TrackInfo]],
        provider: MuQProvider,
        store: MilvusEmbeddingStore,
    ) -> None:
        self.track_loader = track_loader
        self.provider = provider
        self.store = store
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._progress = BatchProgress()

    def start(self, *, models: Sequence[str], clear_existing: bool) -> dict[str, object]:
        normalized = normalize_model_names(models)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("batch job is already running")

            tracks = self.track_loader()
            self._cancel_event = threading.Event()
            self._progress = BatchProgress(
                status="started",
                models=list(normalized),
                clear_existing=clear_existing,
                total_tracks=len(tracks),
                total_operations=len(tracks) * len(normalized),
                started_at=time.time(),
            )
            self._thread = threading.Thread(
                target=self._run,
                args=(tracks, list(normalized), clear_existing, self._cancel_event),
                daemon=True,
            )
            self._thread.start()
            return self._progress.to_dict()

    def progress(self) -> dict[str, object]:
        with self._lock:
            return self._progress.to_dict()

    def cancel(self) -> dict[str, object]:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                return self._progress.to_dict()
            self._cancel_event.set()
            self._progress.status = "cancelling"
            return self._progress.to_dict()

    def _run(
        self,
        tracks: list[TrackInfo],
        models: list[str],
        clear_existing: bool,
        cancel_event: threading.Event,
    ) -> None:
        completed_by_track: dict[str, int] = {}
        failed_track_keys: set[str] = set()
        status = "completed"

        try:
            self.store.reset_for_run(models, clear_existing=clear_existing)
            self._update(status="running")
            for model in models:
                for track in tracks:
                    if cancel_event.is_set():
                        status = "cancelled"
                        break
                    self._update(current_model=model, current_track=canonical_track_name(track))
                    try:
                        vector = self.provider.embed_audio_files([track.full_path], model)[0]
                        self.store.upsert_embeddings(
                            model,
                            [
                                EmbeddingRow(
                                    name=track_storage_key(track),
                                    embedding=vector,
                                    model_id=model,
                                )
                            ],
                        )
                        key = track_storage_key(track)
                        completed_by_track[key] = completed_by_track.get(key, 0) + 1
                        processed_tracks = sum(1 for count in completed_by_track.values() if count >= len(models))
                        self._update(
                            processed_operations=self._progress.processed_operations + 1,
                            processed_tracks=processed_tracks,
                        )
                    except Exception as exc:
                        failed_track_keys.add(track_storage_key(track))
                        self._update(
                            processed_operations=self._progress.processed_operations + 1,
                            failed_tracks=len(failed_track_keys),
                            last_error=str(exc),
                        )
                self.store.flush(model)
                if status == "cancelled":
                    break
        except Exception as exc:
            status = "failed"
            self._update(last_error=str(exc))

        with self._lock:
            self._progress.failed_tracks = len(failed_track_keys)
            if status == "completed" and failed_track_keys:
                self._progress.status = "completed_with_errors"
            else:
                self._progress.status = status
            self._progress.current_track = ""
            self._progress.finished_at = time.time()
            self._progress.progress_percent = self._calculate_progress_percent()
            self._progress.estimated_completion = 0.0

    def _update(self, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._progress, key, value)
            self._progress.progress_percent = self._calculate_progress_percent()
            if self._progress.status in {"running", "started", "cancelling"}:
                self._progress.estimated_completion = self._calculate_eta()

    def _calculate_progress_percent(self) -> float:
        if self._progress.total_operations <= 0:
            return 0.0
        return min(100.0, (self._progress.processed_operations / self._progress.total_operations) * 100.0)

    def _calculate_eta(self) -> float:
        if self._progress.processed_operations <= 0 or self._progress.total_operations <= 0:
            return 0.0
        elapsed = max(0.001, time.time() - self._progress.started_at)
        per_operation = elapsed / self._progress.processed_operations
        remaining = self._progress.total_operations - self._progress.processed_operations
        return time.time() + (per_operation * remaining)


def save_track_embedding(output_dir: str, track: TrackInfo, model: str, embedding: Sequence[float]) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    payload_path = path / f"{_safe_filename(track_storage_key(track))}.pt"
    if payload_path.exists():
        payload = torch.load(payload_path, map_location="cpu")
    else:
        payload = {
            "track_id": track.id,
            "path": track.path,
            "name": canonical_track_name(track),
            "embeddings": {},
        }
    payload["embeddings"][model] = torch.tensor(list(embedding), dtype=torch.float32)
    torch.save(payload, payload_path)
    return str(payload_path)


def _safe_filename(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
