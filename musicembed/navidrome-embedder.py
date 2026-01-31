import argparse
import gc
import json
import logging
import multiprocessing as mp
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, List, Optional, Sequence

import torch
import torch.nn.functional as F
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from enrichment import enrich_and_concatenate
from model import MusicFlamingo, QwenEmbedder

VERSION = "1.0.0"
MILVUS_COLLECTION = "flamingo_audio_embedding"
MILVUS_LYRICS_COLLECTION = "lyrics_embedding"
MILVUS_DESCRIPTION_COLLECTION = "description_embedding"
MODEL_ID = "music_flamingo_enriched"
TEXT_MODEL_ID = "qwen3_embedding_4b"
MILVUS_MAX_DIM = 32768


@dataclass(frozen=True)
class TrackInfo:
    id: str
    path: str
    title: str
    artist: str
    full_path: str


@dataclass
class WorkerStats:
    processed: int
    skipped: int
    failed: int
    elapsed: float
    device: str


def get_all_music(
    inital_path: str,
    *,
    music_dir: str | None = None,
    sort_length: str | None = None,
) -> List[TrackInfo]:
    if not inital_path:
        raise ValueError("database path is required")
    if _is_postgres_uri(inital_path):
        raise ValueError("postgres databases are not supported by this script")

    order_clause = "ORDER BY id"
    if sort_length:
        normalized = sort_length.strip().lower()
        if normalized == "long":
            order_clause = "ORDER BY duration IS NULL, duration DESC"
        elif normalized == "short":
            order_clause = "ORDER BY duration IS NULL, duration ASC"
        else:
            raise ValueError(f"Invalid sort_length value: {sort_length}")

    query = (
        "SELECT id, path, COALESCE(title, ''), COALESCE(artist, '') "
        "FROM media_file WHERE path IS NOT NULL AND path != '' "
        f"{order_clause}"
    )

    with sqlite3.connect(inital_path) as conn:
        rows = conn.execute(query).fetchall()

    tracks: List[TrackInfo] = []
    for row in rows:
        track_id, path, title, artist = row
        full_path = _resolve_track_path(path, music_dir)
        tracks.append(
            TrackInfo(
                id=track_id,
                path=path,
                title=title,
                artist=artist,
                full_path=full_path,
            )
        )
    return tracks


def _is_postgres_uri(value: str) -> bool:
    value = value.lower()
    return value.startswith("postgres://") or value.startswith("postgresql://")


def _get_env_or_default(key: str, default_value: str) -> str:
    return os.environ.get(key, default_value)


def _parse_optional_bool(value: str) -> Optional[bool]:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    return None


def _resolve_model_path(relative_path: str) -> str:
    if os.path.isabs(relative_path):
        return relative_path
    if os.path.exists(relative_path):
        return relative_path
    parent_path = os.path.join("..", relative_path)
    if os.path.exists(parent_path):
        return parent_path
    return relative_path


def _default_config() -> dict[str, Any]:
    return {
        "database": {
            "path": _get_env_or_default("ND_DBPATH", "navidrome.db"),
            "type": "sqlite3",
        },
        "milvus": {
            "uri": _get_env_or_default("MILVUS_URI", "http://localhost:19530"),
            "collection": MILVUS_COLLECTION,
        },
        "models": {
            "music_flamingo": _resolve_model_path("musicembed/music_flamingo_fp8"),
            "library_path": _resolve_model_path("musicembed/llama-lib"),
            "text_model": _resolve_model_path("musicembed/models/qwen-embedder-4b.gguf"),
            "audio_model": _resolve_model_path("musicembed/models/music-flamingo.gguf"),
            "projector": _resolve_model_path("musicembed/models/mmproj-music-flamingo.gguf"),
        },
        "embedder": {
            "batch_size": 50,
            "gpu_layers": 99,
            "threads": 0,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "max_audio_seconds": 0.0,
            "audio_gpus": _get_env_or_default("MF_AUDIO_GPUS", ""),
            "text_gpu": _get_env_or_default("MF_TEXT_GPU", ""),
            "attn_impl": _get_env_or_default("MF_ATTN_IMPL", "flash_attention_2"),
            "dequantize_fp8": _parse_optional_bool(_get_env_or_default("MF_DEQUANTIZE_FP8", "0")) or False,
            "max_tracks": int(_get_env_or_default("MF_MAX_TRACKS", "0")),
            "skip_text": _get_env_or_default("MF_SKIP_TEXT", "").lower() in ("1", "true", "yes"),
        },
        "logging": {
            "level": _get_env_or_default("LOG_LEVEL", "info"),
        },
        "cli": {
            "music_dir": "./music",
            "output_dir": "./embeddings",
            "dry_run": False,
            "force": False,
            "upload_milvus": True,
            "sort_length": "",
        },
    }


def _merge_config(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_config(base[key], value)
        else:
            base[key] = value


def _normalize_config(cfg: dict[str, Any]) -> None:
    db = cfg.get("database", {})
    if not db.get("type"):
        db["type"] = "postgres" if _is_postgres_uri(db.get("path", "")) else "sqlite3"


def _resolve_track_path(path: str, music_dir: str | None) -> str:
    if os.path.isabs(path) or not music_dir:
        return path
    return os.path.join(music_dir, path)


def _canonical_track_name(track: TrackInfo) -> str:
    artist = track.artist.strip()
    title = track.title.strip()
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return title
    return track.path


def _parse_gpu_list(value: str) -> List[int]:
    if not value:
        return []
    parts = [item.strip() for item in value.split(",") if item.strip()]
    gpus: List[int] = []
    for part in parts:
        try:
            gpus.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid GPU id: {part}") from exc
    return gpus


def _available_gpus() -> List[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def _gpu_total_memory(gpu_id: int) -> int:
    props = torch.cuda.get_device_properties(gpu_id)
    return int(props.total_memory)


def _select_text_gpu(available: Sequence[int], requested: Optional[int]) -> Optional[int]:
    if not available:
        return None
    if requested is not None:
        if requested not in available:
            raise ValueError(f"Requested text GPU {requested} is not available; available={list(available)}")
        return requested
    return max(available, key=_gpu_total_memory)


def _split_tracks(tracks: Sequence[TrackInfo], shards: int) -> List[List[TrackInfo]]:
    if shards <= 1:
        return [list(tracks)]
    buckets: List[List[TrackInfo]] = [[] for _ in range(shards)]
    for idx, track in enumerate(tracks):
        buckets[idx % shards].append(track)
    return buckets


def _setup_logging(level: str) -> None:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    logging.basicConfig(
        level=level_map.get(level.lower(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _build_in_filter(field: str, values: Sequence[str]) -> str:
    encoded = ", ".join(json.dumps(value) for value in values)
    return f"{field} in [{encoded}]"


def _get_collection_embedding_dim(collection: Collection) -> Optional[int]:
    for field in collection.schema.fields:
        if field.name != "embedding":
            continue
        dim_value = field.params.get("dim")
        if dim_value is None:
            return None
        try:
            return int(dim_value)
        except (TypeError, ValueError):
            return None
    return None


def _get_collection_missing_fields(collection: Collection, required_fields: Sequence[str]) -> List[str]:
    if not required_fields:
        return []
    existing = {field.name for field in collection.schema.fields}
    return [field for field in required_fields if field not in existing]


class MilvusWriter:
    def __init__(
        self,
        uri: str,
        collection: str,
        *,
        extra_fields: Optional[Sequence[FieldSchema]] = None,
    ) -> None:
        self.uri = uri
        self.collection_name = collection
        self._connected = False
        self._collection: Collection | None = None
        self._dim: Optional[int] = None
        self._extra_fields = list(extra_fields or [])
        self._extra_field_names = [field.name for field in self._extra_fields]

    def connect(self) -> None:
        if self._connected:
            return
        connections.connect(alias="default", uri=self.uri)
        self._connected = True

    def collection_exists(self) -> bool:
        self.connect()
        return utility.has_collection(self.collection_name)

    def ensure_collection(self, dim: int) -> None:
        self.connect()
        if utility.has_collection(self.collection_name):
            existing = Collection(self.collection_name)
            existing_dim = _get_collection_embedding_dim(existing)
            missing_fields = _get_collection_missing_fields(existing, self._extra_field_names)
            dimension_mismatch = existing_dim is not None and existing_dim != dim
            if dimension_mismatch or missing_fields:
                logging.warning(
                    "Milvus collection schema mismatch; recreating %s (expected_dim=%d actual_dim=%s missing_fields=%s)",
                    self.collection_name,
                    dim,
                    existing_dim if existing_dim is not None else "unknown",
                    ", ".join(missing_fields) if missing_fields else "none",
                )
                utility.drop_collection(self.collection_name)
                self._collection = None
            else:
                self._collection = existing
                self._dim = existing_dim or dim
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="name", dtype=DataType.VARCHAR, is_primary=True, max_length=512),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="offset", dtype=DataType.FLOAT),
                FieldSchema(name="model_id", dtype=DataType.VARCHAR, max_length=256),
            ]
            fields.extend(self._extra_fields)
            schema = CollectionSchema(fields, auto_id=False)
            self._collection = Collection(self.collection_name, schema)
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 50, "efConstruction": 250},
            }
            self._collection.create_index("embedding", index_params)
            self._dim = dim
        if self._collection is None:
            self._collection = Collection(self.collection_name)
        self._collection.load()

    def existing_names(self, names: Sequence[str]) -> set[str]:
        if not names:
            return set()
        if not self.collection_exists():
            return set()
        collection = self._collection or Collection(self.collection_name)
        collection.load()
        expr = _build_in_filter("name", names)
        results = collection.query(expr, output_fields=["name"])
        return {item["name"] for item in results}

    def upsert(
        self,
        name: str,
        embedding: Sequence[float],
        model_id: str,
        *,
        extra_fields: Optional[dict[str, Any]] = None,
    ) -> None:
        dim = len(embedding)
        if self._collection is None or self._dim != dim:
            self.ensure_collection(dim)
        if self._collection is None:
            raise RuntimeError("Milvus collection is not initialized")
        if self._extra_field_names:
            if extra_fields is None:
                raise ValueError(f"Missing extra fields for {self.collection_name}: {self._extra_field_names}")
            missing = [name for name in self._extra_field_names if name not in extra_fields]
            if missing:
                raise ValueError(f"Missing required extra fields for {self.collection_name}: {missing}")
        payload: dict[str, Any] = {
            "name": name,
            "embedding": list(embedding),
            "offset": 0.0,
            "model_id": model_id,
        }
        if extra_fields:
            payload.update(extra_fields)
        self._collection.upsert([payload])
        self._collection.flush()


def _chunked(values: Sequence[str], size: int) -> List[List[str]]:
    if size <= 0:
        return [list(values)]
    return [list(values[i : i + size]) for i in range(0, len(values), size)]


def _prefetch_existing_names(milvus: MilvusWriter, names: Sequence[str], chunk_size: int = 512) -> set[str]:
    if not names:
        return set()
    existing: set[str] = set()
    for chunk in _chunked(list(names), chunk_size):
        try:
            existing.update(milvus.existing_names(chunk))
        except Exception as exc:
            logging.warning("Milvus prefetch failed (%s); continuing without skipping", exc)
            break
    return existing


def _enrich_embedding(embedding: torch.Tensor, device: str) -> torch.Tensor:
    if embedding.dim() == 3 and embedding.shape[0] == 1:
        embedding = embedding.squeeze(0)
    if embedding.dim() != 2:
        raise ValueError(f"Expected 2D embedding, got shape {tuple(embedding.shape)}")
    embedding = embedding.to(device=device, dtype=torch.float32)
    return enrich_and_concatenate(embedding)


def _prepare_milvus_embedding(embedding: torch.Tensor, max_dim: int) -> torch.Tensor:
    flat = embedding.flatten()
    if flat.numel() <= max_dim:
        return flat
    logging.warning(
        "Embedding dim %d exceeds Milvus limit %d; downsampling for Milvus.",
        flat.numel(),
        max_dim,
    )
    vec = flat.to(dtype=torch.float32).view(1, 1, -1)
    down = F.interpolate(vec, size=max_dim, mode="linear", align_corners=False)
    return down.view(-1)


def _safe_filename(filename: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in filename)


def _save_embedding(
    output_dir: str,
    track: TrackInfo,
    embedding: torch.Tensor,
    name: str,
    raw_embedding: Optional[torch.Tensor] = None,
    milvus_embedding: Optional[torch.Tensor] = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{track.id}.pt" if track.id else f"{name}.pt"
    safe_filename = _safe_filename(filename)
    output_path = os.path.join(output_dir, safe_filename)
    payload = {
        "track_id": track.id,
        "path": track.path,
        "name": name,
        "embedding": embedding.cpu(),
        "embedding_dim": int(embedding.numel()),
    }
    if raw_embedding is not None:
        payload["raw_embedding"] = raw_embedding.cpu()
        payload["raw_embedding_dim"] = tuple(raw_embedding.shape)
    if milvus_embedding is not None and milvus_embedding.numel() != embedding.numel():
        payload["milvus_embedding"] = milvus_embedding.cpu()
        payload["milvus_dim"] = int(milvus_embedding.numel())
    torch.save(payload, output_path)
    _write_track_path_file(output_path, track.full_path)
    return output_path


def _text_payload_path(output_dir: str, track: TrackInfo, name: str) -> str:
    filename = f"{track.id}.text.json" if track.id else f"{name}.text.json"
    safe_filename = _safe_filename(filename)
    return os.path.join(output_dir, safe_filename)


def _save_text_payload(
    output_dir: str,
    track: TrackInfo,
    name: str,
    description: str,
    lyrics: str,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "track_id": track.id,
        "path": track.path,
        "name": name,
        "description": description,
        "lyrics": lyrics,
    }
    output_path = _text_payload_path(output_dir, track, name)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.write("\n")
    return output_path


def _load_text_payload(path: str) -> Optional[dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        logging.warning("Failed to parse text payload %s: %s", path, exc)
        return None


def _audio_worker(
    tracks: Sequence[TrackInfo],
    cfg: dict[str, Any],
    gpu_id: Optional[int],
    upload_milvus: bool,
    milvus_lock: Optional[Any],
) -> WorkerStats:
    _setup_logging(cfg["logging"]["level"])
    device = cfg["embedder"]["device"]
    if gpu_id is not None and torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"
    model_path = cfg["models"]["music_flamingo"]
    max_audio_seconds = cfg["embedder"]["max_audio_seconds"]
    logging.info("Worker starting on %s with %d tracks", device, len(tracks))
    music_model = MusicFlamingo(
        model_path,
        device=device,
        audio_device=device,
        llm_device="cpu",
        audio_only=True,
        offload_audio_tower=False,
        log_tps=False,
        log_lyrics_check=False,
        max_audio_seconds=max_audio_seconds if max_audio_seconds and max_audio_seconds > 0 else None,
    )

    milvus = MilvusWriter(cfg["milvus"]["uri"], cfg["milvus"]["collection"]) if upload_milvus else None
    audio_collection_ready = False
    existing_audio: set[str] = set()
    if upload_milvus and milvus is not None and not cfg["cli"]["force"]:
        names = [
            _canonical_track_name(track)
            for track in tracks
            if track.full_path and os.path.exists(track.full_path)
        ]
        existing_audio = _prefetch_existing_names(milvus, names)

    processed = 0
    skipped = 0
    failed = 0
    start = time.time()

    for track in tracks:
        name = _canonical_track_name(track)
        if not os.path.exists(track.full_path):
            logging.warning("Skipping missing file: %s", track.full_path)
            skipped += 1
            continue
        if upload_milvus and not cfg["cli"]["force"] and name in existing_audio:
            logging.info("Skipping existing audio embedding: %s", name)
            skipped += 1
            continue

        try:
            raw_embedding = music_model.extract_embedding(track.full_path)
            enriched = _enrich_embedding(raw_embedding, device)
            enriched_cpu = enriched.to(device="cpu", dtype=torch.float32)
            milvus_embedding = None
            if upload_milvus and milvus is not None:
                milvus_embedding = _prepare_milvus_embedding(enriched_cpu, MILVUS_MAX_DIM)
                if not audio_collection_ready:
                    if milvus_lock is not None:
                        with milvus_lock:
                            milvus.ensure_collection(milvus_embedding.numel())
                    else:
                        milvus.ensure_collection(milvus_embedding.numel())
                    audio_collection_ready = True
            output_path = _save_embedding(
                cfg["cli"]["output_dir"],
                track,
                enriched_cpu,
                name,
                raw_embedding=raw_embedding,
                milvus_embedding=milvus_embedding,
            )
            if upload_milvus and milvus is not None and milvus_embedding is not None:
                milvus.upsert(name, milvus_embedding.tolist(), MODEL_ID)
            processed += 1
            logging.info("Embedded %s -> %s", name, output_path)
        except Exception as exc:
            failed += 1
            logging.exception("Failed to embed %s: %s", name, exc)

    elapsed = time.time() - start
    logging.info(
        "Worker finished on %s. processed=%d skipped=%d failed=%d elapsed=%.2fs",
        device,
        processed,
        skipped,
        failed,
        elapsed,
    )

    music_model.unload_all()
    del music_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return WorkerStats(processed=processed, skipped=skipped, failed=failed, elapsed=elapsed, device=device)


def _run_audio_pass(
    tracks: Sequence[TrackInfo],
    cfg: dict[str, Any],
    audio_gpus: Sequence[int],
    upload_milvus: bool,
) -> WorkerStats:
    if not tracks:
        return WorkerStats(processed=0, skipped=0, failed=0, elapsed=0.0, device="cpu")
    if not audio_gpus:
        return _audio_worker(tracks, cfg, None, upload_milvus, None)
    if len(audio_gpus) == 1:
        return _audio_worker(tracks, cfg, audio_gpus[0], upload_milvus, None)

    shards = _split_tracks(tracks, len(audio_gpus))
    ctx = mp.get_context("spawn")
    milvus_lock = ctx.Lock() if upload_milvus else None
    start = time.time()
    results: List[WorkerStats] = []
    with ProcessPoolExecutor(max_workers=len(audio_gpus), mp_context=ctx) as executor:
        futures = []
        for gpu_id, shard in zip(audio_gpus, shards):
            if not shard:
                continue
            futures.append(
                executor.submit(
                    _audio_worker,
                    shard,
                    cfg,
                    gpu_id,
                    upload_milvus,
                    milvus_lock,
                )
            )
        for future in as_completed(futures):
            results.append(future.result())
    elapsed = time.time() - start
    processed = sum(item.processed for item in results)
    skipped = sum(item.skipped for item in results)
    failed = sum(item.failed for item in results)
    return WorkerStats(processed=processed, skipped=skipped, failed=failed, elapsed=elapsed, device="multi")


def _run_text_generation_pass(
    tracks: Sequence[TrackInfo],
    cfg: dict[str, Any],
    text_gpu: Optional[int],
) -> WorkerStats:
    if not tracks:
        return WorkerStats(processed=0, skipped=0, failed=0, elapsed=0.0, device="cpu")
    device = cfg["embedder"]["device"]
    if text_gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(text_gpu)
        device = f"cuda:{text_gpu}"
    model_path = cfg["models"]["music_flamingo"]
    max_audio_seconds = cfg["embedder"]["max_audio_seconds"]
    attn_impl = cfg["embedder"]["attn_impl"] or None
    if attn_impl == "flash_attention_2" and (device.startswith("cpu") or not torch.cuda.is_available()):
        logging.warning("flash_attention_2 requested but CUDA is unavailable; falling back to eager attention")
        attn_impl = "eager"
    dequantize_fp8 = cfg["embedder"].get("dequantize_fp8", False)
    logging.info(
        "Starting MusicFlamingo text generation on %s (attn_impl=%s dequantize_fp8=%s)",
        device,
        attn_impl or "default",
        bool(dequantize_fp8),
    )
    music_model = MusicFlamingo(
        model_path,
        device=device,
        audio_device=device,
        llm_device=device,
        attn_implementation=attn_impl,
        dequantize_fp8=bool(dequantize_fp8),
        audio_only=False,
        offload_audio_tower=True,
        log_tps=True,
        log_lyrics_check=False,
        max_audio_seconds=max_audio_seconds if max_audio_seconds and max_audio_seconds > 0 else None,
    )

    processed = 0
    skipped = 0
    failed = 0
    start = time.time()

    for track in tracks:
        name = _canonical_track_name(track)
        if not os.path.exists(track.full_path):
            logging.warning("Skipping missing file: %s", track.full_path)
            skipped += 1
            continue
        payload_path = _text_payload_path(cfg["cli"]["output_dir"], track, name)
        if not cfg["cli"]["force"] and os.path.exists(payload_path):
            skipped += 1
            continue
        try:
            description, lyrics = music_model.describe_text_only(track.full_path)
            _save_text_payload(
                cfg["cli"]["output_dir"],
                track,
                name,
                (description or "").strip(),
                (lyrics or "").strip(),
            )
            processed += 1
            logging.info("Generated text for %s", name)
        except Exception as exc:
            failed += 1
            logging.exception("Failed to generate text for %s: %s", name, exc)

    elapsed = time.time() - start
    logging.info(
        "Text generation done. processed=%d skipped=%d failed=%d elapsed=%.2fs",
        processed,
        skipped,
        failed,
        elapsed,
    )

    music_model.unload_all()
    del music_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return WorkerStats(processed=processed, skipped=skipped, failed=failed, elapsed=elapsed, device=device)


def _run_qwen_pass(
    tracks: Sequence[TrackInfo],
    cfg: dict[str, Any],
    text_gpu: Optional[int],
) -> None:
    device = cfg["embedder"]["device"]
    if text_gpu is not None and torch.cuda.is_available():
        torch.cuda.set_device(text_gpu)
        device = f"cuda:{text_gpu}"

    lyrics_milvus = MilvusWriter(
        cfg["milvus"]["uri"],
        MILVUS_LYRICS_COLLECTION,
        extra_fields=[FieldSchema(name="lyrics", dtype=DataType.VARCHAR, max_length=32768)],
    )
    description_milvus = MilvusWriter(
        cfg["milvus"]["uri"],
        MILVUS_DESCRIPTION_COLLECTION,
        extra_fields=[FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=4096)],
    )

    logging.info("Starting Qwen text embedding pass on %s", device)
    qwen_embedder = QwenEmbedder(batch_size=cfg["embedder"]["batch_size"], device=device)
    text_processed = 0
    text_skipped = 0
    text_failed = 0
    text_start = time.time()
    text_batch: List[tuple[str, str, str]] = []
    lyrics_collection_ready = False
    description_collection_ready = False

    names_for_lookup: List[str] = []
    payloads: List[tuple[str, str, str]] = []
    for track in tracks:
        name = _canonical_track_name(track)
        payload_path = _text_payload_path(cfg["cli"]["output_dir"], track, name)
        payload = _load_text_payload(payload_path)
        if payload is None:
            continue
        description = (payload.get("description") or "").strip()
        lyrics = (payload.get("lyrics") or "").strip()
        if description or lyrics:
            names_for_lookup.append(name)
            payloads.append((name, description, lyrics))

    existing_descriptions: set[str] = set()
    existing_lyrics: set[str] = set()
    if not cfg["cli"]["force"]:
        existing_descriptions = _prefetch_existing_names(description_milvus, names_for_lookup)
        existing_lyrics = _prefetch_existing_names(lyrics_milvus, names_for_lookup)

    def flush_text_batch() -> None:
        nonlocal text_processed, text_failed, lyrics_collection_ready, description_collection_ready
        if not text_batch:
            return
        texts = [item[2] for item in text_batch]
        try:
            embeddings = qwen_embedder.encode_documents(texts)
        except Exception as exc:
            text_failed += len(text_batch)
            logging.exception("Failed to embed text batch (%d items): %s", len(text_batch), exc)
            text_batch.clear()
            return
        if hasattr(embeddings, "tolist"):
            embeddings_list = embeddings.tolist()
        else:
            embeddings_list = list(embeddings)
        for (track_name, kind, text), embedding in zip(text_batch, embeddings_list):
            try:
                if kind == "description":
                    if not description_collection_ready:
                        description_milvus.ensure_collection(len(embedding))
                        description_collection_ready = True
                    description_milvus.upsert(
                        track_name,
                        embedding,
                        TEXT_MODEL_ID,
                        extra_fields={"description": text},
                    )
                    text_processed += 1
                elif kind == "lyrics":
                    if not lyrics_collection_ready:
                        lyrics_milvus.ensure_collection(len(embedding))
                        lyrics_collection_ready = True
                    lyrics_milvus.upsert(
                        track_name,
                        embedding,
                        TEXT_MODEL_ID,
                        extra_fields={"lyrics": text},
                    )
                    text_processed += 1
            except Exception as exc:
                text_failed += 1
                logging.exception("Failed to upsert %s embedding for %s: %s", kind, track_name, exc)
        text_batch.clear()

    for name, description, lyrics in payloads:
        queued = False
        if description and (cfg["cli"]["force"] or name not in existing_descriptions):
            text_batch.append((name, "description", description))
            queued = True
        if lyrics and (cfg["cli"]["force"] or name not in existing_lyrics):
            text_batch.append((name, "lyrics", lyrics))
            queued = True
        if not queued:
            text_skipped += 1
        if len(text_batch) >= cfg["embedder"]["batch_size"]:
            flush_text_batch()

    flush_text_batch()
    text_elapsed = time.time() - text_start
    logging.info(
        "Qwen pass done. processed=%d skipped=%d failed=%d elapsed=%.2fs",
        text_processed,
        text_skipped,
        text_failed,
        text_elapsed,
    )


def _write_track_path_file(embedding_path: str, track_path: str) -> None:
    base, _ = os.path.splitext(embedding_path)
    path_file = f"{base}.path"
    with open(path_file, "w", encoding="utf-8") as handle:
        handle.write(track_path)
        handle.write("\n")


def _ensure_output_dir(output_dir: str) -> None:
    if not output_dir:
        raise ValueError("output directory is required")
    os.makedirs(output_dir, exist_ok=True)
    test_path = os.path.join(output_dir, ".write_test")
    try:
        with open(test_path, "w", encoding="utf-8") as handle:
            handle.write("ok\n")
    except OSError as exc:
        raise OSError(f"Output directory not writable: {output_dir}") from exc
    else:
        os.remove(test_path)


def _parse_args() -> dict[str, Any]:
    cfg = _default_config()
    parser = argparse.ArgumentParser(description="Navidrome Embedder (Python)")

    parser.add_argument("--config", dest="config_file", default="", help="Path to TOML config file")
    parser.add_argument("--db-path", dest="db_path", default=cfg["database"]["path"], help="Path to Navidrome database")
    parser.add_argument("--milvus-uri", dest="milvus_uri", default=cfg["milvus"]["uri"], help="Milvus connection URI")

    parser.add_argument("--library-path", dest="library_path", default=cfg["models"]["library_path"], help="Path to llama.cpp libraries")
    parser.add_argument("--text-model", dest="text_model", default=cfg["models"]["text_model"], help="Path to text embedding model")
    parser.add_argument("--audio-model", dest="audio_model", default=cfg["models"]["audio_model"], help="Path to audio model")
    parser.add_argument("--music-model", dest="music_model", default=cfg["models"]["music_flamingo"], help="Path to MusicFlamingo model directory")
    parser.add_argument("--projector", dest="projector", default=cfg["models"]["projector"], help="Path to audio projector (mmproj)")

    parser.add_argument("--batch-size", dest="batch_size", type=int, default=cfg["embedder"]["batch_size"], help="Batch size for processing")
    parser.add_argument("--gpu-layers", dest="gpu_layers", type=int, default=cfg["embedder"]["gpu_layers"], help="GPU layers to offload")
    parser.add_argument("--threads", dest="threads", type=int, default=cfg["embedder"]["threads"], help="CPU threads for inference (0 = auto)")
    parser.add_argument("--device", dest="device", default=cfg["embedder"]["device"], help="Inference device (cuda or cpu)")
    parser.add_argument(
        "--audio-gpus",
        dest="audio_gpus",
        default=cfg["embedder"]["audio_gpus"],
        help="Comma-separated GPU ids for audio embedding workers (default: all available)",
    )
    parser.add_argument(
        "--text-gpu",
        dest="text_gpu",
        type=str,
        default=cfg["embedder"]["text_gpu"],
        help="GPU id for MusicFlamingo text generation (default: GPU with most VRAM)",
    )
    parser.add_argument(
        "--attn-impl",
        dest="attn_impl",
        default=cfg["embedder"]["attn_impl"],
        help="Attention implementation for MusicFlamingo LLM (e.g. flash_attention_2, sdpa, eager)",
    )
    parser.add_argument(
        "--dequantize-fp8",
        dest="dequantize_fp8",
        action="store_true",
        default=None,
        help="Dequantize FP8 weights to bf16 for MusicFlamingo text generation",
    )
    parser.add_argument(
        "--no-dequantize-fp8",
        dest="dequantize_fp8",
        action="store_false",
        help="Keep FP8 weights for MusicFlamingo text generation",
    )
    parser.add_argument(
        "--max-tracks",
        dest="max_tracks",
        type=int,
        default=cfg["embedder"]["max_tracks"],
        help="Process at most N tracks (0 = no limit)",
    )
    parser.add_argument(
        "--skip-text",
        dest="skip_text",
        action="store_true",
        help="Skip description/lyrics generation and Qwen text embeddings",
    )
    parser.add_argument(
        "--max-audio-seconds",
        dest="max_audio_seconds",
        type=float,
        default=cfg["embedder"]["max_audio_seconds"],
        help="Truncate audio to N seconds before embedding (0 = no truncation)",
    )

    parser.add_argument("--log-level", dest="log_level", default=cfg["logging"]["level"], help="Log level (debug, info, warn, error)")

    parser.add_argument("--music-dir", dest="music_dir", default=cfg["cli"]["music_dir"], help="Path to music directory")
    parser.add_argument("--output-dir", dest="output_dir", default=cfg["cli"]["output_dir"], help="Directory to write .pt embeddings")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true", help="Show what would be embedded without actually doing it")
    parser.add_argument("--force", dest="force", action="store_true", help="Re-embed tracks even if embeddings exist")
    parser.add_argument("--no-milvus", dest="no_milvus", action="store_true", help="Skip uploading embeddings to Milvus")
    parser.add_argument(
        "--sort-length",
        dest="sort_length",
        choices=("long", "short"),
        default=cfg["cli"]["sort_length"],
        help="Process tracks ordered by duration (long = longest first, short = shortest first)",
    )

    parser.add_argument("--version", dest="version", action="store_true", help="Show version and exit")

    args = parser.parse_args()

    if args.version:
        print(f"navidrome-embedder version {VERSION}")
        sys.exit(0)

    cfg["database"]["path"] = args.db_path
    cfg["milvus"]["uri"] = args.milvus_uri
    cfg["models"]["library_path"] = args.library_path
    cfg["models"]["text_model"] = args.text_model
    cfg["models"]["audio_model"] = args.audio_model
    cfg["models"]["projector"] = args.projector
    cfg["models"]["music_flamingo"] = args.music_model
    cfg["embedder"]["batch_size"] = args.batch_size
    cfg["embedder"]["gpu_layers"] = args.gpu_layers
    cfg["embedder"]["threads"] = args.threads
    cfg["embedder"]["device"] = args.device
    cfg["embedder"]["audio_gpus"] = args.audio_gpus
    cfg["embedder"]["text_gpu"] = args.text_gpu
    cfg["embedder"]["attn_impl"] = args.attn_impl
    if args.dequantize_fp8 is not None:
        cfg["embedder"]["dequantize_fp8"] = args.dequantize_fp8
    cfg["embedder"]["max_tracks"] = args.max_tracks
    cfg["embedder"]["skip_text"] = args.skip_text or cfg["embedder"]["skip_text"]
    cfg["embedder"]["max_audio_seconds"] = args.max_audio_seconds
    cfg["logging"]["level"] = args.log_level
    cfg["cli"]["music_dir"] = args.music_dir
    cfg["cli"]["output_dir"] = args.output_dir
    cfg["cli"]["dry_run"] = args.dry_run
    cfg["cli"]["force"] = args.force
    cfg["cli"]["upload_milvus"] = not args.no_milvus
    cfg["cli"]["config_file"] = args.config_file
    cfg["cli"]["sort_length"] = args.sort_length

    _normalize_config(cfg)
    return cfg


def _check_milvus_available(cfg: dict[str, Any]) -> bool:
    writer = MilvusWriter(cfg["milvus"]["uri"], cfg["milvus"]["collection"])
    try:
        writer.connect()
    except Exception as exc:
        logging.warning(
            "Milvus unavailable at %s (%s); disabling Milvus upload",
            cfg["milvus"]["uri"],
            exc,
        )
        return False
    return True


def main():
    cfg = _parse_args()
    _setup_logging(cfg["logging"]["level"])
    _ensure_output_dir(cfg["cli"]["output_dir"])
    tracks = get_all_music(
        cfg["database"]["path"],
        music_dir=cfg["cli"]["music_dir"],
        sort_length=cfg["cli"]["sort_length"],
    )
    if not tracks:
        logging.info("No tracks found in database.")
        return

    max_tracks = cfg["embedder"]["max_tracks"]
    if max_tracks and max_tracks > 0:
        tracks = tracks[:max_tracks]
        logging.info("Limiting to first %d tracks", max_tracks)
        if not tracks:
            logging.info("No tracks left after applying max-tracks limit.")
            return

    if cfg["cli"]["dry_run"]:
        for track in tracks:
            name = _canonical_track_name(track)
            logging.info("Dry run: would embed %s (%s)", name, track.full_path)
        return

    upload_milvus = cfg["cli"]["upload_milvus"]
    skip_text = cfg["embedder"]["skip_text"]

    if upload_milvus and not _check_milvus_available(cfg):
        upload_milvus = False
        cfg["cli"]["upload_milvus"] = False

    available_gpus = _available_gpus()
    audio_gpus = _parse_gpu_list(cfg["embedder"]["audio_gpus"])
    if not audio_gpus:
        audio_gpus = available_gpus
    if audio_gpus and not available_gpus:
        logging.warning(
            "Requested audio GPU ids %s but no CUDA GPUs are available; falling back to CPU",
            ", ".join(str(gpu) for gpu in audio_gpus),
        )
        audio_gpus = []
    if audio_gpus:
        invalid = [gpu for gpu in audio_gpus if gpu not in available_gpus]
        if invalid:
            raise ValueError(f"Audio GPU ids not available: {invalid} (available={available_gpus})")
    text_gpu = None
    if cfg["embedder"]["text_gpu"]:
        parsed_text_gpu = _parse_gpu_list(str(cfg["embedder"]["text_gpu"]))
        if parsed_text_gpu:
            text_gpu = parsed_text_gpu[0]
    text_gpu = _select_text_gpu(available_gpus, text_gpu)

    if available_gpus:
        for gpu_id in available_gpus:
            props = torch.cuda.get_device_properties(gpu_id)
            logging.info("GPU %d: %s (%.1f GB)", gpu_id, props.name, props.total_memory / (1024 ** 3))
    if audio_gpus:
        logging.info("Audio embedding GPUs: %s", ", ".join(str(gpu) for gpu in audio_gpus))
    else:
        logging.info("Audio embedding on CPU")
    if text_gpu is not None:
        logging.info("Text generation GPU: %d", text_gpu)
    else:
        logging.info("Text generation on CPU")

    logging.info("Starting audio embedding pass")
    audio_stats = _run_audio_pass(tracks, cfg, audio_gpus, upload_milvus)
    logging.info(
        "Audio pass done. processed=%d skipped=%d failed=%d elapsed=%.2fs",
        audio_stats.processed,
        audio_stats.skipped,
        audio_stats.failed,
        audio_stats.elapsed,
    )

    if skip_text:
        logging.info("Text generation disabled; skipping text and Qwen passes.")
        return

    text_stats = _run_text_generation_pass(tracks, cfg, text_gpu)
    logging.info(
        "Text generation pass done. processed=%d skipped=%d failed=%d elapsed=%.2fs",
        text_stats.processed,
        text_stats.skipped,
        text_stats.failed,
        text_stats.elapsed,
    )

    if not upload_milvus:
        logging.info("Milvus upload disabled; skipping Qwen text embedding pass.")
        return

    _run_qwen_pass(tracks, cfg, text_gpu)



if __name__ == "__main__":
    main()
