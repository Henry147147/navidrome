from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable, Sequence

MODEL_MUQ_AUDIO = "muq_audio"
MODEL_MUQ_MULAN = "muq_mulan"
MODEL_MUSIC_FLAMINGO_AUDIO = "music_flamingo_audio"

COLLECTION_MUQ_AUDIO = "muq_audio_embedding"
COLLECTION_MUQ_MULAN = "muq_mulan_embedding"
COLLECTION_MUSIC_FLAMINGO_AUDIO = "flamingo_audio_embedding"

LEGACY_COLLECTIONS = (
    "lyrics_embedding",
    "description_embedding",
)

AUDIO_SAMPLE_RATE = 24_000
DEFAULT_AUDIO_DIMENSION = 1_024
DEFAULT_MULAN_DIMENSION = 512
DEFAULT_MUSIC_FLAMINGO_AUDIO_DIMENSION = 3_584

_AUDIO_ALIASES = {
    "",
    MODEL_MUQ_AUDIO,
    "audio",
    "flamingo",
    "music-flamingo",
    "music_flamingo",
}
_MULAN_ALIASES = {
    MODEL_MUQ_MULAN,
    "lyrics",
    "lyric",
    "description",
    "desc",
    "qwen3",
    "qwen8b",
    "qwen-8b",
}
_MUSIC_FLAMINGO_AUDIO_ALIASES = {
    MODEL_MUSIC_FLAMINGO_AUDIO,
    "music_flamingo_audio",
    "music-flamingo-audio",
    "flamingo_audio",
}


@dataclass(frozen=True)
class TrackInfo:
    id: str
    path: str
    title: str
    artist: str
    full_path: str


def get_env_value(name: str, default: str, *aliases: str) -> str:
    for key in (name, *aliases):
        value = os.environ.get(key)
        if value is not None and value.strip():
            return value
    return default


def default_database_path() -> str:
    return get_env_value("ND_DBPATH", "navidrome.db")


def default_music_dir() -> str:
    return get_env_value("ND_MUSICDIR", "", "MF_MUSIC_DIR")


def default_milvus_uri() -> str:
    return get_env_value("MILVUS_URI", "http://127.0.0.1:19530")


def default_muq_audio_ref() -> str:
    return get_env_value("MUQ_AUDIO_MODEL_REF", "OpenMuQ/MuQ-large-msd-iter", "MF_AUDIO_MODEL")


def default_muq_mulan_ref() -> str:
    return get_env_value("MUQ_MULAN_MODEL_REF", "OpenMuQ/MuQ-MuLan-large", "MF_TEXT_MODEL")


def default_music_flamingo_ref() -> str:
    return get_env_value(
        "MUSIC_FLAMINGO_MODEL_REF",
        "nvidia/music-flamingo-hf",
        "MUSIC_FLAMINGO_MODEL_PATH",
    )


def default_muq_cache_dir() -> str:
    return get_env_value("MUQ_CACHE_DIR", "")


def default_muq_max_audio_seconds() -> float:
    raw = get_env_value("MUQ_MAX_AUDIO_SECONDS", "180")
    return max(0.0, float(raw))


def default_device() -> str:
    explicit = os.environ.get("MUQ_DEVICE")
    if explicit and explicit.strip():
        return explicit.strip()

    legacy_text_gpu = os.environ.get("MF_TEXT_GPU", "").strip()
    if legacy_text_gpu:
        return f"cuda:{legacy_text_gpu}"

    legacy_audio_gpus = os.environ.get("MF_AUDIO_GPUS", "").strip()
    if legacy_audio_gpus:
        first_gpu = legacy_audio_gpus.split(",")[0].strip()
        if first_gpu:
            return f"cuda:{first_gpu}"

    return _best_cuda_device() if _cuda_available() else "cpu"


def _cuda_available() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def _best_cuda_device() -> str:
    import torch

    best_index = 0
    best_free = -1
    for index in range(torch.cuda.device_count()):
        free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        if free_bytes > best_free:
            best_free = free_bytes
            best_index = index
    return f"cuda:{best_index}"


def normalize_model_name(name: str | None) -> str:
    normalized = (name or "").strip().lower()
    if normalized in _MUSIC_FLAMINGO_AUDIO_ALIASES:
        return MODEL_MUSIC_FLAMINGO_AUDIO
    if normalized in _AUDIO_ALIASES:
        return MODEL_MUQ_AUDIO
    if normalized in _MULAN_ALIASES:
        return MODEL_MUQ_MULAN
    raise ValueError(f"unsupported model: {name}")


def normalize_model_names(models: Iterable[str] | None) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in models or (MODEL_MUQ_AUDIO, MODEL_MUQ_MULAN):
        canonical = normalize_model_name(item)
        if canonical in seen:
            continue
        seen.add(canonical)
        ordered.append(canonical)
    if not ordered:
        return [MODEL_MUQ_AUDIO, MODEL_MUQ_MULAN]
    return ordered


def model_collection_name(model: str) -> str:
    canonical = normalize_model_name(model)
    if canonical == MODEL_MUSIC_FLAMINGO_AUDIO:
        return COLLECTION_MUSIC_FLAMINGO_AUDIO
    if canonical == MODEL_MUQ_AUDIO:
        return COLLECTION_MUQ_AUDIO
    return COLLECTION_MUQ_MULAN


def model_dimension(model: str) -> int:
    canonical = normalize_model_name(model)
    if canonical == MODEL_MUSIC_FLAMINGO_AUDIO:
        return DEFAULT_MUSIC_FLAMINGO_AUDIO_DIMENSION
    if canonical == MODEL_MUQ_AUDIO:
        return DEFAULT_AUDIO_DIMENSION
    return DEFAULT_MULAN_DIMENSION


def resolve_track_path(path: str, music_dir: str | None) -> str:
    if os.path.isabs(path) or not music_dir:
        return path
    return os.path.join(music_dir, path)


def canonical_track_name(track: TrackInfo) -> str:
    artist = track.artist.strip()
    title = track.title.strip()
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return title
    return track.path


def track_storage_key(track: TrackInfo) -> str:
    for candidate in (track.id, track.path, canonical_track_name(track)):
        value = candidate.strip()
        if value:
            return value
    raise ValueError("track has no usable storage key")


def get_all_music(database_path: str, *, music_dir: str | None = None, limit: int = 0) -> list[TrackInfo]:
    if not database_path:
        raise ValueError("database path is required")
    if database_path.startswith("postgres://") or database_path.startswith("postgresql://"):
        raise ValueError("postgres databases are not supported by the Python MuQ tools")

    query = (
        "SELECT id, path, COALESCE(title, ''), COALESCE(artist, '') "
        "FROM media_file WHERE path IS NOT NULL AND path != '' "
        "ORDER BY id"
    )
    params: Sequence[int] = ()
    if limit > 0:
        query += " LIMIT ?"
        params = (limit,)

    with sqlite3.connect(database_path) as conn:
        rows = conn.execute(query, params).fetchall()

    tracks: list[TrackInfo] = []
    for track_id, path, title, artist in rows:
        tracks.append(
            TrackInfo(
                id=str(track_id),
                path=path,
                title=title,
                artist=artist,
                full_path=resolve_track_path(path, music_dir),
            )
        )
    return tracks
