"""Helpers for mapping Navidrome track identifiers to embedding names."""

from __future__ import annotations

import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def _default_db_path() -> Path:
    env_path = os.getenv("NAVIDROME_DB_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parents[1] / "navidrome.db"


class TrackNameResolver:
    """Caches mappings between Navidrome track ids and Milvus embedding names."""

    def __init__(self, db_path: Optional[Path | str] = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _default_db_path()
        self._id_to_name: Optional[Dict[str, str]] = None
        self._name_to_ids: Optional[Dict[str, List[str]]] = None

    @staticmethod
    def _normalize_component(value: str) -> str:
        normalized = value.replace("•", "&")
        normalized = normalized.replace("/", "_")
        normalized = normalized.replace("\\", "_")
        normalized = " ".join(normalized.split())
        return normalized.strip()

    @classmethod
    def canonical_name(cls, artist: str, title: str) -> str:
        artist_norm = cls._normalize_component(artist or "")
        title_norm = cls._normalize_component(title or "")
        return f"{artist_norm} - {title_norm}".strip()

    def _ensure_loaded(self) -> None:
        if self._id_to_name is not None and self._name_to_ids is not None:
            return

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, artist, title FROM media_file")
            id_to_name: Dict[str, str] = {}
            name_to_ids: Dict[str, List[str]] = defaultdict(list)
            for track_id, artist, title in cursor.fetchall():
                canonical = self.canonical_name(artist or "", title or "")
                id_to_name[track_id] = canonical
                name_to_ids[canonical].append(track_id)
        finally:
            conn.close()

        self._id_to_name = id_to_name
        self._name_to_ids = name_to_ids

    def ids_to_names(self, track_ids: Iterable[str]) -> Dict[str, str]:
        self._ensure_loaded()
        assert self._id_to_name is not None
        result: Dict[str, str] = {}
        for track_id in track_ids:
            name = self._id_to_name.get(track_id)
            if name:
                result[track_id] = name
        return result

    def names_to_ids(self, names: Iterable[str]) -> Dict[str, str]:
        self._ensure_loaded()
        assert self._name_to_ids is not None
        result: Dict[str, str] = {}
        for name in names:
            ids = self._name_to_ids.get(name)
            if ids:
                result[name] = ids[0]
        return result

    def name_to_id(self, name: str) -> Optional[str]:
        self._ensure_loaded()
        assert self._name_to_ids is not None
        ids = self._name_to_ids.get(name)
        return ids[0] if ids else None

