#!/usr/bin/env python3
"""Track sampling helpers for playback investigation matrix."""

from __future__ import annotations

from typing import Any

import requests

LOSSY_SUFFIXES = {"mp3", "aac", "opus", "m4a"}


class TrackSamplingError(RuntimeError):
    pass


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_song(song: dict[str, Any]) -> dict[str, Any]:
    suffix = str(song.get("suffix", "")).lower()
    bitrate = _safe_int(song.get("bitRate") or song.get("bitrate"), 0)

    return {
        "id": str(song.get("id", "")),
        "title": str(song.get("title", "")),
        "artist": str(song.get("artist", "")),
        "album": str(song.get("album", "")),
        "suffix": suffix,
        "bitrate_kbps": bitrate,
        "duration_s": float(song.get("duration", 0) or 0),
    }


def _build_subsonic_params(auth: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "u": auth["username"],
        "t": auth["subsonic_token"],
        "s": auth["subsonic_salt"],
        "v": "1.16.1",
        "c": "PlaybackProbeSampler",
        "f": "json",
    }
    if extra:
        params.update(extra)
    return params


def _get_random_songs(
    base_url: str,
    auth: dict[str, Any],
    *,
    count: int,
    timeout_sec: int,
) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/rest/getRandomSongs.view"
    params = _build_subsonic_params(auth, {"size": count})
    response = requests.get(url, params=params, timeout=timeout_sec)
    response.raise_for_status()

    payload = response.json().get("subsonic-response", {})
    if payload.get("status") != "ok":
        err = payload.get("error", {})
        raise TrackSamplingError(
            f"Subsonic getRandomSongs failed: {err.get('message', 'unknown')} (code={err.get('code')})"
        )

    songs = payload.get("randomSongs", {}).get("song")
    return [_normalize_song(song) for song in _ensure_list(songs)]


def sample_tracks(
    base_url: str,
    auth: dict[str, Any],
    *,
    max_batches: int = 60,
    batch_size: int = 200,
    timeout_sec: int = 30,
) -> dict[str, Any]:
    """Pick flac_high, flac_any, and lossy_ref tracks for the matrix."""

    discovered: dict[str, dict[str, Any]] = {}

    for _ in range(max_batches):
        songs = _get_random_songs(
            base_url,
            auth,
            count=batch_size,
            timeout_sec=timeout_sec,
        )
        for song in songs:
            if song["id"]:
                discovered[song["id"]] = song

        flacs = sorted(
            [s for s in discovered.values() if s["suffix"] == "flac"],
            key=lambda s: s["bitrate_kbps"],
            reverse=True,
        )
        lossy = sorted(
            [s for s in discovered.values() if s["suffix"] in LOSSY_SUFFIXES],
            key=lambda s: s["bitrate_kbps"],
            reverse=True,
        )

        if len(flacs) >= 2 and lossy:
            break

    flacs = sorted(
        [s for s in discovered.values() if s["suffix"] == "flac"],
        key=lambda s: s["bitrate_kbps"],
        reverse=True,
    )
    lossy = sorted(
        [s for s in discovered.values() if s["suffix"] in LOSSY_SUFFIXES],
        key=lambda s: s["bitrate_kbps"],
        reverse=True,
    )

    if len(flacs) < 2:
        raise TrackSamplingError(
            f"Need at least 2 FLAC tracks, found {len(flacs)} after {max_batches} batches"
        )
    if not lossy:
        raise TrackSamplingError(
            f"Need at least 1 lossy reference track ({sorted(LOSSY_SUFFIXES)}), found 0 after {max_batches} batches"
        )

    selected = {
        "flac_high": flacs[0],
        "flac_any": flacs[1],
        "lossy_ref": lossy[0],
    }

    return {
        "selected_tracks": selected,
        "sampling": {
            "discovered_total": len(discovered),
            "flac_count": len(flacs),
            "lossy_count": len(lossy),
            "lossy_suffixes": sorted(LOSSY_SUFFIXES),
            "max_batches": max_batches,
            "batch_size": batch_size,
        },
    }
