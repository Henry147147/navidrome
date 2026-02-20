#!/usr/bin/env python3
"""Resolve Navidrome player/transcoding profiles for playback investigations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ResolvedProfile:
    id: str
    profile_name: str
    client_name: str
    user_agent: str
    max_bit_rate: int
    last_seen: str
    transcoding_id: str | None
    transcoding_name: str | None
    transcoding_target_format: str | None
    transcoding_default_bit_rate: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _parse_last_seen(value: Any) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)

    text = str(value).strip()
    if not text:
        return datetime.fromtimestamp(0, tz=timezone.utc)

    # RFC3339Nano style values may use trailing Z
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_transcoding_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item.get("id", ""),
        "name": item.get("name", ""),
        "targetFormat": item.get("targetFormat")
        or item.get("target_format")
        or item.get("targetformat"),
        "defaultBitRate": _safe_int(
            item.get("defaultBitRate")
            if item.get("defaultBitRate") is not None
            else item.get("default_bit_rate")
        ),
    }


def _build_profile(
    profile_name: str,
    player: dict[str, Any],
    transcoding_map: dict[str, dict[str, Any]],
) -> ResolvedProfile:
    transcoding_id = player.get("transcodingId") or player.get("transcoding_id")
    transcoding = transcoding_map.get(str(transcoding_id)) if transcoding_id else None

    return ResolvedProfile(
        id=str(player.get("id", "")),
        profile_name=profile_name,
        client_name=str(player.get("client", "") or ""),
        user_agent=str(player.get("userAgent", "") or player.get("user_agent", "") or ""),
        max_bit_rate=_safe_int(player.get("maxBitRate") or player.get("max_bit_rate"), 0),
        last_seen=str(player.get("lastSeen") or player.get("last_seen") or ""),
        transcoding_id=str(transcoding_id) if transcoding_id else None,
        transcoding_name=str(transcoding.get("name", "")) if transcoding else None,
        transcoding_target_format=(
            str(transcoding.get("targetFormat", "")) if transcoding else None
        ),
        transcoding_default_bit_rate=(
            _safe_int(transcoding.get("defaultBitRate"), 0) if transcoding else None
        ),
    )


def resolve_profiles(
    players: list[dict[str, Any]],
    transcodings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Resolve GUI and app-equivalent profiles from player/transcoding snapshots."""

    transcoding_map: dict[str, dict[str, Any]] = {}
    for item in transcodings:
        normalized = _normalize_transcoding_item(item)
        if normalized["id"]:
            transcoding_map[normalized["id"]] = normalized

    sorted_players = sorted(
        players,
        key=lambda p: _parse_last_seen(p.get("lastSeen") or p.get("last_seen")),
        reverse=True,
    )

    gui_player = next(
        (p for p in sorted_players if str(p.get("client", "")) == "NavidromeUI"),
        None,
    )
    app_player = next(
        (
            p
            for p in sorted_players
            if str(p.get("client", ""))
            and str(p.get("client", "")) != "NavidromeUI"
        ),
        None,
    )

    gui_profile = (
        _build_profile("gui_profile", gui_player, transcoding_map).to_dict()
        if gui_player
        else None
    )
    app_profile = (
        _build_profile("app_profile", app_player, transcoding_map).to_dict()
        if app_player
        else None
    )

    return {
        "gui_profile": gui_profile,
        "app_profile": app_profile,
        "app_profile_available": app_profile is not None,
    }


def player_cookie_name(username: str) -> str:
    """Return the Navidrome Subsonic middleware player-cookie name for the user."""
    return f"nd-player-{username.encode('utf-8').hex()}"
