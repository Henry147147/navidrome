#!/usr/bin/env python3
"""Direct Subsonic stream probing utilities."""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

from profile_resolver import player_cookie_name


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _canonical_to_header_ua(canonical_ua: str) -> str:
    ua = canonical_ua or ""
    if "Chrome" in ua:
        return (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    if "Firefox" in ua:
        return "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"
    if "Safari" in ua:
        return (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_4) AppleWebKit/605.1.15 "
            "(KHTML, like Gecko) Version/16.5 Safari/605.1.15"
        )
    return "NavidromePlaybackProbe/1.0"


def _build_subsonic_params(
    auth: dict[str, Any],
    track_id: str,
    client_name: str,
    extra_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "u": auth["username"],
        "t": auth["subsonic_token"],
        "s": auth["subsonic_salt"],
        "v": "1.16.1",
        "c": client_name,
        "id": track_id,
    }
    if extra_params:
        params.update(extra_params)
    return params


def _get_header_case_insensitive(headers: dict[str, Any], key: str) -> str | None:
    wanted = key.lower()
    for k, v in headers.items():
        if k.lower() == wanted:
            return str(v)
    return None


def run_stream_probe(
    *,
    base_url: str,
    auth: dict[str, Any],
    track: dict[str, Any],
    scenario_id: str,
    profile_name: str,
    client_name: str,
    user_agent: str,
    player_id: str | None,
    extra_params: dict[str, Any] | None = None,
    timeout_sec: int = 60,
) -> dict[str, Any]:
    """Probe one /rest/stream request and capture headers + timing."""

    track_id = str(track["id"])
    stream_url = f"{base_url.rstrip('/')}/rest/stream"

    session = requests.Session()
    if player_id:
        cookie_name = player_cookie_name(auth["username"])
        session.cookies.set(cookie_name, player_id, path="/")

    params = _build_subsonic_params(
        auth,
        track_id=track_id,
        client_name=client_name,
        extra_params=extra_params,
    )

    headers = {
        "User-Agent": _canonical_to_header_ua(user_agent),
    }

    http_status: int | None = None
    content_type: str | None = None
    accept_ranges: str | None = None
    content_length: str | None = None
    transfer_encoding: str | None = None
    ttfb_ms: float | None = None
    start_success = False

    error_code: str | None = None

    try:
        started_at = time.perf_counter()
        response = session.get(
            stream_url,
            params=params,
            headers=headers,
            stream=True,
            timeout=timeout_sec,
        )
        http_status = response.status_code

        headers_obj = dict(response.headers)
        content_type = _get_header_case_insensitive(headers_obj, "Content-Type")
        accept_ranges = _get_header_case_insensitive(headers_obj, "Accept-Ranges")
        content_length = _get_header_case_insensitive(headers_obj, "Content-Length")
        transfer_encoding = _get_header_case_insensitive(headers_obj, "Transfer-Encoding")

        if http_status and 200 <= http_status < 400:
            for chunk in response.iter_content(chunk_size=1):
                if chunk is None:
                    continue
                ttfb_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
                start_success = True
                break

        response.close()
    except requests.RequestException as exc:
        error_code = f"request_error:{exc.__class__.__name__}"

    query = parse_qs(urlparse(stream_url).query)

    return {
        "scenario_id": scenario_id,
        "track_id": track_id,
        "suffix": track.get("suffix"),
        "bitrate_kbps": _safe_int(track.get("bitrate_kbps"), 0),
        "ttfb_ms": ttfb_ms,
        "time_to_play_ms": None,
        "start_success": start_success,
        "audio_error_code": error_code,
        "waiting_events": 0,
        "stalled_events": 0,
        "http_status": http_status,
        "content_type": content_type,
        "accept_ranges": accept_ranges,
        "content_length": content_length,
        "transfer_encoding": transfer_encoding,
        "profile_name": profile_name,
        "client_name": client_name,
        "user_agent": user_agent,
        "stream_query": query,
    }
