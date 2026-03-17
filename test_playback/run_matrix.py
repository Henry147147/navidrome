#!/usr/bin/env python3
"""Run Navidrome playback investigation matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from analyze_results import analyze_results_dir
from profile_resolver import resolve_profiles
from selenium_probe import FAST_3G, NO_THROTTLE, SeleniumProbeError, SeleniumProbeSession
from subsonic_probe import run_stream_probe
from track_sampler import TrackSamplingError, sample_tracks

REQUIRED_FIELDS = [
    "scenario_id",
    "track_id",
    "suffix",
    "bitrate_kbps",
    "ttfb_ms",
    "time_to_play_ms",
    "start_success",
    "audio_error_code",
    "waiting_events",
    "stalled_events",
    "http_status",
    "content_type",
    "accept_ranges",
    "content_length",
    "transfer_encoding",
    "profile_name",
    "client_name",
    "user_agent",
]


@dataclass(frozen=True)
class GuiScenario:
    scenario_id: str
    track_key: str
    cold_reps: int
    warm_reps: int
    throttle: str


@dataclass(frozen=True)
class ApiScenario:
    scenario_id: str
    mode: str
    reps: int


def _timestamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    for key in REQUIRED_FIELDS:
        out.setdefault(key, None)
    return out


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True))
        fh.write("\n")


def _request_json(
    method: str,
    url: str,
    *,
    timeout_sec: int,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> Any:
    response = requests.request(
        method,
        url,
        headers=headers,
        params=params,
        json=json_body,
        timeout=timeout_sec,
    )
    response.raise_for_status()
    return response.json()


def _preflight_auth(
    *,
    base_url: str,
    username: str,
    password: str,
    timeout_sec: int,
) -> dict[str, Any]:
    login_url = f"{base_url.rstrip('/')}/auth/login"
    login_response = requests.post(
        login_url,
        json={"username": username, "password": password},
        timeout=timeout_sec,
    )
    if login_response.status_code == 401:
        raise RuntimeError("Preflight failed: invalid username/password for /auth/login")
    login_response.raise_for_status()
    login = login_response.json()

    token = login.get("token")
    subsonic_token = login.get("subsonicToken")
    subsonic_salt = login.get("subsonicSalt")
    if not token or not subsonic_token or not subsonic_salt:
        raise RuntimeError("Preflight failed: login response missing token/subsonicToken/subsonicSalt")

    ping_url = f"{base_url.rstrip('/')}/rest/ping.view"
    ping = _request_json(
        "GET",
        ping_url,
        timeout_sec=timeout_sec,
        params={
            "u": username,
            "t": subsonic_token,
            "s": subsonic_salt,
            "v": "1.16.1",
            "c": "PlaybackProbe",
            "f": "json",
        },
    )
    ping_payload = ping.get("subsonic-response", {})
    if ping_payload.get("status") != "ok":
        error = ping_payload.get("error", {})
        raise RuntimeError(
            f"Preflight failed: Subsonic ping status={ping_payload.get('status')} "
            f"code={error.get('code')} message={error.get('message')}"
        )

    return {
        "username": username,
        "token": token,
        "subsonic_token": subsonic_token,
        "subsonic_salt": subsonic_salt,
    }


def _native_auth_headers(token: str) -> dict[str, str]:
    return {
        "X-ND-Authorization": f"Bearer {token}",
        "X-ND-Client-Unique-Id": "playback-probe",
        "Accept": "application/json",
    }


def _fetch_native_list(
    *,
    base_url: str,
    token: str,
    resource: str,
    timeout_sec: int,
    sort_field: str,
) -> list[dict[str, Any]]:
    url = f"{base_url.rstrip('/')}/api/{resource}"
    payload = _request_json(
        "GET",
        url,
        timeout_sec=timeout_sec,
        headers=_native_auth_headers(token),
        params={
            "sort": sort_field,
            "order": "DESC",
            "range": "[0,999]",
            "filter": "{}",
        },
    )

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise RuntimeError(f"Unexpected /api/{resource} response shape: {type(payload)}")


def _build_error_row(
    *,
    scenario_id: str,
    track: dict[str, Any],
    profile_name: str,
    client_name: str,
    user_agent: str,
    error_code: str,
) -> dict[str, Any]:
    return _normalize_row(
        {
            "scenario_id": scenario_id,
            "track_id": str(track.get("id", "")),
            "suffix": track.get("suffix"),
            "bitrate_kbps": _safe_int(track.get("bitrate_kbps"), 0),
            "start_success": False,
            "audio_error_code": error_code,
            "waiting_events": 0,
            "stalled_events": 0,
            "profile_name": profile_name,
            "client_name": client_name,
            "user_agent": user_agent,
        }
    )


def _gui_profile_or_default(resolved_profiles: dict[str, Any]) -> dict[str, Any]:
    profile = resolved_profiles.get("gui_profile")
    if profile:
        return profile
    return {
        "id": None,
        "profile_name": "gui_profile",
        "client_name": "NavidromeUI",
        "user_agent": "Chrome/Windows",
        "max_bit_rate": 0,
        "transcoding_id": None,
        "transcoding_name": None,
        "transcoding_target_format": None,
        "transcoding_default_bit_rate": None,
    }


def _run_gui_scenarios(
    *,
    base_url: str,
    username: str,
    password: str,
    selected_tracks: dict[str, dict[str, Any]],
    resolved_profiles: dict[str, Any],
    headless: bool,
    timeout_sec: int,
    results_file: Path,
) -> list[dict[str, Any]]:
    gui_profile = _gui_profile_or_default(resolved_profiles)

    scenarios = [
        GuiScenario("GUI-1", "flac_high", cold_reps=3, warm_reps=2, throttle="none"),
        GuiScenario("GUI-2", "flac_any", cold_reps=3, warm_reps=2, throttle="none"),
        GuiScenario("GUI-3", "lossy_ref", cold_reps=3, warm_reps=2, throttle="none"),
        GuiScenario("GUI-4", "flac_high", cold_reps=3, warm_reps=0, throttle="fast3g"),
        GuiScenario("GUI-5", "lossy_ref", cold_reps=3, warm_reps=0, throttle="fast3g"),
    ]

    collected: list[dict[str, Any]] = []

    for scenario in scenarios:
        track = selected_tracks[scenario.track_key]
        network_profile = FAST_3G if scenario.throttle == "fast3g" else NO_THROTTLE

        for rep in range(1, scenario.cold_reps + 1):
            session = SeleniumProbeSession(
                base_url=base_url,
                username=username,
                password=password,
                headless=headless,
                timeout_sec=timeout_sec,
            )
            try:
                session.start()
                session.apply_network_profile(network_profile)
                row = session.run_track_probe(
                    scenario_id=scenario.scenario_id,
                    track=track,
                    profile_name=str(gui_profile.get("profile_name", "gui_profile")),
                    client_name=str(gui_profile.get("client_name", "NavidromeUI")),
                    user_agent=str(gui_profile.get("user_agent", "Chrome/Windows")),
                    playback_timeout_sec=timeout_sec,
                )
                row["rep_kind"] = "cold"
                row["rep_index"] = rep
            except Exception as exc:
                row = _build_error_row(
                    scenario_id=scenario.scenario_id,
                    track=track,
                    profile_name=str(gui_profile.get("profile_name", "gui_profile")),
                    client_name=str(gui_profile.get("client_name", "NavidromeUI")),
                    user_agent=str(gui_profile.get("user_agent", "Chrome/Windows")),
                    error_code=f"gui_probe_error:{exc.__class__.__name__}",
                )
                row["rep_kind"] = "cold"
                row["rep_index"] = rep
            finally:
                session.close()

            row = _normalize_row(row)
            collected.append(row)
            _append_jsonl(results_file, row)

        if scenario.warm_reps > 0:
            session = SeleniumProbeSession(
                base_url=base_url,
                username=username,
                password=password,
                headless=headless,
                timeout_sec=timeout_sec,
            )
            try:
                session.start()
                session.apply_network_profile(network_profile)
                session.ensure_logged_in()

                for rep in range(1, scenario.warm_reps + 1):
                    try:
                        row = session.run_track_probe(
                            scenario_id=scenario.scenario_id,
                            track=track,
                            profile_name=str(gui_profile.get("profile_name", "gui_profile")),
                            client_name=str(gui_profile.get("client_name", "NavidromeUI")),
                            user_agent=str(gui_profile.get("user_agent", "Chrome/Windows")),
                            playback_timeout_sec=timeout_sec,
                        )
                        row["rep_kind"] = "warm"
                        row["rep_index"] = rep
                    except SeleniumProbeError as exc:
                        row = _build_error_row(
                            scenario_id=scenario.scenario_id,
                            track=track,
                            profile_name=str(gui_profile.get("profile_name", "gui_profile")),
                            client_name=str(gui_profile.get("client_name", "NavidromeUI")),
                            user_agent=str(gui_profile.get("user_agent", "Chrome/Windows")),
                            error_code=f"gui_probe_error:{exc.__class__.__name__}",
                        )
                        row["rep_kind"] = "warm"
                        row["rep_index"] = rep

                    row = _normalize_row(row)
                    collected.append(row)
                    _append_jsonl(results_file, row)
            finally:
                session.close()

    return collected


def _run_api_scenarios(
    *,
    base_url: str,
    auth: dict[str, Any],
    selected_tracks: dict[str, dict[str, Any]],
    resolved_profiles: dict[str, Any],
    timeout_sec: int,
    results_file: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    scenarios = [
        ApiScenario("API-1", mode="gui_profile", reps=3),
        ApiScenario("API-2", mode="app_profile", reps=3),
        ApiScenario("API-3", mode="force_raw", reps=3),
        ApiScenario("API-4", mode="force_transcode", reps=3),
    ]

    notes: list[str] = []
    collected: list[dict[str, Any]] = []

    gui_profile = _gui_profile_or_default(resolved_profiles)
    app_profile = resolved_profiles.get("app_profile")

    for scenario in scenarios:
        if scenario.mode == "gui_profile":
            profile = gui_profile
            extra_params = None
        elif scenario.mode == "app_profile":
            if not app_profile:
                notes.append("API-2 skipped: app-equivalent profile not available")
                continue
            profile = app_profile
            extra_params = None
        elif scenario.mode == "force_raw":
            profile = gui_profile
            extra_params = {"format": "raw"}
        elif scenario.mode == "force_transcode":
            profile = gui_profile
            extra_params = {"format": "opus", "maxBitRate": 128}
        else:
            notes.append(f"Unknown API mode skipped: {scenario.mode}")
            continue

        for track in selected_tracks.values():
            for rep in range(1, scenario.reps + 1):
                try:
                    row = run_stream_probe(
                        base_url=base_url,
                        auth=auth,
                        track=track,
                        scenario_id=scenario.scenario_id,
                        profile_name=str(profile.get("profile_name", scenario.mode)),
                        client_name=str(profile.get("client_name", "PlaybackProbe")),
                        user_agent=str(profile.get("user_agent", "")),
                        player_id=profile.get("id"),
                        extra_params=extra_params,
                        timeout_sec=timeout_sec,
                    )
                except Exception as exc:
                    row = _build_error_row(
                        scenario_id=scenario.scenario_id,
                        track=track,
                        profile_name=str(profile.get("profile_name", scenario.mode)),
                        client_name=str(profile.get("client_name", "PlaybackProbe")),
                        user_agent=str(profile.get("user_agent", "")),
                        error_code=f"api_probe_error:{exc.__class__.__name__}",
                    )

                row["rep_kind"] = "api"
                row["rep_index"] = rep
                row = _normalize_row(row)
                collected.append(row)
                _append_jsonl(results_file, row)

    return collected, notes


def _should_run_playwright_parity(rows: list[dict[str, Any]]) -> bool:
    gui_failures = [
        row
        for row in rows
        if str(row.get("scenario_id", "")).startswith("GUI-")
        and not bool(row.get("start_success"))
        and str(row.get("suffix", "")).lower() == "flac"
    ]
    if not gui_failures:
        return False

    for failure in gui_failures:
        track_id = str(failure.get("track_id", ""))
        healthy_api = [
            row
            for row in rows
            if str(row.get("scenario_id", "")).startswith("API-")
            and str(row.get("track_id", "")) == track_id
            and bool(row.get("start_success"))
            and (row.get("ttfb_ms") is None or float(row.get("ttfb_ms")) < 2000)
        ]
        if healthy_api:
            return True

    return False


def _run_playwright_parity(
    *,
    base_url: str,
    username: str,
    password: str,
    run_dir: Path,
    selected_tracks: dict[str, dict[str, Any]],
) -> int:
    script = THIS_DIR / "playwright_parity.sh"
    flac_title = selected_tracks["flac_high"]["title"]
    lossy_title = selected_tracks["lossy_ref"]["title"]

    cmd = [
        "bash",
        str(script),
        "--base-url",
        base_url,
        "--username",
        username,
        "--password",
        password,
        "--flac-title",
        flac_title,
        "--lossy-title",
        lossy_title,
        "--output-dir",
        str(run_dir / "playwright_parity"),
    ]

    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Navidrome playback investigation matrix")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument(
        "--results-root",
        default=str(THIS_DIR / "results"),
        help="Root output directory. A timestamped run folder will be created under this path.",
    )
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--max-random-batches", type=int, default=60)
    parser.add_argument("--random-batch-size", type=int, default=200)
    parser.add_argument("--headed", action="store_true", help="Run Selenium with visible Chrome")
    parser.add_argument(
        "--playwright-parity",
        choices=["auto", "always", "never"],
        default="auto",
        help="Run optional Playwright parity scenario",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    results_root = Path(args.results_root)
    run_dir = results_root / _timestamp_id()
    run_dir.mkdir(parents=True, exist_ok=True)

    results_file = run_dir / "results.jsonl"
    run_meta: dict[str, Any] = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "username": args.username,
        "timeout_sec": args.timeout_sec,
        "max_random_batches": args.max_random_batches,
        "random_batch_size": args.random_batch_size,
        "headed": bool(args.headed),
        "playwright_parity": args.playwright_parity,
        "notes": [],
    }

    try:
        auth = _preflight_auth(
            base_url=args.base_url,
            username=args.username,
            password=args.password,
            timeout_sec=args.timeout_sec,
        )
        run_meta["auth_preflight"] = "ok"
    except Exception as exc:
        run_meta["auth_preflight"] = f"failed:{exc.__class__.__name__}"
        run_meta["notes"].append(str(exc))
        _write_json(run_dir / "run_meta.json", run_meta)
        raise

    players = _fetch_native_list(
        base_url=args.base_url,
        token=auth["token"],
        resource="player",
        timeout_sec=args.timeout_sec,
        sort_field="lastSeen",
    )
    transcodings = _fetch_native_list(
        base_url=args.base_url,
        token=auth["token"],
        resource="transcoding",
        timeout_sec=args.timeout_sec,
        sort_field="name",
    )

    _write_json(run_dir / "players.json", players)
    _write_json(run_dir / "transcoding.json", transcodings)

    resolved_profiles = resolve_profiles(players, transcodings)
    _write_json(run_dir / "resolved_profiles.json", resolved_profiles)

    try:
        sampled = sample_tracks(
            args.base_url,
            auth,
            max_batches=args.max_random_batches,
            batch_size=args.random_batch_size,
            timeout_sec=args.timeout_sec,
        )
    except TrackSamplingError as exc:
        run_meta["notes"].append(str(exc))
        _write_json(run_dir / "run_meta.json", run_meta)
        raise

    selected_tracks = sampled["selected_tracks"]
    _write_json(run_dir / "selected_tracks.json", sampled)

    gui_rows = _run_gui_scenarios(
        base_url=args.base_url,
        username=args.username,
        password=args.password,
        selected_tracks=selected_tracks,
        resolved_profiles=resolved_profiles,
        headless=not bool(args.headed),
        timeout_sec=args.timeout_sec,
        results_file=results_file,
    )

    api_rows, api_notes = _run_api_scenarios(
        base_url=args.base_url,
        auth=auth,
        selected_tracks=selected_tracks,
        resolved_profiles=resolved_profiles,
        timeout_sec=args.timeout_sec,
        results_file=results_file,
    )
    run_meta["notes"].extend(api_notes)

    all_rows = gui_rows + api_rows

    parity_rc = None
    if args.playwright_parity == "always" or (
        args.playwright_parity == "auto" and _should_run_playwright_parity(all_rows)
    ):
        parity_rc = _run_playwright_parity(
            base_url=args.base_url,
            username=args.username,
            password=args.password,
            run_dir=run_dir,
            selected_tracks=selected_tracks,
        )

    run_meta["playwright_parity_rc"] = parity_rc

    analysis = analyze_results_dir(run_dir)
    run_meta["analysis"] = analysis
    run_meta["completed_at"] = datetime.now(timezone.utc).isoformat()

    _write_json(run_dir / "run_meta.json", run_meta)

    print(json.dumps({"run_dir": str(run_dir), "analysis": analysis}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
