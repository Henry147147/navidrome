#!/usr/bin/env python3
"""Selenium-based GUI playback probe for Navidrome."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


AUDIO_PROBE_JS = r"""
(() => {
  if (!window.__ndPlaybackProbe) {
    window.__ndPlaybackProbe = {
      installedAt: performance.now(),
      clickAt: null,
      playingAt: null,
      errorCode: null,
      events: [],
      attached: false,
    };
  }

  const state = window.__ndPlaybackProbe;
  const eventNames = [
    'loadstart', 'loadedmetadata', 'loadeddata', 'canplay', 'canplaythrough',
    'play', 'playing', 'pause', 'waiting', 'stalled', 'suspend', 'abort',
    'error', 'emptied', 'progress', 'timeupdate', 'ended'
  ];

  const attach = (audio) => {
    if (!audio || audio.__ndProbeAttached) {
      return;
    }
    audio.__ndProbeAttached = true;
    state.attached = true;

    const push = (name) => {
      const err = audio.error || null;
      const code = err && typeof err.code === 'number' ? err.code : null;
      const evt = {
        name,
        t: performance.now(),
        readyState: audio.readyState,
        networkState: audio.networkState,
        currentTime: audio.currentTime,
        errorCode: code,
      };
      state.events.push(evt);

      if (name === 'playing' && state.playingAt === null) {
        state.playingAt = evt.t;
      }
      if (name === 'error') {
        state.errorCode = code;
      }
    };

    for (const name of eventNames) {
      audio.addEventListener(name, () => push(name), { passive: true });
    }
  };

  const findAudio = () =>
    document.querySelector('audio.music-player-audio') || document.querySelector('audio');

  const candidate = findAudio();
  if (candidate) {
    attach(candidate);
  }

  if (!window.__ndPlaybackProbeObserver) {
    const observer = new MutationObserver(() => {
      const audio = findAudio();
      if (audio) {
        attach(audio);
      }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
    window.__ndPlaybackProbeObserver = observer;
  }
})();
"""


@dataclass
class NetworkProfile:
    name: str
    offline: bool
    latency_ms: int
    download_bps: int
    upload_bps: int
    connection_type: str


FAST_3G = NetworkProfile(
    name="fast3g",
    offline=False,
    latency_ms=150,
    download_bps=1_600 * 1024 // 8,
    upload_bps=750 * 1024 // 8,
    connection_type="cellular3g",
)

NO_THROTTLE = NetworkProfile(
    name="none",
    offline=False,
    latency_ms=0,
    download_bps=-1,
    upload_bps=-1,
    connection_type="none",
)


class SeleniumProbeError(RuntimeError):
    pass


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_stream_request(
    logs: list[dict[str, Any]],
    track_id: str,
) -> dict[str, Any]:
    requests: dict[str, dict[str, Any]] = {}

    for entry in logs:
        try:
            message = json.loads(entry["message"])["message"]
        except (KeyError, ValueError, TypeError):
            continue

        method = message.get("method")
        params = message.get("params", {})
        request_id = params.get("requestId")

        if method == "Network.requestWillBeSent":
            request = params.get("request", {})
            url = str(request.get("url", ""))
            if "/rest/stream" not in url:
                continue

            requests[request_id] = {
                "url": url,
                "request_ts": params.get("timestamp"),
            }
            continue

        if method == "Network.responseReceived" and request_id in requests:
            response = params.get("response", {})
            requests[request_id]["response_ts"] = params.get("timestamp")
            requests[request_id]["status"] = response.get("status")
            requests[request_id]["headers"] = response.get("headers", {})
            continue

    if not requests:
        return {
            "ttfb_ms": None,
            "http_status": None,
            "content_type": None,
            "accept_ranges": None,
            "content_length": None,
            "transfer_encoding": None,
            "matched_stream_url": None,
        }

    def is_track_match(req: dict[str, Any]) -> bool:
        try:
            query = parse_qs(urlparse(req["url"]).query)
        except Exception:
            return False
        return track_id in query.get("id", [])

    candidates = [r for r in requests.values() if is_track_match(r)]
    if not candidates:
        candidates = list(requests.values())

    candidates.sort(key=lambda item: item.get("response_ts") or 0, reverse=True)
    selected = candidates[0]

    request_ts = selected.get("request_ts")
    response_ts = selected.get("response_ts")
    ttfb_ms = None
    if isinstance(request_ts, (int, float)) and isinstance(response_ts, (int, float)):
        ttfb_ms = round((response_ts - request_ts) * 1000.0, 3)

    headers = selected.get("headers", {}) or {}

    def header(name: str) -> str | None:
        wanted = name.lower()
        for key, value in headers.items():
            if str(key).lower() == wanted:
                return str(value)
        return None

    return {
        "ttfb_ms": ttfb_ms,
        "http_status": _safe_int(selected.get("status"), 0) or None,
        "content_type": header("content-type"),
        "accept_ranges": header("accept-ranges"),
        "content_length": header("content-length"),
        "transfer_encoding": header("transfer-encoding"),
        "matched_stream_url": selected.get("url"),
    }


class SeleniumProbeSession:
    def __init__(
        self,
        *,
        base_url: str,
        username: str,
        password: str,
        headless: bool,
        timeout_sec: int,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.headless = headless
        self.timeout_sec = timeout_sec
        self.driver: webdriver.Chrome | None = None

    def start(self) -> None:
        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--window-size=1600,1000")
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.set_capability("goog:loggingPrefs", {"performance": "ALL", "browser": "ALL"})

        self.driver = webdriver.Chrome(options=options)
        self.driver.set_page_load_timeout(self.timeout_sec)
        self.driver.execute_cdp_cmd("Network.enable", {})

    def close(self) -> None:
        if self.driver is not None:
            self.driver.quit()
            self.driver = None

    def _require_driver(self) -> webdriver.Chrome:
        if self.driver is None:
            raise SeleniumProbeError("Selenium session not started")
        return self.driver

    def apply_network_profile(self, profile: NetworkProfile) -> None:
        driver = self._require_driver()
        driver.execute_cdp_cmd(
            "Network.emulateNetworkConditions",
            {
                "offline": profile.offline,
                "latency": profile.latency_ms,
                "downloadThroughput": profile.download_bps,
                "uploadThroughput": profile.upload_bps,
                "connectionType": profile.connection_type,
            },
        )

    def ensure_logged_in(self) -> None:
        driver = self._require_driver()

        driver.get(f"{self.base_url}/app/#/login")

        if "/login" in driver.current_url:
            wait = WebDriverWait(driver, self.timeout_sec)
            username_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="username"]'))
            )
            password_input = wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'input[name="password"]'))
            )

            username_input.clear()
            username_input.send_keys(self.username)
            password_input.clear()
            password_input.send_keys(self.password)

            sign_in = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Sign in')]") )
            )
            sign_in.click()

            try:
                wait.until(lambda d: "/login" not in d.current_url)
            except TimeoutException as exc:
                raise SeleniumProbeError("GUI login failed or timed out") from exc

    def _go_to_song_filter(self, track: dict[str, Any]) -> None:
        driver = self._require_driver()
        title = str(track.get("title", ""))
        if not title:
            raise SeleniumProbeError("Track title is required for GUI probe")

        filter_payload = {"title": [title], "missing": False}
        encoded_filter = quote(json.dumps(filter_payload, separators=(",", ":")))
        song_url = (
            f"{self.base_url}/app/#/song"
            f"?filter={encoded_filter}&sort=title&order=ASC&perPage=15"
        )
        driver.get(song_url)

        WebDriverWait(driver, self.timeout_sec).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tbody tr"))
        )

    def _find_row_for_track(self, track: dict[str, Any]) -> Any:
        driver = self._require_driver()
        title = str(track.get("title", "")).strip().lower()

        def _locator(drv: webdriver.Chrome) -> Any:
            rows = drv.find_elements(By.CSS_SELECTOR, "tbody tr")
            if not rows:
                return False

            for row in rows:
                text = row.text.strip().lower()
                if title and title in text:
                    return row
            return rows[0]

        return WebDriverWait(driver, self.timeout_sec).until(_locator)

    def _collect_probe_result(self) -> dict[str, Any]:
        driver = self._require_driver()
        payload = driver.execute_script("return window.__ndPlaybackProbe || null;")
        if not isinstance(payload, dict):
            return {
                "clickAt": None,
                "playingAt": None,
                "errorCode": None,
                "events": [],
                "attached": False,
            }
        payload.setdefault("events", [])
        return payload

    def run_track_probe(
        self,
        *,
        scenario_id: str,
        track: dict[str, Any],
        profile_name: str,
        client_name: str,
        user_agent: str,
        playback_timeout_sec: int,
    ) -> dict[str, Any]:
        driver = self._require_driver()
        self.ensure_logged_in()
        self._go_to_song_filter(track)

        driver.execute_script(AUDIO_PROBE_JS)
        driver.execute_script("window.__ndPlaybackProbe.events = []; window.__ndPlaybackProbe.playingAt = null; window.__ndPlaybackProbe.errorCode = null;")

        # Flush any stale performance events before this attempt.
        _ = driver.get_log("performance")

        row = self._find_row_for_track(track)
        driver.execute_script("window.__ndPlaybackProbe.clickAt = performance.now();")
        row.click()

        start_wait = time.time()
        while True:
            probe = self._collect_probe_result()
            if probe.get("playingAt") is not None or probe.get("errorCode") is not None:
                break
            if (time.time() - start_wait) > playback_timeout_sec:
                break
            time.sleep(0.25)

        # Give the network log stream a short settle window.
        time.sleep(0.8)
        perf_logs = driver.get_log("performance")
        stream = _parse_stream_request(perf_logs, str(track["id"]))

        probe = self._collect_probe_result()
        events = probe.get("events", [])

        waiting_events = sum(1 for evt in events if evt.get("name") == "waiting")
        stalled_events = sum(1 for evt in events if evt.get("name") == "stalled")

        click_at = probe.get("clickAt")
        playing_at = probe.get("playingAt")
        time_to_play_ms = None
        if isinstance(click_at, (int, float)) and isinstance(playing_at, (int, float)):
            time_to_play_ms = round(playing_at - click_at, 3)

        start_success = playing_at is not None

        return {
            "scenario_id": scenario_id,
            "track_id": str(track["id"]),
            "suffix": track.get("suffix"),
            "bitrate_kbps": _safe_int(track.get("bitrate_kbps"), 0),
            "ttfb_ms": stream.get("ttfb_ms"),
            "time_to_play_ms": time_to_play_ms,
            "start_success": bool(start_success),
            "audio_error_code": probe.get("errorCode"),
            "waiting_events": waiting_events,
            "stalled_events": stalled_events,
            "http_status": stream.get("http_status"),
            "content_type": stream.get("content_type"),
            "accept_ranges": stream.get("accept_ranges"),
            "content_length": stream.get("content_length"),
            "transfer_encoding": stream.get("transfer_encoding"),
            "profile_name": profile_name,
            "client_name": client_name,
            "user_agent": user_agent,
            "matched_stream_url": stream.get("matched_stream_url"),
            "audio_events": events,
        }
