#!/usr/bin/env python3
"""Analyze playback probe results and produce summary artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


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


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).lower() in {"true", "1", "yes"}


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 3)


def _pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def load_results(results_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with results_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def summarize_by_scenario(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("scenario_id", "unknown"))].append(row)

    summaries: list[dict[str, Any]] = []
    for scenario_id, items in sorted(grouped.items()):
        starts = [_to_bool(item.get("start_success")) for item in items]
        successes = sum(1 for s in starts if s)
        failures = len(items) - successes

        ttfb_values = [
            value
            for value in (_to_float(item.get("ttfb_ms")) for item in items)
            if value is not None
        ]
        ttp_values = [
            value
            for value in (_to_float(item.get("time_to_play_ms")) for item in items)
            if value is not None
        ]

        waiting_total = sum(int(item.get("waiting_events") or 0) for item in items)
        stalled_total = sum(int(item.get("stalled_events") or 0) for item in items)

        summaries.append(
            {
                "scenario_id": scenario_id,
                "runs": len(items),
                "successes": successes,
                "failures": failures,
                "failure_rate_pct": _pct(failures, len(items)),
                "ttfb_ms_median": _safe_median(ttfb_values),
                "time_to_play_ms_median": _safe_median(ttp_values),
                "waiting_events_total": waiting_total,
                "stalled_events_total": stalled_total,
            }
        )

    return summaries


def _pick_likely_causes(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    gui = [r for r in rows if str(r.get("scenario_id", "")).startswith("GUI-")]
    api = [r for r in rows if str(r.get("scenario_id", "")).startswith("API-")]

    gui_flac = [r for r in gui if str(r.get("suffix", "")).lower() == "flac"]
    gui_lossy = [r for r in gui if str(r.get("suffix", "")).lower() in {"mp3", "aac", "opus", "m4a"}]

    def success_rate(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(1 for x in items if _to_bool(x.get("start_success"))) / len(items)

    def median_ttfb(items: list[dict[str, Any]]) -> float | None:
        vals = [_to_float(x.get("ttfb_ms")) for x in items]
        clean = [v for v in vals if v is not None]
        return _safe_median(clean)

    causes: list[dict[str, str]] = []

    api_success = success_rate(api)
    gui_flac_success = success_rate(gui_flac)
    gui_lossy_success = success_rate(gui_lossy)

    api_ttfb = median_ttfb(api)
    gui_ttfb = median_ttfb(gui)

    api1 = [r for r in api if str(r.get("scenario_id")) == "API-1"]
    api2 = [r for r in api if str(r.get("scenario_id")) == "API-2"]
    api1_sr = success_rate(api1)
    api2_sr = success_rate(api2)
    api1_t = median_ttfb(api1)
    api2_t = median_ttfb(api2)

    gui4 = [r for r in rows if str(r.get("scenario_id")) == "GUI-4"]
    gui5 = [r for r in rows if str(r.get("scenario_id")) == "GUI-5"]
    gui4_sr = success_rate(gui4)
    gui5_sr = success_rate(gui5)

    if api1 and api2 and (
        abs(api1_sr - api2_sr) >= 0.25
        or (
            api1_t is not None
            and api2_t is not None
            and max(api1_t, api2_t) >= 2 * max(min(api1_t, api2_t), 1.0)
        )
    ):
        causes.append(
            {
                "cause": "Transcoding/profile mismatch",
                "confidence": "high",
                "signal": "Large API-1 vs API-2 success/TTFB divergence indicates player-profile configuration differences.",
            }
        )

    if api_ttfb is not None and api_ttfb > 2000:
        causes.append(
            {
                "cause": "Server TTFB bottleneck",
                "confidence": "high",
                "signal": "Direct API probe median TTFB exceeds 2000ms, indicating backend stream startup delay.",
            }
        )

    if api_success >= 0.8 and gui_flac and gui_flac_success < 0.6:
        causes.append(
            {
                "cause": "Browser decode/playback issue",
                "confidence": "medium",
                "signal": "API stream probes are healthy while GUI FLAC starts frequently fail.",
            }
        )

    if api_success >= 0.8 and gui and gui_ttfb is not None and gui_ttfb < 1500:
        gui_fail_silent = [
            r
            for r in gui
            if not _to_bool(r.get("start_success")) and not r.get("audio_error_code")
        ]
        if gui_fail_silent:
            causes.append(
                {
                    "cause": "Frontend player state issue",
                    "confidence": "medium",
                    "signal": "GUI failures occur without audio error code and with acceptable stream TTFB.",
                }
            )

    if gui4 and gui5 and gui5_sr > gui4_sr and (gui5_sr - gui4_sr) >= 0.25:
        causes.append(
            {
                "cause": "Bandwidth/format pressure",
                "confidence": "medium",
                "signal": "Under Fast-3G throttle, FLAC scenarios regress more than lossy controls.",
            }
        )

    if not causes:
        causes.append(
            {
                "cause": "Insufficient evidence",
                "confidence": "low",
                "signal": "Current run does not clearly isolate one dominant bottleneck class.",
            }
        )

    return causes


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze_results_dir(results_dir: Path) -> dict[str, Any]:
    results_file = results_dir / "results.jsonl"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    rows = load_results(results_file)
    scenario_summary = summarize_by_scenario(rows)
    likely_causes = _pick_likely_causes(rows)

    _write_csv(results_dir / "scenario_summary.csv", scenario_summary)

    summary_lines = [
        "# Playback Investigation Summary",
        "",
        f"- Total observations: {len(rows)}",
        f"- GUI observations: {sum(1 for r in rows if str(r.get('scenario_id', '')).startswith('GUI-'))}",
        f"- API observations: {sum(1 for r in rows if str(r.get('scenario_id', '')).startswith('API-'))}",
        "",
        "## Scenario Metrics",
        "",
        "| Scenario | Runs | Successes | Failures | Failure % | Median TTFB (ms) | Median Time-to-Play (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in scenario_summary:
        summary_lines.append(
            "| {scenario_id} | {runs} | {successes} | {failures} | {failure_rate_pct} | {ttfb_ms_median} | {time_to_play_ms_median} |".format(
                **row
            )
        )

    summary_lines.extend(
        [
            "",
            "## Ranked Likely Causes",
            "",
        ]
    )

    for idx, cause in enumerate(likely_causes, start=1):
        summary_lines.append(
            f"{idx}. **{cause['cause']}** ({cause['confidence']}) - {cause['signal']}"
        )

    summary_lines.append("")

    (results_dir / "summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (results_dir / "likely_causes.json").write_text(
        json.dumps(likely_causes, indent=2),
        encoding="utf-8",
    )

    return {
        "rows": len(rows),
        "scenario_summary": scenario_summary,
        "likely_causes": likely_causes,
        "summary_path": str(results_dir / "summary.md"),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze playback probe results")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to run output directory containing results.jsonl",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    result = analyze_results_dir(Path(args.results_dir))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
