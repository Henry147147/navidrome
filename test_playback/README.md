# Playback Investigation Harness (`test_playback`)

This harness reproduces and diagnoses slow/never-starting playback in Navidrome GUI, with explicit FLAC vs non-FLAC comparison.

## What It Runs
- GUI probe (Selenium + Chrome) for `GUI-1..GUI-5`
- Direct Subsonic stream probe for `API-1..API-4`
- Optional Playwright parity check when Selenium failures occur but API looks healthy

## Outputs
Each run writes to:
- `test_playback/results/<run_id>/results.jsonl`
- `test_playback/results/<run_id>/players.json`
- `test_playback/results/<run_id>/transcoding.json`
- `test_playback/results/<run_id>/resolved_profiles.json`
- `test_playback/results/<run_id>/selected_tracks.json`
- `test_playback/results/<run_id>/scenario_summary.csv`
- `test_playback/results/<run_id>/summary.md`
- `test_playback/results/<run_id>/run_meta.json`

`results.jsonl` rows include these fields:
- `scenario_id`, `track_id`, `suffix`, `bitrate_kbps`, `ttfb_ms`, `time_to_play_ms`, `start_success`, `audio_error_code`, `waiting_events`, `stalled_events`, `http_status`, `content_type`, `accept_ranges`, `content_length`, `transfer_encoding`, `profile_name`, `client_name`, `user_agent`

## Prerequisites
- Python 3.10+
- Google Chrome
- Dependencies:

```bash
pip install -r python_services/requirements.txt
```

Notes:
- Selenium 4.27+ uses Selenium Manager to locate/download a compatible ChromeDriver.
- If your environment blocks driver download, install ChromeDriver manually and ensure it is on `PATH`.

## Run Matrix

```bash
python test_playback/run_matrix.py \
  --base-url http://127.0.0.1:4533 \
  --username henry \
  --password FastMusic
```

Useful flags:
- `--headed` to run visible Chrome
- `--playwright-parity auto|always|never` (default: `auto`)
- `--results-root test_playback/results`
- `--timeout-sec 45`

## Analyze Existing Results

```bash
python test_playback/analyze_results.py --results-dir test_playback/results/<run_id>
```

## Optional Playwright Parity Script

```bash
bash test_playback/playwright_parity.sh \
  --base-url http://127.0.0.1:4533 \
  --username henry \
  --password FastMusic \
  --flac-title "Some FLAC Title" \
  --lossy-title "Some MP3 Title" \
  --output-dir test_playback/results/<run_id>/playwright_parity
```

## Interpretation Guide
- If `API-*` TTFB is high: likely backend stream startup bottleneck.
- If API is healthy but FLAC GUI fails: likely browser/UI playback path issue.
- If `API-1` and `API-2` diverge strongly: likely player-profile/transcoding mismatch.
- If `GUI-4` degrades much more than `GUI-5`: likely bandwidth/bitrate pressure on FLAC path.
