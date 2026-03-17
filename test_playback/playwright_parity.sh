#!/usr/bin/env bash
set -euo pipefail

BASE_URL=""
USERNAME=""
PASSWORD=""
FLAC_TITLE=""
LOSSY_TITLE=""
OUTPUT_DIR=""
HEADED="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --username)
      USERNAME="$2"
      shift 2
      ;;
    --password)
      PASSWORD="$2"
      shift 2
      ;;
    --flac-title)
      FLAC_TITLE="$2"
      shift 2
      ;;
    --lossy-title)
      LOSSY_TITLE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --headed)
      HEADED="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$USERNAME" || -z "$PASSWORD" || -z "$FLAC_TITLE" || -z "$LOSSY_TITLE" || -z "$OUTPUT_DIR" ]]; then
  echo "Missing required arguments" >&2
  echo "Usage: $0 --base-url URL --username USER --password PASS --flac-title TITLE --lossy-title TITLE --output-dir DIR [--headed]" >&2
  exit 1
fi

if ! command -v npx >/dev/null 2>&1; then
  echo "npx is required for playwright-cli" >&2
  exit 2
fi

export CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
export PWCLI="$CODEX_HOME/skills/playwright/scripts/playwright_cli.sh"

if [[ ! -x "$PWCLI" ]]; then
  echo "Playwright wrapper not found: $PWCLI" >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"
LOG_FILE="$OUTPUT_DIR/playwright_parity.log"
RESULT_FILE="$OUTPUT_DIR/parity_result.json"

OPEN_ARGS=(open "$BASE_URL")
if [[ "$HEADED" == "true" ]]; then
  OPEN_ARGS+=(--headed)
fi
"$PWCLI" "${OPEN_ARGS[@]}" >/dev/null

JS_CODE=$(cat <<'JS'
const base = (process.env.PW_BASE_URL || '').replace(/\/$/, '');
const username = process.env.PW_USERNAME || '';
const password = process.env.PW_PASSWORD || '';
const flacTitle = process.env.PW_FLAC_TITLE || '';
const lossyTitle = process.env.PW_LOSSY_TITLE || '';

const runOne = async (title) => {
  await page.goto(base + '/app/#/song?filter=' + encodeURIComponent(JSON.stringify({ title: [title], missing: false })) + '&sort=title&order=ASC&perPage=15');
  await page.waitForSelector('tbody tr', { timeout: 15000 });

  await page.evaluate(() => {
    if (!window.__ndPlaywrightProbe) {
      window.__ndPlaywrightProbe = { clickAt: null, playingAt: null, errorCode: null, events: [] };
    }
    const state = window.__ndPlaywrightProbe;
    state.clickAt = null;
    state.playingAt = null;
    state.errorCode = null;
    state.events = [];

    const names = ['loadstart', 'loadedmetadata', 'canplay', 'play', 'playing', 'waiting', 'stalled', 'error'];
    const attach = () => {
      const audio = document.querySelector('audio.music-player-audio') || document.querySelector('audio');
      if (!audio || audio.__ndPWProbeAttached) {
        return;
      }
      audio.__ndPWProbeAttached = true;
      for (const name of names) {
        audio.addEventListener(name, () => {
          const code = audio.error && typeof audio.error.code === 'number' ? audio.error.code : null;
          const evt = { name, t: performance.now(), errorCode: code };
          state.events.push(evt);
          if (name === 'playing' && state.playingAt === null) {
            state.playingAt = evt.t;
          }
          if (name === 'error') {
            state.errorCode = code;
          }
        }, { passive: true });
      }
    };

    attach();
    if (!window.__ndPWObserver) {
      window.__ndPWObserver = new MutationObserver(() => attach());
      window.__ndPWObserver.observe(document.documentElement, { childList: true, subtree: true });
    }
  });

  const rows = page.locator('tbody tr');
  const count = await rows.count();
  const needle = title.toLowerCase();

  let clicked = false;
  for (let i = 0; i < count; i++) {
    const text = (await rows.nth(i).innerText()).toLowerCase();
    if (needle && text.includes(needle)) {
      await page.evaluate(() => { window.__ndPlaywrightProbe.clickAt = performance.now(); });
      await rows.nth(i).click();
      clicked = true;
      break;
    }
  }

  if (!clicked && count > 0) {
    await page.evaluate(() => { window.__ndPlaywrightProbe.clickAt = performance.now(); });
    await rows.first().click();
    clicked = true;
  }

  const start = Date.now();
  while ((Date.now() - start) < 25000) {
    const done = await page.evaluate(() => {
      const p = window.__ndPlaywrightProbe || {};
      return p.playingAt !== null || p.errorCode !== null;
    });
    if (done) {
      break;
    }
    await page.waitForTimeout(250);
  }

  const probe = await page.evaluate(() => window.__ndPlaywrightProbe || null);
  const timeToPlay = (probe && probe.clickAt !== null && probe.playingAt !== null)
    ? Math.round((probe.playingAt - probe.clickAt) * 1000) / 1000
    : null;

  return {
    title,
    clicked,
    start_success: !!(probe && probe.playingAt !== null),
    audio_error_code: probe ? probe.errorCode : null,
    time_to_play_ms: timeToPlay,
    waiting_events: probe ? probe.events.filter((x) => x.name === 'waiting').length : 0,
    stalled_events: probe ? probe.events.filter((x) => x.name === 'stalled').length : 0,
  };
};

await page.goto(base + '/app/#/login');
await page.locator('input[name="username"]').fill(username);
await page.locator('input[name="password"]').fill(password);
await page.getByRole('button', { name: 'Sign in' }).click();
await page.waitForURL((url) => !url.toString().includes('/login'), { timeout: 20000 });

const flac = await runOne(flacTitle);
const lossy = await runOne(lossyTitle);

console.log('PLAYWRIGHT_PARITY_RESULT=' + JSON.stringify({ flac, lossy }));
JS
)

PW_BASE_URL="$BASE_URL" \
PW_USERNAME="$USERNAME" \
PW_PASSWORD="$PASSWORD" \
PW_FLAC_TITLE="$FLAC_TITLE" \
PW_LOSSY_TITLE="$LOSSY_TITLE" \
"$PWCLI" run-code "$JS_CODE" | tee "$LOG_FILE"

RESULT_LINE=$(grep -E 'PLAYWRIGHT_PARITY_RESULT=' "$LOG_FILE" | tail -n 1 || true)
if [[ -z "$RESULT_LINE" ]]; then
  echo '{"error":"no_result_line"}' > "$RESULT_FILE"
else
  printf '%s\n' "${RESULT_LINE#PLAYWRIGHT_PARITY_RESULT=}" > "$RESULT_FILE"
fi

"$PWCLI" close-all >/dev/null || true

echo "Playwright parity output: $RESULT_FILE"
