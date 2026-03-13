#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${ND_BASE_URL:-http://127.0.0.1:4500}"
API_URL="${BASE_URL%/}/api"
TOKEN="${ND_TOKEN:-}"
CLIENT_ID="${ND_CLIENT_ID:-smoke-test}"
SEED_TRACK_ID="${ND_SEED_TRACK_ID:-}"
TEXT_PROMPT="${ND_TEXT_PROMPT:-late night jazz with warm bass}"

if [[ -z "${TOKEN}" ]]; then
  echo "ND_TOKEN is required" >&2
  exit 1
fi

auth_headers=(
  -H "Content-Type: application/json"
  -H "X-ND-Authorization: Bearer ${TOKEN}"
  -H "X-ND-Client-Unique-Id: ${CLIENT_ID}"
)

call_json() {
  local method="$1"
  local path="$2"
  local body="${3:-}"
  if [[ -n "${body}" ]]; then
    curl -fsS -X "${method}" "${API_URL}${path}" "${auth_headers[@]}" -d "${body}"
  else
    curl -fsS -X "${method}" "${API_URL}${path}" "${auth_headers[@]}"
  fi
}

validate_success() {
  local label="$1"
  local payload
  payload="$(cat)"
  PAYLOAD="${payload}" python3 - "$label" <<'PY'
import json
import os
import re
import sys

label = sys.argv[1]
payload = json.loads(os.environ["PAYLOAD"])

if payload.get("resultSource") != "semantic":
    raise SystemExit(f"{label}: resultSource was not semantic")
if payload.get("degraded") is True:
    raise SystemExit(f"{label}: response was degraded")
track_ids = payload.get("trackIds") or []
if not track_ids:
    raise SystemExit(f"{label}: no trackIds returned")

raw_warning_patterns = [
    re.compile(r"schema mismatch", re.I),
    re.compile(r"recommendation service unavailable", re.I),
    re.compile(r"connection refused", re.I),
]
for warning in payload.get("warnings") or []:
    if any(pattern.search(str(warning)) for pattern in raw_warning_patterns):
        raise SystemExit(f"{label}: raw infrastructure warning leaked to response: {warning}")

print(f"{label}: ok ({len(track_ids)} tracks)")
PY
}

echo "Checking recommendation health"
health_payload="$(call_json GET /recommendations/health)"
PAYLOAD="${health_payload}" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["PAYLOAD"])
if payload.get("status") != "ready":
    raise SystemExit(f"health not ready: {payload}")
print("health: ready")
PY

echo "Checking recent recommendations"
recent_payload="$(call_json POST /recommendations/recent '{"limit":5}')"
validate_success "recent" <<<"${recent_payload}"

echo "Checking discovery recommendations"
discovery_payload="$(call_json POST /recommendations/discovery '{"limit":5}')"
validate_success "discovery" <<<"${discovery_payload}"

if [[ -n "${SEED_TRACK_ID}" ]]; then
  echo "Checking custom recommendations"
  custom_payload="$(call_json POST /recommendations/custom "{\"limit\":5,\"songIds\":[\"${SEED_TRACK_ID}\"]}")"
  validate_success "custom" <<<"${custom_payload}"
else
  echo "Skipping custom recommendations because ND_SEED_TRACK_ID is not set"
fi

echo "Checking text recommendations"
text_payload="$(call_json POST /recommendations/text "$(printf '{"limit":5,"text":"%s"}' "${TEXT_PROMPT//\"/\\\"}")")"
validate_success "text" <<<"${text_payload}"

echo "Recommendation smoke test passed"
