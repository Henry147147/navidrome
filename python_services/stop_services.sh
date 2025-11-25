#!/usr/bin/env bash
# Stop the unified Navidrome Python service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR="${PID_DIR:-$SCRIPT_DIR/.pids}"
PID_FILE="$PID_DIR/navidrome_service.pid"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [[ ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}Service is not running (no PID file).${NC}"
    exit 0
fi

pid=$(cat "$PID_FILE")
if ! ps -p "$pid" >/dev/null 2>&1; then
    echo -e "${YELLOW}Service not running (stale PID). Removing PID file.${NC}"
    rm -f "$PID_FILE"
    exit 0
fi

echo -e "${GREEN}Stopping service (PID: $pid)...${NC}"
kill -TERM "$pid" 2>/dev/null || true

for _ in {1..10}; do
    if ! ps -p "$pid" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Service stopped${NC}"
        rm -f "$PID_FILE"
        exit 0
    fi
    sleep 1
done

echo -e "${YELLOW}Force stopping service...${NC}"
kill -KILL "$pid" 2>/dev/null || true
sleep 1

if ps -p "$pid" >/dev/null 2>&1; then
    echo -e "${RED}✗ Failed to stop service${NC}"
    exit 1
fi

rm -f "$PID_FILE"
echo -e "${GREEN}✓ Service stopped${NC}"
