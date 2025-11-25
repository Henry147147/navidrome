#!/usr/bin/env bash
# Display status of the unified Navidrome Python service

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR="${PID_DIR:-$SCRIPT_DIR/.pids}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_FILE="$PID_DIR/navidrome_service.pid"
LOG_FILE="$LOG_DIR/navidrome_service.log"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "Navidrome Python Service Status"
echo "=========================================="

if [[ ! -f "$PID_FILE" ]]; then
    echo -e "${YELLOW}Service not running (no PID file).${NC}"
    exit 0
fi

pid=$(cat "$PID_FILE")
if ps -p "$pid" >/dev/null 2>&1; then
    started=$(ps -p "$pid" -o lstart=)
    echo -e "${GREEN}Service running${NC}"
    echo "  PID     : $pid"
    echo "  Started : $started"
    echo "  Log     : $LOG_FILE"
else
    echo -e "${RED}Stale PID file found (PID $pid not running).${NC}"
fi
