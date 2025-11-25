#!/usr/bin/env bash
#
# Start the unified Navidrome Python service (text + recommender + embed)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/.pids}"
mkdir -p "$LOG_DIR" "$PID_DIR"

SERVICE_SCRIPT="navidrome_service.py"
LOG_FILE="$LOG_DIR/navidrome_service.log"
PID_FILE="$PID_DIR/navidrome_service.pid"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

is_running() {
    [[ -f "$PID_FILE" ]] && ps -p "$(cat "$PID_FILE")" >/dev/null 2>&1
}

if is_running; then
    echo -e "${YELLOW}Unified service already running (PID $(cat "$PID_FILE"))${NC}"
    exit 0
fi

echo -e "${GREEN}Starting Navidrome unified service...${NC}"
python3 "$SERVICE_SCRIPT" >"$LOG_FILE" 2>&1 &
pid=$!
echo $pid >"$PID_FILE"

sleep 2
if ps -p "$pid" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Service started (PID: $pid)${NC}"
    echo "  Log file: $LOG_FILE"
else
    echo -e "${RED}✗ Failed to start service. See log: $LOG_FILE${NC}"
    rm -f "$PID_FILE"
    exit 1
fi
