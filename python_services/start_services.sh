#!/usr/bin/env bash
#
# Startup script for Navidrome Python Backend Services
#
# This script starts the following services:
# - Text Embedding Service (port 9003)
# - Recommender API (port 9002)
# - Embedding Socket Server (Unix socket at /tmp/navidrome_embed.sock)
#
# Usage:
#   ./start_services.sh [options]
#
# Options:
#   --text-only         Start only the text embedding service
#   --recommender-only  Start only the recommender API
#   --socket-only       Start only the embedding socket server
#   --no-socket         Start all services except the socket server
#   --help              Show this help message
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default ports (can be overridden by environment variables)
TEXT_EMBEDDING_PORT="${TEXT_EMBEDDING_PORT:-9003}"
NAVIDROME_RECOMMENDER_PORT="${NAVIDROME_RECOMMENDER_PORT:-9002}"

# Log directory
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
mkdir -p "$LOG_DIR"

# PID directory for tracking running services
PID_DIR="${PID_DIR:-$SCRIPT_DIR/.pids}"
mkdir -p "$PID_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service flags
START_TEXT=true
START_RECOMMENDER=true
START_SOCKET=true

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --text-only)
            START_TEXT=true
            START_RECOMMENDER=false
            START_SOCKET=false
            shift
            ;;
        --recommender-only)
            START_TEXT=false
            START_RECOMMENDER=true
            START_SOCKET=false
            shift
            ;;
        --socket-only)
            START_TEXT=false
            START_RECOMMENDER=false
            START_SOCKET=true
            shift
            ;;
        --no-socket)
            START_SOCKET=false
            shift
            ;;
        --help)
            sed -n '2,15p' "$0" | sed 's/^# //; s/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Function to check if a service is already running
is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$pid_file"
        fi
    fi
    return 1
}

# Function to start a service
start_service() {
    local name="$1"
    local script="$2"
    local log_file="$3"
    local pid_file="$4"

    if is_running "$pid_file"; then
        echo -e "${YELLOW}$name is already running (PID: $(cat "$pid_file"))${NC}"
        return 0
    fi

    echo -e "${GREEN}Starting $name...${NC}"
    python3 "$script" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"

    # Wait a moment and check if it's still running
    sleep 2
    if ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name started successfully (PID: $pid)${NC}"
        echo -e "  Log file: $log_file"
        return 0
    else
        echo -e "${RED}✗ $name failed to start. Check log file: $log_file${NC}"
        rm -f "$pid_file"
        return 1
    fi
}

# Main startup sequence
echo "=========================================="
echo "Navidrome Python Backend Services"
echo "=========================================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if required Python packages are installed
echo "Checking Python dependencies..."
if ! python3 -c "import fastapi, uvicorn, pymilvus" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Some required packages may not be installed${NC}"
    echo "Install with: pip install -r requirements.txt"
    echo ""
fi

echo ""

# Start services
FAILED=0

if [[ "$START_TEXT" == true ]]; then
    if ! start_service \
        "Text Embedding Service" \
        "text_embedding_service.py" \
        "$LOG_DIR/text_embedding_service.log" \
        "$PID_DIR/text_embedding_service.pid"; then
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

if [[ "$START_RECOMMENDER" == true ]]; then
    if ! start_service \
        "Recommender API" \
        "recommender_api.py" \
        "$LOG_DIR/recommender_api.log" \
        "$PID_DIR/recommender_api.pid"; then
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

if [[ "$START_SOCKET" == true ]]; then
    if ! start_service \
        "Embedding Socket Server" \
        "python_embed_server.py" \
        "$LOG_DIR/python_embed_server.log" \
        "$PID_DIR/python_embed_server.pid"; then
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# Summary
echo "=========================================="
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All services started successfully!${NC}"
    echo ""
    echo "Service endpoints:"
    [[ "$START_TEXT" == true ]] && echo "  - Text Embedding Service: http://localhost:$TEXT_EMBEDDING_PORT"
    [[ "$START_RECOMMENDER" == true ]] && echo "  - Recommender API: http://localhost:$NAVIDROME_RECOMMENDER_PORT"
    [[ "$START_SOCKET" == true ]] && echo "  - Embedding Socket Server: /tmp/navidrome_embed.sock"
    echo ""
    echo "To stop services, run: ./stop_services.sh"
    echo "To view logs: tail -f $LOG_DIR/*.log"
else
    echo -e "${RED}$FAILED service(s) failed to start${NC}"
    echo "Check the log files in $LOG_DIR for details"
    exit 1
fi
