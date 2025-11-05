#!/usr/bin/env bash
#
# Stop script for Navidrome Python Backend Services
#
# Usage:
#   ./stop_services.sh [options]
#
# Options:
#   --text              Stop only the text embedding service
#   --recommender       Stop only the recommender API
#   --socket            Stop only the embedding socket server
#   --help              Show this help message
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# PID directory
PID_DIR="${PID_DIR:-$SCRIPT_DIR/.pids}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Service flags
STOP_TEXT=true
STOP_RECOMMENDER=true
STOP_SOCKET=true

# Parse command-line arguments
if [[ $# -gt 0 ]]; then
    STOP_TEXT=false
    STOP_RECOMMENDER=false
    STOP_SOCKET=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --text)
                STOP_TEXT=true
                shift
                ;;
            --recommender)
                STOP_RECOMMENDER=true
                shift
                ;;
            --socket)
                STOP_SOCKET=true
                shift
                ;;
            --help)
                sed -n '2,13p' "$0" | sed 's/^# //; s/^#//'
                exit 0
                ;;
            *)
                echo -e "${RED}Error: Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
fi

# Function to stop a service
stop_service() {
    local name="$1"
    local pid_file="$2"

    if [[ ! -f "$pid_file" ]]; then
        echo -e "${YELLOW}$name is not running (no PID file)${NC}"
        return 0
    fi

    local pid=$(cat "$pid_file")

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${YELLOW}$name is not running (stale PID file)${NC}"
        rm -f "$pid_file"
        return 0
    fi

    echo -e "${GREEN}Stopping $name (PID: $pid)...${NC}"

    # Try graceful shutdown first
    kill -TERM "$pid" 2>/dev/null || true

    # Wait up to 10 seconds for graceful shutdown
    for i in {1..10}; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ $name stopped successfully${NC}"
            rm -f "$pid_file"
            return 0
        fi
        sleep 1
    done

    # Force kill if still running
    echo -e "${YELLOW}Service did not stop gracefully, forcing...${NC}"
    kill -KILL "$pid" 2>/dev/null || true
    sleep 1

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $name stopped (forced)${NC}"
        rm -f "$pid_file"
        return 0
    else
        echo -e "${RED}✗ Failed to stop $name${NC}"
        return 1
    fi
}

# Main stop sequence
echo "=========================================="
echo "Stopping Navidrome Python Backend Services"
echo "=========================================="
echo ""

FAILED=0

if [[ "$STOP_TEXT" == true ]]; then
    if ! stop_service \
        "Text Embedding Service" \
        "$PID_DIR/text_embedding_service.pid"; then
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

if [[ "$STOP_RECOMMENDER" == true ]]; then
    if ! stop_service \
        "Recommender API" \
        "$PID_DIR/recommender_api.pid"; then
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

if [[ "$STOP_SOCKET" == true ]]; then
    if ! stop_service \
        "Embedding Socket Server" \
        "$PID_DIR/python_embed_server.pid"; then
        FAILED=$((FAILED + 1))
    fi
    echo ""
fi

# Clean up socket file if it exists
if [[ -S "/tmp/navidrome_embed.sock" ]]; then
    echo "Cleaning up socket file..."
    rm -f /tmp/navidrome_embed.sock
    echo -e "${GREEN}✓ Socket file removed${NC}"
    echo ""
fi

# Summary
echo "=========================================="
if [[ $FAILED -eq 0 ]]; then
    echo -e "${GREEN}All services stopped successfully!${NC}"
else
    echo -e "${RED}$FAILED service(s) failed to stop${NC}"
    exit 1
fi
