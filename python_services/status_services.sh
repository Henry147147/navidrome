#!/usr/bin/env bash
#
# Status script for Navidrome Python Backend Services
#
# Shows the current status of all services
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# PID directory
PID_DIR="${PID_DIR:-$SCRIPT_DIR/.pids}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check service status
check_service() {
    local name="$1"
    local pid_file="$2"
    local port="$3"

    echo -n "  $name: "

    if [[ ! -f "$pid_file" ]]; then
        echo -e "${RED}NOT RUNNING${NC} (no PID file)"
        return 1
    fi

    local pid=$(cat "$pid_file")

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}NOT RUNNING${NC} (stale PID file)"
        return 1
    fi

    # Get process info
    local cmd=$(ps -p "$pid" -o command= 2>/dev/null || echo "unknown")
    local runtime=$(ps -p "$pid" -o etime= 2>/dev/null | xargs || echo "unknown")

    echo -e "${GREEN}RUNNING${NC}"
    echo "    PID: $pid"
    echo "    Runtime: $runtime"
    if [[ -n "$port" ]]; then
        echo "    Port: $port"
        # Check if port is listening
        if command -v netstat &> /dev/null; then
            if netstat -ln 2>/dev/null | grep -q ":$port "; then
                echo -e "    Status: ${GREEN}Listening${NC}"
            else
                echo -e "    Status: ${YELLOW}Not listening${NC}"
            fi
        fi
    fi

    return 0
}

# Function to check socket status
check_socket() {
    local name="$1"
    local pid_file="$2"
    local socket_path="$3"

    echo -n "  $name: "

    if [[ ! -f "$pid_file" ]]; then
        echo -e "${RED}NOT RUNNING${NC} (no PID file)"
        return 1
    fi

    local pid=$(cat "$pid_file")

    if ! ps -p "$pid" > /dev/null 2>&1; then
        echo -e "${RED}NOT RUNNING${NC} (stale PID file)"
        return 1
    fi

    local runtime=$(ps -p "$pid" -o etime= 2>/dev/null | xargs || echo "unknown")

    echo -e "${GREEN}RUNNING${NC}"
    echo "    PID: $pid"
    echo "    Runtime: $runtime"
    echo "    Socket: $socket_path"

    if [[ -S "$socket_path" ]]; then
        echo -e "    Status: ${GREEN}Socket exists${NC}"
    else
        echo -e "    Status: ${YELLOW}Socket not found${NC}"
    fi

    return 0
}

# Main status display
echo "=========================================="
echo "Navidrome Python Backend Services - Status"
echo "=========================================="
echo ""

RUNNING=0
TOTAL=3

echo -e "${BLUE}HTTP Services:${NC}"
if check_service "Text Embedding Service" "$PID_DIR/text_embedding_service.pid" "${TEXT_EMBEDDING_PORT:-9003}"; then
    RUNNING=$((RUNNING + 1))
fi
echo ""

if check_service "Recommender API" "$PID_DIR/recommender_api.pid" "${NAVIDROME_RECOMMENDER_PORT:-9002}"; then
    RUNNING=$((RUNNING + 1))
fi
echo ""

echo -e "${BLUE}Socket Services:${NC}"
if check_socket "Embedding Socket Server" "$PID_DIR/python_embed_server.pid" "/tmp/navidrome_embed.sock"; then
    RUNNING=$((RUNNING + 1))
fi
echo ""

# Log file information
echo "=========================================="
echo -e "${BLUE}Log Files:${NC}"
if [[ -d "$LOG_DIR" ]]; then
    for log in "$LOG_DIR"/*.log; do
        if [[ -f "$log" ]]; then
            local size=$(du -h "$log" | cut -f1)
            local modified=$(stat -c %y "$log" 2>/dev/null | cut -d'.' -f1 || echo "unknown")
            echo "  $(basename "$log")"
            echo "    Size: $size"
            echo "    Modified: $modified"
        fi
    done
else
    echo "  No log directory found"
fi

echo ""
echo "=========================================="
echo -e "${BLUE}Summary:${NC} $RUNNING/$TOTAL services running"
echo ""

if [[ $RUNNING -eq 0 ]]; then
    echo "No services are currently running."
    echo "Start services with: ./start_services.sh"
elif [[ $RUNNING -eq $TOTAL ]]; then
    echo -e "${GREEN}All services are running!${NC}"
else
    echo -e "${YELLOW}Some services are not running.${NC}"
    echo "Start all services with: ./start_services.sh"
fi

exit 0
