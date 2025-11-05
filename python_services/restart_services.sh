#!/usr/bin/env bash
#
# Restart script for Navidrome Python Backend Services
#
# Usage:
#   ./restart_services.sh [options]
#
# All options from start_services.sh are supported
#

set -euo pipefail

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Restarting Navidrome Python Backend Services..."
echo ""

# Stop all services
./stop_services.sh "$@"

echo ""
echo "Waiting 2 seconds before restart..."
sleep 2
echo ""

# Start services with the same options
./start_services.sh "$@"
