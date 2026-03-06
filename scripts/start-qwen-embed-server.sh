#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"

echo "scripts/start-qwen-embed-server.sh is deprecated; starting the MuQ service instead."
exec "$root_dir/scripts/start-muq-service.sh" "$@"
