#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${PYTHON_BIN:-python3}"
host="${MUQ_SERVICE_HOST:-127.0.0.1}"
port="${MUQ_SERVICE_PORT:-9002}"

export PYTHONPATH="$root_dir:${PYTHONPATH:-}"

echo "Starting MuQ service at http://$host:$port"
echo "Set ND_RECOMMENDATIONS_TEXTBASEURL=http://$host:$port for text embeddings."
echo "Set ND_RECOMMENDATIONS_BATCHBASEURL=http://$host:$port for batch endpoints."

exec "$python_bin" -m python_services.muq_service --host "$host" --port "$port" "$@"
