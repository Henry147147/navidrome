#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"

llama_server_bin="${QWEN_LLAMA_SERVER_BIN:-$root_dir/musicembed/llama-lib/llama-server}"
if [[ ! -x "$llama_server_bin" ]]; then
  if command -v llama-server >/dev/null 2>&1; then
    llama_server_bin="$(command -v llama-server)"
  else
    echo "llama-server not found. Build it with: bash scripts/build-llama-cpp.sh" >&2
    exit 1
  fi
fi

host="${QWEN_EMBED_HOST:-127.0.0.1}"
port="${QWEN_EMBED_PORT:-9002}"
pooling="${QWEN_POOLING:-last}"
gpu_layers="${QWEN_GPU_LAYERS:-all}"
ctx_size="${QWEN_CTX_SIZE:-8192}"
ubatch_size="${QWEN_UBATCH_SIZE:-8192}"
batch_size="${QWEN_BATCH_SIZE:-2048}"

hf_repo="${QWEN_HF_REPO:-Qwen/Qwen3-Embedding-4B-GGUF}"
hf_file="${QWEN_HF_FILE:-Qwen3-Embedding-4B-Q8_0.gguf}"
model_path="${QWEN_MODEL_PATH:-}"

lib_dir="$(dirname "$llama_server_bin")"
export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"

args=(
  --host "$host"
  --port "$port"
  --embeddings
  --pooling "$pooling"
  --n-gpu-layers "$gpu_layers"
  --ctx-size "$ctx_size"
  --ubatch-size "$ubatch_size"
  --batch-size "$batch_size"
  --no-webui
)

if [[ -n "$model_path" ]]; then
  args+=(--model "$model_path")
else
  args+=(--hf-repo "$hf_repo" --hf-file "$hf_file")
fi

echo "Starting Qwen embedding server at http://$host:$port"
echo "Set ND_RECOMMENDATIONS_TEXTBASEURL=http://$host:$port for Navidrome."

exec "$llama_server_bin" "${args[@]}"
