#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "$0")/.." && pwd)"
models_dir="${MUQ_MODELS_DIR:-$root_dir/musicembed/models}"

if ! command -v hf >/dev/null 2>&1; then
  echo "installing huggingface cli"
  curl -LsSf https://hf.co/cli/install.sh | bash
fi

mkdir -p "$models_dir"

download_repo() {
  local repo="$1"
  local destination="$2"
  if [[ -d "$destination" && -n "$(ls -A "$destination" 2>/dev/null)" ]]; then
    echo "$destination exists, skipping"
    return
  fi
  mkdir -p "$destination"
  hf download "$repo" --local-dir "$destination"
}

download_repo "${MUQ_AUDIO_REPO:-OpenMuQ/MuQ-large-msd-iter}" "$models_dir/MuQ-large-msd-iter"
download_repo "${MUQ_MULAN_REPO:-OpenMuQ/MuQ-MuLan-large}" "$models_dir/MuQ-MuLan-large"

echo
echo "Models downloaded."
echo "Set MUQ_AUDIO_MODEL_REF=$models_dir/MuQ-large-msd-iter to use the local MuQ audio checkpoint."
echo "Set MUQ_MULAN_MODEL_REF=$models_dir/MuQ-MuLan-large to use the local MuQ-MuLan checkpoint."
