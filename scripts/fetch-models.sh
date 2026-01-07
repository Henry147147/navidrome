if ! command -v hf >/dev/null 2>&1
then
    echo "installing hugginface cli"
    curl -LsSf https://hf.co/cli/install.sh | bash
fi
if [[ -f "models/mmproj-music-flamingo.gguf" ]]; then
  echo "mmproj-music-flamingo.gguf exists, skipping"
else
  hf download henry1477/music-flamingo-gguf mmproj-music-flamingo-bf16.gguf --local-dir ./models
  mv ./models/mmproj-music-flamingo-*.gguf ./models/mmproj-music-flamingo.gguf
fi
if [[ -f "models/music-flamingo.gguf" ]]; then
  echo "music-flamingo.gguf exists, skipping"
else
  hf download henry1477/music-flamingo-gguf music-flamingo-Q8_0.gguf --local-dir ./models
  mv ./models/music-flamingo-*.gguf ./models/music-flamingo.gguf
fi


if [[ -f "models/qwen-embedder-4b.gguf" ]]; then
  echo "qwen-embedder-4b.gguf exists, skipping"
else
  hf download Qwen/Qwen3-Embedding-4B-GGUF Qwen3-Embedding-4B-f16.gguf --local-dir ./models
  mv ./models/Qwen3-Embedding-4B-*.gguf ./models/qwen-embedder-4b.gguf
fi
