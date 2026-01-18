#!/usr/bin/env python
import gc
import os
import time
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Optional

import librosa
import torch
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

MODEL_DIR = "./music_flamingo_fp8"
AUDIO_URL = "https://huggingface.co/datasets/nvidia/AudioSkills/resolve/main/assets/song_1.mp3"

# Keep benchmark settings centralized
MAX_NEW_TOKENS = 256
WARMUP_NEW_TOKENS = 32
TRY_UNSAFE_CONFIGS = False


@dataclass
class BenchConfig:
    name: str
    attn_implementation: Optional[str]
    use_cache: bool
    cache_implementation: Optional[str]
    compile_mode: Optional[str] = None
    sdp_kernel: Optional[dict] = None
    max_new_tokens: Optional[int] = None


def _download_audio(url: str) -> str:
    tmp_dir = tempfile.gettempdir()
    local_path = os.path.join(tmp_dir, "music_flamingo_song_1.mp3")
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(url, local_path)
    return local_path


def prepare_inputs(processor: AutoProcessor):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Describe this track in full detail - tell me the genre, tempo, and key, "
                        "then dive into the instruments, production style, and overall mood it creates."
                    ),
                },
                {"type": "audio", "path": AUDIO_URL},
            ],
        }
    ]

    # Avoid processor.apply_chat_template(tokenize=True) so we can control audio loading.
    prompt = processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )

    audio_path = _download_audio(AUDIO_URL)
    audio, _ = librosa.load(audio_path, sr=16000)
    inputs = processor(
        text=prompt,
        audio=audio,
        return_tensors="pt",
    )
    return inputs


def configure_cache(model, use_cache: bool, cache_implementation: Optional[str]):
    for cfg in (model.config, getattr(model.config, "text_config", None), model.language_model.config):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = use_cache

    model.generation_config.use_cache = use_cache
    model.language_model.generation_config.use_cache = use_cache

    if cache_implementation is not None:
        model.generation_config.cache_implementation = cache_implementation
        model.language_model.generation_config.cache_implementation = cache_implementation


def load_model(attn_implementation: Optional[str]):
    kwargs = {
        "device_map": "cuda",
        "dtype": "auto",
    }
    if attn_implementation is not None:
        kwargs["attn_implementation"] = attn_implementation

    model = MusicFlamingoForConditionalGeneration.from_pretrained(MODEL_DIR, **kwargs)
    model.eval()
    return model


def bench_one(config: BenchConfig, processor_inputs):
    torch.cuda.empty_cache()
    gc.collect()

    model = load_model(config.attn_implementation)
    configure_cache(model, config.use_cache, config.cache_implementation)

    if config.compile_mode:
        model.language_model = torch.compile(model.language_model, mode=config.compile_mode)

    inputs = {k: v.to(model.device) for k, v in processor_inputs.items()}

    def _generate(max_new_tokens: int):
        if config.sdp_kernel is None:
            return model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        with torch.backends.cuda.sdp_kernel(**config.sdp_kernel):
            return model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )

    with torch.inference_mode():
        _ = _generate(WARMUP_NEW_TOKENS)

    torch.cuda.synchronize()
    bench_tokens = config.max_new_tokens or MAX_NEW_TOKENS
    start = time.perf_counter()
    with torch.inference_mode():
        output = _generate(bench_tokens)
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]
    tokens_per_s = new_tokens / elapsed

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return tokens_per_s, elapsed, new_tokens, bench_tokens


def main():
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    inputs = prepare_inputs(processor)

    configs = [
        BenchConfig(
            name="cache_dynamic_sdpa",
            attn_implementation="sdpa",
            use_cache=True,
            cache_implementation="dynamic",
        ),
        BenchConfig(
            name="cache_dynamic_flash_attention_2",
            attn_implementation="flash_attention_2",
            use_cache=True,
            cache_implementation="dynamic",
        ),
        BenchConfig(
            name="cache_dynamic_sdpa_flash_only",
            attn_implementation="sdpa",
            use_cache=True,
            cache_implementation="dynamic",
            sdp_kernel={"enable_flash": True, "enable_mem_efficient": False, "enable_math": False},
        ),
        BenchConfig(
            name="cache_static_sdpa",
            attn_implementation="sdpa",
            use_cache=True,
            cache_implementation="static",
        ),
    ]

    if TRY_UNSAFE_CONFIGS:
        configs.extend(
            [
                BenchConfig(
                    name="baseline_no_cache",
                    attn_implementation=None,
                    use_cache=False,
                    cache_implementation=None,
                    max_new_tokens=64,
                ),
                BenchConfig(
                    name="cache_dynamic_default_attn",
                    attn_implementation=None,
                    use_cache=True,
                    cache_implementation="dynamic",
                ),
                BenchConfig(
                    name="cache_static_sdpa_compile_reduce_overhead",
                    attn_implementation="sdpa",
                    use_cache=True,
                    cache_implementation="static",
                    compile_mode="reduce-overhead",
                ),
                BenchConfig(
                    name="cache_static_sdpa_compile_max_autotune",
                    attn_implementation="sdpa",
                    use_cache=True,
                    cache_implementation="static",
                    compile_mode="max-autotune",
                ),
            ]
        )

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
    print(f"Benchmarking {len(configs)} configs; max_new_tokens={MAX_NEW_TOKENS}\n")

    results = []
    for cfg in configs:
        tokens_per_s, elapsed, new_tokens, bench_tokens = bench_one(cfg, inputs)
        results.append((cfg.name, tokens_per_s, elapsed, new_tokens))
        print(
            f"{cfg.name:45s} | {tokens_per_s:8.2f} tok/s | {elapsed:6.2f}s | {new_tokens:4d} tokens | max_new_tokens={bench_tokens}"
        )

    best = max(results, key=lambda x: x[1])
    print("\nBest:")
    print(f"  {best[0]} => {best[1]:.2f} tok/s")


if __name__ == "__main__":
    main()
