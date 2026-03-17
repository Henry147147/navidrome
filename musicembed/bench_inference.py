#!/usr/bin/env python
import argparse
import gc
import json
import os
import sys
import time
import tempfile
import urllib.request
from dataclasses import dataclass
from typing import Optional

import librosa
import torch
from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")
os.environ.setdefault("SGLANG_JIT_DEEPGEMM_PRECOMPILE", "0")

SGLANG_PYTHON_PATH = os.path.join(os.path.dirname(__file__), "sglang", "python")
if SGLANG_PYTHON_PATH not in sys.path:
    sys.path.insert(0, SGLANG_PYTHON_PATH)

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("transformers", "sglang"),
        default="transformers",
        help="Choose the decoding backend for Qwen.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS,
        help="Number of new tokens for the benchmark run.",
    )
    parser.add_argument(
        "--warmup-new-tokens",
        type=int,
        default=WARMUP_NEW_TOKENS,
        help="Number of new tokens for the warmup run.",
    )
    parser.add_argument(
        "--sglang-attention-backend",
        type=str,
        default="triton",
        help="Optional SGLang attention backend override (e.g., triton, fa3).",
    )
    parser.add_argument(
        "--sglang-tp",
        type=int,
        default=1,
        help="Tensor parallel size for SGLang.",
    )
    parser.add_argument(
        "--sglang-max-input-tokens",
        type=int,
        default=2048,
        help="Optional cap for input embeddings before sending to SGLang (set to 0 to disable).",
    )
    return parser.parse_args()


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


def bench_one(config: BenchConfig, processor_inputs, warmup_tokens: int, max_new_tokens: int):
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
        _ = _generate(warmup_tokens)

    torch.cuda.synchronize()
    bench_tokens = config.max_new_tokens or max_new_tokens
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


def _build_input_embeds(model, processor_inputs):
    input_ids = processor_inputs["input_ids"].to(model.device)
    input_features = processor_inputs["input_features"].to(model.device)
    input_features_mask = processor_inputs["input_features_mask"].to(model.device)
    audio_times = processor_inputs.get("audio_times")
    if audio_times is not None:
        audio_times = audio_times.to(model.device)

    with torch.inference_mode():
        audio_embeds = model.get_audio_features(
            input_features,
            input_features_mask,
            audio_times=audio_times,
        )
        inputs_embeds = model.get_input_embeddings()(input_ids)
        audio_token_mask = (input_ids == model.config.audio_token_id).unsqueeze(-1)
        inputs_embeds = inputs_embeds.masked_scatter(audio_token_mask, audio_embeds)

    inputs_embeds = inputs_embeds.squeeze(0).float().cpu().tolist()
    return inputs_embeds


def _sglang_make_req(input_embeds, max_new_tokens: int):
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    fake_input_ids = [1] * len(input_embeds)
    sampling_params = SamplingParams(temperature=0.0, max_new_tokens=max_new_tokens)
    req = Req(
        rid="bench-0",
        origin_input_text="",
        origin_input_ids=fake_input_ids,
        sampling_params=sampling_params,
        input_embeds=input_embeds,
    )
    req.fill_ids = req.origin_input_ids
    req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
    req.logprob_start_len = len(req.origin_input_ids) - 1
    return [req]


def _sglang_run_tokens(model_runner, input_embeds, max_new_tokens: int):
    if max_new_tokens <= 0:
        return 0
    from sglang.bench_one_batch import decode, extend

    reqs = _sglang_make_req(input_embeds, max_new_tokens)
    next_token_ids, _, batch = extend(reqs, model_runner)
    generated = 1
    for _ in range(max_new_tokens - 1):
        next_token_ids, _ = decode(next_token_ids, batch, model_runner)
        generated += 1
    return generated


def _load_sglang_runner(attention_backend: Optional[str], tp_size: int):
    from sglang.bench_one_batch import load_model
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
    from sglang.srt.server_args import PortArgs, ServerArgs

    model_override = {
        "architectures": ["MusicFlamingoQwen2ForCausalLM"],
        "model_type": "qwen2",
    }

    server_args_kwargs = {
        "model_path": MODEL_DIR,
        "tokenizer_path": MODEL_DIR,
        "trust_remote_code": True,
        "tp_size": tp_size,
        "disable_radix_cache": True,
        "json_model_override_args": json.dumps(model_override),
        "fp8_gemm_runner_backend": "cutlass",
        "disable_cuda_graph": True,
        "sampling_backend": "pytorch",
        "grammar_backend": "none",
    }
    if attention_backend:
        server_args_kwargs["attention_backend"] = attention_backend

    server_args = ServerArgs(**server_args_kwargs)

    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)

    port_args = PortArgs.init_new(server_args)
    model_runner, _tokenizer = load_model(server_args, port_args, gpu_id=0, tp_rank=0)
    return model_runner


def bench_one_sglang(
    input_embeds,
    warmup_tokens: int,
    max_new_tokens: int,
    attention_backend: Optional[str],
    tp_size: int,
    max_input_tokens: int,
):
    from sglang.srt.distributed.parallel_state import destroy_distributed_environment

    model_runner = _load_sglang_runner(attention_backend, tp_size)
    caps = [
        cap
        for cap in (
            getattr(model_runner.model_config, "context_len", None),
            max_input_tokens or None,
        )
        if cap
    ]
    cap_len = min(caps) if caps else None
    if cap_len and len(input_embeds) > cap_len:
        input_embeds = input_embeds[:cap_len]
        print(f"Truncated input embeddings to {cap_len} tokens to fit KV cache limits.")

    _ = _sglang_run_tokens(model_runner, input_embeds, warmup_tokens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    new_tokens = _sglang_run_tokens(model_runner, input_embeds, max_new_tokens)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens_per_s = new_tokens / elapsed if elapsed > 0 else 0.0

    destroy_distributed_environment()
    return tokens_per_s, elapsed, new_tokens, len(input_embeds)


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    processor = AutoProcessor.from_pretrained(MODEL_DIR)
    inputs = prepare_inputs(processor)

    if args.backend == "sglang":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
        print(
            f"SGLang benchmark; max_new_tokens={args.max_new_tokens} warmup_new_tokens={args.warmup_new_tokens}\n"
        )

        model = load_model(attn_implementation=None)
        input_embeds = _build_input_embeds(model, inputs)

        del model
        torch.cuda.empty_cache()
        gc.collect()

        tokens_per_s, elapsed, new_tokens, prompt_len = bench_one_sglang(
            input_embeds,
            args.warmup_new_tokens,
            args.max_new_tokens,
            args.sglang_attention_backend,
            args.sglang_tp,
            args.sglang_max_input_tokens,
        )
        print(
            f"sglang_qwen2{args.sglang_attention_backend or 'default':>18s} | {tokens_per_s:8.2f} tok/s | {elapsed:6.2f}s | {new_tokens:4d} tokens | prompt_len={prompt_len}"
        )
        return

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
    print(f"Benchmarking {len(configs)} configs; max_new_tokens={args.max_new_tokens}\n")

    results = []
    for cfg in configs:
        tokens_per_s, elapsed, new_tokens, bench_tokens = bench_one(
            cfg, inputs, args.warmup_new_tokens, args.max_new_tokens
        )
        results.append((cfg.name, tokens_per_s, elapsed, new_tokens))
        print(
            f"{cfg.name:45s} | {tokens_per_s:8.2f} tok/s | {elapsed:6.2f}s | {new_tokens:4d} tokens | max_new_tokens={bench_tokens}"
        )

    best = max(results, key=lambda x: x[1])
    print("\nBest:")
    print(f"  {best[0]} => {best[1]:.2f} tok/s")


if __name__ == "__main__":
    main()
