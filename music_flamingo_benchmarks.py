#!/usr/bin/env python3
"""
Run Music Flamingo benchmarks from the Music Flamingo paper.

This script is intentionally a single file and is designed to:
  - run base and quantized Music Flamingo side-by-side,
  - load a mix of Hugging Face datasets and local manifests,
  - save per-sample outputs + aggregate metrics.

Many benchmarks in the paper require data access or LLM-as-judge scoring.
This script supports optional local manifests for those cases.

Manifest format (JSONL) for any benchmark:
  {
    "id": "unique-id",
    "audio": "/absolute/or/relative/path.wav",
    "question": "optional question",
    "choices": ["A", "B", "C"],
    "answer": "B",
    "reference": "reference text",
    "prompt": "override prompt"
  }
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "0"


PAPER_BENCHMARKS = [
    # Music QA and reasoning
    "mmau_music_full",
    "mmau_music_test_mini",
    "mmau_pro_music",
    "muchomusic",
    "mmar_music",
    "music_instruct",
    "music_avqa",
    "songcaps",
    # Music Information Retrieval
    "nsynth_source",
    "nsynth_instrument",
    "gtzan",
    "medley_solos_db",
    "musiccaps",
    # Lyrics transcription
    "opencpop",
    "musdb18_lyrics",
]


@dataclass
class BenchmarkSpec:
    name: str
    kind: str  # "mcq", "classification", "caption", "transcription"
    loader: str  # loader key
    prompt: str
    dataset_id: Optional[str] = None
    split: Optional[str] = None
    audio_field: Optional[str] = None
    label_field: Optional[str] = None
    reference_field: Optional[str] = None


DEFAULT_SPECS: Dict[str, BenchmarkSpec] = {
    "mmau_music_full": BenchmarkSpec(
        name="mmau_music_full",
        kind="mcq",
        loader="manifest",
        prompt="Answer the multiple-choice question about the music. Respond with only the option letter.",
    ),
    "mmau_music_test_mini": BenchmarkSpec(
        name="mmau_music_test_mini",
        kind="mcq",
        loader="manifest",
        prompt="Answer the multiple-choice question about the music. Respond with only the option letter.",
    ),
    "mmau_pro_music": BenchmarkSpec(
        name="mmau_pro_music",
        kind="mcq",
        loader="manifest",
        prompt="Answer the multiple-choice question about the music. Respond with only the option letter.",
    ),
    "muchomusic": BenchmarkSpec(
        name="muchomusic",
        kind="mcq",
        loader="manifest",
        prompt="Answer the multiple-choice question about the music. Respond with only the option letter.",
    ),
    "mmar_music": BenchmarkSpec(
        name="mmar_music",
        kind="mcq",
        loader="manifest",
        prompt="Answer the multiple-choice question about the music. Respond with only the option letter.",
    ),
    "music_instruct": BenchmarkSpec(
        name="music_instruct",
        kind="caption",
        loader="manifest",
        prompt="Follow the instruction about the music and provide a helpful response.",
    ),
    "music_avqa": BenchmarkSpec(
        name="music_avqa",
        kind="mcq",
        loader="manifest",
        prompt="Answer the multiple-choice question about the music. Respond with only the option letter.",
    ),
    "songcaps": BenchmarkSpec(
        name="songcaps",
        kind="caption",
        loader="manifest",
        prompt="Write a rich, detailed caption describing the music.",
    ),
    "nsynth_source": BenchmarkSpec(
        name="nsynth_source",
        kind="classification",
        loader="nsynth",
        prompt="Identify the instrument source (acoustic, electronic, or synthetic). Answer with the label only.",
        dataset_id="jg583/NSynth",
        split="test",
        audio_field="audio",
        label_field="instrument_source_str",
    ),
    "nsynth_instrument": BenchmarkSpec(
        name="nsynth_instrument",
        kind="classification",
        loader="nsynth",
        prompt="Identify the instrument family. Answer with the label only.",
        dataset_id="jg583/NSynth",
        split="test",
        audio_field="audio",
        label_field="instrument_family_str",
    ),
    "gtzan": BenchmarkSpec(
        name="gtzan",
        kind="classification",
        loader="hf_audio_classification",
        prompt="Identify the music genre. Answer with the label only.",
        dataset_id="storylinez/gtzan-music-genre-dataset",
        split="train",
        audio_field="audio",
        label_field="genre",
    ),
    "medley_solos_db": BenchmarkSpec(
        name="medley_solos_db",
        kind="classification",
        loader="hf_audio_classification",
        prompt="Identify the solo instrument. Answer with the label only.",
        dataset_id="vtsouval/medley-solos-db",
        split="train",
        audio_field="audio",
        label_field=None,
    ),
    "musiccaps": BenchmarkSpec(
        name="musiccaps",
        kind="caption",
        loader="musiccaps",
        prompt="Write a concise caption describing the music.",
        dataset_id="google/MusicCaps",
        split="train",
    ),
    "opencpop": BenchmarkSpec(
        name="opencpop",
        kind="transcription",
        loader="opencpop",
        prompt="Transcribe the sung lyrics verbatim. Output only the lyrics.",
    ),
    "musdb18_lyrics": BenchmarkSpec(
        name="musdb18_lyrics",
        kind="transcription",
        loader="musdb18_lyrics",
        prompt="Transcribe the sung lyrics verbatim. Output only the lyrics.",
        dataset_id="jazasyed/musdb-alt",
        split="test",
        reference_field="text",
    ),
}


def eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def ensure_pkg(import_name: str, pip_name: Optional[str] = None) -> Any:
    try:
        return __import__(import_name)
    except Exception:  # pragma: no cover - optional dependency
        pkg = pip_name or import_name
        eprint(f"Missing dependency '{pkg}'. Install with: pip install {pkg}")
        sys.exit(1)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_mcq_answer(output: str, choices: List[str]) -> Optional[str]:
    if not output:
        return None
    text = output.strip()
    # Prefer explicit letter A/B/C...
    m = re.search(r"\b([A-Z])\b", text.upper())
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    # Try numeric option
    m = re.search(r"\b(\d+)\b", text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(choices):
            return choices[idx]
    # Match by label text
    norm_out = normalize_text(text)
    best = None
    best_len = 0
    for c in choices:
        nc = normalize_text(c)
        if nc and nc in norm_out and len(nc) > best_len:
            best = c
            best_len = len(nc)
    return best


def parse_label(output: str, labels: List[str]) -> Optional[str]:
    return parse_mcq_answer(output, labels)


def is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def compute_wer(ref: str, hyp: str, cjk_as_chars: bool = True) -> float:
    jiwer = ensure_pkg("jiwer")
    if cjk_as_chars and is_cjk(ref):
        ref_tokens = " ".join([ch for ch in ref if not ch.isspace()])
        hyp_tokens = " ".join([ch for ch in hyp if not ch.isspace()])
        return jiwer.wer(ref_tokens, hyp_tokens)
    return jiwer.wer(ref, hyp)


def ensure_audio_path(audio_obj: Any, tmp_dir: str) -> str:
    # Accept str path or dict with path/array/sampling_rate.
    if isinstance(audio_obj, str):
        return audio_obj
    if isinstance(audio_obj, dict):
        path = audio_obj.get("path")
        if path and os.path.exists(path):
            return path
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate")
        if arr is not None and sr is not None:
            sf = ensure_pkg("soundfile")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"audio_{int(time.time() * 1e6)}.wav")
            sf.write(tmp_path, arr, sr)
            return tmp_path
    raise ValueError("Could not resolve audio path from dataset entry.")


def load_manifest(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    samples: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            samples.append(obj)
    return samples


def hf_load_dataset(dataset_id: str, split: Optional[str], cache_dir: Optional[str], trust_remote_code: bool):
    datasets = ensure_pkg("datasets")
    if split:
        return datasets.load_dataset(dataset_id, split=split, cache_dir=cache_dir, trust_remote_code=trust_remote_code)
    return datasets.load_dataset(dataset_id, cache_dir=cache_dir, trust_remote_code=trust_remote_code)


def load_nsynth(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(spec.dataset_id, spec.split, args.hf_cache_dir, args.trust_remote_code)
    labels = sorted({ex[spec.label_field] for ex in ds})
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        audio_path = ensure_audio_path(ex[spec.audio_field], args.tmp_audio_dir)
        samples.append(
            {
                "id": str(ex.get("note", len(samples))),
                "audio": audio_path,
                "label": ex[spec.label_field],
            }
        )
    return samples, labels


def load_hf_audio_classification(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(spec.dataset_id, spec.split, args.hf_cache_dir, args.trust_remote_code)
    # Try to infer label field if not provided.
    label_field = spec.label_field
    if label_field is None:
        for cand in ["label", "genre", "instrument", "instrument_name", "class", "class_name"]:
            if cand in ds.column_names:
                label_field = cand
                break
    if label_field is None:
        raise ValueError(f"Could not infer label field for {spec.name}. Use --dataset-config to set it.")
    labels = sorted({ex[label_field] for ex in ds})
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        audio_path = ensure_audio_path(ex[spec.audio_field], args.tmp_audio_dir)
        samples.append(
            {
                "id": str(ex.get("id", len(samples))),
                "audio": audio_path,
                "label": ex[label_field],
            }
        )
    return samples, labels


def resolve_musiccaps_audio(audio_dir: str, ytid: str, start_s: int, end_s: int) -> Optional[str]:
    if not audio_dir:
        return None
    candidates = [
        f"{ytid}_{start_s}_{end_s}.wav",
        f"{ytid}_{start_s}_{end_s}.mp3",
        f"{ytid}.wav",
        f"{ytid}.mp3",
    ]
    for c in candidates:
        path = os.path.join(audio_dir, c)
        if os.path.exists(path):
            return path
    return None


def load_musiccaps(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(spec.dataset_id, spec.split, args.hf_cache_dir, args.trust_remote_code)
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        ytid = ex.get("ytid") or ex.get("youtube_id") or ex.get("youtubeid")
        start_s = ex.get("start_s") or ex.get("start_seconds") or ex.get("start")
        end_s = ex.get("end_s") or ex.get("end_seconds") or ex.get("end")
        caption = ex.get("caption") or ex.get("text") or ex.get("description", "")
        if ytid is None or start_s is None or end_s is None:
            continue
        audio_path = resolve_musiccaps_audio(args.musiccaps_audio_dir, ytid, int(start_s), int(end_s))
        if not audio_path:
            continue
        samples.append(
            {
                "id": ytid,
                "audio": audio_path,
                "reference": caption,
            }
        )
    return samples, []


def load_opencpop(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not args.opencpop_root:
        raise ValueError("opencpop requires --opencpop-root pointing to the dataset root.")
    seg_dir = os.path.join(args.opencpop_root, "segments")
    candidates = [
        os.path.join(seg_dir, "test.txt"),
        os.path.join(seg_dir, "transcriptions.txt"),
    ]
    meta_path = None
    for c in candidates:
        if os.path.exists(c):
            meta_path = c
            break
    if not meta_path:
        raise FileNotFoundError("Could not find opencpop transcription file in segments/ (test.txt or transcriptions.txt).")
    samples: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split("|")
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = parts[1]
            # Common locations for utterance audio
            audio_candidates = [
                os.path.join(seg_dir, "wavs", f"{utt_id}.wav"),
                os.path.join(args.opencpop_root, "wavs", f"{utt_id}.wav"),
            ]
            audio_path = None
            for ap in audio_candidates:
                if os.path.exists(ap):
                    audio_path = ap
                    break
            if not audio_path:
                continue
            samples.append({"id": utt_id, "audio": audio_path, "reference": text})
    return samples, []


def load_musdb18_lyrics(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not args.musdb18_root:
        raise ValueError("musdb18_lyrics requires --musdb18-root pointing to MUSDB18.")
    ds = hf_load_dataset(spec.dataset_id, spec.split, args.hf_cache_dir, args.trust_remote_code)
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        name = ex.get("name") or ex.get("track") or ex.get("title")
        if not name:
            continue
        text = ex.get(spec.reference_field or "text") or ex.get("lyrics") or ex.get("text_tagged", "")
        audio_path = os.path.join(args.musdb18_root, "test", name, f"{args.musdb18_audio_stem}.wav")
        if not os.path.exists(audio_path):
            fallback = os.path.join(args.musdb18_root, "test", name, "vocals.wav")
            if os.path.exists(fallback):
                audio_path = fallback
            else:
                continue
        samples.append({"id": name, "audio": audio_path, "reference": text})
    return samples, []


LOADERS = {
    "manifest": None,
    "nsynth": load_nsynth,
    "hf_audio_classification": load_hf_audio_classification,
    "musiccaps": load_musiccaps,
    "opencpop": load_opencpop,
    "musdb18_lyrics": load_musdb18_lyrics,
}


class ModelRunner:
    def __init__(self, model_id: str, torch_dtype: str, device_map: str, quantization: str, cache_dir: Optional[str]):
        transformers = ensure_pkg("transformers")
        self.processor = transformers.AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
        quant_config = None
        if quantization in ("4bit", "8bit"):
            bnb = ensure_pkg("transformers")
            BitsAndBytesConfig = getattr(bnb, "BitsAndBytesConfig", None)
            if BitsAndBytesConfig is None:
                raise ValueError("transformers BitsAndBytesConfig is required for 4bit/8bit loading.")
            quant_config = BitsAndBytesConfig(load_in_4bit=quantization == "4bit", load_in_8bit=quantization == "8bit")
        dtype = None
        if torch_dtype and torch_dtype != "auto":
            torch = ensure_pkg("torch")
            dtype = getattr(torch, torch_dtype)
        self.model = transformers.AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quant_config,
            cache_dir=cache_dir,
        )

    def generate(self, audio_path: str, prompt: str, max_new_tokens: int, temperature: float) -> str:
        torch = ensure_pkg("torch")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "audio", "path": audio_path},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})
        decoded = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return decoded[0] if decoded else ""


def run_classification(
    runner: ModelRunner,
    spec: BenchmarkSpec,
    samples: List[Dict[str, Any]],
    labels: List[str],
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, Any]:
    records = []
    correct = 0
    total = 0
    label_list = labels
    label_prompt = spec.prompt
    for ex in iter_samples(samples, args):
        prompt = ex.get("prompt") or f"{label_prompt}\nLabels: {', '.join(label_list)}"
        output = runner.generate(ex["audio"], prompt, args.max_new_tokens, args.temperature)
        pred = parse_label(output, label_list)
        gold = ex["label"]
        is_correct = pred is not None and normalize_text(pred) == normalize_text(gold)
        correct += int(is_correct)
        total += 1
        records.append(
            {
                "id": ex.get("id"),
                "audio": ex.get("audio"),
                "label": gold,
                "prediction": pred,
                "raw_output": output,
                "correct": is_correct,
            }
        )
    save_records(out_dir, records)
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "num_samples": total}


def run_mcq(
    runner: ModelRunner,
    spec: BenchmarkSpec,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, Any]:
    records = []
    correct = 0
    total = 0
    for ex in iter_samples(samples, args):
        choices = ex.get("choices") or []
        if not choices:
            raise ValueError(f"MCQ sample missing choices in {spec.name}")
        options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
        question = ex.get("question", "")
        prompt = ex.get("prompt") or f"{question}\nOptions:\n{options}\nAnswer with the option letter only."
        output = runner.generate(ex["audio"], prompt, args.max_new_tokens, args.temperature)
        pred = parse_mcq_answer(output, choices)
        gold = ex.get("answer")
        is_correct = pred is not None and gold is not None and normalize_text(pred) == normalize_text(gold)
        correct += int(is_correct)
        total += 1
        records.append(
            {
                "id": ex.get("id"),
                "audio": ex.get("audio"),
                "question": question,
                "choices": choices,
                "answer": gold,
                "prediction": pred,
                "raw_output": output,
                "correct": is_correct,
            }
        )
    save_records(out_dir, records)
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "num_samples": total}


def run_transcription(
    runner: ModelRunner,
    spec: BenchmarkSpec,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, Any]:
    records = []
    wers = []
    for ex in iter_samples(samples, args):
        prompt = ex.get("prompt") or spec.prompt
        output = runner.generate(ex["audio"], prompt, args.max_new_tokens, args.temperature)
        ref = ex.get("reference", "")
        wer = compute_wer(ref, output, cjk_as_chars=args.cjk_as_chars)
        wers.append(wer)
        records.append(
            {
                "id": ex.get("id"),
                "audio": ex.get("audio"),
                "reference": ref,
                "prediction": output,
                "wer": wer,
            }
        )
    save_records(out_dir, records)
    avg_wer = sum(wers) / len(wers) if wers else 0.0
    return {"wer": avg_wer, "num_samples": len(wers)}


def run_caption(
    runner: ModelRunner,
    spec: BenchmarkSpec,
    samples: List[Dict[str, Any]],
    args: argparse.Namespace,
    out_dir: str,
) -> Dict[str, Any]:
    records = []
    for ex in iter_samples(samples, args):
        prompt = ex.get("prompt") or spec.prompt
        output = runner.generate(ex["audio"], prompt, args.max_new_tokens, args.temperature)
        records.append(
            {
                "id": ex.get("id"),
                "audio": ex.get("audio"),
                "reference": ex.get("reference"),
                "prediction": output,
            }
        )
    save_records(out_dir, records)
    return {"num_samples": len(records), "note": "Caption metrics require LLM/human judge; predictions saved."}


def iter_samples(samples: List[Dict[str, Any]], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if args.shuffle:
        random.Random(args.seed).shuffle(samples)
    if args.max_samples:
        return samples[: args.max_samples]
    return samples


def save_records(out_dir: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def load_samples_for_benchmark(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if spec.loader == "manifest":
        manifest = os.path.join(args.data_root, spec.name, "manifest.jsonl")
        samples = load_manifest(manifest)
        return samples, []
    loader_fn = LOADERS.get(spec.loader)
    if not loader_fn:
        raise ValueError(f"Unknown loader: {spec.loader}")
    return loader_fn(spec, args)


def parse_dataset_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_config(specs: Dict[str, BenchmarkSpec], overrides: Dict[str, Any]) -> Dict[str, BenchmarkSpec]:
    for name, cfg in overrides.items():
        if name not in specs:
            continue
        spec = specs[name]
        for k, v in cfg.items():
            if hasattr(spec, k):
                setattr(spec, k, v)
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Music Flamingo benchmarks")
    parser.add_argument("--benchmarks", default="all", help="Comma-separated list or 'all'")
    parser.add_argument("--data-root", default="benchmarks", help="Root folder for manifest-based benchmarks")
    parser.add_argument("--dataset-config", default=None, help="JSON file overriding dataset ids/fields")
    parser.add_argument("--hf-cache-dir", default=None, help="Hugging Face cache dir for datasets/models")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow datasets with custom loading scripts")
    parser.add_argument("--tmp-audio-dir", default=".cache/audio_tmp", help="Temp dir for audio arrays")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit samples per benchmark")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before limiting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cjk-as-chars", action="store_true", help="Compute WER as CER for CJK")
    # Model configs
    parser.add_argument("--model", default="nvidia/music-flamingo-hf")
    parser.add_argument("--model-label", default="base")
    parser.add_argument("--model-quant", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--model2", default=None)
    parser.add_argument("--model2-label", default="quant")
    parser.add_argument("--model2-quant", default="none", choices=["none", "4bit", "8bit"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto", help="e.g., float16, bfloat16, auto")
    parser.add_argument("--output-dir", default="mf_benchmark_results")
    # Dataset-specific paths
    parser.add_argument("--musiccaps-audio-dir", default=None, help="Directory with MusicCaps audio files")
    parser.add_argument("--opencpop-root", default=None, help="Opencpop dataset root")
    parser.add_argument("--musdb18-root", default=None, help="MUSDB18 dataset root")
    parser.add_argument("--musdb18-audio-stem", default="mixture", help="mixture|vocals|other stem")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_dataset_config(args.dataset_config)
    specs = merge_config(DEFAULT_SPECS.copy(), overrides)

    if args.benchmarks == "all":
        bench_names = PAPER_BENCHMARKS
    else:
        bench_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    models = [
        {"id": args.model, "label": args.model_label, "quant": args.model_quant},
    ]
    if args.model2:
        models.append({"id": args.model2, "label": args.model2_label, "quant": args.model2_quant})

    for model_cfg in models:
        eprint(f"Loading model {model_cfg['id']} ({model_cfg['label']}) ...")
        runner = ModelRunner(
            model_id=model_cfg["id"],
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            quantization=model_cfg["quant"],
            cache_dir=args.hf_cache_dir,
        )
        model_out_dir = os.path.join(args.output_dir, model_cfg["label"])
        os.makedirs(model_out_dir, exist_ok=True)
        summary = {}
        for name in bench_names:
            if name not in specs:
                eprint(f"Unknown benchmark: {name}")
                continue
            spec = specs[name]
            eprint(f"Running {name} ...")
            try:
                samples, labels = load_samples_for_benchmark(spec, args)
            except Exception as exc:
                eprint(f"Skipping {name}: {exc}")
                continue
            bench_out_dir = os.path.join(model_out_dir, name)
            if spec.kind == "classification":
                metrics = run_classification(runner, spec, samples, labels, args, bench_out_dir)
            elif spec.kind == "mcq":
                metrics = run_mcq(runner, spec, samples, args, bench_out_dir)
            elif spec.kind == "transcription":
                metrics = run_transcription(runner, spec, samples, args, bench_out_dir)
            elif spec.kind == "caption":
                metrics = run_caption(runner, spec, samples, args, bench_out_dir)
            else:
                eprint(f"Unknown kind for {name}: {spec.kind}")
                continue
            summary[name] = metrics
            with open(os.path.join(bench_out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=True, indent=2))
        with open(os.path.join(model_out_dir, "summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=True, indent=2))
        eprint(f"Finished model {model_cfg['label']}. Results in {model_out_dir}")


if __name__ == "__main__":
    main()
