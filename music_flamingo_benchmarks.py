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
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import wave
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_DATASETS_OFFLINE"] = "0"


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


MCQ_PROMPT_TEMPLATE = "Question: {question} Options: {options} The correct answer is:"


DEFAULT_SPECS: Dict[str, BenchmarkSpec] = {
    "mmau_music_full": BenchmarkSpec(
        name="mmau_music_full",
        kind="mcq",
        loader="mmau",
        prompt=MCQ_PROMPT_TEMPLATE,
        dataset_id="pbcong/mmau",
        split="test",
    ),
    "mmau_music_test_mini": BenchmarkSpec(
        name="mmau_music_test_mini",
        kind="mcq",
        loader="mmau",
        prompt=MCQ_PROMPT_TEMPLATE,
        dataset_id="pbcong/mmau",
        split="test",
    ),
    "mmau_pro_music": BenchmarkSpec(
        name="mmau_pro_music",
        kind="mcq",
        loader="mmau_pro",
        prompt=MCQ_PROMPT_TEMPLATE,
        dataset_id="gamma-lab-umd/MMAU-Pro",
        split="test",
    ),
    "muchomusic": BenchmarkSpec(
        name="muchomusic",
        kind="mcq",
        loader="muchomusic",
        prompt=MCQ_PROMPT_TEMPLATE,
        dataset_id="AudioLLMs/mu_chomusic_test",
        split="test",
        audio_field="context",
    ),
    "mmar_music": BenchmarkSpec(
        name="mmar_music",
        kind="mcq",
        loader="mmar",
        prompt=MCQ_PROMPT_TEMPLATE,
        dataset_id="BoJack/MMAR",
        split="test",
    ),
    "music_instruct": BenchmarkSpec(
        name="music_instruct",
        kind="caption",
        loader="music_instruct",
        prompt="Follow the instruction about the music and provide a helpful response.",
        dataset_id="m-a-p/Music-Instruct",
        split="train",
    ),
    "music_avqa": BenchmarkSpec(
        name="music_avqa",
        kind="classification",
        loader="music_avqa",
        prompt="Answer the question about the music with a short phrase.",
        split="test",
    ),
    "songcaps": BenchmarkSpec(
        name="songcaps",
        kind="caption",
        loader="songcaps",
        prompt="Write a caption describing the music.",
    ),
    "nsynth_source": BenchmarkSpec(
        name="nsynth_source",
        kind="classification",
        loader="nsynth",
        prompt="Identify the instrument source (acoustic, electronic, or synthetic). Answer with the label only.",
        dataset_id="mteb/nsynth-mini",
        split="test",
        audio_field="audio",
        label_field="instrument_source_str",
    ),
    "nsynth_instrument": BenchmarkSpec(
        name="nsynth_instrument",
        kind="classification",
        loader="nsynth",
        prompt="Identify the instrument family. Answer with the label only.",
        dataset_id="mteb/nsynth-mini",
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
        label_field="label",
    ),
    "medley_solos_db": BenchmarkSpec(
        name="medley_solos_db",
        kind="classification",
        loader="medley_solos_db",
        prompt="Identify the solo instrument. Answer with the label only.",
        split="test",
    ),
    "musiccaps": BenchmarkSpec(
        name="musiccaps",
        kind="caption",
        loader="musiccaps",
        prompt="Write a caption describing the music.",
        dataset_id="CLAPv2/MusicCaps",
        split="train",
    ),
    "opencpop": BenchmarkSpec(
        name="opencpop",
        kind="transcription",
        loader="opencpop",
        prompt="Transcribe the sung lyrics verbatim. Output only the lyrics.",
        dataset_id="espnet/ace-opencpop-segments",
        split="test",
        audio_field="audio",
        reference_field="transcription",
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


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def ensure_pkg(import_name: str, pip_name: Optional[str] = None) -> Any:
    try:
        return __import__(import_name)
    except Exception:  # pragma: no cover - optional dependency
        pkg = pip_name or import_name
        eprint(f"Missing dependency '{pkg}'. Install with: pip install {pkg}")
        sys.exit(1)


def has_torchcodec() -> bool:
    try:
        __import__("torchcodec")
        return True
    except Exception:
        return False


def force_hf_online() -> None:
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        import huggingface_hub.constants as hfconst

        hfconst.HF_HUB_OFFLINE = False
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def download_url(url: str, dest_path: str, overwrite: bool = False) -> str:
    if os.path.exists(dest_path) and not overwrite:
        return dest_path
    ensure_dir(os.path.dirname(dest_path))
    with urllib.request.urlopen(url) as resp, open(dest_path, "wb") as out:
        total = resp.headers.get("Content-Length")
        total_size = int(total) if total else None
        downloaded = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                eprint(f"\rDownloading {os.path.basename(dest_path)}: {pct:5.1f}% ({downloaded/1e9:.2f} GB)", end="")
        if total_size:
            eprint("")
    return dest_path


def download_gdrive_file(file_id: str, dest_path: str, overwrite: bool = False) -> str:
    if os.path.exists(dest_path) and not overwrite:
        return dest_path
    ensure_dir(os.path.dirname(dest_path))
    gdown = ensure_pkg("gdown")
    gdown.download(id=file_id, output=dest_path, quiet=False)
    return dest_path


def download_gdrive_folder(folder_url: str, dest_dir: str) -> None:
    ensure_dir(dest_dir)
    gdown = ensure_pkg("gdown")
    gdown.download_folder(url=folder_url, output=dest_dir, quiet=False, use_cookies=True)


def download_zenodo_record(record_id: int, dest_dir: str, only_files: Optional[List[str]] = None) -> List[str]:
    ensure_dir(dest_dir)
    url = f"https://zenodo.org/api/records/{record_id}"
    rec = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))
    downloaded = []
    for f in rec.get("files", []):
        key = f.get("key")
        if only_files and key not in only_files:
            continue
        link = f.get("links", {}).get("self")
        if not link:
            continue
        dest_path = os.path.join(dest_dir, key)
        download_url(link, dest_path)
        downloaded.append(dest_path)
    return downloaded


def extract_archive(path: str, dest_dir: str) -> None:
    print(path)
    ensure_dir(dest_dir)
    lower = path.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest_dir)
        return
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        with tarfile.open(path, "r:gz") as tf:
            tf.extractall(dest_dir)
        return
    if lower.endswith(".tar"):
        with tarfile.open(path, "r:") as tf:
            tf.extractall(dest_dir)
        return
    raise ValueError(f"Unsupported archive format: {path}")


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found on PATH; install ffmpeg to extract audio from videos.")
    return ffmpeg


def extract_audio_from_video(video_path: str, out_path: str, sample_rate: int = 16000) -> str:
    if os.path.exists(out_path):
        return out_path
    ffmpeg = ensure_ffmpeg()
    ensure_dir(os.path.dirname(out_path))
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        out_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path


def resolve_audio_by_ytid(audio_dir: Optional[str], ytid: Optional[str]) -> Optional[str]:
    if not audio_dir or not ytid:
        return None
    candidates = [
        f"{ytid}.wav",
        f"{ytid}.mp3",
        f"{ytid}.flac",
        f"{ytid}.m4a",
    ]
    for c in candidates:
        path = os.path.join(audio_dir, c)
        if os.path.exists(path):
            return path
    return None


def resolve_audio_by_video_id(audio_dir: Optional[str], video_id: Optional[str]) -> Optional[str]:
    return resolve_audio_by_ytid(audio_dir, video_id)


def parse_timestamp_range(ts: str) -> Tuple[Optional[float], Optional[float]]:
    if not ts:
        return None, None
    parts = ts.split(",")
    if len(parts) != 2:
        return None, None

    def to_seconds(t: str) -> float:
        items = [float(x) for x in t.strip().split(":")]
        if len(items) == 3:
            return items[0] * 3600 + items[1] * 60 + items[2]
        if len(items) == 2:
            return items[0] * 60 + items[1]
        return items[0]

    return to_seconds(parts[0]), to_seconds(parts[1])


def download_youtube_audio_segment(url: str, start_s: Optional[float], end_s: Optional[float], out_path: str) -> str:
    yt_dlp = ensure_pkg("yt_dlp", "yt-dlp")
    ensure_dir(os.path.dirname(out_path))
    base = out_path[:-4] if out_path.lower().endswith(".wav") else out_path
    outtmpl = base + ".%(ext)s"
    opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
    }
    if start_s is not None and end_s is not None:
        opts["download_sections"] = [f"*{start_s}-{end_s}"]
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    wav_path = base + ".wav"
    if wav_path != out_path and os.path.exists(wav_path):
        os.replace(wav_path, out_path)
    return out_path


MUSDB18_HQ_RECORD_ID = 3338373
MUSDB18_HQ_ARCHIVE = "musdb18hq.zip"


def find_musdb18_root(base_dir: str) -> Optional[str]:
    if not base_dir:
        return None
    if os.path.exists(os.path.join(base_dir, "train")) and os.path.exists(os.path.join(base_dir, "test")):
        return base_dir
    for root, dirs, _files in os.walk(base_dir):
        if "train" in dirs and "test" in dirs:
            return root
    return None

def parse_choices(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, tuple):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        text = raw.strip()
        # Try JSON list
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
        # Split on common delimiters
        for delim in ["\n", "|", ";"]:
            if delim in text:
                parts = [p.strip() for p in text.split(delim) if p.strip()]
                if parts:
                    return [strip_choice_prefix(p) for p in parts]
        return [strip_choice_prefix(text)]
    return [str(raw)]


def strip_choice_prefix(text: str) -> str:
    return re.sub(r"^[A-Da-d][\).:\-]\s*", "", text).strip()


def normalize_gold_answer(answer: Any, choices: List[str]) -> Optional[str]:
    if answer is None:
        return None
    if isinstance(answer, (int, float)):
        idx = int(answer)
        if idx < 0 and choices:
            return None
        if idx < len(choices):
            return choices[idx]
        if idx - 1 < len(choices):
            return choices[idx - 1]
        return None
    if isinstance(answer, str):
        text = answer.strip()
        if not text:
            return None
        # If letter or numeric index
        if re.fullmatch(r"[A-Da-d]", text):
            idx = ord(text.upper()) - ord("A")
            return choices[idx] if 0 <= idx < len(choices) else None
        if re.fullmatch(r"\d+", text):
            idx = int(text)
            if idx - 1 < len(choices):
                return choices[idx - 1]
        # Try JSON wrapper
        try:
            parsed = json.loads(text)
            if isinstance(parsed, (int, float)):
                return normalize_gold_answer(parsed, choices)
            if isinstance(parsed, str):
                return normalize_gold_answer(parsed, choices)
        except Exception:
            pass
        # Direct match to choice text
        for c in choices:
            if normalize_text(c) == normalize_text(text):
                return c
        return text
    return str(answer)


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def format_prompt(template: str, **kwargs: Any) -> str:
    if not template:
        return ""
    try:
        class _SafeDict(dict):
            def __missing__(self, key: str) -> str:
                return "{" + key + "}"

        return template.format_map(_SafeDict(**kwargs))
    except Exception:
        return template


def format_mcq_options(choices: List[str]) -> str:
    parts = []
    for i, choice in enumerate(choices):
        label = chr(65 + i)
        text = str(choice).strip()
        text = re.sub(r"^\([A-Da-d]\)\s*", "", text)
        text = strip_choice_prefix(text)
        if text and text[-1] not in ".?!":
            text = text + "."
        parts.append(f"({label}) {text}")
    return " ".join(parts)


def extract_choice_index(text: Optional[str], n_choices: int) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b([A-Z])\b", text.upper())
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < n_choices:
            return idx
    m = re.search(r"\b(\d+)\b", text)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_choices:
            return idx
    return None


def choice_index_for_gold(gold: Optional[str], choices: List[str]) -> Optional[int]:
    if gold is None:
        return None
    text = str(gold).strip()
    if not text:
        return None
    if re.fullmatch(r"[A-Da-d]", text):
        idx = ord(text.upper()) - ord("A")
        if 0 <= idx < len(choices):
            return idx
    if re.fullmatch(r"\d+", text):
        idx = int(text)
        if 1 <= idx <= len(choices):
            return idx - 1
    gold_norm = normalize_text(text)
    for i, choice in enumerate(choices):
        if normalize_text(choice) == gold_norm:
            return i
    return None


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
    if labels:
        return parse_mcq_answer(output, labels)
    return output.strip() if output else None


def mcq_is_correct(output: str, pred: Optional[str], gold: Optional[str], choices: List[str]) -> bool:
    if gold is None:
        return False
    gold_norm = normalize_text(gold)
    if pred and normalize_text(pred) == gold_norm:
        return True
    if output and normalize_text(output) == gold_norm:
        return True
    pred_idx = extract_choice_index(output, len(choices))
    gold_idx = choice_index_for_gold(gold, choices)
    return pred_idx is not None and gold_idx is not None and pred_idx == gold_idx


def is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def compute_wer(ref: str, hyp: str, cjk_as_chars: bool = True) -> float:
    try:
        import jiwer  # type: ignore
    except Exception:
        jiwer = None

    def _tokenize(text: str) -> List[str]:
        if cjk_as_chars and is_cjk(text):
            return [ch for ch in text if not ch.isspace()]
        return text.split()

    def _edit_distance(a: List[str], b: List[str]) -> int:
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, token in enumerate(a, start=1):
            curr = [i] + [0] * len(b)
            for j, btok in enumerate(b, start=1):
                cost = 0 if token == btok else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[-1]

    if jiwer is not None:
        if cjk_as_chars and is_cjk(ref):
            ref_tokens = " ".join(_tokenize(ref))
            hyp_tokens = " ".join(_tokenize(hyp))
            return jiwer.wer(ref_tokens, hyp_tokens)
        return jiwer.wer(ref, hyp)
    ref_tokens = _tokenize(ref)
    hyp_tokens = _tokenize(hyp)
    if not ref_tokens:
        return 0.0
    return _edit_distance(ref_tokens, hyp_tokens) / float(len(ref_tokens))


def ensure_audio_path(audio_obj: Any, tmp_dir: str, cache_dir: Optional[str] = None, token: Optional[str] = None) -> str:
    # Accept str path or dict with path/bytes/array/sampling_rate.
    if isinstance(audio_obj, str):
        return resolve_audio_path(audio_obj, tmp_dir, cache_dir, token)
    if isinstance(audio_obj, (bytes, bytearray)):
        return write_audio_bytes(bytes(audio_obj), tmp_dir)
    if isinstance(audio_obj, dict):
        audio_bytes = audio_obj.get("bytes")
        if audio_bytes:
            return write_audio_bytes(audio_bytes, tmp_dir)
        path = audio_obj.get("path")
        if path:
            return resolve_audio_path(path, tmp_dir, cache_dir, token)
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate")
        if arr is not None and sr is not None:
            sf = ensure_pkg("soundfile")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"audio_{int(time.time() * 1e6)}.wav")
            sf.write(tmp_path, arr, sr)
            return tmp_path
    raise ValueError("Could not resolve audio path from dataset entry.")


def resolve_audio_path(path: str, tmp_dir: str, cache_dir: Optional[str], token: Optional[str]) -> str:
    if path.startswith("hf://"):
        return download_hf_uri(path, cache_dir, token)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Audio path not found: {path}")


def download_hf_uri(uri: str, cache_dir: Optional[str], token: Optional[str]) -> str:
    # Example: hf://datasets/owner/name@revision/path/to/file.wav
    m = re.match(r"^hf://datasets/([^@]+)@([^/]+)/(.+)$", uri)
    if not m:
        raise ValueError(f"Unsupported HF URI: {uri}")
    repo_id, revision, filename = m.group(1), m.group(2), m.group(3)
    hub = ensure_pkg("huggingface_hub")
    return hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )


def download_hf_file(
    repo_id: str,
    filename: str,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    repo_type: str = "dataset",
) -> str:
    hub = ensure_pkg("huggingface_hub")
    return hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
    )


def ensure_mmau_pro_audio_dir(args: argparse.Namespace) -> str:
    root = os.path.join(args.data_root, "mmau_pro")
    data_dir = os.path.join(root, "data")
    if os.path.exists(data_dir):
        return data_dir
    ensure_dir(root)
    zip_path = download_hf_file(
        repo_id="gamma-lab-umd/MMAU-Pro",
        filename="data.zip",
        cache_dir=args.hf_cache_dir,
        token=args.hf_token,
        repo_type="dataset",
    )
    extract_archive(zip_path, root)
    if not os.path.exists(data_dir):
        raise ValueError(f"Failed to extract MMAU-Pro audio to {data_dir}")
    return data_dir


def write_audio_bytes(data: bytes, tmp_dir: str) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    digest = hashlib.md5(data).hexdigest()
    tmp_path = os.path.join(tmp_dir, f"audio_{digest}.wav")
    if not os.path.exists(tmp_path):
        with open(tmp_path, "wb") as f:
            f.write(data)
    return tmp_path


def create_sine_wav(tmp_dir: str, duration_s: float = 2.0, sr: int = 16000, freq: float = 440.0) -> str:
    os.makedirs(tmp_dir, exist_ok=True)
    key = f"sine_{duration_s}_{sr}_{freq}"
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    path = os.path.join(tmp_dir, f"{key}_{digest}.wav")
    if os.path.exists(path):
        return path
    n_samples = int(duration_s * sr)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        for i in range(n_samples):
            val = int(0.2 * 32767 * math.sin(2 * math.pi * freq * i / sr))
            wf.writeframes(struct.pack("<h", val))
    return path


def build_smoke_samples(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    audio_path = create_sine_wav(args.tmp_audio_dir)
    samples: List[Dict[str, Any]] = []
    labels: List[str] = []
    for i in range(max(1, args.smoke_samples)):
        sid = f"smoke-{spec.name}-{i}"
        if spec.kind == "mcq":
            choices = ["Option A", "Option B", "Option C", "Option D"]
            samples.append(
                {
                    "id": sid,
                    "audio": audio_path,
                    "question": "Select the correct option for this audio.",
                    "choices": choices,
                    "answer": choices[0],
                }
            )
        elif spec.kind == "classification":
            labels = labels or ["label_a", "label_b"]
            samples.append({"id": sid, "audio": audio_path, "label": labels[0]})
        elif spec.kind == "transcription":
            samples.append({"id": sid, "audio": audio_path, "reference": "la la la"})
        elif spec.kind == "caption":
            samples.append({"id": sid, "audio": audio_path, "reference": ""})
    return samples, labels


def load_manifest(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Manifest not found: {path}")
    samples: List[Dict[str, Any]] = []
    base_dir = os.path.dirname(path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            print("before get")
            print(obj)
            audio_path = obj.get("audio")
            if isinstance(audio_path, str) and audio_path and not os.path.isabs(audio_path):
                obj["audio"] = os.path.join(base_dir, audio_path)
            samples.append(obj)
    return samples


def hf_load_dataset(
    dataset_id: str,
    split: Optional[str],
    cache_dir: Optional[str],
    trust_remote_code: bool,
    streaming: bool,
    token: Optional[str],
    respect_offline: bool,
):
    if not respect_offline:
        force_hf_online()
    datasets = ensure_pkg("datasets")
    kwargs: Dict[str, Any] = {
        "cache_dir": cache_dir,
        "streaming": streaming,
    }
    if split:
        kwargs["split"] = split
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if token:
        kwargs["token"] = token
    try:
        return datasets.load_dataset(dataset_id, **kwargs)
    except TypeError:
        # Older datasets uses use_auth_token
        if "token" in kwargs:
            kwargs["use_auth_token"] = kwargs.pop("token")
        return datasets.load_dataset(dataset_id, **kwargs)


def maybe_cast_audio(ds: Any, audio_field: Optional[str]) -> Any:
    if not audio_field:
        return ds
    try:
        if getattr(ds, "features", None) and audio_field in ds.features:
            datasets = ensure_pkg("datasets")
            Audio = getattr(datasets, "Audio", None)
            if Audio:
                ds = ds.cast_column(audio_field, Audio(decode=False))
    except Exception:
        pass
    return ds


def is_streaming_dataset(ds: Any) -> bool:
    return not isinstance(ds, list) and hasattr(ds, "__iter__") and not hasattr(ds, "__len__")


def load_nsynth(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    ds = maybe_cast_audio(ds, spec.audio_field)
    labels: List[str] = []
    try:
        feat = ds.features.get(spec.label_field) if getattr(ds, "features", None) else None
        if getattr(feat, "names", None):
            labels = list(feat.names)
    except Exception:
        pass
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        audio_path = ensure_audio_path(ex[spec.audio_field], args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
        samples.append(
            {
                "id": str(ex.get("note", len(samples))),
                "audio": audio_path,
                "label": ex[spec.label_field],
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
    return samples, labels


def load_hf_audio_classification(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    ds = maybe_cast_audio(ds, spec.audio_field)
    # Try to infer label field if not provided.
    label_field = spec.label_field
    if label_field is None:
        columns = getattr(ds, "column_names", None)
        if columns is None and getattr(ds, "features", None):
            columns = list(ds.features.keys())
        for cand in ["label", "genre", "instrument", "instrument_name", "class", "class_name"]:
            if columns and cand in columns:
                label_field = cand
                break
    if label_field is None:
        raise ValueError(f"Could not infer label field for {spec.name}. Use --dataset-config to set it.")
    labels: List[str] = []
    try:
        feat = ds.features.get(label_field) if getattr(ds, "features", None) else None
        if getattr(feat, "names", None):
            labels = list(feat.names)
    except Exception:
        pass
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        audio_path = ensure_audio_path(ex[spec.audio_field], args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
        raw_label = ex[label_field]
        label = raw_label
        if labels and isinstance(raw_label, int) and 0 <= raw_label < len(labels):
            label = labels[raw_label]
        samples.append(
            {
                "id": str(ex.get("id", len(samples))),
                "audio": audio_path,
                "label": label,
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
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
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    audio_field = spec.audio_field or "audio"
    ds = maybe_cast_audio(ds, audio_field)
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        caption = ex.get("caption") or ex.get("text") or ex.get("description", "")
        if audio_field in ex:
            audio_path = ensure_audio_path(ex[audio_field], args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
            sample_id = ex.get("index") or ex.get("ytid") or ex.get("youtube_id") or ex.get("id") or len(samples)
            samples.append({"id": str(sample_id), "audio": audio_path, "reference": caption})
        else:
            ytid = ex.get("ytid") or ex.get("youtube_id") or ex.get("youtubeid")
            start_s = ex.get("start_s") or ex.get("start_seconds") or ex.get("start")
            end_s = ex.get("end_s") or ex.get("end_seconds") or ex.get("end")
            if ytid is None or start_s is None or end_s is None:
                continue
            audio_path = resolve_musiccaps_audio(args.musiccaps_audio_dir, ytid, int(start_s), int(end_s))
            if not audio_path and args.allow_ytdlp:
                audio_path = os.path.join(args.tmp_audio_dir, "musiccaps", f"{ytid}_{int(start_s)}_{int(end_s)}.wav")
                url = f"https://www.youtube.com/watch?v={ytid}"
                try:
                    download_youtube_audio_segment(url, float(start_s), float(end_s), audio_path)
                except Exception as exc:
                    eprint(f"MusicCaps yt-dlp failed for {ytid}: {exc}")
                    audio_path = None
            if not audio_path:
                continue
            samples.append({"id": ytid, "audio": audio_path, "reference": caption})
        if args.max_samples and len(samples) >= args.max_samples:
            break
    return samples, []


def load_opencpop(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if args.opencpop_root:
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

    if not spec.dataset_id:
        raise ValueError("opencpop requires --opencpop-root or a dataset_id in the spec.")
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    ds = maybe_cast_audio(ds, spec.audio_field or "audio")
    ref_field = spec.reference_field or "transcription"
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        if ref_field not in ex:
            continue
        audio_path = ensure_audio_path(ex[spec.audio_field or "audio"], args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
        sample_id = ex.get("segment_id") or ex.get("id") or len(samples)
        samples.append({"id": str(sample_id), "audio": audio_path, "reference": ex[ref_field]})
        if args.max_samples and len(samples) >= args.max_samples:
            break
    return samples, []


def load_musdb18_lyrics(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not args.musdb18_root:
        default_root = os.path.join(args.data_root, "musdb18_hq")
        found = find_musdb18_root(default_root)
        if not found:
            archive_path = os.path.join(default_root, MUSDB18_HQ_ARCHIVE)
            if os.path.exists(archive_path):
                extract_archive(archive_path, default_root)
                found = find_musdb18_root(default_root)
        if found:
            args.musdb18_root = found
    if not args.musdb18_root and args.musdb18_download:
        default_root = os.path.join(args.data_root, "musdb18_hq")
        ensure_dir(default_root)
        archive_path = os.path.join(default_root, MUSDB18_HQ_ARCHIVE)
        if not os.path.exists(archive_path):
            eprint("Downloading MUSDB18-HQ from Zenodo (large download)...")
            download_zenodo_record(MUSDB18_HQ_RECORD_ID, default_root, only_files=[MUSDB18_HQ_ARCHIVE])
        extract_archive(archive_path, default_root)
        found = find_musdb18_root(default_root)
        if found:
            args.musdb18_root = found
    if not args.musdb18_root:
        raise ValueError("musdb18_lyrics requires --musdb18-root pointing to MUSDB18 (or --musdb18-download).")
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
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
        if args.max_samples and len(samples) >= args.max_samples:
            break
    return samples, []


MMAU_JSON_URLS = {
    "test": "https://raw.githubusercontent.com/Sakshi113/MMAU/main/mmau-test.json",
    "test-mini": "https://raw.githubusercontent.com/Sakshi113/MMAU/main/mmau-test-mini.json",
}
MMAU_AUDIO_FILE_IDS = {
    "test": "1XqkRupC723zAeyDn4dYniqNv4uO-8rEg",
    "test-mini": "1fERNIyTa0HWry6iIG1X-1ACPlUlhlRWA",
}


def mmau_split_name(spec: BenchmarkSpec) -> str:
    return "test-mini" if spec.name.endswith("test_mini") else "test"


def load_mmau_official(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    split_name = mmau_split_name(spec)
    mmau_root = args.mmau_root or os.path.join(args.data_root, "mmau")
    ensure_dir(mmau_root)
    json_path = os.path.join(mmau_root, f"mmau-{split_name}.json")
    if not os.path.exists(json_path):
        if args.mmau_download:
            download_url(MMAU_JSON_URLS[split_name], json_path)
        else:
            raise ValueError(f"MMAU json not found: {json_path} (set --mmau-download to fetch).")
    audio_root = args.mmau_audio_root or mmau_root
    if args.mmau_download:
        expected_dir = os.path.join(audio_root, f"{split_name}-audios")
        if not os.path.exists(expected_dir):
            archive_path = os.path.join(mmau_root, f"mmau-{split_name}-audios.tar.gz")
            download_gdrive_file(MMAU_AUDIO_FILE_IDS[split_name], archive_path)
            try:
                extract_archive(archive_path, audio_root)
            except Exception as exc:
                raise ValueError(f"Failed to extract MMAU audio archive: {exc}") from exc
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples: List[Dict[str, Any]] = []
    for ex in data:
        if args.mmau_task:
            task = ex.get("task") or ""
            if task and task != args.mmau_task:
                continue
        choices = parse_choices(ex.get("choices"))
        if not choices:
            continue
        question = ex.get("question") or ex.get("prompt") or ""
        answer = normalize_gold_answer(ex.get("answer"), choices)
        audio_id = ex.get("audio_id") or ex.get("audio") or ""
        audio_rel = audio_id.lstrip("./")
        audio_path = os.path.join(audio_root, audio_rel)
        if not os.path.exists(audio_path):
            raise ValueError(f"MMAU audio not found: {audio_path}")
        samples.append(
            {
                "id": ex.get("id") or str(len(samples)),
                "audio": audio_path,
                "question": question,
                "choices": choices,
                "answer": answer,
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("No MMAU samples loaded.")
    return samples, []


def load_mmau_hf(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    ds = maybe_cast_audio(ds, "audio")
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        if args.mmau_task and ex.get("task") and ex.get("task") != args.mmau_task:
            continue
        choices = parse_choices(ex.get("choices"))
        if not choices:
            continue
        question = ex.get("question") or ex.get("prompt") or ""
        answer = normalize_gold_answer(ex.get("answer"), choices)
        audio_entry = ex.get("audio")
        audio_path = None
        if audio_entry:
            audio_path = ensure_audio_path(audio_entry, args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
        else:
            audio_entry = ex.get("audio_path")
            if isinstance(audio_entry, list):
                audio_entry = audio_entry[0] if audio_entry else None
            if isinstance(audio_entry, str):
                filename = audio_entry.lstrip("./")
                try:
                    audio_path = download_hf_file(
                        repo_id=spec.dataset_id or "",
                        filename=filename,
                        cache_dir=args.hf_cache_dir,
                        token=args.hf_token,
                        repo_type="dataset",
                    )
                except Exception:
                    if spec.dataset_id == "gamma-lab-umd/MMAU-Pro":
                        data_dir = ensure_mmau_pro_audio_dir(args)
                        audio_path = os.path.join(args.data_root, "mmau_pro", filename)
                        if not os.path.exists(audio_path):
                            alt = os.path.join(data_dir, os.path.basename(filename))
                            if os.path.exists(alt):
                                audio_path = alt
                            else:
                                audio_path = None
                    else:
                        audio_path = None
        if not audio_path:
            raise ValueError("Could not resolve audio path from dataset entry.")
        samples.append(
            {
                "id": ex.get("id") or str(len(samples)),
                "audio": audio_path,
                "question": question,
                "choices": choices,
                "answer": answer,
            }
        )
        if spec.name.endswith("test_mini") and args.mmau_test_mini_size and len(samples) >= args.mmau_test_mini_size:
            break
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("No MMAU samples loaded.")
    return samples, []


def load_mmau(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if args.mmau_root or args.mmau_audio_root or args.mmau_download:
        return load_mmau_official(spec, args)
    default_root = os.path.join(args.data_root, "mmau")
    if os.path.exists(default_root):
        args.mmau_root = default_root
        return load_mmau_official(spec, args)
    return load_mmau_hf(spec, args)


def load_mmau_pro(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    return load_mmau_hf(spec, args)


def load_muchomusic(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    audio_field = spec.audio_field or "audio"
    ds = maybe_cast_audio(ds, audio_field)
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        choices = parse_choices(ex.get("choices"))
        if not choices:
            continue
        question = ex.get("instruction") or ex.get("question") or ""
        answer = normalize_gold_answer(ex.get("answer"), choices)
        audio_path = ensure_audio_path(ex.get(audio_field), args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
        samples.append(
            {
                "id": ex.get("id") or str(len(samples)),
                "audio": audio_path,
                "question": question,
                "choices": choices,
                "answer": answer,
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("No MuChoMusic samples loaded.")
    return samples, []


MUSIC_AVQA_JSON_URL = "https://raw.githubusercontent.com/GeWu-Lab/MUSIC-AVQA/main/data/json/avqa-{split}.json"
MUSIC_AVQA_VIDEOS_FOLDER_URL = "https://drive.google.com/drive/folders/1WAryZZE0srLIZG8VHl22uZ3tpbGHtsrQ?usp=sharing"


def resolve_music_avqa_json(args: argparse.Namespace, split: str) -> str:
    if args.music_avqa_annotations:
        return args.music_avqa_annotations
    root = os.path.join(args.data_root, "music_avqa")
    ensure_dir(root)
    path = os.path.join(root, f"avqa-{split}.json")
    if not os.path.exists(path):
        if args.music_avqa_download:
            download_url(MUSIC_AVQA_JSON_URL.format(split=split), path)
        else:
            raise ValueError(f"MusicAVQA annotations not found: {path} (set --music-avqa-download to fetch).")
    return path


def find_video_file(video_root: str, video_id: str) -> Optional[str]:
    if not video_root:
        return None
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        cand = os.path.join(video_root, f"{video_id}{ext}")
        if os.path.exists(cand):
            return cand
    return None


def build_video_index(video_root: str) -> Dict[str, str]:
    index: Dict[str, str] = {}
    if not video_root or not os.path.exists(video_root):
        return index
    for root, _dirs, files in os.walk(video_root):
        if "__MACOSX" in root:
            continue
        for fname in files:
            if fname.startswith("._"):
                continue
            base, ext = os.path.splitext(fname)
            if ext.lower() in (".mp4", ".mkv", ".webm", ".mov"):
                index[base] = os.path.join(root, fname)
    return index


def load_music_avqa(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not args.music_avqa_root and not args.music_avqa_audio_dir:
        default_root = os.path.join(args.data_root, "music_avqa_videos")
        if os.path.exists(default_root):
            args.music_avqa_root = default_root
        elif args.music_avqa_download_videos:
            args.music_avqa_root = default_root
            if not os.path.exists(args.music_avqa_root):
                eprint("Downloading MusicAVQA videos from Google Drive (large download)...")
                download_gdrive_folder(MUSIC_AVQA_VIDEOS_FOLDER_URL, args.music_avqa_root)
        else:
            raise ValueError("music_avqa requires --music-avqa-root (videos) or --music-avqa-audio-dir (audio).")
    video_index: Dict[str, str] = {}
    if args.music_avqa_root:
        video_index = build_video_index(args.music_avqa_root)
        if not video_index:
            zip_paths = [
                os.path.join(args.music_avqa_root, name)
                for name in os.listdir(args.music_avqa_root)
                if name.lower().endswith(".zip")
            ]
            if zip_paths:
                for zip_path in zip_paths:
                    extract_archive(zip_path, args.music_avqa_root)
                video_index = build_video_index(args.music_avqa_root)
    split = spec.split or "test"
    ann_path = resolve_music_avqa_json(args, split)
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples: List[Dict[str, Any]] = []
    missing_videos = 0
    for ex in data:
        if ex.get("question_deleted"):
            continue
        video_id = ex.get("video_id")
        if not video_id:
            continue
        audio_path = resolve_audio_by_video_id(args.music_avqa_audio_dir, str(video_id))
        if not audio_path:
            video_path = video_index.get(str(video_id))
            if not video_path:
                video_path = find_video_file(args.music_avqa_root, str(video_id))
            if not video_path:
                missing_videos += 1
                if video_index:
                    continue
                raise ValueError(f"MusicAVQA video not found for id {video_id} in {args.music_avqa_root}")
            audio_out = os.path.join(args.tmp_audio_dir, "music_avqa", f"{video_id}.wav")
            audio_path = extract_audio_from_video(video_path, audio_out)
        question = ex.get("question_content") or ex.get("question") or ""
        answer = ex.get("anser") or ex.get("answer") or ""
        samples.append(
            {
                "id": f"{video_id}_{ex.get('question_id', len(samples))}",
                "audio": audio_path,
                "label": answer,
                "question": question,
                "prompt": question,
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("No MusicAVQA samples loaded.")
    if missing_videos:
        eprint(f"MusicAVQA: skipped {missing_videos} items without local videos.")
    return samples, []


def load_music_instruct(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    samples: List[Dict[str, Any]] = []
    audio_dir = args.music_instruct_audio_dir or args.musiccaps_audio_dir
    for ex in ds:
        qa = ex.get("QA") or ex
        question = qa.get("question") or qa.get("instruction") or ""
        answer = qa.get("answer") or qa.get("response") or ""
        audio_path = None
        audio_obj = qa.get("audio") or ex.get("audio")
        if audio_obj is not None:
            try:
                audio_path = ensure_audio_path(audio_obj, args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
            except Exception:
                audio_path = None
        ytid = qa.get("ytid") or qa.get("youtube_id") or qa.get("video_id")
        if not audio_path:
            audio_path = resolve_audio_by_ytid(audio_dir, ytid)
        if not audio_path and args.allow_ytdlp and ytid:
            audio_path = os.path.join(args.tmp_audio_dir, "music_instruct", f"{ytid}.wav")
            url = f"https://www.youtube.com/watch?v={ytid}"
            try:
                download_youtube_audio_segment(url, None, None, audio_path)
            except Exception as exc:
                eprint(f"Music-Instruct yt-dlp failed for {ytid}: {exc}")
                audio_path = None
        if not audio_path:
            continue
        samples.append(
            {
                "id": ytid or str(len(samples)),
                "audio": audio_path,
                "reference": answer,
                "prompt": question or spec.prompt,
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("No Music-Instruct samples loaded (audio not found).")
    return samples, []


def load_songcaps(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    manifest_path = args.songcaps_manifest
    if not manifest_path:
        manifest_path = os.path.join(args.data_root, spec.name, "manifest.jsonl")
    return load_manifest(manifest_path), []


def load_medley_solos_db(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    root = args.medley_solos_root or os.path.join(args.data_root, "medley_solos_db")
    ensure_dir(root)
    meta_path = os.path.join(root, "Medley-solos-DB_metadata.csv")
    if not os.path.exists(meta_path):
        if args.medley_solos_download:
            download_zenodo_record(3464194, root, only_files=["Medley-solos-DB_metadata.csv"])
        else:
            raise ValueError("Medley-solos-DB metadata not found. Set --medley-solos-download to fetch.")
    audio_archive = os.path.join(root, "Medley-solos-DB.tar.gz")
    has_audio = any(
        name.startswith("Medley-solos-DB_") and name.endswith(".wav") for name in os.listdir(root) if os.path.isfile(os.path.join(root, name))
    )
    if args.medley_solos_download and not has_audio:
        if not os.path.exists(audio_archive):
            download_zenodo_record(3464194, root, only_files=["Medley-solos-DB.tar.gz"])
        extract_archive(audio_archive, root)
    samples: List[Dict[str, Any]] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subset = row.get("subset", "")
            if spec.split and subset != spec.split:
                continue
            instrument = row.get("instrument") or ""
            instrument_id = row.get("instrument_id") or ""
            uuid4 = row.get("uuid4") or ""
            if not uuid4:
                continue
            filename = f"Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav"
            audio_path = os.path.join(root, filename)
            if not os.path.exists(audio_path):
                # Try nested directory.
                alt = os.path.join(root, "Medley-solos-DB", filename)
                if os.path.exists(alt):
                    audio_path = alt
                else:
                    raise ValueError(f"Medley-solos-DB audio not found: {audio_path}")
            samples.append(
                {
                    "id": uuid4,
                    "audio": audio_path,
                    "label": instrument,
                }
            )
            if args.max_samples and len(samples) >= args.max_samples:
                break
    if not samples:
        raise ValueError("No Medley-solos-DB samples loaded.")
    return samples, []


def load_mmar(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    ds = hf_load_dataset(
        spec.dataset_id,
        spec.split,
        args.hf_cache_dir,
        args.trust_remote_code,
        streaming=args.streaming,
        token=args.hf_token,
        respect_offline=args.respect_offline,
    )
    samples: List[Dict[str, Any]] = []
    for ex in ds:
        choices = parse_choices(ex.get("choices"))
        if not choices:
            continue
        question = ex.get("question") or ex.get("prompt") or ""
        answer = normalize_gold_answer(ex.get("answer"), choices)
        audio_path = ex.get("audio_path") or ex.get("audio")
        audio_root = args.mmar_audio_root or os.path.join(args.data_root, "mmar_audio")
        if audio_path:
            if isinstance(audio_path, str) and not os.path.isabs(audio_path):
                if args.mmar_audio_root:
                    audio_path = os.path.join(args.mmar_audio_root, audio_path.lstrip("./"))
            if isinstance(audio_path, dict) or (isinstance(audio_path, str) and os.path.exists(str(audio_path))):
                audio_path = ensure_audio_path(audio_path, args.tmp_audio_dir, args.hf_cache_dir, args.hf_token)
            else:
                audio_path = None
        if not audio_path and args.allow_ytdlp and ex.get("url"):
            rel = ex.get("audio_path") or f"{ex.get('id', len(samples))}.wav"
            filename = os.path.basename(rel)
            out_path = os.path.join(audio_root, filename)
            ensure_dir(audio_root)
            start_s, end_s = parse_timestamp_range(ex.get("timestamp", ""))
            try:
                download_youtube_audio_segment(ex["url"], start_s, end_s, out_path)
                audio_path = out_path
            except Exception as exc:
                eprint(f"MMAR yt-dlp failed for {ex.get('id', '')}: {exc}")
        if not audio_path:
            continue
        samples.append(
            {
                "id": ex.get("id") or str(len(samples)),
                "audio": audio_path,
                "question": question,
                "choices": choices,
                "answer": answer,
            }
        )
        if args.max_samples and len(samples) >= args.max_samples:
            break
    if not samples:
        raise ValueError("No MMAR samples loaded (audio not found).")
    return samples, []


LOADERS = {
    "manifest": None,
    "mmau": load_mmau,
    "mmau_pro": load_mmau_pro,
    "muchomusic": load_muchomusic,
    "mmar": load_mmar,
    "music_avqa": load_music_avqa,
    "music_instruct": load_music_instruct,
    "songcaps": load_songcaps,
    "nsynth": load_nsynth,
    "hf_audio_classification": load_hf_audio_classification,
    "medley_solos_db": load_medley_solos_db,
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
        outputs = self.generate_batch([audio_path], [prompt], max_new_tokens, temperature)
        return outputs[0] if outputs else ""

    def generate_batch(
        self,
        audio_paths: List[str],
        prompts: List[str],
        max_new_tokens: int,
        temperature: float,
    ) -> List[str]:
        if len(audio_paths) != len(prompts):
            raise ValueError("audio_paths and prompts must have the same length")
        torch = ensure_pkg("torch")
        conversations = []
        for audio_path, prompt in zip(audio_paths, prompts):
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            content.append({"type": "audio", "path": audio_path})
            conversations.append([{"role": "user", "content": content}])
        inputs = self.processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        # Cast only floating tensors (audio/features) to model dtype; keep token IDs as int.
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype=self.model.dtype)
        inputs = inputs.to(self.model.device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
        }
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})
        decoded = self.processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        return decoded


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
    tqdm = ensure_pkg("tqdm").tqdm
    total_count = None
    if isinstance(samples, list):
        total_count = len(samples)
        if args.max_samples:
            total_count = min(total_count, args.max_samples)
    with tqdm(total=total_count, desc=spec.name, unit="sample", leave=True, ascii=True) as pbar:
        for batch in iter_batches(samples, args):
            prompts = []
            audio_paths = []
            for ex in batch:
                if label_list:
                    prompt = ex.get("prompt") or f"{label_prompt}\nLabels: {', '.join(label_list)}"
                else:
                    prompt = ex.get("prompt") or label_prompt
                prompts.append(prompt)
                audio_paths.append(ex["audio"])
            outputs = runner.generate_batch(audio_paths, prompts, args.max_new_tokens, args.temperature)
            for ex, output in zip(batch, outputs):
                pred = parse_label(output, label_list)
                gold = ex["label"]
                if label_list:
                    is_correct = pred is not None and normalize_text(pred) == normalize_text(gold)
                else:
                    out_norm = normalize_text(output or "")
                    gold_norm = normalize_text(gold or "")
                    is_correct = bool(gold_norm) and gold_norm in out_norm
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
            pbar.update(len(batch))
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
    tqdm = ensure_pkg("tqdm").tqdm
    total_count = None
    if isinstance(samples, list):
        total_count = len(samples)
        if args.max_samples:
            total_count = min(total_count, args.max_samples)
    with tqdm(total=total_count, desc=spec.name, unit="sample", leave=True, ascii=True) as pbar:
        for batch in iter_batches(samples, args):
            prompts = []
            audio_paths = []
            choices_list = []
            questions = []
            golds = []
            for ex in batch:
                choices = parse_choices(ex.get("choices"))
                if not choices:
                    raise ValueError(f"MCQ sample missing choices in {spec.name}")
                options = format_mcq_options(choices)
                question = ex.get("question", "")
                prompt = ex.get("prompt")
                if not prompt:
                    template = spec.prompt or MCQ_PROMPT_TEMPLATE
                    if "{question}" in template or "{options}" in template:
                        prompt = format_prompt(template, question=question, options=options)
                    else:
                        prompt = f"{template}\nQuestion: {question} Options: {options} The correct answer is:"
                prompts.append(prompt)
                audio_paths.append(ex["audio"])
                choices_list.append(choices)
                questions.append(question)
                golds.append(normalize_gold_answer(ex.get("answer"), choices))
            outputs = runner.generate_batch(audio_paths, prompts, args.max_new_tokens, args.temperature)
            for ex, output, choices, question, gold in zip(batch, outputs, choices_list, questions, golds):
                pred = parse_mcq_answer(output, choices)
                is_correct = mcq_is_correct(output or "", pred, gold, choices)
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
            pbar.update(len(batch))
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
    tqdm = ensure_pkg("tqdm").tqdm
    total_count = None
    if isinstance(samples, list):
        total_count = len(samples)
        if args.max_samples:
            total_count = min(total_count, args.max_samples)
    with tqdm(total=total_count, desc=spec.name, unit="sample", leave=True, ascii=True) as pbar:
        for batch in iter_batches(samples, args):
            prompts = []
            audio_paths = []
            refs = []
            for ex in batch:
                prompts.append(ex.get("prompt") or spec.prompt)
                audio_paths.append(ex["audio"])
                refs.append(ex.get("reference") or "")
            outputs = runner.generate_batch(audio_paths, prompts, args.max_new_tokens, args.temperature)
            for ex, output, ref in zip(batch, outputs, refs):
                wer = compute_wer(ref, output, cjk_as_chars=args.cjk_as_chars) if ref else 0.0
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
            pbar.update(len(batch))
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
    tqdm = ensure_pkg("tqdm").tqdm
    total_count = None
    if isinstance(samples, list):
        total_count = len(samples)
        if args.max_samples:
            total_count = min(total_count, args.max_samples)
    with tqdm(total=total_count, desc=spec.name, unit="sample", leave=True, ascii=True) as pbar:
        for batch in iter_batches(samples, args):
            prompts = []
            audio_paths = []
            refs = []
            for ex in batch:
                prompts.append(ex.get("prompt") or spec.prompt)
                audio_paths.append(ex["audio"])
                refs.append(ex.get("reference"))
            outputs = runner.generate_batch(audio_paths, prompts, args.max_new_tokens, args.temperature)
            for ex, output, ref in zip(batch, outputs, refs):
                records.append(
                    {
                        "id": ex.get("id"),
                        "audio": ex.get("audio"),
                        "reference": ref,
                        "prediction": output,
                    }
                )
            pbar.update(len(batch))
    save_records(out_dir, records)
    return {"num_samples": len(records), "note": "Caption metrics require LLM/human judge; predictions saved."}


def iter_samples(samples: Iterable[Dict[str, Any]], args: argparse.Namespace) -> Iterable[Dict[str, Any]]:
    if isinstance(samples, list):
        if args.shuffle:
            random.Random(args.seed).shuffle(samples)
        limit = args.max_samples or len(samples)
        for ex in samples[:limit]:
            yield ex
        return
    if args.shuffle:
        eprint("Shuffle requested but streaming dataset in use; ignoring shuffle.")
    count = 0
    for ex in samples:
        yield ex
        count += 1
        if args.max_samples and count >= args.max_samples:
            break


def iter_batches(samples: Iterable[Dict[str, Any]], args: argparse.Namespace) -> Iterable[List[Dict[str, Any]]]:
    batch_size = max(1, int(getattr(args, "batch", 1)))
    batch: List[Dict[str, Any]] = []
    for ex in iter_samples(samples, args):
        batch.append(ex)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_records(out_dir: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "predictions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def load_samples_for_benchmark(spec: BenchmarkSpec, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], List[str]]:
    try:
        if spec.loader == "manifest":
            manifest = os.path.join(args.data_root, spec.name, "manifest.jsonl")
            samples = load_manifest(manifest)
            return samples, []
        loader_fn = LOADERS.get(spec.loader)
        if not loader_fn:
            raise ValueError(f"Unknown loader: {spec.loader}")
        samples, labels = loader_fn(spec, args)
        if not hasattr(args, "smoke_used"):
            args.smoke_used = {}
        args.smoke_used[spec.name] = False
        if not samples:
            raise ValueError("No samples loaded.")
        return samples, labels
    except Exception as exc:
        if args.allow_smoke:
            eprint(f"Falling back to smoke samples for {spec.name}: {exc}")
            if not hasattr(args, "smoke_used"):
                args.smoke_used = {}
            args.smoke_used[spec.name] = True
            return build_smoke_samples(spec, args)
        raise


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
    parser.add_argument("--hf-token", default=None, help="Hugging Face token for gated datasets")
    parser.add_argument("--trust-remote-code", action="store_true", help="Allow datasets with custom loading scripts")
    parser.add_argument("--respect-offline", action="store_true", help="Respect HF offline environment variables")
    parser.add_argument("--no-streaming", action="store_true", help="Disable HF streaming (requires full downloads)")
    parser.add_argument("--no-smoke", action="store_true", help="Disable synthetic fallback samples")
    parser.add_argument("--smoke-samples", type=int, default=1, help="Number of synthetic samples per benchmark when falling back")
    parser.add_argument("--tmp-audio-dir", default=".cache/audio_tmp", help="Temp dir for audio arrays")
    parser.add_argument("--prepare-only", action="store_true", help="Only download/prepare datasets; skip model inference")
    parser.add_argument("--download-all", action="store_true", help="Download public datasets where possible")
    parser.add_argument("--allow-ytdlp", dest="allow_ytdlp", action="store_true", help="Allow yt-dlp downloads for YouTube-sourced audio")
    parser.add_argument("--no-ytdlp", dest="allow_ytdlp", action="store_false", help="Disable yt-dlp downloads for YouTube-sourced audio")
    parser.set_defaults(allow_ytdlp=True)
    parser.add_argument("--skip-missing", action="store_true", help="Skip benchmarks when required data is missing")
    parser.add_argument("--batch", "--batch-size", dest="batch", type=int, default=1, help="Batch size for model inference")
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
    parser.add_argument("--music-instruct-audio-dir", default=None, help="Directory with Music-Instruct audio clips (ytid.wav)")
    parser.add_argument("--music-avqa-root", default=None, help="Root folder containing MusicAVQA videos")
    parser.add_argument("--music-avqa-audio-dir", default=None, help="Directory with extracted MusicAVQA audio (video_id.wav)")
    parser.add_argument("--music-avqa-annotations", default=None, help="Path to MusicAVQA annotations JSON")
    parser.add_argument("--music-avqa-download", action="store_true", help="Download MusicAVQA annotations JSON")
    parser.add_argument("--music-avqa-download-videos", action="store_true", help="Download MusicAVQA videos from Google Drive")
    parser.add_argument("--songcaps-manifest", default=None, help="Path to SongCaps manifest JSONL")
    parser.add_argument("--medley-solos-root", default=None, help="Root folder for Medley-solos-DB")
    parser.add_argument("--medley-solos-download", action="store_true", help="Download Medley-solos-DB from Zenodo")
    parser.add_argument("--opencpop-root", default=None, help="Opencpop dataset root")
    parser.add_argument("--musdb18-root", default=None, help="MUSDB18 dataset root")
    parser.add_argument("--musdb18-audio-stem", default="mixture", help="mixture|vocals|other stem")
    parser.add_argument("--musdb18-download", action="store_true", help="Download MUSDB18-HQ from Zenodo")
    parser.add_argument("--mmau-task", default="", help="Filter MMAU by task (e.g., music)")
    parser.add_argument("--mmau-test-mini-size", type=int, default=100, help="Size of MMAU test-mini subset when not provided")
    parser.add_argument("--mmau-root", default=None, help="Root folder for MMAU json/audio downloads")
    parser.add_argument("--mmau-audio-root", default=None, help="Root folder containing MMAU audio files")
    parser.add_argument("--mmau-download", action="store_true", help="Download MMAU json/audio from official sources")
    parser.add_argument("--mmar-audio-root", default=None, help="Root folder containing MMAR audio files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.allow_smoke = not args.no_smoke
    args.streaming = not args.no_streaming
    if args.no_streaming and not has_torchcodec():
        eprint("torchcodec not available; forcing streaming mode.")
        args.streaming = True
    if args.download_all:
        args.mmau_download = True
        args.medley_solos_download = True
        args.music_avqa_download = True
    args.smoke_used = {}
    overrides = parse_dataset_config(args.dataset_config)
    specs = merge_config(DEFAULT_SPECS.copy(), overrides)

    if args.benchmarks == "all":
        bench_names = PAPER_BENCHMARKS
    else:
        bench_names = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    if args.prepare_only:
        summaries = []
        for name in bench_names:
            spec = specs[name]
            try:
                samples, labels = load_samples_for_benchmark(spec, args)
                summaries.append({"benchmark": name, "num_samples": len(samples), "num_labels": len(labels)})
                eprint(f"Prepared {name}: {len(samples)} samples")
            except Exception as exc:
                summaries.append({"benchmark": name, "error": str(exc)})
                eprint(f"Failed to prepare {name}: {exc}")
        out_dir = args.output_dir
        ensure_dir(out_dir)
        with open(os.path.join(out_dir, "prepare_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        eprint(f"Prepared datasets summary written to {os.path.join(out_dir, 'prepare_summary.json')}")
        return

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
                if args.skip_missing:
                    eprint(f"Skipping {name}: {exc}")
                    continue
                raise
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
            if args.smoke_used.get(name):
                metrics = dict(metrics)
                metrics["status"] = "smoke"
                metrics["note"] = (metrics.get("note", "") + " Synthetic fallback data used.").strip()
            summary[name] = metrics
            with open(os.path.join(bench_out_dir, "metrics.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=True, indent=2))
        with open(os.path.join(model_out_dir, "summary.json"), "w", encoding="utf-8") as f:
            f.write(json.dumps(summary, ensure_ascii=True, indent=2))
        eprint(f"Finished model {model_cfg['label']}. Results in {model_out_dir}")


if __name__ == "__main__":
    main()
