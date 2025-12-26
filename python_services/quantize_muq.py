import os
import random
import signal
from glob import glob
from typing import Generator, List

import librosa
import numpy as np
import torch
import tqdm
from muq import MuQ
from optimum.quanto import Calibration, freeze, quantization_map, quantize
from optimum.quanto.tensor.qtype import qfloat
from safetensors.torch import save_file

MODEL_ID = "OpenMuQ/MuQ-large-msd-iter"
SAMPLE_RATE = 240
WINDOW_SECONDS = 120
HOP_SECONDS = 15
CHUNK_BATCH_SIZE = 1
MAX_SONGS = 1000


def iter_audio_chunks(audio: np.ndarray) -> Generator[np.ndarray, None, None]:
    chunk_size = int(WINDOW_SECONDS * SAMPLE_RATE)
    hop_size = int(HOP_SECONDS * SAMPLE_RATE)
    if chunk_size <= 0 or hop_size <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")
    total_samples = int(audio.shape[0])
    if total_samples <= 0:
        return
    last_start = max(total_samples - chunk_size, 0)
    for start_sample in range(0, last_start + 1, hop_size):
        end_sample = min(start_sample + chunk_size, total_samples)
        chunk = audio[start_sample:end_sample]
        observed = int(chunk.shape[0])
        if observed == 0:
            continue
        if observed < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - observed))
        yield chunk.astype("float32", copy=False)
    if last_start % hop_size != 0:
        start_sample = last_start
        end_sample = min(start_sample + chunk_size, total_samples)
        chunk = audio[start_sample:end_sample]
        observed = int(chunk.shape[0])
        if observed == 0:
            return
        if observed < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - observed))
        yield chunk.astype("float32", copy=False)


def load_audio(path: str) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    return audio.astype(np.float32, copy=False)


def run_chunk_batch(model: torch.nn.Module, batch: List[np.ndarray], device: str, dtype: torch.dtype) -> None:
    chunk_matrix = np.stack(batch, axis=0)
    chunk_tensor = torch.from_numpy(chunk_matrix).to(device=device)
    if dtype != torch.float32:
        chunk_tensor = chunk_tensor.to(dtype=dtype)
    with torch.inference_mode():
        _ = model(chunk_tensor)
    if device.startswith("cuda") and torch.cuda.is_available():
        del chunk_tensor
        torch.cuda.empty_cache()


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_dtype = torch.float32

    model = MuQ.from_pretrained(MODEL_ID)
    model = model.to(device=device, dtype=compute_dtype).eval()

    quantize(model, weights=qfloat(torch.float16), activations=qfloat(torch.float16))

    flacs = list(glob("/mnt/data/share/hosted/music/**/*.flac"))
    mp3s = list(glob("/mnt/data/share/hosted/music/**/*.mp3"))

    all_songs = []
    all_songs.extend(flacs)
    all_songs.extend(mp3s)

    all_songs = random.sample(all_songs, min(len(all_songs), MAX_SONGS))

    print("starting calibration")

    stop_requested = False

    def _handle_sigint(signum, frame):
        nonlocal stop_requested
        if not stop_requested:
            print(
                "\nCtrl-C received: will stop after current calibration step and save results..."
            )
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_sigint)

    with Calibration():
        for idx, song in enumerate(tqdm.tqdm(all_songs), start=1):
            if stop_requested:
                print(f"Stopping calibration early at step {idx - 1}/{len(all_songs)}.")
                break
            try:
                audio = load_audio(song)
            except Exception as exc:
                print(f"Failed to load {song}: {exc}")
                continue
            if audio.size == 0:
                continue

            batch: List[np.ndarray] = []
            for chunk in iter_audio_chunks(audio):
                batch.append(chunk)
                if len(batch) >= CHUNK_BATCH_SIZE:
                    run_chunk_batch(model, batch, device, compute_dtype)
                    batch = []
                    if stop_requested:
                        break
            if batch and not stop_requested:
                run_chunk_batch(model, batch, device, compute_dtype)

    if stop_requested:
        print("calibration interrupted, saving results to disk")
    else:
        print("done calibrating")

    freeze(model)
    save_file(model.state_dict(), "muq_fp16.safetensor")

    with open("muq_fp16_quantization_map.json", "w") as f:
        import json

        json.dump(quantization_map(model), f)


if __name__ == "__main__":
    main()
