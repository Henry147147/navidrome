"""
Integration test for MuQ embeddings on a real audio file.

Looks for a local music file and skips if none are available.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torchaudio

from embedding_models import MuQEmbeddingModel


def _find_sample_song() -> Path:
    env_path = os.getenv("NAVIDROME_MUQ_TEST_SONG", "").strip()
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path
        pytest.skip(f"NAVIDROME_MUQ_TEST_SONG not found: {path}")

    root = Path("/mnt/data/share/hosted/music")
    if not root.exists():
        pytest.skip("No hosted music directory found for MuQ integration test")

    for ext in (".flac", ".mp3", ".wav", ".m4a", ".ogg"):
        matches = sorted(root.rglob(f"*{ext}"))
        if matches:
            return matches[0]

    pytest.skip("No audio files found for MuQ integration test")


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.requires_models
def test_muq_embedding_real_song_fp16_no_nan():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for fp16 MuQ integration test")

    song_path = _find_sample_song()

    duration_seconds = 30.0
    try:
        info = torchaudio.info(str(song_path))
        num_frames = int(info.sample_rate * duration_seconds)
        waveform, sample_rate = torchaudio.load(
            str(song_path), frame_offset=0, num_frames=num_frames
        )
    except Exception:
        import librosa

        audio, sample_rate = librosa.load(
            str(song_path), sr=None, mono=True, duration=duration_seconds
        )
        waveform = torch.from_numpy(audio).unsqueeze(0)

    model = MuQEmbeddingModel(
        device="cuda",
        model_dtype=torch.float16,
        storage_dtype=torch.float32,
        window_seconds=int(duration_seconds),
        hop_seconds=int(duration_seconds),
        chunk_batch_size=1,
    )

    try:
        embedding = model.embed_audio_tensor(
            waveform, sample_rate, apply_enrichment=True
        )
    finally:
        model.shutdown()

    assert embedding.dtype == torch.float32
    assert torch.isfinite(embedding).all()
    embedding_dim = embedding.shape[0]
    assert embedding_dim % 3 == 0
    assert embedding_dim // 3 in (512, 1024)
