"""
Shared pytest fixtures for embedding_models test suite.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torchaudio

# Ensure torchaudio uses soundfile backend (avoids torchcodec dependency in tests)
os.environ.setdefault("TORCHAUDIO_USE_SOUNDFILE", "1")

# Provide missing torchaudio.info/save shims when backend is minimal
if not hasattr(torchaudio, "info"):

    def _dummy_info(path):
        class _Info:
            sample_rate = 44100
            num_channels = 2
            num_frames = 0

        return _Info()

    torchaudio.info = _dummy_info  # type: ignore[attr-defined]

if not hasattr(torchaudio, "_original_save"):
    torchaudio._original_save = getattr(torchaudio, "save", None)


def _sf_save(path, waveform, sample_rate, format=None, **kwargs):
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - fallback
        raise RuntimeError(
            "soundfile backend required for torchaudio.save in tests"
        ) from exc
    # waveform expected as torch.Tensor [channels, frames]
    data = waveform.detach().cpu().numpy()
    if data.ndim == 2:
        data = data.T  # soundfile expects frames x channels
    sf.write(path, data, sample_rate, format=format or "FLAC")


# Replace torchaudio.save if it fails to import torchcodec
try:
    # quick sanity check to ensure save callable
    getattr(torchaudio, "save")
except Exception:
    torchaudio.save = _sf_save  # type: ignore[assignment]
else:
    # Even if present, force use of soundfile to avoid codec dependency
    torchaudio.save = _sf_save  # type: ignore[assignment]

from models import TrackSegment


@pytest.fixture
def logger() -> logging.Logger:
    """Provide a test logger."""
    test_logger = logging.getLogger("test_embedding_models")
    test_logger.setLevel(logging.DEBUG)
    return test_logger


@pytest.fixture
def mock_milvus_client() -> Mock:
    """
    Mock MilvusClient with recording capabilities.
    Tracks method calls for verification in tests.
    """
    client = Mock()
    client.list_collections.return_value = []
    client.describe_collection.return_value = {"indexes": []}
    client.create_collection = Mock()
    client.create_index = Mock()
    client.insert = Mock()
    client.search = Mock(return_value=[])
    return client


@pytest.fixture
def sample_track_segment() -> TrackSegment:
    """Provide a standard TrackSegment for testing."""
    return TrackSegment(
        index=1,
        title="Test Track",
        start=0.0,
        end=None,
    )


@pytest.fixture
def short_track_segment() -> TrackSegment:
    """Provide a short TrackSegment (10 seconds)."""
    return TrackSegment(
        index=1,
        title="Short Track",
        start=0.0,
        end=10.0,
    )


@pytest.fixture
def long_track_segment() -> TrackSegment:
    """Provide a long TrackSegment (180 seconds / 3 minutes)."""
    return TrackSegment(
        index=1,
        title="Long Track",
        start=0.0,
        end=180.0,
    )


def generate_sine_wave(
    duration_seconds: float,
    sample_rate: int = 24000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Generate a sine wave audio signal.

    Args:
        duration_seconds: Length of audio in seconds
        sample_rate: Sample rate in Hz
        frequency: Frequency of sine wave in Hz (default: A4 = 440 Hz)
        amplitude: Amplitude (0.0 to 1.0)

    Returns:
        1D numpy array of float32 audio samples
    """
    num_samples = int(duration_seconds * sample_rate)
    t = np.linspace(0, duration_seconds, num_samples, dtype=np.float32)
    audio = amplitude * np.sin(2 * np.pi * frequency * t)
    return audio.astype(np.float32)


@pytest.fixture
def synthetic_audio_generator():
    """
    Fixture that provides the generate_sine_wave function.
    Use this in tests that need to generate audio on demand.
    """
    return generate_sine_wave


@pytest.fixture
def short_audio_array(synthetic_audio_generator) -> np.ndarray:
    """Generate 1 second of audio at 24kHz."""
    return synthetic_audio_generator(duration_seconds=1.0, sample_rate=24000)


@pytest.fixture
def medium_audio_array(synthetic_audio_generator) -> np.ndarray:
    """Generate 30 seconds of audio at 24kHz."""
    return synthetic_audio_generator(duration_seconds=30.0, sample_rate=24000)


@pytest.fixture
def long_audio_array(synthetic_audio_generator) -> np.ndarray:
    """Generate 120 seconds of audio at 24kHz."""
    return synthetic_audio_generator(duration_seconds=120.0, sample_rate=24000)


@pytest.fixture
def temp_audio_file(
    tmp_path: Path, synthetic_audio_generator
) -> Generator[Path, None, None]:
    """
    Create a temporary audio file for testing.
    Returns path to WAV file containing 10 seconds of sine wave.
    File is automatically cleaned up after test.
    """
    import soundfile as sf

    audio_path = tmp_path / "test_audio.wav"
    audio = synthetic_audio_generator(duration_seconds=10.0, sample_rate=24000)

    try:
        sf.write(str(audio_path), audio, 24000)
        yield audio_path
    except ImportError:
        # If soundfile not available, create a dummy file
        audio_path.write_text("")
        yield audio_path


@pytest.fixture
def mock_torch_device() -> str:
    """
    Provide a mock device string for testing.
    Always returns 'cpu' to avoid CUDA requirements in tests.
    """
    return "cpu"


@pytest.fixture
def mock_muq_model():
    """
    Mock MuQ MuLan model for testing.
    Returns embeddings of shape [batch, 512].
    """
    model = Mock()
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value=model)

    def forward(wavs=None, texts=None):
        """Mock forward pass."""
        if wavs is not None:
            batch_size = wavs.shape[0] if isinstance(wavs, torch.Tensor) else 1
            return torch.randn(batch_size, 512)
        if texts is not None:
            return torch.randn(1, 512)
        return torch.randn(1, 512)

    model.__call__ = forward
    return model


@pytest.fixture(autouse=False)
def no_cuda(monkeypatch):
    """
    Fixture to disable CUDA for testing.
    Use this with @pytest.mark.usefixtures("no_cuda") to force CPU execution.
    """
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
