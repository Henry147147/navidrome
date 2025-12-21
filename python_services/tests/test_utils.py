"""
Test utilities and helper functions for embedding_models tests.
"""

from typing import Sequence, Union
import numpy as np
import torch


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


def create_mock_audio_array(
    duration_seconds: float,
    sample_rate: int = 24000,
    channels: int = 1,
    noise_level: float = 0.1,
) -> np.ndarray:
    """
    Create a mock audio array with optional noise.

    Args:
        duration_seconds: Length in seconds
        sample_rate: Sample rate in Hz
        channels: Number of channels (1=mono, 2=stereo)
        noise_level: Amount of random noise to add (0.0 to 1.0)

    Returns:
        Audio array of shape (num_samples,) for mono or (channels, num_samples) for multi-channel
    """
    num_samples = int(duration_seconds * sample_rate)

    if channels == 1:
        # Generate sine wave + noise
        audio = generate_sine_wave(duration_seconds, sample_rate)
        if noise_level > 0:
            noise = np.random.randn(num_samples).astype(np.float32) * noise_level
            audio = audio + noise
        return audio
    else:
        # Generate multi-channel audio
        audio = np.zeros((channels, num_samples), dtype=np.float32)
        for c in range(channels):
            # Use different frequencies for each channel
            freq = 440.0 * (c + 1)
            audio[c] = generate_sine_wave(duration_seconds, sample_rate, frequency=freq)
            if noise_level > 0:
                noise = np.random.randn(num_samples).astype(np.float32) * noise_level
                audio[c] = audio[c] + noise
        return audio


def assert_embedding_valid(
    embedding: Union[Sequence[float], torch.Tensor, np.ndarray],
    expected_dim: int,
    check_normalization: bool = True,
    tolerance: float = 1e-4,
) -> None:
    """
    Assert that an embedding has expected properties.

    Args:
        embedding: The embedding vector to validate
        expected_dim: Expected dimensionality
        check_normalization: Whether to check for L2 normalization
        tolerance: Tolerance for normalization check

    Raises:
        AssertionError: If embedding is invalid
    """
    # Convert to tensor if needed
    if isinstance(embedding, (list, tuple)):
        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
    elif isinstance(embedding, np.ndarray):
        emb_tensor = torch.from_numpy(embedding).float()
    else:
        emb_tensor = embedding.float()

    # Check dimension
    assert emb_tensor.numel() == expected_dim, (
        f"Expected embedding dimension {expected_dim}, " f"but got {emb_tensor.numel()}"
    )

    # Check for NaN or Inf
    assert not torch.isnan(emb_tensor).any(), "Embedding contains NaN values"
    assert not torch.isinf(emb_tensor).any(), "Embedding contains Inf values"

    # Check normalization if requested
    if check_normalization:
        norm = emb_tensor.norm().item()
        assert abs(norm - 1.0) < tolerance, (
            f"Embedding not L2 normalized: norm = {norm:.6f}, "
            f"expected 1.0 ± {tolerance}"
        )


def assert_l2_normalized(
    vector: Union[torch.Tensor, np.ndarray, Sequence[float]],
    tolerance: float = 1e-4,
) -> None:
    """
    Assert that a vector is L2 normalized (norm = 1.0).

    Args:
        vector: Vector to check
        tolerance: Tolerance for norm check

    Raises:
        AssertionError: If vector is not normalized
    """
    if isinstance(vector, (list, tuple)):
        vec_tensor = torch.tensor(vector, dtype=torch.float32)
    elif isinstance(vector, np.ndarray):
        vec_tensor = torch.from_numpy(vector).float()
    else:
        vec_tensor = vector.float()

    norm = vec_tensor.norm().item()
    assert (
        abs(norm - 1.0) < tolerance
    ), f"Vector not L2 normalized: norm = {norm:.6f}, expected 1.0 ± {tolerance}"


class MockMuQModel:
    """
    Mock MuQ MuLan model for testing.
    Simulates audio and text embedding.
    """

    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.device_type = "cpu"
        self.dtype = torch.float32

    def to(self, device):
        """Mock device placement."""
        if isinstance(device, str):
            self.device_type = device
        else:
            self.dtype = device
        return self

    def eval(self):
        """Mock eval mode."""
        return self

    def __call__(self, wavs=None, texts=None):
        """
        Mock forward pass.

        Args:
            wavs: Audio tensor of shape [batch, samples]
            texts: Text string or list of strings

        Returns:
            Embedding tensor of shape [batch, embedding_dim]
        """
        if wavs is not None:
            batch_size = wavs.shape[0] if isinstance(wavs, torch.Tensor) else 1
            return torch.randn(batch_size, self.embedding_dim)
        elif texts is not None:
            if isinstance(texts, str):
                batch_size = 1
            else:
                batch_size = len(texts)
            return torch.randn(batch_size, self.embedding_dim)
        else:
            return torch.randn(1, self.embedding_dim)


def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Assert that two tensors are close in value.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Raises:
        AssertionError: If tensors are not close
    """
    assert (
        tensor1.shape == tensor2.shape
    ), f"Shape mismatch: {tensor1.shape} vs {tensor2.shape}"
    assert torch.allclose(
        tensor1, tensor2, rtol=rtol, atol=atol
    ), "Tensors not close within tolerance"


def count_model_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total parameter count
    """
    return sum(p.numel() for p in model.parameters())


def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    """
    Get approximate memory size of a tensor in bytes.

    Args:
        tensor: PyTorch tensor

    Returns:
        Memory size in bytes
    """
    return tensor.element_size() * tensor.numel()
