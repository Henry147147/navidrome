"""
Unit tests for multi-GPU support in the GPU model coordinator.

These tests use mocking to simulate different GPU configurations without
requiring actual hardware. They verify:
- Device assignment based on VRAM
- Offload behavior between models on same/different devices
- Backward compatibility with single-GPU systems
- Edge cases (no CUDA, equal VRAM, etc.)
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock


class MockDeviceProperties:
    """Mock for torch.cuda.get_device_properties()."""

    def __init__(self, total_memory_gb: float):
        self.total_memory = int(total_memory_gb * (1024**3))


def create_coordinator_with_mock_gpus(
    monkeypatch, device_count: int, vram_per_device: list[float]
):
    """
    Create a fresh GPUModelCoordinator with mocked GPU configuration.

    Args:
        monkeypatch: pytest monkeypatch fixture
        device_count: Number of CUDA devices to simulate
        vram_per_device: List of VRAM sizes in GB for each device
    """
    # Must mock before importing to avoid module-level initialization issues
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: device_count)

    def mock_get_device_properties(device_index):
        if device_index < len(vram_per_device):
            return MockDeviceProperties(vram_per_device[device_index])
        raise RuntimeError(f"Invalid device index: {device_index}")

    monkeypatch.setattr(torch.cuda, "get_device_properties", mock_get_device_properties)

    # Import fresh module to pick up mocked values
    import importlib
    import gpu_model_coordinator

    importlib.reload(gpu_model_coordinator)
    return gpu_model_coordinator.GPUModelCoordinator()


class TestDeviceDetection:
    """Tests for GPU detection and device assignment."""

    def test_no_cuda_returns_cpu(self, monkeypatch):
        """When CUDA is unavailable, all devices should be CPU."""
        coord = create_coordinator_with_mock_gpus(monkeypatch, device_count=0, vram_per_device=[])

        assert coord.get_flamingo_device() == "cpu"
        assert coord.get_default_device() == "cpu"
        assert coord.get_available_devices() == ["cpu"]

    def test_single_gpu_same_device_for_all(self, monkeypatch):
        """With 1 GPU, all models share the same device."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[12.0]
        )

        assert coord.get_flamingo_device() == "cuda:0"
        assert coord.get_default_device() == "cuda:0"
        assert coord.get_available_devices() == ["cuda:0"]

    def test_two_gpus_flamingo_gets_larger(self, monkeypatch):
        """With 2 GPUs of different sizes, Flamingo gets the larger one."""
        # Device 0: 12GB, Device 1: 24GB
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        # Flamingo should get cuda:1 (24GB - the larger one)
        assert coord.get_flamingo_device() == "cuda:1"
        # Other models should get cuda:0 (12GB)
        assert coord.get_default_device() == "cuda:0"

    def test_two_gpus_reversed_order(self, monkeypatch):
        """Verify correct assignment when larger GPU is device 0."""
        # Device 0: 24GB, Device 1: 12GB
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[24.0, 12.0]
        )

        # Flamingo should get cuda:0 (24GB - the larger one)
        assert coord.get_flamingo_device() == "cuda:0"
        # Other models should get cuda:1 (12GB)
        assert coord.get_default_device() == "cuda:1"

    def test_equal_vram_uses_lower_index_for_flamingo(self, monkeypatch):
        """When GPUs have equal VRAM, use lower index for stability."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[16.0, 16.0]
        )

        # With equal VRAM, cuda:0 should be picked for flamingo (lower index after sorting)
        assert coord.get_flamingo_device() == "cuda:0"
        assert coord.get_default_device() == "cuda:1"

    def test_three_gpus_flamingo_gets_largest(self, monkeypatch):
        """With 3+ GPUs, Flamingo still gets the largest."""
        # Device 0: 8GB, Device 1: 24GB, Device 2: 16GB
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=3, vram_per_device=[8.0, 24.0, 16.0]
        )

        # Flamingo should get cuda:1 (24GB - the largest)
        assert coord.get_flamingo_device() == "cuda:1"
        # Default should get the second largest (cuda:2 = 16GB)
        assert coord.get_default_device() == "cuda:2"


class TestDeviceAcquisition:
    """Tests for the acquire() method."""

    def test_acquire_returns_correct_device_for_flamingo(self, monkeypatch):
        """Flamingo acquisition returns the largest GPU."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        device = coord.acquire("music_flamingo_captioner")
        assert device == "cuda:1"

    def test_acquire_returns_correct_device_for_muq(self, monkeypatch):
        """MuQ acquisition returns the default GPU."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        device = coord.acquire("MuQEmbeddingModel")
        assert device == "cuda:0"

    def test_acquire_returns_correct_device_for_qwen3(self, monkeypatch):
        """Qwen3 acquisition returns the default GPU (same as MuQ)."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        device = coord.acquire("qwen3_embedder")
        assert device == "cuda:0"

    def test_acquire_records_ownership(self, monkeypatch):
        """Acquisition records the owner of each device."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        coord.acquire("music_flamingo_captioner")
        coord.acquire("MuQEmbeddingModel")

        assert coord.current_owner("cuda:1") == "music_flamingo_captioner"
        assert coord.current_owner("cuda:0") == "MuQEmbeddingModel"

    def test_acquire_single_gpu_all_same_device(self, monkeypatch):
        """With single GPU, all models get cuda:0."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        assert coord.acquire("music_flamingo_captioner") == "cuda:0"
        assert coord.acquire("MuQEmbeddingModel") == "cuda:0"
        assert coord.acquire("qwen3_embedder") == "cuda:0"


class TestOffloadBehavior:
    """Tests for model offloading behavior."""

    def test_same_device_triggers_offload(self, monkeypatch):
        """Models on the same device should trigger offloading."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        # Register offload callbacks
        muq_offloaded = []
        qwen3_offloaded = []
        coord.register("MuQEmbeddingModel", lambda: muq_offloaded.append(True))
        coord.register("qwen3_embedder", lambda: qwen3_offloaded.append(True))

        # MuQ acquires first
        coord.acquire("MuQEmbeddingModel")
        assert len(muq_offloaded) == 0
        assert len(qwen3_offloaded) == 0

        # Qwen3 acquires - should trigger MuQ offload
        coord.acquire("qwen3_embedder")
        assert len(muq_offloaded) == 1
        assert len(qwen3_offloaded) == 0

        # MuQ acquires again - should trigger Qwen3 offload
        coord.acquire("MuQEmbeddingModel")
        assert len(muq_offloaded) == 1
        assert len(qwen3_offloaded) == 1

    def test_different_devices_no_offload(self, monkeypatch):
        """Models on different devices should NOT trigger offloading."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        # Register offload callbacks
        muq_offloaded = []
        flamingo_offloaded = []
        coord.register("MuQEmbeddingModel", lambda: muq_offloaded.append(True))
        coord.register("music_flamingo_captioner", lambda: flamingo_offloaded.append(True))

        # MuQ acquires cuda:0
        coord.acquire("MuQEmbeddingModel")
        # Flamingo acquires cuda:1 - should NOT trigger MuQ offload
        coord.acquire("music_flamingo_captioner")

        assert len(muq_offloaded) == 0
        assert len(flamingo_offloaded) == 0

    def test_same_owner_no_offload(self, monkeypatch):
        """Re-acquiring the same device by the same owner shouldn't offload."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        offloaded = []
        coord.register("MuQEmbeddingModel", lambda: offloaded.append(True))

        coord.acquire("MuQEmbeddingModel")
        coord.acquire("MuQEmbeddingModel")
        coord.acquire("MuQEmbeddingModel")

        assert len(offloaded) == 0

    def test_offload_failure_handled_gracefully(self, monkeypatch):
        """Offload callback failures should be caught and logged."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        def failing_offload():
            raise RuntimeError("Offload failed!")

        coord.register("MuQEmbeddingModel", failing_offload)

        coord.acquire("MuQEmbeddingModel")
        # This should not raise despite the failing offload callback
        coord.acquire("qwen3_embedder")


class TestMaxMemoryGeneration:
    """Tests for generating HuggingFace-compatible max_memory dicts."""

    def test_max_memory_for_cuda_device(self, monkeypatch):
        """max_memory dict should use correct device index."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        # For cuda:1
        mem = coord.get_max_memory_for_device("cuda:1", gpu_cap_gb=8.0)
        assert mem == {1: "8.0GiB", "cpu": "32.0GiB"}

        # For cuda:0
        mem = coord.get_max_memory_for_device("cuda:0", gpu_cap_gb=6.0)
        assert mem == {0: "6.0GiB", "cpu": "32.0GiB"}

    def test_max_memory_for_cpu_returns_none(self, monkeypatch):
        """CPU device should return None for max_memory."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=0, vram_per_device=[]
        )

        mem = coord.get_max_memory_for_device("cpu")
        assert mem is None

    def test_max_memory_custom_cpu_cap(self, monkeypatch):
        """Custom CPU cap should be respected."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        mem = coord.get_max_memory_for_device("cuda:0", gpu_cap_gb=10.0, cpu_cap_gb=64.0)
        assert mem == {0: "10.0GiB", "cpu": "64.0GiB"}


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_claim_is_alias_for_acquire(self, monkeypatch):
        """claim() should behave identically to acquire()."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        device1 = coord.claim("music_flamingo_captioner")
        device2 = coord.acquire("music_flamingo_captioner")

        assert device1 == device2 == "cuda:1"

    def test_claim_returns_device_string(self, monkeypatch):
        """claim() now returns device string (changed from None in old API)."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        device = coord.claim("MuQEmbeddingModel")
        assert device == "cuda:0"
        assert isinstance(device, str)


class TestHelperMethods:
    """Tests for helper methods."""

    def test_get_device_for_owner(self, monkeypatch):
        """get_device_for_owner returns correct device after acquisition."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        coord.acquire("MuQEmbeddingModel")
        assert coord.get_device_for_owner("MuQEmbeddingModel") == "cuda:0"

    def test_get_device_vram_gb(self, monkeypatch):
        """get_device_vram_gb returns correct VRAM values."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=2, vram_per_device=[12.0, 24.0]
        )

        assert coord.get_device_vram_gb("cuda:0") == 12.0
        assert coord.get_device_vram_gb("cuda:1") == 24.0
        assert coord.get_device_vram_gb("cuda:99") == 0.0  # Unknown device

    def test_parse_device_index(self, monkeypatch):
        """_parse_device_index extracts correct index."""
        coord = create_coordinator_with_mock_gpus(
            monkeypatch, device_count=1, vram_per_device=[16.0]
        )

        assert coord._parse_device_index("cuda:0") == 0
        assert coord._parse_device_index("cuda:1") == 1
        assert coord._parse_device_index("cuda:42") == 42
        assert coord._parse_device_index("cpu") == 0
        assert coord._parse_device_index("invalid") == 0


class TestGPUSettingsHelpers:
    """Tests for gpu_settings helper functions."""

    def test_get_cuda_device_count_no_cuda(self, monkeypatch):
        """get_cuda_device_count returns 0 when CUDA unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        from gpu_settings import get_cuda_device_count

        assert get_cuda_device_count() == 0

    def test_get_cuda_device_count_with_cuda(self, monkeypatch):
        """get_cuda_device_count returns correct count."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

        from gpu_settings import get_cuda_device_count

        assert get_cuda_device_count() == 3

    def test_get_device_vram_gb_no_cuda(self, monkeypatch):
        """get_device_vram_gb returns 0 when CUDA unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        from gpu_settings import get_device_vram_gb

        assert get_device_vram_gb(0) == 0.0

    def test_get_device_vram_gb_with_cuda(self, monkeypatch):
        """get_device_vram_gb returns correct VRAM."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda idx: MockDeviceProperties(24.0),
        )

        from gpu_settings import get_device_vram_gb

        assert get_device_vram_gb(0) == 24.0

    def test_get_cuda_memory_stats_with_device_param(self, monkeypatch):
        """get_cuda_memory_stats accepts device parameter."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda dev: 2 * (1024**3))
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda dev: MockDeviceProperties(16.0),
        )

        from gpu_settings import get_cuda_memory_stats

        stats = get_cuda_memory_stats(device=1)
        assert stats["device"] == 1
        assert stats["total_gib"] == 16.0
        assert stats["allocated_gib"] == 2.0
