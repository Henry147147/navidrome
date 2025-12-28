"""
Integration tests for multi-GPU support.

These tests require actual GPU hardware and will be skipped if:
- CUDA is not available
- Fewer than 2 GPUs are present (for multi-GPU tests)

Run with: pytest tests/test_multi_gpu_integration.py -v
"""

import pytest
import torch

from gpu_settings import get_cuda_device_count, get_device_vram_gb, get_cuda_memory_stats
from gpu_model_coordinator import GPU_COORDINATOR, GPUModelCoordinator


# Skip entire module if CUDA unavailable
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


class TestSingleGPUIntegration:
    """Integration tests that work with any number of GPUs (1+)."""

    def test_coordinator_detects_real_gpus(self):
        """Verify coordinator detects actual GPU hardware."""
        device_count = get_cuda_device_count()
        assert device_count >= 1

        devices = GPU_COORDINATOR.get_available_devices()
        assert len(devices) >= 1
        assert all(d.startswith("cuda:") for d in devices)

    def test_real_vram_detection(self):
        """Verify VRAM detection returns reasonable values."""
        for i in range(get_cuda_device_count()):
            vram = get_device_vram_gb(i)
            # All modern GPUs have at least 1GB VRAM
            assert vram > 1.0
            # Unlikely to have > 100GB VRAM
            assert vram < 100.0

    def test_memory_stats_per_device(self):
        """Verify memory stats work for each device."""
        for i in range(get_cuda_device_count()):
            stats = get_cuda_memory_stats(device=i)
            assert "total_gib" in stats
            assert "allocated_gib" in stats
            assert "free_gib" in stats
            assert stats["device"] == i
            # Free + allocated should roughly equal total
            assert abs(stats["free_gib"] + stats["allocated_gib"] - stats["total_gib"]) < 0.1

    def test_acquire_returns_valid_cuda_device(self):
        """Verify acquire returns a valid CUDA device string."""
        coord = GPUModelCoordinator()
        device = coord.acquire("test_model")

        assert device.startswith("cuda:")
        device_index = int(device.split(":")[1])
        assert device_index < get_cuda_device_count()

    def test_flamingo_device_is_largest_vram(self):
        """Verify Flamingo is assigned to the GPU with most VRAM."""
        coord = GPUModelCoordinator()
        flamingo_device = coord.get_flamingo_device()

        # Get VRAM for all devices
        device_count = get_cuda_device_count()
        vram_per_device = {
            f"cuda:{i}": get_device_vram_gb(i)
            for i in range(device_count)
        }

        # Flamingo's device should have the maximum VRAM
        flamingo_vram = vram_per_device[flamingo_device]
        max_vram = max(vram_per_device.values())
        assert flamingo_vram == max_vram


@pytest.mark.skipif(
    get_cuda_device_count() < 2,
    reason="Requires 2+ GPUs"
)
class TestMultiGPUIntegration:
    """Integration tests requiring 2+ GPUs."""

    def test_different_devices_for_flamingo_and_default(self):
        """Verify Flamingo and default models get different devices."""
        coord = GPUModelCoordinator()

        flamingo_device = coord.get_flamingo_device()
        default_device = coord.get_default_device()

        # With 2+ GPUs, they should be different
        assert flamingo_device != default_device

    def test_flamingo_gets_larger_vram_gpu(self):
        """Verify Flamingo is on the GPU with more VRAM than default."""
        coord = GPUModelCoordinator()

        flamingo_device = coord.get_flamingo_device()
        default_device = coord.get_default_device()

        flamingo_vram = coord.get_device_vram_gb(flamingo_device)
        default_vram = coord.get_device_vram_gb(default_device)

        # Flamingo should have >= VRAM (>= handles equal case)
        assert flamingo_vram >= default_vram

    def test_concurrent_models_different_devices(self):
        """Verify models can be acquired on different devices simultaneously."""
        coord = GPUModelCoordinator()

        # Track offloads
        offloads = {"flamingo": 0, "muq": 0}

        def flamingo_offload():
            offloads["flamingo"] += 1

        def muq_offload():
            offloads["muq"] += 1

        coord.register("music_flamingo_captioner", flamingo_offload)
        coord.register("MuQEmbeddingModel", muq_offload)

        # Acquire both - should be on different devices
        muq_device = coord.acquire("MuQEmbeddingModel")
        flamingo_device = coord.acquire("music_flamingo_captioner")

        # Devices should be different
        assert muq_device != flamingo_device

        # No offloads should have occurred (different devices)
        assert offloads["flamingo"] == 0
        assert offloads["muq"] == 0

    def test_same_device_models_offload_each_other(self):
        """Verify MuQ and Qwen3 (same device) trigger offloads."""
        coord = GPUModelCoordinator()

        offloads = {"muq": 0, "qwen3": 0}

        def muq_offload():
            offloads["muq"] += 1

        def qwen3_offload():
            offloads["qwen3"] += 1

        coord.register("MuQEmbeddingModel", muq_offload)
        coord.register("qwen3_embedder", qwen3_offload)

        # Both should go to default device
        muq_device = coord.acquire("MuQEmbeddingModel")
        qwen3_device = coord.acquire("qwen3_embedder")

        assert muq_device == qwen3_device
        # MuQ should have been offloaded when Qwen3 acquired
        assert offloads["muq"] == 1
        assert offloads["qwen3"] == 0

    def test_max_memory_uses_correct_device_index(self):
        """Verify max_memory dict uses correct device indices."""
        coord = GPUModelCoordinator()

        flamingo_device = coord.get_flamingo_device()
        default_device = coord.get_default_device()

        flamingo_mem = coord.get_max_memory_for_device(flamingo_device)
        default_mem = coord.get_max_memory_for_device(default_device)

        # Extract indices
        flamingo_index = int(flamingo_device.split(":")[1])
        default_index = int(default_device.split(":")[1])

        # Verify the memory dicts have the correct indices
        assert flamingo_index in flamingo_mem
        assert default_index in default_mem
        assert "cpu" in flamingo_mem
        assert "cpu" in default_mem


class TestTensorPlacement:
    """Tests for actual tensor placement on GPUs."""

    def test_tensor_to_acquired_device(self):
        """Verify tensors can be placed on acquired device."""
        coord = GPUModelCoordinator()
        device = coord.acquire("test_placement")

        # Create tensor on acquired device
        tensor = torch.zeros(100, 100, device=device)

        assert str(tensor.device) == device
        assert tensor.is_cuda

    @pytest.mark.skipif(
        get_cuda_device_count() < 2,
        reason="Requires 2+ GPUs"
    )
    def test_tensors_on_different_devices(self):
        """Verify tensors can be placed on different devices simultaneously."""
        coord = GPUModelCoordinator()

        flamingo_device = coord.acquire("music_flamingo_captioner")
        muq_device = coord.acquire("MuQEmbeddingModel")

        # Create tensors on each device
        flamingo_tensor = torch.zeros(100, 100, device=flamingo_device)
        muq_tensor = torch.zeros(100, 100, device=muq_device)

        assert str(flamingo_tensor.device) == flamingo_device
        assert str(muq_tensor.device) == muq_device
        assert flamingo_tensor.device != muq_tensor.device

        # Both should be on CUDA
        assert flamingo_tensor.is_cuda
        assert muq_tensor.is_cuda


class TestVRAMReporting:
    """Tests for VRAM reporting functionality."""

    def test_vram_logging_output(self, caplog):
        """Verify VRAM is logged during offload."""
        import logging

        coord = GPUModelCoordinator()
        logger = logging.getLogger("test_vram")
        logger.setLevel(logging.INFO)

        coord.register("model_a", lambda: None)
        coord.register("model_b", lambda: None)

        coord.acquire("model_a", logger)
        coord.acquire("model_b", logger)

        # Check that VRAM stats were logged
        log_text = caplog.text
        assert "GiB" in log_text or "offloading" in log_text.lower()
