"""
Coordinator to manage GPU access across multiple models with multi-GPU support.

Each model registers an offload callback that moves its weights to CPU. The coordinator
tracks which model owns each GPU device and handles offloading when a new model needs
to claim a device that's already occupied.

Multi-GPU behavior:
- Music Flamingo is assigned to the GPU with the most VRAM
- Other models (MuQ, Qwen3) share the remaining GPU(s)
- Models on different GPUs can run simultaneously without offloading each other
- Single-GPU systems behave exactly as before (all models swap on the same device)
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple


# Owner name for Music Flamingo (gets priority for largest GPU)
FLAMINGO_OWNER = "music_flamingo_captioner"


class GPUModelCoordinator:
    """
    Manages GPU device assignment and model offloading for multi-GPU systems.

    The coordinator automatically detects available GPUs at initialization and
    assigns devices based on VRAM size:
    - Music Flamingo always gets the GPU with the most VRAM
    - Other models share the remaining GPU(s), with offloading between them
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._offloaders: Dict[str, Callable[[], None]] = {}
        # Per-device current owner: {"cuda:0": "muq", "cuda:1": "flamingo"}
        self._device_owners: Dict[str, Optional[str]] = {}
        # Reverse mapping: owner -> device
        self._owner_devices: Dict[str, str] = {}
        # Device assignment computed at init
        self._flamingo_device: str = "cpu"
        self._default_device: str = "cpu"
        self._available_devices: List[str] = []
        self._device_vram: Dict[str, float] = {}
        self._initialize_devices()

    def _initialize_devices(self) -> None:
        """Detect GPUs and assign roles based on VRAM."""
        from gpu_settings import get_cuda_device_count, get_device_vram_gb

        device_count = get_cuda_device_count()
        if device_count == 0:
            # No CUDA available - everything runs on CPU
            self._flamingo_device = "cpu"
            self._default_device = "cpu"
            self._available_devices = ["cpu"]
            self._device_owners["cpu"] = None
            return

        # Build list of (device_str, vram_gb) sorted by VRAM descending
        devices_with_vram: List[Tuple[str, float]] = []
        for i in range(device_count):
            device_str = f"cuda:{i}"
            vram = get_device_vram_gb(i)
            devices_with_vram.append((device_str, vram))
            self._device_vram[device_str] = vram

        # Sort by VRAM descending, then by device index ascending for stability
        devices_with_vram.sort(key=lambda x: (-x[1], x[0]))

        self._available_devices = [d[0] for d in devices_with_vram]

        # Flamingo gets the largest GPU
        self._flamingo_device = devices_with_vram[0][0]

        # Other models get the second largest (or same device if only one GPU)
        if len(devices_with_vram) > 1:
            self._default_device = devices_with_vram[1][0]
        else:
            self._default_device = self._flamingo_device

        # Initialize owner tracking for all devices
        for device_str, _ in devices_with_vram:
            self._device_owners[device_str] = None

    def register(self, owner: str, offload_fn: Callable[[], None]) -> None:
        """
        Register an owner with its offload callback.

        Args:
            owner: Unique identifier for the model (e.g., "muq", "music_flamingo_captioner")
            offload_fn: Callback to move model to CPU and free VRAM
        """
        if not owner:
            return
        with self._lock:
            self._offloaders[owner] = offload_fn

    def _get_device_for_owner(self, owner: str) -> str:
        """Determine which device this owner should use."""
        if owner == FLAMINGO_OWNER:
            return self._flamingo_device
        return self._default_device

    def acquire(self, owner: str, logger: Optional[logging.Logger] = None) -> str:
        """
        Request GPU access for owner. Returns the device string to use.

        This method:
        1. Determines the correct device for the owner
        2. Offloads the previous owner on that device (if different)
        3. Records the new owner
        4. Returns the device string (e.g., "cuda:0", "cuda:1", or "cpu")

        Args:
            owner: Unique identifier for the model
            logger: Optional logger for debug output

        Returns:
            Device string to use (e.g., "cuda:1")
        """
        if not owner:
            return self._default_device

        device = self._get_device_for_owner(owner)
        offload_prev: Optional[Callable[[], None]] = None
        prev_owner: Optional[str] = None

        with self._lock:
            current = self._device_owners.get(device)
            if current and current != owner:
                offload_prev = self._offloaders.get(current)
                prev_owner = current
            self._device_owners[device] = owner
            self._owner_devices[owner] = device

        if offload_prev:
            self._do_offload(offload_prev, prev_owner, owner, device, logger)

        return device

    def _do_offload(
        self,
        offload_fn: Callable[[], None],
        prev_owner: Optional[str],
        new_owner: str,
        device: str,
        logger: Optional[logging.Logger],
    ) -> None:
        """Execute offload callback with logging and memory cleanup."""
        from gpu_settings import force_cuda_memory_release, get_cuda_memory_stats

        device_index = self._parse_device_index(device)

        if logger:
            stats_before = get_cuda_memory_stats(device_index)
            logger.info(
                "GPU acquire [%s]: offloading %s (%.2f/%.2f GiB before)",
                device,
                prev_owner,
                stats_before.get("allocated_gib", 0),
                stats_before.get("total_gib", 0),
            )
        try:
            offload_fn()
        except Exception:
            if logger:
                logger.exception(
                    "Failed to offload previous GPU owner %s on %s", prev_owner, device
                )
        force_cuda_memory_release()
        if logger:
            stats_after = get_cuda_memory_stats(device_index)
            logger.info(
                "GPU acquire [%s]: %s -> %s (%.2f/%.2f GiB after offload)",
                device,
                prev_owner,
                new_owner,
                stats_after.get("allocated_gib", 0),
                stats_after.get("total_gib", 0),
            )

    def _parse_device_index(self, device: str) -> int:
        """Extract device index from device string like 'cuda:1'."""
        if device == "cpu":
            return 0
        if ":" in device:
            try:
                return int(device.split(":")[1])
            except (ValueError, IndexError):
                return 0
        return 0

    def get_max_memory_for_device(
        self, device: str, gpu_cap_gb: float = 7.5, cpu_cap_gb: float = 32.0
    ) -> Optional[Dict[Any, str]]:
        """
        Return a HuggingFace-compatible max_memory dict for the specified device.

        Args:
            device: Device string (e.g., "cuda:0", "cuda:1")
            gpu_cap_gb: Maximum GPU memory to use in GiB
            cpu_cap_gb: Maximum CPU memory for offloading in GiB

        Returns:
            Dict like {0: "7.5GiB", "cpu": "32GiB"} for device_map="auto",
            or None if device is CPU
        """
        if device == "cpu":
            return None
        device_index = self._parse_device_index(device)
        return {device_index: f"{gpu_cap_gb:.1f}GiB", "cpu": f"{cpu_cap_gb:.1f}GiB"}

    def current_owner(self, device: Optional[str] = None) -> Optional[str]:
        """
        Get current owner of a device.

        Args:
            device: Device to query. If None, returns owner of default device.

        Returns:
            Owner name or None if no owner
        """
        with self._lock:
            if device is None:
                device = self._default_device
            return self._device_owners.get(device)

    def get_device_for_owner(self, owner: str) -> str:
        """
        Get the device assigned to an owner.

        Args:
            owner: Owner name

        Returns:
            Device string, or default device if owner not registered
        """
        with self._lock:
            return self._owner_devices.get(owner, self._default_device)

    def get_flamingo_device(self) -> str:
        """Return the device assigned to Music Flamingo (largest VRAM)."""
        return self._flamingo_device

    def get_default_device(self) -> str:
        """Return the default device for non-Flamingo models."""
        return self._default_device

    def get_available_devices(self) -> List[str]:
        """Return list of available devices sorted by VRAM descending."""
        return self._available_devices.copy()

    def get_device_vram_gb(self, device: str) -> float:
        """Return VRAM in GiB for a device, or 0.0 if unknown."""
        return self._device_vram.get(device, 0.0)

    # Backward compatibility: alias for existing code using claim()
    def claim(self, owner: str, logger: Optional[logging.Logger] = None) -> str:
        """
        Backward-compatible alias for acquire().

        Returns the device string (new behavior) instead of None (old behavior).
        """
        return self.acquire(owner, logger)


GPU_COORDINATOR = GPUModelCoordinator()
