"""
Centralized GPU memory settings and helpers for Navidrome Python services.

These settings are loaded at service startup (and after restarts) so the UI can
adjust VRAM usage without code changes. Defaults target a 10GB GPU by keeping
the working cap at ~9GB and using fp16 precision with CPU offload enabled.
"""

from __future__ import annotations

import gc
import json
import configparser
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Default path can be overridden from env var so all Python services stay in sync.
_DEFAULT_SETTINGS_PATH = Path(__file__).parent / "gpu_settings.conf"
GPU_SETTINGS_PATH = Path(
    os.getenv("NAVIDROME_GPU_SETTINGS_PATH", _DEFAULT_SETTINGS_PATH)
).expanduser()


@dataclass
class GPUSettings:
    """Configurable GPU limits exposed to the UI."""

    max_gpu_memory_gb: float = 7.0  # Safer default for 10GB cards with headroom
    precision: str = "fp16"  # fp16 | bf16 | fp32
    enable_cpu_offload: bool = True
    device: str = "auto"  # auto|cuda|cpu

    def torch_dtype(self) -> torch.dtype:
        if self.precision == "bf16":
            return torch.bfloat16
        if self.precision == "fp32":
            return torch.float32
        return torch.float16

    def device_target(self) -> str:
        return resolve_device(self.device)

    def max_memory_map(
        self, device_index: Optional[int] = None
    ) -> Optional[Dict[Any, str]]:
        """Return a huggingface-compatible max_memory map."""
        if not torch.cuda.is_available():
            return None
        if device_index is None:
            device_index = 0
        mem_gb = max(float(self.max_gpu_memory_gb), 1.0)
        mapping: Dict[Any, str] = {device_index: f"{mem_gb:.1f}GiB"}
        if self.enable_cpu_offload:
            # Allow offload to CPU when GPU cap is reached
            mapping["cpu"] = "32GiB"
        return mapping

    def max_memory_map_all_gpus(self) -> Optional[Dict[Any, str]]:
        """Return a max_memory map that spans every visible GPU."""
        if not torch.cuda.is_available():
            return None
        mapping: Dict[Any, str] = {}
        for idx in range(get_cuda_device_count()):
            total_gb = get_device_vram_gb(idx)
            cap_gb = min(self.max_gpu_memory_gb, total_gb * 0.80)
            cap_gb = max(cap_gb, 2.0)
            mapping[idx] = f"{cap_gb:.1f}GiB"
        if self.enable_cpu_offload:
            mapping["cpu"] = "32GiB"
        return mapping

    def apply_runtime_limits(self, device_index: Optional[int] = None) -> None:
        """Attempt to cap process VRAM usage so OOMs fail fast and cleanly."""
        if not torch.cuda.is_available():
            return
        if device_index is None:
            device_index = 0
        try:
            total_gb = torch.cuda.get_device_properties(device_index).total_memory / (
                1024**3
            )
            fraction = min(self.max_gpu_memory_gb / total_gb, 0.97)
            fraction = max(fraction, 0.1)
            torch.cuda.set_per_process_memory_fraction(fraction, device_index)
        except Exception:
            # Best-effort only; not all torch builds expose this API
            pass

    def estimated_vram_gb(self, device_index: Optional[int] = None) -> float:
        """Return an approximate peak VRAM usage for display purposes."""
        if not torch.cuda.is_available():
            return 0.0
        if device_index is None:
            device_index = 0
        try:
            total_gb = torch.cuda.get_device_properties(device_index).total_memory / (
                1024**3
            )
            return round(min(total_gb, self.max_gpu_memory_gb), 2)
        except Exception:
            return round(self.max_gpu_memory_gb, 2)


def _load_from_file(path: Path) -> Optional[GPUSettings]:
    if not path.exists():
        return None
    # First try ini/conf format
    try:
        cp = configparser.ConfigParser()
        cp.read(path)
        if "gpu" in cp:
            section = cp["gpu"]
            return GPUSettings(
                max_gpu_memory_gb=float(section.get("max_gpu_memory_gb", 9.0)),
                precision=section.get("precision", "fp16"),
                enable_cpu_offload=section.getboolean("enable_cpu_offload", True),
                device=section.get("device", "auto"),
            )
    except Exception:
        pass

    # Backward compatibility: accept legacy JSON
    try:
        data = json.loads(path.read_text())
        return GPUSettings(
            max_gpu_memory_gb=float(data.get("maxGpuMemoryGb", 9.0)),
            precision=str(data.get("precision", "fp16")),
            enable_cpu_offload=bool(data.get("enableCpuOffload", True)),
            device=str(data.get("device", "auto")),
        )
    except Exception:
        return None


def load_gpu_settings(path: Optional[Path] = None) -> GPUSettings:
    """Load GPU settings from disk or return sane defaults."""
    settings_path = path or GPU_SETTINGS_PATH
    loaded = _load_from_file(settings_path)
    cfg = loaded or GPUSettings()

    # Clamp to available GPU memory if possible to avoid immediate OOM on load
    if torch.cuda.is_available():
        try:
            device_index = parse_device_index(resolve_device(cfg.device))
            total_bytes = torch.cuda.get_device_properties(device_index).total_memory
            total_gb = total_bytes / (1024**3)
            max_cap = max(min(cfg.max_gpu_memory_gb, total_gb * 0.85), 2.0)
            cfg.max_gpu_memory_gb = round(max_cap, 2)
        except Exception:
            pass
    return cfg


def save_gpu_settings(settings: GPUSettings, path: Optional[Path] = None) -> None:
    """Persist GPU settings to an .conf (ini) file."""
    settings_path = path or GPU_SETTINGS_PATH
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    cp = configparser.ConfigParser()
    cp["gpu"] = {
        "max_gpu_memory_gb": str(settings.max_gpu_memory_gb),
        "precision": settings.precision,
        "enable_cpu_offload": str(settings.enable_cpu_offload).lower(),
        "device": settings.device,
    }
    with settings_path.open("w") as fh:
        cp.write(fh)


def is_oom_error(exc: Exception) -> bool:
    """Heuristic to spot CUDA/CPU OOM errors."""
    msg = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in msg


def force_cuda_memory_release() -> None:
    """
    Aggressively release CUDA memory with proper synchronization.

    This ensures:
    1. All pending CUDA operations complete (synchronize)
    2. Python garbage collector runs to release tensor references
    3. PyTorch's CUDA cache is emptied
    """
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def get_cuda_memory_stats(device: int = 0) -> Dict[str, Any]:
    """Get current CUDA memory statistics for debugging.

    Args:
        device: CUDA device index (default: 0)
    """
    if not torch.cuda.is_available():
        return {"available": False}
    try:
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        total = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        return {
            "device": device,
            "allocated_gib": round(allocated, 2),
            "total_gib": round(total, 2),
            "free_gib": round(total - allocated, 2),
        }
    except Exception:
        return {"error": "Unable to query"}


def get_cuda_device_count() -> int:
    """Return number of available CUDA devices, or 0 if CUDA unavailable."""
    if not torch.cuda.is_available():
        return 0
    try:
        return torch.cuda.device_count()
    except Exception:
        return 0


def resolve_device(device: Optional[str]) -> str:
    """
    Resolve a device string to a concrete target (cpu or cuda:<index>).

    For auto/cuda targets on multi-GPU systems, the GPU with the most VRAM
    is selected to maximize headroom for large models.
    """
    if device is None:
        device = "auto"
    device = str(device).strip()
    if device == "cpu":
        return "cpu"
    if device.startswith("cuda:"):
        if not torch.cuda.is_available():
            return "cpu"
        try:
            index = int(device.split(":")[1])
        except (ValueError, IndexError):
            return "cpu"
        if index < 0 or index >= get_cuda_device_count():
            return "cpu"
        return f"cuda:{index}"
    if device in {"cuda", "auto"}:
        if not torch.cuda.is_available():
            return "cpu"
        best_index = 0
        best_vram = 0.0
        for idx in range(get_cuda_device_count()):
            vram = get_device_vram_gb(idx)
            if vram > best_vram:
                best_vram = vram
                best_index = idx
        return f"cuda:{best_index}"
    return device


def get_device_vram_gb(device_index: int) -> float:
    """Return total VRAM in GiB for the specified CUDA device.

    Args:
        device_index: CUDA device index

    Returns:
        Total VRAM in GiB, or 0.0 if device unavailable
    """
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(device_index)
        return props.total_memory / (1024**3)
    except Exception:
        return 0.0


def parse_device_index(device: str) -> int:
    if device.startswith("cuda:"):
        try:
            return int(device.split(":")[1])
        except (ValueError, IndexError):
            return 0
    if device == "cuda":
        return 0
    return 0


# Encourage PyTorch to use expandable segments to reduce fragmentation OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
