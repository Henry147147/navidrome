"""
Centralized GPU memory settings and helpers for Navidrome Python services.

These settings are loaded at service startup (and after restarts) so the UI can
adjust VRAM usage without code changes. Defaults target a 10GB GPU by keeping
the working cap at ~9GB and using fp16 precision with CPU offload enabled.
"""

from __future__ import annotations

import json
import configparser
import os
from dataclasses import asdict, dataclass
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
        if self.device == "cpu":
            return "cpu"
        if self.device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        # auto
        return "cuda" if torch.cuda.is_available() else "cpu"

    def max_memory_map(self) -> Optional[Dict[Any, str]]:
        """Return a huggingface-compatible max_memory map."""
        if not torch.cuda.is_available():
            return None
        mem_gb = max(float(self.max_gpu_memory_gb), 1.0)
        mapping: Dict[Any, str] = {0: f"{mem_gb:.1f}GiB"}
        if self.enable_cpu_offload:
            # Allow offload to CPU when GPU cap is reached
            mapping["cpu"] = "32GiB"
        return mapping

    def apply_runtime_limits(self) -> None:
        """Attempt to cap process VRAM usage so OOMs fail fast and cleanly."""
        if not torch.cuda.is_available():
            return
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            fraction = min(self.max_gpu_memory_gb / total_gb, 0.97)
            fraction = max(fraction, 0.1)
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
        except Exception:
            # Best-effort only; not all torch builds expose this API
            pass

    def estimated_vram_gb(self) -> float:
        """Return an approximate peak VRAM usage for display purposes."""
        if not torch.cuda.is_available():
            return 0.0
        try:
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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
            total_bytes = torch.cuda.get_device_properties(0).total_memory
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


# Encourage PyTorch to use expandable segments to reduce fragmentation OOMs.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
