import torch

from gpu_settings import GPUSettings, is_oom_error, load_gpu_settings, save_gpu_settings


def test_load_default_when_missing(tmp_path):
    path = tmp_path / "missing.json"
    settings = load_gpu_settings(path)
    assert isinstance(settings, GPUSettings)
    assert settings.max_gpu_memory_gb == 7.0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "gpu.json"
    settings = GPUSettings(max_gpu_memory_gb=7.5, precision="bf16", enable_cpu_offload=False, device="cpu")
    save_gpu_settings(settings, path)
    loaded = load_gpu_settings(path)
    assert loaded.max_gpu_memory_gb == 7.5
    assert loaded.precision == "bf16"
    assert loaded.enable_cpu_offload is False
    assert loaded.device == "cpu"


def test_is_oom_error_detects_cuda():
    err = torch.cuda.OutOfMemoryError("CUDA out of memory")
    assert is_oom_error(err)
    assert is_oom_error(RuntimeError("CUDA out of memory. Tried to allocate"))
