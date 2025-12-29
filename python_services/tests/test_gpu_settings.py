import torch

from gpu_settings import GPUSettings, is_oom_error, load_gpu_settings, save_gpu_settings


def test_load_default_when_missing(tmp_path):
    path = tmp_path / "missing.conf"
    settings = load_gpu_settings(path)
    assert isinstance(settings, GPUSettings)
    assert settings.max_gpu_memory_gb == 7.0


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "gpu.conf"
    settings = GPUSettings(
        max_gpu_memory_gb=7.5, precision="bf16", enable_cpu_offload=False, device="cpu"
    )
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


def test_device_target_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = GPUSettings(device="auto")
    assert settings.device_target() == "cpu"


def test_max_memory_map_with_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class DummyProps:
        total_memory = 8 * 1024**3

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_args: DummyProps()
    )

    settings = GPUSettings(max_gpu_memory_gb=4.0, enable_cpu_offload=False)
    mapping = settings.max_memory_map()

    assert mapping == {0: "4.0GiB"}


def test_apply_runtime_limits_sets_fraction(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    class DummyProps:
        total_memory = 10 * 1024**3

    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *_args: DummyProps()
    )

    calls = []

    def fake_set(fraction, device):
        calls.append((fraction, device))

    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", fake_set)

    settings = GPUSettings(max_gpu_memory_gb=5.0)
    settings.apply_runtime_limits()

    assert calls


def test_resolve_device_prefers_largest_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    class DummyProps:
        def __init__(self, total_memory):
            self.total_memory = total_memory

    def fake_props(index):
        if index == 0:
            return DummyProps(8 * 1024**3)
        return DummyProps(12 * 1024**3)

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake_props)

    from gpu_settings import resolve_device

    assert resolve_device("auto") == "cuda:1"
