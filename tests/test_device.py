"""Tests for solweig_gpu.device (CUDA / ROCm / Intel XPU / Apple MPS / CPU selection)."""
import os

import torch
import pytest

from solweig_gpu import device as dev


def test_backend_is_known():
    assert dev.backend() in ("cuda", "xpu", "mps", "cpu")
    assert isinstance(dev.get_device(), torch.device)
    assert dev.get_device().type == dev.backend()


def test_env_override(monkeypatch):
    monkeypatch.setenv("SOLWEIG_DEVICE", "cpu")
    assert dev.backend() == "cpu" and dev.get_device() == torch.device("cpu")
    assert dev.device_count() == 0 and dev.visible_devices() == [0]
    monkeypatch.setenv("SOLWEIG_DEVICE", "xpu:1")
    assert dev.backend() == "xpu" and str(dev.get_device()) == "xpu:1"


def test_empty_cache_and_synchronize_are_safe_on_cpu(monkeypatch):
    monkeypatch.setenv("SOLWEIG_DEVICE", "cpu")
    dev.empty_cache()
    dev.synchronize()
    assert dev.describe() == "cpu"


def test_mask_device_sets_all_vendor_masks(monkeypatch):
    for k in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ZE_AFFINITY_MASK"):
        monkeypatch.delenv(k, raising=False)
    dev.mask_device(3)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"
    assert os.environ["HIP_VISIBLE_DEVICES"] == "3"
    assert os.environ["ZE_AFFINITY_MASK"] == "3"


def test_modules_use_device_layer():
    """No module may hard-code CUDA any more."""
    import re, glob
    here = os.path.dirname(dev.__file__)
    for f in glob.glob(os.path.join(here, "*.py")):
        if os.path.basename(f) == "device.py":
            continue
        src = open(f, encoding="utf-8").read()
        assert not re.search(r"torch\.cuda\.", src), f"{os.path.basename(f)} still calls torch.cuda directly"


@pytest.mark.gpu
@pytest.mark.skipif(dev.backend() == "cpu", reason="no GPU backend available")
def test_tensor_on_selected_device():
    t = torch.ones(4, device=dev.get_device())
    assert t.device.type == dev.backend()
    dev.synchronize(); dev.empty_cache()
