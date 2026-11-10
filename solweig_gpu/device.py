"""Device selection for solweig_gpu: CUDA, AMD ROCm (torch HIP build, appears as "cuda"), Intel XPU (oneAPI / Level Zero), Apple MPS (macOS Metal) or CPU.

Precedence: the SOLWEIG_DEVICE environment variable ('cuda', 'xpu', 'mps', 'cpu', or a
specific device such as 'xpu:1'), then CUDA if available, then XPU, then MPS, then CPU.
All modules call get_device() instead of hard-coding "cuda".
MPS notes: one device, no float64 tensors (the package uses float32 only),
torch.mps.empty_cache()/synchronize() exist since torch 2.0; set PYTORCH_ENABLE_MPS_FALLBACK=1 to let any
unimplemented op run on the CPU instead of raising.
"""
import os
import torch


def _xpu_available():
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def _mps_available():
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


def backend():
    """Return 'cuda', 'xpu' or 'cpu'."""
    forced = os.environ.get("SOLWEIG_DEVICE", "").strip().lower()
    if forced:
        return forced.split(":")[0]
    if torch.cuda.is_available():
        return "cuda"
    if _xpu_available():
        return "xpu"
    if _mps_available():
        return "mps"
    return "cpu"


def get_device():
    forced = os.environ.get("SOLWEIG_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    return torch.device(backend())


def device_count():
    b = backend()
    if b == "cuda":
        return torch.cuda.device_count()
    if b == "xpu":
        return torch.xpu.device_count()
    if b == "mps":
        return 1
    return 0


def empty_cache():
    b = backend()
    if b == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif b == "xpu" and _xpu_available():
        torch.xpu.empty_cache()
    elif b == "mps" and _mps_available():
        torch.mps.empty_cache()


def synchronize():
    b = backend()
    if b == "cuda":
        torch.cuda.synchronize()
    elif b == "xpu":
        torch.xpu.synchronize()
    elif b == "mps":
        torch.mps.synchronize()


def mask_device(index):
    """Restrict this process to one physical GPU before torch initialises it.

    Sets CUDA_VISIBLE_DEVICES for NVIDIA and ZE_AFFINITY_MASK for Intel GPUs
    (Level Zero). Call it in a worker process before importing anything that
    touches torch devices. Intel Max 1550 cards expose two tiles; with the
    default flat device hierarchy each tile is its own index. MPS has a single
    device, so the masks are harmless there.
    """
    index = str(index)
    os.environ["CUDA_VISIBLE_DEVICES"] = index
    os.environ["HIP_VISIBLE_DEVICES"] = index   # AMD ROCm (torch HIP build); the HIP runtime also honours CUDA_VISIBLE_DEVICES
    os.environ["ZE_AFFINITY_MASK"] = index


def visible_devices():
    """List of device indices usable by drivers that fan out one worker per GPU."""
    n = device_count()
    return list(range(n)) if n else [0]


def describe():
    b = backend()
    if b == "cuda":
        hip = getattr(torch.version, "hip", None)
        kind = f"rocm (HIP {hip}, torch cuda backend)" if hip else "cuda"
        return f"{kind}: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
    if b == "xpu":
        return f"xpu: {torch.xpu.device_count()} x {torch.xpu.get_device_name(0)}"
    if b == "mps":
        return "mps: 1 x Apple GPU (Metal)"
    return "cpu"


def _install_mps_float64_guard():
    """Apple MPS has no float64 tensors. The package builds its scalars with torch.tensor(x, device=device) where x may be a
    numpy float64 (sun position, met-file values); on MPS that raises TypeError. When the backend is mps, wrap torch.tensor so a
    float64 result requested on an mps device is cast to float32 first. No-op on cuda/xpu/cpu."""
    # Decided WITHOUT touching the CUDA/XPU runtimes: torch.xpu.is_available() in a parent process poisons forked
    # workers ("Cannot re-initialize XPU in forked subprocess"), so only the env override and the MPS probe are consulted.
    forced = os.environ.get("SOLWEIG_DEVICE", "").strip().lower().split(":")[0]
    if getattr(torch, "_solweig_mps_guard", False) or (forced and forced != "mps") or (not forced and not _mps_available()):
        return
    _orig = torch.tensor

    def _tensor(data, *args, **kwargs):
        dev = kwargs.get("device")
        if dev is not None and str(dev).startswith("mps") and "dtype" not in kwargs:
            t = _orig(data, *args)
            if t.dtype == torch.float64:
                t = t.float()
            return t.to(dev)
        return _orig(data, *args, **kwargs)

    torch.tensor = _tensor
    torch._solweig_mps_guard = True


_install_mps_float64_guard()
