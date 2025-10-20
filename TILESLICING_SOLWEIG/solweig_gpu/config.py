# solweig_gpu/config.py
from __future__ import annotations
import os
from typing import Mapping

DEFAULT_ENV_VARS: Mapping[str, str] = {
    # Core stability
    "SOLWEIG_FORCE_UTC": "1",
    "SOLWEIG_LOG_EVERY": "6",
    "SOLWEIG_LOG_MEM": "0",

    # GPU / AMP / tiling
    "SOLWEIG_USE_AMP": "1",
    "SOLWEIG_TARGET_TILE_MPX": "2.25",
    "SOLWEIG_MIN_SIDE": "700",
    "SOLWEIG_MAX_TILES_PER_GPU": "16",
    "SOLWEIG_FORCE_TILES_PER_GPU": "",

    # Writer / Repack defaults
    "SOLWEIG_REPACK_TRIM_INTERVAL": "1",
    "SOLWEIG_REPACK_THREADS": "ALL_CPUS",
    "SOLWEIG_REPACK_WORKERS": "auto",  # "auto" = pick based on RAM/CPUs/GPUs
    "SOLWEIG_REPACK_HYGIENE": "1",
    "SOLWEIG_REPACK_FAST_PATH": "1",
    "SOLWEIG_VALIDATE_TIME": "1",
    "SOLWEIG_WRITER_GDAL_CACHE_MB": "256",
    "SOLWEIG_INTERIM_ZSTD_LEVEL": "1",
    "SOLWEIG_WRITER_FLUSH_EVERY": "4",
    "SOLWEIG_BACKPRESSURE_MAXQ": "2000",
    "SOLWEIG_BACKPRESSURE_SLEEP_MS": "50",
    "SOLWEIG_REPACK_DELAY_WHEN_QSIZE": "1000",
    "SOLWEIG_REPACK_PAUSE_MS": "100",   
    # CPU execution
    "SOLWEIG_CPU_WORKERS": "auto",

    # Compression
    "SOLWEIG_ZSTD_LEVEL": "12",
    "SOLWEIG_POST_LERC_ERR_UTCI": "0.12",
    "SOLWEIG_POST_LERC_ERR_TMRT": "0.07",
    "SOLWEIG_POST_LERC_ERR_RAD": "2.0",

    # GDAL / IO
    "GDAL_CACHEMAX": "16384",
    "GDAL_NUM_THREADS": "1",
    "GTIFF_FORCE_BASELINE": "NO",
    "SOLWEIG_GDAL_DISABLE_READDIR": "1",

    # Allocator hygiene
    "PYTHONMALLOC": "malloc",
    "MALLOC_ARENA_MAX": "2",
    "GLIBC_TUNABLES": "glibc.malloc.trim_threshold=131072:glibc.malloc.mmap_threshold=131072",

    # PyTorch CUDA
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:64",
}

def apply_runtime_env_defaults() -> None:
    """Set sane defaults for GDAL, PyTorch and compression related env vars."""
    for key, value in DEFAULT_ENV_VARS.items():
        os.environ.setdefault(key, value)

__all__ = ["apply_runtime_env_defaults", "DEFAULT_ENV_VARS"]
