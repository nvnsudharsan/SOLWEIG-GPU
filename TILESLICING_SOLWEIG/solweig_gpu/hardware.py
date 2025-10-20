"""GPU discovery utilities."""

from __future__ import annotations

import os
from typing import List

import torch


def visible_gpu_ids() -> List[int]:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if mask:
        try:
            return [int(x) for x in mask.replace(",", " ").split() if x]
        except Exception:
            pass
    slurm = os.environ.get("SLURM_JOB_GPUS", "").strip()
    if slurm:
        try:
            return [int(x) for x in slurm.replace(",", " ").split() if x]
        except Exception:
            pass
    try:
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


__all__ = ["visible_gpu_ids"]
