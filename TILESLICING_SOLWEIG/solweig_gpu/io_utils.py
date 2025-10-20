"""Basic IO helpers shared across the SOLWEIG GPU toolchain."""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np
from osgeo import gdal


def load_tif(path: str) -> Tuple[np.ndarray, gdal.Dataset]:
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Unable to open raster: {path}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    return arr, ds


def is_missing_or_empty(path: str) -> bool:
    if not os.path.exists(path):
        return True
    if os.path.isdir(path):
        return len(os.listdir(path)) == 0
    return False


__all__ = ["load_tif", "is_missing_or_empty"]
