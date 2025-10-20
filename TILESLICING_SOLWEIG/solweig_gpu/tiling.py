"""Utilities to split large rasters into GPU sized tiles."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from osgeo import gdal

Grid = Tuple[int, int]
Slice = Tuple[int, int]
Tile = Tuple[int, int, int, int]


def best_grid(rows: int, cols: int, ntiles: int) -> Grid:
    """Return factorisation (rows_splits, cols_splits) that yields near-square tiles."""
    best: Tuple[float, float, int, int, int] | None = None
    for r_s in range(1, ntiles + 1):
        if ntiles % r_s != 0:
            continue
        c_s = ntiles // r_s
        sub_h = rows / float(r_s)
        sub_w = cols / float(c_s)
        aspect = sub_h / max(sub_w, 1e-9)
        aspect_score = abs(math.log(max(aspect, 1e-9)))
        size_diff = abs(sub_h - sub_w) / max(sub_h, sub_w)
        prefer_ok = 0
        if cols >= rows and c_s < r_s:
            prefer_ok = 1
        elif rows > cols and r_s < c_s:
            prefer_ok = 1
        cand = (aspect_score, size_diff, prefer_ok, r_s, c_s)
        if best is None or cand < best:
            best = cand
    if best is None:
        return (1, ntiles)
    return best[3], best[4]


def grid_slices(n: int, parts: int) -> List[Slice]:
    base = n // parts
    rem = n % parts
    sizes = [base + (1 if i < rem else 0) for i in range(parts)]
    offsets = [0]
    for s in sizes[:-1]:
        offsets.append(offsets[-1] + s)
    spans = [(offsets[i], offsets[i] + sizes[i]) for i in range(parts)]
    return spans


def suggest_tiles_per_gpu(rows: int, cols: int, num_gpus: int) -> int:
    try:
        target_mpx = float(_read_env("SOLWEIG_TARGET_TILE_MPX", 1.0))
    except Exception:
        target_mpx = 1.0
    try:
        max_tiles = int(_read_env("SOLWEIG_MAX_TILES_PER_GPU", 16))
    except Exception:
        max_tiles = 16
    try:
        min_side = int(_read_env("SOLWEIG_MIN_SIDE", 256))
    except Exception:
        min_side = 256
    force_tiles = str(_read_env("SOLWEIG_FORCE_TILES_PER_GPU", "")).strip()
    if force_tiles.isdigit():
        return max(1, int(force_tiles))

    total_pixels = float(rows) * float(cols)
    tile_pixels_target = max(1.0, target_mpx * 1e6)
    ideal_tiles = total_pixels / (tile_pixels_target * max(1, num_gpus))

    # Candidate values: powers of two starting at 1 up to max_tiles
    candidates = {1}
    t = 4
    while t <= max_tiles:
        candidates.add(t)
        t *= 2
    candidates.add(max_tiles)
    candidates = sorted(c for c in candidates if c >= 1 and c <= max_tiles)

    best = candidates[0]
    best_score = float("inf")
    for cand in candidates:
        tiles_per_gpu = max(1, cand)
        total_tiles = tiles_per_gpu * max(1, num_gpus)
        r_s, c_s = best_grid(rows, cols, max(1, total_tiles))
        tile_rows = int(math.ceil(rows / max(1, r_s)))
        tile_cols = int(math.ceil(cols / max(1, c_s)))
        if tiles_per_gpu > 1 and (tile_rows < min_side or tile_cols < min_side):
            continue
        score = abs(math.log(max(tiles_per_gpu, 1e-6) / max(ideal_tiles, 1e-6)))
        if score < best_score:
            best_score = score
            best = tiles_per_gpu
    return max(1, min(max_tiles, best))


def tile_georef_rect(ds_ref: gdal.Dataset, r_start: int, c_start: int, nrows: int, ncols: int):
    gt = ds_ref.GetGeoTransform()
    x0, px_w, rot_x, y0, rot_y, px_h = gt
    x0_new = x0 + c_start * px_w + r_start * rot_x
    y0_new = y0 + c_start * rot_y + r_start * px_h
    return (x0_new, px_w, rot_x, y0_new, rot_y, px_h), ds_ref.GetProjection()


def tile_center_xy(ds_ref: gdal.Dataset, r_start: int, c_start: int, nrows: int, ncols: int) -> Tuple[float, float]:
    gt = ds_ref.GetGeoTransform()
    x0, px_w, rot_x, y0, rot_y, px_h = gt
    row_c = r_start + (nrows - 1) / 2.0
    col_c = c_start + (ncols - 1) / 2.0
    x_c = x0 + col_c * px_w + row_c * rot_x
    y_c = y0 + col_c * rot_y + row_c * px_h
    return x_c, y_c


def _read_env(name: str, fallback):
    import os

    return os.environ.get(name, fallback)


__all__ = [
    "best_grid",
    "grid_slices",
    "suggest_tiles_per_gpu",
    "tile_center_xy",
    "tile_georef_rect",
]
