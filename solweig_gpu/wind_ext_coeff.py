#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform
from scipy.ndimage import gaussian_filter, grey_erosion, label, find_objects, rotate


PathLike = Union[str, Path]


def _tlog(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _find_first_file(input_dir: Path, patterns: Sequence[str], label: str) -> Path:
    """
    Find the first matching file in the top level of input_dir.

    This intentionally does not use rglob because this full-domain calculator
    should not discover or process tile folders.
    """
    for pattern in patterns:
        matches = sorted(
            p for p in input_dir.glob(pattern)
            if p.is_file() and not p.name.startswith("WindCoeff_")
        )
        if matches:
            return matches[0]

    available = ", ".join(sorted(p.name for p in input_dir.iterdir() if p.is_file()))
    raise FileNotFoundError(
        f"Could not find {label} in {input_dir}. "
        f"Tried patterns: {list(patterns)}. "
        f"Files found: {available or 'none'}"
    )


def _find_building_raster(input_dir: Path) -> Path:
    """
    Find the building raster.

    Prefer Buildings.tif because the wind calculation requires building height
    above ground. Building_DSM.tif is accepted only as a fallback.
    """
    path = _find_first_file(
        input_dir,
        patterns=[
            "Buildings.tif",
            "buildings.tif",
            "Building_Height.tif",
            "building_height.tif",
            "BuildingHeight.tif",
            "buildingheight.tif",
            "Building_DSM.tif",
            "building_dsm.tif",
            "*Building*DSM*.tif",
            "*building*dsm*.tif",
            "*Buildings*.tif",
            "*buildings*.tif",
        ],
        label="building raster",
    )

    if path.name.lower().startswith("building_dsm"):
        _tlog(
            "[WindCoeff] WARNING: using Building_DSM.tif without DEM subtraction. "
            "This is only correct if it already represents obstacle height above ground. "
            "If it is DEM + building height, use Buildings.tif instead."
        )

    return path


def _find_tree_raster(input_dir: Path) -> Path:
    """Find the full-domain tree DSM raster."""
    return _find_first_file(
        input_dir,
        patterns=[
            "Trees.tif",
            "trees.tif",
            "Tree_DSM.tif",
            "tree_dsm.tif",
            "*Tree*DSM*.tif",
            "*tree*dsm*.tif",
            "*Trees*.tif",
            "*trees*.tif",
        ],
        label="tree DSM raster",
    )


def _find_era5_fsr_file(era5_dir: Path) -> Path:
    """Find the required ERA5 instant NetCDF file in the ERA5 directory."""
    met_path = era5_dir / "data_stream-oper_stepType-instant.nc"
    if met_path.is_file():
        return met_path

    available = ", ".join(sorted(p.name for p in era5_dir.iterdir() if p.is_file()))
    raise FileNotFoundError(
        f"Could not find data_stream-oper_stepType-instant.nc in {era5_dir}. "
        f"Files found: {available or 'none'}"
    )


def _building_raster_midpoint_lonlat(building_fp: Path) -> Tuple[float, float]:
    """
    Return the midpoint of the building raster bounds as lon/lat in EPSG:4326.

    The midpoint is computed from the raster bounds, then transformed from the
    raster CRS to geographic lon/lat. This avoids assuming the raster is already
    in latitude/longitude coordinates.
    """
    with rasterio.open(building_fp) as src:
        if src.crs is None:
            raise ValueError(
                f"Building raster has no CRS, so its midpoint cannot be converted to lon/lat: {building_fp}"
            )

        bounds = src.bounds
        x_mid = 0.5 * (bounds.left + bounds.right)
        y_mid = 0.5 * (bounds.bottom + bounds.top)

        if src.crs.to_string().upper() in {"EPSG:4326", "WGS84"} or src.crs.to_epsg() == 4326:
            lon, lat = x_mid, y_mid
        else:
            lon_vals, lat_vals = rio_transform(src.crs, "EPSG:4326", [x_mid], [y_mid])
            lon, lat = lon_vals[0], lat_vals[0]

    return float(lon), float(lat)


def _coord_name(ds, candidates: Sequence[str], label: str) -> str:
    """Return the first matching coordinate/dimension name from common candidates."""
    for name in candidates:
        if name in ds.coords or name in ds.dims or name in ds.variables:
            return name

    available = sorted(set(ds.coords) | set(ds.dims) | set(ds.variables))
    raise KeyError(
        f"Could not identify {label} coordinate. Tried {list(candidates)}. "
        f"Available names: {available}"
    )


def _era5_target_lon(lon: float, lon_values: np.ndarray) -> float:
    """Convert target longitude to match the longitude convention of the ERA5 file."""
    lon_values = np.asarray(lon_values, dtype=np.float64)
    finite = lon_values[np.isfinite(lon_values)]
    if finite.size == 0:
        return float(lon)

    lon_min = float(np.nanmin(finite))
    lon_max = float(np.nanmax(finite))

    if lon_min >= 0.0 and lon < 0.0:
        return float(lon + 360.0)
    if lon_max <= 180.0 and lon > 180.0:
        return float(lon - 360.0)
    return float(lon)


def _read_z0_from_fsr_at_raster_midpoint(
    met_path: Path,
    building_fp: Path,
    fallback: float,
) -> float:
    """
    Read fsr at the first time and nearest ERA5 grid point to the building raster midpoint.

    The returned fsr value is used as z0_ref in meters. If fsr is missing or the
    extracted value is invalid, the supplied fallback roughness length is used.
    """
    z0_ref = float(fallback)
    try:
        import xarray as xr

        lon, lat = _building_raster_midpoint_lonlat(building_fp)

        with xr.open_dataset(met_path) as ds:
            if "fsr" not in ds.variables:
                _tlog(f"[WindCoeff] fsr not found in {met_path.name}; using z0_ref={z0_ref}")
                return z0_ref

            lat_name = _coord_name(ds, ("latitude", "lat", "y"), "latitude")
            lon_name = _coord_name(ds, ("longitude", "lon", "x"), "longitude")

            da = ds["fsr"]

            # Use the first time value. ERA5 files often use valid_time, but keep
            # this generic for files using time.
            for time_name in ("valid_time", "time"):
                if time_name in da.dims:
                    da = da.isel({time_name: 0})
                    break
            else:
                # If a different time-like dimension exists, use its first value.
                for dim in list(da.dims):
                    if dim in ds.coords and np.issubdtype(ds[dim].dtype, np.datetime64):
                        da = da.isel({dim: 0})
                        break

            # If any non-spatial dimensions remain, select their first element.
            extra_dims = [dim for dim in da.dims if dim not in (lat_name, lon_name)]
            if extra_dims:
                da = da.isel({dim: 0 for dim in extra_dims})

            target_lon = _era5_target_lon(lon, ds[lon_name].values)
            selected = da.sel({lat_name: lat, lon_name: target_lon}, method="nearest")
            z0_candidate = float(selected.values)

            if np.isfinite(z0_candidate) and z0_candidate > 0.0:
                z0_ref = z0_candidate

                nearest_lat = float(selected[lat_name].values) if lat_name in selected.coords else float("nan")
                nearest_lon = float(selected[lon_name].values) if lon_name in selected.coords else float("nan")
                _tlog(
                    "[WindCoeff] using fsr as z0_ref from "
                    f"{met_path.name}: {z0_ref:.4f} m "
                    f"at raster midpoint lon={lon:.6f}, lat={lat:.6f}; "
                    f"nearest ERA5 lon={nearest_lon:.6f}, lat={nearest_lat:.6f}"
                )
            else:
                _tlog(f"[WindCoeff] invalid fsr in {met_path.name}; using z0_ref={z0_ref}")

    except Exception as exc:
        _tlog(
            f"[WindCoeff] WARNING: could not read fsr from {met_path.name}; "
            f"using z0_ref={z0_ref}. Error: {exc}"
        )

    return z0_ref

def _center_crop_or_pad(arr: np.ndarray, target_shape: Tuple[int, int], fill_value=0.0) -> np.ndarray:
    """Center-crop or pad a 2-D array to the requested shape."""
    arr = np.asarray(arr)
    th, tw = target_shape
    ah, aw = arr.shape
    if (ah, aw) == (th, tw):
        return arr

    out = np.full((th, tw), fill_value, dtype=arr.dtype)
    copy_h = min(ah, th)
    copy_w = min(aw, tw)

    src_y0 = max((ah - copy_h) // 2, 0)
    src_x0 = max((aw - copy_w) // 2, 0)
    dst_y0 = max((th - copy_h) // 2, 0)
    dst_x0 = max((tw - copy_w) // 2, 0)

    out[dst_y0:dst_y0 + copy_h, dst_x0:dst_x0 + copy_w] = arr[src_y0:src_y0 + copy_h, src_x0:src_x0 + copy_w]
    return out


def _rotate_full_extent(
    arr: np.ndarray,
    angle: float,
    *,
    order: int = 0,
    mode: str = "constant",
    cval=0.0,
    target_shape: Optional[Tuple[int, int]] = None,
    fill_value=None,
) -> np.ndarray:
    """Rotate array keeping full extent; optionally center-crop/pad to target_shape."""
    arr = np.asarray(arr)

    # Fast path for cardinal angles.
    a = int(round(angle)) % 360
    if abs(angle - round(angle)) < 1e-9 and a in (0, 90, 180, 270):
        if a == 0:
            rotated = arr
        elif a == 90:
            rotated = np.rot90(arr, 1)
        elif a == 180:
            rotated = np.rot90(arr, 2)
        else:  # 270
            rotated = np.rot90(arr, 3)

        rotated = np.ascontiguousarray(rotated)

        if target_shape is None:
            return rotated

        return _center_crop_or_pad(
            rotated,
            target_shape,
            fill_value=cval if fill_value is None else fill_value,
        )

    rotated = rotate(arr, angle, reshape=True, order=order, mode=mode, cval=cval)

    if target_shape is None:
        return rotated

    return _center_crop_or_pad(
        rotated,
        target_shape,
        fill_value=cval if fill_value is None else fill_value,)


def _coeff_at_z_trees(
    H_raw,
    H_mean,
    lp_t=None,
    *,
    z_eval=10.0,
    zref=10.0,
    z0_ref=0.7,
    LAI=4.0,
    a0=0.5,
    a1=0.2,
    alpha_min=0.6,
    alpha_max=3.5,
    hmin=1.0,
    lp_min_open=0.0,
    clamp=(0.01, 1.5),
):
    """Window-based vegetation coefficient without pixel overrides."""
    if H_mean is not None:
        Hm = np.asarray(H_mean, dtype=np.float32)
    else:
        Hm = np.asarray(H_raw, dtype=np.float32)

    if lp_t is None:
        lam_eff = np.where(np.asarray(H_raw, dtype=np.float32) > 0.0, 1.0, 0.0).astype(np.float32)
    else:
        lam_eff = np.clip(np.asarray(lp_t, dtype=np.float32), 0.0, 1.0)

    z = float(z_eval)
    clamp_min, clamp_max = clamp
    C = np.ones_like(Hm, dtype=np.float32)

    active = np.isfinite(Hm) & np.isfinite(lam_eff) & (lam_eff > lp_min_open)
    if not np.any(active):
        return C

    den_ref = math.log(max(z, 1.01 * z0_ref) / max(z0_ref, 1e-6))
    if den_ref == 0:
        den_ref = 1e-6

    lai_eff = LAI * lam_eff
    alpha = np.clip(a0 + a1 * lai_eff, alpha_min, alpha_max).astype(np.float32)

    inside = active & (Hm > z)
    outside = active & (~inside)

    if np.any(inside):
        H_i = np.maximum(Hm[inside], 1e-6)
        z0_i = np.maximum(0.1 * H_i, 0.05)
        d_i = 0.7 * H_i
        ratio_top = np.maximum((H_i - d_i) / z0_i, 1.01)
        corr = np.log(ratio_top) / den_ref
        frac = np.maximum(0.0, 1.0 - (z / H_i))
        atten = np.exp(-alpha[inside] * frac)
        C[inside] = (atten * corr).astype(np.float32)

    if np.any(outside):
        H_o = np.maximum(Hm[outside], 1e-6)
        z0_o = np.maximum(0.1 * H_o, 0.05)
        d_o = 0.7 * H_o
        ratio = (z - d_o) / z0_o
        ratio = np.where(ratio <= 1.0, 1.01, ratio)
        C[outside] = (np.log(ratio) / den_ref).astype(np.float32)

    return np.clip(C, clamp_min, clamp_max)


def _building_wake_lr_from_rot(mask_rot, height_rot, px_size, alpha, out_shape):
    """Compute building wake using pre-rotated mask and heights."""
    mask_rot = mask_rot.astype(bool)
    if not np.any(mask_rot):
        return np.ones(out_shape, dtype=np.float32)

    height_rot = np.where(mask_rot, np.maximum(height_rot, 0.0), 0.0).astype(np.float32)
    labels, num = label(mask_rot.astype(np.uint8))
    if num <= 0:
        return np.ones(out_shape, dtype=np.float32)

    wake_rot = np.ones_like(mask_rot, dtype=np.float32)
    px = float(px_size)
    slices = find_objects(labels)

    for lbl, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        rows_slice, cols_slice = slc
        building_mask = labels[slc] == lbl
        if not np.any(building_mask):
            continue

        h_vals = height_rot[slc][building_mask]
        h_valid = h_vals[np.isfinite(h_vals) & (h_vals > 0.0)]
        if h_valid.size == 0:
            continue

        H_eff = float(np.nanmean(h_valid))
        if H_eff <= 0.0:
            continue

        col_mask = np.any(building_mask, axis=0)
        row_mask = np.any(building_mask, axis=1)
        width_cells = int(np.count_nonzero(col_mask))
        depth_cells = int(np.count_nonzero(row_mask))
        if width_cells <= 0 or depth_cells <= 0:
            continue

        width_m = width_cells * px
        depth_m = depth_cells * px
        ratio_l = depth_m / H_eff
        if ratio_l <= 0.0:
            continue

        denom = (ratio_l ** 0.3) * (1.0 + 0.24 * ratio_l)
        if denom <= 0.0:
            continue

        lr = 3.0 * 1.8 * width_m / denom
        lf = 1.5 * width_m / (1.0 + 0.8 * (width_m / H_eff))
        lr = float(lr) if np.isfinite(lr) and lr > 0.0 else 0.0
        lf = float(lf) if np.isfinite(lf) and lf > 0.0 else 0.0

        ramp_back_full = None
        if lr > 0.0:
            ramp_back_full = np.clip(
                (np.arange(int(math.ceil(lr / px)) + 1, dtype=np.float32) * px) / lr,
                0.0,
                1.0,
            )

        ramp_front_full = None
        if lf > 0.0:
            ramp_front_full = np.clip(
                (np.arange(int(math.ceil(lf / px)) + 1, dtype=np.float32) * px) / lf,
                0.0,
                1.0,
            ) ** 1.5

        col_indices = np.where(col_mask)[0]
        for col_idx in col_indices:
            rows_true = np.flatnonzero(building_mask[:, col_idx])
            if rows_true.size == 0:
                continue

            splits = np.where(np.diff(rows_true) > 1)[0] + 1
            segments = np.split(rows_true, splits)

            for seg_idx, seg in enumerate(segments):
                seg_end_global = rows_slice.start + seg[-1]
                seg_start_global = rows_slice.start + seg[0]
                global_col = cols_slice.start + col_idx

                if ramp_back_full is not None:
                    start_row = seg_end_global + 1
                    if start_row < wake_rot.shape[0]:
                        end_row_limit = min(start_row + ramp_back_full.size, wake_rot.shape[0])
                        if seg_idx + 1 < len(segments):
                            next_start_global = rows_slice.start + segments[seg_idx + 1][0]
                            end_row_limit = min(end_row_limit, next_start_global)
                        row_count = end_row_limit - start_row
                        if row_count > 0:
                            current = wake_rot[start_row:end_row_limit, global_col]
                            wake_rot[start_row:end_row_limit, global_col] = np.minimum(current, ramp_back_full[:row_count])

                if seg_idx == 0 and ramp_front_full is not None:
                    start_row_front = seg_start_global - 1
                    if start_row_front >= 0:
                        min_row = max(start_row_front - (ramp_front_full.size - 1), 0)
                        rows_range = np.arange(min_row, start_row_front + 1, dtype=np.int32)
                        if rows_range.size > 0:
                            idx_front = start_row_front - rows_range
                            current_front = wake_rot[rows_range, global_col]
                            wake_rot[rows_range, global_col] = np.minimum(current_front, ramp_front_full[idx_front])

    wake_back = _rotate_full_extent(
        wake_rot,
        -alpha,
        order=1,
        mode="nearest",
        cval=1.0,
        target_shape=out_shape,
        fill_value=1.0,
    )
    wake_back = np.asarray(wake_back, dtype=np.float32)
    wake_back = np.where(np.isfinite(wake_back), wake_back, 1.0).astype(np.float32)
    return np.clip(wake_back, 0.0, 1.0)


def _trees_wake_lr_from_rot(mask_rot, height_rot, base_rot, px_size, alpha, out_shape):
    """Compute tree wake using pre-rotated mask, heights, and base coefficient."""
    mask_rot = mask_rot.astype(bool)
    if not np.any(mask_rot):
        return np.ones(out_shape, dtype=np.float32)

    height_rot = np.where(mask_rot, np.maximum(height_rot, 0.0), 0.0).astype(np.float32)
    base_rot = np.where(np.isfinite(base_rot), base_rot, 1.0).astype(np.float32)

    labels, num = label(mask_rot.astype(np.uint8))
    if num <= 0:
        wake_back = _rotate_full_extent(
            np.ones_like(mask_rot, dtype=np.float32),
            -alpha,
            order=1,
            mode="nearest",
            cval=1.0,
            target_shape=out_shape,
            fill_value=1.0,
        )
        wake_back = np.asarray(wake_back, dtype=np.float32)
        wake_back = np.where(np.isfinite(wake_back), wake_back, 1.0).astype(np.float32)
        return np.clip(wake_back, 0.0, 1.0)

    px = float(px_size)
    slices = find_objects(labels)
    wake_rot = np.ones_like(mask_rot, dtype=np.float32)

    for lbl, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        rows_slice, cols_slice = slc
        tree_mask = labels[slc] == lbl
        if not np.any(tree_mask):
            continue

        h_vals = height_rot[slc][tree_mask]
        h_valid = h_vals[np.isfinite(h_vals) & (h_vals > 0.0)]
        if h_valid.size == 0:
            continue

        H_eff = float(np.nanmean(h_valid))
        if H_eff <= 0.0:
            continue

        col_mask = np.any(tree_mask, axis=0)
        row_mask = np.any(tree_mask, axis=1)
        width_cells = int(np.count_nonzero(col_mask))
        depth_cells = int(np.count_nonzero(row_mask))
        if width_cells <= 0 or depth_cells <= 0:
            continue

        width_m = width_cells * px
        depth_m = depth_cells * px
        ratio_l = depth_m / H_eff
        if ratio_l <= 0.0:
            continue

        denom = (ratio_l ** 0.3) * (1.0 + 0.24 * ratio_l)
        if denom <= 0.0:
            continue

        lr = 3.0 * 1.8 * width_m / denom
        lf = 1.5 * width_m / (1.0 + 0.8 * (width_m / H_eff))
        lr = float(lr) if np.isfinite(lr) and lr > 0.0 else 0.0
        lf = float(lf) if np.isfinite(lf) and lf > 0.0 else 0.0

        ramp_back_full = None
        if lr > 0.0:
            ramp_back_full = np.clip(
                (np.arange(int(math.ceil(lr / px)) + 1, dtype=np.float32) * px) / lr,
                0.0,
                1.0,
            )

        ramp_front_full = None
        if lf > 0.0:
            ramp_front_full = np.clip(
                (np.arange(int(math.ceil(lf / px)) + 1, dtype=np.float32) * px) / lf,
                0.0,
                1.0,
            ) ** 1.5

        col_indices = np.where(col_mask)[0]
        for col_idx in col_indices:
            rows_true = np.flatnonzero(tree_mask[:, col_idx])
            if rows_true.size == 0:
                continue

            splits = np.where(np.diff(rows_true) > 1)[0] + 1
            segments = np.split(rows_true, splits)

            for seg_idx, seg in enumerate(segments):
                seg_end_global = rows_slice.start + seg[-1]
                seg_start_global = rows_slice.start + seg[0]
                global_col = cols_slice.start + col_idx
                base_end_val = float(np.clip(base_rot[seg_end_global, global_col], 0.0, 1.0))
                base_start_val = float(np.clip(base_rot[seg_start_global, global_col], 0.0, 1.0))

                if ramp_back_full is not None and base_end_val < 1.0:
                    start_row = seg_end_global + 1
                    if start_row < wake_rot.shape[0]:
                        end_row_limit = min(start_row + ramp_back_full.size, wake_rot.shape[0])
                        if seg_idx + 1 < len(segments):
                            next_start_global = rows_slice.start + segments[seg_idx + 1][0]
                            end_row_limit = min(end_row_limit, next_start_global)
                        row_count = end_row_limit - start_row
                        if row_count > 0:
                            coeff_profile = base_end_val + (1.0 - base_end_val) * ramp_back_full[:row_count]
                            current = wake_rot[start_row:end_row_limit, global_col]
                            wake_rot[start_row:end_row_limit, global_col] = np.minimum(current, coeff_profile.astype(np.float32))

                if seg_idx == 0 and ramp_front_full is not None and base_start_val < 1.0:
                    start_row_front = seg_start_global - 1
                    if start_row_front >= 0:
                        min_row = max(start_row_front - (ramp_front_full.size - 1), 0)
                        rows_range = np.arange(min_row, start_row_front + 1, dtype=np.int32)
                        if rows_range.size > 0:
                            idx_front = start_row_front - rows_range
                            coeff_front = base_start_val + (1.0 - base_start_val) * ramp_front_full[idx_front]
                            current_front = wake_rot[rows_range, global_col]
                            wake_rot[rows_range, global_col] = np.minimum(current_front, coeff_front.astype(np.float32))

    wake_rot[mask_rot] = np.clip(base_rot[mask_rot], 0.0, 1.0)

    wake_back = _rotate_full_extent(
        wake_rot,
        -alpha,
        order=1,
        mode="nearest",
        cval=1.0,
        target_shape=out_shape,
        fill_value=1.0,
    )
    wake_back = np.asarray(wake_back, dtype=np.float32)
    wake_back = np.where(np.isfinite(wake_back), wake_back, 1.0).astype(np.float32)
    return np.clip(wake_back, 0.0, 1.0)


def _gaussian_smooth(arr, mask_nan, px_size, window_m=4.0, preserve_mask=None):
    """Apply Gaussian smoothing while averaging values ignoring NaNs."""
    sigma = max(window_m / (2.0 * max(px_size, 1e-6)), 0.5)
    arr = arr.astype(np.float32)

    if mask_nan is not None:
        arr = arr.copy()
        arr[mask_nan] = np.nan

    valid = np.isfinite(arr)
    if not np.any(valid):
        smooth = arr
    else:
        data = np.where(valid, arr, 0.0).astype(np.float32)
        weight = gaussian_filter(valid.astype(np.float32), sigma=sigma, mode="nearest")
        smoothed = gaussian_filter(data, sigma=sigma, mode="nearest")
        with np.errstate(invalid="ignore", divide="ignore"):
            smooth = np.where(weight > 1e-6, smoothed / weight, arr)
        smooth = np.where(valid, smooth, np.nan)

    if preserve_mask is not None:
        smooth = np.where(preserve_mask, arr, smooth)
    if mask_nan is not None:
        smooth = np.where(mask_nan, np.nan, smooth)
    return smooth.astype(np.float32)


def _save_like_meta(meta_ref, out_fp: Path, arr: np.ndarray) -> None:
    """Save a single-band float32 raster using a reference metadata template."""
    arr = np.asarray(arr, dtype=np.float32)
    meta = {
        **meta_ref,
        "dtype": "float32",
        "count": 1,
        "compress": "zstd",
        "zstd_level": 3,
        "tiled": True,
        "blockxsize": min(meta_ref.get("width", 256), 256),
        "blockysize": min(meta_ref.get("height", 256), 256),
        "nodata": np.nan,
    }

    try:
        with rasterio.open(out_fp, "w", **meta) as dst:
            dst.write(arr, 1)
    except Exception:
        fallback_meta = {
            **meta_ref,
            "dtype": "float32",
            "count": 1,
            "compress": "deflate",
            "predictor": 3,
            "zlevel": 3,
            "tiled": True,
            "blockxsize": min(meta_ref.get("width", 256), 256),
            "blockysize": min(meta_ref.get("height", 256), 256),
            "nodata": np.nan,
        }
        with rasterio.open(out_fp, "w", **fallback_meta) as dst:
            dst.write(arr, 1)



def _read_building_height(
    building_fp: Path,
    *,
    hmin_b: float,
) -> Tuple[np.ndarray, dict, rasterio.Affine]:
    """
    Read building/obstacle height raster.

    No DEM is accepted in this full-domain version. Therefore the raster found
    in input_dir must already be usable as obstacle height above ground.
    """
    with rasterio.open(building_fp) as src_b:
        building = src_b.read(1, masked=True).filled(np.nan).astype(np.float32)
        transform = src_b.transform
        meta_ref = src_b.profile

    height = np.where(np.isfinite(building) & (building >= hmin_b), building, 0.0).astype(np.float32)
    return height, meta_ref, transform


def _compute_wind_full_domain(
    *,
    building_fp: Path,
    tree_fp: Path,
    output_dir: Path,
    directions: Sequence[int] = tuple(range(0, 360, 30)),
    hmin_b: float = 1.0,
    hmin_t: float = 1.0,
    z_eval: float = 10.0,
    zref: float = 10.0,
    z0_ref: float = 0.7,
    LAI_t: float = 2.0,
    a0_t: float = 0.5,
    a1_t: float = 0.4,
    alpha_min_t: float = 0.2,
    alpha_max_t: float = 2.5,
    coeff_min: float = 0.1,
    coeff_max: float = 1.00,
    lp_min_open: float = 0.02,
    max_workers: Optional[int] = None,
) -> List[Path]:
    """
    Compute full-domain directional wind coefficients.

    This function reads exactly one building raster and one tree raster and
    writes the output rasters directly into output_dir.
    """
    _ensure_dir(output_dir)

    Hb, meta_ref, transform = _read_building_height(
        building_fp,
        hmin_b=hmin_b,
    )

    with rasterio.open(tree_fp) as src_t:
        T = src_t.read(1, masked=True).filled(np.nan).astype(np.float32)
        if src_t.shape != Hb.shape:
            raise ValueError(
                f"Tree raster shape {src_t.shape} differs from building raster shape {Hb.shape}: {tree_fp}"
            )

    px = abs(transform.a)
    Ht_raw = np.where(np.isfinite(T) & (T >= hmin_t), T, 0.0).astype(np.float32)
    Ht = Ht_raw

    Mb = Hb > 0
    Mt_wake = Ht_raw > 0

    coeff_min_internal = 0.01

    Ct_local = _coeff_at_z_trees(
        H_raw=Ht,
        H_mean=Ht,
        z_eval=z_eval,
        zref=zref,
        z0_ref=z0_ref,
        LAI=LAI_t,
        a0=a0_t,
        a1=a1_t,
        alpha_min=alpha_min_t,
        alpha_max=alpha_max_t,
        hmin=hmin_t,
        lp_min_open=lp_min_open,
        clamp=(coeff_min, coeff_max),
    ).astype(np.float32)

    Ct_base = np.where(Mt_wake, Ct_local, 1.0).astype(np.float32)
    Ct_base = np.clip(Ct_base, coeff_min_internal, coeff_max)

    tree_footprint = None
    Ct_for_wake = Ct_base.copy()
    if np.any(Mt_wake):
        r_pix = max(1, int(round(10.0 / max(px, 1e-6))))
        y, x = np.ogrid[-r_pix:r_pix + 1, -r_pix:r_pix + 1]
        tree_footprint = (x * x + y * y) <= (r_pix * r_pix)
        Ct_for_wake = np.where(
            Mt_wake,
            np.minimum(Ct_for_wake, grey_erosion(Ct_base, footprint=tree_footprint, mode="nearest")),
            Ct_for_wake,
        )

    Ct_for_wake = np.clip(Ct_for_wake, coeff_min_internal, coeff_max)

    Mb_u8 = Mb.astype(np.uint8)
    Hb_f32 = Hb.astype(np.float32)
    Mt_u8 = Mt_wake.astype(np.uint8)
    Ht_f32 = Ht.astype(np.float32)
    Ct_wake_f32 = Ct_for_wake.astype(np.float32)

    valid_smooth_mask = (~Mb).astype(bool)

    sigma6 = max(6.0 / (2.0 * max(px, 1e-6)), 0.5)
    sigma40 = max(40.0 / (2.0 * max(px, 1e-6)), 0.5)

    smooth_weight6 = gaussian_filter(
        valid_smooth_mask.astype(np.float32),
        sigma=sigma6,
        mode="nearest",
    ).astype(np.float32)

    smooth_weight40 = gaussian_filter(
        valid_smooth_mask.astype(np.float32),
        sigma=sigma40,
        mode="nearest",
    ).astype(np.float32)


    def _smooth_combined(arr):
        arr = arr.astype(np.float32, copy=False)

        data6 = np.where(valid_smooth_mask & np.isfinite(arr), arr, 0.0).astype(np.float32)
        sm6_num = gaussian_filter(data6, sigma=sigma6, mode="nearest").astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            sm6 = np.where(smooth_weight6 > 1e-6, sm6_num / smooth_weight6, np.nan)

        sm6 = np.where(np.isnan(arr), np.nan, sm6)
        sm6[Mb] = np.nan

        data40 = np.where(valid_smooth_mask & np.isfinite(sm6), sm6, 0.0).astype(np.float32)
        sm40_num = gaussian_filter(data40, sigma=sigma40, mode="nearest").astype(np.float32)

        with np.errstate(invalid="ignore", divide="ignore"):
            sm40 = np.where(smooth_weight40 > 1e-6, sm40_num / smooth_weight40, np.nan)

        sm40 = np.where(np.isnan(sm6), np.nan, sm40)
        sm40[Mb] = np.nan

        return np.clip(sm40, coeff_min_internal, coeff_max).astype(np.float32)

    def _process_direction(ang: int) -> Tuple[int, np.ndarray]:
        alpha_dir = float(ang)

        b_mask_rot = _rotate_full_extent(Mb_u8, alpha_dir, order=0, mode="constant", cval=0.0).astype(bool)
        b_height_rot = _rotate_full_extent(
            Hb_f32,
            alpha_dir,
            order=1,
            mode="nearest",
            cval=0.0,
            target_shape=b_mask_rot.shape,
            fill_value=0.0,
        ).astype(np.float32)

        t_mask_rot = _rotate_full_extent(Mt_u8, alpha_dir, order=0, mode="constant", cval=0.0).astype(bool)
        t_height_rot = _rotate_full_extent(
            Ht_f32,
            alpha_dir,
            order=1,
            mode="nearest",
            cval=0.0,
            target_shape=t_mask_rot.shape,
            fill_value=0.0,
        ).astype(np.float32)
        t_base_rot = _rotate_full_extent(
            Ct_wake_f32,
            alpha_dir,
            order=1,
            mode="nearest",
            cval=1.0,
            target_shape=t_mask_rot.shape,
            fill_value=1.0,
        ).astype(np.float32)

        Cbuilding = _building_wake_lr_from_rot(b_mask_rot, b_height_rot, px, alpha_dir, Mb.shape)
        Cbuilding = np.clip(Cbuilding.astype(np.float32), coeff_min_internal, coeff_max)
        Cbuilding[Mb] = np.nan

        Ctree = _trees_wake_lr_from_rot(t_mask_rot, t_height_rot, t_base_rot, px, alpha_dir, Mt_wake.shape)
        if tree_footprint is not None:
            local_min = grey_erosion(Ctree, footprint=tree_footprint, mode="nearest")
            Ctree = np.where(Mt_wake, np.minimum(Ctree, local_min), Ctree)
        Ctree = np.clip(Ctree, coeff_min_internal, coeff_max)

        Ccombined = np.where(
            np.isnan(Cbuilding),
            np.nan,
            np.clip(Ctree * Cbuilding, coeff_min_internal, coeff_max),
        ).astype(np.float32)

        return int(ang), _smooth_combined(Ccombined)

    cpu_total = os.cpu_count() or 1
    if max_workers is None:
        max_workers = max(1, min(len(directions), cpu_total))
    else:
        max_workers = max(1, int(max_workers))

    written: List[Path] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_process_direction, int(ang)): int(ang) for ang in directions}
        for future in as_completed(future_map):
            ang = future_map[future]
            try:
                out_ang, arr = future.result()
                final_arr = np.clip(arr.astype(np.float32), coeff_min, coeff_max)
                out_fp = output_dir / f"WindCoeff_dir{out_ang:03d}.tif"
                _save_like_meta(meta_ref, out_fp, final_arr)
                written.append(out_fp)
                _tlog(f"[WindCoeff] direction {out_ang:03d} → {out_fp}")
            except Exception as err:
                raise RuntimeError(f"Wind coefficient direction {ang:03d} failed for {building_fp}") from err

    return sorted(written)


def calculate_wind_ext_coeff(
    input_dir: PathLike,
    era5_dir: PathLike,
    *,
    directions: Sequence[int] = tuple(range(0, 360, 30)),
    z0_ref: float = 0.03,
    hmin_b: float = 1.0,
    hmin_t: float = 1.0,
    z_eval: float = 10.0,
    zref: float = 10.0,
    LAI_t: float = 2.0,
    a0_t: float = 0.5,
    a1_t: float = 0.4,
    alpha_min_t: float = 0.2,
    alpha_max_t: float = 2.5,
    coeff_min: float = 0.1,
    coeff_max: float = 1.0,
    lp_min_open: float = 0.02,
    max_workers: Optional[int] = None,
) -> List[Path]:
    """
    Calculate full-domain directional wind-extension coefficients.

    Parameters
    ----------
    input_dir
        Directory containing the processed SOLWEIG raster inputs. The function
        searches only this directory, not subdirectories. It expects:

            Buildings.tif or Building_DSM.tif
            Trees.tif

        The output rasters are written into this same directory.

    era5_dir
        Directory containing the ERA5 instant NetCDF file named exactly:

            data_stream-oper_stepType-instant.nc

        The function reads variable ``fsr`` from the first time step at the
        nearest ERA5 grid cell to the midpoint of the building raster. That
        extracted ``fsr`` value is used as the reference roughness length
        ``z0_ref`` in meters.

    directions
        Wind-from directions in degrees. Default is 12 directions every 30 degrees.

    z0_ref
        Fallback reference roughness length in meters. This is used only if fsr
        cannot be read or the extracted fsr value is invalid.

    hmin_b, hmin_t
        Minimum building/tree height thresholds in meters.

    z_eval, zref
        Wind evaluation height and reference height in meters.

    LAI_t, a0_t, a1_t, alpha_min_t, alpha_max_t
        Tree attenuation parameters.

    coeff_min, coeff_max
        Output coefficient clipping bounds.

    lp_min_open
        Minimum local plan fraction threshold.

    max_workers
        Optional number of parallel workers for directional calculations.

    Returns
    -------
    list[pathlib.Path]
        Written WindCoeff_dir*.tif files.
    """
    input_dir = Path(input_dir)
    era5_dir = Path(era5_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"input_dir must be a directory: {input_dir}")
    if not era5_dir.is_dir():
        raise NotADirectoryError(f"era5_dir must be a directory: {era5_dir}")

    building_fp = _find_building_raster(input_dir)
    tree_fp = _find_tree_raster(input_dir)
    met_fp = _find_era5_fsr_file(era5_dir)

    _tlog(f"[WindCoeff] input_dir: {input_dir}")
    _tlog(f"[WindCoeff] era5_dir: {era5_dir}")
    _tlog(f"[WindCoeff] building raster: {building_fp.name}")
    _tlog(f"[WindCoeff] tree raster: {tree_fp.name}")
    _tlog(f"[WindCoeff] ERA5 fsr file: {met_fp.name}")

    z0_ref = _read_z0_from_fsr_at_raster_midpoint(
        met_path=met_fp,
        building_fp=building_fp,
        fallback=z0_ref,
    )

    return _compute_wind_full_domain(
        building_fp=building_fp,
        tree_fp=tree_fp,
        output_dir=input_dir,
        directions=directions,
        hmin_b=hmin_b,
        hmin_t=hmin_t,
        z_eval=z_eval,
        zref=zref,
        z0_ref=z0_ref,
        LAI_t=LAI_t,
        a0_t=a0_t,
        a1_t=a1_t,
        alpha_min_t=alpha_min_t,
        alpha_max_t=alpha_max_t,
        coeff_min=coeff_min,
        coeff_max=coeff_max,
        lp_min_open=lp_min_open,
        max_workers=max_workers,
    )
