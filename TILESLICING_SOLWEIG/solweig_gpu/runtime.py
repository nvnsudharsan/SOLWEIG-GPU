"""High-level orchestration for running SOLWEIG with optional multi-GPU support."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from osgeo import gdal, osr

from .config import apply_runtime_env_defaults
from .hardware import visible_gpu_ids
from .io_utils import is_missing_or_empty
from .preprocessor import (
    create_hgt_dem_diff_tile,
    finalize_outfile_subset,
    mosaic_hgt_dem_diffs,
    ppr,
)
from .tiling import best_grid, grid_slices, suggest_tiles_per_gpu, tile_center_xy, tile_georef_rect
from .walls_aspect import mosaic_and_cleanup_tiles, run_parallel_processing
from .workers import gpu_worker
from .writers import TiffWriter


def _set_mp_start_method() -> None:
    try:
        mp.set_start_method("forkserver", force=True)
    except Exception:
        try:
            mp.set_start_method("spawn", force=True)
        except Exception:
            pass


def _clean_parent_memory() -> None:
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _first_positive_int(text: str | int | None) -> Optional[int]:
    if text is None:
        return None
    try:
        if isinstance(text, int):
            return int(text) if text > 0 else None
    except Exception:
        return None
    matches = re.findall(r"\d+", str(text))
    for piece in matches:
        try:
            value = int(piece)
            if value > 0:
                return value
        except Exception:
            continue
    return None


def _detect_cpu_workers(default: int = 1) -> int:
    """Determine how many CPU worker processes to launch."""
    env_val = os.environ.get("SOLWEIG_CPU_WORKERS", "").strip()
    if env_val and env_val.lower() not in ("", "auto", "max", "all"):
        try:
            parsed = int(env_val)
            if parsed > 0:
                return parsed
        except Exception:
            pass
    cpus_per_task = _first_positive_int(os.environ.get("SLURM_CPUS_PER_TASK"))
    ntasks = _first_positive_int(os.environ.get("SLURM_NTASKS"))
    job_cpus = _first_positive_int(os.environ.get("SLURM_JOB_CPUS_PER_NODE"))
    cpus_on_node = _first_positive_int(os.environ.get("SLURM_CPUS_ON_NODE"))
    omp_threads = _first_positive_int(os.environ.get("OMP_NUM_THREADS"))

    cap = cpus_on_node or job_cpus or (os.cpu_count() or default)

    def _clamp(val: Optional[int]) -> Optional[int]:
        if not val or val <= 0:
            return None
        if cap and cap > 0:
            return min(val, cap)
        return val

    combined = None
    if cpus_per_task and ntasks:
        combined = cpus_per_task * ntasks

    for candidate in (
        combined,
        ntasks,
        cpus_per_task,
        job_cpus,
        omp_threads,
        cpus_on_node,
    ):
        val = _clamp(candidate)
        if val:
            return int(val)

    cpu_count = os.cpu_count() or default
    return max(1, cpu_count if cpu_count and cpu_count > 0 else default)


def _tile_windows(rows: int, cols: int, num_workers: int, force_one_per_gpu: bool) -> List[Tuple[int, int, int, int]]:
    if force_one_per_gpu:
        tiles_per_gpu = 1
        total_tiles = max(1, num_workers)
        print(
            f"[TILING] FORCED: {tiles_per_gpu}/GPU x {total_tiles // max(1, tiles_per_gpu)} = {total_tiles} tiles"
        )
    else:
        tiles_per_gpu = suggest_tiles_per_gpu(rows, cols, max(1, num_workers))
        total_tiles = max(1, num_workers) * tiles_per_gpu
        print(f"[TILING] {tiles_per_gpu}/GPU x {max(1, num_workers)} = {total_tiles} tiles")
    r_s, c_s = best_grid(rows, cols, total_tiles)
    row_spans = grid_slices(rows, r_s)
    col_spans = grid_slices(cols, c_s)
    tiles: List[Tuple[int, int, int, int]] = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tiles.append((r0, r1, c0, c1))
    assert len(tiles) == r_s * c_s
    return tiles


def _precompute_walls_aspect(
    ds_ref: gdal.Dataset,
    input_data_path: str,
    tile_list: Sequence[Tuple[int, int, int, int]],
    walls_path: str,
    aspect_path: str,
) -> None:
    if not (is_missing_or_empty(walls_path) or is_missing_or_empty(aspect_path)):
        return
    t0 = datetime.now()
    ntiles = len(tile_list)
    print(f"[PRECOMPUTE] Walls/Aspect: generating {ntiles} sub-tiles")
    for sid, (r0, r1, c0, c1) in enumerate(tile_list, start=1):
        r0i, r1i, c0i, c1i = map(int, (r0, r1, c0, c1))
        sub_rows, sub_cols = r1i - r0i, c1i - c0i
        t_tile = time.time()
        DSM_tile = ds_ref.ReadAsArray(c0i, r0i, sub_cols, sub_rows).astype(np.float32)
        out_w = os.path.join(input_data_path, f"walls_{sid:02d}.tif")
        out_a = os.path.join(input_data_path, f"aspect_{sid:02d}.tif")
        print(
            f"[PRECOMPUTE] Walls/Aspect tile {sid}/{ntiles} r[{r0i}:{r1i}] c[{c0i}:{c1i}] size {sub_rows}x{sub_cols}"
        )
        run_parallel_processing(
            DSM_tile,
            out_w,
            out_a,
            ds_ref.GetGeoTransform(),
            ds_ref.GetProjection(),
            r_start=r0i,
            c_start=c0i,
        )
        print(
            f"[PRECOMPUTE] Walls/Aspect tile {sid}/{ntiles} done in {time.time() - t_tile:.1f}s"
            f" → {os.path.basename(out_w)}, {os.path.basename(out_a)}"
        )
    mosaic_and_cleanup_tiles(input_data_path, walls_path, aspect_path)
    dt = (datetime.now() - t0).total_seconds()
    print(
        f"[PRECOMPUTE] Walls/Aspect: mosaic done in {dt:.1f}s → {os.path.basename(walls_path)}, "
        f"{os.path.basename(aspect_path)}"
    )


def _choose_processed_nc(input_data_path: str, data_source_type: str | None) -> str | None:
    dstype = (data_source_type or "").strip().lower()
    patterns: List[str]
    if dstype in ("cosmo",):
        patterns = ["cosmo"]
    elif dstype in ("ecmwf", "era5", "era-5"):
        patterns = ["era5"]
    else:
        patterns = ["cosmo", "era5"]

    for pfx in patterns:
        candidates = sorted(
            f
            for f in os.listdir(input_data_path)
            if f.lower().startswith(pfx) and f.lower().endswith((".nc", ".nc4"))
        )
        if candidates:
            return os.path.join(input_data_path, candidates[0])
    return None


def _precompute_hgt_dem(
    ds_ref: gdal.Dataset,
    dem_ds: gdal.Dataset,
    input_data_path: str,
    tile_list: Sequence[Tuple[int, int, int, int]],
    data_source_type: str | None,
) -> str:
    hgt_mosaic_path = os.path.join(input_data_path, "HGT_minus_DEM.tif")
    if not is_missing_or_empty(hgt_mosaic_path):
        return hgt_mosaic_path

    t0 = datetime.now()
    processed_nc_file = _choose_processed_nc(input_data_path, data_source_type)
    if not processed_nc_file:
        raise RuntimeError("Unable to locate processed NetCDF (cosmo*/era5*) for HGT-DEM computation")
    print(f"[PRECOMPUTE] HGT-DEM: using {os.path.basename(processed_nc_file)}")

    sref = osr.SpatialReference()
    sref.ImportFromWkt(ds_ref.GetProjection())
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(4326)
    to_ll = osr.CoordinateTransformation(sref, tgt)

    ntiles = len(tile_list)
    print(f"[PRECOMPUTE] HGT-DEM: generating {ntiles} sub-tiles")
    for sid, (r0, r1, c0, c1) in enumerate(tile_list, start=1):
        r0i, r1i, c0i, c1i = map(int, (r0, r1, c0, c1))
        sub_rows, sub_cols = r1i - r0i, c1i - c0i
        t_tile = time.time()
        print(f"[PRECOMPUTE] HGT-DEM tile {sid}/{ntiles} r[{r0i}:{r1i}] c[{c0i}:{c1i}] size {sub_rows}x{sub_cols}")
        DEM_tile = dem_ds.ReadAsArray(c0i, r0i, sub_cols, sub_rows).astype(np.float32)
        cx, cy = tile_center_xy(ds_ref, r0, c0, r1 - r0, c1 - c0)
        try:
            lonlat = to_ll.TransformPoint(cx, cy)
            if str(gdal.__version__).startswith("3"):
                center_lon, center_lat = lonlat[1], lonlat[0]
            else:
                center_lon, center_lat = lonlat[0], lonlat[1]
        except Exception:
            center_lon, center_lat = cx, cy
        try:
            create_hgt_dem_diff_tile(
                processed_nc_file,
                DEM_tile,
                center_lon,
                center_lat,
                r0i,
                r1i,
                c0i,
                c1i,
                input_data_path,
            )
        except Exception as exc:
            print(f"[PRECOMPUTE][ERROR] HGT-DEM tile {sid}/{ntiles} failed: {exc}")
        else:
            print(f"[PRECOMPUTE] HGT-DEM tile {sid}/{ntiles} done in {time.time() - t_tile:.1f}s")
    try:
        mosaic_hgt_dem_diffs(input_data_path, cleanup=True)
    except Exception as exc:
        print(f"[PRECOMPUTE][ERROR] HGT-DEM mosaic failed: {exc}")
    dt = (datetime.now() - t0).total_seconds()
    print(f"[PRECOMPUTE] HGT-DEM: mosaic done in {dt:.1f}s → {os.path.basename(hgt_mosaic_path)}")
    return hgt_mosaic_path


def _resolve_windcoeff_path(input_data_path: str) -> Optional[str]:
    """Return a usable WindCoeff raster path if present, else None."""
    preferred_names = ("WindCoeff.tif", "Windcoeff.tif")
    for name in preferred_names:
        candidate = os.path.join(input_data_path, name)
        if not is_missing_or_empty(candidate):
            return candidate
    try:
        for fname in os.listdir(input_data_path):
            if fname.lower() == "windcoeff.tif":
                candidate = os.path.join(input_data_path, fname)
                if not is_missing_or_empty(candidate):
                    return candidate
    except FileNotFoundError:
        pass
    return None


def _assign_tiles(
    tile_list: Sequence[Tuple[int, int, int, int]],
    num_workers: int,
    cpu_mode: bool,
    phys_ids: Sequence[int],
    ds_ref: gdal.Dataset,
    rows: int,
    cols: int,
    input_data_path: str,
    date_tag: str,
    start_time: str,
    end_time: str,
    data_source_type: str | None,
    output_path: str,
    dsm_path: str,
    dem_path: str,
    trees_path: str,
    landuse_path: str,
    walls_path: str,
    aspect_path: str,
    windcoeff_path: Optional[str],
    hgt_path: str,
    save_flags: Dict[str, bool],
    log_every: int,
    use_amp: bool,
    zstd_level: int,
    use_windcoeff: bool,
    use_uhi_cycle: bool,
) -> Dict[int, List[dict]]:
    tasks: Dict[int, List[dict]] = {gid: [] for gid in range(max(1, num_workers))}
    full_gt = ds_ref.GetGeoTransform()
    full_wkt = ds_ref.GetProjection()
    met_dir = os.path.join(input_data_path, f"metfiles_{date_tag}")
    os.makedirs(met_dir, exist_ok=True)
    # Track number of days per tile (used to compute expected tiles per day)
    per_tile_days: List[int] = []

    for sid, (r0, r1, c0, c1) in enumerate(tile_list):
        number = f"{r0}_{r1}_{c0}_{c1}"
        center_x, center_y = tile_center_xy(ds_ref, r0, c0, r1 - r0, c1 - c0)
        gt, wkt = tile_georef_rect(ds_ref, r0, c0, r1 - r0, c1 - c0)
        metfile_path = os.path.join(met_dir, f"metfile_{number}.txt")
        if not os.path.isfile(metfile_path):
            ppr(
                r0,
                r1,
                c0,
                c1,
                input_data_path,
                0,
                data_source_type,
                start_time,
                end_time,
                wkt,
                center_x=center_x,
                center_y=center_y,
            )
        else:
            print(f"Metfile exists: {os.path.basename(metfile_path)}")
        met_columns: Optional[List[str]] = None
        try:
            with open(metfile_path, "r", encoding="utf-8") as f:
                header_line = f.readline()
            header_tokens = [tok for tok in header_line.strip().split() if tok]
            met_columns = header_tokens if header_tokens else None
        except Exception:
            met_columns = None
        # Determine number of days available for this tile
        n_days = 0
        try:
            with open(metfile_path, "r", encoding="utf-8") as f:
                n_lines = sum(1 for _ in f)
            n_hours = max(0, n_lines - 1)
            n_days = int(np.ceil(n_hours / 24.0)) if n_hours > 0 else 0
        except Exception:
            n_days = 0
        per_tile_days.append(n_days)
        gid = sid % max(1, num_workers)
        task = dict(
            gpu_id=gid,
            device_type=("cpu" if cpu_mode else "gpu"),
            input_data_path=input_data_path,
            DSM=dsm_path,
            DEM=dem_path,
            Trees=trees_path,
            Landuse=landuse_path,
            Walls=walls_path,
            Aspect=aspect_path,
            Windcoeff=(None if not use_windcoeff else windcoeff_path),
            HGTDEM=hgt_path,
            tile_coords=(int(r0), int(r1), int(c0), int(c1)),
            met_path=metfile_path,
            output_path=output_path,
            number=number,
            start_time=start_time,
            gt=gt,
            wkt=wkt,
            full_gt=full_gt,
            full_wkt=full_wkt,
            full_rows=rows,
            full_cols=cols,
            num_tiles=len(tile_list),
            log_every=log_every,
            use_amp=use_amp,
            zstd_level=zstd_level,
            use_windcoeff=bool(use_windcoeff),
            use_uhi_cycle=bool(use_uhi_cycle),
            save_tmrt=save_flags["tmrt"],
            save_svf=save_flags["svf"],
            save_kup=save_flags["kup"],
            save_kdown=save_flags["kdown"],
            save_lup=save_flags["lup"],
            save_ldown=save_flags["ldown"],
            save_shadow=save_flags["shadow"],
            met_columns=met_columns,
        )
        tasks[gid].append(task)

    # Compute expected tiles per day across all tiles and attach to every task
    expected_by_date: Dict[str, int] = {}
    try:
        start_dt = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    except Exception:
        start_dt = datetime.utcnow()
    max_days = int(max(per_tile_days) if per_tile_days else 0)
    for d in range(max_days):
        count_tiles = sum(1 for nd in per_tile_days if nd > d)
        if count_tiles <= 0:
            continue
        date_key = (start_dt + timedelta(days=d)).strftime('%Y%m%d')
        expected_by_date[date_key] = int(count_tiles)

    for gid in list(tasks.keys()):
        for t in tasks[gid]:
            t['expected_tiles_by_date'] = expected_by_date

    try:
        worker_ids = range(max(1, num_workers))
        if cpu_mode:
            mapping = ", ".join(
                [f"w{gid}->CPU tiles={len(tasks.get(gid, []))}" for gid in worker_ids]
            )
        else:
            mapping = ", ".join(
                [
                    f"w{gid}->GPU{phys_ids[gid]} tiles={len(tasks.get(gid, []))}"
                    for gid in worker_ids
                ]
            )
        print(f"[DISPATCH] {mapping or 'no tasks'}", flush=True)
    except Exception:
        pass
    return tasks


def _finalize_outfile(input_data_path: str, date_tag: str) -> None:
    try:
        ok = finalize_outfile_subset(
            input_data_path,
            outfile_name=f"Outfile_{date_tag}.nc",
            output_name=f"Outfile_selected_{date_tag}.nc",
        )
        if not ok:
            print("ℹ️ Skipping Outfile subsetting (no selection or missing source).")
    except Exception as exc:
        print(f"⚠️ finalize_outfile_subset failed: {exc}")


def _start_writer() -> Tuple[mp.Queue, TiffWriter]:
    ctx = mp.get_context()
    writer_q: mp.Queue = ctx.Queue()
    writer = TiffWriter(writer_q)
    writer.start()
    print("[WRITER] Queue: mp.Queue (unbounded)")
    return writer_q, writer


def thermal_comfort(
    input_data_path: str,
    start_time: str,
    end_time: str,
    output_path: str,
    data_source_type: str | None = None,
    save_tmrt: bool = True,
    save_svf: bool = False,
    save_kup: bool = False,
    save_kdown: bool = False,
    save_lup: bool = False,
    save_ldown: bool = False,
    save_shadow: bool = False,
    one_tile_per_gpu: bool | None = None,
    use_windcoeff: bool = True,
    use_uhi_cycle: bool = True,
) -> None:
    os.makedirs(output_path, exist_ok=True)
    apply_runtime_env_defaults()
    _set_mp_start_method()
    if os.environ.get("SOLWEIG_GDAL_DISABLE_READDIR", "") not in ("", "0", "False", "false"):
        try:
            gdal.SetConfigOption("GDAL_DISABLE_READDIR_ON_OPEN", "YES")
        except Exception:
            pass

    num_gpus = torch.cuda.device_count()
    cpu_mode = (num_gpus == 0) or (not torch.cuda.is_available())
    phys_ids: List[int] = [] if cpu_mode else visible_gpu_ids()
    if not cpu_mode and not phys_ids:
        phys_ids = list(range(num_gpus))
    if cpu_mode:
        num_workers = max(1, _detect_cpu_workers())
        print(f"CPU mode ({num_workers} worker(s))")
        os.environ.setdefault("SOLWEIG_DISABLE_REPACK", "1")
    else:
        num_workers = max(1, len(phys_ids))
        print(f"GPUs: {len(phys_ids)} visible")
        if os.environ.get("SOLWEIG_DISABLE_REPACK"):
            try:
                del os.environ["SOLWEIG_DISABLE_REPACK"]
            except Exception:
                os.environ["SOLWEIG_DISABLE_REPACK"] = "0"

    start_dt_global = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
    date_tag = f"{start_dt_global.year:04d}_{start_dt_global.month:02d}_{start_dt_global.day:02d}"

    log_every = max(1, int(os.environ.get("SOLWEIG_LOG_EVERY", "1")))
    use_amp = os.environ.get("SOLWEIG_USE_AMP", "1") != "0"
    zstd_level = int(os.environ.get("SOLWEIG_ZSTD_LEVEL", "12"))

    dsm_path = os.path.join(input_data_path, "Building_DSM.tif")
    dem_path = os.path.join(input_data_path, "DEM.tif")
    trees_path = os.path.join(input_data_path, "Trees.tif")
    landuse_path = os.path.join(input_data_path, "Landuse.tif")
    walls_path = os.path.join(input_data_path, "Walls.tif")
    aspect_path = os.path.join(input_data_path, "Aspect.tif")
    windcoeff_path = _resolve_windcoeff_path(input_data_path)
    if not use_windcoeff:
        windcoeff_path = None
        print("[INFO] WindCoeff disabled by switch; ignoring WindCoeff.tif", flush=True)
    ds_ref = gdal.Open(dsm_path, gdal.GA_ReadOnly)
    dem_ds = gdal.Open(dem_path, gdal.GA_ReadOnly)
    if ds_ref is None:
        raise RuntimeError("Unable to open Building_DSM.tif")
    rows, cols = ds_ref.RasterYSize, ds_ref.RasterXSize

    sidecar_to_reset = os.path.join(input_data_path, ".selected_indices.json")
    if os.path.exists(sidecar_to_reset):
        os.remove(sidecar_to_reset)
        print("Reset .selected_indices.json for fresh selection")

    tile_list = _tile_windows(rows, cols, num_workers, bool(one_tile_per_gpu))
    _precompute_walls_aspect(ds_ref, input_data_path, tile_list, walls_path, aspect_path)

    hgt_mosaic_path = _precompute_hgt_dem(ds_ref, dem_ds, input_data_path, tile_list, data_source_type)

    if not (
        os.path.exists(walls_path)
        and os.path.exists(aspect_path)
        and os.path.exists(hgt_mosaic_path)
    ):
        raise RuntimeError("Error loading Walls.tif / Aspect.tif / HGT_minus_DEM.tif")

    if windcoeff_path is None:
        print("[WARN] WindCoeff.tif not found; proceeding without wind coefficient scaling")

    _clean_parent_memory()

    save_flags = {
        "tmrt": save_tmrt,
        "svf": save_svf,
        "kup": save_kup,
        "kdown": save_kdown,
        "lup": save_lup,
        "ldown": save_ldown,
        "shadow": save_shadow,
    }

    tasks = _assign_tiles(
        tile_list,
        num_workers,
        cpu_mode,
        phys_ids,
        ds_ref,
        rows,
        cols,
        input_data_path,
        date_tag,
        start_time,
        end_time,
        data_source_type,
        output_path,
        dsm_path,
        dem_path,
        trees_path,
        landuse_path,
        walls_path,
        aspect_path,
        windcoeff_path,
        hgt_mosaic_path,
        save_flags,
        log_every,
        use_amp,
        zstd_level,
        use_windcoeff,
        use_uhi_cycle,
    )

    _finalize_outfile(input_data_path, date_tag)

    ds_ref = None
    dem_ds = None
    _clean_parent_memory()

    writer_q, writer = _start_writer()
    processes: List[mp.Process] = []
    try:
        worker_ids = range(max(1, num_workers))
        for gid in worker_ids:
            task_list = tasks.get(gid, [])
            if not task_list:
                continue
            phys_id = phys_ids[gid] if (not cpu_mode and gid < len(phys_ids)) else gid
            proc = mp.Process(target=gpu_worker, args=(task_list, gid, phys_id, writer_q))
            proc.start()
            processes.append(proc)
        if not processes and cpu_mode:
            # Fallback: run inline if no separate worker was launched (e.g., no tiles)
            gpu_worker([], 0, 0, writer_q)
        for proc in processes:
            proc.join()
    finally:
        try:
            writer_q.put(("shutdown",))
        except Exception:
            pass
        try:
            writer.join()
        except Exception:
            pass
        try:
            writer_q.close()
        except Exception:
            pass
        try:
            writer_q.join_thread()
        except Exception:
            pass
        for proc in processes:
            try:
                proc.join(timeout=1)
            except Exception:
                pass


__all__ = ["thermal_comfort"]
