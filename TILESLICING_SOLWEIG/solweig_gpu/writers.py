"""Writer processes to serialize GeoTIFF outputs without exhausting memory."""

from __future__ import annotations

import threading
import sys
import ctypes
import ctypes.util
import gc
import multiprocessing as mp
import os
import time
from queue import Queue, Empty
from typing import Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal


def _determine_default_workers() -> int:
    total_bytes, avail_bytes = _system_memory_bytes()
    avail_gb = (avail_bytes or 0) / (1024 ** 3)

    def _visible_gpu_count() -> Optional[int]:
        mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if mask:
            devices = [dev.strip() for dev in mask.split(",") if dev.strip()]
            if devices:
                if all(dev == "" for dev in devices):
                    return None
                return max(0, len(devices))
        env_override = os.environ.get("SOLWEIG_GPU_COUNT")
        if env_override:
            try:
                return max(0, int(env_override))
            except Exception:
                return None
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return int(torch.cuda.device_count())
        except Exception:
            return None
        return None

    gpu_cnt = _visible_gpu_count()
    if gpu_cnt is not None and gpu_cnt > 0:
        if avail_gb >= 40:
            return min(2, gpu_cnt)
        return 1

    try:
        cpu_cnt = mp.cpu_count()
    except NotImplementedError:
        cpu_cnt = 1
    cpu_cnt = max(1, int(cpu_cnt or 1))
    if avail_gb >= 32:
        return min(2, cpu_cnt)
    return 1


def _system_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    total = None
    available = None
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total = int(getattr(vm, "total", 0)) or None
        available = int(getattr(vm, "available", 0)) or None
        if total is not None and available is not None:
            return total, available
    except Exception:
        pass

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        phys_pages = os.sysconf("SC_PHYS_PAGES")
        total = page_size * phys_pages
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        available = page_size * avail_pages
    except Exception:
        pass
    return total, available




def _malloc_trim_os() -> None:
    func = getattr(_malloc_trim_os, "_func", None)
    if func is False:
        return
    if func is None:
        libc_path = ctypes.util.find_library("c")
        if not libc_path:
            setattr(_malloc_trim_os, "_func", False)
            return
        try:
            libc = ctypes.CDLL(libc_path)
            func = getattr(libc, "malloc_trim", None)
            if func is None:
                setattr(_malloc_trim_os, "_func", False)
                return
            func.argtypes = [ctypes.c_size_t]
            func.restype = ctypes.c_int
            setattr(_malloc_trim_os, "_func", func)
        except Exception:
            setattr(_malloc_trim_os, "_func", False)
            return
    func = getattr(_malloc_trim_os, "_func", None)
    if not func or func is False:
        return
    try:
        func(0)
    except Exception:
        setattr(_malloc_trim_os, "_func", False)

# Aggressively free Python objects and return freed pages to the OS after fast CreateCopy.
def _post_fastcopy_cleanup() -> None:
    """Aggressively free Python objects and return freed pages to the OS after fast CreateCopy."""
    try:
        gc.collect()
    except Exception:
        pass
    try:
        _malloc_trim_os()
    except Exception:
        pass
    # Best-effort GDAL-side resets (safe if not available)
    try:
        gdal.ErrorReset()
    except Exception:
        pass



def _fast_repack_via_createcopy(path: str, var: str, *, verbose: bool = True) -> bool:
    """
    Fast repack via GDAL CreateCopy using LERC+ZSTD.
    No temporal/size validation, no fallbacks.
    """
    v = str(var).upper()
    if v == "SVF":
        return True  # SVF static: nothing to recompress

    base = os.path.basename(path)
    start_ts = time.time()
    ds_in = gdal.Open(path, gdal.GA_ReadOnly)
    if ds_in is None:
        if verbose:
            print(f"[POST][fastcopy] open failed for {base}", flush=True)
        return False

    # MAX_Z_ERROR by variable
    max_err_map = {
        "UTCI": os.environ.get("SOLWEIG_POST_LERC_ERR_UTCI", "0.15"),
        "TMRT": os.environ.get("SOLWEIG_POST_LERC_ERR_TMRT", "0.1"),
    }
    if v in ("KUP", "KDOWN", "LUP", "LDOWN", "SHADOW"):
        max_err = os.environ.get("SOLWEIG_POST_LERC_ERR_RAD", "2.0")
    else:
        max_err = max_err_map.get(v, os.environ.get("SOLWEIG_POST_LERC_ERR_MISC", "0.10"))

    zstd_level = int(os.environ.get("SOLWEIG_ZSTD_LEVEL", "12"))
    threads_opt = os.environ.get("SOLWEIG_REPACK_THREADS", "1") or "1"

    tmp = path + ".fast"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
    except Exception:
        pass

    driver = gdal.GetDriverByName("GTiff")
    opts = [
        "COMPRESS=LERC_ZSTD",
        f"MAX_Z_ERROR={max_err}",
        f"ZSTD_LEVEL={zstd_level}",
        "TILED=YES",
        "SPARSE_OK=YES",
        "PREDICTOR=3",
        "BIGTIFF=YES",
        f"NUM_THREADS={threads_opt}",
    ]
    out = None
    try:
        out = driver.CreateCopy(tmp, ds_in, strict=0, options=opts)
        if out is None:
            raise RuntimeError("CreateCopy returned None")
        try:
            out.FlushCache()
        except Exception:
            pass
    except Exception as exc:
        if verbose:
            print(f"[POST][fastcopy] CreateCopy failed for {base}: {exc}", flush=True)
        try:
            if out is not None:
                out = None
            os.remove(tmp)
        except Exception:
            pass
        ds_in = None
        _post_fastcopy_cleanup()
        return False
    finally:
        try:
            out = None
        except Exception:
            pass
        try:
            ds_in = None
        except Exception:
            pass
        _post_fastcopy_cleanup()

    # Replace original file
    try:
        os.replace(tmp, path)
    except Exception as exc:
        try:
            os.remove(tmp)
        except Exception:
            pass
        if verbose:
            print(f"[POST][fastcopy] replace failed for {base}: {exc}", flush=True)
        _post_fastcopy_cleanup()
        return False

    _post_fastcopy_cleanup()
    elapsed = time.time() - start_ts
    if verbose:
        print(f"[POST][fastcopy] {base} ok in {elapsed:.1f}s", flush=True)
    return True


def _recompress_one_task(path: str, var: str) -> bool:
    """Fast-path only (CreateCopy). No block-wise fallback."""
    try:
        return _fast_repack_via_createcopy(path, var, verbose=True)
    except Exception as exc:
        print(f"[POST][fastcopy] error for {os.path.basename(path)}: {exc}", flush=True)
        return False




def _repack_subprocess_main(path: str, var: str) -> None:
    ok = _recompress_one_task(path, var)
    # Aggressive RAM cleanup before exiting the repack subprocess
    try:
        gc.collect()
    except Exception:
        pass
    try:
        _malloc_trim_os()
    except Exception:
        pass
    try:
        gdal.ErrorReset()
    except Exception:
        pass
    sys.exit(0 if ok else 1)


class TiffWriter(mp.Process):
    """Serialize GeoTIFF writes from worker tiles via a queue."""

    def __init__(self, q: mp.Queue):
        super().__init__()
        self.q = q
        self.paths: Dict[str, str] = {}
        self.done_counts: Dict[str, int] = {}
        self.expected: Dict[str, int] = {}
        self._repack_trim_interval = max(1, int(os.environ.get("SOLWEIG_REPACK_TRIM_INTERVAL", "4")))
        self._repack_blocks_since_trim = 0

        cache_mb = float(os.environ.get("SOLWEIG_WRITER_GDAL_CACHE_MB", "256"))
        cache_bytes = int(max(32.0, cache_mb) * 1024 * 1024)
        try:
            gdal.SetCacheMax(cache_bytes)
        except Exception:
            try:
                gdal.SetConfigOption("GDAL_CACHEMAX", str(cache_bytes // (1024 * 1024)))
            except Exception:
                pass
        if not gdal.GetConfigOption("GDAL_NUM_THREADS"):
            try:
                gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
            except Exception:
                pass

        self._libc_path = ctypes.util.find_library("c") if hasattr(ctypes, "util") else None
        self._malloc_trim = None
        fallback_workers = _determine_default_workers()
        env_workers = os.environ.get("SOLWEIG_REPACK_WORKERS")
        # Allow human-friendly "auto" to enable dynamic selection based on RAM/CPUs/GPUs
        if env_workers is None or env_workers.strip().lower() in ("", "auto"):
            parsed_workers = fallback_workers
        else:
            try:
                parsed_workers = int(env_workers)
            except Exception:
                parsed_workers = fallback_workers
        self._repack_workers = max(1, parsed_workers)
        try:
            print(f"[writer] repack workers = {self._repack_workers} (env={env_workers!r}, fallback={fallback_workers})", flush=True)
        except Exception:
            pass
        self._repack_lock: Optional[threading.Lock] = None
        self._repack_queue: Optional[Queue[Tuple[str, str]]] = None
        self._repack_thread: Optional[threading.Thread] = None

        # Keep datasets open during intermediate writes to avoid open/close on every tile write
        # This drastically reduces latency of the first-pass (intermediate) writer.
        self._open_ds: Dict[str, gdal.Dataset] = {}
        self._writes_since_flush: Dict[str, int] = {}
        # Flush to disk every N writes per dataset to bound cache pressure without killing throughput
        try:
            self._flush_every = max(2, int(os.environ.get("SOLWEIG_WRITER_FLUSH_EVERY", "4")))
        except Exception:
            self._flush_every = 4
        self._disable_repack = str(os.environ.get("SOLWEIG_DISABLE_REPACK", "0")).strip().lower() not in ("", "0", "false", "no")

    def _key(self, var: str, date_str: str) -> str:
        return f"{var}|{date_str}"

    def _ensure_runtime_state(self) -> None:
        if self._disable_repack:
            return
        if self._repack_lock is None:
            self._repack_lock = threading.Lock()
        if self._repack_queue is None:
            self._repack_queue = Queue()
        if self._repack_thread is None or not self._repack_thread.is_alive():
            self._repack_thread = threading.Thread(
                target=self._repack_worker_loop,
                name="solweig-repack",
                daemon=True,
            )
            self._repack_thread.start()


    def _ensure_created(
        self,
        var: str,
        date_str: str,
        bands: int,
        rows: int,
        cols: int,
        gt,
        wkt,
        out_dir: str,
        expected_tiles: int,
    ) -> None:
        k = self._key(var, date_str)
        if k in self.paths:
            self.expected[k] = max(int(self.expected.get(k, 0)), int(expected_tiles))
            return
        driver = gdal.GetDriverByName("GTiff")
        block_x = max(64, min(int(cols), 512))
        block_y = max(64, min(int(rows), 512))
        if "SVF" in var.upper():
            opts = [
                "COMPRESS=NONE",
                "TILED=YES",
                f"BLOCKXSIZE={int(block_x)}",
                f"BLOCKYSIZE={int(block_y)}",
                "INTERLEAVE=BAND",
                "BIGTIFF=YES",
                "NUM_THREADS=ALL_CPUS",
            ]
        else:
            interim_lvl = int(os.environ.get("SOLWEIG_INTERIM_ZSTD_LEVEL", "2"))
            opts = [
                "COMPRESS=ZSTD",
                f"ZSTD_LEVEL={interim_lvl}",
                "TILED=YES",
                f"BLOCKXSIZE={int(block_x)}",
                f"BLOCKYSIZE={int(block_y)}",
                "INTERLEAVE=BAND",
                "BIGTIFF=YES",
                "NUM_THREADS=ALL_CPUS",
            ]
        out_path = os.path.join(out_dir, f"{var}_{date_str}.tif")
        ds = None
        if not os.path.exists(out_path):
            # Create brand-new intermediate file (light compression to keep writes fast)
            ds = driver.Create(out_path, int(cols), int(rows), int(bands), gdal.GDT_Float32, options=opts)
            ds.SetGeoTransform(gt)
            ds.SetProjection(wkt)
            nodata = -9999.0
            for b in range(1, int(bands) + 1):
                ds.GetRasterBand(b).SetNoDataValue(nodata)
            try:
                ds.FlushCache()
            except Exception:
                pass
        else:
            # Re-open existing file for update if already created in a previous tile
            ds = gdal.Open(out_path, gdal.GA_Update)
            if ds is None:
                raise RuntimeError(f"Failed to re-open for update: {out_path}")
        # Keep handle open in cache to avoid reopening per write
        self._open_ds[self._key(var, date_str)] = ds
        self._writes_since_flush[self._key(var, date_str)] = 0

        self.paths[k] = out_path
        self.done_counts[k] = 0
        self.expected[k] = int(expected_tiles)
        print(f"[writer] open {os.path.basename(out_path)} {rows}x{cols} bands={bands}")

    # --- Memory housekeeping (during repack only) -------------------------
    def _ensure_malloc_trim(self) -> None:
        if self._malloc_trim is not None or not self._libc_path:
            return
        try:
            libc = ctypes.CDLL(self._libc_path)
            if hasattr(libc, "malloc_trim"):
                func = libc.malloc_trim
                func.argtypes = [ctypes.c_size_t]
                func.restype = ctypes.c_int
                self._malloc_trim = func
            else:
                self._libc_path = None
        except Exception:
            self._libc_path = None
            self._malloc_trim = None

    def _maybe_trim_repack(self, force: bool = False) -> bool:
        if self._repack_lock is None:
            return False
        with self._repack_lock:
            trimmed = False
            if force:
                self._repack_blocks_since_trim = 0
                trimmed = True
            else:
                self._repack_blocks_since_trim += 1
                if self._repack_blocks_since_trim < self._repack_trim_interval:
                    return False
                self._repack_blocks_since_trim = 0
                trimmed = True
        gc.collect()
        self._ensure_malloc_trim()
        if self._malloc_trim:
            try:
                self._malloc_trim(0)
            except Exception:
                self._malloc_trim = None
                self._libc_path = None
        return trimmed

    def _recompress_one(self, path: str, var: str) -> bool:
        ok = _recompress_one_task(path, var)
        self._maybe_trim_repack(force=True)
        return ok

    def _write(self, var, date_str, band_idx, x0, y0, array):
        k = self._key(var, date_str)
        if k not in self.paths:
            raise RuntimeError(f"[WRITER] Dataset path unknown for {k}")
        # Retrieve or open persistent dataset handle
        ds = self._open_ds.get(k)
        if ds is None:
            # Fallback: if not cached (e.g., after restart), open and cache it now
            ds = gdal.Open(self.paths[k], gdal.GA_Update)
            if ds is None:
                raise RuntimeError(f"[WRITER] Failed to open dataset for {k}")
            self._open_ds[k] = ds
            self._writes_since_flush[k] = 0

        arr = np.asarray(array, dtype=np.float32, order="C")
        arr = np.nan_to_num(arr, copy=False, nan=-9999.0, posinf=-9999.0, neginf=-9999.0)
        band = ds.GetRasterBand(int(band_idx))
        band.WriteArray(arr, xoff=int(x0), yoff=int(y0))
        # Periodic flush (not every write) to avoid heavy sync cost
        try:
            cnt = (self._writes_since_flush.get(k, 0) or 0) + 1
            self._writes_since_flush[k] = cnt
            if cnt >= self._flush_every:
                ds.FlushCache()
                self._writes_since_flush[k] = 0
        except Exception:
            pass
        band = None

    def _tile_done(self, var, date_str):
        k = self._key(var, date_str)
        if k not in self.done_counts:
            return
        self.done_counts[k] += 1
        expected = int(self.expected.get(k, 0))
        if expected and self.done_counts[k] >= expected:
            exp = self.expected.get(k, 0)
            out_path = self.paths.get(k)
            print(f"[writer] {var}_{date_str} tiles: {self.done_counts[k]}/{exp}")
            print(f"[writer] close {var}_{date_str}.tif")

            # Flush & close persistent dataset handle before scheduling repack
            ds_cached = self._open_ds.pop(k, None)
            if ds_cached is not None:
                try:
                    ds_cached.FlushCache()
                except Exception:
                    pass
                ds_cached = None
            self._writes_since_flush.pop(k, None)

            # NEW: write a hidden ready marker (e.g., ".UTCI_YYYYMMDD.tif.ready") so it does not appear in default ls
            try:
                if out_path:
                    dname = os.path.dirname(out_path)
                    bname = os.path.basename(out_path)
                    hidden_marker = os.path.join(dname, f".{bname}.ready")
                    with open(hidden_marker, "w") as _f:
                        _f.write("ready\n")
            except Exception:
                pass

            # Defer repack while backlog is high (GPU→CPU backpressure)
            try:
                thr = int(os.environ.get("SOLWEIG_REPACK_DELAY_WHEN_QSIZE", "0"))
            except Exception:
                thr = 0
            if thr > 0:
                try:
                    pause_ms = int(os.environ.get("SOLWEIG_REPACK_PAUSE_MS", "100"))
                except Exception:
                    pause_ms = 100
                while True:
                    try:
                        qsz = self.q.qsize()
                    except Exception:
                        qsz = 0
                    if qsz <= thr:
                        break
                    time.sleep(max(1, pause_ms) / 1000.0)

            # Enqueue repack task (fast-path only)
            if out_path and not self._disable_repack:
                self._ensure_runtime_state()
                self._repack_queue.put((out_path, var))

            # Cleanup maps for this dataset key
            self.paths.pop(k, None)
            self.done_counts.pop(k, None)
            self.expected.pop(k, None)

    def _schedule_recompress(self, path: Optional[str], var: str) -> None:
        # No-op: repack is now handled only via the dedicated _repack_queue worker.
        return

    def _repack_worker_loop(self) -> None:
        ctx = mp.get_context("spawn")
        queue = self._repack_queue
        if queue is None:
            return

        active: List[Tuple[mp.Process, str]] = []
        sentinel_seen = False

        while True:
            item_obtained = False
            item = None
            if not sentinel_seen:
                try:
                    item = queue.get(timeout=0.1)
                    item_obtained = True
                except Empty:
                    item_obtained = False
                except Exception:
                    item_obtained = False

            if item_obtained:
                if item is None:
                    sentinel_seen = True
                else:
                    path, var = item
                    while self._repack_workers > 0 and len(active) >= self._repack_workers:
                        proc, job_path = active.pop(0)
                        proc.join()
                        if proc.exitcode not in (0, None):
                            print(
                                f"[writer][repack] process failed for {os.path.basename(job_path)} (code={proc.exitcode})",
                                flush=True,
                            )
                        queue.task_done()

                    try:
                        proc = ctx.Process(target=_repack_subprocess_main, args=(path, var))
                        proc.start()
                        active.append((proc, path))
                    except Exception as exc:
                        print(
                            f"[writer][repack] failed to launch process for {os.path.basename(path)}: {exc}",
                            flush=True,
                        )
                        queue.task_done()

            survivors: List[Tuple[mp.Process, str]] = []
            for proc, job_path in active:
                if proc.is_alive():
                    survivors.append((proc, job_path))
                    continue
                proc.join()
                if proc.exitcode not in (0, None):
                    print(
                        f"[writer][repack] process failed for {os.path.basename(job_path)} (code={proc.exitcode})",
                        flush=True,
                    )
                queue.task_done()
            active = survivors

            if sentinel_seen and not active:
                break

        if sentinel_seen:
            queue.task_done()

    def run(self):
        self._ensure_runtime_state()
        while True:
            msg = self.q.get()
            if not msg:
                continue
            op = msg[0]
            if op == "shutdown":
                for k, path in list(self.paths.items()):
                    var, date_str = k.split("|", 1)
                    done = int(self.done_counts.get(k, 0))
                    exp = int(self.expected.get(k, 0))
                    print(f"[writer][shutdown] finalize {var}_{date_str} tiles_done={done} expected={exp}")
                    # Ensure persistent handle is closed prior to repack
                    ds = self._open_ds.pop(self._key(var, date_str), None)
                    if ds is not None:
                        try:
                            ds.FlushCache()
                        except Exception:
                            pass
                        ds = None
                    self._writes_since_flush.pop(self._key(var, date_str), None)
                    if not self._disable_repack:
                        try:
                            if not self._recompress_one(path, var):
                                raise RuntimeError("recompress returned False")
                        except Exception as e:
                            print(f"[writer][shutdown] recompress failed for {var}_{date_str}: {e}")
                    self._gc(aggressive=True)
                print("[writer] shutdown")
                if (not self._disable_repack) and self._repack_queue is not None:
                    try:
                        self._repack_queue.join()
                    except Exception:
                        pass
                    try:
                        self._repack_queue.put(None)
                        self._repack_queue.join()
                    except Exception:
                        pass
                if (not self._disable_repack) and self._repack_thread is not None:
                    try:
                        self._repack_thread.join(timeout=10.0)
                    except Exception:
                        pass
                self._repack_thread = None
                self._repack_queue = None
                break
            elif op == "open":
                try:
                    _, var, date_str, bands, rows, cols, gt, wkt, out_dir, expected_tiles = msg
                    self._ensure_created(var, date_str, bands, rows, cols, gt, wkt, out_dir, expected_tiles)
                except Exception as e:
                    print(f"[writer] ERROR open {msg[1:3]}: {e}")
            elif op == "write":
                try:
                    _, var, date_str, band_idx, x0, y0, array = msg
                    self._write(var, date_str, band_idx, x0, y0, array)
                except Exception as e:
                    print(f"[writer] ERROR write {msg[1:4]}: {e}")
            elif op == "tile_done":
                try:
                    _, var, date_str = msg
                    self._tile_done(var, date_str)
                except Exception as e:
                    print(f"[writer] ERROR tile_done {msg[1:3]}: {e}")
            else:
                print(f"[writer] unknown: {msg!r}")


__all__ = ["TiffWriter"]
