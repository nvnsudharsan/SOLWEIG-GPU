"""Worker orchestration for SOLWEIG GPU tasks."""

from __future__ import annotations

import math
import os
import time
from typing import List

import torch

from .utci_process import compute_utci, _wait_day_ready


def gpu_worker(task_list: List[dict], local_rank: int, phys_gpu_id: int, writer_q) -> None:
    first_task = task_list[0] if task_list else {}
    device_type = str(first_task.get("device_type", "gpu")).lower()
    use_cuda = torch.cuda.is_available() and device_type == "gpu"
    device_prefix = "GPU" if use_cuda else "CPU"
    device_tag = f"{device_prefix}{phys_gpu_id}"

    if use_cuda:
        try:
            parent_mask = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            try:
                if parent_mask:
                    # Rispetta la maschera del parent (es. "0,1,2,3")
                    torch.cuda.set_device(int(local_rank) % max(1, torch.cuda.device_count()))
                else:
                    # Non tagliare la visibilità: seleziona solo il device target
                    torch.cuda.set_device(int(phys_gpu_id))
            except Exception:
                torch.cuda.set_device(0)
        except Exception:
            torch.cuda.set_device(0)
    try:
        if use_cuda:
            cur = torch.cuda.current_device()
            dev_name = torch.cuda.get_device_name(cur)
            nvis = torch.cuda.device_count()
            print(
                f"[WORKER] rank={local_rank} phys={phys_gpu_id} visible={nvis} set_device={cur} name={dev_name}",
                flush=True,
            )
        else:
            print(f"[WORKER] rank={local_rank} CPU worker={phys_gpu_id}", flush=True)
    except Exception as err:
        print(f"[WORKER] rank={local_rank} device-info error: {err}", flush=True)
    torch.set_num_threads(1)
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    tile_infos = []
    for args in task_list:
        met_path = args.get("met_path")
        if met_path and os.path.isfile(met_path):
            try:
                with open(met_path, "r") as f:
                    n_lines = sum(1 for _ in f)
                n_hours = max(0, n_lines - 1)
            except Exception:
                n_hours = 0
        else:
            n_hours = 24
        n_days = int(math.ceil(n_hours / 24.0)) if n_hours > 0 else 0
        tile_infos.append({"args": args, "n_days": n_days})

    if len(tile_infos) <= 1:
        # Single tile per worker: run once per tile with internal per-day sync (fast path)
        for ti in tile_infos:
            args = dict(ti["args"])
            n_days = ti["n_days"]
            if n_days <= 0:
                tc = args.get("tile_coords")
                if tc is not None and len(tc) == 4:
                    r0, r1, c0, c1 = map(int, tc)
                    print(
                        f"[WORKER] {device_tag} skip sub_tile r[{r0}:{r1}] c[{c0}:{c1}] (no meteorological data)",
                        flush=True,
                    )
                else:
                    print(
                        f"[WORKER] {device_tag} skip tile {args.get('number','?')} (no meteorological data)",
                        flush=True,
                    )
                continue
            args["internal_sync"] = True
            tile_t0 = time.time()
            try:
                tc = args.get("tile_coords")
                if tc is not None and len(tc) == 4:
                    r0, r1, c0, c1 = map(int, tc)
                    print(
                        f"[WORKER] {device_tag} start sub_tile r[{r0}:{r1}] c[{c0}:{c1}] days={n_days}",
                        flush=True,
                    )
                else:
                    print(
                        f"[WORKER] {device_tag} start tile {args.get('number','?')} days={n_days}",
                        flush=True,
                    )
                compute_utci(writer_q, **args)
                dt = time.time() - tile_t0
                if tc is not None and len(tc) == 4:
                    print(
                        f"[WORKER] {device_tag} done sub_tile r[{r0}:{r1}] c[{c0}:{c1}] days={n_days} in {dt:.1f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[WORKER] {device_tag} done tile {args.get('number','?')} days={n_days} in {dt:.1f}s",
                        flush=True,
                    )
            except Exception as exc:
                import traceback
                print(
                    f"{device_prefix} {local_rank}/{phys_gpu_id} ERROR on tile {args.get('number','?')}: {exc}",
                    flush=True,
                )
                traceback.print_exc()
            finally:
                if use_cuda:
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
    else:
        # Multi-tile per worker: process day-by-day across tiles, with global per-day barrier via writer
        # Build union of variables to wait for and common out path + start time
        needed_vars = {"UTCI", "Ta", "Va10m"}
        out_path = None
        start_time_str = None
        for ti in tile_infos:
            a = ti["args"]
            out_path = out_path or a.get("output_path")
            start_time_str = start_time_str or a.get("start_time")
            if a.get("save_tmrt", False):
                needed_vars.add("TMRT")
            if a.get("save_kup", False):
                needed_vars.add("Kup")
            if a.get("save_kdown", False):
                needed_vars.add("Kdown")
            if a.get("save_lup", False):
                needed_vars.add("Lup")
            if a.get("save_ldown", False):
                needed_vars.add("Ldown")
            if a.get("save_shadow", False):
                needed_vars.add("Shadow")

        try:
            from datetime import datetime, timedelta
            start_dt = datetime.strptime(start_time_str or "1970-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
        except Exception:
            from datetime import datetime as _dt, timedelta
            start_dt = _dt.utcnow()

        max_days = max((ti["n_days"] for ti in tile_infos), default=0)

        for day_idx in range(max_days):
            for ti in tile_infos:
                if ti["n_days"] <= day_idx:
                    continue
                args = dict(ti["args"])  # shallow copy
                tc = args.get("tile_coords")
                if tc is not None and len(tc) == 4:
                    r0, r1, c0, c1 = map(int, tc)
                    print(
                        f"[WORKER] {device_tag} day {day_idx+1} start sub_tile r[{r0}:{r1}] c[{c0}:{c1}]",
                        flush=True,
                    )
                else:
                    print(
                        f"[WORKER] {device_tag} day {day_idx+1} start tile {args.get('number','?')}",
                        flush=True,
                    )
                args["day_index"] = int(day_idx)
                args["internal_sync"] = False  # barrier handled here
                # evict cached SVF after last day of this tile
                args["cache_evict"] = (day_idx == int(ti["n_days"]) - 1)
                t0 = time.time()
                try:
                    compute_utci(writer_q, **args)
                except Exception as exc:
                    import traceback
                    print(
                        f"{device_prefix} {local_rank}/{phys_gpu_id} ERROR day {day_idx+1} on tile {args.get('number','?')}: {exc}",
                        flush=True,
                    )
                    traceback.print_exc()
                finally:
                    if use_cuda:
                        try:
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        except Exception:
                            pass
                    dt = time.time() - t0
                    if tc is not None and len(tc) == 4:
                        print(
                            f"[WORKER] {device_tag} day {day_idx+1} done sub_tile r[{r0}:{r1}] c[{c0}:{c1}] in {dt:.1f}s",
                            flush=True,
                        )
                    else:
                        print(
                            f"[WORKER] {device_tag} day {day_idx+1} done tile {args.get('number','?')} in {dt:.1f}s",
                            flush=True,
                        )

            date_str = (start_dt + timedelta(days=day_idx)).strftime('%Y%m%d')
            print(f"[WORKER] {device_tag} waiting day {date_str} writer ready...", flush=True)
            _wait_day_ready(date_str, sorted(needed_vars), out_path or ".")


__all__ = ["gpu_worker"]
