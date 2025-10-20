"""Deferred GeoTIFF recompression utility for SOLWEIG outputs."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import os
from pathlib import Path
from typing import Iterable

from .writers import recompress_geotiff


def _iter_files(roots: Iterable[str], pattern: str, recursive: bool) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        p = Path(root)
        if not p.exists():
            raise FileNotFoundError(root)
        if p.is_file():
            files.append(p)
        else:
            iterator = p.rglob(pattern) if recursive else p.glob(pattern)
            files.extend(sorted(f for f in iterator if f.is_file()))
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in files:
        if f not in seen:
            unique.append(f)
            seen.add(f)
    return unique


def _repack_path(
    path: Path,
    threads_opt: str,
    zstd_level: int | None,
    block_x: int | None,
    block_y: int | None,
    target_mb: float | None,
    verbose: bool,
) -> Path:
    recompress_geotiff(
        path,
        threads_opt=threads_opt,
        zstd_level=zstd_level,
        block_x=block_x,
        block_y=block_y,
        target_mb=target_mb,
        verbose=verbose,
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompress SOLWEIG GeoTIFF outputs (deferred mode).")
    parser.add_argument("roots", nargs="+", help="Directories or files to process.")
    parser.add_argument("--pattern", default="*.tif", help="Glob pattern for files inside directories (default: *.tif).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when scanning.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers for recompression (default: 1).")
    parser.add_argument("--threads", default=os.environ.get("SOLWEIG_REPACK_THREADS", "1"), help="GDAL NUM_THREADS option (default: env or 1).")
    parser.add_argument("--zstd-level", type=int, default=None, help="Override ZSTD level (default: env SOLWEIG_ZSTD_LEVEL or 12).")
    parser.add_argument("--block-x", type=int, default=None, help="Override block X size.")
    parser.add_argument("--block-y", type=int, default=None, help="Override block Y size.")
    parser.add_argument("--target-mb", type=float, default=None, help="Target in-memory block size in MB (default: env or 32).")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-block progress output.")

    args = parser.parse_args()

    files = _iter_files(args.roots, args.pattern, args.recursive)
    if not files:
        print("No files matched the selection.")
        return

    verbose = not args.quiet
    jobs = max(1, int(args.jobs))

    print(f"Repacking {len(files)} file(s) with {jobs} worker(s)…", flush=True)

    if jobs == 1:
        for path in files:
            _repack_path(path, args.threads, args.zstd_level, args.block_x, args.block_y, args.target_mb, verbose)
    else:
        with futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = [
                pool.submit(
                    _repack_path,
                    path,
                    args.threads,
                    args.zstd_level,
                    args.block_x,
                    args.block_y,
                    args.target_mb,
                    verbose,
                )
                for path in files
            ]
            for fut in futures.as_completed(futs):
                fut.result()

    print("Repacking complete.", flush=True)


if __name__ == "__main__":
    main()
