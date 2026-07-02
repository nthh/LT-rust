#!/usr/bin/env python3
"""Throughput benchmark: original LandTrendr IDL (under GDL) vs the Rust port.

Runs the SAME scene through three paths and reports pixels/second:
  1. LT-IDL       — fit_trajectory_v2 over the whole raster in one GDL session
                    (binary I/O, startup amortized over all pixels).
  2. Rust per-pixel — landtrendr.pixel() in a Python loop (single-threaded).
  3. Rust raster    — landtrendr.raster_summary() (rayon across all cores).

Needs GDL + the LandTrendr-2012 IDL source; see idl-harness/README.md.
Run: python python/bench_idl_vs_rust.py

Caveat: on the small bundled scenes (a few thousand pixels) the multithreaded
wall-clock is a couple of milliseconds, so trust the px/s columns, not the
raw speedup ratio — the latter carries real timer noise at that scale. The
px/s figures are stable and match a large-stack run.
"""
import sys
import time

import numpy as np
import rasterio

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import landtrendr  # noqa: E402
from idl_env import ROOT  # noqa: E402
from idl_vs_gee_vs_rust_map import idl_map, rust_map, YEARS, NODATA  # noqa: E402

SCENES = [("forest_or", "forest"), ("ag_ia", "cropland"), ("arid_nv", "arid")]


def bench(tag, label):
    src = rasterio.open(ROOT / "data" / f"{tag}_source.tif").read().astype(np.float32)
    T, H, W = src.shape
    npix = H * W
    flat_raw = src.reshape(T, npix).T.copy()          # (npix, T) — idl_map/rust_map layout

    stack = src.reshape(T, npix).astype(np.float32).copy()  # (n_years, n_pixels)
    stack[stack == NODATA] = np.nan
    stack /= 1000.0
    yrs = YEARS.astype(np.int32)
    landtrendr.raster_summary(stack[:, :64], yrs)     # warm up the extension

    t = time.perf_counter(); idl_map(flat_raw, npix, T);            t_idl = time.perf_counter() - t
    t = time.perf_counter(); rust_map(flat_raw, npix);              t_loop = time.perf_counter() - t
    t = time.perf_counter(); landtrendr.raster_summary(stack, yrs); t_mt = time.perf_counter() - t

    print(f"\n=== {label} ({tag}): {npix:,} px x {T} yr ===")
    print(f"  LT-IDL (GDL, 1 session)  : {t_idl:8.3f}s   {npix / t_idl:12,.0f} px/s")
    print(f"  Rust per-pixel (Py loop) : {t_loop:8.3f}s   {npix / t_loop:12,.0f} px/s"
          f"   {t_idl / t_loop:6.0f}x")
    print(f"  Rust raster_summary (MT) : {t_mt:8.3f}s   {npix / t_mt:12,.0f} px/s"
          f"   {t_idl / t_mt:6.0f}x")


if __name__ == "__main__":
    for tag, label in SCENES:
        bench(tag, label)
