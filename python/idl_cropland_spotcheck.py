#!/usr/bin/env python3
"""Spot-check: on cropland pixels where LT-rust (post stage-③/④ fix) calls a
disturbance but GEE does not, does LT-IDL agree with GEE (no disturbance) or with
LT-rust? Answers whether the fix over-detects on cropland relative to the true IDL
algorithm — i.e. whether the unported stage-② recovery veto is the missing piece.
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
import lt_rust

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idl_compare import idl_fit, CANON  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DELTA1000 = 150.0  # 0.15 NBR drop (loss-down) = disturbance call, matching compare_maps

src = rasterio.open(ROOT / "data" / "ag_ia_source.tif").read().astype(np.float32)
src[src == -32768] = np.nan
_dy = rasterio.open(ROOT / "data" / "ag_ia_distyear.tif")
gee_mag = _dy.read(2).astype(float)   # band 2 = GEE loss magnitude (NBRx1000), compare_maps criterion
T, _, _ = src.shape
years = np.arange(1984, 1984 + T).astype(np.int32)
flat = src.reshape(T, -1).T
gm = gee_mag.reshape(-1)


def max_drop(fit1000):
    d = np.diff(fit1000)
    return float(-np.nanmin(d)) if np.isfinite(d).any() else 0.0


picks = []
for idx in range(min(flat.shape[0], 40000)):
    s = flat[idx]
    if np.isfinite(s).sum() < 12:               # need enough obs for a meaningful fit
        continue
    if np.isfinite(gm[idx]) and gm[idx] >= DELTA1000:   # GEE detected -> skip
        continue
    fit, _, _ = lt_rust.landtrendr_pixel(
        np.ascontiguousarray(s / 1000.0, np.float32), years, **CANON)
    rdrop = max_drop(np.asarray(fit) * 1000.0)
    if rdrop >= DELTA1000:                       # rust detects, GEE doesn't
        picks.append((idx, s, rdrop))
    if len(picks) >= 8:
        break

print(f"checking {len(picks)} cropland pixels where rust calls disturbance, GEE does not\n")
print(f"{'pixel':>8} {'rustDrop':>8} {'GEE':>6} {'IDLdrop':>8} {'IDL?':>6}")
idl_agrees_gee = 0
for idx, s, rdrop in picks:
    _, ifit = idl_fit(years, s)
    idrop = max_drop(ifit)
    idl_det = idrop >= DELTA1000
    idl_agrees_gee += int(not idl_det)
    print(f"{idx:>8} {rdrop:>8.0f} {'none':>6} {idrop:>8.0f} {'DET' if idl_det else 'none':>6}")

if picks:
    verdict = ("stage-② recovery veto is the cropland fix (IDL suppresses these)"
               if idl_agrees_gee > len(picks) // 2
               else "GEE itself diverges on cropland; LT-rust may be MORE IDL-faithful")
    print(f"\nIDL agrees with GEE (no disturbance) on {idl_agrees_gee}/{len(picks)}  ->  {verdict}")
