#!/usr/bin/env python3
"""Localize the cropland over-detection: full IDL-vs-LT-rs tape on one cropland
pixel where LT-rs calls a disturbance but GEE/IDL do not. Dumps stage-②
candidates, stage-③ vetted vertices, selected n_segments, and the final fit drop
for BOTH pipelines, so we see at which stage they diverge before porting anything.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import landtrendr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from idl_compare import CANON  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GDL = os.path.expanduser("~/Applications/GNU Data Language.app/Contents/Resources/bin/gdl")
HARNESS = ROOT / "idl-harness"
LTSRC = Path.home() / "projects" / "LandTrendr-2012"
DELTA1000 = 150.0

DRIVER = """!QUIET=1
!PATH = expand_path('+{harness}') + path_sep(/search_path) + expand_path('+{ltsrc}') + path_sep(/search_path) + !PATH
catch, err
if err ne 0 then begin
  print, 'CAUGHT_ERROR: ', !error_state.msg
  catch, /cancel
  exit
endif
resolve_routine, 'tbcd_v2', /is_function
years = [{yrs}]
goods = [{goods}]
vvals = [{sv}]
n_all = n_elements(years)
miny = min(years)
all_x = years - miny
x = all_x[goods]
y = desawtooth(vvals, stopat=0.9) * (-1.0)
v1 = find_vertices(x, y, 10, 2.0)
print, 'CAND_START'
for i=0,n_elements(v1)-1 do print, x[v1[i]]+miny
print, 'CAND_END'
v = vet_verts3(x, y, v1, 7, 2.0)
print, 'VET_START'
for i=0,n_elements(v)-1 do print, x[v[i]]+miny
print, 'VET_END'
res = fit_trajectory_v2(years, goods, vvals, 6, 0.0, -1.0, 42L, 0.9, 0.05, 6, 0.25, 2.0, 3, 0.75)
bm = res.best_model
print, 'NSEG=', bm.n_segments
print, 'YFIT_START'
for i=0,n_all-1 do print, bm.yfit[i]
print, 'YFIT_END'
exit
end
"""


def _between(lines, a, b, cast):
    out, on = [], False
    for ln in lines:
        if a in ln:
            on = True
        elif b in ln:
            on = False
        elif on:
            try:
                out.append(cast(float(ln.strip())))
            except ValueError:
                pass
    return out


def idl_debug(years, source):
    source = np.asarray(source, float)
    good = np.where(np.isfinite(source))[0]
    drv = DRIVER.format(
        harness=HARNESS, ltsrc=LTSRC,
        yrs=",".join(str(int(y)) for y in years),
        goods=",".join(str(int(g)) for g in good),
        sv=",".join(f"{float(source[g]):.4f}" for g in good),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pro", delete=False) as f:
        f.write(drv)
        path = f.name
    try:
        out = subprocess.run([GDL], input=f".run {path}\nexit\n",
                             capture_output=True, text=True, timeout=120).stdout
    finally:
        os.unlink(path)
    lines = out.splitlines()
    cand = sorted(set(_between(lines, "CAND_START", "CAND_END", lambda v: int(round(v)))))
    vet = sorted(set(_between(lines, "VET_START", "VET_END", lambda v: int(round(v)))))
    yfit = np.array(_between(lines, "YFIT_START", "YFIT_END", float))
    return cand, vet, yfit


src = rasterio.open(ROOT / "data" / "ag_ia_source.tif").read().astype(np.float32)
src[src == -32768] = np.nan
gee_mag = rasterio.open(ROOT / "data" / "ag_ia_distyear.tif").read(2).astype(float)
T, _, _ = src.shape
years = np.arange(1984, 1984 + T).astype(np.int32)
flat = src.reshape(T, -1).T
gm = gee_mag.reshape(-1)


def drop1000(fit1000):
    d = np.diff(fit1000)
    return float(-np.nanmin(d)) if np.isfinite(d).any() else 0.0


# first cropland pixel where rust detects, GEE doesn't
target = None
for idx in range(min(flat.shape[0], 40000)):
    s = flat[idx]
    if np.isfinite(s).sum() < 12:
        continue
    if np.isfinite(gm[idx]) and gm[idx] >= DELTA1000:
        continue
    fit, _, _ = landtrendr.pixel(np.ascontiguousarray(s / 1000.0, np.float32), years, **CANON)
    if drop1000(np.asarray(fit) * 1000.0) >= DELTA1000:
        target = (idx, s)
        break

idx, s = target
desp, rc_idx, rv_idx = landtrendr.pixel_debug(
    np.ascontiguousarray(s / 1000.0, np.float32), years, **CANON)
rfit, rvtx, _ = landtrendr.pixel(
    np.ascontiguousarray(s / 1000.0, np.float32), years, **CANON)
rfit1000 = np.asarray(rfit) * 1000.0
rc = sorted(int(years[i]) for i in rc_idx)
rv = sorted(int(years[i]) for i in rv_idx)
rfinal = sorted(int(y) for y, v in zip(years, np.asarray(rvtx)) if v)

icand, ivet, iyfit = idl_debug(years, s)

print(f"\n===== cropland pixel {idx} (rust detects, GEE does not) =====")
print(f"valid obs: {int(np.isfinite(s).sum())}/{T}")
print(f"② candidates  rust: {rc}")
print(f"② candidates  IDL : {icand}")
print(f"③ vetted      rust: {rv}")
print(f"③ vetted      IDL : {ivet}")
print(f"   final verts rust: {rfinal}  ({len(rfinal)-1} segs)")
print(f"FINAL fit max single-yr DROP (x1000):  rust {drop1000(rfit1000):.0f}   IDL {drop1000(iyfit):.0f}")
print(f"  -> rust {'DETECTS' if drop1000(rfit1000) >= DELTA1000 else 'none'}, "
      f"IDL {'DETECTS' if drop1000(iyfit) >= DELTA1000 else 'none'}")
