#!/usr/bin/env python3
"""Root-cause one LT-rs arid false positive: dump the full tape for both
pipelines (despiked series, stage-② candidates, stage-③ vetted, final fit) on the
same pixel and find the FIRST stage where LT-rs and LT-IDL diverge.
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
from idl_compare import CANON, GDL, HARNESS, LTSRC  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TAG = os.environ.get("LT_SCENE", "arid_nv")
PIX = int(os.environ.get("LT_PIX", "1781"))

src = rasterio.open(ROOT / "data" / f"{TAG}_source.tif").read().astype(np.float32)
src[src == -32768] = np.nan
T, H, W = src.shape
years = np.arange(1984, 1984 + T)
raw = src.reshape(T, -1).T[PIX]               # x1000, NaN nodata
good = np.where(np.isfinite(raw))[0]

# ---- LT-rs tape ----
s = np.ascontiguousarray(raw / 1000.0, np.float32)
desp_r, cand_r, vet_r = landtrendr.pixel_debug(s, years.astype(np.int32), **CANON)
fit_r, vtx_r, _ = landtrendr.pixel(s, years.astype(np.int32), **CANON)
desp_r = np.asarray(desp_r) * 1000.0
fit_r = np.asarray(fit_r) * 1000.0


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


DRV = f"""!QUIET=1
!PATH = expand_path('+{HARNESS}') + path_sep(/search_path) + expand_path('+{LTSRC}') + path_sep(/search_path) + !PATH
resolve_routine, 'tbcd_v2', /is_function
years = [{",".join(str(int(y)) for y in years)}]
goods = [{",".join(str(int(g)) for g in good)}]
vvals = [{",".join(f"{float(raw[g]):.4f}" for g in good)}]
nall = n_elements(years)
miny = min(years)
allx = years - miny
x = allx[goods]
desp = desawtooth(vvals, stopat=0.9)
print, 'DESP_START' & for i=0,n_elements(desp)-1 do print, desp[i] & print, 'DESP_END'
y = desp * (-1.0)
v1 = find_vertices(x, y, 10, 2.0)
print, 'CAND_START' & for i=0,n_elements(v1)-1 do print, x[v1[i]]+miny & print, 'CAND_END'
v = vet_verts3(x, y, v1, 7, 2.0)
print, 'VET_START' & for i=0,n_elements(v)-1 do print, x[v[i]]+miny & print, 'VET_END'
res = fit_trajectory_v2(years, goods, vvals, 6, 0.0, -1.0, 42L, 0.9, 0.05, 6, 0.25, 2.0, 3, 0.75)
print, 'NSEG=', res.best_model.n_segments
print, 'YFIT_START' & for i=0,nall-1 do print, res.best_model.yfit[i] & print, 'YFIT_END'
fbt = find_best_trace(x, y, v, n_elements(v)-1)
print, 'FBT_START' & for i=0,n_elements(fbt.yfit)-1 do print, fbt.yfit[i] & print, 'FBT_END'
exit
end
"""
with tempfile.NamedTemporaryFile("w", suffix=".pro", delete=False) as f:
    f.write(DRV)
    drv = f.name
out = subprocess.run([GDL], input=f".run {drv}\nexit\n", capture_output=True, text=True, timeout=120).stdout
os.unlink(drv)

desp_i = np.array(_between(out.splitlines(), "DESP_START", "DESP_END", float))   # good-obs only
cand_i = sorted(set(_between(out.splitlines(), "CAND_START", "CAND_END", lambda v: int(round(v)))))
vet_i = sorted(set(_between(out.splitlines(), "VET_START", "VET_END", lambda v: int(round(v)))))
fit_i = np.array(_between(out.splitlines(), "YFIT_START", "YFIT_END", float))
fbt_i = -np.array(_between(out.splitlines(), "FBT_START", "FBT_END", float))  # loss-up -> loss-down


def drop(f):
    d = np.diff(f)
    return float(-np.nanmin(d)) if np.isfinite(d).any() else 0.0


# align rust despiked to good-obs positions for a like-for-like compare
desp_r_good = desp_r[good]
dmax = float(np.nanmax(np.abs(desp_r_good - desp_i))) if len(desp_i) == len(good) else float("nan")

print(f"\n===== {TAG} pixel {PIX}  ({len(good)}/{T} valid) =====")
print(f"① despike   max|rust-IDL| = {dmax:.1f} (NBRx1000)   "
      f"[rust npts {len(desp_r_good)}, IDL npts {len(desp_i)}]")
if len(desp_i) == len(good):
    print("   year     raw  rust-desp  IDL-desp   diff")
    for k in range(len(good)):
        mark = "  <<<" if abs(desp_r_good[k] - desp_i[k]) > 1 else ""
        print(f"   {years[good[k]]}  {raw[good[k]]:6.0f}  {desp_r_good[k]:8.1f}  {desp_i[k]:8.1f}  {desp_r_good[k]-desp_i[k]:6.1f}{mark}")
rc = sorted(int(years[i]) for i in cand_r)
rv = sorted(int(years[i]) for i in vet_r)
print(f"② candidates rust {rc}")
print(f"② candidates IDL  {cand_i}")
print(f"③ vetted     rust {rv}")
print(f"③ vetted     IDL  {vet_i}")
rfinal = [int(years[i]) for i, v in enumerate(np.asarray(vtx_r)) if v]
nseg_idl = next((l.split('=', 1)[1].strip() for l in out.splitlines() if l.startswith('NSEG=')), '?')
print(f"rust FINAL verts: {rfinal}  ({len(rfinal)-1} seg)")
print(f"IDL  FINAL n_segments (post-flatten): {nseg_idl}")
print(f"final DROP   rust {drop(fit_r):.0f}   IDL {drop(fit_i):.0f}")
print(f"IDL find_best_trace (raw full model, PRE-flatten) DROP = {drop(fbt_i):.0f}")
if len(fbt_i) == len(fit_r):
    print(f"   max|rust_final - IDL_fbt| = {np.nanmax(np.abs(fit_r - fbt_i)):.1f} (NBRx1000)")
if len(fbt_i) == len(fit_r):
    print("   year   rust-fit  IDL-fbt    diff   (vertices marked *)")
    vset = set(rv)
    for k in range(len(fit_r)):
        d = fit_r[k] - fbt_i[k]
        mark = "  <<<" if abs(d) > 2 else ""
        vmark = " *" if int(years[k]) in vset else "  "
        print(f"   {years[k]}{vmark} {fit_r[k]:8.1f}  {fbt_i[k]:8.1f}  {d:6.1f}{mark}")
