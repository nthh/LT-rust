#!/usr/bin/env python3
"""Three-panel year-of-disturbance MAPS — LT-IDL (GDL) vs LT-GEE vs LT-rs — for
every validation scene (forest, cropland, arid). Each scene's whole raster is run
through the original LandTrendr IDL in a SINGLE GDL session (binary I/O, one
compile), so IDL≈GEE≈Rust agreement is shown at the map level, not just at the 5
validation pixels. One figure per scene: <scene>_idl_gee_rust_distyear.png.

Run: .venv-lazy/bin/python python/idl_vs_gee_vs_rust_map.py
"""
import os
import subprocess
import tempfile

import numpy as np
import rasterio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
import landtrendr  # noqa: E402

from idl_env import ROOT, require_gdl  # noqa: E402

GDL, LTSRC, HARNESS = require_gdl()
START, END = 1984, 2016
DELTA1000 = 150.0      # 0.15 NBR drop = disturbance call (compare_maps convention)
NODATA = -32768.0
CANON = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
             p_value_threshold=0.05, best_model_proportion=0.75,
             min_observations_needed=6, vertex_count_overshoot=3,
             prevent_one_year_recovery=True)

SCENES = [
    ("gee", "forest", "Oregon Coast Range — forest"),
    ("ag_ia", "cropland", "central Iowa — cropland"),
    ("arid_nv", "arid", "northern Nevada — arid / shrub"),
]

YEARS = np.arange(START, END + 1)


def distyear_from_fit(fit1000):
    d = np.diff(fit1000)
    if not np.isfinite(d).any():
        return np.nan
    i = int(np.nanargmin(d))
    return YEARS[i + 1] if d[i] < -DELTA1000 else np.nan


def rust_map(flat_raw, npix):
    out = np.full(npix, np.nan)
    for p in range(npix):
        s = flat_raw[p].copy()
        s[s == NODATA] = np.nan
        s = (s / 1000.0).astype(np.float32)
        if np.isfinite(s).sum() < CANON["min_observations_needed"]:
            continue
        fit, _, _ = landtrendr.pixel(np.ascontiguousarray(s), YEARS.astype(np.int32), **CANON)
        out[p] = distyear_from_fit(np.asarray(fit) * 1000.0)
    return out


def idl_map(flat_raw, npix, T):
    """Run the whole raster through LandTrendr IDL in one GDL session."""
    src_path, out_path = "/tmp/lt3_src.bin", "/tmp/lt3_out.bin"
    flat_raw.astype(np.float32).tofile(src_path)   # IDL reads as fltarr(T, npix)
    driver = f"""!QUIET=1
!PATH = expand_path('+{HARNESS}') + path_sep(/search_path) + expand_path('+{LTSRC}') + path_sep(/search_path) + !PATH
resolve_routine, 'fit_trajectory_v2', /is_function
npix = {npix}L
nyr = {T}L
years = [{",".join(str(int(y)) for y in YEARS)}]
src = fltarr(nyr, npix)
openr, lun, '{src_path}', /get_lun
readu, lun, src
free_lun, lun
distyr = fltarr(npix)
distmag = fltarr(npix)
for p = 0L, npix-1L do begin
  catch, err
  if err eq 0 then begin
    vals = reform(src[*, p])
    goods = where(vals ne {NODATA}, ng)
    if ng ge 6 then begin
      res = fit_trajectory_v2(years, goods, vals[goods], 6, 0.0, -1.0, 42L, 0.9, 0.05, 6, 0.25, 2.0, 3, 0.75)
      if res.ok eq 1 then begin
        yfit = res.best_model.yfit
        d = yfit[1:nyr-1] - yfit[0:nyr-2]
        mn = min(d, imn)
        if mn lt 0 then begin
          distmag[p] = -mn
          distyr[p] = years[imn+1]
        endif
      endif
    endif
  endif
  catch, /cancel
endfor
openw, lun, '{out_path}', /get_lun
writeu, lun, distyr, distmag
free_lun, lun
print, 'IDL_BATCH_DONE'
exit
end
"""
    with tempfile.NamedTemporaryFile("w", suffix=".pro", delete=False) as f:
        f.write(driver)
        drv = f.name
    r = subprocess.run([GDL], input=f".run {drv}\nexit\n", capture_output=True, text=True, timeout=600)
    os.unlink(drv)
    if "IDL_BATCH_DONE" not in r.stdout:
        print("  WARN: IDL batch marker missing.\n", r.stdout[-400:], "\n", r.stderr[-400:])
    out = np.fromfile(out_path, np.float32)
    return np.where(out[npix:2 * npix] >= DELTA1000, out[:npix], np.nan)


def iou(a, b):
    A, B = np.isfinite(a), np.isfinite(b)
    return (A & B).sum() / max(1, (A | B).sum())


CMAP = LinearSegmentedColormap.from_list("dy", ["#2a9d8f", "#e9c46a", "#e76f51"])
CMAP.set_bad("#0a0a0a")


def run_scene(tag, scene, label):
    src = rasterio.open(ROOT / "data" / f"{tag}_source.tif").read().astype(np.float32)
    T, H, W = src.shape
    npix = H * W
    flat_raw = np.ascontiguousarray(src.reshape(T, npix).T)

    dy = rasterio.open(ROOT / "data" / f"{tag}_distyear.tif")
    gee_mag = dy.read(2).astype(float).ravel()
    gee_year = np.where(gee_mag >= DELTA1000, dy.read(1).astype(float).ravel(), np.nan)

    print(f"[{scene}] {npix} px — rust ...", flush=True)
    rust_year = rust_map(flat_raw, npix)
    print(f"[{scene}] LT-IDL over {npix} px in one GDL session ...", flush=True)
    idl_year = idl_map(flat_raw, npix, T)

    pct = lambda a: np.isfinite(a).mean() * 100
    ig, rg, ir = iou(idl_year, gee_year), iou(rust_year, gee_year), iou(idl_year, rust_year)
    print(f"[{scene}] disturbed%%: IDL {pct(idl_year):.0f} GEE {pct(gee_year):.0f} rust {pct(rust_year):.0f}"
          f"  |  IoU IDL-GEE {ig:.2f} rust-GEE {rg:.2f} IDL-rust {ir:.2f}")

    panels = [("LT-IDL (GDL)", idl_year), ("LT-GEE", gee_year), ("LT-rs", rust_year)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))
    im = None
    for ax, (lab, arr) in zip(axes, panels):
        ax.set_facecolor("#0a0a0a")
        im = ax.imshow(np.ma.masked_invalid(arr.reshape(H, W)), cmap=CMAP,
                       vmin=START, vmax=END, interpolation="nearest")
        ax.set_title(f"{lab}   ({pct(arr):.0f}% disturbed)", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    if max(pct(idl_year), pct(gee_year), pct(rust_year)) < 2:
        sub = "essentially no disturbance in any (all three agree it is undisturbed)"
    else:
        sub = f"disturbed-pixel IoU:  IDL–GEE {ig:.2f}  ·  rust–GEE {rg:.2f}  ·  IDL–rust {ir:.2f}"
    fig.suptitle(f"Year of disturbance, {label} — original LandTrendr IDL vs GEE vs the Rust port\n{sub}",
                 fontsize=12)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[1985, 1995, 2005, 2016])
    cb.set_label("year of disturbance")
    out_png = ROOT / "images" / f"{scene}_idl_gee_rust.png"
    fig.savefig(out_png, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[{scene}] wrote {out_png.name}\n")


if __name__ == "__main__":
    for tag, scene, label in SCENES:
        run_scene(tag, scene, label)
