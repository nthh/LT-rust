#!/usr/bin/env python3
"""Three-way LandTrendr comparison: LT-IDL (run in GDL) vs LT-GEE vs LT-rust.

This is the open-source IDL reference harness for validating the Rust port.
It runs the *original* LandTrendr-2012 IDL segmentation (fit_trajectory_v2 ->
tbcd_v2 -> find_vertices/vet_verts3/anchored_regression/desawtooth) under GNU
Data Language (GDL) on the same NBR series GEE used, then compares vertices and
the fitted trajectory against GEE's native output and our Rust kernel.

Why it matters: GEE is a black box validated in the literature as a faithful
IDL translation. Running IDL directly lets us check that claim *and* gives a
white-box gold standard to validate a future faithful tbcd_v2 port against,
instead of guessing against GEE.

Setup (one-time):
  - GDL prebuilt headless app at ~/Applications/GNU Data Language.app
  - LandTrendr-2012 IDL source cloned at ~/projects/LandTrendr-2012
  - idl-harness/ shims (regress.pro, f_test1.pro) on the GDL path

Run: .venv-lazy/bin/python python/idl_compare.py
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import lt_rust

ROOT = Path(__file__).resolve().parent.parent
GDL = os.path.expanduser(
    "~/Applications/GNU Data Language.app/Contents/Resources/bin/gdl"
)
HARNESS = ROOT / "idl-harness"
LTSRC = Path.home() / "projects" / "LandTrendr-2012"
GEE = json.load(open(ROOT / "data" / "gee_truth.json"))

CANON = dict(
    max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
    p_value_threshold=0.05, best_model_proportion=0.75,
    min_observations_needed=6, vertex_count_overshoot=3,
    prevent_one_year_recovery=True,
)

# IDL fit_trajectory_v2 arg order:
#   all_years, goods, vvals, minneeded, background, modifier, seed,
#   desawtooth_val, pval, max_segments, recovery_threshold, distweightfactor,
#   vertexcountovershoot, bestmodelproportion
# modifier=-1 flips NBR (loss-down) to loss-up internally; yfit comes back
# loss-down (x1000), comparable to GEE 'fitted'. distweightfactor=2.0 is the
# IDL default (GEE's preventOneYearRecovery has no separate IDL arg here).
DRIVER = """!QUIET=1
!PATH = expand_path('+{harness}') + path_sep(/search_path) + expand_path('+{ltsrc}') + path_sep(/search_path) + !PATH
catch, err
if err ne 0 then begin
  print, 'CAUGHT_ERROR: ', !error_state.msg
  catch, /cancel
  exit
endif
years = [{yrs}]
goods = [{goods}]
vvals = [{sv}]
n = n_elements(years)
res = fit_trajectory_v2(years, goods, vvals, 6, 0.0, -1.0, 42L, 0.9, 0.05, 6, 0.25, 2.0, 3, 0.75)
print, 'RESOK=', res.ok
bm = res.best_model
ns = bm.n_segments
print, 'NSEG=', ns
print, 'VERTYEARS=', bm.vertices[0:ns]
print, 'YFIT_START'
for i=0,n-1 do print, bm.yfit[i]
print, 'YFIT_END'
exit
end
"""


def idl_fit(years, source):
    """Run the LT-IDL fit under GDL; return (vertex_years, yfit x1000)."""
    source = np.asarray(source, float)
    good = np.where(np.isfinite(source))[0]   # 0-based indices of valid obs into years
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
        out = subprocess.run(
            [GDL], input=f".run {path}\nexit\n",
            capture_output=True, text=True, timeout=120,
        ).stdout
    finally:
        os.unlink(path)

    if "CAUGHT_ERROR" in out:
        msg = next(l for l in out.splitlines() if "CAUGHT_ERROR" in l)
        raise RuntimeError(f"IDL error: {msg}")

    vy, yfit, in_yfit = [], [], False
    for ln in out.splitlines():
        if ln.startswith("VERTYEARS="):
            vy = [int(x) for x in ln.split("=", 1)[1].split()]
        elif "YFIT_START" in ln:
            in_yfit = True
        elif "YFIT_END" in ln:
            in_yfit = False
        elif in_yfit:
            try:
                yfit.append(float(ln.strip()))
            except ValueError:
                pass
    return vy, np.array(yfit)


def dist_year(years, fitted):
    """Year of greatest single-year fitted DROP (loss-down). None if no drop."""
    d = np.diff(fitted)
    if not np.isfinite(d).any() or np.nanmin(d) >= 0:
        return None
    return int(years[int(np.nanargmin(d)) + 1])


def main():
    rows = []
    center_panel = None
    for p in GEE["pixels"]:
        years = np.array(p["years"], int)
        gsrc = np.array([np.nan if v is None else v for v in p["source"]], float)
        gfit = np.array([np.nan if v is None else v for v in p["fitted"]], float)
        gvy = p["vertex_years"]

        ivy, ifit = idl_fit(years, gsrc)

        rfit_raw, rvtx, _ = lt_rust.landtrendr_pixel(
            np.ascontiguousarray(gsrc / 1000.0, np.float32),
            years.astype(np.int32), **CANON,
        )
        rfit = np.asarray(rfit_raw) * 1000.0
        rvy = [int(y) for y, v in zip(years, np.asarray(rvtx)) if v]

        mm = np.isfinite(gfit) & np.isfinite(ifit)
        mae_idl_gee = float(np.mean(np.abs(ifit[mm] - gfit[mm])))
        mae_rust_gee = float(np.mean(np.abs(rfit[mm] - gfit[mm])))

        gy, iy, ry = (dist_year(years, f) for f in (gfit, ifit, rfit))
        trough = lambda f: float(np.nanmin(f))
        rows.append(dict(
            name=p["name"], gvy=gvy, ivy=ivy, rvy=rvy,
            idl_eq_gee=(ivy == gvy),
            gy=gy, iy=iy, ry=ry,
            gtr=trough(gfit), itr=trough(ifit), rtr=trough(rfit),
            mae_ig=mae_idl_gee, mae_rg=mae_rust_gee,
        ))
        if p["name"] == "center":
            center_panel = (years, gsrc, gfit, ifit, rfit, gvy, ivy, rvy)

    # ---- table ----
    print("\n=== LT-IDL (GDL) vs LT-GEE vs LT-rust — 5 GEE-truth pixels ===\n")
    print(f"{'pixel':>10} {'IDLvtx=GEE?':>11} {'troughGEE':>9} {'troughIDL':>9} "
          f"{'troughRust':>10} {'MAE(IDL,GEE)':>12} {'MAE(Rust,GEE)':>13}")
    for r in rows:
        print(f"{r['name']:>10} {('YES' if r['idl_eq_gee'] else 'no'):>11} "
              f"{r['gtr']:>9.0f} {r['itr']:>9.0f} {r['rtr']:>10.0f} "
              f"{r['mae_ig']:>12.1f} {r['mae_rg']:>13.1f}")
    n_eq = sum(r["idl_eq_gee"] for r in rows)
    print(f"\nIDL vertices exactly match GEE: {n_eq}/{len(rows)} pixels")
    print(f"mean MAE(IDL, GEE)  = {np.mean([r['mae_ig'] for r in rows]):.1f}  (NBRx1000)")
    print(f"mean MAE(Rust, GEE) = {np.mean([r['mae_rg'] for r in rows]):.1f}  (NBRx1000)")
    print("\nReading: IDL≈GEE confirms GEE is a faithful IDL translation; the gap")
    print("between MAE(Rust,GEE) and MAE(IDL,GEE) is the Rust port's true error vs")
    print("the original algorithm, not vs a black box.")

    # ---- figure: center pixel, 3-way overlay ----
    if center_panel is not None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        years, gsrc, gfit, ifit, rfit, gvy, ivy, rvy = center_panel
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(years, gsrc, "o", ms=4, color="0.6", label="source NBR×1000", zorder=1)
        ax.plot(years, gfit, "-", lw=2.4, color="#1b7837", label="LT-GEE fitted", zorder=3)
        ax.plot(years, ifit, "--", lw=2.4, color="#762a83",
                label="LT-IDL (GDL) fitted", zorder=4)
        ax.plot(years, rfit, "-", lw=1.8, color="#d95f02", label="LT-rust fitted", zorder=2)
        for vy, c, m in [(gvy, "#1b7837", "v"), (ivy, "#762a83", "^")]:
            yy = np.interp(vy, years, gfit if c == "#1b7837" else ifit)
            ax.plot(vy, yy, m, ms=9, color=c, zorder=5)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_title("Center pixel: LT-IDL (GDL) vs LT-GEE vs LT-rust\n"
                     "LT-rust now tracks the IDL/GEE reference fit",
                     fontsize=12)
        ax.set_xlabel("year")
        ax.set_ylabel("NBR × 1000 (loss-down)")
        ax.legend(loc="lower left", fontsize=10)
        itr, gtr, rtr = (np.nanmin(f) for f in (ifit, gfit, rfit))
        ax.annotate(f"trough: GEE {gtr:.0f} · IDL {itr:.0f} · rust {rtr:.0f}",
                    xy=(0.98, 0.04), xycoords="axes fraction", ha="right",
                    fontsize=10, color="0.3")
        out = Path.home() / "Downloads" / "landtrendr_idl_vs_gee_vs_rust.png"
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
