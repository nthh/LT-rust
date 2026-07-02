#!/usr/bin/env python3
"""Compare the standalone Rust LandTrendr against native GEE LandTrendr.

Two comparisons, both on the reference LT-GEE pixel (-123.845, 45.889):

  A. ALGORITHM (the one that matters): feed GEE's OWN annual source series into the
     Rust kernel, so the only variable is the segmentation. Overlays GEE fitted vs
     Rust fitted, scores vertex agreement + whether the 2001 disturbance trough is
     captured. This isolates the fit from any compositing difference.

  B. COMPOSITING: our MPC annual NBR (data/nbr_1984_2016.npz) vs GEE's
     source series — confirms the local read path reproduces GEE's inputs.

Needs data/gee_truth.json (run python/gee_truth.py first) and the landtrendr
module (maturin build --features python; pip install the wheel).

Run: python python/compare.py
"""
import json
from pathlib import Path
import numpy as np
import landtrendr

ROOT = Path(__file__).resolve().parent.parent
GEE = json.load(open(ROOT / "data" / "gee_truth.json"))
LON, LAT = -123.845, 45.889
RUN = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
           p_value_threshold=0.05, best_model_proportion=0.75, min_observations_needed=6)

g = GEE["pixels"][0]
gyears = np.array(g["years"], int)
gsrc = np.array([np.nan if v is None else v for v in g["source"]], np.float32)   # NBRx1000
gfit = np.array([np.nan if v is None else v for v in g["fitted"]], float)
gvtx = np.array(g["vertex"], int)
gvy = g["vertex_years"]

# --- A. algorithm: feed GEE's source into the Rust kernel ----------------------
# Rust kernel is loss-down on raw NBR; GEE source is NBRx1000 (already un-negated). /1000.
rsrc = np.ascontiguousarray(gsrc / 1000.0, np.float32)
rfit, rvtx, rmse = landtrendr.pixel(rsrc, gyears.astype(np.int32), **RUN)
rfit = np.asarray(rfit) * 1000.0
rvtx = np.asarray(rvtx).astype(int)
rvy = [int(gyears[i]) for i in range(len(gyears)) if rvtx[i]]

gset, rset = set(gvy), set(rvy)
jac = len(gset & rset) / max(1, len(gset | rset))

# disturbance-trough capture: source minimum year, fitted value there
m = np.isfinite(gsrc)
trough_year = int(gyears[m][np.argmin(gsrc[m])])
ti = list(gyears).index(trough_year)
print(f"=== A. ALGORITHM (identical GEE source -> each kernel) ===")
print(f"{'year':>5} {'source':>7} {'GEE fit':>8} {'Rust fit':>9} {'GEEv':>5} {'Rustv':>6}")
for i, y in enumerate(gyears):
    s = "" if not np.isfinite(gsrc[i]) else f"{gsrc[i]:7.0f}"
    gv = "V" if gvtx[i] else ""; rv = "V" if rvtx[i] else ""
    print(f"{y:>5} {s:>7} {gfit[i]:8.0f} {rfit[i]:9.0f} {gv:>5} {rv:>6}")
print(f"\nGEE vertices : {sorted(gvy)}")
print(f"Rust vertices: {sorted(rvy)}")
print(f"vertex Jaccard: {jac:.2f}")
print(f"trough @ {trough_year}: source {gsrc[ti]:.0f} | GEE fit {gfit[ti]:.0f} | Rust fit {rfit[ti]:.0f}")
mm = np.isfinite(gfit) & np.isfinite(rfit)
print(f"fitted MAD (GEE vs Rust): {np.mean(np.abs(gfit[mm]-rfit[mm])):.0f} NBRx1000")
print(f"Rust RMSE: {rmse*1000:.0f}")

# --- B. compositing: our MPC source vs GEE source ------------------------------
z = np.load(ROOT / "data" / "nbr_1984_2016.npz")
mpc = z["annual"][:, 26, 26] * 1000.0; myears = z["years"].astype(int)
mser = {int(y): mpc[i] for i, y in enumerate(myears)}
both = [(y, mser[y], gsrc[i]) for i, y in enumerate(gyears) if y in mser
        and np.isfinite(mser[y]) and np.isfinite(gsrc[i])]
if both:
    a = np.array([b[1] for b in both]); bb = np.array([b[2] for b in both])
    print(f"\n=== B. COMPOSITING (MPC NBR vs GEE NBR source) ===")
    print(f"overlap {len(both)} yrs  corr {np.corrcoef(a,bb)[0,1]:.3f}  MAD {np.mean(np.abs(a-bb)):.0f} NBRx1000")

# --- C. END-TO-END: Rust LandTrendr on our MPC NBR vs GEE ----------------------
# The combination the shipped demo actually computes: the Rust kernel fed OUR
# composited NBR over OUR years (not GEE's source). Scored against GEE fitted /
# vertices on the overlapping years. This is the only path A and B never compose.
mraw = np.ascontiguousarray(mpc / 1000.0, np.float32)        # raw NBR (loss-down), our years
cfit_full, cvtx_full, crmse = landtrendr.pixel(mraw, myears.astype(np.int32), **RUN)
cfit_full = np.asarray(cfit_full) * 1000.0
cvtx_full = np.asarray(cvtx_full).astype(int)
cfit_by_year = {int(y): cfit_full[i] for i, y in enumerate(myears)}
cvy = [int(myears[i]) for i in range(len(myears)) if cvtx_full[i]]
cset = set(cvy)
cjac = len(gset & cset) / max(1, len(gset | cset))
cfit_on_g = np.array([cfit_by_year.get(int(y), np.nan) for y in gyears], float)  # align to GEE axis
cmm = np.isfinite(gfit) & np.isfinite(cfit_on_g)
ctrough = cfit_by_year.get(trough_year, float("nan"))
print(f"\n=== C. END-TO-END (Rust LandTrendr on our MPC NBR vs GEE) ===")
print(f"GEE vertices      : {sorted(gvy)}")
print(f"Rust(MPC) vertices: {sorted(cvy)}")
print(f"vertex Jaccard vs GEE: {cjac:.2f}")
print(f"trough @ {trough_year}: GEE fit {gfit[ti]:.0f} | Rust(MPC) fit {ctrough:.0f}")
if cmm.any():
    print(f"fitted MAD (GEE vs Rust-on-MPC, {int(cmm.sum())} yrs): "
          f"{np.mean(np.abs(gfit[cmm]-cfit_on_g[cmm])):.0f} NBRx1000")
print(f"Rust(MPC) RMSE: {crmse*1000:.0f}")

# --- overlay plot --------------------------------------------------------------
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(gyears[m], gsrc[m], "o", color="#6b7280", ms=4, label="GEE annual NBR (source)")
    ax.plot(gyears, gfit, "-", color="#2563eb", lw=2.2, label="GEE LandTrendr fitted")
    ax.plot(gyears, rfit, "--", color="#c53030", lw=2.2, label="Rust fitted (on GEE source)")
    # C: our composited NBR + Rust fitted on it (what the demo computes).
    fm = np.isfinite(mpc)
    ax.plot(myears[fm], mpc[fm], "o", color="#9ae6b4", ms=3.5, label="our MPC annual NBR (source)")
    ax.plot(myears, cfit_full, ":", color="#2f855a", lw=2.4, label="Rust fitted (on our MPC NBR)")
    ax.plot([y for i,y in enumerate(gyears) if gvtx[i]], [gfit[i] for i in range(len(gyears)) if gvtx[i]],
            "o", color="#2563eb", ms=8, mfc="none", mew=2)
    ax.plot([y for i,y in enumerate(gyears) if rvtx[i]], [rfit[i] for i in range(len(gyears)) if rvtx[i]],
            "s", color="#c53030", ms=6)
    ax.plot([myears[i] for i in range(len(myears)) if cvtx_full[i]],
            [cfit_full[i] for i in range(len(myears)) if cvtx_full[i]],
            "^", color="#2f855a", ms=7)
    ax.set_xlabel("year"); ax.set_ylabel("NBR x1000")
    ax.set_title(f"LandTrendr @ reference pixel {LON},{LAT} — GEE vs Rust(GEE src) vs Rust(our NBR)")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.2); fig.tight_layout()
    out = ROOT / "images" / "compare_gee_vs_rust.png"; fig.savefig(out, dpi=120)
    print(f"\n[plot] {out.name}")
except Exception as e:
    print(f"[plot] skipped ({e})")
