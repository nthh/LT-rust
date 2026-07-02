#!/usr/bin/env python3
"""Which LandTrendr param set matches GEE better: the old Rust default (1.25/0.01)
or the README/Python default (0.75/0.05)?

Runs the kernel on the same GEE composites at both parameter sets, scores each
against GEE's disturbance year, and renders GEE | rust@1.25/0.01 | rust@0.75/0.05.
Saves to ~/Downloads/landtrendr_param_compare.png.
"""
from pathlib import Path
import numpy as np
import rasterio
import landtrendr as lt
import warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
START, END = 1984, 2016
yrs = np.arange(START, END + 1).astype(np.int32)
DELTA = 0.15

src = rasterio.open(ROOT / "data" / "gee_source.tif").read().astype(np.float32)
src[src == -32768] = np.nan
T, H, W = src.shape
dy = rasterio.open(ROOT / "data" / "gee_distyear.tif")
gee_year = dy.read(1).astype(float)
gee_mag = dy.read(2).astype(float) / 1000.0
gee_year = np.where(gee_mag >= DELTA, gee_year, np.nan)
gee_d = np.isfinite(gee_year)

PARAMS = {
    "Rust default\np=0.01, bmp=1.25": dict(p_value_threshold=0.01, best_model_proportion=1.25),
    "README / Python\np=0.05, bmp=0.75": dict(p_value_threshold=0.05, best_model_proportion=0.75),
}
BASE = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
            min_observations_needed=6, vertex_count_overshoot=3, prevent_one_year_recovery=True)


def run(p):
    yr = np.full((H, W), np.nan)
    drop = np.full((H, W), np.nan)
    for r in range(H):
        for c in range(W):
            s = (src[:, r, c] / 1000.0).astype(np.float32)
            if np.isfinite(s).sum() < 6:
                continue
            fit, _, _ = lt.pixel(np.ascontiguousarray(s), yrs, **BASE, **p)
            d = np.diff(np.asarray(fit))
            i = int(np.argmin(d))
            drop[r, c] = -d[i]
            if -d[i] >= DELTA:
                yr[r, c] = yrs[i + 1]
    return yr, drop


def score(yr, drop):
    fd = np.isfinite(yr)
    both = gee_d & fd
    iou = both.sum() / max(1, (gee_d | fd).sum())
    y1 = float((np.abs(gee_year[both] - yr[both]) <= 1).mean()) if both.any() else float("nan")
    overall = (both.sum() + (~gee_d & ~fd).sum()) / (H * W)
    depth = float(np.nanmean(drop[both])) if both.any() else float("nan")
    return iou, y1, overall, depth


cmap = LinearSegmentedColormap.from_list("dy", ["#2a9d8f", "#e9c46a", "#e76f51"])
cmap.set_bad("#0a0a0a")
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
panels = [("LT-GEE (reference)", gee_year, None)]
results = {}
for name, p in PARAMS.items():
    yr, drop = run(p)
    iou, y1, overall, depth = score(yr, drop)
    results[name] = (iou, y1, overall, depth)
    panels.append((name, yr, (iou, y1, overall, depth)))

im = None
for ax, (lab, arr, sc) in zip(axes, panels):
    ax.set_facecolor("#0a0a0a")
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=START, vmax=END, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    title = lab if sc is None else f"{lab}\nIoU {sc[0]:.3f} · yr±1 {sc[1]:.3f} · overall {sc[2]:.3f}"
    ax.set_title(title, fontsize=10)

cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, ticks=[1985, 1995, 2005, 2016])
cb.set_label("year of forest disturbance")

# verdict
(a_iou, a_y1, a_ov, a_dep) = results["Rust default\np=0.01, bmp=1.25"]
(b_iou, b_y1, b_ov, b_dep) = results["README / Python\np=0.05, bmp=0.75"]
if abs(a_ov - b_ov) < 0.005 and abs(a_iou - b_iou) < 0.005:
    verdict = "Verdict: effectively IDENTICAL on this scene — the param split is hygiene, not fidelity."
else:
    better = "0.75/0.05 (README/Python)" if (b_iou + b_ov) > (a_iou + a_ov) else "1.25/0.01 (Rust default)"
    verdict = f"Verdict: {better} matches GEE better here."
fig.suptitle("LandTrendr parameter sets vs GEE — Oregon Coast Range (~1.5 km)\n" + verdict, fontsize=11)

out = Path.home() / "Downloads" / "landtrendr_param_compare.png"
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print(f"wrote {out}")
for name, (iou, y1, ov, dep) in results.items():
    print(f"  {name.replace(chr(10),' '):34} IoU {iou:.3f}  yr±1 {y1:.3f}  overall {ov:.3f}  depth {dep:.3f}")
