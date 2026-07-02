#!/usr/bin/env python3
"""LT-GEE paper Figure 5, for our port: year-of-disturbance, GEE vs LT-rs,
on the SAME GEE composites + the Table-2-style agreement.

Reads gee_source.tif (the GEE composite NBR stack) and gee_distyear.tif (GEE's
disturbance year + loss magnitude) from gee_dist_map.py, runs the Rust kernel on
the identical source stack, extracts disturbance year identically on both sides,
and renders the two panels + agreement stats. Same input, two implementations:
the comparison is pure algorithm.

Run: .venv-lazy/bin/python python/compare_maps.py
"""
from pathlib import Path
import os
import numpy as np
import rasterio
import landtrendr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
TAG = os.environ.get("LT_TAG", "gee")            # reads data/<TAG>_source.tif, <TAG>_distyear.tif
LABEL = os.environ.get("LT_LABEL", TAG)          # human label for the figure title
START = int(os.environ.get("LT_START", 1984))
END = int(os.environ.get("LT_END", 2016))
years = np.arange(START, END + 1)
DELTA = 0.15  # NBR drop (loss-down) to call a disturbance; GEE side uses DELTA*1000
RUN = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
           p_value_threshold=0.05, best_model_proportion=0.75,
           min_observations_needed=6, vertex_count_overshoot=3,
           prevent_one_year_recovery=True)

src = rasterio.open(ROOT / "data" / f"{TAG}_source.tif").read().astype(np.float32)  # [33,H,W] NBRx1000
src[src == -32768] = np.nan
_, H, W = src.shape

dy = rasterio.open(ROOT / "data" / f"{TAG}_distyear.tif")
gee_year = dy.read(1).astype(float)
gee_mag = dy.read(2).astype(float)                       # NBRx1000 loss-up step
gee_year = np.where(gee_mag >= DELTA * 1000, gee_year, np.nan)

# Rust LandTrendr on the identical GEE composites (NBR scale, like compare.py).
rust_year = np.full((H, W), np.nan)
for r in range(H):
    for c in range(W):
        s = (src[:, r, c] / 1000.0).astype(np.float32)
        if np.isfinite(s).sum() < RUN["min_observations_needed"]:
            continue
        fit, _, _ = landtrendr.pixel(np.ascontiguousarray(s), years.astype(np.int32), **RUN)
        d = np.diff(np.asarray(fit))
        i = int(np.argmin(d))
        if d[i] < -DELTA:
            rust_year[r, c] = years[i + 1]

gd, fd = np.isfinite(gee_year), np.isfinite(rust_year)
both = gd & fd
iou = both.sum() / max(1, (gd | fd).sum())
yr_within1 = float((np.abs(gee_year[both] - rust_year[both]) <= 1).mean()) if both.any() else float("nan")
overall = (both.sum() + (~gd & ~fd).sum()) / (H * W)

cmap = LinearSegmentedColormap.from_list("dy", ["#2a9d8f", "#e9c46a", "#e76f51"])
cmap.set_bad("#0a0a0a")
fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
im = None
for ax, (lab, arr) in zip(axes, [("LT-GEE", gee_year), ("LT-rs", rust_year)]):
    ax.set_facecolor("#0a0a0a")
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=START, vmax=END, interpolation="nearest")
    ax.set_title(lab, fontsize=12); ax.set_xticks([]); ax.set_yticks([])
bar = 300 / 30
axes[0].plot([2, 2 + bar], [H - 3, H - 3], color="white", lw=3)
axes[0].text(2 + bar / 2, H - 4.5, "300 m", color="white", ha="center", va="bottom", fontsize=8)
cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02, ticks=[1985, 1995, 2005, 2016])
cb.set_label("year of forest disturbance")
fig.suptitle(
    f"Year of disturbance on the same GEE composites - {LABEL}\n"
    f"disturbed IoU {iou:.2f}  |  year-within-1yr (co-detected) {yr_within1:.2f}  |  overall pixel agreement {overall:.2f}",
    fontsize=10)
out = ROOT / "images" / (f"gee_vs_rust_distyear.png" if TAG == "gee" else f"{TAG}_compare.png")
fig.savefig(out, dpi=130, facecolor="white", bbox_inches="tight")
print(f"wrote {out.name}")
print(f"GEE disturbed {gd.mean()*100:.0f}%  rust disturbed {fd.mean()*100:.0f}%  "
      f"both {int(both.sum())}  GEE-only {int((gd&~fd).sum())}  rust-only {int((~gd&fd).sum())}")
print(f"RESULT\t{TAG}\t{LABEL}\tIoU={iou:.2f}\tyr_within1={yr_within1:.2f}\toverall={overall:.2f}"
      f"\tgee_dist%={gd.mean()*100:.0f}\trust_dist%={fd.mean()*100:.0f}")
