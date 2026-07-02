#!/usr/bin/env python3
"""Year-of-disturbance map (the LT-rs side of LT-GEE paper Figure 5).

Runs LandTrendr per pixel over the cached NBR box, takes the year of the largest
fitted NBR drop as the disturbance year (masking pixels with no significant drop),
and renders it with a teal->red year colormap on black, like the paper.

The paper's Figure 5 is TWO panels (LT-IDL vs LT-GEE) + an agreement table. The
GEE panel + table need a GEE disturbance-year raster over the whole AOI (we only
have GEE truth at 5 pixels), i.e. a billed GEE run. This script produces our panel.

Run: .venv-lazy/bin/python python/make_dist_map.py
"""
from pathlib import Path
import numpy as np
import landtrendr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
z = np.load(ROOT / "data" / "nbr_1984_2016.npz")
annual, years = z["annual"], z["years"].astype(int)
T, H, W = annual.shape
px_m = abs(float(z["transform"][0])) if "transform" in z.files else 30.0

RUN = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
           p_value_threshold=0.05, best_model_proportion=0.75,
           min_observations_needed=6, vertex_count_overshoot=3,
           prevent_one_year_recovery=True)
DELTA = 0.15  # minimum fitted NBR drop (loss-down) to call a disturbance

dist_year = np.full((H, W), np.nan, np.float32)
for r in range(H):
    for c in range(W):
        s = annual[:, r, c].astype(np.float32)
        if np.isfinite(s).sum() < RUN["min_observations_needed"]:
            continue
        fit, _, _ = landtrendr.pixel(
            np.ascontiguousarray(s), years.astype(np.int32), **RUN)
        d = np.diff(np.asarray(fit))
        i = int(np.argmin(d))
        if d[i] < -DELTA:
            dist_year[r, c] = years[i + 1]

cmap = LinearSegmentedColormap.from_list("distyr", ["#2a9d8f", "#e9c46a", "#e76f51"])
cmap.set_bad("#0a0a0a")
fig, ax = plt.subplots(figsize=(6.2, 6.4))
ax.set_facecolor("#0a0a0a")
im = ax.imshow(np.ma.masked_invalid(dist_year), cmap=cmap, vmin=1985, vmax=2016,
               interpolation="nearest")
ax.set_xticks([]); ax.set_yticks([])

# north arrow + scale bar
ax.annotate("N", xy=(0.06, 0.93), xytext=(0.06, 0.83), xycoords="axes fraction",
            color="white", ha="center", fontsize=12, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5))
bar_px = 300.0 / px_m
ax.plot([3, 3 + bar_px], [H - 4, H - 4], color="white", lw=3)
ax.text(3 + bar_px / 2, H - 5.5, "300 m", color="white", ha="center", va="bottom", fontsize=9)

ax.set_title("LT-rs LandTrendr - year of disturbance\nOregon Coast Range (validation box)",
             fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[1985, 1995, 2005, 2016])
cb.set_label("year of disturbance")
fig.tight_layout()
out = ROOT / "images" / "rust_disturbance_year.png"
fig.savefig(out, dpi=130, facecolor="white")
print(f"wrote {out.name}  |  disturbed {int(np.isfinite(dist_year).sum())}/{H*W} px  "
      f"| pixel {px_m:.0f} m | years {years[0]}-{years[-1]}")
