#!/usr/bin/env python3
"""Score our LandTrendr vs GEE in the *paper's own* metrics (Kennedy et al. 2018,
Remote Sens. 10(5):691, the LT-GEE technical note).

The paper benchmarks LT-GEE against LT-IDL with:
  - vertex COUNT difference (Fig 3; dominated by 0)
  - MAE of fitted NBR as % of NBR range -1..1 (Fig 4a; "<3% for 5/6 regions")
  - for co-detected disturbance: agreement on the disturbance YEAR (">97%")

We compute the same over our 5 GEE-truth pixels (our kernel on GEE's own source,
standard params), so the numbers are directly comparable to the paper's bar.

NOTE: 5 pixels at one site is far thinner than the paper's 6 x ~184km regions.
This is indicative, not a regional reproduction.

Run: .venv-lazy/bin/python python/paper_compare.py
"""
import json
from pathlib import Path
import numpy as np
import landtrendr

ROOT = Path(__file__).resolve().parent.parent
GEE = json.load(open(ROOT / "data" / "gee_truth.json"))
CANON = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
             p_value_threshold=0.05, best_model_proportion=0.75,
             min_observations_needed=6, vertex_count_overshoot=3,
             prevent_one_year_recovery=True)
NBR_RANGE = 2000.0  # NBRx1000 over -1..1


def arr(p, key, dt=float):
    return np.array([np.nan if v is None else v for v in p[key]], dt)


def dist_year(years, fitted):
    """Year of greatest single-year fitted DROP (loss). NaN if no drop."""
    d = np.diff(fitted)
    if not np.isfinite(d).any() or np.nanmin(d) >= 0:
        return None
    return int(years[int(np.nanargmin(d)) + 1])


print(f"{'pixel':>10} {'GEEvtx':>6} {'ourvtx':>6} {'dVtx':>5} "
      f"{'GEEyr':>6} {'ouryr':>6} {'yr=':>4} {'MAE%':>5}")
cnt_diffs, mae_pcts = [], []
yr_hits = yr_total = 0
for p in GEE["pixels"]:
    years = np.array(p["years"], int)
    gsrc = arr(p, "source", np.float32)
    gfit = arr(p, "fitted")
    gvtx = np.array(p["vertex"], int)
    fit, vtx, _ = landtrendr.pixel(
        np.ascontiguousarray(gsrc / 1000.0, np.float32), years.astype(np.int32), **CANON)
    rfit = np.asarray(fit) * 1000.0
    rvtx = np.asarray(vtx).astype(int)

    gvc, rvc = int(gvtx.sum()), int(rvtx.sum())
    cnt_diffs.append(rvc - gvc)
    mm = np.isfinite(gfit) & np.isfinite(rfit)
    mae = float(np.mean(np.abs(gfit[mm] - rfit[mm])))
    mae_pct = 100 * mae / NBR_RANGE
    mae_pcts.append(mae_pct)

    gy, ry = dist_year(years, gfit), dist_year(years, rfit)
    match = (gy is not None and ry is not None and abs(gy - ry) <= 1)  # within 1 yr, paper-style
    if gy is not None and ry is not None:
        yr_total += 1; yr_hits += int(match)
    print(f"{p['name']:>10} {gvc:>6} {rvc:>6} {rvc-gvc:>+5} "
          f"{str(gy):>6} {str(ry):>6} {'Y' if match else 'n':>4} {mae_pct:>5.1f}")

print("\n--- paper's metrics, our 5 pixels (standard params, GEE source) ---")
print(f"mean |vertex-count diff| : {np.mean(np.abs(cnt_diffs)):.1f}  "
      f"(exact-count match: {sum(d==0 for d in cnt_diffs)}/5)")
print(f"MAE as % of NBR range    : {np.mean(mae_pcts):.2f}%   (paper bar: <3% for 5/6 regions)")
print(f"disturbance-year agree   : {yr_hits}/{yr_total} within 1yr   (paper: >97% on co-detected)")
