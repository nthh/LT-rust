#!/usr/bin/env python3
"""How close is our LandTrendr, and which knob drives the gap?

Two parts, both on GEE's OWN source series (algorithm isolated from compositing),
over all 5 GEE-truth pixels (center + 4 neighbors). Truth: data/gee_truth.json.

  1. BASELINE  — our kernel at standard params vs GEE, averaged over the 5 pixels:
                 vertex Jaccard, fitted MAD (NBRx1000), disturbance-depth capture.
  2. ABLATION  — sweep each parameter one at a time (others at standard default)
                 and report the same three metrics, to localize the divergence.

Hypothesis (from VALIDATION.md Findings 1-3): the disturbance depth is under-
captured because the fit smooths the sharp single-year trough. Prime suspects are
`spike_threshold` (despike) and `prevent_one_year_recovery`; if relaxing either
recovers depth, that localizes it. best_model_proportion / p_value are expected
to be ~neutral (shown output-neutral at the reference pixel).

NOTE: 5 pixels around ONE site is a localization probe, not a multi-site
validation. A real refit needs GEE truth over >=3 sites (a billed GEE run).

Run: .venv-lazy/bin/python python/ablate.py
"""
import json
from pathlib import Path
import numpy as np
import landtrendr

ROOT = Path(__file__).resolve().parent.parent
GEE = json.load(open(ROOT / "data" / "gee_truth.json"))
PX = GEE["pixels"]

CANON = dict(max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
             p_value_threshold=0.05, best_model_proportion=0.75,
             min_observations_needed=6, vertex_count_overshoot=3,
             prevent_one_year_recovery=True)


def arr(p, key, dtype=float):
    return np.array([np.nan if v is None else v for v in p[key]], dtype)


def metrics_for_pixel(p, params):
    """Run our kernel on GEE's source for one pixel; score vs GEE fitted/vertices."""
    years = np.array(p["years"], int)
    gsrc = arr(p, "source", np.float32)            # NBRx1000
    gfit = arr(p, "fitted")                         # NBRx1000
    gvtx = np.array(p["vertex"], int)
    gvy = set(int(years[i]) for i in range(len(years)) if gvtx[i])

    fit, vtx, _ = landtrendr.pixel(
        np.ascontiguousarray(gsrc / 1000.0, np.float32), years.astype(np.int32), **params)
    rfit = np.asarray(fit) * 1000.0
    rvy = set(int(years[i]) for i in range(len(years)) if np.asarray(vtx).astype(int)[i])

    jac = len(gvy & rvy) / max(1, len(gvy | rvy))
    mm = np.isfinite(gfit) & np.isfinite(rfit)
    mad = float(np.mean(np.abs(gfit[mm] - rfit[mm]))) if mm.any() else np.nan

    # disturbance-depth capture: at GEE's peak/trough YEARS, our drop / GEE drop
    fg = gfit.copy()
    pk_i, tr_i = int(np.nanargmax(fg)), int(np.nanargmin(fg))
    gee_depth = fg[pk_i] - fg[tr_i]
    our_depth = rfit[pk_i] - rfit[tr_i]
    cap = float(our_depth / gee_depth) if gee_depth > 1e-6 else np.nan
    return jac, mad, cap


def agg(params):
    js, ms, cs = [], [], []
    for p in PX:
        j, m, c = metrics_for_pixel(p, params)
        js.append(j); ms.append(m)
        if np.isfinite(c): cs.append(c)
    return np.mean(js), np.mean(ms), (np.mean(cs) if cs else np.nan)


def show(label, params):
    j, m, c = agg(params)
    print(f"  {label:34} Jaccard {j:.2f}   MAD {m:5.0f}   depth-capture {c*100:4.0f}%")


print(f"5 pixels: {[p['name'] for p in PX]}\n")
print("=== BASELINE (standard params, our kernel on GEE source) ===")
show("standard default", CANON)

SWEEPS = {
    "spike_threshold":        [0.75, 0.85, 0.90, 0.95, 1.00],
    "prevent_one_year_recovery": [True, False],
    "max_segments":           [6, 8, 10],
    "recovery_threshold":     [0.25, 0.50, 1.00],
    "vertex_count_overshoot": [3, 5, 7],
    "best_model_proportion":  [0.75, 1.25],
    "p_value_threshold":      [0.05, 0.01, 0.10],
}

for knob, vals in SWEEPS.items():
    print(f"\n=== ablate {knob} (others reference) ===")
    for v in vals:
        params = dict(CANON); params[knob] = v
        tag = f"{knob}={v}" + ("  <- default" if CANON[knob] == v else "")
        show(tag, params)
