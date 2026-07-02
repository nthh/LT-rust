#!/usr/bin/env python3
"""Reproduce the reference LandTrendr example (LT-GEE Fig 2.1) with LT-rs.

The textbook pixel: Lon -123.845, Lat 45.889 (Oregon Coast Range, conifer-dominated
industrial forest): ~17 years stable mature conifer (1984-2000), a service road in
2000-2001, clearcut harvest 2001-2002, then 14 years of regrowth to 2016. LandTrendr
should place a disturbance vertex at ~2001.

Pipeline (runs locally from cloud-native Landsat):
  MPC STAC (landsat-c2-l2, all sensors via common asset names nir08/swir22/qa_pixel)
    -> windowed COG reads -> cloud-mask + annual median NBR composite (Jun-Sep)
    -> landtrendr.pixel  (the Rust kernel; same engine the browser runs via WASM)

Window 1984-2016 (matches the published figure: the 1984-2000 stable baseline is the
first segment that anchors the 2001 vertex; truncating to 2000 loses it).

First run fetches + caches the annual NBR stack to data/nbr_1984_2016.npz
(committed, small). Subsequent runs are offline.

A GEE-truth side-by-side (source/fitted/isVertex from native ee.Algorithms.
TemporalSegmentation.LandTrendr) drops in via data/gee_truth.json (see gee_truth.py);
this script prints the comparison when that file exists.

Run: python python/fetch_nbr.py
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import landtrendr as fc

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data"; DATA.mkdir(exist_ok=True)
CACHE = DATA / "nbr_1984_2016.npz"
GEE_TRUTH = DATA / "gee_truth.json"

LON, LAT = -123.845, 45.889
# ~1.5 km box around the reference pixel: enough for the pixel + neighborhood, small npz
BBOX = (-123.855, 45.882, -123.835, 45.896)
START, END = 1984, 2016
TARGET_EPSG, RES = "EPSG:32610", 30.0
SR_SCALE, SR_OFFSET = 2.75e-5, -0.2
QA_BAD_BITS = [1, 2, 3, 4]                  # dilated-cloud, cirrus, cloud, cloud-shadow
COLLECTION = "landsat-c2-l2"
BANDS = ("nir08", "swir22", "qa_pixel")     # MPC common names: right band per sensor


def target_grid():
    from pyproj import Transformer
    from rasterio.transform import from_origin
    tf = Transformer.from_crs("EPSG:4326", TARGET_EPSG, always_xy=True)
    xs, ys = [], []
    for lon in (BBOX[0], BBOX[2]):
        for lat in (BBOX[1], BBOX[3]):
            x, y = tf.transform(lon, lat); xs.append(x); ys.append(y)
    xmin = np.floor(min(xs) / RES) * RES; xmax = np.ceil(max(xs) / RES) * RES
    ymin = np.floor(min(ys) / RES) * RES; ymax = np.ceil(max(ys) / RES) * RES
    W = int(round((xmax - xmin) / RES)); H = int(round((ymax - ymin) / RES))
    return from_origin(xmin, ymax, RES, RES), W, H


def qa_bad(qa):
    qi = qa.astype(np.uint32); bad = np.zeros(qi.shape, bool)
    for b in QA_BAD_BITS:
        bad |= ((qi >> b) & 1).astype(bool)
    return bad


def nbr_from(nir_dn, swir_dn, qa):
    nir = nir_dn.astype("f4") * SR_SCALE + SR_OFFSET
    swir = swir_dn.astype("f4") * SR_SCALE + SR_OFFSET
    denom = nir + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        nbr = (nir - swir) / denom
    nbr[qa_bad(qa) | (nir_dn == 0) | (denom == 0)] = np.nan
    return nbr


def fetch_stack(max_cloud=60.0, workers=8):
    import planetary_computer as pc
    from pystac_client import Client
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.enums import Resampling
    from concurrent.futures import ThreadPoolExecutor, as_completed
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
    os.environ.setdefault("VSI_CACHE", "TRUE")

    transform, W, H = target_grid()
    print(f"[read] grid {W}x{H} @ {RES:.0f} m, {TARGET_EPSG}; bbox {BBOX}")
    cat = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=pc.sign_inplace)
    items = None
    for attempt in range(4):
        try:
            items = list(cat.search(collections=[COLLECTION], bbox=BBOX,
                          datetime=f"{START}-01-01/{END}-12-31",
                          query={"eo:cloud_cover": {"lt": max_cloud},
                                 "platform": {"in": ["landsat-5", "landsat-7", "landsat-8"]}}).items())
            break
        except Exception as e:
            print(f"[read] STAC retry {attempt+1}/4: {e}"); time.sleep(2 * (attempt + 1))
    if not items:
        sys.exit("[read] no scenes (STAC unreachable?)")
    # growing season Jun-Sep, to match eMapR summer compositing
    items = [it for it in items if 6 <= it.datetime.month <= 9]
    print(f"[read] {len(items)} summer scenes, cloud<{max_cloud}%, {START}-{END}")

    def band(href):
        with rasterio.open(href) as src, WarpedVRT(
            src, crs=TARGET_EPSG, transform=transform, width=W, height=H,
            resampling=Resampling.nearest) as vrt:
            return vrt.read(1)

    def read_item(it):
        nir, swir, qa = (band(it.assets[b].href) for b in BANDS)
        return it.datetime.year, nbr_from(nir, swir, qa)

    by_year = defaultdict(list)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(read_item, it): it for it in items}
        done = 0
        for fut in as_completed(futs):
            it = futs[fut]; done += 1
            try:
                yr, nbr = fut.result(); by_year[yr].append(nbr)
            except Exception as e:
                print(f"[read]   skip {it.id}: {e}"); continue
            if done % 50 == 0:
                print(f"[read]   {done}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    years = [y for y in range(START, END + 1) if by_year.get(y)]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        annual = np.stack([np.nanmedian(np.stack(by_year[y]), axis=0) for y in years]).astype("f4")
    transform, W, H = target_grid()
    np.savez_compressed(CACHE, annual=annual, years=np.asarray(years, np.int32),
                        transform=np.asarray(transform[:6], float), bbox=np.asarray(BBOX, float),
                        scenes_per_year=np.asarray([len(by_year[y]) for y in years], np.int32))
    print(f"[cache] saved {CACHE.name}: {annual.shape}, {len(years)} years, "
          f"{annual.nbytes/1e3:.0f} KB in {time.time()-t0:.0f}s")
    return annual, np.asarray(years, np.int32), transform


def pixel_index(transform, lon, lat):
    from pyproj import Transformer
    from rasterio.transform import rowcol
    tf = Transformer.from_crs("EPSG:4326", TARGET_EPSG, always_xy=True)
    x, y = tf.transform(lon, lat)
    r, c = rowcol(transform, x, y)
    return int(r), int(c)


def main():
    from rasterio.transform import Affine
    if CACHE.exists():
        z = np.load(CACHE); annual = z["annual"]; years = z["years"]
        transform = Affine(*z["transform"]); spy = z["scenes_per_year"]
        print(f"[cache] loaded {CACHE.name}: {annual.shape}, {years[0]}-{years[-1]}")
    else:
        annual, years, transform = main_fetch()
        spy = None

    r, c = pixel_index(transform, LON, LAT)
    H, W = annual.shape[1:]
    r = min(max(r, 0), H - 1); c = min(max(c, 0), W - 1)
    series = annual[:, r, c].astype(np.float32)
    print(f"\n[pixel] {LON},{LAT} -> grid ({r},{c}) of {H}x{W}")

    # standard LT-GEE defaults (emapr.github.io/LT-GEE/running-lt-gee.html):
    # maxSegments 6, spike 0.9, vertexCountOvershoot 3, preventOneYearRecovery true,
    # recovery 0.25, pval 0.05, bestModelProportion 0.75, minObs 6. (overshoot +
    # preventOneYearRecovery are the Rust defaults.) NBR is loss-DOWN, which matches
    # the Rust kernel's loss=decrease convention -> feed raw NBR, no negation.
    fit, vtx, rmse = fc.pixel(
        np.ascontiguousarray(series), years.astype(np.int32),
        max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
        p_value_threshold=0.05, best_model_proportion=0.75, min_observations_needed=6)
    fit = np.asarray(fit); vtx = np.asarray(vtx).astype(int)
    vyears = [int(years[i]) for i in range(len(years)) if vtx[i] == 1]

    # NBR scaled to thousands for readability, like the figures
    print(f"\n{'year':>5} {'NBR':>7} {'fitted':>7} {'vertex':>6}" + ("  scenes" if spy is not None else ""))
    for i, y in enumerate(years):
        s = "" if not np.isfinite(series[i]) else f"{series[i]*1000:7.0f}"
        mark = " <-- VERTEX" if vtx[i] else ""
        extra = f"  {spy[i]:6d}" if spy is not None else ""
        print(f"{y:>5} {s:>7} {fit[i]*1000:7.0f} {vtx[i]:>6}{extra}{mark}")

    # disturbance = largest single-year fitted DROP (NBR loss)
    dfit = np.diff(fit); di = int(np.argmin(dfit))
    print(f"\n[result] vertices at: {vyears}")
    print(f"[result] largest NBR drop {dfit[di]*1000:.0f} between {years[di]}->{years[di+1]} "
          f"(eMapR: clearcut ~2001-2002)  rmse {rmse*1000:.0f}")

    # Fig 2.1-style plot
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.2))
        m = np.isfinite(series)
        ax.plot(years[m], series[m]*1000, "o", color="#6b7280", ms=4, label="annual NBR (source)")
        ax.plot(years, fit*1000, "-", color="#2f855a", lw=2, label="LandTrendr fitted (Rust)")
        ax.plot([years[i] for i in range(len(years)) if vtx[i]],
                [fit[i]*1000 for i in range(len(years)) if vtx[i]],
                "s", color="#c53030", ms=7, label="vertices")
        ax.set_xlabel("year"); ax.set_ylabel("NBR x1000"); ax.set_title(
            f"LandTrendr reference pixel {LON},{LAT} (LT-rs)")
        ax.legend(frameon=False); ax.grid(alpha=0.2); fig.tight_layout()
        out = HERE / "fig21.png"; fig.savefig(out, dpi=120)
        print(f"[plot] {out.name}")
    except Exception as e:
        print(f"[plot] skipped ({e})")

    # GEE side-by-side when the truth cache exists
    if GEE_TRUTH.exists():
        g = json.load(open(GEE_TRUTH))
        gv = set(g.get("vertex_years", [])); rv = set(vyears)
        jac = len(gv & rv) / max(1, len(gv | rv))
        print(f"\n[vs GEE] GEE vertices {sorted(gv)} / Rust {sorted(rv)}  Jaccard {jac:.2f}")
    else:
        print(f"\n[vs GEE] no truth cache yet ({GEE_TRUTH.name}); run gee_truth.py to add it")


def main_fetch():
    return fetch_stack()


if __name__ == "__main__":
    main()
