#!/usr/bin/env python3
"""Native GEE LandTrendr ground truth on GEE's OWN annual NBR composites.

The reference LT-rs is validated against: GEE builds the annual composites AND runs
LandTrendr, exactly as in LT-GEE. We capture both so compare.py can check two things
separately:
  - compositing: GEE's annual NBR vs our MPC annual NBR (data/nbr_*.npz)
  - algorithm:   GEE LandTrendr vs Rust LandTrendr on identical input

Memory-safe: each sensor is filterBounds(region) BEFORE merge/median, so GEE only
loads scenes over the ~1.5 km AOI (the unfiltered 3-sensor merge over 1984-2016 hits
"user memory limit exceeded"). LandTrendr then runs on GEE's 33 composite images fed
back as constant images per pixel (cheap, no re-compositing).

native LandTrendr is loss-up -> feed -NBR; source/fitted stored un-negated (NBR space).

Caches [year, source(GEE composite), fitted, isVertex] per pixel to
../data/gee_truth.json.

Run: EE_PROJECT=your-cloud-project python python/gee_truth.py   # needs an Earth Engine account
"""
import ee, json, os, time
from pathlib import Path
import numpy as np

# Use your own Earth Engine cloud project: set EE_PROJECT, or rely on the gcloud default.
_proj = os.environ.get("EE_PROJECT")
ee.Initialize(project=_proj) if _proj else ee.Initialize()
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "gee_truth.json"

z = np.load(ROOT / "data" / "nbr_1984_2016.npz")
years = z["years"].astype(int)
W, S, E, N = [float(v) for v in z["bbox"]]
region = ee.Geometry.Rectangle([W, S, E, N])
START, END = int(years[0]), int(years[-1])
# reference pixel (26,26) + 4 neighbors, as lon/lat from the npz transform
from rasterio.transform import Affine, xy
T = Affine(*z["transform"])
from pyproj import Transformer
to_ll = Transformer.from_crs("EPSG:32610", "EPSG:4326", always_xy=True)
PIX = {"center": (26, 26), "n_up": (25, 26), "n_down": (27, 26),
       "n_left": (26, 25), "n_right": (26, 27)}
def pix_lonlat(r, c):
    x, y = xy(T, r, c)            # UTM center
    return to_ll.transform(x, y)

RUN = dict(maxSegments=6, spikeThreshold=0.9, vertexCountOvershoot=3,
           preventOneYearRecovery=True, recoveryThreshold=0.25, pvalThreshold=0.05,
           bestModelProportion=0.75, minObservationsNeeded=6)

# --- GEE's own annual NBR composites (NBRx1000), memory-safe via filterBounds ---
def mask_sr(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0))
               .And(qa.bitwiseAnd(1 << 3).eq(0)).And(qa.bitwiseAnd(1 << 4).eq(0)))
    sr = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    return ee.Image(sr.updateMask(clear).copyProperties(img, ["system:time_start"]))

def nbr(coll, nir, swir2):
    coll = coll.filterBounds(region)                      # <-- the memory fix
    def f(img):
        m = mask_sr(img); t = img.get("system:time_start")
        return (m.normalizedDifference([nir, swir2]).multiply(1000).rename("nbr")
                  .set("year", ee.Date(t).get("year")).set("system:time_start", t))
    return coll.map(f)

l5 = nbr(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"), "SR_B4", "SR_B7")
l7 = nbr(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"), "SR_B4", "SR_B7")
l8 = nbr(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"), "SR_B5", "SR_B7")
allc = l5.merge(l7).merge(l8).filter(ee.Filter.calendarRange(6, 9, "month"))

def annual(y):
    y = ee.Number(y)
    return (allc.filter(ee.Filter.eq("year", y)).median().rename(ee.String("y").cat(y.format("%d")))
              .set("system:time_start", ee.Date.fromYMD(y, 8, 1).millis()))
comp = ee.ImageCollection(ee.List.sequence(START, END).map(annual))
stack = comp.toBands()   # 33 bands y1984..y2016 (band names get a prefix)

# sample GEE composites at all pixels in one call
fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(pix_lonlat(r, c)), {"name": nm})
                           for nm, (r, c) in PIX.items()])
print("[gee] sampling GEE annual NBR composites...", flush=True)
samp = None
for a in range(4):
    try:
        samp = stack.sampleRegions(collection=fc, scale=30, geometries=False).getInfo()
        break
    except Exception as e:
        print(f"  composite retry {a+1}/4: {type(e).__name__} {str(e)[:80]}", flush=True)
        time.sleep(3 * (a + 1))
if samp is None:
    raise SystemExit("[gee] composite sampling failed")

# band names from toBands are like '0_y1984','1_y1985',... map by suffix yYYYY
def series_for(props):
    out = np.full(len(years), np.nan)
    for k, v in props.items():
        if "y" in k and v is not None:
            yr = int(k.split("y")[-1])
            if START <= yr <= END:
                out[yr - START] = v
    return out
gee_src = {f["properties"]["name"]: series_for(f["properties"]) for f in samp["features"]}

# --- native LandTrendr on GEE's composites (constant images, cheap) ------------
def gee_landtrendr(series):
    imgs = []
    for y, v in zip(years, series):
        fin = np.isfinite(v)
        imgs.append(ee.Image.constant(float(-v) if fin else 0.0).rename("idx").float()
                    .updateMask(ee.Image.constant(1 if fin else 0))
                    .set("system:time_start", ee.Date.fromYMD(int(y), 8, 1).millis()))
    lt = ee.Algorithms.TemporalSegmentation.LandTrendr(timeSeries=ee.ImageCollection(imgs), **RUN)
    for a in range(4):
        try:
            return lt.reduceRegion(ee.Reducer.first(), ee.Geometry.Point(0, 0), 1).getInfo()["LandTrendr"]
        except Exception as e:
            print(f"    LT retry {a+1}/4: {type(e).__name__}", flush=True); time.sleep(2 * (a + 1))
    return None

out = {"start": START, "end": END, "params": RUN,
       "note": "GEE native LandTrendr on GEE's own filterBounds composites; NBRx1000 (loss-down)",
       "pixels": []}
t0 = time.time()
for name in PIX:
    series = gee_src[name]
    arr = gee_landtrendr(series)
    if arr is None:
        print(f"  {name} LT FAILED"); continue
    yrs, src_neg, fit_neg, vtx = arr
    rec = {"name": name,
           "years": [int(y) for y in yrs],
           "source": [round(float(series[list(years).index(int(y))]), 1)
                      if int(y) in years and np.isfinite(series[list(years).index(int(y))]) else None
                      for y in yrs],   # GEE composite NBRx1000
           "fitted": [(-v if v is not None else None) for v in fit_neg],
           "vertex": [int(v) for v in vtx],
           "vertex_years": [int(yrs[k]) for k in range(len(yrs)) if vtx[k] == 1]}
    out["pixels"].append(rec)
    print(f"  {name} vertices {rec['vertex_years']} ({time.time()-t0:.0f}s)", flush=True)

out["vertex_years"] = out["pixels"][0]["vertex_years"] if out["pixels"] else []
json.dump(out, open(OUT, "w"), indent=2)
print(f"\nsaved {OUT.name}: {len(out['pixels'])} pixels in {time.time()-t0:.0f}s")
print(f"center GEE vertices: {out['vertex_years']}")
