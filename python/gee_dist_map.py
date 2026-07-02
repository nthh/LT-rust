#!/usr/bin/env python3
"""GEE LandTrendr over a small AOI -> downloads the fitted + source NBR stacks.

Produces the GEE side of the LT-GEE paper's Figure 5 comparison: it runs the
native GEE LandTrendr (ee.Algorithms.TemporalSegmentation.LandTrendr, the same
engine eMapR's LandTrendr.js wraps) and exports the per-year FITTED and SOURCE
NBR as GeoTIFF stacks. Disturbance-year extraction is then done locally and
*identically* for GEE and for the LT-rs kernel (see compare_maps.py), so the
two maps differ only by the segmentation, not by the change-extraction logic.

Self-contained: needs only `earthengine-api` + an authenticated EE account.
  eMapR: edit AOI / START / END / COMPOSITE below and run. Nothing else to change.

GEE LandTrendr is loss-UP, so we feed -NBR and negate the fitted/source back to
NBR (loss-down), matching the Rust kernel's convention.

Run: EE_PROJECT=<your-cloud-project> python python/gee_dist_map.py
"""
import ee, os, json, urllib.request
from pathlib import Path

_proj = os.environ.get("EE_PROJECT")
ee.Initialize(project=_proj) if _proj else ee.Initialize()

# ============================ CHANGE ME (eMapR) ===============================
# Edit AOI / window here, or override per run with env vars LT_AOI (JSON bbox),
# LT_TAG (output prefix), LT_START, LT_END (used by the multi-site batch).
AOI = json.loads(os.environ["LT_AOI"]) if os.environ.get("LT_AOI") else \
    [-123.855, 45.882, -123.835, 45.896]     # ~1.5 km Oregon Coast Range (reference box)
TAG = os.environ.get("LT_TAG", "gee")        # outputs data/<TAG>_source.tif, <TAG>_distyear.tif
START = int(os.environ.get("LT_START", 1984))
END = int(os.environ.get("LT_END", 2016))
COMPOSITE = "median"          # paper uses medoid; median is simpler + matches our prior truth
# Output grid. scale+region keeps the synchronous download bounded to the AOI
# (crsTransform makes GEE size the request to the full Landsat footprint -> blows
# the 50 MB getDownloadURL limit). For large AOIs use ee.batch.Export instead.
CRS = "EPSG:32610"
SCALE = 30
# LT-GEE standard runParams (Kennedy et al. 2018, Table 1; +preventOneYearRecovery).
RUN = dict(maxSegments=6, spikeThreshold=0.9, vertexCountOvershoot=3,
           preventOneYearRecovery=True, recoveryThreshold=0.25, pvalThreshold=0.05,
           bestModelProportion=0.75, minObservationsNeeded=6)
# =============================================================================

OUTDIR = Path(__file__).resolve().parent.parent / "data"
region = ee.Geometry.Rectangle(AOI)
years = list(range(START, END + 1))


def mask_sr(img):
    qa = img.select("QA_PIXEL")
    clear = (qa.bitwiseAnd(1 << 1).eq(0).And(qa.bitwiseAnd(1 << 2).eq(0))
               .And(qa.bitwiseAnd(1 << 3).eq(0)).And(qa.bitwiseAnd(1 << 4).eq(0)))
    sr = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    return sr.updateMask(clear).copyProperties(img, ["system:time_start"])


def nbr(coll, nir, swir2):
    coll = coll.filterBounds(region)  # memory-safe: only scenes over the AOI
    def f(img):
        m = mask_sr(img); t = img.get("system:time_start")
        return (ee.Image(m).normalizedDifference([nir, swir2]).multiply(1000).rename("nbr")
                .set("year", ee.Date(t).get("year")).set("system:time_start", t))
    return coll.map(f)


l5 = nbr(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2"), "SR_B4", "SR_B7")
l7 = nbr(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2"), "SR_B4", "SR_B7")
l8 = nbr(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2"), "SR_B5", "SR_B7")
allc = l5.merge(l7).merge(l8).filter(ee.Filter.calendarRange(6, 9, "month"))


def annual(y):
    y = ee.Number(y)
    c = allc.filter(ee.Filter.eq("year", y))
    img = c.median() if COMPOSITE == "median" else c.mean()
    return (img.rename("nbr").set("system:time_start", ee.Date.fromYMD(y, 8, 1).millis()))


comp = ee.ImageCollection(ee.List.sequence(START, END).map(annual))

# loss-UP for GEE LandTrendr: feed -NBR.
ts = comp.map(lambda im: ee.Image(im).multiply(-1).rename("idx")
              .copyProperties(im, ["system:time_start"]))
lt = ee.Algorithms.TemporalSegmentation.LandTrendr(timeSeries=ts, **RUN)

NODATA = -32768

# (1) GEE's composite SOURCE stack = the LandTrendr input, NBR (loss-down). Fixed
# 33 bands via toBands() so the download never hits the ragged-array problem that
# arrayFlatten / position arrayGet do (a fully cloud-masked year shortens a pixel's
# LandTrendr array to 32 -> "index 32 out of bounds"). compare_maps.py feeds this
# same stack into the Rust kernel, so both algorithms see identical input.
source = comp.toBands().unmask(NODATA)

# (2) GEE disturbance-year, computed in GEE (length-agnostic). GEE LandTrendr is
# loss-up (we fed -NBR), so a forest loss is an INCREASE in the fitted series; the
# disturbance year is the year AFTER the largest such step (matches the local
# extraction: argmin of the loss-down diff, year[i+1]). loss_mag is that step in
# NBRx1000; compare_maps.py masks non-disturbance by loss_mag < threshold.
fitted = lt.select("LandTrendr").arraySlice(0, 2, 3).arrayProject([1])   # loss-up fitted
yrow = lt.select("LandTrendr").arraySlice(0, 0, 1).arrayProject([1])     # the pixel's own years
delta = fitted.arraySlice(0, 1).subtract(fitted.arraySlice(0, 0, -1))    # + = loss
idx = delta.arrayArgmax().arrayProject([0]).arrayFlatten([["i"]]).toInt()
lossmag = delta.arrayReduce(ee.Reducer.max(), [0]).arrayProject([0]).arrayFlatten([["m"]])
distmap = (yrow.arrayGet(idx.add(1)).rename("dist_year").unmask(0)
           .addBands(lossmag.rename("loss_mag").unmask(0)))

OUTDIR.mkdir(exist_ok=True)
DL = {"crs": CRS, "scale": SCALE, "region": region, "format": "GEO_TIFF"}
for name, img in ((f"{TAG}_source", source), (f"{TAG}_distyear", distmap)):
    url = img.clip(region).getDownloadURL(DL)
    dst = OUTDIR / f"{name}.tif"
    print(f"[gee] downloading {dst.name} ...", flush=True)
    urllib.request.urlretrieve(url, dst)
    print(f"      wrote {dst}")
print(f"done. tag={TAG}  AOI {AOI}  years {START}-{END}  composite {COMPOSITE}")
