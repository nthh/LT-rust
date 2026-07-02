# LT-IDL validation harness

The `idl_*` scripts in `python/` validate the Rust port against the **original**
LandTrendr-2012 IDL segmentation (`fit_trajectory_v2` → `tbcd_v2`), run under
[GNU Data Language (GDL)](https://github.com/gnudatalanguage/gdl) — an open-source
IDL interpreter, so no IDL license is needed. This directory holds two `.pro`
shims the headless GDL build is missing; the setup below wires everything together.

Neither GDL nor the IDL source is bundled (GDL is a platform binary; the IDL
source is a separate repository). Both are one-time installs.

## What you need

1. **GDL** — the interpreter that runs the IDL code.
2. **LandTrendr-2012 IDL source** — the unmodified algorithm.
3. **The shims in this directory** (`regress.pro`, `f_test1.pro`) — on the GDL
   search path. The scripts add this automatically; nothing to do by hand.

`regress.pro` and `f_test1.pro` replace two routines absent from the headless
GDL build; each is documented in its header as mathematically identical to the
routine it stands in for (ordinary least squares, and the F-distribution CDF via
the regularized incomplete beta), so they preserve the algorithm rather than
approximate it.

## Install

**GDL.** Use your platform package manager, or a prebuilt app:

```bash
# macOS (Homebrew)
brew install gnudatalanguage
# Debian / Ubuntu
sudo apt-get install gnudatalanguage
```

Confirm it runs: `gdl --version`. Note the interpreter's full path (`which gdl`,
or `.../Contents/Resources/bin/gdl` inside a macOS .app).

**LandTrendr-2012 IDL source.** Clone the original algorithm anywhere:

```bash
git clone https://github.com/KennedyResearch/LandTrendr-2012 ~/projects/LandTrendr-2012
```

The segmentation algorithm lives under `segmentation/` in that repo
(`tbcd_v2.pro`, `fit_trajectory_v2.pro`, `desawtooth.pro`, `vet_verts3.pro`) —
the routines the scripts drive and the Rust kernel is validated against.

## Point the scripts at both

The scripts read two environment variables (falling back to the author's macOS
layout if unset — set them so they work on your machine):

```bash
export GDL_BIN="$(which gdl)"                        # or the full .app path
export LANDTRENDR_IDL="$HOME/projects/LandTrendr-2012"
```

`python/idl_env.py` resolves these and validates them; if either path is wrong,
every `idl_*` script exits immediately with which one is missing.

## Run

From the repo root, with the package installed (`maturin develop --features python`)
and `pip install -r python/requirements.txt`:

```bash
python python/idl_compare.py            # 5 GEE-truth pixels: IDL vs GEE vs Rust vertices + fit
python python/idl_tape_diff.py          # stage-by-stage vertex-selection tape, IDL vs Rust
python python/idl_pixel_debug.py        # full despike→vertices→fit tape for one pixel
python python/idl_vs_gee_vs_rust_map.py # 3-panel year-of-disturbance maps (regenerates images/)
```
