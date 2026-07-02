#!/usr/bin/env python3
"""Stage-by-stage tape diff: LT-rs vs LT-IDL vertex selection.

Confirms the first-divergence localization: dumps each pipeline's intermediate
vertex sets (stage ② candidates from detection, stage ③ vetted after culling)
on the same pixel and shows exactly where they first disagree. Uses LT-rs's
pixel_debug binding and the LT-IDL find_vertices/vet_verts3 under GDL.
"""
import json
import os
import subprocess
import tempfile

import numpy as np
import landtrendr

from idl_env import ROOT, require_gdl

GDL, LTSRC, HARNESS = require_gdl()
GEE = json.load(open(ROOT / "data" / "gee_truth.json"))
CANON = dict(
    max_segments=6, spike_threshold=0.9, recovery_threshold=0.25,
    p_value_threshold=0.05, best_model_proportion=0.75,
    min_observations_needed=6, vertex_count_overshoot=3,
    prevent_one_year_recovery=True,
)

# max_count = max_segments+1 = 7; detection target = 7+overshoot(3) = 10; cull to 7.
IDL_TAPE = """!QUIET=1
!PATH = expand_path('+{harness}') + path_sep(/search_path) + expand_path('+{ltsrc}') + path_sep(/search_path) + !PATH
catch, err
if err ne 0 then begin
  print, 'CAUGHT_ERROR: ', !error_state.msg
  catch, /cancel
  exit
endif
resolve_routine, 'tbcd_v2', /is_function
years = [{yrs}]
src   = [{sv}]
n = n_elements(years)
miny = min(years)
x = years - miny
desp = desawtooth(src, stopat=0.9)
y = desp * (-1.0)
v1 = find_vertices(x, y, 10, 2.0)
v  = vet_verts3(x, y, v1, 7, 2.0)
print, 'CAND_START'
for i=0,n_elements(v1)-1 do print, x[v1[i]]+miny
print, 'CAND_END'
print, 'VET_START'
for i=0,n_elements(v)-1 do print, x[v[i]]+miny
print, 'VET_END'
exit
end
"""


def _between(lines, start, end):
    out, on = [], False
    for ln in lines:
        if start in ln:
            on = True
        elif end in ln:
            on = False
        elif on:
            try:
                out.append(int(round(float(ln.strip()))))
            except ValueError:
                pass
    return sorted(set(out))


def idl_tape(years, src):
    drv = IDL_TAPE.format(
        harness=HARNESS, ltsrc=LTSRC,
        yrs=",".join(str(int(y)) for y in years),
        sv=",".join(f"{float(v):.4f}" for v in src),
    )
    with tempfile.NamedTemporaryFile("w", suffix=".pro", delete=False) as f:
        f.write(drv)
        path = f.name
    try:
        out = subprocess.run(
            [GDL], input=f".run {path}\nexit\n",
            capture_output=True, text=True, timeout=120,
        ).stdout
    finally:
        os.unlink(path)
    lines = out.splitlines()
    return _between(lines, "CAND_START", "CAND_END"), _between(lines, "VET_START", "VET_END")


for name in ("center", "n_down"):
    p = next(p for p in GEE["pixels"] if p["name"] == name)
    years = np.array(p["years"], int)
    src = np.array(p["source"], float)

    _, rc_idx, rv_idx = landtrendr.pixel_debug(
        np.ascontiguousarray(src / 1000.0, np.float32),
        years.astype(np.int32), **CANON,
    )
    rc = sorted(int(years[i]) for i in rc_idx)
    rv = sorted(int(years[i]) for i in rv_idx)
    ic, iv = idl_tape(years, src)

    print(f"\n===== {name} =====")
    print(f"② candidates  IDL : {ic}")
    print(f"② candidates  rust: {rc}")
    print(f"     rust proposes, IDL never does : {sorted(set(rc) - set(ic))}")
    print(f"     IDL proposes, rust never does : {sorted(set(ic) - set(rc))}")
    print(f"③ vetted (final-7) IDL : {iv}")
    print(f"③ vetted (final-7) rust: {rv}")
    print(f"     rust KEEPS, IDL culls : {sorted(set(rv) - set(iv))}")
    print(f"     IDL KEEPS, rust culls : {sorted(set(iv) - set(rv))}")
