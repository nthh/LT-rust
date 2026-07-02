"""Shared paths for the IDL/GDL validation harness.

The LandTrendr-IDL comparison scripts need two things this repo does not bundle:
a GDL interpreter and the original LandTrendr-2012 IDL source. Point at them
with environment variables (see idl-harness/README.md for setup):

    export GDL_BIN=/path/to/gdl
    export LANDTRENDR_IDL=/path/to/LandTrendr-2012

If unset, these fall back to the author's macOS layout, so `gdl_paths()` always
returns something; `require_gdl()` validates and prints actionable guidance.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HARNESS = ROOT / "idl-harness"

_DEFAULT_GDL = os.path.expanduser(
    "~/Applications/GNU Data Language.app/Contents/Resources/bin/gdl"
)
_DEFAULT_LTSRC = Path.home() / "projects" / "LandTrendr-2012"


def gdl_paths():
    """(gdl_bin, ltsrc, harness) from $GDL_BIN / $LANDTRENDR_IDL, else defaults."""
    gdl = os.environ.get("GDL_BIN", _DEFAULT_GDL)
    ltsrc = Path(os.environ.get("LANDTRENDR_IDL", _DEFAULT_LTSRC))
    return gdl, ltsrc, HARNESS


def require_gdl():
    """As gdl_paths(), but exit with setup guidance if either path is missing."""
    gdl, ltsrc, harness = gdl_paths()
    problems = []
    if not Path(gdl).exists():
        problems.append(f"GDL interpreter not found at: {gdl}")
    if not ltsrc.exists():
        problems.append(f"LandTrendr-2012 IDL source not found at: {ltsrc}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        print(
            "\nSet GDL_BIN and LANDTRENDR_IDL, or see idl-harness/README.md "
            "for one-time setup.",
            file=sys.stderr,
        )
        sys.exit(1)
    return gdl, ltsrc, harness
