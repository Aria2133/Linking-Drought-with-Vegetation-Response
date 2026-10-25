"""MOD17A2H monthly GPP: deterministic unit conversion and validation of monthly Earth Engine exports.

The GEE scripts export monthly totals in kg C m-2 month-1. FLUXCOM-X-BASE is in g C m-2 d-1, so every MOD17 month is
converted with x 1000 / days_in_month. The conversion is applied by product, never by a value threshold: the earlier
rule "convert if the raster median < 0.2" left Natural Forests 2023-05 and 2023-06 unconverted
(results/audit/gpp_table_s8/mod17_months_left_unconverted.csv).

Usage: python src/ecological/gpp/mod17_monthly.py verify --folder <GeoTIFF folder> --year 2022 [--out report.csv]
Checks: 12 files present; no two months byte-identical or pixel-identical; plausible range after conversion.
"""
import argparse
import calendar
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PATTERN = "GPP_MOD17A2H_CA_5km_{year}_{month:02d}.tif"
# California monthly means of FLUXCOM / MOD17 lie well inside this range (g C m-2 d-1); used only as a sanity check.
PLAUSIBLE_GC_DAY = (0.0, 15.0)


def kgc_month_to_gc_day(values, year: int, month: int):
    """kg C m-2 month-1 -> g C m-2 d-1 for the given calendar month (always applied to MOD17 monthly totals)."""
    return np.asarray(values, dtype="float64") * 1000.0 / calendar.monthrange(year, month)[1]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_months(arrays: dict, hashes: dict, year: int) -> pd.DataFrame:
    """arrays/hashes keyed by month (1..12). Returns one row per month with the failed checks (empty string = ok)."""
    rows = []
    for m in range(1, 13):
        problems = []
        if m not in arrays:
            rows.append({"year": year, "month": m, "problems": "missing"})
            continue
        a = arrays[m]
        same_bytes = [o for o in arrays if o != m and hashes[o] == hashes[m]]
        same_pixels = [o for o in arrays if o != m and arrays[o].shape == a.shape
                       and np.array_equal(arrays[o], a, equal_nan=True)]
        if same_bytes:
            problems.append("byte-identical to month(s) " + ",".join(map(str, same_bytes)))
        if same_pixels:
            problems.append("pixel-identical to month(s) " + ",".join(map(str, same_pixels)))
        gc = kgc_month_to_gc_day(a, year, m)
        finite = gc[np.isfinite(gc)]
        mean = float(finite.mean()) if finite.size else np.nan
        if not finite.size:
            problems.append("no finite pixels")
        elif not (PLAUSIBLE_GC_DAY[0] <= mean <= PLAUSIBLE_GC_DAY[1]):
            problems.append(f"implausible mean {mean:.3f} gC m-2 d-1")
        rows.append({"year": year, "month": m, "sha256": hashes[m], "n_finite": int(finite.size),
                     "mean_kgC_m2_month": float(np.nanmean(a)) if finite.size else np.nan,
                     "mean_gC_m2_day": mean, "problems": "; ".join(problems)})
    return pd.DataFrame(rows)


def read_folder(folder: Path, year: int):
    import rasterio

    arrays, hashes = {}, {}
    for m in range(1, 13):
        path = folder / PATTERN.format(year=year, month=m)
        if not path.exists():
            continue
        with rasterio.open(path) as src:
            a = src.read(1, masked=True).astype("float64").filled(np.nan)
        arrays[m], hashes[m] = a, file_hash(path)
    return arrays, hashes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--folder", required=True, type=Path)
    v.add_argument("--year", required=True, type=int)
    v.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    report = check_months(*read_folder(args.folder, args.year), args.year)
    if args.out:
        report.to_csv(args.out, index=False)
    print(report.to_string(index=False))
    bad = report[report.problems.fillna("") != ""]
    print("OK: all 12 months valid" if bad.empty else f"FAILED: {len(bad)} month(s) invalid")
    return 0 if bad.empty else 1


if __name__ == "__main__":
    sys.exit(main())
