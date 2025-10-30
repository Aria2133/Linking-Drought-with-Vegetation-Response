#!/usr/bin/env python3
"""
Simple, terminal-first dataset diagnostics for CA drought project.
Focus: quick truths you can paste back to me.

What it checks (per land type):
- File discovery + sample filenames when missing
- Shapes encountered (incl. coarse rasters) and common reference shape
- NaN/Inf/nodata rates and per-year missing fractions
- Temporal gaps by variable
- Value distributions (min/median/max, 1%–99% range) with outlier warnings
- EWBI/PGSI label feasibility counts per split (train/val/test)
- Target drift (train vs val vs test), simple correlations (drivers vs PGSI/EWBI)
- Pixel-selection stress test vs. min_valid_frac (how many nodes you can keep)

No plotting, just tidy terminal output + tiny CSVs.

Author: you & me
"""

import os, glob, warnings, math, json
from collections import Counter, defaultdict
from statistics import median
import numpy as np
import rasterio

warnings.filterwarnings("ignore")

# -------------------------------
# CONFIG (match your project)
# -------------------------------
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
VARIABLES  = ["GNDVI","GOSIF","GPP","LST","NDVI","NIRv","SPEI","TWS","VPD"]

TRAIN_YEARS = range(2001, 2022)  # inclusive end
VAL_YEARS   = range(2019, 2022)
TEST_YEARS  = range(2022, 2025)

# Paths like your training/testing structure
def get_train_base_path(land_type):
    base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
    land_map = {
        "GRASSLANDS": "CALIFORNIA GRASSLANDS",
        "NATURAL_FORESTS": "CALIFORNIA NATURAL FORESTS",
        "SAVANNAS": "CALIFORNIA SAVANNAS",
        "SHRUBLANDS": "CALIFORNIA SHRUBLANDS",
    }
    return os.path.join(base, land_map[land_type])

def get_test_base_path(land_type):
    base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"
    land_map = {
        "GRASSLANDS": "Grasslands",
        "NATURAL_FORESTS": "NaturalForests",
        "SAVANNAS": "Savannas",
        "SHRUBLANDS": "Shrublands",
    }
    return os.path.join(base, land_map[land_type])

# Robust-ish filename resolver (kept simple; prints samples if not found)
def find_data_file(base_path, variable, year, month, land_type, is_test=False):
    var_path = os.path.join(base_path, variable)
    if not os.path.isdir(var_path):
        return None

    # Test patterns
    if is_test:
        land_tag = {
            "GRASSLANDS": "Grasslands",
            "NATURAL_FORESTS": "NaturalForests",
            "SAVANNAS": "Savannas",
            "SHRUBLANDS": "Shrublands",
        }[land_type]
        patt = {
            "GOSIF": f"GOSIF_{year}.M{month:02d}_{land_tag}.tif",
            "GNDVI": f"GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            "GPP":   f"GPP_MOD17A2H_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            "LST":   f"MYD21C1_CA_LST_{year}_{month:02d}_{land_tag}.tif",
            "NDVI":  f"NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            "NIRv":  f"NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            "SPEI":  f"SPEI1_CSIC_CA_{year}_{month:02d}_{land_tag}.tif",
            "TWS":   f"TWS_GRACE_CA_{year}_{month:02d}_{land_tag}.tif",
            "VPD":   f"VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_{land_tag}.tif",
        }[variable]
        fp = os.path.join(var_path, patt)
        return fp if os.path.exists(fp) else None

    # Train patterns (handle Savanna/Savannas)
    land_tags = {
        "GRASSLANDS": ["Grasslands"],
        "NATURAL_FORESTS": ["NaturalForests"],
        "SAVANNAS": ["Savannas", "Savanna"],
        "SHRUBLANDS": ["Shrublands"],
    }[land_type]
    patterns = []
    for tag in land_tags:
        if variable == "GOSIF":
            patterns.append(f"GOSIF_{tag}_CALIFORNIA_GOSIF_{year}{month:02d}.tif")
        elif variable == "GNDVI":
            patterns.append(f"GNDVI_{tag}_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif")
        elif variable == "GPP":
            patterns.append(f"{tag}_GPP_{year}_{month:02d}.tif")
        elif variable == "LST":
            patterns.append(f"LST_{tag}_MYD21C1_CA_LST_{year}_{month:02d}.tif")
        elif variable == "NDVI":
            patterns.append(f"NDVI_{tag}_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif")
        elif variable == "NIRv":
            patterns.append(f"NIRv_{tag}_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif")
        elif variable == "SPEI":
            patterns.append(f"SPEI_{tag}_SPEI1_CSIC_CA_{year}_{month:02d}.tif")
        elif variable == "TWS":
            patterns.append(f"TWS_{tag}_TWS_GRACE_CA_{year}_{month:02d}.tif")
        elif variable == "VPD":
            patterns.append(f"VPD_{tag}_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif")
    for patt in patterns:
        fp = os.path.join(var_path, patt)
        if os.path.exists(fp):
            return fp
    return None

def read_geotiff(fp):
    with rasterio.open(fp) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
    arr[~np.isfinite(arr)] = np.nan
    return arr

# Simple stats helpers
def pct(x, total): return (100.0*x/total) if total else 0.0
def finite_ratio(a):
    n = a.size
    f = np.isfinite(a).sum()
    return f/n if n else 0.0

def p01_99(a):
    v = a[np.isfinite(a)]
    if v.size == 0: return (np.nan, np.nan)
    return (float(np.percentile(v,1)), float(np.percentile(v,99)))

def safe_corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return np.nan
    xb, yb = x[m], y[m]
    if xb.std() == 0 or yb.std() == 0: return np.nan
    return float(np.corrcoef(xb, yb)[0,1])

# -------------------------------
# CORE DIAGNOSTIC
# -------------------------------
def diagnose_land_type(land_type):
    print("\n" + "="*90)
    print(f"LAND TYPE: {land_type}")
    print("="*90)

    train_base = get_train_base_path(land_type)
    test_base  = get_test_base_path(land_type)

    # pass 1: scan files, shapes, coverage
    shapes_seen = Counter()
    shapes_by_var = defaultdict(Counter)
    files_found = 0
    files_missing = []

    # store data lite: per timestep per var -> shape, finite ratio, value stats (min/median/max)
    meta = {}
    def regkey(y,m): return f"{y}_{m:02d}"

    # Find a reference shape (most frequent) from anything 2001–2021
    for y in TRAIN_YEARS:
        for m in range(1,13):
            for v in VARIABLES:
                if v == "LST" and y < 2003:
                    continue
                if v == "TWS" and (y < 2002 or (y == 2002 and m < 4)):
                    continue
                fp = find_data_file(train_base, v, y, m, land_type, is_test=False)
                if not fp:
                    continue
                arr = read_geotiff(fp)
                sh = arr.shape
                shapes_seen[sh] += 1
                shapes_by_var[v][sh] += 1
                files_found += 1
                k = regkey(y,m)
                meta.setdefault(k, {})
                meta[k][v] = {
                    "shape": sh,
                    "finite_frac": finite_ratio(arr),
                    "min": float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan,
                    "median": float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan,
                    "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan,
                    "p01": p01_99(arr)[0], "p99": p01_99(arr)[1],
                    "nfinite": int(np.isfinite(arr).sum()),
                    "n": int(arr.size),
                }
    # Also check test to catch reference if train sparse
    for y in TEST_YEARS:
        for m in range(1,13):
            if y==2024 and m>10: break
            for v in VARIABLES:
                fp = find_data_file(test_base, v, y, m, land_type, is_test=True)
                if not fp:
                    continue
                arr = read_geotiff(fp)
                sh = arr.shape
                shapes_seen[sh] += 1
                shapes_by_var[v][sh] += 1
                files_found += 1
                k = regkey(y,m)
                meta.setdefault(k, {})
                meta[k][v] = {
                    "shape": sh,
                    "finite_frac": finite_ratio(arr),
                    "min": float(np.nanmin(arr)) if np.isfinite(arr).any() else np.nan,
                    "median": float(np.nanmedian(arr)) if np.isfinite(arr).any() else np.nan,
                    "max": float(np.nanmax(arr)) if np.isfinite(arr).any() else np.nan,
                    "p01": p01_99(arr)[0], "p99": p01_99(arr)[1],
                    "nfinite": int(np.isfinite(arr).sum()),
                    "n": int(arr.size),
                }

    if files_found == 0:
        print("!! No files found at all. Check paths.")
        return

    ref_shape, ref_count = shapes_seen.most_common(1)[0]
    print(f"Reference shape (most frequent): {ref_shape}  [{ref_count} occurrences]")
    print("Unique shapes per variable:")
    for v in VARIABLES:
        if shapes_by_var[v]:
            top = shapes_by_var[v].most_common(3)
            print(f"  - {v:<5}: " + ", ".join([f"{sh}×{cnt}" for sh,cnt in top]))
        else:
            print(f"  - {v:<5}: (none)")

    # Coarse detection flag
    def is_coarse(sh):
        H,W = sh
        return (H*W) <= 64 or H<=10 or W<=10

    coarse_flags = {v: any(is_coarse(sh) for sh in shapes_by_var[v]) for v in VARIABLES}
    coarse_vars = [v for v,f in coarse_flags.items() if f]

    print("\nCOARSE RASTER DETECTION:")
    if coarse_vars:
        for v in coarse_vars:
            ex = ", ".join([str(sh) for sh,_ in shapes_by_var[v].most_common(3)])
            print(f"  * {v}: coarse shapes seen -> {ex}")
    else:
        print("  none")

    # -----------------------
    # Coverage & temporal gaps
    # -----------------------
    def month_iter(years):
        for y in years:
            for m in range(1,13):
                yield y,m

    print("\nCOVERAGE SUMMARY (fraction of months with a file & finite pixels):")
    for v in VARIABLES:
        total_months = 0
        ok_months = 0
        for y,m in month_iter(range(2001,2025)):
            if y==2024 and m>10: break
            k = regkey(y,m)
            if k in meta and v in meta[k]:
                total_months += 1
                if meta[k][v]["finite_frac"] > 0:
                    ok_months += 1
        if total_months==0:
            print(f"  - {v:<5}: no months found")
        else:
            print(f"  - {v:<5}: {ok_months}/{total_months} months ({pct(ok_months,total_months):.1f}%)")

    # Per-year missing fraction (all vars pooled as mean)
    by_year_missing = defaultdict(list)
    for y in range(2001,2025):
        for m in range(1,13):
            if y==2024 and m>10: break
            k = regkey(y,m)
            fracs=[]
            for v in VARIABLES:
                if k in meta and v in meta[k]:
                    fracs.append(1.0 - meta[k][v]["finite_frac"])
            if fracs:
                by_year_missing[y].append(np.mean(fracs))
    print("\nMEAN MISSING FRACTION BY YEAR (avg over available vars per month):")
    for y in sorted(by_year_missing.keys()):
        print(f"  {y}: {np.mean(by_year_missing[y]):.3f}")

    # -----------------------
    # Label feasibility (EWBI/PGSI)
    # -----------------------
    def ewbi_ready(k):
        need = ["SPEI","TWS","VPD","LST"]
        have = 0
        for v in need:
            if k in meta and v in meta[k] and meta[k][v]["finite_frac"]>0:
                have += 1
        return have>=2  # relaxed: consider >=2 drivers usable

    def pgsi_ready(k):
        veg = ["GOSIF","GPP","NIRv","NDVI","GNDVI"]
        have = 0
        for v in veg:
            if k in meta and v in meta[k] and meta[k][v]["finite_frac"]>0:
                have += 1
        return have>=3

    def count_ready(years, pred="EWBI"):
        cnt=0
        for y,m in month_iter(years):
            if y==2024 and m>10: break
            k=regkey(y,m)
            if (ewbi_ready(k) if pred=="EWBI" else pgsi_ready(k)):
                cnt+=1
        return cnt

    print("\nLABEL FEASIBILITY COUNTS (months usable):")
    tr_ew = count_ready(TRAIN_YEARS, "EWBI")
    tr_pg = count_ready(TRAIN_YEARS, "PGSI")
    va_ew = count_ready(VAL_YEARS, "EWBI")
    va_pg = count_ready(VAL_YEARS, "PGSI")
    te_ew = count_ready(TEST_YEARS, "EWBI")
    te_pg = count_ready(TEST_YEARS, "PGSI")
    print(f"  Train (2001–2018): EWBI={tr_ew:3d}  PGSI={tr_pg:3d}")
    print(f"  Val   (2019–2021): EWBI={va_ew:3d}  PGSI={va_pg:3d}")
    print(f"  Test  (2022–2024): EWBI={te_ew:3d}  PGSI={te_pg:3d}")

    # -----------------------
    # Drift & correlations (time-series of spatial means)
    # -----------------------
    # we approximate spatial mean by the median across finite pixels reported in meta
    # (not exact, but sufficient for drift/corr signal)
    def monthly_value(k, vname):
        if k in meta and vname in meta[k]:
            return meta[k][vname]["median"]
        return np.nan

    def zscore_series(vals):
        vals = np.array(vals, dtype=float)
        m = np.isfinite(vals)
        if m.sum()<5: return vals
        mu, sd = np.nanmean(vals[m]), np.nanstd(vals[m]) + 1e-8
        z = (vals - mu)/sd
        z[~np.isfinite(z)] = np.nan
        return z

    # PGSI proxy = z-mean of available veg (>=3)
    # EWBI proxy = signed z-mean of drivers (>=2)
    months_all = []
    pgsi_series, ewbi_series = [], []
    for y,m in month_iter(range(2001,2025)):
        if y==2024 and m>10: break
        k = regkey(y,m)
        months_all.append((y,m))
        vegvals = [monthly_value(k,v) for v in ["GOSIF","GPP","NIRv","NDVI","GNDVI"]]
        drvvals = [monthly_value(k,"SPEI"), monthly_value(k,"TWS"),
                   monthly_value(k,"VPD"), monthly_value(k,"LST")]
        # z per var across time (compute once lazily)
        pgsi_series.append(np.nan)  # placeholder
        ewbi_series.append(np.nan)

    # compute per-variable z across whole time
    per_var_z = {}
    for v in VARIABLES:
        arr = [monthly_value(regkey(y,m), v) for (y,m) in months_all]
        per_var_z[v] = zscore_series(arr)

    for i,(y,m) in enumerate(months_all):
        # PGSI z-mean
        zlist = [per_var_z[v][i] for v in ["GOSIF","GPP","NIRv","NDVI","GNDVI"]]
        ok = [z for z in zlist if np.isfinite(z)]
        pgsi_series[i] = float(np.mean(ok)) if len(ok)>=3 else np.nan
        # EWBI signed z-mean
        zdrv = []
        if np.isfinite(per_var_z["SPEI"][i]): zdrv.append(+per_var_z["SPEI"][i])
        if np.isfinite(per_var_z["TWS"][i]):  zdrv.append(+per_var_z["TWS"][i])
        if np.isfinite(per_var_z["VPD"][i]):  zdrv.append(-per_var_z["VPD"][i])
        if np.isfinite(per_var_z["LST"][i]):  zdrv.append(-per_var_z["LST"][i])
        ewbi_series[i] = float(np.mean(zdrv)) if len(zdrv)>=2 else np.nan

    # Drift by split
    def split_mask(years):
        years=set(years)
        return np.array([y in years for (y,m) in months_all])

    ms_pg = np.array(pgsi_series, dtype=float)
    ms_ew = np.array(ewbi_series, dtype=float)

    def split_stats(name, mask, arr):
        a = arr[mask]
        a = a[np.isfinite(a)]
        if a.size==0:
            print(f"  {name:<5}: n=0")
        else:
            print(f"  {name:<5}: n={a.size:<3d}  mean={a.mean():+.3f}  std={a.std():.3f}  p5={np.percentile(a,5):+.3f}  p95={np.percentile(a,95):+.3f}")

    print("\nTEMPORAL DRIFT (spatial-mean z-scores) — PGSI proxy:")
    split_stats("Train", split_mask(range(2001,2019)), ms_pg)
    split_stats("Val",   split_mask(range(2019,2022)), ms_pg)
    split_stats("Test",  split_mask(range(2022,2025)), ms_pg)

    print("TEMPORAL DRIFT (spatial-mean z-scores) — EWBI proxy:")
    split_stats("Train", split_mask(range(2001,2019)), ms_ew)
    split_stats("Val",   split_mask(range(2019,2022)), ms_ew)
    split_stats("Test",  split_mask(range(2022,2025)), ms_ew)

    # Correlations: drivers vs PGSI/EWBI series (over whole timeline)
    print("\nSIMPLE CORRELATIONS (time-series of spatial means):")
    for v in ["SPEI","TWS","VPD","LST","NIRv","NDVI","GPP","GOSIF","GNDVI"]:
        r_pg = safe_corr(per_var_z[v], ms_pg)
        r_ew = safe_corr(per_var_z[v], ms_ew)
        print(f"  {v:<5}  vs PGSI: {r_pg:+.3f}   vs EWBI: {r_ew:+.3f}")

    # -----------------------
    # Pixel-selection stress test
    # -----------------------
    # Approximate per-pixel monthly coverage feasibility using reference shape stats.
    # We can’t load all rasters here for speed; so we infer from meta whether each timestep
    # would count as EWBI/PGSI-feasible. Then estimate how many pixels would survive
    # for a given min_valid_frac assuming uniformity over space (upper bound).
    months_all_idx = [(y,m) for (y,m) in months_all if not (y==2024 and m>10)]
    ew_months = sum(1 for (y,m) in months_all_idx if ewbi_ready(regkey(y,m)))
    pg_months = sum(1 for (y,m) in months_all_idx if pgsi_ready(regkey(y,m)))

    # Reference grid size upper bound:
    H,W = ref_shape
    grid_pixels = H*W

    print("\nPIXEL-SELECTION STRESS TEST (UPPER BOUNDS):")
    print(f"  Reference grid has {grid_pixels} pixels")
    print(f"  Months with EWBI-feasible drivers: {ew_months}")
    print(f"  Months with PGSI-feasible veg:     {pg_months}")
    for frac in [0.8, 0.7, 0.6, 0.5, 0.4]:
        ew_need = math.ceil(frac*ew_months)
        pg_need = math.ceil(frac*pg_months)
        print(f"   If min_valid_frac={frac:.1f}: need ≥{ew_need} EWBI months and ≥{pg_need} PGSI months per pixel (actual kept pixels ≤ {grid_pixels})")

    # -----------------------
    # Actionable flags
    # -----------------------
    print("\nACTIONABLE FLAGS:")
    # 1) coarse vars
    if any(coarse_flags.values()):
        print("  - Coarse drivers present:", ", ".join([v for v in ["SPEI","TWS"] if coarse_flags.get(v,False)]) or "none")
        print("    Suggest: include them with coarse-aware upscaling (NN + light smooth), do NOT drop.")
    # 2) label feasibility
    if tr_ew < 60:
        print(f"  - Few EWBI-usable train months ({tr_ew}). Consider relaxing EWBI requirement to ≥2 drivers and/or imputing SPEI/TWS per-month trends.")
    if tr_pg < 150:
        print(f"  - Limited PGSI-usable train months ({tr_pg}). Check MODIS/SIF coverage and masks.")
    # 3) drift
    def drift_flag(arr):
        mtr = split_mask(range(2001,2019))
        mte = split_mask(range(2022,2025))
        a = arr[mtr]; b = arr[mte]
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if a.size<10 or b.size<10: return False
        return abs(a.mean()-b.mean())>0.3
    if drift_flag(ms_pg):
        print("  - PGSI proxy shows distribution shift train→test (>0.3 z). Consider monthwise standardization anchored to 2001–2018.")
    if drift_flag(ms_ew):
        print("  - EWBI proxy shows distribution shift train→test (>0.3 z). Recheck normalization/climatology.")
    # 4) value outliers
    for v in VARIABLES:
        # collect p01/p99 across timeline
        p01s, p99s = [], []
        for k in meta:
            if v in meta[k] and np.isfinite(meta[k][v]["p01"]):
                p01s.append(meta[k][v]["p01"]); p99s.append(meta[k][v]["p99"])
        if p01s and p99s and (np.nanmax(p99s) - np.nanmin(p01s) > 1e6):
            print(f"  - {v}: extreme dynamic range detected. Clip or log-transform before z-scoring.")
    print("\nDone.\n")


def main():
    for lt in LAND_TYPES:
        try:
            diagnose_land_type(lt)
        except Exception as e:
            print(f"\n[ERROR] {lt}: {e}")

if __name__ == "__main__":
    main()
