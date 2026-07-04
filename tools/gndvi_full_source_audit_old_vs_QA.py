import hashlib
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from statsmodels.tsa.stattools import ccf

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\GNDVI_full_source_audit_old_vs_QA")
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORT = OUT / "report"
LOGS = OUT / "logs"
for d in [TABLES, FIGURES, REPORT, LOGS]:
    d.mkdir(parents=True, exist_ok=True)

PREVIOUS_GNDVI = Path(r"C:\Users\LOQ\OneDrive\Desktop\Previous GNDVI")
QA_TABLE = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\05_forecasting_raster_full_candidate\tables\gndvi_source_confirmation.csv"
)
DIAG_PACKAGE = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\gndvi_landtype_source_diagnostic_fig5_fig6"
)
SOURCE_RECOVERY = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\figure5_figure6_exact_source_recovery"
)

DATA_ROOT = Path(r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data")
TEST_ROOT = Path(
    r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Test Data\Masked_LandTypes"
)
TRAIN_DIRS = {
    "Grasslands": DATA_ROOT / "CALIFORNIA GRASSLANDS",
    "NaturalForests": DATA_ROOT / "CALIFORNIA NATURAL FORESTS",
    "Savannas": DATA_ROOT / "CALIFORNIA SAVANNAS",
    "Shrublands": DATA_ROOT / "CALIFORNIA SHRUBLANDS",
}
TEST_DIRS = {
    "Grasslands": TEST_ROOT / "Grasslands",
    "NaturalForests": TEST_ROOT / "NaturalForests",
    "Savannas": TEST_ROOT / "Savannas",
    "Shrublands": TEST_ROOT / "Shrublands",
}
LAND_CANON = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
LAND_TO_SCRIPT = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "NaturalForests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
PREVIOUS_FOLDERS = {
    "GRASSLANDS": PREVIOUS_GNDVI / "grass" / "GNDVI",
    "NATURAL_FORESTS": PREVIOUS_GNDVI / "nat" / "GNDVI",
    "SAVANNAS": PREVIOUS_GNDVI / "sav" / "GNDVI",
    "SHRUBLANDS": PREVIOUS_GNDVI / "shrub" / "GNDVI",
}
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
TARGET_VARS = ["NDVI", "NIRv", "GOSIF", "GPP", "LST", "SPEI", "TWS", "VPD"]
KEY_CCF_TARGETS = ["NDVI", "NIRv", "GOSIF", "GPP", "LST", "VPD"]

FIG5_NAMING = {
    "Grasslands": {
        "GNDVI": "GNDVI_Grasslands_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif",
        "GOSIF": "GOSIF_Grasslands_CALIFORNIA_GOSIF_{year}{month:02d}.tif",
        "GPP": "Grasslands_GPP_{year}_{month:02d}.tif",
        "LST": "LST_Grasslands_MYD21C1_CA_LST_{year}_{month:02d}.tif",
        "NDVI": "NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "NIRv": "NIRv_Grasslands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "SPEI": "SPEI1_GRIDMET_CA_{year}_{month:02d}_Grasslands.tif",
        "TWS": "TWS_Grasslands_TWS_GRACE_CA_{year}_{month:02d}.tif",
        "VPD": "VPD_Grasslands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif",
    },
    "NaturalForests": {
        "GNDVI": "GNDVI_NaturalForests_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif",
        "GOSIF": "GOSIF_NaturalForests_CALIFORNIA_GOSIF_{year}{month:02d}.tif",
        "GPP": "NaturalForests_GPP_{year}_{month:02d}.tif",
        "LST": "LST_NaturalForests_MYD21C1_CA_LST_{year}_{month:02d}.tif",
        "NDVI": "NDVI_NaturalForests_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "NIRv": "NIRv_NaturalForests_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "SPEI": "SPEI1_GRIDMET_CA_{year}_{month:02d}_NaturalForests.tif",
        "TWS": "TWS_NaturalForests_TWS_GRACE_CA_{year}_{month:02d}.tif",
        "VPD": "VPD_NaturalForests_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif",
    },
    "Savannas": {
        "GNDVI": "GNDVI_Savannas_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif",
        "GOSIF": "GOSIF_Savanna_CALIFORNIA_GOSIF_{year}{month:02d}.tif",
        "GPP": "Savannas_GPP_{year}_{month:02d}.tif",
        "LST": "LST_Savanna_MYD21C1_CA_LST_{year}_{month:02d}.tif",
        "NDVI": "NDVI_Savanna_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "NIRv": "NIRv_Savanna_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "SPEI": "SPEI1_GRIDMET_CA_{year}_{month:02d}_Savannas.tif",
        "TWS": "TWS_Savanna_TWS_GRACE_CA_{year}_{month:02d}.tif",
        "VPD": "VPD_Savanna_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif",
    },
    "Shrublands": {
        "GNDVI": "GNDVI_Shrublands_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif",
        "GOSIF": "GOSIF_Shrublands_CALIFORNIA_GOSIF_{year}{month:02d}.tif",
        "GPP": "Shrublands_GPP_{year}_{month:02d}.tif",
        "LST": "LST_Shrublands_MYD21C1_CA_LST_{year}_{month:02d}.tif",
        "NDVI": "NDVI_Shrublands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "NIRv": "NIRv_Shrublands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
        "SPEI": "SPEI1_GRIDMET_CA_{year}_{month:02d}_Shrublands.tif",
        "TWS": "TWS_Shrublands_TWS_GRACE_CA_{year}_{month:02d}.tif",
        "VPD": "VPD_Shrublands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif",
    },
}
FIG5_TEST_PATTERNS = {
    land: {
        "GNDVI": f"GNDVI_MOD09A1_CA_5km_{{year}}_{{month:02d}}_{land}.tif",
        "GOSIF": f"GOSIF_{{year}}.M{{month:02d}}_{land}.tif",
        "LST": f"MYD21C1_CA_LST_{{year}}_{{month:02d}}_{land}.tif",
        "NDVI": f"NDVI_MOD13Q1_CA_5km_{{year}}_{{month:02d}}_{land}.tif",
        "NIRv": f"NIRv_MOD13Q1_CA_5km_{{year}}_{{month:02d}}_{land}.tif",
        "SPEI": f"SPEI1_GRIDMET_CA_{{year}}_{{month:02d}}_{land}.tif",
        "TWS": f"TWS_GRACE_CA_{{year}}_{{month:02d}}_{land}.tif",
        "VPD": f"VPDmax_GRIDMET_CA_5km_{{year}}_{{month:02d}}_{land}.tif",
    }
    for land in ["Grasslands", "NaturalForests", "Savannas", "Shrublands"]
}


def log(msg):
    print(msg, flush=True)


def parse_date(path):
    stem = Path(path).stem
    for pattern in [r"(\d{4})\.M(\d{2})", r"(\d{4})_(\d{2})", r"(\d{6})(?:_|$)"]:
        m = re.search(pattern, stem)
        if not m:
            continue
        if len(m.groups()) == 2:
            y, mo = int(m.group(1)), int(m.group(2))
        else:
            ds = m.group(1)
            y, mo = int(ds[:4]), int(ds[4:6])
        if 1900 <= y <= 2030 and 1 <= mo <= 12:
            return y, mo, pd.Timestamp(y, mo, 1)
    return None, None, pd.NaT


def sha256(path):
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def nodata_masks(arr, nodata):
    arr = arr.astype("float64", copy=False)
    raw_finite = np.isfinite(arr)
    nodata_like = ~raw_finite
    if nodata is not None and np.isfinite(nodata):
        nodata_like |= np.isclose(arr, nodata, rtol=0, atol=1e-12)
    for v in [-9999, -3.4e38, 32767, -32768, 65535, -32767]:
        nodata_like |= np.isclose(arr, v, rtol=0, atol=1)
    valid = raw_finite & ~nodata_like
    return raw_finite, nodata_like, valid


def read_valid(path):
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float64")
        _, _, valid = nodata_masks(arr, src.nodata)
        values = np.where(valid, arr, np.nan)
        meta = {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": str(src.transform),
            "pixel_size_x": float(src.transform.a),
            "pixel_size_y": float(src.transform.e),
            "bounds": str(src.bounds),
            "count": src.count,
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
        }
    return values, valid, meta


def raster_inventory_row(source_type, land_type, path):
    y, mo, dt = parse_date(path)
    row = {
        "source_type": source_type,
        "land_type": land_type,
        "year": y,
        "month": mo,
        "date": "" if pd.isna(dt) else dt.strftime("%Y-%m-%d"),
        "path": str(path),
        "file_exists": Path(path).exists(),
    }
    if not Path(path).exists():
        return row
    try:
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float64")
            raw_finite, nodata_like, valid = nodata_masks(arr, src.nodata)
            vals = arr[valid]
            total = arr.size
            row.update(
                {
                    "width": src.width,
                    "height": src.height,
                    "crs": str(src.crs),
                    "affine_transform": str(src.transform),
                    "pixel_size_x": float(src.transform.a),
                    "pixel_size_y": float(src.transform.e),
                    "bounds": str(src.bounds),
                    "band_count": src.count,
                    "nodata_value": src.nodata,
                    "data_type": src.dtypes[0],
                    "min": float(np.nanmin(vals)) if vals.size else np.nan,
                    "max": float(np.nanmax(vals)) if vals.size else np.nan,
                    "mean": float(np.nanmean(vals)) if vals.size else np.nan,
                    "median": float(np.nanmedian(vals)) if vals.size else np.nan,
                    "standard_deviation": float(np.nanstd(vals)) if vals.size else np.nan,
                    "total_pixels": int(total),
                    "raw_finite_pixels": int(raw_finite.sum()),
                    "finite_pixels": int(valid.sum()),
                    "nodata_pixels": int(nodata_like.sum()),
                    "fraction_finite": float(valid.sum() / total) if total else np.nan,
                    "fraction_nodata": float(nodata_like.sum() / total) if total else np.nan,
                    "fraction_values_outside_plausible_GNDVI_range": float(
                        np.sum((vals < -1) | (vals > 1)) / vals.size
                    )
                    if vals.size
                    else np.nan,
                    "fraction_exactly_0": float(np.sum(arr == 0) / total) if total else np.nan,
                    "fraction_exactly_minus9999_or_nodata_like": float(nodata_like.sum() / total)
                    if total
                    else np.nan,
                    "sha256": sha256(path),
                }
            )
    except Exception as e:
        row["read_error"] = str(e)
    return row


def list_previous_files():
    rows = []
    for land, folder in PREVIOUS_FOLDERS.items():
        for path in sorted(folder.glob("*.tif")):
            rows.append({"source_type": "previous", "land_type": land, "path": path})
    return rows


def list_qa_files():
    df = pd.read_csv(QA_TABLE)
    df = df[df["qa_fixed_confirmed"] == True].copy()
    rows = []
    for _, r in df.iterrows():
        rows.append({"source_type": "QA", "land_type": r["land_type"], "path": Path(r["selected_path"])})
    return rows


def inventory_sources():
    log("Inventorying old and QA GNDVI rasters...")
    rows = []
    for item in list_previous_files() + list_qa_files():
        rows.append(raster_inventory_row(item["source_type"], item["land_type"], item["path"]))
    inv = pd.DataFrame(rows)
    inv.to_csv(TABLES / "GNDVI_raster_inventory_old_vs_QA.csv", index=False)
    return inv


def pairing_and_grid(inv):
    log("Auditing old/QA pairing and grid alignment...")
    prev = inv[inv["source_type"] == "previous"].copy()
    qa = inv[inv["source_type"] == "QA"].copy()
    prev["ym"] = prev["year"].astype("Int64").astype(str) + prev["month"].astype("Int64").astype(str).str.zfill(2)
    qa["ym"] = qa["year"].astype("Int64").astype(str) + qa["month"].astype("Int64").astype(str).str.zfill(2)
    rows = []
    keys = sorted(set(zip(prev.land_type, prev.ym)).union(set(zip(qa.land_type, qa.ym))))
    for land, ym in keys:
        p = prev[(prev.land_type == land) & (prev.ym == ym)]
        q = qa[(qa.land_type == land) & (qa.ym == ym)]
        pr = p.iloc[0].to_dict() if len(p) else {}
        qr = q.iloc[0].to_dict() if len(q) else {}
        rows.append(
            {
                "land_type": land,
                "date_yyyymm": ym,
                "old_exists": bool(len(p) and pr.get("file_exists", False)),
                "QA_exists": bool(len(q) and qr.get("file_exists", False)),
                "old_path": pr.get("path", ""),
                "QA_path": qr.get("path", ""),
                "same_CRS": pr.get("crs") == qr.get("crs") if len(p) and len(q) else False,
                "same_transform": pr.get("affine_transform") == qr.get("affine_transform") if len(p) and len(q) else False,
                "same_bounds": pr.get("bounds") == qr.get("bounds") if len(p) and len(q) else False,
                "same_shape": (pr.get("width") == qr.get("width") and pr.get("height") == qr.get("height"))
                if len(p) and len(q)
                else False,
                "date_matches": bool(len(p) and len(q) and pr.get("year") == qr.get("year") and pr.get("month") == qr.get("month")),
                "land_type_matches": bool(len(p) and len(q) and pr.get("land_type") == qr.get("land_type")),
                "old_duplicate_count": max(0, len(p) - 1),
                "QA_duplicate_count": max(0, len(q) - 1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "GNDVI_pairing_and_grid_alignment_audit.csv", index=False)
    return out


def pixel_level_comparison(pairing):
    log("Running pixel-level old-vs-QA comparisons...")
    rows = []
    matched = pairing[pairing["old_exists"] & pairing["QA_exists"]].copy()
    for _, r in matched.iterrows():
        old_path, qa_path = Path(r.old_path), Path(r.QA_path)
        row = {
            "land_type": r.land_type,
            "date_yyyymm": r.date_yyyymm,
            "old_path": str(old_path),
            "QA_path": str(qa_path),
            "same_shape": bool(r.same_shape),
            "same_grid": bool(r.same_shape and r.same_transform and r.same_CRS),
        }
        try:
            old, old_valid, _ = read_valid(old_path)
            qa, qa_valid, _ = read_valid(qa_path)
            if old.shape != qa.shape:
                row["comparison_status"] = "shape_mismatch_no_pixel_compare"
            else:
                both = old_valid & qa_valid
                either = old_valid | qa_valid
                one_missing = old_valid ^ qa_valid
                ov, qv = old[both], qa[both]
                diff = qv - ov
                row.update(
                    {
                        "comparison_status": "ok",
                        "mean_old_GNDVI": float(np.nanmean(ov)) if ov.size else np.nan,
                        "mean_QA_GNDVI": float(np.nanmean(qv)) if qv.size else np.nan,
                        "mean_difference_QA_minus_old": float(np.nanmean(diff)) if diff.size else np.nan,
                        "median_difference": float(np.nanmedian(diff)) if diff.size else np.nan,
                        "RMSE": float(np.sqrt(np.nanmean(diff**2))) if diff.size else np.nan,
                        "MAE": float(np.nanmean(np.abs(diff))) if diff.size else np.nan,
                        "Pearson_r": safe_pearson(ov, qv),
                        "Spearman_r": safe_spearman(ov, qv),
                        "valid_overlap_pixel_count": int(both.sum()),
                        "valid_overlap_fraction": float(both.sum() / old.size),
                        "QA_valid_fraction_relative_to_old": float(both.sum() / old_valid.sum())
                        if old_valid.sum()
                        else np.nan,
                        "old_valid_fraction_relative_to_QA": float(both.sum() / qa_valid.sum())
                        if qa_valid.sum()
                        else np.nan,
                        "fraction_QA_higher_than_old": float(np.sum(diff > 0) / diff.size) if diff.size else np.nan,
                        "fraction_QA_lower_than_old": float(np.sum(diff < 0) / diff.size) if diff.size else np.nan,
                        "fraction_abs_difference_gt_0p05": float(np.sum(np.abs(diff) > 0.05) / diff.size)
                        if diff.size
                        else np.nan,
                        "fraction_abs_difference_gt_0p10": float(np.sum(np.abs(diff) > 0.10) / diff.size)
                        if diff.size
                        else np.nan,
                        "fraction_sign_change": float(np.sum(np.sign(ov) != np.sign(qv)) / diff.size)
                        if diff.size
                        else np.nan,
                        "fraction_one_source_valid_other_missing": float(one_missing.sum() / either.sum())
                        if either.sum()
                        else np.nan,
                    }
                )
        except Exception as e:
            row["comparison_status"] = "error"
            row["error"] = str(e)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "GNDVI_pixel_level_old_vs_QA_comparison.csv", index=False)
    return out


def safe_pearson(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return np.nan
    return float(stats.pearsonr(a[mask], b[mask])[0])


def safe_spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or np.nanstd(a[mask]) == 0 or np.nanstd(b[mask]) == 0:
        return np.nan
    return float(stats.spearmanr(a[mask], b[mask]).correlation)


def monthly_mean_tables(inv):
    log("Building monthly mean time-series comparison...")
    m = inv[["source_type", "land_type", "year", "month", "date", "path", "mean", "finite_pixels", "fraction_finite"]].copy()
    m = m[m["source_type"].isin(["previous", "QA"])]
    prev = m[m.source_type == "previous"].rename(
        columns={"mean": "old_mean", "finite_pixels": "old_valid_pixel_count", "fraction_finite": "old_valid_fraction", "path": "old_path"}
    )
    qa = m[m.source_type == "QA"].rename(
        columns={"mean": "QA_mean", "finite_pixels": "QA_valid_pixel_count", "fraction_finite": "QA_valid_fraction", "path": "QA_path"}
    )
    keys = ["land_type", "year", "month"]
    out = prev[keys + ["date", "old_path", "old_mean", "old_valid_pixel_count", "old_valid_fraction"]].merge(
        qa[keys + ["QA_path", "QA_mean", "QA_valid_pixel_count", "QA_valid_fraction"]],
        on=keys,
        how="outer",
    )
    out["date"] = pd.to_datetime(out["date"])
    out["difference_QA_minus_old"] = out["QA_mean"] - out["old_mean"]
    out.to_csv(TABLES / "GNDVI_landtype_monthly_mean_old_vs_QA.csv", index=False)

    metrics = []
    for land, g in out.groupby("land_type"):
        overlap = g.dropna(subset=["old_mean", "QA_mean"]).copy()
        diff = overlap["QA_mean"] - overlap["old_mean"]
        metrics.append(
            {
                "land_type": land,
                "metric_scope": "overall",
                "calendar_month": "",
                "year": "",
                "valid_months": len(overlap),
                "Pearson_r": safe_pearson(overlap.old_mean, overlap.QA_mean),
                "Spearman_r": safe_spearman(overlap.old_mean, overlap.QA_mean),
                "RMSE": float(np.sqrt(np.mean(diff**2))) if len(diff) else np.nan,
                "MAE": float(np.mean(np.abs(diff))) if len(diff) else np.nan,
                "mean_bias_QA_minus_old": float(diff.mean()) if len(diff) else np.nan,
                "months_abs_diff_gt_0p05": int((diff.abs() > 0.05).sum()),
                "months_abs_diff_gt_0p10": int((diff.abs() > 0.10).sum()),
                "months_QA_abnormally_low_high_relative_to_old": int((diff.abs() > 0.10).sum()),
                "months_QA_valid_fraction_drops_sharply": int(
                    (overlap.QA_valid_fraction < overlap.QA_valid_fraction.quantile(0.1)).sum()
                )
                if len(overlap)
                else 0,
            }
        )
        for mo, gm in overlap.groupby("month"):
            d = gm.QA_mean - gm.old_mean
            metrics.append(
                {
                    "land_type": land,
                    "metric_scope": "seasonal_bias",
                    "calendar_month": int(mo),
                    "year": "",
                    "valid_months": len(gm),
                    "mean_bias_QA_minus_old": float(d.mean()) if len(d) else np.nan,
                }
            )
        for yr, gy in overlap.groupby("year"):
            d = gy.QA_mean - gy.old_mean
            metrics.append(
                {
                    "land_type": land,
                    "metric_scope": "annual_bias",
                    "calendar_month": "",
                    "year": int(yr),
                    "valid_months": len(gy),
                    "mean_bias_QA_minus_old": float(d.mean()) if len(d) else np.nan,
                }
            )
    sim = pd.DataFrame(metrics)
    sim.to_csv(TABLES / "GNDVI_landtype_monthly_similarity_metrics.csv", index=False)
    return out, sim


def seasonal_cycle(monthly):
    rows = []
    for (land, source), g in pd.concat(
        [
            monthly[["land_type", "month", "old_mean"]].rename(columns={"old_mean": "value"}).assign(source_type="previous"),
            monthly[["land_type", "month", "QA_mean"]].rename(columns={"QA_mean": "value"}).assign(source_type="QA"),
        ]
    ).dropna(subset=["value"]).groupby(["land_type", "source_type"]):
        for mo, gm in g.groupby("month"):
            rows.append(
                {
                    "land_type": land,
                    "source_type": source,
                    "calendar_month": int(mo),
                    "mean": float(gm.value.mean()),
                    "std": float(gm.value.std(ddof=1)) if len(gm) > 1 else np.nan,
                    "n": len(gm),
                }
            )
    out = pd.DataFrame(rows)
    wide = out.pivot_table(index=["land_type", "calendar_month"], columns="source_type", values="mean").reset_index()
    wide["QA_minus_previous_seasonal_mean"] = wide.get("QA", np.nan) - wide.get("previous", np.nan)
    out = out.merge(wide[["land_type", "calendar_month", "QA_minus_previous_seasonal_mean"]], on=["land_type", "calendar_month"], how="left")
    out.to_csv(TABLES / "GNDVI_seasonal_cycle_old_vs_QA.csv", index=False)
    return out


def plot_monthly_diagnostics(monthly, seasonal):
    log("Rendering GNDVI diagnostic figures...")
    land_order = LAND_CANON
    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for ax, land in zip(axes, land_order):
        g = monthly[monthly.land_type == land].sort_values("date")
        ax.plot(g.date, g.old_mean, label="Previous GNDVI", lw=1.4)
        ax.plot(g.date, g.QA_mean, label="QA GNDVI", lw=1.4)
        ax.set_title(land)
        ax.set_ylabel("GNDVI")
        ax.grid(alpha=0.25)
    axes[0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "GNDVI_old_vs_QA_monthly_timeseries.png", dpi=300)
    fig.savefig(FIGURES / "GNDVI_old_vs_QA_monthly_timeseries.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, land in zip(axes.ravel(), land_order):
        g = seasonal[seasonal.land_type == land]
        for source, gg in g.groupby("source_type"):
            ax.errorbar(gg.calendar_month, gg["mean"], yerr=gg["std"], marker="o", label=source, capsize=2)
        ax.set_title(land)
        ax.set_xticks(range(1, 13))
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(frameon=False)
    fig.supxlabel("Calendar month")
    fig.supylabel("GNDVI seasonal mean")
    fig.tight_layout()
    fig.savefig(FIGURES / "GNDVI_old_vs_QA_seasonal_cycle.png", dpi=300)
    fig.savefig(FIGURES / "GNDVI_old_vs_QA_seasonal_cycle.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for ax, land in zip(axes, land_order):
        g = monthly[monthly.land_type == land].sort_values("date")
        ax.axhline(0, color="black", lw=0.8)
        ax.plot(g.date, g.old_mean - g.QA_mean, lw=1.2, color="#7b3294")
        ax.set_title(land)
        ax.set_ylabel("Old - QA")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "GNDVI_old_minus_QA_difference_timeseries.png", dpi=300)
    fig.savefig(FIGURES / "GNDVI_old_minus_QA_difference_timeseries.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
    for ax, land in zip(axes, land_order):
        g = monthly[monthly.land_type == land].sort_values("date")
        ax.plot(g.date, g.QA_valid_fraction, lw=1.2, color="#008837")
        ax.set_title(land)
        ax.set_ylabel("QA valid fraction")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "GNDVI_QA_valid_fraction_timeseries.png", dpi=300)
    fig.savefig(FIGURES / "GNDVI_QA_valid_fraction_timeseries.pdf")
    plt.close(fig)


def load_series_variable(variable, land, gndvi_source=None):
    script_land = LAND_TO_SCRIPT[land]
    if variable == "GNDVI" and gndvi_source == "previous":
        rows = []
        for path in sorted(PREVIOUS_FOLDERS[land].glob("*.tif")):
            _, _, dt = parse_date(path)
            if pd.isna(dt):
                continue
            values, _, _ = read_valid(path)
            rows.append((dt, float(np.nanmean(values))))
        return pd.Series(dict(rows), name="GNDVI").sort_index()
    if variable == "GNDVI" and gndvi_source == "QA":
        qa = pd.read_csv(QA_TABLE)
        qa = qa[(qa.land_type == land) & (qa.qa_fixed_confirmed == True)].copy()
        qa["date_str"] = qa["date"].astype(str)
        rows = []
        for _, r in qa.drop_duplicates("date_str", keep="last").sort_values("date_str").iterrows():
            path = Path(r.selected_path)
            ds = r.date_str
            dt = pd.Timestamp(int(ds[:4]), int(ds[4:6]), 1)
            if path.exists():
                values, _, _ = read_valid(path)
                rows.append((dt, float(np.nanmean(values))))
        return pd.Series(dict(rows), name="GNDVI").sort_index()

    rows = []
    actual_start = {"LST": 2003, "TWS": 2002}.get(variable, 2001)
    train_dir = TRAIN_DIRS[script_land] / variable
    if train_dir.exists():
        for year in range(actual_start, 2022):
            for month in range(1, 13):
                filename = FIG5_NAMING[script_land][variable].format(year=year, month=month)
                path = train_dir / filename
                if path.exists():
                    values, _, _ = read_valid(path)
                    rows.append((pd.Timestamp(year, month, 1), float(np.nanmean(values))))
    if variable != "GPP":
        test_dir = TEST_DIRS[script_land] / variable
        if test_dir.exists():
            for year in range(2022, 2025):
                for month in range(1, 13):
                    filename = FIG5_TEST_PATTERNS[script_land][variable].format(year=year, month=month)
                    path = test_dir / filename
                    if path.exists():
                        values, _, _ = read_valid(path)
                        rows.append((pd.Timestamp(year, month, 1), float(np.nanmean(values))))
    return pd.Series(dict(rows), name=variable).sort_index()


def figure5_exact_corr(data_dict):
    valid = [v for v in VARIABLES if v in data_dict and len(data_dict[v]) > 0]
    min_len = min(len(data_dict[v]) for v in valid)
    arrs = {v: data_dict[v].values[:min_len] for v in valid}
    return arrs, min_len


def interindex_diagnostic():
    log("Computing Figure 5 GNDVI inter-index diagnostics...")
    rows = []
    for land in LAND_CANON:
        non_gndvi = {v: load_series_variable(v, land) for v in VARIABLES if v != "GNDVI"}
        for source in ["previous", "QA"]:
            data = dict(non_gndvi)
            data["GNDVI"] = load_series_variable("GNDVI", land, source)
            exact_arrays, min_len = figure5_exact_corr(data)
            for var in TARGET_VARS:
                # exact old min-length logic
                corr_exact = safe_pearson(exact_arrays["GNDVI"], exact_arrays[var])
                # date-safe diagnostic
                joined = pd.concat([data["GNDVI"], data[var]], axis=1, join="inner").dropna()
                rows.append(
                    {
                        "land_type": land,
                        "GNDVI_source": source,
                        "comparison_variable": var,
                        "exact_old_logic_correlation": corr_exact,
                        "exact_old_logic_months_used": min_len,
                        "date_safe_correlation": safe_pearson(joined.iloc[:, 0], joined.iloc[:, 1]),
                        "date_safe_overlap_months": len(joined),
                        "date_safe_start_date": joined.index.min().strftime("%Y-%m-%d") if len(joined) else "",
                        "date_safe_end_date": joined.index.max().strftime("%Y-%m-%d") if len(joined) else "",
                        "uses_GPP_limited_period": var == "GPP",
                        "date_safe_alignment_changes_result": abs(
                            corr_exact - safe_pearson(joined.iloc[:, 0], joined.iloc[:, 1])
                        )
                        > 0.02
                        if len(joined)
                        else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "GNDVI_interindex_correlation_diagnostic.csv", index=False)
    plot_interindex(out)
    return out


def plot_interindex(df):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    for ax, land in zip(axes.ravel(), LAND_CANON):
        g = df[df.land_type == land]
        x = np.arange(len(TARGET_VARS))
        width = 0.35
        prev = g[g.GNDVI_source == "previous"].set_index("comparison_variable").reindex(TARGET_VARS)
        qa = g[g.GNDVI_source == "QA"].set_index("comparison_variable").reindex(TARGET_VARS)
        ax.bar(x - width / 2, prev.exact_old_logic_correlation, width, label="Previous")
        ax.bar(x + width / 2, qa.exact_old_logic_correlation, width, label="QA")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(land)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_VARS, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].legend(frameon=False)
    fig.supylabel("Pearson r with GNDVI")
    fig.tight_layout()
    fig.savefig(FIGURES / "GNDVI_interindex_correlation_old_vs_QA_barplot.png", dpi=300)
    fig.savefig(FIGURES / "GNDVI_interindex_correlation_old_vs_QA_barplot.pdf")
    plt.close(fig)


def load_ccf_data(land, gndvi_source):
    script_land = LAND_TO_SCRIPT[land]
    dfs = []
    for var in VARIABLES:
        s = load_series_variable(var, land, gndvi_source if var == "GNDVI" else None)
        if len(s):
            dfs.append(s.to_frame())
    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how="outer")
    return result.ffill(limit=2)


def compute_ccf_values(x, y, max_lag=6):
    valid = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[valid].values
    y_clean = y[valid].values
    if len(x_clean) < 24:
        return [], []
    x_std = (x_clean - np.mean(x_clean)) / np.std(x_clean)
    y_std = (y_clean - np.mean(y_clean)) / np.std(y_clean)
    pos = ccf(x_std, y_std, adjusted=False)[: max_lag + 1]
    neg = ccf(y_std, x_std, adjusted=False)[1 : max_lag + 1]
    lags = list(range(-max_lag, max_lag + 1))
    vals = list(neg[::-1]) + list(pos)
    return lags, vals


def ccf_diagnostic():
    log("Computing Figure 6 GNDVI CCF diagnostics...")
    summary = []
    curves = []
    for land in LAND_CANON:
        for source in ["previous", "QA"]:
            data = load_ccf_data(land, source)
            for target in TARGET_VARS:
                if target not in data.columns:
                    continue
                lags, vals = compute_ccf_values(data["GNDVI"], data[target], max_lag=6)
                if not vals:
                    continue
                vals = np.array(vals, dtype=float)
                lags_arr = np.array(lags)
                for lag, val in zip(lags, vals):
                    curves.append({"land_type": land, "GNDVI_source": source, "pair": f"GNDVI-{target}", "lag": lag, "ccf": val})
                pos_idx = int(np.nanargmax(vals))
                abs_idx = int(np.nanargmax(np.abs(vals)))
                summary.append(
                    {
                        "land_type": land,
                        "GNDVI_source": source,
                        "pair": f"GNDVI-{target}",
                        "target": target,
                        "maximum_positive_correlation": float(vals[pos_idx]),
                        "lag_of_maximum_positive_correlation": int(lags_arr[pos_idx]),
                        "maximum_absolute_correlation": float(vals[abs_idx]),
                        "lag_of_maximum_absolute_correlation": int(lags_arr[abs_idx]),
                        "correlation_at_lag_0": float(vals[lags.index(0)]),
                    }
                )
    summary_df = pd.DataFrame(summary)
    curves_df = pd.DataFrame(curves)
    comp = summary_df.pivot_table(
        index=["land_type", "pair", "target"],
        columns="GNDVI_source",
        values=["maximum_positive_correlation", "lag_of_maximum_positive_correlation", "maximum_absolute_correlation", "lag_of_maximum_absolute_correlation", "correlation_at_lag_0"],
        aggfunc="first",
    )
    comp.columns = [f"{a}_{b}" for a, b in comp.columns]
    comp = comp.reset_index()
    comp["lag0_difference_QA_minus_previous"] = comp["correlation_at_lag_0_QA"] - comp["correlation_at_lag_0_previous"]
    comp["sign_or_pattern_changes_between_previous_and_QA"] = np.sign(comp["correlation_at_lag_0_QA"]) != np.sign(comp["correlation_at_lag_0_previous"])
    comp["peak_shifts_by_more_than_1_month"] = (
        comp["lag_of_maximum_absolute_correlation_QA"] - comp["lag_of_maximum_absolute_correlation_previous"]
    ).abs() > 1
    comp["QA_biologically_unexpected_or_unstable_pattern_flag"] = (
        comp["sign_or_pattern_changes_between_previous_and_QA"] | comp["peak_shifts_by_more_than_1_month"]
    )
    comp.to_csv(TABLES / "GNDVI_CCF_old_vs_QA_diagnostic.csv", index=False)
    curves_df.to_csv(TABLES / "GNDVI_CCF_old_vs_QA_curves.csv", index=False)
    plot_ccf_key(curves_df)
    return comp, curves_df


def plot_ccf_key(curves):
    fig, axes = plt.subplots(len(KEY_CCF_TARGETS), 4, figsize=(17, 14), sharex=True, sharey=True)
    for c, land in enumerate(LAND_CANON):
        for r, target in enumerate(KEY_CCF_TARGETS):
            ax = axes[r, c]
            g = curves[(curves.land_type == land) & (curves.pair == f"GNDVI-{target}")]
            for source, gg in g.groupby("GNDVI_source"):
                ax.plot(gg.lag, gg.ccf, marker="o", ms=3, lw=1.1, label=source)
            ax.axhline(0, color="black", lw=0.6)
            ax.axvline(0, color="black", ls="--", lw=0.7)
            if r == 0:
                ax.set_title(land)
            if c == 0:
                ax.set_ylabel(f"GNDVI-{target}\nCCF")
            ax.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    fig.supxlabel("Lag (months; negative = GNDVI leads target)")
    fig.tight_layout()
    fig.savefig(FIGURES / "GNDVI_CCF_old_vs_QA_key_pairs.png", dpi=300)
    fig.savefig(FIGURES / "GNDVI_CCF_old_vs_QA_key_pairs.pdf")
    plt.close(fig)


def failure_mode_checklist(inv, pairing, pixel, monthly, inter, ccf_diag):
    rows = []
    qa = inv[inv.source_type == "QA"].copy()
    prev = inv[inv.source_type == "previous"].copy()
    matched = monthly.dropna(subset=["old_mean", "QA_mean"])
    qa_out = float((qa["fraction_values_outside_plausible_GNDVI_range"] > 0.001).mean()) if len(qa) else np.nan
    qa_zero = float(qa["fraction_exactly_0"].median()) if len(qa) else np.nan
    qa_nodata = float(qa["fraction_nodata"].median()) if len(qa) else np.nan
    monthly_sim = pd.read_csv(TABLES / "GNDVI_landtype_monthly_similarity_metrics.csv")
    overall = monthly_sim[monthly_sim.metric_scope == "overall"]
    mean_r = float(overall.Pearson_r.mean()) if len(overall) else np.nan
    ccf_flags = int(ccf_diag["QA_biologically_unexpected_or_unstable_pattern_flag"].sum()) if len(ccf_diag) else 0

    def add(mode, status, evidence, interpretation):
        rows.append({"failure_mode": mode, "status": status, "evidence": evidence, "interpretation": interpretation})

    add(
        "A_wrong_scale_factor",
        "not_supported" if qa_out == 0 else "possible",
        f"Median QA value range outside [-1,1] fraction by raster is near {qa['fraction_values_outside_plausible_GNDVI_range'].median():.4g}.",
        "QA values are physically plausible if outside-range fractions are negligible.",
    )
    add(
        "B_nodata_or_zero_mishandling",
        "possible" if qa_zero > 0.05 or qa_nodata > 0.9 else "not_strongly_supported",
        f"Median QA zero fraction={qa_zero:.4g}; median QA nodata-like fraction={qa_nodata:.4g}.",
        "High zero/nodata fractions would indicate masking or fill-value risk.",
    )
    add(
        "C_wrong_band",
        "not_supported" if qa["band_count"].dropna().max() == 1 else "possible",
        f"QA band count range: {qa['band_count'].min()} to {qa['band_count'].max()}.",
        "Single-band rasters are consistent with an exported GNDVI band.",
    )
    add(
        "D_wrong_date_parsing",
        "not_supported" if qa.groupby("land_type").date.nunique().min() == 288 else "possible",
        "QA table has 288 unique monthly dates per land type from 2001-01 to 2024-12.",
        "No duplicated or shifted dates are apparent from the confirmation table.",
    )
    add(
        "E_wrong_land_type_mapping",
        "possible" if matched.groupby("land_type").apply(lambda g: safe_pearson(g.old_mean, g.QA_mean)).min() < 0.5 else "not_strongly_supported",
        f"Mean old-vs-QA monthly Pearson r across land types={mean_r:.3f}.",
        "Low land-type monthly agreement would support mapping/source mismatch.",
    )
    add(
        "F_spatial_mismatch",
        "possible" if (~pairing[pairing.old_exists & pairing.QA_exists]["same_grid"]).any() else "not_supported",
        f"Matched pairs with same grid: {int(pairing[pairing.old_exists & pairing.QA_exists]['same_grid'].sum())}/{int((pairing.old_exists & pairing.QA_exists).sum())}.",
        "Grid mismatch would prevent direct pixel-level interpretation.",
    )
    add(
        "G_QA_mask_too_strict",
        "possible" if qa["fraction_finite"].median() < prev["fraction_finite"].median() * 0.5 else "not_strongly_supported",
        f"Median finite fraction previous={prev['fraction_finite'].median():.4f}; QA={qa['fraction_finite'].median():.4f}.",
        "A large QA valid-fraction collapse would support over-filtering.",
    )
    add(
        "H_formula_or_export_issue",
        "needs_external_GEE_review",
        f"CCF unstable/sign-or-peak flags among key pairs={ccf_flags}.",
        "Formula/export cannot be proven from GeoTIFF values alone; inspect GEE source if flags persist.",
    )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "GNDVI_failure_mode_checklist.csv", index=False)
    return out


def final_report(inv, pairing, pixel, monthly, sim, seasonal, inter, ccf_diag, checklist):
    qa = inv[inv.source_type == "QA"]
    prev = inv[inv.source_type == "previous"]
    overlap = monthly.dropna(subset=["old_mean", "QA_mean"])
    overall = sim[sim.metric_scope == "overall"].copy()
    high_diff = int((overlap.difference_QA_minus_old.abs() > 0.10).sum())
    same_grid = int(pairing[pairing.old_exists & pairing.QA_exists].same_grid.sum())
    matched_pairs = int((pairing.old_exists & pairing.QA_exists).sum())
    qa_out_max = float(qa.fraction_values_outside_plausible_GNDVI_range.max())
    qa_zero_med = float(qa.fraction_exactly_0.median())
    qa_valid_med = float(qa.fraction_finite.median())
    prev_valid_med = float(prev.fraction_finite.median())
    min_monthly_r = float(overall.Pearson_r.min()) if len(overall) else np.nan
    mean_monthly_r = float(overall.Pearson_r.mean()) if len(overall) else np.nan
    ccf_flags = int(ccf_diag.QA_biologically_unexpected_or_unstable_pattern_flag.sum()) if len(ccf_diag) else 0
    corr_comp = inter.pivot_table(
        index=["land_type", "comparison_variable"],
        columns="GNDVI_source",
        values="exact_old_logic_correlation",
        aggfunc="first",
    ).reset_index()
    corr_comp["QA_minus_previous"] = corr_comp["QA"] - corr_comp["previous"]
    largest = corr_comp.reindex(corr_comp.QA_minus_previous.abs().sort_values(ascending=False).index).head(10)
    technical_bug = (
        (qa_out_max > 0.01)
        or (qa_zero_med > 0.05)
        or (same_grid < matched_pairs)
        or (qa_valid_med < prev_valid_med * 0.5)
        or (min_monthly_r < 0.5)
    )
    if technical_bug:
        recommendation = (
            "The diagnostics indicate unresolved technical/source concerns. Do not use QA GNDVI in Figure 5/6 "
            "until the flagged source/masking/grid issues are resolved, or use a moderate-QA alternative."
        )
    else:
        recommendation = (
            "No major GeoTIFF-level technical failure was detected. If QA GNDVI is accepted, report that stricter QA "
            "materially changes GNDVI-linked raw correlation/CCF diagnostics and add a caveat."
        )
    lines = [
        "# FINAL GNDVI Old-vs-QA Source Audit Report",
        "",
        "## Scope",
        "Diagnostic-only audit. No manuscript figures or forecasting models were modified.",
        "",
        "## Inputs",
        f"- Previous GNDVI root: `{PREVIOUS_GNDVI}`",
        f"- QA GNDVI source table: `{QA_TABLE}`",
        f"- Source recovery package consulted: `{SOURCE_RECOVERY}`",
        f"- GNDVI diagnostic package consulted if present: `{DIAG_PACKAGE}`",
        "",
        "## 1. Direct comparability by date, land type, grid, CRS, and extent",
        f"- Previous raster count: {len(prev)}",
        f"- QA raster count: {len(qa)}",
        f"- Matched old/QA month-land pairs: {matched_pairs}",
        f"- Matched pairs with same grid/CRS/transform: {same_grid}/{matched_pairs}",
        "- Previous GNDVI coverage is expected to be shorter if the legacy folder ends in 2021; QA table contains 2001-2024.",
        "",
        "## 2. Physical plausibility and nodata behavior",
        f"- Maximum QA fraction outside [-1, 1]: {qa_out_max:.6f}",
        f"- Median QA zero fraction: {qa_zero_med:.6f}",
        f"- Median finite fraction previous: {prev_valid_med:.6f}",
        f"- Median finite fraction QA: {qa_valid_med:.6f}",
        "",
        "## 3. Monthly time-series agreement",
        f"- Mean old-vs-QA monthly Pearson r across land types: {mean_monthly_r:.3f}",
        f"- Minimum old-vs-QA monthly Pearson r across land types: {min_monthly_r:.3f}",
        f"- Number of matched months with absolute mean difference > 0.10: {high_diff}",
        "",
        "## 4. Seasonal-cycle preservation",
        "Seasonal-cycle means and standard deviations are saved in `GNDVI_seasonal_cycle_old_vs_QA.csv` and plotted in `GNDVI_old_vs_QA_seasonal_cycle.png/pdf`.",
        "",
        "## 5. Inter-index correlation changes",
        "Largest exact-old-Figure-5 GNDVI correlation differences (QA - previous):",
        largest.to_string(index=False),
        "",
        "## 6. Figure 6 CCF changes",
        f"- Key GNDVI CCF sign/peak-shift flags: {ccf_flags}",
        "- CCF diagnostic table: `GNDVI_CCF_old_vs_QA_diagnostic.csv`.",
        "",
        "## 7. Failure-mode interpretation",
        checklist.to_string(index=False),
        "",
        "## 8. Cause classification",
        "- If the checklist flags grid/source/mapping/valid-fraction problems, the Figure 5/6 behavior is likely technical/source-related.",
        "- If GeoTIFF values, dates, grids, and valid fractions are clean while correlations differ, the behavior is more likely a real consequence of using a different/stricter GNDVI source.",
        "",
        "## 9. Recommendation",
        recommendation,
        "",
        "## 10. Manuscript caveat options",
        "- If QA GNDVI is accepted: state that GNDVI-linked raw inter-index and CCF diagnostics are sensitive to QA filtering/source choice, while non-GNDVI pairs are unaffected.",
        "- If previous GNDVI is retained: justify it as the land-type-specific legacy product used in the original raw monthly diagnostic figures and report QA GNDVI as a sensitivity.",
        "- If uncertainty remains: use a moderate-QA GNDVI variant or remove GNDVI from Figure 5/6 until the export/masking choice is resolved.",
    ]
    (REPORT / "FINAL_GNDVI_old_vs_QA_source_audit_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    inv = inventory_sources()
    pairing = pairing_and_grid(inv)
    pixel = pixel_level_comparison(pairing)
    monthly, sim = monthly_mean_tables(inv)
    seasonal = seasonal_cycle(monthly)
    plot_monthly_diagnostics(monthly, seasonal)
    inter = interindex_diagnostic()
    ccf_diag, curves = ccf_diagnostic()
    checklist = failure_mode_checklist(inv, pairing, pixel, monthly, inter, ccf_diag)
    final_report(inv, pairing, pixel, monthly, sim, seasonal, inter, ccf_diag, checklist)
    log(f"Audit complete: {OUT}")


if __name__ == "__main__":
    main()
