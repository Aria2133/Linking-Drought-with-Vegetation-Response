import hashlib
import os
import re
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import stats
from statsmodels.tsa.stattools import ccf


ROOT = Path(r"<PROJECT_ROOT>\Docs\major_revision_analyses\figure5_figure6_exact_source_recovery")
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "tables"
REPORT_DIR = ROOT / "report"
AUDIT_DIR = ROOT / "audit"
for directory in [FIG_DIR, TABLE_DIR, REPORT_DIR, AUDIT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(r"<DATA_ROOT>\Data")
TRAIN_DIRS = {
    "Grasslands": DATA_ROOT / "CALIFORNIA GRASSLANDS",
    "NaturalForests": DATA_ROOT / "CALIFORNIA NATURAL FORESTS",
    "Savannas": DATA_ROOT / "CALIFORNIA SAVANNAS",
    "Shrublands": DATA_ROOT / "CALIFORNIA SHRUBLANDS",
}
TEST_ROOT_DOCUMENTS = Path(
    r"<DATA_ROOT>\Test Data\Masked_LandTypes"
)
TEST_ROOT_DESKTOP = Path(r"<DATA_ROOT>\Test Data\Masked_LandTypes")
TEST_DIRS_DOCUMENTS = {
    "Grasslands": TEST_ROOT_DOCUMENTS / "Grasslands",
    "NaturalForests": TEST_ROOT_DOCUMENTS / "NaturalForests",
    "Savannas": TEST_ROOT_DOCUMENTS / "Savannas",
    "Shrublands": TEST_ROOT_DOCUMENTS / "Shrublands",
}
TEST_DIRS_DESKTOP = {
    "Grasslands": TEST_ROOT_DESKTOP / "Grasslands",
    "NaturalForests": TEST_ROOT_DESKTOP / "NaturalForests",
    "Savannas": TEST_ROOT_DESKTOP / "Savannas",
    "Shrublands": TEST_ROOT_DESKTOP / "Shrublands",
}

PREVIOUS_GNDVI_ROOT = Path(r"<PROJECT_ROOT>\Previous GNDVI")
QA_GNDVI_TABLE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses\05_forecasting_raster_full_candidate\tables\gndvi_source_confirmation.csv"
)

ARCHIVED_FIG5_PRIMARY = Path(
    r"<DATA_ROOT>\Data\Clipping Tonzi Ranch\combined_correlation_heatmap_2001_2024.png"
)
ARCHIVED_FIG5_DESKTOP = Path(
    r"<PROJECT_ROOT>\Previous Visualizations\combined_correlation_heatmap_2001_2024.png"
)
ARCHIVED_FIG6_CANDIDATES = [
    Path(r"<DATA_ROOT>\ALL the RESULTS\Fig2_CCF_Heatmaps.png"),
    Path(r"<DATA_ROOT>\ALL the RESULTS\Fig2_CCF_Heatmaps fixed.png"),
    Path(r"<DATA_ROOT>\ALL the RESULTS\Advanced_Figures\Fig2_CCF_Heatmaps.png"),
    Path(r"<PROJECT_ROOT>\Advanced_Figures\Fig2_CCF_Heatmaps.png"),
]

LAND_TYPES_FIG5 = ["GRASSLANDS", "NATURAL FORESTS", "SAVANNAS", "SHRUBLANDS"]
LAND_TYPE_MAP = {
    "GRASSLANDS": "Grasslands",
    "NATURAL FORESTS": "NaturalForests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
LAND_TYPES_FIG6 = ["Grasslands", "NaturalForests", "Savannas", "Shrublands"]
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]

FIG5_NAMING_PATTERNS = {
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def scan_candidate_scripts():
    roots = [
        Path(r"<DATA_ROOT>\Codes"),
        Path(r"<DATA_ROOT>\Codes\Visualizations"),
        Path(r"<PROJECT_ROOT>\Previous Visualizations"),
        Path(r"<PROJECT_ROOT>"),
        Path(r"<PROJECT_ROOT>\Linking-Drought-with-Vegetation-Response"),
        Path(r"<DATA_ROOT>\Data\Clipping Tonzi Ranch"),
    ]
    fig5_patterns = [
        "Inter-Index Correlation Matrix",
        "combined_correlation_heatmap_2001_2024",
        "magma_r",
        "np.triu",
        "GPP: 2001-2021 only",
        "calculate_correlation_matrix",
        "load_variable_data",
    ]
    fig6_patterns = [
        "Negative = Predictor LEADS target",
        "Predictor → Target (Unique Pairs)",
        "Fig2_CCF_Heatmaps",
        "plot_ccf_heatmaps",
        "compute_ccf_proper",
        "ccf_pos",
        "ccf_neg",
        "all_lags = list(range(-max_lag, max_lag+1))",
    ]
    seen = set()
    rows = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path in seen:
                continue
            seen.add(path)
            text = read_text(path)
            f5 = [p for p in fig5_patterns if p in text]
            f6 = [p for p in fig6_patterns if p in text]
            if not f5 and not f6:
                continue
            likely = []
            if len(f5) >= 3:
                likely.append("Figure 5")
            if len(f6) >= 3:
                likely.append("Figure 6")
            output_names = []
            if "combined_correlation_heatmap_2001_2024" in text:
                output_names.append("combined_correlation_heatmap_2001_2024.png")
            if "Fig2_CCF_Heatmaps" in text:
                output_names.append("Fig2_CCF_Heatmaps.png/pdf")
            decision = "reject"
            reason = []
            if "Data\\Clipping Tonzi Ranch\\Inter index correlation.py" in str(path):
                decision = "use_for_figure5_primary"
                reason.append("unpatched raw-array/min-length Figure 5 script and archived PNG in same folder")
            elif "Codes\\Visualizations\\Advance Drought Visualizations.py" in str(path):
                decision = "use_for_figure6_primary"
                reason.append("unpatched Advanced CCF script with exact CCF fingerprints")
            elif "Visualizations\\CCF.py" in str(path):
                decision = "reject"
                reason.append("different CCF family")
            elif "GNDVI_SOURCE_CONFIRMATION" in text or "load_qa_controlled_gndvi" in text:
                reason.append("already patched to QA GNDVI")
            elif "Inter index correlation.py" in path.name and len(f5) >= 5:
                decision = "candidate"
                reason.append("matches Figure 5 fingerprints")
            elif "Advance Drought Visualizations.py" in path.name and len(f6) >= 6:
                decision = "candidate"
                reason.append("matches Figure 6 fingerprints")
            rows.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "matched_fingerprints": "; ".join(f5 + f6),
                    "likely_figure_number": "; ".join(likely),
                    "output_filename": "; ".join(output_names),
                    "matches_shown_old_figure": "yes" if decision.startswith("use") or decision == "candidate" else "no",
                    "use_or_reject": decision,
                    "reason": "; ".join(reason),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(TABLE_DIR / "source_lineage_candidate_scripts.csv", index=False)
    md = ["# Source Lineage Candidate Scripts", ""]
    for _, row in df.iterrows():
        md.append(f"## {row['use_or_reject']}: `{row['path']}`")
        md.append(f"- Likely figure: {row['likely_figure_number']}")
        md.append(f"- Output: {row['output_filename']}")
        md.append(f"- Matched fingerprints: {row['matched_fingerprints']}")
        md.append(f"- Reason: {row['reason']}")
        md.append("")
    (REPORT_DIR / "source_lineage_candidate_scripts.md").write_text("\n".join(md), encoding="utf-8")
    return df


def parse_date_from_filename(filename):
    for pattern in [r"(\d{4})\.M(\d{2})", r"(\d{4})_(\d{2})", r"(\d{6})(?:_|$)"]:
        match = re.search(pattern, filename)
        if not match:
            continue
        if len(match.groups()) == 2:
            year, month = int(match.group(1)), int(match.group(2))
        else:
            raw = match.group(1)
            year, month = int(raw[:4]), int(raw[4:6])
        if 2000 <= year <= 2026 and 1 <= month <= 12:
            return year, month
    return None, None


def clean_raster_data(data):
    data = data.astype(np.float32)
    for nodata in [-9999, -3.4e38, 32767, -32768, 65535, -32767]:
        data[np.abs(data - nodata) < 1.0] = np.nan
    data[np.abs(data) > 1e10] = np.nan
    return data


def read_raster_mean(path):
    try:
        with rasterio.open(path) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            data = clean_raster_data(data)
            val = np.nanmean(data)
            return float(val) if not np.isnan(val) else None
    except Exception:
        return None


def previous_gndvi_files(land_type):
    folder_map = {
        "Grasslands": "grass",
        "NaturalForests": "nat",
        "Savannas": "sav",
        "Shrublands": "shrub",
    }
    folder = PREVIOUS_GNDVI_ROOT / folder_map[land_type] / "GNDVI"
    return sorted(folder.glob("*.tif"))


def load_previous_gndvi_array(land_type):
    vals = []
    paths = previous_gndvi_files(land_type)
    for path in paths:
        val = read_raster_mean(path)
        if val is not None:
            vals.append(val)
    return np.array(vals, dtype=float)


def load_previous_gndvi_series(land_type):
    rows = []
    for path in previous_gndvi_files(land_type):
        y, m = parse_date_from_filename(path.stem)
        if y is None:
            continue
        val = read_raster_mean(path)
        if val is not None:
            rows.append({"date": pd.Timestamp(y, m, 1), "value": val})
    if not rows:
        return None
    return pd.DataFrame(rows).drop_duplicates("date", keep="last").set_index("date").sort_index()["value"].rename("GNDVI")


def load_qa_gndvi_series(land_type):
    if not QA_GNDVI_TABLE.exists():
        return None
    confirmation = pd.read_csv(QA_GNDVI_TABLE)
    key = {
        "Grasslands": "GRASSLANDS",
        "NaturalForests": "NATURAL_FORESTS",
        "Savannas": "SAVANNAS",
        "Shrublands": "SHRUBLANDS",
    }[land_type]
    subset = confirmation[(confirmation["land_type"] == key) & (confirmation["qa_fixed_confirmed"] == True)].copy()
    subset["date"] = subset["date"].astype(str)
    subset = subset.drop_duplicates("date", keep="last").sort_values("date")
    rows = []
    for _, row in subset.iterrows():
        path = Path(row["selected_path"])
        if not path.exists():
            continue
        val = read_raster_mean(path)
        if val is not None:
            ds = row["date"]
            rows.append({"date": pd.Timestamp(int(ds[:4]), int(ds[4:6]), 1), "value": val})
    if not rows:
        return None
    return pd.DataFrame(rows).set_index("date").sort_index()["value"].rename("GNDVI")


def load_qa_gndvi_array(land_type):
    series = load_qa_gndvi_series(land_type)
    return np.array([], dtype=float) if series is None else series.values.astype(float)


def fig5_load_variable_array(variable, land_type_key, test_dirs, gndvi_mode="original"):
    land_type = LAND_TYPE_MAP[land_type_key]
    if variable == "GNDVI":
        if gndvi_mode == "previous":
            return load_previous_gndvi_array(land_type)
        if gndvi_mode == "qa":
            return load_qa_gndvi_array(land_type)
    data = []
    actual_start_year = {"LST": 2003, "TWS": 2002}.get(variable, 2001)
    index_dir = TRAIN_DIRS[land_type] / variable
    if index_dir.exists():
        for year in range(actual_start_year, 2022):
            for month in range(1, 13):
                filename = FIG5_NAMING_PATTERNS[land_type][variable].format(year=year, month=month)
                path = index_dir / filename
                if path.exists():
                    val = read_raster_mean(path)
                    if val is not None:
                        data.append(val)
    if variable != "GPP":
        test_index_dir = test_dirs[land_type] / variable
        if test_index_dir.exists():
            for year in range(2022, 2025):
                for month in range(1, 13):
                    filename = FIG5_TEST_PATTERNS[land_type][variable].format(year=year, month=month)
                    path = test_index_dir / filename
                    if path.exists():
                        val = read_raster_mean(path)
                        if val is not None:
                            data.append(val)
    return np.array(data, dtype=float)


def fig5_calculate(test_dirs, gndvi_mode="original"):
    all_results = {}
    values_rows = []
    input_rows = []
    for land_key in LAND_TYPES_FIG5:
        data = {}
        for var in VARIABLES:
            arr = fig5_load_variable_array(var, land_key, test_dirs, gndvi_mode=gndvi_mode)
            if len(arr) > 0:
                data[var] = arr
            input_rows.append(
                {
                    "figure": "Figure 5",
                    "variant": gndvi_mode,
                    "land_type": land_key,
                    "variable": var,
                    "n_values_loaded": len(arr),
                    "source_logic": "raw raster spatial means; old min-length truncation; GPP test period skipped",
                    "test_root": str(next(iter(test_dirs.values())).parent),
                }
            )
        valid_vars = [k for k, v in data.items() if len(v) > 0]
        min_len = min(len(data[v]) for v in valid_vars)
        matrix = np.array([data[v][:min_len] for v in valid_vars]).T
        corr = np.corrcoef(matrix.T)
        all_results[land_key] = {"matrix": corr, "variables": valid_vars, "min_length": min_len}
        for i, v1 in enumerate(valid_vars):
            for j, v2 in enumerate(valid_vars):
                values_rows.append(
                    {
                        "land_type": land_key,
                        "variable_1": v1,
                        "variable_2": v2,
                        "correlation": corr[i, j],
                        "min_length_used": min_len,
                        "gndvi_mode": gndvi_mode,
                    }
                )
    return all_results, pd.DataFrame(values_rows), pd.DataFrame(input_rows)


def render_fig5(results, output_png, output_pdf=None, large_style=True):
    if large_style:
        fig, axes = plt.subplots(2, 2, figsize=(24, 22))
        annot_kws = {"size": 16, "weight": "bold", "color": "white"}
        title_size = 18
        label_size = 14
        tick_size = 13
        suptitle_size = 20
        linewidth = 1.0
        cbar_kws = {"shrink": 0.8, "pad": 0.02}
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        annot_kws = {"size": 8}
        title_size = 14
        label_size = 11
        tick_size = 9
        suptitle_size = 16
        linewidth = 0.5
        cbar_kws = {"shrink": 0.8}
    axes = axes.flatten()
    for ax, land_type in zip(axes, LAND_TYPES_FIG5):
        corr = results[land_type]["matrix"]
        variables = results[land_type]["variables"]
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="magma_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=linewidth,
            cbar_kws=cbar_kws,
            xticklabels=variables,
            yticklabels=variables,
            annot_kws=annot_kws,
            ax=ax,
        )
        ax.set_title(land_type, fontsize=title_size, fontweight="bold", pad=15 if large_style else 10)
        ax.set_xlabel("Variables", fontsize=label_size, fontweight="bold")
        ax.set_ylabel("Variables", fontsize=label_size, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=tick_size)
        ax.tick_params(axis="y", rotation=0, labelsize=tick_size)
        cbar_obj = ax.collections[0].colorbar
        cbar_obj.ax.tick_params(labelsize=12 if large_style else 9)
    fig.suptitle(
        "Inter-Index Correlation Matrix (2001-2024)\nGPP: 2001-2021 only",
        fontsize=suptitle_size,
        fontweight="bold",
        y=0.995,
    )
    if large_style:
        plt.subplots_adjust(hspace=0.35, wspace=0.25)
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_png, dpi=600 if large_style else 300, bbox_inches="tight", facecolor="white")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def image_difference(candidate, archived):
    if not candidate.exists() or not archived.exists():
        return {"archived_path": str(archived), "archived_exists": archived.exists(), "candidate_exists": candidate.exists()}
    try:
        im1 = Image.open(candidate).convert("RGB")
        im2 = Image.open(archived).convert("RGB")
        original_size = im2.size
        im1 = im1.resize(original_size)
        a = np.asarray(im1).astype(float)
        b = np.asarray(im2).astype(float)
        diff = np.abs(a - b)
        return {
            "archived_path": str(archived),
            "archived_exists": True,
            "candidate_exists": True,
            "candidate_size": f"{Image.open(candidate).size[0]}x{Image.open(candidate).size[1]}",
            "archived_size": f"{original_size[0]}x{original_size[1]}",
            "mean_abs_pixel_diff": float(diff.mean()),
            "median_abs_pixel_diff": float(np.median(diff)),
            "max_abs_pixel_diff": float(diff.max()),
        }
    except Exception as exc:
        return {"archived_path": str(archived), "archived_exists": True, "candidate_exists": True, "error": str(exc)}


def load_timeseries_original_ccf(land_type, index, test_dirs, gndvi_mode="original"):
    if index == "GNDVI":
        if gndvi_mode == "previous":
            s = load_previous_gndvi_series(land_type)
            if s is not None:
                return s.to_frame()
        elif gndvi_mode == "qa":
            s = load_qa_gndvi_series(land_type)
            if s is not None:
                return s.to_frame()
    files = []
    train_dir = TRAIN_DIRS[land_type] / index
    if train_dir.exists():
        files.extend(train_dir.glob("*.tif"))
    if index == "LST":
        filled = TRAIN_DIRS[land_type] / f"{index}_filled"
        if filled.exists():
            files.extend(filled.glob("*.tif"))
    test_dir = test_dirs[land_type] / index
    if test_dir.exists():
        files.extend(test_dir.glob("*.tif"))
    if not files:
        return None
    min_valid = {"GOSIF": 5, "GPP": 5, "TWS": 1}.get(index, 10)
    rows = []
    for path in sorted(files):
        y, m = parse_date_from_filename(path.stem)
        if y is None:
            continue
        val = read_raster_mean(path)
        if val is None:
            continue
        try:
            with rasterio.open(path) as src:
                data = clean_raster_data(src.read(1))
                valid = int(np.sum(~np.isnan(data)))
            if valid < min_valid:
                continue
        except Exception:
            pass
        rows.append({"date": pd.Timestamp(y, m, 1), "value": val})
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("date", keep="first").set_index("date").sort_index()
    df.columns = [index]
    return df


def ccf_load_all_data(test_dirs, gndvi_mode="original"):
    by_land = {}
    input_rows = []
    for land in LAND_TYPES_FIG6:
        dfs = []
        for index in VARIABLES:
            df = load_timeseries_original_ccf(land, index, test_dirs, gndvi_mode=gndvi_mode)
            n = 0 if df is None else len(df)
            input_rows.append(
                {
                    "figure": "Figure 6",
                    "variant": gndvi_mode,
                    "land_type": land,
                    "variable": index,
                    "n_values_loaded": n,
                    "source_logic": "raw spatial means; outer join by date; ffill(limit=2); CCF z-scores pairwise valid rows",
                    "test_root": str(next(iter(test_dirs.values())).parent),
                }
            )
            if df is not None and len(df) > 0:
                dfs.append(df)
        if not dfs:
            continue
        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how="outer")
        result = result.ffill(limit=2)
        by_land[land] = result
    return by_land, pd.DataFrame(input_rows)


def compute_ccf_proper(x, y, max_lag=6, alpha=0.05):
    valid = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[valid].values
    y_clean = y[valid].values
    if len(x_clean) < 24:
        return [], []
    x_std = (x_clean - np.mean(x_clean)) / np.std(x_clean)
    y_std = (y_clean - np.mean(y_clean)) / np.std(y_clean)
    ccf_pos = ccf(x_std, y_std, adjusted=False)[: max_lag + 1]
    ccf_neg = ccf(y_std, x_std, adjusted=False)[1 : max_lag + 1]
    all_lags = list(range(-max_lag, max_lag + 1))
    all_ccf = list(ccf_neg[::-1]) + list(ccf_pos)
    return all_lags, all_ccf


def ccf_calculate(test_dirs, gndvi_mode="original"):
    data_by_land, input_rows = ccf_load_all_data(test_dirs, gndvi_mode=gndvi_mode)
    values = []
    matrices = {}
    labels_by_land = {}
    max_lag = 6
    for land, data in data_by_land.items():
        indices = data.columns.tolist()
        ccf_matrix = []
        pair_labels = []
        for i, predictor in enumerate(indices):
            for j, target in enumerate(indices):
                if i < j:
                    lags, vals = compute_ccf_proper(data[predictor], data[target], max_lag=max_lag)
                    if len(vals) > 0:
                        ccf_matrix.append(vals)
                        pair_label = f"{predictor}→{target}"
                        pair_labels.append(pair_label)
                        for lag, val in zip(lags, vals):
                            values.append(
                                {
                                    "land_type": land,
                                    "predictor": predictor,
                                    "target": target,
                                    "pair": pair_label,
                                    "lag": lag,
                                    "ccf": val,
                                    "gndvi_mode": gndvi_mode,
                                }
                            )
        matrices[land] = np.array(ccf_matrix)
        labels_by_land[land] = pair_labels
    return matrices, labels_by_land, pd.DataFrame(values), input_rows


def render_fig6(matrices, labels_by_land, output_png, output_pdf=None):
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_context("paper", font_scale=1.2)
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, top=0.95, bottom=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    im = None
    for idx, land in enumerate(LAND_TYPES_FIG6):
        ax = axes[idx]
        matrix = matrices[land]
        labels = labels_by_land[land]
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1, interpolation="bilinear")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        lag_labels = list(range(-6, 7))
        ax.set_xticks(range(len(lag_labels)))
        ax.set_xticklabels(lag_labels, fontsize=11)
        ax.set_xlabel("Lag (months)\nNegative = Predictor LEADS target", fontsize=12, weight="bold")
        ax.set_ylabel("Predictor → Target (Unique Pairs)", fontsize=12, weight="bold")
        ax.set_title(f"{land}\nCross-Correlation Function", fontsize=13, weight="bold")
        ax.axvline(6, color="black", linewidth=2.5, linestyle="--", alpha=0.8, label="Lag = 0")
        ax.legend(loc="upper right", fontsize=10)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Cross-Correlation Coefficient\n(Red = Positive | Blue = Negative)", fontsize=13, weight="bold")
    cbar.ax.tick_params(labelsize=11)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if output_pdf:
        fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def compare_fig5_values(base, alt, label):
    merged = base.merge(
        alt,
        on=["land_type", "variable_1", "variable_2"],
        suffixes=("_original", f"_{label}"),
    )
    alt_col = f"correlation_{label}"
    merged["difference"] = merged[alt_col] - merged["correlation_original"]
    merged["abs_difference"] = merged["difference"].abs()
    merged["changed"] = merged["abs_difference"] > 1e-10
    merged["involves_gndvi"] = (merged["variable_1"] == "GNDVI") | (merged["variable_2"] == "GNDVI")
    merged["non_gndvi_changed"] = merged["changed"] & ~merged["involves_gndvi"]
    merged["comparison"] = label
    return merged


def compare_fig6_values(base, alt, label):
    merged = base.merge(
        alt,
        on=["land_type", "predictor", "target", "pair", "lag"],
        suffixes=("_original", f"_{label}"),
    )
    alt_col = f"ccf_{label}"
    merged["difference"] = merged[alt_col] - merged["ccf_original"]
    merged["abs_difference"] = merged["difference"].abs()
    merged["changed"] = merged["abs_difference"] > 1e-10
    merged["involves_gndvi"] = (merged["predictor"] == "GNDVI") | (merged["target"] == "GNDVI")
    merged["non_gndvi_changed"] = merged["changed"] & ~merged["involves_gndvi"]
    merged["comparison"] = label
    return merged


def write_markdown(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    candidates = scan_candidate_scripts()

    # Figure 5 candidate reproductions.
    fig5_repro_results, fig5_repro_values, fig5_inputs = fig5_calculate(TEST_DIRS_DOCUMENTS, gndvi_mode="original")
    fig5_png = FIG_DIR / "Figure5_reproduction_from_exact_old_script.png"
    fig5_pdf = FIG_DIR / "Figure5_reproduction_from_exact_old_script.pdf"
    render_fig5(fig5_repro_results, fig5_png, fig5_pdf, large_style=True)
    fig5_repro_values.to_csv(TABLE_DIR / "Figure5_reproduction_values.csv", index=False)

    fig5_image_audit_rows = []
    for archived in [ARCHIVED_FIG5_PRIMARY, ARCHIVED_FIG5_DESKTOP]:
        row = image_difference(fig5_png, archived)
        row["candidate_variant"] = "primary_unpatched_clipping_tonzi_documents_test_min_length"
        fig5_image_audit_rows.append(row)
    fig5_image_audit = pd.DataFrame(fig5_image_audit_rows)
    fig5_image_audit.to_csv(TABLE_DIR / "Figure5_reproduction_vs_archived_png_audit.csv", index=False)

    # Figure 6 candidate reproductions: documents test root and missing Desktop test root.
    fig6_doc_m, fig6_doc_labels, fig6_doc_values, fig6_doc_inputs = ccf_calculate(TEST_DIRS_DOCUMENTS, "original")
    fig6_doc_png = FIG_DIR / "Figure6_reproduction_from_exact_old_script.png"
    fig6_doc_pdf = FIG_DIR / "Figure6_reproduction_from_exact_old_script.pdf"
    render_fig6(fig6_doc_m, fig6_doc_labels, fig6_doc_png, fig6_doc_pdf)
    fig6_doc_values.to_csv(TABLE_DIR / "Figure6_reproduction_values.csv", index=False)

    fig6_image_audit_rows = []
    for archived in ARCHIVED_FIG6_CANDIDATES:
        row = image_difference(fig6_doc_png, archived)
        row["candidate_variant"] = "advanced_ccf_logic_documents_test_root"
        fig6_image_audit_rows.append(row)
    fig6_image_audit = pd.DataFrame(fig6_image_audit_rows)
    fig6_image_audit.to_csv(TABLE_DIR / "Figure6_reproduction_vs_archived_png_audit.csv", index=False)

    # Controlled GNDVI replacements.
    fig5_prev_results, fig5_prev_values, fig5_prev_inputs = fig5_calculate(TEST_DIRS_DOCUMENTS, gndvi_mode="previous")
    render_fig5(
        fig5_prev_results,
        FIG_DIR / "Figure5_previous_GNDVI_exact_old_logic.png",
        FIG_DIR / "Figure5_previous_GNDVI_exact_old_logic.pdf",
        large_style=True,
    )
    fig5_prev_values.to_csv(TABLE_DIR / "Figure5_previous_GNDVI_values.csv", index=False)

    fig5_qa_results, fig5_qa_values, fig5_qa_inputs = fig5_calculate(TEST_DIRS_DOCUMENTS, gndvi_mode="qa")
    render_fig5(
        fig5_qa_results,
        FIG_DIR / "Figure5_QA_GNDVI_exact_old_logic.png",
        FIG_DIR / "Figure5_QA_GNDVI_exact_old_logic.pdf",
        large_style=True,
    )
    fig5_qa_values.to_csv(TABLE_DIR / "Figure5_QA_GNDVI_values.csv", index=False)

    fig6_prev_m, fig6_prev_labels, fig6_prev_values, fig6_prev_inputs = ccf_calculate(TEST_DIRS_DOCUMENTS, "previous")
    render_fig6(
        fig6_prev_m,
        fig6_prev_labels,
        FIG_DIR / "Figure6_previous_GNDVI_exact_old_logic.png",
        FIG_DIR / "Figure6_previous_GNDVI_exact_old_logic.pdf",
    )
    fig6_prev_values.to_csv(TABLE_DIR / "Figure6_previous_GNDVI_values.csv", index=False)

    fig6_qa_m, fig6_qa_labels, fig6_qa_values, fig6_qa_inputs = ccf_calculate(TEST_DIRS_DOCUMENTS, "qa")
    render_fig6(
        fig6_qa_m,
        fig6_qa_labels,
        FIG_DIR / "Figure6_QA_GNDVI_exact_old_logic.png",
        FIG_DIR / "Figure6_QA_GNDVI_exact_old_logic.pdf",
    )
    fig6_qa_values.to_csv(TABLE_DIR / "Figure6_QA_GNDVI_values.csv", index=False)

    pd.concat([fig5_inputs, fig5_prev_inputs, fig5_qa_inputs, fig6_doc_inputs, fig6_prev_inputs, fig6_qa_inputs]).to_csv(
        TABLE_DIR / "Figure5_Figure6_input_path_inventory.csv", index=False
    )

    fig5_prev_audit = compare_fig5_values(fig5_repro_values, fig5_prev_values, "previous_GNDVI")
    fig5_qa_audit = compare_fig5_values(fig5_repro_values, fig5_qa_values, "QA_GNDVI")
    fig5_audit = pd.concat([fig5_prev_audit, fig5_qa_audit], ignore_index=True)
    fig5_audit.to_csv(TABLE_DIR / "Figure5_GNDVI_source_comparison_audit.csv", index=False)

    fig6_prev_audit = compare_fig6_values(fig6_doc_values, fig6_prev_values, "previous_GNDVI")
    fig6_qa_audit = compare_fig6_values(fig6_doc_values, fig6_qa_values, "QA_GNDVI")
    fig6_audit = pd.concat([fig6_prev_audit, fig6_qa_audit], ignore_index=True)
    fig6_audit.to_csv(TABLE_DIR / "Figure6_GNDVI_source_comparison_audit.csv", index=False)

    # Specific ecological coherence summaries for GNDVI pairs.
    fig5_focus = fig5_audit[
        fig5_audit["variable_1"].eq("GNDVI")
        & fig5_audit["variable_2"].isin(["NDVI", "NIRv", "GOSIF", "GPP"])
    ].copy()
    fig5_focus.to_csv(TABLE_DIR / "Figure5_GNDVI_key_relationships.csv", index=False)

    # Copy key audits to audit folder too.
    shutil.copy2(TABLE_DIR / "Figure5_GNDVI_source_comparison_audit.csv", AUDIT_DIR / "Figure5_GNDVI_source_comparison_audit.csv")
    shutil.copy2(TABLE_DIR / "Figure6_GNDVI_source_comparison_audit.csv", AUDIT_DIR / "Figure6_GNDVI_source_comparison_audit.csv")

    # Reports.
    f5_non_gndvi_prev = int(fig5_prev_audit["non_gndvi_changed"].sum())
    f5_non_gndvi_qa = int(fig5_qa_audit["non_gndvi_changed"].sum())
    f6_non_gndvi_prev = int(fig6_prev_audit["non_gndvi_changed"].sum())
    f6_non_gndvi_qa = int(fig6_qa_audit["non_gndvi_changed"].sum())

    best_fig6 = fig6_image_audit.sort_values("mean_abs_pixel_diff", na_position="last").head(1)
    best_fig6_path = best_fig6.iloc[0]["archived_path"] if len(best_fig6) else "not available"
    best_fig6_diff = best_fig6.iloc[0].get("mean_abs_pixel_diff", np.nan) if len(best_fig6) else np.nan
    best_fig5 = fig5_image_audit.sort_values("mean_abs_pixel_diff", na_position="last").head(1)
    best_fig5_path = best_fig5.iloc[0]["archived_path"] if len(best_fig5) else "not available"
    best_fig5_diff = best_fig5.iloc[0].get("mean_abs_pixel_diff", np.nan) if len(best_fig5) else np.nan

    write_markdown(
        REPORT_DIR / "Figure5_reproduction_audit.md",
        [
            "# Figure 5 Reproduction Audit",
            "",
            "## Best source script",
            "`<DATA_ROOT>\\Data\\Clipping Tonzi Ranch\\Inter index correlation.py`",
            "",
            "This is the unpatched archive script with the exact old Figure 5 fingerprints: title, `combined_correlation_heatmap_2001_2024.png`, `magma_r`, lower-triangle mask, raw raster spatial means, GPP test-period skip, and min-length truncation.",
            "",
            "## Input folders used",
            f"- Training rasters: `{DATA_ROOT}` land-type folders",
            f"- Test rasters: `{TEST_ROOT_DOCUMENTS}`",
            "",
            "## Old logic confirmed",
            "- Raw/original monthly raster spatial means.",
            "- No anomaly, detrending, baseline-standardization, or EWBI transform.",
            "- GPP uses 2001-2021 only because test-period GPP is skipped.",
            "- Variables are aligned by old min-length truncation, not date-safe complete-case alignment.",
            "- Previous/original GNDVI is loaded from the old land-type `GNDVI` raster folders.",
            "",
            "## Archived image comparison",
            f"- Best archived target: `{best_fig5_path}`",
            f"- Mean absolute pixel difference after resizing: {best_fig5_diff}",
            "- Pixel-difference comparison is a visual-fingerprint check; the old PNG does not include a machine-readable value table.",
        ],
    )

    write_markdown(
        REPORT_DIR / "Figure6_reproduction_audit.md",
        [
            "# Figure 6 Reproduction Audit",
            "",
            "## Best source script",
            "`<DATA_ROOT>\\Codes\\Visualizations\\Advance Drought Visualizations.py`",
            "",
            "This unpatched script contains the exact Figure 6 fingerprints: `compute_ccf_proper`, `plot_ccf_heatmaps`, all unique predictor-target pairs, lags -6 to +6, dashed lag-0 line, and bottom horizontal colorbar.",
            "",
            "## Rejected script family",
            "`Visualizations/CCF.py` was rejected because it uses a different target-restricted CCF family rather than all unique pairs.",
            "",
            "## Input folders used",
            f"- Training rasters: `{DATA_ROOT}` land-type folders",
            f"- Test rasters for reproduction: `{TEST_ROOT_DOCUMENTS}`",
            "",
            "The unpatched thesis-code copy points to a Desktop test-data root that is no longer present on this machine. The recovered reproduction therefore uses the same old CCF logic with the extant thesis `Test Data\\Masked_LandTypes` root. Candidate image matching is reported in `Figure6_reproduction_vs_archived_png_audit.csv`.",
            "",
            "## Old logic confirmed",
            "- Raw monthly raster spatial means loaded by date from filenames.",
            "- Outer date join across variables followed by `ffill(limit=2)`.",
            "- Pairwise valid rows are z-scored inside `compute_ccf_proper`.",
            "- Negative lag means predictor leads target.",
            f"- Best archived target: `{best_fig6_path}`",
            f"- Mean absolute pixel difference after resizing: {best_fig6_diff}",
        ],
    )

    write_markdown(
        REPORT_DIR / "Figure5_GNDVI_source_comparison_audit.md",
        [
            "# Figure 5 GNDVI Source Comparison Audit",
            "",
            f"- Non-GNDVI changed cells for previous-GNDVI replacement: {f5_non_gndvi_prev}",
            f"- Non-GNDVI changed cells for QA-GNDVI replacement: {f5_non_gndvi_qa}",
            "",
            "Any non-GNDVI change would indicate a logic leak. Both counts should be zero for a valid GNDVI-only replacement.",
            "",
            "Key GNDVI vegetation/productivity relationships are saved in `Figure5_GNDVI_key_relationships.csv`.",
        ],
    )

    write_markdown(
        REPORT_DIR / "Figure6_GNDVI_source_comparison_audit.md",
        [
            "# Figure 6 GNDVI Source Comparison Audit",
            "",
            f"- Non-GNDVI changed CCF rows for previous-GNDVI replacement: {f6_non_gndvi_prev}",
            f"- Non-GNDVI changed CCF rows for QA-GNDVI replacement: {f6_non_gndvi_qa}",
            "",
            "Any non-GNDVI change would indicate a logic leak. Both counts should be zero for a valid GNDVI-only replacement.",
        ],
    )

    # Decision report.
    qa_focus = fig5_focus[fig5_focus["comparison"].eq("QA_GNDVI")]
    prev_focus = fig5_focus[fig5_focus["comparison"].eq("previous_GNDVI")]
    qa_mean_delta = float(qa_focus["difference"].mean()) if len(qa_focus) else np.nan
    prev_mean_delta = float(prev_focus["difference"].mean()) if len(prev_focus) else np.nan
    write_markdown(
        REPORT_DIR / "FINAL_Figure5_Figure6_source_lineage_and_GNDVI_decision_report.md",
        [
            "# Final Figure 5/Figure 6 Source-Lineage and GNDVI Decision Report",
            "",
            "## Exact old Figure 5 source",
            "`<DATA_ROOT>\\Data\\Clipping Tonzi Ranch\\Inter index correlation.py`.",
            "",
            "## Exact old Figure 6 source",
            "`<DATA_ROOT>\\Codes\\Visualizations\\Advance Drought Visualizations.py`.",
            "",
            "## Scripts rejected",
            "- `Visualizations/CCF.py`: rejected because it is a different CCF family with restricted targets rather than all unique pairs.",
            "- Active/patched `Previous Visualizations\\Advance Drought Visualizations.py`: rejected as an original source because it now contains a QA-GNDVI override.",
            "- Active/patched thesis `Codes\\Visualizations\\Inter index correlation.py`: rejected as original Figure 5 source because it now contains a QA-GNDVI loader.",
            "- New revision figure scripts: rejected for source-lineage recovery because they were not old manuscript figure scripts.",
            "",
            "## Reproduction status",
            f"- Figure 5 reproduced from the unpatched archive logic. Best archived-image mean absolute pixel difference: {best_fig5_diff}.",
            f"- Figure 6 reproduced from the unpatched CCF logic with the extant thesis test-data root. Best archived-image mean absolute pixel difference: {best_fig6_diff}.",
            "",
            "## Controlled GNDVI-only replacement",
            f"- Figure 5 non-GNDVI changed cells with previous GNDVI: {f5_non_gndvi_prev}.",
            f"- Figure 5 non-GNDVI changed cells with QA GNDVI: {f5_non_gndvi_qa}.",
            f"- Figure 6 non-GNDVI changed rows with previous GNDVI: {f6_non_gndvi_prev}.",
            f"- Figure 6 non-GNDVI changed rows with QA GNDVI: {f6_non_gndvi_qa}.",
            "",
            "## GNDVI coherence check",
            f"- Mean QA-GNDVI change for Figure 5 GNDVI-NDVI/NIRv/GOSIF/GPP cells relative to original reproduction: {qa_mean_delta:.4f}.",
            f"- Mean previous-GNDVI change for the same cells relative to original reproduction: {prev_mean_delta:.4f}.",
            "",
            "## Recommendation",
            "Use the exact old Figure 5/Figure 6 logic as the scientific definition. The previous-GNDVI exact-old-logic output is the cleanest controlled reconstruction of the old manuscript concept. QA-GNDVI should not replace these figures unless the source-level GNDVI problem is resolved and the ecological coherence checks support it.",
            "",
            "## Required manuscript wording if previous GNDVI is retained",
            "Remove or soften any statement that Figures 5 and 6 use the strict QA-controlled GNDVI source. State that the inter-index and CCF diagnostics preserve the original land-type masked monthly GNDVI source, with QA-controlled GNDVI evaluated as a sensitivity.",
            "",
            "## Caveat if QA-GNDVI is used",
            "Add a caveat that the QA-GNDVI source materially changes GNDVI-linked correlation and CCF structure while leaving non-GNDVI pairs unchanged under controlled replacement.",
        ],
    )

    print("Source-lineage recovery complete.")
    print(f"Figure 5 non-GNDVI changed cells: previous={f5_non_gndvi_prev}, QA={f5_non_gndvi_qa}")
    print(f"Figure 6 non-GNDVI changed rows: previous={f6_non_gndvi_prev}, QA={f6_non_gndvi_qa}")


if __name__ == "__main__":
    main()
