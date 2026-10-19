"""Rebuild figures using previous/no-QA GNDVI including masked 2022-2024 test rasters.

Outputs are written to a new package and then copied into the final figure
folder with backups. Scientific logic is preserved from the accepted source
lineage; only the GNDVI source is completed with test-period land-mask means.
"""

from __future__ import annotations

import hashlib
import itertools
import runpy
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.lines import Line2D
from rasterio.enums import Resampling
from rasterio.warp import reproject


PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_previous_GNDVI_with_test_rebuild"
)
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"
REPORT = PACKAGE / "report"
TOOLS = PACKAGE / "tools"
AUDIT = PACKAGE / "audit"
DATA_OUT = PACKAGE / "data_for_forecasting"
FINAL_FIGURES = Path(
    r"<PROJECT_ROOT>\ECOINF_revision_final_figures_package"
    r"\1_paper_main_figures"
)

LEGACY_CONFIRMATION = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\legacy_gndvi_sensitivity_package\tables\legacy_gndvi_source_confirmation.csv"
)
PREVIOUS_GNDVI_ROOT = Path(r"<PROJECT_ROOT>\Previous GNDVI")
PREVIOUS_GNDVI_TEST = PREVIOUS_GNDVI_ROOT / "Test" / "GNDVI"
LAND_MASK_ROOT = Path(
    r"<DATA_ROOT>\Data\CA_LandMasks"
)
FINAL_PAPER_TABLE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_paper_figures_corrected\tables\paper_standardized_anomaly_series.csv"
)
RECOVERY_SCRIPT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\figure5_figure6_exact_source_recovery\scripts\recover_figure5_figure6_source_lineage.py"
)
FIG6_REFERENCE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\figure5_figure6_exact_source_recovery\figures\Figure6_previous_GNDVI_exact_old_logic.png"
)

LAND_TYPES_DISPLAY = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_TYPES_OLD = ["Grasslands", "NaturalForests", "Savannas", "Shrublands"]
LAND_TYPES_FIG5 = ["GRASSLANDS", "NATURAL FORESTS", "SAVANNAS", "SHRUBLANDS"]
LAND_COLORS = {
    "Grasslands": "#1F77B4",
    "Natural Forests": "#FF7F0E",
    "Savannas": "#2CA02C",
    "Shrublands": "#D62728",
}
VEGETATION = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP"]
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
FOLDER_MAP = {
    "Grasslands": "grass",
    "NaturalForests": "nat",
    "Savannas": "sav",
    "Shrublands": "shrub",
}
DISPLAY_FROM_OLD = {
    "Grasslands": "Grasslands",
    "NaturalForests": "Natural Forests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
CONFIRM_TO_OLD = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "NaturalForests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
MASKS = {
    "Grasslands": LAND_MASK_ROOT / "CA_Grasslands_2021.tif",
    "NaturalForests": LAND_MASK_ROOT / "CA_NaturalForests_2021.tif",
    "Savannas": LAND_MASK_ROOT / "CA_Savannas_2021.tif",
    "Shrublands": LAND_MASK_ROOT / "CA_Shrublands_2021.tif",
}
HISTORY_DIR_NAMES = {
    "Grasslands": "CALIFORNIA GRASSLANDS",
    "NaturalForests": "CALIFORNIA NATURAL FORESTS",
    "Savannas": "CALIFORNIA SAVANNAS",
    "Shrublands": "CALIFORNIA SHRUBLANDS",
}
TEST_DIR_NAMES = {
    "Grasslands": "Grasslands",
    "NaturalForests": "NaturalForests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, REPORT, TOOLS, AUDIT, DATA_OUT]:
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_array(array: np.ndarray, nodata=None) -> np.ndarray:
    data = array.astype("float32", copy=True)
    if nodata is not None:
        data[np.isclose(data, nodata)] = np.nan
    for fill in [-9999, -3.4e38, 32767, -32768, 65535, -32767]:
        data[np.abs(data - fill) < 1.0] = np.nan
    data[np.abs(data) > 1e10] = np.nan
    return data


def parse_date_from_name(path: Path) -> pd.Timestamp:
    import re

    match = re.search(r"(20\d{2})_(\d{2})", path.stem)
    if not match:
        raise ValueError(f"Cannot parse date from {path.name}")
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def reproject_mask_to_source(mask_path: Path, source) -> np.ndarray:
    with rasterio.open(mask_path) as mask_src:
        source_mask = mask_src.read(1).astype("float32")
        dest = np.zeros((source.height, source.width), dtype="float32")
        reproject(
            source=source_mask,
            destination=dest,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=source.transform,
            dst_crs=source.crs,
            resampling=Resampling.nearest,
        )
    return dest > 0.5


def write_masked_test_raster(source_path: Path, land_type: str, mask: np.ndarray, data: np.ndarray) -> Path:
    out_dir = DATA_OUT / "Test Data" / "Masked_LandTypes" / TEST_DIR_NAMES[land_type] / "GNDVI"
    out_dir.mkdir(parents=True, exist_ok=True)
    date = parse_date_from_name(source_path)
    out_path = out_dir / f"GNDVI_MOD09A1_CA_5km_{date.year}_{date.month:02d}_{land_type}.tif"
    with rasterio.open(source_path) as src:
        profile = src.profile.copy()
        profile.update(dtype="float32", nodata=-9999.0, compress="lzw")
        masked = np.where(mask & np.isfinite(data), data, -9999.0).astype("float32")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(masked, 1)
    return out_path


def build_full_previous_gndvi() -> dict[str, pd.Series]:
    completed_table = TABLES / "previous_GNDVI_full_2001_2024_landtype_monthly_means.csv"
    masked_complete = all(
        len(list((DATA_OUT / "Test Data" / "Masked_LandTypes" / TEST_DIR_NAMES[land] / "GNDVI").glob("*.tif"))) == 36
        for land in LAND_TYPES_OLD
    )
    if completed_table.exists() and masked_complete:
        full = pd.read_csv(completed_table, parse_dates=["date"])
        series = {}
        for land_type, group in full.groupby("land_type"):
            s = (
                group.dropna(subset=["mean"])
                .drop_duplicates("date", keep="last")
                .set_index("date")
                .sort_index()["mean"]
                .astype(float)
            )
            s.name = "GNDVI"
            series[land_type] = s
        return series

    history = pd.read_csv(LEGACY_CONFIRMATION)
    history["old_land_type"] = history["land_type"].map(CONFIRM_TO_OLD)
    history["date_ts"] = pd.to_datetime(history["date"].astype(str), format="%Y%m")

    rows = []
    for _, row in history.iterrows():
        rows.append(
            {
                "land_type": row["old_land_type"],
                "date": row["date_ts"],
                "source_period": "historical",
                "source_path": row["selected_path"],
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "valid_pixel_count": int(row["valid_pixel_count"]),
                "valid_pixel_fraction": float(row["valid_pixel_fraction"]),
                "masked_test_raster": "",
                "source_sha256": row.get("hash", ""),
            }
        )

    test_rows = []
    for raster_path in sorted(PREVIOUS_GNDVI_TEST.glob("*.tif")):
        date = parse_date_from_name(raster_path)
        with rasterio.open(raster_path) as src:
            data = clean_array(src.read(1), src.nodata)
            finite = np.isfinite(data)
            for land_type in LAND_TYPES_OLD:
                mask = reproject_mask_to_source(MASKS[land_type], src)
                valid = finite & mask
                valid_count = int(valid.sum())
                masked_path = write_masked_test_raster(raster_path, land_type, mask, data)
                values = data[valid]
                record = {
                    "land_type": land_type,
                    "date": date,
                    "source_period": "test_masked_by_land_type",
                    "source_path": str(raster_path),
                    "mean": float(np.nanmean(values)) if valid_count else np.nan,
                    "std": float(np.nanstd(values, ddof=1)) if valid_count > 1 else np.nan,
                    "min": float(np.nanmin(values)) if valid_count else np.nan,
                    "max": float(np.nanmax(values)) if valid_count else np.nan,
                    "valid_pixel_count": valid_count,
                    "valid_pixel_fraction": float(valid_count / max(1, int(mask.sum()))),
                    "masked_test_raster": str(masked_path),
                    "source_sha256": sha256(raster_path),
                }
                rows.append(record)
                test_rows.append(record)

    full = pd.DataFrame(rows).sort_values(["land_type", "date"])
    full.to_csv(TABLES / "previous_GNDVI_full_2001_2024_landtype_monthly_means.csv", index=False)
    pd.DataFrame(test_rows).to_csv(TABLES / "previous_GNDVI_test_landmask_means_2022_2024.csv", index=False)

    # Copy historical rasters into a forecasting-style data tree.
    for land_type in LAND_TYPES_OLD:
        src_dir = PREVIOUS_GNDVI_ROOT / FOLDER_MAP[land_type] / "GNDVI"
        dst_dir = DATA_OUT / "Data" / HISTORY_DIR_NAMES[land_type] / "GNDVI"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for path in src_dir.glob("*.tif"):
            shutil.copy2(path, dst_dir / path.name)

    series = {}
    for land_type, group in full.groupby("land_type"):
        s = (
            group.dropna(subset=["mean"])
            .drop_duplicates("date", keep="last")
            .set_index("date")
            .sort_index()["mean"]
            .astype(float)
        )
        s.name = "GNDVI"
        series[land_type] = s
    coverage = (
        full.groupby("land_type")
        .agg(start_date=("date", "min"), end_date=("date", "max"), months=("date", "count"))
        .reset_index()
    )
    coverage.to_csv(TABLES / "previous_GNDVI_full_coverage_audit.csv", index=False)
    return series


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def previous_gndvi_standardized_rows(series_by_land: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for old_land, series in series_by_land.items():
        frame = series.rename("original_value").reset_index().rename(columns={"index": "date"})
        frame["date"] = pd.to_datetime(frame["date"])
        frame["year"] = frame["date"].dt.year
        frame["month"] = frame["date"].dt.month
        baseline = frame[(frame["year"] >= 2001) & (frame["year"] <= 2011)]
        month_means = baseline.groupby("month")["original_value"].mean()
        frame["processed_value"] = frame.apply(
            lambda row: row["original_value"] - month_means.loc[int(row["month"])], axis=1
        )
        base_processed = frame.loc[(frame["year"] >= 2001) & (frame["year"] <= 2011), "processed_value"]
        base_mean = float(base_processed.mean())
        base_sd = float(base_processed.std(ddof=1))
        frame["standardized_anomaly"] = (frame["processed_value"] - base_mean) / base_sd
        frame["baseline_anomaly_mean"] = base_mean
        frame["baseline_anomaly_sd"] = base_sd
        frame["land_type"] = DISPLAY_FROM_OLD[old_land]
        frame["variable"] = "GNDVI"
        frame["seasonal_reference"] = "2001-2011 calendar-month mean; confirmed previous/no-QA GNDVI with test land masks"
        frame["standardization_reference"] = "2001-2011 deseasonalized-anomaly mean and standard deviation"
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)[
        [
            "date",
            "year",
            "month",
            "land_type",
            "variable",
            "original_value",
            "processed_value",
            "standardized_anomaly",
            "baseline_anomaly_mean",
            "baseline_anomaly_sd",
            "seasonal_reference",
            "standardization_reference",
        ]
    ]


def build_figure2_table(series_by_land: dict[str, pd.Series]) -> pd.DataFrame:
    current = pd.read_csv(FINAL_PAPER_TABLE, parse_dates=["date"])
    gndvi = previous_gndvi_standardized_rows(series_by_land)
    non_gndvi = current[current["variable"] != "GNDVI"].copy()
    combined = pd.concat([non_gndvi, gndvi], ignore_index=True)
    figure2 = combined[combined["variable"].isin(VEGETATION)].copy()
    figure2 = figure2.sort_values(["variable", "land_type", "date"])
    figure2.to_csv(TABLES / "Figure_2_previous_GNDVI_with_test_monthly_series.csv", index=False)
    return figure2


def save_figure(fig, stem: str, dpi: int = 450, dpi600: bool = False) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    if dpi600:
        fig.savefig(FIGURES / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def render_figure2(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(VEGETATION), 1, figsize=(11.5, 2.15 * len(VEGETATION) + 1.2), sharex=True)
    axes = np.atleast_1d(axes)
    for axis, variable in zip(axes, VEGETATION):
        subset = data[data["variable"] == variable]
        for land_type in LAND_TYPES_DISPLAY:
            series = (
                subset[subset["land_type"] == land_type]
                .set_index("date")["standardized_anomaly"]
                .sort_index()
            )
            axis.plot(series.index, series, color=LAND_COLORS[land_type], linewidth=0.55, alpha=0.24)
            axis.plot(
                series.index,
                series.rolling(3, center=True, min_periods=2).mean(),
                color=LAND_COLORS[land_type],
                linewidth=1.45,
                alpha=0.98,
                label=land_type,
            )
        axis.axhline(0, color="#333333", linewidth=0.65)
        axis.axvspan(pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-31"), color="#F4CCCC", alpha=0.34, linewidth=0)
        axis.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31"), color="#FFF2CC", alpha=0.42, linewidth=0)
        axis.set_ylabel(f"{variable}\n(SD)")
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)

    land_handles = [Line2D([0], [0], color=LAND_COLORS[value], lw=2, label=value) for value in LAND_TYPES_DISPLAY]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=0.7, alpha=0.45, label="Monthly"),
        Line2D([0], [0], color="#555555", lw=2, label="3-month mean"),
    ]
    fig.legend(handles=land_handles + style_handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=6, frameon=False)
    fig.suptitle("Vegetation and productivity standardized anomalies (2001-2024)", y=0.998)
    axes[-1].set_xlabel("Year")
    fig.text(0.5, 0.012, "Shading: 2012-2016 historic drought and 2020-2024 recent period", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_figure(fig, "Figure_2_previous_GNDVI_with_test", dpi=450, dpi600=True)


def render_fig6_font_fixed(values: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    import seaborn as sns

    sns.set_context("paper", font_scale=1.25)
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, top=0.95, bottom=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    im = None
    lags = list(range(-6, 7))
    for idx, land in enumerate(LAND_TYPES_OLD):
        ax = axes[idx]
        sub = values[values["land_type"] == land].copy()
        pairs = sub[["predictor", "target", "pair"]].drop_duplicates()
        matrix = []
        labels = []
        for _, pair_row in pairs.iterrows():
            row_values = sub[(sub["predictor"] == pair_row["predictor"]) & (sub["target"] == pair_row["target"])]
            lookup = dict(zip(row_values["lag"], row_values["ccf"]))
            matrix.append([lookup.get(lag, np.nan) for lag in lags])
            labels.append(f"{pair_row['predictor']}\u2192{pair_row['target']}")
        matrix = np.asarray(matrix, dtype=float)
        im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1, interpolation="bilinear")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels(lags, fontsize=12.5)
        ax.set_xlabel("Lag (months)\nNegative = Predictor LEADS target", fontsize=14, weight="bold")
        ax.set_ylabel("Predictor \u2192 Target (Unique Pairs)", fontsize=14, weight="bold")
        ax.set_title(f"{DISPLAY_FROM_OLD[land]}\nCross-Correlation Function", fontsize=15, weight="bold")
        ax.axvline(6, color="black", linewidth=2.5, linestyle="--", alpha=0.8, label="Lag = 0")
        ax.legend(loc="upper right", fontsize=11.5)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Cross-Correlation Coefficient\n(Red = Positive | Blue = Negative)", fontsize=15, weight="bold")
    cbar.ax.tick_params(labelsize=12.5)
    save_figure(fig, "Figure_6_previous_GNDVI_exact_old_logic_font_fixed", dpi=450, dpi600=True)


def rebuild_fig5_fig6(series_by_land: dict[str, pd.Series]) -> None:
    module = runpy.run_path(str(RECOVERY_SCRIPT))

    def load_previous_gndvi_series(land_type: str):
        return series_by_land[land_type].copy()

    def load_previous_gndvi_array(land_type: str):
        return series_by_land[land_type].dropna().to_numpy(dtype=float)

    module["load_previous_gndvi_series"] = load_previous_gndvi_series
    module["load_previous_gndvi_array"] = load_previous_gndvi_array
    module["fig5_load_variable_array"].__globals__["load_previous_gndvi_array"] = load_previous_gndvi_array
    module["load_timeseries_original_ccf"].__globals__["load_previous_gndvi_series"] = load_previous_gndvi_series

    fig5_results, fig5_values, fig5_inputs = module["fig5_calculate"](module["TEST_DIRS_DOCUMENTS"], gndvi_mode="previous")
    module["render_fig5"](
        fig5_results,
        FIGURES / "Figure_5_previous_GNDVI_with_test_exact_old_logic.png",
        FIGURES / "Figure_5_previous_GNDVI_with_test_exact_old_logic.pdf",
        large_style=True,
    )
    fig5_values.to_csv(TABLES / "Figure_5_previous_GNDVI_with_test_values.csv", index=False)
    fig5_inputs.to_csv(TABLES / "Figure_5_previous_GNDVI_with_test_input_inventory.csv", index=False)

    fig6_m, fig6_labels, fig6_values, fig6_inputs = module["ccf_calculate"](module["TEST_DIRS_DOCUMENTS"], "previous")
    fig6_values.to_csv(TABLES / "Figure_6_previous_GNDVI_with_test_values.csv", index=False)
    fig6_inputs.to_csv(TABLES / "Figure_6_previous_GNDVI_with_test_input_inventory.csv", index=False)
    render_fig6_font_fixed(fig6_values)


def backup_and_copy_to_final_folder() -> None:
    copy_map = {
        FIGURES / "Figure_2_previous_GNDVI_with_test.png": FINAL_FIGURES / "Figure_2_final.png",
        FIGURES / "Figure_2_previous_GNDVI_with_test.pdf": FINAL_FIGURES / "Figure_2_final.pdf",
        FIGURES / "Figure_5_previous_GNDVI_with_test_exact_old_logic.png": FINAL_FIGURES / "Figure_5_final.png",
        FIGURES / "Figure_5_previous_GNDVI_with_test_exact_old_logic.pdf": FINAL_FIGURES / "Figure_5_final.pdf",
        FIGURES / "Figure_6_previous_GNDVI_exact_old_logic_font_fixed.png": FINAL_FIGURES / "Figure_6_final.png",
        FIGURES / "Figure_6_previous_GNDVI_exact_old_logic_font_fixed.pdf": FINAL_FIGURES / "Figure_6_final.pdf",
    }
    rows = []
    backup_dir = PACKAGE / "backups_from_final_figures_folder"
    backup_dir.mkdir(exist_ok=True)
    for source, dest in copy_map.items():
        if dest.exists():
            backup = backup_dir / f"{dest.stem}_backup_before_previous_GNDVI_with_test{dest.suffix}"
            shutil.copy2(dest, backup)
        shutil.copy2(source, dest)
        rows.append({"source": str(source), "destination": str(dest), "destination_exists": dest.exists()})
    pd.DataFrame(rows).to_csv(AUDIT / "final_figure_folder_replacement_manifest.csv", index=False)


def build_manifest() -> None:
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file():
            rows.append({"relative_path": str(path.relative_to(PACKAGE)), "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(AUDIT / "package_file_manifest.csv", index=False)


def write_reports(series_by_land: dict[str, pd.Series]) -> None:
    coverage = []
    for land_type, series in series_by_land.items():
        coverage.append(
            {
                "land_type": land_type,
                "start": series.index.min(),
                "end": series.index.max(),
                "months": len(series),
            }
        )
    coverage_df = pd.DataFrame(coverage)
    coverage_df.to_csv(TABLES / "previous_GNDVI_series_coverage_final.csv", index=False)
    report = [
        "# Final Previous-GNDVI With Test Rebuild Report",
        "",
        "## What Changed",
        "",
        "- Completed the previous/no-QA GNDVI series by masking statewide 2022-2024 test rasters with the CA land-type masks.",
        "- Rebuilt Figure 2 with the completed previous/no-QA GNDVI series and unchanged non-GNDVI data.",
        "- Rebuilt Figure 5 using the exact old Figure 5 logic with completed previous/no-QA GNDVI.",
        "- Rebuilt Figure 6 using the exact old Figure 6 CCF logic with completed previous/no-QA GNDVI and only presentation fixes.",
        "- Copied Figure 2, Figure 5, and Figure 6 into the final main-figure folder after creating backups in this package.",
        "",
        "## GNDVI Coverage",
        "",
        coverage_df.to_markdown(index=False),
        "",
        "## Notes",
        "",
        "- Non-GNDVI Figure 2 rows came unchanged from the accepted final paper standardized anomaly table.",
        "- The 2022-2024 GNDVI rasters were statewide input rasters; land-type means and land-type masked rasters were created using `CA_LandMasks`.",
        "- The forecasting-ready masked GNDVI data tree is saved under `data_for_forecasting`.",
    ]
    (REPORT / "FINAL_previous_GNDVI_with_test_rebuild_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    shutil.copy2(Path(__file__), TOOLS / Path(__file__).name)


def main() -> None:
    ensure_dirs()
    apply_publication_style()
    series_by_land = build_full_previous_gndvi()
    fig2 = build_figure2_table(series_by_land)
    render_figure2(fig2)
    rebuild_fig5_fig6(series_by_land)
    backup_and_copy_to_final_folder()
    write_reports(series_by_land)
    build_manifest()


if __name__ == "__main__":
    main()
