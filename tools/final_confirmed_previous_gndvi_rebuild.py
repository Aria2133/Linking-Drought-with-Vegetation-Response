"""Build final confirmed previous-GNDVI figure/audit package.

This script intentionally writes only to the new package requested by the user.
It does not modify the manuscript figure folders. It preserves the accepted
Figure 2/6 definitions as closely as possible while replacing only the Figure 2
GNDVI time series with the confirmed previous/no-QA source.
"""

from __future__ import annotations

import hashlib
import itertools
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PACKAGE = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"
    r"\final_confirmed_previous_GNDVI_rebuild"
)
FIGURES = PACKAGE / "figures"
TABLES = PACKAGE / "tables"
REPORT = PACKAGE / "report"
TOOLS = PACKAGE / "tools"
AUDIT = PACKAGE / "audit"

FINAL_FIG_ROOT = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_revision_final_figures_package"
    r"\1_paper_main_figures"
)
PAPER_ROOT = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"
    r"\final_paper_figures_corrected"
)
PAPER_TABLES = PAPER_ROOT / "tables"
PAPER_SCRIPTS = PAPER_ROOT / "scripts"
LEGACY_GNDVI = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"
    r"\legacy_gndvi_sensitivity_package\tables\legacy_gndvi_source_confirmation.csv"
)
FORECAST_ROOT = Path(
    r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses"
    r"\05_forecasting_raster_reviewer_safe_candidate"
)

LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#1F77B4",
    "Natural Forests": "#FF7F0E",
    "Savannas": "#2CA02C",
    "Shrublands": "#D62728",
}
VEGETATION = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP"]
LEGACY_VARIABLE_ORDER = [
    "GNDVI",
    "GOSIF",
    "GPP",
    "LST",
    "NDVI",
    "NIRv",
    "SPEI",
    "TWS",
    "VPD",
]
LAND_MAP = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "Natural Forests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}


def ensure_dirs() -> None:
    for directory in [FIGURES, TABLES, REPORT, TOOLS, AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def save_all(fig, stem: str, dpi: int = 450, also_600: bool = False) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    if also_600:
        fig.savefig(FIGURES / f"{stem}_600dpi.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def compute_previous_gndvi_standardized() -> pd.DataFrame:
    legacy = pd.read_csv(LEGACY_GNDVI)
    legacy["land_type"] = legacy["land_type"].map(LAND_MAP)
    legacy["date"] = pd.to_datetime(legacy["date"].astype(str), format="%Y%m")
    rows: list[pd.DataFrame] = []

    for land_type, group in legacy.groupby("land_type"):
        group = group.sort_values("date").copy()
        group["year"] = group["date"].dt.year
        group["month"] = group["date"].dt.month
        baseline = group[(group["year"] >= 2001) & (group["year"] <= 2011)]
        month_means = baseline.groupby("month")["mean"].mean()
        group["original_value"] = group["mean"].astype(float)
        group["processed_value"] = group.apply(
            lambda row: row["original_value"] - month_means.loc[int(row["month"])],
            axis=1,
        )
        base_processed = group.loc[
            (group["year"] >= 2001) & (group["year"] <= 2011), "processed_value"
        ]
        baseline_mean = float(base_processed.mean())
        baseline_sd = float(base_processed.std(ddof=1))
        group["standardized_anomaly"] = (
            group["processed_value"] - baseline_mean
        ) / baseline_sd
        group["baseline_anomaly_mean"] = baseline_mean
        group["baseline_anomaly_sd"] = baseline_sd
        group["variable"] = "GNDVI"
        group["seasonal_reference"] = (
            "2001-2011 calendar-month mean; confirmed previous/no-QA GNDVI"
        )
        group["standardization_reference"] = (
            "2001-2011 deseasonalized-anomaly mean and standard deviation; "
            "confirmed previous/no-QA GNDVI"
        )
        rows.append(
            group[
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
                    "selected_path",
                    "valid_pixel_count",
                    "valid_pixel_fraction",
                    "source_type",
                ]
            ]
        )

    return pd.concat(rows, ignore_index=True)


def build_figure2_table() -> pd.DataFrame:
    current = pd.read_csv(
        PAPER_TABLES / "paper_standardized_anomaly_series.csv",
        parse_dates=["date"],
    )
    previous = compute_previous_gndvi_standardized()

    # Preserve all non-GNDVI rows from the accepted final paper data table.
    non_gndvi = current[current["variable"] != "GNDVI"].copy()
    gndvi_template = current[current["variable"] == "GNDVI"].copy()
    gndvi_template[
        [
            "original_value",
            "processed_value",
            "standardized_anomaly",
            "baseline_anomaly_mean",
            "baseline_anomaly_sd",
        ]
    ] = np.nan
    gndvi_template["seasonal_reference"] = (
        "missing confirmed previous/no-QA GNDVI after 2021"
    )
    gndvi_template["standardization_reference"] = (
        "missing confirmed previous/no-QA GNDVI after 2021"
    )

    previous_core = previous[
        [
            "date",
            "land_type",
            "original_value",
            "processed_value",
            "standardized_anomaly",
            "baseline_anomaly_mean",
            "baseline_anomaly_sd",
            "seasonal_reference",
            "standardization_reference",
        ]
    ].copy()
    previous_core["variable"] = "GNDVI"
    merged_gndvi = gndvi_template.drop(
        columns=[
            "original_value",
            "processed_value",
            "standardized_anomaly",
            "baseline_anomaly_mean",
            "baseline_anomaly_sd",
            "seasonal_reference",
            "standardization_reference",
        ]
    ).merge(
        previous_core,
        on=["date", "land_type", "variable"],
        how="left",
    )

    combined = pd.concat([non_gndvi, merged_gndvi], ignore_index=True)
    combined = combined.sort_values(["variable", "land_type", "date"])
    figure2 = combined[combined["variable"].isin(VEGETATION)].copy()
    figure2.to_csv(TABLES / "Figure_2_confirmed_previous_GNDVI_monthly_series.csv", index=False)

    previous.to_csv(TABLES / "confirmed_previous_GNDVI_standardized_series.csv", index=False)
    coverage = (
        previous.groupby("land_type")
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            observed_months=("date", "count"),
            min_valid_fraction=("valid_pixel_fraction", "min"),
            median_valid_fraction=("valid_pixel_fraction", "median"),
            max_valid_fraction=("valid_pixel_fraction", "max"),
        )
        .reset_index()
    )
    coverage["missing_2022_2024_months"] = 36
    coverage["source_folder"] = r"C:\Users\LOQ\OneDrive\Desktop\Previous GNDVI"
    coverage.to_csv(TABLES / "confirmed_previous_GNDVI_coverage_audit.csv", index=False)
    return figure2


def render_figure2(data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(VEGETATION),
        1,
        figsize=(11.5, 2.15 * len(VEGETATION) + 1.2),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, variable in zip(axes, VEGETATION):
        subset = data[data["variable"] == variable]
        for land_type in LAND_TYPES:
            series = (
                subset[subset["land_type"] == land_type]
                .set_index("date")["standardized_anomaly"]
                .sort_index()
            )
            axis.plot(
                series.index,
                series,
                color=LAND_COLORS[land_type],
                linewidth=0.55,
                alpha=0.24,
            )
            axis.plot(
                series.index,
                series.rolling(3, center=True, min_periods=2).mean(),
                color=LAND_COLORS[land_type],
                linewidth=1.45,
                alpha=0.98,
                label=land_type,
            )
        axis.axhline(0, color="#333333", linewidth=0.65)
        axis.axvspan(
            pd.Timestamp("2012-01-01"),
            pd.Timestamp("2016-12-31"),
            color="#F4CCCC",
            alpha=0.34,
            linewidth=0,
        )
        axis.axvspan(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2024-12-31"),
            color="#FFF2CC",
            alpha=0.42,
            linewidth=0,
        )
        axis.set_ylabel(f"{variable}\n(SD)")
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)

    land_handles = [
        Line2D([0], [0], color=LAND_COLORS[value], lw=2, label=value)
        for value in LAND_TYPES
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=0.7, alpha=0.45, label="Monthly"),
        Line2D([0], [0], color="#555555", lw=2, label="3-month mean"),
    ]
    fig.legend(
        handles=land_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
    )
    fig.suptitle("Vegetation and productivity standardized anomalies (2001-2024)", y=0.998)
    axes[-1].set_xlabel("Year")
    fig.text(
        0.5,
        0.012,
        "Shading: 2012-2016 historic drought and 2020-2024 recent period",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_all(fig, "Figure_2_confirmed_previous_GNDVI", dpi=450, also_600=True)


def copy_figure5() -> None:
    for ext in ["png", "pdf"]:
        source = FINAL_FIG_ROOT / f"Figure_5_final.{ext}"
        if source.exists():
            shutil.copy2(source, FIGURES / f"Figure_5_accepted_as_is.{ext}")


def render_figure6_from_accepted_table() -> None:
    values = pd.read_csv(PAPER_TABLES / "figure_6_cross_correlation_lag_values.csv")
    values["land_type"] = values["land_type"].replace({"NaturalForests": "Natural Forests"})
    values.to_csv(TABLES / "Figure_6_accepted_data_font_fixed_values.csv", index=False)
    lags = np.arange(-6, 7)
    pairs = list(itertools.combinations(LEGACY_VARIABLE_ORDER, 2))
    pair_labels = [f"{first}\u2192{second}" for first, second in pairs]

    fig = plt.figure(figsize=(24, 24))
    grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, top=0.95, bottom=0.12)
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(2)]
    image = None

    for axis, land_type in zip(axes, LAND_TYPES):
        matrix = np.full((len(pairs), len(lags)), np.nan)
        sub = values[values["land_type"] == land_type]
        for row, (first, second) in enumerate(pairs):
            row_values = sub[(sub["first_series"] == first) & (sub["second_series"] == second)]
            lookup = dict(zip(row_values["lag_months"], row_values["pearson_r"]))
            for col, lag in enumerate(lags):
                matrix[row, col] = lookup.get(int(lag), np.nan)
        image = axis.imshow(
            matrix,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            interpolation="bilinear",
        )
        axis.set_yticks(range(len(pair_labels)))
        axis.set_yticklabels(pair_labels, fontsize=9.5)
        axis.set_xticks(range(len(lags)), lags)
        axis.set_xticklabels(lags, fontsize=12.5)
        axis.set_xlabel(
            "Lag (months)\nNegative = Predictor LEADS target",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_ylabel(
            "Predictor \u2192 Target (Unique Pairs)",
            fontsize=14,
            fontweight="bold",
        )
        axis.set_title(
            f"{land_type}\nCross-Correlation Function",
            fontsize=15,
            fontweight="bold",
        )
        axis.axvline(
            np.where(lags == 0)[0][0],
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
            label="Lag = 0",
        )
        axis.legend(loc="upper right", fontsize=11.5)
        axis.grid(color="#D0D0D0", linewidth=0.55, alpha=0.65)

    colorbar_axis = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(
        "Cross-Correlation Coefficient\n(Red = Positive | Blue = Negative)",
        fontsize=15,
        fontweight="bold",
    )
    colorbar.ax.tick_params(labelsize=12.5)
    fig.text(
        0.5,
        0.012,
        "Empirical lag associations are descriptive and do not establish causality.",
        ha="center",
        fontsize=11,
    )
    save_all(fig, "Figure_6_accepted_data_font_fixed", dpi=450, also_600=True)


def build_forecasting_dependency_audit() -> None:
    rows = []
    table_dir = FORECAST_ROOT / "tables"
    for csv_name in [
        "reviewer_safe_all_pixel_metrics.csv",
        "reviewer_safe_all_monthly_aggregated_metrics.csv",
        "reviewer_safe_baseline_pixel_metrics.csv",
        "reviewer_safe_baseline_monthly_aggregated_metrics.csv",
        "reviewer_safe_lstm_pixel_metrics.csv",
        "reviewer_safe_lstm_monthly_aggregated_metrics.csv",
        "reviewer_safe_xgboost_pixel_metrics.csv",
        "reviewer_safe_xgboost_monthly_aggregated_metrics.csv",
        "reviewer_safe_remaining_models_pixel_metrics.csv",
        "reviewer_safe_remaining_models_monthly_aggregated_metrics.csv",
    ]:
        path = table_dir / csv_name
        if not path.exists():
            rows.append(
                {
                    "table": csv_name,
                    "exists": False,
                    "gndvi_rows": np.nan,
                    "non_gndvi_rows": np.nan,
                    "models_with_gndvi": "",
                    "rerun_status": "not_run",
                    "reason": "source table missing",
                }
            )
            continue
        data = pd.read_csv(path)
        variable_col = "variable" if "variable" in data.columns else None
        gndvi = data[data[variable_col] == "GNDVI"] if variable_col else pd.DataFrame()
        models = (
            ";".join(sorted(gndvi["model"].dropna().astype(str).unique()))
            if "model" in gndvi.columns and not gndvi.empty
            else ""
        )
        rows.append(
            {
                "table": csv_name,
                "exists": True,
                "gndvi_rows": len(gndvi),
                "non_gndvi_rows": len(data) - len(gndvi),
                "models_with_gndvi": models,
                "rerun_status": "blocked_not_run",
                "reason": (
                    "Confirmed previous/no-QA GNDVI inventory covers 2001-2021 only; "
                    "reviewer-safe folds require GNDVI observations through 2024. "
                    "Exact rerun would require producing/installing 2022-2024 previous/no-QA "
                    "GNDVI rasters before rerunning only GNDVI-dependent tasks."
                ),
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "forecasting_GNDVI_dependency_audit.csv", index=False)

    current_metrics = table_dir / "reviewer_safe_all_pixel_metrics.csv"
    if current_metrics.exists():
        current = pd.read_csv(current_metrics)
        gndvi_current = current[current["variable"] == "GNDVI"].copy()
        gndvi_current["confirmed_previous_GNDVI_rerun_status"] = "not_run_blocked_by_2022_2024_missing_source"
        gndvi_current.to_csv(TABLES / "forecasting_current_GNDVI_metrics_for_before_after_reference.csv", index=False)


def write_reports(figure2: pd.DataFrame) -> None:
    gndvi = figure2[figure2["variable"] == "GNDVI"].copy()
    observed = gndvi[gndvi["standardized_anomaly"].notna()]
    missing = gndvi[gndvi["standardized_anomaly"].isna()]
    coverage = observed.groupby("land_type")["date"].agg(["min", "max", "count"]).reset_index()

    (REPORT / "Figure_2_confirmed_previous_GNDVI_audit.md").write_text(
        "# Figure 2 Confirmed Previous-GNDVI Audit\n\n"
        "- Source render logic: `final_paper_figures_corrected/scripts/01_render_figures_1_3.py`, "
        "specifically the accepted `render_time_series` style and panel structure.\n"
        "- GNDVI source folder: `C:\\Users\\LOQ\\OneDrive\\Desktop\\Previous GNDVI` via "
        "`legacy_gndvi_source_confirmation.csv`.\n"
        "- GNDVI source decision: confirmed previous/no-QA GNDVI.\n"
        "- Non-GNDVI inputs: copied unchanged from the accepted final paper standardized anomaly table.\n"
        "- Visual style: preserved from the accepted Figure 2 renderer; only output filenames differ.\n"
        f"- Observed confirmed previous/no-QA GNDVI rows: {len(observed)}.\n"
        f"- Missing confirmed previous/no-QA GNDVI rows in the 2001-2024 Figure 2 frame: {len(missing)} "
        "(all are 2022-2024 months because the confirmed previous/no-QA inventory stops at 2021-12).\n\n"
        "## GNDVI Coverage By Land Type\n\n"
        + coverage.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )

    (REPORT / "Figure_5_accepted_as_is_audit.md").write_text(
        "# Figure 5 Accepted As-Is Audit\n\n"
        "- Figure 5 was accepted as-is and copied from the current final manuscript figure package.\n"
        "- It was not recalculated in this task.\n"
        "- It uses the recovered/accepted original raw monthly correlation definition selected by the user.\n"
        "- No new GNDVI substitution was performed for Figure 5 in this task.\n",
        encoding="utf-8",
    )

    (REPORT / "Figure_6_font_fix_audit.md").write_text(
        "# Figure 6 Font/Label Fix Audit\n\n"
        "- Figure 6 was rendered from the accepted existing CCF value table, not recalculated from raster data.\n"
        "- Only presentation edits were made: font sizes were increased slightly and the title label "
        "`NaturalForests` was changed to `Natural Forests`.\n"
        "- CCF values were not changed.\n"
        "- Pair order was not changed.\n"
        "- Lag range remained -6 to +6 months.\n"
        "- The dashed lag-zero line and `RdBu_r` color scale from -1 to 1 were preserved.\n"
        "- No strict-QA GNDVI source was introduced; this render used the accepted CCF table directly.\n",
        encoding="utf-8",
    )

    (REPORT / "FINAL_confirmed_previous_GNDVI_forecasting_rerun_audit.md").write_text(
        "# Forecasting / DL Confirmed Previous-GNDVI Rerun Audit\n\n"
        "No forecasting, machine-learning, or deep-learning models were rerun in this package.\n\n"
        "## Reason\n\n"
        "The confirmed previous/no-QA GNDVI source currently verified in "
        "`legacy_gndvi_source_confirmation.csv` covers 2001-01 through 2021-12 for all four "
        "land types. The reviewer-safe forecasting benchmark uses four rolling-origin folds with "
        "test periods through 2024-12, including folds that require GNDVI observations in 2022, "
        "2023, and 2024. Running the GNDVI target benchmark now would either fail or require "
        "changing fold coverage, imputation behavior, or the GNDVI source. Those changes would "
        "violate the user's rule that architecture, folds, preprocessing, target definitions, and "
        "non-GNDVI results remain unchanged.\n\n"
        "## Minimum Required Corrective Action\n\n"
        "To complete an exact confirmed-previous-GNDVI forecasting rerun, first produce or install "
        "land-type-specific previous/no-QA GNDVI rasters for 2022-01 through 2024-12 on the same "
        "grid and with the same date naming conventions. Then rerun only GNDVI-dependent tasks: "
        "Persistence, seasonal persistence, monthly climatology, SARIMA where applicable at monthly "
        "aggregate level, XGBoost, LSTM, BiLSTM, TCN, N-BEATS, and N-HiTS for the GNDVI target under "
        "the existing four folds/seeds/settings. Non-GNDVI tasks can remain unchanged.\n",
        encoding="utf-8",
    )

    final_report = (
        "# Final Confirmed Previous-GNDVI Rebuild Report\n\n"
        "## Summary\n\n"
        "- Figure 2 was regenerated using confirmed previous/no-QA GNDVI for the GNDVI panel.\n"
        "- Figure 5 was accepted as-is and copied without recalculation.\n"
        "- Figure 6 was accepted scientifically and only font/title formatting was changed.\n"
        "- Forecasting / DL outputs involving GNDVI were not rerun because the confirmed previous/no-QA "
        "GNDVI inventory stops at 2021-12 while the reviewer-safe folds require 2022-2024.\n"
        "- No model architecture, hyperparameters, split design, or visualization design was changed.\n\n"
        "## Safety Assessment\n\n"
        "The figure outputs in this package are safe for manuscript review with one caveat: Figure 2's "
        "GNDVI series uses confirmed previous/no-QA GNDVI only through 2021-12. The manuscript should "
        "not imply that previous/no-QA GNDVI values were available for 2022-2024 unless those rasters "
        "are later produced. The forecasting update is not complete until 2022-2024 previous/no-QA "
        "GNDVI is available and the GNDVI-only benchmark is rerun under the unchanged reviewer-safe "
        "settings.\n"
    )
    (REPORT / "FINAL_confirmed_previous_GNDVI_rebuild_report.md").write_text(final_report, encoding="utf-8")


def copy_tools() -> None:
    script_path = Path(__file__).resolve()
    shutil.copy2(script_path, TOOLS / script_path.name)
    for name in ["01_render_figures_1_3.py", "02_render_figures_5_6.py", "00_prepare_paper_data.py", "paper_figure_utils.py"]:
        source = PAPER_SCRIPTS / name
        if source.exists():
            shutil.copy2(source, TOOLS / f"source_{name}")


def build_manifest() -> None:
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(PACKAGE)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).to_csv(AUDIT / "package_file_manifest.csv", index=False)


def main() -> None:
    ensure_dirs()
    apply_publication_style()
    figure2 = build_figure2_table()
    render_figure2(figure2)
    copy_figure5()
    render_figure6_from_accepted_table()
    build_forecasting_dependency_audit()
    write_reports(figure2)
    copy_tools()
    build_manifest()


if __name__ == "__main__":
    main()
