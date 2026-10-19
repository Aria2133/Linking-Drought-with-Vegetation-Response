from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


ROOT = Path(
    r"<PROJECT_ROOT>\ECOINF_supplementary_information_final_separate_figures"
)
PNG_DIR = ROOT / "figures_png"
PDF_DIR = ROOT / "figures_pdf"
CAPTION_DIR = ROOT / "captions"
MAP_DIR = ROOT / "source_mapping"
REPORT_DIR = ROOT / "report"
SUPP_DIR = ROOT / "supplementary_pdf"

DESKTOP = Path(r"<PROJECT_ROOT>")
EXAMPLE_PDF = DESKTOP / "supplementary.pdf"
POLISHED = DESKTOP / "ECOINF_compact_supplementary_figures_FINAL_POLISHED"
COMPACT = DESKTOP / "ECOINF_compact_supplementary_figures_final"
PANEL_DIR = COMPACT / "source_mapping"
FINAL_FIGURES_PACKAGE = DESKTOP / "ECOINF_revision_final_figures_package"
SOURCE_TABLES = FINAL_FIGURES_PACKAGE / "4_source_tables_used"
GAP_TABLES = DESKTOP / "GapFilling_Comparison_Revision" / "tables"
FULL_MONTHLY_METRICS = (
    DESKTOP
    / "major_revision_analyses"
    / "05_forecasting_raster_reviewer_safe_candidate"
    / "tables"
    / "reviewer_safe_all_monthly_aggregated_metrics.csv"
)

MANUSCRIPT_TITLE = (
    "Integrating Multi-Sensor Earth Observation and Deep Learning for Drought "
    "Prediction and Vegetation Resilience Assessment in California"
)
AUTHOR_LINE = "Aria Ahmadi"

LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#007C91",
    "Natural Forests": "#2E7D32",
    "Savannas": "#E58B00",
    "Shrublands": "#B23A67",
}
MODEL_COLORS = {
    "Persistence": "#6F6F6F",
    "Seasonal persistence": "#A3A3A3",
    "Monthly climatology": "#B8A900",
    "SARIMA": "#9A5D45",
    "LSTM": "#2F6DAE",
    "BiLSTM": "#8556A8",
    "TCN": "#3B8C4A",
    "N-BEATS": "#F28E2B",
    "N-HiTS": "#D84A4A",
    "XGBoost": "#00A6A6",
}
PERIOD_ORDER = [
    "Baseline 2001-2011",
    "Historic drought 2012-2016",
    "Recent period 2020-2024",
]
METHOD_ORDER = [
    "current_monthwise_cross_year",
    "seasonal_mean",
    "linear_time",
    "polynomial_regression",
    "spline_interpolation",
    "kalman_state_space",
]
METHOD_LABELS = {
    "current_monthwise_cross_year": "Calendar\nmonth",
    "seasonal_mean": "Seasonal\nmean",
    "linear_time": "Linear\ntime",
    "polynomial_regression": "Trend +\nseason",
    "spline_interpolation": "Spline",
    "kalman_state_space": "State\nspace",
}
METHOD_LABELS_FLAT = {
    "current_monthwise_cross_year": "Calendar month",
    "seasonal_mean": "Seasonal mean",
    "linear_time": "Linear time",
    "polynomial_regression": "Trend + season",
    "spline_interpolation": "Spline",
    "kalman_state_space": "State space",
}

SOURCE_MAP_ROWS: list[dict[str, str]] = []
READ_FILES: list[Path] = []


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def register_read(path: Path) -> None:
    READ_FILES.append(path)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    register_read(path)
    return pd.read_csv(path)


def panel_csv(name: str) -> pd.DataFrame:
    return read_csv(PANEL_DIR / name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def add_source_map(
    fig: str,
    panel: str,
    plotted: str,
    source: str,
    concern: str,
    notes: str,
) -> None:
    SOURCE_MAP_ROWS.append(
        {
            "supplementary_figure": fig,
            "panel": panel,
            "source_compact_figure": source.split("|")[0].strip(),
            "source_table_or_file": source,
            "reviewer_concern_addressed": concern,
            "notes": notes,
            "plotted_content": plotted,
        }
    )


def add_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str = ".2f",
    annotations: np.ndarray | None = None,
    colorbar_label: str | None = None,
    rotate_x: float = 0,
    annotation_size: float = 8.2,
) -> None:
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=7)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    plt.setp(
        ax.get_xticklabels(),
        rotation=rotate_x,
        ha="right" if rotate_x else "center",
        rotation_mode="anchor",
    )
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "NA" if np.isnan(value) else format(value, fmt)
            if annotations is not None:
                label = str(annotations[row, col])
            rgba = mpl.colormaps[cmap](norm(value)) if not np.isnan(value) else (1, 1, 1, 1)
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                col,
                row,
                label,
                ha="center",
                va="center",
                fontsize=annotation_size,
                color="white" if lum < 0.50 else "#202020",
            )
    if colorbar_label:
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.040, pad=0.022, aspect=25)
        colorbar.ax.tick_params(labelsize=8)
        colorbar.set_label(colorbar_label, fontsize=9)


def save_fig(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = PNG_DIR / f"{stem}.png"
    pdf = PDF_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return png, pdf


def parse_yyyymm(value: int | str) -> float:
    text = str(int(value))
    year = int(text[:4])
    month = int(text[4:6])
    return year + (month - 1) / 12.0


def build_s1() -> tuple[Path, Path]:
    period = panel_csv("S1a_period_response_panel_data.csv")
    row_order = [
        f"{land.replace('Natural Forests', 'Natural forests')} | {period_name}"
        for land in LAND_ORDER
        for period_name in ("baseline", "historic", "recent")
    ]
    responses = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
    values = (
        period.pivot(index="row_label", columns="response", values="pearson_r")
        .reindex(index=row_order, columns=responses)
    )
    sig = (
        period.pivot(index="row_label", columns="response", values="ci_excludes_zero")
        .reindex(index=row_order, columns=responses)
    )
    annotations = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            annotations[i, j] = f"{values.iloc[i, j]:.2f}{'*' if bool(sig.iloc[i, j]) else ''}"

    fig, ax = plt.subplots(figsize=(8.9, 6.3), layout="constrained")
    add_heatmap(
        ax,
        values.to_numpy(),
        row_order,
        responses,
        "EWBI-response association",
        "RdBu_r",
        -1,
        1,
        annotations=annotations,
        colorbar_label="Pearson r",
        rotate_x=25,
        annotation_size=8.0,
    )
    add_source_map(
        "Figure S1",
        "single panel",
        "EWBI-response Pearson correlations by land type and drought period",
        "Figure_S1_EWBI_validation_robustness_FINAL | S1a_period_response_panel_data.csv; ewbi_period_stratified_response_correlations_bootstrap_ci.csv",
        "EWBI validation against vegetation and productivity responses",
        "Asterisks mark moving-block/bootstrap 95% confidence intervals excluding zero.",
    )
    return save_fig(fig, "Figure_S1_EWBI_response_association")


def build_s2() -> tuple[Path, Path]:
    component = panel_csv("S1b_component_omission_panel_data.csv")
    weighting = panel_csv("S1c_weighting_panel_data.csv")
    separation = panel_csv("S1d_period_separation_panel_data.csv")
    fig = plt.figure(figsize=(12.7, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 1.18])

    ax_a = fig.add_subplot(gs[0, 0])
    comps = ["SPEI", "TWS", "VPD", "LST"]
    x = np.arange(len(comps))
    styles = [
        ("Individual component", "Component alone", "#2166AC", "o", -0.09),
        ("EWBI after omission", "EWBI omitting component", "#B2182B", "s", 0.09),
    ]
    for series, label, color, marker, offset in styles:
        subset = component[component["series"] == series].set_index("component").reindex(comps)
        ax_a.errorbar(
            x + offset,
            subset["mean"],
            yerr=subset["std"],
            fmt=marker,
            ms=5.8,
            lw=1.2,
            capsize=3,
            color=color,
            label=label,
        )
    ax_a.axhline(0, color="#4A4A4A", lw=0.8)
    ax_a.set_xticks(x, comps)
    ax_a.set_ylabel("Mean Pearson r")
    ax_a.set_title("Component sensitivity")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "(a)")

    ax_b = fig.add_subplot(gs[0, 1])
    weights = ["Equal", "PCA", "Regression"]
    for land in LAND_ORDER:
        subset = weighting[weighting["land_type"] == land].set_index("weighting").reindex(weights)
        ax_b.errorbar(
            np.arange(3),
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            ms=5.0,
            capsize=2.8,
            lw=1.2,
            color=LAND_COLORS[land],
            label=land,
        )
    ax_b.axhline(0, color="#4A4A4A", lw=0.8)
    ax_b.set_xticks(np.arange(3), weights)
    ax_b.set_ylabel("Mean Pearson r")
    ax_b.set_title("Weighting robustness")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, ncol=2, fontsize=7.6, loc="lower center")
    panel_label(ax_b, "(b)")

    ax_c = fig.add_subplot(gs[0, 2])
    offsets = np.linspace(-0.21, 0.21, len(LAND_ORDER))
    labels = ["Baseline\n2001-2011", "Historic drought\n2012-2016", "Recent drought\n2020-2024"]
    for offset, land in zip(offsets, LAND_ORDER):
        subset = separation[separation["land_type"] == land].set_index("period").reindex(PERIOD_ORDER)
        y = subset["mean_ewbi"].to_numpy()
        err = np.vstack([y - subset["ci95_low"].to_numpy(), subset["ci95_high"].to_numpy() - y])
        ax_c.errorbar(
            np.arange(3) + offset,
            y,
            yerr=err,
            fmt="o",
            ms=5.0,
            capsize=2.8,
            lw=1.15,
            color=LAND_COLORS[land],
        )
    ax_c.axhline(0, color="#4A4A4A", lw=0.8)
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.set_ylabel("Mean EWBI forcing score")
    ax_c.set_title("Drought-period separation")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)")

    for panel, plotted, source, concern, notes in [
        ("(a)", "Component-alone and EWBI omission associations", "Figure_S1_EWBI_validation_robustness_FINAL | S1b_component_omission_panel_data.csv", "EWBI component sensitivity", "Error bars show sample SD across land-type-response tasks."),
        ("(b)", "Alternative weighting response associations", "Figure_S1_EWBI_validation_robustness_FINAL | S1c_weighting_panel_data.csv", "EWBI weighting robustness", "Equal, PCA-derived, and regression-derived weights are compared."),
        ("(c)", "Drought-period EWBI separation", "Figure_S1_EWBI_validation_robustness_FINAL | S1d_period_separation_panel_data.csv", "Drought-period discrimination", "Error bars show bootstrap 95% confidence intervals."),
    ]:
        add_source_map("Figure S2", panel, plotted, source, concern, notes)
    return save_fig(fig, "Figure_S2_EWBI_component_weighting_robustness")


def build_s3() -> tuple[Path, Path]:
    scale = panel_csv("S2a_tws_scale_panel_data.csv")
    resampling = panel_csv("S2b_resampling_panel_data.csv")
    comparison_labels = {
        "TWS nearest 5 km versus bilinear 5 km": "Nearest-\nbilinear",
        "TWS bilinear 5 km versus coarse fractional-overlap": "Bilinear-\ncoarse",
        "TWS nearest 5 km versus coarse fractional-overlap": "Nearest-\ncoarse",
        "TWS reverse block average versus coarse fractional-overlap": "Reverse-\ncoarse",
    }
    resampling["comparison_clean"] = resampling["comparison"].map(comparison_labels)
    comparison_order = list(comparison_labels.values())
    resamp = (
        resampling.pivot(index="land_type", columns="comparison_clean", values="normalized_rmse_by_reference_sd")
        .reindex(index=LAND_ORDER, columns=comparison_order)
    )
    fig = plt.figure(figsize=(12.2, 4.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 0.75, 1.75])
    ax_a = fig.add_subplot(gs[0, 0])
    add_heatmap(
        ax_a,
        scale[["pearson_r"]].to_numpy(),
        LAND_ORDER,
        ["r"],
        "TWS scale: Pearson r",
        "YlGn",
        0.95,
        1.0,
        fmt=".3f",
        annotation_size=8.2,
    )
    panel_label(ax_a, "(a)", x=-0.20)
    ax_b = fig.add_subplot(gs[0, 1])
    add_heatmap(
        ax_b,
        scale[["normalized_rmse_by_reference_sd"]].to_numpy(),
        LAND_ORDER,
        ["nRMSE"],
        "TWS scale: nRMSE",
        "YlOrRd",
        0,
        max(0.02, scale["normalized_rmse_by_reference_sd"].max()),
        fmt=".3f",
        annotation_size=8.2,
    )
    ax_b.set_yticklabels([])
    panel_label(ax_b, "(b)", x=-0.20)
    ax_c = fig.add_subplot(gs[0, 2])
    add_heatmap(
        ax_c,
        resamp.to_numpy(),
        LAND_ORDER,
        comparison_order,
        "Resampling sensitivity",
        "YlOrRd",
        0,
        max(0.03, float(np.nanmax(resamp.to_numpy()))),
        fmt=".3f",
        colorbar_label="nRMSE / reference SD",
        annotation_size=8.0,
    )
    panel_label(ax_c, "(c)", x=-0.10)
    add_source_map("Figure S3", "(a)", "TWS scale Pearson r", "Figure_S2_preprocessing_harmonization_sensitivity_FINAL | S2a_tws_scale_panel_data.csv", "TWS scale and harmonization uncertainty", "Pearson r values are rounded to three decimals.")
    add_source_map("Figure S3", "(b)", "TWS scale normalized RMSE", "Figure_S2_preprocessing_harmonization_sensitivity_FINAL | S2a_tws_scale_panel_data.csv", "TWS scale and harmonization uncertainty", "High agreement supports stable temporal summaries, not independent fine-scale TWS data.")
    add_source_map("Figure S3", "(c)", "TWS resampling-method normalized disagreement", "Figure_S2_preprocessing_harmonization_sensitivity_FINAL | S2b_resampling_panel_data.csv", "Resampling-method sensitivity", "The common grid is interpreted as an analytical overlay.")
    return save_fig(fig, "Figure_S3_TWS_scale_resampling_sensitivity")


def build_s4() -> tuple[Path, Path]:
    validation = panel_csv("S2c_gapfill_panel_data.csv")
    decision = read_csv(SOURCE_TABLES / "gapfill_decision_summary.csv")
    ewbi = read_csv(GAP_TABLES / "ewbi_gapfill_sensitivity_metrics.csv")
    state = read_csv(SOURCE_TABLES / "state_gapfill_sensitivity.csv")

    fig = plt.figure(figsize=(13.0, 4.8), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.05, 1.05])

    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(METHOD_ORDER))
    for variable, offset, color, marker in [("LST", -0.10, "#2F6DAE", "o"), ("TWS", 0.10, "#D84A4A", "s")]:
        subset = validation[validation["variable"] == variable].set_index("method").reindex(METHOD_ORDER)
        ax_a.errorbar(
            x + offset,
            subset["mean"],
            yerr=subset["std"],
            fmt=marker,
            ms=5.3,
            capsize=2.8,
            lw=1.15,
            color=color,
            label=variable,
        )
    ax_a.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER])
    ax_a.set_ylabel("Artificial-gap nRMSE")
    ax_a.set_title("Artificial-gap validation")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False)
    panel_label(ax_a, "(a)")

    ax_b = fig.add_subplot(gs[0, 1])
    ordered = decision.set_index("method").reindex(METHOD_ORDER)
    ax_b.plot(
        np.arange(len(METHOD_ORDER)),
        ordered["ewbi_minimum_pearson_r_vs_current"],
        marker="o",
        color="#2166AC",
        label="Minimum EWBI r",
    )
    ax_b.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
    ax_b.set_ylim(0.94, 1.005)
    ax_b.set_ylabel("Worst-case Pearson r")
    ax_b.set_title("EWBI downstream agreement")
    ax_b.grid(axis="y")
    panel_label(ax_b, "(b)")

    ax_c = fig.add_subplot(gs[0, 2])
    state_summary = state.groupby("method", as_index=False)["state_agreement_percent"].min().set_index("method").reindex(METHOD_ORDER)
    ax_c.plot(
        np.arange(len(METHOD_ORDER)),
        state_summary["state_agreement_percent"],
        marker="s",
        color="#2E7D32",
        label="Minimum state agreement",
    )
    ax_c.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER], rotation=25, ha="right")
    ax_c.set_ylim(88, 101)
    ax_c.set_ylabel("Minimum state agreement (%)")
    ax_c.set_title("State downstream agreement")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)")

    # Register the detailed EWBI table as read and provenance even though the plotted summary uses the decision table.
    _ = ewbi.shape
    add_source_map("Figure S4", "(a)", "Artificial-gap normalized RMSE for LST and TWS", "Figure_S2_preprocessing_harmonization_sensitivity_FINAL | S2c_gapfill_panel_data.csv; gapfill_validation_metrics.csv", "Gap-filling method comparison", "Calendar-month, seasonal mean, linear, trend-seasonality, spline, and state-space methods were evaluated.")
    add_source_map("Figure S4", "(b)", "Worst-case EWBI agreement relative to retained calendar-month fill", "gapfill_decision_summary.csv; ewbi_gapfill_sensitivity_metrics.csv", "Downstream EWBI gap-fill sensitivity", "The retained calendar-month method remains the reference for continuity.")
    add_source_map("Figure S4", "(c)", "Minimum ecosystem-state agreement relative to retained calendar-month fill", "state_gapfill_sensitivity.csv", "Downstream state-classification gap-fill sensitivity", "State agreement remained high across tested alternatives.")
    return save_fig(fig, "Figure_S4_gap_filling_validation_downstream_sensitivity")


def build_s5() -> tuple[Path, Path]:
    detrending = panel_csv("S2d_detrending_panel_data.csv")
    prep_order = ["raw_monthly", "deseasonalized_only", "month_specific_detrended", "ssa_detrended"]
    labels = ["Raw", "Deseasonalized", "Month\n detrended", "SSA\n detrended"]
    ordered = detrending.set_index("preprocessing").reindex(prep_order)
    fig = plt.figure(figsize=(9.0, 4.5), layout="constrained")
    gs = fig.add_gridspec(1, 2)
    x = np.arange(len(prep_order))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(
        x,
        ordered["mean_trend_removed"],
        yerr=ordered["sd_trend_removed"],
        fmt="o",
        ms=5.8,
        lw=1.2,
        capsize=3,
        color="#B2182B",
    )
    ax_a.set_xticks(x, labels)
    ax_a.set_ylim(-5, 105)
    ax_a.set_ylabel("Absolute trend removed (%)")
    ax_a.set_title("TWS low-frequency trend removal")
    ax_a.grid(axis="y")
    panel_label(ax_a, "(a)")
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.errorbar(
        x - 0.08,
        ordered["mean_ewbi_r"] * 100,
        yerr=ordered["sd_ewbi_r"] * 100,
        fmt="o",
        ms=5.5,
        capsize=3,
        color="#2166AC",
        label=r"EWBI association ($r \times 100$)",
    )
    ax_b.errorbar(
        x + 0.08,
        ordered["mean_state_agreement"],
        yerr=ordered["sd_state_agreement"],
        fmt="s",
        ms=5.5,
        capsize=3,
        color="#2E7D32",
        label="State agreement (%)",
    )
    ax_b.set_xticks(x, labels)
    ax_b.set_ylim(0, 105)
    ax_b.set_ylabel("Agreement with reference (%)")
    ax_b.set_title("Downstream agreement")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, loc="lower left")
    panel_label(ax_b, "(b)")
    add_source_map("Figure S5", "(a)", "TWS low-frequency trend removed by preprocessing option", "Figure_S2_preprocessing_harmonization_sensitivity_FINAL | S2d_detrending_panel_data.csv; detrending_trend_removal_summary.csv", "Detrending may remove meaningful climate signals", "Raw, deseasonalized-only, month-specific detrended, and SSA-detrended versions are compared.")
    add_source_map("Figure S5", "(b)", "EWBI and state agreement relative to deseasonalized reference", "Figure_S2_preprocessing_harmonization_sensitivity_FINAL | S2d_detrending_panel_data.csv; ewbi_detrending_sensitivity_metrics.csv; state_detrending_sensitivity.csv", "Downstream detrending sensitivity", "Deseasonalized-only is retained as the main ecological preprocessing choice.")
    return save_fig(fig, "Figure_S5_detrending_preprocessing_sensitivity")


def aggregate_monthly_models() -> pd.DataFrame:
    models = [
        "Persistence",
        "Seasonal persistence",
        "Monthly climatology",
        "SARIMA",
        "LSTM",
        "BiLSTM",
        "TCN",
        "N-BEATS",
        "N-HiTS",
        "XGBoost",
    ]
    monthly = read_csv(FULL_MONTHLY_METRICS)
    monthly = monthly[
        monthly["model"].isin(models)
        & monthly["r2"].notna()
        & monthly["normalized_rmse"].notna()
    ].copy()
    task_seed_mean = (
        monthly.groupby(["model", "land_type", "variable", "fold"], as_index=False)[["r2", "normalized_rmse"]]
        .mean()
    )
    return (
        task_seed_mean.groupby("model", as_index=False)
        .agg(
            n_tasks=("r2", "size"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
            sd_normalized_rmse=("normalized_rmse", "std"),
        )
        .set_index("model")
        .reindex(models)
        .reset_index()
    )


def build_s6() -> tuple[Path, Path]:
    baseline = aggregate_monthly_models()
    folds = panel_csv("S3b_fold_performance_panel_data.csv")
    timeline = panel_csv("S3d_rolling_origin_timeline_data.csv")
    fig = plt.figure(figsize=(13.0, 7.3), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])

    sub_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.12)
    y = np.arange(len(baseline))
    colors_model = [MODEL_COLORS[m] for m in baseline["model"]]
    ax_a1 = fig.add_subplot(sub_a[0, 0])
    ax_a1.errorbar(baseline["mean_r2"], y, xerr=baseline["sd_r2"], fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1.0)
    ax_a1.scatter(baseline["mean_r2"], y, c=colors_model, s=38, zorder=3)
    ax_a1.axvline(0, color="#8B1A1A", lw=0.85)
    ax_a1.set_yticks(y, baseline["model"])
    ax_a1.invert_yaxis()
    ax_a1.set_xlabel(r"Mean $R^2$")
    ax_a1.set_title(r"Monthly-aggregate $R^2$")
    ax_a1.grid(axis="x")
    panel_label(ax_a1, "(a)", x=-0.28)
    ax_a2 = fig.add_subplot(sub_a[0, 1], sharey=ax_a1)
    ax_a2.errorbar(baseline["mean_normalized_rmse"], y, xerr=baseline["sd_normalized_rmse"], fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1.0)
    ax_a2.scatter(baseline["mean_normalized_rmse"], y, c=colors_model, s=38, zorder=3)
    ax_a2.set_xlabel("Mean nRMSE")
    ax_a2.set_title("Monthly-aggregate nRMSE")
    ax_a2.tick_params(labelleft=False)
    ax_a2.grid(axis="x")

    sub_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.28)
    xb = np.arange(len(folds))
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b1.errorbar(xb, folds["mean_r2"], yerr=folds["sd_r2"], fmt="o-", ms=5.2, capsize=2.8, lw=1.2, color="#2F6DAE")
    ax_b1.axhline(0, color="#4A4A4A", lw=0.8)
    ax_b1.set_xticks(xb, folds["test_period"])
    ax_b1.set_ylabel(r"Mean pixel-level $R^2$")
    ax_b1.set_title(r"Pixel-level $R^2$ by fold")
    ax_b1.grid(axis="y")
    panel_label(ax_b1, "(b)", x=-0.26, y=1.18)
    ax_b2 = fig.add_subplot(sub_b[0, 1])
    ax_b2.errorbar(xb, folds["mean_nrmse"], yerr=folds["sd_nrmse"], fmt="s-", ms=5.2, capsize=2.8, lw=1.2, color="#D84A4A")
    ax_b2.set_xticks(xb, folds["test_period"])
    ax_b2.set_ylabel("Mean pixel-level nRMSE")
    ax_b2.set_title("Pixel-level nRMSE by fold")
    ax_b2.grid(axis="y")

    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_xlim(2000.5, 2025.0)
    ax_c.set_ylim(0.4, 4.6)
    ax_c.set_yticks([1, 2, 3, 4], [f"Fold {i}" for i in [1, 2, 3, 4]])
    for row in timeline.itertuples(index=False):
        fold = int(row.fold)
        core_end = parse_yyyymm(row.core_model_training_end) + 1 / 12
        val_start = parse_yyyymm(row.internal_validation_start)
        val_end = parse_yyyymm(row.internal_validation_end) + 1 / 12
        test_start = parse_yyyymm(row.test_start)
        test_end = parse_yyyymm(row.test_end) + 1 / 12
        ax_c.add_patch(Rectangle((2001.0, fold - 0.20), core_end - 2001.0, 0.40, facecolor="#8A8A8A", edgecolor="none"))
        ax_c.add_patch(Rectangle((val_start, fold - 0.20), val_end - val_start, 0.40, facecolor="#E9B949", edgecolor="none"))
        ax_c.add_patch(Rectangle((test_start, fold - 0.20), test_end - test_start, 0.40, facecolor="#D95D39", edgecolor="none"))
    ax_c.set_xticks([2001, 2005, 2009, 2013, 2017, 2021, 2025])
    ax_c.set_xlabel("Calendar year")
    ax_c.set_title("Rolling-origin design")
    ax_c.grid(axis="x")
    ax_c.legend(handles=[Patch(facecolor="#8A8A8A", label="Training"), Patch(facecolor="#E9B949", label="Validation"), Patch(facecolor="#D95D39", label="Test")], frameon=False, ncol=3, loc="lower center")
    panel_label(ax_c, "(c)", x=-0.03)

    baseline.to_csv(MAP_DIR / "Figure_S6_monthly_aggregate_model_summary.csv", index=False)
    add_source_map("Figure S6", "(a)", "Monthly aggregate R2 and nRMSE for statistical baselines and six original forecasting models", "Figure_S3_forecasting_robustness_baselines_FINAL | reviewer_safe_all_monthly_aggregated_metrics.csv; table_SF2_monthly_baseline_metrics_with_SARIMA.csv", "Benchmark baselines including SARIMA", "SARIMA is monthly land-type aggregate only; XGBoost is non-deep machine learning.")
    add_source_map("Figure S6", "(b)", "Pixel-level performance by rolling-origin fold", "Figure_S3_forecasting_robustness_baselines_FINAL | S3b_fold_performance_panel_data.csv; figure7_reviewer_safe_all_pixel_metrics.csv", "Validation beyond a single 2024 test year", "Main Figure 7 remains the pixel-level reviewer-safe benchmark.")
    add_source_map("Figure S6", "(c)", "Rolling-origin training, validation, and test design", "Figure_S3_forecasting_robustness_baselines_FINAL | S3d_rolling_origin_timeline_data.csv; reviewer_safe_fold_manifest.csv", "Leakage-free rolling-origin evaluation", "Preprocessing and validation are confined to training data within each fold.")
    return save_fig(fig, "Figure_S6_forecasting_baselines_rolling_origin")


def build_s7() -> tuple[Path, Path]:
    mv = panel_csv("S3c_multivariate_panel_data.csv")
    fig, ax = plt.subplots(figsize=(7.0, 4.4), layout="constrained")
    targets = ["GPP", "NIRv", "SPEI"]
    x = np.arange(len(targets))
    offsets = {"BiLSTM": -0.10, "XGBoost": 0.10}
    for model in ["BiLSTM", "XGBoost"]:
        subset = mv[mv["model"] == model].set_index("target").reindex(targets)
        x_model = x + offsets[model]
        ax.scatter(x_model, subset["mean_delta_r2_multivariate_minus_univariate"], marker="o", s=56, color=MODEL_COLORS[model], zorder=3)
        ax.scatter(x_model, subset["nrmse_benefit"], marker="^", s=62, facecolor="white", edgecolor=MODEL_COLORS[model], linewidth=1.4, zorder=3)
    ax.axhline(0, color="#3F3F3F", lw=0.8)
    ax.set_xticks(x, targets)
    ax.set_ylabel("Change favoring multivariate input")
    ax.set_title("")
    ax.grid(axis="y")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["BiLSTM"], markeredgecolor=MODEL_COLORS["BiLSTM"], label=r"BiLSTM $\Delta R^2$"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=MODEL_COLORS["BiLSTM"], markeredgewidth=1.4, label=r"BiLSTM $-\Delta$nRMSE"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS["XGBoost"], markeredgecolor=MODEL_COLORS["XGBoost"], label=r"XGBoost $\Delta R^2$"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="white", markeredgecolor=MODEL_COLORS["XGBoost"], markeredgewidth=1.4, label=r"XGBoost $-\Delta$nRMSE"),
        ],
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        fontsize=7.6,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )
    add_source_map("Figure S7", "single panel", "Targeted multivariate-minus-univariate sensitivity", "Figure_S3_forecasting_robustness_baselines_FINAL | S3c_multivariate_panel_data.csv; table_SF4_multivariate_sensitivity.csv", "Univariate forecasting concern", "Color denotes model; marker shape denotes metric; positive values indicate improvement from multivariate input.")
    png, pdf = save_fig(fig, "Figure_S7_targeted_multivariate_forecasting_sensitivity")
    final_png = PNG_DIR / "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL.png"
    final_pdf = PDF_DIR / "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL.pdf"
    shutil.copy2(png, final_png)
    shutil.copy2(pdf, final_pdf)
    return png, pdf


def build_s8() -> tuple[Path, Path]:
    stress = panel_csv("S4a_stress_threshold_panel_data.csv")
    fig = plt.figure(figsize=(9.0, 4.5), layout="constrained")
    gs = fig.add_gridspec(1, 2, wspace=0.24)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    thresholds = [-30, -40, -50]
    x = np.arange(3)
    for land in LAND_ORDER:
        subset = stress[stress["land_type"] == land].set_index("threshold").reindex(thresholds)
        y_area = subset["difference_km2"].to_numpy()
        y_pp = subset["difference_percentage_points"].to_numpy()
        area_err = np.vstack([y_area - subset["ci95_low_km2"].to_numpy(), subset["ci95_high_km2"].to_numpy() - y_area])
        pp_err = np.vstack([y_pp - subset["ci95_low_percentage_points"].to_numpy(), subset["ci95_high_percentage_points"].to_numpy() - y_pp])
        ax_a.errorbar(x, y_area, yerr=area_err, marker="o", ms=5.0, capsize=2.8, lw=1.15, color=LAND_COLORS[land], label=land)
        ax_b.errorbar(x, y_pp, yerr=pp_err, marker="o", ms=5.0, capsize=2.8, lw=1.15, color=LAND_COLORS[land])
    for ax in [ax_a, ax_b]:
        ax.axhline(0, color="#3F3F3F", lw=0.8)
        ax.set_xticks(x, ["30%", "40%", "50%"])
        ax.set_xlabel("NIRv decline threshold")
        ax.grid(axis="y")
    ax_a.set_title("Absolute area")
    ax_a.set_ylabel(r"Area change (km$^2$)")
    ax_a.legend(frameon=False, fontsize=8.0, loc="upper right")
    panel_label(ax_a, "(a)")
    ax_b.set_title("Eligible-area share")
    ax_b.set_ylabel("Change (percentage points)")
    panel_label(ax_b, "(b)")
    add_source_map("Figure S8", "(a)", "Absolute stressed-area change across NIRv decline thresholds", "Figure_S4_stress_state_sensitivity_FINAL | S4a_stress_threshold_panel_data.csv; stress_zone_authoritative_values_final.csv", "Stress-zone threshold sensitivity and uncertainty", "Areas use fractional masks and eligible analyzed land area; intervals are spatial block-bootstrap intervals.")
    add_source_map("Figure S8", "(b)", "Eligible analyzed-area percentage-point change across NIRv decline thresholds", "Figure_S4_stress_state_sensitivity_FINAL | S4a_stress_threshold_panel_data.csv; stress_zone_authoritative_values_final.csv", "Stress-zone threshold sensitivity and uncertainty", "Percentages refer to eligible analyzed land area.")
    return save_fig(fig, "Figure_S8_NIRv_stress_zone_threshold_sensitivity")


def build_s9() -> tuple[Path, Path]:
    agreement = panel_csv("S4b_classification_agreement_panel_data.csv")
    occupancy = panel_csv("S4c_occupancy_sensitivity_panel_data.csv")
    persistence = panel_csv("S4d_persistence_panel_data.csv")
    method_order = ["33_67_extremes", "kmeans", "gmm"]
    persistence_order = ["median", "33_67_extremes", "kmeans", "gmm"]
    labels = {"median": "Median", "33_67_extremes": "33/67\nref.", "kmeans": "k-means", "gmm": "GMM"}
    agreement_p = (
        agreement[agreement["method"].isin(method_order)]
        .pivot(index="land_type", columns="method", values="cohen_kappa")
        .reindex(index=LAND_ORDER, columns=method_order)
    )
    occ_p = (
        occupancy.pivot(index="land_type", columns="method", values="max_absolute_occupancy_change_pp")
        .reindex(index=LAND_ORDER, columns=method_order)
    )
    pers_p = (
        persistence.pivot(index="land_type", columns="method", values="probability_percent")
        .reindex(index=LAND_ORDER, columns=persistence_order)
    )
    fig = plt.figure(figsize=(13.0, 4.5), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.18])
    ax_a = fig.add_subplot(gs[0, 0])
    add_heatmap(ax_a, agreement_p.to_numpy(), LAND_ORDER, [labels[m] for m in method_order], "Classification agreement", "YlGnBu", 0, 1, fmt=".2f", colorbar_label="Cohen's kappa")
    panel_label(ax_a, "(a)")
    ax_b = fig.add_subplot(gs[0, 1])
    add_heatmap(ax_b, occ_p.to_numpy(), LAND_ORDER, ["33/67\nref.", "k-means", "GMM"], "Occupancy sensitivity", "YlOrRd", 0, max(30, float(np.nanmax(occ_p.to_numpy()))), fmt=".1f", colorbar_label="Maximum change (percentage points)")
    panel_label(ax_b, "(b)")
    ax_c = fig.add_subplot(gs[0, 2])
    add_heatmap(ax_c, pers_p.to_numpy(), LAND_ORDER, [labels[m] for m in persistence_order], "Dry/stressed persistence", "PuBuGn", 0, 100, fmt=".1f", colorbar_label="Self-transition probability (%)")
    panel_label(ax_c, "(c)")
    add_source_map("Figure S9", "(a)", "Classification agreement relative to median-quadrant reference", "Figure_S4_stress_state_sensitivity_FINAL | S4b_classification_agreement_panel_data.csv; state_classification_agreement.csv", "Classification-threshold arbitrariness", "33/67 reference is a self-comparison on comparable extreme months.")
    add_source_map("Figure S9", "(b)", "Maximum state-occupancy change across classification methods", "Figure_S4_stress_state_sensitivity_FINAL | S4c_occupancy_sensitivity_panel_data.csv; state_occupancy_all_methods.csv", "State-occupancy sensitivity", "Changes are percentage points relative to the median classification.")
    add_source_map("Figure S9", "(c)", "Dry/stressed self-transition probability across classification methods", "Figure_S4_stress_state_sensitivity_FINAL | S4d_persistence_panel_data.csv; state_transition_matrices_all_methods.csv", "State-transition sensitivity", "State labels are descriptive and classification-dependent, not mechanistic regime shifts.")
    return save_fig(fig, "Figure_S9_ecosystem_state_classification_sensitivity")


CAPTIONS = {
    "Figure S1": {
        "title": "EWBI-response association.",
        "text": (
            "Period-stratified Pearson correlations between the Ecohydrological Water Balance Index (EWBI) and NIRv, NDVI, QA-controlled GNDVI, GOSIF/SIF, and GPP across land types and drought periods. Asterisks indicate associations whose moving-block/bootstrap 95% confidence intervals exclude zero. EWBI is interpreted as a forcing-side composite; this association-based validation supports reviewer-requested EWBI evaluation but does not establish causality."
        ),
    },
    "Figure S2": {
        "title": "EWBI component and weighting robustness.",
        "text": (
            "(a) Mean response association for individual forcing components and for EWBI recalculated after omitting each component. (b) Mean response association under equal, PCA-derived, and regression-derived EWBI weights. (c) EWBI drought-period separation for the 2001-2011 baseline, 2012-2016 drought, and 2020-2024 recent drought period. Error bars show sample SD or bootstrap confidence intervals as indicated by the source analyses. EWBI is not interpreted as a mechanistic drought index; these panels support component, weighting, and period-discrimination robustness checks, not causal proof."
        ),
    },
    "Figure S3": {
        "title": "TWS scale and resampling sensitivity.",
        "text": (
            "(a-b) TWS agreement between land-type summaries on the common analytical grid and coarse fractional-overlap summaries, shown as Pearson r and normalized RMSE. Pearson r values are rounded to three decimals. (c) Normalized disagreement across nearest-neighbor, bilinear, coarse-overlap, and reverse-aggregation representations. High agreement indicates stability of land-type temporal summaries, not independent fine-scale TWS hydrologic information; the common grid is an analytical overlay."
        ),
    },
    "Figure S4": {
        "title": "Gap-filling validation and downstream sensitivity.",
        "text": (
            "(a) Artificial-gap normalized RMSE for calendar-month interpolation, seasonal mean, linear interpolation, trend-seasonality regression, spline interpolation, and Kalman/state-space gap filling. (b) Worst-case EWBI agreement relative to the retained calendar-month method. (c) Minimum ecosystem-state agreement relative to the retained calendar-month method. Artificial-gap validation was used to compare predictive reconstruction, and downstream EWBI/state sensitivity was assessed to determine whether conclusions were stable."
        ),
    },
    "Figure S5": {
        "title": "Detrending and preprocessing sensitivity.",
        "text": (
            "(a) Percentage of the absolute low-frequency TWS trend removed under raw monthly, deseasonalized-only, month-specific linearly detrended, and SSA-detrended preprocessing. (b) EWBI association and ecosystem-state agreement relative to the deseasonalized reference. Deseasonalized-only anomalies were retained as the main ecological preprocessing choice. This detrending analysis is interpreted as preprocessing robustness evidence rather than as a new scientific result."
        ),
    },
    "Figure S6": {
        "title": "Forecasting baselines and rolling-origin validation.",
        "text": (
            "(a) Monthly land-type aggregate R2 and normalized RMSE for persistence, seasonal persistence, monthly climatology, SARIMA, LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost. SARIMA monthly land-type aggregate results are shown only as a baseline and SARIMA is not part of the pixel-level Figure 7 benchmark. XGBoost is a non-deep machine-learning comparator. (b) Pixel-level reviewer-safe performance across four rolling-origin folds. (c) Expanding training, training-only validation, and 24-month test periods. Main Figure 7 remains the pixel-level reviewer-safe benchmark; rolling-origin validation was leakage-free."
        ),
    },
    "Figure S7": {
        "title": "Targeted multivariate forecasting sensitivity.",
        "text": (
            "The figure summarizes changes in forecasting performance when selected targets were evaluated with multivariate rather than univariate inputs. Filled circles indicate Delta R2 and open triangles indicate negative Delta nRMSE; therefore, positive values indicate improved performance under multivariate input. Colors denote model family. This analysis was used as a targeted sensitivity test for selected targets and models, not as a full all-target/all-model multivariate reconstruction. Gains were target-dependent, and the analysis does not support an unqualified model-family dominance claim."
        ),
    },
    "Figure S8": {
        "title": "NIRv stress-zone threshold sensitivity.",
        "text": (
            "(a) Absolute recent-minus-historic stressed-area change and (b) percentage-point change in eligible analyzed land area across 30%, 40%, and 50% NIRv decline thresholds. Stress areas use fractional masks and eligible analyzed land area. Error bars show spatial block-bootstrap uncertainty from the authoritative stress-zone source. This figure addresses stress-zone significance and threshold sensitivity."
        ),
    },
    "Figure S9": {
        "title": "Ecosystem-state classification sensitivity.",
        "text": (
            "(a) Classification agreement, (b) maximum state-occupancy sensitivity, and (c) dry/stressed self-transition probability under 33/67 percentile, k-means, and Gaussian mixture alternatives. The 33/67 reference/self-comparison equals 1.00 by definition where retained because it is evaluated on comparable extreme months. Ecosystem-state labels are descriptive and classification-dependent and are not interpreted as mechanistic regime shifts."
        ),
    },
}


FIGURE_ORDER = [
    ("Figure S1", "Figure_S1_EWBI_response_association"),
    ("Figure S2", "Figure_S2_EWBI_component_weighting_robustness"),
    ("Figure S3", "Figure_S3_TWS_scale_resampling_sensitivity"),
    ("Figure S4", "Figure_S4_gap_filling_validation_downstream_sensitivity"),
    ("Figure S5", "Figure_S5_detrending_preprocessing_sensitivity"),
    ("Figure S6", "Figure_S6_forecasting_baselines_rolling_origin"),
    ("Figure S7", "Figure_S7_targeted_multivariate_forecasting_sensitivity"),
    ("Figure S8", "Figure_S8_NIRv_stress_zone_threshold_sensitivity"),
    ("Figure S9", "Figure_S9_ecosystem_state_classification_sensitivity"),
]


def write_captions() -> None:
    for fig_id, _ in FIGURE_ORDER:
        payload = CAPTIONS[fig_id]
        number = fig_id.replace("Figure ", "Figure_")
        (CAPTION_DIR / f"{number}_caption.md").write_text(
            f"# {fig_id}. {payload['title']}\n\n{fig_id}. {payload['title']} {payload['text']}\n",
            encoding="utf-8",
        )


def write_source_map() -> None:
    df = pd.DataFrame(SOURCE_MAP_ROWS)
    df = df[
        [
            "supplementary_figure",
            "panel",
            "source_compact_figure",
            "source_table_or_file",
            "reviewer_concern_addressed",
            "notes",
        ]
    ]
    df.to_csv(MAP_DIR / "separate_supplementary_figure_source_map.csv", index=False)


def inspect_example_pdf() -> dict[str, str]:
    register_read(EXAMPLE_PDF)
    reader = PdfReader(str(EXAMPLE_PDF))
    page_count = len(reader.pages)
    first_text = (reader.pages[0].extract_text() or "").replace("\x00", " ")
    summary = {
        "path": str(EXAMPLE_PDF),
        "pages": str(page_count),
        "style_summary": "Title/contents page followed by separate numbered supplementary figures with captions.",
        "first_page_excerpt": re.sub(r"\s+", " ", first_text[:900]).strip(),
    }
    return summary


def build_supplementary_pdf() -> Path:
    pdf_path = SUPP_DIR / "ECOINF_supplementary_information_final.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="Supplementary Information",
        author=AUTHOR_LINE,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER, fontSize=20, leading=24, spaceAfter=10))
    styles.add(ParagraphStyle(name="CenterSub", parent=styles["Heading2"], alignment=TA_CENTER, fontSize=11, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle(name="FigCaption", parent=styles["BodyText"], fontSize=8.2, leading=10.4, spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="Contents", parent=styles["BodyText"], fontSize=9.2, leading=12.0, leftIndent=4 * mm))
    story = []
    story.append(Paragraph("Supplementary Information", styles["CenterTitle"]))
    story.append(Paragraph(MANUSCRIPT_TITLE, styles["CenterSub"]))
    story.append(Paragraph(AUTHOR_LINE, styles["CenterSub"]))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("<b>Contents of this file:</b>", styles["Heading2"]))
    for fig_id, _stem in FIGURE_ORDER:
        story.append(Paragraph(f"{fig_id}: {CAPTIONS[fig_id]['title'].rstrip('.')}", styles["Contents"]))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("The supplementary figures provide reviewer-requested robustness and sensitivity evidence for EWBI validation, preprocessing, forecasting validation, stress-zone uncertainty, and ecosystem-state classification sensitivity. Scientific results were not recomputed for this file; the figures are restructured from validated revision outputs.", styles["BodyText"]))
    story.append(PageBreak())

    for idx, (fig_id, stem) in enumerate(FIGURE_ORDER):
        png = PNG_DIR / f"{stem}.png"
        story.append(Paragraph(f"{fig_id}. {CAPTIONS[fig_id]['title']}", styles["Heading2"]))
        image = RLImage(str(png))
        max_width = 175 * mm
        max_height = 185 * mm if fig_id not in {"Figure S6"} else 170 * mm
        scale = min(max_width / image.imageWidth, max_height / image.imageHeight)
        image.drawWidth = image.imageWidth * scale
        image.drawHeight = image.imageHeight * scale
        story.append(image)
        caption = f"<b>{fig_id}. {CAPTIONS[fig_id]['title']}</b> {CAPTIONS[fig_id]['text']}"
        story.append(Paragraph(caption, styles["FigCaption"]))
        if idx < len(FIGURE_ORDER) - 1:
            story.append(PageBreak())

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(16 * mm, 8 * mm, "ECOINF-D-26-00382 | Supplementary Information")
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    register_read(pdf_path)
    return pdf_path


def write_report(example: dict[str, str], pdf_path: Path) -> None:
    source_map = pd.DataFrame(SOURCE_MAP_ROWS)
    read_unique = []
    seen = set()
    for path in READ_FILES:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            read_unique.append(path)
    inv = []
    for path in read_unique:
        inv.append(
            {
                "file": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
                "sha256": sha256(path) if path.exists() and path.is_file() else "",
            }
        )
    pd.DataFrame(inv).to_csv(MAP_DIR / "build_input_inventory.csv", index=False)

    lines = [
        "# Separate Supplementary Figure Build Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## What was read",
        "",
        f"- Example supplementary PDF: `{example['path']}` ({example['pages']} pages).",
        f"- Style reference used: {example['style_summary']}",
        f"- Final polished compact package: `{POLISHED}`.",
        f"- Validated panel CSVs: `{PANEL_DIR}`.",
        f"- Consolidated source tables: `{SOURCE_TABLES}`.",
        f"- Gap-fill downstream tables: `{GAP_TABLES}`.",
        "",
        "## Figures created",
        "",
    ]
    for fig_id, stem in FIGURE_ORDER:
        lines.append(f"- {fig_id}: `{PNG_DIR / (stem + '.png')}` and `{PDF_DIR / (stem + '.pdf')}`")
    lines.append(f"- Figure S7 final clarity exports: `{PNG_DIR / 'Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL.png'}` and `{PDF_DIR / 'Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL.pdf'}`.")
    lines.extend(
        [
            "",
            "## Supplementary PDF",
            "",
            f"- `{pdf_path}`",
            "",
            "## Source mapping",
            "",
            f"- Panel-level source map: `{MAP_DIR / 'separate_supplementary_figure_source_map.csv'}` ({len(source_map)} rows).",
            f"- Input inventory: `{MAP_DIR / 'build_input_inventory.csv'}`.",
            "",
            "## Scientific safeguards",
            "",
            "- Scientific analysis rerun: no.",
            "- Forecasting or deep-learning models rerun: no.",
            "- Raster extraction rerun: no.",
            "- Scientific values changed: no; figures were re-rendered or split from validated panel tables.",
            "- Figure S7 legend clarified: color denotes model and marker shape denotes metric; plotted values are unchanged.",
            "- SARIMA is described only as a monthly land-type aggregate baseline.",
            "- XGBoost is not described as a deep-learning model.",
            "- TWS common-grid representation is described as an analytical overlay, not fine-scale hydrologic information.",
            "- EWBI is described as a forcing-side composite, not a mechanistic drought index.",
            "- Ecosystem-state classes are described as descriptive and classification-dependent.",
            "- Stress-zone uncertainty is linked to spatial block bootstrap, consistent with the authoritative stress-zone table.",
            "",
            "## Remaining manual checks",
            "",
            "- Confirm final Supplement numbering after manuscript layout.",
            "- Inspect the assembled PDF at the journal's preferred page size.",
            "- Insert cross-references to these supplementary figures in the revised manuscript and response letter.",
        ]
    )
    (REPORT_DIR / "separate_supplementary_figure_build_report.md").write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(pdf_path: Path) -> None:
    expected = [stem for _, stem in FIGURE_ORDER]
    missing = []
    for stem in expected:
        if not (PNG_DIR / f"{stem}.png").exists():
            missing.append(str(PNG_DIR / f"{stem}.png"))
        if not (PDF_DIR / f"{stem}.pdf").exists():
            missing.append(str(PDF_DIR / f"{stem}.pdf"))
    if missing:
        raise RuntimeError("Missing outputs: " + "; ".join(missing))
    reader = PdfReader(str(pdf_path))
    if len(reader.pages) != 10:
        raise RuntimeError(f"Unexpected supplementary PDF page count: {len(reader.pages)}")
    checks = {
        "SARIMA monthly land-type aggregate": CAPTIONS["Figure S6"]["text"],
        "XGBoost is a non-deep machine-learning comparator": CAPTIONS["Figure S6"]["text"],
        "analytical overlay": CAPTIONS["Figure S3"]["text"],
        "forcing-side composite": CAPTIONS["Figure S1"]["text"],
        "descriptive and classification-dependent": CAPTIONS["Figure S9"]["text"],
        "spatial block-bootstrap": CAPTIONS["Figure S8"]["text"],
    }
    for phrase, text in checks.items():
        if phrase not in text:
            raise RuntimeError(f"Missing required phrase: {phrase}")


def main() -> None:
    for directory in [PNG_DIR, PDF_DIR, CAPTION_DIR, MAP_DIR, REPORT_DIR, SUPP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    example = inspect_example_pdf()
    builders = [build_s1, build_s2, build_s3, build_s4, build_s5, build_s6, build_s7, build_s8, build_s9]
    for builder in builders:
        builder()
    write_captions()
    write_source_map()
    pdf_path = build_supplementary_pdf()
    validate_outputs(pdf_path)
    write_report(example, pdf_path)
    print(f"OUTPUT_ROOT={ROOT}")
    print(f"FIGURES={len(list(PNG_DIR.glob('*.png')))} PNG + {len(list(PDF_DIR.glob('*.pdf')))} PDF")
    print(f"SUPPLEMENTARY_PDF={pdf_path}")
    print("SCIENTIFIC_ANALYSIS_RERUN=no")
    print("MODELS_RERUN=no")
    print("RASTER_EXTRACTION_RERUN=no")
    print("VALUES_CHANGED=no")


if __name__ == "__main__":
    main()
