"""Presentation-only readability fixes for supplementary Figures S2-S6 and S9.

The script uses the already-final supplementary source tables and redraws only
the requested figures with larger fonts/annotations and wider axes where needed.
No upstream analysis, model fitting, raster extraction, or metric recomputation
is performed.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle


DOCS = Path(r"<PROJECT_ROOT>\Docs")
OUT = DOCS / "major_revision_analyses" / "FINAL_supplementary_readability_fixes"
PNG_DIR = OUT / "figures_png"
PDF_DIR = OUT / "figures_pdf"
REPORT_DIR = OUT / "report"
TABLE_DIR = OUT / "tables"
TOOL_DIR = OUT / "tools"

FINAL_SUPP = DOCS / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
COMPACT = DOCS / "ECOINF_compact_supplementary_figures_final"
PANEL_DIR = COMPACT / "source_mapping"
FINAL_PACKAGE_TABLES = DOCS / "ECOINF_revision_final_figures_package" / "4_source_tables_used"

S2_TABLE_DIR = FINAL_SUPP / "tables"
S6_MAP_DIR = FINAL_SUPP / "source_mapping"

LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#007C91",
    "Natural Forests": "#2E7D32",
    "Savannas": "#E58B00",
    "Shrublands": "#B23A67",
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
MONTHLY_MODELS = [
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

READ_FILES: list[Path] = []


def ensure_dirs() -> None:
    for directory in (PNG_DIR, PDF_DIR, REPORT_DIR, TABLE_DIR, TOOL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def configure_style(font_scale: float = 1.0) -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.5 * font_scale,
            "axes.titlesize": 17.0 * font_scale,
            "axes.titleweight": "semibold",
            "axes.labelsize": 15.0 * font_scale,
            "xtick.labelsize": 13.5 * font_scale,
            "ytick.labelsize": 13.5 * font_scale,
            "legend.fontsize": 12.5 * font_scale,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.75,
            "grid.alpha": 0.60,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    READ_FILES.append(path)
    return pd.read_csv(path)


def panel_csv(name: str) -> pd.DataFrame:
    return read_csv(PANEL_DIR / name)


def save_fig(fig: plt.Figure, stem: str, dpi: int = 500) -> tuple[Path, Path]:
    png = PNG_DIR / f"{stem}.png"
    pdf = PDF_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08, size: float = 20) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=size,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_yyyymm(value: object) -> float:
    text = str(value)
    if "-" in text:
        year, month = text.split("-")[:2]
    elif len(text) >= 6 and text[:6].isdigit():
        year, month = text[:4], text[4:6]
    else:
        raise ValueError(f"Cannot parse YYYYMM value: {value!r}")
    return int(year) + (int(month) - 1) / 12


def add_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    *,
    fmt: str = ".2f",
    colorbar_label: str | None = None,
    annotation_size: float = 18,
    tick_size: float = 16,
    title_size: float = 21,
) -> None:
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=title_size, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=tick_size)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=tick_size)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = mpl.colormaps[cmap]
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            rgba = cmap_obj(norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.50 else "#222222"
            ax.text(
                j,
                i,
                format(value, fmt),
                ha="center",
                va="center",
                color=color,
                fontsize=annotation_size,
            )
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    if colorbar_label:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.028)
        cbar.set_label(colorbar_label, fontsize=tick_size, fontweight="normal")
        cbar.ax.tick_params(labelsize=tick_size - 1)


def build_s2() -> tuple[Path, Path]:
    configure_style(1.10)
    component_panel = read_csv(S2_TABLE_DIR / "S2_component_omission_panel_data_previous_GNDVI_with_test.csv")
    weighting_panel = read_csv(S2_TABLE_DIR / "S2_weighting_panel_data_previous_GNDVI_with_test.csv")
    separation = read_csv(S2_TABLE_DIR / "S2_period_separation_panel_data_unchanged.csv")

    fig = plt.figure(figsize=(19.5, 6.4), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.04, 1.10, 1.34], wspace=0.18)
    comps = ["SPEI", "TWS", "VPD", "LST"]
    x = np.arange(len(comps))

    ax_a = fig.add_subplot(gs[0, 0])
    styles = [
        ("Individual component", "Component alone", "#2166AC", "o", -0.10),
        ("EWBI after omission", "EWBI omitting component", "#B2182B", "s", 0.10),
    ]
    for series, label, color, marker, offset in styles:
        subset = component_panel[component_panel["series"] == series].set_index("component").reindex(comps)
        ax_a.errorbar(
            x + offset,
            subset["mean"],
            yerr=subset["std"],
            fmt=marker,
            ms=8.5,
            lw=1.8,
            capsize=4.5,
            color=color,
            label=label,
        )
    ax_a.axhline(0, color="#4A4A4A", lw=1.2)
    ax_a.set_xticks(x, comps)
    ax_a.set_ylabel("Mean Pearson r")
    ax_a.set_title("Component sensitivity")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="lower right", fontsize=13.0)
    panel_label(ax_a, "(a)", x=-0.16)

    ax_b = fig.add_subplot(gs[0, 1])
    weights_order = ["Equal", "PCA", "Regression"]
    for land in LAND_ORDER:
        subset = weighting_panel[weighting_panel["land_type"] == land].set_index("weighting").reindex(weights_order)
        ax_b.errorbar(
            np.arange(3),
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            ms=8.0,
            capsize=4.0,
            lw=1.8,
            color=LAND_COLORS[land],
            label=land,
        )
    ax_b.axhline(0, color="#4A4A4A", lw=1.2)
    ax_b.set_xticks(np.arange(3), weights_order)
    ax_b.set_ylabel("Mean Pearson r")
    ax_b.set_title("Weighting robustness")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, ncol=2, fontsize=12.4, loc="lower center")
    panel_label(ax_b, "(b)", x=-0.14)

    ax_c = fig.add_subplot(gs[0, 2])
    offsets = np.linspace(-0.24, 0.24, len(LAND_ORDER))
    labels = ["Baseline\n2001-2011", "Historic drought\n2012-2016", "Recent drought\n2020-2024"]
    for offset, land in zip(offsets, LAND_ORDER):
        subset = separation[separation["land_type"] == land].set_index("period").reindex(PERIOD_ORDER)
        y = subset["mean_ewbi"].to_numpy(dtype=float)
        err = np.vstack(
            [
                y - subset["ci95_low"].to_numpy(dtype=float),
                subset["ci95_high"].to_numpy(dtype=float) - y,
            ]
        )
        ax_c.errorbar(
            np.arange(3) + offset,
            y,
            yerr=err,
            fmt="o",
            ms=8.0,
            capsize=4.0,
            lw=1.7,
            color=LAND_COLORS[land],
        )
    ax_c.axhline(0, color="#4A4A4A", lw=1.2)
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.tick_params(axis="x", labelsize=15)
    ax_c.set_ylabel("Mean EWBI forcing score")
    ax_c.set_title("Drought-period separation")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)", x=-0.12)
    return save_fig(fig, "Figure_S2_EWBI_component_weighting_robustness")


def build_s3() -> tuple[Path, Path]:
    configure_style(1.05)
    scale = panel_csv("S2a_tws_scale_panel_data.csv").set_index("land_type").reindex(LAND_ORDER)
    resamp_raw = panel_csv("S2b_resampling_panel_data.csv")
    comparison_order = [
        "Nearest vs\nbilinear",
        "Bilinear vs\ncoarse overlap",
        "Nearest vs\ncoarse overlap",
        "Reverse aggregate\nvs coarse",
    ]
    comparison_labels = [
        "Nearest\nvs\nbilinear",
        "Bilinear\nvs coarse\noverlap",
        "Nearest\nvs coarse\noverlap",
        "Reverse\naggregate\nvs coarse",
    ]
    resamp = (
        resamp_raw.pivot(index="land_type", columns="comparison_short", values="normalized_rmse_by_reference_sd")
        .reindex(index=LAND_ORDER, columns=comparison_order)
    )

    fig = plt.figure(figsize=(23.0, 6.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[0.78, 0.78, 2.55], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    add_heatmap(
        ax_a,
        scale[["pearson_r"]].to_numpy(),
        LAND_ORDER,
        ["r"],
        "(a) TWS scale: Pearson r",
        "YlGn",
        0.95,
        1.0,
        fmt=".3f",
        annotation_size=18.5,
        tick_size=16.2,
        title_size=21,
    )
    ax_b = fig.add_subplot(gs[0, 1])
    add_heatmap(
        ax_b,
        scale[["normalized_rmse_by_reference_sd"]].to_numpy(),
        LAND_ORDER,
        ["nRMSE"],
        "(b) TWS scale: nRMSE",
        "YlOrRd",
        0,
        max(0.02, scale["normalized_rmse_by_reference_sd"].max()),
        fmt=".3f",
        annotation_size=18.5,
        tick_size=16.2,
        title_size=21,
    )
    ax_b.set_yticklabels([])
    ax_c = fig.add_subplot(gs[0, 2])
    add_heatmap(
        ax_c,
        resamp.to_numpy(),
        LAND_ORDER,
        comparison_labels,
        "(c) Resampling sensitivity",
        "YlOrRd",
        0,
        max(0.03, float(np.nanmax(resamp.to_numpy()))),
        fmt=".3f",
        colorbar_label="nRMSE / reference SD",
        annotation_size=18.5,
        tick_size=14.4,
        title_size=21,
    )
    return save_fig(fig, "Figure_S3_TWS_scale_resampling_sensitivity")


def build_s4() -> tuple[Path, Path]:
    configure_style(1.08)
    validation = panel_csv("S2c_gapfill_panel_data.csv")
    decision = read_csv(FINAL_PACKAGE_TABLES / "gapfill_decision_summary.csv")
    state = read_csv(FINAL_PACKAGE_TABLES / "state_gapfill_sensitivity.csv")

    fig = plt.figure(figsize=(26.0, 7.0), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.58, 1.55, 1.55], wspace=0.26)

    ax_a = fig.add_subplot(gs[0, 0])
    x = np.arange(len(METHOD_ORDER))
    for variable, offset, color, marker in [("LST", -0.11, "#2F6DAE", "o"), ("TWS", 0.11, "#D84A4A", "s")]:
        subset = validation[validation["variable"] == variable].set_index("method").reindex(METHOD_ORDER)
        ax_a.errorbar(
            x + offset,
            subset["mean"],
            yerr=subset["std"],
            fmt=marker,
            ms=8.0,
            capsize=4.0,
            lw=1.8,
            color=color,
            label=variable,
        )
    ax_a.set_xticks(x, [METHOD_LABELS[m] for m in METHOD_ORDER])
    ax_a.tick_params(axis="x", labelsize=14.2, pad=7)
    ax_a.set_ylabel("Artificial-gap nRMSE")
    ax_a.set_title("Artificial-gap validation")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, fontsize=14)
    panel_label(ax_a, "(a)", x=-0.10)

    ax_b = fig.add_subplot(gs[0, 1])
    ordered = decision.set_index("method").reindex(METHOD_ORDER)
    ax_b.plot(
        np.arange(len(METHOD_ORDER)),
        ordered["ewbi_minimum_pearson_r_vs_current"],
        marker="o",
        markersize=8.0,
        linewidth=2.0,
        color="#2166AC",
    )
    ax_b.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER])
    for label in ax_b.get_xticklabels():
        label.set_rotation(22)
        label.set_ha("right")
    ax_b.tick_params(axis="x", labelsize=14.0, pad=7)
    ax_b.set_ylim(0.94, 1.005)
    ax_b.set_ylabel("Worst-case Pearson r")
    ax_b.set_title("EWBI downstream agreement")
    ax_b.grid(axis="y")
    panel_label(ax_b, "(b)", x=-0.12)

    ax_c = fig.add_subplot(gs[0, 2])
    state_summary = (
        state.groupby("method", as_index=False)["state_agreement_percent"]
        .min()
        .set_index("method")
        .reindex(METHOD_ORDER)
    )
    ax_c.plot(
        np.arange(len(METHOD_ORDER)),
        state_summary["state_agreement_percent"],
        marker="s",
        markersize=8.0,
        linewidth=2.0,
        color="#2E7D32",
    )
    ax_c.set_xticks(np.arange(len(METHOD_ORDER)), [METHOD_LABELS[m] for m in METHOD_ORDER])
    for label in ax_c.get_xticklabels():
        label.set_rotation(22)
        label.set_ha("right")
    ax_c.tick_params(axis="x", labelsize=14.0, pad=7)
    ax_c.set_ylim(88, 101)
    ax_c.set_ylabel("Minimum state agreement (%)")
    ax_c.set_title("State downstream agreement")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)", x=-0.12)
    return save_fig(fig, "Figure_S4_gap_filling_validation_downstream_sensitivity")


def build_s5() -> tuple[Path, Path]:
    configure_style(1.12)
    detrending = panel_csv("S2d_detrending_panel_data.csv")
    prep_order = ["raw_monthly", "deseasonalized_only", "month_specific_detrended", "ssa_detrended"]
    labels = ["Raw", "Deseasonalized", "Month\n detrended", "SSA\n detrended"]
    ordered = detrending.set_index("preprocessing").reindex(prep_order)
    fig = plt.figure(figsize=(15.5, 6.4), layout="constrained")
    gs = fig.add_gridspec(1, 2, wspace=0.18)
    x = np.arange(len(prep_order))
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.errorbar(
        x,
        ordered["mean_trend_removed"],
        yerr=ordered["sd_trend_removed"],
        fmt="o",
        ms=9.0,
        lw=2.0,
        capsize=4.5,
        color="#B2182B",
    )
    ax_a.set_xticks(x, labels)
    ax_a.tick_params(axis="x", labelsize=16, pad=8)
    ax_a.set_ylim(-5, 105)
    ax_a.set_ylabel("Absolute trend removed (%)")
    ax_a.set_title("TWS low-frequency trend removal")
    ax_a.grid(axis="y")
    panel_label(ax_a, "(a)", x=-0.10, size=22)

    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.errorbar(
        x - 0.08,
        ordered["mean_ewbi_r"] * 100,
        yerr=ordered["sd_ewbi_r"] * 100,
        fmt="o",
        ms=8.8,
        capsize=4.5,
        linewidth=2.0,
        color="#2166AC",
        label=r"EWBI association ($r \times 100$)",
    )
    ax_b.errorbar(
        x + 0.08,
        ordered["mean_state_agreement"],
        yerr=ordered["sd_state_agreement"],
        fmt="s",
        ms=8.8,
        capsize=4.5,
        linewidth=2.0,
        color="#2E7D32",
        label="State agreement (%)",
    )
    ax_b.set_xticks(x, labels)
    ax_b.tick_params(axis="x", labelsize=16, pad=8)
    ax_b.set_ylim(0, 105)
    ax_b.set_ylabel("Agreement with reference (%)")
    ax_b.set_title("Downstream agreement")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, loc="lower left", fontsize=14)
    panel_label(ax_b, "(b)", x=-0.10, size=22)
    return save_fig(fig, "Figure_S5_detrending_preprocessing_sensitivity")


def build_s6() -> tuple[Path, Path]:
    configure_style(1.08)
    baseline = read_csv(S6_MAP_DIR / "Figure_S6_monthly_aggregate_model_summary.csv")
    folds = read_csv(S6_MAP_DIR / "Figure_S6_fold_performance_previous_GNDVI_with_test.csv")
    timeline = panel_csv("S3d_rolling_origin_timeline_data.csv")

    fig = plt.figure(figsize=(24.0, 12.0), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.05], width_ratios=[1.34, 1.66], hspace=0.20, wspace=0.18)

    sub_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.10)
    y = np.arange(len(baseline))
    colors_model = [MODEL_COLORS[m] for m in baseline["model"]]
    ax_a1 = fig.add_subplot(sub_a[0, 0])
    ax_a1.errorbar(baseline["mean_r2"], y, xerr=baseline["sd_r2"], fmt="none", ecolor="#3A3A3A", capsize=4.0, lw=1.5)
    ax_a1.scatter(baseline["mean_r2"], y, c=colors_model, s=72, zorder=3)
    ax_a1.axvline(0, color="#8B1A1A", lw=1.2)
    ax_a1.set_yticks(y, baseline["model"])
    ax_a1.tick_params(axis="y", labelsize=14.5)
    ax_a1.invert_yaxis()
    ax_a1.set_xlabel(r"Mean $R^2$")
    ax_a1.set_title(r"Monthly-aggregate $R^2$")
    ax_a1.grid(axis="x")
    panel_label(ax_a1, "(a)", x=-0.33, size=22)

    ax_a2 = fig.add_subplot(sub_a[0, 1], sharey=ax_a1)
    ax_a2.errorbar(baseline["mean_normalized_rmse"], y, xerr=baseline["sd_normalized_rmse"], fmt="none", ecolor="#3A3A3A", capsize=4.0, lw=1.5)
    ax_a2.scatter(baseline["mean_normalized_rmse"], y, c=colors_model, s=72, zorder=3)
    ax_a2.set_xlabel("Mean nRMSE")
    ax_a2.set_title("Monthly-aggregate nRMSE")
    ax_a2.tick_params(labelleft=False)
    ax_a2.grid(axis="x")

    sub_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.34)
    xb = np.arange(len(folds))
    tick_labels = folds["test_period"].astype(str).tolist()
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b1.errorbar(xb, folds["mean_r2"], yerr=folds["sd_r2"], fmt="o-", ms=8.0, capsize=4.2, lw=2.0, color="#2F6DAE")
    ax_b1.axhline(0, color="#4A4A4A", lw=1.1)
    ax_b1.set_xticks(xb, tick_labels)
    ax_b1.tick_params(axis="x", labelsize=15, pad=8)
    ax_b1.set_ylabel(r"Mean pixel-level $R^2$")
    ax_b1.set_title(r"Pixel-level $R^2$ by fold")
    ax_b1.grid(axis="y")
    panel_label(ax_b1, "(b)", x=-0.16, y=1.16, size=22)

    ax_b2 = fig.add_subplot(sub_b[0, 1])
    ax_b2.errorbar(xb, folds["mean_nrmse"], yerr=folds["sd_nrmse"], fmt="s-", ms=8.0, capsize=4.2, lw=2.0, color="#D84A4A")
    ax_b2.set_xticks(xb, tick_labels)
    ax_b2.tick_params(axis="x", labelsize=15, pad=8)
    ax_b2.set_ylabel("Mean pixel-level nRMSE")
    ax_b2.set_title("Pixel-level nRMSE by fold")
    ax_b2.grid(axis="y")

    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_xlim(2000.5, 2025.0)
    ax_c.set_ylim(0.4, 4.6)
    ax_c.set_yticks([1, 2, 3, 4], [f"Fold {i}" for i in [1, 2, 3, 4]])
    ax_c.tick_params(axis="y", labelsize=15.5)
    for row in timeline.itertuples(index=False):
        fold = int(row.fold)
        core_end = parse_yyyymm(row.core_model_training_end) + 1 / 12
        val_start = parse_yyyymm(row.internal_validation_start)
        val_end = parse_yyyymm(row.internal_validation_end) + 1 / 12
        test_start = parse_yyyymm(row.test_start)
        test_end = parse_yyyymm(row.test_end) + 1 / 12
        ax_c.add_patch(Rectangle((2001.0, fold - 0.21), core_end - 2001.0, 0.42, facecolor="#8A8A8A", edgecolor="none"))
        ax_c.add_patch(Rectangle((val_start, fold - 0.21), val_end - val_start, 0.42, facecolor="#E9B949", edgecolor="none"))
        ax_c.add_patch(Rectangle((test_start, fold - 0.21), test_end - test_start, 0.42, facecolor="#D95D39", edgecolor="none"))
    ax_c.set_xticks([2001, 2005, 2009, 2013, 2017, 2021, 2025])
    ax_c.tick_params(axis="x", labelsize=16, pad=7)
    ax_c.set_xlabel("Calendar year")
    ax_c.set_title("Rolling-origin design")
    ax_c.grid(axis="x")
    ax_c.legend(
        handles=[
            Patch(facecolor="#8A8A8A", label="Training"),
            Patch(facecolor="#E9B949", label="Validation"),
            Patch(facecolor="#D95D39", label="Test"),
        ],
        frameon=False,
        ncol=3,
        loc="lower center",
        fontsize=14.5,
    )
    panel_label(ax_c, "(c)", x=-0.028, size=22)
    return save_fig(fig, "Figure_S6_forecasting_baselines_rolling_origin")


def build_s9() -> tuple[Path, Path]:
    configure_style(1.05)
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
    fig = plt.figure(figsize=(22.0, 6.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.05, 1.28], wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    add_heatmap(
        ax_a,
        agreement_p.to_numpy(),
        LAND_ORDER,
        [labels[m] for m in method_order],
        "(a) Classification agreement",
        "YlGnBu",
        0,
        1,
        fmt=".2f",
        colorbar_label="Cohen's kappa",
        annotation_size=18.8,
        tick_size=16.5,
        title_size=21,
    )
    ax_b = fig.add_subplot(gs[0, 1])
    add_heatmap(
        ax_b,
        occ_p.to_numpy(),
        LAND_ORDER,
        ["33/67\nref.", "k-means", "GMM"],
        "(b) Occupancy sensitivity",
        "YlOrRd",
        0,
        max(30, float(np.nanmax(occ_p.to_numpy()))),
        fmt=".1f",
        colorbar_label="Maximum change (percentage points)",
        annotation_size=18.8,
        tick_size=16.5,
        title_size=21,
    )
    ax_c = fig.add_subplot(gs[0, 2])
    add_heatmap(
        ax_c,
        pers_p.to_numpy(),
        LAND_ORDER,
        [labels[m] for m in persistence_order],
        "(c) Dry/stressed persistence",
        "PuBuGn",
        0,
        100,
        fmt=".1f",
        colorbar_label="Self-transition probability (%)",
        annotation_size=18.8,
        tick_size=15.7,
        title_size=21,
    )
    return save_fig(fig, "Figure_S9_ecosystem_state_classification_sensitivity")


def write_report(outputs: dict[str, tuple[Path, Path]]) -> None:
    rows = []
    for fig_id, (png, pdf) in outputs.items():
        rows.append(
            {
                "figure": fig_id,
                "png": str(png),
                "pdf": str(pdf),
                "png_sha256": sha256(png),
                "pdf_sha256": sha256(pdf),
            }
        )
    pd.DataFrame(rows).to_csv(TABLE_DIR / "supplementary_readability_fix_outputs.csv", index=False)
    read_rows = []
    for path in sorted(set(READ_FILES)):
        read_rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path) if path.exists() else "",
            }
        )
    pd.DataFrame(read_rows).to_csv(TABLE_DIR / "supplementary_readability_fix_source_manifest.csv", index=False)

    report = f"""# Supplementary Figure Readability Fix Report

Completed: {datetime.now().isoformat(timespec='seconds')}

## Scope

Presentation-only fixes were applied to Figures S2, S3, S4, S5, S6, and S9.
No scientific analysis, forecasting model, raster extraction, source table, or
metric value was changed.

## Changes Made

- Figure S2: enlarged x-axis text, tick labels, panel labels, markers, and titles.
- Figure S3: enlarged heatmap cell numbers, tick labels, titles, and colorbar text.
- Figure S4: widened the figure and enlarged all x-axis method labels and y-axis text.
- Figure S5: enlarged all text and markers; x-axis labels remain separated and readable.
- Figure S6: widened/stretched the top-row fold panels so test-year labels no longer overlap; enlarged x/y-axis fonts, markers, and timeline labels.
- Figure S9: enlarged heatmap cell numbers, tick labels, titles, and colorbar text.

## Output Folder

`{OUT}`

## Figures

"""
    for fig_id, (png, pdf) in outputs.items():
        report += f"- {fig_id}: `{png}` and `{pdf}`\n"
    report += f"""
## Source Tables

The source manifest is saved at:
`{TABLE_DIR / 'supplementary_readability_fix_source_manifest.csv'}`

The output manifest is saved at:
`{TABLE_DIR / 'supplementary_readability_fix_outputs.csv'}`
"""
    (REPORT_DIR / "FINAL_supplementary_readability_fix_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    outputs = {
        "Figure S2": build_s2(),
        "Figure S3": build_s3(),
        "Figure S4": build_s4(),
        "Figure S5": build_s5(),
        "Figure S6": build_s6(),
        "Figure S9": build_s9(),
    }
    write_report(outputs)
    shutil.copy2(__file__, TOOL_DIR / Path(__file__).name)
    print(OUT)
    for _, (png, pdf) in outputs.items():
        print(png)
        print(pdf)


if __name__ == "__main__":
    main()
