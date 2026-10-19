"""Minimal presentation-only fixes for final manuscript Figures 7 and 8.

Authoritative source figures:
    Figure_7_final.png
    Figure_8_final.png

This script rejects the prior readability redesign and performs only the
requested minimal rendering changes:

1. Figure 7: same final layout/logic/data, exported larger with modest font and
   marker scaling for Word readability.
2. Figure 8: same final maps/layout/data/colorbars, with only the in-panel
   annotation/statistic boxes removed.

No upstream analysis, forecasting model, raster extraction, or source-table
change is performed.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DOCS = Path(r"<PROJECT_ROOT>\Docs")
OUT = DOCS / "major_revision_analyses" / "FINAL_minimal_figure7_figure8_fix"
FIGURES = OUT / "figures"
REPORT = OUT / "report"
TOOLS = OUT / "tools"
TABLES = OUT / "tables"

FINAL_FIGURES = DOCS / "ECOINF_revision_final_figures_package" / "1_paper_main_figures"
FIG7_ORIGINAL = FINAL_FIGURES / "Figure_7_final.png"
FIG8_ORIGINAL = FINAL_FIGURES / "Figure_8_final.png"

SOURCE_TABLES = DOCS / "ECOINF_revision_final_figures_package" / "4_source_tables_used"
FIG7_PANEL_PATH = SOURCE_TABLES / "figure7_audited_panel_values.csv"
FIG7_COUNT_PATH = SOURCE_TABLES / "figure7_global_best_counts_verified.csv"

TWS_ROOT = DOCS / "TWS_Harmonization_Uncertainty_Revision"
DETREND_ROOT = DOCS / "major_revision_analyses" / "04_detrending_sensitivity_fixed"
FIG8_MAP_ARCHIVE = TWS_ROOT / "cache" / "nirv_period_maps.npz"
FIG8_STRESS_TABLE = TWS_ROOT / "tables" / "stress_zone_area_uncertainty.csv"
FIG8_MONTHLY_RAW = DETREND_ROOT / "tables" / "monthly_landtype_series_raw.csv"

TABLE4_VALUES = (
    DOCS
    / "major_revision_analyses"
    / "FINAL_table4_figure8_verification_rebuild"
    / "tables"
    / "Table4_final_nirv_response_and_severe_area.csv"
)


MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
TARGETS = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
LAND_TYPES_FIG7 = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
LAND_LABELS = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "Natural Forests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
COLORS = {
    "LSTM": "#0072B2",
    "BiLSTM": "#CC79A7",
    "TCN": "#009E73",
    "N-BEATS": "#E69F00",
    "N-HiTS": "#D55E00",
    "XGBoost": "#00A6A6",
}
CAPTION = (
    "Points show means across available reviewer-safe rolling-origin fold/seed "
    "rows; error bars show sample SD. Global count panels show the number of "
    "land-type × target tasks, out of 36, for which each model achieved the "
    "best R² or lowest normalized RMSE."
)

LAND_TYPES_FIG8 = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]


def ensure_directories() -> None:
    for directory in (FIGURES, REPORT, TOOLS, TABLES):
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_land_type(value: str) -> str:
    text = str(value).replace("_", " ").replace("-", " ").strip()
    collapsed = " ".join(text.split()).lower()
    mapping = {
        "grasslands": "Grasslands",
        "naturalforests": "Natural Forests",
        "natural forests": "Natural Forests",
        "savannas": "Savannas",
        "shrublands": "Shrublands",
    }
    return mapping.get(collapsed, text)


def archive_key(land_type: str) -> str:
    return "NaturalForests" if land_type == "Natural Forests" else land_type


def add_target_guides(axis, lower, upper):
    for index in range(len(TARGETS)):
        if index % 2 == 0:
            axis.axvspan(
                index - 0.5,
                index + 0.5,
                color="#EDF2F5",
                alpha=0.58,
                zorder=0,
            )
    for boundary in np.arange(0.5, len(TARGETS) - 0.49, 1.0):
        axis.axvline(
            boundary,
            color="#9EA7B0",
            linewidth=0.62,
            alpha=0.52,
            zorder=1,
        )
    axis.set_ylim(lower, upper)


def style_metric_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#C5CBD1", linewidth=0.60, alpha=0.66)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4F555B")
    axis.spines["bottom"].set_color("#4F555B")
    axis.tick_params(axis="both", colors="#25282B", width=0.75)


def draw_count_panel(axis, counts, column, title, font_scale: float):
    values = counts.set_index("model").reindex(MODELS)[column]
    y = np.arange(len(MODELS))
    maximum = float(values.max())
    bars = axis.barh(
        y,
        values,
        height=0.53,
        color=[COLORS[model] for model in MODELS],
        edgecolor="white",
        linewidth=0.45,
        alpha=0.94,
        zorder=2,
    )
    for bar, value in zip(bars, values):
        label = f"{value:.1f}" if not float(value).is_integer() else str(int(value))
        axis.text(
            value + max(0.22, maximum * 0.022),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left",
            fontsize=9.1 * font_scale,
            fontweight="bold",
            color="#2E3236",
        )
    axis.set_yticks(y)
    axis.set_yticklabels(MODELS, fontsize=8.9 * font_scale)
    axis.invert_yaxis()
    axis.set_xlim(0, maximum + max(1.55, maximum * 0.125))
    axis.set_xlabel("Task wins (out of 36)", fontsize=8.4 * font_scale)
    axis.set_title(title, fontsize=10.8 * font_scale, fontweight="bold", pad=8)
    axis.grid(axis="x", color="#D1D5DA", linewidth=0.55, alpha=0.64)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#72787E")
    axis.tick_params(axis="y", length=0)


def render_figure7_minimal() -> tuple[Path, Path]:
    panel = pd.read_csv(FIG7_PANEL_PATH)
    counts = pd.read_csv(FIG7_COUNT_PATH)
    if len(panel) != 216 or set(panel["model"]) != set(MODELS):
        raise ValueError("Approved audited Figure 7 plotting input is incomplete.")
    if not np.isclose(counts["best_r2_count"].sum(), 36):
        raise ValueError("Best R2 counts do not sum to 36.")
    if not np.isclose(counts["best_normalized_rmse_count"].sum(), 36):
        raise ValueError("Best nRMSE counts do not sum to 36.")

    r2_lower = float((panel["mean_r2"] - panel["sd_r2"]).min())
    r2_bottom = max(min(-0.25, np.floor((r2_lower - 0.05) * 10) / 10), -2.1)
    r2_top = max(
        1.05,
        np.ceil(float((panel["mean_r2"] + panel["sd_r2"]).max() + 0.03) * 20)
        / 20,
    )
    nrmse_upper = float((panel["mean_normalized_rmse"] + panel["sd_normalized_rmse"]).max())
    nrmse_top = np.ceil((nrmse_upper + 0.08) * 10) / 10

    # Minimal readability scaling while preserving the exact final layout.
    # The prior export was mostly a higher-resolution version; for Word
    # insertion the text must be larger relative to the panel area.
    font_scale = 1.55
    size_scale = 1.18
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0 * font_scale,
            "axes.titlesize": 11.6 * font_scale,
            "axes.labelsize": 10.8 * font_scale,
            "xtick.labelsize": 9.1 * font_scale,
            "ytick.labelsize": 9.3 * font_scale,
            "legend.fontsize": 10.0 * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(21.8 * size_scale, 15.2 * size_scale))
    grid = fig.add_gridspec(
        4,
        3,
        width_ratios=[6.25, 6.25, 1.38],
        left=0.048,
        right=0.988,
        top=0.902,
        bottom=0.078,
        hspace=0.36,
        wspace=0.135,
    )
    metric_axes = [
        (fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1]))
        for row in range(4)
    ]
    count_r2_axis = fig.add_subplot(grid[0:2, 2])
    count_nrmse_axis = fig.add_subplot(grid[2:4, 2])

    x = np.arange(len(TARGETS), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(MODELS))
    panel_letters = "abcdefgh"
    for row, land_type in enumerate(LAND_TYPES_FIG7):
        r2_axis, nrmse_axis = metric_axes[row]
        add_target_guides(r2_axis, r2_bottom, r2_top)
        add_target_guides(nrmse_axis, 0, nrmse_top)
        for model, offset in zip(MODELS, offsets):
            data = (
                panel[(panel["land_type"] == land_type) & (panel["model"] == model)]
                .set_index("target")
                .reindex(TARGETS)
            )
            positions = x + offset
            for axis, means, errors in (
                (r2_axis, data["mean_r2"], data["sd_r2"]),
                (nrmse_axis, data["mean_normalized_rmse"], data["sd_normalized_rmse"]),
            ):
                axis.errorbar(
                    positions,
                    means,
                    yerr=errors,
                    fmt="none",
                    capsize=4.1 * font_scale,
                    elinewidth=1.40 * font_scale,
                    capthick=1.26 * font_scale,
                    ecolor=COLORS[model],
                    alpha=0.98,
                    zorder=5,
                    clip_on=False,
                )
                axis.scatter(
                    positions,
                    means,
                    s=47 * (font_scale**2),
                    color=COLORS[model],
                    edgecolor="white",
                    linewidth=0.80 * font_scale,
                    zorder=6,
                )
        r2_axis.axhspan(r2_bottom, 0, color="#F4D9D9", alpha=0.42, zorder=0.5)
        r2_axis.axhline(0, color="#9B3030", linewidth=0.95, zorder=2)
        r2_axis.set_title(
            f"{panel_letters[2 * row]}) {LAND_LABELS[land_type]}: $R^2$",
            loc="left",
            fontweight="bold",
            pad=8,
        )
        nrmse_axis.set_title(
            f"{panel_letters[2 * row + 1]}) {LAND_LABELS[land_type]}: normalized RMSE",
            loc="left",
            fontweight="bold",
            pad=8,
        )
        r2_axis.set_ylabel(r"$R^2$")
        nrmse_axis.set_ylabel("Normalized RMSE")
        for axis in (r2_axis, nrmse_axis):
            axis.set_xlim(-0.58, len(TARGETS) - 0.42)
            axis.set_xticks(x)
            axis.set_xticklabels(TARGETS)
            axis.tick_params(axis="x", pad=5)
            style_metric_axis(axis)

    draw_count_panel(count_r2_axis, counts, "best_r2_count", "Best $R^2$ Count", font_scale)
    draw_count_panel(
        count_nrmse_axis,
        counts,
        "best_normalized_rmse_count",
        "Best nRMSE Count",
        font_scale,
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=7.6 * font_scale,
            markerfacecolor=COLORS[model],
            markeredgecolor="white",
            markeredgewidth=0.72,
            label=model,
        )
        for model in MODELS
    ]
    fig.legend(
        handles=handles,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.949),
        frameon=False,
        columnspacing=2.0,
        handlelength=0.8,
        handletextpad=0.48,
    )
    fig.suptitle(
        "Reviewer-safe forecasting performance across land types and targets",
        fontsize=15.3 * font_scale,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.022,
        CAPTION,
        ha="center",
        fontsize=8.8 * font_scale,
        color="#44494E",
    )
    png = FIGURES / "Figure_7_final_large_readable.png"
    pdf = FIGURES / "Figure_7_final_large_readable.pdf"
    fig.savefig(png, dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png, pdf


def reference_extent(land_type: str) -> list[float]:
    import rasterio

    raw = pd.read_csv(FIG8_MONTHLY_RAW)
    raw["land_type"] = raw["land_type"].map(normalize_land_type)
    row = raw[
        (raw["land_type"] == land_type)
        & (raw["variable"] == "NIRv")
        & raw["source_path"].notna()
    ].iloc[0]
    with rasterio.open(row["source_path"]) as source:
        bounds = source.bounds
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def render_figure8_no_boxes() -> tuple[Path, Path]:
    archive = np.load(FIG8_MAP_ARCHIVE)
    stress = pd.read_csv(FIG8_STRESS_TABLE)
    stress["land_type"] = stress["land_type"].map(normalize_land_type)
    main = stress[stress["threshold"] == -40].set_index("land_type").loc[LAND_TYPES_FIG8]

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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, axes = plt.subplots(4, 2, figsize=(18, 24))
    for row, land_type in enumerate(LAND_TYPES_FIG8):
        key = archive_key(land_type)
        extent = reference_extent(land_type)
        # Keep the final source table loaded for lineage validation, but do not
        # draw the statistic box that used these values in the original render.
        _summary = main.loc[land_type]
        for col, (period_key, _period_label) in enumerate(
            [("historic_decline", "2012-2016"), ("recent_decline", "2020-2024")]
        ):
            values = archive[f"{key}__{period_key}"].astype(float)
            axis = axes[row, col]
            image = axis.imshow(
                np.ma.masked_invalid(values),
                extent=extent,
                origin="upper",
                cmap="RdYlGn",
                vmin=-60,
                vmax=30,
                interpolation="nearest",
                aspect="auto",
            )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Change (%)", fontsize=10, fontweight="bold")
            period_title = (
                "2012-2016 Historic Drought" if col == 0 else "2020-2024 Recent Period"
            )
            axis.set_title(
                f"{land_type}\n{period_title}",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
            axis.set_xlabel("Longitude (degrees)", fontsize=10, fontweight="bold")
            axis.set_ylabel("Latitude (degrees)", fontsize=10, fontweight="bold")
            axis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    figure.suptitle(
        "Vegetation Response: NIRv Changes\n"
        "Historic Drought (2012-2016) vs Recent Period (2020-2024)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        "Red = Decline • Green = Increase/Stable • Percentage change from "
        "the 2001-2011 baseline\n"
        "Severe-area changes use latitude-aware area and paired spatial-block "
        "bootstrap 95% intervals",
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.4},
    )
    figure.tight_layout(rect=[0, 0.025, 1, 0.985])
    png = FIGURES / "Figure_8_final_no_boxes.png"
    pdf = FIGURES / "Figure_8_final_no_boxes.pdf"
    figure.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png, pdf


def write_report(fig7_png: Path, fig7_pdf: Path, fig8_png: Path, fig8_pdf: Path) -> None:
    rows = []
    for path in [
        FIG7_ORIGINAL,
        FIG8_ORIGINAL,
        FIG7_PANEL_PATH,
        FIG7_COUNT_PATH,
        FIG8_MAP_ARCHIVE,
        FIG8_STRESS_TABLE,
        TABLE4_VALUES,
        fig7_png,
        fig7_pdf,
        fig8_png,
        fig8_pdf,
    ]:
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "bytes": path.stat().st_size if path.exists() else "",
                "sha256": sha256(path) if path.exists() else "",
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "FINAL_minimal_fix_file_manifest.csv", index=False)

    report = f"""# FINAL Minimal Figure 7 / Figure 8 Fix Report

Completed: {datetime.now().isoformat(timespec='seconds')}

## Original Files Used

- Figure 7 original: `{FIG7_ORIGINAL}`
- Figure 8 original: `{FIG8_ORIGINAL}`
- Figure 7 panel source table: `{FIG7_PANEL_PATH}`
- Figure 7 count source table: `{FIG7_COUNT_PATH}`
- Figure 8 map archive: `{FIG8_MAP_ARCHIVE}`
- Figure 8 stress-zone table used only for lineage validation: `{FIG8_STRESS_TABLE}`

## Figure 7 Minimal Fix

Figure 7 was only enlarged / exported with a larger canvas and modest font,
marker, and error-bar scaling using the same final rendering structure.

Confirmed unchanged:

- data values,
- panel arrangement,
- model list and model order,
- target order,
- colors,
- legend contents,
- right-side best-count panels,
- interpretation and scientific logic.

No SARIMA or baseline rows were added.

Outputs:

- PNG: `{fig7_png}`
- PDF: `{fig7_pdf}`

## Figure 8 Minimal Fix

Figure 8 was changed only by removing the unreadable in-panel corner annotation
boxes. The severe-response statistics remain reported in Table 4 and Section
4.3.

Confirmed unchanged:

- map arrays and source data,
- land-type order,
- period order,
- subplot layout,
- color scale,
- per-panel colorbars,
- titles and axis labels,
- interpretation and scientific logic.

Outputs:

- PNG: `{fig8_png}`
- PDF: `{fig8_pdf}`

## Notes

The previously generated redesign-style readability outputs are not used in
this package. This package is the minimal presentation-only fix requested by
the author.
"""
    (REPORT / "FINAL_minimal_figure7_figure8_fix_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    ensure_directories()
    fig7_png, fig7_pdf = render_figure7_minimal()
    fig8_png, fig8_pdf = render_figure8_no_boxes()
    write_report(fig7_png, fig7_pdf, fig8_png, fig8_pdf)
    shutil.copy2(__file__, TOOLS / Path(__file__).name)
    print(fig7_png)
    print(fig7_pdf)
    print(fig8_png)
    print(fig8_pdf)
    print(REPORT / "FINAL_minimal_figure7_figure8_fix_report.md")


if __name__ == "__main__":
    main()
