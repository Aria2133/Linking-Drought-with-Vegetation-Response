"""Merge previous/no-QA GNDVI forecasting rerun into frozen Figure 7 layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


DL_ROOT = Path(r"<PROJECT_ROOT>\DL_GNDVI_previous_noQA_rerun")
ORIG_FIG7_ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_raster_reviewer_safe_candidate\figure7_final_layout_revision"
)
OUT_FIG = DL_ROOT / "figures"
OUT_TABLE = DL_ROOT / "tables"
OUT_REPORT = DL_ROOT / "report"

MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
TARGETS = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
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


def summarize_gndvi() -> pd.DataFrame:
    metrics = pd.read_csv(DL_ROOT / "tables" / "reviewer_safe_all_pixel_metrics.csv")
    metrics = metrics[(metrics["variable"] == "GNDVI") & (metrics["model"].isin(MODELS))].copy()
    grouped = metrics.groupby(["land_type", "variable", "model"], dropna=False)
    rows = []
    for (land, variable, model), group in grouped:
        rows.append(
            {
                "land_type": land,
                "target": variable,
                "model": model,
                "selected_configuration": "common_reviewer_safe_previous_noQA_GNDVI",
                "mean_normalized_rmse": group["normalized_rmse"].mean(),
                "mean_r2": group["r2"].mean(),
                "available_folds": group["fold"].nunique(),
                "available_rows": len(group),
                "available_seeds": group["seed"].nunique(),
                "sd_normalized_rmse": group["normalized_rmse"].std(ddof=1),
                "sd_r2": group["r2"].std(ddof=1),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLE / "figure7_previous_noQA_GNDVI_panel_rows.csv", index=False)
    return out


def recompute_counts(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = {model: {"best_r2_count": 0.0, "best_normalized_rmse_count": 0.0} for model in MODELS}
    winners = []
    for (land, target), group in panel.groupby(["land_type", "target"]):
        r2_max = group["mean_r2"].max()
        r2_models = group[np.isclose(group["mean_r2"], r2_max, atol=1e-12)]["model"].tolist()
        nrmse_min = group["mean_normalized_rmse"].min()
        nrmse_models = group[np.isclose(group["mean_normalized_rmse"], nrmse_min, atol=1e-12)]["model"].tolist()
        for model in r2_models:
            counts[model]["best_r2_count"] += 1.0 / len(r2_models)
        for model in nrmse_models:
            counts[model]["best_normalized_rmse_count"] += 1.0 / len(nrmse_models)
        winners.append(
            {
                "land_type": land,
                "target": target,
                "best_r2_value": r2_max,
                "best_r2_models": ";".join(r2_models),
                "best_r2_tie_count": len(r2_models),
                "best_normalized_rmse_value": nrmse_min,
                "best_normalized_rmse_models": ";".join(nrmse_models),
                "best_normalized_rmse_tie_count": len(nrmse_models),
                "tie_rule": "models equal within 1e-12 share one task credit equally",
            }
        )
    count_rows = []
    for model in MODELS:
        count_rows.append(
            {
                "model": model,
                **counts[model],
                "tasks_total": 36,
                "tie_rule": "models equal within 1e-12 share one task credit equally",
            }
        )
    return pd.DataFrame(count_rows), pd.DataFrame(winners)


def add_target_guides(axis, lower, upper):
    for index in range(len(TARGETS)):
        if index % 2 == 0:
            axis.axvspan(index - 0.5, index + 0.5, color="#EDF2F5", alpha=0.58, zorder=0)
    for boundary in np.arange(0.5, len(TARGETS) - 0.49, 1.0):
        axis.axvline(boundary, color="#9EA7B0", linewidth=0.62, alpha=0.52, zorder=1)
    axis.set_ylim(lower, upper)


def style_metric_axis(axis):
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#C5CBD1", linewidth=0.60, alpha=0.66)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#4F555B")
    axis.spines["bottom"].set_color("#4F555B")
    axis.tick_params(axis="both", colors="#25282B", width=0.75)


def draw_count_panel(axis, counts, column, title):
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
        axis.text(value + max(0.22, maximum * 0.022), bar.get_y() + bar.get_height() / 2, label, va="center", ha="left", fontsize=9.1, fontweight="bold", color="#2E3236")
    axis.set_yticks(y)
    axis.set_yticklabels(MODELS, fontsize=8.9)
    axis.invert_yaxis()
    axis.set_xlim(0, maximum + max(1.55, maximum * 0.125))
    axis.set_xlabel("Task wins (out of 36)", fontsize=8.4)
    axis.set_title(title, fontsize=10.8, fontweight="bold", pad=8)
    axis.grid(axis="x", color="#D1D5DA", linewidth=0.55, alpha=0.64)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#72787E")
    axis.tick_params(axis="y", length=0)


def render(panel: pd.DataFrame, counts: pd.DataFrame) -> None:
    r2_lower = float((panel["mean_r2"] - panel["sd_r2"]).min())
    r2_bottom = max(min(-0.25, np.floor((r2_lower - 0.05) * 10) / 10), -2.1)
    r2_top = max(1.05, np.ceil(float((panel["mean_r2"] + panel["sd_r2"]).max() + 0.03) * 20) / 20)
    nrmse_upper = float((panel["mean_normalized_rmse"] + panel["sd_normalized_rmse"]).max())
    nrmse_top = np.ceil((nrmse_upper + 0.08) * 10) / 10

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.6,
            "axes.labelsize": 10.8,
            "xtick.labelsize": 9.1,
            "ytick.labelsize": 9.3,
            "legend.fontsize": 10.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(21.8, 15.2))
    grid = fig.add_gridspec(4, 3, width_ratios=[6.25, 6.25, 1.38], left=0.048, right=0.988, top=0.902, bottom=0.078, hspace=0.36, wspace=0.135)
    metric_axes = [(fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1])) for row in range(4)]
    count_r2_axis = fig.add_subplot(grid[0:2, 2])
    count_nrmse_axis = fig.add_subplot(grid[2:4, 2])

    x = np.arange(len(TARGETS), dtype=float)
    offsets = np.linspace(-0.30, 0.30, len(MODELS))
    panel_letters = "abcdefgh"
    for row, land_type in enumerate(LAND_TYPES):
        r2_axis, nrmse_axis = metric_axes[row]
        add_target_guides(r2_axis, r2_bottom, r2_top)
        add_target_guides(nrmse_axis, 0, nrmse_top)
        for model, offset in zip(MODELS, offsets):
            data = panel[(panel["land_type"] == land_type) & (panel["model"] == model)].set_index("target").reindex(TARGETS)
            positions = x + offset
            for axis, means, errors in (
                (r2_axis, data["mean_r2"], data["sd_r2"]),
                (nrmse_axis, data["mean_normalized_rmse"], data["sd_normalized_rmse"]),
            ):
                axis.errorbar(positions, means, yerr=errors, fmt="none", capsize=4.1, elinewidth=1.40, capthick=1.26, ecolor=COLORS[model], alpha=0.98, zorder=5, clip_on=False)
                axis.scatter(positions, means, s=47, color=COLORS[model], edgecolor="white", linewidth=0.80, zorder=6)
        r2_axis.axhspan(r2_bottom, 0, color="#F4D9D9", alpha=0.42, zorder=0.5)
        r2_axis.axhline(0, color="#9B3030", linewidth=0.95, zorder=2)
        r2_axis.set_title(f"{panel_letters[2 * row]}) {LAND_LABELS[land_type]}: $R^2$", loc="left", fontweight="bold", pad=8)
        nrmse_axis.set_title(f"{panel_letters[2 * row + 1]}) {LAND_LABELS[land_type]}: normalized RMSE", loc="left", fontweight="bold", pad=8)
        r2_axis.set_ylabel(r"$R^2$")
        nrmse_axis.set_ylabel("Normalized RMSE")
        for axis in (r2_axis, nrmse_axis):
            axis.set_xlim(-0.58, len(TARGETS) - 0.42)
            axis.set_xticks(x)
            axis.set_xticklabels(TARGETS)
            axis.tick_params(axis="x", pad=5)
            style_metric_axis(axis)

    draw_count_panel(count_r2_axis, counts, "best_r2_count", "Best $R^2$ Count")
    draw_count_panel(count_nrmse_axis, counts, "best_normalized_rmse_count", "Best nRMSE Count")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=7.6, markerfacecolor=COLORS[model], markeredgecolor="white", markeredgewidth=0.72, label=model) for model in MODELS]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.949), frameon=False, columnspacing=2.0, handlelength=0.8, handletextpad=0.48)
    fig.suptitle("Reviewer-safe forecasting performance across land types and targets", fontsize=15.3, fontweight="bold", y=0.985)
    fig.text(0.5, 0.022, CAPTION, ha="center", fontsize=8.8, color="#44494E")
    fig.savefig(OUT_FIG / "Figure_7_previous_noQA_GNDVI_same_visualization.png", dpi=800, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_FIG / "Figure_7_previous_noQA_GNDVI_same_visualization.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_FIG.mkdir(exist_ok=True)
    OUT_TABLE.mkdir(exist_ok=True)
    OUT_REPORT.mkdir(exist_ok=True)
    old_panel = pd.read_csv(ORIG_FIG7_ROOT / "figure7_audited_panel_values_used.csv")
    new_gndvi = summarize_gndvi()
    combined = pd.concat([old_panel[old_panel["target"] != "GNDVI"], new_gndvi], ignore_index=True)
    combined = combined.sort_values(["land_type", "target", "model"])
    combined.to_csv(OUT_TABLE / "figure7_combined_previous_noQA_GNDVI_panel_values.csv", index=False)
    counts, winners = recompute_counts(combined)
    counts.to_csv(OUT_TABLE / "figure7_previous_noQA_GNDVI_global_best_counts.csv", index=False)
    winners.to_csv(OUT_TABLE / "figure7_previous_noQA_GNDVI_task_winners.csv", index=False)
    render(combined, counts)
    before = old_panel[old_panel["target"] == "GNDVI"].merge(
        new_gndvi,
        on=["land_type", "target", "model"],
        suffixes=("_old_strictQA", "_previous_noQA"),
    )
    before["delta_mean_r2"] = before["mean_r2_previous_noQA"] - before["mean_r2_old_strictQA"]
    before["delta_mean_normalized_rmse"] = before["mean_normalized_rmse_previous_noQA"] - before["mean_normalized_rmse_old_strictQA"]
    before.to_csv(OUT_TABLE / "figure7_GNDVI_before_after_previous_noQA_audit.csv", index=False)
    (OUT_REPORT / "Figure7_previous_noQA_GNDVI_render_report.md").write_text(
        "# Figure 7 Previous/no-QA GNDVI Render Report\n\n"
        f"Completed: {datetime.now().isoformat()}\n\n"
        "- Visualization format copied from the frozen final Figure 7 renderer.\n"
        "- Non-GNDVI panel rows are unchanged from the accepted reviewer-safe Figure 7 source table.\n"
        "- GNDVI rows were replaced with the GNDVI-only previous/no-QA reviewer-safe rerun metrics.\n"
        "- Global count panels were recomputed across the full 36 land-type x target tasks after replacing GNDVI rows.\n"
        "- SARIMA remains outside main Figure 7, as in the accepted design.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
