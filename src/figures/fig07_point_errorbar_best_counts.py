from __future__ import annotations

import hashlib
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "source_tables" / "Figure_7_source.csv"

TARGETS = ["GNDVI", "GOSIF/SIF", "GPP", "LST", "NDVI", "NIRv", "SPEI-1", "TWS", "VPDmax"]
LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
MODELS = ["N-HiTS", "LSTM", "XGBoost", "TCN", "N-BEATS", "BiLSTM"]
COLORS = {
    "N-HiTS": "#E24A4A",
    "LSTM": "#FF8C32",
    "XGBoost": "#55B455",
    "TCN": "#9F7BC3",
    "N-BEATS": "#9A7267",
    "BiLSTM": "#4B95C4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(SOURCE)
    required = {
        "land_type_display",
        "target_display",
        "model",
        "mean_r2",
        "sd_r2",
        "mean_normalized_rmse",
        "sd_normalized_rmse",
        "available_folds",
        "available_rows",
        "available_seeds",
        "data_status",
        "source_metric_table",
        "metric_uncertainty_definition",
    }
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing corrected-source columns: {sorted(missing)}")
    if len(source) != 216:
        raise ValueError(f"Expected 216 model-task rows, found {len(source)}")
    if set(source["target_display"]) != set(TARGETS):
        raise ValueError("Target set does not match the finalized corrected Figure 7 source")
    if set(source["land_type_display"]) != set(LANDS):
        raise ValueError("Land-type set does not match the finalized corrected Figure 7 source")
    if set(source["model"]) != set(MODELS):
        raise ValueError("Model set does not match the finalized corrected Figure 7 source")
    if not source["available_folds"].eq(4).all():
        raise ValueError("Every plotted model-task row must summarize four rolling-origin folds")
    duplicates = source.duplicated(["land_type_display", "target_display", "model"])
    if duplicates.any():
        raise ValueError("Duplicate target x land type x model rows found")

    target_rank = {name: i for i, name in enumerate(TARGETS)}
    land_rank = {name: i for i, name in enumerate(LANDS)}
    model_rank = {name: i for i, name in enumerate(MODELS)}
    target_gap = 0.72
    model_offsets = dict(zip(MODELS, np.linspace(-0.18, 0.18, len(MODELS))))

    plot = source.copy()
    plot["target_order"] = plot["target_display"].map(target_rank)
    plot["land_order"] = plot["land_type_display"].map(land_rank)
    plot["model_order"] = plot["model"].map(model_rank)
    plot["land_x"] = plot["target_order"] * (len(LANDS) + target_gap) + plot["land_order"]
    plot["plot_x"] = plot["land_x"] + plot["model"].map(model_offsets)
    plot = plot.sort_values(["target_order", "land_order", "model_order"]).reset_index(drop=True)

    task_group = ["land_type_display", "target_display"]
    winner_rows = []
    for (land, target), task in plot.groupby(task_group, sort=False):
        best_r2 = task[task.mean_r2.eq(task.mean_r2.max())]
        best_nrmse = task[task.mean_normalized_rmse.eq(task.mean_normalized_rmse.min())]
        if len(best_r2) != 1 or len(best_nrmse) != 1:
            raise ValueError(f"Non-unique descriptive winner for {land} / {target}")
        winner_rows.append(
            {
                "land_type": land,
                "target": target,
                "highest_mean_r2_model": best_r2.iloc[0].model,
                "highest_mean_r2": best_r2.iloc[0].mean_r2,
                "lowest_mean_nrmse_model": best_nrmse.iloc[0].model,
                "lowest_mean_nrmse": best_nrmse.iloc[0].mean_normalized_rmse,
            }
        )
    winners = pd.DataFrame(winner_rows)
    counts = pd.DataFrame({"model": MODELS})
    counts["highest_mean_r2_task_count"] = counts.model.map(winners.highest_mean_r2_model.value_counts()).fillna(0).astype(int)
    counts["lowest_mean_nrmse_task_count"] = counts.model.map(winners.lowest_mean_nrmse_model.value_counts()).fillna(0).astype(int)
    if counts.highest_mean_r2_task_count.sum() != 36 or counts.lowest_mean_nrmse_task_count.sum() != 36:
        raise ValueError("Descriptive task counts must sum to 36 for each metric")
    return source, plot, counts, winners


def add_two_level_xlabels(axis, land_x: dict[tuple[str, str], float], centers: dict[str, float]) -> None:
    axis.set_xticks([])
    for target in TARGETS:
        axis.text(
            centers[target],
            -0.035,
            target,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            clip_on=False,
        )
        for land in LANDS:
            axis.text(
                land_x[(target, land)],
                -0.092,
                land,
                transform=axis.get_xaxis_transform(),
                ha="right",
                va="top",
                fontsize=7.3,
                rotation=40,
                clip_on=False,
            )


def count_panel(axis, counts: pd.DataFrame, column: str, title: str) -> None:
    values = counts.set_index("model").reindex(MODELS)[column]
    y = np.arange(len(MODELS))
    bars = axis.barh(
        y,
        values,
        color=[COLORS[model] for model in MODELS],
        edgecolor="#333333",
        linewidth=0.75,
        alpha=0.94,
        height=0.72,
    )
    for bar, value in zip(bars, values):
        axis.text(float(value) + 0.22, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center", fontsize=9, fontweight="bold")
    axis.set_yticks(y, MODELS)
    axis.invert_yaxis()
    axis.set_xlim(0, max(4.0, float(values.max()) + 2.0))
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axis.set_xlabel("Task count (out of 36)", fontsize=9, fontweight="semibold")
    axis.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    axis.grid(axis="x", alpha=0.25, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(axis="both", labelsize=8.5)


def render(plot: pd.DataFrame, counts: pd.DataFrame) -> None:
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(19.0, 10.6), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[5.8, 1.28],
        height_ratios=[1, 1],
        left=0.065,
        right=0.985,
        top=0.865,
        bottom=0.115,
        hspace=0.52,
        wspace=0.10,
    )
    ax_r2 = fig.add_subplot(grid[0, 0])
    ax_nrmse = fig.add_subplot(grid[1, 0])
    ax_r2_count = fig.add_subplot(grid[0, 1])
    ax_nrmse_count = fig.add_subplot(grid[1, 1])

    land_x = plot.drop_duplicates(["target_display", "land_type_display"]).set_index(["target_display", "land_type_display"]).land_x.to_dict()
    centers = {target: np.mean([land_x[(target, land)] for land in LANDS]) for target in TARGETS}
    first_x = min(land_x.values()) - 0.55
    last_x = max(land_x.values()) + 0.55

    for target_index, target in enumerate(TARGETS):
        left = min(land_x[(target, land)] for land in LANDS) - 0.52
        right = max(land_x[(target, land)] for land in LANDS) + 0.52
        if target_index % 2 == 0:
            for axis in (ax_r2, ax_nrmse):
                axis.axvspan(left, right, color="#808080", alpha=0.075, zorder=0)
        for land in LANDS:
            xpos = land_x[(target, land)]
            for axis in (ax_r2, ax_nrmse):
                axis.axvline(xpos, color="#8C8C8C", linestyle="--", linewidth=0.55, alpha=0.36, zorder=1)

    r2_bottom = np.floor((plot.mean_r2 - plot.sd_r2).min() * 10) / 10 - 0.05
    r2_top = max(1.05, float((plot.mean_r2 + plot.sd_r2).max()) + 0.04)
    nrmse_top = np.ceil((plot.mean_normalized_rmse + plot.sd_normalized_rmse).max() * 10) / 10 + 0.05
    ax_r2.axhspan(r2_bottom, 0, facecolor="#F4B8B8", alpha=0.42, zorder=0)
    ax_r2.axhline(0, color="#A33A3A", linestyle="--", linewidth=1.0, alpha=0.85, zorder=2)

    for model in MODELS:
        data = plot[plot.model.eq(model)]
        ax_r2.errorbar(
            data.plot_x,
            data.mean_r2,
            yerr=data.sd_r2,
            fmt="o",
            markersize=4.8,
            color=COLORS[model],
            ecolor=COLORS[model],
            elinewidth=0.65,
            capsize=1.6,
            capthick=0.65,
            markeredgecolor="#242424",
            markeredgewidth=0.45,
            alpha=0.92,
            zorder=4,
        )
        ax_nrmse.errorbar(
            data.plot_x,
            data.mean_normalized_rmse,
            yerr=data.sd_normalized_rmse,
            fmt="o",
            markersize=4.8,
            color=COLORS[model],
            ecolor=COLORS[model],
            elinewidth=0.65,
            capsize=1.6,
            capthick=0.65,
            markeredgecolor="#242424",
            markeredgewidth=0.45,
            alpha=0.92,
            zorder=4,
        )

    for axis, ylabel, ylim in [
        (ax_r2, r"Mean $R^2$", (r2_bottom, r2_top)),
        (ax_nrmse, "Mean nRMSE", (0, nrmse_top)),
    ]:
        axis.set_xlim(first_x, last_x)
        axis.set_ylim(*ylim)
        axis.set_ylabel(ylabel, fontsize=12, fontweight="bold")
        axis.grid(axis="y", alpha=0.30, linewidth=0.65)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", labelsize=8.5)
        add_two_level_xlabels(axis, land_x, centers)

    count_panel(ax_r2_count, counts, "highest_mean_r2_task_count", "Highest mean $R^2$\n(descriptive)")
    count_panel(ax_nrmse_count, counts, "lowest_mean_nrmse_task_count", "Lowest mean nRMSE\n(descriptive)")

    legend = [Patch(facecolor=COLORS[model], edgecolor="#333333", linewidth=0.7, label=model) for model in MODELS]
    fig.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.066, 0.914),
        ncol=6,
        frameon=True,
        edgecolor="#8A8A8A",
        fontsize=9.2,
        columnspacing=1.25,
        handlelength=1.8,
    )
    fig.suptitle("Forecasting performance across land types and targets", fontsize=15, fontweight="bold", y=0.974)
    fig.text(
        0.5,
        0.944,
        "Four expanding rolling-origin test periods: 2017–2018, 2019–2020, 2021–2022, and 2023–2024",
        ha="center",
        fontsize=10.3,
        color="#3F3F3F",
    )
    fig.text(
        0.40,
        0.025,
        "Points show corrected model-task means; error bars show the finalized SD across available fold/seed metric rows. Higher R² and lower nRMSE indicate better fit.",
        ha="center",
        fontsize=8.2,
        color="#454545",
    )
    fig.text(
        0.895,
        0.025,
        "Best-model counts are descriptive only; they do not establish statistical superiority.",
        ha="center",
        fontsize=8.0,
        color="#454545",
    )
    fig.savefig(HERE / "Figure_7_original_design_corrected_TRUE_BiLSTM.png", dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(HERE / "Figure_7_original_design_corrected_TRUE_BiLSTM.pdf", bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def verify_and_write(source: pd.DataFrame, plot: pd.DataFrame, counts: pd.DataFrame, winners: pd.DataFrame) -> None:
    output_columns = [
        "target_display",
        "land_type_display",
        "model",
        "mean_r2",
        "sd_r2",
        "mean_normalized_rmse",
        "sd_normalized_rmse",
        "available_folds",
        "available_rows",
        "available_seeds",
        "metric_uncertainty_definition",
        "data_status",
        "source_metric_table",
        "target_order",
        "land_order",
        "model_order",
        "land_x",
        "plot_x",
    ]
    plot_path = HERE / "Figure_7_original_design_corrected_plot_values.csv"
    count_path = HERE / "Figure_7_original_design_corrected_best_model_counts.csv"
    winner_path = HERE / "Figure_7_original_design_corrected_task_winners.csv"
    plot[output_columns].to_csv(plot_path, index=False)
    counts.to_csv(count_path, index=False)
    winners.to_csv(winner_path, index=False)

    reloaded = pd.read_csv(plot_path)
    keys = ["target_display", "land_type_display", "model"]
    metric_columns = ["mean_r2", "sd_r2", "mean_normalized_rmse", "sd_normalized_rmse"]
    comparison = reloaded.merge(source[keys + metric_columns], on=keys, suffixes=("_plot", "_source"), validate="one_to_one")
    metric_diffs = {column: float((comparison[f"{column}_plot"] - comparison[f"{column}_source"]).abs().max()) for column in metric_columns}

    source_r2 = source.loc[source.groupby(["land_type_display", "target_display"]).mean_r2.idxmax(), "model"].value_counts().reindex(MODELS, fill_value=0).astype(int)
    source_nrmse = source.loc[source.groupby(["land_type_display", "target_display"]).mean_normalized_rmse.idxmin(), "model"].value_counts().reindex(MODELS, fill_value=0).astype(int)
    plotted_counts = counts.set_index("model").reindex(MODELS)
    checks = [
        ("source_row_count_is_216", len(source) == 216, f"rows={len(source)}"),
        ("source_has_36_tasks", source.groupby(["land_type_display", "target_display"]).ngroups == 36, f"tasks={source.groupby(['land_type_display', 'target_display']).ngroups}"),
        ("six_models_per_task", source.groupby(["land_type_display", "target_display"]).size().eq(6).all(), "all tasks have six models"),
        ("all_rows_cover_four_folds", source.available_folds.eq(4).all(), f"fold values={[int(value) for value in sorted(source.available_folds.unique())]}"),
        ("true_bilstm_rows_complete", len(source[source.model.eq("BiLSTM")]) == 36, f"BiLSTM rows={len(source[source.model.eq('BiLSTM')])}"),
        ("plotted_means_and_sd_exact", max(metric_diffs.values()) == 0.0, f"maximum absolute differences={metric_diffs}"),
        ("r2_counts_match_source", np.array_equal(plotted_counts.highest_mean_r2_task_count.to_numpy(), source_r2.to_numpy()), f"counts={plotted_counts.highest_mean_r2_task_count.to_dict()}"),
        ("nrmse_counts_match_source", np.array_equal(plotted_counts.lowest_mean_nrmse_task_count.to_numpy(), source_nrmse.to_numpy()), f"counts={plotted_counts.lowest_mean_nrmse_task_count.to_dict()}"),
        ("r2_counts_sum_to_36", int(plotted_counts.highest_mean_r2_task_count.sum()) == 36, f"sum={int(plotted_counts.highest_mean_r2_task_count.sum())}"),
        ("nrmse_counts_sum_to_36", int(plotted_counts.lowest_mean_nrmse_task_count.sum()) == 36, f"sum={int(plotted_counts.lowest_mean_nrmse_task_count.sum())}"),
    ]
    verification = pd.DataFrame(checks, columns=["check", "passed", "detail"])
    verification.to_csv(HERE / "Figure_7_original_design_corrected_numerical_verification.csv", index=False)
    pd.DataFrame(
        [
            {
                "authoritative_source": str(SOURCE),
                "source_sha256": sha256(SOURCE),
                "source_rows": len(source),
                "source_tasks": 36,
                "models_per_task": 6,
                "rolling_origin_test_periods": "2017–2018; 2019–2020; 2021–2022; 2023–2024",
            }
        ]
    ).to_csv(HERE / "Figure_7_original_design_corrected_source_provenance.csv", index=False)
    if not verification.passed.all():
        raise RuntimeError("Numerical verification failed")

    count_text = "; ".join(
        f"{row.model}: {row.highest_mean_r2_task_count} R² / {row.lowest_mean_nrmse_task_count} nRMSE"
        for row in counts.itertuples(index=False)
    )
    report = f"""# Revised Figure 7 — original-design recreation

## Outcome

The original “Models' Performances (2024 test data)” visual design was recreated with the finalized corrected forecasting results. No old Figure-7 numerical value was used.

- Upper main panel: corrected mean R² for all 36 target × land-type tasks and six models.
- Lower main panel: corrected mean nRMSE for the same tasks and models.
- Error bars: finalized SD across the available fold/seed metric rows for each model-task combination.
- Test periods: 2017–2018, 2019–2020, 2021–2022, and 2023–2024.
- TRUE-BiLSTM: all 36 BiLSTM rows come from the finalized accepted/corrected branch recorded in the source table.
- Right panels: descriptive highest-mean-R² and lowest-mean-nRMSE counts across 36 tasks.

## Descriptive counts

{count_text}.

These counts are descriptive only and do not establish statistical superiority.

## Numerical verification

All {len(verification)} checks passed. Reloaded plotted means and uncertainty values match the finalized source exactly (maximum absolute difference = 0), both count columns match independent recomputation from the authoritative table, and each count column sums to 36.

## Files

- `Figure_7_original_design_corrected_TRUE_BiLSTM.png`
- `Figure_7_original_design_corrected_TRUE_BiLSTM.pdf`
- `Figure_7_original_design_corrected_plot_values.csv`
- `Figure_7_original_design_corrected_task_winners.csv`
- `Figure_7_original_design_corrected_best_model_counts.csv`
- `Figure_7_original_design_corrected_numerical_verification.csv`
- `Figure_7_original_design_corrected_source_provenance.csv`
- `render_figure7_original_design_corrected.py`

Previous Figure 7 files were not overwritten.
"""
    (HERE / "FIGURE_7_ORIGINAL_DESIGN_RECREATION.md").write_text(report, encoding="utf-8")


def main() -> None:
    source, plot, counts, winners = prepare()
    verify_and_write(source, plot, counts, winners)
    render(plot, counts)
    print("Recreated corrected Figure 7 and passed all numerical checks.")


if __name__ == "__main__":
    main()
