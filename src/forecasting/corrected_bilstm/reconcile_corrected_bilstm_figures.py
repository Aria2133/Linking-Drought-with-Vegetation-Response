from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
DP = HERE.parents[2]
CORRECTED = DP / "Docs" / "major_revision_analyses" / "corrected_true_BiLSTM_rerun_20260728"
DOWNSTREAM = CORRECTED / "08_downstream"
OLD_F7 = DP / "Docs" / "major_revision_analyses" / "ALTERNATIVE_Figure7_polar_redesign_no_detrend_update" / "Source tables used for plotting" / "figure7_alternative_plotting_table.csv"
S9_ROOT = DP / "Docs" / "major_revision_analyses" / "EWBI_from_existing_forecasts_scale_recovery_and_figure_20260728"
S9_SCRIPT = S9_ROOT / "scripts" / "finish_forecast_support_ewbi_analysis.py"
ECOLOGICAL_EWBI = DP / "Docs" / "major_revision_analyses" / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv"
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]


def render_figure7(plot: pd.DataFrame) -> None:
    colors = {"LSTM": "#1f77b4", "BiLSTM": "#9467bd", "TCN": "#2ca02c", "N-BEATS": "#ff7f0e", "N-HiTS": "#d62728", "XGBoost": "#17becf"}
    targets = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
    target_display = {"GNDVI": "GNDVI", "GOSIF": "GOSIF/SIF", "GPP": "GPP", "LST": "LST", "NDVI": "NDVI", "NIRv": "NIRv", "SPEI": "SPEI-1", "TWS": "TWS", "VPD": "VPDmax"}
    lands = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
    land_display = {"GRASSLANDS": "Grasslands", "NATURAL_FORESTS": "Natural Forests", "SAVANNAS": "Savannas", "SHRUBLANDS": "Shrublands"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.titlesize": 20, "axes.labelsize": 16, "legend.fontsize": 17})
    fig, axes = plt.subplots(4, 2, subplot_kw={"projection": "polar"}, figsize=(24, 33))
    fig.subplots_adjust(top=0.975, bottom=0.11, left=0.035, right=0.965, hspace=0.25, wspace=0.11)
    angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False)
    spacing = 2 * np.pi / len(targets)
    offsets = dict(zip(MODELS, np.linspace(-0.255 * spacing, 0.255 * spacing, len(MODELS))))
    angle_by_target = dict(zip(targets, angles))
    target_rank = {target: i for i, target in enumerate(targets)}
    nrmse_rmax = max(np.ceil(float((plot["mean_normalized_rmse"] + plot["sd_normalized_rmse"].fillna(0)).max()) * 4) / 4, 1.5)
    for row_i, land in enumerate(lands):
        for col_i, metric in enumerate(["R2", "nRMSE"]):
            ax = axes[row_i, col_i]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels([target_display[t] for t in targets], fontsize=14, fontweight="bold")
            ax.tick_params(axis="x", pad=17)
            if metric == "R2":
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                metric_col, sd_col = "mean_r2", "sd_r2"
                ax.set_title(f"{land_display[land]} - R$^2$", pad=28, fontweight="bold")
            else:
                ax.set_ylim(0, nrmse_rmax)
                ax.set_yticks(np.arange(0, nrmse_rmax + 0.001, 0.5))
                metric_col, sd_col = "mean_normalized_rmse", "sd_normalized_rmse"
                ax.set_title(f"{land_display[land]} - nRMSE", pad=28, fontweight="bold")
            ax.set_rlabel_position(235)
            ax.grid(True, alpha=0.30, linewidth=1.0)
            sub = plot[plot["land_type_internal"].eq(land)]
            for model in MODELS:
                m = sub[sub["model"].eq(model)].sort_values("target_internal", key=lambda s: s.map(target_rank))
                theta = np.array([angle_by_target[t] + offsets[model] for t in m["target_internal"]])
                values = m[metric_col].to_numpy(float)
                errors = m[sd_col].to_numpy(float)
                if metric == "R2":
                    positive = values >= 0
                    if positive.any():
                        ax.errorbar(theta[positive], values[positive], yerr=errors[positive], fmt="o", ms=10, capsize=4, color=colors[model], ecolor=colors[model], elinewidth=1.6, zorder=3)
                    if (~positive).any():
                        ax.scatter(theta[~positive], np.full((~positive).sum(), 0.026), marker="x", s=110, color=colors[model], linewidths=2.4, zorder=4)
                else:
                    ax.errorbar(theta, values, yerr=errors, fmt="o", ms=10, capsize=4, color=colors[model], ecolor=colors[model], elinewidth=1.6, zorder=3)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=colors[m], markeredgecolor=colors[m], markersize=13, label=m) for m in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 0.047), title="Learned models")
    fig.text(0.5, 0.022, "Higher R² and lower nRMSE indicate better performance. Error bars are SD across available seed/fold metric rows; × at center denotes mean R² < 0.", ha="center", fontsize=13)
    fig.savefig(HERE / "Figure_7_corrected_TRUE_BiLSTM.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_yyyymm(value: object) -> float:
    text = str(value)
    if "-" in text:
        year, month = text.split("-")[:2]
    else:
        year, month = text[:4], text[4:6]
    return int(year) + (int(month) - 1) / 12


def render_s7(panel_a: pd.DataFrame, panel_b: pd.DataFrame, timeline: pd.DataFrame) -> None:
    model_colors = {"Persistence": "#6F6F6F", "Seasonal persistence": "#A3A3A3", "Monthly climatology": "#B8A900", "RBF-epsilon-SVR": "#4DAF4A", "LSTM": "#2F6DAE", "BiLSTM": "#8556A8", "TCN": "#3B8C4A", "N-BEATS": "#F28E2B", "N-HiTS": "#D84A4A", "XGBoost": "#00A6A6"}
    mpl.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11.5, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.0, 7.3), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    sub_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.12)
    y = np.arange(len(panel_a))
    colors = [model_colors[m] for m in panel_a["model"]]
    ax_a1 = fig.add_subplot(sub_a[0, 0])
    ax_a1.errorbar(panel_a["mean_r2"], y, xerr=panel_a["sd_r2"], fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1)
    ax_a1.scatter(panel_a["mean_r2"], y, c=colors, s=38, zorder=3)
    ax_a1.axvline(0, color="#8B1A1A", lw=0.85)
    ax_a1.set_yticks(y, panel_a["model"]); ax_a1.invert_yaxis(); ax_a1.set_xlabel("Mean R²"); ax_a1.set_title("Monthly-aggregate R²"); ax_a1.grid(axis="x")
    ax_a1.text(-0.28, 1.07, "(a)", transform=ax_a1.transAxes, fontsize=12.5, fontweight="bold", va="top")
    ax_a2 = fig.add_subplot(sub_a[0, 1], sharey=ax_a1)
    ax_a2.errorbar(panel_a["mean_normalized_rmse"], y, xerr=panel_a["sd_normalized_rmse"], fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1)
    ax_a2.scatter(panel_a["mean_normalized_rmse"], y, c=colors, s=38, zorder=3)
    ax_a2.set_xlabel("Mean nRMSE"); ax_a2.set_title("Monthly-aggregate nRMSE"); ax_a2.tick_params(labelleft=False); ax_a2.grid(axis="x")
    sub_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.28)
    x = np.arange(len(panel_b))
    ax_b1 = fig.add_subplot(sub_b[0, 0]); ax_b1.errorbar(x, panel_b["mean_r2"], yerr=panel_b["sd_r2"], fmt="o-", ms=5.2, capsize=2.8, lw=1.2, color="#2F6DAE")
    ax_b1.axhline(0, color="#4A4A4A", lw=0.8); ax_b1.set_xticks(x, panel_b["test_period"]); ax_b1.set_ylabel("Mean pixel-level R²"); ax_b1.set_title("Pixel-level R² by fold"); ax_b1.grid(axis="y")
    ax_b1.text(0.03, 0.98, "(b)", transform=ax_b1.transAxes, fontsize=12.5, fontweight="bold", va="top")
    ax_b2 = fig.add_subplot(sub_b[0, 1]); ax_b2.errorbar(x, panel_b["mean_nrmse"], yerr=panel_b["sd_nrmse"], fmt="s-", ms=5.2, capsize=2.8, lw=1.2, color="#D84A4A")
    ax_b2.set_xticks(x, panel_b["test_period"]); ax_b2.set_ylabel("Mean pixel-level nRMSE"); ax_b2.set_title("Pixel-level nRMSE by fold"); ax_b2.grid(axis="y")
    ax_c = fig.add_subplot(gs[1, :]); ax_c.set_xlim(2000.5, 2025); ax_c.set_ylim(0.4, 4.6); ax_c.set_yticks([1, 2, 3, 4], [f"Fold {i}" for i in [1, 2, 3, 4]])
    for row in timeline.itertuples(index=False):
        fold = int(row.fold); core_end = parse_yyyymm(row.core_model_training_end) + 1/12; val_start = parse_yyyymm(row.internal_validation_start); val_end = parse_yyyymm(row.internal_validation_end) + 1/12; test_start = parse_yyyymm(row.test_start); test_end = parse_yyyymm(row.test_end) + 1/12
        ax_c.add_patch(Rectangle((2001, fold-0.2), core_end-2001, 0.4, facecolor="#8A8A8A", edgecolor="none")); ax_c.add_patch(Rectangle((val_start, fold-0.2), val_end-val_start, 0.4, facecolor="#E9B949", edgecolor="none")); ax_c.add_patch(Rectangle((test_start, fold-0.2), test_end-test_start, 0.4, facecolor="#D95D39", edgecolor="none"))
    ax_c.set_xticks([2001, 2005, 2009, 2013, 2017, 2021, 2025]); ax_c.set_xlabel("Calendar year"); ax_c.set_title("Rolling-origin design"); ax_c.grid(axis="x")
    ax_c.legend(handles=[Patch(facecolor="#8A8A8A", label="Training"), Patch(facecolor="#E9B949", label="Validation"), Patch(facecolor="#D95D39", label="Test")], frameon=False, ncol=3, loc="lower center")
    ax_c.text(-0.03, 1.07, "(c)", transform=ax_c.transAxes, fontsize=12.5, fontweight="bold", va="top")
    fig.savefig(HERE / "Figure_S7_corrected_TRUE_BiLSTM.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build_change_table(old_f7: pd.DataFrame, new_f7: pd.DataFrame) -> pd.DataFrame:
    keys = ["land_type_internal", "target_internal", "model"]
    metrics = ["mean_r2", "mean_normalized_rmse", "sd_r2", "sd_normalized_rmse"]
    old = old_f7[keys + metrics].rename(columns={m: f"old_{m}" for m in metrics})
    new = new_f7[keys + metrics].rename(columns={m: f"corrected_{m}" for m in metrics})
    comparison = old.merge(new, on=keys, validate="one_to_one")
    rows = []
    for row in comparison[comparison["model"].eq("BiLSTM")].itertuples(index=False):
        for metric in metrics:
            old_value = getattr(row, f"old_{metric}"); new_value = getattr(row, f"corrected_{metric}")
            if not np.isclose(old_value, new_value, equal_nan=True, atol=1e-15, rtol=0):
                rows.append({"figure": "Figure 7", "cell": f"{row.land_type_internal}|{row.target_internal}|BiLSTM", "metric": metric, "old_value": old_value, "corrected_value": new_value, "change": new_value-old_value, "changed": True})
    run_level = pd.read_csv(CORRECTED / "06_metrics" / "old_vs_corrected_BiLSTM_comparison.csv")
    for row in run_level.itertuples(index=False):
        for metric, old_col, new_col in [("pixel_r2", "old_pixel_r2", "corrected_pixel_r2"), ("pixel_nrmse", "old_pixel_nrmse", "corrected_pixel_nrmse"), ("monthly_r2", "old_monthly_r2", "corrected_monthly_r2"), ("monthly_nrmse", "old_monthly_nrmse", "corrected_monthly_nrmse")]:
            old_value, new_value = getattr(row, old_col), getattr(row, new_col)
            rows.append({"figure": "Underlying corrected run", "cell": f"{row.land_type}|{row.variable}|fold{row.fold}|seed{row.seed}", "metric": metric, "old_value": old_value, "corrected_value": new_value, "change": new_value-old_value, "changed": not np.isclose(old_value, new_value, equal_nan=True, atol=1e-15, rtol=0)})
    return pd.DataFrame(rows)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    corrected_f7 = pd.read_csv(DOWNSTREAM / "final_corrected_Figure_7_source_table.csv")
    old_f7 = pd.read_csv(OLD_F7)
    corrected_f7.to_csv(HERE / "Figure_7_corrected_source_table.csv", index=False)
    render_figure7(corrected_f7)

    panel_a = pd.read_csv(DOWNSTREAM / "final_corrected_Figure_S6_panel_a_source.csv")
    panel_b = pd.read_csv(DOWNSTREAM / "final_corrected_Figure_S6_panel_b_source.csv")
    panel_c = pd.read_csv(DOWNSTREAM / "final_corrected_Figure_S6_panel_c_source.csv")
    panel_a.to_csv(HERE / "Figure_S7_corrected_panel_a_source.csv", index=False)
    panel_b.to_csv(HERE / "Figure_S7_corrected_panel_b_source.csv", index=False)
    panel_c.to_csv(HERE / "Figure_S7_corrected_panel_c_source.csv", index=False)
    pd.read_csv(DOWNSTREAM / "final_corrected_Figure_S6_combined_monthly_metrics_with_SVR.csv").to_csv(HERE / "Figure_S7_corrected_exact_monthly_metrics.csv", index=False)
    pd.read_csv(DOWNSTREAM / "final_corrected_Figure_S6_combined_pixel_metrics_learned_source.csv").to_csv(HERE / "Figure_S7_corrected_exact_pixel_metrics.csv", index=False)
    render_s7(panel_a, panel_b, panel_c)

    # S9 was created after the VPD/LST corrected rerun. Recompute its metrics and
    # render directly from its saved monthly EWBI source; SPEI and TWS are retained
    # because their accepted implementations were already truly bidirectional.
    s9_out = HERE / "Figure_S9_rebuild"
    module = runpy.run_path(str(S9_SCRIPT))
    module["ROOT"] = s9_out
    module["ECOLOGICAL_EWBI"] = ECOLOGICAL_EWBI
    for function_name in ["ensure_dirs", "compute_forecast_metrics", "compute_support_metrics", "plot_figure"]:
        module[function_name].__globals__["ROOT"] = s9_out
        module[function_name].__globals__["ECOLOGICAL_EWBI"] = ECOLOGICAL_EWBI
    module["ensure_dirs"]()
    ewbi = pd.read_csv(S9_ROOT / "03_ewbi" / "observed_and_forecast_derived_EWBI_monthly.csv")
    metrics, fold_metrics = module["compute_forecast_metrics"](ewbi)
    support, support_joined = module["compute_support_metrics"](ewbi)
    ewbi.to_csv(HERE / "Figure_S9_corrected_exact_monthly_source.csv", index=False)
    metrics.to_csv(HERE / "Figure_S9_corrected_source_table.csv", index=False)
    fold_metrics.to_csv(HERE / "Figure_S9_corrected_fold_source_table.csv", index=False)
    support.to_csv(HERE / "Figure_S9_corrected_support_source_table.csv", index=False)
    module["plot_figure"](metrics, ewbi, support_joined)
    generated = s9_out / "04_figure" / "Figure_forecast_derived_EWBI.png"
    generated.replace(HERE / "Figure_S9_corrected_TRUE_BiLSTM.png")

    changes = build_change_table(old_f7, corrected_f7)
    changes.to_csv(HERE / "bilstm_old_vs_corrected_values.csv", index=False)
    figure7_changed_cells = changes[(changes["figure"].eq("Figure 7")) & changes["changed"]]["cell"].nunique()
    underlying_changed = changes[(changes["figure"].eq("Underlying corrected run")) & changes["changed"]]["cell"].nunique()
    affected_targets = sorted(corrected_f7[(corrected_f7["model"].eq("BiLSTM")) & corrected_f7["data_status"].eq("patched_corrected_true_BiLSTM")]["target_internal"].unique())
    lines = [
        "# TRUE BiLSTM reconciliation",
        "",
        "## Result",
        "",
        f"Figure 7 and Figure S7 were rebuilt from the authoritative corrected TRUE-BiLSTM downstream tables. Figure 7 contains {figure7_changed_cells} changed land-type × target BiLSTM cells across {', '.join(affected_targets)}; the long audit table records every changed metric value. At the underlying fold/seed level, {underlying_changed} run cells differ from the old implementation.",
        "",
        "Figure S9 was regenerated from its exact saved monthly EWBI table. That table was created after the correction and already uses corrected TRUE-BiLSTM predictions for the affected EWBI components VPDmax and LST. SPEI-1 and TWS retain their accepted saved predictions because those two original implementations were already bidirectional and were intentionally not rerun in the corrected seven-target package. Thus no S9 numerical cell was fabricated or retrained.",
        "",
        "## Important branch definition",
        "",
        "The authoritative corrected branch patches the seven targets whose old model was not bidirectional (GNDVI, GOSIF/SIF, GPP, LST, NDVI, NIRv, VPDmax) and deliberately preserves SPEI-1 and TWS, which were already bidirectional. This is the same branch used by final corrected Figure S8 and the corrected best-model counts.",
        "",
        "## Outputs",
        "",
        "- `Figure_7_corrected_TRUE_BiLSTM.png` with its exact 216-row source table.",
        "- `Figure_S7_corrected_TRUE_BiLSTM.png` with panel and full metric source tables.",
        "- `Figure_S9_corrected_TRUE_BiLSTM.png` with monthly, metric, fold, and support source tables.",
        "- `bilstm_old_vs_corrected_values.csv` documents every old/corrected run metric and every changed Figure-7 cell value.",
        "",
        "No models were fitted, resumed, or retrained, and no pixels were manually edited or cropped.",
    ]
    (HERE / "BILSTM_RECONCILIATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Figure 7 changed cells: {figure7_changed_cells}")
    print(f"Underlying changed run cells: {underlying_changed}")
    print(f"S9 rows: {len(ewbi)}; metrics: {len(metrics)}")


if __name__ == "__main__":
    main()
