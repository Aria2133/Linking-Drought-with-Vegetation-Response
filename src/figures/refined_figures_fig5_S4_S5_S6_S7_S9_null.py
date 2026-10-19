from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT.parents[1]
COMP = PAPER / "AUDIT_TEMP" / "REVISION_COMPUTATIONAL_PHASE"
MAJOR = PAPER / "Docs" / "major_revision_analyses"
PNG = ROOT / "figures_png"
PDF = ROOT / "figures_pdf"
TABLES = ROOT / "source_tables"
NOTES = ROOT / "notes"
LOGS = ROOT / "logs"

LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#16869A",
    "Natural Forests": "#3B873E",
    "Savannas": "#E68A00",
    "Shrublands": "#B83A63",
}
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
RESPONSES = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
MODEL_COLORS = {
    "LSTM": "#1f77b4",
    "BiLSTM": "#9467bd",
    "TCN": "#2ca02c",
    "N-BEATS": "#ff7f0e",
    "N-HiTS": "#d62728",
    "XGBoost": "#17becf",
}
STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]


SOURCE_MAP = {
    "Figure_5_correlations.csv": COMP / "task1_figure5" / "figure5_raw_correlations.csv",
    "Figure_5_pairwise_date_coverage.csv": COMP / "task1_figure5" / "figure5_pairwise_date_coverage.csv",
    "Figure_5_exact_monthly_input.csv": COMP / "task1_figure5" / "figure5_exact_input_monthly_series.csv",
    "Figure_7_source.csv": COMP / "task5_bilstm_reconciliation" / "Figure_7_corrected_source_table.csv",
    "Figure_S4_validation_correlations.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_validation_correlations.csv",
    "Figure_S4_component_correlations.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_component_correlations.csv",
    "Figure_S4_leave_one_out.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_leave_one_out.csv",
    "Figure_S4_weights.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_weights.csv",
    "Figure_S4_period_summary.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_period_summary.csv",
    "Figure_S4_period_tests.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_period_tests.csv",
    "Figure_S4_exact_monthly_input.csv": COMP / "task6_supplement_harmonization" / "Figure_S4_harmonized_exact_monthly_source.csv",
    "Figure_S5_correlations.csv": COMP / "task6_supplement_harmonization" / "Figure_S5_harmonized_correlations.csv",
    "Figure_S5_exact_monthly_input.csv": COMP / "task6_supplement_harmonization" / "Figure_S5_harmonized_exact_monthly_source.csv",
    "Figure_S6_agreement.csv": COMP / "task6_supplement_harmonization" / "Figure_S6_harmonized_agreement.csv",
    "Figure_S6_occupancy.csv": COMP / "task6_supplement_harmonization" / "Figure_S6_harmonized_occupancy.csv",
    "Figure_S6_transition_matrices.csv": COMP / "task6_supplement_harmonization" / "Figure_S6_harmonized_transition_matrices.csv",
    "Figure_S6_exact_monthly_states.csv": COMP / "task6_supplement_harmonization" / "Figure_S6_harmonized_exact_monthly_states.csv",
    "Figure_S7_panel_a_source.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S7_corrected_panel_a_source.csv",
    "Figure_S7_panel_b_source.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S7_corrected_panel_b_source.csv",
    "Figure_S7_panel_c_source.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S7_corrected_panel_c_source.csv",
    "Figure_S7_exact_monthly_metrics.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S7_corrected_exact_monthly_metrics.csv",
    "Figure_S7_exact_pixel_metrics.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S7_corrected_exact_pixel_metrics.csv",
    "Figure_S9_metrics.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S9_corrected_source_table.csv",
    "Figure_S9_exact_monthly_source.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S9_corrected_exact_monthly_source.csv",
    "Figure_S9_fold_metrics.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S9_corrected_fold_source_table.csv",
    "Figure_S9_support_summary.csv": COMP / "task5_bilstm_reconciliation" / "Figure_S9_corrected_support_source_table.csv",
    "Figure_S_new_Figure5_deseasonalized_correlations.csv": COMP / "task1_figure5" / "figure5_deseasonalized_correlations.csv",
    "Figure_S_new_Figure5_deseasonalized_exact_monthly_input.csv": COMP / "task1_figure5" / "figure5_deseasonalized_exact_input_monthly_series.csv",
    "Figure_S_new_state_transition_null_summary.csv": COMP / "task3_transition_null" / "state_transition_null_summary.csv",
    "Figure_S_new_state_transition_null_distributions.csv": COMP / "task3_transition_null" / "state_transition_null_distributions.csv",
    "Figure_S_new_state_transition_null_exact_input.csv": COMP / "task3_transition_null" / "state_transition_null_exact_input_series.csv",
}


def reset_style() -> None:
    mpl.rcdefaults()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_sources() -> None:
    for directory in (PNG, PDF, TABLES, NOTES, LOGS):
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for output_name, source in SOURCE_MAP.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = TABLES / output_name
        shutil.copy2(source, destination)
        source_hash = sha256(source)
        copied_hash = sha256(destination)
        if source_hash != copied_hash:
            raise RuntimeError(f"Source-table copy mismatch: {source}")
        rows.append(
            {
                "copied_table": output_name,
                "authoritative_source": str(source),
                "source_sha256": source_hash,
                "copied_sha256": copied_hash,
                "byte_identical": True,
            }
        )
    pd.DataFrame(rows).to_csv(TABLES / "SOURCE_TABLE_MANIFEST.csv", index=False)


def save_figure(fig: plt.Figure, stem: str, dpi: int = 400, pad: float = 0.08) -> None:
    fig.savefig(PNG / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white")
    fig.savefig(PDF / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)


def correlation_matrix(table: pd.DataFrame, land: str) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=VARIABLES, columns=VARIABLES)
    for row in table[table.land_type.eq(land)].itertuples(index=False):
        matrix.loc[row.variable_1, row.variable_2] = row.pearson_r
        matrix.loc[row.variable_2, row.variable_1] = row.pearson_r
    return matrix


def render_correlation_figure(table_name: str, stem: str, sensitivity: bool) -> None:
    reset_style()
    table = pd.read_csv(TABLES / table_name)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 10.8))
    mask = np.triu(np.ones((len(VARIABLES), len(VARIABLES)), dtype=bool), k=1)
    for ax, land in zip(axes.ravel(), LANDS):
        matrix = correlation_matrix(table, land)
        heat = sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="magma_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.65,
            linecolor="white",
            cbar_kws={"shrink": 0.76, "pad": 0.025, "label": "Pearson r"},
            annot_kws={"size": 8.6, "weight": "semibold"},
            ax=ax,
        )
        ax.set_title(land.upper(), fontsize=11.5, fontweight="bold", pad=9)
        ax.set_xlabel("Variables", fontweight="semibold")
        ax.set_ylabel("Variables", fontweight="semibold")
        ax.tick_params(axis="x", rotation=45, labelsize=8.5, pad=1)
        ax.tick_params(axis="y", rotation=0, labelsize=8.5, pad=1)
        heat.collections[0].colorbar.ax.tick_params(labelsize=8)
        heat.collections[0].colorbar.ax.yaxis.label.set_size(8.5)
    if sensitivity:
        title = "Supplementary sensitivity: inter-index correlations after removing calendar-month climatology"
        subtitle = "Pairwise YYYY-MM alignment; GPP coverage ends in 2021"
    else:
        title = "Inter-index correlations of raw monthly values"
        subtitle = "Pairwise YYYY-MM alignment (N = 194–288 months); GPP coverage: 2001–2021"
    fig.suptitle(title, fontsize=13.2, fontweight="bold", y=0.995)
    fig.text(0.5, 0.969, subtitle, ha="center", va="top", fontsize=9.2, color="#444444")
    fig.subplots_adjust(top=0.92, bottom=0.065, left=0.065, right=0.97, hspace=0.28, wspace=0.20)
    save_figure(fig, stem, dpi=450, pad=0.06)


def render_figure7() -> None:
    reset_style()
    data = pd.read_csv(TABLES / "Figure_7_source.csv")
    targets = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
    target_display = {"GNDVI": "GNDVI", "GOSIF": "GOSIF/SIF", "GPP": "GPP", "LST": "LST", "NDVI": "NDVI", "NIRv": "NIRv", "SPEI": "SPEI-1", "TWS": "TWS", "VPD": "VPDmax"}
    land_internal = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
    land_display = dict(zip(land_internal, LANDS))
    mpl.rcParams.update({"font.size": 8.3, "axes.titlesize": 10.8, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(4, 2, subplot_kw={"projection": "polar"}, figsize=(12.0, 19.5))
    fig.subplots_adjust(top=0.985, bottom=0.065, left=0.035, right=0.965, hspace=0.25, wspace=0.10)
    angles = np.linspace(0, 2 * np.pi, len(targets), endpoint=False)
    spacing = 2 * np.pi / len(targets)
    offsets = dict(zip(MODELS, np.linspace(-0.255 * spacing, 0.255 * spacing, len(MODELS))))
    angle_by_target = dict(zip(targets, angles))
    target_rank = {target: i for i, target in enumerate(targets)}
    nrmse_rmax = max(np.ceil(float((data.mean_normalized_rmse + data.sd_normalized_rmse.fillna(0)).max()) * 4) / 4, 1.5)
    for row_index, land in enumerate(land_internal):
        sub = data[data.land_type_internal.eq(land)]
        for column_index, metric in enumerate(["R2", "nRMSE"]):
            ax = axes[row_index, column_index]
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.set_xticks(angles)
            ax.set_xticklabels([target_display[target] for target in targets], fontsize=8.2, fontweight="bold")
            ax.tick_params(axis="x", pad=10)
            ax.spines["polar"].set_linewidth(0.9)
            if metric == "R2":
                ax.set_ylim(0, 1)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
                ax.set_title(f"{land_display[land]} - R$^2$", pad=16, fontweight="bold")
                value_col, error_col = "mean_r2", "sd_r2"
            else:
                ax.set_ylim(0, nrmse_rmax)
                ax.set_yticks(np.arange(0, nrmse_rmax + 0.001, 0.5))
                ax.set_title(f"{land_display[land]} - nRMSE", pad=16, fontweight="bold")
                value_col, error_col = "mean_normalized_rmse", "sd_normalized_rmse"
            ax.set_rlabel_position(235)
            ax.tick_params(axis="y", labelsize=7.2)
            ax.grid(True, alpha=0.28, linewidth=0.6)
            for model in MODELS:
                m = sub[sub.model.eq(model)].sort_values("target_internal", key=lambda series: series.map(target_rank))
                theta = np.array([angle_by_target[target] + offsets[model] for target in m.target_internal])
                values = m[value_col].to_numpy(float)
                errors = m[error_col].to_numpy(float)
                if metric == "R2":
                    positive = values >= 0
                    if positive.any():
                        ax.errorbar(theta[positive], values[positive], yerr=errors[positive], fmt="o", ms=4.0, capsize=2.1, color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model], elinewidth=0.8, zorder=3)
                    if (~positive).any():
                        ax.scatter(theta[~positive], np.full((~positive).sum(), 0.022), marker="x", s=25, color=MODEL_COLORS[model], linewidths=1.1, zorder=4)
                else:
                    ax.errorbar(theta, values, yerr=errors, fmt="o", ms=4.0, capsize=2.1, color=MODEL_COLORS[model], ecolor=MODEL_COLORS[model], elinewidth=0.8, zorder=3)
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=MODEL_COLORS[model], markeredgecolor=MODEL_COLORS[model], markersize=6.5, label=model) for model in MODELS]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=True, bbox_to_anchor=(0.5, 0.022), title="Learned models", columnspacing=1.6, handletextpad=0.45)
    save_figure(fig, "Figure_7_refined_TRUE_BiLSTM", dpi=400, pad=0.06)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left", clip_on=False)


def render_s4() -> None:
    reset_style()
    validation = pd.read_csv(TABLES / "Figure_S4_validation_correlations.csv")
    components = pd.read_csv(TABLES / "Figure_S4_component_correlations.csv")
    loco = pd.read_csv(TABLES / "Figure_S4_leave_one_out.csv")
    periods = pd.read_csv(TABLES / "Figure_S4_period_summary.csv")

    component_rows = []
    for component in ["SPEI", "TWS", "VPD", "LST"]:
        direct = components[components.predictor.eq(component)].pearson_r
        omitted = loco[loco.omitted_component.eq(component)].pearson_r
        component_rows.extend(
            [
                {"component": component, "series": "Individual component", "mean": direct.mean(), "std": direct.std(ddof=1), "n_tasks": len(direct)},
                {"component": component, "series": "EWBI after omission", "mean": omitted.mean(), "std": omitted.std(ddof=1), "n_tasks": len(omitted)},
            ]
        )
    panel_a = pd.DataFrame(component_rows)
    panel_a.to_csv(TABLES / "Figure_S4_panel_a_plot_source.csv", index=False)

    weighting_rows = []
    for land in LANDS:
        for weighting in ["Equal", "PCA", "Regression"]:
            values = validation[(validation.land_type.eq(land)) & (validation["index"].eq(weighting))].pearson_r
            weighting_rows.append({"land_type": land, "weighting": weighting, "mean": values.mean(), "std": values.std(ddof=1), "n_responses": len(values)})
    panel_b = pd.DataFrame(weighting_rows)
    panel_b.to_csv(TABLES / "Figure_S4_panel_b_plot_source.csv", index=False)

    fig = plt.figure(figsize=(12.7, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 1.18])
    ax_a = fig.add_subplot(gs[0, 0])
    comps = ["SPEI", "TWS", "VPD", "LST"]
    x = np.arange(len(comps))
    styles = [("Individual component", "Component alone", "#2166AC", "o", -0.09), ("EWBI after omission", "EWBI without component", "#B2182B", "s", 0.09)]
    for series, label, color, marker, offset in styles:
        d = panel_a[panel_a.series.eq(series)].set_index("component").reindex(comps)
        ax_a.errorbar(x + offset, d["mean"], yerr=d["std"], fmt=marker, ms=5.8, lw=1.2, capsize=3, color=color, label=label)
    ax_a.axhline(0, color="#4A4A4A", lw=0.8)
    ax_a.set_xticks(x, comps)
    ax_a.set_ylabel("Mean Pearson r")
    ax_a.set_title("Component sensitivity")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="lower right", fontsize=7.6, handletextpad=0.4)
    panel_label(ax_a, "(a)")

    ax_b = fig.add_subplot(gs[0, 1])
    weight_order = ["Equal", "PCA", "Regression"]
    for land in LANDS:
        d = panel_b[panel_b.land_type.eq(land)].set_index("weighting").reindex(weight_order)
        ax_b.errorbar(np.arange(3), d["mean"], yerr=d["std"], marker="o", ms=5.0, capsize=2.8, lw=1.2, color=LAND_COLORS[land], label=land)
    ax_b.axhline(0, color="#4A4A4A", lw=0.8)
    ax_b.set_xticks(np.arange(3), weight_order)
    ax_b.set_ylabel("Mean Pearson r")
    ax_b.set_title("Weighting robustness")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, ncol=2, fontsize=7.4, loc="lower center")
    panel_label(ax_b, "(b)")

    ax_c = fig.add_subplot(gs[0, 2])
    period_order = ["Baseline 2001–2011", "Historic drought 2012–2016", "Recent drought 2020–2024"]
    labels = ["Baseline\n2001–2011", "Historic drought\n2012–2016", "Recent drought\n2020–2024"]
    offsets = np.linspace(-0.21, 0.21, len(LANDS))
    for offset, land in zip(offsets, LANDS):
        d = periods[periods.land_type.eq(land)].set_index("period").reindex(period_order)
        y = d.mean_ewbi.to_numpy()
        err = np.vstack([y - d.block_bootstrap_ci95_low.to_numpy(), d.block_bootstrap_ci95_high.to_numpy() - y])
        ax_c.errorbar(np.arange(3) + offset, y, yerr=err, fmt="o", ms=5.0, capsize=2.8, lw=1.15, color=LAND_COLORS[land])
    ax_c.axhline(0, color="#4A4A4A", lw=0.8)
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.set_ylabel("Mean EWBI forcing score")
    ax_c.set_title("Drought-period separation")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)")
    save_figure(fig, "Figure_S4_refined", dpi=500, pad=0.06)


def heatmap_annotations(values: pd.DataFrame, significance: pd.DataFrame) -> np.ndarray:
    out = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            out[i, j] = f"{value:.2f}{'*' if bool(significance.iloc[i, j]) else ''}"
    return out


def add_heatmap(ax: plt.Axes, values: pd.DataFrame, annotations: np.ndarray, title: str, cmap: str, vmin: float, vmax: float, colorbar_label: str, rotate_x: float = 0) -> None:
    image = ax.imshow(values.to_numpy(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=7)
    ax.set_xticks(np.arange(len(values.columns)), labels=list(values.columns))
    ax.set_yticks(np.arange(len(values.index)), labels=list(values.index))
    plt.setp(ax.get_xticklabels(), rotation=rotate_x, ha="right" if rotate_x else "center", rotation_mode="anchor")
    ax.set_xticks(np.arange(-0.5, len(values.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values.index), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    norm = Normalize(vmin=vmin, vmax=vmax)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values.iloc[row, col]
            rgba = mpl.colormaps[cmap](norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(col, row, annotations[row, col], ha="center", va="center", fontsize=8.0, color="white" if luminance < 0.50 else "#202020")
    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.040, pad=0.022, aspect=25)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.set_label(colorbar_label, fontsize=9)


def render_s5() -> None:
    reset_style()
    table = pd.read_csv(TABLES / "Figure_S5_correlations.csv")
    period_map = {"Baseline 2001–2011": "baseline", "Historic drought 2012–2016": "historic", "Recent drought 2020–2024": "recent"}
    display_land = {"Natural Forests": "Natural forests"}
    table["row_label"] = table.apply(lambda row: f"{display_land.get(row.land_type, row.land_type)} | {period_map[row.period]}", axis=1)
    row_order = [f"{display_land.get(land, land)} | {period}" for land in LANDS for period in ["baseline", "historic", "recent"]]
    values = table.pivot(index="row_label", columns="response", values="pearson_r").reindex(index=row_order, columns=RESPONSES)
    sig = table.pivot(index="row_label", columns="response", values="significant_bootstrap_ci_excludes_zero").reindex(index=row_order, columns=RESPONSES)
    plot_source = values.stack().rename("pearson_r").reset_index().merge(sig.stack().rename("significant_bootstrap_ci_excludes_zero").reset_index(), on=["row_label", "response"])
    plot_source.to_csv(TABLES / "Figure_S5_plot_source.csv", index=False)
    annotations = heatmap_annotations(values, sig)
    fig, ax = plt.subplots(figsize=(8.9, 6.3), layout="constrained")
    add_heatmap(ax, values, annotations, "EWBI-response association", "RdBu_r", -1, 1, "Pearson r", rotate_x=25)
    ax.text(0.0, -0.14, "* 95% paired 12-month moving-block bootstrap CI excludes zero", transform=ax.transAxes, fontsize=8.2, color="#404040", ha="left")
    save_figure(fig, "Figure_S5_refined", dpi=500, pad=0.07)


def render_s6() -> None:
    reset_style()
    agreement = pd.read_csv(TABLES / "Figure_S6_agreement.csv")
    occupancy = pd.read_csv(TABLES / "Figure_S6_occupancy.csv")
    transitions = pd.read_csv(TABLES / "Figure_S6_transition_matrices.csv")
    alternatives = ["33_67_extremes", "kmeans", "gmm"]
    labels = ["33/67", "k-means", "GMM"]
    a = agreement[agreement.method.isin(alternatives)].pivot(index="land_type", columns="method", values="cohen_kappa").reindex(LANDS, columns=alternatives)
    a.columns = labels
    median_occ = occupancy[occupancy.method.eq("median")].set_index(["land_type", "state"]).occupancy_percent
    change_rows = []
    for land in LANDS:
        for method in alternatives:
            current = occupancy[(occupancy.land_type.eq(land)) & (occupancy.method.eq(method))].set_index("state").occupancy_percent
            change_rows.append({"land_type": land, "method": method, "maximum_change_percentage_points": float((current - median_occ.loc[land]).abs().max())})
    change_long = pd.DataFrame(change_rows)
    change_long.to_csv(TABLES / "Figure_S6_panel_b_plot_source.csv", index=False)
    b = change_long.pivot(index="land_type", columns="method", values="maximum_change_percentage_points").reindex(LANDS, columns=alternatives)
    b.columns = labels
    c = transitions[(transitions.from_state.eq("Dry / Stressed")) & (transitions.to_state.eq("Dry / Stressed"))].pivot(index="land_type", columns="method", values="probability_percent").reindex(LANDS, columns=["median", *alternatives])
    c.columns = ["Median", "33/67", "k-means", "GMM"]

    fig = plt.figure(figsize=(14.4, 4.7), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.24)
    panels = [
        (fig.add_subplot(gs[0, 0]), a, "(a) Classification agreement", "YlGnBu", 0, 1, "Cohen's kappa"),
        (fig.add_subplot(gs[0, 1]), b, "(b) Occupancy sensitivity", "YlOrRd", 0, max(30, float(np.nanmax(b.to_numpy()))), "Maximum change (percentage points)"),
        (fig.add_subplot(gs[0, 2]), c, "(c) Dry/stressed persistence", "PuBuGn", 0, 100, "Self-transition probability (%)"),
    ]
    formats = [".2f", ".1f", ".1f"]
    for (ax, data, title, cmap, vmin, vmax, cbar_label), fmt in zip(panels, formats):
        im = ax.imshow(data.to_numpy(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11.4, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(len(data.columns)), data.columns)
        ax.set_yticks(np.arange(len(data.index)), data.index)
        ax.set_xticks(np.arange(-0.5, len(data.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data.iloc[i, j]
                rgba = mpl.colormaps[cmap](norm(value))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8.6, color="white" if lum < 0.48 else "#202020")
        colorbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.022, aspect=22)
        colorbar.set_label(cbar_label, fontsize=8.5)
        colorbar.ax.tick_params(labelsize=7.8)
    panels[0][0].set_xlabel("Alternative method vs Figure-9 median")
    panels[1][0].set_xlabel("Alternative method vs median")
    panels[2][0].set_xlabel("Classification method")
    save_figure(fig, "Figure_S6_refined", dpi=500, pad=0.07)


def parse_yyyymm(value: object) -> float:
    text = str(value)
    if "-" in text:
        year, month = text.split("-")[:2]
    else:
        text = str(int(float(text)))
        year, month = text[:4], text[4:6]
    return int(year) + (int(month) - 1) / 12


def render_s7() -> None:
    reset_style()
    panel_a_data = pd.read_csv(TABLES / "Figure_S7_panel_a_source.csv")
    panel_b_data = pd.read_csv(TABLES / "Figure_S7_panel_b_source.csv")
    timeline = pd.read_csv(TABLES / "Figure_S7_panel_c_source.csv")
    model_colors = {"Persistence": "#6F6F6F", "Seasonal persistence": "#A3A3A3", "Monthly climatology": "#B8A900", "RBF-epsilon-SVR": "#4DAF4A", **MODEL_COLORS}
    fig = plt.figure(figsize=(13.0, 7.3), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    sub_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.12)
    y = np.arange(len(panel_a_data))
    colors = [model_colors[model] for model in panel_a_data.model]
    ax_a1 = fig.add_subplot(sub_a[0, 0])
    ax_a1.errorbar(panel_a_data.mean_r2, y, xerr=panel_a_data.sd_r2, fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1.0)
    ax_a1.scatter(panel_a_data.mean_r2, y, c=colors, s=38, zorder=3)
    ax_a1.axvline(0, color="#8B1A1A", lw=0.85)
    ax_a1.set_yticks(y, panel_a_data.model)
    ax_a1.invert_yaxis()
    ax_a1.set_xlabel(r"Mean $R^2$")
    ax_a1.set_title(r"Monthly-aggregate $R^2$")
    ax_a1.grid(axis="x")
    panel_label(ax_a1, "(a)", x=-0.28)
    ax_a2 = fig.add_subplot(sub_a[0, 1], sharey=ax_a1)
    ax_a2.errorbar(panel_a_data.mean_normalized_rmse, y, xerr=panel_a_data.sd_normalized_rmse, fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1.0)
    ax_a2.scatter(panel_a_data.mean_normalized_rmse, y, c=colors, s=38, zorder=3)
    ax_a2.set_xlabel("Mean nRMSE")
    ax_a2.set_title("Monthly-aggregate nRMSE")
    ax_a2.tick_params(labelleft=False)
    ax_a2.grid(axis="x")

    sub_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.28)
    x = np.arange(len(panel_b_data))
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b1.errorbar(x, panel_b_data.mean_r2, yerr=panel_b_data.sd_r2, fmt="o-", ms=5.2, capsize=2.8, lw=1.2, color="#2F6DAE")
    ax_b1.axhline(0, color="#4A4A4A", lw=0.8)
    ax_b1.set_xticks(x, panel_b_data.test_period)
    ax_b1.tick_params(axis="x", labelsize=7.5, pad=3)
    ax_b1.set_ylabel(r"Mean pixel-level $R^2$")
    ax_b1.set_title(r"Pixel-level $R^2$ by fold")
    ax_b1.grid(axis="y")
    panel_label(ax_b1, "(b)", x=0.03, y=0.98)
    ax_b2 = fig.add_subplot(sub_b[0, 1])
    ax_b2.errorbar(x, panel_b_data.mean_nrmse, yerr=panel_b_data.sd_nrmse, fmt="s-", ms=5.2, capsize=2.8, lw=1.2, color="#D84A4A")
    ax_b2.set_xticks(x, panel_b_data.test_period)
    ax_b2.tick_params(axis="x", labelsize=7.5, pad=3)
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
        validation_start = parse_yyyymm(row.internal_validation_start)
        validation_end = parse_yyyymm(row.internal_validation_end) + 1 / 12
        test_start = parse_yyyymm(row.test_start)
        test_end = parse_yyyymm(row.test_end) + 1 / 12
        ax_c.add_patch(Rectangle((2001.0, fold - 0.20), core_end - 2001.0, 0.40, facecolor="#8A8A8A", edgecolor="none"))
        ax_c.add_patch(Rectangle((validation_start, fold - 0.20), validation_end - validation_start, 0.40, facecolor="#E9B949", edgecolor="none"))
        ax_c.add_patch(Rectangle((test_start, fold - 0.20), test_end - test_start, 0.40, facecolor="#D95D39", edgecolor="none"))
    ax_c.set_xticks([2001, 2005, 2009, 2013, 2017, 2021, 2025])
    ax_c.set_xlabel("Calendar year")
    ax_c.set_title("Rolling-origin design")
    ax_c.grid(axis="x")
    ax_c.legend(handles=[Patch(facecolor="#8A8A8A", label="Training"), Patch(facecolor="#E9B949", label="Validation"), Patch(facecolor="#D95D39", label="Test")], frameon=False, ncol=3, loc="lower center")
    panel_label(ax_c, "(c)", x=-0.03)
    save_figure(fig, "Figure_S7_refined_TRUE_BiLSTM", dpi=500, pad=0.06)


def add_s9_heatmap(ax: plt.Axes, data: pd.DataFrame, value_col: str, title: str, cmap: str, best: str):
    matrix = data.pivot(index="model", columns="land_cover", values=value_col).loc[MODELS, LANDS]
    image = ax.imshow(matrix.to_numpy(float), cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(LANDS)), LANDS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(MODELS)), MODELS)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=8)
    ax.tick_params(length=0)
    for y in np.arange(0.5, len(MODELS), 1):
        ax.axhline(y, color="white", lw=0.8)
    for x in np.arange(0.5, len(LANDS), 1):
        ax.axvline(x, color="white", lw=0.8)
    for sep in [2.5, 4.5]:
        ax.axhline(sep, color="black", lw=1.0)
    values = matrix.to_numpy(float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            rgba = image.cmap(image.norm(values[i, j]))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=8.2, fontweight="bold", color="white" if lum < 0.48 else "black")
    for j in range(values.shape[1]):
        target = np.nanmin(values[:, j]) if best == "min" else np.nanmax(values[:, j])
        for i in np.where(np.isclose(values[:, j], target, atol=1e-12, rtol=0))[0]:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6, clip_on=False))
    return image


def render_s9() -> None:
    reset_style()
    metrics = pd.read_csv(TABLES / "Figure_S9_metrics.csv")
    monthly = pd.read_csv(TABLES / "Figure_S9_exact_monthly_source.csv")
    ecological = pd.read_csv(MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv")
    support = monthly.groupby(["land_cover", "date"], as_index=False).agg(forecast_support_observed_ewbi=("observed_ewbi", "mean"), duplicate_observed_min=("observed_ewbi", "min"), duplicate_observed_max=("observed_ewbi", "max"), contributing_rows=("observed_ewbi", "size"), contributing_models=("model", "nunique"), contributing_seeds=("seed", "nunique"))
    support["date_month"] = pd.to_datetime(support.date.astype(str), format="%Y%m")
    ecological = ecological.rename(columns={"land_type": "land_cover", "ewbi": "ecological_observed_ewbi"})
    ecological["date_month"] = pd.to_datetime(ecological.date)
    joined = support.merge(ecological[["land_cover", "date_month", "ecological_observed_ewbi"]], on=["land_cover", "date_month"], how="inner")
    joined.to_csv(TABLES / "Figure_S9_support_joined_plot_source.csv", index=False)

    fig = plt.figure(figsize=(16, 11.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.34, wspace=0.24)
    ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(gs[i, j]) for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]]
    nrmse_image = add_s9_heatmap(ax_a, metrics, "nrmse", "(a) EWBI nRMSE", "YlGnBu_r", "min")
    r2_image = add_s9_heatmap(ax_b, metrics, "r2", r"(b) EWBI $R^2$", "RdYlBu", "max")
    fig.colorbar(nrmse_image, ax=ax_a, fraction=0.046, pad=0.035).set_label("nRMSE")
    fig.colorbar(r2_image, ax=ax_b, fraction=0.046, pad=0.035).set_label(r"$R^2$")

    plot_models = ["BiLSTM", "XGBoost"]
    colors = {"BiLSTM": "#7b4ab8", "XGBoost": "#17aebf"}
    markers = {"Grasslands": "o", "Natural Forests": "s", "Savannas": "^", "Shrublands": "D"}
    panel_c = monthly[monthly.model.isin(plot_models)]
    for model in plot_models:
        for land in LANDS:
            group = panel_c[(panel_c.model.eq(model)) & (panel_c.land_cover.eq(land))]
            ax_c.scatter(group.observed_ewbi, group.predicted_ewbi, s=22, alpha=0.46, color=colors[model], marker=markers[land], linewidths=0)
    finite = pd.concat([panel_c.observed_ewbi, panel_c.predicted_ewbi]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = float(finite.quantile(0.005)), float(finite.quantile(0.995))
    pad = (hi - lo) * 0.08
    ax_c.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.1, ls="--")
    ax_c.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel="Forecast-support observed EWBI", ylabel="Forecast-derived EWBI")
    ax_c.set_title("(c) Observed vs forecast-derived EWBI", loc="left", fontweight="bold", pad=8)
    ax_c.grid(True)
    model_handles = [Line2D([0], [0], marker="o", linestyle="", color=colors[m], label=m, markersize=6) for m in plot_models]
    land_handles = [Line2D([0], [0], marker=markers[l], linestyle="", color="0.35", label=l, markersize=5.5) for l in LANDS]
    first = ax_c.legend(handles=model_handles, title="Displayed model", loc="upper left", frameon=True, fontsize=7.8, title_fontsize=8.2)
    ax_c.add_artist(first)
    ax_c.legend(handles=land_handles, title="Land cover", loc="lower right", frameon=True, fontsize=7.6, title_fontsize=8.0)

    land_colors = {"Grasslands": "#1b9e77", "Natural Forests": "#7570b3", "Savannas": "#d95f02", "Shrublands": "#666666"}
    for land in LANDS:
        group = joined[joined.land_cover.eq(land)]
        ax_d.scatter(group.ecological_observed_ewbi, group.forecast_support_observed_ewbi, s=24, alpha=0.65, color=land_colors[land], marker=markers[land], linewidths=0, label=land)
    finite = pd.concat([joined.ecological_observed_ewbi, joined.forecast_support_observed_ewbi]).replace([np.inf, -np.inf], np.nan).dropna()
    lo, hi = float(finite.quantile(0.005)), float(finite.quantile(0.995))
    pad = (hi - lo) * 0.08
    ax_d.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1.1, ls="--")
    ax_d.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), xlabel="Ecological observed EWBI", ylabel="Forecast-support observed EWBI")
    ax_d.set_title("(d) Spatial-support comparison", loc="left", fontweight="bold", pad=8)
    ax_d.grid(True)
    ax_d.legend(loc="upper left", frameon=True, fontsize=7.8, title="Land cover", title_fontsize=8.2)
    fig.subplots_adjust(left=0.075, right=0.965, top=0.965, bottom=0.075)
    save_figure(fig, "Figure_S9_refined_TRUE_BiLSTM", dpi=400, pad=0.06)


def render_state_null() -> None:
    reset_style()
    summary = pd.read_csv(TABLES / "Figure_S_new_state_transition_null_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True, sharey=True)
    panel_names = ["(a)", "(b)", "(c)", "(d)"]
    y = np.arange(len(STATES))[::-1]
    for ax, land, panel in zip(axes.ravel(), LANDS, panel_names):
        data = summary[summary.land_type.eq(land)].set_index("state").reindex(STATES)
        for ypos, state in zip(y, STATES):
            row = data.loc[state]
            ax.hlines(ypos, row.null_ci95_low, row.null_ci95_high, color="#A2A9B3", lw=5.5, zorder=1)
            ax.plot([row.null_ci95_low, row.null_ci95_low], [ypos - 0.09, ypos + 0.09], color="#7E8794", lw=1.0)
            ax.plot([row.null_ci95_high, row.null_ci95_high], [ypos - 0.09, ypos + 0.09], color="#7E8794", lw=1.0)
            ax.scatter(row.null_median, ypos, s=35, facecolor="white", edgecolor="#34495E", linewidth=1.2, zorder=3)
            ax.scatter(row.observed_diagonal_probability, ypos, s=46, marker="D", color="#C8322B", edgecolor="white", linewidth=0.45, zorder=4)
        ax.set_yticks(y, STATES)
        ax.set_xlim(0.30, 0.80)
        ax.set_xticks(np.arange(0.3, 0.81, 0.1))
        ax.set_title(f"{panel} {land}", loc="left", fontsize=11, fontweight="bold")
        ax.grid(axis="x")
        ax.set_axisbelow(True)
        ax.set_xlabel("Diagonal transition probability")
    axes[0, 0].set_ylabel("State")
    axes[1, 0].set_ylabel("State")
    handles = [
        Line2D([0], [0], color="#A2A9B3", lw=5.5, label="Null 95% interval"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#34495E", markersize=6, label="Null median"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="#C8322B", markeredgecolor="white", markersize=6.5, label="Observed persistence"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Observed state persistence relative to an autocorrelation-preserving null", fontsize=13, fontweight="bold", y=0.995)
    fig.text(0.5, 0.945, "IAAFT surrogates; 500 independent EWBI–NIRv pairs per land type", ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.17, right=0.98, hspace=0.30, wspace=0.18)
    save_figure(fig, "Figure_S_new_state_transition_null", dpi=450, pad=0.07)


def write_notes() -> None:
    notes = {
        "Figure_5_README.md": """# Figure 5 refined candidate\n\n- Placement: main paper.\n- Classification: C — rebuilt using the submitted Figure 5 as the visual template.\n- Data: unchanged from the completed computational phase; raw monthly Pearson correlations with pairwise YYYY-MM alignment.\n- Visualization: lower-triangle four-panel heatmap retained; typography, whitespace, title, and coverage statement refined.\n- GPP scope: 2001–2021 is intentional and matches the submitted Figure-5 computation. The recovered original code skipped test-period GPP and its removed title stated this range. It is not an accidental import from the later GPP sensitivity.\n- Exact sources: `Figure_5_correlations.csv`, `Figure_5_pairwise_date_coverage.csv`, and `Figure_5_exact_monthly_input.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_correlation_figure`).\n""",
        "Figure_7_README.md": """# Figure 7 refined candidate\n\n- Placement: main paper.\n- Classification: B — visual refinement only.\n- Data: unchanged from the corrected TRUE-BiLSTM computational table.\n- Visualization: original eight-panel polar layout, target order, colors, uncertainty bars, and legend retained; output size and typography normalized for submission.\n- Exact source: `Figure_7_source.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_figure7`).\n""",
        "Figure_S4_README.md": """# Figure S4 refined candidate\n\n- Placement: supplement.\n- Classification: C — rebuilt using submitted S4 panel structure.\n- Data: unchanged from the harmonized authoritative no-QA/Figure-9 branch.\n- Visualization: restored component-sensitivity, weighting-robustness, and drought-period separation panels instead of the interim heatmap redesign.\n- Exact sources: all `Figure_S4_*` CSV files, including generated panel-A/panel-B plot summaries.\n- Script: `scripts/render_refined_figures.py` (`render_s4`).\n""",
        "Figure_S5_README.md": """# Figure S5 refined candidate\n\n- Placement: supplement.\n- Classification: C — rebuilt using submitted S5 single-heatmap structure.\n- Data: unchanged from the harmonized previous/no-QA GNDVI analysis.\n- Visualization: restored 12-row land-type/period heatmap. Asterisks use one rule only: the paired moving-block bootstrap 95% CI excludes zero.\n- Exact sources: `Figure_S5_correlations.csv`, `Figure_S5_exact_monthly_input.csv`, and `Figure_S5_plot_source.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_s5`).\n""",
        "Figure_S6_README.md": """# Figure S6 refined candidate\n\n- Placement: supplement.\n- Classification: C — corrected values rendered in the submitted three-panel heatmap template.\n- Data: unchanged from the exact Figure-9 EWBI/NIRv harmonization.\n- Visualization: restored submitted proportions, panel-title style, independent colorbars, and compact labeling. The reference is the Figure-9 median classification.\n- Exact sources: `Figure_S6_agreement.csv`, `Figure_S6_occupancy.csv`, `Figure_S6_transition_matrices.csv`, `Figure_S6_exact_monthly_states.csv`, and `Figure_S6_panel_b_plot_source.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_s6`).\n""",
        "Figure_S7_README.md": """# Figure S7 refined candidate\n\n- Placement: supplement.\n- Classification: C — rebuilt with the original submitted S7 renderer as the visual template.\n- Data: unchanged from the corrected TRUE-BiLSTM S7 tables.\n- Visualization: original panel geometry, proportions, labels, fold plots, timeline, legend placement, colors, and typography restored. The interim oversized text was caused by inherited Figure-7 plotting settings and is removed.\n- Exact sources: `Figure_S7_panel_a_source.csv`, `Figure_S7_panel_b_source.csv`, `Figure_S7_panel_c_source.csv`, plus the exact monthly/pixel metric tables.\n- Script: `scripts/render_refined_figures.py` (`render_s7`).\n""",
        "Figure_S9_README.md": """# Figure S9 refined candidate\n\n- Placement: supplement.\n- Classification: C — returned to the submitted four-panel visual balance.\n- Data: unchanged from the corrected TRUE-BiLSTM S9 branch.\n- Visualization: retained heatmaps, observed-versus-derived scatter, and spatial-support comparison; removed the interim oversized title/footer and preserved the submitted panel hierarchy.\n- Exact sources: `Figure_S9_metrics.csv`, `Figure_S9_exact_monthly_source.csv`, `Figure_S9_fold_metrics.csv`, `Figure_S9_support_summary.csv`, and `Figure_S9_support_joined_plot_source.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_s9`).\n""",
        "Figure_S_new_Figure5_deseasonalized_sensitivity_README.md": """# New Figure-5 deseasonalized sensitivity\n\n- Placement: supplement only.\n- Classification: D — new supplementary design, intentionally matched to Figure 5 for direct comparison.\n- Data: unchanged from the completed calendar-month-climatology removal sensitivity.\n- Visualization: title explicitly identifies it as supplementary sensitivity so it cannot be mistaken for the raw main analysis.\n- Exact sources: `Figure_S_new_Figure5_deseasonalized_correlations.csv` and `Figure_S_new_Figure5_deseasonalized_exact_monthly_input.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_correlation_figure`).\n""",
        "Figure_S_new_state_transition_null_README.md": """# New state-transition null figure\n\n- Placement: supplement only; Figure 9 remains untouched.\n- Classification: D — new supplementary figure design.\n- Data: unchanged from the completed 500-pair-per-land IAAFT null analysis.\n- Visualization: compact forest-plot panels distinguish observed persistence, null median, and null 95% interval without implying validated Markov dynamics.\n- Exact sources: `Figure_S_new_state_transition_null_summary.csv`, `Figure_S_new_state_transition_null_distributions.csv`, and `Figure_S_new_state_transition_null_exact_input.csv`.\n- Script: `scripts/render_refined_figures.py` (`render_state_null`).\n""",
    }
    for name, text in notes.items():
        (NOTES / name).write_text(text, encoding="utf-8")


def write_reports() -> None:
    summary = """# Figure refinement summary

This phase changed figure presentation only. All plotted scientific values come from the completed computational revision tables.

| Figure | Intended location | Current issue | Classification / action | Data changed in this phase? | Visualization changed? | Original design preserved? | Source tables | Source script | Confidence |
|---|---|---|---|---:|---:|---:|---|---|---|
| Figure 5 | Main paper | Correct values but excessive title whitespace; coverage needed clarification | C — rebuilt in submitted four-panel lower-triangle template | No | Yes | Yes | `Figure_5_*` | `render_refined_figures.py` | High |
| Figure 7 | Main paper | Correct polar figure needed consistent submission packaging | B — visual refinement only | No | Yes, minor | Yes | `Figure_7_source.csv` | `render_refined_figures.py` | High |
| Figure S4 | Supplement | Interim heatmap redesign departed substantially from submitted S4 | C — restored three original analytical panels | No | Yes | Yes | `Figure_S4_*` | `render_refined_figures.py` | High |
| Figure S5 | Supplement | Interim three-panel heatmap departed from submitted single-matrix layout | C — restored submitted row-by-period matrix | No | Yes | Yes | `Figure_S5_*` | `render_refined_figures.py` | High |
| Figure S6 | Supplement | Correct structure but rough draft typography/spacing | C — corrected values in submitted three-panel heatmap template | No | Yes | Yes | `Figure_S6_*` | `render_refined_figures.py` | High |
| Figure S7 | Supplement | Corrected version inherited oversized Figure-7 fonts and lost original balance | C — rebuilt directly from original S7 visual template | No | Yes | Yes, closely | `Figure_S7_*` | `render_refined_figures.py` | High |
| Figure S9 | Supplement | Interim title/footer enlarged canvas and weakened submitted balance | C — restored submitted four-panel hierarchy | No | Yes | Yes | `Figure_S9_*` | `render_refined_figures.py` | High |
| Figure S_new Figure-5 deseasonalized sensitivity | Supplement | New figure required clear separation from main raw analysis | D — new supplementary figure matched to Figure 5 | No | Yes | Not applicable; new figure | `Figure_S_new_Figure5_*` | `render_refined_figures.py` | High |
| Figure S_new state-transition null | Supplement | Existing draft was visually immature | D — new dot-and-whisker/forest-plot design | No | Yes | Not applicable; new figure | `Figure_S_new_state_transition_null_*` | `render_refined_figures.py` | High |

## Figure 5 GPP scope decision

The 2001–2021 GPP scope in the corrected main Figure 5 was investigated and retained. It is not an accidental import from the later product-transition sensitivity: the recovered submitted Figure-5 code explicitly skipped test-period GPP, and its removed title stated “GPP: 2001–2021 only.” The refined figure now states that coverage visibly. The stitched FLUXCOM/MOD17 sensitivity remains separate and is not silently substituted into the main correlation matrix.
"""
    (ROOT / "FIGURE_REFINEMENT_SUMMARY.md").write_text(summary, encoding="utf-8")

    placement = """# Figure placement plan

## Main paper

- Figure 5: `figures_png/Figure_5_refined_raw_date_aligned.png`
- Figure 7: `figures_png/Figure_7_refined_TRUE_BiLSTM.png`

## Supplementary material

- Figure S4: `figures_png/Figure_S4_refined.png`
- Figure S5: `figures_png/Figure_S5_refined.png`
- Figure S6: `figures_png/Figure_S6_refined.png`
- Figure S7: `figures_png/Figure_S7_refined_TRUE_BiLSTM.png`
- Figure S9: `figures_png/Figure_S9_refined_TRUE_BiLSTM.png`
- Temporary new supplementary figure — deseasonalized Figure-5 sensitivity: `figures_png/Figure_S_new_Figure5_deseasonalized_sensitivity.png`
- Temporary new supplementary figure — state-transition null: `figures_png/Figure_S_new_state_transition_null.png`

Final supplementary numbering should be assigned only when the manuscript and supplementary file are edited. Figure 9 is not replaced or modified.
"""
    (ROOT / "FIGURE_PLACEMENT_PLAN.md").write_text(placement, encoding="utf-8")

    differences = """# Visual difference notes

## Intentionally follows the original design

- Figure 5 retains the submitted 2 × 2 lower-triangle correlation matrices, magma-reversed palette, annotations, land-cover ordering, and separate colorbars. A compact title and explicit pairwise/GPP coverage line were added because the submitted title was manually cropped away.
- Figure 7 retains the submitted eight-panel polar layout, spoke order, model colors, point/error-bar encoding, land-cover ordering, and bottom model legend.
- Figure S4 restores the submitted component-sensitivity, weighting-robustness, and drought-period-separation panels. The interim heatmap redesign is not retained.
- Figure S5 restores the submitted single heatmap with land type × period rows. The interim three-panel layout is not retained.
- Figure S6 retains the submitted three side-by-side heatmaps and independent colorbars, but uses the corrected Figure-9-consistent values.
- Figure S7 follows the submitted layout especially closely. Its panel grid, proportions, labels, fold plots, rolling-origin timeline, colors, and legend placement are restored. The previous replacement looked different because Figure-7 font settings leaked into the S7 renderer.
- Figure S9 restores the submitted four-panel balance and removes the interim oversized page title and explanatory footer; the explanation belongs in the caption/notes.

## Intentionally new designs

- The deseasonalized Figure-5 sensitivity is new. It intentionally mirrors Figure 5 for comparison, but its title explicitly marks it as supplementary sensitivity.
- The state-transition null figure is new. A horizontal dot-and-whisker design replaces the rough vertical interval bars so observed persistence, null median, and null 95% interval can be compared cleanly.

## Data-change statement

No numerical value or analysis branch was changed during figure refinement. Changes relative to the submitted figures arose in the preceding computational phase and are merely visualized here.
"""
    (ROOT / "VISUAL_DIFFERENCE_NOTES.md").write_text(differences, encoding="utf-8")


def write_output_manifest() -> None:
    rows = []
    for path in sorted([*PNG.glob("*.png"), *PDF.glob("*.pdf"), *TABLES.glob("*.csv"), *NOTES.glob("*.md"), ROOT / "FIGURE_REFINEMENT_SUMMARY.md", ROOT / "FIGURE_PLACEMENT_PLAN.md", ROOT / "VISUAL_DIFFERENCE_NOTES.md"]):
        if path.is_file():
            rows.append({"relative_path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(rows).to_csv(ROOT / "FIGURE_REFINEMENT_OUTPUT_MANIFEST.csv", index=False)


def main() -> None:
    copy_sources()
    render_correlation_figure("Figure_5_correlations.csv", "Figure_5_refined_raw_date_aligned", sensitivity=False)
    render_figure7()
    render_s4()
    render_s5()
    render_s6()
    render_s7()
    render_s9()
    render_correlation_figure("Figure_S_new_Figure5_deseasonalized_correlations.csv", "Figure_S_new_Figure5_deseasonalized_sensitivity", sensitivity=True)
    render_state_null()
    write_notes()
    write_reports()
    write_output_manifest()
    print(f"Rendered {len(list(PNG.glob('*.png')))} PNG and {len(list(PDF.glob('*.pdf')))} PDF figures.")


if __name__ == "__main__":
    main()
