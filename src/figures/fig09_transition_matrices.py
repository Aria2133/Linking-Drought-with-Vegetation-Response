"""Render label-only Figure 9 and dependent supplementary updates.

No classification, transition, surrogate, or sensitivity calculation is run here.
The script reads finalized tables, replaces four categorical display labels, and
plots the already-finalized values.
"""

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
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PAPER = ROOT.parents[1]
RESTORED = PAPER / "AUDIT_TEMP" / "RESTORED_FINAL_FIGURE_CODES" / "Figure_9"
REFINED = PAPER / "AUDIT_TEMP" / "FIGURE_REFINEMENT_PHASE"
PNG = ROOT / "figures_png"
PDF = ROOT / "figures_pdf"
TABLES = ROOT / "source_tables"
REPORTS = ROOT / "reports"

LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
OLD_STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]
LABEL_MAP = {
    "Wet / Healthy": "Wet / Higher NIRv",
    "Wet / Suppressed": "Wet / Lower NIRv",
    "Dry / Resilient": "Dry / Higher NIRv",
    "Dry / Stressed": "Dry / Lower NIRv",
}
NEW_STATES = [LABEL_MAP[value] for value in OLD_STATES]

FIG9_SOURCE = (
    RESTORED
    / "03_reproduced_outputs"
    / "tables"
    / "figure_9_transition_probabilities.csv"
)
SOURCE_MAP = {
    "Figure_9_transition_probabilities_neutral_labels.csv": FIG9_SOURCE,
    "Figure_S_state_transition_null_summary_neutral_labels.csv": REFINED / "source_tables" / "Figure_S_new_state_transition_null_summary.csv",
    "Figure_S_state_transition_null_distributions_neutral_labels.csv": REFINED / "source_tables" / "Figure_S_new_state_transition_null_distributions.csv",
    "Figure_S_state_transition_null_exact_input_neutral_labels.csv": REFINED / "source_tables" / "Figure_S_new_state_transition_null_exact_input.csv",
    "Figure_S6_occupancy_neutral_labels.csv": REFINED / "source_tables" / "Figure_S6_occupancy.csv",
    "Figure_S6_transition_matrices_neutral_labels.csv": REFINED / "source_tables" / "Figure_S6_transition_matrices.csv",
    "Figure_S6_exact_monthly_states_neutral_labels.csv": REFINED / "source_tables" / "Figure_S6_exact_monthly_states.csv",
}
UNCHANGED_S6_SOURCES = {
    "Figure_S6_agreement_unchanged.csv": REFINED / "source_tables" / "Figure_S6_agreement.csv",
    "Figure_S6_panel_b_plot_source_unchanged.csv": REFINED / "source_tables" / "Figure_S6_panel_b_plot_source.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reset_style(*, original_figure9: bool = False) -> None:
    mpl.rcdefaults()
    settings = {
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    if original_figure9:
        settings.update(
            {
                "font.size": 10,
                "axes.titlesize": 12,
                "axes.labelsize": 11,
                "axes.linewidth": 0.8,
                "xtick.labelsize": 9,
                "ytick.labelsize": 9,
                "legend.fontsize": 9,
                "figure.titlesize": 15,
            }
        )
    else:
        settings.update(
            {
                "font.size": 9.5,
                "axes.titlesize": 11.5,
                "axes.titleweight": "semibold",
                "axes.labelsize": 10.0,
                "xtick.labelsize": 9.0,
                "ytick.labelsize": 9.0,
                "legend.fontsize": 8.5,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.linewidth": 0.8,
                "grid.color": "#D9D9D9",
                "grid.linewidth": 0.55,
                "grid.alpha": 0.65,
            }
        )
    mpl.rcParams.update(settings)


def replace_labels_only(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy(deep=True)
    object_columns = output.select_dtypes(include=["object", "string"]).columns
    for column in object_columns:
        output[column] = output[column].replace(LABEL_MAP)
    return output


def verify_label_only(source: pd.DataFrame, output: pd.DataFrame, source_name: str, output_name: str) -> dict[str, object]:
    assert list(source.columns) == list(output.columns)
    assert len(source) == len(output)
    numeric_columns = list(source.select_dtypes(include=[np.number]).columns)
    numeric_identical = source[numeric_columns].equals(output[numeric_columns])
    expected = replace_labels_only(source)
    all_cells_as_expected = expected.equals(output)
    if not numeric_identical or not all_cells_as_expected:
        raise RuntimeError(f"Label-only verification failed for {source_name}")
    changed_cells = int((source.astype(str) != output.astype(str)).to_numpy().sum())
    return {
        "source_file": source_name,
        "output_file": output_name,
        "rows": len(source),
        "numeric_columns": len(numeric_columns),
        "numeric_values_identical": numeric_identical,
        "row_and_column_order_identical": True,
        "changed_text_cells": changed_cells,
        "all_changes_exactly_match_label_map": all_cells_as_expected,
    }


def prepare_source_tables() -> None:
    for directory in (PNG, PDF, TABLES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    for output_name, source_path in SOURCE_MAP.items():
        source = pd.read_csv(source_path)
        output = replace_labels_only(source)
        checks.append(verify_label_only(source, output, str(source_path), output_name))
        output.to_csv(TABLES / output_name, index=False)
    for output_name, source_path in UNCHANGED_S6_SOURCES.items():
        shutil.copy2(source_path, TABLES / output_name)
    pd.DataFrame(checks).to_csv(REPORTS / "NUMERIC_INVARIANCE_CHECK.csv", index=False)


def save_figure(fig: plt.Figure, stem: str, *, dpi: int, pad: float = 0.08) -> None:
    fig.savefig(PNG / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white")
    fig.savefig(PDF / f"{stem}.pdf", bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)


def render_figure9() -> None:
    reset_style(original_figure9=True)
    transitions = pd.read_csv(TABLES / "Figure_9_transition_probabilities_neutral_labels.csv")
    transitions = transitions[transitions["period"].eq("all")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 11.0))
    image = None
    plotted_rows = []
    for axis, land_type in zip(axes.flat, LANDS):
        subset = transitions[transitions["land_type"].eq(land_type)]
        matrix = (
            subset.pivot(index="from_state", columns="to_state", values="probability_percent")
            .reindex(index=NEW_STATES, columns=NEW_STATES)
        )
        if matrix.isna().any().any() or matrix.shape != (4, 4):
            raise RuntimeError(f"Incomplete Figure 9 matrix for {land_type}")
        image = axis.imshow(matrix.to_numpy(float), cmap="YlGnBu", vmin=0, vmax=100)
        xlabels = [value.replace(" / ", "/\n") for value in NEW_STATES]
        axis.set_xticks(range(4), xlabels)
        axis.set_yticks(range(4), NEW_STATES)
        axis.set_xlabel("State at month t+1")
        axis.set_ylabel("State at month t")
        axis.set_title(land_type, fontweight="bold")
        for source_index in range(4):
            for target_index in range(4):
                value = float(matrix.iloc[source_index, target_index])
                axis.text(
                    target_index,
                    source_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="white" if value > 58 else "black",
                )
                plotted_rows.append(
                    {
                        "land_type": land_type,
                        "from_state": NEW_STATES[source_index],
                        "to_state": NEW_STATES[target_index],
                        "probability_percent_plotted": value,
                    }
                )
    fig.subplots_adjust(left=0.12, right=0.86, bottom=0.08, top=0.92, wspace=0.38, hspace=0.42)
    colorbar_axis = fig.add_axes([0.90, 0.20, 0.02, 0.58])
    fig.colorbar(image, cax=colorbar_axis, label="Observed transition proportion (%)")
    save_figure(fig, "Figure_9_neutral_state_labels", dpi=450, pad=0.0)
    plotted = pd.DataFrame(plotted_rows)
    plotted.to_csv(TABLES / "Figure_9_exact_plotted_matrix_values.csv", index=False)
    merged = plotted.merge(
        transitions[["land_type", "from_state", "to_state", "probability_percent"]],
        on=["land_type", "from_state", "to_state"],
        validate="one_to_one",
    )
    if not merged["probability_percent_plotted"].equals(merged["probability_percent"]):
        raise RuntimeError("Figure 9 plotted values differ from the finalized source table")


def render_state_null() -> None:
    reset_style()
    summary = pd.read_csv(TABLES / "Figure_S_state_transition_null_summary_neutral_labels.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), sharex=True, sharey=True)
    panel_names = ["(a)", "(b)", "(c)", "(d)"]
    y = np.arange(len(NEW_STATES))[::-1]
    for ax, land, panel in zip(axes.ravel(), LANDS, panel_names):
        data = summary[summary["land_type"].eq(land)].set_index("state").reindex(NEW_STATES)
        if data.isna().all(axis=1).any():
            raise RuntimeError(f"Incomplete null-summary states for {land}")
        for ypos, state in zip(y, NEW_STATES):
            row = data.loc[state]
            ax.hlines(ypos, row.null_ci95_low, row.null_ci95_high, color="#A2A9B3", lw=5.5, zorder=1)
            ax.plot([row.null_ci95_low, row.null_ci95_low], [ypos - 0.09, ypos + 0.09], color="#7E8794", lw=1.0)
            ax.plot([row.null_ci95_high, row.null_ci95_high], [ypos - 0.09, ypos + 0.09], color="#7E8794", lw=1.0)
            ax.scatter(row.null_median, ypos, s=35, facecolor="white", edgecolor="#34495E", linewidth=1.2, zorder=3)
            ax.scatter(row.observed_diagonal_probability, ypos, s=46, marker="D", color="#C8322B", edgecolor="white", linewidth=0.45, zorder=4)
        ax.set_yticks(y, NEW_STATES)
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
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.205, right=0.98, hspace=0.30, wspace=0.18)
    save_figure(fig, "Figure_S_state_transition_null_neutral_state_labels", dpi=450, pad=0.07)


def render_s6() -> None:
    reset_style()
    agreement = pd.read_csv(TABLES / "Figure_S6_agreement_unchanged.csv")
    occupancy = pd.read_csv(TABLES / "Figure_S6_occupancy_neutral_labels.csv")
    transitions = pd.read_csv(TABLES / "Figure_S6_transition_matrices_neutral_labels.csv")
    panel_b = pd.read_csv(TABLES / "Figure_S6_panel_b_plot_source_unchanged.csv")
    alternatives = ["33_67_extremes", "kmeans", "gmm"]
    labels = ["33/67", "k-means", "GMM"]
    a = agreement[agreement["method"].isin(alternatives)].pivot(index="land_type", columns="method", values="cohen_kappa").reindex(LANDS, columns=alternatives)
    a.columns = labels
    b = panel_b.pivot(index="land_type", columns="method", values="maximum_change_percentage_points").reindex(LANDS, columns=alternatives)
    b.columns = labels
    state = "Dry / Lower NIRv"
    c = transitions[(transitions["from_state"].eq(state)) & (transitions["to_state"].eq(state))].pivot(index="land_type", columns="method", values="probability_percent").reindex(LANDS, columns=["median", *alternatives])
    c.columns = ["Median", "33/67", "k-means", "GMM"]

    fig = plt.figure(figsize=(14.4, 4.9), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], wspace=0.24)
    panels = [
        (fig.add_subplot(gs[0, 0]), a, "(a) Classification agreement*", "YlGnBu", 0, 1, "Cohen's kappa"),
        (fig.add_subplot(gs[0, 1]), b, "(b) Occupancy sensitivity", "YlOrRd", 0, max(30, float(np.nanmax(b.to_numpy()))), "Maximum change (percentage points)"),
        (fig.add_subplot(gs[0, 2]), c, "(c) Dry / Lower NIRv persistence", "PuBuGn", 0, 100, "Self-transition probability (%)"),
    ]
    formats = [".2f", ".1f", ".1f"]
    for (ax, data, title, cmap, vmin, vmax, colorbar_label), fmt in zip(panels, formats):
        image = ax.imshow(data.to_numpy(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11.4, fontweight="bold", pad=8)
        ax.set_xticks(np.arange(len(data.columns)), data.columns)
        ax.set_yticks(np.arange(len(data.index)), data.index)
        ax.set_xticks(np.arange(-0.5, len(data.columns), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(data.index), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = float(data.iloc[row, col])
                rgba = mpl.colormaps[cmap](norm(value))
                luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(col, row, format(value, fmt), ha="center", va="center", fontsize=8.6, color="white" if luminance < 0.48 else "#202020")
        colorbar = fig.colorbar(image, ax=ax, fraction=0.040, pad=0.022, aspect=22)
        colorbar.set_label(colorbar_label, fontsize=8.5)
        colorbar.ax.tick_params(labelsize=7.8)
    panels[0][0].set_xlabel("Alternative method vs Figure-9 median")
    panels[1][0].set_xlabel("Alternative method vs median")
    panels[2][0].set_xlabel("Classification method")
    fig.text(0.01, -0.065, "* For 33/67, agreement is calculated only for retained extreme/comparable months.", fontsize=8.2, color="#404040", ha="left")
    save_figure(fig, "Figure_S6_neutral_state_labels", dpi=500, pad=0.07)


def write_reports() -> None:
    caption = """# Updated captions

## Figure 9 — exact updated caption

Figure 9. Ecosystem-state transition probabilities across four California land types during 2001–2024. Rows indicate the descriptive state at month (t), and columns indicate the state at month (t+1). Values are row-normalized transition probabilities, so each row sums to 100%. States are defined from median-based combinations of EWBI and NIRv anomaly classes: Wet / Higher NIRv, Wet / Lower NIRv, Dry / Higher NIRv, and Dry / Lower NIRv. The labels denote relative forcing–response combinations and should be interpreted as descriptive, threshold-dependent states rather than mechanistic ecological regimes.

## Dependent supplementary caption updates

Figure S6. Sensitivity of ecosystem-state results to the classification method. (a) Agreement between the reference median rule and the 33/67 percentile, k-means, and Gaussian mixture model alternatives, measured with Cohen's kappa; larger values mean closer agreement. For the 33/67 method, agreement is calculated only for retained extreme/comparable months. (b) Largest change in the proportion assigned to any state, in percentage points, relative to the reference method; smaller values mean more stable state occupancy. (c) Probability that a Dry / Lower NIRv case remains Dry / Lower NIRv in the next month under each method. These state classes are descriptive and method-dependent, not fixed ecological regimes.

Figure Sx. Observed diagonal persistence of the four median-defined EWBI–NIRv states relative to autocorrelation-preserving null distributions. Red diamonds show observed persistence, open circles show null medians, and gray intervals show the null 2.5th–97.5th percentile range from 500 independent IAAFT surrogate pairs per land type. State labels indicate relative EWBI forcing and NIRv response combinations and do not imply validated Markov ecosystem dynamics.
"""
    (ROOT / "UPDATED_CAPTIONS.md").write_text(caption, encoding="utf-8")

    replacements = pd.DataFrame(
        [
            {"old_label": old, "new_label": new, "state_order": index + 1}
            for index, (old, new) in enumerate(LABEL_MAP.items())
        ]
    )
    replacements.to_csv(ROOT / "STATE_LABEL_MAPPING.csv", index=False)

    inventory_rows = [
        ("Figure", "figures_png/Figure_9_neutral_state_labels.png", "All x/y state tick labels"),
        ("Figure", "figures_pdf/Figure_9_neutral_state_labels.pdf", "All x/y state tick labels"),
        ("Table", "source_tables/Figure_9_transition_probabilities_neutral_labels.csv", "from_state; to_state"),
        ("Table", "source_tables/Figure_9_exact_plotted_matrix_values.csv", "from_state; to_state"),
        ("Figure", "figures_png/Figure_S_state_transition_null_neutral_state_labels.png", "All y-axis state labels"),
        ("Figure", "figures_pdf/Figure_S_state_transition_null_neutral_state_labels.pdf", "All y-axis state labels"),
        ("Table", "source_tables/Figure_S_state_transition_null_summary_neutral_labels.csv", "state"),
        ("Table", "source_tables/Figure_S_state_transition_null_distributions_neutral_labels.csv", "state"),
        ("Table", "source_tables/Figure_S_state_transition_null_exact_input_neutral_labels.csv", "ecosystem_state"),
        ("Figure", "figures_png/Figure_S6_neutral_state_labels.png", "Panel (c) title"),
        ("Figure", "figures_pdf/Figure_S6_neutral_state_labels.pdf", "Panel (c) title"),
        ("Table", "source_tables/Figure_S6_occupancy_neutral_labels.csv", "state"),
        ("Table", "source_tables/Figure_S6_transition_matrices_neutral_labels.csv", "from_state; to_state"),
        ("Table", "source_tables/Figure_S6_exact_monthly_states_neutral_labels.csv", "ecosystem_state; state_median; state_33_67_extremes; state_kmeans; state_gmm"),
        ("Caption", "UPDATED_CAPTIONS.md", "Figure 9 and dependent supplementary wording"),
    ]
    inventory = pd.DataFrame(inventory_rows, columns=["item_type", "relative_path", "location_of_replacement"])
    inventory.to_csv(ROOT / "STATE_LABEL_REPLACEMENT_INVENTORY.csv", index=False)

    summary = """# Figure 9 neutral-label update

This is a **label-only** update. The finalized continuous inputs, median classification logic, state assignments, transition counts, transition probabilities, null results, and S6 sensitivity values were not recomputed or changed.

## State order retained

1. Wet / Higher NIRv
2. Wet / Lower NIRv
3. Dry / Higher NIRv
4. Dry / Lower NIRv

## Updated figures

- Main paper: `figures_png/Figure_9_neutral_state_labels.png` and PDF counterpart.
- Supplement: `figures_png/Figure_S_state_transition_null_neutral_state_labels.png` and PDF counterpart.
- Supplement: `figures_png/Figure_S6_neutral_state_labels.png` and PDF counterpart; only panel (c) terminology was changed, while the prior 33/67 clarification was retained.

## Verification

- Figure 9 contains the same 64 finalized matrix values in the same land-type panels and state order.
- Every numeric column in each label-updated source table is exactly identical in memory to its finalized source.
- `reports/NUMERIC_INVARIANCE_CHECK.csv` records the row counts and checks.
- `STATE_LABEL_REPLACEMENT_INVENTORY.csv` lists every updated figure, table, and caption artifact in this package.
- Originals in `FINALIZED FIGURES`, `REVISION_COMPUTATIONAL_PHASE`, and `FIGURE_REFINEMENT_PHASE` were not overwritten.
"""
    (ROOT / "README.md").write_text(summary, encoding="utf-8")


def write_manifest() -> None:
    rows = []
    for path in sorted(
        item for item in ROOT.rglob("*")
        if item.is_file() and item.name != "OUTPUT_MANIFEST.csv"
    ):
        rows.append(
            {
                "relative_path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(ROOT / "OUTPUT_MANIFEST.csv", index=False)


def main() -> None:
    prepare_source_tables()
    render_figure9()
    render_state_null()
    render_s6()
    write_reports()
    write_manifest()
    print("Rendered label-only Figure 9, null/persistence, and S6 updates.")


if __name__ == "__main__":
    main()
