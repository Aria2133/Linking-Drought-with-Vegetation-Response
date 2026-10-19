from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_ROOT = Path(r"<PROJECT_ROOT>\Supplementary_Figure_S7_axis_wording_patch")
FIG_DIR = OUT_ROOT / "figures"
CAPTION_DIR = OUT_ROOT / "captions"
SOURCE_DIR = OUT_ROOT / "source"
PANEL_SOURCE = SOURCE_DIR / "Figure_S7_threshold_panel_source_values.csv"

LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#007C91",
    "Natural Forests": "#2E7D32",
    "Savannas": "#E58B00",
    "Shrublands": "#B23A67",
}

CAPTION = (
    "Figure S7. Sensitivity of severe NIRv-response area to the decline "
    "threshold. (a) Change in severe-response area, in km², between the "
    "historic and recent periods when a severe decline is defined using "
    "three threshold levels: mild (30%), medium (40%), and severe (50%). "
    "(b) The same change expressed as percentage points of eligible land "
    "area. Positive values indicate an increase in severe-response area or "
    "eligible-area share in the recent period relative to the historic "
    "period, whereas negative values indicate a decrease. Error bars are "
    "95% uncertainty intervals from the spatial-block bootstrap. The 40% "
    "threshold was retained in the main analysis as the medium threshold "
    "for the scope of this study. A similar direction across thresholds "
    "indicates that the conclusion is not tied to the selected threshold. "
    "Numerical values are in Supplementary Table S8."
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11.2,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "legend.fontsize": 8.2,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.75,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render() -> None:
    for directory in (FIG_DIR, CAPTION_DIR, SOURCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    configure_style()
    stress = pd.read_csv(PANEL_SOURCE)
    x = np.array([30, 40, 50])
    threshold_order = [-30, -40, -50]

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8), constrained_layout=False)
    ax_a, ax_b = axes

    for land in LAND_ORDER:
        subset = (
            stress[stress["land_type"] == land]
            .set_index("threshold")
            .reindex(threshold_order)
        )

        y_area = subset["difference_km2"].to_numpy()
        y_pp = subset["difference_percentage_points"].to_numpy()
        area_err = np.vstack(
            [
                y_area - subset["ci95_low_km2"].to_numpy(),
                subset["ci95_high_km2"].to_numpy() - y_area,
            ]
        )
        pp_err = np.vstack(
            [
                y_pp - subset["ci95_low_percentage_points"].to_numpy(),
                subset["ci95_high_percentage_points"].to_numpy() - y_pp,
            ]
        )

        ax_a.errorbar(
            x,
            y_area,
            yerr=area_err,
            marker="o",
            ms=5.2,
            capsize=3,
            lw=1.25,
            color=LAND_COLORS[land],
            label=land,
        )
        ax_b.errorbar(
            x,
            y_pp,
            yerr=pp_err,
            marker="o",
            ms=5.2,
            capsize=3,
            lw=1.25,
            color=LAND_COLORS[land],
        )

    severity_labels = {30: "Mild", 40: "Medium", 50: "Severe"}
    for axis in axes:
        axis.axhline(0, color="#3F3F3F", lw=0.8)
        axis.set_xticks(x, ["30%", "40%", "50%"])
        for threshold, label in severity_labels.items():
            axis.text(
                threshold,
                -0.095,
                label,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=7.2,
                color="#4A4A4A",
                clip_on=False,
            )
        axis.set_xlabel("NIRv decline threshold", labelpad=18)
        axis.grid(axis="y")
        axis.tick_params(axis="x", pad=2)

    ax_a.set_ylabel("Change in severe-response area (km²)")
    ax_a.set_title("(a) Absolute area")
    ax_a.legend(frameon=False, fontsize=7.7, loc="upper right")
    ax_b.set_ylabel("Change in eligible-area share (percentage points)")
    ax_b.set_title("(b) Eligible-area share")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.88, bottom=0.22, wspace=0.30)

    fig.savefig(
        FIG_DIR / "Figure_S7_NIRv_threshold_sensitivity_axis_wording.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        FIG_DIR / "Figure_S7_NIRv_threshold_sensitivity_axis_wording.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)

    (CAPTION_DIR / "Figure_S7_caption_axis_wording.md").write_text(
        CAPTION + "\n", encoding="utf-8"
    )
    (OUT_ROOT / "README_label_only_revision.md").write_text(
        "# Figure S7 Label-Only Revision\n\n"
        "- Data changed: NO.\n"
        "- Calculations rerun: NO.\n"
        "- Bootstrap results recomputed: NO.\n"
        "- Plotted values, thresholds, uncertainty intervals, colors, line styles, legend, panel structure, and layout style were preserved.\n"
        "- Edits made: updated the two y-axis labels and added concise caption support text explaining positive and negative values.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    render()
