"""Render Figure 6 from the accepted previous-GNDVI exact-old-logic values table.

This intentionally does not recompute CCF values. It only improves fonts and
changes NaturalForests to Natural Forests.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_TABLE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\figure5_figure6_exact_source_recovery\tables\Figure6_previous_GNDVI_values.csv"
)
OUT_DIR = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_previous_GNDVI_with_test_rebuild\figures"
)
FINAL_DIR = Path(
    r"<PROJECT_ROOT>\ECOINF_revision_final_figures_package"
    r"\1_paper_main_figures"
)
AUDIT_DIR = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_previous_GNDVI_with_test_rebuild\audit"
)

LANDS = ["Grasslands", "NaturalForests", "Savannas", "Shrublands"]
DISPLAY = {
    "Grasslands": "Grasslands",
    "NaturalForests": "Natural Forests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}


def render() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    values = pd.read_csv(SOURCE_TABLE)
    lags = list(range(-6, 7))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, top=0.95, bottom=0.12)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
    im = None
    for idx, land in enumerate(LANDS):
        ax = axes[idx]
        sub = values[values["land_type"] == land].copy()
        pairs = sub[["predictor", "target", "pair"]].drop_duplicates()
        matrix = []
        labels = []
        for _, pair_row in pairs.iterrows():
            row_values = sub[
                (sub["predictor"] == pair_row["predictor"])
                & (sub["target"] == pair_row["target"])
            ]
            lookup = dict(zip(row_values["lag"], row_values["ccf"]))
            matrix.append([lookup.get(lag, np.nan) for lag in lags])
            labels.append(f"{pair_row['predictor']}\u2192{pair_row['target']}")
        matrix = np.asarray(matrix, dtype=float)
        im = ax.imshow(
            matrix,
            cmap="RdBu_r",
            aspect="auto",
            vmin=-1,
            vmax=1,
            interpolation="bilinear",
        )
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9.5)
        ax.set_xticks(range(len(lags)))
        ax.set_xticklabels(lags, fontsize=12.5)
        ax.set_xlabel(
            "Lag (months)\nNegative = Predictor LEADS target",
            fontsize=14,
            weight="bold",
        )
        ax.set_ylabel("Predictor \u2192 Target (Unique Pairs)", fontsize=14, weight="bold")
        ax.set_title(f"{DISPLAY[land]}\nCross-Correlation Function", fontsize=15, weight="bold")
        ax.axvline(6, color="black", linewidth=2.5, linestyle="--", alpha=0.8, label="Lag = 0")
        ax.legend(loc="upper right", fontsize=11.5)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Cross-Correlation Coefficient\n(Red = Positive | Blue = Negative)",
        fontsize=15,
        weight="bold",
    )
    cbar.ax.tick_params(labelsize=12.5)

    png = OUT_DIR / "Figure_6_previous_GNDVI_exact_old_logic_REFERENCE_font_fixed.png"
    pdf = OUT_DIR / "Figure_6_previous_GNDVI_exact_old_logic_REFERENCE_font_fixed.pdf"
    png600 = OUT_DIR / "Figure_6_previous_GNDVI_exact_old_logic_REFERENCE_font_fixed_600dpi.png"
    fig.savefig(png, dpi=450, bbox_inches="tight")
    fig.savefig(png600, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    # Replace final Figure 6 only with the accepted-value, font-fixed version.
    backup_png = FINAL_DIR / "Figure_6_final_backup_before_reference_font_fix.png"
    backup_pdf = FINAL_DIR / "Figure_6_final_backup_before_reference_font_fix.pdf"
    if (FINAL_DIR / "Figure_6_final.png").exists() and not backup_png.exists():
        (FINAL_DIR / "Figure_6_final.png").replace(backup_png)
    if (FINAL_DIR / "Figure_6_final.pdf").exists() and not backup_pdf.exists():
        (FINAL_DIR / "Figure_6_final.pdf").replace(backup_pdf)
    import shutil

    shutil.copy2(png, FINAL_DIR / "Figure_6_final.png")
    shutil.copy2(pdf, FINAL_DIR / "Figure_6_final.pdf")
    pd.DataFrame(
        [
            {
                "source_values": str(SOURCE_TABLE),
                "output_png": str(png),
                "output_pdf": str(pdf),
                "final_png": str(FINAL_DIR / "Figure_6_final.png"),
                "final_pdf": str(FINAL_DIR / "Figure_6_final.pdf"),
                "change": "font/title only; no CCF recomputation",
            }
        ]
    ).to_csv(AUDIT_DIR / "Figure6_reference_font_fix_replacement_manifest.csv", index=False)


if __name__ == "__main__":
    render()
