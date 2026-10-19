from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(r"<PROJECT_ROOT>\Docs\major_revision_analyses\corrected_true_BiLSTM_rerun_20260728")
ACCEPTED_PDF = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses\ALTERNATIVE_Figure7_polar_redesign_no_detrend_update\Figures\Figure_7_alternative_polar_large_readable_v2.pdf"
)
SOURCE_TABLE = ROOT / "08_downstream" / "final_corrected_Figure_7_source_table.csv"
ACCEPTED_SOURCE_TABLE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses\ALTERNATIVE_Figure7_polar_redesign_no_detrend_update\Source tables used for plotting\figure7_alternative_plotting_table.csv"
)
OUTDIR = ROOT / "08_downstream"

TARGET_ORDER = ["GNDVI", "GOSIF/SIF", "GPP", "LST", "NDVI", "NIRv", "SPEI-1", "TWS", "VPDmax"]
LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
MODEL_ORDER = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
MODEL_COLORS = {
    "LSTM": "#1f77b4",
    "BiLSTM": "#9467bd",
    "TCN": "#2ca02c",
    "N-BEATS": "#ff7f0e",
    "N-HiTS": "#d62728",
    "XGBoost": "#17becf",
}

# Exact page size of the accepted v2 PDF, read with pypdf.
ACCEPTED_WIDTH_IN = 23.838197686072224
ACCEPTED_HEIGHT_IN = 41.65309895833334


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def render(table: pd.DataFrame, png: Path, pdf: Path) -> None:
    configure_style()
    target_angles = np.linspace(0, 2 * np.pi, len(TARGET_ORDER), endpoint=False)
    offsets = np.linspace(-0.105, 0.105, len(MODEL_ORDER))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        subplot_kw={"projection": "polar"},
        figsize=(ACCEPTED_WIDTH_IN, ACCEPTED_HEIGHT_IN),
        dpi=300,
    )
    # Keep the accepted v2 page geometry, but leave a clearer buffer between
    # the bottom-row polar panels and the global legend.
    fig.subplots_adjust(left=0.055, right=0.945, top=0.955, bottom=0.112, wspace=0.16, hspace=0.31)

    for row_idx, land in enumerate(LAND_ORDER):
        for col_idx, metric in enumerate(["R2", "nRMSE"]):
            ax = axes[row_idx, col_idx]
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(target_angles)
            ax.set_xticklabels(TARGET_ORDER, fontsize=21, fontweight="bold")
            ax.tick_params(axis="x", pad=27)
            ax.grid(True, color="#CFCFCF", alpha=0.55, linewidth=1.0)
            ax.spines["polar"].set_color("#333333")
            ax.spines["polar"].set_linewidth(1.6)
            ax.set_rlabel_position(225)

            if metric == "R2":
                ax.set_ylim(0, 1.0)
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"], fontsize=18)
                title = f"{land} - R$^2$"
                value_col = "r2_plot_radius"
                err_col = "sd_r2"
            else:
                ax.set_ylim(0, 2.0)
                ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
                ax.set_yticklabels(["0", "0.5", "1.0", "1.5", "2"], fontsize=18)
                title = f"{land} - nRMSE"
                value_col = "nrmse_plot_radius"
                err_col = "sd_normalized_rmse"
            ax.set_title(title, fontsize=28, fontweight="bold", pad=30)

            sub = table[table["land_type_display"] == land]
            for model_idx, model in enumerate(MODEL_ORDER):
                model_sub = sub[sub["model"] == model].set_index("target_display")
                color = MODEL_COLORS[model]
                for target_idx, target in enumerate(TARGET_ORDER):
                    if target not in model_sub.index:
                        continue
                    data = model_sub.loc[target]
                    theta = target_angles[target_idx] + offsets[model_idx]
                    radius = float(data[value_col])
                    err = float(data[err_col]) if pd.notna(data[err_col]) else np.nan

                    if metric == "R2" and bool(data["r2_negative_flag"]):
                        ax.scatter(theta, 0.02, s=130, marker="x", color=color, linewidths=3.0, zorder=5)
                    else:
                        yerr = None
                        if np.isfinite(err) and err > 0:
                            yerr = [[min(err, max(radius, 0))], [err]]
                        ax.errorbar(
                            theta,
                            radius,
                            yerr=yerr,
                            fmt="o",
                            markersize=14.5,
                            markerfacecolor=color,
                            markeredgecolor="white",
                            markeredgewidth=1.05,
                            ecolor=color,
                            elinewidth=2.4,
                            capsize=5.2,
                            alpha=0.95,
                            zorder=4,
                        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=16,
            markerfacecolor=MODEL_COLORS[m],
            markeredgecolor="white",
            label=m,
        )
        for m in MODEL_ORDER
    ]
    legend = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.022),
        ncol=6,
        title="Learned models",
        frameon=True,
        fancybox=False,
        edgecolor="#333333",
        facecolor="white",
        fontsize=20,
        title_fontsize=21,
        borderpad=0.9,
        labelspacing=0.9,
        handletextpad=0.7,
        columnspacing=1.8,
    )
    legend.get_title().set_fontweight("bold")
    fig.text(
        0.5,
        0.0035,
        "Radius gives the plotted metric value. Error bars are SD across available seed/fold metric rows. x at center = mean R^2 < 0.",
        ha="center",
        fontsize=15,
        color="#333333",
    )

    # Do not use bbox_inches='tight': the accepted v2 PDF canvas size is the visual authority.
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)


def main() -> None:
    accepted_table = pd.read_csv(ACCEPTED_SOURCE_TABLE)
    table = pd.read_csv(SOURCE_TABLE)
    accepted_reconstructed_png = OUTDIR / "Figure7_accepted_v2_source_larger_text_dots_reference.png"
    accepted_reconstructed_pdf = OUTDIR / "Figure7_accepted_v2_source_larger_text_dots_reference.pdf"
    v2_png = OUTDIR / "final_corrected_Figure_7_accepted_v2_larger_text_dots.png"
    v2_pdf = OUTDIR / "final_corrected_Figure_7_accepted_v2_larger_text_dots.pdf"
    render(accepted_table, accepted_reconstructed_png, accepted_reconstructed_pdf)
    render(table, v2_png, v2_pdf)

    # Refresh the generic corrected Figure 7 outputs so they point to the accepted-v2 layout.
    render(table, OUTDIR / "final_corrected_Figure_7_polar_larger_text_dots.png", OUTDIR / "final_corrected_Figure_7_polar_larger_text_dots.pdf")

    try:
        from PIL import Image, ImageDraw

        left = Image.open(accepted_reconstructed_png).convert("RGB")
        right = Image.open(v2_png).convert("RGB")
        max_h = 2600
        scale = min(max_h / left.height, max_h / right.height, 1.0)

        def resize(img: Image.Image) -> Image.Image:
            return img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

        left_r = resize(left)
        right_r = resize(right)
        pad = 90
        header_h = 90
        out = Image.new("RGB", (left_r.width + right_r.width + pad * 3, max(left_r.height, right_r.height) + header_h + pad), "white")
        draw = ImageDraw.Draw(out)
        draw.text((pad, 25), "Accepted v2 source", fill="black")
        draw.text((left_r.width + pad * 2, 25), "Corrected true-BiLSTM patch", fill="black")
        out.paste(left_r, (pad, header_h))
        out.paste(right_r, (left_r.width + pad * 2, header_h))
        out.save(OUTDIR / "Figure7_accepted_v2_previous_vs_corrected_larger_text_dots_side_by_side.png", dpi=(180, 180))
    except Exception as exc:
        print(f"Side-by-side comparison skipped: {exc}")

    report = OUTDIR / "Figure7_accepted_v2_BiLSTM_patch_note.md"
    report.write_text(
        "\n".join(
            [
                "# Figure 7 accepted-v2 BiLSTM patch note",
                "",
                f"Accepted visual reference: `{ACCEPTED_PDF}`",
                f"Accepted plotting table: `{ACCEPTED_SOURCE_TABLE}`",
                f"Corrected plotting table: `{SOURCE_TABLE}`",
                "",
                "This rendering uses the accepted v2 polar Figure 7 canvas size and layout, while replacing only the corrected true-BiLSTM values already present in the patched source table.",
                f"The accepted and corrected PDF page sizes are `{ACCEPTED_WIDTH_IN:.6f} x {ACCEPTED_HEIGHT_IN:.6f}` inches.",
                "",
                "No forecasting model was run. No accepted original figure file was overwritten.",
                "",
                "Outputs:",
                f"- Reconstructed accepted-v2 PNG: `{accepted_reconstructed_png}`",
                f"- Reconstructed accepted-v2 PDF: `{accepted_reconstructed_pdf}`",
                f"- PNG: `{v2_png}`",
                f"- PDF: `{v2_pdf}`",
                f"- Side-by-side: `{OUTDIR / 'Figure7_accepted_v2_previous_vs_corrected_larger_text_dots_side_by_side.png'}`",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
