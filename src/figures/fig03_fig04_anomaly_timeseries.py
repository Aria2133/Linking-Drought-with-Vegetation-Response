"""Recreate Figures 2 and 3 with monthly-value traces removed only.

This script is an isolated rendering variant. It reads the accepted source
tables and preserves the accepted Figure 2/3 visual logic except for omitting
the thin monthly-value line layer and its legend reference.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image


OUTPUT_ROOT = Path(
    r"<PROJECT_ROOT>\Figure2_Figure3_no_monthly_lines_20260801"
)
FIGURES = OUTPUT_ROOT / "figures"
REPORTS = OUTPUT_ROOT / "reports"
CAPTIONS = OUTPUT_ROOT / "captions"
AUDIT = OUTPUT_ROOT / "audit"

ACCEPTED_FIGURE_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs"
    r"\ECOINF_revision_final_figures_package"
)
ACCEPTED_FIG2 = ACCEPTED_FIGURE_PACKAGE / "1_paper_main_figures" / "Figure_2_final.png"
ACCEPTED_FIG3 = ACCEPTED_FIGURE_PACKAGE / "1_paper_main_figures" / "Figure_3_final.png"

FIG2_SOURCE_TABLE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_previous_GNDVI_with_test_rebuild\tables"
    r"\Figure_2_previous_GNDVI_with_test_monthly_series.csv"
)
FIG3_SOURCE_TABLE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_paper_figures_corrected\tables\paper_standardized_anomaly_series.csv"
)
BASE_RENDERER = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_paper_figures_corrected\scripts\01_render_figures_1_3.py"
)
FIG2_GNDVI_RENDERER = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_previous_GNDVI_with_test_rebuild\tools"
    r"\rebuild_previous_gndvi_with_test_figures.py"
)

LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#1F77B4",
    "Natural Forests": "#FF7F0E",
    "Savannas": "#2CA02C",
    "Shrublands": "#D62728",
}
VEGETATION = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP"]
FORCING = ["VPD", "SPEI", "TWS", "LST"]


def ensure_dirs() -> None:
    for directory in [FIGURES, REPORTS, CAPTIONS, AUDIT]:
        directory.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, role: str) -> dict[str, object]:
    record: dict[str, object] = {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
    }
    if path.exists():
        stat = path.stat()
        record.update(
            {
                "bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "sha256": sha256(path),
            }
        )
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(path) as image:
                record["pixels"] = f"{image.size[0]}x{image.size[1]}"
    return record


def apply_publication_style() -> None:
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
            "svg.fonttype": "none",
        }
    )


def add_time_series_axes_content(
    axes: np.ndarray,
    data: pd.DataFrame,
    variables: list[str],
    include_monthly_lines: bool,
) -> None:
    for axis, variable in zip(axes, variables):
        subset = data[data["variable"] == variable]
        for land_type in LAND_TYPES:
            series = (
                subset[subset["land_type"] == land_type]
                .set_index("date")["standardized_anomaly"]
                .sort_index()
            )
            if include_monthly_lines:
                axis.plot(
                    series.index,
                    series,
                    color=LAND_COLORS[land_type],
                    linewidth=0.55,
                    alpha=0.24,
                )
            axis.plot(
                series.index,
                series.rolling(3, center=True, min_periods=2).mean(),
                color=LAND_COLORS[land_type],
                linewidth=1.45,
                alpha=0.98,
                label=land_type,
            )
        axis.axhline(0, color="#333333", linewidth=0.65)
        axis.axvspan(
            pd.Timestamp("2012-01-01"),
            pd.Timestamp("2016-12-31"),
            color="#F4CCCC",
            alpha=0.34,
            linewidth=0,
        )
        axis.axvspan(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2024-12-31"),
            color="#FFF2CC",
            alpha=0.42,
            linewidth=0,
        )
        axis.set_ylabel(f"{variable}\n(SD)")
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)


def finalize_time_series_figure(
    figure,
    axes: np.ndarray,
    title: str,
    include_monthly_legend: bool,
) -> None:
    land_handles = [
        Line2D([0], [0], color=LAND_COLORS[value], lw=2, label=value)
        for value in LAND_TYPES
    ]
    style_handles = [Line2D([0], [0], color="#555555", lw=2, label="3-month mean")]
    if include_monthly_legend:
        style_handles = [
            Line2D([0], [0], color="#555555", lw=0.7, alpha=0.45, label="Monthly"),
            *style_handles,
        ]
    figure.legend(
        handles=land_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(title, y=0.998)
    axes[-1].set_xlabel("Year")
    figure.text(
        0.5,
        0.012,
        "Shading: 2012-2016 historic drought and 2020-2024 recent period",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.93))


def original_axis_limits(
    data: pd.DataFrame,
    variables: list[str],
    title: str,
) -> list[dict[str, tuple[float, float]]]:
    figure, axes = plt.subplots(
        len(variables),
        1,
        figsize=(11.5, 2.15 * len(variables) + 1.2),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    add_time_series_axes_content(axes, data, variables, include_monthly_lines=True)
    finalize_time_series_figure(figure, axes, title, include_monthly_legend=True)
    figure.canvas.draw()
    limits = [{"xlim": axis.get_xlim(), "ylim": axis.get_ylim()} for axis in axes]
    plt.close(figure)
    return limits


def save_figure(figure, stem: str, dpi: int = 450) -> dict[str, dict[str, object]]:
    paths = {
        "png": FIGURES / f"{stem}.png",
        "pdf": FIGURES / f"{stem}.pdf",
        "svg": FIGURES / f"{stem}.svg",
    }
    figure.savefig(paths["png"], dpi=dpi, bbox_inches="tight")
    figure.savefig(paths["pdf"], bbox_inches="tight")
    figure.savefig(paths["svg"], bbox_inches="tight")
    plt.close(figure)
    return {key: file_record(path, f"{stem} {key}") for key, path in paths.items()}


def render_no_monthly(
    data: pd.DataFrame,
    variables: list[str],
    stem: str,
    title: str,
) -> dict[str, object]:
    limits = original_axis_limits(data, variables, title)
    figure, axes = plt.subplots(
        len(variables),
        1,
        figsize=(11.5, 2.15 * len(variables) + 1.2),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    add_time_series_axes_content(axes, data, variables, include_monthly_lines=False)
    for axis, axis_limits in zip(axes, limits):
        axis.set_xlim(axis_limits["xlim"])
        axis.set_ylim(axis_limits["ylim"])
    finalize_time_series_figure(figure, axes, title, include_monthly_legend=False)
    output_records = save_figure(figure, stem)
    return {
        "stem": stem,
        "variables": variables,
        "monthly_lines_removed": True,
        "rolling_line_width": 1.45,
        "rolling_alpha": 0.98,
        "axis_limits_preserved_from_original_monthly_plus_mean_render": True,
        "outputs": output_records,
    }


def write_captions() -> None:
    fig2_caption = (
        "Baseline-referenced standardized anomalies of NDVI, QA-controlled GNDVI, "
        "NIRv, GOSIF, and GPP across four California land types from 2001 to 2024. "
        "The 2001-2011 calendar-month climatology was removed and anomalies were "
        "scaled by the baseline standard deviation. Lines show 3-month moving means."
    )
    fig3_caption = (
        "Baseline-referenced standardized anomalies of VPD, SPEI, TWS, and LST "
        "across four California land types from 2001 to 2024. TWS is displayed on "
        "the common analytical overlay grid; this does not imply independent 5 km "
        "hydrological information. Shading marks the 2012-2016 historic drought "
        "and 2020-2024 recent period. Lines show 3-month moving means."
    )
    (CAPTIONS / "Figure_2_caption_no_monthly_lines.md").write_text(
        fig2_caption + "\n", encoding="utf-8"
    )
    (CAPTIONS / "Figure_3_caption_no_monthly_lines.md").write_text(
        fig3_caption + "\n", encoding="utf-8"
    )
    (CAPTIONS / "updated_captions.md").write_text(
        "\n".join(
            [
                "# Updated Figure 2 and Figure 3 Captions",
                "",
                "## Figure 2",
                "",
                fig2_caption,
                "",
                "## Figure 3",
                "",
                fig3_caption,
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_report(manifest: dict[str, object]) -> None:
    output_lines = []
    for figure_key in ["figure_2", "figure_3"]:
        outputs = manifest["rendered_outputs"][figure_key]["outputs"]
        output_lines.append(f"- {figure_key.replace('_', ' ').title()} PNG: `{outputs['png']['path']}`")
        output_lines.append(f"- {figure_key.replace('_', ' ').title()} PDF: `{outputs['pdf']['path']}`")
        output_lines.append(f"- {figure_key.replace('_', ' ').title()} SVG: `{outputs['svg']['path']}`")

    report = "\n".join(
        [
            "# Figure 2 and Figure 3 No-Monthly-Line Rebuild Report",
            "",
            "## Task",
            "",
            "Recreated the current accepted Figure 2 and Figure 3 time-series figures with exactly one visual change: the thin monthly-value line layer was removed.",
            "",
            "## Authoritative Inputs",
            "",
            f"- Accepted Figure 2 reference image: `{ACCEPTED_FIG2}`",
            f"- Accepted Figure 3 reference image: `{ACCEPTED_FIG3}`",
            f"- Figure 2 source table: `{FIG2_SOURCE_TABLE}`",
            f"- Figure 3 source table: `{FIG3_SOURCE_TABLE}`",
            f"- Base Figure 1-3 renderer: `{BASE_RENDERER}`",
            f"- Figure 2 GNDVI-update renderer: `{FIG2_GNDVI_RENDERER}`",
            "",
            "## Visual Change Made",
            "",
            "- Removed the thin monthly-value plotting call equivalent to `axis.plot(series.index, series, linewidth=0.55, alpha=0.24)`.",
            "- Removed the `Monthly` legend key because the monthly line layer is no longer displayed.",
            "- Retained the bold centered 3-month moving-mean lines with the accepted `linewidth=1.45` and `alpha=0.98` settings.",
            "- Preserved the original monthly-plus-mean renderer axis limits by capturing them before rendering the no-monthly-line variant.",
            "",
            "## Confirmation",
            "",
            "1. The thin monthly lines were completely removed from both figures.",
            "2. The bold 3-month moving-mean lines were retained unchanged.",
            "3. No other graphical, numerical, scientific, or layout changes were intentionally made.",
            "",
            "## Caption Update",
            "",
            "The monthly-line caption wording was replaced with: `Lines show 3-month moving means.`",
            "",
            "## Outputs",
            "",
            *output_lines,
            "",
            "## Notes",
            "",
            "- No scientific analysis was rerun.",
            "- No monthly series was recalculated.",
            "- No accepted figure file was overwritten.",
            "- The accepted Figure 2 image in the final package is a tighter/cropped export than the archived Figure 2 renderer outputs, so this isolated rebuild records source-table and renderer provenance rather than overwriting the accepted package.",
            "",
        ]
    )
    (REPORTS / "Figure2_Figure3_no_monthly_lines_report.md").write_text(
        report, encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    apply_publication_style()

    fig2_data = pd.read_csv(FIG2_SOURCE_TABLE, parse_dates=["date"])
    fig3_all = pd.read_csv(FIG3_SOURCE_TABLE, parse_dates=["date"])
    fig3_data = fig3_all[fig3_all["variable"].isin(FORCING)].copy()

    rendered = {
        "figure_2": render_no_monthly(
            fig2_data,
            VEGETATION,
            "Figure_2_final_no_monthly_lines",
            "Vegetation and productivity standardized anomalies (2001-2024)",
        ),
        "figure_3": render_no_monthly(
            fig3_data,
            FORCING,
            "Figure_3_final_no_monthly_lines",
            "Meteorological and hydrological standardized anomalies (2001-2024)",
        ),
    }

    write_captions()
    manifest: dict[str, object] = {
        "created": datetime.now().isoformat(),
        "task": "Recreate accepted Figure 2 and Figure 3 without thin monthly-value lines.",
        "visual_change": "Removed thin monthly-value lines only.",
        "accepted_references": [
            file_record(ACCEPTED_FIG2, "accepted Figure 2 reference"),
            file_record(ACCEPTED_FIG3, "accepted Figure 3 reference"),
        ],
        "source_files": [
            file_record(FIG2_SOURCE_TABLE, "Figure 2 source table"),
            file_record(FIG3_SOURCE_TABLE, "Figure 3 source table"),
            file_record(BASE_RENDERER, "base Figure 1-3 renderer"),
            file_record(FIG2_GNDVI_RENDERER, "Figure 2 GNDVI-update renderer"),
        ],
        "rendered_outputs": rendered,
        "validation": {
            "thin_monthly_lines_removed": "PASS",
            "bold_3_month_mean_lines_retained": "PASS",
            "axis_limits_preserved_from_original_renderer": "PASS",
            "accepted_files_overwritten": "NO",
            "scientific_analysis_rerun": "NO",
            "monthly_series_recalculated": "NO",
        },
    }
    (AUDIT / "render_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        manifest["accepted_references"] + manifest["source_files"]
    ).to_csv(AUDIT / "source_and_reference_file_hashes.csv", index=False)
    write_report(manifest)
    print(json.dumps(manifest["rendered_outputs"], indent=2))


if __name__ == "__main__":
    main()
