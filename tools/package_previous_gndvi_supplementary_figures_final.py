from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd


SOURCE = Path(r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\full_previous_GNDVI_supplementary_rebuild")
CURRENT_SUPP = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_supplementary_information_final_separate_figures")
OUT = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_supplementary_figures_previous_GNDVI_final")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "png": OUT / "figures_png",
        "pdf": OUT / "figures_pdf",
        "captions": OUT / "captions",
        "tables": OUT / "tables_and_audits",
        "notes": OUT / "notes",
        "source": OUT / "source_package_files",
        "pending": OUT / "figure_S6_pending_reference_only",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def copy_file(src: Path, dst: Path, role: str, rows: list[dict[str, str]]) -> None:
    if not src.exists():
        rows.append(
            {
                "role": role,
                "source_path": str(src),
                "copied_path": "",
                "status": "missing",
                "sha256": "",
            }
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    rows.append(
        {
            "role": role,
            "source_path": str(src),
            "copied_path": str(dst),
            "status": "copied",
            "sha256": sha256(dst),
        }
    )


def main() -> None:
    dirs = ensure_dirs()
    rows: list[dict[str, str]] = []

    final_figures = [
        "Figure_S1_EWBI_response_association",
        "Figure_S2_EWBI_component_weighting_robustness",
        "Figure_S3_TWS_scale_resampling_sensitivity",
        "Figure_S4_gap_filling_validation_downstream_sensitivity",
        "Figure_S5_detrending_preprocessing_sensitivity",
        "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL",
        "Figure_S8_NIRv_stress_zone_threshold_sensitivity",
        "Figure_S9_ecosystem_state_classification_sensitivity",
    ]

    for stem in final_figures:
        copy_file(SOURCE / "figures" / f"{stem}.png", dirs["png"] / f"{stem}.png", "final_supplementary_png", rows)
        copy_file(SOURCE / "figures" / f"{stem}.pdf", dirs["pdf"] / f"{stem}.pdf", "final_supplementary_pdf", rows)

    # S6 is intentionally not treated as final previous-GNDVI-consistent output.
    blocked_stem = "Figure_S6_FORECASTING_BLOCKED_previous_GNDVI_requires_target_rerun"
    copy_file(SOURCE / "figures" / f"{blocked_stem}.png", dirs["png"] / f"{blocked_stem}.png", "blocked_s6_notice_png", rows)
    copy_file(SOURCE / "figures" / f"{blocked_stem}.pdf", dirs["pdf"] / f"{blocked_stem}.pdf", "blocked_s6_notice_pdf", rows)

    # Copy the current S6 only as reference, never as a previous-GNDVI-final figure.
    current_s6 = "Figure_S6_forecasting_baselines_rolling_origin"
    copy_file(
        CURRENT_SUPP / "figures_png" / f"{current_s6}.png",
        dirs["pending"] / f"{current_s6}_CURRENT_STRICT_QA_REFERENCE_ONLY.png",
        "current_s6_reference_only",
        rows,
    )
    copy_file(
        CURRENT_SUPP / "figures_pdf" / f"{current_s6}.pdf",
        dirs["pending"] / f"{current_s6}_CURRENT_STRICT_QA_REFERENCE_ONLY.pdf",
        "current_s6_reference_only",
        rows,
    )

    for cap in sorted((SOURCE / "captions").glob("*.md")):
        copy_file(cap, dirs["captions"] / cap.name, "caption", rows)

    for table in [
        "Supplementary_figure_full_inventory.csv",
        "Supplementary_figure_rebuild_audit.csv",
        "S1a_previous_GNDVI_period_response_panel_data.csv",
        "S1b_previous_GNDVI_component_omission_panel_data.csv",
        "S1c_previous_GNDVI_weighting_panel_data.csv",
        "S2_previous_GNDVI_component_comparison_rows.csv",
        "S2_previous_GNDVI_leave_one_component_rows.csv",
        "S2_previous_GNDVI_weighting_validation_rows.csv",
    ]:
        copy_file(SOURCE / "tables" / table, dirs["tables"] / table, "supporting_table", rows)

    for audit in [
        "figure_panel_comparison_summary.csv",
        "package_file_manifest.csv",
        "unchanged_figure_copy_log.csv",
    ]:
        copy_file(SOURCE / "audit" / audit, dirs["tables"] / audit, "audit_table", rows)

    for report in [
        "FINAL_full_previous_GNDVI_supplementary_rebuild_report.md",
        "Supplementary_figure_source_lineage_report.md",
    ]:
        copy_file(SOURCE / "report" / report, dirs["notes"] / report, "report", rows)

    copy_file(SOURCE / "tools" / "full_previous_gndvi_supplementary_rebuild.py", dirs["source"] / "full_previous_gndvi_supplementary_rebuild.py", "source_script", rows)

    s6_note = """# Figure S6 Status

Figure S6 is not included as a final previous-GNDVI-consistent supplementary figure.

Reason: Figure S6 contains GNDVI target forecasting/baseline outputs. Reusing the current Figure S6 would silently retain strict-QA GNDVI model results, which would be inconsistent if the manuscript switches to previous/original GNDVI.

This package therefore includes:

- `figures_png/Figure_S6_FORECASTING_BLOCKED_previous_GNDVI_requires_target_rerun.png`
- `figures_pdf/Figure_S6_FORECASTING_BLOCKED_previous_GNDVI_requires_target_rerun.pdf`
- `figure_S6_pending_reference_only/` with the current strict-QA S6 for visual reference only.

Minimum remaining action: locate existing previous-GNDVI GNDVI-target forecasting/baseline outputs or run a targeted GNDVI-only forecasting/baseline rerun.
"""
    (dirs["notes"] / "Figure_S6_status_note.md").write_text(s6_note, encoding="utf-8")

    readme = """# ECOINF Supplementary Figures - Previous GNDVI Final Package

This folder gathers the supplementary figures prepared after choosing the previous/original GNDVI source as the working manuscript option.

Final previous-GNDVI-ready figures:

- Figure S1: rebuilt with previous GNDVI.
- Figure S2: rebuilt with previous-GNDVI GNDVI-only EWBI rows; non-GNDVI rows unchanged.
- Figures S3, S4, S5, S7, S8, S9: copied unchanged because no GNDVI dependency was identified.

Important caveat:

- Figure S6 is not final previous-GNDVI-consistent because it depends on GNDVI-target forecasting/baseline outputs. A clear blocked notice and a current-reference-only copy are included.

No forecasting models, ML/deep-learning models, or raster extraction workflows were rerun while preparing this folder.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    pd.DataFrame(rows).to_csv(OUT / "package_manifest.csv", index=False)
    print(f"OUTPUT_ROOT={OUT}")
    print(f"COPIED_FILES={sum(row['status'] == 'copied' for row in rows)}")
    print("S6_FINAL_PREVIOUS_GNDVI_CONSISTENT=no")
    print("MODELS_RERUN=no")


if __name__ == "__main__":
    main()
