from __future__ import annotations

import csv
import hashlib
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
AUDIT_ROOT = DESKTOP / "major_revision_analyses" / "FINAL_submission_figure_directory_audit"
MAIN_DIR = DESKTOP / "ECOINF_revision_final_figures_package" / "1_paper_main_figures"
SUPP_DIR = DESKTOP / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
PREV_GNDVI_REBUILD = DESKTOP / "major_revision_analyses" / "final_previous_GNDVI_with_test_rebuild"
DL_GNDVI = DESKTOP / "DL_GNDVI_previous_noQA_rerun"

ACCEPTED_MAIN_SOURCES = {
    "Figure 2": {
        "png": PREV_GNDVI_REBUILD / "figures" / "Figure_2_previous_GNDVI_with_test.png",
        "pdf": PREV_GNDVI_REBUILD / "figures" / "Figure_2_previous_GNDVI_with_test.pdf",
        "decision": "Previous/no-QA GNDVI with completed 2022-2024 coverage; vegetation/productivity standardized anomalies.",
    },
    "Figure 5": {
        "png": PREV_GNDVI_REBUILD / "figures" / "Figure_5_previous_GNDVI_with_test_exact_old_logic.png",
        "pdf": PREV_GNDVI_REBUILD / "figures" / "Figure_5_previous_GNDVI_with_test_exact_old_logic.pdf",
        "decision": "Original raw monthly inter-index correlation logic; previous/no-QA GNDVI; not standardized-anomaly correlation.",
    },
    "Figure 6": {
        "png": PREV_GNDVI_REBUILD / "figures" / "Figure_6_previous_GNDVI_exact_old_logic_REFERENCE_font_fixed.png",
        "pdf": PREV_GNDVI_REBUILD / "figures" / "Figure_6_previous_GNDVI_exact_old_logic_REFERENCE_font_fixed.pdf",
        "decision": "Accepted previous/no-QA CCF values; only font and Natural Forests presentation fixes.",
    },
    "Figure 7": {
        "png": DL_GNDVI / "figures" / "Figure_7_previous_noQA_GNDVI_same_visualization.png",
        "pdf": DL_GNDVI / "figures" / "Figure_7_previous_noQA_GNDVI_same_visualization.pdf",
        "decision": "Reviewer-safe benchmark with only GNDVI target rows replaced by previous/no-QA GNDVI rerun; non-GNDVI rows unchanged.",
    },
}

MAIN_EXPECTATIONS = {
    "Figure 1": "Keep corrected/enlarged final version unless missing.",
    "Figure 2": ACCEPTED_MAIN_SOURCES["Figure 2"]["decision"],
    "Figure 3": "Keep current accepted final version unless source-lineage evidence says wrong.",
    "Figure 4": "Keep rolling-origin/design figure if it matches reviewer-safe setup.",
    "Figure 5": ACCEPTED_MAIN_SOURCES["Figure 5"]["decision"],
    "Figure 6": ACCEPTED_MAIN_SOURCES["Figure 6"]["decision"],
    "Figure 7": ACCEPTED_MAIN_SOURCES["Figure 7"]["decision"],
    "Figure 8": "Keep accepted NIRv stress-zone threshold/uncertainty figure unless missing.",
    "Figure 9": "Keep corrected/enlarged final version unless missing.",
}

SUPP_EXPECTATIONS = {
    "Figure S1": "Rebuilt with previous/no-QA GNDVI.",
    "Figure S2": "Rebuilt where GNDVI-dependent EWBI rows/cells changed.",
    "Figure S3": "TWS scale/resampling sensitivity copied unchanged.",
    "Figure S4": "Gap-filling validation/downstream sensitivity copied unchanged.",
    "Figure S5": "Detrending/preprocessing sensitivity copied unchanged.",
    "Figure S6": "Rebuilt after GNDVI-only forecasting rerun; no placeholder and no old strict-QA GNDVI forecasting reuse.",
    "Figure S7": "Targeted multivariate forecasting sensitivity copied unchanged; legend clarity retained.",
    "Figure S8": "NIRv stress-zone sensitivity copied unchanged.",
    "Figure S9": "Ecosystem-state classification sensitivity copied unchanged.",
}

SUPP_STEMS = {
    "Figure S1": "Figure_S1_EWBI_response_association",
    "Figure S2": "Figure_S2_EWBI_component_weighting_robustness",
    "Figure S3": "Figure_S3_TWS_scale_resampling_sensitivity",
    "Figure S4": "Figure_S4_gap_filling_validation_downstream_sensitivity",
    "Figure S5": "Figure_S5_detrending_preprocessing_sensitivity",
    "Figure S6": "Figure_S6_forecasting_baselines_rolling_origin",
    "Figure S7": "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL",
    "Figure S8": "Figure_S8_NIRv_stress_zone_threshold_sensitivity",
    "Figure S9": "Figure_S9_ecosystem_state_classification_sensitivity",
}

RISK_PATTERNS = [
    "QA-corrected GNDVI",
    "QA-controlled GNDVI",
    "strict-QA GNDVI",
    "standardized anomaly correlation",
    "deep learning dominated",
    "BiLSTM won 34/36",
    "six deep-learning models",
    "GRU",
    "2024-only test",
    "17-fold",
    "regime shift",
    "causal transition",
    "5 km TWS",
    "fine-scale TWS",
    "anomaly correlation matrix",
    "Figure S6 not rebuilt",
]


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "tables": AUDIT_ROOT / "tables",
        "report": AUDIT_ROOT / "report",
        "audit": AUDIT_ROOT / "audit",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_dimensions(path: Path) -> str:
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        return ""
    try:
        with Image.open(path) as img:
            return f"{img.width}x{img.height}"
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def pdf_page_count(path: Path) -> str:
    if path.suffix.lower() != ".pdf":
        return ""
    if PdfReader is None:
        return "pypdf_unavailable"
    try:
        reader = PdfReader(str(path))
        return str(len(reader.pages))
    except Exception as exc:
        return f"error:{type(exc).__name__}"


def infer_role(path: Path) -> str:
    name = path.name
    parent = path.parent.name
    if re.search(r"Figure_\d+_final", name):
        return "main manuscript figure"
    if name.startswith("Figure_S"):
        if parent in {"figures_png", "figures_pdf"}:
            return "supplementary figure"
        if parent == "captions":
            return "supplementary caption"
    if "source_map" in name or "lineage" in name:
        return "source lineage/provenance"
    if "audit" in name.lower() or "manifest" in name.lower():
        return "audit/manifest"
    if path.suffix.lower() == ".csv":
        return "source/supporting table"
    if path.suffix.lower() == ".md":
        return "report/caption text"
    return ""


def source_package(path: Path) -> str:
    for root, label in [
        (MAIN_DIR, "main_final_directory"),
        (SUPP_DIR, "supplementary_final_previous_GNDVI_with_test"),
        (PREV_GNDVI_REBUILD, "final_previous_GNDVI_with_test_rebuild"),
        (DL_GNDVI, "DL_GNDVI_previous_noQA_rerun"),
    ]:
        try:
            path.relative_to(root)
            return label
        except ValueError:
            pass
    return ""


def inventory_file(path: Path) -> dict[str, str]:
    stat = path.stat()
    return {
        "file_path": str(path),
        "file_name": path.name,
        "file_size": str(stat.st_size),
        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": sha256(path),
        "image_dimensions": image_dimensions(path),
        "pdf_page_count": pdf_page_count(path),
        "role_purpose": infer_role(path),
        "source_package_folder": source_package(path),
    }


def inventory(root: Path, recursive: bool = True, exts: set[str] | None = None) -> pd.DataFrame:
    exts = exts or {".png", ".pdf", ".csv", ".md"}
    paths = root.rglob("*") if recursive else root.glob("*")
    rows = [inventory_file(p) for p in sorted(paths) if p.is_file() and p.suffix.lower() in exts]
    return pd.DataFrame(rows)


def write_csv(path: Path, rows: list[dict[str, str]] | pd.DataFrame) -> None:
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    pd.DataFrame(rows).to_csv(path, index=False)


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def compare_hash(path_a: Path, path_b: Path) -> bool:
    return path_a.exists() and path_b.exists() and sha256(path_a) == sha256(path_b)


def make_backup() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = AUDIT_ROOT / f"backup_before_replacement_{stamp}"
    backup.mkdir(parents=True, exist_ok=True)
    # Keep the timestamped backup folder, but use short child folder names to
    # avoid Windows path-length failures for long supplementary filenames.
    copy_tree(MAIN_DIR, backup / "main")
    copy_tree(SUPP_DIR, backup / "supp")
    return backup


def replace_file(src: Path, dst: Path, reason: str, rows: list[dict[str, str]]) -> None:
    before_hash = sha256(dst) if dst.exists() else ""
    shutil.copy2(src, dst)
    after_hash = sha256(dst)
    rows.append(
        {
            "file": str(dst),
            "source": str(src),
            "action": "replaced",
            "reason": reason,
            "before_sha256": before_hash,
            "after_sha256": after_hash,
        }
    )


def audit_main_and_replace(replacement_rows: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for i in range(1, 10):
        fig = f"Figure {i}"
        final_png = MAIN_DIR / f"Figure_{i}_final.png"
        final_pdf = MAIN_DIR / f"Figure_{i}_final.pdf"
        expectation = MAIN_EXPECTATIONS[fig]
        source_entry = ACCEPTED_MAIN_SOURCES.get(fig)
        source_png = source_entry["png"] if source_entry else None
        source_pdf = source_entry["pdf"] if source_entry else None
        replacement_needed = "no"
        action = "left unchanged"
        evidence = "current file exists; accepted unless missing or explicit source replacement required"
        source_folder = ""
        if source_entry:
            source_folder = str(source_png.parent)
            png_match = compare_hash(final_png, source_png)
            pdf_match = compare_hash(final_pdf, source_pdf)
            evidence = f"PNG hash match={png_match}; PDF hash match={pdf_match}"
            if fig == "Figure 7":
                # Explicitly approved replacement source after GNDVI-only rerun.
                if not png_match:
                    replace_file(source_png, final_png, "Approved Figure 7 previous/no-QA GNDVI rerun PNG", replacement_rows)
                if not pdf_match:
                    replace_file(source_pdf, final_pdf, "Approved Figure 7 previous/no-QA GNDVI rerun PDF", replacement_rows)
                replacement_needed = "yes" if (not png_match or not pdf_match) else "no"
                action = "replaced from accepted DL GNDVI rerun source" if replacement_needed == "yes" else "left unchanged; already matched accepted source"
                evidence = f"before replacement: {evidence}"
            elif not png_match or not pdf_match:
                # Do not blindly replace non-Figure 7; record if mismatch. For 2/5/6, replacement is okay if source is accepted.
                if fig in {"Figure 2", "Figure 5", "Figure 6"}:
                    if not png_match:
                        replace_file(source_png, final_png, f"Approved {fig} previous/no-QA GNDVI source PNG", replacement_rows)
                    if not pdf_match:
                        replace_file(source_pdf, final_pdf, f"Approved {fig} previous/no-QA GNDVI source PDF", replacement_rows)
                    replacement_needed = "yes"
                    action = "replaced from accepted previous/no-QA source"
                else:
                    replacement_needed = "needs_attention"
                    action = "left unchanged; source mismatch requires manual review"
        else:
            if not final_png.exists() or not final_pdf.exists():
                replacement_needed = "needs_attention"
                action = "missing PNG or PDF"
        rows.append(
            {
                "figure_number": fig,
                "current_png": str(final_png),
                "current_pdf": str(final_pdf),
                "accepted_source_file_or_folder": source_folder if source_folder else "current final directory accepted unless missing",
                "accepted_decision": expectation,
                "matches_accepted_source_or_lineage": evidence,
                "replacement_needed": replacement_needed,
                "action_taken": action,
            }
        )
    return pd.DataFrame(rows)


def audit_supplementary() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    source_map = SUPP_DIR / "source_mapping" / "previous_GNDVI_with_test_supplementary_source_map.csv"
    rebuild_audit = SUPP_DIR / "tables" / "Supplementary_figure_rebuild_audit_previous_GNDVI_with_test.csv"
    source_text = source_map.read_text(encoding="utf-8", errors="ignore") if source_map.exists() else ""
    rebuild = pd.read_csv(rebuild_audit) if rebuild_audit.exists() else pd.DataFrame()
    for fig, stem in SUPP_STEMS.items():
        png = SUPP_DIR / "figures_png" / f"{stem}.png"
        pdf = SUPP_DIR / "figures_pdf" / f"{stem}.pdf"
        caption = SUPP_DIR / "captions" / f"{fig.replace(' ', '_')}_caption.md"
        audit_row = rebuild[rebuild["figure_id"] == fig] if not rebuild.empty and "figure_id" in rebuild.columns else pd.DataFrame()
        decision = SUPP_EXPECTATIONS[fig]
        if not audit_row.empty:
            src_decision = str(audit_row.iloc[0].get("rebuilt_or_copied", ""))
            non_gndvi = str(audit_row.iloc[0].get("non_GNDVI_changes_observed", ""))
            acceptable = str(audit_row.iloc[0].get("acceptable_for_manuscript", ""))
        else:
            src_decision = "missing audit row"
            non_gndvi = ""
            acceptable = ""
        replacement_needed = "no"
        action = "left unchanged"
        caveat = "caption present" if caption.exists() else "caption missing"
        if not png.exists() or not pdf.exists():
            replacement_needed = "needs_attention"
            action = "missing PNG or PDF"
        if fig == "Figure S6":
            blocked_hits = scan_text_for_patterns([caption, source_map, rebuild_audit], ["Figure S6 not rebuilt"])
            if blocked_hits:
                replacement_needed = "needs_attention"
                action = "S6 red flag detected; inspect manually"
        rows.append(
            {
                "figure_number": fig,
                "current_png": str(png),
                "current_pdf": str(pdf),
                "source_decision_expected": decision,
                "source_decision_observed": src_decision,
                "matches_accepted_source_package": "yes" if png.exists() and pdf.exists() and acceptable in {"yes", ""} else "needs_attention",
                "replacement_needed": replacement_needed,
                "action_taken": action,
                "caption_caveat_needed": caveat,
                "non_GNDVI_changes_observed": non_gndvi,
                "source_map_mentions_figure": "yes" if fig in source_text else "no",
            }
        )
    return pd.DataFrame(rows)


def scan_text_for_patterns(paths: list[Path], patterns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lower_patterns = [(p, p.lower()) for p in patterns]
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        lower = text.lower()
        for original, pat in lower_patterns:
            if pat in lower:
                for line_no, line in enumerate(text.splitlines(), start=1):
                    if pat in line.lower():
                        rows.append(
                            {
                                "file": str(path),
                                "line": str(line_no),
                                "pattern": original,
                                "text": line.strip()[:500],
                                "risk": risk_label(original),
                                "recommended_action": recommended_action(original),
                            }
                        )
    return rows


def risk_label(pattern: str) -> str:
    if pattern in {"QA-corrected GNDVI", "QA-controlled GNDVI", "strict-QA GNDVI"}:
        return "May conflict with final previous/no-QA GNDVI source."
    if pattern in {"standardized anomaly correlation", "anomaly correlation matrix"}:
        return "May conflict with accepted raw monthly Figure 5 definition."
    if pattern in {"deep learning dominated", "BiLSTM won 34/36", "six deep-learning models"}:
        return "May overclaim forecasting or misclassify XGBoost."
    if pattern in {"2024-only test"}:
        return "May conflict with rolling-origin validation."
    if pattern in {"5 km TWS", "fine-scale TWS"}:
        return "May overstate TWS spatial support."
    if pattern in {"regime shift", "causal transition"}:
        return "May overstate descriptive ecosystem-state analysis."
    if pattern in {"Figure S6 not rebuilt"}:
        return "May conflict with rebuilt post-GNDVI-rerun S6."
    return "Manual review."


def recommended_action(pattern: str) -> str:
    if pattern in {"QA-corrected GNDVI", "QA-controlled GNDVI", "strict-QA GNDVI"}:
        return "If referring to final figures, revise to previous/no-QA GNDVI; if historical audit text, leave as context."
    if pattern == "Figure S6 not rebuilt":
        return "Ensure this phrase is not in final captions/source maps; old package references can remain archived only."
    return "Inspect manually before manuscript submission."


def red_flag_scan() -> pd.DataFrame:
    paths = []
    for root in [MAIN_DIR.parent, SUPP_DIR, PREV_GNDVI_REBUILD, DL_GNDVI]:
        if root.exists():
            paths.extend([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".md", ".csv", ".txt"}])
    rows = scan_text_for_patterns(sorted(set(paths)), RISK_PATTERNS)
    return pd.DataFrame(rows)


def make_source_lineage_map(main_audit: pd.DataFrame, supp_audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in main_audit.itertuples(index=False):
        rows.append(
            {
                "figure": row.figure_number,
                "final_location": f"{row.current_png}; {row.current_pdf}",
                "accepted_source": row.accepted_source_file_or_folder,
                "decision": row.accepted_decision,
                "action": row.action_taken,
            }
        )
    for row in supp_audit.itertuples(index=False):
        rows.append(
            {
                "figure": row.figure_number,
                "final_location": f"{row.current_png}; {row.current_pdf}",
                "accepted_source": str(SUPP_DIR),
                "decision": row.source_decision_expected,
                "action": row.action_taken,
            }
        )
    return pd.DataFrame(rows)


def write_reports(
    backup: Path,
    replacement_log: pd.DataFrame,
    main_audit: pd.DataFrame,
    supp_audit: pd.DataFrame,
    red_flags: pd.DataFrame,
    dirs: dict[str, Path],
) -> None:
    replaced = replacement_log[replacement_log["action"] == "replaced"] if not replacement_log.empty else pd.DataFrame()
    main_needs = main_audit[main_audit["replacement_needed"].astype(str).str.contains("needs_attention", case=False, na=False)]
    supp_needs = supp_audit[supp_audit["replacement_needed"].astype(str).str.contains("needs_attention", case=False, na=False)]
    main_status = "PASS" if main_needs.empty else "NEEDS ATTENTION"
    supp_status = "PASS" if supp_needs.empty else "NEEDS ATTENTION"

    def md_table(df: pd.DataFrame, cols: list[str]) -> str:
        if df.empty:
            return "_None._"
        sub = df[cols].copy()
        return sub.to_markdown(index=False)

    (dirs["report"] / "FINAL_main_figure_directory_audit.md").write_text(
        "\n".join(
            [
                "# Final Main Figure Directory Audit",
                "",
                f"Backup created at: `{backup}`",
                "",
                f"Status: **{main_status}**",
                "",
                md_table(main_audit, ["figure_number", "accepted_decision", "replacement_needed", "action_taken", "matches_accepted_source_or_lineage"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dirs["report"] / "FINAL_supplementary_figure_directory_audit.md").write_text(
        "\n".join(
            [
                "# Final Supplementary Figure Directory Audit",
                "",
                f"Backup created at: `{backup}`",
                "",
                f"Status: **{supp_status}**",
                "",
                md_table(supp_audit, ["figure_number", "source_decision_expected", "source_decision_observed", "replacement_needed", "action_taken", "caption_caveat_needed"]),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dirs["report"] / "FINAL_replacement_log.md").write_text(
        "\n".join(
            [
                "# Final Replacement Log",
                "",
                f"Backup created before replacement: `{backup}`",
                "",
                md_table(replacement_log, ["file", "source", "action", "reason"] if not replacement_log.empty else []),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manual_phrases = [
        "previous/no-QA GNDVI, not strict-QA or QA-controlled GNDVI",
        "Figure 5 is raw/original monthly inter-index correlation, not standardized-anomaly correlation",
        "Figure 6 negative lag means predictor leads target; values use accepted previous/no-QA CCF source",
        "Figure 7 is reviewer-safe benchmark; only GNDVI target rows replaced; non-GNDVI targets unchanged",
        "SARIMA is monthly land-type aggregate only",
        "XGBoost is non-deep machine learning, not a baseline replacement",
        "Avoid universal deep-learning dominance claims",
        "EWBI is a forcing-side composite, not causal proof of regime shift",
        "Common grid is analytical overlay, not independent fine-scale TWS hydrologic information",
    ]
    (dirs["report"] / "FINAL_submission_readiness_report.md").write_text(
        "\n".join(
            [
                "# Final Submission Readiness Report",
                "",
                f"Main final figure directory status: **{main_status}**",
                f"Supplementary final figure directory status: **{supp_status}**",
                "",
                "## Files Replaced",
                "",
                md_table(replaced, ["file", "source", "reason"] if not replaced.empty else []),
                "",
                "## Files Left Unchanged",
                "",
                md_table(main_audit[main_audit["action_taken"].astype(str).str.contains("left unchanged", case=False, na=False)], ["figure_number", "action_taken", "accepted_decision"]),
                "",
                "## Supplementary Decisions",
                "",
                md_table(supp_audit, ["figure_number", "source_decision_expected", "source_decision_observed", "action_taken"]),
                "",
                "## Red-Flag Phrase Hits",
                "",
                md_table(red_flags, ["file", "line", "pattern", "risk", "recommended_action"] if not red_flags.empty else []),
                "",
                "## Manual Caption/Manuscript Phrases To Check",
                "",
                "\n".join([f"- {phrase}" for phrase in manual_phrases]),
                "",
                "## Final Note",
                "",
                "No scientific models or raster extraction workflows were rerun by this audit. Replacement was limited to approved source files after backing up the current final directories.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    dirs = ensure_dirs()
    # Pre-replacement inventories.
    main_inv = inventory(MAIN_DIR, recursive=False)
    supp_inv = inventory(SUPP_DIR, recursive=True)
    source_roots = [PREV_GNDVI_REBUILD, DL_GNDVI, SUPP_DIR]
    source_inv = pd.concat([inventory(root, recursive=True) for root in source_roots if root.exists()], ignore_index=True)
    write_csv(dirs["tables"] / "main_final_directory_inventory.csv", main_inv)
    write_csv(dirs["tables"] / "supplementary_final_directory_inventory.csv", supp_inv)
    write_csv(dirs["tables"] / "candidate_source_inventory.csv", source_inv)

    backup = make_backup()
    replacement_rows: list[dict[str, str]] = []
    main_audit = audit_main_and_replace(replacement_rows)
    supp_audit = audit_supplementary()
    replacement_log = pd.DataFrame(replacement_rows)
    if replacement_log.empty:
        replacement_log = pd.DataFrame(columns=["file", "source", "action", "reason", "before_sha256", "after_sha256"])

    # Post-replacement manifests.
    write_csv(dirs["tables"] / "replacement_log.csv", replacement_log)
    write_csv(dirs["tables"] / "final_main_figure_manifest_after_replacement.csv", inventory(MAIN_DIR, recursive=False))
    write_csv(dirs["tables"] / "final_supplementary_figure_manifest_after_replacement.csv", inventory(SUPP_DIR, recursive=True))
    write_csv(dirs["tables"] / "figure_source_lineage_map.csv", make_source_lineage_map(main_audit, supp_audit))
    red_flags = red_flag_scan()
    if red_flags.empty:
        red_flags = pd.DataFrame(columns=["file", "line", "pattern", "text", "risk", "recommended_action"])
    write_csv(dirs["tables"] / "remaining_risks_and_caption_checks.csv", red_flags)
    write_csv(dirs["tables"] / "main_figure_audit.csv", main_audit)
    write_csv(dirs["tables"] / "supplementary_figure_audit.csv", supp_audit)
    write_reports(backup, replacement_log, main_audit, supp_audit, red_flags, dirs)

    main_status = "PASS" if not (main_audit["replacement_needed"].astype(str).str.contains("needs_attention", case=False, na=False)).any() else "NEEDS ATTENTION"
    supp_status = "PASS" if not (supp_audit["replacement_needed"].astype(str).str.contains("needs_attention", case=False, na=False)).any() else "NEEDS ATTENTION"
    print(f"MAIN_STATUS={main_status}")
    print(f"SUPPLEMENTARY_STATUS={supp_status}")
    print(f"AUDIT_ROOT={AUDIT_ROOT}")
    print(f"REPLACED_FILES={len(replacement_log)}")
    for row in replacement_log.itertuples(index=False):
        print(f"REPLACED: {row.file} <- {row.source}")


if __name__ == "__main__":
    main()
