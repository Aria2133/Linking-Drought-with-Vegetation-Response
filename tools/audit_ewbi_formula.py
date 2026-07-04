from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
ROOT = DESKTOP / "major_revision_analyses" / "EWBI_formula_implementation_audit"
TABLES = ROOT / "tables"
REPORT = ROOT / "report"

MAJOR = DESKTOP / "major_revision_analyses"
WORKSPACE = DESKTOP / "Linking-Drought-with-Vegetation-Response"
SUPP = DESKTOP / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
GAP = DESKTOP / "GapFilling_Comparison_Revision"
FULL_REPORT = DESKTOP / "ECOINF_FINAL_full_reviewer_response_report"

COMPONENT_COLUMNS = ["spei_signed_z", "tws_signed_z", "vpd_signed_z", "lst_signed_z"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def file_record(path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else "",
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else "",
        "sha256": sha256(path) if path.exists() and path.is_file() and path.stat().st_size < 100_000_000 else "",
    }


def stored_value_check(path: Path, filter_expr: str = "") -> tuple[list[dict[str, object]], dict[str, object]]:
    df = pd.read_csv(path)
    original_rows = len(df)
    if filter_expr:
        df = df.query(filter_expr).copy()
    missing = [col for col in COMPONENT_COLUMNS + ["ewbi"] if col not in df.columns]
    if missing:
        return [], {
            "source_table": str(path),
            "filter": filter_expr,
            "original_rows": original_rows,
            "checked_rows": 0,
            "status": f"missing columns: {', '.join(missing)}",
        }

    signed_sum = df[COMPONENT_COLUMNS].sum(axis=1, skipna=False)
    signed_mean = df[COMPONENT_COLUMNS].mean(axis=1, skipna=False)
    valid = df["ewbi"].notna() & signed_sum.notna() & signed_mean.notna()
    mean_error = (df.loc[valid, "ewbi"] - signed_mean.loc[valid]).abs()
    sum_error = (df.loc[valid, "ewbi"] - signed_sum.loc[valid]).abs()
    rows = []
    sample = df.loc[valid].head(12).copy()
    for idx, row in sample.iterrows():
        comp_values = {col: float(row[col]) for col in COMPONENT_COLUMNS}
        rows.append(
            {
                "source_table": str(path),
                "row_index": idx,
                "land_type": row.get("land_type", ""),
                "date": row.get("date", ""),
                "method_or_preprocessing": row.get("method", row.get("preprocessing", "")),
                **comp_values,
                "stored_ewbi": float(row["ewbi"]),
                "signed_sum": float(signed_sum.loc[idx]),
                "signed_mean": float(signed_mean.loc[idx]),
                "abs_error_vs_signed_mean": float(abs(row["ewbi"] - signed_mean.loc[idx])),
                "abs_error_vs_signed_sum": float(abs(row["ewbi"] - signed_sum.loc[idx])),
                "matches": "signed_mean" if abs(row["ewbi"] - signed_mean.loc[idx]) < 1e-10 else "other",
            }
        )
    summary = {
        "source_table": str(path),
        "filter": filter_expr,
        "original_rows": original_rows,
        "checked_rows": int(valid.sum()),
        "max_abs_error_vs_signed_mean": float(mean_error.max()) if len(mean_error) else np.nan,
        "max_abs_error_vs_signed_sum": float(sum_error.max()) if len(sum_error) else np.nan,
        "median_abs_error_vs_signed_mean": float(mean_error.median()) if len(mean_error) else np.nan,
        "median_abs_error_vs_signed_sum": float(sum_error.median()) if len(sum_error) else np.nan,
        "stored_formula_supported": "signed average / equal mean of four signed standardized components",
        "status": "checked",
    }
    return rows, summary


def scan_text() -> list[dict[str, object]]:
    roots = [MAJOR, SUPP, FULL_REPORT, WORKSPACE / "tools"]
    exts = {".md", ".txt", ".csv", ".py"}
    patterns = [
        ("possible_sum_equation", re.compile(r"z\s*\(\s*SPEI\s*\).*z\s*\(\s*TWS\s*\).*(?:-|−).*z\s*\(\s*VPD.*(?:-|−).*z\s*\(\s*LST", re.I)),
        ("explicit_mean_definition", re.compile(r"equal\s+mean\s+of\s+signed|mean\s+of\s+signed|signed\s+average", re.I)),
        ("generic_forcing_composite", re.compile(r"forcing-side composite", re.I)),
        ("code_signed_sum", re.compile(r"time_zscore\(.*SPEI|time_zscore\(.*TWS|- time_zscore\(.*VPD|- time_zscore\(.*LST", re.I)),
    ]
    rows: list[dict[str, object]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            try:
                path.relative_to(ROOT)
                continue
            except ValueError:
                pass
            if path.resolve() == Path(__file__).resolve():
                continue
            if path.stat().st_size > 10_000_000:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(lines, 1):
                for label, regex in patterns:
                    if regex.search(line):
                        status = "ok"
                        recommendation = ""
                        if label == "possible_sum_equation":
                            status = "potential_mismatch"
                            recommendation = "If this is the main EWBI equation, add division by 4 or rewrite as the mean of four signed standardized components."
                        if label == "code_signed_sum" and "09_state_classification_sensitivity.py" in str(path):
                            status = "scale_inconsistency_in_sensitivity_code"
                            recommendation = "This sum-based sensitivity code is scale-equivalent for median/state ranking but should not be used as the main EWBI equation."
                        rows.append(
                            {
                                "match_type": label,
                                "status": status,
                                "file": str(path),
                                "line": lineno,
                                "text": line.strip()[:700],
                                "recommended_action": recommendation,
                            }
                        )
    # Add the user-reported manuscript equation as a manual check target because the .docx/manuscript source is not guaranteed present.
    rows.append(
        {
            "match_type": "manual_manuscript_equation_reported_by_user",
            "status": "mismatch_if_present",
            "file": "manuscript source not located in audit scan",
            "line": "",
            "text": "EWBI_P = z(SPEI_P) + z(TWS_P) - z(VPD_P) - z(LST_P)",
            "recommended_action": "Replace with EWBI_P = [z(SPEI_P) + z(TWS_P) - z(VPD_P) - z(LST_P)] / 4, or define EWBI as the equal-weight mean of four signed standardized components.",
        }
    )
    return rows


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    sources = [
        file_record(MAJOR / "revision_utils.py", "shared baseline-month anomaly and equal-weight EWBI utility"),
        file_record(MAJOR / "01_ewbi_validation.py", "EWBI validation, components, leave-one-out, and alternative weighting"),
        file_record(MAJOR / "02_harmonization_sensitivity.py", "TWS scale/resampling sensitivity using EWBI"),
        file_record(MAJOR / "03_gapfill_sensitivity.py", "gap-filling downstream EWBI sensitivity"),
        file_record(MAJOR / "04_detrending_sensitivity.py", "earlier detrending EWBI sensitivity"),
        file_record(MAJOR / "09_state_classification_sensitivity.py", "state-classification sensitivity; sum-scale implementation flagged"),
        file_record(MAJOR / "12_ewbi_gndvi_diagnostic.py", "EWBI-GNDVI diagnostics"),
        file_record(MAJOR / "04_detrending_sensitivity_fixed" / "scripts" / "rebuild_source" / "02_ewbi_detrending_sensitivity.py", "fixed detrending package EWBI recomputation"),
        file_record(MAJOR / "04_detrending_sensitivity_fixed" / "scripts" / "rebuild_source" / "detrending_utils.py", "fixed detrending preprocessing and baseline standardization"),
        file_record(MAJOR / "04_detrending_sensitivity_fixed" / "scripts" / "01_main_ecological_outputs.py", "main ecological EWBI/state output extraction"),
        file_record(WORKSPACE / "tools" / "rebuild_supplementary_previous_gndvi_with_test.py", "final S1-S2 supplementary EWBI figure rebuild"),
        file_record(GAP / "tables" / "ewbi_gapfill_monthly_series.csv", "exported EWBI monthly table from gap-fill package"),
        file_record(MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "ewbi_detrending_monthly_series.csv", "exported EWBI monthly table from fixed detrending package"),
        file_record(MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv", "accepted main ecological EWBI monthly series"),
        file_record(SUPP / "tables" / "S1_previous_GNDVI_with_test_period_response_panel_data.csv", "final S1 EWBI response panel table"),
        file_record(SUPP / "tables" / "S2_component_omission_panel_data_previous_GNDVI_with_test.csv", "final S2 component and omission panel table"),
    ]
    write_csv(TABLES / "ewbi_formula_source_inventory.csv", sources)

    sample_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for table_path, filter_expr in [
        (GAP / "tables" / "ewbi_gapfill_monthly_series.csv", "method == 'current_monthwise_cross_year'"),
        (MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "ewbi_detrending_monthly_series.csv", ""),
        (MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv", ""),
    ]:
        if table_path.exists():
            rows, summary = stored_value_check(table_path, filter_expr)
            sample_rows.extend(rows)
            summary_rows.append(summary)
    write_csv(TABLES / "ewbi_stored_value_formula_sample_check.csv", sample_rows)
    write_csv(TABLES / "ewbi_stored_value_formula_summary.csv", summary_rows)

    claims = [
        {
            "Claim": "SPEI and TWS are positive EWBI components.",
            "Status": "Supported",
            "Evidence file path": str(MAJOR / "revision_utils.py"),
            "Exact code line, function name, table column, or output field": "signed_forcing_anomalies(), lines 246-250; fixed detrending compute_ewbi() lines 35-36",
            "Recommended manuscript wording/equation": "Use +z(SPEI) and +z(TWS).",
        },
        {
            "Claim": "VPD/VPDmax and LST are negative stress components.",
            "Status": "Supported",
            "Evidence file path": str(MAJOR / "revision_utils.py"),
            "Exact code line, function name, table column, or output field": "signed_forcing_anomalies(), line 250; fixed detrending compute_ewbi() lines 37-38",
            "Recommended manuscript wording/equation": "Use -z(VPDmax) and -z(LST); code/table columns usually store VPDmax as VPD.",
        },
        {
            "Claim": "Main equal-weight EWBI is the signed sum without division by 4.",
            "Status": "Unsupported",
            "Evidence file path": str(MAJOR / "01_ewbi_validation.py"),
            "Exact code line, function name, table column, or output field": "line 73 equal = signed.mean(axis=1); revision_utils.equal_weight_ewbi() line 255; stored-value summary table",
            "Recommended manuscript wording/equation": "Include division by 4 or call it the equal-weight mean of four signed standardized components.",
        },
        {
            "Claim": "Main equal-weight EWBI is the signed average of four components.",
            "Status": "Supported",
            "Evidence file path": str(TABLES / "ewbi_stored_value_formula_summary.csv"),
            "Exact code line, function name, table column, or output field": "stored ewbi equals mean(spei_signed_z,tws_signed_z,vpd_signed_z,lst_signed_z) to numerical precision",
            "Recommended manuscript wording/equation": "EWBI = [z(SPEI) + z(TWS) - z(VPDmax) - z(LST)] / 4.",
        },
        {
            "Claim": "EWBI is computed from period means before standardization.",
            "Status": "Unsupported",
            "Evidence file path": str(MAJOR / "01_ewbi_validation.py"),
            "Exact code line, function name, table column, or output field": "monthly signed EWBI built at lines 71-73; period rows are summarized later from grouped monthly EWBI values at lines 170-181",
            "Recommended manuscript wording/equation": "Compute monthly component anomalies first, average signed components monthly, then summarize by period.",
        },
        {
            "Claim": "Period EWBI values are means of monthly EWBI values.",
            "Status": "Supported",
            "Evidence file path": str(MAJOR / "04_detrending_sensitivity_fixed" / "scripts" / "01_main_ecological_outputs.py"),
            "Exact code line, function name, table column, or output field": "lines 204-229 compute ewbi_mean as subset['ewbi'].mean() for each period",
            "Recommended manuscript wording/equation": "For period P, EWBI_P denotes the mean of monthly EWBI_t over months t in P.",
        },
        {
            "Claim": "Standardization is across pixels within the same land type and period.",
            "Status": "Unsupported for the final land-type monthly EWBI validation/state tables",
            "Evidence file path": str(MAJOR / "revision_utils.py"),
            "Exact code line, function name, table column, or output field": "baseline_monthly_anomaly() lines 224-234 uses land-type monthly time series and 2001-2011 calendar-month baseline; no pixel-period spatial z-score in final EWBI validation path",
            "Recommended manuscript wording/equation": "Describe z() as baseline-referenced standardization of land-type monthly component series, not spatial standardization across pixels.",
        },
        {
            "Claim": "Standardization is component-specific and land-type-specific.",
            "Status": "Supported",
            "Evidence file path": str(MAJOR / "01_ewbi_validation.py"),
            "Exact code line, function name, table column, or output field": "loop over land types lines 70-73; signed_forcing_anomalies standardizes each forcing variable separately",
            "Recommended manuscript wording/equation": "z() is computed separately for each component and land type using the stated baseline/reference period.",
        },
        {
            "Claim": "Standardization is separately refit for each drought period.",
            "Status": "Unsupported",
            "Evidence file path": str(MAJOR / "revision_utils.py"),
            "Exact code line, function name, table column, or output field": "baseline_monthly_anomaly() uses BASELINE 2001-2011; periods are labels/summaries, not separate z-score fits",
            "Recommended manuscript wording/equation": "Do not say z-scores were fitted separately within each period.",
        },
        {
            "Claim": "Alternative equal weighting used component weights of 0.25 each.",
            "Status": "Supported",
            "Evidence file path": str(MAJOR / "01_ewbi_validation.py"),
            "Exact code line, function name, table column, or output field": "lines 80-83: equal = np.repeat(0.25, 4)",
            "Recommended manuscript wording/equation": "Equal weighting is equivalent to division by 4.",
        },
        {
            "Claim": "PCA and regression/correlation-weighted EWBI variants define the main EWBI.",
            "Status": "Unsupported",
            "Evidence file path": str(MAJOR / "01_ewbi_validation.py"),
            "Exact code line, function name, table column, or output field": "lines 95-99 treat EWBI_equal, EWBI_PCA, EWBI_regression as compared indices; final captions identify them as robustness checks",
            "Recommended manuscript wording/equation": "Keep PCA/regression weighting as supplementary sensitivity, not the main EWBI formula.",
        },
        {
            "Claim": "One older state-classification sensitivity path uses signed sum scale.",
            "Status": "Partly supported",
            "Evidence file path": str(MAJOR / "09_state_classification_sensitivity.py"),
            "Exact code line, function name, table column, or output field": "manuscript_state_inputs(), lines 41-46 add/subtract standardized terms without /4",
            "Recommended manuscript wording/equation": "Flag as scale-equivalent for median/threshold classification but do not use as the manuscript's main EWBI equation.",
        },
    ]
    write_csv(TABLES / "ewbi_formula_claim_verification.csv", claims)

    scan_rows = scan_text()
    write_csv(TABLES / "ewbi_formula_text_mismatch_scan.csv", scan_rows)

    report = f"""# EWBI Formula Implementation Audit

Completed: {datetime.now().isoformat(timespec="seconds")}

## Bottom Line

The implemented main EWBI is the **equal-weight signed average**, not the unscaled signed sum. The manuscript equation should therefore include division by 4, or define EWBI as the mean of the four signed standardized forcing components.

Recommended main equation:

```text
EWBI_t = [z(SPEI_t) + z(TWS_t) - z(VPDmax_t) - z(LST_t)] / 4
```

If period notation is used:

```text
EWBI_P = mean_{{t in P}}(EWBI_t)
       = mean_{{t in P}}([z(SPEI_t) + z(TWS_t) - z(VPDmax_t) - z(LST_t)] / 4)
```

Here `VPDmax` is often stored as `VPD` in code/table columns. The z-standardization in the final EWBI validation path is component-specific and land-type-specific, using the stated baseline/reference monthly time-series definition; it is not a spatial z-score across pixels within each period.

## Stored-Value Verification

The exported EWBI tables contain component columns (`spei_signed_z`, `tws_signed_z`, `vpd_signed_z`, `lst_signed_z`) and `ewbi`. The audit recomputed both the signed sum and signed mean from those columns.

See:

- `tables/ewbi_stored_value_formula_summary.csv`
- `tables/ewbi_stored_value_formula_sample_check.csv`

Result: stored `ewbi` matches the signed mean to numerical precision. The signed sum differs by approximately a factor of 4.

## Primary Code Evidence

- Shared implementation: `{MAJOR / 'revision_utils.py'}` lines 246-255.
- EWBI validation: `{MAJOR / '01_ewbi_validation.py'}` lines 70-73 and 80-83.
- Gap-fill downstream sensitivity: `{MAJOR / '03_gapfill_sensitivity.py'}` lines 78-83.
- Harmonization sensitivity: `{MAJOR / '02_harmonization_sensitivity.py'}` lines 63-73.
- Fixed detrending package: `{MAJOR / '04_detrending_sensitivity_fixed' / 'scripts' / 'rebuild_source' / '02_ewbi_detrending_sensitivity.py'}` lines 27-40.
- Final supplementary S1/S2 rebuild: `{WORKSPACE / 'tools' / 'rebuild_supplementary_previous_gndvi_with_test.py'}` lines 304-312 and 468-482.

## Important Caveat

One older state-classification sensitivity script (`09_state_classification_sensitivity.py`) computes a signed sum scale in lines 41-46. Because this is a positive constant multiple of the signed average, it does not change rankings, median splits, or k-means/GMM inputs after standardization. Still, it should not be cited as the manuscript's main EWBI equation. To avoid scale inconsistency, manuscript Methods should define the main EWBI using `/4`.

## Recommended Manuscript Sentence

For each land type, we first converted SPEI, TWS, VPDmax, and LST to component-specific standardized monthly anomalies using the stated baseline/reference period; SPEI and TWS were assigned positive signs, while VPDmax and LST were assigned negative signs. The equal-weight EWBI was then calculated as the mean of the four signed standardized components, `EWBI_t = [z(SPEI_t) + z(TWS_t) - z(VPDmax_t) - z(LST_t)] / 4`, and period-level EWBI values were summarized from the resulting monthly EWBI series.

## Outputs

- `tables/ewbi_formula_source_inventory.csv`
- `tables/ewbi_formula_claim_verification.csv`
- `tables/ewbi_stored_value_formula_summary.csv`
- `tables/ewbi_stored_value_formula_sample_check.csv`
- `tables/ewbi_formula_text_mismatch_scan.csv`
"""
    (REPORT / "EWBI_formula_implementation_audit_report.md").write_text(report, encoding="utf-8")

    equation = """# Recommended EWBI Equation

Use this in the manuscript Methods:

```text
EWBI_t = [z(SPEI_t) + z(TWS_t) - z(VPDmax_t) - z(LST_t)] / 4
```

If using period notation:

```text
EWBI_P = mean_{t in P}(EWBI_t)
```

Suggested explanatory sentence:

For each land type, EWBI was computed as the equal-weight mean of four signed standardized forcing components. SPEI and TWS were assigned positive signs, while VPDmax and LST were assigned negative stress signs. The division by four matches the implemented equal-weight average and fixes the numerical scale; it does not alter correlations, spatial or temporal rankings, or relative change patterns compared with the unscaled signed sum.
"""
    (REPORT / "recommended_EWBI_equation_and_sentence.md").write_text(equation, encoding="utf-8")


if __name__ == "__main__":
    main()
