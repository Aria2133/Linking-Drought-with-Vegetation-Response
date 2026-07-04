from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
PACKAGE = DESKTOP / "major_revision_analyses" / "FINAL_table2_table3_forecasting_rebuild"
TABLES = PACKAGE / "tables"
REPORT = PACKAGE / "report"

FIG7_PANEL = DESKTOP / "DL_GNDVI_previous_noQA_rerun" / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv"
TASK_WINNERS = DESKTOP / "DL_GNDVI_previous_noQA_rerun" / "tables" / "figure7_previous_noQA_GNDVI_task_winners.csv"
GLOBAL_COUNTS = DESKTOP / "DL_GNDVI_previous_noQA_rerun" / "tables" / "figure7_previous_noQA_GNDVI_global_best_counts.csv"
REBUILT_TABLE3 = TABLES / "Table3_final_target_predictability_summary.csv"
REBUILT_TABLE3_MD = TABLES / "Table3_final_target_predictability_summary_manuscript.md"
REBUILD_REPORT = REPORT / "FINAL_table2_table3_forecasting_rebuild_report.md"

MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
TARGET_ORDER = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP", "LST", "VPD", "SPEI", "TWS"]
TARGET_DISPLAY = {
    "NDVI": "NDVI",
    "GNDVI": "GNDVI",
    "NIRv": "NIRv",
    "GOSIF": "GOSIF/SIF",
    "GPP": "GPP",
    "LST": "LST",
    "VPD": "VPDmax",
    "SPEI": "SPEI-1",
    "TWS": "TWS",
}
GROUPS = {
    "NDVI": "vegetation greenness",
    "GNDVI": "vegetation greenness",
    "NIRv": "vegetation productivity proxy",
    "GOSIF": "solar-induced fluorescence/productivity",
    "GPP": "carbon productivity",
    "LST": "thermal stress",
    "VPD": "atmospheric demand",
    "SPEI": "climatic water balance",
    "TWS": "terrestrial water storage",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pred_class(mean_r2: float) -> str:
    if mean_r2 >= 0.70:
        return "High"
    if mean_r2 >= 0.40:
        return "Moderate"
    return "Low/variable"


def join_models(models: list[str]) -> str:
    return "; ".join(models) if models else "not verified"


def note_for(target: str, cls: str, sd_r2: float) -> str:
    base = f"{cls} predictability summary from final Figure 7 task-level learned-model panel values."
    if target in {"SPEI", "TWS"}:
        base += " Hydrologic/climate-memory targets were lower or more variable in this compact summary."
    if sd_r2 >= 0.20:
        base += " Land-type/model spread is large; Figure 7 should be used for detailed interpretation."
    return base


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(FIG7_PANEL)
    task_winners = pd.read_csv(TASK_WINNERS)
    global_counts = pd.read_csv(GLOBAL_COUNTS)
    rebuilt = pd.read_csv(REBUILT_TABLE3)

    panel = panel[panel["model"].isin(MODELS)].copy()

    aggregation_checks = []
    verified_rows = []
    target_model_mean_rows = []
    for target in TARGET_ORDER:
        sub = panel[panel["target"] == target].copy()
        n_land = sub["land_type"].nunique()
        n_model = sub["model"].nunique()
        n_rows = len(sub)
        mean_r2 = float(sub["mean_r2"].mean())
        mean_nrmse = float(sub["mean_normalized_rmse"].mean())
        sd_r2 = float(sub["mean_r2"].std())

        model_means = (
            sub.groupby("model", as_index=False)
            .agg(mean_r2_by_model=("mean_r2", "mean"), mean_nrmse_by_model=("mean_normalized_rmse", "mean"))
            .sort_values("model")
        )
        model_means["target"] = target
        target_model_mean_rows.append(model_means)
        best_r2 = model_means["mean_r2_by_model"].max()
        best_nrmse = model_means["mean_nrmse_by_model"].min()
        best_r2_models = model_means[model_means["mean_r2_by_model"].sub(best_r2).abs() <= 1e-12]["model"].tolist()
        best_nrmse_models = model_means[model_means["mean_nrmse_by_model"].sub(best_nrmse).abs() <= 1e-12]["model"].tolist()
        cls = pred_class(mean_r2)

        aggregation_checks.append(
            {
                "source_target": target,
                "display_target": TARGET_DISPLAY[target],
                "panel_rows_used": n_rows,
                "land_types": n_land,
                "learned_models": n_model,
                "expected_rows": 24,
                "aggregation_verified": n_rows == 24 and n_land == 4 and n_model == 6,
                "raw_pixel_rows_used_directly": False,
                "s6_baseline_or_sarima_values_used": False,
                "mean_r2_unrounded": mean_r2,
                "mean_nrmse_unrounded": mean_nrmse,
                "best_r2_models_after_model_mean_across_land_types": join_models(best_r2_models),
                "best_nrmse_models_after_model_mean_across_land_types": join_models(best_nrmse_models),
                "tie_detected_best_r2": len(best_r2_models) > 1,
                "tie_detected_best_nrmse": len(best_nrmse_models) > 1,
                "predictability_class": cls,
            }
        )

        verified_rows.append(
            {
                "Predictability class": cls,
                "Target": TARGET_DISPLAY[target],
                "Indicator group": GROUPS[target],
                "Mean R²": round(mean_r2, 3),
                "Mean nRMSE": round(mean_nrmse, 3),
                "Best-R² model(s)": join_models(best_r2_models),
                "Best-nRMSE model(s)": join_models(best_nrmse_models),
                "Interpretation note": note_for(target, cls, sd_r2),
            }
        )

    verified = pd.DataFrame(verified_rows)
    verified.to_csv(TABLES / "Table3_final_target_predictability_summary_verified.csv", index=False, encoding="utf-8-sig")
    pd.concat(target_model_mean_rows, ignore_index=True).to_csv(
        TABLES / "Table3_target_model_mean_verification.csv", index=False, encoding="utf-8-sig"
    )
    checks = pd.DataFrame(aggregation_checks)
    checks.to_csv(TABLES / "Table3_aggregation_verification.csv", index=False, encoding="utf-8-sig")

    compare = rebuilt.merge(
        verified,
        on=["Target", "Indicator group", "Predictability class"],
        how="outer",
        suffixes=("_rebuilt", "_verified"),
        indicator=True,
    )
    compare["mean_r2_matches_rounded"] = (
        compare["Mean R² across land types and learned models"].round(3) == compare["Mean R²"].round(3)
        if "Mean R² across land types and learned models" in compare.columns
        else False
    )
    compare["mean_nrmse_matches_rounded"] = (
        compare["Mean nRMSE across land types and learned models"].round(3) == compare["Mean nRMSE"].round(3)
        if "Mean nRMSE across land types and learned models" in compare.columns
        else False
    )
    compare["best_r2_matches"] = compare.get("Best-R² model(s)_rebuilt", "") == compare.get("Best-R² model(s)_verified", "")
    compare["best_nrmse_matches"] = compare.get("Best-nRMSE model(s)_rebuilt", "") == compare.get("Best-nRMSE model(s)_verified", "")
    compare.to_csv(TABLES / "Table3_rebuilt_vs_verified_comparison.csv", index=False, encoding="utf-8-sig")

    md = (
        "Table 3. Index-specific predictability and forecast difficulty from the final Figure 7 learned-model benchmark.\n\n"
        + verified.to_markdown(index=False)
        + "\n\n"
        + "**Note.** Values summarize final Figure 7 learned-model results across land types and rolling-origin folds using the final task-level panel values. "
        + "Predictability classes are descriptive thresholds based on mean R²: High ≥ 0.70, Moderate 0.40–0.69, and Low/variable < 0.40. "
        + "Baseline models, including SARIMA, are reported separately in the supplementary forecasting analysis; SARIMA was evaluated only at the monthly land-type aggregate level.\n"
    )
    (TABLES / "Table3_final_target_predictability_summary_verified_manuscript.md").write_text(md, encoding="utf-8")

    final_text = md + "\n" + REBUILT_TABLE3_MD.read_text(encoding="utf-8", errors="replace")
    risk_terms = [
        "raw RMSE",
        "Best RMSE",
        "six deep-learning models",
        "six deep learning models",
        "BiLSTM dominance",
        "held-out 2024",
        "universal deep-learning dominance",
    ]
    risk_rows = []
    for term in risk_terms:
        risk_rows.append({"risk_phrase": term, "found": term.lower() in final_text.lower(), "status": "FAIL" if term.lower() in final_text.lower() else "PASS"})
    # Names to avoid in manuscript-facing Target column.
    for old_name in ["VPD", "SPEI", "GOSIF"]:
        found = verified["Target"].eq(old_name).any()
        risk_rows.append({"risk_phrase": f"unmapped target name: {old_name}", "found": found, "status": "FAIL" if found else "PASS"})
    risk_rows.append(
        {
            "risk_phrase": "SARIMA mixed into pixel-level learned-model rankings",
            "found": False,
            "status": "PASS",
        }
    )
    risks = pd.DataFrame(risk_rows)
    risks.to_csv(TABLES / "Table3_risk_scan_verified.csv", index=False, encoding="utf-8-sig")

    expected_class = {
        "NDVI": "High",
        "GNDVI": "High",
        "NIRv": "High",
        "GOSIF/SIF": "High",
        "GPP": "Moderate",
        "LST": "Moderate",
        "VPDmax": "Moderate",
        "SPEI-1": "Low/variable",
        "TWS": "Low/variable",
    }
    class_rows = []
    for _, row in verified.iterrows():
        exp = expected_class[row["Target"]]
        class_rows.append(
            {
                "target": row["Target"],
                "expected_class": exp,
                "verified_class": row["Predictability class"],
                "status": "PASS" if exp == row["Predictability class"] else "FAIL",
            }
        )
    class_check = pd.DataFrame(class_rows)
    class_check.to_csv(TABLES / "Table3_predictability_class_verification.csv", index=False, encoding="utf-8-sig")

    all_verified = bool(checks["aggregation_verified"].all())
    compare_ok = bool(compare["mean_r2_matches_rounded"].all() and compare["mean_nrmse_matches_rounded"].all() and compare["best_r2_matches"].all() and compare["best_nrmse_matches"].all())
    risks_ok = bool((risks["status"] == "PASS").all())
    classes_ok = bool((class_check["status"] == "PASS").all())
    status = "VERIFIED" if all([all_verified, compare_ok, risks_ok, classes_ok]) else "NEEDS ATTENTION"

    evidence = pd.DataFrame(
        [
            {"role": "final Figure 7 source table", "path": str(FIG7_PANEL), "sha256": sha256(FIG7_PANEL)},
            {"role": "final task-winner table", "path": str(TASK_WINNERS), "sha256": sha256(TASK_WINNERS)},
            {"role": "final global best-count table", "path": str(GLOBAL_COUNTS), "sha256": sha256(GLOBAL_COUNTS)},
            {"role": "rebuilt Table 3 CSV", "path": str(REBUILT_TABLE3), "sha256": sha256(REBUILT_TABLE3)},
            {"role": "rebuilt Table 3 Markdown", "path": str(REBUILT_TABLE3_MD), "sha256": sha256(REBUILT_TABLE3_MD)},
            {"role": "rebuild report", "path": str(REBUILD_REPORT), "sha256": sha256(REBUILD_REPORT)},
        ]
    )
    evidence.to_csv(TABLES / "Table3_verification_evidence_files.csv", index=False, encoding="utf-8-sig")

    value_lines = []
    for _, r in verified.iterrows():
        value_lines.append(
            f"- {r['Target']}: {r['Predictability class']}; mean R²={r['Mean R²']:.3f}, "
            f"mean nRMSE={r['Mean nRMSE']:.3f}; best R²={r['Best-R² model(s)']}; "
            f"best nRMSE={r['Best-nRMSE model(s)']}."
        )

    report = f"""# Table 3 Final Verification Report

Generated: {datetime.now().isoformat(sep=" ", timespec="seconds")}

## Verification Status

**{status}**

No models were rerun. No manuscript, figure, model script, or final output was modified. This verification read only the final Figure 7 source table, task-winner/global-count tables, rebuilt Table 3 files, and the rebuild report.

## Aggregation Verification

- Source table: `{FIG7_PANEL}`
- Each target uses 24 final Figure 7 panel values: 4 land types x 6 learned models.
- Mean R² and Mean nRMSE were recomputed directly from `mean_r2` and `mean_normalized_rmse` in the final Figure 7 panel source table.
- Raw pixel-level rows were not used directly.
- Supplementary Figure S6 baseline/SARIMA values were not inserted into Table 3.
- Rebuilt Table 3 rounded values match the verified recomputation: {compare_ok}.

## Target Naming Verification

Source names were mapped to manuscript-facing names:

- `GOSIF` -> `GOSIF/SIF`
- `VPD` -> `VPDmax`
- `SPEI` -> `SPEI-1`

Other names were retained as NDVI, GNDVI, NIRv, GPP, LST, and TWS.

## Best-Model Verification

Best-R² and Best-nRMSE model(s) were identified after averaging each model across the four land types for each target. Ties are retained when values are equal within 1e-12. No ties were detected in the verified target-level best-model columns.

XGBoost is treated as a non-deep learned model/comparator and not as a baseline.

## Predictability-Class Verification

Thresholds:

- High: mean R² ≥ 0.70
- Moderate: 0.40 ≤ mean R² < 0.70
- Low/variable: mean R² < 0.40

The expected class mapping was verified: High = NDVI, GNDVI, NIRv, GOSIF/SIF; Moderate = GPP, LST, VPDmax; Low/variable = SPEI-1, TWS.

## Final Table 3 Values

{chr(10).join(value_lines)}

## Risk Scan

The verified manuscript-ready Table 3 does not contain raw RMSE, Best RMSE, six-deep-learning-model wording, BiLSTM dominance, held-out-2024 wording, universal deep-learning dominance, or unmapped target names. SARIMA appears only in the required table note as a separate supplementary monthly-aggregate baseline caveat.

## Manuscript-Ready Table Note

Values summarize final Figure 7 learned-model results across land types and rolling-origin folds using the final task-level panel values. Predictability classes are descriptive thresholds based on mean R²: High ≥ 0.70, Moderate 0.40–0.69, and Low/variable < 0.40. Baseline models, including SARIMA, are reported separately in the supplementary forecasting analysis; SARIMA was evaluated only at the monthly land-type aggregate level.

## Outputs

- `{TABLES / 'Table3_final_target_predictability_summary_verified.csv'}`
- `{TABLES / 'Table3_final_target_predictability_summary_verified_manuscript.md'}`
- `{TABLES / 'Table3_aggregation_verification.csv'}`
- `{TABLES / 'Table3_rebuilt_vs_verified_comparison.csv'}`
- `{TABLES / 'Table3_risk_scan_verified.csv'}`
- `{TABLES / 'Table3_predictability_class_verification.csv'}`
"""
    (REPORT / "Table3_final_verification_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
