from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path

import pandas as pd


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
REPO = DESKTOP / "Linking-Drought-with-Vegetation-Response"
OUT = DESKTOP / "major_revision_analyses" / "FINAL_table2_table3_forecasting_rebuild"
TABLES = OUT / "tables"
REPORT = OUT / "report"

FORECAST_AUDIT = DESKTOP / "major_revision_analyses" / "FINAL_forecasting_framework_technical_audit"
SECTION4_AUDIT = DESKTOP / "major_revision_analyses" / "FINAL_section4_results_technical_audit"
DL_GNDVI = DESKTOP / "DL_GNDVI_previous_noQA_rerun"
S6_PACKAGE = DESKTOP / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
FIG7_RENDERER = REPO / "tools" / "build_previous_gndvi_figure7.py"

FIG7_PANEL = DL_GNDVI / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv"
FIG7_COUNTS = DL_GNDVI / "tables" / "figure7_previous_noQA_GNDVI_global_best_counts.csv"
FIG7_WINNERS = DL_GNDVI / "tables" / "figure7_previous_noQA_GNDVI_task_winners.csv"
S6_MONTHLY = S6_PACKAGE / "tables" / "S6_combined_monthly_metrics_previous_GNDVI_with_test.csv"
S6_PIXEL = S6_PACKAGE / "tables" / "S6_combined_pixel_metrics_previous_GNDVI_with_test.csv"

MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
MODEL_FAMILY = {
    "LSTM": "deep learning",
    "BiLSTM": "deep learning",
    "TCN": "deep learning",
    "N-BEATS": "deep learning",
    "N-HiTS": "deep learning",
    "XGBoost": "non-deep machine learning",
}
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
TARGET_GROUP = {
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


def ensure_dirs() -> None:
    for d in [OUT, TABLES, REPORT]:
        d.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_row(path: Path, role: str) -> dict:
    row = {"role": role, "path": str(path), "exists": path.exists(), "sha256": "", "size_bytes": "", "modified_time": ""}
    if path.exists():
        st = path.stat()
        row.update(
            {
                "sha256": sha256(path),
                "size_bytes": st.st_size,
                "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(sep=" ", timespec="seconds"),
            }
        )
    return row


def read_line_refs(path: Path, needles: list[str]) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for i, line in enumerate(lines, 1):
        for n in needles:
            if n in line:
                rows.append({"file": str(path), "line": i, "needle": n, "text": line.strip()})
    return rows


def fmt(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


def model_note(model: str, best_r2: float, best_nrmse: float) -> str:
    if model == "XGBoost":
        return "Largest final best-count totals; non-deep ML comparator, not a baseline."
    if best_r2 == 0 and best_nrmse == 0:
        return "No task-level wins in the final Figure 7 panel aggregation."
    return "Learned model; performance is task- and metric-dependent."


def predictability_class(mean_r2: float) -> str:
    if mean_r2 >= 0.70:
        return "High"
    if mean_r2 >= 0.40:
        return "Moderate"
    return "Low/variable"


def join_models(models: list[str]) -> str:
    return "; ".join(models) if models else "not verified"


def main() -> None:
    ensure_dirs()
    panel = pd.read_csv(FIG7_PANEL)
    counts = pd.read_csv(FIG7_COUNTS)
    winners = pd.read_csv(FIG7_WINNERS)

    panel = panel[panel["model"].isin(MODELS)].copy()
    if len(panel) != 216:
        raise RuntimeError(f"Expected 216 Figure 7 panel rows, found {len(panel)}")

    # Table 2: counts from final global count table; means across the same 36 task-level panel values per model.
    mean_by_model = (
        panel.groupby("model", as_index=False)
        .agg(
            mean_r2=("mean_r2", "mean"),
            mean_nrmse=("mean_normalized_rmse", "mean"),
            task_model_rows=("mean_r2", "size"),
        )
    )
    table2 = counts.merge(mean_by_model, on="model", how="left")
    table2["Model"] = table2["model"]
    table2["Model family"] = table2["model"].map(MODEL_FAMILY)
    table2["Best R² count"] = table2["best_r2_count"]
    table2["Best nRMSE count"] = table2["best_normalized_rmse_count"]
    table2["Mean R²"] = table2["mean_r2"].round(3)
    table2["Mean nRMSE"] = table2["mean_nrmse"].round(3)
    table2["Interpretation note"] = table2.apply(lambda r: model_note(r["model"], r["best_r2_count"], r["best_normalized_rmse_count"]), axis=1)
    table2_out = table2[["Model", "Model family", "Best R² count", "Best nRMSE count", "Mean R²", "Mean nRMSE", "Interpretation note"]]
    table2_out.to_csv(TABLES / "Table2_final_model_performance_rankings.csv", index=False, encoding="utf-8-sig")

    # Target-level summaries use the same final Figure 7 task-level panel values.
    target_mean = (
        panel.groupby("target", as_index=False)
        .agg(
            mean_r2=("mean_r2", "mean"),
            mean_nrmse=("mean_normalized_rmse", "mean"),
            sd_r2_across_land_model=("mean_r2", "std"),
            sd_nrmse_across_land_model=("mean_normalized_rmse", "std"),
            n_land_model_rows=("mean_r2", "size"),
        )
    )
    target_model_mean = (
        panel.groupby(["target", "model"], as_index=False)
        .agg(mean_r2_by_model=("mean_r2", "mean"), mean_nrmse_by_model=("mean_normalized_rmse", "mean"))
    )
    rows = []
    for _, r in target_mean.iterrows():
        target = r["target"]
        sub = target_model_mean[target_model_mean["target"] == target]
        max_r2 = sub["mean_r2_by_model"].max()
        min_nrmse = sub["mean_nrmse_by_model"].min()
        best_r2_models = sub[sub["mean_r2_by_model"].sub(max_r2).abs() <= 1e-12]["model"].tolist()
        best_nrmse_models = sub[sub["mean_nrmse_by_model"].sub(min_nrmse).abs() <= 1e-12]["model"].tolist()
        cls = predictability_class(float(r["mean_r2"]))
        note = "Summarizes Figure 7 panel values across four land types and six learned models."
        if float(r["sd_r2_across_land_model"]) >= 0.20:
            note += " High spread across land/model combinations; use Figure 7 for detail."
        rows.append(
            {
                "Target": TARGET_DISPLAY.get(target, target),
                "source_target": target,
                "Indicator group": TARGET_GROUP.get(target, "not verified"),
                "Mean R² across land types and learned models": round(float(r["mean_r2"]), 3),
                "Mean nRMSE across land types and learned models": round(float(r["mean_nrmse"]), 3),
                "Best-R² model(s)": join_models(best_r2_models),
                "Best-nRMSE model(s)": join_models(best_nrmse_models),
                "Predictability class": cls,
                "Notes": note,
                "n_land_model_rows": int(r["n_land_model_rows"]),
                "sd_R2_across_land_model": round(float(r["sd_r2_across_land_model"]), 3),
            }
        )
    table3 = pd.DataFrame(rows)
    # Keep manuscript-facing order.
    order = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP", "LST", "VPD", "SPEI", "TWS"]
    table3["order"] = table3["source_target"].map({v: i for i, v in enumerate(order)})
    table3 = table3.sort_values("order")
    table3_out = table3[
        [
            "Target",
            "Indicator group",
            "Mean R² across land types and learned models",
            "Mean nRMSE across land types and learned models",
            "Best-R² model(s)",
            "Best-nRMSE model(s)",
            "Predictability class",
            "Notes",
        ]
    ]
    table3_out.to_csv(TABLES / "Table3_final_target_predictability_summary.csv", index=False, encoding="utf-8-sig")

    # Markdown manuscript-ready versions.
    table2_md = (
        "Table 2. Model performance rankings across the final Figure 7 learned-model benchmark.\n\n"
        + table2_out.to_markdown(index=False)
        + "\n\n"
        + "**Note.** Counts indicate the number of land-type x target tasks, out of 36, for which each learned model achieved the highest R² or lowest nRMSE in the final pixel-level learned-model benchmark. "
        + "Models equal within 1e-12 share one task credit equally. XGBoost is a non-deep machine-learning comparator. nRMSE denotes RMSE normalized by the population standard deviation of observed values. Mean values are averages of the final Figure 7 task-level panel means across 36 land-type x target tasks for each model.\n"
    )
    (TABLES / "Table2_final_model_performance_rankings_manuscript.md").write_text(table2_md, encoding="utf-8")

    table3_md = (
        "Table 3. Index-specific predictability summary from the final Figure 7 learned-model benchmark.\n\n"
        + table3_out.to_markdown(index=False)
        + "\n\n"
        + "**Note.** Values summarize final Figure 7 learned-model results across land types and rolling-origin folds, using the final task-level panel values. "
        + "Predictability classes are descriptive thresholds based on mean R²: High ≥ 0.70, Moderate 0.40-0.69, and Low/variable < 0.40. "
        + "Baseline models, including SARIMA, are reported separately in the supplementary forecasting analysis; SARIMA was evaluated only at the monthly land-type aggregate level.\n"
    )
    (TABLES / "Table3_final_target_predictability_summary_manuscript.md").write_text(table3_md, encoding="utf-8")

    forbidden = [
        "six deep-learning models",
        "six deep learning models",
        "Best RMSE Count",
        "Mean RMSE",
        "held-out 2024-only",
        "held-out 2024",
        "BiLSTM 34/36",
        "34/36",
        "94%",
        "TCN 36%",
        "deep-learning dominance",
        "SARIMA",
    ]
    verification_rows = []
    combined_text = table2_md + "\n\n" + table3_md
    for phrase in forbidden:
        hit = phrase.lower() in combined_text.lower()
        allowed_note = ""
        if phrase == "SARIMA" and hit:
            allowed_note = "Allowed only in Table 3 note as supplementary monthly-aggregate baseline caveat."
        if allowed_note:
            note = allowed_note
        elif hit:
            note = "Forbidden phrase present; inspect manually."
        else:
            note = "Forbidden phrase absent."
        verification_rows.append(
            {
                "claim_or_phrase": phrase,
                "found_in_final_tables": hit,
                "status": "allowed caveat" if allowed_note else ("FAIL" if hit else "PASS"),
                "note": note,
            }
        )
    verification = pd.DataFrame(verification_rows)
    verification.to_csv(TABLES / "table2_table3_claim_verification.csv", index=False, encoding="utf-8-sig")

    evidence_files = pd.DataFrame(
        [
            source_row(FORECAST_AUDIT / "report" / "FINAL_forecasting_framework_technical_audit.md", "forecasting framework audit report"),
            source_row(FORECAST_AUDIT / "tables" / "final_best_model_counts.csv", "final best model counts"),
            source_row(FORECAST_AUDIT / "tables" / "forecasting_claim_verification.csv", "forecasting claim verification"),
            source_row(FORECAST_AUDIT / "tables" / "rolling_origin_fold_design.csv", "rolling-origin fold design"),
            source_row(FORECAST_AUDIT / "tables" / "baseline_level_summary.csv", "baseline level summary"),
            source_row(FORECAST_AUDIT / "tables" / "reviewer_risk_audit.csv", "reviewer risk audit"),
            source_row(SECTION4_AUDIT / "report" / "FINAL_section4_results_technical_audit.md", "Section 4 technical audit"),
            source_row(SECTION4_AUDIT / "tables" / "figure7_final_best_model_counts.csv", "Section 4 Figure 7 best counts mirror"),
            source_row(SECTION4_AUDIT / "tables" / "claim_by_claim_verification.csv", "Section 4 claim-by-claim verification"),
            source_row(SECTION4_AUDIT / "tables" / "section4_table_recommendations.csv", "Section 4 table recommendations"),
            source_row(FIG7_PANEL, "final Figure 7 panel values"),
            source_row(FIG7_COUNTS, "final Figure 7 global best counts"),
            source_row(FIG7_WINNERS, "final Figure 7 task winners"),
            source_row(FIG7_RENDERER, "final Figure 7 renderer"),
            source_row(S6_MONTHLY, "Supplementary Figure S6 monthly metrics"),
            source_row(S6_PIXEL, "Supplementary Figure S6 pixel metrics"),
            source_row(S6_PACKAGE / "captions" / "Figure_S6_caption.md", "Supplementary Figure S6 caption"),
        ]
    )
    evidence_files.to_csv(TABLES / "table2_table3_evidence_sources.csv", index=False, encoding="utf-8-sig")

    line_refs = pd.DataFrame(
        read_line_refs(
            FIG7_RENDERER,
            [
                'MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]',
                "mean_normalized_rmse",
                "mean_r2",
                "best_normalized_rmse_count",
                "best_r2_count",
                "tie_rule",
                "combined.to_csv",
            ],
        )
    )
    line_refs.to_csv(TABLES / "figure7_renderer_line_evidence.csv", index=False, encoding="utf-8-sig")

    # Report.
    xgb = table2_out[table2_out["Model"] == "XGBoost"].iloc[0]
    report = f"""# Final Table 2 / Table 3 Forecasting Rebuild Report

Generated: {datetime.now().isoformat(sep=" ", timespec="seconds")}

## Evidence Sources Used

Primary final Figure 7 source table:

`{FIG7_PANEL}`

This table has {len(panel)} rows, representing 4 land types x 9 targets x 6 learned models. It contains `mean_r2` and `mean_normalized_rmse`, already averaged over available reviewer-safe rolling-origin fold/seed rows for each land-type x target x model combination.

Final count table:

`{FIG7_COUNTS}`

Final task-winner table:

`{FIG7_WINNERS}`

Supplementary Figure S6 sources were inspected for baseline/aggregate-level provenance:

- `{S6_MONTHLY}`
- `{S6_PIXEL}`
- `{S6_PACKAGE / 'captions' / 'Figure_S6_caption.md'}`

S6 values were not inserted into Tables 2 or 3 because the requested replacement tables are Figure 7 learned-model summaries. S6 should be cited separately for baselines, monthly aggregates, and SARIMA caveats.

## Table 2 Computation

Best R² Count and Best nRMSE Count were read from the final Figure 7 global best-count table, which was produced by the final Figure 7 renderer after replacing only GNDVI target rows with the confirmed previous/no-QA GNDVI rerun.

Tie rule: models equal within 1e-12 share one task credit equally.

Mean R² and Mean nRMSE were computed from `{FIG7_PANEL.name}` by averaging each model's 36 task-level panel means. This matches the final Figure 7 reviewer-facing aggregation and avoids raw pixel-level reaggregation.

XGBoost has the largest final best-count totals: Best R² Count = {xgb['Best R² count']}, Best nRMSE Count = {xgb['Best nRMSE count']}. This should be reported carefully as a best-count result, not as universal model-family dominance.

## Table 3 Computation

For each target, Mean R² and Mean nRMSE were computed as averages over the 24 final Figure 7 panel values for that target: 4 land types x 6 learned models. Best-R² and best-nRMSE models were identified after first averaging each model across the four land types for that target.

Predictability classes are descriptive only:

- High: mean R² >= 0.70
- Moderate: 0.40 <= mean R² < 0.70
- Low/variable: mean R² < 0.40

These classes should be treated as a compact manuscript summary; Figure 7 remains the source for land-type-specific detail.

## Reviewer-Risk Check

The rebuilt tables use:

- five deep-learning models plus XGBoost
- Best nRMSE Count, not Best RMSE Count
- Mean nRMSE, not Mean RMSE
- final 36 land-type x target task summaries
- no held-out-2024-only wording
- no BiLSTM 34/36 or 94% claim
- no TCN 36% claim
- no universal deep-learning dominance claim
- no SARIMA mixed into pixel-level Figure 7 model rankings

`table2_table3_claim_verification.csv` records the phrase scan. SARIMA appears only in the Table 3 note as a supplementary monthly-aggregate baseline caveat.

## Manuscript-Ready Table Notes

Table 2 note:

Counts indicate the number of land-type x target tasks, out of 36, for which each learned model achieved the highest R² or lowest nRMSE in the final pixel-level learned-model benchmark. Models equal within 1e-12 share one task credit equally. XGBoost is a non-deep machine-learning comparator. nRMSE denotes RMSE normalized by the population standard deviation of observed values. Mean values are averages of the final Figure 7 task-level panel means across 36 land-type x target tasks for each model.

Table 3 note:

Values summarize final Figure 7 learned-model results across land types and rolling-origin folds, using the final task-level panel values. Predictability classes are descriptive thresholds based on mean R²: High >= 0.70, Moderate 0.40-0.69, and Low/variable < 0.40. Baseline models, including SARIMA, are reported separately in the supplementary forecasting analysis; SARIMA was evaluated only at the monthly land-type aggregate level.

## Outputs

- `{TABLES / 'Table2_final_model_performance_rankings.csv'}`
- `{TABLES / 'Table2_final_model_performance_rankings_manuscript.md'}`
- `{TABLES / 'Table3_final_target_predictability_summary.csv'}`
- `{TABLES / 'Table3_final_target_predictability_summary_manuscript.md'}`
- `{TABLES / 'table2_table3_claim_verification.csv'}`
"""
    (REPORT / "FINAL_table2_table3_forecasting_rebuild_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
