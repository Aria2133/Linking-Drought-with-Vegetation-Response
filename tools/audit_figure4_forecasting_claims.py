from __future__ import annotations

import csv
import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
ROOT = DESKTOP / "major_revision_analyses" / "Figure4_forecasting_claim_audit"
TABLES = ROOT / "tables"
REPORT = ROOT / "report"

ORIG = DESKTOP / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate"
DL = DESKTOP / "DL_GNDVI_previous_noQA_rerun"
SUPP = DESKTOP / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
WORKSPACE = DESKTOP / "Linking-Drought-with-Vegetation-Response"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path, role: str) -> dict[str, str]:
    return {
        "role": role,
        "path": str(path),
        "exists": str(path.exists()),
        "size_bytes": str(path.stat().st_size) if path.exists() else "",
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") if path.exists() else "",
        "sha256": sha256(path) if path.exists() and path.is_file() and path.stat().st_size < 50_000_000 else "",
    }


def fmt_ym(value: int | str) -> str:
    value = int(value)
    return f"{value // 100:04d}-{value % 100:02d}"


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def scan_phrases(paths: list[Path]) -> list[dict[str, str]]:
    phrases = [
        "Resample to 5 km",
        "2024-only test",
        "variable-specific BiLSTM",
        "six deep-learning models",
        "Independent Models",
        "model-specific lookback windows",
        "model-specific preprocessing strategies",
        "No test-derived model selection",
        "Gap handling within training folds",
        "RMSE",
    ]
    exts = {".py", ".md", ".txt", ".csv", ".json", ".yml", ".yaml"}
    rows: list[dict[str, str]] = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            if path.stat().st_size > 5_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:
                continue
            for lineno, line in enumerate(text, 1):
                low = line.lower()
                for phrase in phrases:
                    if phrase.lower() in low:
                        rows.append(
                            {
                                "phrase": phrase,
                                "file": str(path),
                                "line": str(lineno),
                                "text": line.strip()[:500],
                                "risk_note": risk_note(phrase),
                            }
                        )
    return rows


def risk_note(phrase: str) -> str:
    mapping = {
        "RMSE": "RMSE is recorded, but Figure 7/S6 plot normalized RMSE; Figure 4 should say R2 and nRMSE when describing plotted benchmark metrics.",
        "No test-derived model selection": "Supported only in the sense of training-period validation and no test-month early stopping; safer wording recommended.",
        "Gap handling within training folds": "Supported for imputation/preprocessing fit through fold training endpoints; safer compact wording is training-only preprocessing/scaling.",
        "Independent Models": "Ambiguous; replace with 36 land-type x target tasks.",
        "six deep-learning models": "Incorrect because XGBoost is not deep learning.",
        "2024-only test": "Explicitly prohibited by reviewer-safe fold manifest.",
        "Resample to 5 km": "Avoid in forecasting schematic unless paired with analytical-overlay/TWS-support caveat.",
    }
    return mapping.get(phrase, "Unsafe or potentially obsolete Figure 4 wording; verify before use.")


def main() -> None:
    for directory in (TABLES, REPORT):
        directory.mkdir(parents=True, exist_ok=True)

    source_files = [
        file_record(ORIG / "scripts" / "candidate_config.py", "full reviewer-safe benchmark configuration"),
        file_record(DL / "scripts" / "candidate_config.py", "GNDVI-only previous/no-QA rerun configuration"),
        file_record(DL / "scripts" / "01_define_reviewer_safe_folds.py", "fold manifest and metric definitions generator"),
        file_record(DL / "scripts" / "raster_pipeline.py", "preprocessing, imputation, detrending, scaling, supervised sequence builder"),
        file_record(DL / "scripts" / "run_helpers.py", "fold sample construction and XGBoost validation/test separation"),
        file_record(DL / "scripts" / "model_core.py", "deep-learning training loop and internal validation"),
        file_record(DL / "scripts" / "02_run_reviewer_safe_baselines.py", "baseline and SARIMA benchmark script"),
        file_record(DL / "scripts" / "06_compile_reviewer_safe_benchmark.py", "benchmark compiler and SARIMA availability check"),
        file_record(WORKSPACE / "tools" / "build_previous_gndvi_figure7.py", "final Figure 7 renderer after previous/no-QA GNDVI replacement"),
        file_record(WORKSPACE / "tools" / "rebuild_supplementary_previous_gndvi_with_test.py", "final supplementary S6 builder after previous/no-QA GNDVI replacement"),
        file_record(DL / "tables" / "reviewer_safe_fold_manifest.csv", "final fold manifest"),
        file_record(DL / "tables" / "reviewer_safe_all_pixel_metrics.csv", "pixel-level metrics after GNDVI-only rerun"),
        file_record(DL / "tables" / "reviewer_safe_all_monthly_aggregated_metrics.csv", "monthly aggregate metrics after GNDVI-only rerun"),
        file_record(DL / "tables" / "forecasting_GNDVI_only_completion_audit.csv", "GNDVI-only rerun completion audit"),
        file_record(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv", "final Figure 7 source values"),
        file_record(DL / "report" / "Figure7_previous_noQA_GNDVI_render_report.md", "Figure 7 lineage report"),
        file_record(DL / "report" / "FINAL_confirmed_previous_GNDVI_forecasting_rerun_audit.md", "forecasting rerun audit report"),
        file_record(SUPP / "source_mapping" / "Figure_S6_monthly_aggregate_model_summary.csv", "S6 monthly aggregate model summary"),
        file_record(SUPP / "source_mapping" / "Figure_S6_fold_performance_previous_GNDVI_with_test.csv", "S6 rolling-origin fold performance summary"),
    ]
    write_csv(TABLES / "located_forecasting_sources.csv", source_files)

    fold_path = DL / "tables" / "reviewer_safe_fold_manifest.csv"
    folds = pd.read_csv(fold_path)
    fold_rows = []
    for _, row in folds.iterrows():
        fold_rows.append(
            {
                "fold": str(row["fold"]),
                "train_window": f"{row['expanding_history_start']} through {fmt_ym(row['fold_training_end'])}",
                "core_model_training_end": fmt_ym(row["core_model_training_end"]),
                "internal_validation_period": f"{fmt_ym(row['internal_validation_start'])} to {fmt_ym(row['internal_validation_end'])}",
                "test_period": f"{fmt_ym(row['test_start'])} to {fmt_ym(row['test_end'])}",
                "lookback": str(row["lookback"]),
                "preprocessing_fit_end": fmt_ym(row["preprocessing_fit_end"]),
                "preprocessing_rule": row["preprocessing_rule"],
                "validation_rule": row["validation_rule"],
            }
        )
    write_csv(TABLES / "rolling_origin_fold_verification.csv", fold_rows)

    pixel = pd.read_csv(DL / "tables" / "reviewer_safe_all_pixel_metrics.csv")
    monthly = pd.read_csv(DL / "tables" / "reviewer_safe_all_monthly_aggregated_metrics.csv")
    fig7 = pd.read_csv(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv")
    s6_monthly = pd.read_csv(SUPP / "source_mapping" / "Figure_S6_monthly_aggregate_model_summary.csv")
    s6_folds = pd.read_csv(SUPP / "source_mapping" / "Figure_S6_fold_performance_previous_GNDVI_with_test.csv")

    models = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
    baselines = ["Persistence", "Seasonal persistence", "Monthly climatology", "SARIMA"]
    target_count = fig7["target"].nunique()
    land_count = fig7["land_type"].nunique()
    task_count = target_count * land_count
    fig7_models = sorted(fig7["model"].unique())
    pixel_models = sorted(pixel["model"].dropna().unique())
    monthly_models = sorted(monthly["model"].dropna().unique())
    baseline_levels = []
    for model in baselines:
        pix = pixel[pixel["model"] == model]
        mon = monthly[monthly["model"] == model]
        baseline_levels.append(
            {
                "baseline": model,
                "pixel_rows": str(len(pix)),
                "monthly_rows": str(len(mon)),
                "pixel_task_folds": str(pix[["land_type", "variable", "fold"]].drop_duplicates().shape[0]) if not pix.empty else "0",
                "monthly_task_folds": str(mon[["land_type", "variable", "fold"]].drop_duplicates().shape[0]) if not mon.empty else "0",
                "availability": "monthly aggregate only" if model == "SARIMA" else "pixel and monthly aggregate",
            }
        )
    write_csv(TABLES / "baseline_level_verification.csv", baseline_levels)

    claims = [
        {
            "Claim": "Benchmark includes 9 target variables/indicators.",
            "Status": "Supported",
            "Evidence file path": str(ORIG / "scripts" / "candidate_config.py"),
            "Exact line number or table/column name": "lines 44-45 define 9 VARIABLES and 4 LAND_TYPES; final Figure 7 table target column has 9 unique values",
            "Recommended Figure 4 wording": "9 targets",
        },
        {
            "Claim": "Benchmark includes 4 land types and 36 land-type x target tasks.",
            "Status": "Supported",
            "Evidence file path": str(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv"),
            "Exact line number or table/column name": f"columns land_type and target: {land_count} land types x {target_count} targets = {task_count} tasks",
            "Recommended Figure 4 wording": "4 land types x 9 targets = 36 tasks",
        },
        {
            "Claim": "Learned-model set is LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost.",
            "Status": "Supported",
            "Evidence file path": str(ORIG / "scripts" / "candidate_config.py"),
            "Exact line number or table/column name": "line 46 ORIGINAL_MODELS; Figure 7 renderer lines 23-24",
            "Recommended Figure 4 wording": "Five deep-learning models plus XGBoost",
        },
        {
            "Claim": "XGBoost is a deep-learning model.",
            "Status": "Unsupported",
            "Evidence file path": str(ORIG / "scripts" / "candidate_config.py"),
            "Exact line number or table/column name": "line 46 lists XGBoost with original models; line 47 baselines exclude it; S6 caption states non-deep machine-learning comparator",
            "Recommended Figure 4 wording": "XGBoost: non-deep machine-learning comparator",
        },
        {
            "Claim": "Common 12-month input history/lookback was used.",
            "Status": "Supported",
            "Evidence file path": str(ORIG / "scripts" / "candidate_config.py"),
            "Exact line number or table/column name": "line 49 MAIN_LOOKBACK = 12; fold manifest lookback column = 12",
            "Recommended Figure 4 wording": "Common 12-month lookback",
        },
        {
            "Claim": "Forecasting is 1-month-ahead single-target.",
            "Status": "Supported",
            "Evidence file path": str(DL / "scripts" / "raster_pipeline.py"),
            "Exact line number or table/column name": "lines 473-487 build each sample from previous lookback months to the target month; Figure 7 target column is single variable",
            "Recommended Figure 4 wording": "One-month-ahead single-target forecasting",
        },
        {
            "Claim": "Training-only scaling/preprocessing without test-data leakage.",
            "Status": "Supported",
            "Evidence file path": str(DL / "scripts" / "raster_pipeline.py"),
            "Exact line number or table/column name": "lines 451-458 pass train_end to imputation, detrending, and scaling; fold manifest preprocessing_rule",
            "Recommended Figure 4 wording": "Training-only preprocessing/scaling",
        },
        {
            "Claim": "Gap handling within training folds.",
            "Status": "Supported",
            "Evidence file path": str(DL / "scripts" / "raster_pipeline.py"),
            "Exact line number or table/column name": "lines 451-455 impute_calendar_month(... train_end=train_end); fold manifest preprocessing_rule includes imputation through fold training dates only",
            "Recommended Figure 4 wording": "Training-only preprocessing/scaling; describe imputation detail in Methods, not the schematic table",
        },
        {
            "Claim": "No test-derived model selection.",
            "Status": "Partly supported",
            "Evidence file path": str(DL / "scripts" / "01_define_reviewer_safe_folds.py"),
            "Exact line number or table/column name": "lines 72-75 prohibit test-month early stopping; model_core lines 394-433 and run_helpers lines 274-296 use internal validation, not test",
            "Recommended Figure 4 wording": "Training-period validation only; test periods held out",
        },
        {
            "Claim": "Four rolling-origin folds with tests 2017-2018, 2019-2020, 2021-2022, and 2023-2024.",
            "Status": "Supported",
            "Evidence file path": str(DL / "tables" / "reviewer_safe_fold_manifest.csv"),
            "Exact line number or table/column name": "columns fold, test_start, test_end",
            "Recommended Figure 4 wording": "Expanding folds: test 2017-18, 2019-20, 2021-22, 2023-24",
        },
        {
            "Claim": "Train window expands through time.",
            "Status": "Supported",
            "Evidence file path": str(DL / "tables" / "reviewer_safe_fold_manifest.csv"),
            "Exact line number or table/column name": "fold_training_end increases from 201612 to 202212; expanding_history_start column",
            "Recommended Figure 4 wording": "Expanding training history through each fold endpoint",
        },
        {
            "Claim": "Persistence, seasonal persistence, monthly climatology, and SARIMA baselines were included.",
            "Status": "Supported",
            "Evidence file path": str(DL / "scripts" / "02_run_reviewer_safe_baselines.py"),
            "Exact line number or table/column name": "lines 172-178 define persistence, seasonal persistence, monthly climatology; lines 190-205 run SARIMA",
            "Recommended Figure 4 wording": "Baselines: persistence, seasonal persistence, monthly climatology, SARIMA",
        },
        {
            "Claim": "SARIMA was included in the pixel-level Figure 7 benchmark.",
            "Status": "Unsupported",
            "Evidence file path": str(DL / "scripts" / "02_run_reviewer_safe_baselines.py"),
            "Exact line number or table/column name": "lines 202-204 metric_level_availability = monthly aggregated only; compile script lines 184-190 expected_pixel = 0 for SARIMA",
            "Recommended Figure 4 wording": "SARIMA: monthly land-type aggregate only",
        },
        {
            "Claim": "Figure 7 uses R2 and raw RMSE.",
            "Status": "Unsupported",
            "Evidence file path": str(WORKSPACE / "tools" / "build_previous_gndvi_figure7.py"),
            "Exact line number or table/column name": "lines 199-210 plot mean_r2 and mean_normalized_rmse; count panel line 219 says Best nRMSE Count",
            "Recommended Figure 4 wording": "Metrics: R2 and nRMSE",
        },
        {
            "Claim": "Supplementary Figure S6 uses R2 and nRMSE.",
            "Status": "Supported",
            "Evidence file path": str(WORKSPACE / "tools" / "rebuild_supplementary_previous_gndvi_with_test.py"),
            "Exact line number or table/column name": "lines 722-730 monthly R2/nRMSE; lines 740-748 pixel-level R2/nRMSE by fold",
            "Recommended Figure 4 wording": "S6 summarizes R2 and nRMSE for aggregate baselines and rolling-origin fold performance",
        },
        {
            "Claim": "2024-only test remains the final evidence.",
            "Status": "Unsupported",
            "Evidence file path": str(DL / "scripts" / "01_define_reviewer_safe_folds.py"),
            "Exact line number or table/column name": "lines 60-63 mark 2024-only final evidence prohibited",
            "Recommended Figure 4 wording": "Four rolling-origin test periods",
        },
        {
            "Claim": "Variable-specific/model-specific lookbacks or preprocessing are used in the final reviewer-safe benchmark.",
            "Status": "Unsupported",
            "Evidence file path": str(ORIG / "scripts" / "candidate_config.py"),
            "Exact line number or table/column name": "line 49 common MAIN_LOOKBACK = 12; fold manifest preprocessing_fit_end by fold, not by model-specific test tuning",
            "Recommended Figure 4 wording": "Common lookback and training-only preprocessing for all model comparisons",
        },
        {
            "Claim": "Evaluation box should say Independent Models.",
            "Status": "Unsupported",
            "Evidence file path": str(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv"),
            "Exact line number or table/column name": "tasks are land_type x target combinations; no column defines independent models",
            "Recommended Figure 4 wording": "36 land-type x target tasks",
        },
    ]
    write_csv(TABLES / "figure4_forecasting_claim_verification.csv", claims)

    # Basic validation summary used by the report.
    validation_rows = [
        {"check": "Figure 7 unique targets", "value": str(target_count), "source": str(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv")},
        {"check": "Figure 7 unique land types", "value": str(land_count), "source": str(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv")},
        {"check": "Figure 7 learned models", "value": ", ".join(fig7_models), "source": str(DL / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv")},
        {"check": "Pixel metrics models", "value": ", ".join(pixel_models), "source": str(DL / "tables" / "reviewer_safe_all_pixel_metrics.csv")},
        {"check": "Monthly metrics models", "value": ", ".join(monthly_models), "source": str(DL / "tables" / "reviewer_safe_all_monthly_aggregated_metrics.csv")},
        {"check": "S6 monthly models", "value": ", ".join(s6_monthly["model"].astype(str).tolist()), "source": str(SUPP / "source_mapping" / "Figure_S6_monthly_aggregate_model_summary.csv")},
        {"check": "S6 fold periods", "value": ", ".join(s6_folds["test_period"].astype(str).tolist()), "source": str(SUPP / "source_mapping" / "Figure_S6_fold_performance_previous_GNDVI_with_test.csv")},
    ]
    write_csv(TABLES / "forecasting_metadata_validation_summary.csv", validation_rows)

    scan_roots = [
        DESKTOP / "ECOINF_revision_final_figures_package",
        SUPP,
        DL / "report",
        DL / "scripts",
        WORKSPACE / "tools",
        ROOT,
    ]
    phrase_hits = scan_phrases(scan_roots)
    write_csv(TABLES / "unsafe_figure4_phrase_scan.csv", phrase_hits)

    recommended = f"""# Recommended Figure 4 Forecasting Text

## Central Three-Column Table

| Forecasting setup | Expanding rolling-origin evaluation | Benchmark baselines |
|---|---|---|
| One-month-ahead single-target forecasting | Expanding training history through each fold endpoint | Persistence |
| 4 land types x 9 targets = 36 tasks | Fold 1: train through 2016; validation 2015-2016; test 2017-2018 | Seasonal persistence |
| Five deep-learning models plus XGBoost | Fold 2: train through 2018; validation 2017-2018; test 2019-2020 | Monthly climatology |
| XGBoost is a non-deep machine-learning comparator | Fold 3: train through 2020; validation 2019-2020; test 2021-2022 | SARIMA: monthly land-type aggregate only |
| Common 12-month lookback; training-only preprocessing/scaling | Fold 4: train through 2022; validation 2021-2022; test 2023-2024 | Baselines reported separately from the pixel-level learned-model panels |

## Evaluation Box

Metrics: R2 and nRMSE

Pixel-level learned-model benchmark: five deep-learning models plus XGBoost across 36 land-type x target tasks.

Global best-count panels: best R2 count and best nRMSE count across the 36 tasks.

Monthly aggregate baseline check: persistence, seasonal persistence, monthly climatology, and SARIMA. SARIMA is monthly land-type aggregate only and is not inserted into the pixel-level Figure 7 panels.

## Wording To Remove Or Replace

- Replace `Metrics (R2, RMSE)` with `Metrics (R2, nRMSE)` when describing Figure 7/S6.
- Replace `Independent Models` with `36 land-type x target tasks`.
- Avoid `six deep-learning models`; use `five deep-learning models plus XGBoost`.
- Avoid a table layout that visually pairs individual baselines with individual test periods.
- Prefer `Training-period validation only; test periods held out` over `No test-derived model selection`.
- Prefer `Training-only preprocessing/scaling` over the longer `Gap handling within training folds` inside the figure; keep gap-handling details in Methods or caption.
"""
    (REPORT / "Figure4_recommended_text.md").write_text(recommended, encoding="utf-8")

    report = f"""# Figure 4 Forecasting-Method Claim Audit

Completed: {datetime.now().isoformat(timespec="seconds")}

## Scope

This audit checked the Figure 4 forecasting-method wording against the final reviewer-safe forecasting scripts, the previous/no-QA GNDVI rerun package, Figure 7 source tables, Supplementary Figure S6 source tables, and final audit reports. No forecasting models or scientific analyses were rerun.

## Key Findings

- The reviewer-safe learned-model benchmark is supported as a common 12-month, single-target monthly forecasting benchmark with five deep-learning models plus XGBoost.
- The full accepted Figure 7 source table contains {target_count} targets and {land_count} land types, giving {task_count} land-type x target tasks.
- XGBoost is part of the original learned-model comparison but is not a deep-learning model and should not be described as a benchmark baseline.
- The rolling-origin design is supported by four folds with test periods 2017-2018, 2019-2020, 2021-2022, and 2023-2024.
- Persistence, seasonal persistence, and monthly climatology are available at pixel and monthly aggregate levels. SARIMA is monthly land-type aggregate only.
- Figure 7 and Supplementary Figure S6 use R2 and normalized RMSE/nRMSE in the plotted evidence, even though raw RMSE is also stored in metric tables.
- `Gap handling within training folds` is supported by preprocessing code and the fold manifest, but is too detailed for the central schematic table. Use `Training-only preprocessing/scaling` in Figure 4.
- `No test-derived model selection` is directionally supported by internal validation and prohibited-practice tables, but safer figure wording is `Training-period validation only; test periods held out`.

## Main Evidence Files

- Full benchmark configuration: `{ORIG / 'scripts' / 'candidate_config.py'}`
- Previous/no-QA GNDVI rerun configuration: `{DL / 'scripts' / 'candidate_config.py'}`
- Fold manifest: `{DL / 'tables' / 'reviewer_safe_fold_manifest.csv'}`
- Baseline script: `{DL / 'scripts' / '02_run_reviewer_safe_baselines.py'}`
- Benchmark compiler: `{DL / 'scripts' / '06_compile_reviewer_safe_benchmark.py'}`
- Final Figure 7 source table: `{DL / 'tables' / 'figure7_combined_previous_noQA_GNDVI_panel_values.csv'}`
- S6 monthly source table: `{SUPP / 'source_mapping' / 'Figure_S6_monthly_aggregate_model_summary.csv'}`
- S6 fold source table: `{SUPP / 'source_mapping' / 'Figure_S6_fold_performance_previous_GNDVI_with_test.csv'}`

## Output Tables

- `tables/figure4_forecasting_claim_verification.csv`
- `tables/rolling_origin_fold_verification.csv`
- `tables/baseline_level_verification.csv`
- `tables/forecasting_metadata_validation_summary.csv`
- `tables/located_forecasting_sources.csv`
- `tables/unsafe_figure4_phrase_scan.csv`

## Recommendation

Revise Figure 4 wording, but do not alter Figure 7 or Supplementary Figure S6 source lineage. The main correction is textual: remove old or ambiguous schematic phrases, especially `RMSE` in the evaluation box, `Independent Models`, and anything implying one baseline per fold or six deep-learning models.
"""
    (REPORT / "Figure4_forecasting_claim_audit.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
