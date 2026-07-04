from __future__ import annotations

import csv
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
REPO = DESKTOP / "Linking-Drought-with-Vegetation-Response"
OUT = DESKTOP / "major_revision_analyses" / "FINAL_forecasting_framework_technical_audit"
TABLES = OUT / "tables"
REPORT = OUT / "report"
TOOLS = OUT / "tools"

FULL = DESKTOP / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate"
GNDVI = DESKTOP / "DL_GNDVI_previous_noQA_rerun"
SUPP = DESKTOP / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
SUBMISSION_AUDIT = DESKTOP / "major_revision_analyses" / "FINAL_submission_figure_directory_audit"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_row(path: Path, role: str) -> dict:
    return {
        "file_path": str(path),
        "file_name": path.name,
        "role": role,
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else "",
        "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        if path.exists()
        else "",
        "sha256": sha256(path) if path.exists() and path.is_file() else "",
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "_No rows._\n"
    out = ["|" + "|".join(columns) + "|", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        out.append(
            "|"
            + "|".join(str(row.get(col, "")).replace("\n", "<br>").replace("|", "\\|") for col in columns)
            + "|"
        )
    return "\n".join(out) + "\n"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    for d in (TABLES, REPORT, TOOLS):
        d.mkdir(parents=True, exist_ok=True)

    full_config = FULL / "scripts" / "candidate_config.py"
    gndvi_config = GNDVI / "scripts" / "candidate_config.py"
    fold_script = GNDVI / "scripts" / "01_define_reviewer_safe_folds.py"
    baseline_script = GNDVI / "scripts" / "02_run_reviewer_safe_baselines.py"
    compile_script = GNDVI / "scripts" / "06_compile_reviewer_safe_benchmark.py"
    fig7_script = REPO / "tools" / "build_previous_gndvi_figure7.py"
    s6_script = REPO / "tools" / "rebuild_supplementary_previous_gndvi_with_test.py"
    raster_script = GNDVI / "scripts" / "raster_pipeline.py"
    model_core = GNDVI / "scripts" / "model_core.py"
    run_helpers = GNDVI / "scripts" / "run_helpers.py"

    source_files = [
        (full_config, "full reviewer-safe 9-variable benchmark configuration"),
        (gndvi_config, "targeted previous/no-QA GNDVI rerun configuration"),
        (fold_script, "rolling-origin fold and metric manifest writer"),
        (baseline_script, "baseline runner"),
        (compile_script, "metric compiler and best-row selector"),
        (fig7_script, "final Figure 7 GNDVI replacement renderer"),
        (s6_script, "final Supplementary Figure S6 renderer"),
        (raster_script, "raster cache, preprocessing, scaling, supervised sample builder"),
        (model_core, "deep-learning model definitions and metric calculations"),
        (run_helpers, "learned-model run helpers and XGBoost settings"),
        (FULL / "tables" / "reviewer_safe_model_completion_status.csv", "full benchmark completion status"),
        (FULL / "tables" / "reviewer_safe_fold_manifest.csv", "full benchmark fold manifest"),
        (FULL / "tables" / "reviewer_safe_metric_definitions.csv", "metric definitions"),
        (GNDVI / "tables" / "forecasting_GNDVI_only_completion_audit.csv", "GNDVI-only rerun completion audit"),
        (GNDVI / "tables" / "figure7_previous_noQA_GNDVI_global_best_counts.csv", "final Figure 7 global best counts"),
        (GNDVI / "tables" / "figure7_combined_previous_noQA_GNDVI_panel_values.csv", "final Figure 7 panel source table"),
        (GNDVI / "tables" / "figure7_GNDVI_before_after_previous_noQA_audit.csv", "GNDVI Figure 7 before/after audit"),
        (GNDVI / "report" / "FINAL_confirmed_previous_GNDVI_forecasting_rerun_audit.md", "GNDVI-only rerun audit report"),
        (GNDVI / "report" / "Figure7_previous_noQA_GNDVI_render_report.md", "Figure 7 render lineage report"),
        (SUPP / "captions" / "Figure_S6_caption.md", "final S6 caption"),
        (SUPP / "tables" / "S6_combined_monthly_metrics_previous_GNDVI_with_test.csv", "S6 combined monthly metrics"),
        (SUPP / "tables" / "S6_combined_pixel_metrics_previous_GNDVI_with_test.csv", "S6 combined pixel metrics"),
        (SUBMISSION_AUDIT / "tables" / "replacement_log.csv", "final directory replacement log"),
    ]
    inventory = [file_row(p, role) for p, role in source_files]
    write_csv(TABLES / "source_file_inventory.csv", inventory)

    full_completion_path = FULL / "tables" / "reviewer_safe_model_completion_status.csv"
    full_completion = load_csv(full_completion_path)
    gndvi_completion = load_csv(GNDVI / "tables" / "forecasting_GNDVI_only_completion_audit.csv")
    folds = load_csv(FULL / "tables" / "reviewer_safe_fold_manifest.csv")
    metrics_defs = load_csv(FULL / "tables" / "reviewer_safe_metric_definitions.csv")
    counts = load_csv(GNDVI / "tables" / "figure7_previous_noQA_GNDVI_global_best_counts.csv")
    s6_model_summary = load_csv(SUPP / "source_mapping" / "Figure_S6_monthly_aggregate_model_summary.csv")
    s6_fold = load_csv(SUPP / "source_mapping" / "Figure_S6_fold_performance_previous_GNDVI_with_test.csv")

    folds.to_csv(TABLES / "rolling_origin_fold_design.csv", index=False)
    metrics_defs.to_csv(TABLES / "metric_definitions.csv", index=False)
    counts.to_csv(TABLES / "final_best_model_counts.csv", index=False)
    s6_model_summary.to_csv(TABLES / "supplementary_s6_monthly_model_summary.csv", index=False)
    s6_fold.to_csv(TABLES / "supplementary_s6_fold_performance_summary.csv", index=False)

    hyper_rows = [
        {
            "model": "LSTM",
            "family": "deep learning",
            "implementation": str(model_core),
            "evidence": "OriginalLSTM lines 28-42; MODEL_SETTINGS lines 282-292",
            "key_settings": "2-layer LSTM; hidden=128; dropout=0.2; batch=128; cap=8 epochs; patience=3; AdamW lr=0.001, weight_decay=0.0001; training sample cap=4000",
            "fixed_or_tuned": "fixed settings with internal-validation early stopping; no test tuning verified",
        },
        {
            "model": "BiLSTM",
            "family": "deep learning",
            "implementation": str(model_core),
            "evidence": "OriginalBiLSTM lines 45-61; make_model lines 339-342; MODEL_SETTINGS lines 293-302",
            "key_settings": "two recurrent layers; units=128 normally, 150 for TWS, 200 for SPEI; bidirectional only for SPEI/TWS; batch=64; cap=8 epochs; patience=3; AdamW lr=0.001",
            "fixed_or_tuned": "fixed variable-specific architecture inherited from prior code; no test tuning verified",
        },
        {
            "model": "TCN",
            "family": "deep learning",
            "implementation": str(model_core),
            "evidence": "OriginalTCN lines 99-119; MODEL_SETTINGS lines 303-312",
            "key_settings": "3 causal residual blocks; channels 32/32/32; kernel=3; dropout=0.3; batch=256; cap=3 epochs; patience=2; AdamW lr=0.001, weight_decay=0.0001",
            "fixed_or_tuned": "fixed settings with internal-validation early stopping; no test tuning verified",
        },
        {
            "model": "N-BEATS",
            "family": "deep learning",
            "implementation": str(model_core),
            "evidence": "OriginalNBeats lines 195-224; MODEL_SETTINGS lines 313-322",
            "key_settings": "generic/trend/seasonality stacks; 2 blocks each; width=64; batch=128; cap=8 epochs; patience=3; AdamW lr=0.001, weight_decay=0.0001",
            "fixed_or_tuned": "fixed settings with internal-validation early stopping; no test tuning verified",
        },
        {
            "model": "N-HiTS",
            "family": "deep learning",
            "implementation": str(model_core),
            "evidence": "OriginalNHiTS lines 262-279; MODEL_SETTINGS lines 323-331",
            "key_settings": "3 stacks; pooling 2/2/1; 2 layers; width=512; dropout=0.1; batch=128; cap=8 epochs; patience=3; AdamW lr=0.001, weight_decay=0.0001",
            "fixed_or_tuned": "fixed settings with internal-validation early stopping; no test tuning verified",
        },
        {
            "model": "XGBoost",
            "family": "non-deep machine learning",
            "implementation": str(run_helpers),
            "evidence": "XGB_SETTINGS lines 241-253; model fit lines 279-295",
            "key_settings": "n_estimators=300; max_depth=6; learning_rate=0.05; subsample=0.8; colsample_bytree=0.8; min_child_weight=3; reg_alpha=0.1; reg_lambda=1.0; objective=reg:squarederror; tree_method=hist; early_stopping_rounds=20",
            "fixed_or_tuned": "fixed settings with internal-validation early stopping; no test tuning verified",
        },
    ]
    write_csv(TABLES / "learned_model_hyperparameter_summary.csv", hyper_rows)

    baseline_rows = [
        {
            "baseline": "Persistence",
            "level": "pixel and monthly aggregate",
            "evidence": f"{baseline_script}, lines 172-178; full completion table completed_pixel_tasks=144 and completed_monthly_tasks=144",
            "figure7": "not included in main Figure 7 learned-model panels",
            "supplementary_s6": "included in monthly aggregate panel",
        },
        {
            "baseline": "Seasonal persistence",
            "level": "pixel and monthly aggregate",
            "evidence": f"{baseline_script}, lines 90-99 and 172-178; full completion table completed_pixel_tasks=144 and completed_monthly_tasks=144",
            "figure7": "not included in main Figure 7 learned-model panels",
            "supplementary_s6": "included in monthly aggregate panel",
        },
        {
            "baseline": "Monthly climatology",
            "level": "pixel and monthly aggregate",
            "evidence": f"{baseline_script}, lines 77-87 and 172-178; full completion table completed_pixel_tasks=144 and completed_monthly_tasks=144",
            "figure7": "not included in main Figure 7 learned-model panels",
            "supplementary_s6": "included in monthly aggregate panel",
        },
        {
            "baseline": "SARIMA",
            "level": "monthly land-type aggregate only",
            "evidence": f"{baseline_script}, lines 190-205 set metric_level_availability; {compile_script}, lines 184-190 set expected_pixel=0",
            "figure7": "not included in main Figure 7 learned-model panels",
            "supplementary_s6": "included in monthly aggregate panel only",
        },
    ]
    write_csv(TABLES / "baseline_level_summary.csv", baseline_rows)

    lineage_rows = [
        {
            "item": "Figure 7 final",
            "status": "verified",
            "source": str(GNDVI / "figures" / "Figure_7_previous_noQA_GNDVI_same_visualization.png"),
            "evidence": f"{fig7_script}, lines 233-257; replacement log copied this file into final Figure_7_final.png/pdf",
            "content": "pixel-level learned-model benchmark; R2 and normalized RMSE; non-GNDVI rows retained from accepted reviewer-safe source; GNDVI rows replaced from previous/no-QA rerun; SARIMA outside Figure 7",
        },
        {
            "item": "Supplementary Figure S6 final",
            "status": "verified",
            "source": str(SUPP / "figures_png" / "Figure_S6_forecasting_baselines_rolling_origin.png"),
            "evidence": f"{s6_script}, lines 632-648, 667-700, 722-748, 845-849",
            "content": "monthly aggregate baseline/model comparison including SARIMA; pixel-level fold summary for learned models; previous/no-QA GNDVI rows replaced only",
        },
    ]
    write_csv(TABLES / "figure7_s6_lineage.csv", lineage_rows)

    claim_rows = [
        {
            "claim": "Final task is one-month-ahead forecasting.",
            "status": "Supported",
            "evidence_file": str(raster_script),
            "line_or_column": "build_supervised lines 473-487 uses previous lookback observations to predict target month.",
            "recommended_wording": "one-month-ahead forecasting",
        },
        {
            "claim": "Final benchmark is single-target/univariate.",
            "status": "Supported",
            "evidence_file": str(run_helpers),
            "line_or_column": "lines 63-78 call prepare_task(land_type, variable) for one variable; model_core input_size=1 for recurrent/TCN models.",
            "recommended_wording": "one-month-ahead single-target forecasting using each target's own lagged raster series",
        },
        {
            "claim": "There are 9 target variables and 4 land types, giving 36 land-type x target tasks.",
            "status": "Supported",
            "evidence_file": str(full_config),
            "line_or_column": "lines 44-45 define 9 VARIABLES and 4 LAND_TYPES.",
            "recommended_wording": "4 land types x 9 targets = 36 tasks",
        },
        {
            "claim": "Learned models are LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost.",
            "status": "Supported",
            "evidence_file": str(full_config),
            "line_or_column": "line 46 ORIGINAL_MODELS.",
            "recommended_wording": "five deep-learning models plus XGBoost",
        },
        {
            "claim": "XGBoost is a deep-learning model.",
            "status": "Unsupported",
            "evidence_file": str(full_completion_path),
            "line_or_column": "model_family column lists XGBoost as machine_learning.",
            "recommended_wording": "XGBoost is a non-deep machine-learning comparator among the learned models",
        },
        {
            "claim": "Common 12-month lookback is used.",
            "status": "Supported",
            "evidence_file": str(full_config),
            "line_or_column": "line 49 MAIN_LOOKBACK = 12; fold_manifest lookback column = 12.",
            "recommended_wording": "common 12-month lookback",
        },
        {
            "claim": "Folds are expanding rolling-origin with test periods 2017-2018, 2019-2020, 2021-2022, and 2023-2024.",
            "status": "Supported",
            "evidence_file": str(FULL / "tables" / "reviewer_safe_fold_manifest.csv"),
            "line_or_column": "test_start/test_end columns.",
            "recommended_wording": "four expanding rolling-origin test periods: 2017-2018, 2019-2020, 2021-2022, and 2023-2024",
        },
        {
            "claim": "Validation is selected inside each fold.",
            "status": "Supported",
            "evidence_file": str(fold_script),
            "line_or_column": "lines 10-12 and 19-20 define last 24 months of the fold training period as internal validation.",
            "recommended_wording": "the last 24 months of each fold's training period were used for internal validation",
        },
        {
            "claim": "Scaling/preprocessing are training-only with no test statistics.",
            "status": "Supported with caveat",
            "evidence_file": str(raster_script),
            "line_or_column": "lines 341-415 fit imputation, detrending, and monthwise z scaling through train_end; fold manifest says no test-derived statistics.",
            "recommended_wording": "preprocessing/scaling parameters were fit through each fold's training endpoint; test periods were held out",
        },
        {
            "claim": "No test-derived model selection.",
            "status": "Partly supported",
            "evidence_file": str(fold_script),
            "line_or_column": "lines 72-75 prohibit test-month early stopping; model_core lines 394-433 use internal validation loss.",
            "recommended_wording": "internal training-period validation was used for early stopping; test periods were held out",
        },
        {
            "claim": "Gap handling within training folds.",
            "status": "Supported but too detailed for Figure 4",
            "evidence_file": str(raster_script),
            "line_or_column": "lines 341-363 impute calendar-month gaps; lines 451-455 pass train_end.",
            "recommended_wording": "training-only preprocessing/scaling",
        },
        {
            "claim": "Learned models are pixel-level.",
            "status": "Supported",
            "evidence_file": str(raster_script),
            "line_or_column": "lines 222-230 select pixels; build_supervised returns pixel_id rows; full completion table has 144 pixel tasks/model.",
            "recommended_wording": "pixel-level learned-model benchmark",
        },
        {
            "claim": "Valid pixel threshold was 30% and selected pixels capped at 100 per task.",
            "status": "Supported with caveat",
            "evidence_file": str(gndvi_config),
            "line_or_column": "lines 47-48 MAX_PIXELS=100 and MIN_VALID_FRACTION=0.30; raster_pipeline lines 195-202 define coverage mask.",
            "recommended_wording": "fixed selected-pixel set with minimum data-availability screen; disclose that availability screening used source-record coverage",
        },
        {
            "claim": "SARIMA is pixel-level.",
            "status": "Unsupported",
            "evidence_file": str(baseline_script),
            "line_or_column": "lines 202-204 state monthly aggregated only; compile script lines 184-190 expected_pixel=0.",
            "recommended_wording": "SARIMA was evaluated only at monthly land-type aggregate level",
        },
        {
            "claim": "Figure 7 uses R2 and raw RMSE.",
            "status": "Unsupported",
            "evidence_file": str(fig7_script),
            "line_or_column": "lines 199-210 plot mean_r2 and mean_normalized_rmse; line 219 best nRMSE count.",
            "recommended_wording": "Figure 7 reports R2 and nRMSE",
        },
        {
            "claim": "Supplementary Figure S6 uses R2 and nRMSE.",
            "status": "Supported",
            "evidence_file": str(s6_script),
            "line_or_column": "lines 722-730 and 740-748.",
            "recommended_wording": "S6 reports R2 and nRMSE at monthly aggregate and pixel-fold levels",
        },
        {
            "claim": "Final Figure 7 uses previous/no-QA GNDVI rerun with only GNDVI rows replaced.",
            "status": "Supported",
            "evidence_file": str(fig7_script),
            "line_or_column": "lines 233-257 concatenate non-GNDVI old rows and new GNDVI rows; replacement_log records final Figure 7 replacement.",
            "recommended_wording": "GNDVI target rows were replaced using the confirmed previous/no-QA GNDVI rerun; non-GNDVI rows were retained",
        },
        {
            "claim": "BiLSTM achieved best RMSE in 34/36 tasks.",
            "status": "Unsupported / obsolete",
            "evidence_file": str(GNDVI / "tables" / "figure7_previous_noQA_GNDVI_global_best_counts.csv"),
            "line_or_column": "final counts: BiLSTM best R2=7, best nRMSE=6; XGBoost best R2=18, best nRMSE=19.",
            "recommended_wording": "model rankings were target-, land-type-, fold-, metric-, and scale-dependent",
        },
    ]
    write_csv(TABLES / "forecasting_claim_verification.csv", claim_rows)

    risk_rows = [
        {
            "unsafe_claim_or_phrase": "six deep-learning models",
            "risk": "Incorrect because XGBoost is non-deep machine learning.",
            "local_hit_or_status": "Found in older audit/recommendation files as text to avoid; not supported by final completion table.",
            "replacement": "five deep-learning models plus XGBoost",
        },
        {
            "unsafe_claim_or_phrase": "2024-only test data / 2024-only final evidence",
            "risk": "Contradicts four rolling-origin folds.",
            "local_hit_or_status": "Fold script lines 60-63 explicitly marks 2024-only final evidence as prohibited.",
            "replacement": "four expanding rolling-origin test periods",
        },
        {
            "unsafe_claim_or_phrase": "12-24 month lookback / model-specific lookback windows",
            "risk": "Contradicts MAIN_LOOKBACK=12 for reviewer-safe benchmark.",
            "local_hit_or_status": "No final benchmark support; common lookback verified in config and fold manifest.",
            "replacement": "common 12-month lookback",
        },
        {
            "unsafe_claim_or_phrase": "BiLSTM achieved optimal RMSE on 94% / BiLSTM won 34/36",
            "risk": "Contradicted by final best-count table and metric naming.",
            "local_hit_or_status": "Search found this only in audit scripts as a red-flag phrase, but remove if present in manuscript.",
            "replacement": "XGBoost has the largest final best-count totals; rankings remain metric- and scale-dependent",
        },
        {
            "unsafe_claim_or_phrase": "TCN best R2 36%",
            "risk": "Not supported by final best-count table.",
            "local_hit_or_status": "Not verified in final outputs.",
            "replacement": "Use final best-count table or avoid single-model dominance claim",
        },
        {
            "unsafe_claim_or_phrase": "RMSE in Figure 7/S6 captions or Figure 4 evaluation box",
            "risk": "Raw RMSE is stored, but final visual metrics use normalized RMSE.",
            "local_hit_or_status": "Figure 7/S6 renderers plot normalized_rmse.",
            "replacement": "nRMSE or normalized RMSE",
        },
        {
            "unsafe_claim_or_phrase": "SARIMA included in pixel-level Figure 7",
            "risk": "False; SARIMA monthly aggregate only.",
            "local_hit_or_status": "Baseline script and S6 caption confirm monthly aggregate only.",
            "replacement": "SARIMA reported separately at monthly land-type aggregate level",
        },
        {
            "unsafe_claim_or_phrase": "GRU",
            "risk": "GRU is not in the final learned-model set.",
            "local_hit_or_status": "README.md contains an old GRU mention; final config excludes GRU.",
            "replacement": "Remove GRU from final methods unless explicitly describing historical repository contents",
        },
        {
            "unsafe_claim_or_phrase": "Independent Models",
            "risk": "Ambiguous and unsupported as a task definition.",
            "local_hit_or_status": "Old Figure 4 audit flags this phrase.",
            "replacement": "36 land-type x target tasks",
        },
        {
            "unsafe_claim_or_phrase": "unqualified deep-learning dominance",
            "risk": "Final evidence shows XGBoost highest best-count totals and aggregate-level performance depends on scale/metric.",
            "local_hit_or_status": "S6 caption explicitly says no unqualified model-family dominance claim.",
            "replacement": "model rankings were target-, land-type-, fold-, metric-, and scale-dependent",
        },
    ]
    write_csv(TABLES / "reviewer_risk_audit.csv", risk_rows)

    fig4_rows = [
        {
            "Forecasting setup": "One-month-ahead single-target forecasting",
            "Expanding rolling-origin evaluation": "Expanding training history through each fold endpoint",
            "Benchmark baselines": "Persistence",
        },
        {
            "Forecasting setup": "4 land types x 9 targets = 36 tasks",
            "Expanding rolling-origin evaluation": "Fold 1: train through 2016; validation 2015-2016; test 2017-2018",
            "Benchmark baselines": "Seasonal persistence",
        },
        {
            "Forecasting setup": "Five deep-learning models plus XGBoost",
            "Expanding rolling-origin evaluation": "Fold 2: train through 2018; validation 2017-2018; test 2019-2020",
            "Benchmark baselines": "Monthly climatology",
        },
        {
            "Forecasting setup": "XGBoost is a non-deep machine-learning comparator",
            "Expanding rolling-origin evaluation": "Fold 3: train through 2020; validation 2019-2020; test 2021-2022",
            "Benchmark baselines": "SARIMA: monthly land-type aggregate only",
        },
        {
            "Forecasting setup": "Common 12-month lookback; training-only preprocessing/scaling",
            "Expanding rolling-origin evaluation": "Fold 4: train through 2022; validation 2021-2022; test 2023-2024",
            "Benchmark baselines": "Baselines reported separately from pixel-level learned-model panels",
        },
    ]
    write_csv(TABLES / "recommended_figure4_table_text.csv", fig4_rows)

    manuscript_summary = (
        "The final reviewer-safe forecasting benchmark evaluated one-month-ahead single-target forecasting for "
        "nine indicators across four land types, yielding 36 land-type x target tasks. Five deep-learning models "
        "(LSTM, BiLSTM, TCN, N-BEATS, and N-HiTS) and the non-deep machine-learning comparator XGBoost were "
        "evaluated with a common 12-month lookback under four expanding rolling-origin test periods "
        "(2017-2018, 2019-2020, 2021-2022, and 2023-2024). Preprocessing and scaling parameters were fit through "
        "each fold's training endpoint, with the final 24 months of the fold training period used for internal "
        "validation and test periods held out for evaluation. Persistence, seasonal persistence, monthly "
        "climatology, and aggregate-level SARIMA baselines were evaluated separately, with SARIMA restricted to "
        "monthly land-type aggregate series. Final reviewer-facing metrics are R2 and nRMSE. Model rankings were "
        "target-, land-type-, fold-, metric-, and scale-dependent, so the revised manuscript should avoid an "
        "unqualified model-family dominance claim."
    )
    (REPORT / "manuscript_ready_section_3_3_summary.md").write_text(
        "# Manuscript-Ready Section 3.3 Summary\n\n" + manuscript_summary + "\n",
        encoding="utf-8",
    )

    final_counts_table = counts.to_dict("records")
    best_xgb_r2 = float(counts.loc[counts["model"] == "XGBoost", "best_r2_count"].iloc[0])
    best_xgb_nrmse = float(
        counts.loc[counts["model"] == "XGBoost", "best_normalized_rmse_count"].iloc[0]
    )

    report = f"""# Final Forecasting Framework Technical Audit

Generated: {datetime.now().isoformat()}

Scope: evidence-based audit only. No manuscript files, figures, models, or raster workflows were modified.

## 1. Final Forecasting Objective

The final reviewer-safe benchmark is verified as one-month-ahead, single-target forecasting. The full benchmark configuration defines the target variables as `GNDVI`, `GOSIF`, `GPP`, `LST`, `NDVI`, `NIRv`, `SPEI`, `TWS`, and `VPD`, and the four land types as `GRASSLANDS`, `NATURAL_FORESTS`, `SAVANNAS`, and `SHRUBLANDS` ({full_config}, lines 44-45). This gives 4 x 9 = 36 land-type x target tasks.

The code evidence supports a univariate/single-target setup: each task is prepared as `prepare_task(land_type, variable)` and supervised samples are built from one variable's prior lookback sequence ({run_helpers}, lines 63-78; {raster_script}, lines 473-487). I found no final benchmark evidence that the main Figure 7 benchmark is multivariate.

## 2. Input Structure and Lookback

The full benchmark uses `MAIN_LOOKBACK = 12` ({full_config}, line 49). The GNDVI-only replacement rerun preserves this setting ({gndvi_config}, line 46). The fold manifest also records `lookback = 12` for all four folds.

Important caveat: the report from the reviewer-safe package states "12 previous available raster observations." Because some products have missing months, especially TWS, this should not be described as always 12 consecutive calendar months.

## 3. Rolling-Origin Evaluation Design

{md_table(folds.to_dict("records"), list(folds.columns))}

The fold script computes the internal validation window as the last 24 months of the fold training period ({fold_script}, lines 10-12 and 19-20). The model-training samples end at `core_model_training_end`, while preprocessing/scaling parameters are fit through `fold_training_end` (`preprocessing_fit_end`) ({fold_script}, lines 17-24; {run_helpers}, lines 63-78). Test periods are held out from early stopping and model fit. The fold script explicitly prohibits test-month early stopping ({fold_script}, lines 72-75).

## 4. Models Included

The full final learned-model set is verified as LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost ({full_config}, line 46). XGBoost is a learned non-deep machine-learning comparator, not a deep-learning model and not one of the statistical baselines. The full completion table classifies XGBoost as `machine_learning`, while LSTM/BiLSTM/TCN/N-BEATS/N-HiTS are `deep_learning`.

## 5. Model Architecture and Hyperparameter Evidence

{md_table(hyper_rows, ["model", "family", "implementation", "evidence", "key_settings", "fixed_or_tuned"])}

The code supports fixed settings plus internal-validation early stopping. I did not find evidence of an exhaustive grid search in the final benchmark, and there is no support for test-period tuning.

## 6. Preprocessing, Scaling, and Leakage Prevention

The final reviewer-safe mode applies:

- fixed GPP unit conversion only, not a test-derived train/test ratio ({raster_script}, lines 324-338);
- calendar-month gap imputation with `train_end` passed from the fold ({raster_script}, lines 341-363 and 451-455);
- month-specific detrending fitted using dates through `train_end` ({raster_script}, lines 366-384);
- monthwise z scaling fitted through the fold training endpoint ({raster_script}, lines 387-415).

Test-derived scaling, detrending, and early stopping are explicitly prohibited by the fold manifest writer ({fold_script}, lines 52-80). Recommended wording: "training-only preprocessing/scaling; internal validation was restricted to the training fold." Avoid the broader phrase "no test-derived model selection" unless it is clarified as model-parameter/early-stopping selection, because pixel availability screening uses source-record coverage as described below.

## 7. Pixel-Level Versus Land-Type Aggregate Evaluation

Learned models are evaluated at pixel level and also summarized after monthly land-type aggregation. Pixel selection is performed by raster cache construction: `_coverage_mask` requires sufficient finite observations and selected pixels are capped at 100 per task ({gndvi_config}, lines 47-48; {raster_script}, lines 195-230).

Caveat: `_coverage_mask` uses dates <= 202112 and a 2024 availability screen ({raster_script}, lines 195-202). That appears to be an availability/coverage screen, not model fitting or scaling, but it should be disclosed cautiously as fixed selected-pixel availability screening. Do not describe the pixel set as selected solely from training folds unless that is separately verified.

The full completion table verifies 144 pixel tasks per learned model across 9 variables x 4 land types x 4 folds. SARIMA has no pixel-level tasks.

## 8. Benchmark Baselines

{md_table(baseline_rows, ["baseline", "level", "evidence", "figure7", "supplementary_s6"])}

Persistence, seasonal persistence, and monthly climatology are calculated at pixel level and monthly aggregate level. SARIMA is monthly land-type aggregate only; the script explicitly records `metric_level_availability = monthly aggregated only; per-pixel SARIMA infeasible` ({baseline_script}, lines 202-204).

## 9. Metrics

{md_table(metrics_defs.to_dict("records"), list(metrics_defs.columns))}

The metric implementation confirms normalized RMSE is `RMSE / population standard deviation of observed values` ({model_core}, lines 463-475). Raw RMSE is stored in output tables, but final Figure 7 and S6 use R2 and normalized RMSE/nRMSE for reviewer-facing visual summaries.

## 10. Figure 7 and Supplementary Figure S6

Figure 7: verified as pixel-level learned-model-only. The final renderer plots `mean_r2` and `mean_normalized_rmse` ({fig7_script}, lines 199-210), and the count panels are best R2 count and best nRMSE count ({fig7_script}, lines 218-219). SARIMA remains outside Figure 7.

Supplementary Figure S6: verified as monthly aggregate model/baseline comparison plus rolling-origin fold summary. It includes persistence, seasonal persistence, monthly climatology, SARIMA, XGBoost, and the five deep-learning models ({s6_script}, lines 112-124 and 845-849). The S6 renderer combines original non-GNDVI rows with new previous/no-QA GNDVI rows ({s6_script}, lines 632-648).

{md_table(lineage_rows, ["item", "status", "source", "evidence", "content"])}

## 11. GNDVI Forecasting Rerun

The GNDVI rerun configuration points to the previous/no-QA data package: `HISTORY_ROOT` and `TEST_ROOT` are under `final_previous_GNDVI_with_test_rebuild/data_for_forecasting` ({gndvi_config}, lines 28-30). The GNDVI rerun package intentionally sets `VARIABLES = ["GNDVI"]` ({gndvi_config}, line 41).

The rerun audit states that no non-GNDVI target was rerun and that model architectures, hyperparameters, 12-month lookback, folds, split logic, random seeds, scaling, preprocessing, and visualization format were preserved. The Figure 7 renderer then concatenates accepted non-GNDVI rows with new GNDVI rows ({fig7_script}, lines 233-236) and writes a before/after GNDVI audit ({fig7_script}, lines 242-249).

The final submission replacement log verifies that `Figure_7_final.png/pdf` were replaced by the previous/no-QA GNDVI Figure 7 files from the DL rerun package.

## 12. Final Best-Model Counts

{md_table(final_counts_table, list(counts.columns))}

After the GNDVI replacement, XGBoost has the largest final best-count totals: best R2 count = {best_xgb_r2:g}, best nRMSE count = {best_xgb_nrmse:g}. Claims such as "BiLSTM achieved best RMSE in 34/36 tasks" are obsolete/unsupported by the final count table.

## 13. Reviewer-Risk Audit

{md_table(risk_rows, ["unsafe_claim_or_phrase", "risk", "local_hit_or_status", "replacement"])}

Editable manuscript files were not clearly located in the active repository during this audit. I scanned local markdown/text/caption/report files and recorded verified local risks where found. Any manuscript draft should still be manually checked for the same phrases.

## 14. Manuscript-Ready Summary

{manuscript_summary}

## Recommended Figure 4 Central Table Text

{md_table(fig4_rows, ["Forecasting setup", "Expanding rolling-origin evaluation", "Benchmark baselines"])}

## Recommended Figure 4 Evaluation Box

Metrics: R2 and nRMSE.

Pixel-level learned-model benchmark: five deep-learning models plus XGBoost across 36 land-type x target tasks.

Global best-count panels: best R2 count and best nRMSE count across the 36 tasks.

Monthly aggregate baseline check: persistence, seasonal persistence, monthly climatology, and SARIMA. SARIMA is monthly land-type aggregate only and is not inserted into the pixel-level Figure 7 panels.

## Final Notes

- Do not write "six deep-learning models"; use "five deep-learning models plus XGBoost."
- Do not describe SARIMA as pixel-level.
- Do not claim a 2024-only test.
- Do not claim universal deep-learning dominance.
- Do not use RMSE alone when describing Figure 7/S6; use nRMSE/normalized RMSE.
- Do disclose the fixed selected-pixel availability screen if discussing pixel-level sampling details.
"""
    (REPORT / "FINAL_forecasting_framework_technical_audit.md").write_text(report, encoding="utf-8")

    shutil.copy2(Path(__file__), TOOLS / Path(__file__).name)


if __name__ == "__main__":
    main()
