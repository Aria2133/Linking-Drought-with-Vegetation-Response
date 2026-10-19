import numpy as np
import pandas as pd

from candidate_config import (
    BASELINES,
    CSV_CANDIDATE_PACKAGE,
    ORIGINAL_MODELS,
    SOURCE_PACKAGE,
    TABLES_DIR,
)

PIXEL_FILES = [
    "reviewer_safe_baseline_pixel_metrics.csv",
    "reviewer_safe_xgboost_pixel_metrics.csv",
    "reviewer_safe_lstm_pixel_metrics.csv",
    "reviewer_safe_remaining_models_pixel_metrics.csv",
]
MONTHLY_FILES = [
    "reviewer_safe_baseline_monthly_aggregated_metrics.csv",
    "reviewer_safe_xgboost_monthly_aggregated_metrics.csv",
    "reviewer_safe_lstm_monthly_aggregated_metrics.csv",
    "reviewer_safe_remaining_models_monthly_aggregated_metrics.csv",
]
METRICS = ["r2", "rmse", "mae", "pearson_r", "normalized_rmse", "bias"]


def read_many(names):
    frames = []
    for name in names:
        path = TABLES_DIR / name
        if path.exists() and path.stat().st_size:
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def task_seed_mean(frame):
    group = ["model", "land_type", "variable", "fold"]
    return frame.groupby(group, as_index=False)[METRICS].mean()


def summary(frame, metric_level):
    tasks = task_seed_mean(frame)
    aggregate = tasks.groupby("model")[METRICS].agg(["mean", "std", "count"])
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    output = aggregate.reset_index()
    output["metric_level"] = metric_level
    output["model_family"] = output["model"].map(
        lambda value: (
            "deep_learning"
            if value in {"LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS"}
            else "machine_learning"
            if value == "XGBoost"
            else "statistical_baseline"
        )
    )
    return output


def variable_summary(frame, metric_level):
    tasks = task_seed_mean(frame)
    output = (
        tasks.groupby(["variable", "model"], as_index=False)[METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    output.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in output.columns
    ]
    output["metric_level"] = metric_level
    return output


def best_rows(frame, metric_level):
    tasks = task_seed_mean(frame)
    rows = []
    for keys, group in tasks.groupby(["land_type", "variable", "fold"]):
        land_type, variable, fold = keys
        for metric, ascending in [("r2", False), ("normalized_rmse", True)]:
            valid = group.dropna(subset=[metric]).sort_values(metric, ascending=ascending)
            if valid.empty:
                continue
            best = valid.iloc[0]
            rows.append(
                {
                    "metric_level": metric_level,
                    "land_type": land_type,
                    "variable": variable,
                    "fold": fold,
                    "metric": metric,
                    "best_model": best["model"],
                    "best_value": best[metric],
                    "candidate_status": "pending_human_approval",
                }
            )
    return pd.DataFrame(rows)


def mode_a_comparison(reviewer_pixel):
    old = pd.read_csv(SOURCE_PACKAGE / "tables" / "all_models_pixel_metrics.csv")
    old = old.query("mode == 'Mode_A_2024' and model in @ORIGINAL_MODELS")
    old_summary = (
        old.groupby(["model", "land_type", "variable"], as_index=False)
        .agg(
            mode_a_r2=("r2", "mean"),
            mode_a_rmse=("rmse", "mean"),
            mode_a_nrmse=("normalized_rmse", "mean"),
        )
    )
    reviewer_summary = (
        reviewer_pixel[reviewer_pixel["model"].isin(ORIGINAL_MODELS)]
        .groupby(["model", "land_type", "variable"], as_index=False)
        .agg(
            reviewer_safe_r2=("r2", "mean"),
            reviewer_safe_rmse=("rmse", "mean"),
            reviewer_safe_nrmse=("normalized_rmse", "mean"),
        )
    )
    output = reviewer_summary.merge(
        old_summary, on=["model", "land_type", "variable"], how="left"
    )
    output["r2_change_reviewer_safe_minus_mode_a"] = (
        output["reviewer_safe_r2"] - output["mode_a_r2"]
    )
    output["rmse_change_reviewer_safe_minus_mode_a"] = (
        output["reviewer_safe_rmse"] - output["mode_a_rmse"]
    )
    output["comparison_note"] = (
        "Combined effect of rolling-origin periods and removal of test-derived Mode A rescaling; "
        "not a single-factor causal estimate."
    )
    return output


def csv_comparison(reviewer_pixel):
    path = CSV_CANDIDATE_PACKAGE / "tables" / "fair_forecasting_all_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    csv = pd.read_csv(path).rename(
        columns={"target": "variable", "nrmse": "normalized_rmse"}
    )
    csv["land_type"] = csv["land_type"].astype(str).str.upper().str.replace(" ", "_")
    csv = csv[csv["lookback"] == 12]
    csv_summary = (
        csv.groupby(["model", "land_type", "variable"], as_index=False)
        .agg(
            csv_candidate_r2=("r2", "mean"),
            csv_candidate_rmse=("rmse", "mean"),
            csv_candidate_nrmse=("normalized_rmse", "mean"),
        )
    )
    reviewer_summary = (
        reviewer_pixel.groupby(["model", "land_type", "variable"], as_index=False)
        .agg(
            reviewer_safe_r2=("r2", "mean"),
            reviewer_safe_rmse=("rmse", "mean"),
            reviewer_safe_nrmse=("normalized_rmse", "mean"),
        )
    )
    output = reviewer_summary.merge(
        csv_summary, on=["model", "land_type", "variable"], how="inner"
    )
    output["r2_change_reviewer_safe_minus_csv"] = (
        output["reviewer_safe_r2"] - output["csv_candidate_r2"]
    )
    output["rmse_change_reviewer_safe_minus_csv"] = (
        output["reviewer_safe_rmse"] - output["csv_candidate_rmse"]
    )
    return output


def completion_status(pixel, monthly):
    rows = []
    for model in ORIGINAL_MODELS + BASELINES:
        pixel_model = pixel[pixel["model"] == model]
        monthly_model = monthly[monthly["model"] == model]
        if model in BASELINES:
            pixel_tasks = pixel_model[
                ["land_type", "variable", "fold"]
            ].drop_duplicates().shape[0]
            monthly_tasks = monthly_model[
                ["land_type", "variable", "fold"]
            ].drop_duplicates().shape[0]
            expected_pixel = 0 if model == "SARIMA" else 144
            expected_monthly = 144
            full = monthly_tasks == expected_monthly and (
                model == "SARIMA" or pixel_tasks == expected_pixel
            )
            note = (
                "SARIMA is monthly-aggregate only; per-pixel fitting was computationally infeasible."
                if model == "SARIMA"
                else ""
            )
        else:
            seed42 = pixel_model[
                pd.to_numeric(pixel_model["seed"], errors="coerce") == 42
            ]
            pixel_tasks = seed42[
                ["land_type", "variable", "fold"]
            ].drop_duplicates().shape[0]
            seed42_monthly = monthly_model[
                pd.to_numeric(monthly_model["seed"], errors="coerce") == 42
            ]
            monthly_tasks = seed42_monthly[
                ["land_type", "variable", "fold"]
            ].drop_duplicates().shape[0]
            expected_pixel = expected_monthly = 144
            full = pixel_tasks == 144 and monthly_tasks == 144
            note = ""
        rows.append(
            {
                "model": model,
                "model_family": (
                    "deep_learning"
                    if model in {"LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS"}
                    else "machine_learning"
                    if model == "XGBoost"
                    else "statistical_baseline"
                ),
                "expected_pixel_tasks": expected_pixel,
                "completed_pixel_tasks": pixel_tasks,
                "expected_monthly_tasks": expected_monthly,
                "completed_monthly_tasks": monthly_tasks,
                "full_rolling_origin_complete": full,
                "seeds_present": ";".join(
                    sorted(set(pixel_model["seed"].astype(str)))
                ),
                "status": "full" if full else "partial",
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def main():
    pixel = read_many(PIXEL_FILES)
    monthly = read_many(MONTHLY_FILES)
    pixel.to_csv(TABLES_DIR / "reviewer_safe_all_pixel_metrics.csv", index=False)
    monthly.to_csv(
        TABLES_DIR / "reviewer_safe_all_monthly_aggregated_metrics.csv", index=False
    )

    by_model = pd.concat(
        [summary(pixel, "pixel"), summary(monthly, "monthly_aggregated")],
        ignore_index=True,
    )
    by_model.to_csv(TABLES_DIR / "reviewer_safe_summary_by_model.csv", index=False)
    pd.concat(
        [
            variable_summary(pixel, "pixel"),
            variable_summary(monthly, "monthly_aggregated"),
        ],
        ignore_index=True,
    ).to_csv(TABLES_DIR / "reviewer_safe_summary_by_variable.csv", index=False)
    pd.concat(
        [best_rows(pixel, "pixel"), best_rows(monthly, "monthly_aggregated")],
        ignore_index=True,
    ).to_csv(TABLES_DIR / "reviewer_safe_best_model_by_task.csv", index=False)

    mode_a = mode_a_comparison(pixel)
    mode_a.to_csv(TABLES_DIR / "comparison_modeA_vs_reviewer_safe.csv", index=False)
    csv = csv_comparison(pixel)
    csv.to_csv(TABLES_DIR / "comparison_csv_candidate_vs_reviewer_safe.csv", index=False)

    pixel_tasks = task_seed_mean(pixel)
    monthly_tasks = task_seed_mean(monthly)
    level_compare = pixel_tasks.merge(
        monthly_tasks,
        on=["model", "land_type", "variable", "fold"],
        suffixes=("_pixel", "_monthly"),
        how="inner",
    )
    level_compare["r2_pixel_minus_monthly"] = (
        level_compare["r2_pixel"] - level_compare["r2_monthly"]
    )
    level_compare["rmse_pixel_minus_monthly"] = (
        level_compare["rmse_pixel"] - level_compare["rmse_monthly"]
    )
    level_compare["nrmse_pixel_minus_monthly"] = (
        level_compare["normalized_rmse_pixel"]
        - level_compare["normalized_rmse_monthly"]
    )
    level_compare.to_csv(
        TABLES_DIR / "reviewer_safe_pixel_vs_monthly_comparison.csv", index=False
    )
    diagnosis = [
        {
            "diagnostic": "pixel_vs_monthly_aggregation",
            "evidence_rows": len(level_compare),
            "mean_r2_change": (
                level_compare["r2_pixel"] - level_compare["r2_monthly"]
            ).mean(),
            "median_r2_change": (
                level_compare["r2_pixel"] - level_compare["r2_monthly"]
            ).median(),
            "interpretation": "Positive values mean pixel-level R2 is stronger.",
        },
        {
            "diagnostic": "mode_a_vs_reviewer_safe",
            "evidence_rows": len(mode_a),
            "mean_r2_change": mode_a[
                "r2_change_reviewer_safe_minus_mode_a"
            ].mean(),
            "median_r2_change": mode_a[
                "r2_change_reviewer_safe_minus_mode_a"
            ].median(),
            "interpretation": (
                "Combined rolling-period and leakage-removal effect; not separately identifiable."
            ),
        },
        {
            "diagnostic": "csv_candidate_vs_reviewer_safe",
            "evidence_rows": len(csv),
            "mean_r2_change": (
                csv["r2_change_reviewer_safe_minus_csv"].mean()
                if len(csv)
                else np.nan
            ),
            "median_r2_change": (
                csv["r2_change_reviewer_safe_minus_csv"].median()
                if len(csv)
                else np.nan
            ),
            "interpretation": "Positive values favor raster-selected pixel forecasting.",
        },
        {
            "diagnostic": "fixed_gndvi_mode_a_vs_reviewer_safe",
            "evidence_rows": int((mode_a["variable"] == "GNDVI").sum()),
            "mean_r2_change": mode_a.query("variable == 'GNDVI'")[
                "r2_change_reviewer_safe_minus_mode_a"
            ].mean(),
            "median_r2_change": mode_a.query("variable == 'GNDVI'")[
                "r2_change_reviewer_safe_minus_mode_a"
            ].median(),
            "interpretation": (
                "Both sides use fixed GNDVI; change reflects evaluation/preprocessing, not GNDVI replacement."
            ),
        },
    ]
    pd.DataFrame(diagnosis).to_csv(
        TABLES_DIR / "reviewer_safe_diagnosis.csv", index=False
    )
    completion_status(pixel, monthly).to_csv(
        TABLES_DIR / "reviewer_safe_model_completion_status.csv", index=False
    )


if __name__ == "__main__":
    main()
