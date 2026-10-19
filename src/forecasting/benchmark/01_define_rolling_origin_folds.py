import pandas as pd

from candidate_config import MAIN_LOOKBACK, ROLLING_FOLDS, TABLES_DIR
from run_helpers import previous_month


def main() -> None:
    folds = []
    for item in ROLLING_FOLDS:
        validation_end = item["train_end"]
        validation_start = previous_month(validation_end, 23)
        core_train_end = previous_month(validation_start, 1)
        folds.append(
            {
                "fold": item["fold"],
                "expanding_history_start": "variable-specific first raster",
                "fold_training_end": item["train_end"],
                "core_model_training_end": core_train_end,
                "internal_validation_start": validation_start,
                "internal_validation_end": validation_end,
                "test_start": item["test_start"],
                "test_end": item["test_end"],
                "lookback": MAIN_LOOKBACK,
                "preprocessing_fit_end": item["train_end"],
                "preprocessing_rule": (
                    "unit conversion, imputation, detrending, and scaling use fold training "
                    "dates only; no test-derived statistics"
                ),
                "scaling": "calendar-month z score fitted through fold_training_end",
                "unit_correction": (
                    "fixed documented GPP conversion from 2022 onward; no observed train/test ratio"
                ),
                "validation_rule": "last 24 months of fold training period; never test months",
            }
        )
    pd.DataFrame(folds).to_csv(
        TABLES_DIR / "reviewer_safe_fold_manifest.csv", index=False
    )

    metrics = [
        ("R2", "1 - sum((observed-predicted)^2) / sum((observed-mean observed)^2)", "higher"),
        ("RMSE", "sqrt(mean((predicted-observed)^2))", "lower"),
        ("MAE", "mean(abs(predicted-observed))", "lower"),
        ("Pearson r", "linear correlation of observed and predicted values", "higher"),
        ("normalized RMSE", "RMSE / population standard deviation of observed values", "lower"),
        ("bias", "mean(predicted-observed)", "absolute value closer to zero"),
    ]
    pd.DataFrame(metrics, columns=["metric", "definition", "preferred_direction"]).to_csv(
        TABLES_DIR / "reviewer_safe_metric_definitions.csv", index=False
    )

    prohibited = [
        (
            "test-period statistical unit rescaling",
            "Mode A estimated train/test mean or median ratios using 2022-2024 values",
            "prohibited",
            "Only the fixed documented GPP conversion is allowed.",
        ),
        (
            "2024-only final evidence",
            "Mode A tested one calendar year after a fixed split",
            "prohibited",
            "Four rolling-origin folds are the main benchmark.",
        ),
        (
            "test-fitted scaling or detrending",
            "Any parameter estimated from fold test dates",
            "prohibited",
            "Parameters are fitted through each fold training endpoint only.",
        ),
        (
            "test-month early stopping",
            "Using fold test loss to select epochs or trees",
            "prohibited",
            "The last 24 months inside the fold training period are validation.",
        ),
        (
            "paper anomaly CSV model input",
            "paper_standardized_anomaly_series.csv or land-type mean anomaly CSV",
            "prohibited",
            "Verified selected-pixel GeoTIFF raster caches are used.",
        ),
    ]
    pd.DataFrame(
        prohibited,
        columns=["practice", "old_or_risky_behavior", "reviewer_safe_status", "replacement_rule"],
    ).to_csv(TABLES_DIR / "prohibited_mode_a_practices.csv", index=False)


if __name__ == "__main__":
    main()
