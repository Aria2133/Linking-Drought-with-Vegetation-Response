import json

import pandas as pd

from candidate_config import (
    BILSTM_CONFIG,
    NBEATS_SPEI_CONFIG,
    TABLES_DIR,
    TARGETS,
    XGBOOST_BASE_CONFIG,
    XGBOOST_SPEI_CONFIG,
)


def main():
    configurations = []
    for variable in TARGETS:
        configurations.append(
            {
                "model": "BiLSTM",
                "target": variable,
                "configuration": "common_reviewer_safe",
                "lookback": 12,
                "capacity": "existing reviewer-safe BiLSTM capacity",
                "scaling": "training-only monthwise z-score",
                "gap_handling": "available observations",
                "source": "existing reviewer-safe result table; not recomputed",
            }
        )
        cfg = BILSTM_CONFIG[variable]
        configurations.append(
            {
                "model": "BiLSTM",
                "target": variable,
                "configuration": "hydroclimate_special",
                "lookback": cfg["lookback"],
                "capacity": json.dumps(cfg, sort_keys=True),
                "scaling": "training-only global MinMax",
                "gap_handling": (
                    "available observations and training-only state-space sensitivity"
                    if variable == "TWS"
                    else "available observations"
                ),
                "source": "BiLSTM for Meteorlogical data.py",
            }
        )
        configurations.append(
            {
                "model": "XGBoost",
                "target": variable,
                "configuration": "common_reviewer_safe",
                "lookback": 12,
                "capacity": "300 trees, depth 6, learning rate 0.05",
                "scaling": "training-only monthwise z-score",
                "gap_handling": "available observations",
                "source": "existing reviewer-safe result table; not recomputed",
            }
        )
        xgb = XGBOOST_SPEI_CONFIG if variable == "SPEI" else XGBOOST_BASE_CONFIG
        configurations.append(
            {
                "model": "XGBoost",
                "target": variable,
                "configuration": "hydroclimate_special",
                "lookback": xgb["lookback"],
                "capacity": json.dumps(xgb, sort_keys=True),
                "scaling": "training-only monthwise z-score",
                "gap_handling": (
                    "available observations and training-only state-space sensitivity"
                    if variable == "TWS"
                    else "available observations"
                ),
                "source": "XGBoost 2024 Test only (SPEI, TWS, VPD, LST).py",
            }
        )
    configurations.extend(
        [
            {
                "model": "N-BEATS",
                "target": "SPEI",
                "configuration": "common_reviewer_safe",
                "lookback": 12,
                "capacity": "existing generic/trend/seasonality reviewer-safe model",
                "scaling": "training-only monthwise z-score",
                "gap_handling": "available observations",
                "source": "existing reviewer-safe result table; not recomputed",
            },
            {
                "model": "N-BEATS",
                "target": "SPEI",
                "configuration": "hydroclimate_special",
                "lookback": NBEATS_SPEI_CONFIG["lookback"],
                "capacity": json.dumps(NBEATS_SPEI_CONFIG, sort_keys=True),
                "scaling": "training-only calendar-month median/IQR",
                "gap_handling": "available observations",
                "source": "N-BEATS SPEI.py",
            },
        ]
    )
    pd.DataFrame(configurations).to_csv(
        TABLES_DIR / "hydroclimate_model_configurations.csv", index=False
    )

    feature_rows = [
        {
            "model": "XGBoost",
            "target": "SPEI",
            "feature": name,
            "definition": definition,
            "leakage_control": "computed only from the pre-forecast lookback window",
        }
        for name, definition in [
            ("lag values", "18 prior scaled SPEI observations"),
            ("3-month trend", "last lag minus third-last lag"),
            ("6-month trend", "last lag minus sixth-last lag"),
            ("3-month mean", "mean of final 3 lags"),
            ("6-month mean", "mean of final 6 lags"),
            ("6-month standard deviation", "standard deviation of final 6 lags"),
        ]
    ]
    feature_rows.append(
        {
            "model": "BiLSTM/N-BEATS",
            "target": "SPEI/TWS/VPD",
            "feature": "univariate lag sequence",
            "definition": "only observations available before the forecast month",
            "leakage_control": "no contemporaneous or future covariates",
        }
    )
    pd.DataFrame(feature_rows).to_csv(
        TABLES_DIR / "hydroclimate_feature_engineering_rules.csv", index=False
    )

    rules = [
        ("Fold-local preprocessing", "Fit only through each fold train end", True),
        ("Internal validation", "Final 24 months of fold training period", True),
        ("Test-derived unit ratio", "Prohibited", True),
        ("Test-derived imputation", "Prohibited", True),
        ("Test-derived scaling/detrending", "Prohibited", True),
        ("Test-based early stopping/tuning", "Prohibited", True),
        (
            "TWS state-space sensitivity",
            "Fit aggregate state-space and pixel offsets on training dates only",
            True,
        ),
        (
            "TWS scoring",
            "Filled values may be lag inputs; only observed raster targets are scored",
            True,
        ),
        (
            "Paper anomaly CSV",
            "Prohibited as model input and not imported by run scripts",
            True,
        ),
    ]
    pd.DataFrame(
        rules, columns=["rule", "implementation", "reviewer_safe"]
    ).to_csv(TABLES_DIR / "hydroclimate_leakage_control_rules.csv", index=False)


if __name__ == "__main__":
    main()
