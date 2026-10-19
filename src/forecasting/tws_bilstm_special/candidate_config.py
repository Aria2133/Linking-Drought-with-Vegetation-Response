from pathlib import Path

PACKAGE_ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_hydroclimate_special_candidate"
)
REVIEWER_SAFE_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_raster_reviewer_safe_candidate"
)
RASTER_FULL_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_raster_full_candidate"
)
GAP_FILL_PACKAGE = Path(
    r"<PROJECT_ROOT>\GapFilling_Comparison_Revision"
)

SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
TABLES_DIR = PACKAGE_ROOT / "tables"
FIGURES_DIR = PACKAGE_ROOT / "figures"
REPORT_DIR = PACKAGE_ROOT / "report"
LOGS_DIR = PACKAGE_ROOT / "logs"
OUTPUT_CACHE_DIR = PACKAGE_ROOT / "cache"
MODELS_DIR = PACKAGE_ROOT / "models"
PREDICTIONS_DIR = PACKAGE_ROOT / "predictions"

# raster_pipeline.py imports CACHE_DIR. This points to the verified read-only
# task cache from the raster candidate package.
CACHE_DIR = RASTER_FULL_PACKAGE / "cache"
SOURCE_TABLES_DIR = RASTER_FULL_PACKAGE / "tables"
SOURCE_PREDICTIONS_DIR = RASTER_FULL_PACKAGE / "predictions"
SOURCE_SCRIPTS_DIR = RASTER_FULL_PACKAGE / "scripts"

THESIS_ROOT = Path(r"<DATA_ROOT>")
HISTORY_ROOT = THESIS_ROOT / "Data"
TEST_ROOT = THESIS_ROOT / "Test Data" / "Masked_LandTypes"
ORIGINAL_SCRIPT_ROOT = THESIS_ROOT / "Codes" / "Deep Leaning Codes for Forecasting"
DISALLOWED_CSV_INPUT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_paper_figures_corrected\tables\paper_standardized_anomaly_series.csv"
)

TARGETS = ["SPEI", "TWS", "VPD"]
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
SEEDS = [42, 123, 2026]
MAIN_LOOKBACK = 12
MAX_PIXELS = 100
MIN_VALID_FRACTION = 0.30
TARGET_CLIP = 4.0

LAND_CONFIG = {
    "GRASSLANDS": {
        "history": "CALIFORNIA GRASSLANDS",
        "test": "Grasslands",
        "file_tag": "Grasslands",
    },
    "NATURAL_FORESTS": {
        "history": "CALIFORNIA NATURAL FORESTS",
        "test": "NaturalForests",
        "file_tag": "NaturalForests",
    },
    "SAVANNAS": {
        "history": "CALIFORNIA SAVANNAS",
        "test": "Savannas",
        "file_tag": "Savannas",
    },
    "SHRUBLANDS": {
        "history": "CALIFORNIA SHRUBLANDS",
        "test": "Shrublands",
        "file_tag": "Shrublands",
    },
}

ROLLING_FOLDS = [
    {"fold": 1, "train_end": 201612, "test_start": 201701, "test_end": 201812},
    {"fold": 2, "train_end": 201812, "test_start": 201901, "test_end": 202012},
    {"fold": 3, "train_end": 202012, "test_start": 202101, "test_end": 202212},
    {"fold": 4, "train_end": 202212, "test_start": 202301, "test_end": 202412},
]

# Included only for compatibility with the copied raster helper. This package
# never runs Mode A.
MODE_A = {
    "name": "PROHIBITED_Mode_A_2024",
    "train_end": 202112,
    "validation_start": 202201,
    "validation_end": 202312,
    "test_start": 202401,
    "test_end": 202412,
}

ORIGINAL_SCRIPTS = {
    "BiLSTM_special": ORIGINAL_SCRIPT_ROOT / "BiLSTM for Meteorlogical data.py",
    "XGBoost_special": (
        ORIGINAL_SCRIPT_ROOT / "XGBoost 2024 Test only (SPEI, TWS, VPD, LST).py"
    ),
    "NBEATS_SPEI_special": ORIGINAL_SCRIPT_ROOT / "N-BEATS SPEI.py",
    "ARIMA_reference": ORIGINAL_SCRIPT_ROOT / "ARIMA Final.py",
}

BILSTM_CONFIG = {
    "SPEI": {
        "lookback": 24,
        "units": 200,
        "layers": 2,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs_original": 500,
        "epochs_cap": 30,
        "learning_rate": 0.001,
        "patience": 7,
        "bidirectional": True,
    },
    "TWS": {
        "lookback": 18,
        "units": 150,
        "layers": 2,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs_original": 300,
        "epochs_cap": 30,
        "learning_rate": 0.001,
        "patience": 7,
        "bidirectional": True,
    },
    "VPD": {
        "lookback": 12,
        "units": 128,
        "layers": 2,
        "dropout": 0.2,
        "batch_size": 64,
        "epochs_original": 300,
        "epochs_cap": 30,
        "learning_rate": 0.001,
        "patience": 7,
        "bidirectional": False,
    },
}

XGBOOST_BASE_CONFIG = {
    "lookback": 12,
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.03,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 1,
    "gamma": 0.1,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "early_stopping_rounds": 30,
}
XGBOOST_SPEI_CONFIG = {
    **XGBOOST_BASE_CONFIG,
    "lookback": 18,
    "n_estimators": 600,
    "max_depth": 10,
    "learning_rate": 0.02,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.3,
    "reg_lambda": 1.5,
}

NBEATS_SPEI_CONFIG = {
    "lookback": 6,
    "stack_types": ["trend", "seasonality"],
    "blocks_per_stack": 3,
    "theta_dims": [4, 8],
    "hidden_units": 64,
    "harmonics": 3,
    "batch_size": 32,
    "epochs_original": 40,
    "epochs_cap": 40,
    "learning_rate": 0.001,
    "weight_decay": 0.001,
    "dropout_documented_but_unused": 0.4,
    "grad_clip": 1.0,
    "patience": 10,
    "loss": "Huber",
    "ema_decay": 0.995,
    "warmup_epochs": 10,
    "target_clip": 3.0,
}

for directory in [
    SCRIPTS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    REPORT_DIR,
    LOGS_DIR,
    OUTPUT_CACHE_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
