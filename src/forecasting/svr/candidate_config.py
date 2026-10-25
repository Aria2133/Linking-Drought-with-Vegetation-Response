from pathlib import Path

PACKAGE_ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\SVR_reviewer_safe_candidate_20260722_185851"
)
AUTHORITATIVE_REVIEWER_SAFE_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_raster_reviewer_safe_candidate"
)
SOURCE_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_raster_full_candidate"
)
PREVIOUS_SVR_AUDIT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\SVR_feasibility_audit"
)

SCRIPTS_DIR = PACKAGE_ROOT / "scripts"
CONFIGS_DIR = PACKAGE_ROOT / "configs"
LOGS_DIR = PACKAGE_ROOT / "logs"
PILOT_DIR = PACKAGE_ROOT / "pilot"
PREDICTIONS_DIR = PACKAGE_ROOT / "predictions"
METRICS_DIR = PACKAGE_ROOT / "metrics"
TABLES_DIR = PACKAGE_ROOT / "tables"
FIGURES_DIR = PACKAGE_ROOT / "figures"
FIGURE_SOURCE_DATA_DIR = PACKAGE_ROOT / "figure_source_data"
AUDITS_DIR = PACKAGE_ROOT / "audits"
MANIFESTS_DIR = PACKAGE_ROOT / "manifests"
REPORTS_DIR = PACKAGE_ROOT / "reports"

# raster_pipeline.py imports CACHE_DIR. Point it at the verified read-only cache.
CACHE_DIR = SOURCE_PACKAGE / "cache"
SOURCE_TABLES_DIR = SOURCE_PACKAGE / "tables"
SOURCE_PREDICTIONS_DIR = SOURCE_PACKAGE / "predictions"
SOURCE_SCRIPTS_DIR = SOURCE_PACKAGE / "scripts"

THESIS_ROOT = Path(r"<DATA_ROOT>")
HISTORY_ROOT = THESIS_ROOT / "Data"
TEST_ROOT = THESIS_ROOT / "Test Data" / "Masked_LandTypes"
DISALLOWED_CSV_INPUT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_paper_figures_corrected\tables\paper_standardized_anomaly_series.csv"
)
GNDVI_INSTALL_MANIFEST = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\tables_revision\gndvi_fixed_installed_files.csv"
)
ORIGINAL_SCRIPT_ROOT = THESIS_ROOT / "Codes" / "Deep Leaning Codes for Forecasting"

VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
LEARNED_MODELS_EXISTING = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
SVR_MODEL_NAME = "RBF-epsilon-SVR"
CONVENTIONAL_ML_MODELS = ["XGBoost", SVR_MODEL_NAME]
DEEP_LEARNING_MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS"]
SIMPLE_BASELINES = ["Persistence", "Seasonal persistence", "Monthly climatology"]

# SARIMA is intentionally archived-only and excluded from new SVR candidate outputs.
BASELINES = SIMPLE_BASELINES
SEEDS = [42, 123, 2026]
MAIN_LOOKBACK = 12
# Preprocessing boundary for rolling-origin folds (src/forecasting/fold_boundaries.py).
# "fit_subset": imputation/detrending/scaling fitted on core-training dates only (default for new runs).
# "legacy_through_validation": fitted through the end of the internal-validation window; reproduces the
# published benchmark runs (Figure 7, Tables 4-5, statistical comparison). Outputs carry the label in
# their `preprocessing` column ("reviewer_mode_b_fitsubset" vs "reviewer_mode_b").
PREPROCESSING_BOUNDARY = "fit_subset"
MAX_PIXELS = 100
MIN_VALID_FRACTION = 0.30
TARGET_CLIP = 4.0

SVR_C_VALUES = [1.0, 3.0, 10.0]
SVR_EPSILON_VALUES = [0.05, 0.10, 0.20]
SVR_GAMMA_MULTIPLIERS = [0.5, 1.0, 2.0]
SVR_VALIDATION_NRMSE_RELATIVE_TOLERANCE = 0.01
SVR_FIT_TIMEOUT_SECONDS = 180

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

# Included for compatibility with the reused helper module. Final scripts call Mode B only.
MODE_A = {
    "name": "PROHIBITED_Mode_A_2024",
    "train_end": 202112,
    "validation_start": 202201,
    "validation_end": 202312,
    "test_start": 202401,
    "test_end": 202412,
}

ROLLING_FOLDS = [
    {"fold": 1, "train_end": 201612, "test_start": 201701, "test_end": 201812},
    {"fold": 2, "train_end": 201812, "test_start": 201901, "test_end": 202012},
    {"fold": 3, "train_end": 202012, "test_start": 202101, "test_end": 202212},
    {"fold": 4, "train_end": 202212, "test_start": 202301, "test_end": 202412},
]

ORIGINAL_SCRIPTS = {
    "LSTM": ORIGINAL_SCRIPT_ROOT / "LSTM Final.py",
    "BiLSTM": ORIGINAL_SCRIPT_ROOT / "BiLSTM.py",
    "TCN": ORIGINAL_SCRIPT_ROOT / "TCN With Uncertainty Analysis Final.py",
    "N-BEATS": ORIGINAL_SCRIPT_ROOT / "N-BEATS 2024 test.py",
    "N-HiTS": ORIGINAL_SCRIPT_ROOT / "N-HiTS 2024 test.py",
    "XGBoost": ORIGINAL_SCRIPT_ROOT / "XGBoost 2024 test.py",
}

for directory in [
    SCRIPTS_DIR,
    CONFIGS_DIR,
    LOGS_DIR,
    PILOT_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    FIGURE_SOURCE_DATA_DIR,
    AUDITS_DIR,
    MANIFESTS_DIR,
    REPORTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)
