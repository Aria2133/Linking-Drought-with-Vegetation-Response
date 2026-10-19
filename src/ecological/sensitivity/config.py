from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parent
DESKTOP = ANALYSIS_ROOT.parent
THESIS_ROOT = (
    Path.home()
    / "OneDrive"
    / "Documents"
    / "Msc Water Resources Management"
    / "Thesis"
)
DATA_ROOT = THESIS_ROOT / "Data"
TEST_ROOT_CANDIDATES = [
    THESIS_ROOT / "Test Data" / "Masked_LandTypes",
    THESIS_ROOT / "ARIA DATA" / "Test Data" / "Masked_LandTypes",
    DATA_ROOT / "Masked_LandTypes",
]
TEST_ROOT = next((path for path in TEST_ROOT_CANDIDATES if path.exists()), TEST_ROOT_CANDIDATES[0])
LAND_MASK_ROOT = DATA_ROOT / "CA_LandMasks"
RESULTS_ROOT = THESIS_ROOT / "ALL the RESULTS"

TABLES = ANALYSIS_ROOT / "tables_revision"
FIGURES = ANALYSIS_ROOT / "figures_revision"
SUPPLEMENTARY = ANALYSIS_ROOT / "supplementary_revision"
CACHE = ANALYSIS_ROOT / "cache"
LOGS = ANALYSIS_ROOT / "logs"

LAND_TYPES = ["Grasslands", "NaturalForests", "Savannas", "Shrublands"]
LAND_LABELS = {
    "Grasslands": "Grasslands",
    "NaturalForests": "Natural Forests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
TRAIN_FOLDERS = {
    "Grasslands": "CALIFORNIA GRASSLANDS",
    "NaturalForests": "CALIFORNIA NATURAL FORESTS",
    "Savannas": "CALIFORNIA SAVANNAS",
    "Shrublands": "CALIFORNIA SHRUBLANDS",
}
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
FORCING = ["SPEI", "TWS", "VPD", "LST"]
VEGETATION = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]

DATE_START = "2001-01-01"
DATE_END = "2024-12-01"
BASELINE = (2001, 2011)
HISTORIC_DROUGHT = (2012, 2016)
RECENT_DROUGHT = (2020, 2024)

ROLLING_FOLDS = [
    {"fold": 1, "train_end": 2014, "validation": (2015, 2016), "test": (2017, 2018)},
    {"fold": 2, "train_end": 2016, "validation": (2017, 2018), "test": (2019, 2020)},
    {"fold": 3, "train_end": 2018, "validation": (2019, 2020), "test": (2021, 2022)},
    {"fold": 4, "train_end": 2020, "validation": (2021, 2022), "test": (2023, 2024)},
]

NATIVE_DATASET_TABLE = [
    ["NIRv", "MODIS MOD13Q1 V6.1", "250 m", "16-day", "Monthly mean", "5 km"],
    ["NDVI", "MODIS MOD13Q1 V6.1", "250 m", "16-day", "Monthly mean", "5 km"],
    ["GNDVI", "MODIS MOD09A1 V6.1", "500 m", "8-day", "Monthly mean", "5 km"],
    ["GOSIF", "GOSIF v2", "0.05 deg (~5.5 km)", "8-day", "Monthly mean", "5 km"],
    ["GPP", "FLUXCOM / MODIS MOD17A2H", "0.0833 deg / 500 m", "Monthly / 8-day", "Monthly", "5 km"],
    ["SPEI", "GRIDMET drought", "~4.6 km", "5-day", "Monthly mean", "5 km"],
    ["TWS", "GRACE and GRACE-FO mascon", "0.5 deg (~55.7 km)", "Monthly", "Monthly", "5 km analysis grid"],
    ["VPD", "GRIDMET", "~4 km", "Daily", "Monthly mean", "5 km"],
    ["LST", "MODIS Aqua MYD21C1 V6.1", "0.05 deg (~5.6 km)", "Daily", "Monthly mean", "5 km"],
]

for directory in [TABLES, FIGURES, SUPPLEMENTARY, CACHE, LOGS]:
    directory.mkdir(parents=True, exist_ok=True)
