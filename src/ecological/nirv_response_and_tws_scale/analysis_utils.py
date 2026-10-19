from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling, reproject
from scipy import stats


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ANALYSIS_ROOT / "scripts"
TABLES = ANALYSIS_ROOT / "tables"
FIGURES = ANALYSIS_ROOT / "figures"
REPORT = ANALYSIS_ROOT / "report"
CACHE = ANALYSIS_ROOT / "cache"
LOGS = ANALYSIS_ROOT / "logs"

THESIS_ROOT = (
    Path.home()
    / "OneDrive"
    / "Documents"
    / "Msc Water Resources Management"
    / "Thesis"
)
DATA_ROOT = THESIS_ROOT / "Data"
TEST_ROOT = THESIS_ROOT / "Test Data"
TEST_MASKED_ROOT = TEST_ROOT / "Masked_LandTypes"
LAND_MASK_ROOT = DATA_ROOT / "CA_LandMasks"

LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_KEYS = {
    "Grasslands": "Grasslands",
    "Natural Forests": "NaturalForests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
TRAIN_FOLDERS = {
    "Grasslands": "CALIFORNIA GRASSLANDS",
    "Natural Forests": "CALIFORNIA NATURAL FORESTS",
    "Savannas": "CALIFORNIA SAVANNAS",
    "Shrublands": "CALIFORNIA SHRUBLANDS",
}
LAND_MASK_FILES = {
    "Grasslands": "CA_Grasslands_2021.tif",
    "Natural Forests": "CA_NaturalForests_2021.tif",
    "Savannas": "CA_Savannas_2021.tif",
    "Shrublands": "CA_Shrublands_2021.tif",
}
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]

BASELINE = (2001, 2011)
HISTORIC = (2012, 2016)
RECENT = (2020, 2024)
PERIODS = {
    "baseline_2001_2011": BASELINE,
    "historic_2012_2016": HISTORIC,
    "recent_2020_2024": RECENT,
}

DATE_PATTERN = re.compile(
    r"(?<!\d)(20\d{2})[._-]?(?:M)?(0[1-9]|1[0-2])(?!\d)",
    flags=re.IGNORECASE,
)


def ensure_output_directories() -> None:
    for directory in (SCRIPTS, TABLES, FIGURES, REPORT, CACHE, LOGS):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logger(name: str, filename: str) -> logging.Logger:
    ensure_output_directories()
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(LOGS / filename, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def parse_year_month(path: str | Path) -> pd.Timestamp | None:
    match = DATE_PATTERN.search(Path(path).name)
    if not match:
        return None
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def discover_land_files(land_type: str, variable: str) -> dict[pd.Timestamp, Path]:
    files: dict[pd.Timestamp, Path] = {}
    train = DATA_ROOT / TRAIN_FOLDERS[land_type] / variable
    test = TEST_MASKED_ROOT / LAND_KEYS[land_type] / variable
    for directory, prefer in ((train, False), (test, True)):
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.tif")):
            date = parse_year_month(path)
            if date is None:
                continue
            if date not in files or prefer:
                files[date] = path
    return dict(sorted(files.items()))


def discover_tws_source_files() -> dict[pd.Timestamp, Path]:
    files: dict[pd.Timestamp, Path] = {}
    sources = [
        (DATA_ROOT / "CALIFORNIA INDICES" / "TWS_GRACE_TIFs", False),
        (TEST_ROOT / "TWS", True),
    ]
    for directory, prefer in sources:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.tif")):
            date = parse_year_month(path)
            if date is None:
                continue
            if date not in files or prefer:
                files[date] = path
    return dict(sorted(files.items()))


def land_mask_path(land_type: str) -> Path:
    return LAND_MASK_ROOT / LAND_MASK_FILES[land_type]


def read_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        masked = src.read(1, masked=True)
        array = np.asarray(masked.filled(np.nan), dtype=np.float64)
        for sentinel in (-9999.0, -32768.0, 32767.0, 65535.0, -3.4e38):
            array[np.isclose(array, sentinel, rtol=0.0, atol=1e-6)] = np.nan
        array[~np.isfinite(array)] = np.nan
        metadata = {
            "shape": src.shape,
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
            "resolution_x": abs(src.res[0]),
            "resolution_y": abs(src.res[1]),
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "count": src.count,
        }
    return array, metadata


def raster_signature(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "resolution_x": abs(src.res[0]),
            "resolution_y": abs(src.res[1]),
            "height": src.height,
            "width": src.width,
            "shape": f"{src.height}x{src.width}",
            "nodata": src.nodata,
            "dtype": src.dtypes[0],
            "bounds": (
                float(src.bounds.left),
                float(src.bounds.bottom),
                float(src.bounds.right),
                float(src.bounds.top),
            ),
        }


def grid_from_path(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {
            "shape": src.shape,
            "height": src.height,
            "width": src.width,
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
        }


def nirv_reference_grid(land_type: str) -> dict:
    files = discover_land_files(land_type, "NIRv")
    if not files:
        raise FileNotFoundError(f"No NIRv files found for {land_type}.")
    return grid_from_path(next(iter(files.values())))


def grids_match(source_meta: dict, target_meta: dict, tolerance: float = 1e-10) -> bool:
    if tuple(source_meta["shape"]) != tuple(target_meta["shape"]):
        return False
    if CRS.from_user_input(source_meta["crs"]) != CRS.from_user_input(target_meta["crs"]):
        return False
    return bool(
        np.allclose(
            tuple(source_meta["transform"]),
            tuple(target_meta["transform"]),
            rtol=0.0,
            atol=tolerance,
        )
    )


def reproject_array(
    array: np.ndarray,
    source_meta: dict,
    target_meta: dict,
    method: str,
) -> np.ndarray:
    methods = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
    }
    destination = np.full(target_meta["shape"], np.nan, dtype=np.float64)
    reproject(
        source=array,
        destination=destination,
        src_transform=source_meta["transform"],
        src_crs=source_meta["crs"],
        src_nodata=np.nan,
        dst_transform=target_meta["transform"],
        dst_crs=target_meta["crs"],
        dst_nodata=np.nan,
        resampling=methods[method],
        init_dest_nodata=True,
    )
    return destination


def fractional_land_mask(land_type: str, target_meta: dict) -> np.ndarray:
    path = land_mask_path(land_type)
    with rasterio.open(path) as src:
        source = src.read(1).astype(np.float64)
        destination = np.full(target_meta["shape"], np.nan, dtype=np.float64)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_meta["transform"],
            dst_crs=target_meta["crs"],
            resampling=Resampling.average,
            dst_nodata=np.nan,
            init_dest_nodata=True,
        )
    destination = np.clip(destination, 0.0, 1.0)
    destination[~np.isfinite(destination)] = 0.0
    return destination


def cell_area_km2(meta: dict) -> np.ndarray:
    transform = meta["transform"]
    rows, cols = meta["shape"]
    crs = CRS.from_user_input(meta["crs"])
    if crs.is_projected:
        area = abs(transform.a * transform.e - transform.b * transform.d) / 1_000_000.0
        return np.full((rows, cols), area, dtype=np.float64)

    radius_km = 6371.0088
    dlon = math.radians(abs(transform.a))
    row_areas = np.empty(rows, dtype=np.float64)
    for row in range(rows):
        north = transform.f + row * transform.e
        south = north + transform.e
        row_areas[row] = radius_km**2 * dlon * abs(
            math.sin(math.radians(north)) - math.sin(math.radians(south))
        )
    return np.repeat(row_areas[:, None], cols, axis=1)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def period_mask(dates: Iterable[pd.Timestamp], years: tuple[int, int]) -> np.ndarray:
    index = pd.DatetimeIndex(dates)
    return (index.year >= years[0]) & (index.year <= years[1])


def linear_slope_per_year(dates: Iterable[pd.Timestamp], values: Iterable[float]) -> float:
    index = pd.DatetimeIndex(dates)
    y = np.asarray(list(values), dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return np.nan
    x = (index.year.to_numpy() - index.year.min()) + (index.month.to_numpy() - 1) / 12.0
    return float(stats.linregress(x[valid], y[valid]).slope)


def comparison_metrics(
    dates: Iterable[pd.Timestamp],
    candidate: Iterable[float],
    reference: Iterable[float],
) -> dict[str, float]:
    index = pd.DatetimeIndex(dates)
    x = np.asarray(list(candidate), dtype=float)
    y = np.asarray(list(reference), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    xv = x[valid]
    yv = y[valid]
    if len(xv) < 3:
        return {
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mean_bias": np.nan,
            "standard_deviation_ratio": np.nan,
            "normalized_rmse_by_reference_sd": np.nan,
            "trend_slope_candidate_per_year": np.nan,
            "trend_slope_reference_per_year": np.nan,
            "trend_slope_difference_per_year": np.nan,
            "maximum_absolute_monthly_difference": np.nan,
            "number_valid_months": int(len(xv)),
        }
    difference = xv - yv
    reference_sd = float(np.std(yv, ddof=1))
    candidate_sd = float(np.std(xv, ddof=1))
    candidate_slope = linear_slope_per_year(index[valid], xv)
    reference_slope = linear_slope_per_year(index[valid], yv)
    return {
        "pearson_r": float(stats.pearsonr(xv, yv).statistic),
        "spearman_r": float(stats.spearmanr(xv, yv).statistic),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mae": float(np.mean(np.abs(difference))),
        "mean_bias": float(np.mean(difference)),
        "standard_deviation_ratio": (
            candidate_sd / reference_sd if reference_sd > 0 else np.nan
        ),
        "normalized_rmse_by_reference_sd": (
            float(np.sqrt(np.mean(difference**2))) / reference_sd
            if reference_sd > 0
            else np.nan
        ),
        "trend_slope_candidate_per_year": candidate_slope,
        "trend_slope_reference_per_year": reference_slope,
        "trend_slope_difference_per_year": candidate_slope - reference_slope,
        "maximum_absolute_monthly_difference": float(np.max(np.abs(difference))),
        "number_valid_months": int(len(xv)),
    }


def period_mean_differences(
    dates: Iterable[pd.Timestamp],
    candidate: Iterable[float],
    reference: Iterable[float],
) -> dict[str, float]:
    index = pd.DatetimeIndex(dates)
    x = np.asarray(list(candidate), dtype=float)
    y = np.asarray(list(reference), dtype=float)
    output: dict[str, float] = {}
    for label, years in PERIODS.items():
        mask = period_mask(index, years) & np.isfinite(x) & np.isfinite(y)
        output[f"mean_difference_{label}"] = (
            float(np.mean(x[mask] - y[mask])) if mask.any() else np.nan
        )
        output[f"valid_months_{label}"] = int(mask.sum())
    return output


def dataframe_to_markdown(frame: pd.DataFrame, float_format: str = ".3f") -> str:
    if frame.empty:
        return "_No records available._"
    return frame.to_markdown(index=False, floatfmt=float_format)


def nominal_resolution_km(meta: dict) -> float:
    crs = CRS.from_user_input(meta["crs"])
    transform = meta["transform"]
    if crs.is_projected:
        return float(np.sqrt(abs(transform.a * transform.e - transform.b * transform.d)) / 1000)
    center_lat = meta["bounds"].bottom + (
        meta["bounds"].top - meta["bounds"].bottom
    ) / 2
    x_km = abs(transform.a) * 111.320 * math.cos(math.radians(center_lat))
    y_km = abs(transform.e) * 110.574
    return float(math.sqrt(x_km * y_km))

