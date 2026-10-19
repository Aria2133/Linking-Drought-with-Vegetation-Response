"""Shared utilities for the Reviewer #1 detrending sensitivity package."""

from __future__ import annotations

import calendar
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import pearsonr, spearmanr


ROOT = Path(r"<PROJECT_ROOT>\Detrending_Sensitivity_Revision")
THESIS_ROOT = Path(r"<DATA_ROOT>")
TEST_ROOT = THESIS_ROOT / "Test Data" / "Masked_LandTypes"
LAND_FOLDERS = {
    "Grasslands": THESIS_ROOT / "Data" / "CALIFORNIA GRASSLANDS",
    "Natural Forests": THESIS_ROOT / "Data" / "CALIFORNIA NATURAL FORESTS",
    "Savannas": THESIS_ROOT / "Data" / "CALIFORNIA SAVANNAS",
    "Shrublands": THESIS_ROOT / "Data" / "CALIFORNIA SHRUBLANDS",
}
TEST_NAMES = {
    "Grasslands": "Grasslands",
    "Natural Forests": "NaturalForests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
VARIABLES = ["TWS", "SPEI", "VPD", "LST", "NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
FORECAST_TARGETS = ["TWS", "SPEI", "VPD", "LST", "NIRv", "GPP"]
VERSIONS = [
    "raw_monthly",
    "deseasonalized_only",
    "month_specific_detrended",
    "ssa_detrended",
]
DATE_INDEX = pd.date_range("2001-01-01", "2024-12-01", freq="MS")
BASELINE = (pd.Timestamp("2001-01-01"), pd.Timestamp("2011-12-01"))
PERIODS = {
    "baseline": BASELINE,
    "historic": (pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-01")),
    "recent": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-01")),
}
ROLLING_FOLDS = [
    ("2017-2018", pd.Timestamp("2017-01-01"), pd.Timestamp("2018-12-01")),
    ("2019-2020", pd.Timestamp("2019-01-01"), pd.Timestamp("2020-12-01")),
    ("2021-2022", pd.Timestamp("2021-01-01"), pd.Timestamp("2022-12-01")),
    ("2023-2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-01")),
]
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[._-]?(?:M)?(0[1-9]|1[0-2])(?!\d)", re.I)
VALID_RANGES = {
    "GNDVI": (-1.0, 1.5),
    "GOSIF": (-100.0, 3000.0),
    "GPP": (0.0, 100.0),
    "LST": (-50.0, 70.0),
    "NDVI": (-1.0, 1.5),
    "NIRv": (-0.2, 0.8),
    "SPEI": (-6.0, 6.0),
    "TWS": (-100.0, 100.0),
    "VPD": (0.0, 15.0),
}
SSA_WINDOW = 60
SSA_REMOVED_COMPONENTS = 2
SEED = 42


def ensure_directories() -> None:
    for folder in ("scripts", "tables", "figures", "report", "logs", "cache"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def configure_logging(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_date(path: Path) -> pd.Timestamp | None:
    match = DATE_RE.search(path.name)
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1) if match else None


def source_directories(land_type: str, variable: str) -> list[tuple[str, Path]]:
    return [
        ("train", LAND_FOLDERS[land_type] / variable),
        ("test", TEST_ROOT / TEST_NAMES[land_type] / variable),
    ]


def discover_files(land_type: str, variable: str) -> tuple[dict[pd.Timestamp, Path], list[str]]:
    files: dict[pd.Timestamp, Path] = {}
    warnings_out: list[str] = []
    for period, folder in source_directories(land_type, variable):
        if not folder.exists():
            warnings_out.append(f"Missing {period} folder: {folder}")
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            if "filled" in path.as_posix().lower():
                continue
            date = parse_date(path)
            if date is None:
                warnings_out.append(f"Unparsed date: {path}")
                continue
            if date not in files or period == "test":
                files[date] = path
    return files, warnings_out


def read_raster_mean(path: Path, variable: str, date: pd.Timestamp) -> tuple[float, dict]:
    with rasterio.open(path) as src:
        array = src.read(1).astype("float64")
        if src.nodata is not None and np.isfinite(src.nodata):
            array[np.isclose(array, src.nodata)] = np.nan
        array[~np.isfinite(array)] = np.nan
        for invalid in (-9999, -3.4e38, 32767, -32768, 65535, -32767):
            array[np.isclose(array, invalid, rtol=0, atol=1e-5)] = np.nan
        low, high = VALID_RANGES[variable]
        array[(array < low) | (array > high)] = np.nan
        unit_action = "as stored"
        valid = array[np.isfinite(array)]
        if variable == "GPP" and "MOD17" in path.name.upper() and valid.size:
            if float(np.nanmedian(valid)) < 0.2:
                array *= 1000.0 / calendar.monthrange(date.year, date.month)[1]
                unit_action = "MOD17 kgC m-2 month-1 to gC m-2 day-1"
                valid = array[np.isfinite(array)]
        metadata = {
            "crs": str(src.crs) if src.crs else "",
            "resolution_x": abs(float(src.res[0])),
            "resolution_y": abs(float(src.res[1])),
            "rows": int(src.height),
            "cols": int(src.width),
            "nodata": src.nodata,
            "valid_pixels": int(valid.size),
            "unit_action": unit_action,
        }
    return (float(np.mean(valid)) if valid.size else np.nan), metadata


def monthwise_fill(series: pd.Series) -> tuple[pd.Series, int]:
    """Current same-calendar-month cross-year interpolation with monthly fallback."""
    result = series.astype(float).copy()
    fallback = 0
    for month in range(1, 13):
        monthly = result[result.index.month == month]
        observed = monthly.dropna()
        if observed.empty:
            continue
        years = observed.index.year.to_numpy(dtype=float)
        values = observed.to_numpy(dtype=float)
        for date in monthly.index[monthly.isna()]:
            if len(observed) >= 2 and years.min() <= date.year <= years.max():
                result.loc[date] = np.interp(float(date.year), years, values)
            else:
                result.loc[date] = float(np.mean(values))
                fallback += 1
    if result.isna().any():
        result = result.fillna(result.mean())
    return result, fallback


def load_raw_table() -> pd.DataFrame:
    path = ROOT / "tables" / "monthly_landtype_series_raw.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run 00_inventory_and_series.py first: {path}")
    return pd.read_csv(path, parse_dates=["date"])


def analysis_series(frame: pd.DataFrame, land_type: str, variable: str) -> pd.Series:
    subset = frame[(frame["land_type"] == land_type) & (frame["variable"] == variable)]
    output = subset.set_index("date")["analysis_value"].astype(float).reindex(DATE_INDEX)
    output.index.name = "date"
    return output


def raw_observation_series(frame: pd.DataFrame, land_type: str, variable: str) -> pd.Series:
    subset = frame[(frame["land_type"] == land_type) & (frame["variable"] == variable)]
    output = subset.set_index("date")["raw_value"].astype(float).reindex(DATE_INDEX)
    output.index.name = "date"
    return output


def linear_slope_per_year(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 3:
        return np.nan
    x = valid.index.year + (valid.index.month - 1) / 12.0
    return float(np.polyfit(x.to_numpy(dtype=float), valid.to_numpy(dtype=float), 1)[0])


def diagonal_average(matrix: np.ndarray) -> np.ndarray:
    rows, cols = matrix.shape
    output = np.zeros(rows + cols - 1, dtype=float)
    counts = np.zeros_like(output)
    for col in range(cols):
        output[col : col + rows] += matrix[:, col]
        counts[col : col + rows] += 1
    return output / counts


def ssa_low_frequency_component(
    series: pd.Series,
    window: int = SSA_WINDOW,
    components: int = SSA_REMOVED_COMPONENTS,
) -> tuple[pd.Series, dict]:
    values = series.interpolate(method="time", limit_direction="both").to_numpy(dtype=float)
    n = len(values)
    window = min(window, n // 2)
    trajectory = np.column_stack([values[i : i + window] for i in range(n - window + 1)])
    u, singular, vt = np.linalg.svd(trajectory, full_matrices=False)
    rank = min(components, len(singular))
    reconstruction = (u[:, :rank] * singular[:rank]) @ vt[:rank]
    trend = diagonal_average(reconstruction)
    explained = float(np.sum(singular[:rank] ** 2) / np.sum(singular**2))
    metadata = {
        "ssa_window_months": window,
        "ssa_removed_components": rank,
        "ssa_removed_trajectory_variance_fraction": explained,
    }
    return pd.Series(trend, index=series.index), metadata


def preprocess_ecological(series: pd.Series, version: str) -> dict[str, object]:
    original = series.astype(float).copy()
    zeros = pd.Series(0.0, index=original.index)
    metadata: dict[str, object] = {
        "seasonal_reference": "",
        "trend_fit_period": "",
        "ssa_window_months": "",
        "ssa_removed_components": "",
        "ssa_removed_trajectory_variance_fraction": "",
        "warning": "",
    }
    if version == "raw_monthly":
        processed, seasonal, trend = original.copy(), zeros.copy(), zeros.copy()
    elif version == "deseasonalized_only":
        baseline = original.loc[BASELINE[0] : BASELINE[1]]
        climatology = baseline.groupby(baseline.index.month).mean()
        seasonal = pd.Series([climatology.loc[d.month] for d in original.index], index=original.index)
        trend = zeros.copy()
        processed = original - seasonal
        metadata["seasonal_reference"] = "2001-2011 calendar-month mean"
    elif version == "month_specific_detrended":
        seasonal = pd.Series(index=original.index, dtype=float)
        trend = pd.Series(index=original.index, dtype=float)
        processed = pd.Series(index=original.index, dtype=float)
        fit = original.loc["2001-01-01":"2018-12-01"]
        for month in range(1, 13):
            fit_month = fit[fit.index.month == month].dropna()
            all_month = original[original.index.month == month]
            x_fit = fit_month.index.year.to_numpy(dtype=float)
            slope, intercept = np.polyfit(x_fit, fit_month.to_numpy(dtype=float), 1)
            reference_level = slope * 2001.0 + intercept
            seasonal.loc[all_month.index] = reference_level
            trend.loc[all_month.index] = slope * (all_month.index.year.to_numpy(dtype=float) - 2001.0)
            processed.loc[all_month.index] = all_month - (
                seasonal.loc[all_month.index] + trend.loc[all_month.index]
            )
        metadata["seasonal_reference"] = "month-specific fitted intercept at 2001"
        metadata["trend_fit_period"] = "2001-2018, matching forecasting scripts"
    elif version == "ssa_detrended":
        baseline = original.loc[BASELINE[0] : BASELINE[1]]
        climatology = baseline.groupby(baseline.index.month).mean()
        seasonal = pd.Series([climatology.loc[d.month] for d in original.index], index=original.index)
        anomaly = original - seasonal
        trend, ssa_meta = ssa_low_frequency_component(anomaly)
        processed = anomaly - trend
        metadata.update(ssa_meta)
        metadata["seasonal_reference"] = "2001-2011 calendar-month mean before SSA"
        metadata["trend_fit_period"] = "2001-2024 exploratory full-series decomposition"
        metadata["warning"] = (
            "Exploratory SSA sensitivity; component grouping is not uniquely identified."
        )
    else:
        raise ValueError(f"Unknown preprocessing version: {version}")
    return {
        "processed": processed,
        "seasonal": seasonal,
        "trend": trend,
        "residual": processed,
        "metadata": metadata,
    }


def standardize_baseline(series: pd.Series) -> pd.Series:
    baseline = series.loc[BASELINE[0] : BASELINE[1]]
    mean = float(baseline.mean())
    std = float(baseline.std(ddof=1))
    return (series - mean) / std if std > 0 else series * np.nan


def comparison_metrics(reference: pd.Series, comparison: pd.Series) -> dict[str, float | int]:
    joined = pd.concat([reference.rename("reference"), comparison.rename("comparison")], axis=1).dropna()
    if len(joined) < 3:
        return {key: np.nan for key in (
            "pearson_r", "spearman_r", "rmse", "normalized_rmse", "mean_bias",
            "standard_deviation_ratio", "maximum_monthly_difference"
        )} | {"n": len(joined)}
    error = joined["comparison"] - joined["reference"]
    ref_sd = float(joined["reference"].std(ddof=1))
    return {
        "pearson_r": float(pearsonr(joined["reference"], joined["comparison"]).statistic),
        "spearman_r": float(spearmanr(joined["reference"], joined["comparison"]).statistic),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / ref_sd) if ref_sd > 0 else np.nan,
        "mean_bias": float(np.mean(error)),
        "standard_deviation_ratio": float(joined["comparison"].std(ddof=1) / ref_sd)
        if ref_sd > 0 else np.nan,
        "maximum_monthly_difference": float(np.max(np.abs(error))),
        "n": int(len(joined)),
    }


def period_differences(reference: pd.Series, comparison: pd.Series) -> dict[str, float]:
    output = {}
    for period, (start, end) in PERIODS.items():
        ref = float(reference.loc[start:end].mean())
        comp = float(comparison.loc[start:end].mean())
        output[f"{period}_reference_mean"] = ref
        output[f"{period}_comparison_mean"] = comp
        output[f"{period}_mean_difference"] = comp - ref
    return output

