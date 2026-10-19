"""Shared utilities for the Reviewer #1 gap-filling comparison package."""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import rasterio
from scipy.interpolate import CubicSpline
from scipy.stats import pearsonr, spearmanr


ROOT = Path(r"<PROJECT_ROOT>\GapFilling_Comparison_Revision")
THESIS_ROOT = Path(r"<DATA_ROOT>")
TEST_ROOT = THESIS_ROOT / "Test Data" / "Masked_LandTypes"

LAND_FOLDERS = {
    "Grasslands": THESIS_ROOT / "Data" / "CALIFORNIA GRASSLANDS",
    "Natural Forests": THESIS_ROOT / "Data" / "CALIFORNIA NATURAL FORESTS",
    "Savannas": THESIS_ROOT / "Data" / "CALIFORNIA SAVANNAS",
    "Shrublands": THESIS_ROOT / "Data" / "CALIFORNIA SHRUBLANDS",
}
TEST_LAND_NAMES = {
    "Grasslands": "Grasslands",
    "Natural Forests": "NaturalForests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
FOCUS_VARIABLES = ["TWS", "LST"]
METHODS = [
    "current_monthwise_cross_year",
    "seasonal_mean",
    "linear_time",
    "polynomial_regression",
    "spline_interpolation",
    "kalman_state_space",
]
DATE_INDEX = pd.date_range("2001-01-01", "2024-12-01", freq="MS")
PERIODS = {
    "baseline": (pd.Timestamp("2001-01-01"), pd.Timestamp("2011-12-01")),
    "historic": (pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-01")),
    "recent": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-01")),
}
SEED = 42
DATE_RE = re.compile(r"(?<!\d)(20\d{2})[._-]?(?:M)?(0[1-9]|1[0-2])(?!\d)", re.IGNORECASE)
RASTER_EXTENSIONS = {".tif", ".tiff"}
PLAUSIBLE_RANGES = {
    "GNDVI": (-1.0, 1.5),
    "GOSIF": (-100.0, 3000.0),
    "GPP": (0.0, 10000.0),
    "LST": (-80.0, 80.0),
    "NDVI": (-1.0, 1.5),
    "NIRv": (-0.5, 1.0),
    "SPEI": (-10.0, 10.0),
    "TWS": (-200.0, 200.0),
    "VPD": (0.0, 30.0),
}


def ensure_directories() -> None:
    for name in ("scripts", "tables", "figures", "report", "logs", "cache"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)


def configure_logging(log_path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_month(path: Path) -> pd.Timestamp | None:
    match = DATE_RE.search(path.stem)
    if not match:
        return None
    return pd.Timestamp(int(match.group(1)), int(match.group(2)), 1)


def source_directories(land_type: str, variable: str) -> list[tuple[str, Path]]:
    train = LAND_FOLDERS[land_type] / variable
    test = TEST_ROOT / TEST_LAND_NAMES[land_type] / variable
    return [("train", train), ("test", test)]


def discover_raw_files(land_type: str, variable: str) -> tuple[dict[pd.Timestamp, Path], list[str]]:
    """Find raw rasters, explicitly excluding prefilled derivative folders."""
    files: dict[pd.Timestamp, Path] = {}
    warnings_out: list[str] = []
    for source_period, folder in source_directories(land_type, variable):
        if not folder.exists():
            warnings_out.append(f"Missing {source_period} folder: {folder}")
            continue
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in RASTER_EXTENSIONS:
                continue
            if "filled" in path.as_posix().lower():
                continue
            date = parse_month(path)
            if date is None:
                warnings_out.append(f"Could not parse date: {path}")
                continue
            if date in files:
                preferred = path if source_period == "test" else files[date]
                warnings_out.append(
                    f"Duplicate {land_type} {variable} {date:%Y-%m}; using {preferred}"
                )
                files[date] = preferred
            else:
                files[date] = path
    return files, warnings_out


def read_raster_mean(path: Path, variable: str) -> tuple[float, dict[str, object]]:
    with rasterio.open(path) as src:
        array = src.read(1, masked=True).astype("float64")
        data = np.asarray(array.filled(np.nan), dtype=float)
        mask = ~np.isfinite(data)
        mask |= np.abs(data) > 1.0e20
        low, high = PLAUSIBLE_RANGES[variable]
        mask |= (data < low) | (data > high)
        valid = data[~mask]
        mean = float(np.mean(valid)) if valid.size else np.nan
        metadata = {
            "crs": str(src.crs) if src.crs else "",
            "resolution_x": abs(float(src.transform.a)),
            "resolution_y": abs(float(src.transform.e)),
            "rows": int(src.height),
            "cols": int(src.width),
            "nodata": src.nodata,
            "valid_pixel_count": int(valid.size),
        }
    return mean, metadata


def monthly_summary_cache_path() -> Path:
    return ROOT / "cache" / "raw_monthly_landtype_summary.csv"


def load_monthly_summary() -> pd.DataFrame:
    path = monthly_summary_cache_path()
    if not path.exists():
        raise FileNotFoundError(f"Run 00_gap_inventory.py first: {path}")
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame


def series_for(frame: pd.DataFrame, land_type: str, variable: str) -> pd.Series:
    subset = frame[(frame["land_type"] == land_type) & (frame["variable"] == variable)]
    series = subset.set_index("date")["mean_value"].astype(float).reindex(DATE_INDEX)
    series.index.name = "date"
    return series


def _seasonal_fill(series: pd.Series) -> pd.Series:
    result = series.copy()
    seasonal = series.groupby(series.index.month).mean()
    for date in result.index[result.isna()]:
        result.loc[date] = seasonal.get(date.month, np.nan)
    if result.isna().any():
        result = result.fillna(series.mean())
    return result


def _current_monthwise(series: pd.Series) -> tuple[pd.Series, int]:
    result = series.copy()
    fallback_count = 0
    for month in range(1, 13):
        mask = result.index.month == month
        monthly = result.loc[mask]
        observed = monthly.dropna()
        if observed.empty:
            continue
        x_obs = observed.index.year.to_numpy(dtype=float)
        y_obs = observed.to_numpy(dtype=float)
        for date in monthly.index[monthly.isna()]:
            if x_obs.min() <= date.year <= x_obs.max() and len(observed) >= 2:
                result.loc[date] = np.interp(float(date.year), x_obs, y_obs)
            else:
                result.loc[date] = float(np.mean(y_obs))
                fallback_count += 1
    remaining = int(result.isna().sum())
    if remaining:
        result = _seasonal_fill(result)
        fallback_count += remaining
    return result, fallback_count


def _linear_time(series: pd.Series) -> tuple[pd.Series, int]:
    result = series.interpolate(method="time", limit_area="inside")
    fallback_count = int(result.isna().sum())
    if fallback_count:
        result = _seasonal_fill(result)
    return result, fallback_count


def _polynomial_regression(series: pd.Series, degree: int = 2) -> tuple[pd.Series, int, str]:
    n = len(series)
    t = np.linspace(-1.0, 1.0, n)
    month = series.index.month.to_numpy()
    columns = [np.ones(n)]
    columns.extend(t**power for power in range(1, degree + 1))
    columns.extend((month == value).astype(float) for value in range(2, 13))
    design = np.column_stack(columns)
    observed = series.notna().to_numpy()
    if int(observed.sum()) < design.shape[1] + 5:
        return _seasonal_fill(series), int(series.isna().sum()), "insufficient observations"
    ridge = 1.0e-6 * np.eye(design.shape[1])
    ridge[0, 0] = 0.0
    coefficients = np.linalg.solve(
        design[observed].T @ design[observed] + ridge,
        design[observed].T @ series.to_numpy()[observed],
    )
    predictions = design @ coefficients
    result = series.copy()
    result.loc[result.isna()] = predictions[result.isna().to_numpy()]
    return result, 0, ""


def _spline(series: pd.Series) -> tuple[pd.Series, int, str]:
    x = np.arange(len(series), dtype=float)
    observed = series.notna().to_numpy()
    if int(observed.sum()) < 8:
        return _seasonal_fill(series), int(series.isna().sum()), "insufficient observations"
    spline = CubicSpline(x[observed], series.to_numpy()[observed], bc_type="natural", extrapolate=False)
    predictions = spline(x)
    result = series.copy()
    missing = result.isna().to_numpy()
    result.iloc[np.flatnonzero(missing)] = predictions[missing]
    fallback_count = int(result.isna().sum())
    if fallback_count:
        result = _seasonal_fill(result)
    return result, fallback_count, ""


def _kalman(series: pd.Series) -> tuple[pd.Series, int, str]:
    try:
        from statsmodels.tsa.statespace.structural import UnobservedComponents

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = UnobservedComponents(
                series.astype(float),
                level="local linear trend",
                seasonal=12,
                irregular=True,
            )
            fitted = model.fit(disp=False, maxiter=300)
            try:
                predictions = fitted.get_prediction(information_set="smoothed").predicted_mean
            except (TypeError, ValueError, NotImplementedError):
                predictions = fitted.get_prediction().predicted_mean
        result = series.copy()
        missing = result.isna()
        result.loc[missing] = np.asarray(predictions)[missing.to_numpy()]
        fallback_count = int(result.isna().sum())
        if fallback_count:
            result = _seasonal_fill(result)
        warning_text = "; ".join(str(item.message) for item in caught)
        if not bool(fitted.mle_retvals.get("converged", True)):
            warning_text = (warning_text + "; optimizer did not converge").strip("; ")
        return result, fallback_count, warning_text
    except Exception as exc:
        return _seasonal_fill(series), int(series.isna().sum()), f"state-space failed: {exc}"


def fill_series(series: pd.Series, method: str) -> tuple[pd.Series, dict[str, object]]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    start = perf_counter()
    original = series.astype(float).copy()
    warning_text = ""
    fallback_count = 0
    if method == "current_monthwise_cross_year":
        filled, fallback_count = _current_monthwise(original)
    elif method == "seasonal_mean":
        filled = _seasonal_fill(original)
    elif method == "linear_time":
        filled, fallback_count = _linear_time(original)
    elif method == "polynomial_regression":
        filled, fallback_count, warning_text = _polynomial_regression(original, degree=2)
    elif method == "spline_interpolation":
        filled, fallback_count, warning_text = _spline(original)
    else:
        filled, fallback_count, warning_text = _kalman(original)

    filled.loc[original.notna()] = original.loc[original.notna()]
    imputed = filled.loc[original.isna()]
    observed = original.dropna()
    observed_sd = float(observed.std(ddof=1)) if len(observed) > 1 else np.nan
    lower = float(observed.min() - 3.0 * observed_sd) if np.isfinite(observed_sd) else -np.inf
    upper = float(observed.max() + 3.0 * observed_sd) if np.isfinite(observed_sd) else np.inf
    unstable_count = int(((imputed < lower) | (imputed > upper) | ~np.isfinite(imputed)).sum())
    unstable = unstable_count > 0 or filled.isna().any()
    if unstable_count:
        extra = f"{unstable_count} imputed values outside observed range plus/minus 3 SD"
        warning_text = f"{warning_text}; {extra}".strip("; ")
    diagnostics = {
        "method": method,
        "observed_count": int(original.notna().sum()),
        "missing_count": int(original.isna().sum()),
        "filled_count": int(original.isna().sum() - filled.isna().sum()),
        "remaining_missing": int(filled.isna().sum()),
        "fallback_count": int(fallback_count),
        "imputed_min": float(imputed.min()) if len(imputed) else np.nan,
        "imputed_max": float(imputed.max()) if len(imputed) else np.nan,
        "observed_min": float(observed.min()) if len(observed) else np.nan,
        "observed_max": float(observed.max()) if len(observed) else np.nan,
        "unstable": bool(unstable),
        "unstable_count": int(unstable_count),
        "warning": warning_text,
        "runtime_seconds": round(perf_counter() - start, 4),
    }
    return filled, diagnostics


def comparison_metrics(reference: pd.Series, comparison: pd.Series) -> dict[str, float | int]:
    joined = pd.concat([reference.rename("reference"), comparison.rename("comparison")], axis=1).dropna()
    if len(joined) < 3:
        return {
            "pearson_r": np.nan,
            "spearman_r": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "mean_bias": np.nan,
            "normalized_rmse": np.nan,
            "standard_deviation_ratio": np.nan,
            "n": int(len(joined)),
        }
    error = joined["comparison"] - joined["reference"]
    ref_sd = float(joined["reference"].std(ddof=1))
    return {
        "pearson_r": float(pearsonr(joined["reference"], joined["comparison"]).statistic),
        "spearman_r": float(spearmanr(joined["reference"], joined["comparison"]).statistic),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mean_bias": float(np.mean(error)),
        "normalized_rmse": float(np.sqrt(np.mean(error**2)) / ref_sd) if ref_sd > 0 else np.nan,
        "standard_deviation_ratio": float(joined["comparison"].std(ddof=1) / ref_sd)
        if ref_sd > 0
        else np.nan,
        "n": int(len(joined)),
    }


def trend_slope(series: pd.Series) -> float:
    valid = series.dropna()
    if len(valid) < 3:
        return np.nan
    x = (valid.index.year + (valid.index.month - 1) / 12.0).to_numpy(dtype=float)
    return float(np.polyfit(x, valid.to_numpy(dtype=float), 1)[0])


def period_mean_differences(reference: pd.Series, comparison: pd.Series) -> dict[str, float]:
    output: dict[str, float] = {}
    for name, (start, end) in PERIODS.items():
        ref_mean = float(reference.loc[start:end].mean())
        cmp_mean = float(comparison.loc[start:end].mean())
        output[f"{name}_reference_mean"] = ref_mean
        output[f"{name}_comparison_mean"] = cmp_mean
        output[f"{name}_mean_difference"] = cmp_mean - ref_mean
    return output


def baseline_monthly_zscore(series: pd.Series) -> pd.Series:
    baseline = series.loc[PERIODS["baseline"][0] : PERIODS["baseline"][1]]
    monthly_mean = baseline.groupby(baseline.index.month).mean()
    monthly_sd = baseline.groupby(baseline.index.month).std(ddof=1).replace(0.0, np.nan)
    output = pd.Series(index=series.index, dtype=float)
    for date, value in series.items():
        output.loc[date] = (value - monthly_mean.get(date.month, np.nan)) / monthly_sd.get(
            date.month, np.nan
        )
    return output


def monthly_median_anomaly(series: pd.Series) -> pd.Series:
    medians = series.groupby(series.index.month).median()
    return pd.Series(
        [value - medians.get(date.month, np.nan) for date, value in series.items()],
        index=series.index,
        dtype=float,
    )


METHOD_DESCRIPTIONS = {
    "current_monthwise_cross_year": (
        "Linear interpolation among observed values for the same calendar month across adjacent "
        "years; same-month climatological mean is used at unsupported boundaries."
    ),
    "seasonal_mean": "Long-term mean of the corresponding calendar month.",
    "linear_time": (
        "Linear interpolation on the continuous monthly time axis; same-month mean is used for "
        "unsupported leading or trailing gaps."
    ),
    "polynomial_regression": (
        "Degree-2 time trend with monthly indicator terms, fit by lightly regularized least squares."
    ),
    "spline_interpolation": (
        "Natural cubic spline through observed months; same-month mean is used outside the observed range."
    ),
    "kalman_state_space": (
        "Unobserved-components state-space model with local linear trend, annual seasonality, and irregular error."
    ),
}
