from __future__ import annotations

import calendar
import json
import math
import re
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from config import (
    BASELINE,
    CACHE,
    DATA_ROOT,
    FORCING,
    LAND_TYPES,
    TEST_ROOT,
    TRAIN_FOLDERS,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATE_PATTERN = re.compile(r"(20\d{2})[._-]?(?:M)?(0[1-9]|1[0-2])")

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


def parse_year_month(path: str | Path) -> pd.Timestamp | None:
    matches = DATE_PATTERN.findall(Path(path).name)
    for year, month in matches:
        year_i = int(year)
        if 2000 <= year_i <= 2025:
            return pd.Timestamp(year_i, int(month), 1)
    return None


def source_directory(land_type: str, variable: str, period: str) -> Path:
    if period == "train":
        return DATA_ROOT / TRAIN_FOLDERS[land_type] / variable
    return TEST_ROOT / land_type / variable


def discover_files(land_type: str, variable: str) -> dict[pd.Timestamp, Path]:
    discovered: dict[pd.Timestamp, Path] = {}
    for period in ("train", "test"):
        directory = source_directory(land_type, variable, period)
        if not directory.exists():
            continue
        for path in directory.glob("*.tif"):
            date = parse_year_month(path)
            if date is None:
                continue
            if date not in discovered or period == "test":
                discovered[date] = path
    return dict(sorted(discovered.items()))


def clean_array(array: np.ndarray, variable: str, nodata: float | None = None) -> np.ndarray:
    output = array.astype(np.float64, copy=True)
    if nodata is not None and np.isfinite(nodata):
        output[np.isclose(output, nodata)] = np.nan
    output[~np.isfinite(output)] = np.nan
    for invalid in (-9999, -3.4e38, 32767, -32768, 65535, -32767):
        output[np.isclose(output, invalid, rtol=0, atol=1e-5)] = np.nan
    low, high = VALID_RANGES[variable]
    output[(output < low) | (output > high)] = np.nan
    return output


def harmonize_gpp_units(array: np.ndarray, date: pd.Timestamp, path: Path) -> tuple[np.ndarray, str]:
    valid = array[np.isfinite(array)]
    if not valid.size:
        return array, "unknown"
    name = path.name.upper()
    if "MOD17" in name and np.nanmedian(valid) < 0.2:
        days = calendar.monthrange(date.year, date.month)[1]
        return array * (1000.0 / days), "MOD17 kgC m-2 month-1 to gC m-2 day-1"
    return array, "as stored"


def read_raster(path: Path, variable: str, date: pd.Timestamp | None = None) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        array = clean_array(src.read(1), variable, src.nodata)
        meta = {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
            "resolution_x": abs(src.res[0]),
            "resolution_y": abs(src.res[1]),
            "nodata": src.nodata,
        }
    unit_action = "as stored"
    if variable == "GPP" and date is not None:
        array, unit_action = harmonize_gpp_units(array, date, path)
    meta["unit_action"] = unit_action
    return array, meta


def reference_profile(land_type: str) -> dict:
    files = discover_files(land_type, "NIRv")
    path = next(iter(files.values()))
    with rasterio.open(path) as src:
        return {
            "shape": src.shape,
            "transform": src.transform,
            "crs": src.crs,
            "bounds": src.bounds,
        }


def reproject_array(
    array: np.ndarray,
    source_meta: dict,
    target_meta: dict,
    method: str = "bilinear",
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
        dst_transform=target_meta["transform"],
        dst_crs=target_meta["crs"],
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=methods[method],
    )
    return destination


def load_summary() -> pd.DataFrame:
    path = CACHE / "monthly_landtype_summary.csv"
    if not path.exists():
        raise FileNotFoundError("Run 00_build_analysis_cache.py first.")
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame


def summary_wide(statistic: str = "mean") -> pd.DataFrame:
    frame = load_summary()
    wide = frame.pivot_table(
        index=["date", "land_type"],
        columns="variable",
        values=statistic,
        aggfunc="first",
    ).sort_index()
    return wide


def monthwise_linear_fill(series: pd.Series) -> pd.Series:
    result = series.copy().astype(float)
    for month in range(1, 13):
        mask = result.index.month == month
        values = result.loc[mask]
        result.loc[mask] = values.interpolate(method="linear", limit_direction="both")
    return result


def fill_series(series: pd.Series, method: str) -> pd.Series:
    values = series.astype(float).copy()
    if method == "monthwise":
        return monthwise_linear_fill(values)
    if method == "linear":
        return values.interpolate(method="time", limit_direction="both")
    if method == "climatology":
        climatology = values.groupby(values.index.month).transform("mean")
        return values.fillna(climatology).fillna(values.mean())
    if method == "spline":
        try:
            return values.interpolate(method="spline", order=3, limit_direction="both")
        except Exception:
            return values.interpolate(method="time", limit_direction="both")
    if method == "mask":
        return values
    raise ValueError(f"Unknown fill method: {method}")


def complete_landtype_frame(
    land_type: str,
    fill_method: str = "monthwise",
    statistic: str = "mean",
) -> pd.DataFrame:
    wide = summary_wide(statistic).xs(land_type, level="land_type").copy()
    full_index = pd.date_range("2001-01-01", "2024-12-01", freq="MS")
    wide = wide.reindex(full_index)
    for column in wide.columns:
        wide[column] = fill_series(wide[column], fill_method)
    return wide


def baseline_monthly_anomaly(
    series: pd.Series,
    baseline: tuple[int, int] = BASELINE,
    standardize: bool = True,
) -> pd.Series:
    base_mask = series.index.year.astype(int)
    base = series[(base_mask >= baseline[0]) & (base_mask <= baseline[1])]
    monthly_mean = base.groupby(base.index.month).mean()
    monthly_std = base.groupby(base.index.month).std(ddof=1).replace(0, np.nan)
    anomaly = pd.Series(index=series.index, dtype=float)
    for month in range(1, 13):
        mask = series.index.month == month
        anomaly.loc[mask] = series.loc[mask] - monthly_mean.get(month, np.nan)
        if standardize:
            anomaly.loc[mask] /= monthly_std.get(month, np.nan)
    return anomaly.replace([np.inf, -np.inf], np.nan)


def standardize_training(series: pd.Series, train_mask: np.ndarray) -> tuple[pd.Series, float, float]:
    train = series.loc[train_mask]
    mean = float(train.mean())
    std = float(train.std(ddof=0))
    if not np.isfinite(std) or std == 0:
        std = 1.0
    return (series - mean) / std, mean, std


def signed_forcing_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    signed = pd.DataFrame(index=frame.index)
    for variable in FORCING:
        anomaly = baseline_monthly_anomaly(frame[variable])
        signed[variable] = anomaly if variable in ("SPEI", "TWS") else -anomaly
    return signed


def equal_weight_ewbi(frame: pd.DataFrame) -> pd.Series:
    return signed_forcing_anomalies(frame).mean(axis=1, skipna=False)


def period_label(index: pd.DatetimeIndex) -> pd.Series:
    labels = np.full(len(index), "Other", dtype=object)
    years = index.year
    labels[(years >= 2001) & (years <= 2011)] = "Baseline 2001-2011"
    labels[(years >= 2012) & (years <= 2016)] = "Historic drought 2012-2016"
    labels[(years >= 2020) & (years <= 2024)] = "Recent period 2020-2024"
    return pd.Series(labels, index=index)


def moving_block_bootstrap(
    values: np.ndarray,
    statistic=np.nanmean,
    block_length: int = 12,
    repetitions: int = 2000,
    seed: int = 42,
) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(values)
    starts = np.arange(max(1, n - block_length + 1))
    samples = np.empty(repetitions)
    for i in range(repetitions):
        pieces = []
        while sum(len(piece) for piece in pieces) < n:
            start = int(rng.choice(starts))
            pieces.append(values[start : start + block_length])
        sample = np.concatenate(pieces)[:n]
        samples[i] = statistic(sample)
    estimate = float(statistic(values))
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return estimate, float(low), float(high)


def correlation_with_ci(
    x: Iterable[float],
    y: Iterable[float],
    repetitions: int = 2000,
    block_length: int = 12,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    x = np.asarray(list(x), dtype=float)
    y = np.asarray(list(y), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 4:
        return np.nan, np.nan, np.nan, np.nan
    r, p = stats.pearsonr(x, y)
    rng = np.random.default_rng(seed)
    n = len(x)
    starts = np.arange(max(1, n - block_length + 1))
    boot = []
    for _ in range(repetitions):
        indices = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + block_length, n)))
        indices = np.asarray(indices[:n])
        if np.std(x[indices]) > 0 and np.std(y[indices]) > 0:
            boot.append(stats.pearsonr(x[indices], y[indices]).statistic)
    low, high = np.nanpercentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return float(r), float(p), float(low), float(high)


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = np.asarray(y_true)[valid], np.asarray(y_pred)[valid]
    if len(y_true) < 2:
        return {"n": len(y_true), "rmse": np.nan, "mae": np.nan, "r2": np.nan, "nse": np.nan}
    mse = mean_squared_error(y_true, y_pred)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    nse = 1 - np.sum((y_true - y_pred) ** 2) / denominator if denominator > 0 else np.nan
    return {
        "n": int(len(y_true)),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "nse": float(nse),
    }


def make_sequences(
    values: np.ndarray,
    dates: pd.DatetimeIndex,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    x, y, target_dates = [], [], []
    for i in range(lookback, len(values)):
        window = values[i - lookback : i]
        target = values[i]
        if np.all(np.isfinite(window)) and np.isfinite(target):
            x.append(window)
            y.append(target)
            target_dates.append(dates[i])
    return np.asarray(x), np.asarray(y), pd.DatetimeIndex(target_dates)


def save_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(labels_a, labels_b))
