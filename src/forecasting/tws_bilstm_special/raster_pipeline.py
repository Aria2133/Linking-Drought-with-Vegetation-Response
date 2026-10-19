import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import zoom

from candidate_config import (
    CACHE_DIR,
    HISTORY_ROOT,
    LAND_CONFIG,
    MAIN_LOOKBACK,
    MAX_PIXELS,
    MIN_VALID_FRACTION,
    TARGET_CLIP,
    TEST_ROOT,
)

DATE_RE = re.compile(r"(?<!\d)(20\d{2})[._-]?[Mm]?(\d{2})(?!\d)")


@dataclass
class RasterMeta:
    path: Path
    height: int
    width: int
    bounds: Tuple[float, float, float, float]
    transform: object
    crs: str
    nodata: Optional[float]

    @property
    def area(self) -> float:
        left, bottom, right, top = self.bounds
        return abs((right - left) * (top - bottom))


def parse_date(path: Path) -> Optional[int]:
    matches = DATE_RE.findall(path.stem)
    for year, month in reversed(matches):
        month_i = int(month)
        if 1 <= month_i <= 12:
            return int(year) * 100 + month_i
    return None


def expected_start(variable: str) -> int:
    if variable == "LST":
        return 200301
    if variable == "TWS":
        return 200204
    return 200101


def month_range(start: int, end: int) -> List[int]:
    year, month = divmod(start, 100)
    out = []
    while year * 100 + month <= end:
        out.append(year * 100 + month)
        month += 1
        if month == 13:
            year += 1
            month = 1
    return out


def source_directories(land_type: str, variable: str) -> Tuple[Path, Path]:
    cfg = LAND_CONFIG[land_type]
    history = HISTORY_ROOT / cfg["history"] / variable
    test = TEST_ROOT / cfg["test"] / variable
    return history, test


def discover_files(land_type: str, variable: str) -> pd.DataFrame:
    history_dir, test_dir = source_directories(land_type, variable)
    rows = []
    for period, directory in [("historical", history_dir), ("test", test_dir)]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.tif")):
            date = parse_date(path)
            if date is None:
                continue
            rows.append(
                {
                    "land_type": land_type,
                    "variable": variable,
                    "period": period,
                    "date": date,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "mtime_utc": pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat(),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["date", "period", "path"]).drop_duplicates("date", keep="last")
    return frame.reset_index(drop=True)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: str(item).lower()):
        stat = path.stat()
        record = f"{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def read_meta(path: Path) -> RasterMeta:
    with rasterio.open(path) as src:
        return RasterMeta(
            path=path,
            height=src.height,
            width=src.width,
            bounds=(src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top),
            transform=src.transform,
            crs=str(src.crs),
            nodata=src.nodata,
        )


def choose_target_meta(paths: Sequence[Path]) -> RasterMeta:
    if not paths:
        raise ValueError("No raster paths supplied")
    candidates = [read_meta(paths[0])]
    for path in paths[1:]:
        meta = read_meta(path)
        if meta.area < candidates[0].area:
            candidates[0] = meta
    return candidates[0]


def _old_style_crop_window(src, target: RasterMeta) -> Optional[Window]:
    src_area = abs((src.bounds.right - src.bounds.left) * (src.bounds.top - src.bounds.bottom))
    if src_area <= target.area * 1.01:
        return None
    res_x = abs(src.transform.a)
    res_y = abs(src.transform.e)
    left, bottom, right, top = target.bounds
    col_start = max(0, int((left - src.bounds.left) / res_x))
    col_end = min(src.width, int((right - src.bounds.left) / res_x))
    row_start = max(0, int((src.bounds.top - top) / res_y))
    row_end = min(src.height, int((src.bounds.top - bottom) / res_y))
    if col_end <= col_start or row_end <= row_start:
        return None
    return Window(col_start, row_start, col_end - col_start, row_end - row_start)


def read_aligned(path: Path, target: RasterMeta) -> np.ndarray:
    with rasterio.open(path) as src:
        window = _old_style_crop_window(src, target)
        data = src.read(1, window=window).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[np.isclose(data, nodata)] = np.nan
        data[~np.isfinite(data)] = np.nan
        data[np.abs(data) > 1e20] = np.nan
    if data.shape != (target.height, target.width):
        factors = (target.height / data.shape[0], target.width / data.shape[1])
        valid = np.isfinite(data).astype(np.float32)
        filled = np.nan_to_num(data, nan=0.0)
        numerator = zoom(filled, factors, order=1, prefilter=False)
        denominator = zoom(valid, factors, order=1, prefilter=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            data = numerator / denominator
        data[denominator < 0.25] = np.nan
        data = data[: target.height, : target.width]
        if data.shape != (target.height, target.width):
            padded = np.full((target.height, target.width), np.nan, dtype=np.float32)
            padded[: data.shape[0], : data.shape[1]] = data
            data = padded
    return data.astype(np.float32)


def _coverage_mask(stack: np.ndarray, dates: np.ndarray) -> np.ndarray:
    train = dates <= 202112
    test = (dates >= 202401) & (dates <= 202412)
    train_required = max(1, int(math.floor(train.sum() * MIN_VALID_FRACTION)))
    test_required = max(1, int(math.floor(test.sum() * MIN_VALID_FRACTION)))
    train_ok = np.isfinite(stack[train]).sum(axis=0) >= train_required
    test_ok = np.isfinite(stack[test]).sum(axis=0) >= test_required
    return train_ok & test_ok


def build_task_cache(land_type: str, variable: str, overwrite: bool = False) -> Dict:
    output = CACHE_DIR / "task_data" / f"{land_type}_{variable}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        cached = np.load(output, allow_pickle=False)
        return json.loads(str(cached["metadata_json"]))

    files = discover_files(land_type, variable)
    if files.empty:
        raise FileNotFoundError(f"No files found for {land_type} {variable}")
    paths = [Path(value) for value in files["path"]]
    target = choose_target_meta([paths[0], paths[-1]])
    dates = files["date"].to_numpy(dtype=np.int32)
    stack = np.empty((len(paths), target.height, target.width), dtype=np.float32)
    for index, path in enumerate(paths):
        stack[index] = read_aligned(path, target)

    mask = _coverage_mask(stack, dates)
    flat_indices = np.flatnonzero(mask.ravel())
    if len(flat_indices) == 0:
        raise RuntimeError(f"No pixels passed coverage for {land_type} {variable}")
    rng = np.random.default_rng(42)
    if len(flat_indices) > MAX_PIXELS:
        flat_indices = np.sort(rng.choice(flat_indices, size=MAX_PIXELS, replace=False))
    rows, cols = np.unravel_index(flat_indices, mask.shape)
    selected = stack[:, rows, cols].astype(np.float32)

    legacy_corrected, legacy_factor, legacy_factor_reason = apply_unit_correction(
        stack, dates, variable, "legacy_mode_a"
    )
    legacy_imputed = impute_calendar_month(
        legacy_corrected,
        dates,
        train_end=202112,
        legacy_all_periods=True,
    )
    legacy_detrended, legacy_trends = detrend_by_calendar_month(
        legacy_imputed, dates, train_end=202112
    )
    legacy_z, legacy_z_params = scale_values(
        legacy_detrended, dates, train_end=202112, scaling="monthwise_z"
    )
    legacy_minmax, legacy_minmax_params = scale_values(
        legacy_detrended, dates, train_end=202112, scaling="global_minmax"
    )

    metadata = {
        "land_type": land_type,
        "variable": variable,
        "source": "GeoTIFF rasters",
        "target_height": target.height,
        "target_width": target.width,
        "target_bounds": list(target.bounds),
        "target_crs": target.crs,
        "selected_pixels": int(len(rows)),
        "available_start": int(dates.min()),
        "available_end": int(dates.max()),
        "available_months": int(len(dates)),
        "missing_expected_months": [
            int(value)
            for value in sorted(set(month_range(expected_start(variable), 202412)) - set(dates.tolist()))
        ],
        "history_root": str(HISTORY_ROOT),
        "test_root": str(TEST_ROOT),
        "legacy_unit_scale_factor": legacy_factor,
        "legacy_unit_scale_reason": legacy_factor_reason,
        "legacy_trends": legacy_trends,
        "legacy_monthwise_z_params": legacy_z_params,
        "legacy_global_minmax_params": legacy_minmax_params,
    }
    np.savez_compressed(
        output,
        dates=dates,
        values=selected,
        legacy_monthwise_z_values=legacy_z[:, rows, cols].astype(np.float32),
        legacy_global_minmax_values=legacy_minmax[:, rows, cols].astype(np.float32),
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        metadata_json=np.array(json.dumps(metadata)),
    )
    return metadata


def load_task_cache(land_type: str, variable: str) -> Dict:
    path = CACHE_DIR / "task_data" / f"{land_type}_{variable}.npz"
    if not path.exists():
        build_task_cache(land_type, variable)
    data = np.load(path, allow_pickle=False)
    return {
        "dates": data["dates"].astype(np.int32),
        "values": data["values"].astype(np.float32),
        "legacy_monthwise_z_values": data["legacy_monthwise_z_values"].astype(np.float32),
        "legacy_global_minmax_values": data["legacy_global_minmax_values"].astype(np.float32),
        "rows": data["rows"].astype(np.int32),
        "cols": data["cols"].astype(np.int32),
        "metadata": json.loads(str(data["metadata_json"])),
    }


def _legacy_scale_factor(values: np.ndarray, dates: np.ndarray, variable: str) -> Tuple[float, str]:
    if variable in {"SPEI", "NDVI", "GNDVI", "NIRv"}:
        return 1.0, "old script: explicitly no correction"
    train = values[dates <= 202112]
    test = values[dates >= 202201]
    train_valid = train[np.isfinite(train)]
    test_valid = test[np.isfinite(test)]
    if train_valid.size == 0 or test_valid.size == 0:
        return 1.0, "insufficient values"
    mean_ratio = float(np.mean(train_valid) / (np.mean(test_valid) + 1e-10))
    median_ratio = float(np.median(train_valid) / (np.median(test_valid) + 1e-10))
    if variable == "GPP" and abs(mean_ratio - 1.0) > 0.1:
        if abs(31.56 - mean_ratio) / (abs(mean_ratio) + 1e-10) < 0.2:
            return 31.56, "old script: known GPP conversion"
        return median_ratio, "old script: observed test-period median ratio"
    if abs(mean_ratio - 1.0) > 0.1 or abs(median_ratio - 1.0) > 0.1:
        return median_ratio, "old script: observed test-period median ratio"
    return 1.0, "old script: distributions considered consistent"


def apply_unit_correction(
    values: np.ndarray,
    dates: np.ndarray,
    variable: str,
    preprocessing_mode: str,
) -> Tuple[np.ndarray, float, str]:
    corrected = values.copy()
    if preprocessing_mode == "legacy_mode_a":
        factor, reason = _legacy_scale_factor(corrected, dates, variable)
        corrected[dates >= 202201] *= factor
        return corrected, factor, reason
    if variable == "GPP":
        corrected[dates >= 202201] *= 31.56
        return corrected, 31.56, "reviewer mode: documented fixed GPP conversion only"
    return corrected, 1.0, "reviewer mode: no test-derived unit scaling"


def impute_calendar_month(
    values: np.ndarray,
    dates: np.ndarray,
    train_end: int,
    legacy_all_periods: bool,
) -> np.ndarray:
    result = values.copy()
    months = dates % 100
    for month in range(1, 13):
        indices = np.flatnonzero(months == month)
        if not legacy_all_periods:
            indices = indices[dates[indices] <= train_end]
        if len(indices) < 3:
            continue
        subset = result[indices].copy()
        for offset in range(1, len(indices) - 1):
            missing = ~np.isfinite(subset[offset])
            can_fill = missing & np.isfinite(subset[offset - 1]) & np.isfinite(subset[offset + 1])
            subset[offset, can_fill] = 0.5 * (
                subset[offset - 1, can_fill] + subset[offset + 1, can_fill]
            )
        result[indices] = subset
    return result


def detrend_by_calendar_month(values: np.ndarray, dates: np.ndarray, train_end: int) -> Tuple[np.ndarray, Dict]:
    result = values.copy()
    years = dates // 100
    months = dates % 100
    trends = {}
    for month in range(1, 13):
        indices = np.flatnonzero(months == month)
        spatial_axes = tuple(range(1, result.ndim))
        means = np.nanmean(result[indices], axis=spatial_axes)
        fit = np.isfinite(means) & (dates[indices] <= train_end)
        if fit.sum() < 5:
            continue
        design = np.vstack([years[indices][fit], np.ones(fit.sum())]).T
        slope, intercept = np.linalg.lstsq(design, means[fit], rcond=None)[0]
        trend = slope * years[indices] + intercept
        trend_shape = (len(indices),) + (1,) * (result.ndim - 1)
        result[indices] -= trend.reshape(trend_shape)
        trends[str(month)] = {"slope": float(slope), "intercept": float(intercept)}
    return result, trends


def scale_values(
    values: np.ndarray,
    dates: np.ndarray,
    train_end: int,
    scaling: str = "monthwise_z",
) -> Tuple[np.ndarray, Dict]:
    result = values.copy()
    params = {}
    train = dates <= train_end
    if scaling == "global_minmax":
        valid = result[train][np.isfinite(result[train])]
        minimum = float(np.min(valid))
        maximum = float(np.max(valid))
        span = maximum - minimum + 1e-8
        result = (result - minimum) / span
        params = {"minimum": minimum, "maximum": maximum}
        return np.clip(result, -TARGET_CLIP, TARGET_CLIP), params
    months = dates % 100
    for month in range(1, 13):
        fit = train & (months == month)
        valid = result[fit][np.isfinite(result[fit])]
        if valid.size == 0:
            continue
        mean = float(np.mean(valid))
        std = float(np.std(valid) + 1e-8)
        indices = months == month
        result[indices] = (result[indices] - mean) / std
        params[str(month)] = {"mean": mean, "std": std}
    return np.clip(result, -TARGET_CLIP, TARGET_CLIP), params


def prepare_task(
    land_type: str,
    variable: str,
    train_end: int,
    preprocessing_mode: str,
    scaling: str = "monthwise_z",
) -> Dict:
    cached = load_task_cache(land_type, variable)
    if preprocessing_mode == "legacy_mode_a":
        processed_key = (
            "legacy_global_minmax_values"
            if scaling == "global_minmax"
            else "legacy_monthwise_z_values"
        )
        cached.update(
            {
                "processed_values": cached[processed_key].copy(),
                "unit_scale_factor": cached["metadata"]["legacy_unit_scale_factor"],
                "unit_scale_reason": cached["metadata"]["legacy_unit_scale_reason"],
                "trends": cached["metadata"]["legacy_trends"],
                "scale_params": cached["metadata"][
                    "legacy_global_minmax_params"
                    if scaling == "global_minmax"
                    else "legacy_monthwise_z_params"
                ],
                "scaling": scaling,
                "preprocessing_mode": preprocessing_mode,
            }
        )
        return cached
    values, factor, factor_reason = apply_unit_correction(
        cached["values"], cached["dates"], variable, preprocessing_mode
    )
    values = impute_calendar_month(
        values,
        cached["dates"],
        train_end=train_end,
        legacy_all_periods=preprocessing_mode == "legacy_mode_a",
    )
    values, trends = detrend_by_calendar_month(values, cached["dates"], train_end)
    values, scale_params = scale_values(values, cached["dates"], train_end, scaling=scaling)
    cached.update(
        {
            "processed_values": values.astype(np.float32),
            "unit_scale_factor": factor,
            "unit_scale_reason": factor_reason,
            "trends": trends,
            "scale_params": scale_params,
            "scaling": scaling,
            "preprocessing_mode": preprocessing_mode,
        }
    )
    return cached


def build_supervised(
    prepared: Dict,
    target_start: int,
    target_end: int,
    lookback: int = MAIN_LOOKBACK,
) -> Dict:
    dates = prepared["dates"]
    values = prepared["processed_values"]
    target_indices = np.flatnonzero((dates >= target_start) & (dates <= target_end))
    target_indices = target_indices[target_indices >= lookback]
    n_pixels = values.shape[1]
    x_rows, y_rows, mask_rows, date_rows, pixel_rows = [], [], [], [], []
    for index in target_indices:
        sequence = values[index - lookback : index].T
        target = values[index]
        mask = np.isfinite(target)
        x_rows.append(np.nan_to_num(sequence, nan=0.0))
        y_rows.append(np.nan_to_num(target, nan=0.0))
        mask_rows.append(mask.astype(np.float32))
        date_rows.append(np.full(n_pixels, dates[index], dtype=np.int32))
        pixel_rows.append(np.arange(n_pixels, dtype=np.int32))
    if not x_rows:
        return {
            "X": np.empty((0, lookback), dtype=np.float32),
            "y": np.empty(0, dtype=np.float32),
            "mask": np.empty(0, dtype=np.float32),
            "date": np.empty(0, dtype=np.int32),
            "pixel_id": np.empty(0, dtype=np.int32),
        }
    return {
        "X": np.concatenate(x_rows, axis=0).astype(np.float32),
        "y": np.concatenate(y_rows).astype(np.float32),
        "mask": np.concatenate(mask_rows).astype(np.float32),
        "date": np.concatenate(date_rows).astype(np.int32),
        "pixel_id": np.concatenate(pixel_rows).astype(np.int32),
    }
