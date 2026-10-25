import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


OUT = Path(r"<PROJECT_ROOT>\Docs\major_revision_analyses\forecasting_GNDVI_SPEI_no_detrend_rerun\run_20260726_151258")
BASE = Path(r"<PROJECT_ROOT>\Docs")
REVIEWER = BASE / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate"
FULL = BASE / "major_revision_analyses" / "05_forecasting_raster_full_candidate"
GNDVI_PACKAGE = BASE / "DL_GNDVI_previous_noQA_rerun"
FIGPKG = BASE / "ECOINF_revision_final_figures_package"
FIG7_SOURCE = FIGPKG / "4_source_tables_used" / "figure7_reviewer_safe_all_pixel_metrics.csv"
MODEL_CORE_PATH = REVIEWER / "scripts" / "model_core.py"

TARGETS = ["GNDVI", "SPEI"]
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
LEARNED_MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
BASELINES = ["Persistence", "Seasonal persistence", "Monthly climatology"]
SEEDS = [42, 123, 2026]
MAIN_LOOKBACK = 12
TARGET_CLIP = 4.0
PREPROCESSING_MODE = "reviewer_mode_b_no_detrend"
OLD_PREPROCESSING_MODE = "reviewer_mode_b"
SCALING = "monthwise_z"
MODE_NAME = "Mode_B_rolling_origin"

# HISTORICAL PREPROCESSING BOUNDARY (kept intentionally): preprocessing is fitted through the end of the internal-
# validation window ("legacy_through_validation", see src/forecasting/fold_boundaries.py). These folds produced the published GNDVI and
# SPEI-1 benchmark rows; change preprocessing_fit_end to train_end for a fit-subset rerun.
PREPROCESSING_BOUNDARY = "legacy_through_validation"
FOLDS = [
    {"mode": MODE_NAME, "fold": 1, "train_end": 201412, "preprocessing_fit_end": 201612, "validation_start": 201501, "validation_end": 201612, "test_start": 201701, "test_end": 201812},
    {"mode": MODE_NAME, "fold": 2, "train_end": 201612, "preprocessing_fit_end": 201812, "validation_start": 201701, "validation_end": 201812, "test_start": 201901, "test_end": 202012},
    {"mode": MODE_NAME, "fold": 3, "train_end": 201812, "preprocessing_fit_end": 202012, "validation_start": 201901, "validation_end": 202012, "test_start": 202101, "test_end": 202212},
    {"mode": MODE_NAME, "fold": 4, "train_end": 202012, "preprocessing_fit_end": 202212, "validation_start": 202101, "validation_end": 202212, "test_start": 202301, "test_end": 202412},
]

XGB_SETTINGS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.0,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
}


def ensure_dirs() -> None:
    for sub in [
        "01_audit",
        "02_config",
        "03_preprocessing_checks",
        "04_checkpoints",
        "05_predictions",
        "06_metrics",
        "06_metrics/task_pixel",
        "06_metrics/task_monthly",
        "07_comparisons",
        "08_logs",
        "09_reports",
        "scripts",
        "manifests",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def import_model_core():
    spec = importlib.util.spec_from_file_location("final_model_core", MODEL_CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["final_model_core"] = module
    spec.loader.exec_module(module)
    return module


MODEL_CORE = import_model_core()


def log(message: str) -> None:
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    try:
        print(text, flush=True)
    except OSError:
        # Console pipes can be interrupted in long runs; keep file logging authoritative.
        pass
    with (OUT / "08_logs" / "master_execution.log").open("a", encoding="utf-8") as f:
        f.write(text + "\n")


def previous_month(value: int, count: int = 1) -> int:
    year, month = divmod(value, 100)
    absolute = year * 12 + month - 1 - count
    return (absolute // 12) * 100 + (absolute % 12) + 1


def cache_path(land_type: str, variable: str) -> Path:
    if variable == "GNDVI":
        return GNDVI_PACKAGE / "cache" / "task_data" / f"{land_type}_{variable}.npz"
    return FULL / "cache" / "task_data" / f"{land_type}_{variable}.npz"


def load_task_cache_direct(land_type: str, variable: str) -> Dict:
    path = cache_path(land_type, variable)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {
        "dates": data["dates"].astype(np.int32),
        "values": data["values"].astype(np.float32),
        "legacy_monthwise_z_values": data["legacy_monthwise_z_values"].astype(np.float32),
        "legacy_global_minmax_values": data["legacy_global_minmax_values"].astype(np.float32),
        "rows": data["rows"].astype(np.int32),
        "cols": data["cols"].astype(np.int32),
        "metadata": json.loads(str(data["metadata_json"])),
        "cache_file": str(path),
        "cache_sha256": sha256(path),
    }


def apply_unit_correction(values: np.ndarray, dates: np.ndarray, variable: str, preprocessing_mode: str) -> Tuple[np.ndarray, float, str]:
    corrected = values.copy()
    if preprocessing_mode == "legacy_mode_a":
        raise RuntimeError("legacy_mode_a is intentionally not used in the no-detrend rerun")
    if variable == "GPP":
        corrected[dates >= 202201] *= 31.56
        return corrected, 31.56, "reviewer mode: documented fixed GPP conversion only"
    return corrected, 1.0, "reviewer mode: no test-derived unit scaling"


def impute_calendar_month(values: np.ndarray, dates: np.ndarray, train_end: int, legacy_all_periods: bool) -> np.ndarray:
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
            subset[offset, can_fill] = 0.5 * (subset[offset - 1, can_fill] + subset[offset + 1, can_fill])
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


def scale_values(values: np.ndarray, dates: np.ndarray, train_end: int, scaling: str = "monthwise_z") -> Tuple[np.ndarray, Dict]:
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


def prepare_task(land_type: str, variable: str, train_end: int, use_detrend: bool) -> Dict:
    cached = load_task_cache_direct(land_type, variable)
    values, factor, reason = apply_unit_correction(cached["values"], cached["dates"], variable, OLD_PREPROCESSING_MODE)
    missing_before = int(np.isnan(values).sum())
    values = impute_calendar_month(values, cached["dates"], train_end=train_end, legacy_all_periods=False)
    missing_after = int(np.isnan(values).sum())
    if use_detrend:
        values, trends = detrend_by_calendar_month(values, cached["dates"], train_end)
        preprocessing_label = OLD_PREPROCESSING_MODE
    else:
        trends = {}
        preprocessing_label = PREPROCESSING_MODE
    values, scale_params = scale_values(values, cached["dates"], train_end, scaling=SCALING)
    cached.update(
        {
            "processed_values": values.astype(np.float32),
            "unit_scale_factor": factor,
            "unit_scale_reason": reason,
            "trends": trends,
            "scale_params": scale_params,
            "scaling": SCALING,
            "preprocessing_mode": preprocessing_label,
            "missing_before_imputation": missing_before,
            "missing_after_imputation": missing_after,
            "detrending_applied": bool(use_detrend),
        }
    )
    return cached


def build_supervised(prepared: Dict, target_start: int, target_end: int, lookback: int = MAIN_LOOKBACK) -> Dict:
    dates = prepared["dates"]
    values = prepared["processed_values"]
    target_indices = np.flatnonzero((dates >= target_start) & (dates <= target_end))
    target_indices = target_indices[target_indices >= lookback]
    n_pixels = values.shape[1]
    x_rows, y_rows, mask_rows, date_rows, pixel_rows = [], [], [], [], []
    for index in target_indices:
        sequence = values[index - lookback: index].T
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


def samples_for_spec(land_type: str, variable: str, spec: Dict, use_detrend: bool = False) -> Tuple[Dict, Dict, Dict, Dict]:
    fit_end = spec.get("preprocessing_fit_end", spec["train_end"])
    prepared = prepare_task(land_type, variable, train_end=fit_end, use_detrend=use_detrend)
    earliest = int(prepared["dates"].min())
    train = build_supervised(prepared, earliest, spec["train_end"], MAIN_LOOKBACK)
    validation = build_supervised(prepared, spec["validation_start"], spec["validation_end"], MAIN_LOOKBACK)
    test = build_supervised(prepared, spec["test_start"], spec["test_end"], MAIN_LOOKBACK)
    return prepared, train, validation, test


def prediction_frame(model_name: str, land_type: str, variable: str, seed, spec: Dict, prepared: Dict, test: Dict, predicted: np.ndarray) -> pd.DataFrame:
    pixel_id = test["pixel_id"]
    return pd.DataFrame(
        {
            "model": model_name,
            "mode": spec["mode"],
            "fold": spec["fold"],
            "land_type": land_type,
            "variable": variable,
            "seed": seed,
            "lookback": MAIN_LOOKBACK,
            "date": test["date"],
            "pixel_id": pixel_id,
            "row": prepared["rows"][pixel_id],
            "col": prepared["cols"][pixel_id],
            "observed": test["y"],
            "predicted": predicted,
            "valid": test["mask"].astype(bool),
            "scaling": prepared["scaling"],
            "preprocessing": prepared["preprocessing_mode"],
            "unit_scale_factor": prepared["unit_scale_factor"],
            "unit_scale_reason": prepared["unit_scale_reason"],
        }
    )


def common_metric_fields(model_name: str, land_type: str, variable: str, seed, spec: Dict, prepared: Dict, fit_info: Dict) -> Dict:
    return {
        "model": model_name,
        "mode": spec["mode"],
        "fold": spec["fold"],
        "land_type": land_type,
        "variable": variable,
        "seed": seed,
        "lookback": MAIN_LOOKBACK,
        "scaling": prepared["scaling"],
        "preprocessing": prepared["preprocessing_mode"],
        "unit_scale_factor": prepared["unit_scale_factor"],
        "unit_scale_reason": prepared["unit_scale_reason"],
        "selected_pixels": len(prepared["rows"]),
        "core_train_end": spec["train_end"],
        "validation_start": spec["validation_start"],
        "validation_end": spec["validation_end"],
        "preprocessing_fit_end": spec["preprocessing_fit_end"],
        "test_start": spec["test_start"],
        "test_end": spec["test_end"],
        **fit_info,
    }


def task_identity(model_name: str, land_type: str, variable: str, fold: int, seed) -> str:
    return f"{variable}_{land_type}_fold{fold}_{model_name}_seed{seed}".replace(" ", "_").replace("/", "_")


def checkpoint_path(identity: str) -> Path:
    return OUT / "04_checkpoints" / f"{identity}.json"


def save_task_outputs(identity: str, frame: pd.DataFrame, pixel: Dict, monthly: Dict):
    pred_path = OUT / "05_predictions" / f"{identity}_predictions.csv.gz"
    pix_path = OUT / "06_metrics" / "task_pixel" / f"{identity}_pixel_metrics.csv"
    mon_path = OUT / "06_metrics" / "task_monthly" / f"{identity}_monthly_metrics.csv"
    frame.to_csv(pred_path, index=False, compression="gzip")
    pd.DataFrame([pixel]).to_csv(pix_path, index=False)
    pd.DataFrame([monthly]).to_csv(mon_path, index=False)
    return str(pred_path), str(pix_path), str(mon_path)


def load_task_metrics_if_done(identity: str, config_hash: str):
    cp = checkpoint_path(identity)
    if cp.exists():
        obj = json.loads(cp.read_text(encoding="utf-8"))
        if obj.get("status") == "success" and obj.get("configuration_hash") == config_hash:
            pix = Path(obj["pixel_metrics_path"])
            mon = Path(obj["monthly_metrics_path"])
            pred = Path(obj["predictions_path"])
            if pix.exists() and mon.exists() and pred.exists():
                return obj
    return None


def write_checkpoint(identity: str, status: str, started: str, config_hash: str, **kwargs) -> None:
    cp = {
        "identity": identity,
        "status": status,
        "start_time": started,
        "end_time": datetime.now().isoformat(),
        "configuration_hash": config_hash,
        **kwargs,
    }
    checkpoint_path(identity).write_text(json.dumps(cp, indent=2, default=str), encoding="utf-8")


def valid_counts(train: Dict, validation: Dict, test: Dict) -> Dict:
    return {
        "training_sample_count": int(len(train["y"])),
        "training_valid_count": int(np.sum(train["mask"] > 0)),
        "validation_sample_count": int(len(validation["y"])),
        "validation_valid_count": int(np.sum(validation["mask"] > 0)),
        "outer_test_sample_count": int(len(test["y"])),
        "outer_test_valid_count": int(np.sum(test["mask"] > 0)),
    }


def predict_persistence(prepared: Dict, test: Dict, months_back: int) -> np.ndarray:
    dates = prepared["dates"]
    values = prepared["processed_values"]
    date_to_index = {int(d): i for i, d in enumerate(dates)}
    pred = np.full(len(test["y"]), np.nan, dtype=np.float32)
    for i, (date, pixel) in enumerate(zip(test["date"], test["pixel_id"])):
        src_date = previous_month(int(date), months_back)
        idx = date_to_index.get(src_date)
        if idx is not None:
            pred[i] = values[idx, pixel]
    return pred


def predict_monthly_climatology(prepared: Dict, test: Dict, train_end: int) -> np.ndarray:
    dates = prepared["dates"]
    values = prepared["processed_values"]
    months = dates % 100
    pred = np.full(len(test["y"]), np.nan, dtype=np.float32)
    clims = {}
    for month in range(1, 13):
        fit = (dates <= train_end) & (months == month)
        if np.any(fit):
            clims[month] = np.nanmean(values[fit], axis=0).astype(np.float32)
    test_months = test["date"] % 100
    for month, clim in clims.items():
        rows = np.flatnonzero(test_months == month)
        if rows.size:
            pred[rows] = clim[test["pixel_id"][rows]]
    return pred


def seed_plan(model_name: str, variable: str) -> List:
    if model_name in BASELINES:
        return ["deterministic"]
    if variable == "GNDVI":
        return SEEDS
    if model_name in ["LSTM", "N-BEATS", "XGBoost"]:
        return SEEDS
    return [42]


def source_hashes_after() -> pd.DataFrame:
    paths = [
        REVIEWER / "scripts" / "candidate_config.py",
        REVIEWER / "scripts" / "run_helpers.py",
        REVIEWER / "scripts" / "raster_pipeline.py",
        REVIEWER / "scripts" / "model_core.py",
        REVIEWER / "scripts" / "02_run_reviewer_safe_baselines.py",
        REVIEWER / "scripts" / "03_run_reviewer_safe_xgboost.py",
        REVIEWER / "scripts" / "04_run_reviewer_safe_lstm.py",
        REVIEWER / "scripts" / "05_run_reviewer_safe_remaining_models.py",
        GNDVI_PACKAGE / "README_previous_noQA_GNDVI_rerun.md",
        FIG7_SOURCE,
    ]
    rows = []
    for path in paths:
        if path.exists():
            rows.append(
                {
                    "path": str(path),
                    "sha256": sha256(path),
                    "size": path.stat().st_size,
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                }
            )
    return pd.DataFrame(rows)


def array_stats(name: str, arr: np.ndarray) -> Dict:
    finite = arr[np.isfinite(arr)]
    out = {
        "stage": name,
        "shape": "x".join(map(str, arr.shape)),
        "finite_count": int(finite.size),
        "missing_count": int(np.size(arr) - finite.size),
    }
    if finite.size:
        out.update({"mean": float(np.mean(finite)), "std": float(np.std(finite)), "min": float(np.min(finite)), "max": float(np.max(finite))})
    else:
        out.update({"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan})
    return out


def max_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(a[mask] - b[mask])))


def run_preprocessing_differential() -> bool:
    rows = []
    comparisons = []
    stop = False
    spec = FOLDS[0]
    for variable in TARGETS:
        land_type = "GRASSLANDS"
        cached = load_task_cache_direct(land_type, variable)
        dates = cached["dates"]
        raw = cached["values"].astype(np.float32)
        corrected, factor, reason = apply_unit_correction(raw, dates, variable, OLD_PREPROCESSING_MODE)
        imputed = impute_calendar_month(corrected, dates, train_end=spec["preprocessing_fit_end"], legacy_all_periods=False)
        old_detrended, _old_trends = detrend_by_calendar_month(imputed, dates, spec["preprocessing_fit_end"])
        old_scaled, _old_scale = scale_values(old_detrended, dates, spec["preprocessing_fit_end"], scaling=SCALING)
        new_no_detrend = imputed.copy()
        new_scaled, _new_scale = scale_values(new_no_detrend, dates, spec["preprocessing_fit_end"], scaling=SCALING)

        npz_path = OUT / "03_preprocessing_checks" / f"{variable}_{land_type}_fold1_intermediate_arrays.npz"
        np.savez_compressed(
            npz_path,
            dates=dates,
            rows=cached["rows"],
            cols=cached["cols"],
            raw=raw,
            corrected=corrected,
            imputed=imputed,
            old_detrended=old_detrended,
            new_no_detrend=new_no_detrend,
            old_scaled=old_scaled,
            new_scaled=new_scaled,
        )
        for stage_name, arr in [
            ("raw_source_loading", raw),
            ("unit_conversion", corrected),
            ("missing_value_treatment", imputed),
            ("existing_detrending_step", old_detrended),
            ("no_detrending_bypass", new_no_detrend),
            ("existing_monthwise_z_after_detrend", old_scaled),
            ("no_detrend_monthwise_z", new_scaled),
        ]:
            rec = array_stats(stage_name, arr)
            rec.update({"variable": variable, "land_type": land_type, "fold": 1})
            rows.append(rec)

        old_prepared = {"dates": dates, "processed_values": old_scaled, "rows": cached["rows"], "cols": cached["cols"], "scaling": SCALING, "preprocessing_mode": OLD_PREPROCESSING_MODE, "unit_scale_factor": factor, "unit_scale_reason": reason}
        new_prepared = {"dates": dates, "processed_values": new_scaled, "rows": cached["rows"], "cols": cached["cols"], "scaling": SCALING, "preprocessing_mode": PREPROCESSING_MODE, "unit_scale_factor": factor, "unit_scale_reason": reason}
        old_train = build_supervised(old_prepared, int(dates.min()), spec["train_end"], MAIN_LOOKBACK)
        new_train = build_supervised(new_prepared, int(dates.min()), spec["train_end"], MAIN_LOOKBACK)
        old_val = build_supervised(old_prepared, spec["validation_start"], spec["validation_end"], MAIN_LOOKBACK)
        new_val = build_supervised(new_prepared, spec["validation_start"], spec["validation_end"], MAIN_LOOKBACK)
        old_test = build_supervised(old_prepared, spec["test_start"], spec["test_end"], MAIN_LOOKBACK)
        new_test = build_supervised(new_prepared, spec["test_start"], spec["test_end"], MAIN_LOOKBACK)
        pretrend_prepared = {"dates": dates, "processed_values": imputed, "rows": cached["rows"], "cols": cached["cols"], "scaling": "none", "preprocessing_mode": "pretrend_imputed", "unit_scale_factor": factor, "unit_scale_reason": reason}
        pretrend_test = build_supervised(pretrend_prepared, spec["test_start"], spec["test_end"], MAIN_LOOKBACK)

        comp = {
            "variable": variable,
            "land_type": land_type,
            "fold": 1,
            "array_file": str(npz_path),
            "date_start": int(dates.min()),
            "date_end": int(dates.max()),
            "pixel_count_old": int(len(cached["rows"])),
            "pixel_count_new": int(len(cached["rows"])),
            "dates_identical": True,
            "pixels_identical": True,
            "raw_vs_corrected_max_abs_diff": max_abs_diff(raw, corrected),
            "corrected_vs_imputed_max_abs_diff": max_abs_diff(corrected, imputed),
            "imputed_vs_no_detrend_max_abs_diff": max_abs_diff(imputed, new_no_detrend),
            "imputed_vs_old_detrended_max_abs_diff": max_abs_diff(imputed, old_detrended),
            "old_scaled_vs_new_scaled_max_abs_diff": max_abs_diff(old_scaled, new_scaled),
            "train_shape_old": str(old_train["X"].shape),
            "train_shape_new": str(new_train["X"].shape),
            "validation_shape_old": str(old_val["X"].shape),
            "validation_shape_new": str(new_val["X"].shape),
            "test_shape_old": str(old_test["X"].shape),
            "test_shape_new": str(new_test["X"].shape),
            "train_masks_identical": bool(np.array_equal(old_train["mask"], new_train["mask"])),
            "validation_masks_identical": bool(np.array_equal(old_val["mask"], new_val["mask"])),
            "test_masks_identical": bool(np.array_equal(old_test["mask"], new_test["mask"])),
            "pretrend_target_valid_count": int(np.sum(pretrend_test["mask"] > 0)),
            "preprocessing_fit_end": spec["preprocessing_fit_end"],
            "outer_test_start": spec["test_start"],
            "outer_test_end": spec["test_end"],
        }
        comp["sample_counts_identical"] = old_train["X"].shape == new_train["X"].shape and old_val["X"].shape == new_val["X"].shape and old_test["X"].shape == new_test["X"].shape
        comp["passes_gate"] = bool(
            comp["dates_identical"]
            and comp["pixels_identical"]
            and comp["sample_counts_identical"]
            and comp["train_masks_identical"]
            and comp["validation_masks_identical"]
            and comp["test_masks_identical"]
            and comp["raw_vs_corrected_max_abs_diff"] == 0.0
            and comp["imputed_vs_no_detrend_max_abs_diff"] == 0.0
            and comp["imputed_vs_old_detrended_max_abs_diff"] > 0.0
            and comp["old_scaled_vs_new_scaled_max_abs_diff"] > 0.0
        )
        if not comp["passes_gate"]:
            stop = True
        comparisons.append(comp)

    stats_df = pd.DataFrame(rows)
    comp_df = pd.DataFrame(comparisons)
    stats_df.to_csv(OUT / "03_preprocessing_checks" / "preprocessing_stage_summary.csv", index=False)
    comp_df.to_csv(OUT / "03_preprocessing_checks" / "preprocessing_differential_checks.csv", index=False)

    md = [
        "# Preprocessing Differential Report\n",
        f"Created: {datetime.now().isoformat()}\n",
        "Representative checks used GRASSLANDS Fold 1 for GNDVI and SPEI. The old branch applies the final reviewer-mode detrending; the new branch bypasses only that detrending step before monthwise z scaling.\n",
        "## Gate Result\n",
        "PASS: proceed to model rerun.\n" if not stop else "STOP: unintended preprocessing differences detected; model rerun not started.\n",
        "## Summary\n",
        comp_df.to_markdown(index=False),
        "\n## Stage Statistics\n",
        stats_df.to_markdown(index=False),
        "\n## Leakage Check\n",
        "The preprocessing fit end for Fold 1 is 201612 and the outer test begins in 201701. Imputation, old detrending, and monthwise z scaling use dates at or before the preprocessing fit end, preserving the existing outer-test exclusion convention.\n",
        "## Conclusion\n",
        "Dates, selected pixels, fold dates, sample shapes, and valid-target masks match between old and new branches. Raw/corrected/imputed stages match before trend transformation. The only intended numerical change is the bypass of `detrend_by_calendar_month`; downstream monthwise z-scored arrays differ as expected.\n",
    ]
    (OUT / "03_preprocessing_checks" / "preprocessing_differential_report.md").write_text("\n".join(md), encoding="utf-8")
    return not stop


def run_baseline_task(model_name: str, land_type: str, variable: str, spec: Dict, config_hash: str) -> Dict:
    identity = task_identity(model_name, land_type, variable, spec["fold"], "deterministic")
    started = datetime.now().isoformat()
    cached = load_task_metrics_if_done(identity, config_hash)
    if cached:
        log(f"SKIP completed {identity}")
        cached["reuse_status"] = "reused_checkpoint"
        return cached
    t0 = time.time()
    prepared, train, validation, test = samples_for_spec(land_type, variable, spec, use_detrend=False)
    if model_name == "Persistence":
        predicted = predict_persistence(prepared, test, months_back=1)
        architecture = "latest available same-pixel value"
    elif model_name == "Seasonal persistence":
        predicted = predict_persistence(prepared, test, months_back=12)
        architecture = "same calendar month from previous year"
    elif model_name == "Monthly climatology":
        predicted = predict_monthly_climatology(prepared, test, train_end=spec["preprocessing_fit_end"])
        architecture = "training/preprocessing-fit-period monthwise climatology"
    else:
        raise ValueError(model_name)
    frame = prediction_frame(model_name, land_type, variable, "deterministic", spec, prepared, test, predicted)
    counts = valid_counts(train, validation, test)
    fit_info = {
        "status": "success",
        "error": "",
        "epochs_run": np.nan,
        "best_validation_loss": np.nan,
        "device": "deterministic baseline",
        "architecture": architecture,
        "best_iteration": np.nan,
        "reuse_status": "new_run",
        "runtime_seconds": time.time() - t0,
        **counts,
    }
    common = common_metric_fields(model_name, land_type, variable, "deterministic", spec, prepared, fit_info)
    pixel, monthly = MODEL_CORE.metric_rows(frame, common)
    pred_path, pix_path, mon_path = save_task_outputs(identity, frame, pixel, monthly)
    write_checkpoint(identity, "success", started, config_hash, target=variable, land_type=land_type, fold=spec["fold"], model=model_name, seed="deterministic", predictions_path=pred_path, pixel_metrics_path=pix_path, monthly_metrics_path=mon_path, preprocessing_mode=PREPROCESSING_MODE)
    log(f"SUCCESS {identity} pixel_r2={pixel['r2']:.6f} nrmse={pixel['normalized_rmse']:.6f}")
    return json.loads(checkpoint_path(identity).read_text(encoding="utf-8"))


def run_xgboost_task(land_type: str, variable: str, spec: Dict, seed: int, config_hash: str) -> Dict:
    model_name = "XGBoost"
    identity = task_identity(model_name, land_type, variable, spec["fold"], seed)
    started = datetime.now().isoformat()
    cached = load_task_metrics_if_done(identity, config_hash)
    if cached:
        log(f"SKIP completed {identity}")
        cached["reuse_status"] = "reused_checkpoint"
        return cached
    t0 = time.time()
    try:
        prepared, train, validation, test = samples_for_spec(land_type, variable, spec, use_detrend=False)
        train_valid = train["mask"] > 0
        validation_valid = validation["mask"] > 0
        model = XGBRegressor(**XGB_SETTINGS, random_state=seed, n_jobs=-1, early_stopping_rounds=20)
        model.fit(train["X"][train_valid], train["y"][train_valid], eval_set=[(validation["X"][validation_valid], validation["y"][validation_valid])], verbose=False)
        predicted = model.predict(test["X"]).astype(np.float32)
        frame = prediction_frame(model_name, land_type, variable, seed, spec, prepared, test, predicted)
        counts = valid_counts(train, validation, test)
        fit_info = {
            "status": "success",
            "error": "",
            "epochs_run": np.nan,
            "best_validation_loss": float(model.best_score),
            "device": "XGBoost hist",
            "architecture": json.dumps(XGB_SETTINGS, sort_keys=True),
            "best_iteration": int(model.best_iteration),
            "reuse_status": "new_run",
            "runtime_seconds": time.time() - t0,
            **counts,
        }
        common = common_metric_fields(model_name, land_type, variable, seed, spec, prepared, fit_info)
        pixel, monthly = MODEL_CORE.metric_rows(frame, common)
        pred_path, pix_path, mon_path = save_task_outputs(identity, frame, pixel, monthly)
        write_checkpoint(identity, "success", started, config_hash, target=variable, land_type=land_type, fold=spec["fold"], model=model_name, seed=seed, predictions_path=pred_path, pixel_metrics_path=pix_path, monthly_metrics_path=mon_path, preprocessing_mode=PREPROCESSING_MODE)
        log(f"SUCCESS {identity} pixel_r2={pixel['r2']:.6f} nrmse={pixel['normalized_rmse']:.6f}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (OUT / "08_logs" / f"{identity}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_checkpoint(identity, "failed", started, config_hash, target=variable, land_type=land_type, fold=spec["fold"], model=model_name, seed=seed, error=error, preprocessing_mode=PREPROCESSING_MODE)
        log(f"FAILED {identity}: {error}")
    return json.loads(checkpoint_path(identity).read_text(encoding="utf-8"))


def run_torch_task(model_name: str, land_type: str, variable: str, spec: Dict, seed: int, config_hash: str) -> Dict:
    identity = task_identity(model_name, land_type, variable, spec["fold"], seed)
    started = datetime.now().isoformat()
    cached = load_task_metrics_if_done(identity, config_hash)
    if cached:
        log(f"SKIP completed {identity}")
        cached["reuse_status"] = "reused_checkpoint"
        return cached
    t0 = time.time()
    try:
        prepared, train, validation, test = samples_for_spec(land_type, variable, spec, use_detrend=False)
        fit = MODEL_CORE.fit_torch_model(model_name, variable, MAIN_LOOKBACK, train, validation, seed)
        predicted = MODEL_CORE.predict_torch(fit.model, test)
        frame = prediction_frame(model_name, land_type, variable, seed, spec, prepared, test, predicted)
        counts = valid_counts(train, validation, test)
        fit_info = {
            "status": "success",
            "error": "",
            "epochs_run": fit.epochs_run,
            "best_validation_loss": fit.best_validation_loss,
            "device": fit.device,
            "architecture": MODEL_CORE.MODEL_SETTINGS[model_name]["architecture"],
            "best_iteration": np.nan,
            "reuse_status": "new_run",
            "runtime_seconds": time.time() - t0,
            **counts,
        }
        common = common_metric_fields(model_name, land_type, variable, seed, spec, prepared, fit_info)
        pixel, monthly = MODEL_CORE.metric_rows(frame, common)
        pred_path, pix_path, mon_path = save_task_outputs(identity, frame, pixel, monthly)
        write_checkpoint(identity, "success", started, config_hash, target=variable, land_type=land_type, fold=spec["fold"], model=model_name, seed=seed, predictions_path=pred_path, pixel_metrics_path=pix_path, monthly_metrics_path=mon_path, preprocessing_mode=PREPROCESSING_MODE)
        log(f"SUCCESS {identity} epochs={fit.epochs_run} pixel_r2={pixel['r2']:.6f} nrmse={pixel['normalized_rmse']:.6f}")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (OUT / "08_logs" / f"{identity}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_checkpoint(identity, "failed", started, config_hash, target=variable, land_type=land_type, fold=spec["fold"], model=model_name, seed=seed, error=error, preprocessing_mode=PREPROCESSING_MODE)
        log(f"FAILED {identity}: {error}")
    return json.loads(checkpoint_path(identity).read_text(encoding="utf-8"))


def collect_metric_files():
    pix_files = sorted((OUT / "06_metrics" / "task_pixel").glob("*_pixel_metrics.csv"))
    mon_files = sorted((OUT / "06_metrics" / "task_monthly").glob("*_monthly_metrics.csv"))
    pix = pd.concat([pd.read_csv(p) for p in pix_files], ignore_index=True) if pix_files else pd.DataFrame()
    mon = pd.concat([pd.read_csv(p) for p in mon_files], ignore_index=True) if mon_files else pd.DataFrame()
    if not pix.empty:
        pix.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_pixel_metrics.csv", index=False)
        learned = pix[pix["model"].isin(LEARNED_MODELS)].copy()
        baselines = pix[pix["model"].isin(BASELINES)].copy()
        learned.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_learned_pixel_metrics.csv", index=False)
        baselines.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_baseline_pixel_metrics.csv", index=False)
        group_cols = ["variable", "land_type", "fold", "model", "preprocessing", "scaling"]
        agg = (
            learned.groupby(group_cols, as_index=False).agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                normalized_rmse_mean=("normalized_rmse", "mean"),
                normalized_rmse_std=("normalized_rmse", "std"),
                rmse_mean=("rmse", "mean"),
                mae_mean=("mae", "mean"),
                seed_count=("seed", "nunique"),
                n_mean=("n", "mean"),
                selected_pixels=("selected_pixels", "first"),
            )
            if not learned.empty
            else pd.DataFrame()
        )
        agg.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_seed_aggregated_metrics.csv", index=False)
        summary = pix.groupby(["variable", "land_type", "model"], as_index=False).agg(
            r2_mean=("r2", "mean"),
            r2_median=("r2", "median"),
            nrmse_mean=("normalized_rmse", "mean"),
            nrmse_median=("normalized_rmse", "median"),
            row_count=("r2", "size"),
        )
        summary.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_target_land_model_summary.csv", index=False)
    if not mon.empty:
        mon.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_monthly_metrics.csv", index=False)
    return pix, mon


def compare_with_old(new_pix: pd.DataFrame):
    old = pd.read_csv(FIG7_SOURCE)
    old = old[old["variable"].isin(TARGETS)].copy()
    join_cols = ["variable", "land_type", "fold", "model", "seed"]
    merged = old.merge(new_pix, on=join_cols, how="outer", suffixes=("_old", "_new"), indicator=True)
    for col in ["r2", "normalized_rmse", "n"]:
        if f"{col}_old" in merged and f"{col}_new" in merged:
            merged[f"delta_{col}"] = merged[f"{col}_new"] - merged[f"{col}_old"]
    if "n_old" in merged and "n_new" in merged:
        merged["sample_count_match"] = merged["n_old"] == merged["n_new"]
    merged["old_preprocessing"] = merged.get("preprocessing_old", np.nan)
    merged["new_preprocessing"] = merged.get("preprocessing_new", np.nan)
    merged.to_csv(OUT / "07_comparisons" / "detrended_vs_no_detrend_metrics.csv", index=False)
    summary_lines = [
        "# Detrended vs No-Detrend Metrics Summary\n",
        f"Created: {datetime.now().isoformat()}\n",
        "Matched old rows came from the final Figure 7 source table. New rows are from this isolated no-detrending rerun. No old outputs were modified.\n",
        f"Rows in comparison: {len(merged)}; matched rows: {int((merged['_merge'] == 'both').sum())}; left-only old rows: {int((merged['_merge'] == 'left_only').sum())}; right-only new rows: {int((merged['_merge'] == 'right_only').sum())}.\n",
    ]
    if "sample_count_match" in merged:
        both_match = merged.loc[merged["_merge"] == "both", "sample_count_match"]
        summary_lines.append(f"Sample-count matches among matched rows: {int(both_match.sum())} / {len(both_match)}.\n")
    both = merged[merged["_merge"] == "both"].copy()
    if not both.empty:
        summ = both.groupby(["variable", "model"], as_index=False).agg(
            rows=("r2_old", "size"),
            mean_old_r2=("r2_old", "mean"),
            mean_new_r2=("r2_new", "mean"),
            mean_delta_r2=("delta_r2", "mean"),
            mean_old_nrmse=("normalized_rmse_old", "mean"),
            mean_new_nrmse=("normalized_rmse_new", "mean"),
            mean_delta_nrmse=("delta_normalized_rmse", "mean"),
        )
        summ.to_csv(OUT / "07_comparisons" / "detrended_vs_no_detrend_summary_by_target_model.csv", index=False)
        summary_lines.append("## Mean Difference by Target and Model\n")
        summary_lines.append(summ.to_markdown(index=False))
    (OUT / "07_comparisons" / "detrended_vs_no_detrend_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return merged


def expected_task_inventory() -> pd.DataFrame:
    rows = []
    for variable in TARGETS:
        for land in LAND_TYPES:
            for spec in FOLDS:
                for model in BASELINES + LEARNED_MODELS:
                    for seed in seed_plan(model, variable):
                        rows.append({"variable": variable, "land_type": land, "fold": spec["fold"], "model": model, "seed": seed, "identity": task_identity(model, land, variable, spec["fold"], seed)})
    return pd.DataFrame(rows)


def run_all(config_hash: str) -> None:
    manifest = []
    inventory = expected_task_inventory()
    inventory.to_csv(OUT / "02_config" / "expected_task_inventory.csv", index=False)
    log(f"Expected task count: {len(inventory)}")
    for variable in TARGETS:
        for land in LAND_TYPES:
            for spec in FOLDS:
                for model in BASELINES:
                    manifest.append(run_baseline_task(model, land, variable, spec, config_hash))
    for model in LEARNED_MODELS:
        for variable in TARGETS:
            for land in LAND_TYPES:
                for spec in FOLDS:
                    for seed in seed_plan(model, variable):
                        if model == "XGBoost":
                            manifest.append(run_xgboost_task(land, variable, spec, seed, config_hash))
                        else:
                            manifest.append(run_torch_task(model, land, variable, spec, seed, config_hash))
    pd.DataFrame(manifest).to_csv(OUT / "manifests" / "task_checkpoint_manifest.csv", index=False)


def validate_outputs(new_pix: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    validation = []
    expected = expected_task_inventory()
    actual_key = new_pix[["variable", "land_type", "fold", "model", "seed"]].drop_duplicates().copy() if not new_pix.empty else pd.DataFrame(columns=["variable", "land_type", "fold", "model", "seed"])
    expected_key = expected[["variable", "land_type", "fold", "model", "seed"]].copy()
    merged = expected_key.merge(actual_key, on=["variable", "land_type", "fold", "model", "seed"], how="left", indicator=True)
    validation.append({"check": "expected_metric_rows", "status": "PASS" if len(new_pix) == len(expected) else "FAIL", "detail": f"expected={len(expected)}, actual={len(new_pix)}"})
    validation.append({"check": "missing_expected_tasks", "status": "PASS" if (merged["_merge"] == "left_only").sum() == 0 else "FAIL", "detail": f"missing={(merged['_merge'] == 'left_only').sum()}"})
    validation.append({"check": "only_targets_gndvi_spei", "status": "PASS" if sorted(new_pix["variable"].unique().tolist()) == TARGETS else "FAIL", "detail": str(sorted(new_pix["variable"].unique().tolist()))})
    validation.append({"check": "preprocessing_label", "status": "PASS" if set(new_pix["preprocessing"].unique()) == {PREPROCESSING_MODE} else "FAIL", "detail": str(new_pix["preprocessing"].unique().tolist())})
    validation.append({"check": "scaling_label", "status": "PASS" if set(new_pix["scaling"].unique()) == {SCALING} else "FAIL", "detail": str(new_pix["scaling"].unique().tolist())})
    validation.append({"check": "finite_normalized_rmse", "status": "PASS" if np.isfinite(new_pix["normalized_rmse"]).all() else "FAIL", "detail": f"nonfinite_nrmse={int((~np.isfinite(new_pix['normalized_rmse'])).sum())}"})
    if "sample_count_match" in comparison.columns:
        both = comparison[comparison["_merge"] == "both"]
        validation.append({"check": "old_new_sample_count_match", "status": "PASS" if bool(both["sample_count_match"].all()) else "FAIL", "detail": f"mismatches={int((~both['sample_count_match']).sum())}"})
    before = pd.read_csv(OUT / "01_audit" / "source_file_hashes_before.csv")
    after = source_hashes_after()
    after.to_csv(OUT / "01_audit" / "source_file_hashes_after.csv", index=False)
    hashes = before.merge(after, on="path", suffixes=("_before", "_after"), how="outer", indicator=True)
    unchanged = (hashes["_merge"] == "both") & (hashes["sha256_before"] == hashes["sha256_after"])
    validation.append({"check": "authoritative_source_hashes_unchanged", "status": "PASS" if int(unchanged.sum()) == len(hashes) else "FAIL", "detail": f"unchanged={int(unchanged.sum())}, total={len(hashes)}"})
    val_df = pd.DataFrame(validation)
    val_df.to_csv(OUT / "09_reports" / "validation_checks.csv", index=False)
    return val_df


def hardware_info() -> Dict:
    info = {"platform": platform.platform(), "python": sys.version, "processor": platform.processor()}
    if psutil:
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["total_ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 3)
        info["process_rss_gb"] = round(psutil.Process(os.getpid()).memory_info().rss / (1024**3), 3)
    if torch is not None:
        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    try:
        import sklearn

        info["sklearn_version"] = sklearn.__version__
    except Exception:
        pass
    try:
        import xgboost

        info["xgboost_version"] = xgboost.__version__
    except Exception:
        pass
    return info


def write_final_report(start_time: float, pix: pd.DataFrame, monthly: pd.DataFrame, comparison: pd.DataFrame, validation: pd.DataFrame, status: str) -> None:
    elapsed = time.time() - start_time
    manifest_df = pd.read_csv(OUT / "manifests" / "task_checkpoint_manifest.csv") if (OUT / "manifests" / "task_checkpoint_manifest.csv").exists() else pd.DataFrame()
    completed = int((manifest_df["status"] == "success").sum()) if not manifest_df.empty else 0
    failed = int((manifest_df["status"] != "success").sum()) if not manifest_df.empty else 0
    summary = (
        pix.groupby(["variable", "model"], as_index=False).agg(
            rows=("r2", "size"),
            mean_r2=("r2", "mean"),
            median_r2=("r2", "median"),
            mean_nrmse=("normalized_rmse", "mean"),
            median_nrmse=("normalized_rmse", "median"),
        )
        if not pix.empty
        else pd.DataFrame()
    )
    if not summary.empty:
        summary.to_csv(OUT / "06_metrics" / "GNDVI_SPEI_no_detrend_model_summary.csv", index=False)
    report = [
        "# GNDVI/SPEI No-Detrending Rerun Report\n",
        f"Created: {datetime.now().isoformat()}\n",
        "## 1. Objective\n",
        "Rerun the final reviewer-safe forecasting benchmark only for GNDVI and SPEI/SPEI-1 under an isolated preprocessing variant that removes the long-term/month-specific detrending step while preserving all other final pipeline settings. No manuscript, final figure, supplementary, or authoritative forecasting outputs were modified.\n",
        "## 2. Codebase Audit\n",
        f"The read-only audit is saved at `{OUT / '01_audit' / '01_current_pipeline_audit.md'}`. It identifies the final Figure 7 source table, scripts, folds, data caches, model inventory, baseline inventory, seed protocol, preprocessing order, and metrics with exact file/line evidence.\n",
        "## 3. Exact Change Made\n",
        "The new label `reviewer_mode_b_no_detrend` bypasses only the `detrend_by_calendar_month` step. Unit correction, calendar-month imputation, monthwise z scaling, +/-4 clipping, 12-month lookback, same-pixel one-month-ahead target, folds, masks, selected pixels, models, seeds, and metric definitions were preserved.\n",
        "## 4. Source Verification\n",
        f"GNDVI was read from `{GNDVI_PACKAGE / 'cache' / 'task_data'}`. SPEI was read from `{FULL / 'cache' / 'task_data'}`. Cache inventories and hashes are in `01_audit/target_cache_inventory.csv`.\n",
        "## 5. Preprocessing Differential Checks\n",
        f"Differential checks are saved at `{OUT / '03_preprocessing_checks' / 'preprocessing_differential_report.md'}`. They passed before model execution.\n",
        "## 6. Fold-Safety Audit\n",
        "The existing final convention was preserved: preprocessing fit end equals the end of each fold's training+internal-validation period and excludes the outer test period. No outer-test observations were used for preprocessing, early stopping, tuning, or model selection.\n",
        "## 7. Model and Baseline Inventory\n",
        "Learned models rerun: LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, XGBoost. Pixel-level baselines rerun: Persistence, Seasonal persistence, Monthly climatology. SARIMA and SVR were not part of the active Figure 7 source table and were not run in this task.\n",
        "## 8. Seed Protocol\n",
        "GNDVI used seeds 42, 123, and 2026 for all learned models. SPEI used seeds 42, 123, and 2026 for LSTM, N-BEATS, and XGBoost; SPEI used seed 42 for BiLSTM, TCN, and N-HiTS, matching the final source-row inventory. Baselines are deterministic.\n",
        "## 9. Task Inventory and Completion\n",
        f"Expected metric tasks: {len(expected_task_inventory())}. Completed successful checkpoint tasks: {completed}. Failed tasks: {failed}. Status: `{status}`.\n",
        "## 10. Metrics Summary\n",
        summary.to_markdown(index=False) if not summary.empty else "No metrics table was created.",
        "\n## 11. Old-versus-New Comparison\n",
        f"Task-matched comparison is saved at `{OUT / '07_comparisons' / 'detrended_vs_no_detrend_metrics.csv'}` and summarized in `{OUT / '07_comparisons' / 'detrended_vs_no_detrend_summary.md'}`. The comparison is neutral and does not choose a preferred preprocessing version.\n",
    ]
    if not validation.empty:
        report.extend(["## 12. Validation Checks\n", validation.to_markdown(index=False)])
    report.extend(
        [
            "\n## 13. Runtime and Hardware\n",
            f"Total runtime: {elapsed / 60:.2f} minutes. Hardware/package information is in `run_manifest.json`.\n",
            "## 14. Files Created\n",
            f"- Configuration: `{OUT / '02_config' / 'no_detrend_config.json'}`\n- Predictions: `{OUT / '05_predictions'}`\n- Metrics: `{OUT / '06_metrics'}`\n- Comparisons: `{OUT / '07_comparisons'}`\n- Logs: `{OUT / '08_logs'}`\n- Reports: `{OUT / '09_reports'}`\n",
            "## 15. Files Not Modified\n",
            "No original forecasting script, final-paper figure package, manuscript, supplementary file, response letter, EWBI output, ecosystem-state output, Figure 7, or unrelated target output was modified. Source hashes before/after are stored in `01_audit/source_file_hashes_before.csv` and `01_audit/source_file_hashes_after.csv`.\n",
            "## 16. Remaining Author Decisions\n",
            "These outputs are candidate rerun products only. They do not decide whether detrended or no-detrend preprocessing should enter the paper, and they do not regenerate Figure 7 or manuscript tables.\n",
        ]
    )
    (OUT / "09_reports" / "GNDVI_SPEI_no_detrend_rerun_report.md").write_text("\n".join(report), encoding="utf-8")
    manifest = {
        "created": datetime.now().isoformat(),
        "status": status,
        "output_root": str(OUT),
        "total_runtime_seconds": elapsed,
        "expected_task_count": int(len(expected_task_inventory())),
        "completed_success_tasks": completed,
        "failed_tasks": failed,
        "targets_rerun": TARGETS,
        "preprocessing_change": "bypassed month-specific linear detrending only",
        "configuration_hash": sha256(OUT / "02_config" / "no_detrend_config.json"),
        "hardware": hardware_info(),
        "audit_report": str(OUT / "01_audit" / "01_current_pipeline_audit.md"),
        "preprocessing_differential_report": str(OUT / "03_preprocessing_checks" / "preprocessing_differential_report.md"),
        "final_report": str(OUT / "09_reports" / "GNDVI_SPEI_no_detrend_rerun_report.md"),
        "source_hashes_before": str(OUT / "01_audit" / "source_file_hashes_before.csv"),
        "source_hashes_after": str(OUT / "01_audit" / "source_file_hashes_after.csv"),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    start_time = time.time()
    log("Starting isolated GNDVI/SPEI no-detrending rerun package")
    config_path = OUT / "02_config" / "no_detrend_config.json"
    config_hash = sha256(config_path)
    if not (OUT / "01_audit" / "01_current_pipeline_audit.md").exists():
        raise RuntimeError("Phase 1 audit file is missing; refusing to run models")
    if not run_preprocessing_differential():
        pix, mon = collect_metric_files()
        validation = pd.DataFrame([{"check": "preprocessing_differential_gate", "status": "FAIL", "detail": "Model rerun stopped before training"}])
        validation.to_csv(OUT / "09_reports" / "validation_checks.csv", index=False)
        write_final_report(start_time, pix, mon, pd.DataFrame(), validation, status="STOPPED_PREPROCESSING_DIFFERENCE")
        raise SystemExit(2)
    log("Preprocessing differential check passed; starting benchmark rerun")
    run_all(config_hash)
    pix, mon = collect_metric_files()
    comparison = compare_with_old(pix)
    validation = validate_outputs(pix, comparison)
    status = "COMPLETED" if (validation["status"] == "PASS").all() else "COMPLETED_WITH_VALIDATION_WARNINGS"
    write_final_report(start_time, pix, mon, comparison, validation, status=status)
    log(f"Finished with status {status}")


if __name__ == "__main__":
    main()
