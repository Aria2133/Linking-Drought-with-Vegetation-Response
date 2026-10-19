import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from candidate_config import MAIN_LOOKBACK, MODE_A, ROLLING_FOLDS
from model_core import (
    MODEL_SETTINGS,
    fit_torch_model,
    metric_rows,
    predict_torch,
    set_seed,
)
from raster_pipeline import build_supervised, prepare_task


def previous_month(value: int, count: int = 1) -> int:
    year, month = divmod(value, 100)
    absolute = year * 12 + month - 1 - count
    return (absolute // 12) * 100 + (absolute % 12) + 1


def split_specs(include_mode_b: bool, include_mode_a: bool = True) -> List[Dict]:
    specs = []
    if include_mode_a:
        specs.append(
            {
                "mode": MODE_A["name"],
                "fold": 0,
                "train_end": MODE_A["train_end"],
                "validation_start": MODE_A["validation_start"],
                "validation_end": MODE_A["validation_end"],
                "test_start": MODE_A["test_start"],
                "test_end": MODE_A["test_end"],
                "preprocessing_mode": "legacy_mode_a",
            }
        )
    if include_mode_b:
        for fold in ROLLING_FOLDS:
            validation_end = fold["train_end"]
            validation_start = previous_month(validation_end, 23)
            specs.append(
                {
                    "mode": "Mode_B_rolling_origin",
                    "fold": fold["fold"],
                    "train_end": previous_month(validation_start, 1),
                    "preprocessing_fit_end": fold["train_end"],
                    "validation_start": validation_start,
                    "validation_end": validation_end,
                    "test_start": fold["test_start"],
                    "test_end": fold["test_end"],
                    "preprocessing_mode": "reviewer_mode_b",
                }
            )
    return specs


def _samples_for_spec(land_type: str, variable: str, spec: Dict, scaling: str) -> Tuple:
    fit_end = spec.get("preprocessing_fit_end", spec["train_end"])
    prepared = prepare_task(
        land_type,
        variable,
        train_end=fit_end,
        preprocessing_mode=spec["preprocessing_mode"],
        scaling=scaling,
    )
    earliest = int(prepared["dates"].min())
    train = build_supervised(prepared, earliest, spec["train_end"], MAIN_LOOKBACK)
    validation = build_supervised(
        prepared, spec["validation_start"], spec["validation_end"], MAIN_LOOKBACK
    )
    test = build_supervised(prepared, spec["test_start"], spec["test_end"], MAIN_LOOKBACK)
    return prepared, train, validation, test


def prediction_frame(
    model_name: str,
    land_type: str,
    variable: str,
    seed: int,
    spec: Dict,
    prepared: Dict,
    test: Dict,
    predicted: np.ndarray,
) -> pd.DataFrame:
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


def common_metric_fields(
    model_name: str,
    land_type: str,
    variable: str,
    seed: int,
    spec: Dict,
    prepared: Dict,
    fit_info: Dict,
) -> Dict:
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
        **fit_info,
    }


def run_torch_tasks(
    model_names: Iterable[str],
    land_types: Iterable[str],
    variables: Iterable[str],
    seeds_for_task,
    include_mode_b_for_model,
    log_path: Path,
    include_mode_a: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pixel_metrics, monthly_metrics, predictions, run_manifest = [], [], [], []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started {datetime.now().isoformat()}\n")
        for model_name in model_names:
            for land_type in land_types:
                for variable in variables:
                    include_mode_b = bool(
                        include_mode_b_for_model(model_name, land_type, variable)
                    )
                    for spec in split_specs(include_mode_b, include_mode_a=include_mode_a):
                        scaling = "monthwise_z"
                        for seed in seeds_for_task(model_name, land_type, variable, spec):
                            started = datetime.now()
                            identity = f"{model_name}/{land_type}/{variable}/{spec['mode']}/fold{spec['fold']}/seed{seed}"
                            try:
                                prepared, train, validation, test = _samples_for_spec(
                                    land_type, variable, spec, scaling
                                )
                                fit = fit_torch_model(
                                    model_name,
                                    variable,
                                    MAIN_LOOKBACK,
                                    train,
                                    validation,
                                    seed,
                                )
                                predicted = predict_torch(fit.model, test)
                                frame = prediction_frame(
                                    model_name,
                                    land_type,
                                    variable,
                                    seed,
                                    spec,
                                    prepared,
                                    test,
                                    predicted,
                                )
                                common = common_metric_fields(
                                    model_name,
                                    land_type,
                                    variable,
                                    seed,
                                    spec,
                                    prepared,
                                    {
                                        "status": "success",
                                        "epochs_run": fit.epochs_run,
                                        "best_validation_loss": fit.best_validation_loss,
                                        "device": fit.device,
                                        "architecture": MODEL_SETTINGS[model_name]["architecture"],
                                    },
                                )
                                pixel, monthly = metric_rows(frame, common)
                                pixel_metrics.append(pixel)
                                monthly_metrics.append(monthly)
                                predictions.append(frame)
                                status = "success"
                                error = ""
                                log.write(
                                    f"SUCCESS {identity} epochs={fit.epochs_run} "
                                    f"pixel_r2={pixel['r2']:.6f} monthly_r2={monthly['r2']:.6f}\n"
                                )
                            except Exception as exc:
                                status = "failed"
                                error = f"{type(exc).__name__}: {exc}"
                                log.write(f"FAILED {identity}: {error}\n{traceback.format_exc()}\n")
                            run_manifest.append(
                                {
                                    "model": model_name,
                                    "land_type": land_type,
                                    "variable": variable,
                                    "mode": spec["mode"],
                                    "fold": spec["fold"],
                                    "seed": seed,
                                    "status": status,
                                    "error": error,
                                    "started": started.isoformat(),
                                    "finished": datetime.now().isoformat(),
                                }
                            )
                            log.flush()
        log.write(f"Finished {datetime.now().isoformat()}\n")
    return (
        pd.DataFrame(pixel_metrics),
        pd.DataFrame(monthly_metrics),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(run_manifest),
    )


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


def run_xgboost_tasks(
    land_types: Iterable[str],
    variables: Iterable[str],
    seeds_for_task,
    include_mode_b: bool,
    log_path: Path,
    include_mode_a: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pixel_metrics, monthly_metrics, predictions, run_manifest = [], [], [], []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started {datetime.now().isoformat()}\n")
        for land_type in land_types:
            for variable in variables:
                for spec in split_specs(include_mode_b, include_mode_a=include_mode_a):
                    for seed in seeds_for_task("XGBoost", land_type, variable, spec):
                        identity = f"XGBoost/{land_type}/{variable}/{spec['mode']}/fold{spec['fold']}/seed{seed}"
                        started = datetime.now()
                        try:
                            prepared, train, validation, test = _samples_for_spec(
                                land_type, variable, spec, "monthwise_z"
                            )
                            train_valid = train["mask"] > 0
                            validation_valid = validation["mask"] > 0
                            model = XGBRegressor(
                                **XGB_SETTINGS,
                                random_state=seed,
                                n_jobs=-1,
                                early_stopping_rounds=20,
                            )
                            model.fit(
                                train["X"][train_valid],
                                train["y"][train_valid],
                                eval_set=[
                                    (
                                        validation["X"][validation_valid],
                                        validation["y"][validation_valid],
                                    )
                                ],
                                verbose=False,
                            )
                            predicted = model.predict(test["X"]).astype(np.float32)
                            frame = prediction_frame(
                                "XGBoost",
                                land_type,
                                variable,
                                seed,
                                spec,
                                prepared,
                                test,
                                predicted,
                            )
                            common = common_metric_fields(
                                "XGBoost",
                                land_type,
                                variable,
                                seed,
                                spec,
                                prepared,
                                {
                                    "status": "success",
                                    "epochs_run": np.nan,
                                    "best_validation_loss": float(model.best_score),
                                    "device": "XGBoost hist",
                                    "architecture": json.dumps(XGB_SETTINGS, sort_keys=True),
                                    "best_iteration": int(model.best_iteration),
                                },
                            )
                            pixel, monthly = metric_rows(frame, common)
                            pixel_metrics.append(pixel)
                            monthly_metrics.append(monthly)
                            predictions.append(frame)
                            status, error = "success", ""
                            log.write(
                                f"SUCCESS {identity} trees={model.best_iteration + 1} "
                                f"pixel_r2={pixel['r2']:.6f} monthly_r2={monthly['r2']:.6f}\n"
                            )
                        except Exception as exc:
                            status = "failed"
                            error = f"{type(exc).__name__}: {exc}"
                            log.write(f"FAILED {identity}: {error}\n{traceback.format_exc()}\n")
                        run_manifest.append(
                            {
                                "model": "XGBoost",
                                "land_type": land_type,
                                "variable": variable,
                                "mode": spec["mode"],
                                "fold": spec["fold"],
                                "seed": seed,
                                "status": status,
                                "error": error,
                                "started": started.isoformat(),
                                "finished": datetime.now().isoformat(),
                            }
                        )
                        log.flush()
        log.write(f"Finished {datetime.now().isoformat()}\n")
    return (
        pd.DataFrame(pixel_metrics),
        pd.DataFrame(monthly_metrics),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(run_manifest),
    )
