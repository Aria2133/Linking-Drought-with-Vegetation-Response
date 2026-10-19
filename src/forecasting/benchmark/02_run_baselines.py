import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from candidate_config import (
    LAND_TYPES,
    LOGS_DIR,
    MAIN_LOOKBACK,
    PREDICTIONS_DIR,
    TABLES_DIR,
    VARIABLES,
)
from model_core import calculate_metrics, metric_rows
from raster_pipeline import build_supervised, month_range, prepare_task
from run_helpers import split_specs

PIXEL_OUT = TABLES_DIR / "reviewer_safe_baseline_pixel_metrics.csv"
MONTHLY_OUT = TABLES_DIR / "reviewer_safe_baseline_monthly_aggregated_metrics.csv"
PREDICTION_OUT = PREDICTIONS_DIR / "reviewer_safe_baseline_predictions.csv.gz"


def prediction_frame(model, land_type, variable, spec, prepared, test, predicted):
    pixel_id = test["pixel_id"]
    valid = (test["mask"] > 0) & np.isfinite(predicted)
    return pd.DataFrame(
        {
            "model": model,
            "mode": "reviewer_safe_rolling_origin",
            "fold": spec["fold"],
            "land_type": land_type,
            "variable": variable,
            "seed": "deterministic",
            "lookback": MAIN_LOOKBACK,
            "date": test["date"],
            "pixel_id": pixel_id,
            "row": prepared["rows"][pixel_id],
            "col": prepared["cols"][pixel_id],
            "observed": test["y"],
            "predicted": predicted,
            "valid": valid,
            "scaling": "monthwise_z",
            "preprocessing": "reviewer_mode_b",
            "unit_scale_factor": prepared["unit_scale_factor"],
            "unit_scale_reason": prepared["unit_scale_reason"],
            "prediction_level": "pixel",
        }
    )


def common_fields(model, land_type, variable, spec, prepared, status="success", error=""):
    return {
        "model": model,
        "mode": "reviewer_safe_rolling_origin",
        "fold": spec["fold"],
        "land_type": land_type,
        "variable": variable,
        "seed": "deterministic",
        "lookback": MAIN_LOOKBACK,
        "scaling": "monthwise_z",
        "preprocessing": "reviewer_mode_b",
        "unit_scale_factor": prepared["unit_scale_factor"],
        "unit_scale_reason": prepared["unit_scale_reason"],
        "selected_pixels": len(prepared["rows"]),
        "status": status,
        "error": error,
    }


def subtract_year(date):
    return date - 100


def monthly_climatology(prepared, fold_train_end, test):
    dates = prepared["dates"]
    values = prepared["processed_values"]
    output = np.full(len(test["y"]), np.nan, dtype=np.float32)
    for date in np.unique(test["date"]):
        month = date % 100
        fit = (dates <= fold_train_end) & (dates % 100 == month)
        climatology = np.nanmean(values[fit], axis=0)
        rows = test["date"] == date
        output[rows] = climatology[test["pixel_id"][rows]]
    return output


def seasonal_persistence(prepared, test):
    date_to_index = {int(date): index for index, date in enumerate(prepared["dates"])}
    output = np.full(len(test["y"]), np.nan, dtype=np.float32)
    for date in np.unique(test["date"]):
        source_index = date_to_index.get(subtract_year(int(date)))
        rows = test["date"] == date
        if source_index is not None:
            output[rows] = prepared["processed_values"][
                source_index, test["pixel_id"][rows]
            ]
    return output


def sarima_monthly(prepared, spec):
    dates = prepared["dates"]
    values = prepared["processed_values"]
    aggregate = np.nanmean(values, axis=1)
    train_mask = dates <= spec["preprocessing_fit_end"]
    train_dates = dates[train_mask]
    train_values = aggregate[train_mask]
    full_train_dates = month_range(int(train_dates.min()), int(spec["preprocessing_fit_end"]))
    lookup = dict(zip(train_dates.tolist(), train_values.tolist()))
    series = pd.Series([lookup.get(date, np.nan) for date in full_train_dates], dtype=float)
    series = series.interpolate(limit_direction="both")
    if series.notna().sum() < 36:
        raise RuntimeError("insufficient monthly aggregate training values for SARIMA")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            series.to_numpy(),
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 12),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False, maxiter=75)
    forecast_dates = month_range(
        int(spec["preprocessing_fit_end"] // 100 * 100 + spec["preprocessing_fit_end"] % 100),
        int(spec["test_end"]),
    )[1:]
    forecast = fit.forecast(len(forecast_dates))
    forecast_lookup = dict(zip(forecast_dates, forecast))
    test_mask = (dates >= spec["test_start"]) & (dates <= spec["test_end"])
    observed_dates = dates[test_mask]
    observed = aggregate[test_mask]
    predicted = np.array(
        [forecast_lookup.get(int(date), np.nan) for date in observed_dates], dtype=float
    )
    return observed_dates, observed, predicted, bool(getattr(fit, "mle_retvals", {}).get("converged", True))


def main():
    log_path = LOGS_DIR / "reviewer_safe_baselines_log.txt"
    if PIXEL_OUT.exists() and MONTHLY_OUT.exists() and PREDICTION_OUT.exists():
        existing = pd.read_csv(PIXEL_OUT)
        if len(existing) >= 432:
            log_path.write_text(
                "Existing complete deterministic baseline outputs verified; expensive rerun skipped.\n",
                encoding="utf-8",
            )
            return

    pixel_metrics = []
    monthly_metrics = []
    prediction_frames = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {datetime.now().isoformat()}\n")
        for land_type in LAND_TYPES:
            for variable in VARIABLES:
                for spec in split_specs(include_mode_b=True, include_mode_a=False):
                    try:
                        prepared = prepare_task(
                            land_type,
                            variable,
                            train_end=spec["preprocessing_fit_end"],
                            preprocessing_mode="reviewer_mode_b",
                            scaling="monthwise_z",
                        )
                        test = build_supervised(
                            prepared, spec["test_start"], spec["test_end"], MAIN_LOOKBACK
                        )
                        predictions = {
                            "Persistence": test["X"][:, -1].copy(),
                            "Seasonal persistence": seasonal_persistence(prepared, test),
                            "Monthly climatology": monthly_climatology(
                                prepared, spec["preprocessing_fit_end"], test
                            ),
                        }
                        for model, predicted in predictions.items():
                            frame = prediction_frame(
                                model, land_type, variable, spec, prepared, test, predicted
                            )
                            common = common_fields(
                                model, land_type, variable, spec, prepared
                            )
                            pixel, monthly = metric_rows(frame, common)
                            pixel_metrics.append(pixel)
                            monthly_metrics.append(monthly)
                            prediction_frames.append(frame)
                        try:
                            dates, observed, predicted, converged = sarima_monthly(
                                prepared, spec
                            )
                            metrics = calculate_metrics(observed, predicted)
                            metrics.update(
                                common_fields(
                                    "SARIMA", land_type, variable, spec, prepared
                                )
                            )
                            metrics["n_months"] = metrics.pop("n")
                            metrics["sarima_converged"] = converged
                            metrics["metric_level_availability"] = (
                                "monthly aggregated only; per-pixel SARIMA infeasible"
                            )
                            monthly_metrics.append(metrics)
                            prediction_frames.append(
                                pd.DataFrame(
                                    {
                                        "model": "SARIMA",
                                        "mode": "reviewer_safe_rolling_origin",
                                        "fold": spec["fold"],
                                        "land_type": land_type,
                                        "variable": variable,
                                        "seed": "deterministic",
                                        "lookback": MAIN_LOOKBACK,
                                        "date": dates,
                                        "pixel_id": -1,
                                        "row": -1,
                                        "col": -1,
                                        "observed": observed,
                                        "predicted": predicted,
                                        "valid": np.isfinite(observed)
                                        & np.isfinite(predicted),
                                        "scaling": "monthwise_z",
                                        "preprocessing": "reviewer_mode_b",
                                        "unit_scale_factor": prepared[
                                            "unit_scale_factor"
                                        ],
                                        "unit_scale_reason": prepared[
                                            "unit_scale_reason"
                                        ],
                                        "prediction_level": "monthly_aggregated",
                                    }
                                )
                            )
                        except Exception as sarima_exc:
                            log.write(
                                f"SARIMA PARTIAL {land_type}/{variable}/fold{spec['fold']}: "
                                f"{type(sarima_exc).__name__}: {sarima_exc}\n"
                            )
                        log.write(
                            f"SUCCESS {land_type}/{variable}/fold{spec['fold']} "
                            "persistence+seasonal+climatology\n"
                        )
                    except Exception as exc:
                        log.write(
                            f"FAILED {land_type}/{variable}/fold{spec['fold']}: "
                            f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}\n"
                        )
                    log.flush()
        pd.DataFrame(pixel_metrics).to_csv(PIXEL_OUT, index=False)
        pd.DataFrame(monthly_metrics).to_csv(MONTHLY_OUT, index=False)
        pd.concat(prediction_frames, ignore_index=True).to_csv(
            PREDICTION_OUT, index=False, compression="gzip"
        )
        log.write(f"Finished: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
