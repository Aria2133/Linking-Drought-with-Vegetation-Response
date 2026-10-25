from __future__ import annotations

import gzip
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


OUT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\EWBI_from_existing_forecasts_scale_recovery_and_figure_20260728"
)
PREVIOUS_AUDIT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\EWBI_from_existing_forecasts_visualization_20260728"
)
BASE = Path(r"<PROJECT_ROOT>\Docs")
FULL_CACHE = BASE / "major_revision_analyses" / "05_forecasting_raster_full_candidate" / "cache" / "task_data"
REVIEWER = BASE / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate"
NO_DETREND = (
    BASE
    / "major_revision_analyses"
    / "forecasting_GNDVI_SPEI_no_detrend_rerun"
    / "run_20260726_151258"
)
CORRECTED_BILSTM = (
    BASE / "major_revision_analyses" / "corrected_true_BiLSTM_rerun_20260728"
)
SVR_FULL = (
    BASE
    / "major_revision_analyses"
    / "SVR_full_benchmark_supplementary_candidate_20260723_204013"
)
SVR_UPDATED = (
    BASE
    / "major_revision_analyses"
    / "forecasting_hyperparameter_sensitivity_final"
    / "run_20260727_200736"
    / "05_svr"
)
ECO = BASE / "major_revision_analyses" / "04_detrending_sensitivity_fixed"


COMPONENTS = ["SPEI", "TWS", "VPD", "LST"]
DISPLAY_COMPONENTS = {"SPEI": "SPEI-1", "TWS": "TWS", "VPD": "VPDmax", "LST": "LST"}
SIGNS = {"SPEI": 1.0, "TWS": 1.0, "VPD": -1.0, "LST": -1.0}
LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
LAND_DISPLAY = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "Natural Forests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
MODEL_ORDER = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost", "RBF-epsilon-SVR"]
TARGET_CLIP = 4.0
MAIN_LOOKBACK = 12

# HISTORICAL PREPROCESSING BOUNDARY (kept intentionally): preprocessing is fitted through the end of the internal-
# validation window ("legacy_through_validation", see src/forecasting/fold_boundaries.py). It must stay identical to the published runs
# because it inverts their saved predictions back to physical units.
FOLDS = [
    {
        "mode": "Mode_B_rolling_origin",
        "fold": 1,
        "train_end": 201412,
        "preprocessing_fit_end": 201612,
        "validation_start": 201501,
        "validation_end": 201612,
        "test_start": 201701,
        "test_end": 201812,
    },
    {
        "mode": "Mode_B_rolling_origin",
        "fold": 2,
        "train_end": 201612,
        "preprocessing_fit_end": 201812,
        "validation_start": 201701,
        "validation_end": 201812,
        "test_start": 201901,
        "test_end": 202012,
    },
    {
        "mode": "Mode_B_rolling_origin",
        "fold": 3,
        "train_end": 201812,
        "preprocessing_fit_end": 202012,
        "validation_start": 201901,
        "validation_end": 202012,
        "test_start": 202101,
        "test_end": 202212,
    },
    {
        "mode": "Mode_B_rolling_origin",
        "fold": 4,
        "train_end": 202012,
        "preprocessing_fit_end": 202212,
        "validation_start": 202101,
        "validation_end": 202212,
        "test_start": 202301,
        "test_end": 202412,
    },
]
FOLD_BY_ID = {item["fold"]: item for item in FOLDS}


def ensure_dirs() -> None:
    for sub in [
        "01_scale_recovery",
        "02_components",
        "03_ewbi",
        "04_figure",
        "09_reports",
        "scripts",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def yyyymm_to_timestamp(value: int) -> pd.Timestamp:
    year, month = divmod(int(value), 100)
    return pd.Timestamp(year, month, 1)


def load_task_cache_direct(land_type: str, variable: str) -> Dict:
    path = FULL_CACHE / f"{land_type}_{variable}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {
        "dates": data["dates"].astype(np.int32),
        "values": data["values"].astype(np.float32),
        "rows": data["rows"].astype(np.int32),
        "cols": data["cols"].astype(np.int32),
        "metadata": json.loads(str(data["metadata_json"])),
        "cache_file": str(path),
        "cache_sha256": sha256(path),
    }


def apply_unit_correction(values: np.ndarray, dates: np.ndarray, variable: str) -> Tuple[np.ndarray, float, str]:
    corrected = values.copy()
    if variable == "GPP":
        corrected[dates >= 202201] *= 31.56
        return corrected, 31.56, "reviewer mode: documented fixed GPP conversion only"
    return corrected, 1.0, "reviewer mode: no test-derived unit scaling"


def impute_calendar_month(
    values: np.ndarray, dates: np.ndarray, train_end: int, legacy_all_periods: bool = False
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
    trends: Dict[str, Dict[str, float]] = {}
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


def scale_values(values: np.ndarray, dates: np.ndarray, train_end: int) -> Tuple[np.ndarray, Dict]:
    result = values.copy()
    params: Dict[str, Dict[str, float]] = {}
    train = dates <= train_end
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


def prepare_component(land_type: str, variable: str, fold: int) -> Dict:
    spec = FOLD_BY_ID[int(fold)]
    fit_end = spec["preprocessing_fit_end"]
    cached = load_task_cache_direct(land_type, variable)
    values, factor, factor_reason = apply_unit_correction(cached["values"], cached["dates"], variable)
    missing_before = int(np.isnan(values).sum())
    values = impute_calendar_month(values, cached["dates"], train_end=fit_end, legacy_all_periods=False)
    missing_after = int(np.isnan(values).sum())
    detrending_applied = variable != "SPEI"
    if detrending_applied:
        values, trends = detrend_by_calendar_month(values, cached["dates"], fit_end)
        preprocessing_mode = "reviewer_mode_b"
    else:
        trends = {}
        preprocessing_mode = "reviewer_mode_b_no_detrend"
    scaled, scale_params = scale_values(values, cached["dates"], fit_end)
    cached.update(
        {
            "processed_values": scaled.astype(np.float32),
            "pre_scale_values": values.astype(np.float32),
            "unit_scale_factor": factor,
            "unit_scale_reason": factor_reason,
            "trends": trends,
            "scale_params": scale_params,
            "scaling": "monthwise_z",
            "preprocessing_mode": preprocessing_mode,
            "missing_before_imputation": missing_before,
            "missing_after_imputation": missing_after,
            "detrending_applied": detrending_applied,
            "preprocessing_fit_end": fit_end,
            "spec": spec,
        }
    )
    return cached


def build_supervised_targets(prepared: Dict, target_start: int, target_end: int) -> pd.DataFrame:
    dates = prepared["dates"]
    values = prepared["processed_values"]
    target_indices = np.flatnonzero((dates >= target_start) & (dates <= target_end))
    target_indices = target_indices[target_indices >= MAIN_LOOKBACK]
    rows = []
    n_pixels = values.shape[1]
    pixel_ids = np.arange(n_pixels, dtype=np.int32)
    for index in target_indices:
        y = np.nan_to_num(values[index], nan=0.0).astype(np.float32)
        mask = np.isfinite(values[index])
        frame = pd.DataFrame(
            {
                "date": int(dates[index]),
                "pixel_id": pixel_ids,
                "row": prepared["rows"][pixel_ids],
                "col": prepared["cols"][pixel_ids],
                "reconstructed_observed": y,
                "reconstructed_valid": mask.astype(bool),
            }
        )
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def inverse_standardized(z: np.ndarray, dates: np.ndarray, prepared: Dict) -> np.ndarray:
    z = z.astype(float)
    out = np.full_like(z, np.nan, dtype=float)
    for month in range(1, 13):
        idx = (dates % 100) == month
        if not np.any(idx):
            continue
        params = prepared["scale_params"].get(str(month))
        if params is None:
            continue
        value = z[idx] * params["std"] + params["mean"]
        if prepared["detrending_applied"]:
            trend = prepared["trends"].get(str(month))
            if trend is None:
                out[idx] = np.nan
                continue
            years = dates[idx] // 100
            value = value + trend["slope"] * years + trend["intercept"]
        out[idx] = value
    return out


def forward_standardized(raw: np.ndarray, dates: np.ndarray, prepared: Dict) -> np.ndarray:
    raw = raw.astype(float)
    out = np.full_like(raw, np.nan, dtype=float)
    for month in range(1, 13):
        idx = (dates % 100) == month
        if not np.any(idx):
            continue
        value = raw[idx].copy()
        if prepared["detrending_applied"]:
            trend = prepared["trends"].get(str(month))
            if trend is None:
                out[idx] = np.nan
                continue
            years = dates[idx] // 100
            value = value - (trend["slope"] * years + trend["intercept"])
        params = prepared["scale_params"].get(str(month))
        if params is None:
            continue
        out[idx] = (value - params["mean"]) / params["std"]
    return out


def read_prediction_csv(path: Path, variables: Iterable[str], models: Iterable[str]) -> pd.DataFrame:
    cols = [
        "model",
        "fold",
        "land_type",
        "variable",
        "seed",
        "date",
        "pixel_id",
        "row",
        "col",
        "observed",
        "predicted",
        "valid",
        "scaling",
        "preprocessing",
    ]
    frame = pd.read_csv(path, usecols=lambda c: c in cols)
    frame = frame[frame["variable"].isin(list(variables)) & frame["model"].isin(list(models))].copy()
    frame["source_path"] = str(path)
    return frame


def read_prediction_folder(folder: Path, variables: Iterable[str], models: Iterable[str]) -> pd.DataFrame:
    frames = []
    for path in sorted(folder.glob("*.csv.gz")):
        name = path.name
        if not any(name.startswith(f"{var}_") for var in variables):
            continue
        if not any(f"_{model}_" in name for model in models):
            continue
        frames.append(pd.read_csv(path))
        frames[-1]["source_path"] = str(path)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_svr_predictions() -> Tuple[pd.DataFrame, List[Dict]]:
    metrics_path = SVR_FULL / "metrics" / "svr_outer_test_selected_refit_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    frames = []
    inventory = []
    # TWS, VPD, and LST are still compatible with the accepted detrended SVR package.
    compatible = metrics[
        metrics["target"].isin(["TWS", "VPD", "LST"])
        & (metrics["status"] == "success")
    ].copy()
    for _, row in compatible.iterrows():
        pred_path = Path(row["prediction_path"])
        ok = pred_path.exists()
        inventory.append(
            {
                "model": "RBF-epsilon-SVR",
                "target": row["target"],
                "land_type": row["land_type"],
                "fold": int(row["fold"]),
                "prediction_path": str(pred_path),
                "usable": bool(ok),
                "reason": "accepted detrended SVR final outer-test prediction" if ok else "missing prediction_path",
            }
        )
        if not ok:
            continue
        data = np.load(pred_path, allow_pickle=False)
        frame = pd.DataFrame(
            {
                "model": "RBF-epsilon-SVR",
                "fold": int(row["fold"]),
                "land_type": row["land_type"],
                "variable": row["target"],
                "seed": "selected_validation_config",
                "date": data["date"].astype(np.int32),
                "pixel_id": data["pixel_id"].astype(np.int32),
                "row": data["row"].astype(np.int32),
                "col": data["col"].astype(np.int32),
                "observed": data["y_true"].astype(np.float32),
                "predicted": data["y_pred"].astype(np.float32),
                "valid": np.isfinite(data["y_true"]),
                "scaling": "monthwise_z",
                "preprocessing": "reviewer_mode_b",
                "source_path": str(pred_path),
            }
        )
        frames.append(frame)

    selected_path = SVR_UPDATED / "svr_gndvi_spei_no_detrend_selected_configurations.csv"
    grid_path = SVR_UPDATED / "svr_gndvi_spei_no_detrend_validation_grid.csv"
    no_det_outer = []
    if selected_path.exists():
        selected = pd.read_csv(selected_path)
        no_det_outer = selected[selected["target"] == "SPEI"].to_dict("records")
    for row in no_det_outer:
        inventory.append(
            {
                "model": "RBF-epsilon-SVR",
                "target": "SPEI",
                "land_type": row["land_type"],
                "fold": int(row["fold"]),
                "prediction_path": "",
                "usable": False,
                "reason": (
                    "updated no-detrending SVR file contains validation-grid selection only; "
                    "no final outer-test refit prediction file exists, and refitting is forbidden"
                ),
                "selected_config_source": str(selected_path),
                "validation_grid_source": str(grid_path),
            }
        )
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(), inventory)


def load_all_predictions() -> Tuple[pd.DataFrame, pd.DataFrame]:
    original_lstm = REVIEWER / "predictions" / "reviewer_safe_lstm_predictions.csv.gz"
    original_remaining = REVIEWER / "predictions" / "reviewer_safe_remaining_models_predictions.csv.gz"
    original_xgb = REVIEWER / "predictions" / "reviewer_safe_xgboost_predictions.csv.gz"
    no_det_pred = NO_DETREND / "05_predictions"
    corrected_pred = CORRECTED_BILSTM / "05_predictions"

    frames = []
    frames.append(read_prediction_csv(original_lstm, ["TWS", "VPD", "LST"], ["LSTM"]))
    frames.append(read_prediction_csv(original_xgb, ["TWS", "VPD", "LST"], ["XGBoost"]))
    frames.append(read_prediction_csv(original_remaining, ["TWS", "VPD", "LST"], ["TCN", "N-BEATS", "N-HiTS"]))
    frames.append(read_prediction_csv(original_remaining, ["TWS"], ["BiLSTM"]))
    frames.append(read_prediction_folder(no_det_pred, ["SPEI"], ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]))
    frames.append(read_prediction_folder(corrected_pred, ["VPD", "LST"], ["BiLSTM"]))
    svr, svr_inventory = load_svr_predictions()
    if not svr.empty:
        frames.append(svr)

    combined = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    combined = combined[combined["variable"].isin(COMPONENTS)].copy()
    combined["seed"] = combined["seed"].astype(str)
    combined["valid"] = combined["valid"].astype(bool)
    inventory = []
    for (model, variable, land_type, fold), group in combined.groupby(["model", "variable", "land_type", "fold"]):
        inventory.append(
            {
                "model": model,
                "target": variable,
                "land_type": land_type,
                "fold": int(fold),
                "row_count": int(len(group)),
                "valid_count": int(group["valid"].sum()),
                "seed_count": int(group["seed"].nunique()),
                "source_paths": "; ".join(sorted(map(str, group["source_path"].unique()))[:5]),
                "status": "loaded",
            }
        )
    inventory.extend(svr_inventory)
    return combined, pd.DataFrame(inventory)


def calculate_metrics(observed: np.ndarray, predicted: np.ndarray) -> Dict:
    mask = np.isfinite(observed) & np.isfinite(predicted)
    observed = observed[mask]
    predicted = predicted[mask]
    if observed.size < 2:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "nrmse": np.nan,
            "bias": np.nan,
            "pearson_r": np.nan,
            "n": int(observed.size),
        }
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    std = float(np.std(observed))
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "nrmse": rmse / std if std > 0 else np.nan,
        "bias": float(np.mean(predicted - observed)),
        "pearson_r": float(np.corrcoef(observed, predicted)[0, 1])
        if std > 0 and np.std(predicted) > 0
        else np.nan,
        "n": int(observed.size),
    }


def color_for_value(cmap, norm, value: float) -> str:
    if not np.isfinite(value):
        return "black"
    r, g, b, _ = cmap(norm(value))
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 0.58 else "white"


def create_figure(ewbi: pd.DataFrame, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("white")
    land_order = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
    included_models = [m for m in MODEL_ORDER if m in metrics["model"].unique()]
    nrmse = metrics.pivot(index="model", columns="land_cover", values="nrmse").reindex(index=included_models, columns=land_order)
    r2 = metrics.pivot(index="model", columns="land_cover", values="r2").reindex(index=included_models, columns=land_order)

    for ax, data, title, cmap_name, best_func in [
        (axes[0, 0], nrmse, "(a) EWBI nRMSE", "YlGnBu_r", np.nanmin),
        (axes[0, 1], r2, "(b) EWBI R2", "YlGnBu", np.nanmax),
    ]:
        values = data.to_numpy(dtype=float)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#f2f2f2")
        finite = values[np.isfinite(values)]
        vmin, vmax = (0.0, 1.0) if finite.size == 0 else (float(np.nanmin(finite)), float(np.nanmax(finite)))
        if math.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=25, ha="right")
        ax.set_yticks(range(len(data.index)), data.index)
        ax.set_title(title, loc="left", fontweight="bold")
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                text = "NA" if not np.isfinite(value) else f"{value:.3f}"
                ax.text(col, row, text, ha="center", va="center", fontsize=9, color=color_for_value(cmap, norm, value))
        for col in range(values.shape[1]):
            column = values[:, col]
            if np.isfinite(column).any():
                best = best_func(column)
                for row in np.flatnonzero(np.isclose(column, best, rtol=1e-10, atol=1e-10)):
                    ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=2.0))
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)

    ax = axes[1, 0]
    scatter_models = ["BiLSTM", "XGBoost", "RBF-epsilon-SVR"]
    colors = {"BiLSTM": "#9467bd", "XGBoost": "#17becf", "RBF-epsilon-SVR": "#8c564b"}
    for model in scatter_models:
        subset = ewbi[ewbi["model"] == model]
        if subset.empty:
            continue
        ax.scatter(
            subset["observed_ewbi"],
            subset["predicted_ewbi"],
            s=18,
            alpha=0.45,
            label=model,
            color=colors[model],
            edgecolor="none",
        )
    finite_xy = ewbi[["observed_ewbi", "predicted_ewbi"]].to_numpy(dtype=float)
    finite_xy = finite_xy[np.isfinite(finite_xy).all(axis=1)]
    if finite_xy.size:
        lo = float(np.nanmin(finite_xy))
        hi = float(np.nanmax(finite_xy))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1.0, linestyle="--")
    ax.set_title("(c) Observed versus forecast-derived EWBI", loc="left", fontweight="bold")
    ax.set_xlabel("Observed EWBI")
    ax.set_ylabel("Forecast-derived EWBI")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1, 1]
    xgb = ewbi[ewbi["model"] == "XGBoost"].copy()
    xgb["date_dt"] = xgb["date"].map(yyyymm_to_timestamp)
    for land, group in xgb.groupby("land_cover"):
        # Average across seeds for compact display.
        monthly = group.groupby("date_dt", as_index=False)[["observed_ewbi", "predicted_ewbi"]].mean()
        ax.plot(monthly["date_dt"], monthly["observed_ewbi"], color="black", linewidth=1.1, alpha=0.9)
        ax.plot(monthly["date_dt"], monthly["predicted_ewbi"], linewidth=1.0, alpha=0.85, label=land)
    for boundary in ["2019-01-01", "2021-01-01", "2023-01-01"]:
        ax.axvline(pd.Timestamp(boundary), color="0.5", linewidth=0.7, linestyle=":")
    ax.set_title("(d) Monthly observed and XGBoost forecast-derived EWBI", loc="left", fontweight="bold")
    ax.set_xlabel("Held-out month")
    ax.set_ylabel("EWBI")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle("Forecast-derived reconstruction of EWBI", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(OUT / "04_figure" / f"Figure_forecast_derived_EWBI.{ext}", dpi=300)
    plt.close(fig)


def main() -> None:
    started = time.time()
    ensure_dirs()
    status = "STARTED"
    notes: List[str] = []

    previous_files = [
        "01_audit/EWBI_definition_and_lineage.md",
        "01_audit/EWBI_component_prediction_inventory.csv",
        "01_audit/EWBI_scale_recovery_checks.csv",
        "01_audit/EWBI_alignment_checks.csv",
        "09_reports/EWBI_forecast_propagation_report.md",
        "09_reports/EWBI_forecast_propagation_validation_checks.csv",
    ]
    previous_inventory = [
        {"relative_path": item, "exists": (PREVIOUS_AUDIT / item).exists(), "path": str(PREVIOUS_AUDIT / item)}
        for item in previous_files
    ]

    prediction_rows, prediction_inventory = load_all_predictions()
    prediction_inventory.to_csv(OUT / "01_scale_recovery" / "prediction_source_inventory.csv", index=False)

    prepared: Dict[Tuple[str, str, int], Dict] = {}
    test_targets: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    param_rows = []
    for variable in COMPONENTS:
        for land_type in LAND_TYPES:
            for fold in [1, 2, 3, 4]:
                obj = prepare_component(land_type, variable, fold)
                prepared[(variable, land_type, fold)] = obj
                spec = obj["spec"]
                test_targets[(variable, land_type, fold)] = build_supervised_targets(
                    obj, spec["test_start"], spec["test_end"]
                )
                for month in range(1, 13):
                    scale = obj["scale_params"].get(str(month), {})
                    trend = obj["trends"].get(str(month), {})
                    param_rows.append(
                        {
                            "target": variable,
                            "target_label": DISPLAY_COMPONENTS[variable],
                            "land_type": land_type,
                            "fold": fold,
                            "preprocessing_mode": obj["preprocessing_mode"],
                            "detrending_applied": obj["detrending_applied"],
                            "preprocessing_fit_end": obj["preprocessing_fit_end"],
                            "month": month,
                            "z_mean": scale.get("mean", np.nan),
                            "z_std": scale.get("std", np.nan),
                            "trend_slope": trend.get("slope", np.nan),
                            "trend_intercept": trend.get("intercept", np.nan),
                            "unit_scale_factor": obj["unit_scale_factor"],
                            "unit_scale_reason": obj["unit_scale_reason"],
                            "cache_file": obj["cache_file"],
                            "cache_sha256": obj["cache_sha256"],
                            "missing_before_imputation": obj["missing_before_imputation"],
                            "missing_after_imputation": obj["missing_after_imputation"],
                        }
                    )
    pd.DataFrame(param_rows).to_csv(
        OUT / "01_scale_recovery" / "forecast_preprocessing_parameter_inventory.csv",
        index=False,
    )

    reproduction_rows = []
    inverse_monthly_rows = []
    roundtrip_rows = []
    failed_reproduction = False
    failed_roundtrip = False
    max_reproduction_diff = np.nan
    max_roundtrip_diff = np.nan

    for (model, variable, land_type, fold, seed), group in prediction_rows.groupby(
        ["model", "variable", "land_type", "fold", "seed"], dropna=False
    ):
        key = (variable, land_type, int(fold))
        target_df = test_targets[key]
        merged = group.merge(
            target_df,
            on=["date", "pixel_id", "row", "col"],
            how="left",
            validate="many_to_one",
        )
        diff = merged["observed"].astype(float) - merged["reconstructed_observed"].astype(float)
        finite_diff = diff[np.isfinite(diff)]
        corr = (
            float(np.corrcoef(merged["observed"], merged["reconstructed_observed"])[0, 1])
            if len(merged) >= 2 and np.nanstd(merged["observed"]) > 0 and np.nanstd(merged["reconstructed_observed"]) > 0
            else np.nan
        )
        max_abs = float(np.nanmax(np.abs(finite_diff))) if len(finite_diff) else np.nan
        max_reproduction_diff = np.nanmax([max_reproduction_diff, max_abs])
        status_row = (
            "PASS"
            if len(merged) == len(group)
            and int(merged["reconstructed_observed"].isna().sum()) == 0
            and max_abs <= 1e-5
            and (np.isnan(corr) or corr > 0.999999999)
            else "FAIL"
        )
        if status_row != "PASS":
            failed_reproduction = True
        reproduction_rows.append(
            {
                "model": model,
                "target": variable,
                "land_type": land_type,
                "fold": int(fold),
                "seed": seed,
                "matched_row_count": int(merged["reconstructed_observed"].notna().sum()),
                "stored_row_count": int(len(group)),
                "missing_match_count": int(merged["reconstructed_observed"].isna().sum()),
                "maximum_absolute_difference": max_abs,
                "mean_absolute_difference": float(np.nanmean(np.abs(finite_diff))) if len(finite_diff) else np.nan,
                "rmse_difference": float(np.sqrt(np.nanmean(finite_diff**2))) if len(finite_diff) else np.nan,
                "correlation": corr,
                "proportion_equal_within_1e_6": float(np.nanmean(np.abs(finite_diff) <= 1e-6)) if len(finite_diff) else np.nan,
                "proportion_equal_within_1e_5": float(np.nanmean(np.abs(finite_diff) <= 1e-5)) if len(finite_diff) else np.nan,
                "status": status_row,
            }
        )
        if status_row != "PASS":
            continue

        obj = prepared[key]
        dates = merged["date"].to_numpy(dtype=np.int32)
        observed_raw = inverse_standardized(merged["observed"].to_numpy(dtype=float), dates, obj)
        predicted_raw = inverse_standardized(merged["predicted"].to_numpy(dtype=float), dates, obj)
        z_back = forward_standardized(predicted_raw, dates, obj)
        rt_diff = z_back - merged["predicted"].to_numpy(dtype=float)
        finite_rt = rt_diff[np.isfinite(rt_diff)]
        rt_max = float(np.nanmax(np.abs(finite_rt))) if len(finite_rt) else np.nan
        max_roundtrip_diff = np.nanmax([max_roundtrip_diff, rt_max])
        rt_status = "PASS" if rt_max <= 1e-5 else "FAIL"
        if rt_status != "PASS":
            failed_roundtrip = True
        roundtrip_rows.append(
            {
                "model": model,
                "target": variable,
                "land_type": land_type,
                "fold": int(fold),
                "seed": seed,
                "row_count": int(len(merged)),
                "maximum_absolute_roundtrip_difference": rt_max,
                "mean_absolute_roundtrip_difference": float(np.nanmean(np.abs(finite_rt))) if len(finite_rt) else np.nan,
                "proportion_equal_within_1e_6": float(np.nanmean(np.abs(finite_rt) <= 1e-6)) if len(finite_rt) else np.nan,
                "proportion_equal_within_1e_5": float(np.nanmean(np.abs(finite_rt) <= 1e-5)) if len(finite_rt) else np.nan,
                "status": rt_status,
            }
        )
        if rt_status != "PASS":
            continue

        merged["observed_component_original_scale"] = observed_raw
        merged["predicted_component_original_scale"] = predicted_raw
        valid = merged["valid"].astype(bool) & np.isfinite(observed_raw) & np.isfinite(predicted_raw)
        monthly = (
            merged.loc[valid]
            .groupby("date", as_index=False)
            .agg(
                observed_component=("observed_component_original_scale", "mean"),
                predicted_component=("predicted_component_original_scale", "mean"),
                component_rows=("observed_component_original_scale", "size"),
            )
        )
        monthly["model"] = model
        monthly["seed"] = seed
        monthly["land_type"] = land_type
        monthly["land_cover"] = LAND_DISPLAY[land_type]
        monthly["target"] = variable
        monthly["target_label"] = DISPLAY_COMPONENTS[variable]
        monthly["fold"] = int(fold)
        monthly["preprocessing_mode"] = obj["preprocessing_mode"]
        monthly["detrending_applied"] = obj["detrending_applied"]
        monthly["source_paths"] = "; ".join(sorted(map(str, merged["source_path"].unique()))[:3])
        inverse_monthly_rows.append(monthly)

    reproduction = pd.DataFrame(reproduction_rows)
    reproduction.to_csv(
        OUT / "01_scale_recovery" / "forecast_preprocessing_reproduction_checks.csv",
        index=False,
    )
    roundtrip = pd.DataFrame(roundtrip_rows)
    roundtrip.to_csv(
        OUT / "01_scale_recovery" / "inverse_transformation_roundtrip_checks.csv",
        index=False,
    )

    if failed_reproduction or failed_roundtrip:
        status = "STOPPED_REPRODUCTION_OR_ROUNDTRIP_FAILURE"
        component_monthly = pd.DataFrame()
    else:
        component_monthly = pd.concat(inverse_monthly_rows, ignore_index=True)
        component_monthly.to_csv(
            OUT / "02_components" / "inverse_transformed_component_predictions_monthly.csv",
            index=False,
        )
        status = "SCALE_RECOVERY_PASSED"

    ewbi = pd.DataFrame()
    metrics = pd.DataFrame()
    fold_metrics = pd.DataFrame()
    accepted_match_status = "NOT_EVALUATED"
    accepted_max_diff = np.nan

    if status == "SCALE_RECOVERY_PASSED":
        preprocessed = pd.read_csv(ECO / "tables" / "preprocessed_monthly_series.csv", parse_dates=["date"])
        deseason = preprocessed[
            (preprocessed["preprocessing"] == "deseasonalized_only")
            & (preprocessed["variable"].isin(COMPONENTS))
        ].copy()
        seasonal = (
            deseason.groupby(["land_type", "variable", "month"])["seasonal_component"]
            .first()
            .to_dict()
        )
        baseline_stats = {}
        baseline_mask = (deseason["date"] >= pd.Timestamp("2001-01-01")) & (
            deseason["date"] <= pd.Timestamp("2011-12-01")
        )
        for (land, var), subset in deseason[baseline_mask].groupby(["land_type", "variable"]):
            baseline_stats[(land, var)] = {
                "mean": float(subset["processed_value"].mean()),
                "std": float(subset["processed_value"].std(ddof=1)),
            }

        rows = []
        for _, row in component_monthly.iterrows():
            land = row["land_cover"]
            var = row["target"]
            month = int(row["date"]) % 100
            seas = seasonal.get((land, var, month), np.nan)
            stats = baseline_stats.get((land, var), {"mean": np.nan, "std": np.nan})
            for source_col, out_col in [
                ("observed_component", "observed_component_signed_z"),
                ("predicted_component", "predicted_component_signed_z"),
            ]:
                anomaly = float(row[source_col]) - seas
                z = (anomaly - stats["mean"]) / stats["std"] if stats["std"] > 0 else np.nan
                new = row.to_dict()
                new["component_value_type"] = source_col.replace("_component", "")
                new["component_anomaly"] = anomaly
                new["component_signed_z"] = z * SIGNS[var]
                rows.append(new)
        signed_components = pd.DataFrame(rows)
        value_wide = signed_components.pivot_table(
            index=["model", "seed", "land_type", "land_cover", "fold", "date", "component_value_type"],
            columns="target",
            values="component_signed_z",
            aggfunc="first",
        ).reset_index()
        complete_mask = np.ones(len(value_wide), dtype=bool)
        for var in COMPONENTS:
            if var not in value_wide.columns:
                value_wide[var] = np.nan
            complete_mask &= np.isfinite(value_wide[var].to_numpy(dtype=float))
        value_wide = value_wide[complete_mask].copy()
        value_wide["ewbi"] = value_wide[COMPONENTS].mean(axis=1)
        obs = value_wide[value_wide["component_value_type"] == "observed"].copy()
        pred = value_wide[value_wide["component_value_type"] == "predicted"].copy()
        ewbi = obs.merge(
            pred,
            on=["model", "seed", "land_type", "land_cover", "fold", "date"],
            suffixes=("_observed", "_predicted"),
            validate="one_to_one",
        )
        ewbi = ewbi[
            ["model", "seed", "land_type", "land_cover", "fold", "date", "ewbi_observed", "ewbi_predicted"]
        ].rename(columns={"ewbi_observed": "observed_ewbi", "ewbi_predicted": "predicted_ewbi"})
        ewbi.to_csv(OUT / "03_ewbi" / "observed_and_forecast_derived_EWBI_monthly.csv", index=False)

        accepted = pd.read_csv(ECO / "tables" / "main_ewbi_deseasonalized_series.csv")
        accepted["date"] = pd.to_datetime(accepted["date"]).dt.year * 100 + pd.to_datetime(accepted["date"]).dt.month
        accepted_small = accepted[["land_type", "date", "ewbi"]].rename(
            columns={"land_type": "land_cover", "ewbi": "accepted_observed_ewbi"}
        )
        comp = ewbi.merge(accepted_small, on=["land_cover", "date"], how="left")
        comp["observed_minus_accepted"] = comp["observed_ewbi"] - comp["accepted_observed_ewbi"]
        accepted_max_diff = float(np.nanmax(np.abs(comp["observed_minus_accepted"]))) if not comp.empty else np.nan
        accepted_mean_diff = float(np.nanmean(np.abs(comp["observed_minus_accepted"]))) if not comp.empty else np.nan
        accepted_match = comp.groupby(["land_cover"], as_index=False).agg(
            matched_month_rows=("accepted_observed_ewbi", "count"),
            max_abs_observed_ewbi_difference=("observed_minus_accepted", lambda s: float(np.nanmax(np.abs(s)))),
            mean_abs_observed_ewbi_difference=("observed_minus_accepted", lambda s: float(np.nanmean(np.abs(s)))),
        )
        accepted_match["status"] = np.where(
            accepted_match["max_abs_observed_ewbi_difference"] <= 1e-5,
            "PASS",
            "FAIL_SUPPORT_OR_SOURCE_DIFFERENCE",
        )
        accepted_match.to_csv(OUT / "03_ewbi" / "observed_EWBI_match_to_accepted_output.csv", index=False)
        accepted_match_status = "PASS" if (accepted_match["status"] == "PASS").all() else "FAIL"

        if accepted_match_status == "PASS":
            metric_rows = []
            fold_metric_rows = []
            for (model, land), subset in ewbi.groupby(["model", "land_cover"]):
                row = calculate_metrics(subset["observed_ewbi"].to_numpy(), subset["predicted_ewbi"].to_numpy())
                row.update({"model": model, "land_cover": land, "n_folds": int(subset["fold"].nunique()), "n_seed_runs": int(subset["seed"].nunique())})
                metric_rows.append(row)
            for (model, land, fold), subset in ewbi.groupby(["model", "land_cover", "fold"]):
                row = calculate_metrics(subset["observed_ewbi"].to_numpy(), subset["predicted_ewbi"].to_numpy())
                row.update({"model": model, "land_cover": land, "fold": int(fold), "n_seed_runs": int(subset["seed"].nunique())})
                fold_metric_rows.append(row)
            metrics = pd.DataFrame(metric_rows)
            fold_metrics = pd.DataFrame(fold_metric_rows)
            metrics.to_csv(OUT / "03_ewbi" / "EWBI_metrics_by_model_landcover.csv", index=False)
            fold_metrics.to_csv(OUT / "03_ewbi" / "EWBI_metrics_by_model_landcover_fold.csv", index=False)
            create_figure(ewbi.rename(columns={"ewbi_observed": "observed_ewbi"}), metrics)
            caption = (
                "# Figure Forecast-Derived EWBI Caption\n\n"
                "Forecast-derived reconstruction of EWBI from existing one-month-ahead component predictions. "
                "EWBI was calculated as the equal-weight signed mean of baseline-standardized SPEI-1, TWS, "
                "VPDmax, and LST components. Panels (a) and (b) summarize nRMSE and R2 by model and land-cover "
                "class; outlined cells identify the best value within each land-cover column. Panel (c) compares "
                "observed and forecast-derived EWBI for predefined representative model families. Panel (d) shows "
                "monthly observed and XGBoost-derived EWBI across the held-out rolling-origin periods. Lower nRMSE "
                "and higher R2 indicate better performance."
            )
            (OUT / "04_figure" / "Figure_forecast_derived_EWBI_caption.md").write_text(caption, encoding="utf-8")
            status = "COMPLETED_WITH_FIGURE"
        else:
            status = "STOPPED_ACCEPTED_OBSERVED_EWBI_MISMATCH"
            notes.append(
                "Reconstructed forecast-support observed EWBI did not match the accepted ecological EWBI table. "
                "This indicates a support/source difference between selected forecasting pixels and accepted "
                "land-cover monthly EWBI summaries."
            )

    model_component_counts = (
        component_monthly.groupby(["model", "target"])["date"].count().reset_index(name="monthly_rows")
        if not component_monthly.empty
        else pd.DataFrame()
    )
    if not model_component_counts.empty:
        complete_models = []
        excluded_models = []
        for model in MODEL_ORDER:
            have = set(model_component_counts.loc[model_component_counts["model"] == model, "target"])
            if set(COMPONENTS).issubset(have):
                complete_models.append(model)
            else:
                excluded_models.append({"model": model, "missing_components": ";".join(sorted(set(COMPONENTS) - have))})
    else:
        complete_models = []
        excluded_models = [{"model": model, "missing_components": ";".join(COMPONENTS)} for model in MODEL_ORDER]
    pd.DataFrame(excluded_models).to_csv(OUT / "03_ewbi" / "excluded_model_lineages.csv", index=False)

    validation_rows = [
        {"check": "no forecasting model was fitted or retrained", "status": "PASS", "details": "Only saved predictions and preprocessing caches were read."},
        {"check": "no trained model was loaded for prediction", "status": "PASS", "details": "No model checkpoint was loaded and no prediction method was invoked."},
        {"check": "previous audit files read", "status": "PASS" if all(item["exists"] for item in previous_inventory) else "FAIL", "details": json.dumps(previous_inventory)},
        {"check": "stored observed z-scores reproduced", "status": "PASS" if not failed_reproduction else "FAIL", "details": f"max_abs_difference={max_reproduction_diff}"},
        {"check": "inverse transformation roundtrip", "status": "PASS" if not failed_roundtrip else "FAIL", "details": f"max_abs_difference={max_roundtrip_diff}"},
        {"check": "SPEI-1 no detrending", "status": "PASS", "details": "SPEI prepared with reviewer_mode_b_no_detrend."},
        {"check": "TWS VPDmax LST detrending restored", "status": "PASS", "details": "TWS/VPD/LST prepared with month-specific detrending restored during inversion."},
        {"check": "SVR final no-detrend SPEI outer prediction availability", "status": "FAIL" if any(r.get("target") == "SPEI" and r.get("model") == "RBF-epsilon-SVR" and not r.get("usable", False) for r in prediction_inventory.to_dict("records")) else "PASS", "details": "Updated SVR SPEI no-detrend outer prediction files were not found."},
        {"check": "observed EWBI matched accepted EWBI output", "status": accepted_match_status, "details": f"max_abs_difference={accepted_max_diff}"},
        {"check": "figure values reproducible from saved CSV files", "status": "PASS" if status == "COMPLETED_WITH_FIGURE" else "NOT_CREATED_DUE_STOP_CONDITION", "details": status},
    ]
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(OUT / "09_reports" / "EWBI_scale_recovery_and_propagation_validation.csv", index=False)

    if metrics.empty:
        pd.DataFrame([{"status": "not_created", "reason": status}]).to_csv(
            OUT / "03_ewbi" / "EWBI_metrics_by_model_landcover.csv", index=False
        )
        pd.DataFrame([{"status": "not_created", "reason": status}]).to_csv(
            OUT / "03_ewbi" / "EWBI_metrics_by_model_landcover_fold.csv", index=False
        )
    if not (OUT / "04_figure" / "Figure_forecast_derived_EWBI_caption.md").exists():
        (OUT / "04_figure" / "Figure_forecast_derived_EWBI_caption.md").write_text(
            "# Figure Not Created\n\n"
            f"The forecast-derived EWBI figure was not created because the pipeline stopped with status `{status}`.\n",
            encoding="utf-8",
        )

    report = [
        "# EWBI Scale Recovery and Forecast Propagation Report",
        "",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Outcome",
        "",
        f"Status: `{status}`.",
        "",
        "No forecasting model was fitted, retrained, refit, loaded for prediction, or used to generate new component forecasts.",
        "",
        "## EWBI Definition",
        "",
        "The accepted EWBI equation is the equal-weight signed mean of baseline-standardized components:",
        "",
        "`EWBI = [z(SPEI-1) + z(TWS) - z(VPDmax) - z(LST)] / 4`.",
        "",
        "## Preprocessing Recovery",
        "",
        "Forecasting monthwise-z parameters were reconstructed from the original forecasting caches using the audited reviewer-safe preprocessing logic. SPEI was reconstructed with the approved no-detrending branch; TWS, VPDmax, and LST were reconstructed with calendar-month detrending, monthwise z-scaling, and trend restoration during inversion.",
        "",
        f"Maximum stored-observed reproduction difference: `{max_reproduction_diff}`.",
        f"Maximum prediction round-trip difference: `{max_roundtrip_diff}`.",
        "",
        "## SVR Lineage",
        "",
        "TWS, VPDmax, and LST SVR final outer-test prediction files were available from the full SVR package. For SPEI-1, the later no-detrending SVR package contains validation-grid and selected-configuration tables, but no final outer-test refit prediction file. Because refitting SVR is forbidden in this task, the SVR EWBI lineage is incomplete.",
        "",
        "## Observed EWBI Gate",
        "",
        f"Observed forecast-support EWBI versus accepted ecological EWBI match status: `{accepted_match_status}`.",
        f"Maximum absolute difference from accepted EWBI: `{accepted_max_diff}`.",
        "",
        "If this status is FAIL, the likely reason is that accepted ecological EWBI is based on land-cover monthly raster means, whereas the forecasting prediction files are based on selected eligible forecasting pixels. The script therefore does not create the final figure under that stop condition.",
        "",
        "## Models With Complete Component Coverage",
        "",
        ", ".join(complete_models) if complete_models else "None.",
        "",
        "## Notes",
        "",
        *[f"- {note}" for note in notes],
        "",
        "## Outputs",
        "",
        f"- Parameter inventory: `{OUT / '01_scale_recovery' / 'forecast_preprocessing_parameter_inventory.csv'}`",
        f"- Reproduction checks: `{OUT / '01_scale_recovery' / 'forecast_preprocessing_reproduction_checks.csv'}`",
        f"- Roundtrip checks: `{OUT / '01_scale_recovery' / 'inverse_transformation_roundtrip_checks.csv'}`",
        f"- Component monthly predictions: `{OUT / '02_components' / 'inverse_transformed_component_predictions_monthly.csv'}`",
        f"- EWBI monthly table: `{OUT / '03_ewbi' / 'observed_and_forecast_derived_EWBI_monthly.csv'}`",
        f"- Metrics by model land cover: `{OUT / '03_ewbi' / 'EWBI_metrics_by_model_landcover.csv'}`",
        f"- Fold metrics: `{OUT / '03_ewbi' / 'EWBI_metrics_by_model_landcover_fold.csv'}`",
        f"- Validation checks: `{OUT / '09_reports' / 'EWBI_scale_recovery_and_propagation_validation.csv'}`",
    ]
    (OUT / "09_reports" / "EWBI_scale_recovery_and_propagation_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "runtime_seconds": time.time() - started,
        "no_model_rerun": True,
        "max_reproduction_difference": None if np.isnan(max_reproduction_diff) else max_reproduction_diff,
        "max_roundtrip_difference": None if np.isnan(max_roundtrip_diff) else max_roundtrip_diff,
        "accepted_ewbi_max_difference": None if np.isnan(accepted_max_diff) else accepted_max_diff,
        "models_complete": complete_models,
        "outputs_root": str(OUT),
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
