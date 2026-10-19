import copy
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from statsmodels.tsa.statespace.structural import UnobservedComponents
from torch.utils.data import DataLoader, TensorDataset

from candidate_config import (
    BILSTM_CONFIG,
    NBEATS_SPEI_CONFIG,
    ROLLING_FOLDS,
    TARGET_CLIP,
)
from model_core import (
    OriginalBiLSTM,
    SeasonalityBasis,
    TrendBasis,
    calculate_metrics,
    set_seed,
)
from raster_pipeline import (
    build_supervised,
    detrend_by_calendar_month,
    load_task_cache,
    month_range,
    prepare_task,
    scale_values,
)


def previous_month(value: int, count: int = 1) -> int:
    year, month = divmod(value, 100)
    absolute = year * 12 + month - 1 - count
    return (absolute // 12) * 100 + (absolute % 12) + 1


def fold_specs() -> List[Dict]:
    specs = []
    for fold in ROLLING_FOLDS:
        validation_end = fold["train_end"]
        validation_start = previous_month(validation_end, 23)
        specs.append(
            {
                "mode": "reviewer_safe_rolling_origin",
                "fold": fold["fold"],
                "model_train_end": previous_month(validation_start),
                "preprocessing_fit_end": fold["train_end"],
                "validation_start": validation_start,
                "validation_end": validation_end,
                "test_start": fold["test_start"],
                "test_end": fold["test_end"],
            }
        )
    return specs


def _robust_monthly_scale(
    values: np.ndarray, dates: np.ndarray, train_end: int, clip: float = 3.0
) -> Tuple[np.ndarray, Dict]:
    result = values.copy()
    train = dates <= train_end
    months = dates % 100
    params = {}
    for month in range(1, 13):
        fit = train & (months == month)
        valid = result[fit][np.isfinite(result[fit])]
        if valid.size == 0:
            continue
        median = float(np.median(valid))
        q25, q75 = np.percentile(valid, [25, 75])
        robust_std = float(max(1e-8, (q75 - q25) / 1.349))
        use = months == month
        result[use] = (result[use] - median) / robust_std
        params[str(month)] = {"median": median, "robust_std": robust_std}
    return np.clip(result, -clip, clip).astype(np.float32), params


def prepare_available_task(
    land_type: str,
    variable: str,
    spec: Dict,
    scaling: str,
) -> Dict:
    if scaling == "robust_monthly":
        cached = load_task_cache(land_type, variable)
        values = cached["values"].copy()
        # The hydroclimate targets have no fixed cross-product conversion.
        values, trends = detrend_by_calendar_month(
            values, cached["dates"], spec["preprocessing_fit_end"]
        )
        values, params = _robust_monthly_scale(
            values,
            cached["dates"],
            spec["preprocessing_fit_end"],
            clip=NBEATS_SPEI_CONFIG["target_clip"],
        )
        cached.update(
            {
                "processed_values": values,
                "unit_scale_factor": 1.0,
                "unit_scale_reason": "reviewer-safe: no test-derived unit correction",
                "trends": trends,
                "scale_params": params,
                "scaling": scaling,
                "preprocessing_mode": "reviewer_safe_train_only",
                "target_observed_mask": np.isfinite(cached["values"]),
                "gap_handling": "available_observations",
            }
        )
        return cached
    prepared = prepare_task(
        land_type,
        variable,
        train_end=spec["preprocessing_fit_end"],
        preprocessing_mode="reviewer_mode_b",
        scaling=scaling,
    )
    prepared["target_observed_mask"] = np.isfinite(prepared["values"])
    prepared["gap_handling"] = "available_observations"
    return prepared


def _fit_aggregate_state_space(
    aggregate: np.ndarray, train_count: int, future_count: int
) -> Tuple[np.ndarray, str]:
    train = pd.Series(aggregate[:train_count], dtype=float)
    try:
        model = UnobservedComponents(
            train,
            level="local linear trend",
            seasonal=12,
            irregular=True,
        )
        result = model.fit(disp=False, maxiter=200)
        fitted = np.asarray(
            result.get_prediction(start=0, end=train_count - 1).predicted_mean,
            dtype=float,
        )
        future = (
            np.asarray(result.forecast(future_count), dtype=float)
            if future_count > 0
            else np.empty(0, dtype=float)
        )
        combined = np.concatenate([fitted, future])
        return combined, "training-only UnobservedComponents local linear trend + seasonal(12)"
    except Exception as exc:
        months = np.arange(len(aggregate)) % 12
        fill = aggregate.copy()
        for month in range(12):
            train_values = aggregate[:train_count][months[:train_count] == month]
            valid = train_values[np.isfinite(train_values)]
            fallback = float(np.mean(valid)) if valid.size else 0.0
            missing = (~np.isfinite(fill)) & (months == month)
            fill[missing] = fallback
        return fill, f"state-space failed; training-only calendar-month mean fallback: {exc}"


def prepare_tws_state_space_task(land_type: str, spec: Dict, scaling: str) -> Dict:
    cached = load_task_cache(land_type, "TWS")
    start = int(cached["dates"].min())
    end = 202412
    full_dates = np.asarray(month_range(start, end), dtype=np.int32)
    n_pixels = cached["values"].shape[1]
    raw = np.full((len(full_dates), n_pixels), np.nan, dtype=np.float32)
    index = {int(date): i for i, date in enumerate(full_dates)}
    for source_i, date in enumerate(cached["dates"]):
        raw[index[int(date)]] = cached["values"][source_i]
    observed_mask = np.isfinite(raw)
    train_count = int(np.searchsorted(full_dates, spec["preprocessing_fit_end"], side="right"))
    valid_count = np.isfinite(raw).sum(axis=1)
    aggregate = np.divide(
        np.nansum(raw, axis=1),
        valid_count,
        out=np.full(len(raw), np.nan, dtype=float),
        where=valid_count > 0,
    )
    aggregate_fill, method = _fit_aggregate_state_space(
        aggregate, train_count, len(full_dates) - train_count
    )
    aggregate_fill = np.where(np.isfinite(aggregate_fill), aggregate_fill, 0.0)

    offsets = np.zeros((12, n_pixels), dtype=np.float32)
    train_dates = full_dates[:train_count]
    train_raw = raw[:train_count]
    train_aggregate = aggregate[:train_count]
    for month in range(1, 13):
        month_mask = (train_dates % 100) == month
        if not month_mask.any():
            continue
        residual = train_raw[month_mask] - train_aggregate[month_mask, None]
        month_offset = np.nanmean(residual, axis=0)
        offsets[month - 1] = np.nan_to_num(month_offset, nan=0.0)

    filled = raw.copy()
    for time_i, date in enumerate(full_dates):
        missing = ~np.isfinite(filled[time_i])
        if missing.any():
            estimate = aggregate_fill[time_i] + offsets[(date % 100) - 1]
            filled[time_i, missing] = estimate[missing]

    filled, trends = detrend_by_calendar_month(
        filled, full_dates, spec["preprocessing_fit_end"]
    )
    if scaling == "robust_monthly":
        processed, scale_params = _robust_monthly_scale(
            filled, full_dates, spec["preprocessing_fit_end"], clip=3.0
        )
    else:
        processed, scale_params = scale_values(
            filled,
            full_dates,
            spec["preprocessing_fit_end"],
            scaling=scaling,
        )
    return {
        **cached,
        "dates": full_dates,
        "values": raw,
        "processed_values": processed.astype(np.float32),
        "target_observed_mask": observed_mask,
        "unit_scale_factor": 1.0,
        "unit_scale_reason": "reviewer-safe: no test-derived unit correction",
        "trends": trends,
        "scale_params": scale_params,
        "scaling": scaling,
        "preprocessing_mode": "reviewer_safe_train_only",
        "gap_handling": "training_only_state_space",
        "gap_method_detail": method,
    }


def build_supervised_observed(
    prepared: Dict, target_start: int, target_end: int, lookback: int
) -> Dict:
    dates = prepared["dates"]
    values = prepared["processed_values"]
    observed_mask = prepared.get("target_observed_mask", np.isfinite(prepared["values"]))
    target_indices = np.flatnonzero((dates >= target_start) & (dates <= target_end))
    target_indices = target_indices[target_indices >= lookback]
    n_pixels = values.shape[1]
    x_rows, y_rows, mask_rows, date_rows, pixel_rows = [], [], [], [], []
    for index in target_indices:
        sequence = values[index - lookback : index].T
        target = values[index]
        mask = observed_mask[index] & np.isfinite(target)
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
        "X": np.concatenate(x_rows).astype(np.float32),
        "y": np.concatenate(y_rows).astype(np.float32),
        "mask": np.concatenate(mask_rows).astype(np.float32),
        "date": np.concatenate(date_rows).astype(np.int32),
        "pixel_id": np.concatenate(pixel_rows).astype(np.int32),
    }


def samples_for_spec(
    land_type: str,
    variable: str,
    spec: Dict,
    lookback: int,
    scaling: str,
    gap_handling: str = "available_observations",
) -> Tuple[Dict, Dict, Dict, Dict]:
    if variable == "TWS" and gap_handling == "training_only_state_space":
        prepared = prepare_tws_state_space_task(land_type, spec, scaling)
        builder = build_supervised_observed
    else:
        prepared = prepare_available_task(land_type, variable, spec, scaling)
        builder = build_supervised
    earliest = int(prepared["dates"].min())
    train = builder(prepared, earliest, spec["model_train_end"], lookback)
    validation = builder(
        prepared, spec["validation_start"], spec["validation_end"], lookback
    )
    test = builder(prepared, spec["test_start"], spec["test_end"], lookback)
    return prepared, train, validation, test


def prediction_frame(
    model_name: str,
    configuration: str,
    land_type: str,
    variable: str,
    seed: int,
    spec: Dict,
    prepared: Dict,
    samples: Dict,
    predicted: np.ndarray,
    lookback: int,
) -> pd.DataFrame:
    pixel_id = samples["pixel_id"]
    return pd.DataFrame(
        {
            "model": model_name,
            "configuration": configuration,
            "mode": spec["mode"],
            "fold": spec["fold"],
            "land_type": land_type,
            "variable": variable,
            "seed": seed,
            "lookback": lookback,
            "date": samples["date"],
            "pixel_id": pixel_id,
            "row": prepared["rows"][pixel_id],
            "col": prepared["cols"][pixel_id],
            "observed": samples["y"],
            "predicted": predicted,
            "valid": samples["mask"].astype(bool),
            "scaling": prepared["scaling"],
            "preprocessing": prepared["preprocessing_mode"],
            "gap_handling": prepared.get("gap_handling", "available_observations"),
            "unit_scale_factor": prepared["unit_scale_factor"],
            "unit_scale_reason": prepared["unit_scale_reason"],
        }
    )


def metric_pair(predictions: pd.DataFrame, common: Dict) -> Tuple[Dict, Dict]:
    valid = predictions["valid"].astype(bool)
    pixel = calculate_metrics(
        predictions.loc[valid, "observed"].to_numpy(),
        predictions.loc[valid, "predicted"].to_numpy(),
    )
    monthly_frame = (
        predictions.loc[valid]
        .groupby("date", as_index=False)[["observed", "predicted"]]
        .mean()
    )
    monthly = calculate_metrics(
        monthly_frame["observed"].to_numpy(),
        monthly_frame["predicted"].to_numpy(),
    )
    pixel.update(common)
    monthly.update(common)
    monthly["n_months"] = monthly.pop("n")
    return pixel, monthly


@dataclass
class NeuralFit:
    model: nn.Module
    best_validation_loss: float
    epochs_run: int
    device: str


def fit_bilstm_special(
    variable: str, train: Dict, validation: Dict, seed: int
) -> NeuralFit:
    cfg = BILSTM_CONFIG[variable]
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OriginalBiLSTM(
        units=cfg["units"],
        dropout=cfg["dropout"],
        bidirectional=cfg["bidirectional"],
    ).to(device)
    train_valid = np.flatnonzero(train["mask"] > 0)
    validation_valid = validation["mask"] > 0
    if len(train_valid) == 0 or validation_valid.sum() == 0:
        raise RuntimeError("No valid training or validation labels")
    sample_cap = 6000
    if len(train_valid) > sample_cap:
        rng = np.random.default_rng(seed)
        train_valid = np.sort(rng.choice(train_valid, sample_cap, replace=False))
    dataset = TensorDataset(
        torch.from_numpy(train["X"][train_valid]),
        torch.from_numpy(train["y"][train_valid]),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_x = torch.from_numpy(validation["X"][validation_valid]).to(device)
    val_y = torch.from_numpy(validation["y"][validation_valid]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(cfg["epochs_cap"]):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = torch.mean((prediction - batch_y) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(torch.mean((model(val_x) - val_y) ** 2).item())
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return NeuralFit(model, best_loss, epochs_run, str(device))


def predict_torch(model: nn.Module, samples: Dict, batch_size: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    output = np.full(len(samples["y"]), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(output), batch_size):
            end = min(len(output), start + batch_size)
            x = torch.from_numpy(samples["X"][start:end]).to(device)
            output[start:end] = model(x).detach().cpu().numpy().astype(np.float32)
    return output


def augment_spei_features(samples: Dict) -> Dict:
    output = dict(samples)
    x = samples["X"]
    if x.shape[1] < 6:
        raise ValueError("SPEI engineered features require at least 6 lookback months")
    features = np.column_stack(
        [
            x[:, -1] - x[:, -3],
            x[:, -1] - x[:, -6],
            np.mean(x[:, -3:], axis=1),
            np.mean(x[:, -6:], axis=1),
            np.std(x[:, -6:], axis=1),
        ]
    ).astype(np.float32)
    output["X"] = np.column_stack([x, features]).astype(np.float32)
    return output


class SpecialNBeatsBlock(nn.Module):
    def __init__(self, lookback: int, theta_size: int, basis: nn.Module):
        super().__init__()
        hidden = NBEATS_SPEI_CONFIG["hidden_units"]
        self.lookback = lookback
        self.basis = basis
        self.layers = nn.ModuleList(
            [nn.Linear(lookback if i == 0 else hidden, hidden) for i in range(4)]
        )
        self.backcast = nn.Linear(hidden, theta_size)
        self.forecast = nn.Linear(hidden, theta_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        backcast = self.basis(self.backcast(x), self.lookback, False)
        forecast = self.basis(self.forecast(x), 1, True)
        return backcast, forecast


class SpecialSPEINBeats(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = NBEATS_SPEI_CONFIG
        lookback = cfg["lookback"]
        bases = [
            TrendBasis(degree=cfg["theta_dims"][0] - 1),
            SeasonalityBasis(harmonics=cfg["harmonics"]),
        ]
        self.stacks = nn.ModuleList()
        for theta_size, basis in zip(cfg["theta_dims"], bases):
            self.stacks.append(
                nn.ModuleList(
                    [
                        SpecialNBeatsBlock(lookback, theta_size, basis)
                        for _ in range(cfg["blocks_per_stack"])
                    ]
                )
            )

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.shape[0], 1, device=x.device)
        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast
        return forecast.squeeze(-1)


def fit_spei_nbeats_special(
    train: Dict, validation: Dict, seed: int
) -> NeuralFit:
    cfg = NBEATS_SPEI_CONFIG
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpecialSPEINBeats().to(device)
    ema_model = copy.deepcopy(model).to(device)
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    train_valid = np.flatnonzero(train["mask"] > 0)
    validation_valid = validation["mask"] > 0
    if len(train_valid) == 0 or validation_valid.sum() == 0:
        raise RuntimeError("No valid training or validation labels")
    sample_cap = 6000
    if len(train_valid) > sample_cap:
        rng = np.random.default_rng(seed)
        train_valid = np.sort(rng.choice(train_valid, sample_cap, replace=False))
    dataset = TensorDataset(
        torch.from_numpy(train["X"][train_valid]),
        torch.from_numpy(train["y"][train_valid]),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_x = torch.from_numpy(validation["X"][validation_valid]).to(device)
    val_y = torch.from_numpy(validation["y"][validation_valid]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    def lr_factor(epoch: int) -> float:
        warmup = cfg["warmup_epochs"]
        total = cfg["epochs_cap"]
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    best_state = copy.deepcopy(ema_model.state_dict())
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    decay = cfg["ema_decay"]
    for epoch in range(cfg["epochs_cap"]):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            noise_scale = 0.02 * torch.std(batch_x).clamp_min(1e-6)
            noisy_x = batch_x + torch.randn_like(batch_x) * noise_scale
            optimizer.zero_grad(set_to_none=True)
            prediction = model(noisy_x)
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            with torch.no_grad():
                for source, target in zip(model.parameters(), ema_model.parameters()):
                    target.mul_(decay).add_(source, alpha=1.0 - decay)
        scheduler.step()
        ema_model.eval()
        with torch.no_grad():
            val_loss = float(
                torch.nn.functional.smooth_l1_loss(ema_model(val_x), val_y).item()
            )
        epochs_run = epoch + 1
        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = copy.deepcopy(ema_model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= cfg["patience"]:
                break
    ema_model.load_state_dict(best_state)
    return NeuralFit(ema_model, best_loss, epochs_run, str(device))


def common_metric_fields(
    model: str,
    configuration: str,
    land_type: str,
    variable: str,
    seed: int,
    spec: Dict,
    prepared: Dict,
    lookback: int,
    extra: Optional[Dict] = None,
) -> Dict:
    return {
        "model": model,
        "configuration": configuration,
        "mode": spec["mode"],
        "fold": spec["fold"],
        "land_type": land_type,
        "variable": variable,
        "seed": seed,
        "lookback": lookback,
        "scaling": prepared["scaling"],
        "preprocessing": prepared["preprocessing_mode"],
        "gap_handling": prepared.get("gap_handling", "available_observations"),
        "selected_pixels": len(prepared["rows"]),
        "test_derived_preprocessing": False,
        **(extra or {}),
    }


def write_prediction_csv_gz(frames: List[pd.DataFrame], path: Path) -> None:
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(
            path, index=False, compression="gzip"
        )
    else:
        pd.DataFrame().to_csv(path, index=False, compression="gzip")


def log_failure(log, identity: str, exc: Exception) -> str:
    error = f"{type(exc).__name__}: {exc}"
    log.write(f"FAILED {identity}: {error}\n{traceback.format_exc()}\n")
    log.flush()
    return error


def run_manifest_row(
    model: str,
    configuration: str,
    land_type: str,
    variable: str,
    seed: int,
    spec: Dict,
    status: str,
    error: str,
    started: datetime,
) -> Dict:
    return {
        "model": model,
        "configuration": configuration,
        "land_type": land_type,
        "variable": variable,
        "fold": spec["fold"],
        "seed": seed,
        "status": status,
        "error": error,
        "started": started.isoformat(),
        "finished": datetime.now().isoformat(),
    }
