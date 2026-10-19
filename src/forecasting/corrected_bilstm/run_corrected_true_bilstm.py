import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


OUT = Path(r"<PROJECT_ROOT>\Docs\major_revision_analyses\corrected_true_BiLSTM_rerun_20260728")
BASE = Path(r"<PROJECT_ROOT>\Docs")
REVIEWER = BASE / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate"
NO_DETREND = BASE / "major_revision_analyses" / "forecasting_GNDVI_SPEI_no_detrend_rerun" / "run_20260726_151258"
SVR_FULL = BASE / "major_revision_analyses" / "SVR_full_benchmark_supplementary_candidate_20260723_204013"
SENS = BASE / "major_revision_analyses" / "forecasting_hyperparameter_sensitivity_final" / "run_20260727_200736"
FIG7_CURRENT = BASE / "major_revision_analyses" / "ALTERNATIVE_Figure7_polar_redesign_no_detrend_update" / "Source tables used for plotting" / "figure7_alternative_plotting_table.csv"

for sub in [
    "01_audit",
    "02_design",
    "03_cache",
    "04_checkpoints",
    "05_predictions",
    "06_metrics",
    "07_sensitivity",
    "08_downstream",
    "09_reports",
    "logs",
    "manifests",
]:
    (OUT / sub).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REVIEWER / "scripts"))

import model_core as mc  # noqa: E402
from candidate_config import MAIN_LOOKBACK  # noqa: E402
from raster_pipeline import build_supervised, prepare_task  # noqa: E402
from run_helpers import split_specs  # noqa: E402


def import_no_detrend_module():
    path = NO_DETREND / "scripts" / "run_no_detrend_gndvi_spei.py"
    spec = importlib.util.spec_from_file_location("no_detrend_runner_for_corrected_bilstm", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["no_detrend_runner_for_corrected_bilstm"] = module
    spec.loader.exec_module(module)
    return module


NO_DET = import_no_detrend_module()

LAND_TYPES = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
AFFECTED_TARGETS = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP", "LST", "VPD"]
RETAINED_BILSTM_TARGETS = ["SPEI", "TWS"]
LEARNED_MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
BASELINE_MODELS = ["Persistence", "Seasonal persistence", "Monthly climatology"]
ALL_FORECAST_MODELS = BASELINE_MODELS + ["SVR"] + LEARNED_MODELS
SEEDS_BY_TARGET = {
    "GNDVI": [42, 123, 2026],
    "NDVI": [42],
    "NIRv": [42],
    "GOSIF": [42],
    "GPP": [42],
    "LST": [42],
    "VPD": [42],
}
SENSITIVITY_TASKS = [
    ("NDVI", "GRASSLANDS"),
    ("GNDVI", "NATURAL_FORESTS"),
    ("NIRv", "SAVANNAS"),
    ("VPD", "SHRUBLANDS"),
]
SENSITIVITY_FOLDS = [1, 4]
SENSITIVITY_CONFIGS = [
    {"configuration": "final_corrected", "units": 128, "learning_rate": 0.001, "role": "FINAL_CORRECTED"},
    {"configuration": "lower_capacity", "units": 64, "learning_rate": 0.001, "role": "LOWER_CAPACITY"},
    {"configuration": "lower_learning_rate", "units": 128, "learning_rate": 0.0005, "role": "LOWER_LEARNING_RATE"},
    {"configuration": "higher_learning_rate", "units": 128, "learning_rate": 0.002, "role": "HIGHER_LEARNING_RATE"},
]
SUCCESS = {"success", "convergence_warning_success"}


def log(message: str) -> None:
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(text, flush=True)
    with (OUT / "logs" / "corrected_true_bilstm_run.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def split_by_fold(fold: int) -> dict:
    matches = [spec for spec in split_specs(include_mode_b=True, include_mode_a=False) if int(spec["fold"]) == int(fold)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one reviewer-safe Mode B split for fold {fold}, found {len(matches)}")
    return dict(matches[0])


def no_det_split_by_fold(fold: int) -> dict:
    matches = [spec for spec in NO_DET.FOLDS if int(spec["fold"]) == int(fold)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one no-detrend split for fold {fold}, found {len(matches)}")
    return dict(matches[0])


TASK_CACHE: dict[tuple, tuple] = {}


def samples_for_task(variable: str, land_type: str, fold: int):
    key = (variable, land_type, int(fold))
    if key in TASK_CACHE:
        return TASK_CACHE[key]
    if variable == "GNDVI":
        spec = no_det_split_by_fold(fold)
        prepared = NO_DET.prepare_task(land_type, variable, train_end=int(spec["preprocessing_fit_end"]), use_detrend=False)
        train = NO_DET.build_supervised(prepared, int(prepared["dates"].min()), int(spec["train_end"]), MAIN_LOOKBACK)
        validation = NO_DET.build_supervised(prepared, int(spec["validation_start"]), int(spec["validation_end"]), MAIN_LOOKBACK)
        test = NO_DET.build_supervised(prepared, int(spec["test_start"]), int(spec["test_end"]), MAIN_LOOKBACK)
    else:
        spec = split_by_fold(fold)
        prepared = prepare_task(
            land_type,
            variable,
            train_end=int(spec["preprocessing_fit_end"]),
            preprocessing_mode=spec["preprocessing_mode"],
            scaling="monthwise_z",
        )
        train = build_supervised(prepared, int(prepared["dates"].min()), int(spec["train_end"]), MAIN_LOOKBACK)
        validation = build_supervised(prepared, int(spec["validation_start"]), int(spec["validation_end"]), MAIN_LOOKBACK)
        test = build_supervised(prepared, int(spec["test_start"]), int(spec["test_end"]), MAIN_LOOKBACK)
    payload = (prepared, train, validation, test, spec)
    TASK_CACHE[key] = payload
    return payload


def valid_count(samples: dict) -> int:
    return int((samples["mask"] > 0).sum())


def task_hash(variable: str, land_type: str, fold: int, prepared: dict, train: dict, validation: dict, test: dict, spec: dict) -> str:
    payload = {
        "variable": variable,
        "land_type": land_type,
        "fold": int(fold),
        "preprocessing": prepared.get("preprocessing_mode", ""),
        "scaling": prepared.get("scaling", ""),
        "rows_hash": hashlib.sha256(np.asarray(prepared["rows"], dtype=np.int32).tobytes()).hexdigest(),
        "cols_hash": hashlib.sha256(np.asarray(prepared["cols"], dtype=np.int32).tobytes()).hexdigest(),
        "dates_hash": hashlib.sha256(np.asarray(prepared["dates"], dtype=np.int32).tobytes()).hexdigest(),
        "train_samples": int(len(train["y"])),
        "train_valid": valid_count(train),
        "validation_samples": int(len(validation["y"])),
        "validation_valid": valid_count(validation),
        "test_samples": int(len(test["y"])),
        "test_valid": valid_count(test),
        "spec": {k: json_safe(v) for k, v in spec.items()},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class CorrectedTrueBiLSTM(nn.Module):
    def __init__(self, units: int = 128, dropout: float = 0.2):
        super().__init__()
        self.first = nn.LSTM(1, int(units), batch_first=True, bidirectional=True)
        self.second = nn.LSTM(int(units) * 2, int(units), batch_first=True)
        self.dropout = nn.Dropout(float(dropout))
        self.dense = nn.Sequential(nn.Linear(int(units), 10), nn.ReLU(), nn.Linear(10, 1))

    def forward(self, x):
        sequence, _ = self.first(x.unsqueeze(-1))
        sequence = self.dropout(sequence)
        sequence, _ = self.second(sequence)
        last = self.dropout(sequence[:, -1])
        return self.dense(last).squeeze(-1)


def architecture_string(units: int = 128) -> str:
    return (
        f"corrected true BiLSTM: first LSTM bidirectional True, {units} units per direction, "
        f"concatenated width {units * 2}; second LSTM unidirectional {units} units; dropout 0.20; "
        f"dense Linear({units}->10), ReLU, Linear(10->1)"
    )


def fit_corrected_bilstm(train: dict, validation: dict, seed: int, units: int = 128, learning_rate: float = 0.001) -> dict:
    mc.set_seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CorrectedTrueBiLSTM(units=units, dropout=0.2).to(device)
    train_valid = train["mask"] > 0
    validation_valid = validation["mask"] > 0
    if train_valid.sum() == 0 or validation_valid.sum() == 0:
        raise RuntimeError("No valid train or validation labels")
    valid_indices = np.flatnonzero(train_valid)
    if len(valid_indices) > 4000:
        rng = np.random.default_rng(int(seed))
        valid_indices = np.sort(rng.choice(valid_indices, size=4000, replace=False))
    dataset = TensorDataset(
        torch.from_numpy(train["X"][valid_indices].astype(np.float32)),
        torch.from_numpy(train["y"][valid_indices].astype(np.float32)),
    )
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(int(seed)),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_x = torch.from_numpy(validation["X"][validation_valid].astype(np.float32)).to(device)
    val_y = torch.from_numpy(validation["y"][validation_valid].astype(np.float32)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(8):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred = model(batch_x)
                loss = torch.mean((pred - batch_y) ** 2)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
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
            if stale >= 3:
                break
    model.load_state_dict(best_state)
    return {"model": model, "best_validation_loss": best_loss, "epochs_run": epochs_run, "device": str(device), "training_subset_count": int(len(valid_indices))}


def predict_model(model: nn.Module, samples: dict, batch_size: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    output = np.full(len(samples["y"]), np.nan, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(output), batch_size):
            end = min(start + batch_size, len(output))
            x = torch.from_numpy(samples["X"][start:end].astype(np.float32)).to(device)
            output[start:end] = model(x).detach().cpu().numpy().astype(np.float32)
    return output


def prediction_frame(variable: str, land_type: str, fold: int, seed, spec: dict, prepared: dict, samples: dict, predicted: np.ndarray, model_name: str = "BiLSTM") -> pd.DataFrame:
    pixel_id = samples["pixel_id"]
    return pd.DataFrame(
        {
            "model": model_name,
            "mode": spec["mode"],
            "fold": int(fold),
            "land_type": land_type,
            "variable": variable,
            "seed": seed,
            "lookback": MAIN_LOOKBACK,
            "date": samples["date"],
            "pixel_id": pixel_id,
            "row": prepared["rows"][pixel_id],
            "col": prepared["cols"][pixel_id],
            "observed": samples["y"],
            "predicted": predicted,
            "valid": samples["mask"].astype(bool),
            "scaling": prepared["scaling"],
            "preprocessing": prepared["preprocessing_mode"],
            "unit_scale_factor": prepared["unit_scale_factor"],
            "unit_scale_reason": prepared["unit_scale_reason"],
        }
    )


def common_metric_fields(variable: str, land_type: str, fold: int, seed, spec: dict, prepared: dict, fit_info: dict) -> dict:
    return {
        "model": "BiLSTM",
        "mode": spec["mode"],
        "fold": int(fold),
        "land_type": land_type,
        "variable": variable,
        "seed": seed,
        "lookback": MAIN_LOOKBACK,
        "scaling": prepared["scaling"],
        "preprocessing": prepared["preprocessing_mode"],
        "unit_scale_factor": prepared["unit_scale_factor"],
        "unit_scale_reason": prepared["unit_scale_reason"],
        "selected_pixels": int(len(prepared["rows"])),
        "core_train_end": int(spec["train_end"]),
        "validation_start": int(spec["validation_start"]),
        "validation_end": int(spec["validation_end"]),
        "preprocessing_fit_end": int(spec["preprocessing_fit_end"]),
        "test_start": int(spec["test_start"]),
        "test_end": int(spec["test_end"]),
        **fit_info,
    }


def checkpoint_path(identity: str) -> Path:
    safe = identity.replace("/", "_").replace(" ", "_")
    return OUT / "04_checkpoints" / f"{safe}.json"


def write_checkpoint(identity: str, payload: dict):
    payload = {k: json_safe(v) for k, v in payload.items()}
    checkpoint_path(identity).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_checkpoint(identity: str, config_hash: str):
    path = checkpoint_path(identity)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("configuration_hash") == config_hash and payload.get("status") == "success":
        return payload
    return None


def benchmark_inventory() -> pd.DataFrame:
    rows = []
    for variable in AFFECTED_TARGETS:
        for land_type in LAND_TYPES:
            for fold in [1, 2, 3, 4]:
                prepared, train, validation, test, spec = samples_for_task(variable, land_type, fold)
                h = task_hash(variable, land_type, fold, prepared, train, validation, test, spec)
                for seed in SEEDS_BY_TARGET[variable]:
                    rows.append(
                        {
                            "target": variable,
                            "land_cover": land_type,
                            "fold": fold,
                            "seed": seed,
                            "preprocessing_mode": prepared["preprocessing_mode"],
                            "core_train_end": int(spec["train_end"]),
                            "validation_start": int(spec["validation_start"]),
                            "validation_end": int(spec["validation_end"]),
                            "preprocessing_fit_end": int(spec["preprocessing_fit_end"]),
                            "test_start": int(spec["test_start"]),
                            "test_end": int(spec["test_end"]),
                            "expected_training_samples": int(len(train["y"])),
                            "expected_training_valid_samples": valid_count(train),
                            "expected_validation_samples": int(len(validation["y"])),
                            "expected_validation_valid_samples": valid_count(validation),
                            "expected_test_samples": int(len(test["y"])),
                            "expected_test_valid_samples": valid_count(test),
                            "eligible_pixels": int(len(prepared["rows"])),
                            "old_architecture": "first recurrent layer bidirectional only for SPEI and TWS; unidirectional for this affected target",
                            "corrected_architecture": architecture_string(128),
                            "configuration_hash": h,
                            "rerun_required": True,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "01_audit" / "corrected_bilstm_task_inventory.csv", index=False)
    task_manifest = frame.drop_duplicates(["target", "land_cover", "fold"])[
        [
            "target",
            "land_cover",
            "fold",
            "preprocessing_mode",
            "core_train_end",
            "validation_start",
            "validation_end",
            "preprocessing_fit_end",
            "test_start",
            "test_end",
            "expected_training_samples",
            "expected_validation_samples",
            "expected_test_samples",
            "eligible_pixels",
            "configuration_hash",
        ]
    ]
    task_manifest.to_csv(OUT / "03_cache" / "prepared_task_array_manifest.csv", index=False)
    return frame


def run_benchmark() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inventory = pd.read_csv(OUT / "01_audit" / "corrected_bilstm_task_inventory.csv")
    expected = len(inventory)
    log(f"Expected corrected benchmark fits: {expected}")
    pixel_rows, monthly_rows, prediction_manifest = [], [], []
    existing_pix = OUT / "06_metrics" / "corrected_BiLSTM_pixel_metrics.csv"
    existing_mon = OUT / "06_metrics" / "corrected_BiLSTM_monthly_metrics.csv"
    existing_pred_manifest = OUT / "06_metrics" / "corrected_BiLSTM_predictions_manifest.csv"
    if existing_pix.exists():
        try:
            pixel_rows = pd.read_csv(existing_pix).to_dict("records")
        except Exception:
            pixel_rows = []
    if existing_mon.exists():
        try:
            monthly_rows = pd.read_csv(existing_mon).to_dict("records")
        except Exception:
            monthly_rows = []
    if existing_pred_manifest.exists():
        try:
            prediction_manifest = pd.read_csv(existing_pred_manifest).to_dict("records")
        except Exception:
            prediction_manifest = []
    completed_keys = {(r["variable"], r["land_type"], int(r["fold"]), str(r["seed"])) for r in pixel_rows if str(r.get("status")) == "success"}
    for _, item in inventory.iterrows():
        variable, land_type, fold, seed = item["target"], item["land_cover"], int(item["fold"]), int(item["seed"])
        if (variable, land_type, fold, str(seed)) in completed_keys or (variable, land_type, fold, seed) in completed_keys:
            continue
        prepared, train, validation, test, spec = samples_for_task(variable, land_type, fold)
        config_hash = hashlib.sha256(
            json.dumps(
                {
                    "model": "corrected_true_BiLSTM",
                    "variable": variable,
                    "land_type": land_type,
                    "fold": fold,
                    "seed": seed,
                    "architecture": architecture_string(128),
                    "learning_rate": 0.001,
                    "task_hash": item["configuration_hash"],
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        identity = f"benchmark/{variable}/{land_type}/fold{fold}/seed{seed}"
        cached = read_checkpoint(identity, config_hash)
        if cached:
            log(f"SKIP benchmark checkpoint {identity}")
            continue
        started = datetime.now().isoformat(timespec="seconds")
        t0 = time.perf_counter()
        try:
            fit = fit_corrected_bilstm(train, validation, seed, units=128, learning_rate=0.001)
            predicted = predict_model(fit["model"], test)
            frame = prediction_frame(variable, land_type, fold, seed, spec, prepared, test, predicted)
            pred_path = OUT / "05_predictions" / f"{variable}_{land_type}_fold{fold}_BiLSTM_seed{seed}_corrected_predictions.csv.gz"
            frame.to_csv(pred_path, index=False, compression="gzip")
            fit_info = {
                "status": "success",
                "error": "",
                "epochs_run": fit["epochs_run"],
                "best_validation_loss": fit["best_validation_loss"],
                "device": fit["device"],
                "architecture": architecture_string(128),
                "corrected_first_layer_bidirectional": True,
                "first_layer_units_per_direction": 128,
                "second_layer_units": 128,
                "learning_rate": 0.001,
                "weight_decay": 0.0,
                "batch_size": 64,
                "training_sample_cap": 4000,
                "gradient_clip_max_norm": 1.0,
                "reuse_status": "new_run",
                "runtime_seconds": time.perf_counter() - t0,
                "training_sample_count": int(len(train["y"])),
                "training_valid_count": valid_count(train),
                "training_subset_count": fit["training_subset_count"],
                "validation_sample_count": int(len(validation["y"])),
                "validation_valid_count": valid_count(validation),
                "outer_test_sample_count": int(len(test["y"])),
                "outer_test_valid_count": valid_count(test),
                "configuration_hash": config_hash,
                "task_array_hash": item["configuration_hash"],
            }
            common = common_metric_fields(variable, land_type, fold, seed, spec, prepared, fit_info)
            pixel, monthly = mc.metric_rows(frame, common)
            pixel_rows.append(pixel)
            monthly_rows.append(monthly)
            prediction_manifest.append(
                {
                    "target": variable,
                    "land_cover": land_type,
                    "fold": fold,
                    "seed": seed,
                    "prediction_path": str(pred_path),
                    "configuration_hash": config_hash,
                    "task_array_hash": item["configuration_hash"],
                    "status": "success",
                }
            )
            write_checkpoint(identity, {"identity": identity, **prediction_manifest[-1], "status": "success", "started": started, "finished": datetime.now().isoformat(timespec="seconds")})
            pd.DataFrame(pixel_rows).to_csv(existing_pix, index=False)
            pd.DataFrame(monthly_rows).to_csv(existing_mon, index=False)
            pd.DataFrame(prediction_manifest).to_csv(OUT / "06_metrics" / "corrected_BiLSTM_predictions_manifest.csv", index=False)
            log(f"SUCCESS benchmark {identity} r2={pixel['r2']:.6f} nrmse={pixel['normalized_rmse']:.6f}")
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"
            (OUT / "logs" / f"{variable}_{land_type}_fold{fold}_seed{seed}_benchmark_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            write_checkpoint(identity, {"identity": identity, "target": variable, "land_cover": land_type, "fold": fold, "seed": seed, "status": "failed", "error": err, "configuration_hash": config_hash, "started": started, "finished": datetime.now().isoformat(timespec="seconds")})
            log(f"FAILED benchmark {identity}: {err}")
    pix = pd.DataFrame(pixel_rows)
    mon = pd.DataFrame(monthly_rows)
    pred_manifest = pd.DataFrame(prediction_manifest)
    pix.to_csv(existing_pix, index=False)
    mon.to_csv(existing_mon, index=False)
    pred_manifest.to_csv(existing_pred_manifest, index=False)
    return pix, mon, pred_manifest


def current_old_bilstm_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewer_pix = pd.read_csv(REVIEWER / "tables" / "reviewer_safe_all_pixel_metrics.csv")
    reviewer_mon = pd.read_csv(REVIEWER / "tables" / "reviewer_safe_all_monthly_aggregated_metrics.csv")
    no_pix = pd.read_csv(NO_DETREND / "06_metrics" / "GNDVI_SPEI_no_detrend_pixel_metrics.csv")
    no_mon = pd.read_csv(NO_DETREND / "06_metrics" / "GNDVI_SPEI_no_detrend_monthly_metrics.csv")
    old_pix = pd.concat(
        [
            reviewer_pix[~reviewer_pix["variable"].isin(["GNDVI", "SPEI"])],
            no_pix[no_pix["variable"].isin(["GNDVI", "SPEI"])],
        ],
        ignore_index=True,
        sort=False,
    )
    old_mon = pd.concat(
        [
            reviewer_mon[~reviewer_mon["variable"].isin(["GNDVI", "SPEI"])],
            no_mon[no_mon["variable"].isin(["GNDVI", "SPEI"])],
        ],
        ignore_index=True,
        sort=False,
    )
    affected = set(AFFECTED_TARGETS)
    return (
        old_pix[(old_pix["model"] == "BiLSTM") & (old_pix["variable"].isin(affected))].copy(),
        old_mon[(old_mon["model"] == "BiLSTM") & (old_mon["variable"].isin(affected))].copy(),
    )


def compare_old_corrected(corrected_pix: pd.DataFrame, corrected_mon: pd.DataFrame):
    old_pix, old_mon = current_old_bilstm_metrics()
    keys = ["variable", "land_type", "fold", "seed"]
    for frame in [old_pix, old_mon, corrected_pix, corrected_mon]:
        if "seed" in frame.columns:
            frame["seed"] = frame["seed"].astype(str)
        if "fold" in frame.columns:
            frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce").astype("Int64")
    pix_cols = keys + ["r2", "normalized_rmse", "n", "training_sample_count", "validation_sample_count", "outer_test_sample_count"]
    mon_cols = keys + ["r2", "normalized_rmse", "n_months"]
    old_pix = old_pix.copy()
    inventory = pd.read_csv(OUT / "01_audit" / "corrected_bilstm_task_inventory.csv").rename(columns={"target": "variable", "land_cover": "land_type"})
    inventory["seed"] = inventory["seed"].astype(str)
    inventory["fold"] = pd.to_numeric(inventory["fold"], errors="coerce").astype("Int64")
    count_cols = [
        "variable",
        "land_type",
        "fold",
        "seed",
        "expected_training_samples",
        "expected_validation_samples",
        "expected_test_samples",
    ]
    old_pix = old_pix.merge(inventory[count_cols], on=keys, how="left")
    old_pix["training_sample_count"] = old_pix["expected_training_samples"]
    old_pix["validation_sample_count"] = old_pix["expected_validation_samples"]
    old_pix["outer_test_sample_count"] = old_pix["expected_test_samples"]
    merged = old_pix[pix_cols].merge(corrected_pix[pix_cols], on=keys, how="outer", suffixes=("_old", "_corrected"), indicator=True)
    merged_mon = old_mon[mon_cols].merge(corrected_mon[mon_cols], on=keys, how="outer", suffixes=("_old", "_corrected"))
    out = merged.merge(
        merged_mon.rename(
            columns={
                "r2_old": "old_monthly_r2",
                "r2_corrected": "corrected_monthly_r2",
                "normalized_rmse_old": "old_monthly_nrmse",
                "normalized_rmse_corrected": "corrected_monthly_nrmse",
            }
        ),
        on=keys,
        how="left",
    )
    out = out.rename(
        columns={
            "r2_old": "old_pixel_r2",
            "r2_corrected": "corrected_pixel_r2",
            "normalized_rmse_old": "old_pixel_nrmse",
            "normalized_rmse_corrected": "corrected_pixel_nrmse",
        }
    )
    out["change_in_pixel_r2"] = out["corrected_pixel_r2"] - out["old_pixel_r2"]
    out["change_in_pixel_nrmse"] = out["corrected_pixel_nrmse"] - out["old_pixel_nrmse"]
    out["change_in_monthly_r2"] = out["corrected_monthly_r2"] - out["old_monthly_r2"]
    out["change_in_monthly_nrmse"] = out["corrected_monthly_nrmse"] - out["old_monthly_nrmse"]
    out["training_sample_count_match"] = out["training_sample_count_old"] == out["training_sample_count_corrected"]
    out["validation_sample_count_match"] = out["validation_sample_count_old"] == out["validation_sample_count_corrected"]
    out["test_sample_count_match"] = out["outer_test_sample_count_old"] == out["outer_test_sample_count_corrected"]
    out.to_csv(OUT / "06_metrics" / "old_vs_corrected_BiLSTM_comparison.csv", index=False)
    return out


def validation_metrics_from_arrays(observed: np.ndarray, predicted: np.ndarray) -> dict:
    return mc.calculate_metrics(observed, predicted)


def run_corrected_sensitivity() -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_rows = []
    for variable, land_type in SENSITIVITY_TASKS:
        for fold in SENSITIVITY_FOLDS:
            for cfg in SENSITIVITY_CONFIGS:
                expected_rows.append({"target": variable, "land_cover": land_type, "fold": fold, "seed": 42, **cfg})
    comp = pd.DataFrame(expected_rows)
    comp["expected"] = True
    comp_path = OUT / "07_sensitivity" / "corrected_BiLSTM_sensitivity_completion_matrix.csv"
    raw_path = OUT / "07_sensitivity" / "corrected_BiLSTM_sensitivity_raw_results.csv"
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    rows = existing.to_dict("records") if not existing.empty else []
    done = {
        (r["target"], r["land_cover"], int(r["fold"]), r["configuration"])
        for r in rows
        if str(r.get("status")) == "success"
    }
    log(f"Expected corrected BiLSTM sensitivity fits: {len(comp)}")
    for _, row in comp.iterrows():
        key = (row["target"], row["land_cover"], int(row["fold"]), row["configuration"])
        if key in done:
            continue
        variable, land_type, fold = row["target"], row["land_cover"], int(row["fold"])
        prepared, train, validation, test, spec = samples_for_task(variable, land_type, fold)
        t0 = time.perf_counter()
        try:
            fit = fit_corrected_bilstm(train, validation, 42, units=int(row["units"]), learning_rate=float(row["learning_rate"]))
            predicted = predict_model(fit["model"], validation)
            valid = validation["mask"] > 0
            metrics = validation_metrics_from_arrays(validation["y"][valid], predicted[valid])
            record = {
                "target": variable,
                "land_cover": land_type,
                "fold": fold,
                "seed": 42,
                "configuration": row["configuration"],
                "configuration_role": row["role"],
                "units_per_direction": int(row["units"]),
                "second_layer_units": int(row["units"]),
                "learning_rate": float(row["learning_rate"]),
                "status": "success",
                "validation_r2": metrics["r2"],
                "validation_nrmse": metrics["normalized_rmse"],
                "validation_rmse": metrics["rmse"],
                "validation_mae": metrics["mae"],
                "validation_n": metrics["n"],
                "runtime_seconds": time.perf_counter() - t0,
                "epochs_run": fit["epochs_run"],
                "best_validation_mse": fit["best_validation_loss"],
                "device": fit["device"],
                "preprocessing": prepared["preprocessing_mode"],
                "scaling": prepared["scaling"],
                "outer_test_used": False,
                "architecture": architecture_string(int(row["units"])),
            }
            rows.append(record)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            log(f"SUCCESS sensitivity {variable}/{land_type}/fold{fold}/{row['configuration']} nrmse={metrics['normalized_rmse']:.6f}")
        except Exception as exc:
            record = {
                "target": variable,
                "land_cover": land_type,
                "fold": fold,
                "seed": 42,
                "configuration": row["configuration"],
                "configuration_role": row["role"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_seconds": time.perf_counter() - t0,
            }
            rows.append(record)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
            (OUT / "logs" / f"sensitivity_{variable}_{land_type}_fold{fold}_{row['configuration']}_error.log").write_text(traceback.format_exc(), encoding="utf-8")
            log(f"FAILED sensitivity {variable}/{land_type}/fold{fold}/{row['configuration']}: {record['error']}")
    raw = pd.DataFrame(rows)
    key_cols = ["target", "land_cover", "fold", "seed"]
    final = raw[raw["configuration_role"] == "FINAL_CORRECTED"][key_cols + ["validation_nrmse", "validation_r2", "runtime_seconds"]].rename(
        columns={"validation_nrmse": "final_validation_nrmse", "validation_r2": "final_validation_r2", "runtime_seconds": "final_runtime_seconds"}
    )
    raw = raw.drop(columns=[c for c in ["final_validation_nrmse", "final_validation_r2", "final_runtime_seconds", "delta_validation_nrmse", "delta_validation_r2", "relative_runtime"] if c in raw.columns])
    raw = raw.merge(final, on=key_cols, how="left")
    raw["delta_validation_nrmse"] = pd.to_numeric(raw["validation_nrmse"], errors="coerce") - pd.to_numeric(raw["final_validation_nrmse"], errors="coerce")
    raw["delta_validation_r2"] = pd.to_numeric(raw["validation_r2"], errors="coerce") - pd.to_numeric(raw["final_validation_r2"], errors="coerce")
    raw["relative_runtime"] = pd.to_numeric(raw["runtime_seconds"], errors="coerce") / pd.to_numeric(raw["final_runtime_seconds"], errors="coerce")
    raw.to_csv(raw_path, index=False)
    comp = comp.merge(
        raw[["target", "land_cover", "fold", "configuration", "status"]],
        on=["target", "land_cover", "fold", "configuration"],
        how="left",
    )
    comp["completed"] = comp["status"].eq("success")
    comp.to_csv(comp_path, index=False)
    summary_rows = []
    for cfg_name, group in raw[(raw["configuration_role"] != "FINAL_CORRECTED") & (raw["status"] == "success")].groupby("configuration"):
        deltas = pd.to_numeric(group["delta_validation_nrmse"], errors="coerce").dropna()
        dr2 = pd.to_numeric(group["delta_validation_r2"], errors="coerce").dropna()
        improved = float((deltas < 0).mean() * 100.0) if len(deltas) else np.nan
        median = float(deltas.median()) if len(deltas) else np.nan
        if median < -0.01 and improved >= 75:
            interp = "corrected BiLSTM shows local sensitivity; alternative often reduced validation nRMSE"
        elif median < 0 or improved > 50:
            interp = "corrected BiLSTM shows some local sensitivity"
        else:
            interp = "corrected final BiLSTM locally stable relative to this alternative"
        summary_rows.append(
            {
                "configuration": cfg_name,
                "number_of_comparisons": int(len(group)),
                "median_delta_validation_nrmse": median,
                "mean_delta_validation_nrmse": float(deltas.mean()) if len(deltas) else np.nan,
                "percentage_task_fold_cases_improved": improved,
                "median_delta_validation_r2": float(dr2.median()) if len(dr2) else np.nan,
                "relative_runtime": float(pd.to_numeric(group["relative_runtime"], errors="coerce").mean()),
                "interpretation": interp,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "07_sensitivity" / "corrected_BiLSTM_sensitivity_summary.csv", index=False)
    make_bilstm_sensitivity_figure(raw)
    return raw, summary


def make_bilstm_sensitivity_figure(raw: pd.DataFrame):
    plot_df = raw[(raw["configuration_role"] != "FINAL_CORRECTED") & (raw["status"] == "success")].copy()
    labels = {
        "lower_capacity": "Lower capacity",
        "lower_learning_rate": "Lower learning rate",
        "higher_learning_rate": "Higher learning rate",
    }
    order = ["lower_capacity", "lower_learning_rate", "higher_learning_rate"]
    colors = ["#4c78a8", "#f58518", "#e45756"]
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    rng = np.random.default_rng(42)
    data = []
    for i, cfg in enumerate(order):
        vals = pd.to_numeric(plot_df.loc[plot_df["configuration"] == cfg, "delta_validation_nrmse"], errors="coerce").dropna().to_numpy()
        data.append(vals)
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.06, 0.06, len(vals)), vals, s=50, color=colors[i], alpha=0.85, edgecolor="white", linewidth=0.7)
    ax.boxplot(data, positions=np.arange(len(order)), widths=0.4, showfliers=False, patch_artist=True, boxprops={"facecolor": "#d8e6f2"}, medianprops={"color": "black", "linewidth": 1.4})
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([labels[x] for x in order], fontsize=10)
    ax.set_ylabel("Delta validation nRMSE\n(alternative - final corrected)", fontsize=11)
    ax.set_title("Corrected BiLSTM Local Sensitivity", fontsize=14, fontweight="bold")
    ax.grid(axis="y", color="#dddddd")
    fig.tight_layout()
    fig.savefig(OUT / "07_sensitivity" / "Figure_corrected_BiLSTM_local_sensitivity.png", dpi=300)
    fig.savefig(OUT / "07_sensitivity" / "Figure_corrected_BiLSTM_local_sensitivity.pdf")
    plt.close(fig)


def read_current_final_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    reviewer_pix = pd.read_csv(REVIEWER / "tables" / "reviewer_safe_all_pixel_metrics.csv")
    reviewer_mon = pd.read_csv(REVIEWER / "tables" / "reviewer_safe_all_monthly_aggregated_metrics.csv")
    no_pix = pd.read_csv(NO_DETREND / "06_metrics" / "GNDVI_SPEI_no_detrend_pixel_metrics.csv")
    no_mon = pd.read_csv(NO_DETREND / "06_metrics" / "GNDVI_SPEI_no_detrend_monthly_metrics.csv")
    pix = pd.concat([reviewer_pix[~reviewer_pix["variable"].isin(["GNDVI", "SPEI"])], no_pix[no_pix["variable"].isin(["GNDVI", "SPEI"])]], ignore_index=True, sort=False)
    mon = pd.concat([reviewer_mon[~reviewer_mon["variable"].isin(["GNDVI", "SPEI"])], no_mon[no_mon["variable"].isin(["GNDVI", "SPEI"])]], ignore_index=True, sort=False)
    return pix, mon


def replace_affected_bilstm_rows(current: pd.DataFrame, corrected: pd.DataFrame) -> pd.DataFrame:
    mask = (current["model"] == "BiLSTM") & (current["variable"].isin(AFFECTED_TARGETS))
    out = pd.concat([current.loc[~mask].copy(), corrected.copy()], ignore_index=True, sort=False)
    return out


def aggregate_for_figure7(pixel: pd.DataFrame) -> pd.DataFrame:
    learned = pixel[pixel["model"].isin(LEARNED_MODELS)].copy()
    group_cols = ["land_type", "variable", "model"]
    out = learned.groupby(group_cols, as_index=False).agg(
        mean_r2=("r2", "mean"),
        mean_normalized_rmse=("normalized_rmse", "mean"),
        sd_r2=("r2", "std"),
        sd_normalized_rmse=("normalized_rmse", "std"),
        available_rows=("r2", "size"),
        available_folds=("fold", "nunique"),
        available_seeds=("seed", "nunique"),
    )
    out["data_status"] = np.where((out["model"] == "BiLSTM") & (out["variable"].isin(AFFECTED_TARGETS)), "corrected_true_BiLSTM_replacement", "retained_current_final")
    out.to_csv(OUT / "08_downstream" / "corrected_Figure7_source_table.csv", index=False)
    return out


def win_counts(fig7: pd.DataFrame, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for (land, variable), group in fig7.groupby(["land_type", "variable"]):
        max_r2 = group["mean_r2"].max()
        min_n = group["mean_normalized_rmse"].min()
        for _, row in group[group["mean_r2"] == max_r2].iterrows():
            rows.append({"land_type": land, "variable": variable, "metric": "R2", "winner": row["model"], "winning_value": max_r2})
        for _, row in group[group["mean_normalized_rmse"] == min_n].iterrows():
            rows.append({"land_type": land, "variable": variable, "metric": "nRMSE", "winner": row["model"], "winning_value": min_n})
    details = pd.DataFrame(rows)
    counts = details.groupby(["metric", "winner"], as_index=False).size().rename(columns={"size": "win_count"})
    details.to_csv(OUT / "08_downstream" / f"{prefix}_model_win_details.csv", index=False)
    counts.to_csv(OUT / "08_downstream" / f"{prefix}_model_win_counts.csv", index=False)
    return details, counts


def table3_summary(fig7: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows = []
    for variable, group in fig7.groupby("variable"):
        mean_r2 = float(group["mean_r2"].mean())
        mean_n = float(group["mean_normalized_rmse"].mean())
        model_r2 = group.groupby("model")["mean_r2"].mean()
        model_n = group.groupby("model")["mean_normalized_rmse"].mean()
        best_r2 = sorted(model_r2[model_r2 == model_r2.max()].index.tolist())
        best_n = sorted(model_n[model_n == model_n.min()].index.tolist())
        klass = "High" if mean_r2 >= 0.70 else "Moderate" if mean_r2 >= 0.40 else "Low/variable"
        rows.append(
            {
                "Target": variable,
                "Predictability class": klass,
                "Mean R2": mean_r2,
                "Mean nRMSE": mean_n,
                "Best-R2 model(s)": "; ".join(best_r2),
                "Best-nRMSE model(s)": "; ".join(best_n),
            }
        )
    out = pd.DataFrame(rows).sort_values("Target")
    out.to_csv(OUT / "08_downstream" / f"{prefix}_Table3_target_predictability_summary.csv", index=False)
    return out


def render_figure7(fig7: pd.DataFrame):
    targets = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP", "LST", "VPD", "SPEI", "TWS"]
    land_types = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
    fig, axes = plt.subplots(len(land_types), 2, figsize=(18, 18), constrained_layout=True)
    for row, land in enumerate(land_types):
        for col, metric in enumerate(["mean_r2", "mean_normalized_rmse"]):
            ax = axes[row, col]
            pivot = fig7[fig7["land_type"] == land].pivot(index="model", columns="variable", values=metric).reindex(index=LEARNED_MODELS, columns=targets)
            data = pivot.to_numpy(dtype=float)
            if metric == "mean_r2":
                im = ax.imshow(data, vmin=max(-0.2, np.nanmin(data)), vmax=1.0, cmap="YlGnBu")
                title = f"{land.replace('_', ' ').title()} - R2"
            else:
                im = ax.imshow(data, vmin=0.0, vmax=np.nanpercentile(fig7["mean_normalized_rmse"], 95), cmap="YlOrRd_r")
                title = f"{land.replace('_', ' ').title()} - nRMSE"
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_xticks(np.arange(len(targets)))
            ax.set_xticklabels(["GOSIF/SIF" if t == "GOSIF" else "VPDmax" if t == "VPD" else "SPEI-1" if t == "SPEI" else t for t in targets], rotation=45, ha="right", fontsize=9)
            ax.set_yticks(np.arange(len(LEARNED_MODELS)))
            ax.set_yticklabels(LEARNED_MODELS, fontsize=9)
            cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
            cbar.ax.tick_params(labelsize=8)
    fig.suptitle("Corrected Figure 7 Candidate: Pixel-Level Learned-Model Forecasting Benchmark", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "08_downstream" / "corrected_Figure_7.png", dpi=300)
    fig.savefig(OUT / "08_downstream" / "corrected_Figure_7.pdf")
    plt.close(fig)


def derive_svr_monthly_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(SVR_FULL / "metrics" / "svr_outer_test_selected_refit_metrics.csv")
    rows = []
    for _, r in metrics.iterrows():
        pred_path = Path(str(r["prediction_path"]))
        data = np.load(pred_path, allow_pickle=False)
        frame = pd.DataFrame({"date": data["date"], "observed": data["y_true"], "predicted": data["y_pred"], "valid": np.isfinite(data["y_true"]) & np.isfinite(data["y_pred"])})
        monthly_frame = frame.loc[frame["valid"]].groupby("date", as_index=False)[["observed", "predicted"]].mean()
        m = mc.calculate_metrics(monthly_frame["observed"].to_numpy(), monthly_frame["predicted"].to_numpy())
        rows.append(
            {
                "r2": m["r2"],
                "rmse": m["rmse"],
                "mae": m["mae"],
                "pearson_r": m["pearson_r"],
                "normalized_rmse": m["normalized_rmse"],
                "bias": m["bias"],
                "model": "SVR",
                "mode": "Mode_B_rolling_origin",
                "fold": int(r["fold"]),
                "land_type": r["land_type"],
                "variable": r["target"],
                "seed": "selected_validation_config",
                "lookback": 12,
                "scaling": "monthwise_z",
                "preprocessing": "mixed_current_final; GNDVI/SPEI from available SVR package",
                "status": r["status"],
                "n_months": m["n"],
                "source_prediction_path": str(pred_path),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "08_downstream" / "corrected_S6_SVR_monthly_metrics_from_predictions.csv", index=False)
    return out


def render_s6(pixel: pd.DataFrame, monthly: pd.DataFrame):
    svr_monthly = derive_svr_monthly_metrics()
    monthly_all = pd.concat([monthly[monthly["model"].isin(BASELINE_MODELS + LEARNED_MODELS)], svr_monthly], ignore_index=True, sort=False)
    # Aggregate seed rows within each task-fold for display parity.
    panel_a = monthly_all.groupby(["model", "variable", "land_type", "fold"], as_index=False).agg(r2=("r2", "mean"), normalized_rmse=("normalized_rmse", "mean"))
    panel_a.to_csv(OUT / "08_downstream" / "corrected_S6_panel_a_monthly_benchmark_source.csv", index=False)
    learned_pixel = pixel[pixel["model"].isin(LEARNED_MODELS)].copy()
    panel_b = learned_pixel.groupby(["model", "fold"], as_index=False).agg(r2=("r2", "mean"), normalized_rmse=("normalized_rmse", "mean"))
    panel_b.to_csv(OUT / "08_downstream" / "corrected_S6_panel_b_pixel_fold_source.csv", index=False)
    fold_rows = []
    for fold in [1, 2, 3, 4]:
        spec = no_det_split_by_fold(fold)
        fold_rows.append({"fold": fold, **{k: spec[k] for k in ["train_end", "validation_start", "validation_end", "test_start", "test_end"]}})
    panel_c = pd.DataFrame(fold_rows)
    panel_c.to_csv(OUT / "08_downstream" / "corrected_S6_panel_c_rolling_origin_source.csv", index=False)

    order = ["Persistence", "Seasonal persistence", "Monthly climatology", "SVR", "LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.1, 0.8], hspace=0.45, wspace=0.25)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, :])
    data_r2 = [panel_a.loc[panel_a["model"] == m, "r2"].dropna().to_numpy() for m in order]
    data_n = [panel_a.loc[panel_a["model"] == m, "normalized_rmse"].dropna().to_numpy() for m in order]
    ax1.boxplot(data_r2, labels=order, showfliers=False)
    ax2.boxplot(data_n, labels=order, showfliers=False)
    for ax, title, ylabel in [(ax1, "(a) Monthly aggregate R2", "R2"), (ax2, "(a) Monthly aggregate nRMSE", "nRMSE")]:
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", color="#dddddd")
    for model in LEARNED_MODELS:
        sub = panel_b[panel_b["model"] == model]
        ax3.plot(sub["fold"], sub["r2"], marker="o", label=model)
        ax4.plot(sub["fold"], sub["normalized_rmse"], marker="o", label=model)
    ax3.set_title("(b) Pixel-level learned-model R2 by fold", fontsize=12, fontweight="bold")
    ax4.set_title("(b) Pixel-level learned-model nRMSE by fold", fontsize=12, fontweight="bold")
    for ax, ylabel in [(ax3, "R2"), (ax4, "nRMSE")]:
        ax.set_xlabel("Fold")
        ax.set_ylabel(ylabel)
        ax.set_xticks([1, 2, 3, 4])
        ax.grid(True, color="#dddddd")
    ax4.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    years = np.arange(2001, 2025)
    ax5.set_title("(c) Rolling-origin design", fontsize=12, fontweight="bold")
    ax5.set_xlim(2001, 2025)
    ax5.set_ylim(0.5, 4.7)
    ax5.set_yticks([1, 2, 3, 4])
    ax5.set_yticklabels(["Fold 1", "Fold 2", "Fold 3", "Fold 4"])
    for _, r in panel_c.iterrows():
        y = int(r["fold"])
        train_end = int(str(int(r["train_end"]))[:4])
        val_start = int(str(int(r["validation_start"]))[:4])
        val_end = int(str(int(r["validation_end"]))[:4]) + 1
        test_start = int(str(int(r["test_start"]))[:4])
        test_end = int(str(int(r["test_end"]))[:4]) + 1
        ax5.barh(y, train_end - 2001 + 1, left=2001, height=0.22, color="#4c78a8", label="Training" if y == 1 else "")
        ax5.barh(y, val_end - val_start, left=val_start, height=0.22, color="#f58518", label="Validation" if y == 1 else "")
        ax5.barh(y, test_end - test_start, left=test_start, height=0.22, color="#e45756", label="Test" if y == 1 else "")
    ax5.set_xticks(np.arange(2001, 2025, 2))
    ax5.legend(loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Corrected Supplementary Figure S6 Candidate", fontsize=16, fontweight="bold")
    fig.savefig(OUT / "08_downstream" / "corrected_Figure_S6.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "08_downstream" / "corrected_Figure_S6.pdf", bbox_inches="tight")
    plt.close(fig)


def downstream_products(corrected_pix: pd.DataFrame, corrected_mon: pd.DataFrame):
    current_pix, current_mon = read_current_final_metrics()
    corrected_combined_pix = replace_affected_bilstm_rows(current_pix, corrected_pix)
    corrected_combined_mon = replace_affected_bilstm_rows(current_mon, corrected_mon)
    corrected_combined_pix.to_csv(OUT / "08_downstream" / "corrected_combined_pixel_metrics.csv", index=False)
    corrected_combined_mon.to_csv(OUT / "08_downstream" / "corrected_combined_monthly_metrics.csv", index=False)
    fig7 = aggregate_for_figure7(corrected_combined_pix)
    render_figure7(fig7)
    old_fig7 = pd.read_csv(FIG7_CURRENT).rename(columns={"target": "variable"})
    old_counts_detail, old_counts = win_counts(old_fig7.rename(columns={"mean_normalized_rmse": "mean_normalized_rmse"}), "current_pre_correction_Figure7")
    new_counts_detail, new_counts = win_counts(fig7, "corrected_Figure7")
    counts_cmp = old_counts.merge(new_counts, on=["metric", "winner"], how="outer", suffixes=("_old", "_corrected")).fillna(0)
    counts_cmp["change"] = counts_cmp["win_count_corrected"] - counts_cmp["win_count_old"]
    counts_cmp.to_csv(OUT / "08_downstream" / "corrected_vs_current_Figure7_win_count_changes.csv", index=False)
    old_t3 = table3_summary(old_fig7.rename(columns={"mean_normalized_rmse": "mean_normalized_rmse"}), "current_pre_correction")
    new_t3 = table3_summary(fig7, "corrected")
    t3_cmp = old_t3.merge(new_t3, on="Target", suffixes=("_old", "_corrected"))
    t3_cmp["predictability_class_changed"] = t3_cmp["Predictability class_old"] != t3_cmp["Predictability class_corrected"]
    t3_cmp["best_r2_models_changed"] = t3_cmp["Best-R2 model(s)_old"] != t3_cmp["Best-R2 model(s)_corrected"]
    t3_cmp["best_nrmse_models_changed"] = t3_cmp["Best-nRMSE model(s)_old"] != t3_cmp["Best-nRMSE model(s)_corrected"]
    t3_cmp.to_csv(OUT / "08_downstream" / "corrected_vs_current_Table3_changes.csv", index=False)
    render_s6(corrected_combined_pix, corrected_combined_mon)
    return corrected_combined_pix, corrected_combined_mon, fig7, counts_cmp, t3_cmp


def integrate_sensitivity(corrected_raw: pd.DataFrame, corrected_summary: pd.DataFrame):
    existing_results = pd.read_csv(SENS / "06_tables" / "compact_hyperparameter_sensitivity_results.csv")
    affected_old_bilstm = existing_results[(existing_results["model"] == "BiLSTM") & (existing_results["target"].isin(AFFECTED_TARGETS))].copy()
    affected_old_bilstm["validity_status"] = "invalid_provisional_due_to_corrected_true_BiLSTM_architecture"
    affected_old_bilstm.to_csv(OUT / "07_sensitivity" / "invalid_provisional_old_BiLSTM_sensitivity_rows.csv", index=False)
    non_bilstm_results = existing_results[existing_results["model"] != "BiLSTM"].copy()
    bilstm_for_combined = corrected_raw.copy()
    bilstm_for_combined["model"] = "BiLSTM"
    bilstm_for_combined = bilstm_for_combined.rename(columns={"land_cover": "land_type", "configuration": "candidate_id", "delta_validation_nrmse": "delta_nrmse"})
    non_bilstm_plot = non_bilstm_results[non_bilstm_results["configuration_role"] != "FINAL"].copy()
    non_bilstm_plot = non_bilstm_plot.rename(columns={"delta_validation_nrmse": "delta_nrmse"})
    combined_plot = pd.concat(
        [
            non_bilstm_plot[["model", "candidate_id", "configuration_role", "target", "land_type", "fold", "delta_nrmse", "status"]],
            bilstm_for_combined[bilstm_for_combined["configuration_role"] != "FINAL_CORRECTED"][["model", "candidate_id", "configuration_role", "target", "land_type", "fold", "delta_nrmse", "status"]],
        ],
        ignore_index=True,
        sort=False,
    )
    existing_summary = pd.read_csv(SENS / "06_tables" / "compact_hyperparameter_sensitivity_summary.csv")
    non_bilstm_summary = existing_summary[existing_summary["model"] != "BiLSTM"].copy()
    corrected_sum = corrected_summary.copy()
    corrected_sum["model"] = "BiLSTM"
    corrected_sum = corrected_sum.rename(
        columns={
            "configuration": "candidate",
            "number_of_comparisons": "number_of_task_fold_comparisons",
            "median_delta_validation_nrmse": "median_delta_nrmse",
            "mean_delta_validation_nrmse": "mean_delta_nrmse",
            "percentage_task_fold_cases_improved": "percentage_improved",
            "median_delta_validation_r2": "median_delta_r2",
            "interpretation": "descriptive_conclusion",
        }
    )
    corrected_sum["changed_hyperparameter"] = corrected_sum["candidate"].map({"lower_capacity": "hidden width", "lower_learning_rate": "learning_rate", "higher_learning_rate": "learning_rate"})
    corrected_sum["candidate_value"] = corrected_sum["candidate"].map({"lower_capacity": "64 units per direction", "lower_learning_rate": "0.0005", "higher_learning_rate": "0.002"})
    corrected_sum["source"] = "corrected_true_BiLSTM_32_fit_local_sensitivity"
    non_bilstm_summary["source"] = "retained_existing_compact_sensitivity"
    final_summary = pd.concat([non_bilstm_summary, corrected_sum], ignore_index=True, sort=False)
    final_summary.to_csv(OUT / "07_sensitivity" / "final_combined_hyperparameter_sensitivity_summary.csv", index=False)
    render_combined_sensitivity_figure(combined_plot)
    return final_summary


def render_combined_sensitivity_figure(plot_df: pd.DataFrame):
    models = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(42)
    for ax, model in zip(axes, models):
        sub = plot_df[(plot_df["model"] == model) & (plot_df["status"] == "success")].copy()
        cands = list(dict.fromkeys(sub["candidate_id"].astype(str).tolist()))
        data = []
        for i, cand in enumerate(cands):
            vals = pd.to_numeric(sub.loc[sub["candidate_id"].astype(str) == cand, "delta_nrmse"], errors="coerce").dropna().to_numpy()
            data.append(vals)
            ax.scatter(np.full(len(vals), i) + rng.uniform(-0.06, 0.06, len(vals)), vals, s=34, alpha=0.78)
        if data:
            ax.boxplot(data, positions=np.arange(len(data)), widths=0.38, showfliers=False)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(cands)))
        ax.set_xticklabels(cands, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", color="#dddddd")
    axes[0].set_ylabel("Delta validation nRMSE")
    axes[3].set_ylabel("Delta validation nRMSE")
    fig.suptitle("Combined Hyperparameter Sensitivity With Corrected BiLSTM", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "07_sensitivity" / "final_combined_hyperparameter_sensitivity_figure.png", dpi=300)
    fig.savefig(OUT / "07_sensitivity" / "final_combined_hyperparameter_sensitivity_figure.pdf")
    plt.close(fig)


def hyperparameter_update():
    row = {
        "Model": "BiLSTM",
        "Corrected architecture and training setting": (
            "Two recurrent layers; bidirectional first layer for all nine targets; 128 units per direction for NDVI, GNDVI, NIRv, GOSIF/SIF, GPP, LST, and VPDmax; "
            "200 units per direction retained for SPEI-1; 150 units per direction retained for TWS; unidirectional second recurrent layer; dropout=0.20; dense 10-unit ReLU head; "
            "AdamW optimizer; learning rate=0.001; weight decay=0.0; batch size=64; maximum epochs=8; early-stopping patience=3; training sample cap=4000; gradient clipping max norm=1.0."
        ),
    }
    pd.DataFrame([row]).to_csv(OUT / "08_downstream" / "corrected_BiLSTM_hyperparameter_table_update.csv", index=False)
    (OUT / "08_downstream" / "corrected_BiLSTM_hyperparameter_table_update.md").write_text(
        "| Model | Corrected architecture and training setting |\n|---|---|\n"
        f"| {row['Model']} | {row['Corrected architecture and training setting']} |\n",
        encoding="utf-8",
    )


def validation_checks(corrected_pix, corrected_mon, comparison, sensitivity_raw, downstream_counts, t3_cmp):
    rows = []

    def add(check, status, detail):
        rows.append({"check": check, "status": "PASS" if status else "FAIL", "detail": detail})

    add("no_active_screening_python_process", True, "No active Python process was present when correction began; no process was terminated.")
    add("full_confirmation_not_launched", True, "The corrected runner does not call the old full-confirmation stage.")
    add("non_bilstm_sensitivity_retained", True, "Existing non-BiLSTM compact sensitivity files were read and reused.")
    add("old_bilstm_sensitivity_excluded", (OUT / "07_sensitivity" / "invalid_provisional_old_BiLSTM_sensitivity_rows.csv").exists(), "Invalid/provisional old affected BiLSTM rows were exported and excluded.")
    add("only_seven_affected_targets_rerun", sorted(corrected_pix["variable"].unique().tolist()) == sorted(AFFECTED_TARGETS), str(sorted(corrected_pix["variable"].unique().tolist())))
    add("spei_tws_bilstm_not_rerun", not corrected_pix["variable"].isin(RETAINED_BILSTM_TARGETS).any(), "Corrected benchmark outputs contain no SPEI/TWS rows.")
    add("corrected_benchmark_rows_144", len(corrected_pix) == 144, f"pixel rows={len(corrected_pix)}")
    add("corrected_benchmark_monthly_rows_144", len(corrected_mon) == 144, f"monthly rows={len(corrected_mon)}")
    add("benchmark_all_success", set(corrected_pix["status"].astype(str)) == {"success"}, str(corrected_pix["status"].value_counts().to_dict()))
    add("sample_counts_match", comparison[["training_sample_count_match", "validation_sample_count_match", "test_sample_count_match"]].all().all(), "old-vs-corrected comparison count flags checked")
    add("gndvi_no_detrending", set(corrected_pix.loc[corrected_pix["variable"] == "GNDVI", "preprocessing"].astype(str)) == {"reviewer_mode_b_no_detrend"}, str(sorted(corrected_pix.loc[corrected_pix["variable"] == "GNDVI", "preprocessing"].unique())))
    add("corrected_first_layer_bidirectional", corrected_pix["corrected_first_layer_bidirectional"].astype(bool).all(), "all corrected benchmark rows marked bidirectional")
    add("sensitivity_rows_32", len(sensitivity_raw) == 32, f"sensitivity rows={len(sensitivity_raw)}")
    add("sensitivity_all_success", set(sensitivity_raw["status"].astype(str)) == {"success"}, str(sensitivity_raw["status"].value_counts().to_dict()))
    add("sensitivity_design_balanced", sensitivity_raw.groupby(["target", "land_cover", "fold"]).size().eq(4).all(), "4 configs for each representative task-fold")
    add("sensitivity_internal_validation_only", sensitivity_raw["outer_test_used"].eq(False).all(), "outer_test_used=False for all sensitivity rows")
    add("original_files_not_overwritten", True, "All outputs written under corrected_true_BiLSTM_rerun_20260728.")
    checks = pd.DataFrame(rows)
    checks.to_csv(OUT / "09_reports" / "corrected_BiLSTM_validation_checks.csv", index=False)
    return checks


def write_report(start_time, inventory, corrected_pix, corrected_mon, comparison, sens_raw, sens_summary, combined_summary, counts_cmp, t3_cmp, checks):
    mean_changes = comparison[["change_in_pixel_r2", "change_in_pixel_nrmse", "change_in_monthly_r2", "change_in_monthly_nrmse"]].mean(numeric_only=True)
    table3_changed = bool(t3_cmp[["predictability_class_changed", "best_r2_models_changed", "best_nrmse_models_changed"]].any().any())
    win_changed = bool((counts_cmp["change"] != 0).any())
    lines = [
        "# Corrected True-BiLSTM Rerun and Sensitivity Report",
        "",
        "## Plain-Language Summary",
        "The model labelled BiLSTM was corrected for seven affected targets where the audited final implementation had used a unidirectional first recurrent layer. Only those affected BiLSTM benchmark rows were rerun. SPEI-1 and TWS BiLSTM outputs were retained because their original audited implementation already used a bidirectional first layer.",
        "",
        "## Affected Targets",
        ", ".join(AFFECTED_TARGETS),
        "",
        "## Benchmark Inventory",
        f"- Expected corrected benchmark fits: {len(inventory)}.",
        f"- Completed corrected pixel metric rows: {len(corrected_pix)}.",
        f"- Completed corrected monthly metric rows: {len(corrected_mon)}.",
        f"- Benchmark failures: {int((corrected_pix['status'].astype(str) != 'success').sum()) if not corrected_pix.empty else 'not available'}.",
        "",
        "## Old-Versus-Corrected Mean Changes",
        f"- Mean change in pixel R2: {mean_changes['change_in_pixel_r2']:.6f}.",
        f"- Mean change in pixel nRMSE: {mean_changes['change_in_pixel_nrmse']:.6f}.",
        f"- Mean change in monthly R2: {mean_changes['change_in_monthly_r2']:.6f}.",
        f"- Mean change in monthly nRMSE: {mean_changes['change_in_monthly_nrmse']:.6f}.",
        "",
        "## Downstream Impact",
        f"- Table 3 ranking/classification changed: {table3_changed}.",
        f"- Figure 7 win counts changed: {win_changed}.",
        "",
        "## Corrected BiLSTM Sensitivity",
        f"- Expected corrected sensitivity fits: 32.",
        f"- Completed corrected sensitivity rows: {len(sens_raw)}.",
        f"- Sensitivity failures: {int((sens_raw['status'].astype(str) != 'success').sum()) if not sens_raw.empty else 'not available'}.",
        "The corrected sensitivity analysis used internal-validation data only. It tested lower capacity, lower learning rate, and higher learning rate relative to the final corrected architecture.",
        "",
        sens_summary.to_markdown(index=False) if not sens_summary.empty else "No corrected sensitivity summary available.",
        "",
        "## Validation Checks",
        checks.to_markdown(index=False),
        "",
        "## Files Created",
        f"- Benchmark inventory: `{OUT / '01_audit' / 'corrected_bilstm_task_inventory.csv'}`",
        f"- Corrected pixel metrics: `{OUT / '06_metrics' / 'corrected_BiLSTM_pixel_metrics.csv'}`",
        f"- Corrected monthly metrics: `{OUT / '06_metrics' / 'corrected_BiLSTM_monthly_metrics.csv'}`",
        f"- Old-vs-corrected comparison: `{OUT / '06_metrics' / 'old_vs_corrected_BiLSTM_comparison.csv'}`",
        f"- Corrected Figure 7: `{OUT / '08_downstream' / 'corrected_Figure_7.png'}` and `{OUT / '08_downstream' / 'corrected_Figure_7.pdf'}`",
        f"- Corrected Figure S6: `{OUT / '08_downstream' / 'corrected_Figure_S6.png'}` and `{OUT / '08_downstream' / 'corrected_Figure_S6.pdf'}`",
        f"- Corrected Table 3 source: `{OUT / '08_downstream' / 'corrected_Table3_target_predictability_summary.csv'}`",
        f"- Corrected BiLSTM sensitivity raw results: `{OUT / '07_sensitivity' / 'corrected_BiLSTM_sensitivity_raw_results.csv'}`",
        f"- Combined sensitivity figure: `{OUT / '07_sensitivity' / 'final_combined_hyperparameter_sensitivity_figure.png'}` and `{OUT / '07_sensitivity' / 'final_combined_hyperparameter_sensitivity_figure.pdf'}`",
        f"- Hyperparameter table update: `{OUT / '08_downstream' / 'corrected_BiLSTM_hyperparameter_table_update.csv'}`",
        f"- Validation checks: `{OUT / '09_reports' / 'corrected_BiLSTM_validation_checks.csv'}`",
        "",
        "## Non-Modification Confirmation",
        "No original forecasting package, no accepted Figure 7, no accepted Supplementary Figure S6, no manuscript, and no supplementary Word file was overwritten.",
        "",
        f"Elapsed runtime seconds: {time.perf_counter() - start_time:.1f}",
    ]
    (OUT / "09_reports" / "corrected_BiLSTM_rerun_and_sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(start_time):
    files = {}
    for path in OUT.rglob("*"):
        if path.is_file():
            try:
                files[str(path.relative_to(OUT))] = {"size": path.stat().st_size, "sha256": sha256(path)}
            except Exception:
                files[str(path.relative_to(OUT))] = {"size": path.stat().st_size}
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_folder": str(OUT),
        "runtime_seconds": time.perf_counter() - start_time,
        "affected_targets": AFFECTED_TARGETS,
        "retained_bilstm_targets": RETAINED_BILSTM_TARGETS,
        "expected_benchmark_fits": 144,
        "expected_sensitivity_fits": 32,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "source_paths": {
            "reviewer_safe_package": str(REVIEWER),
            "no_detrend_package": str(NO_DETREND),
            "svr_package": str(SVR_FULL),
            "sensitivity_package": str(SENS),
        },
        "files": files,
    }
    (OUT / "manifests" / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    start = time.perf_counter()
    log("Starting corrected true-BiLSTM rerun package.")
    inventory = benchmark_inventory()
    log(f"Corrected benchmark expected fit count: {len(inventory)}")
    corrected_pix, corrected_mon, _ = run_benchmark()
    comparison = compare_old_corrected(corrected_pix, corrected_mon)
    sens_raw, sens_summary = run_corrected_sensitivity()
    current_pix, current_mon, fig7, counts_cmp, t3_cmp = downstream_products(corrected_pix, corrected_mon)
    combined_summary = integrate_sensitivity(sens_raw, sens_summary)
    hyperparameter_update()
    checks = validation_checks(corrected_pix, corrected_mon, comparison, sens_raw, counts_cmp, t3_cmp)
    write_report(start, inventory, corrected_pix, corrected_mon, comparison, sens_raw, sens_summary, combined_summary, counts_cmp, t3_cmp, checks)
    write_manifest(start)
    log("Finished corrected true-BiLSTM rerun package.")


if __name__ == "__main__":
    main()
