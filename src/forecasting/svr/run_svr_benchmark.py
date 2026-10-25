import argparse
import concurrent.futures
import hashlib
import itertools
import json
import math
import subprocess
import sys
import threading
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\SVR_full_benchmark_supplementary_candidate_20260723_204013"
)
SVR_PARENT_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\SVR_reviewer_safe_candidate_20260722_185851"
)
PILOT_PACKAGE = (
    SVR_PARENT_PACKAGE
    / "pilot_continuation_20260723_121924"
    / "pilot_completion_20260723_165914"
)
AUTHORITATIVE_REVIEWER_SAFE_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_forecasting_raster_reviewer_safe_candidate"
)
FINAL_FIGURE7_PACKAGE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\05_final_figure7_reviewer_safe"
)

SCRIPT_DIR = PACKAGE_ROOT / "scripts"
LOGS_DIR = PACKAGE_ROOT / "logs"
PREDICTIONS_DIR = PACKAGE_ROOT / "predictions"
METRICS_DIR = PACKAGE_ROOT / "metrics"
TABLES_DIR = PACKAGE_ROOT / "tables"
FIGURES_DIR = PACKAGE_ROOT / "figures"
FIGURE_SOURCE_DIR = PACKAGE_ROOT / "figure_source_data"
REPORTS_DIR = PACKAGE_ROOT / "reports"
MANIFESTS_DIR = PACKAGE_ROOT / "manifests"
AUDITS_DIR = PACKAGE_ROOT / "audits"
ARRAYS_DIR = METRICS_DIR / "task_arrays"
WORKER_DIR = METRICS_DIR / "worker_jobs"

for directory in [
    LOGS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    FIGURE_SOURCE_DIR,
    REPORTS_DIR,
    MANIFESTS_DIR,
    AUDITS_DIR,
    ARRAYS_DIR,
    WORKER_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SCRIPT_DIR))

import candidate_config as cfg  # noqa: E402
from model_core import calculate_metrics  # noqa: E402
from raster_pipeline import build_supervised, prepare_task  # noqa: E402
from run_helpers import split_specs  # noqa: E402

from sklearn import __version__ as sklearn_version  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None


TIMEOUT_SECONDS = 900
SUCCESS_STATUSES = {"success", "convergence_warning_success"}
FAILURE_STATUSES = {
    "timeout_computationally_infeasible",
    "failed_exception",
    "convergence_warning_failed",
}
MODEL_ORDER = [
    "Persistence",
    "Seasonal persistence",
    "Monthly climatology",
    "XGBoost",
    "RBF\u2013\u03b5-SVR",
    "LSTM",
    "BiLSTM",
    "TCN",
    "N-BEATS",
    "N-HiTS",
]
MODEL_GROUP = {
    "Persistence": "Simple baseline",
    "Seasonal persistence": "Simple baseline",
    "Monthly climatology": "Simple baseline",
    "XGBoost": "Conventional ML benchmark",
    "RBF\u2013\u03b5-SVR": "Conventional ML benchmark",
    "LSTM": "Deep learning",
    "BiLSTM": "Deep learning",
    "TCN": "Deep learning",
    "N-BEATS": "Deep learning",
    "N-HiTS": "Deep learning",
}
TARGET_ORDER = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
LAND_ORDER = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]


def log(message: str) -> None:
    text = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(text, flush=True)
    with (LOGS_DIR / "full_svr_benchmark.log").open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_stem(*parts) -> str:
    text = "_".join(str(part) for part in parts)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        ("(", ""),
        (")", ""),
        (",", ""),
        ("\u03b5", "epsilon"),
        ("\u00d7", "x"),
        ("\u2013", "-"),
        ("\u2014", "-"),
    ]:
        text = text.replace(old, new)
    return text


def valid_xy(samples: dict) -> tuple[np.ndarray, np.ndarray]:
    valid = samples["mask"] > 0
    return samples["X"][valid].astype(np.float32), samples["y"][valid].astype(np.float32)


def valid_test_vectors(prepared: dict, test: dict) -> dict:
    valid = test["mask"] > 0
    pixel_id = test["pixel_id"][valid].astype(np.int32)
    return {
        "x_test": test["X"][valid].astype(np.float32),
        "y_test": test["y"][valid].astype(np.float32),
        "date_test": test["date"][valid].astype(np.int32),
        "pixel_id_test": pixel_id,
        "row_test": prepared["rows"][pixel_id].astype(np.int32),
        "col_test": prepared["cols"][pixel_id].astype(np.int32),
    }


def gamma_scale_from_training(x_train: np.ndarray) -> float:
    variance = float(np.var(x_train))
    if not np.isfinite(variance) or variance <= 0:
        return 1.0
    return 1.0 / (x_train.shape[1] * variance)


def grid_rows(gamma_scale: float) -> list[dict]:
    rows = []
    for index, (c_value, epsilon, multiplier) in enumerate(
        itertools.product(
            cfg.SVR_C_VALUES,
            cfg.SVR_EPSILON_VALUES,
            cfg.SVR_GAMMA_MULTIPLIERS,
        ),
        start=1,
    ):
        rows.append(
            {
                "candidate_index": int(index),
                "C": float(c_value),
                "epsilon": float(epsilon),
                "gamma_multiplier": float(multiplier),
                "gamma_scale": float(gamma_scale),
                "gamma": float(gamma_scale * multiplier),
            }
        )
    return rows


def split_by_fold(fold: int) -> dict:
    matches = [
        item
        for item in split_specs(include_mode_b=True, include_mode_a=False)
        if int(item["fold"]) == int(fold)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Mode B split for fold {fold}, found {len(matches)}")
    return matches[0]


def task_array_stem(target: str, land_type: str, fold: int) -> str:
    # Cached arrays depend on the preprocessing boundary; legacy keeps its historical cache names.
    boundary = split_by_fold(fold).get("preprocessing_boundary", "legacy_through_validation")
    suffix = "" if boundary == "legacy_through_validation" else f"_{boundary}"
    return safe_stem(target, land_type, f"fold{int(fold)}{suffix}")


def task_array_path(target: str, land_type: str, fold: int) -> Path:
    return ARRAYS_DIR / f"{task_array_stem(target, land_type, fold)}_arrays.npz"


def task_metadata_path(target: str, land_type: str, fold: int) -> Path:
    return ARRAYS_DIR / f"{task_array_stem(target, land_type, fold)}_metadata.json"


def prepare_and_cache_task_arrays(target: str, land_type: str, fold: int) -> dict:
    array_path = task_array_path(target, land_type, fold)
    metadata_path = task_metadata_path(target, land_type, fold)
    if array_path.exists() and metadata_path.exists():
        return read_json(metadata_path)

    spec = split_by_fold(fold)
    prepared = prepare_task(
        land_type,
        target,
        train_end=spec["preprocessing_fit_end"],
        preprocessing_mode=spec["preprocessing_mode"],
        scaling="monthwise_z",
    )
    earliest = int(prepared["dates"].min())
    train = build_supervised(prepared, earliest, spec["train_end"], cfg.MAIN_LOOKBACK)
    validation = build_supervised(
        prepared,
        spec["validation_start"],
        spec["validation_end"],
        cfg.MAIN_LOOKBACK,
    )
    # Final refit on training + validation. Under the fit_subset boundary the preprocessing is refitted through
    # validation_end for the refit and the outer test only; selection above used core-training preprocessing.
    refit_fit_end = int(spec.get("refit_preprocessing_fit_end", spec["preprocessing_fit_end"]))
    if refit_fit_end != int(spec["preprocessing_fit_end"]):
        prepared_refit = prepare_task(
            land_type,
            target,
            train_end=refit_fit_end,
            preprocessing_mode=spec["preprocessing_mode"],
            scaling="monthwise_z",
        )
    else:
        prepared_refit = prepared
    refit = build_supervised(prepared_refit, earliest, spec["validation_end"], cfg.MAIN_LOOKBACK)
    test = build_supervised(prepared_refit, spec["test_start"], spec["test_end"], cfg.MAIN_LOOKBACK)

    x_train, y_train = valid_xy(train)
    x_validation, y_validation = valid_xy(validation)
    x_refit, y_refit = valid_xy(refit)
    test_vectors = valid_test_vectors(prepared_refit, test)
    gamma_scale = gamma_scale_from_training(x_train)

    np.savez_compressed(
        array_path,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_refit=x_refit,
        y_refit=y_refit,
        **test_vectors,
    )

    metadata = {
        "target": target,
        "land_type": land_type,
        "fold": int(fold),
        "mode": spec["mode"],
        "core_training_start": earliest,
        "core_training_end": int(spec["train_end"]),
        "preprocessing_fit_end": int(spec["preprocessing_fit_end"]),
        "refit_preprocessing_fit_end": refit_fit_end,
        "validation_start": int(spec["validation_start"]),
        "validation_end": int(spec["validation_end"]),
        "outer_test_start": int(spec["test_start"]),
        "outer_test_end": int(spec["test_end"]),
        "preprocessing_mode": spec["preprocessing_mode"],
        "scaling": prepared["scaling"],
        "unit_scale_factor": float(prepared["unit_scale_factor"]),
        "unit_scale_reason": str(prepared["unit_scale_reason"]),
        "lookback": int(cfg.MAIN_LOOKBACK),
        "unique_pixels": int(len(prepared["rows"])),
        "n_features": int(x_train.shape[1]) if len(x_train) else int(cfg.MAIN_LOOKBACK),
        "core_training_samples": int(len(y_train)),
        "validation_samples": int(len(y_validation)),
        "refit_samples": int(len(y_refit)),
        "outer_test_samples": int(len(test_vectors["y_test"])),
        "gamma_scale": float(gamma_scale),
        "array_path": str(array_path),
    }
    write_json(metadata_path, metadata)
    return metadata


def expected_validation_candidates() -> pd.DataFrame:
    rows = []
    specs = split_specs(include_mode_b=True, include_mode_a=False)
    for target in TARGET_ORDER:
        for land_type in LAND_ORDER:
            for spec in specs:
                fold = int(spec["fold"])
                metadata = prepare_and_cache_task_arrays(target, land_type, fold)
                for grid in grid_rows(float(metadata["gamma_scale"])):
                    rows.append(
                        {
                            "target": target,
                            "land_type": land_type,
                            "fold": fold,
                            "candidate_index": int(grid["candidate_index"]),
                            "train_samples": int(metadata["core_training_samples"]),
                            "validation_samples": int(metadata["validation_samples"]),
                            "outer_test_samples_not_used": int(metadata["outer_test_samples"]),
                            "refit_samples_not_used_for_selection": int(metadata["refit_samples"]),
                            "unique_pixels": int(metadata["unique_pixels"]),
                            "n_features": int(metadata["n_features"]),
                            "core_training_start": int(metadata["core_training_start"]),
                            "core_training_end": int(metadata["core_training_end"]),
                            "preprocessing_fit_end": int(metadata["preprocessing_fit_end"]),
                            "validation_start": int(metadata["validation_start"]),
                            "validation_end": int(metadata["validation_end"]),
                            "outer_test_start_not_used": int(metadata["outer_test_start"]),
                            "outer_test_end_not_used": int(metadata["outer_test_end"]),
                            "array_path": metadata["array_path"],
                            **grid,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(TABLES_DIR / "expected_svr_validation_candidates.csv", index=False)
    return frame


def validation_key_columns() -> list[str]:
    return ["target", "land_type", "fold", "candidate_index"]


def final_key_columns() -> list[str]:
    return ["target", "land_type", "fold"]


def key_tuple(row, columns: list[str]) -> tuple:
    values = []
    for column in columns:
        value = row[column]
        if column in {"fold", "candidate_index"}:
            value = int(value)
        values.append(value)
    return tuple(values)


class ResourceMonitor:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.rows = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._process = psutil.Process() if psutil is not None else None

    def __enter__(self):
        if self._process is not None:
            self._process.cpu_percent(interval=None)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        if self._process is not None:
            self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._process.memory_info().rss / (1024 * 1024)
                cpu = self._process.cpu_percent(interval=None)
                self.rows.append(
                    {
                        "elapsed_seconds": time.perf_counter(),
                        "rss_mb": float(rss),
                        "cpu_percent": float(cpu),
                    }
                )
            except Exception:
                pass
            time.sleep(self.interval)

    def summary(self) -> dict:
        if not self.rows:
            return {
                "rss_mb_max": float("nan"),
                "rss_mb_mean": float("nan"),
                "cpu_percent_mean": float("nan"),
                "cpu_percent_max": float("nan"),
                "resource_sample_count": 0,
            }
        frame = pd.DataFrame(self.rows)
        return {
            "rss_mb_max": float(frame["rss_mb"].max()),
            "rss_mb_mean": float(frame["rss_mb"].mean()),
            "cpu_percent_mean": float(frame["cpu_percent"].mean()),
            "cpu_percent_max": float(frame["cpu_percent"].max()),
            "resource_sample_count": int(len(frame)),
        }


def fit_predict_worker(job_path: Path, out_path: Path) -> None:
    job = read_json(job_path)
    arrays = np.load(job["array_path"], allow_pickle=False)
    mode = job["mode"]
    if mode == "validation":
        x_fit = arrays["x_train"].astype(np.float32)
        y_fit = arrays["y_train"].astype(np.float32)
        x_eval = arrays["x_validation"].astype(np.float32)
        y_eval = arrays["y_validation"].astype(np.float32)
        metric_prefix = "validation"
    elif mode == "final_refit":
        x_fit = arrays["x_refit"].astype(np.float32)
        y_fit = arrays["y_refit"].astype(np.float32)
        x_eval = arrays["x_test"].astype(np.float32)
        y_eval = arrays["y_test"].astype(np.float32)
        metric_prefix = "outer_test"
    else:
        raise ValueError(f"Unknown worker mode: {mode}")

    params = job["params"]
    model = SVR(
        kernel="rbf",
        C=float(params["C"]),
        epsilon=float(params["epsilon"]),
        gamma=float(params["gamma"]),
        shrinking=True,
    )
    started = time.perf_counter()
    status = "success"
    error = ""
    warning_messages = []
    n_iter = float("nan")
    fit_seconds = float("nan")
    predict_seconds = float("nan")
    predicted = None
    with ResourceMonitor() as monitor:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit_start = time.perf_counter()
                model.fit(x_fit, y_fit)
                fit_seconds = time.perf_counter() - fit_start
                pred_start = time.perf_counter()
                predicted = model.predict(x_eval).astype(np.float32)
                predict_seconds = time.perf_counter() - pred_start
                warning_messages = [str(item.message) for item in caught]
            metrics = calculate_metrics(y_eval, predicted)
            n_support = int(len(model.support_))
            fit_status = int(getattr(model, "fit_status_", -999))
            raw_n_iter = getattr(model, "n_iter_", np.nan)
            try:
                n_iter = int(np.asarray(raw_n_iter).ravel()[0])
            except Exception:
                n_iter = float("nan")
            if warning_messages or fit_status != 0:
                status = "convergence_warning_success"
        except Exception as exc:
            status = "failed_exception"
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            metrics = {
                "r2": float("nan"),
                "rmse": float("nan"),
                "mae": float("nan"),
                "pearson_r": float("nan"),
                "normalized_rmse": float("nan"),
                "bias": float("nan"),
                "n": 0,
            }
            n_support = float("nan")
            fit_status = -999
    resource = monitor.summary()

    if mode == "final_refit" and predicted is not None and job.get("prediction_path"):
        np.savez_compressed(
            job["prediction_path"],
            y_true=y_eval.astype(np.float32),
            y_pred=predicted.astype(np.float32),
            date=arrays["date_test"].astype(np.int32),
            pixel_id=arrays["pixel_id_test"].astype(np.int32),
            row=arrays["row_test"].astype(np.int32),
            col=arrays["col_test"].astype(np.int32),
        )

    result = {
        **job["identity"],
        **params,
        "status": status,
        "error": error,
        "fit_status": fit_status,
        "n_iter": n_iter,
        "warning_messages": " | ".join(warning_messages),
        "fit_seconds": fit_seconds,
        f"{metric_prefix}_predict_seconds": predict_seconds,
        "total_worker_seconds": time.perf_counter() - started,
        f"{metric_prefix}_r2": metrics["r2"],
        f"{metric_prefix}_nrmse": metrics["normalized_rmse"],
        f"{metric_prefix}_rmse": metrics["rmse"],
        f"{metric_prefix}_mae": metrics["mae"],
        f"{metric_prefix}_pearson_r": metrics["pearson_r"],
        f"{metric_prefix}_bias": metrics["bias"],
        f"{metric_prefix}_n": metrics["n"],
        "n_support_vectors": n_support,
        "support_vector_fraction": float(n_support / len(y_fit))
        if len(y_fit) and np.isfinite(n_support)
        else float("nan"),
        "sklearn_version": sklearn_version,
        **resource,
    }
    write_json(out_path, result)


def timeout_result(identity: dict, params: dict, mode: str, wall: float, exc) -> dict:
    prefix = "validation" if mode == "validation" else "outer_test"
    return {
        **identity,
        **params,
        "status": "timeout_computationally_infeasible",
        "error": f"Exceeded per-fit timeout of {TIMEOUT_SECONDS} seconds. stdout={exc.stdout}; stderr={exc.stderr}",
        "fit_status": "",
        "n_iter": "",
        "warning_messages": "",
        "fit_seconds": float("nan"),
        f"{prefix}_predict_seconds": float("nan"),
        "total_worker_seconds": float("nan"),
        f"{prefix}_r2": float("nan"),
        f"{prefix}_nrmse": float("nan"),
        f"{prefix}_rmse": float("nan"),
        f"{prefix}_mae": float("nan"),
        f"{prefix}_pearson_r": float("nan"),
        f"{prefix}_bias": float("nan"),
        f"{prefix}_n": 0,
        "n_support_vectors": float("nan"),
        "support_vector_fraction": float("nan"),
        "sklearn_version": sklearn_version,
        "rss_mb_max": float("nan"),
        "rss_mb_mean": float("nan"),
        "cpu_percent_mean": float("nan"),
        "cpu_percent_max": float("nan"),
        "resource_sample_count": 0,
        "orchestrator_wall_seconds": wall,
        "worker_returncode": "",
    }


def subprocess_result(row: pd.Series, mode: str) -> dict:
    if mode == "validation":
        identity_columns = [
            "target",
            "land_type",
            "fold",
            "candidate_index",
            "train_samples",
            "validation_samples",
            "outer_test_samples_not_used",
            "refit_samples_not_used_for_selection",
            "unique_pixels",
            "n_features",
            "core_training_start",
            "core_training_end",
            "preprocessing_fit_end",
            "validation_start",
            "validation_end",
            "outer_test_start_not_used",
            "outer_test_end_not_used",
        ]
        stem = safe_stem(row["target"], row["land_type"], f"fold{int(row['fold'])}", f"candidate{int(row['candidate_index']):02d}")
        prediction_path = ""
    else:
        identity_columns = [
            "target",
            "land_type",
            "fold",
            "selected_candidate_index",
            "refit_samples",
            "outer_test_samples",
            "unique_pixels",
            "n_features",
            "core_training_start",
            "core_training_end",
            "preprocessing_fit_end",
            "validation_start",
            "validation_end",
            "outer_test_start",
            "outer_test_end",
        ]
        stem = safe_stem(row["target"], row["land_type"], f"fold{int(row['fold'])}", "final_refit")
        prediction_path = str(PREDICTIONS_DIR / f"{stem}_predictions.npz")

    identity = {column: json_safe(row[column]) for column in identity_columns}
    for col in ["fold", "candidate_index", "selected_candidate_index"]:
        if col in identity and identity[col] is not None:
            identity[col] = int(identity[col])
    params = {
        "C": float(row["C"] if mode == "validation" else row["selected_C"]),
        "epsilon": float(row["epsilon"] if mode == "validation" else row["selected_epsilon"]),
        "gamma_multiplier": float(row["gamma_multiplier"] if mode == "validation" else row["selected_gamma_multiplier"]),
        "gamma_scale": float(row["gamma_scale"] if mode == "validation" else row["selected_gamma_scale"]),
        "gamma": float(row["gamma"] if mode == "validation" else row["selected_gamma"]),
    }
    job_path = WORKER_DIR / f"{stem}_{mode}_job.json"
    out_path = WORKER_DIR / f"{stem}_{mode}_result.json"
    write_json(
        job_path,
        {
            "mode": mode,
            "array_path": str(row["array_path"]),
            "prediction_path": prediction_path,
            "identity": identity,
            "params": params,
        },
    )
    started = time.perf_counter()
    try:
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", str(job_path), str(out_path)],
            cwd=str(SCRIPT_DIR),
            text=True,
            capture_output=True,
            timeout=TIMEOUT_SECONDS,
        )
        wall = time.perf_counter() - started
        if out_path.exists():
            result = read_json(out_path)
        else:
            result = {
                **identity,
                **params,
                "status": "failed_exception",
                "error": (
                    f"Worker returned {process.returncode} without JSON. "
                    f"stdout={process.stdout}; stderr={process.stderr}"
                ),
            }
        result["orchestrator_wall_seconds"] = wall
        result["worker_returncode"] = process.returncode
        if mode == "final_refit":
            result["prediction_path"] = prediction_path if Path(prediction_path).exists() else ""
    except subprocess.TimeoutExpired as exc:
        wall = time.perf_counter() - started
        result = timeout_result(identity, params, mode, wall, exc)
        if mode == "final_refit":
            result["prediction_path"] = ""
    return result


def append_and_save(results: list[dict], path: Path, sort_columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(results)
    if not frame.empty:
        frame = frame.sort_values(sort_columns).reset_index(drop=True)
    frame.to_csv(path, index=False)
    return frame


def run_validation_candidates(expected: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    out_path = METRICS_DIR / "svr_validation_candidate_results.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    completed_keys = set()
    if not existing.empty:
        completed_keys = {
            key_tuple(row, validation_key_columns())
            for _, row in existing.iterrows()
            if str(row.get("status", "")) in SUCCESS_STATUSES | FAILURE_STATUSES
        }
    pending_rows = [
        row
        for _, row in expected.iterrows()
        if key_tuple(row, validation_key_columns()) not in completed_keys
    ]
    results = existing.to_dict("records") if not existing.empty else []
    log(f"Validation candidates expected={len(expected)} completed={len(completed_keys)} pending={len(pending_rows)}")
    if not pending_rows:
        return append_and_save(results, out_path, ["target", "land_type", "fold", "candidate_index"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(subprocess_result, row, "validation"): key_tuple(row, validation_key_columns())
            for row in pending_rows
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "target": key[0],
                    "land_type": key[1],
                    "fold": key[2],
                    "candidate_index": key[3],
                    "status": "failed_exception",
                    "error": f"orchestrator exception: {type(exc).__name__}: {exc}",
                }
            results.append(result)
            frame = append_and_save(results, out_path, ["target", "land_type", "fold", "candidate_index"])
            status_counts = frame["status"].value_counts().to_dict()
            log(f"Candidate {key} finished as {result.get('status')}; progress={len(frame)}/{len(expected)}; statuses={status_counts}")
    return append_and_save(results, out_path, ["target", "land_type", "fold", "candidate_index"])


def grid_coverage(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expected_levels = {
        "C": {float(v) for v in cfg.SVR_C_VALUES},
        "epsilon": {float(v) for v in cfg.SVR_EPSILON_VALUES},
        "gamma_multiplier": {float(v) for v in cfg.SVR_GAMMA_MULTIPLIERS},
    }
    for (target, land_type, fold), group in results.groupby(["target", "land_type", "fold"]):
        success = group[group["status"].isin(SUCCESS_STATUSES)].copy()
        timeout_count = int((group["status"] == "timeout_computationally_infeasible").sum())
        failure_count = int(group["status"].isin(["failed_exception", "convergence_warning_failed"]).sum())
        level_status = {}
        all_levels_present = True
        for column, levels in expected_levels.items():
            observed = {float(v) for v in pd.to_numeric(success[column], errors="coerce").dropna().unique()}
            missing = sorted(levels - observed)
            level_status[f"successful_{column}_levels"] = ";".join(str(v) for v in sorted(observed))
            level_status[f"missing_{column}_levels"] = ";".join(str(v) for v in missing)
            if missing:
                all_levels_present = False
        if len(success) >= 18 and all_levels_present:
            status = "ADEQUATE_GRID_COVERAGE"
        elif len(success) >= 9 and all_levels_present:
            status = "PARTIAL_BUT_USABLE_GRID_COVERAGE"
        else:
            status = "INADEQUATE_GRID_COVERAGE"
        rows.append(
            {
                "target": target,
                "land_type": land_type,
                "fold": int(fold),
                "total_candidates": int(len(group)),
                "successful_candidates": int(len(success)),
                "timeout_candidates": timeout_count,
                "failed_candidates": failure_count,
                **level_status,
                "grid_coverage_status": status,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["target", "land_type", "fold"])
    frame.to_csv(TABLES_DIR / "svr_full_grid_coverage.csv", index=False)
    return frame


def select_validation_configs(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, land_type, fold), group in results.groupby(["target", "land_type", "fold"]):
        success = group[group["status"].isin(SUCCESS_STATUSES)].dropna(subset=["validation_nrmse"]).copy()
        if success.empty:
            rows.append(
                {
                    "target": target,
                    "land_type": land_type,
                    "fold": int(fold),
                    "selection_status": "no_successful_validation_candidate",
                }
            )
            continue
        best_nrmse = float(success["validation_nrmse"].min())
        eligible = success[success["validation_nrmse"] <= best_nrmse * (1 + cfg.SVR_VALIDATION_NRMSE_RELATIVE_TOLERANCE)].copy()
        eligible["simplicity_C"] = eligible["C"]
        eligible["simplicity_gamma"] = eligible["gamma_multiplier"]
        eligible["simplicity_epsilon"] = -eligible["epsilon"]
        selected = eligible.sort_values(
            [
                "simplicity_C",
                "simplicity_gamma",
                "simplicity_epsilon",
                "validation_r2",
                "validation_nrmse",
            ],
            ascending=[True, True, True, False, True],
        ).iloc[0]
        metadata = read_json(task_metadata_path(target, land_type, int(fold)))
        rows.append(
            {
                "target": target,
                "land_type": land_type,
                "fold": int(fold),
                "selection_status": "selected_from_successful_validation_candidates",
                "total_candidates": int(len(group)),
                "successful_candidates": int(len(success)),
                "timeout_candidates": int((group["status"] == "timeout_computationally_infeasible").sum()),
                "failed_candidates": int(group["status"].isin(["failed_exception", "convergence_warning_failed"]).sum()),
                "best_validation_nrmse": best_nrmse,
                "selected_validation_nrmse": float(selected["validation_nrmse"]),
                "selected_validation_r2": float(selected["validation_r2"]),
                "selected_candidate_index": int(selected["candidate_index"]),
                "selected_C": float(selected["C"]),
                "selected_epsilon": float(selected["epsilon"]),
                "selected_gamma_multiplier": float(selected["gamma_multiplier"]),
                "selected_gamma_scale": float(selected["gamma_scale"]),
                "selected_gamma": float(selected["gamma"]),
                "n_configs_within_1pct_nrmse": int(len(eligible)),
                "selected_n_support_vectors": int(selected["n_support_vectors"]),
                "selected_support_vector_fraction": float(selected["support_vector_fraction"]),
                "selected_validation_fit_seconds": float(selected["fit_seconds"]),
                "array_path": metadata["array_path"],
                "refit_samples": int(metadata["refit_samples"]),
                "outer_test_samples": int(metadata["outer_test_samples"]),
                "unique_pixels": int(metadata["unique_pixels"]),
                "n_features": int(metadata["n_features"]),
                "core_training_start": int(metadata["core_training_start"]),
                "core_training_end": int(metadata["core_training_end"]),
                "preprocessing_fit_end": int(metadata["preprocessing_fit_end"]),
                "validation_start": int(metadata["validation_start"]),
                "validation_end": int(metadata["validation_end"]),
                "outer_test_start": int(metadata["outer_test_start"]),
                "outer_test_end": int(metadata["outer_test_end"]),
                "tie_breaking_rule": (
                    "Filter to configurations within 1% of best validation nRMSE, "
                    "then prefer lower C, lower gamma multiplier, larger epsilon, then higher validation R2."
                ),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["target", "land_type", "fold"])
    frame.to_csv(TABLES_DIR / "svr_full_selected_configurations_validation_only.csv", index=False)
    return frame


def run_final_refits(selected: pd.DataFrame, max_workers: int) -> pd.DataFrame:
    out_path = METRICS_DIR / "svr_outer_test_selected_refit_metrics.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    completed_keys = set()
    if not existing.empty:
        completed_keys = {
            key_tuple(row, final_key_columns())
            for _, row in existing.iterrows()
            if str(row.get("status", "")) in SUCCESS_STATUSES | FAILURE_STATUSES
        }
    usable = selected[selected["selection_status"] == "selected_from_successful_validation_candidates"].copy()
    pending_rows = [
        row
        for _, row in usable.iterrows()
        if key_tuple(row, final_key_columns()) not in completed_keys
    ]
    results = existing.to_dict("records") if not existing.empty else []
    log(f"Final selected SVR refits expected={len(usable)} completed={len(completed_keys)} pending={len(pending_rows)}")
    if not pending_rows:
        return append_and_save(results, out_path, ["target", "land_type", "fold"])

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(subprocess_result, row, "final_refit"): key_tuple(row, final_key_columns())
            for row in pending_rows
        }
        for future in concurrent.futures.as_completed(future_to_key):
            key = future_to_key[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "target": key[0],
                    "land_type": key[1],
                    "fold": key[2],
                    "status": "failed_exception",
                    "error": f"orchestrator exception: {type(exc).__name__}: {exc}",
                }
            results.append(result)
            frame = append_and_save(results, out_path, ["target", "land_type", "fold"])
            status_counts = frame["status"].value_counts().to_dict()
            log(f"Final refit {key} finished as {result.get('status')}; progress={len(frame)}/{len(usable)}; statuses={status_counts}")
    return append_and_save(results, out_path, ["target", "land_type", "fold"])


def verify_pilot_and_sources() -> pd.DataFrame:
    rows = []
    pilot_status_path = PILOT_PACKAGE / "tables" / "pilot_complete_status.json"
    pilot_results_path = PILOT_PACKAGE / "tables" / "pilot_complete_candidate_results.csv"
    pilot_selected_path = PILOT_PACKAGE / "tables" / "pilot_complete_selected_configurations.csv"
    for label, path in [
        ("completed_pilot_status_json", pilot_status_path),
        ("completed_pilot_candidate_results", pilot_results_path),
        ("completed_pilot_selected_configurations", pilot_selected_path),
    ]:
        rows.append({"check": label, "path": str(path), "status": "pass" if path.exists() else "missing"})
    if pilot_status_path.exists():
        status = read_json(pilot_status_path)
        rows.extend(
            [
                {
                    "check": "pilot_complete_candidate_rows",
                    "path": str(pilot_status_path),
                    "status": "pass" if int(status.get("observed_candidate_rows", -1)) == 81 else "review",
                    "value": status.get("observed_candidate_rows"),
                },
                {
                    "check": "pilot_outer_test_not_used",
                    "path": str(pilot_status_path),
                    "status": "pass" if status.get("outer_test_used") is False else "fail",
                    "value": status.get("outer_test_used"),
                },
                {
                    "check": "pilot_complete_benchmark_not_started",
                    "path": str(pilot_status_path),
                    "status": "pass" if status.get("complete_benchmark_not_started") is True else "fail",
                    "value": status.get("complete_benchmark_not_started"),
                },
            ]
        )
    if pilot_results_path.exists():
        pilot_results = pd.read_csv(pilot_results_path)
        rows.append(
            {
                "check": "pilot_timeout_count",
                "path": str(pilot_results_path),
                "status": "pass" if int((pilot_results["status"] == "timeout_computationally_infeasible").sum()) == 2 else "review",
                "value": int((pilot_results["status"] == "timeout_computationally_infeasible").sum()),
            }
        )

    source_manifest = SVR_PARENT_PACKAGE / "manifests" / "source_file_hashes.csv"
    if source_manifest.exists():
        manifest = pd.read_csv(source_manifest)
        for _, row in manifest.iterrows():
            path = Path(row["authoritative_path"])
            current = file_sha256(path) if path.exists() else ""
            rows.append(
                {
                    "check": f"hash_{row['file_name']}",
                    "path": str(path),
                    "status": "pass" if path.exists() and current == row["authoritative_sha256"] else "fail",
                    "value": current,
                    "expected": row["authoritative_sha256"],
                }
            )
    else:
        rows.append({"check": "source_hash_manifest", "path": str(source_manifest), "status": "missing"})

    source_tables = [
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_all_pixel_metrics.csv",
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_model_completion_status.csv",
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_metric_definitions.csv",
        FINAL_FIGURE7_PACKAGE / "tables" / "figure7_landtype_plot_values.csv",
    ]
    for path in source_tables:
        rows.append({"check": f"source_table_{path.name}", "path": str(path), "status": "pass" if path.exists() else "missing"})

    frame = pd.DataFrame(rows)
    frame.to_csv(AUDITS_DIR / "pre_run_framework_and_source_verification.csv", index=False)
    failure = frame[frame["status"].isin(["fail", "missing"])]
    if not failure.empty:
        raise RuntimeError("Pre-run verification failed. See pre_run_framework_and_source_verification.csv")
    return frame


def source_table_inventory() -> pd.DataFrame:
    rows = []
    for path in [
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_all_pixel_metrics.csv",
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_baseline_pixel_metrics.csv",
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_best_model_by_task.csv",
        AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_model_completion_status.csv",
        FINAL_FIGURE7_PACKAGE / "tables" / "figure7_landtype_plot_values.csv",
        FINAL_FIGURE7_PACKAGE / "tables" / "figure7_best_count_summary.csv",
    ]:
        if not path.exists():
            rows.append({"path": str(path), "exists": False})
            continue
        first = pd.read_csv(path, nrows=1)
        count = sum(1 for _ in path.open("r", encoding="utf-8", errors="replace")) - 1
        rows.append(
            {
                "path": str(path),
                "exists": True,
                "rows": int(count),
                "columns": ",".join(first.columns),
                "sha256": file_sha256(path),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(MANIFESTS_DIR / "authoritative_result_source_inventory.csv", index=False)
    return frame


def existing_model_task_level_metrics() -> pd.DataFrame:
    path = AUTHORITATIVE_REVIEWER_SAFE_PACKAGE / "tables" / "reviewer_safe_all_pixel_metrics.csv"
    raw = pd.read_csv(path)
    raw = raw[raw["model"].isin([m for m in MODEL_ORDER if m != "RBF\u2013\u03b5-SVR"])].copy()
    raw["target"] = raw["variable"]
    raw["outer_test_r2"] = pd.to_numeric(raw["r2"], errors="coerce")
    raw["outer_test_nrmse"] = pd.to_numeric(raw["normalized_rmse"], errors="coerce")
    raw["outer_test_rmse"] = pd.to_numeric(raw["rmse"], errors="coerce")
    raw["outer_test_mae"] = pd.to_numeric(raw["mae"], errors="coerce")
    raw["outer_test_n"] = pd.to_numeric(raw["n"], errors="coerce")
    grouped = (
        raw.groupby(["model", "target", "land_type", "fold"], as_index=False)
        .agg(
            outer_test_r2=("outer_test_r2", "mean"),
            outer_test_nrmse=("outer_test_nrmse", "mean"),
            outer_test_rmse=("outer_test_rmse", "mean"),
            outer_test_mae=("outer_test_mae", "mean"),
            outer_test_n=("outer_test_n", "mean"),
            source_seed_rows=("seed", "count"),
            seed_count=("seed", pd.Series.nunique),
            source_statuses=("status", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    grouped["model_group"] = grouped["model"].map(MODEL_GROUP)
    grouped["metric_level"] = "pixel"
    grouped["source_table"] = str(path)
    grouped["selected_C"] = np.nan
    grouped["selected_epsilon"] = np.nan
    grouped["selected_gamma_multiplier"] = np.nan
    grouped["selected_gamma"] = np.nan
    grouped["support_vector_fraction"] = np.nan
    grouped["runtime_seconds"] = np.nan
    grouped["status"] = grouped["source_statuses"]
    return grouped


def svr_task_level_metrics(final_results: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    merged = final_results.merge(
        selected[
            [
                "target",
                "land_type",
                "fold",
                "selected_candidate_index",
                "selected_validation_r2",
                "selected_validation_nrmse",
                "selected_C",
                "selected_epsilon",
                "selected_gamma_multiplier",
                "selected_gamma",
                "selected_gamma_scale",
                "n_configs_within_1pct_nrmse",
            ]
        ],
        on=["target", "land_type", "fold"],
        how="left",
        suffixes=("", "_selected"),
    )
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            {
                "model": "RBF\u2013\u03b5-SVR",
                "target": row["target"],
                "land_type": row["land_type"],
                "fold": int(row["fold"]),
                "outer_test_r2": row.get("outer_test_r2", np.nan),
                "outer_test_nrmse": row.get("outer_test_nrmse", np.nan),
                "outer_test_rmse": row.get("outer_test_rmse", np.nan),
                "outer_test_mae": row.get("outer_test_mae", np.nan),
                "outer_test_n": row.get("outer_test_n", np.nan),
                "source_seed_rows": 1,
                "seed_count": 1,
                "source_statuses": row.get("status", ""),
                "model_group": MODEL_GROUP["RBF\u2013\u03b5-SVR"],
                "metric_level": "pixel",
                "source_table": str(METRICS_DIR / "svr_outer_test_selected_refit_metrics.csv"),
                "selected_C": row.get("selected_C", np.nan),
                "selected_epsilon": row.get("selected_epsilon", np.nan),
                "selected_gamma_multiplier": row.get("selected_gamma_multiplier", np.nan),
                "selected_gamma": row.get("selected_gamma", np.nan),
                "selected_gamma_scale": row.get("selected_gamma_scale", np.nan),
                "selected_validation_r2": row.get("selected_validation_r2", np.nan),
                "selected_validation_nrmse": row.get("selected_validation_nrmse", np.nan),
                "selected_candidate_index": row.get("selected_candidate_index", np.nan),
                "n_configs_within_1pct_nrmse": row.get("n_configs_within_1pct_nrmse", np.nan),
                "support_vector_fraction": row.get("support_vector_fraction", np.nan),
                "runtime_seconds": row.get("fit_seconds", np.nan),
                "status": row.get("status", ""),
                "prediction_path": row.get("prediction_path", ""),
            }
        )
    return pd.DataFrame(rows)


def build_combined_tables(final_results: pd.DataFrame, selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    existing = existing_model_task_level_metrics()
    svr = svr_task_level_metrics(final_results, selected)
    combined = pd.concat([existing, svr], ignore_index=True, sort=False)
    combined["model"] = pd.Categorical(combined["model"], categories=MODEL_ORDER, ordered=True)
    combined["target"] = pd.Categorical(combined["target"], categories=TARGET_ORDER, ordered=True)
    combined["land_type"] = pd.Categorical(combined["land_type"], categories=LAND_ORDER, ordered=True)
    combined = combined.sort_values(["model", "target", "land_type", "fold"]).reset_index(drop=True)
    combined["model"] = combined["model"].astype(str)
    combined["target"] = combined["target"].astype(str)
    combined["land_type"] = combined["land_type"].astype(str)
    combined.to_csv(TABLES_DIR / "candidate_supplementary_task_level_outer_test_results.csv", index=False)

    tie_tolerance = 1e-12
    win_rows = []
    for (target, land_type, fold), group in combined.groupby(["target", "land_type", "fold"], observed=False):
        finite_r2 = group[np.isfinite(pd.to_numeric(group["outer_test_r2"], errors="coerce"))].copy()
        finite_nrmse = group[np.isfinite(pd.to_numeric(group["outer_test_nrmse"], errors="coerce"))].copy()
        if not finite_r2.empty:
            best_r2 = float(finite_r2["outer_test_r2"].max())
            winners = finite_r2[np.abs(finite_r2["outer_test_r2"] - best_r2) <= tie_tolerance]
            for _, row in winners.iterrows():
                win_rows.append({"target": target, "land_type": land_type, "fold": fold, "metric": "R2", "model": row["model"], "best_value": best_r2})
        if not finite_nrmse.empty:
            best_nrmse = float(finite_nrmse["outer_test_nrmse"].min())
            winners = finite_nrmse[np.abs(finite_nrmse["outer_test_nrmse"] - best_nrmse) <= tie_tolerance]
            for _, row in winners.iterrows():
                win_rows.append({"target": target, "land_type": land_type, "fold": fold, "metric": "nRMSE", "model": row["model"], "best_value": best_nrmse})
    wins = pd.DataFrame(win_rows)
    wins.to_csv(TABLES_DIR / "candidate_supplementary_task_win_details.csv", index=False)

    win_counts = (
        wins.groupby(["model", "metric"], as_index=False)
        .size()
        .pivot(index="model", columns="metric", values="size")
        .fillna(0)
        .reset_index()
        .rename(columns={"R2": "task_win_count_R2", "nRMSE": "task_win_count_nRMSE"})
    )
    summary = (
        combined.groupby(["model", "model_group"], as_index=False, observed=False)
        .agg(
            mean_R2=("outer_test_r2", "mean"),
            median_R2=("outer_test_r2", "median"),
            mean_nRMSE=("outer_test_nrmse", "mean"),
            median_nRMSE=("outer_test_nrmse", "median"),
            evaluated_task_fold_combinations=("outer_test_r2", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
            status_values=("status", lambda x: ";".join(sorted(set(map(str, x))))),
        )
    )
    summary = summary.merge(win_counts, on="model", how="left")
    for col in ["task_win_count_R2", "task_win_count_nRMSE"]:
        summary[col] = summary[col].fillna(0).astype(int)
    summary["model_order"] = summary["model"].map({m: i for i, m in enumerate(MODEL_ORDER, start=1)})
    summary["interpretation_note"] = summary["model"].map(
        {
            "RBF\u2013\u03b5-SVR": "Exact RBF epsilon-SVR conventional machine-learning comparator; not a simple baseline.",
            "XGBoost": "Original non-deep learned comparator from the accepted reviewer-safe benchmark.",
            "Persistence": "Simple deterministic benchmark.",
            "Seasonal persistence": "Simple deterministic seasonal benchmark.",
            "Monthly climatology": "Simple deterministic climatological benchmark.",
        }
    ).fillna("Original deep-learning model from the accepted reviewer-safe benchmark.")
    summary = summary.sort_values("model_order").drop(columns=["model_order"])
    summary.to_csv(TABLES_DIR / "candidate_supplementary_full_model_comparison_summary.csv", index=False)

    group_summary = (
        combined.groupby(["model_group"], as_index=False)
        .agg(
            models=("model", lambda x: "; ".join([m for m in MODEL_ORDER if m in set(x)])),
            mean_R2=("outer_test_r2", "mean"),
            median_R2=("outer_test_r2", "median"),
            mean_nRMSE=("outer_test_nrmse", "mean"),
            median_nRMSE=("outer_test_nrmse", "median"),
            evaluated_task_fold_combinations=("outer_test_r2", lambda x: int(pd.to_numeric(x, errors="coerce").notna().sum())),
        )
    )
    group_order = {
        "Simple baseline": 1,
        "Conventional ML benchmark": 2,
        "Deep learning": 3,
    }
    group_summary["group_order"] = group_summary["model_group"].map(group_order)
    group_summary = group_summary.sort_values("group_order").drop(columns=["group_order"])
    group_summary.to_csv(TABLES_DIR / "candidate_supplementary_baseline_benchmark_group_summary.csv", index=False)

    svr_selection = svr.copy()
    svr_selection = svr_selection[
        [
            "target",
            "land_type",
            "fold",
            "selected_candidate_index",
            "selected_C",
            "selected_epsilon",
            "selected_gamma_multiplier",
            "selected_gamma",
            "selected_validation_r2",
            "selected_validation_nrmse",
            "outer_test_r2",
            "outer_test_nrmse",
            "support_vector_fraction",
            "runtime_seconds",
            "status",
            "n_configs_within_1pct_nrmse",
        ]
    ]
    svr_selection.to_csv(TABLES_DIR / "candidate_supplementary_svr_hyperparameter_selection_summary.csv", index=False)
    return combined, summary, group_summary, svr_selection


def consolidate_svr_predictions(final_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, result in final_results.iterrows():
        path_text = str(result.get("prediction_path", ""))
        if not path_text:
            continue
        path = Path(path_text)
        if not path.exists():
            continue
        arrays = np.load(path, allow_pickle=False)
        frame = pd.DataFrame(
            {
                "model": "RBF\u2013\u03b5-SVR",
                "target": result["target"],
                "land_type": result["land_type"],
                "fold": int(result["fold"]),
                "date": arrays["date"],
                "pixel_id": arrays["pixel_id"],
                "row": arrays["row"],
                "col": arrays["col"],
                "observed": arrays["y_true"],
                "predicted": arrays["y_pred"],
            }
        )
        rows.append(frame)
    if rows:
        predictions = pd.concat(rows, ignore_index=True)
    else:
        predictions = pd.DataFrame()
    predictions.to_csv(PREDICTIONS_DIR / "svr_outer_test_predictions.csv", index=False)
    return predictions


def verify_svr_metrics_from_predictions(final_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, result in final_results.iterrows():
        path_text = str(result.get("prediction_path", ""))
        if not path_text or not Path(path_text).exists():
            rows.append(
                {
                    "target": result.get("target"),
                    "land_type": result.get("land_type"),
                    "fold": result.get("fold"),
                    "prediction_path": path_text,
                    "status": "missing_prediction_file",
                }
            )
            continue
        arrays = np.load(path_text, allow_pickle=False)
        metrics = calculate_metrics(arrays["y_true"], arrays["y_pred"])
        rows.append(
            {
                "target": result["target"],
                "land_type": result["land_type"],
                "fold": int(result["fold"]),
                "prediction_path": path_text,
                "stored_outer_test_r2": result.get("outer_test_r2", np.nan),
                "recomputed_outer_test_r2": metrics["r2"],
                "abs_r2_difference": abs(float(result.get("outer_test_r2", np.nan)) - float(metrics["r2"])),
                "stored_outer_test_nrmse": result.get("outer_test_nrmse", np.nan),
                "recomputed_outer_test_nrmse": metrics["normalized_rmse"],
                "abs_nrmse_difference": abs(float(result.get("outer_test_nrmse", np.nan)) - float(metrics["normalized_rmse"])),
                "stored_n": result.get("outer_test_n", np.nan),
                "recomputed_n": metrics["n"],
                "status": "pass"
                if abs(float(result.get("outer_test_r2", np.nan)) - float(metrics["r2"])) < 1e-9
                and abs(float(result.get("outer_test_nrmse", np.nan)) - float(metrics["normalized_rmse"])) < 1e-9
                else "review",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDITS_DIR / "svr_prediction_metric_recalculation_check.csv", index=False)
    return frame


def task_definition_alignment_check(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, land_type, fold), group in combined.groupby(["target", "land_type", "fold"], observed=False):
        n_by_model = group.set_index("model")["outer_test_n"].to_dict()
        finite_counts = [float(v) for v in n_by_model.values() if np.isfinite(pd.to_numeric(v, errors="coerce"))]
        rows.append(
            {
                "target": target,
                "land_type": land_type,
                "fold": int(fold),
                "model_count_present": int(group["model"].nunique()),
                "finite_metric_model_count": int(pd.to_numeric(group["outer_test_r2"], errors="coerce").notna().sum()),
                "min_outer_test_n": min(finite_counts) if finite_counts else np.nan,
                "max_outer_test_n": max(finite_counts) if finite_counts else np.nan,
                "outer_test_n_identical_across_models": bool(len(set(finite_counts)) <= 1) if finite_counts else False,
                "models_present": ";".join([m for m in MODEL_ORDER if m in set(group["model"])]),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(AUDITS_DIR / "task_definition_alignment_check.csv", index=False)
    return frame


def runtime_summaries(validation_results: pd.DataFrame, final_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    success = validation_results[validation_results["status"].isin(SUCCESS_STATUSES)].copy()
    runtime_rows = []
    if not success.empty:
        success["fit_seconds"] = pd.to_numeric(success["fit_seconds"], errors="coerce")
        for label, group in [("all_successful_validation_candidates", success)]:
            fit = group["fit_seconds"].dropna()
            runtime_rows.append(
                {
                    "group": label,
                    "n": int(len(fit)),
                    "min_fit_seconds": float(fit.min()),
                    "median_fit_seconds": float(fit.median()),
                    "mean_fit_seconds": float(fit.mean()),
                    "p75_fit_seconds": float(fit.quantile(0.75)),
                    "p90_fit_seconds": float(fit.quantile(0.90)),
                    "p95_fit_seconds": float(fit.quantile(0.95)),
                    "max_fit_seconds": float(fit.max()),
                }
            )
        for column in ["target", "land_type", "C", "epsilon", "gamma_multiplier"]:
            for value, group in success.groupby(column):
                fit = pd.to_numeric(group["fit_seconds"], errors="coerce").dropna()
                if len(fit):
                    runtime_rows.append(
                        {
                            "group": f"{column}={value}",
                            "n": int(len(fit)),
                            "min_fit_seconds": float(fit.min()),
                            "median_fit_seconds": float(fit.median()),
                            "mean_fit_seconds": float(fit.mean()),
                            "p75_fit_seconds": float(fit.quantile(0.75)),
                            "p90_fit_seconds": float(fit.quantile(0.90)),
                            "p95_fit_seconds": float(fit.quantile(0.95)),
                            "max_fit_seconds": float(fit.max()),
                        }
                    )
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(TABLES_DIR / "svr_full_validation_runtime_summary.csv", index=False)

    timeout = validation_results[validation_results["status"] == "timeout_computationally_infeasible"].copy()
    timeout.to_csv(TABLES_DIR / "svr_full_timeout_summary.csv", index=False)
    memory = (
        success.groupby(["target", "land_type", "fold"], as_index=False)
        .agg(
            validation_candidate_count=("candidate_index", "count"),
            max_observed_rss_mb=("rss_mb_max", "max"),
            mean_observed_rss_mb=("rss_mb_mean", "mean"),
            max_cpu_percent=("cpu_percent_max", "max"),
            median_support_vector_fraction=("support_vector_fraction", "median"),
            max_support_vector_fraction=("support_vector_fraction", "max"),
        )
        if not success.empty
        else pd.DataFrame()
    )
    if not final_results.empty:
        final_mem = final_results.copy()
        final_mem["rss_mb_max"] = pd.to_numeric(final_mem["rss_mb_max"], errors="coerce")
        final_mem = final_mem[["target", "land_type", "fold", "rss_mb_max", "support_vector_fraction", "fit_seconds"]]
        final_mem.to_csv(TABLES_DIR / "svr_full_final_refit_memory_and_runtime.csv", index=False)
    memory.to_csv(TABLES_DIR / "svr_full_memory_summary.csv", index=False)
    return runtime, memory


def make_supplementary_figure(combined: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plot = combined.copy()
    plot["model"] = pd.Categorical(plot["model"], categories=MODEL_ORDER, ordered=True)
    plot = plot.sort_values(["model", "target", "land_type", "fold"])
    plot.to_csv(FIGURE_SOURCE_DIR / "candidate_new_supplementary_forecasting_comparison_panel_values.csv", index=False)
    plot[["model_group", "model", "target", "land_type", "fold", "outer_test_r2"]].to_csv(
        FIGURE_SOURCE_DIR / "candidate_new_supplementary_forecasting_comparison_R2_source.csv",
        index=False,
    )
    plot[["model_group", "model", "target", "land_type", "fold", "outer_test_nrmse"]].to_csv(
        FIGURE_SOURCE_DIR / "candidate_new_supplementary_forecasting_comparison_nRMSE_source.csv",
        index=False,
    )

    colors = {
        "Simple baseline": "#8da0cb",
        "Conventional ML benchmark": "#66c2a5",
        "Deep learning": "#fc8d62",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 9.0), sharey=True)
    y_positions = np.arange(len(MODEL_ORDER))
    jitter = np.linspace(-0.18, 0.18, 144)
    rng = np.random.default_rng(12345)

    for ax, metric, xlabel, title in [
        (axes[0], "outer_test_r2", "Outer-test R\u00b2", "(a) R\u00b2, higher is better"),
        (axes[1], "outer_test_nrmse", "Outer-test nRMSE", "(b) nRMSE, lower is better"),
    ]:
        ax.axhspan(-0.5, 2.5, color=colors["Simple baseline"], alpha=0.10, zorder=0)
        ax.axhspan(2.5, 4.5, color=colors["Conventional ML benchmark"], alpha=0.10, zorder=0)
        ax.axhspan(4.5, 9.5, color=colors["Deep learning"], alpha=0.10, zorder=0)
        for y, model in zip(y_positions, MODEL_ORDER):
            values = pd.to_numeric(plot.loc[plot["model"] == model, metric], errors="coerce").dropna().to_numpy()
            if values.size == 0:
                continue
            group = MODEL_GROUP[model]
            ax.boxplot(
                values,
                positions=[y],
                vert=False,
                widths=0.48,
                patch_artist=True,
                manage_ticks=False,
                boxprops={"facecolor": colors[group], "edgecolor": "#333333", "linewidth": 1.1, "alpha": 0.75},
                medianprops={"color": "#111111", "linewidth": 1.4},
                whiskerprops={"color": "#333333", "linewidth": 1.0},
                capprops={"color": "#333333", "linewidth": 1.0},
                flierprops={"marker": "", "markersize": 0},
            )
            local_jitter = rng.choice(jitter, size=values.size, replace=True) if values.size <= len(jitter) else rng.normal(0, 0.08, size=values.size)
            ax.scatter(
                values,
                np.full(values.size, y) + local_jitter,
                s=14,
                color="#222222",
                alpha=0.36,
                linewidth=0,
                zorder=3,
            )
        if metric == "outer_test_r2":
            ax.axvline(0, color="#555555", linestyle="--", linewidth=1.0)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_title(title, fontsize=15, fontweight="bold", loc="left")
        ax.tick_params(axis="x", labelsize=12)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.8)
        ax.set_axisbelow(True)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(MODEL_ORDER, fontsize=13)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Model", fontsize=14)
    axes[1].tick_params(axis="y", labelleft=False)
    legend_handles = [
        Patch(facecolor=colors[group], edgecolor="#333333", alpha=0.75, label=group)
        for group in ["Simple baseline", "Conventional ML benchmark", "Deep learning"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=13,
        bbox_to_anchor=(0.5, 0.015),
    )
    fig.suptitle(
        "Candidate supplementary forecasting comparison including RBF\u2013\u03b5-SVR",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.055,
        "Each point is one target \u00d7 land-cover \u00d7 rolling-origin fold task. "
        "SVR is a conventional machine-learning comparator, not a simple baseline.",
        ha="center",
        fontsize=11.5,
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.91, bottom=0.14, wspace=0.14)
    for ext in ["png", "pdf", "svg"]:
        save_kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            save_kwargs["dpi"] = 600
        fig.savefig(FIGURES_DIR / f"candidate_new_supplementary_forecasting_comparison.{ext}", **save_kwargs)
    plt.close(fig)


def write_caption_and_interpretation(summary: pd.DataFrame) -> None:
    caption = (
        "Candidate Supplementary Figure Sx. Forecasting comparison including simple baselines, "
        "conventional machine-learning benchmarks, and deep-learning models. Panel (a) shows "
        "outer-test R\u00b2 and panel (b) shows outer-test normalized RMSE (nRMSE) for one point per "
        "target \u00d7 land-cover \u00d7 rolling-origin fold task. Persistence, seasonal persistence, and "
        "monthly climatology are simple baselines; XGBoost and RBF\u2013\u03b5-SVR are conventional "
        "machine-learning benchmarks; LSTM, BiLSTM, TCN, N-BEATS, and N-HiTS are deep-learning "
        "models. Figure 7 remains unchanged and reports the accepted pixel-level learned-model "
        "benchmark separately."
    )
    (REPORTS_DIR / "candidate_new_supplementary_forecasting_comparison_caption_draft.md").write_text(
        caption + "\n",
        encoding="utf-8",
    )
    top = summary.sort_values(["mean_R2", "mean_nRMSE"], ascending=[False, True]).head(3)
    lines = [
        "# Short Interpretation Draft",
        "",
        "This candidate supplementary comparison adds exact RBF\u2013\u03b5-SVR to the accepted reviewer-safe forecasting outputs without altering Figure 7.",
        "Model rankings should be interpreted as descriptive across target, land-cover, and fold tasks, not as universal dominance.",
        "",
        "Highest mean R2 models in this candidate table:",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['model']}: mean R2={row['mean_R2']:.3f}, mean nRMSE={row['mean_nRMSE']:.3f}")
    (REPORTS_DIR / "candidate_new_supplementary_forecasting_comparison_interpretation_draft.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_runtime_estimate(runtime: pd.DataFrame, validation_results: pd.DataFrame) -> pd.DataFrame:
    success = validation_results[validation_results["status"].isin(SUCCESS_STATUSES)].copy()
    success["fit_seconds"] = pd.to_numeric(success["fit_seconds"], errors="coerce")
    fit = success["fit_seconds"].dropna()
    total_validation_fits = 3888
    total_final_refits = 144
    total_all = total_validation_fits + total_final_refits
    rows = []
    if fit.empty:
        estimate = pd.DataFrame()
        estimate.to_csv(TABLES_DIR / "svr_full_runtime_estimate_from_observed.csv", index=False)
        return estimate
    quantiles = {
        "optimistic_median_successful_fit": fit.quantile(0.50),
        "central_p75_successful_fit": fit.quantile(0.75),
        "conservative_p90_successful_fit": fit.quantile(0.90),
        "very_conservative_p95_successful_fit": fit.quantile(0.95),
    }
    max_rss = float(pd.to_numeric(success.get("rss_mb_max", pd.Series(dtype=float)), errors="coerce").max())
    available_mb = float("nan")
    if psutil is not None:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
    safe_workers = 1
    if np.isfinite(max_rss) and max_rss > 0 and np.isfinite(available_mb):
        safe_workers = max(1, int(math.floor((available_mb * 0.70) / max_rss)))
    for label, seconds in quantiles.items():
        for workers in sorted(set([1, 2, 4, safe_workers])):
            rows.append(
                {
                    "scenario": label,
                    "fit_seconds_assumed": float(seconds),
                    "total_validation_fits": total_validation_fits,
                    "total_final_refits": total_final_refits,
                    "total_fits": total_all,
                    "estimated_cpu_hours": float(seconds * total_all / 3600),
                    "worker_count": int(workers),
                    "estimated_wall_clock_hours": float(seconds * total_all / 3600 / workers),
                    "max_observed_rss_mb_per_worker": max_rss,
                    "estimated_combined_rss_mb": max_rss * workers if np.isfinite(max_rss) else np.nan,
                    "available_memory_mb_at_estimation": available_mb,
                    "memory_safe_by_70pct_available_rule": bool(max_rss * workers <= available_mb * 0.70)
                    if np.isfinite(max_rss) and np.isfinite(available_mb)
                    else None,
                    "observed_validation_timeout_count": int((validation_results["status"] == "timeout_computationally_infeasible").sum()),
                }
            )
    estimate = pd.DataFrame(rows)
    estimate.to_csv(TABLES_DIR / "svr_full_runtime_estimate_from_observed.csv", index=False)
    return estimate


def status_payload(validation_results: pd.DataFrame, final_results: pd.DataFrame, coverage: pd.DataFrame) -> dict:
    validation_status_counts = validation_results["status"].value_counts().to_dict()
    final_status_counts = final_results["status"].value_counts().to_dict() if not final_results.empty else {}
    if len(validation_results) != 3888:
        status = "FULL_SVR_INCOMPLETE"
    elif (coverage["grid_coverage_status"] == "INADEQUATE_GRID_COVERAGE").any():
        status = "FULL_SVR_COMPLETED_WITH_INADEQUATE_GRID_COVERAGE_AUTHOR_DECISION_REQUIRED"
    elif len(final_results) != 144:
        status = "FULL_SVR_VALIDATION_COMPLETE_FINAL_REFIT_INCOMPLETE"
    elif int(final_status_counts.get("timeout_computationally_infeasible", 0)) > 0:
        status = "FULL_SVR_COMPLETED_WITH_FINAL_REFIT_TIMEOUTS_AUTHOR_REVIEW"
    elif int(validation_status_counts.get("timeout_computationally_infeasible", 0)) > 0:
        status = "FULL_SVR_COMPLETED_WITH_ISOLATED_VALIDATION_TIMEOUTS"
    else:
        status = "FULL_SVR_COMPLETED"
    payload = {
        "status": status,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "expected_validation_candidate_rows": 3888,
        "observed_validation_candidate_rows": int(len(validation_results)),
        "expected_final_refit_rows": 144,
        "observed_final_refit_rows": int(len(final_results)),
        "validation_status_counts": {str(k): int(v) for k, v in validation_status_counts.items()},
        "final_refit_status_counts": {str(k): int(v) for k, v in final_status_counts.items()},
        "outer_test_used_for_svr_tuning": False,
        "figure7_modified": False,
        "manuscript_modified": False,
        "hyperparameter_grid_changed": False,
        "forecasting_task_changed": False,
    }
    write_json(TABLES_DIR / "svr_full_benchmark_status.json", payload)
    return payload


def write_manifest() -> pd.DataFrame:
    rows = []
    for path in PACKAGE_ROOT.rglob("*"):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(PACKAGE_ROOT)),
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    "sha256": file_sha256(path),
                }
            )
    frame = pd.DataFrame(rows).sort_values("relative_path")
    frame.to_csv(MANIFESTS_DIR / "package_file_manifest.csv", index=False)
    return frame


def write_report(
    precheck: pd.DataFrame,
    source_inventory: pd.DataFrame,
    validation_results: pd.DataFrame,
    selected: pd.DataFrame,
    final_results: pd.DataFrame,
    coverage: pd.DataFrame,
    summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    runtime: pd.DataFrame,
    memory: pd.DataFrame,
    estimate: pd.DataFrame,
    qc: pd.DataFrame,
    alignment: pd.DataFrame,
    status: dict,
) -> None:
    validation_counts = validation_results["status"].value_counts().to_dict()
    final_counts = final_results["status"].value_counts().to_dict() if not final_results.empty else {}
    inadequate = coverage[coverage["grid_coverage_status"] == "INADEQUATE_GRID_COVERAGE"]
    timeout_count = int(validation_counts.get("timeout_computationally_infeasible", 0))
    final_timeout_count = int(final_counts.get("timeout_computationally_infeasible", 0))
    lines = [
        "# SVR Full Benchmark Supplementary Candidate Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Plain-language outcome",
        "",
        f"Full SVR status: `{status['status']}`.",
        f"The validation grid contains {len(validation_results)} rows out of 3,888 expected candidate fits.",
        f"Validation timeouts recorded: {timeout_count}. Final refit timeouts recorded: {final_timeout_count}.",
        "Figure 7, manuscript text, response letter, and authoritative forecasting outputs were not modified.",
        "",
        "## Framework preservation",
        "",
        "- Model: exact scikit-learn epsilon-SVR.",
        "- Kernel: RBF.",
        "- Input: 12 previous monthly observations from the same pixel.",
        "- Target: same variable, same pixel, one month ahead.",
        "- Task structure: one pooled model per target x land-cover x fold.",
        "- Hyperparameter grid unchanged: C in {1, 3, 10}; epsilon in {0.05, 0.10, 0.20}; gamma multiplier in {0.5, 1, 2} x gamma_scale.",
        "- Selection used internal validation nRMSE only, with the approved 1% simplicity tie rule.",
        "- Held-out outer-test data were evaluated only after validation-based selection.",
        "",
        "## Pre-run verification",
        "",
        precheck.to_markdown(index=False),
        "",
        "## Authoritative source tables inspected",
        "",
        source_inventory.to_markdown(index=False),
        "",
        "## Validation candidate status",
        "",
        pd.DataFrame(
            [{"status": k, "count": v} for k, v in validation_counts.items()]
        ).to_markdown(index=False),
        "",
        "## Grid coverage",
        "",
        coverage["grid_coverage_status"].value_counts().rename_axis("grid_coverage_status").reset_index(name="count").to_markdown(index=False),
    ]
    if not inadequate.empty:
        lines.extend(["", "Tasks with inadequate grid coverage:", inadequate.to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Selected SVR configurations",
            "",
            selected[
                [
                    "target",
                    "land_type",
                    "fold",
                    "selected_C",
                    "selected_epsilon",
                    "selected_gamma_multiplier",
                    "selected_validation_nrmse",
                    "selected_validation_r2",
                    "n_configs_within_1pct_nrmse",
                ]
            ].head(30).to_markdown(index=False),
            "",
            "The full selected-configuration table is saved as `tables/svr_full_selected_configurations_validation_only.csv`.",
            "",
            "## Final refit status",
            "",
            pd.DataFrame(
                [{"status": k, "count": v} for k, v in final_counts.items()]
            ).to_markdown(index=False)
            if final_counts
            else "No final refits were completed.",
            "",
            "## New supplementary figure and tables",
            "",
            "- Figure: `figures/candidate_new_supplementary_forecasting_comparison.png/pdf/svg`.",
            "- Source data: `figure_source_data/candidate_new_supplementary_forecasting_comparison_panel_values.csv`.",
            "- Table A: `tables/candidate_supplementary_full_model_comparison_summary.csv`.",
            "- Table B: `tables/candidate_supplementary_task_level_outer_test_results.csv`.",
            "- Table C: `tables/candidate_supplementary_baseline_benchmark_group_summary.csv`.",
            "- Table D: `tables/candidate_supplementary_svr_hyperparameter_selection_summary.csv`.",
            "",
            "## Model comparison summary",
            "",
            summary.to_markdown(index=False),
            "",
            "## Group summary",
            "",
            group_summary.to_markdown(index=False),
            "",
            "## Runtime distribution",
            "",
            runtime.head(30).to_markdown(index=False) if not runtime.empty else "Runtime table is empty.",
            "",
            "## Memory limitations",
            "",
            memory.head(30).to_markdown(index=False) if not memory.empty else "No memory observations were available.",
            "",
            "Timeout subprocesses may not return peak-memory observations; this limitation is preserved rather than filled.",
            "",
            "## Runtime estimate",
            "",
            estimate.head(20).to_markdown(index=False) if not estimate.empty else "Runtime estimate unavailable.",
            "",
            "## Quality control",
            "",
            "- SVR prediction files were recomputed against stored SVR outer-test metrics.",
            "- Existing accepted reviewer-safe metrics were aggregated to one model x target x land-cover x fold value before plotting.",
            "- SARIMA was not inserted into the new pixel-level supplementary comparison.",
            "- SVR is labelled as a conventional machine-learning benchmark/comparator, not a simple baseline.",
            "",
            qc["status"].value_counts().rename_axis("svr_metric_recalculation_status").reset_index(name="count").to_markdown(index=False)
            if not qc.empty
            else "No SVR prediction QC rows were created.",
            "",
            alignment["outer_test_n_identical_across_models"].value_counts().rename_axis("outer_test_n_identical_across_models").reset_index(name="task_count").to_markdown(index=False)
            if not alignment.empty
            else "No task-alignment rows were created.",
            "",
            "## Evidence table",
            "",
            pd.DataFrame(
                [
                    {
                        "Question": "Was the approved grid preserved?",
                        "Evidence-based answer": "Yes.",
                        "Exact file": str(SCRIPT_DIR / "candidate_config.py"),
                        "Exact function": "module constants",
                        "Relevant line range": "SVR_C_VALUES / SVR_EPSILON_VALUES / SVR_GAMMA_MULTIPLIERS",
                        "Numerical evidence": "3 x 3 x 3 candidates per task-fold; 3,888 expected validation rows",
                        "Confidence": "High",
                        "Remaining uncertainty": "None identified",
                    },
                    {
                        "Question": "Was outer-test data used for SVR tuning?",
                        "Evidence-based answer": "No. Candidate selection uses validation metrics only; outer-test metrics are generated after selection.",
                        "Exact file": str(Path(__file__).resolve()),
                        "Exact function": "select_validation_configs; run_final_refits",
                        "Relevant line range": "selection and final-refit functions",
                        "Numerical evidence": f"{len(selected)} selected rows; status flag outer_test_used_for_svr_tuning=False",
                        "Confidence": "High",
                        "Remaining uncertainty": "Depends on unchanged imported preprocessing code, verified by hash check.",
                    },
                    {
                        "Question": "Was Figure 7 modified?",
                        "Evidence-based answer": "No.",
                        "Exact file": str(PACKAGE_ROOT),
                        "Exact function": "not applicable",
                        "Relevant line range": "not applicable",
                        "Numerical evidence": "All outputs were written to this new candidate package.",
                        "Confidence": "High",
                        "Remaining uncertainty": "None identified",
                    },
                    {
                        "Question": "Is SVR included as a simple baseline?",
                        "Evidence-based answer": "No. It is grouped as a conventional ML benchmark.",
                        "Exact file": str(TABLES_DIR / "candidate_supplementary_task_level_outer_test_results.csv"),
                        "Exact function": "build_combined_tables",
                        "Relevant line range": "MODEL_GROUP mapping",
                        "Numerical evidence": "RBF-epsilon-SVR group = Conventional ML benchmark",
                        "Confidence": "High",
                        "Remaining uncertainty": "None identified",
                    },
                ]
            ).to_markdown(index=False),
            "",
            "## Files not modified",
            "",
            "- Figure 7 and its source tables.",
            "- Manuscript files.",
            "- Response letter.",
            "- Existing final figures.",
            "- Existing authoritative forecasting outputs.",
        ]
    )
    (REPORTS_DIR / "FINAL_SVR_full_benchmark_supplementary_candidate_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_readme(status: dict) -> None:
    text = f"""# SVR Full Benchmark Supplementary Candidate

Generated: {datetime.now().isoformat(timespec='seconds')}

Status: `{status['status']}`

This package contains a candidate full RBF-epsilon-SVR benchmark and a new supplementary-only forecasting comparison. It does not modify Figure 7, manuscript files, response-letter files, or authoritative forecasting outputs.

The SVR benchmark used the approved framework:

- exact scikit-learn epsilon-SVR;
- RBF kernel;
- 12 same-pixel monthly lag inputs;
- same-variable, same-pixel one-month-ahead target;
- one pooled model per target x land-cover x rolling-origin fold;
- validation-only hyperparameter selection;
- outer-test evaluation only after selection.

Timed-out validation configurations are retained as `timeout_computationally_infeasible` and excluded from validation-based selection. No validation score is fabricated for a timed-out configuration.
"""
    (PACKAGE_ROOT / "README.md").write_text(text, encoding="utf-8")


def auto_worker_count(requested) -> int:
    if str(requested).lower() != "auto":
        return max(1, int(requested))
    logical = psutil.cpu_count(logical=True) if psutil is not None else None
    logical = int(logical or 4)
    available_mb = (
        psutil.virtual_memory().available / (1024 * 1024)
        if psutil is not None
        else float("nan")
    )
    rss_candidates = []
    for path in [
        TABLES_DIR / "svr_full_memory_summary.csv",
        PILOT_PACKAGE / "tables" / "pilot_complete_candidate_results.csv",
    ]:
        if path.exists():
            try:
                frame = pd.read_csv(path)
                for column in ["rss_mb_max", "max_observed_rss_mb"]:
                    if column in frame.columns:
                        values = pd.to_numeric(frame[column], errors="coerce").dropna()
                        if len(values):
                            rss_candidates.append(float(values.max()))
            except Exception:
                pass
    observed_or_assumed_rss_mb = max(rss_candidates) if rss_candidates else 700.0
    memory_workers = logical
    if np.isfinite(available_mb) and observed_or_assumed_rss_mb > 0:
        memory_workers = max(1, int(math.floor((available_mb * 0.70) / observed_or_assumed_rss_mb)))
    # Leave headroom for Windows, OneDrive, and the parent orchestrator.
    return max(1, min(12, max(1, logical - 4), memory_workers))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", nargs=2, metavar=("JOB_JSON", "OUT_JSON"))
    parser.add_argument(
        "--max-workers",
        default="auto",
        help="Parallel SVR subprocess count. Use an integer or 'auto'.",
    )
    args = parser.parse_args()

    if args.worker:
        fit_predict_worker(Path(args.worker[0]), Path(args.worker[1]))
        return

    max_workers = auto_worker_count(args.max_workers)
    log("Starting full SVR benchmark supplementary candidate.")
    log(f"Using max_workers={max_workers} for orchestration; this changes only execution parallelism.")
    precheck = verify_pilot_and_sources()
    source_inventory = source_table_inventory()

    log("Preparing task arrays and expected validation candidate table.")
    expected = expected_validation_candidates()
    if len(expected) != 3888:
        raise RuntimeError(f"Expected 3,888 validation candidates, found {len(expected)}")

    validation_results = run_validation_candidates(expected, max_workers=max_workers)
    coverage = grid_coverage(validation_results)
    selected = select_validation_configs(validation_results)
    final_results = run_final_refits(selected, max_workers=max_workers)

    consolidate_svr_predictions(final_results)
    qc = verify_svr_metrics_from_predictions(final_results)
    runtime, memory = runtime_summaries(validation_results, final_results)
    estimate = write_runtime_estimate(runtime, validation_results)
    combined, summary, group_summary, svr_selection = build_combined_tables(final_results, selected)
    alignment = task_definition_alignment_check(combined)
    make_supplementary_figure(combined)
    write_caption_and_interpretation(summary)
    status = status_payload(validation_results, final_results, coverage)
    write_report(
        precheck,
        source_inventory,
        validation_results,
        selected,
        final_results,
        coverage,
        summary,
        group_summary,
        runtime,
        memory,
        estimate,
        qc,
        alignment,
        status,
    )
    write_readme(status)
    write_manifest()
    log(f"Finished full SVR benchmark supplementary candidate with status {status['status']}.")


if __name__ == "__main__":
    main()
