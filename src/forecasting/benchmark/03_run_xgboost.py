import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from candidate_config import (
    LAND_TYPES,
    LOGS_DIR,
    PREDICTIONS_DIR,
    SEEDS,
    SOURCE_PREDICTIONS_DIR,
    SOURCE_TABLES_DIR,
    TABLES_DIR,
    VARIABLES,
)
from run_helpers import run_xgboost_tasks

PIXEL_OUT = TABLES_DIR / "reviewer_safe_xgboost_pixel_metrics.csv"
MONTHLY_OUT = TABLES_DIR / "reviewer_safe_xgboost_monthly_aggregated_metrics.csv"
PREDICTION_OUT = PREDICTIONS_DIR / "reviewer_safe_xgboost_predictions.csv.gz"
MANIFEST_OUT = TABLES_DIR / "reviewer_safe_xgboost_run_manifest.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_mode(frame):
    return frame[
        (frame["mode"] == "Mode_B_rolling_origin")
        & (frame["preprocessing"] == "reviewer_mode_b")
        & (frame["scaling"] == "monthwise_z")
        & (~frame["unit_scale_reason"].str.contains("observed|test-period", case=False, na=False))
    ].copy()


def load_verified_seed42(log):
    pixel_path = SOURCE_TABLES_DIR / "xgboost_pixel_metrics.csv"
    monthly_path = SOURCE_TABLES_DIR / "xgboost_monthly_aggregated_metrics.csv"
    prediction_path = SOURCE_PREDICTIONS_DIR / "xgboost_predictions.csv.gz"
    pixel = safe_mode(pd.read_csv(pixel_path))
    monthly = safe_mode(pd.read_csv(monthly_path))
    predictions = safe_mode(pd.read_csv(prediction_path))
    pixel = pixel[pixel["seed"] == 42]
    monthly = monthly[monthly["seed"] == 42]
    predictions = predictions[predictions["seed"] == 42]
    expected_tasks = len(LAND_TYPES) * len(VARIABLES) * 4
    if len(pixel) != expected_tasks or len(monthly) != expected_tasks:
        raise RuntimeError(
            f"Seed-42 reused XGBoost metrics incomplete: pixel={len(pixel)}, monthly={len(monthly)}"
        )
    task_count = predictions[
        ["land_type", "variable", "fold", "seed"]
    ].drop_duplicates().shape[0]
    if task_count != expected_tasks:
        raise RuntimeError(f"Seed-42 reused XGBoost predictions cover {task_count} tasks")
    for frame in [pixel, monthly, predictions]:
        frame["mode"] = "reviewer_safe_rolling_origin"
        frame["reuse_status"] = "verified_reuse_from_raster_full_candidate"
    log.write(
        f"Verified seed-42 reuse: {expected_tasks} tasks; source hashes "
        f"pixel={sha256(pixel_path)}, monthly={sha256(monthly_path)}, "
        f"predictions={sha256(prediction_path)}\n"
    )
    return pixel, monthly, predictions


def save(pixel, monthly, predictions, manifests):
    pixel.to_csv(PIXEL_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    predictions.to_csv(PREDICTION_OUT, index=False, compression="gzip")
    pd.concat(manifests, ignore_index=True).to_csv(MANIFEST_OUT, index=False)


def main():
    log_path = LOGS_DIR / "reviewer_safe_xgboost_log.txt"
    if PIXEL_OUT.exists() and MONTHLY_OUT.exists() and PREDICTION_OUT.exists():
        existing = pd.read_csv(PIXEL_OUT)
        if (
            len(existing) >= len(LAND_TYPES) * len(VARIABLES) * 4 * len(SEEDS)
            and set(pd.to_numeric(existing["seed"]).astype(int)) == set(SEEDS)
        ):
            log_path.write_text(
                "Existing complete three-seed reviewer-safe XGBoost outputs verified; rerun skipped.\n",
                encoding="utf-8",
            )
            return

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {datetime.now().isoformat()}\n")
        base_pixel, base_monthly, base_predictions = load_verified_seed42(log)
        pixel_frames = [base_pixel]
        monthly_frames = [base_monthly]
        prediction_frames = [base_predictions]
        manifest_frames = [
            pd.DataFrame(
                {
                    "model": ["XGBoost"] * 144,
                    "mode": ["reviewer_safe_rolling_origin"] * 144,
                    "fold": base_pixel["fold"].to_numpy(),
                    "land_type": base_pixel["land_type"].to_numpy(),
                    "variable": base_pixel["variable"].to_numpy(),
                    "seed": [42] * 144,
                    "status": ["verified_reuse"] * 144,
                    "error": [""] * 144,
                }
            )
        ]
        for seed in [123, 2026]:
            temp_log = LOGS_DIR / f"_xgboost_seed_{seed}.log"
            pixel, monthly, predictions, manifest = run_xgboost_tasks(
                land_types=LAND_TYPES,
                variables=VARIABLES,
                seeds_for_task=lambda model, land, variable, spec, value=seed: [value],
                include_mode_b=True,
                log_path=temp_log,
                include_mode_a=False,
            )
            for frame in [pixel, monthly, predictions, manifest]:
                if "mode" in frame:
                    frame["mode"] = "reviewer_safe_rolling_origin"
            pixel["reuse_status"] = "new_reviewer_safe_run"
            monthly["reuse_status"] = "new_reviewer_safe_run"
            predictions["reuse_status"] = "new_reviewer_safe_run"
            pixel_frames.append(pixel)
            monthly_frames.append(monthly)
            prediction_frames.append(predictions)
            manifest_frames.append(manifest)
            log.write(temp_log.read_text(encoding="utf-8"))
            temp_log.unlink(missing_ok=True)
            save(
                pd.concat(pixel_frames, ignore_index=True),
                pd.concat(monthly_frames, ignore_index=True),
                pd.concat(prediction_frames, ignore_index=True),
                manifest_frames,
            )
            log.write(f"Checkpoint saved after seed {seed}\n")
            log.flush()
        log.write(f"Finished: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
