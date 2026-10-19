from datetime import datetime

import pandas as pd

from candidate_config import (
    LAND_TYPES,
    LOGS_DIR,
    PREDICTIONS_DIR,
    SEEDS,
    TABLES_DIR,
    VARIABLES,
)
from run_helpers import run_torch_tasks

PIXEL_OUT = TABLES_DIR / "reviewer_safe_lstm_pixel_metrics.csv"
MONTHLY_OUT = TABLES_DIR / "reviewer_safe_lstm_monthly_aggregated_metrics.csv"
PREDICTION_OUT = PREDICTIONS_DIR / "reviewer_safe_lstm_predictions.csv.gz"
MANIFEST_OUT = TABLES_DIR / "reviewer_safe_lstm_run_manifest.csv"


def save(pixel_frames, monthly_frames, prediction_frames, manifest_frames):
    pd.concat(pixel_frames, ignore_index=True).to_csv(PIXEL_OUT, index=False)
    pd.concat(monthly_frames, ignore_index=True).to_csv(MONTHLY_OUT, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        PREDICTION_OUT, index=False, compression="gzip"
    )
    pd.concat(manifest_frames, ignore_index=True).to_csv(MANIFEST_OUT, index=False)


def main():
    log_path = LOGS_DIR / "reviewer_safe_lstm_log.txt"
    expected = len(LAND_TYPES) * len(VARIABLES) * 4 * len(SEEDS)
    if PIXEL_OUT.exists() and MONTHLY_OUT.exists() and PREDICTION_OUT.exists():
        existing = pd.read_csv(PIXEL_OUT)
        if len(existing) >= expected and set(pd.to_numeric(existing["seed"]).astype(int)) == set(SEEDS):
            log_path.write_text(
                "Existing complete three-seed reviewer-safe LSTM outputs verified; rerun skipped.\n",
                encoding="utf-8",
            )
            return

    pixel_frames = []
    monthly_frames = []
    prediction_frames = []
    manifest_frames = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {datetime.now().isoformat()}\n")
        for seed in SEEDS:
            temp_log = LOGS_DIR / f"_lstm_seed_{seed}.log"
            pixel, monthly, predictions, manifest = run_torch_tasks(
                model_names=["LSTM"],
                land_types=LAND_TYPES,
                variables=VARIABLES,
                seeds_for_task=lambda model, land, variable, spec, value=seed: [value],
                include_mode_b_for_model=lambda model, land, variable: True,
                log_path=temp_log,
                include_mode_a=False,
            )
            for frame in [pixel, monthly, predictions, manifest]:
                if "mode" in frame:
                    frame["mode"] = "reviewer_safe_rolling_origin"
            pixel_frames.append(pixel)
            monthly_frames.append(monthly)
            prediction_frames.append(predictions)
            manifest_frames.append(manifest)
            log.write(temp_log.read_text(encoding="utf-8"))
            temp_log.unlink(missing_ok=True)
            save(pixel_frames, monthly_frames, prediction_frames, manifest_frames)
            log.write(f"Checkpoint saved after seed {seed}\n")
            log.flush()
        log.write(f"Finished: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
