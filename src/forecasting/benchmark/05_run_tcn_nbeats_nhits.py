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

MODELS = ["N-BEATS", "BiLSTM", "TCN", "N-HiTS"]
PIXEL_OUT = TABLES_DIR / "reviewer_safe_remaining_models_pixel_metrics.csv"
MONTHLY_OUT = TABLES_DIR / "reviewer_safe_remaining_models_monthly_aggregated_metrics.csv"
PREDICTION_OUT = PREDICTIONS_DIR / "reviewer_safe_remaining_models_predictions.csv.gz"
MANIFEST_OUT = TABLES_DIR / "reviewer_safe_remaining_models_run_manifest.csv"
COMPLETION_OUT = TABLES_DIR / "reviewer_safe_model_completion_status.csv"


def expected_rows():
    full = len(LAND_TYPES) * len(VARIABLES) * 4
    gndvi_extra = len(LAND_TYPES) * 4 * 2
    return full * 3 + (full + gndvi_extra) * 3


def completion_table(pixel, manifest):
    rows = []
    for model in MODELS:
        model_pixel = pixel[pixel["model"] == model]
        seed42 = model_pixel[pd.to_numeric(model_pixel["seed"], errors="coerce") == 42]
        completed_tasks = seed42[
            ["land_type", "variable", "fold"]
        ].drop_duplicates().shape[0]
        failed = manifest[
            (manifest["model"] == model) & (manifest["status"] == "failed")
        ]
        rows.append(
            {
                "model": model,
                "required_seed42_tasks": 144,
                "completed_seed42_tasks": completed_tasks,
                "full_rolling_origin_complete": completed_tasks == 144,
                "seeds_present": ";".join(
                    map(
                        str,
                        sorted(
                            pd.to_numeric(
                                model_pixel["seed"], errors="coerce"
                            ).dropna().astype(int).unique()
                        ),
                    )
                ),
                "gndvi_three_seed_complete": (
                    model_pixel.query("variable == 'GNDVI'")[
                        ["land_type", "fold", "seed"]
                    ].drop_duplicates().shape[0]
                    == 48
                ),
                "failed_runs": len(failed),
                "failure_reason": "; ".join(
                    sorted(set(failed["error"].dropna().astype(str)))
                ),
                "status": (
                    "full"
                    if completed_tasks == 144
                    else "partial"
                    if completed_tasks > 0
                    else "failed"
                ),
            }
        )
    return pd.DataFrame(rows)


def save(pixel_frames, monthly_frames, prediction_frames, manifest_frames):
    pixel = pd.concat(pixel_frames, ignore_index=True) if pixel_frames else pd.DataFrame()
    monthly = (
        pd.concat(monthly_frames, ignore_index=True) if monthly_frames else pd.DataFrame()
    )
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    manifest = (
        pd.concat(manifest_frames, ignore_index=True)
        if manifest_frames
        else pd.DataFrame()
    )
    pixel.to_csv(PIXEL_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    predictions.to_csv(PREDICTION_OUT, index=False, compression="gzip")
    manifest.to_csv(MANIFEST_OUT, index=False)
    completion_table(pixel, manifest).to_csv(COMPLETION_OUT, index=False)


def run_block(model, variables, seed, log):
    temp_log = LOGS_DIR / f"_remaining_{model.replace('-', '').lower()}_{seed}.log"
    outputs = run_torch_tasks(
        model_names=[model],
        land_types=LAND_TYPES,
        variables=variables,
        seeds_for_task=lambda name, land, variable, spec, value=seed: [value],
        include_mode_b_for_model=lambda name, land, variable: True,
        log_path=temp_log,
        include_mode_a=False,
    )
    pixel, monthly, predictions, manifest = outputs
    for frame in [pixel, monthly, predictions, manifest]:
        if "mode" in frame:
            frame["mode"] = "reviewer_safe_rolling_origin"
    log.write(temp_log.read_text(encoding="utf-8"))
    temp_log.unlink(missing_ok=True)
    return pixel, monthly, predictions, manifest


def main():
    log_path = LOGS_DIR / "reviewer_safe_remaining_models_log.txt"
    if PIXEL_OUT.exists() and MONTHLY_OUT.exists() and COMPLETION_OUT.exists():
        completion = pd.read_csv(COMPLETION_OUT)
        completion = completion[completion["model"].isin(MODELS)]
        existing_pixel = pd.read_csv(PIXEL_OUT)
        if (
            len(completion) == 4
            and completion["full_rolling_origin_complete"].astype(bool).all()
            and len(existing_pixel) >= expected_rows()
        ):
            log_path.write_text(
                "Existing full seed-42 rolling-origin outputs for all four remaining models "
                "verified; rerun skipped.\n",
                encoding="utf-8",
            )
            return

    pixel_frames = []
    monthly_frames = []
    prediction_frames = []
    manifest_frames = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started: {datetime.now().isoformat()}\n")
        plan = []
        for seed in SEEDS:
            plan.append(("N-BEATS", VARIABLES, seed))
        for model in ["BiLSTM", "TCN", "N-HiTS"]:
            plan.append((model, VARIABLES, 42))
            plan.append((model, ["GNDVI"], 123))
            plan.append((model, ["GNDVI"], 2026))
        for model, variables, seed in plan:
            try:
                pixel, monthly, predictions, manifest = run_block(
                    model, variables, seed, log
                )
                pixel_frames.append(pixel)
                monthly_frames.append(monthly)
                prediction_frames.append(predictions)
                manifest_frames.append(manifest)
                save(
                    pixel_frames,
                    monthly_frames,
                    prediction_frames,
                    manifest_frames,
                )
                log.write(
                    f"Checkpoint saved: model={model}, seed={seed}, "
                    f"variables={len(variables)}\n"
                )
            except Exception as exc:
                log.write(
                    f"BLOCK FAILED model={model}, seed={seed}: "
                    f"{type(exc).__name__}: {exc}\n"
                )
            log.flush()
        save(pixel_frames, monthly_frames, prediction_frames, manifest_frames)
        log.write(f"Finished: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
