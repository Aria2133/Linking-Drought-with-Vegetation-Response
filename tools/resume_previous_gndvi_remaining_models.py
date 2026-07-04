"""Resume missing GNDVI-only remaining-model blocks without overwriting completed work."""

from pathlib import Path

PACKAGE = Path(r"C:\Users\LOQ\OneDrive\Desktop\DL_GNDVI_previous_noQA_rerun")
SCRIPT_DIR = PACKAGE / "scripts"

RESUME = r'''
from datetime import datetime
from pathlib import Path

import pandas as pd

from candidate_config import LOGS_DIR, PREDICTIONS_DIR, SEEDS, TABLES_DIR, VARIABLES
from run_helpers import run_torch_tasks
from 05_run_reviewer_safe_remaining_models import save  # type: ignore
'''

# Python cannot import a module whose filename starts with a digit via normal
# syntax, so write a self-contained script instead.
RESUME = r'''
from datetime import datetime

import pandas as pd

from candidate_config import LOGS_DIR, PREDICTIONS_DIR, SEEDS, TABLES_DIR, VARIABLES
from run_helpers import run_torch_tasks

MODELS = ["N-BEATS", "BiLSTM", "TCN", "N-HiTS"]
PIXEL_OUT = TABLES_DIR / "reviewer_safe_remaining_models_pixel_metrics.csv"
MONTHLY_OUT = TABLES_DIR / "reviewer_safe_remaining_models_monthly_aggregated_metrics.csv"
PREDICTION_OUT = PREDICTIONS_DIR / "reviewer_safe_remaining_models_predictions.csv.gz"
MANIFEST_OUT = TABLES_DIR / "reviewer_safe_remaining_models_run_manifest.csv"
COMPLETION_OUT = TABLES_DIR / "reviewer_safe_model_completion_status.csv"


def completion_table(pixel, manifest):
    rows = []
    for model in MODELS:
        model_pixel = pixel[pixel["model"] == model]
        seed42 = model_pixel[pd.to_numeric(model_pixel["seed"], errors="coerce") == 42]
        completed_tasks = seed42[["land_type", "variable", "fold"]].drop_duplicates().shape[0]
        failed = manifest[(manifest["model"] == model) & (manifest["status"] == "failed")]
        rows.append(
            {
                "model": model,
                "required_seed42_tasks": 16,
                "completed_seed42_tasks": completed_tasks,
                "full_rolling_origin_complete": completed_tasks == 16,
                "seeds_present": ";".join(
                    map(str, sorted(pd.to_numeric(model_pixel["seed"], errors="coerce").dropna().astype(int).unique()))
                ),
                "gndvi_three_seed_complete": (
                    model_pixel.query("variable == 'GNDVI'")[["land_type", "fold", "seed"]]
                    .drop_duplicates()
                    .shape[0]
                    == 48
                ),
                "failed_runs": len(failed),
                "failure_reason": "; ".join(sorted(set(failed["error"].dropna().astype(str)))),
                "status": "full" if model_pixel[["land_type", "fold", "seed"]].drop_duplicates().shape[0] == 48 else "partial",
            }
        )
    return pd.DataFrame(rows)


def save_all(pixel, monthly, predictions, manifest):
    pixel.to_csv(PIXEL_OUT, index=False)
    monthly.to_csv(MONTHLY_OUT, index=False)
    predictions.to_csv(PREDICTION_OUT, index=False, compression="gzip")
    manifest.to_csv(MANIFEST_OUT, index=False)
    completion_table(pixel, manifest).to_csv(COMPLETION_OUT, index=False)


def run_block(model, variables, seed, log):
    temp_log = LOGS_DIR / f"_resume_{model.replace('-', '').lower()}_{seed}.log"
    outputs = run_torch_tasks(
        model_names=[model],
        land_types=["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"],
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


def block_complete(pixel, model, seed):
    sub = pixel[
        (pixel["model"] == model)
        & (pd.to_numeric(pixel["seed"], errors="coerce") == seed)
        & (pixel["variable"] == "GNDVI")
    ]
    return sub[["land_type", "fold"]].drop_duplicates().shape[0] == 16


def main():
    log_path = LOGS_DIR / "reviewer_safe_remaining_models_resume_log.txt"
    pixel = pd.read_csv(PIXEL_OUT) if PIXEL_OUT.exists() else pd.DataFrame()
    monthly = pd.read_csv(MONTHLY_OUT) if MONTHLY_OUT.exists() else pd.DataFrame()
    predictions = pd.read_csv(PREDICTION_OUT) if PREDICTION_OUT.exists() else pd.DataFrame()
    manifest = pd.read_csv(MANIFEST_OUT) if MANIFEST_OUT.exists() else pd.DataFrame()

    plan = []
    for seed in SEEDS:
        plan.append(("N-BEATS", VARIABLES, seed))
    for model in ["BiLSTM", "TCN", "N-HiTS"]:
        plan.append((model, VARIABLES, 42))
        plan.append((model, ["GNDVI"], 123))
        plan.append((model, ["GNDVI"], 2026))

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Resume started: {datetime.now().isoformat()}\n")
        for model, variables, seed in plan:
            if block_complete(pixel, model, seed):
                log.write(f"SKIP complete model={model} seed={seed}\n")
                continue
            log.write(f"RUN missing model={model} seed={seed}\n")
            new_pixel, new_monthly, new_predictions, new_manifest = run_block(model, variables, seed, log)
            pixel = pd.concat([pixel, new_pixel], ignore_index=True)
            monthly = pd.concat([monthly, new_monthly], ignore_index=True)
            predictions = pd.concat([predictions, new_predictions], ignore_index=True)
            manifest = pd.concat([manifest, new_manifest], ignore_index=True)
            save_all(pixel, monthly, predictions, manifest)
            log.write(f"Checkpoint saved model={model} seed={seed}\n")
            log.flush()
        save_all(pixel, monthly, predictions, manifest)
        log.write(f"Resume finished: {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    (SCRIPT_DIR / "05b_resume_remaining_models_previous_gndvi.py").write_text(RESUME, encoding="utf-8")


if __name__ == "__main__":
    main()
