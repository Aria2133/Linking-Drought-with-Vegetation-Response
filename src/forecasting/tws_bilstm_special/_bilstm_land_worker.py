import json
import sys
from datetime import datetime

import pandas as pd

from candidate_config import BILSTM_CONFIG, LOGS_DIR, OUTPUT_CACHE_DIR, SEEDS
from hydro_core import (
    common_metric_fields,
    fit_bilstm_special,
    fold_specs,
    log_failure,
    metric_pair,
    predict_torch,
    prediction_frame,
    run_manifest_row,
    samples_for_spec,
    write_prediction_csv_gz,
)


def main():
    land_type = sys.argv[1]
    stem = f"bilstm_worker_{land_type.lower()}"
    pixel_rows, monthly_rows, predictions, manifest = [], [], [], []
    with (LOGS_DIR / f"{stem}.log").open("w", encoding="utf-8") as log:
        log.write(f"Started {datetime.now().isoformat()}\n")
        for variable in ["TWS", "VPD"]:
            cfg = BILSTM_CONFIG[variable]
            gap_options = (
                ["available_observations", "training_only_state_space"]
                if variable == "TWS"
                else ["available_observations"]
            )
            for spec in fold_specs():
                for gap_handling in gap_options:
                    configuration = (
                        "bilstm_special_tws_state_space"
                        if gap_handling == "training_only_state_space"
                        else "bilstm_special"
                    )
                    try:
                        prepared, train, validation, test = samples_for_spec(
                            land_type,
                            variable,
                            spec,
                            cfg["lookback"],
                            "global_minmax",
                            gap_handling,
                        )
                    except Exception as exc:
                        for seed in SEEDS:
                            started = datetime.now()
                            identity = (
                                f"BiLSTM/{configuration}/{land_type}/{variable}/"
                                f"fold{spec['fold']}/seed{seed}"
                            )
                            error = log_failure(log, identity, exc)
                            manifest.append(
                                run_manifest_row(
                                    "BiLSTM",
                                    configuration,
                                    land_type,
                                    variable,
                                    seed,
                                    spec,
                                    "failed",
                                    error,
                                    started,
                                )
                            )
                        continue
                    for seed in SEEDS:
                        started = datetime.now()
                        identity = (
                            f"BiLSTM/{configuration}/{land_type}/{variable}/"
                            f"fold{spec['fold']}/seed{seed}"
                        )
                        try:
                            fit = fit_bilstm_special(variable, train, validation, seed)
                            predicted = predict_torch(fit.model, test)
                            frame = prediction_frame(
                                "BiLSTM",
                                configuration,
                                land_type,
                                variable,
                                seed,
                                spec,
                                prepared,
                                test,
                                predicted,
                                cfg["lookback"],
                            )
                            common = common_metric_fields(
                                "BiLSTM",
                                configuration,
                                land_type,
                                variable,
                                seed,
                                spec,
                                prepared,
                                cfg["lookback"],
                                {
                                    "status": "success",
                                    "epochs_run": fit.epochs_run,
                                    "best_validation_loss": fit.best_validation_loss,
                                    "device": fit.device,
                                    "architecture": json.dumps(cfg, sort_keys=True),
                                    "framework_note": (
                                        "PyTorch LSTM translation; original Keras "
                                        "used activation='relu'"
                                    ),
                                },
                            )
                            pixel, monthly = metric_pair(frame, common)
                            pixel_rows.append(pixel)
                            monthly_rows.append(monthly)
                            predictions.append(frame)
                            manifest.append(
                                run_manifest_row(
                                    "BiLSTM",
                                    configuration,
                                    land_type,
                                    variable,
                                    seed,
                                    spec,
                                    "success",
                                    "",
                                    started,
                                )
                            )
                            log.write(
                                f"SUCCESS {identity} epochs={fit.epochs_run} "
                                f"pixel_r2={pixel['r2']:.6f} "
                                f"monthly_r2={monthly['r2']:.6f}\n"
                            )
                            log.flush()
                        except Exception as exc:
                            error = log_failure(log, identity, exc)
                            manifest.append(
                                run_manifest_row(
                                    "BiLSTM",
                                    configuration,
                                    land_type,
                                    variable,
                                    seed,
                                    spec,
                                    "failed",
                                    error,
                                    started,
                                )
                            )
        pd.DataFrame(pixel_rows).to_csv(
            OUTPUT_CACHE_DIR / f"{stem}_pixel.csv", index=False
        )
        pd.DataFrame(monthly_rows).to_csv(
            OUTPUT_CACHE_DIR / f"{stem}_monthly.csv", index=False
        )
        pd.DataFrame(manifest).to_csv(
            OUTPUT_CACHE_DIR / f"{stem}_manifest.csv", index=False
        )
        write_prediction_csv_gz(
            predictions, OUTPUT_CACHE_DIR / f"{stem}_predictions.csv.gz"
        )
        log.write(f"Finished {datetime.now().isoformat()}\n")


if __name__ == "__main__":
    main()
