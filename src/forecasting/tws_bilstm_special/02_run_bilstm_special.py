import json
from datetime import datetime

import pandas as pd

from candidate_config import (
    BILSTM_CONFIG,
    LAND_TYPES,
    LOGS_DIR,
    PREDICTIONS_DIR,
    SEEDS,
    TABLES_DIR,
    TARGETS,
)
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


PIXEL_PATH = TABLES_DIR / "hydroclimate_bilstm_special_pixel_metrics.csv"
MONTHLY_PATH = TABLES_DIR / "hydroclimate_bilstm_special_monthly_metrics.csv"
MANIFEST_PATH = TABLES_DIR / "hydroclimate_bilstm_special_run_manifest.csv"
PREDICTION_PATH = (
    PREDICTIONS_DIR / "hydroclimate_bilstm_special_predictions.csv.gz"
)


def checkpoint(pixel, monthly, predictions, manifest):
    pd.DataFrame(pixel).to_csv(PIXEL_PATH, index=False)
    pd.DataFrame(monthly).to_csv(MONTHLY_PATH, index=False)
    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False)
    write_prediction_csv_gz(predictions, PREDICTION_PATH)


def main():
    pixel_rows, monthly_rows, predictions, manifest = [], [], [], []
    log_path = LOGS_DIR / "hydroclimate_bilstm_special_log.txt"
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Started {datetime.now().isoformat()}\n")
        log.write(
            "Framework: PyTorch translation used by the reviewer-safe package. "
            "Original Keras widths, directionality, dropout, and lookbacks retained; "
            "CPU runtime cap is 30 epochs with fold-local early stopping.\n"
        )
        for variable in TARGETS:
            cfg = BILSTM_CONFIG[variable]
            gap_options = (
                ["available_observations", "training_only_state_space"]
                if variable == "TWS"
                else ["available_observations"]
            )
            for land_type in LAND_TYPES:
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
                                identity = (
                                    f"BiLSTM/{configuration}/{land_type}/{variable}/"
                                    f"fold{spec['fold']}/seed{seed}"
                                )
                                started = datetime.now()
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
                            checkpoint(pixel_rows, monthly_rows, predictions, manifest)
                            continue
                        for seed in SEEDS:
                            started = datetime.now()
                            identity = (
                                f"BiLSTM/{configuration}/{land_type}/{variable}/"
                                f"fold{spec['fold']}/seed{seed}"
                            )
                            try:
                                fit = fit_bilstm_special(
                                    variable, train, validation, seed
                                )
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
                                        "architecture": json.dumps(
                                            cfg, sort_keys=True
                                        ),
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
                        checkpoint(pixel_rows, monthly_rows, predictions, manifest)
        log.write(f"Finished {datetime.now().isoformat()}\n")
    checkpoint(pixel_rows, monthly_rows, predictions, manifest)


if __name__ == "__main__":
    main()
