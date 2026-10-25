"""GPP 2022 fix: corrected true-BiLSTM (128 units, lr 0.001, seed 42) for GPP folds 3-4 with the corrected cache.
Same functions as the published corrected_true_BiLSTM rerun; FORECAST_CACHE may point at the original cache for a
reproduction check (--tag repro)."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import bilstm_module as B

ap = argparse.ArgumentParser()
ap.add_argument("--folds", nargs="+", type=int, default=[3, 4])
ap.add_argument("--lands", nargs="+", default=B.LAND_TYPES)
ap.add_argument("--tag", required=True)
args = ap.parse_args()
out = B.OUT / args.tag
out.mkdir(parents=True, exist_ok=True)
pix, mon = [], []
for land in args.lands:
    for fold in args.folds:
        prepared, train, validation, test, spec = B.samples_for_task("GPP", land, fold)
        fit = B.fit_corrected_bilstm(train, validation, 42, units=128, learning_rate=0.001)
        predicted = B.predict_model(fit["model"], test)
        frame = B.prediction_frame("GPP", land, fold, 42, spec, prepared, test, predicted)
        frame.to_csv(out / f"GPP_{land}_fold{fold}_BiLSTM_seed42_predictions.csv.gz", index=False, compression="gzip")
        info = {"status": "success", "error": "", "epochs_run": fit["epochs_run"],
                "best_validation_loss": fit["best_validation_loss"], "device": fit["device"],
                "architecture": B.architecture_string(128), "learning_rate": 0.001, "reuse_status": "gpp_2022_fix"}
        common = B.common_metric_fields("GPP", land, fold, 42, spec, prepared, info)
        p, m = B.mc.metric_rows(frame, common)
        pix.append(p); mon.append(m)
        pd.DataFrame(pix).to_csv(out / "BiLSTM_pixel_metrics.csv", index=False)
        pd.DataFrame(mon).to_csv(out / "BiLSTM_monthly_metrics.csv", index=False)
        print(f"done {land} fold {fold} r2={p['r2']:.6f}", flush=True)
