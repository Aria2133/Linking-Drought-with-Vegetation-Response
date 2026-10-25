"""GPP 2022 fix: rerun the common reviewer-safe benchmark for target GPP, folds 3-4 only, with the corrected cache.

Same code, settings, seeds and preprocessing boundary (reviewer_mode_b, fitted through validation end) as the published
runs; only the 12 GPP months of 2022 in the cache differ. Folds 1-2 never read 2022 and are not rerun.

  python run_gpp_2022_fix.py --models XGBoost --seeds 123 --lands GRASSLANDS --folds 3 --tag repro   (with FORECAST_CACHE
         pointing at the original cache: reproduction check against the published rows)
  python run_gpp_2022_fix.py --models baselines|XGBoost|LSTM|N-BEATS|TCN|N-HiTS --seeds 42 123 2026 --tag fix
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd  # noqa: E402

import candidate_config  # noqa: E402
import run_helpers  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--lands", nargs="+", default=candidate_config.LAND_TYPES)
    ap.add_argument("--folds", nargs="+", type=int, default=[3, 4])
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    run_helpers.ROLLING_FOLDS = [f for f in candidate_config.ROLLING_FOLDS if f["fold"] in args.folds]
    out = candidate_config.TABLES_DIR / args.tag
    out.mkdir(parents=True, exist_ok=True)
    log = candidate_config.LOGS_DIR / f"{args.tag}_{'_'.join(args.models)}_{'_'.join(map(str, args.seeds))}.log"

    if args.models == ["baselines"]:
        import importlib

        base = importlib.import_module("02_run_reviewer_safe_baselines")
        base.VARIABLES, base.LAND_TYPES = ["GPP"], args.lands
        base.PIXEL_OUT = out / "baseline_pixel_metrics.csv"
        base.MONTHLY_OUT = out / "baseline_monthly_metrics.csv"
        base.PREDICTION_OUT = out / "baseline_predictions.csv.gz"
        base.LOGS_DIR = candidate_config.LOGS_DIR
        base.main()
        return

    frames = {"pixel": [], "monthly": [], "pred": [], "manifest": []}
    for model in args.models:
        for seed in args.seeds:
            seeds = lambda m, l, v, s, value=seed: [value]  # noqa: E731
            if model == "XGBoost":
                res = run_helpers.run_xgboost_tasks(land_types=args.lands, variables=["GPP"], seeds_for_task=seeds,
                                                    include_mode_b=True, log_path=log, include_mode_a=False)
            else:
                res = run_helpers.run_torch_tasks(model_names=[model], land_types=args.lands, variables=["GPP"],
                                                  seeds_for_task=seeds, include_mode_b_for_model=lambda *a: True,
                                                  log_path=log, include_mode_a=False)
            for key, frame in zip(frames, res):
                if "mode" in frame:
                    frame["mode"] = "reviewer_safe_rolling_origin"
                frames[key].append(frame)
            name = f"{'_'.join(args.models)}_{'_'.join(map(str, args.seeds))}"
            pd.concat(frames["pixel"]).to_csv(out / f"{name}_pixel_metrics.csv", index=False)
            pd.concat(frames["monthly"]).to_csv(out / f"{name}_monthly_metrics.csv", index=False)
            pd.concat(frames["pred"]).to_csv(out / f"{name}_predictions.csv.gz", index=False, compression="gzip")
            pd.concat(frames["manifest"]).to_csv(out / f"{name}_manifest.csv", index=False)
            print(f"done {model} seed {seed}", flush=True)


if __name__ == "__main__":
    main()
