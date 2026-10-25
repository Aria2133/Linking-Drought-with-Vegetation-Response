"""Formal comparison of the six learned forecasting models.

Inferential unit: the 36 target x land-type tasks (paired across models). Each task value is the published
task mean of the out-of-sample metric (mean over the rolling-origin fold x seed runs), taken from
results/forecasting/figure7_final_task_metrics.csv (the source table of the current Figure 7).
Pixels are never used as replicates: each run's metric is already computed on all pixel-months of the task.

Procedure, separately for R2 and nRMSE:
  1. Friedman test across the six models (36 blocks).
  2. Only if Friedman p < 0.05: two-sided Wilcoxon signed-rank tests for all 15 model pairs.
  3. Holm correction across those 15 comparisons (per metric).
Outputs: results/forecasting/forecast_friedman_tests.csv, results/forecasting/forecast_pairwise_wilcoxon_holm.csv

Usage:  python src/statistics/forecast_model_comparison.py
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results" / "forecasting" / "figure7_final_task_metrics.csv"
OUT = ROOT / "results" / "forecasting"
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
METRICS = {"R2": ("mean_r2", True), "nRMSE": ("mean_normalized_rmse", False)}  # (column, higher_is_better)
ALPHA = 0.05


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def compare_models(data: pd.DataFrame):
    """Friedman + Wilcoxon-Holm on a 216-row task table (columns target, land_type, model, mean_r2,
    mean_normalized_rmse). Returns (friedman DataFrame, pairwise DataFrame)."""
    data = data.copy()
    data["task"] = data["target"] + " | " + data["land_type"]
    friedman_rows, pair_rows = [], []
    for metric, (column, higher_better) in METRICS.items():
        wide = data.pivot(index="task", columns="model", values=column)[MODELS]
        if wide.shape != (36, 6) or wide.isna().any().any():
            raise ValueError(f"expected a complete 36 x 6 task matrix for {metric}, got {wide.shape}")
        chi2, p = stats.friedmanchisquare(*[wide[m].to_numpy() for m in MODELS])
        ranks = (wide if higher_better else -wide).rank(axis=1, ascending=False)
        friedman_rows.append({"metric": metric, "n_tasks": len(wide), "k_models": len(MODELS), "df": len(MODELS) - 1,
                              "chi2": chi2, "p_value": p, "kendall_W": chi2 / (len(wide) * (len(MODELS) - 1)),
                              **{f"mean_rank_{m}": ranks[m].mean() for m in MODELS}})
        if p >= ALPHA:
            continue
        rows = []
        for a, b in combinations(MODELS, 2):
            stat, pv = stats.wilcoxon(wide[a], wide[b], alternative="two-sided")
            diff = wide[a] - wide[b]
            better_a = (diff > 0) if higher_better else (diff < 0)
            rows.append({"metric": metric, "model_a": a, "model_b": b, "n_tasks": len(diff),
                         "median_difference_a_minus_b": float(diff.median()),
                         "tasks_a_better": int(better_a.sum()), "tasks_b_better": int((~better_a & (diff != 0)).sum()),
                         "wilcoxon_statistic": stat, "p_raw": pv})
        adj = holm([r["p_raw"] for r in rows])
        for r, a in zip(rows, adj):
            r["p_holm"] = a
            r["significant_holm_0.05"] = bool(a < ALPHA)
        pair_rows.extend(rows)
    return pd.DataFrame(friedman_rows), pd.DataFrame(pair_rows)


def main():
    data = pd.read_csv(SOURCE)
    # Provenance gate: only main common-benchmark configurations may enter the comparison.
    sys.path.insert(0, str(ROOT / "src" / "forecasting"))
    from provenance import check_task_table, load_registry  # noqa: E402

    check_task_table(data, load_registry(ROOT / "config" / "forecast_result_registry.csv"))
    friedman, pairs = compare_models(data)
    friedman_rows, pair_rows = friedman.to_dict("records"), pairs.to_dict("records")
    OUT.mkdir(parents=True, exist_ok=True)
    friedman.to_csv(OUT / "forecast_friedman_tests.csv", index=False)
    pairs.to_csv(OUT / "forecast_pairwise_wilcoxon_holm.csv", index=False)
    for r in friedman_rows:
        print(f"{r['metric']}: Friedman chi2({r['df']}) = {r['chi2']:.3f}, p = {r['p_value']:.3g}")
    sig = [r for r in pair_rows if r["significant_holm_0.05"]]
    print(f"Holm-significant pairs: {len(sig)} of {len(pair_rows)}")


if __name__ == "__main__":
    main()
