from __future__ import annotations

import os
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


HERE = Path(__file__).resolve().parent
DP = HERE.parents[2]
SOURCE = (
    DP
    / "Docs"
    / "major_revision_analyses"
    / "corrected_true_BiLSTM_rerun_20260728"
    / "08_downstream"
    / "final_corrected_Figure_7_source_table.csv"
)
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
METRICS = {
    "r2": ("mean_r2", True),
    "nrmse": ("mean_normalized_rmse", False),
}
BOOTSTRAPS = 10_000
RANDOM_SEED = 20261013


def holm_adjust(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted_sorted = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def rank_biserial(oriented_differences: np.ndarray) -> float:
    diff = np.asarray(oriented_differences, dtype=float)
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if not len(diff):
        return 0.0
    ranks = rankdata(np.abs(diff), method="average")
    positive = ranks[diff > 0].sum()
    negative = ranks[diff < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    boot = values[indices].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE)
    source = source[source["model"].isin(MODELS)].copy()
    source["task_id"] = source["target"].astype(str) + "__" + source["land_type"].astype(str)

    expected_tasks = set(source["task_id"])
    assert len(expected_tasks) == 36, f"Expected 36 target x land-type tasks, found {len(expected_tasks)}"
    counts = source.groupby(["task_id", "model"]).size()
    assert counts.eq(1).all(), "Model-task cells are not unique"
    assert source.groupby("task_id")["model"].nunique().eq(len(MODELS)).all(), "Incomplete paired model matrix"

    task_metrics = source[
        [
            "task_id",
            "target",
            "land_type",
            "model",
            "mean_r2",
            "mean_normalized_rmse",
            "sd_r2",
            "sd_normalized_rmse",
            "available_folds",
            "available_rows",
            "available_seeds",
            "data_status",
            "source_metric_table",
        ]
    ].rename(columns={"mean_normalized_rmse": "mean_nrmse"})
    task_metrics.to_csv(HERE / "forecast_model_task_metrics_corrected.csv", index=False)

    friedman_rows = []
    pairwise_rows = []
    effect_rows = []
    rank_rows = []
    rng = np.random.default_rng(RANDOM_SEED)

    for metric, (column, higher_is_better) in METRICS.items():
        matrix = source.pivot(index="task_id", columns="model", values=column)[MODELS]
        statistic, p_value = friedmanchisquare(*(matrix[m].to_numpy() for m in MODELS))
        friedman_rows.append(
            {
                "metric": metric,
                "n_paired_tasks": len(matrix),
                "n_models": len(MODELS),
                "friedman_chi_square": statistic,
                "degrees_of_freedom": len(MODELS) - 1,
                "p_value": p_value,
                "kendalls_W": statistic / (len(matrix) * (len(MODELS) - 1)),
                "posthoc_performed": bool(p_value < 0.05),
            }
        )

        if higher_is_better:
            ranks = matrix.rank(axis=1, ascending=False, method="average")
            winners = matrix.eq(matrix.max(axis=1), axis=0)
        else:
            ranks = matrix.rank(axis=1, ascending=True, method="average")
            winners = matrix.eq(matrix.min(axis=1), axis=0)

        for model in MODELS:
            values = matrix[model]
            rank_rows.append(
                {
                    "metric": metric,
                    "model": model,
                    "n_tasks": len(values),
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "median": values.median(),
                    "q25": values.quantile(0.25),
                    "q75": values.quantile(0.75),
                    "minimum": values.min(),
                    "maximum": values.max(),
                    "mean_rank": ranks[model].mean(),
                    "median_rank": ranks[model].median(),
                    "task_win_count": int(winners[model].sum()),
                }
            )

        metric_pairs = []
        for model_a, model_b in combinations(MODELS, 2):
            raw_difference = matrix[model_a].to_numpy() - matrix[model_b].to_numpy()
            oriented = raw_difference if higher_is_better else -raw_difference
            try:
                w = wilcoxon(oriented, alternative="two-sided", zero_method="wilcox", method="auto")
                statistic_w, raw_p = float(w.statistic), float(w.pvalue)
            except ValueError:
                statistic_w, raw_p = 0.0, 1.0
            metric_pairs.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired_tasks": len(oriented),
                    "wilcoxon_W": statistic_w,
                    "p_value_raw": raw_p,
                    "direction_definition": "positive effect favors model_a",
                }
            )
            ci_low, ci_high = paired_bootstrap_ci(raw_difference, rng)
            effect_rows.append(
                {
                    "metric": metric,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_paired_tasks": len(raw_difference),
                    "mean_paired_difference_a_minus_b": raw_difference.mean(),
                    "median_paired_difference_a_minus_b": np.median(raw_difference),
                    "mean_difference_bootstrap_ci95_low": ci_low,
                    "mean_difference_bootstrap_ci95_high": ci_high,
                    "rank_biserial_effect_positive_favors_a": rank_biserial(oriented),
                    "tasks_favoring_a": int(np.sum(oriented > 0)),
                    "tasks_tied": int(np.sum(oriented == 0)),
                    "tasks_favoring_b": int(np.sum(oriented < 0)),
                }
            )
        adjusted = holm_adjust([row["p_value_raw"] for row in metric_pairs])
        for row, adj in zip(metric_pairs, adjusted):
            row["p_value_holm"] = adj
            row["significant_holm_0_05"] = bool(adj < 0.05 and p_value < 0.05)
        pairwise_rows.extend(metric_pairs)

    friedman = pd.DataFrame(friedman_rows)
    pairwise = pd.DataFrame(pairwise_rows)
    effects = pd.DataFrame(effect_rows)
    pairwise = pairwise.merge(
        effects[
            [
                "metric",
                "model_a",
                "model_b",
                "rank_biserial_effect_positive_favors_a",
                "mean_paired_difference_a_minus_b",
            ]
        ],
        on=["metric", "model_a", "model_b"],
        how="left",
        validate="one_to_one",
    )
    pairwise["favored_model_by_rank_biserial"] = np.where(
        pairwise["rank_biserial_effect_positive_favors_a"] > 0,
        pairwise["model_a"],
        np.where(
            pairwise["rank_biserial_effect_positive_favors_a"] < 0,
            pairwise["model_b"],
            "tie",
        ),
    )
    ranks = pd.DataFrame(rank_rows).sort_values(["metric", "mean_rank", "model"])
    friedman.to_csv(HERE / "forecast_friedman_tests.csv", index=False)
    pairwise.to_csv(HERE / "forecast_pairwise_tests_holm.csv", index=False)
    effects.to_csv(HERE / "forecast_effect_sizes.csv", index=False)
    ranks.to_csv(HERE / "forecast_model_rank_summary.csv", index=False)

    lines = [
        "# Forecast statistical inference",
        "",
        "## Inferential unit",
        "",
        "The paired inferential unit is **target × land type** (9 targets × 4 land types = 36 tasks). "
        "Each cell is the corrected Figure-7 mean over its available rolling-origin folds and seeds. "
        "All six learned models are paired on the same 36 tasks. Individual pixels were not used as inferential replicates: "
        "pixels are repeated observations within a fitted land-type/target task and are spatially and temporally dependent, so treating them as independent would be pseudoreplication.",
        "",
        "The comparison covers the six learned models in the corrected TRUE-BiLSTM Figure-7 branch. Baselines and SVR are not included in this omnibus test because they are not all represented in that exact corrected Figure-7 task table, and the stored SVR branch uses different preprocessing/data for GNDVI and SPEI.",
        "",
        "## Methods",
        "",
        "For R² (larger is better) and nRMSE (smaller is better), the analysis used a Friedman omnibus test across the six paired models. "
        "When the omnibus test was significant, all paired Wilcoxon signed-rank contrasts were evaluated with Holm family-wise correction. "
        "Effect size is the matched-pairs rank-biserial correlation, oriented so positive values favor model A. "
        f"Mean paired differences have fixed-seed ({RANDOM_SEED}) percentile bootstrap 95% confidence intervals over the 36 task units ({BOOTSTRAPS:,} resamples).",
        "",
        "## Results",
        "",
    ]
    for row in friedman.itertuples(index=False):
        lines.append(
            f"- **{row.metric}**: Friedman χ²({int(row.degrees_of_freedom)}) = {row.friedman_chi_square:.3f}, "
            f"p = {row.p_value:.4g}, Kendall's W = {row.kendalls_W:.3f}."
        )
        significant = pairwise[(pairwise["metric"].eq(row.metric)) & pairwise["significant_holm_0_05"]]
        if significant.empty:
            lines.append("  No pair remained statistically distinguishable after Holm correction.")
        else:
            labels = [
                f"{r.favored_model_by_rank_biserial} favored over "
                f"{r.model_b if r.favored_model_by_rank_biserial == r.model_a else r.model_a} "
                f"(Holm p={r.p_value_holm:.4g}, rank-biserial={r.rank_biserial_effect_positive_favors_a:.3f} "
                f"for {r.model_a} vs {r.model_b})"
                for r in significant.itertuples(index=False)
            ]
            lines.append("  Holm-significant pairs: " + "; ".join(labels) + ".")
    lines += [
        "",
        "## Interpretation guardrail",
        "",
        "Task wins and small mean differences are descriptive. They are not treated as evidence of superiority unless the paired contrast remains significant after Holm correction and its paired-difference interval/effect size is considered. Results do not imply causal model superiority outside these stored out-of-sample tasks.",
        "",
        f"Source: `{SOURCE}`",
    ]
    (HERE / "FORECAST_STATISTICAL_INFERENCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(friedman.to_string(index=False))
    print("\nHolm-significant comparisons:")
    print(pairwise[pairwise["significant_holm_0_05"]].to_string(index=False))
    print("\nMean-rank summary:")
    print(ranks[["metric", "model", "mean_rank", "task_win_count"]].to_string(index=False))


if __name__ == "__main__":
    main()
