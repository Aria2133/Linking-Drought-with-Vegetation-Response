"""Validate candidate imputation methods by hiding observed TWS and LST months."""

from __future__ import annotations

import warnings
from itertools import groupby

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from gapfill_utils import (
    FOCUS_VARIABLES,
    LAND_FOLDERS,
    METHODS,
    ROOT,
    SEED,
    configure_logging,
    ensure_directories,
    fill_series,
    load_monthly_summary,
    series_for,
)


REPLICATES_PER_SCENARIO = 12


def gap_run_lengths(series: pd.Series) -> list[int]:
    missing_positions = np.flatnonzero(series.isna().to_numpy())
    if not len(missing_positions):
        return [1]
    lengths: list[int] = []
    for _, group in groupby(enumerate(missing_positions), lambda item: item[1] - item[0]):
        lengths.append(len(list(group)))
    return lengths


def contiguous_observed_windows(series: pd.Series, length: int) -> list[np.ndarray]:
    observed = series.notna().to_numpy()
    return [
        np.arange(start, start + length)
        for start in range(0, len(series) - length + 1)
        if bool(np.all(observed[start : start + length]))
    ]


def build_scenarios(series: pd.Series, rng: np.random.Generator) -> list[dict[str, object]]:
    scenarios: list[dict[str, object]] = []
    lengths = gap_run_lengths(series)
    for replicate in range(1, REPLICATES_PER_SCENARIO + 1):
        requested_length = int(rng.choice(lengths))
        length = requested_length
        windows = contiguous_observed_windows(series, length)
        while not windows and length > 1:
            length -= 1
            windows = contiguous_observed_windows(series, length)
        positions = windows[int(rng.integers(0, len(windows)))] if windows else np.array([], dtype=int)
        scenarios.append(
            {
                "scenario_type": "realistic_contiguous_gap",
                "replicate": replicate,
                "positions": positions,
                "requested_gap_length": requested_length,
                "actual_hidden_count": len(positions),
            }
        )

    observed_positions = np.flatnonzero(series.notna().to_numpy())
    random_count = max(12, int(round(0.10 * len(observed_positions))))
    for replicate in range(1, REPLICATES_PER_SCENARIO + 1):
        positions = np.sort(rng.choice(observed_positions, size=random_count, replace=False))
        scenarios.append(
            {
                "scenario_type": "random_observed_months",
                "replicate": replicate,
                "positions": positions,
                "requested_gap_length": random_count,
                "actual_hidden_count": len(positions),
            }
        )
    return scenarios


def safe_correlations(truth: pd.Series, prediction: pd.Series) -> tuple[float, float]:
    valid = truth.notna() & prediction.notna()
    if int(valid.sum()) < 3:
        return np.nan, np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pearson = float(pearsonr(truth[valid], prediction[valid]).statistic)
        spearman = float(spearmanr(truth[valid], prediction[valid]).statistic)
    return pearson, spearman


def aggregate_metrics(group: pd.DataFrame) -> pd.Series:
    valid = group.dropna(subset=["truth", "prediction"])
    error = valid["prediction"] - valid["truth"]
    truth_sd = float(valid["truth"].std(ddof=1))
    pearson, spearman = safe_correlations(valid["truth"], valid["prediction"])
    return pd.Series(
        {
            "rmse": float(np.sqrt(np.mean(error**2))) if len(valid) else np.nan,
            "mae": float(np.mean(np.abs(error))) if len(valid) else np.nan,
            "bias": float(np.mean(error)) if len(valid) else np.nan,
            "pearson_r": pearson,
            "spearman_r": spearman,
            "normalized_rmse": float(np.sqrt(np.mean(error**2)) / truth_sd)
            if truth_sd > 0
            else np.nan,
            "n_predictions": int(len(valid)),
            "failed_predictions": int(group["prediction"].isna().sum()),
            "unstable_predictions": int(group["prediction_unstable"].sum()),
            "mean_hidden_per_replicate": float(group.groupby("replicate").size().mean()),
        }
    )


def create_heatmap(metrics: pd.DataFrame) -> None:
    overall = metrics[metrics["scenario_type"] == "all"].copy()
    overall["column"] = overall["variable"] + "\n" + overall["land_type"].str.replace(
        "Natural Forests", "Forests", regex=False
    )
    pivot = overall.pivot(index="method", columns="column", values="normalized_rmse").reindex(METHODS)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), [name.replace("_", "\n") for name in pivot.index])
    ax.set_title("Artificial-gap validation: normalized RMSE (lower is better)")
    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.iloc[row, col]
            if np.isfinite(value):
                color = "white" if value > np.nanmedian(pivot.to_numpy()) else "black"
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(image, ax=ax, label="Normalized RMSE")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "gapfill_validation_metric_heatmap.png", dpi=220)
    plt.close(fig)


def create_boxplot(predictions: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=False)
    short_labels = ["Current", "Seasonal", "Linear", "Polynomial", "Spline", "State-space"]
    for ax, variable in zip(axes, FOCUS_VARIABLES):
        subset = predictions[predictions["variable"] == variable].copy()
        values = [
            subset.loc[subset["method"] == method, "absolute_error"].dropna().to_numpy()
            for method in METHODS
        ]
        ax.boxplot(values, tick_labels=short_labels, showfliers=False, patch_artist=True)
        ax.set_title(f"{variable}: absolute artificial-gap errors")
        ax.set_ylabel(f"Absolute error ({variable} units)")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Gap-filling validation across land types and gap scenarios")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "gapfill_validation_boxplot.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    logger = configure_logging(
        ROOT / "logs" / "gapfill_validation_log.txt", "gapfill_validation"
    )
    raw = load_monthly_summary()
    rng = np.random.default_rng(SEED)
    prediction_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        for variable in FOCUS_VARIABLES:
            original = series_for(raw, land_type, variable)
            scenarios = build_scenarios(original, rng)
            for scenario in scenarios:
                hidden = original.copy()
                positions = np.asarray(scenario["positions"], dtype=int)
                truth = original.iloc[positions].copy()
                hidden.iloc[positions] = np.nan
                for method in METHODS:
                    filled, diagnostics = fill_series(hidden, method)
                    observed_sd = float(original.dropna().std(ddof=1))
                    lower = float(original.min() - 3.0 * observed_sd)
                    upper = float(original.max() + 3.0 * observed_sd)
                    for position, true_value in zip(positions, truth.to_numpy()):
                        prediction = filled.iloc[position]
                        unstable = bool(
                            not np.isfinite(prediction) or prediction < lower or prediction > upper
                        )
                        prediction_rows.append(
                            {
                                "land_type": land_type,
                                "variable": variable,
                                "method": method,
                                "scenario_type": scenario["scenario_type"],
                                "replicate": scenario["replicate"],
                                "date": original.index[position],
                                "truth": true_value,
                                "prediction": prediction,
                                "error": prediction - true_value,
                                "absolute_error": abs(prediction - true_value),
                                "prediction_unstable": unstable,
                                "fit_warning": diagnostics["warning"],
                                "requested_gap_length": scenario["requested_gap_length"],
                                "actual_hidden_count": scenario["actual_hidden_count"],
                            }
                        )
            logger.info(
                "Validated %s %s using %d paired scenarios",
                land_type,
                variable,
                len(scenarios),
            )

    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(ROOT / "tables" / "gapfill_validation_predictions.csv", index=False)
    grouped = (
        predictions.groupby(
            ["land_type", "variable", "method", "scenario_type"], sort=False, dropna=False
        )
        .apply(aggregate_metrics, include_groups=False)
        .reset_index()
    )
    overall = (
        predictions.groupby(["land_type", "variable", "method"], sort=False, dropna=False)
        .apply(aggregate_metrics, include_groups=False)
        .reset_index()
    )
    overall.insert(3, "scenario_type", "all")
    metrics = pd.concat([overall, grouped], ignore_index=True)
    metrics["random_seed"] = SEED
    metrics["replicates_per_scenario"] = REPLICATES_PER_SCENARIO
    metrics.to_csv(ROOT / "tables" / "gapfill_validation_metrics.csv", index=False)
    create_heatmap(metrics)
    create_boxplot(predictions)

    ranking = (
        overall.groupby(["variable", "method"], as_index=False)["normalized_rmse"]
        .mean()
        .sort_values(["variable", "normalized_rmse"])
    )
    for variable in FOCUS_VARIABLES:
        best = ranking[ranking["variable"] == variable].iloc[0]
        logger.info(
            "Best mean normalized RMSE for %s: %s (%.4f)",
            variable,
            best["method"],
            best["normalized_rmse"],
        )
    logger.info("Validation used random seed %d", SEED)


if __name__ == "__main__":
    main()
