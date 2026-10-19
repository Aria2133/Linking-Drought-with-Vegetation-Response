"""Recalculate EWBI under each TWS/LST gap-filling method."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gapfill_utils import (
    DATE_INDEX,
    LAND_FOLDERS,
    METHODS,
    ROOT,
    baseline_monthly_zscore,
    comparison_metrics,
    configure_logging,
    ensure_directories,
    load_monthly_summary,
    period_mean_differences,
    series_for,
    trend_slope,
)


CURRENT = "current_monthwise_cross_year"


def load_filled_series() -> pd.DataFrame:
    return pd.read_csv(
        ROOT / "tables" / "gapfilled_monthly_series.csv", parse_dates=["date"]
    )


def filled_for(frame: pd.DataFrame, land_type: str, variable: str, method: str) -> pd.Series:
    subset = frame[
        (frame["land_type"] == land_type)
        & (frame["variable"] == variable)
        & (frame["method"] == method)
    ]
    return subset.set_index("date")["filled_value"].astype(float).reindex(DATE_INDEX)


def compute_ewbi(
    spei: pd.Series,
    tws: pd.Series,
    vpd: pd.Series,
    lst: pd.Series,
) -> pd.DataFrame:
    components = pd.DataFrame(
        {
            "spei_signed_z": baseline_monthly_zscore(spei),
            "tws_signed_z": baseline_monthly_zscore(tws),
            "vpd_signed_z": -baseline_monthly_zscore(vpd),
            "lst_signed_z": -baseline_monthly_zscore(lst),
        }
    )
    components["ewbi"] = components.mean(axis=1, skipna=False)
    return components


def stability_label(metrics: dict[str, float | int], period_diffs: dict[str, float]) -> str:
    max_period = max(
        abs(period_diffs["baseline_mean_difference"]),
        abs(period_diffs["historic_mean_difference"]),
        abs(period_diffs["recent_mean_difference"]),
    )
    stable = (
        metrics["pearson_r"] >= 0.95
        and metrics["spearman_r"] >= 0.95
        and metrics["normalized_rmse"] <= 0.25
        and 0.8 <= metrics["standard_deviation_ratio"] <= 1.2
        and max_period <= 0.25
    )
    return "stable" if stable else "potentially meaningful change"


def create_timeseries_figure(monthly: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(METHODS)))
    for ax, land_type in zip(axes.flat, LAND_FOLDERS):
        subset = monthly[monthly["land_type"] == land_type]
        for color, method in zip(colors, METHODS):
            method_data = subset[subset["method"] == method]
            ax.plot(
                method_data["date"],
                method_data["ewbi"],
                color="black" if method == CURRENT else color,
                linewidth=1.7 if method == CURRENT else 0.8,
                alpha=1.0 if method == CURRENT else 0.7,
                label=method.replace("_", " "),
            )
        ax.axhline(0, color="0.6", linewidth=0.7)
        ax.set_title(land_type)
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("EWBI sensitivity to TWS and LST gap-filling method")
    fig.supxlabel("Month")
    fig.supylabel("Equal-weight baseline-standardized EWBI")
    fig.tight_layout(rect=(0.02, 0.08, 1, 0.96))
    fig.savefig(ROOT / "figures" / "ewbi_gapfill_sensitivity_timeseries.png", dpi=220)
    plt.close(fig)


def create_heatmap(metrics: pd.DataFrame) -> None:
    alternatives = metrics[metrics["method"] != CURRENT].copy()
    alternatives["row"] = alternatives["land_type"].str.replace(
        "Natural Forests", "Forests", regex=False
    ) + "\n" + alternatives["method"].str.replace("_", " ", regex=False)
    selected = [
        "pearson_r",
        "spearman_r",
        "normalized_rmse",
        "standard_deviation_ratio",
        "max_absolute_period_mean_difference",
    ]
    matrix = alternatives.set_index("row")[selected]
    display = matrix.copy()
    display["pearson_r"] = 1.0 - display["pearson_r"]
    display["spearman_r"] = 1.0 - display["spearman_r"]
    display["standard_deviation_ratio"] = abs(display["standard_deviation_ratio"] - 1.0)
    fig, ax = plt.subplots(figsize=(10, 13))
    image = ax.imshow(display.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(
        range(len(selected)),
        ["1-Pearson", "1-Spearman", "NRMSE", "|SD ratio-1|", "Max period mean diff"],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(display.index)), display.index, fontsize=8)
    ax.set_title("EWBI divergence from current month-wise gap filling")
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    fig.colorbar(image, ax=ax, label="Divergence (lower is more stable)")
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "ewbi_gapfill_sensitivity_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    logger = configure_logging(
        ROOT / "logs" / "ewbi_gapfill_sensitivity_log.txt", "ewbi_gapfill"
    )
    raw = load_monthly_summary()
    filled = load_filled_series()
    monthly_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        spei = series_for(raw, land_type, "SPEI")
        vpd = series_for(raw, land_type, "VPD")
        if spei.isna().any() or vpd.isna().any():
            raise RuntimeError(f"SPEI or VPD unexpectedly incomplete for {land_type}")
        for method in METHODS:
            tws = filled_for(filled, land_type, "TWS", method)
            lst = filled_for(filled, land_type, "LST", method)
            components = compute_ewbi(spei, tws, vpd, lst)
            for date, row in components.iterrows():
                monthly_rows.append(
                    {
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "land_type": land_type,
                        "method": method,
                        **row.to_dict(),
                    }
                )
        logger.info("Calculated six EWBI variants for %s", land_type)

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(ROOT / "tables" / "ewbi_gapfill_monthly_series.csv", index=False)

    metric_rows: list[dict[str, object]] = []
    for land_type in LAND_FOLDERS:
        land = monthly[monthly["land_type"] == land_type]
        reference = (
            land[land["method"] == CURRENT].set_index("date")["ewbi"].reindex(DATE_INDEX)
        )
        for method in METHODS:
            comparison = land[land["method"] == method].set_index("date")["ewbi"].reindex(DATE_INDEX)
            metrics = comparison_metrics(reference, comparison)
            periods = period_mean_differences(reference, comparison)
            max_period = max(
                abs(periods["baseline_mean_difference"]),
                abs(periods["historic_mean_difference"]),
                abs(periods["recent_mean_difference"]),
            )
            row = {
                "land_type": land_type,
                "reference_method": CURRENT,
                "method": method,
                **metrics,
                "reference_trend_slope_per_year": trend_slope(reference),
                "comparison_trend_slope_per_year": trend_slope(comparison),
                "trend_slope_difference_per_year": trend_slope(comparison) - trend_slope(reference),
                **periods,
                "max_absolute_period_mean_difference": max_period,
                "max_absolute_monthly_difference": float(abs(comparison - reference).max()),
            }
            row["stability_assessment"] = stability_label(metrics, periods)
            metric_rows.append(row)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(
        ROOT / "tables" / "ewbi_gapfill_sensitivity_metrics.csv", index=False
    )
    create_timeseries_figure(monthly)
    create_heatmap(metrics_frame)
    counts = metrics_frame[metrics_frame["method"] != CURRENT]["stability_assessment"].value_counts()
    logger.info("EWBI stability assessments: %s", counts.to_dict())


if __name__ == "__main__":
    main()
