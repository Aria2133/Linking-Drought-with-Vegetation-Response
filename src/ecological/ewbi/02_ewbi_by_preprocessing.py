"""Recalculate EWBI under each preprocessing version."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from detrending_utils import (
    LAND_FOLDERS,
    PERIODS,
    ROOT,
    VERSIONS,
    comparison_metrics,
    configure_logging,
    ensure_directories,
    linear_slope_per_year,
    period_differences,
    standardize_baseline,
)


COMPONENTS = ["SPEI", "TWS", "VPD", "LST"]
REFERENCES = ["raw_monthly", "deseasonalized_only"]


def compute_ewbi(processed: pd.DataFrame, land_type: str, version: str) -> pd.DataFrame:
    subset = processed[
        (processed["land_type"] == land_type)
        & (processed["preprocessing"] == version)
        & (processed["variable"].isin(COMPONENTS))
    ]
    wide = subset.pivot(index="date", columns="variable", values="processed_value").sort_index()
    signed = pd.DataFrame(index=wide.index)
    signed["spei_signed_z"] = standardize_baseline(wide["SPEI"])
    signed["tws_signed_z"] = standardize_baseline(wide["TWS"])
    signed["vpd_signed_z"] = -standardize_baseline(wide["VPD"])
    signed["lst_signed_z"] = -standardize_baseline(wide["LST"])
    signed["ewbi"] = signed.mean(axis=1, skipna=False)
    return signed


def create_timeseries(monthly: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    colors = ["0.3", "#1f77b4", "#d62728", "#2ca02c"]
    for ax, land_type in zip(axes.flat, LAND_FOLDERS):
        land = monthly[monthly["land_type"] == land_type]
        for color, version in zip(colors, VERSIONS):
            group = land[land["preprocessing"] == version]
            ax.plot(
                group["date"],
                group["ewbi"].rolling(3, center=True, min_periods=1).mean(),
                label=version.replace("_", " "),
                linewidth=1.1,
                color=color,
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(land_type)
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("EWBI sensitivity to preprocessing version")
    figure.supylabel("EWBI")
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.96))
    figure.savefig(ROOT / "figures" / "ewbi_detrending_timeseries.png", dpi=220)
    plt.close(figure)


def create_metric_heatmap(metrics: pd.DataFrame) -> None:
    frame = metrics[
        (metrics["reference_preprocessing"] == "deseasonalized_only")
        & (metrics["preprocessing"] != "deseasonalized_only")
    ].copy()
    frame["row"] = frame["land_type"].str.replace("Natural Forests", "Forests", regex=False) + "\n" + frame[
        "preprocessing"
    ].str.replace("_", " ", regex=False)
    columns = [
        "pearson_r",
        "spearman_r",
        "normalized_rmse",
        "standard_deviation_ratio",
        "maximum_absolute_period_mean_difference",
    ]
    matrix = frame.set_index("row")[columns]
    display = matrix.copy()
    display["pearson_r"] = 1.0 - display["pearson_r"]
    display["spearman_r"] = 1.0 - display["spearman_r"]
    display["standard_deviation_ratio"] = abs(display["standard_deviation_ratio"] - 1.0)
    figure, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(display.to_numpy(dtype=float), aspect="auto", cmap="magma")
    ax.set_xticks(
        range(len(columns)),
        ["1-Pearson", "1-Spearman", "NRMSE", "|SD ratio-1|", "Max period difference"],
        rotation=30,
        ha="right",
    )
    ax.set_yticks(range(len(display.index)), display.index, fontsize=8)
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display.iloc[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", color="white", fontsize=7)
    ax.set_title("EWBI divergence from deseasonalized-only reference")
    figure.colorbar(image, ax=ax, label="Divergence (lower is more similar)")
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "ewbi_detrending_metric_heatmap.png", dpi=220)
    plt.close(figure)


def create_period_figure(monthly: pd.DataFrame) -> None:
    rows = []
    for (land_type, version), group in monthly.groupby(["land_type", "preprocessing"]):
        series = group.set_index("date")["ewbi"]
        for period, (start, end) in PERIODS.items():
            rows.append(
                {
                    "land_type": land_type,
                    "preprocessing": version,
                    "period": period,
                    "mean": float(series.loc[start:end].mean()),
                }
            )
    frame = pd.DataFrame(rows)
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    periods = list(PERIODS)
    x = np.arange(len(periods))
    width = 0.19
    for ax, land_type in zip(axes.flat, LAND_FOLDERS):
        land = frame[frame["land_type"] == land_type]
        for offset, version in enumerate(VERSIONS):
            values = [
                land[
                    (land["preprocessing"] == version) & (land["period"] == period)
                ]["mean"].iloc[0]
                for period in periods
            ]
            ax.bar(x + (offset - 1.5) * width, values, width, label=version.replace("_", " "))
        ax.set_xticks(x, periods)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(land_type)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("EWBI period separation by preprocessing")
    figure.supylabel("Period mean EWBI")
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.96))
    figure.savefig(ROOT / "figures" / "ewbi_period_separation_by_preprocessing.png", dpi=220)
    plt.close(figure)


def main() -> None:
    ensure_directories()
    logger = configure_logging(ROOT / "logs" / "ewbi_detrending_log.txt", "ewbi_detrending")
    processed = pd.read_csv(ROOT / "tables" / "preprocessed_monthly_series.csv", parse_dates=["date"])
    monthly_rows: list[dict[str, object]] = []
    series_store: dict[tuple[str, str], pd.Series] = {}

    for land_type in LAND_FOLDERS:
        for version in VERSIONS:
            components = compute_ewbi(processed, land_type, version)
            series_store[(land_type, version)] = components["ewbi"]
            for date, row in components.iterrows():
                monthly_rows.append(
                    {
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "land_type": land_type,
                        "preprocessing": version,
                        **row.to_dict(),
                    }
                )

    metric_rows: list[dict[str, object]] = []
    for land_type in LAND_FOLDERS:
        for reference_name in REFERENCES:
            reference = series_store[(land_type, reference_name)]
            for version in VERSIONS:
                comparison = series_store[(land_type, version)]
                metrics = comparison_metrics(reference, comparison)
                periods = period_differences(reference, comparison)
                max_period = max(
                    abs(periods["baseline_mean_difference"]),
                    abs(periods["historic_mean_difference"]),
                    abs(periods["recent_mean_difference"]),
                )
                metric_rows.append(
                    {
                        "land_type": land_type,
                        "reference_preprocessing": reference_name,
                        "preprocessing": version,
                        **metrics,
                        "reference_trend_slope_per_year": linear_slope_per_year(reference),
                        "comparison_trend_slope_per_year": linear_slope_per_year(comparison),
                        "trend_slope_difference_per_year": (
                            linear_slope_per_year(comparison) - linear_slope_per_year(reference)
                        ),
                        **periods,
                        "maximum_absolute_period_mean_difference": max_period,
                    }
                )

    monthly = pd.DataFrame(monthly_rows)
    metrics = pd.DataFrame(metric_rows)
    monthly.to_csv(ROOT / "tables" / "ewbi_detrending_monthly_series.csv", index=False)
    metrics.to_csv(ROOT / "tables" / "ewbi_detrending_sensitivity_metrics.csv", index=False)
    create_timeseries(monthly)
    create_metric_heatmap(metrics)
    create_period_figure(monthly)

    focus = metrics[
        (metrics["reference_preprocessing"] == "deseasonalized_only")
        & (metrics["preprocessing"].isin(["month_specific_detrended", "ssa_detrended"]))
    ]
    logger.info(
        "EWBI vs deseasonalized: minimum r=%.3f, maximum NRMSE=%.3f, maximum period difference=%.3f",
        focus["pearson_r"].min(),
        focus["normalized_rmse"].max(),
        focus["maximum_absolute_period_mean_difference"].max(),
    )


if __name__ == "__main__":
    main()
