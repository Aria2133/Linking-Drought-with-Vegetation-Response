"""Create preprocessing versions and quantify trend and variance removal."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from detrending_utils import (
    LAND_FOLDERS,
    ROOT,
    VARIABLES,
    VERSIONS,
    analysis_series,
    configure_logging,
    ensure_directories,
    linear_slope_per_year,
    load_raw_table,
    preprocess_ecological,
)


def create_series_figure(processed: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    colors = {
        "raw_monthly": "0.35",
        "deseasonalized_only": "#1f77b4",
        "month_specific_detrended": "#d62728",
        "ssa_detrended": "#2ca02c",
    }
    for ax, land_type in zip(axes.flat, LAND_FOLDERS):
        subset = processed[
            (processed["land_type"] == land_type) & (processed["variable"] == "TWS")
        ]
        for version in VERSIONS:
            group = subset[subset["preprocessing"] == version].set_index("date")
            values = group["processed_value"]
            standardized = (values - values.mean()) / values.std(ddof=1)
            ax.plot(
                standardized.index,
                standardized.rolling(3, center=True, min_periods=1).mean(),
                label=version.replace("_", " "),
                color=colors[version],
                linewidth=1.2,
            )
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(land_type)
        ax.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("TWS preprocessing comparison (standardized, 3-month smoothing)")
    figure.supylabel("Version-specific standardized value")
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.96))
    figure.savefig(ROOT / "figures" / "detrending_series_comparison.png", dpi=220)
    plt.close(figure)


def create_heatmap(summary: pd.DataFrame) -> None:
    frame = summary.copy()
    frame["row"] = frame["land_type"].str.replace("Natural Forests", "Forests", regex=False) + " | " + frame["variable"]
    pivot = frame.pivot(index="row", columns="preprocessing", values="percent_absolute_trend_removed")
    pivot = pivot.reindex(columns=VERSIONS)
    figure, ax = plt.subplots(figsize=(10, 13))
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-100, vmax=100)
    ax.set_xticks(range(len(VERSIONS)), [value.replace("_", "\n") for value in VERSIONS])
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=7)
    ax.set_title(
        "Change in absolute long-term trend magnitude\n"
        "(positive = removed; negative = amplified or reversed; colors clipped at +/-100%)"
    )
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isfinite(value):
                ax.text(col, row, f"{value:.0f}%", ha="center", va="center", fontsize=6)
    figure.colorbar(image, ax=ax, label="Change in absolute trend magnitude (%)")
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "trend_removal_heatmap.png", dpi=220)
    plt.close(figure)


def main() -> None:
    ensure_directories()
    logger = configure_logging(
        ROOT / "logs" / "preprocessing_versions_log.txt", "preprocessing_versions"
    )
    raw = load_raw_table()
    series_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        for variable in VARIABLES:
            original = analysis_series(raw, land_type, variable)
            original_slope = linear_slope_per_year(original)
            original_variance = float(original.var(ddof=1))
            for version in VERSIONS:
                result = preprocess_ecological(original, version)
                processed = result["processed"]
                seasonal = result["seasonal"]
                trend = result["trend"]
                metadata = result["metadata"]
                for date in original.index:
                    series_rows.append(
                        {
                            "date": date,
                            "year": date.year,
                            "month": date.month,
                            "land_type": land_type,
                            "variable": variable,
                            "preprocessing": version,
                            "original_value": original.loc[date],
                            "processed_value": processed.loc[date],
                            "seasonal_component": seasonal.loc[date],
                            "trend_component": trend.loc[date],
                            "residual_value": result["residual"].loc[date],
                            **metadata,
                        }
                    )
                processed_slope = linear_slope_per_year(processed)
                if np.isfinite(original_slope) and abs(original_slope) > 1.0e-12:
                    percent_removed = (
                        1.0 - abs(processed_slope) / abs(original_slope)
                    ) * 100.0
                else:
                    percent_removed = np.nan
                valid = original.notna() & processed.notna()
                correlation = (
                    float(pearsonr(original[valid], processed[valid]).statistic)
                    if int(valid.sum()) >= 3
                    else np.nan
                )
                variance_retained = (
                    100.0 * float(processed.var(ddof=1)) / original_variance
                    if original_variance > 0
                    else np.nan
                )
                summary_rows.append(
                    {
                        "land_type": land_type,
                        "variable": variable,
                        "preprocessing": version,
                        "original_trend_slope_per_year": original_slope,
                        "processed_trend_slope_per_year": processed_slope,
                        "trend_slope_change_per_year": processed_slope - original_slope,
                        "percent_absolute_trend_removed": percent_removed,
                        "variance_retained_percent": variance_retained,
                        "correlation_with_raw": correlation,
                        "seasonal_reference": metadata["seasonal_reference"],
                        "trend_fit_period": metadata["trend_fit_period"],
                        "ssa_window_months": metadata["ssa_window_months"],
                        "ssa_removed_components": metadata["ssa_removed_components"],
                        "ssa_removed_trajectory_variance_fraction": metadata[
                            "ssa_removed_trajectory_variance_fraction"
                        ],
                        "warning": metadata["warning"],
                    }
                )
            logger.info("Processed %s | %s across four versions", land_type, variable)

    processed_frame = pd.DataFrame(series_rows)
    summary = pd.DataFrame(summary_rows)
    processed_frame.to_csv(ROOT / "tables" / "preprocessed_monthly_series.csv", index=False)
    summary.to_csv(ROOT / "tables" / "detrending_trend_removal_summary.csv", index=False)
    create_series_figure(processed_frame)
    create_heatmap(summary)

    tws = summary[
        (summary["variable"] == "TWS")
        & (summary["preprocessing"] == "month_specific_detrended")
    ]
    logger.warning(
        "Month-specific detrending TWS trend removal by land type: %s",
        dict(zip(tws["land_type"], tws["percent_absolute_trend_removed"].round(1))),
    )


if __name__ == "__main__":
    main()
