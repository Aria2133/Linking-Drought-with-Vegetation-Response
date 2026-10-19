from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    FIGURES,
    LAND_TYPES,
    REPORT,
    TABLES,
    cell_area_km2,
    comparison_metrics,
    configure_logger,
    dataframe_to_markdown,
    discover_land_files,
    discover_tws_source_files,
    ensure_output_directories,
    fractional_land_mask,
    grid_from_path,
    nirv_reference_grid,
    period_mean_differences,
    read_raster,
    reproject_array,
    weighted_mean,
)


FULL_INDEX = pd.date_range("2001-01-01", "2024-12-01", freq="MS")
REPRESENTATION_COLUMNS = {
    "5km_bilinear_vs_coarse_fractional": (
        "harmonized_5km_bilinear_mean",
        "coarse_fractional_overlap_mean",
    ),
    "5km_nearest_vs_coarse_fractional": (
        "harmonized_5km_nearest_mean",
        "coarse_fractional_overlap_mean",
    ),
    "nearest_vs_bilinear_5km": (
        "harmonized_5km_nearest_mean",
        "harmonized_5km_bilinear_mean",
    ),
    "reverse_block_average_vs_coarse_fractional": (
        "block_aggregated_5km_to_coarse_mean",
        "coarse_fractional_overlap_mean",
    ),
    "stored_masked_coarse_vs_fractional_coarse": (
        "stored_landtype_masked_coarse_mean",
        "coarse_fractional_overlap_mean",
    ),
}


def load_stored_masked_mean(
    path,
    coarse_area: np.ndarray,
) -> float:
    if path is None or not path.exists():
        return np.nan
    values, _ = read_raster(path)
    return weighted_mean(values, coarse_area)


def build_monthly_series(logger) -> pd.DataFrame:
    source_files = discover_tws_source_files()
    if not source_files:
        raise FileNotFoundError("No statewide TWS source rasters were found.")

    first_source = next(iter(source_files.values()))
    source_grid = grid_from_path(first_source)
    coarse_area = cell_area_km2(source_grid)

    target_grids = {land: nirv_reference_grid(land) for land in LAND_TYPES}
    target_areas = {land: cell_area_km2(target_grids[land]) for land in LAND_TYPES}
    coarse_fractions = {
        land: fractional_land_mask(land, source_grid) for land in LAND_TYPES
    }
    target_fractions = {
        land: fractional_land_mask(land, target_grids[land]) for land in LAND_TYPES
    }
    stored_files = {land: discover_land_files(land, "TWS") for land in LAND_TYPES}

    logger.info(
        "Coarse source grid: %s, resolution %.6f x %.6f degrees",
        source_grid["shape"],
        abs(source_grid["transform"].a),
        abs(source_grid["transform"].e),
    )
    for land in LAND_TYPES:
        logger.info(
            "%s target grid: %s, resolution %.6f x %.6f degrees",
            land,
            target_grids[land]["shape"],
            abs(target_grids[land]["transform"].a),
            abs(target_grids[land]["transform"].e),
        )

    rows = []
    for date in FULL_INDEX:
        source_path = source_files.get(date)
        if source_path is None:
            for land in LAND_TYPES:
                rows.append(
                    {
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "land_type": land,
                        "source_file": "",
                        "source_month_status": "missing",
                        "coarse_fractional_overlap_mean": np.nan,
                        "stored_landtype_masked_coarse_mean": np.nan,
                        "harmonized_5km_nearest_mean": np.nan,
                        "harmonized_5km_bilinear_mean": np.nan,
                        "block_aggregated_5km_to_coarse_mean": np.nan,
                        "analysis_mode": (
                            "coarse project-source sensitivity; not true native mascon validation"
                        ),
                    }
                )
            continue

        coarse_values, source_meta = read_raster(source_path)
        source_meta["shape"] = coarse_values.shape
        for land in LAND_TYPES:
            coarse_weights = coarse_area * coarse_fractions[land]
            target_weights = target_areas[land] * target_fractions[land]
            coarse_mean = weighted_mean(coarse_values, coarse_weights)

            nearest = reproject_array(
                coarse_values, source_meta, target_grids[land], "nearest"
            )
            bilinear = reproject_array(
                coarse_values, source_meta, target_grids[land], "bilinear"
            )
            nearest_mean = weighted_mean(nearest, target_weights)
            bilinear_mean = weighted_mean(bilinear, target_weights)

            reverse_average = reproject_array(
                bilinear, target_grids[land], source_grid, "average"
            )
            reverse_mean = weighted_mean(reverse_average, coarse_weights)

            stored_path = stored_files[land].get(date)
            stored_mean = load_stored_masked_mean(stored_path, coarse_area)
            rows.append(
                {
                    "date": date,
                    "year": date.year,
                    "month": date.month,
                    "land_type": land,
                    "source_file": str(source_path),
                    "source_month_status": "available",
                    "coarse_fractional_overlap_mean": coarse_mean,
                    "stored_landtype_masked_coarse_mean": stored_mean,
                    "harmonized_5km_nearest_mean": nearest_mean,
                    "harmonized_5km_bilinear_mean": bilinear_mean,
                    "block_aggregated_5km_to_coarse_mean": reverse_mean,
                    "analysis_mode": (
                        "coarse project-source sensitivity; not true native mascon validation"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_metrics(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for land in LAND_TYPES:
        subset = monthly[monthly["land_type"] == land].sort_values("date")
        for comparison, (candidate_column, reference_column) in REPRESENTATION_COLUMNS.items():
            metrics = comparison_metrics(
                subset["date"],
                subset[candidate_column],
                subset[reference_column],
            )
            period_metrics = period_mean_differences(
                subset["date"],
                subset[candidate_column],
                subset[reference_column],
            )
            pattern_stable = bool(
                np.isfinite(metrics["pearson_r"])
                and metrics["pearson_r"] >= 0.95
                and metrics["spearman_r"] >= 0.95
            )
            magnitude_stable = bool(
                np.isfinite(metrics["normalized_rmse_by_reference_sd"])
                and metrics["normalized_rmse_by_reference_sd"] <= 0.25
                and 0.8 <= metrics["standard_deviation_ratio"] <= 1.2
            )
            rows.append(
                {
                    "land_type": land,
                    "comparison": comparison,
                    "candidate_representation": candidate_column,
                    "reference_representation": reference_column,
                    "is_primary_comparison": comparison
                    == "5km_bilinear_vs_coarse_fractional",
                    **metrics,
                    **period_metrics,
                    "temporal_pattern_stable_by_prespecified_rule": pattern_stable,
                    "magnitude_stable_by_prespecified_rule": magnitude_stable,
                    "overall_stable_by_prespecified_rule": (
                        pattern_stable and magnitude_stable
                    ),
                    "stability_rule": (
                        "Pearson and Spearman >= 0.95; normalized RMSE <= 0.25; "
                        "standard-deviation ratio between 0.8 and 1.2"
                    ),
                    "analysis_mode": (
                        "coarse project-source comparison; original native mascon "
                        "geometry unavailable"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_timeseries(monthly: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    for ax, land in zip(axes.ravel(), LAND_TYPES):
        subset = monthly[monthly["land_type"] == land].sort_values("date")
        ax.plot(
            subset["date"],
            subset["coarse_fractional_overlap_mean"],
            color="#111111",
            linewidth=1.7,
            label="Coarse fractional-overlap",
        )
        ax.plot(
            subset["date"],
            subset["harmonized_5km_bilinear_mean"],
            color="#2C7FB8",
            linewidth=1.1,
            alpha=0.9,
            label="5 km bilinear overlay",
        )
        ax.plot(
            subset["date"],
            subset["harmonized_5km_nearest_mean"],
            color="#D95F0E",
            linewidth=0.9,
            alpha=0.75,
            label="5 km nearest overlay",
        )
        ax.set_title(land)
        ax.set_ylabel("TWS anomaly (cm LWE)")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(
        "TWS land-type summaries across coarse and 5 km representations", fontsize=15
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.96))
    figure.savefig(FIGURES / "tws_scale_sensitivity_timeseries.png", dpi=300)
    plt.close(figure)


def plot_scatter(monthly: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, land in zip(axes.ravel(), LAND_TYPES):
        subset = monthly[monthly["land_type"] == land]
        x = subset["coarse_fractional_overlap_mean"].to_numpy(float)
        y = subset["harmonized_5km_bilinear_mean"].to_numpy(float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=18, alpha=0.55, color="#2C7FB8")
        if valid.any():
            limits = [
                min(float(x[valid].min()), float(y[valid].min())),
                max(float(x[valid].max()), float(y[valid].max())),
            ]
            ax.plot(limits, limits, color="black", linestyle="--", linewidth=1)
            ax.set_xlim(limits)
            ax.set_ylim(limits)
        ax.set_title(land)
        ax.set_xlabel("Coarse fractional-overlap mean")
        ax.set_ylabel("5 km bilinear-overlay mean")
        ax.grid(alpha=0.2)
    figure.suptitle("TWS scale-sensitivity scatter comparison", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(FIGURES / "tws_scale_sensitivity_scatter.png", dpi=300)
    plt.close(figure)


def plot_metric_heatmap(metrics: pd.DataFrame) -> None:
    primary = metrics[metrics["is_primary_comparison"]].set_index("land_type").loc[LAND_TYPES]
    correlation = primary[["pearson_r", "spearman_r"]].to_numpy(float)
    error = np.column_stack(
        [
            primary["normalized_rmse_by_reference_sd"].to_numpy(float),
            np.abs(primary["standard_deviation_ratio"].to_numpy(float) - 1.0),
        ]
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    left = axes[0].imshow(correlation, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Correlation")
    axes[0].set_xticks([0, 1], ["Pearson r", "Spearman r"])
    axes[0].set_yticks(range(len(LAND_TYPES)), LAND_TYPES)
    for row in range(correlation.shape[0]):
        for col in range(correlation.shape[1]):
            axes[0].text(col, row, f"{correlation[row, col]:.3f}", ha="center", va="center")
    figure.colorbar(left, ax=axes[0], fraction=0.046)

    right = axes[1].imshow(error, cmap="YlOrRd", vmin=0, aspect="auto")
    axes[1].set_title("Normalized disagreement (lower is better)")
    axes[1].set_xticks([0, 1], ["NRMSE / ref SD", "|SD ratio - 1|"])
    axes[1].set_yticks(range(len(LAND_TYPES)), LAND_TYPES)
    for row in range(error.shape[0]):
        for col in range(error.shape[1]):
            axes[1].text(col, row, f"{error[row, col]:.3f}", ha="center", va="center")
    figure.colorbar(right, ax=axes[1], fraction=0.046)
    figure.suptitle("Primary TWS scale-sensitivity metrics", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(FIGURES / "tws_scale_sensitivity_metric_heatmap.png", dpi=300)
    plt.close(figure)


def write_summary(metrics: pd.DataFrame, monthly: pd.DataFrame) -> None:
    primary = metrics[metrics["is_primary_comparison"]].copy()
    stable_count = int(primary["overall_stable_by_prespecified_rule"].sum())
    missing = (
        monthly[["date", "source_month_status"]]
        .drop_duplicates()
        .query("source_month_status == 'missing'")["date"]
    )
    display_columns = [
        "land_type",
        "pearson_r",
        "spearman_r",
        "rmse",
        "mae",
        "mean_bias",
        "standard_deviation_ratio",
        "normalized_rmse_by_reference_sd",
        "trend_slope_difference_per_year",
        "number_valid_months",
        "overall_stable_by_prespecified_rule",
    ]
    text = [
        "# TWS scale sensitivity summary",
        "",
        "## Analysis status",
        "",
        (
            "Original native GRACE/GRACE-FO mascon geometry was not available. A coarse "
            "statewide project-source export was available and used as the reference. "
            "The stored grid is approximately 0.898 degrees (about 100 km export spacing), "
            "while the effective hydrologic information remains mascon-scale/coarser."
        ),
        "",
        (
            "This is therefore a coarse-source representation sensitivity analysis, not "
            "an independent validation against original native mascon files."
        ),
        "",
        "## Primary comparison",
        "",
        dataframe_to_markdown(primary[display_columns]),
        "",
        (
            f"{stable_count} of {len(primary)} land types met the prespecified joint "
            "pattern-and-magnitude stability rule. This rule is a transparent diagnostic, "
            "not a claim that 5 km TWS pixels are independent observations."
        ),
        "",
        "## Missing months",
        "",
        (
            f"{len(missing)} months between January 2001 and December 2024 lacked a "
            "statewide source TWS raster and were retained as explicit missing records."
        ),
        "",
        "## Interpretation",
        "",
        (
            "The comparison evaluates whether monthly land-type summaries are stable to "
            "the representation and resampling method. It does not imply that resampling "
            "creates 5 km hydrologic information. TWS effective information content "
            "remains coarse."
        ),
        "",
    ]
    (REPORT / "tws_scale_sensitivity_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_output_directories()
    logger = configure_logger("tws_scale", "tws_scale_sensitivity_log.txt")
    monthly = build_monthly_series(logger)
    metrics = build_metrics(monthly)

    monthly.to_csv(
        TABLES / "tws_landtype_monthly_series_5km_vs_coarse.csv", index=False
    )
    metrics.to_csv(TABLES / "tws_scale_sensitivity_metrics.csv", index=False)
    plot_timeseries(monthly)
    plot_scatter(monthly)
    plot_metric_heatmap(metrics)
    write_summary(metrics, monthly)

    logger.info("Monthly records: %d", len(monthly))
    logger.info("Available source months: %d", monthly.query("source_month_status == 'available'")["date"].nunique())
    logger.info("Metric rows: %d", len(metrics))
    logger.info(
        "Primary stable land types: %d of %d",
        int(metrics[metrics["is_primary_comparison"]]["overall_stable_by_prespecified_rule"].sum()),
        len(LAND_TYPES),
    )


if __name__ == "__main__":
    main()

