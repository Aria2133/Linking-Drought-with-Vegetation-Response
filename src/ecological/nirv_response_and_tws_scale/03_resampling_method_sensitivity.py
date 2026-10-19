from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    FIGURES,
    LAND_TYPES,
    REPORT,
    TABLES,
    comparison_metrics,
    configure_logger,
    dataframe_to_markdown,
    ensure_output_directories,
    period_mean_differences,
)


METHOD_PAIRS = {
    "TWS nearest 5 km versus bilinear 5 km": (
        "harmonized_5km_nearest_mean",
        "harmonized_5km_bilinear_mean",
    ),
    "TWS bilinear 5 km versus coarse fractional-overlap": (
        "harmonized_5km_bilinear_mean",
        "coarse_fractional_overlap_mean",
    ),
    "TWS nearest 5 km versus coarse fractional-overlap": (
        "harmonized_5km_nearest_mean",
        "coarse_fractional_overlap_mean",
    ),
    "TWS reverse block average versus coarse fractional-overlap": (
        "block_aggregated_5km_to_coarse_mean",
        "coarse_fractional_overlap_mean",
    ),
}


def build_results(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for land in LAND_TYPES:
        subset = monthly[monthly["land_type"] == land].sort_values("date")
        for comparison, (candidate, reference) in METHOD_PAIRS.items():
            metrics = comparison_metrics(
                subset["date"], subset[candidate], subset[reference]
            )
            rows.append(
                {
                    "variable": "TWS",
                    "land_type": land,
                    "comparison": comparison,
                    "candidate_method": candidate,
                    "reference_method": reference,
                    **metrics,
                    **period_mean_differences(
                        subset["date"], subset[candidate], subset[reference]
                    ),
                    "native_file_status": (
                        "original native mascon file unavailable; coarse project export used"
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_heatmap(results: pd.DataFrame) -> None:
    row_labels = []
    values = []
    for land in LAND_TYPES:
        for comparison in METHOD_PAIRS:
            row = results[
                (results["land_type"] == land)
                & (results["comparison"] == comparison)
            ].iloc[0]
            row_labels.append(f"{land}\n{comparison.replace('TWS ', '')}")
            values.append(
                [
                    row["pearson_r"],
                    row["normalized_rmse_by_reference_sd"],
                    abs(row["standard_deviation_ratio"] - 1),
                ]
            )
    matrix = np.asarray(values, dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(15, 10), sharey=False)
    settings = [
        ("Pearson r", "YlGn", 0, 1),
        ("NRMSE / reference SD", "YlOrRd", 0, np.nanmax(matrix[:, 1])),
        ("|SD ratio - 1|", "YlOrRd", 0, np.nanmax(matrix[:, 2])),
    ]
    for column, (title, cmap, vmin, vmax) in enumerate(settings):
        image = axes[column].imshow(
            matrix[:, [column]], cmap=cmap, vmin=vmin, vmax=max(vmax, 1e-9), aspect="auto"
        )
        axes[column].set_title(title)
        axes[column].set_xticks([])
        axes[column].set_yticks(range(len(row_labels)))
        if column == 0:
            axes[column].set_yticklabels(row_labels, fontsize=8)
        else:
            axes[column].set_yticklabels([])
        for row in range(matrix.shape[0]):
            axes[column].text(
                0,
                row,
                f"{matrix[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
        figure.colorbar(image, ax=axes[column], fraction=0.06)
    figure.suptitle("TWS resampling-method sensitivity", fontsize=15)
    figure.tight_layout(rect=(0.16, 0, 1, 0.97))
    figure.savefig(FIGURES / "resampling_method_sensitivity_heatmap.png", dpi=300)
    plt.close(figure)


def write_summary(results: pd.DataFrame) -> None:
    nearest_bilinear = results[
        results["comparison"] == "TWS nearest 5 km versus bilinear 5 km"
    ]
    coarse = results[
        results["comparison"]
        == "TWS bilinear 5 km versus coarse fractional-overlap"
    ]
    columns = [
        "land_type",
        "comparison",
        "pearson_r",
        "rmse",
        "mae",
        "mean_bias",
        "standard_deviation_ratio",
        "mean_difference_baseline_2001_2011",
        "mean_difference_historic_2012_2016",
        "mean_difference_recent_2020_2024",
        "number_valid_months",
    ]
    text = [
        "# Resampling method sensitivity summary",
        "",
        "## Nearest versus bilinear 5 km overlay",
        "",
        dataframe_to_markdown(nearest_bilinear[columns]),
        "",
        "## Bilinear 5 km overlay versus coarse fractional-overlap reference",
        "",
        dataframe_to_markdown(coarse[columns]),
        "",
        "## Interpretation",
        "",
        (
            "Nearest-neighbor and bilinear expansion were evaluated as alternative "
            "representations of the same coarse TWS field. Agreement of land-type monthly "
            "summaries does not create independent fine-scale hydrologic information. "
            "Reverse average aggregation provides an additional check on whether the "
            "expanded surface preserves the coarse-grid summary."
        ),
        "",
    ]
    (REPORT / "resampling_method_sensitivity_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )


def main() -> None:
    ensure_output_directories()
    logger = configure_logger("resampling", "resampling_method_sensitivity_log.txt")
    monthly_path = TABLES / "tws_landtype_monthly_series_5km_vs_coarse.csv"
    if not monthly_path.exists():
        raise FileNotFoundError("Run 02_tws_scale_sensitivity.py first.")
    monthly = pd.read_csv(monthly_path, parse_dates=["date"])
    results = build_results(monthly)
    results.to_csv(TABLES / "resampling_method_sensitivity.csv", index=False)
    plot_heatmap(results)
    write_summary(results)
    logger.info("Wrote %d resampling sensitivity rows", len(results))


if __name__ == "__main__":
    main()
