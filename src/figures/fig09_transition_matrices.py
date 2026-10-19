"""Render NIRv spatial-response and ecosystem-state figures."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figure_utils import (  # noqa: E402
    DETREND_ROOT,
    LAND_TYPES,
    STATE_ORDER,
    TABLES,
    TWS_ROOT,
    apply_publication_style,
    configure_logging,
    ensure_directories,
    normalize_land_type,
    save_figure,
)


def reference_extent(land_type: str) -> list[float]:
    raw = pd.read_csv(DETREND_ROOT / "tables" / "monthly_landtype_series_raw.csv")
    raw["land_type"] = raw["land_type"].map(normalize_land_type)
    row = raw[
        (raw["land_type"] == land_type)
        & (raw["variable"] == "NIRv")
        & raw["source_path"].notna()
    ].iloc[0]
    with rasterio.open(row["source_path"]) as source:
        bounds = source.bounds
    return [bounds.left, bounds.right, bounds.bottom, bounds.top]


def render_figure_8(logger) -> None:
    archive = np.load(TWS_ROOT / "cache" / "nirv_period_maps.npz")
    stress = pd.read_csv(TWS_ROOT / "tables" / "stress_zone_area_uncertainty.csv")
    stress["land_type"] = stress["land_type"].map(normalize_land_type)
    main = stress[stress["threshold"] == -40].set_index("land_type").loc[LAND_TYPES]

    figure, axes = plt.subplots(4, 2, figsize=(18, 24))
    map_rows = []
    for row, land_type in enumerate(LAND_TYPES):
        archive_key = "NaturalForests" if land_type == "Natural Forests" else land_type
        extent = reference_extent(land_type)
        summary = main.loc[land_type]
        for col, (period_key, period_label) in enumerate(
            [("historic_decline", "2012-2016"), ("recent_decline", "2020-2024")]
        ):
            values = archive[f"{archive_key}__{period_key}"].astype(float)
            axis = axes[row, col]
            image = axis.imshow(
                np.ma.masked_invalid(values),
                extent=extent,
                origin="upper",
                cmap="RdYlGn",
                vmin=-60,
                vmax=30,
                interpolation="nearest",
                aspect="auto",
            )
            median_change = float(np.nanmedian(values))
            mean_change = float(np.nanmean(values))
            if col == 0:
                area = summary["historic_area_km2"]
                percent = summary["historic_percent"]
                annotation = (
                    f"Mean: {mean_change:+.1f}%\n"
                    f"Severe (<-40%): {area:,.0f} km2 ({percent:.2f}%)"
                )
            else:
                area = summary["recent_area_km2"]
                percent = summary["recent_percent"]
                annotation = (
                    f"Mean: {mean_change:+.1f}%\n"
                    f"Severe (<-40%): {area:,.0f} km2 ({percent:.2f}%)\n"
                    f"Change: {summary['difference_km2']:+,.0f} km2 "
                    f"[{summary['ci95_low_km2']:,.0f}, {summary['ci95_high_km2']:,.0f}]\n"
                    f"{summary['difference_percentage_points']:+.2f} pp "
                    f"[{summary['ci95_low_percentage_points']:.2f}, "
                    f"{summary['ci95_high_percentage_points']:.2f}]"
                )
            axis.text(
                0.985,
                0.97,
                annotation,
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=7.6,
                fontfamily="monospace",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": "#555555",
                    "alpha": 0.90,
                },
            )
            colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            colorbar.set_label("Change (%)", fontsize=10, fontweight="bold")
            period_title = (
                "2012-2016 Historic Drought"
                if col == 0
                else "2020-2024 Recent Period"
            )
            axis.set_title(
                f"{land_type}\n{period_title}",
                fontsize=13,
                fontweight="bold",
                pad=10,
            )
            axis.set_xlabel("Longitude (degrees)", fontsize=10, fontweight="bold")
            axis.set_ylabel("Latitude (degrees)", fontsize=10, fontweight="bold")
            axis.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
            map_rows.append(
                {
                    "land_type": land_type,
                    "period": period_label,
                    "mean_nirv_change_percent": mean_change,
                    "median_nirv_change_percent": median_change,
                    "stressed_area_km2": area,
                    "stressed_percent": percent,
                }
            )

    figure.suptitle(
        "Vegetation Response: NIRv Changes\n"
        "Historic Drought (2012-2016) vs Recent Period (2020-2024)",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.01,
        "Red = Decline \u2022 Green = Increase/Stable \u2022 Percentage change from "
        "the 2001-2011 baseline\n"
        "Severe-area changes use latitude-aware area and paired spatial-block "
        "bootstrap 95% intervals",
        ha="center",
        fontsize=11,
        fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.4},
    )
    figure.tight_layout(rect=[0, 0.025, 1, 0.985])
    save_figure(figure, "Figure_8_nirv_spatial_response_comparison", dpi=300)

    output = main.reset_index().merge(
        pd.DataFrame(map_rows),
        on="land_type",
        how="left",
    )
    output.to_csv(TABLES / "figure_8_nirv_spatial_response_summary.csv", index=False)
    logger.info("Rendered Figure 8 from the latest NIRv maps and block-bootstrap table.")


def render_figure_9(logger) -> None:
    transitions = pd.read_csv(
        DETREND_ROOT
        / "tables"
        / "main_state_transition_probabilities_deseasonalized.csv"
    )
    transitions["land_type"] = transitions["land_type"].map(normalize_land_type)
    transitions = transitions[transitions["period"] == "all"].copy()
    transitions["probability_percent"] = 100 * transitions["transition_probability"]

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 11.0))
    image = None
    for axis, land_type in zip(axes.flat, LAND_TYPES):
        subset = transitions[transitions["land_type"] == land_type]
        matrix = (
            subset.pivot(
                index="from_state",
                columns="to_state",
                values="probability_percent",
            )
            .reindex(index=STATE_ORDER, columns=STATE_ORDER)
            .to_numpy(float)
        )
        image = axis.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100)
        labels = [value.replace(" / ", "/\n") for value in STATE_ORDER]
        axis.set_xticks(range(4), labels)
        axis.set_yticks(range(4), STATE_ORDER)
        axis.set_xlabel("State at month t+1")
        axis.set_ylabel("State at month t")
        axis.set_title(land_type, fontweight="bold")
        for source in range(4):
            for target in range(4):
                value = matrix[source, target]
                axis.text(
                    target,
                    source,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="white" if value > 58 else "black",
                )

    figure.subplots_adjust(
        left=0.10, right=0.86, bottom=0.08, top=0.88, wspace=0.38, hspace=0.42
    )
    colorbar_axis = figure.add_axes([0.90, 0.20, 0.02, 0.58])
    figure.colorbar(image, cax=colorbar_axis, label="Transition probability (%)")
    figure.suptitle("Ecosystem-state transition probabilities (2001-2024)", y=0.955)
    save_figure(figure, "Figure_9_ecosystem_state_or_transition")
    transitions.to_csv(TABLES / "figure_9_transition_probabilities.csv", index=False)
    logger.info("Rendered Figure 9 from deseasonalized-only EWBI and NIRv states.")


def main() -> None:
    ensure_directories()
    apply_publication_style()
    logger = configure_logging("render_figures_8_9_log.txt", "render_figures_8_9")
    render_figure_8(logger)
    render_figure_9(logger)


if __name__ == "__main__":
    main()
