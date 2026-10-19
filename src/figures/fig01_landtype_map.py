"""Render manuscript Figures 1, 2, and 3."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figure_utils import (  # noqa: E402
    FORCING,
    LAND_COLORS,
    LAND_TYPES,
    TABLES,
    THESIS_ROOT,
    VEGETATION,
    apply_publication_style,
    configure_logging,
    ensure_directories,
    save_figure,
)


MASKS = {
    "Grasslands": THESIS_ROOT / "Data" / "CA_LandMasks" / "CA_Grasslands_2021.tif",
    "Natural Forests": THESIS_ROOT
    / "Data"
    / "CA_LandMasks"
    / "CA_NaturalForests_2021.tif",
    "Savannas": THESIS_ROOT / "Data" / "CA_LandMasks" / "CA_Savannas_2021.tif",
    "Shrublands": THESIS_ROOT / "Data" / "CA_LandMasks" / "CA_Shrublands_2021.tif",
}
MASK_COLORS = {
    "Grasslands": "#8FBC8F",
    "Natural Forests": "#245501",
    "Savannas": "#DAA520",
    "Shrublands": "#CD853F",
}


def row_pixel_area_km2(transform, row: int) -> float:
    radius = 6371.0088
    north = transform.f + row * transform.e
    south = north + transform.e
    dlon = math.radians(abs(transform.a))
    return radius**2 * dlon * abs(
        math.sin(math.radians(north)) - math.sin(math.radians(south))
    )


def render_figure_1(logger) -> None:
    combined = None
    transform = None
    bounds = None
    area_rows = []
    overlap_pixels = 0
    for index, land_type in enumerate(LAND_TYPES, start=1):
        path = MASKS[land_type]
        with rasterio.open(path) as source:
            array = source.read(1)
            mask = np.isfinite(array) & (array > 0)
            if combined is None:
                combined = np.zeros(source.shape, dtype=np.uint8)
                transform = source.transform
                bounds = source.bounds
                crs = str(source.crs)
            elif source.shape != combined.shape or source.transform != transform:
                raise ValueError(f"Land mask grid mismatch: {path}")
        overlap_pixels += int(np.sum(mask & (combined > 0)))
        combined[mask] = index
        area = sum(
            row_pixel_area_km2(transform, row) * int(mask[row].sum())
            for row in range(mask.shape[0])
        )
        area_rows.append(
            {
                "land_type": land_type,
                "area_km2": area,
                "mask_pixels": int(mask.sum()),
                "mask_file": str(path),
                "crs": crs,
            }
        )

    figure, axis = plt.subplots(figsize=(8.2, 9.2))
    cmap = ListedColormap(["white", *[MASK_COLORS[value] for value in LAND_TYPES]])
    norm = BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
    axis.imshow(
        combined,
        extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
        origin="upper",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )
    axis.set_xlabel("Longitude (degrees)")
    axis.set_ylabel("Latitude (degrees)")
    axis.grid(alpha=0.22, linestyle="--", linewidth=0.6)
    handles = [
        mpatches.Patch(
            facecolor=MASK_COLORS[land_type],
            edgecolor="#333333",
            label=land_type,
        )
        for land_type in LAND_TYPES
    ]
    axis.legend(
        handles=handles,
        title="Land type",
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )
    axis.annotate(
        "N",
        xy=(0.91, 0.72),
        xytext=(0.91, 0.61),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "black"},
    )
    axis.set_title("Spatial distribution of land types in California (2021)", pad=12)
    figure.tight_layout()
    save_figure(figure, "Figure_1_land_types")

    areas = pd.DataFrame(area_rows)
    areas["area_percent_of_four_types"] = (
        100 * areas["area_km2"] / areas["area_km2"].sum()
    )
    areas["overlap_pixels_detected_across_masks"] = overlap_pixels
    areas.to_csv(TABLES / "figure_1_land_type_areas.csv", index=False)
    logger.info("Rendered Figure 1; mask overlap count=%d.", overlap_pixels)


def render_time_series(
    data: pd.DataFrame,
    variables: list[str],
    filename: str,
    title: str,
) -> None:
    figure, axes = plt.subplots(
        len(variables),
        1,
        figsize=(11.5, 2.15 * len(variables) + 1.2),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for axis, variable in zip(axes, variables):
        subset = data[data["variable"] == variable]
        for land_type in LAND_TYPES:
            series = (
                subset[subset["land_type"] == land_type]
                .set_index("date")["standardized_anomaly"]
                .sort_index()
            )
            axis.plot(
                series.index,
                series,
                color=LAND_COLORS[land_type],
                linewidth=0.55,
                alpha=0.24,
            )
            axis.plot(
                series.index,
                series.rolling(3, center=True, min_periods=2).mean(),
                color=LAND_COLORS[land_type],
                linewidth=1.45,
                alpha=0.98,
                label=land_type,
            )
        axis.axhline(0, color="#333333", linewidth=0.65)
        axis.axvspan(
            pd.Timestamp("2012-01-01"),
            pd.Timestamp("2016-12-31"),
            color="#F4CCCC",
            alpha=0.34,
            linewidth=0,
        )
        axis.axvspan(
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2024-12-31"),
            color="#FFF2CC",
            alpha=0.42,
            linewidth=0,
        )
        axis.set_ylabel(f"{variable}\n(SD)")
        axis.grid(axis="y", alpha=0.18, linewidth=0.5)

    land_handles = [
        Line2D([0], [0], color=LAND_COLORS[value], lw=2, label=value)
        for value in LAND_TYPES
    ]
    style_handles = [
        Line2D([0], [0], color="#555555", lw=0.7, alpha=0.45, label="Monthly"),
        Line2D([0], [0], color="#555555", lw=2, label="3-month mean"),
    ]
    figure.legend(
        handles=land_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=6,
        frameon=False,
    )
    figure.suptitle(title, y=0.998)
    axes[-1].set_xlabel("Year")
    figure.text(
        0.5,
        0.012,
        "Shading: 2012-2016 historic drought and 2020-2024 recent period",
        ha="center",
        fontsize=8.5,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.93))
    save_figure(figure, filename)


def main() -> None:
    ensure_directories()
    apply_publication_style()
    logger = configure_logging("render_figures_1_3_log.txt", "render_figures_1_3")
    data = pd.read_csv(TABLES / "paper_standardized_anomaly_series.csv", parse_dates=["date"])
    render_figure_1(logger)
    render_time_series(
        data,
        VEGETATION,
        "Figure_2_vegetation_productivity_anomalies",
        "Vegetation and productivity standardized anomalies (2001-2024)",
    )
    render_time_series(
        data,
        FORCING,
        "Figure_3_meteorological_hydrological_anomalies",
        "Meteorological and hydrological standardized anomalies (2001-2024)",
    )
    logger.info("Rendered Figures 1, 2, and 3 in PNG and PDF.")


if __name__ == "__main__":
    main()
