from __future__ import annotations

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis_utils import (
    BASELINE,
    CACHE,
    FIGURES,
    HISTORIC,
    LAND_KEYS,
    LAND_TYPES,
    RECENT,
    REPORT,
    TABLES,
    cell_area_km2,
    configure_logger,
    dataframe_to_markdown,
    discover_land_files,
    ensure_output_directories,
    fractional_land_mask,
    grids_match,
    nominal_resolution_km,
    nirv_reference_grid,
    read_raster,
    reproject_array,
)


THRESHOLDS = [-30, -40, -50]
MAIN_THRESHOLD = -40
BOOTSTRAP_REPLICATES = 2000
BLOCK_SIZE_PIXELS = 10
RANDOM_SEED = 42
MIN_BASELINE_NIRV = 0.01


def period_for_year(year: int) -> str | None:
    if BASELINE[0] <= year <= BASELINE[1]:
        return "baseline"
    if HISTORIC[0] <= year <= HISTORIC[1]:
        return "historic"
    if RECENT[0] <= year <= RECENT[1]:
        return "recent"
    return None


def build_period_maps(land_type: str, logger) -> tuple[dict[str, np.ndarray], dict]:
    files = discover_land_files(land_type, "NIRv")
    target = nirv_reference_grid(land_type)
    sums = {
        period: np.zeros(target["shape"], dtype=np.float64)
        for period in ("baseline", "historic", "recent")
    }
    counts = {
        period: np.zeros(target["shape"], dtype=np.uint16)
        for period in ("baseline", "historic", "recent")
    }
    month_counts = {period: 0 for period in ("baseline", "historic", "recent")}
    expected_counts = {"baseline": 132, "historic": 60, "recent": 60}
    reprojected_count = 0

    for date, path in files.items():
        period = period_for_year(date.year)
        if period is None:
            continue
        array, metadata = read_raster(path)
        metadata["shape"] = array.shape
        if not grids_match(metadata, target):
            logger.warning(
                "%s %s grid differs from reference; bilinear reprojection applied: %s",
                land_type,
                date.strftime("%Y-%m"),
                path,
            )
            array = reproject_array(array, metadata, target, "bilinear")
            reprojected_count += 1
        valid = np.isfinite(array)
        sums[period][valid] += array[valid]
        counts[period][valid] += 1
        month_counts[period] += 1

    means = {}
    for period in ("baseline", "historic", "recent"):
        means[period] = np.divide(
            sums[period],
            counts[period],
            out=np.full(target["shape"], np.nan, dtype=np.float64),
            where=counts[period] > 0,
        )
        if month_counts[period] != expected_counts[period]:
            logger.warning(
                "%s %s has %d of %d expected monthly rasters",
                land_type,
                period,
                month_counts[period],
                expected_counts[period],
            )

    coverage = {
        "land_type": land_type,
        "baseline_months_found": month_counts["baseline"],
        "historic_months_found": month_counts["historic"],
        "recent_months_found": month_counts["recent"],
        "reprojected_raster_count": reprojected_count,
    }
    return means, {"grid": target, "coverage": coverage}


def block_statistics(
    historic_stress: np.ndarray,
    recent_stress: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_blocks = []
    historic_blocks = []
    recent_blocks = []
    rows, cols = valid.shape
    for row in range(0, rows, block_size):
        for col in range(0, cols, block_size):
            slices = np.s_[row : row + block_size, col : col + block_size]
            block_valid = valid[slices]
            if not block_valid.any():
                continue
            block_area = area[slices]
            total_blocks.append(float(np.sum(block_area[block_valid])))
            historic_blocks.append(
                float(np.sum(block_area[block_valid & historic_stress[slices]]))
            )
            recent_blocks.append(
                float(np.sum(block_area[block_valid & recent_stress[slices]]))
            )
    return (
        np.asarray(total_blocks, dtype=float),
        np.asarray(historic_blocks, dtype=float),
        np.asarray(recent_blocks, dtype=float),
    )


def paired_spatial_block_bootstrap(
    historic_stress: np.ndarray,
    recent_stress: np.ndarray,
    area: np.ndarray,
    valid: np.ndarray,
    repetitions: int,
    block_size: int,
    seed: int,
) -> pd.DataFrame:
    total_blocks, historic_blocks, recent_blocks = block_statistics(
        historic_stress, recent_stress, area, valid, block_size
    )
    if not len(total_blocks):
        raise ValueError("No valid spatial blocks available for bootstrap.")

    observed_total_area = float(np.sum(area[valid]))
    rng = np.random.default_rng(seed)
    samples = np.empty((repetitions, 6), dtype=np.float64)
    for replicate in range(repetitions):
        selected = rng.integers(0, len(total_blocks), size=len(total_blocks))
        sampled_total = float(np.sum(total_blocks[selected]))
        sampled_historic_fraction = (
            float(np.sum(historic_blocks[selected])) / sampled_total
        )
        sampled_recent_fraction = float(np.sum(recent_blocks[selected])) / sampled_total
        historic_area = sampled_historic_fraction * observed_total_area
        recent_area = sampled_recent_fraction * observed_total_area
        samples[replicate] = [
            historic_area,
            recent_area,
            recent_area - historic_area,
            sampled_historic_fraction * 100,
            sampled_recent_fraction * 100,
            (sampled_recent_fraction - sampled_historic_fraction) * 100,
        ]
    return pd.DataFrame(
        samples,
        columns=[
            "historic_area_km2",
            "recent_area_km2",
            "difference_km2",
            "historic_percent",
            "recent_percent",
            "difference_percentage_points",
        ],
    )


def analyze_land_type(land_type: str, logger) -> tuple[list[dict], list[pd.DataFrame], dict]:
    maps, metadata = build_period_maps(land_type, logger)
    grid = metadata["grid"]
    land_fraction = fractional_land_mask(land_type, grid)
    area = cell_area_km2(grid) * land_fraction

    baseline = maps["baseline"]
    historic = maps["historic"]
    recent = maps["recent"]
    valid = (
        np.isfinite(baseline)
        & np.isfinite(historic)
        & np.isfinite(recent)
        & (baseline > MIN_BASELINE_NIRV)
        & np.isfinite(area)
        & (area > 0)
    )
    historic_decline = np.full(grid["shape"], np.nan, dtype=np.float64)
    recent_decline = np.full(grid["shape"], np.nan, dtype=np.float64)
    historic_decline[valid] = (
        (historic[valid] - baseline[valid]) / baseline[valid] * 100
    )
    recent_decline[valid] = (recent[valid] - baseline[valid]) / baseline[valid] * 100

    total_area = float(np.sum(area[valid]))
    block_approx_km = nominal_resolution_km(grid) * BLOCK_SIZE_PIXELS
    summary_rows = []
    sample_frames = []

    for threshold in THRESHOLDS:
        historic_stress = valid & (historic_decline <= threshold)
        recent_stress = valid & (recent_decline <= threshold)
        historic_area = float(np.sum(area[historic_stress]))
        recent_area = float(np.sum(area[recent_stress]))
        historic_percent = historic_area / total_area * 100
        recent_percent = recent_area / total_area * 100

        samples = paired_spatial_block_bootstrap(
            historic_stress,
            recent_stress,
            area,
            valid,
            repetitions=BOOTSTRAP_REPLICATES,
            block_size=BLOCK_SIZE_PIXELS,
            seed=RANDOM_SEED,
        )
        samples.insert(0, "replicate", np.arange(1, BOOTSTRAP_REPLICATES + 1))
        samples.insert(0, "threshold", threshold)
        samples.insert(0, "land_type", land_type)
        sample_frames.append(samples)

        difference_ci = np.percentile(samples["difference_km2"], [2.5, 97.5])
        percentage_ci = np.percentile(
            samples["difference_percentage_points"], [2.5, 97.5]
        )
        near_zero_limit = max(1.0, total_area * 0.001)
        fold_defined = historic_area > near_zero_limit
        summary_rows.append(
            {
                "land_type": land_type,
                "threshold": threshold,
                "historic_area_km2": historic_area,
                "recent_area_km2": recent_area,
                "difference_km2": recent_area - historic_area,
                "historic_percent": historic_percent,
                "recent_percent": recent_percent,
                "difference_percentage_points": recent_percent - historic_percent,
                "ci95_low_km2": float(difference_ci[0]),
                "ci95_high_km2": float(difference_ci[1]),
                "ci95_low_percentage_points": float(percentage_ci[0]),
                "ci95_high_percentage_points": float(percentage_ci[1]),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "random_seed": RANDOM_SEED,
                "block_size_pixels": BLOCK_SIZE_PIXELS,
                "block_size_approx_km": block_approx_km,
                "valid_land_area_km2": total_area,
                "fold_change": recent_area / historic_area if fold_defined else np.nan,
                "fold_change_status": (
                    "defined"
                    if fold_defined
                    else "undefined: historic area zero or near zero"
                ),
                "near_zero_historic_area_rule_km2": near_zero_limit,
                "baseline_denominator_rule": f"baseline NIRv > {MIN_BASELINE_NIRV}",
            }
        )

    cache_record = {
        f"{LAND_KEYS[land_type]}__baseline": baseline,
        f"{LAND_KEYS[land_type]}__historic": historic,
        f"{LAND_KEYS[land_type]}__recent": recent,
        f"{LAND_KEYS[land_type]}__historic_decline": historic_decline,
        f"{LAND_KEYS[land_type]}__recent_decline": recent_decline,
        f"{LAND_KEYS[land_type]}__valid": valid.astype(np.uint8),
    }
    return summary_rows, sample_frames, {
        "coverage": metadata["coverage"],
        "cache": cache_record,
    }


def plot_uncertainty_barplot(results: pd.DataFrame) -> None:
    main = results[results["threshold"] == MAIN_THRESHOLD].set_index("land_type").loc[
        LAND_TYPES
    ]
    values = main["difference_km2"].to_numpy(float)
    lower = values - main["ci95_low_km2"].to_numpy(float)
    upper = main["ci95_high_km2"].to_numpy(float) - values
    colors = ["#4C78A8" if value >= 0 else "#E45756" for value in values]

    figure, ax = plt.subplots(figsize=(10, 6))
    ax.bar(LAND_TYPES, values, color=colors, alpha=0.9)
    ax.errorbar(
        np.arange(len(LAND_TYPES)),
        values,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.4,
    )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_ylabel("Recent minus historic stressed area (km2)")
    ax.set_title(
        "NIRv stress-zone difference with paired spatial-block bootstrap\n"
        "Main threshold: 40% decline relative to baseline"
    )
    ax.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "stress_zone_uncertainty_barplot.png", dpi=300)
    plt.close(figure)


def plot_threshold_sensitivity(results: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "Grasslands": "#1B9E77",
        "Natural Forests": "#D95F02",
        "Savannas": "#7570B3",
        "Shrublands": "#E7298A",
    }
    for land in LAND_TYPES:
        subset = results[results["land_type"] == land].sort_values(
            "threshold", ascending=False
        )
        x = np.abs(subset["threshold"].to_numpy(float))
        y = subset["difference_km2"].to_numpy(float)
        lower = y - subset["ci95_low_km2"].to_numpy(float)
        upper = subset["ci95_high_km2"].to_numpy(float) - y
        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.7,
            capsize=4,
            label=land,
            color=colors[land],
        )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks([30, 40, 50], ["30%", "40%", "50%"])
    ax.set_xlabel("NIRv decline threshold")
    ax.set_ylabel("Recent minus historic stressed area (km2)")
    ax.set_title("Stress-zone threshold sensitivity with 95% block-bootstrap intervals")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(FIGURES / "stress_zone_threshold_sensitivity.png", dpi=300)
    plt.close(figure)


def write_summary(results: pd.DataFrame, coverage: pd.DataFrame) -> None:
    main = results[results["threshold"] == MAIN_THRESHOLD].copy()
    display_columns = [
        "land_type",
        "historic_area_km2",
        "recent_area_km2",
        "difference_km2",
        "historic_percent",
        "recent_percent",
        "difference_percentage_points",
        "ci95_low_km2",
        "ci95_high_km2",
        "ci95_low_percentage_points",
        "ci95_high_percentage_points",
        "fold_change_status",
    ]
    text = [
        "# Spatial aggregation uncertainty summary",
        "",
        "## Method",
        "",
        (
            "NIRv period means were calculated per 5 km analytical pixel for the "
            "2001-2011 baseline, 2012-2016 historic drought, and 2020-2024 recent "
            "period. Pixels with baseline NIRv <= 0.01 were excluded from percentage "
            "decline calculations to avoid unstable denominators."
        ),
        "",
        (
            "Stressed area was calculated at 30%, 40%, and 50% decline thresholds using "
            "latitude-aware pixel area multiplied by fractional land-type coverage. "
            f"Uncertainty used {BOOTSTRAP_REPLICATES} paired spatial-block bootstrap "
            f"replicates, {BLOCK_SIZE_PIXELS} x {BLOCK_SIZE_PIXELS} pixels per block, "
            f"and random seed {RANDOM_SEED}. The same sampled blocks were applied to "
            "historic and recent maps."
        ),
        "",
        "## Main 40% threshold",
        "",
        dataframe_to_markdown(main[display_columns]),
        "",
        "## Raster coverage",
        "",
        dataframe_to_markdown(coverage),
        "",
        "## Interpretation rule",
        "",
        (
            "Absolute km2 and percentage-point changes are reported before any ratio. "
            "Fold change is marked undefined when historic stressed area is <= 1 km2 or "
            "<= 0.1% of valid land area."
        ),
        "",
    ]
    (REPORT / "spatial_aggregation_uncertainty_summary.md").write_text(
        "\n".join(text), encoding="utf-8"
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ensure_output_directories()
    logger = configure_logger(
        "spatial_uncertainty", "spatial_aggregation_uncertainty_log.txt"
    )
    all_rows = []
    all_samples = []
    coverage_rows = []
    cache_arrays = {}

    for land_type in LAND_TYPES:
        logger.info("Processing %s", land_type)
        rows, samples, metadata = analyze_land_type(land_type, logger)
        all_rows.extend(rows)
        all_samples.extend(samples)
        coverage_rows.append(metadata["coverage"])
        cache_arrays.update(metadata["cache"])

    results = pd.DataFrame(all_rows)
    bootstrap = pd.concat(all_samples, ignore_index=True)
    coverage = pd.DataFrame(coverage_rows)

    results.to_csv(TABLES / "stress_zone_area_uncertainty.csv", index=False)
    bootstrap.to_csv(TABLES / "stress_zone_bootstrap_samples.csv", index=False)
    coverage.to_csv(TABLES / "nirv_period_raster_coverage.csv", index=False)
    np.savez_compressed(CACHE / "nirv_period_maps.npz", **cache_arrays)
    (CACHE / "nirv_period_map_settings.json").write_text(
        json.dumps(
            {
                "baseline": BASELINE,
                "historic": HISTORIC,
                "recent": RECENT,
                "thresholds": THRESHOLDS,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "block_size_pixels": BLOCK_SIZE_PIXELS,
                "random_seed": RANDOM_SEED,
                "minimum_baseline_nirv": MIN_BASELINE_NIRV,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    plot_uncertainty_barplot(results)
    plot_threshold_sensitivity(results)
    write_summary(results, coverage)
    logger.info("Wrote %d stress-zone summary rows", len(results))
    logger.info("Wrote %d bootstrap samples", len(bootstrap))


if __name__ == "__main__":
    main()
