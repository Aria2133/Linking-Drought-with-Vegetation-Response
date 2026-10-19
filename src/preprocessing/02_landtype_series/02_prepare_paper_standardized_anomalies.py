"""Prepare date-aligned, baseline-standardized series and source provenance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_figure_utils import (  # noqa: E402
    DETREND_ROOT,
    GAPFILL_ROOT,
    LAND_TYPES,
    MAJOR_ROOT,
    PERIODS,
    SUBMITTED_MANUSCRIPT,
    TABLES,
    TWS_ROOT,
    VARIABLES,
    configure_logging,
    ensure_directories,
    normalize_land_type,
    sha256,
)


SOURCE_FILES = [
    DETREND_ROOT / "tables" / "preprocessed_monthly_series.csv",
    DETREND_ROOT / "tables" / "main_state_transition_probabilities_deseasonalized.csv",
    TWS_ROOT / "tables" / "stress_zone_area_uncertainty.csv",
    TWS_ROOT / "cache" / "nirv_period_maps.npz",
    GAPFILL_ROOT / "tables" / "gapfill_decision_summary.csv",
    MAJOR_ROOT / "tables_revision" / "gndvi_fixed_directory_validation.csv",
    MAJOR_ROOT / "tables_revision" / "forecasting_rolling_origin_summary.csv",
    SUBMITTED_MANUSCRIPT,
]


def build_standardized_series(logger) -> pd.DataFrame:
    source = DETREND_ROOT / "tables" / "preprocessed_monthly_series.csv"
    data = pd.read_csv(source, parse_dates=["date"])
    data["land_type"] = data["land_type"].map(normalize_land_type)
    data = data[
        (data["preprocessing"] == "deseasonalized_only")
        & data["variable"].isin(VARIABLES)
        & data["land_type"].isin(LAND_TYPES)
    ].copy()

    expected = len(LAND_TYPES) * len(VARIABLES) * 288
    if len(data) != expected:
        raise ValueError(f"Expected {expected} deseasonalized rows, found {len(data)}.")
    duplicated = data.duplicated(["date", "land_type", "variable"]).sum()
    if duplicated:
        raise ValueError(f"Found {duplicated} duplicate land-type variable months.")

    rows = []
    baseline_start, baseline_end = PERIODS["baseline"]
    for (land_type, variable), group in data.groupby(["land_type", "variable"]):
        group = group.sort_values("date").copy()
        baseline = group.loc[
            group["date"].between(baseline_start, baseline_end), "processed_value"
        ]
        baseline_mean = float(baseline.mean())
        baseline_sd = float(baseline.std(ddof=1))
        if not np.isfinite(baseline_sd) or baseline_sd <= 0:
            raise ValueError(f"Invalid baseline SD for {land_type} / {variable}.")
        group["standardized_anomaly"] = (
            group["processed_value"] - baseline_mean
        ) / baseline_sd
        group["baseline_anomaly_mean"] = baseline_mean
        group["baseline_anomaly_sd"] = baseline_sd
        group["standardization_reference"] = (
            "2001-2011 deseasonalized-anomaly mean and standard deviation"
        )
        rows.append(
            group[
                [
                    "date",
                    "year",
                    "month",
                    "land_type",
                    "variable",
                    "original_value",
                    "processed_value",
                    "standardized_anomaly",
                    "baseline_anomaly_mean",
                    "baseline_anomaly_sd",
                    "seasonal_reference",
                    "standardization_reference",
                ]
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(TABLES / "paper_standardized_anomaly_series.csv", index=False)

    gndvi = result[result["variable"] == "GNDVI"]
    if len(gndvi) != len(LAND_TYPES) * 288 or gndvi["standardized_anomaly"].isna().any():
        raise ValueError("QA-controlled GNDVI is incomplete in the paper series.")
    logger.info("Prepared %d standardized anomaly observations.", len(result))
    return result


def build_manifest(logger) -> None:
    rows = []
    for path in SOURCE_FILES:
        exists = path.exists()
        rows.append(
            {
                "source_file": str(path),
                "exists": exists,
                "size_bytes": path.stat().st_size if exists else np.nan,
                "modified_time": (
                    pd.Timestamp(path.stat().st_mtime, unit="s").isoformat()
                    if exists
                    else ""
                ),
                "sha256": sha256(path) if exists else "",
                "role": {
                    "preprocessed_monthly_series.csv": "Figures 2, 3, 5, and 6",
                    "main_state_transition_probabilities_deseasonalized.csv": "Figure 9",
                    "stress_zone_area_uncertainty.csv": "Figure 8 uncertainty annotations",
                    "nirv_period_maps.npz": "Figure 8 spatial maps",
                    "gapfill_decision_summary.csv": "Gap-fill interpretation provenance",
                    "gndvi_fixed_directory_validation.csv": "QA-controlled GNDVI provenance",
                    "forecasting_rolling_origin_summary.csv": "Figure 7 deferral audit",
                    SUBMITTED_MANUSCRIPT.name: "Submitted captions and figure numbering",
                }.get(path.name, "Input provenance"),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(TABLES / "source_data_manifest.csv", index=False)
    missing = manifest.loc[~manifest["exists"], "source_file"].tolist()
    if missing:
        raise FileNotFoundError("Missing required source files: " + "; ".join(missing))
    logger.info("Validated %d source inputs.", len(manifest))


def main() -> None:
    ensure_directories()
    logger = configure_logging("prepare_paper_data_log.txt", "prepare_paper_data")
    build_manifest(logger)
    build_standardized_series(logger)


if __name__ == "__main__":
    main()
