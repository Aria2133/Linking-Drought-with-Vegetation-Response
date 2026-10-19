from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CACHE,
    DATE_END,
    DATE_START,
    LAND_TYPES,
    NATIVE_DATASET_TABLE,
    TABLES,
    TEST_ROOT,
    VARIABLES,
)
from revision_utils import discover_files, read_raster


def file_checksum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_raster(path: Path, variable: str, date: pd.Timestamp) -> dict:
    array, meta = read_raster(path, variable, date)
    valid = array[np.isfinite(array)]
    if valid.size:
        quantiles = np.nanpercentile(valid, [25, 50, 75])
        values = {
            "mean": float(np.nanmean(valid)),
            "median": float(quantiles[1]),
            "q25": float(quantiles[0]),
            "q75": float(quantiles[2]),
            "std": float(np.nanstd(valid, ddof=1)) if valid.size > 1 else 0.0,
            "valid_count": int(valid.size),
        }
    else:
        values = {
            "mean": np.nan,
            "median": np.nan,
            "q25": np.nan,
            "q75": np.nan,
            "std": np.nan,
            "valid_count": 0,
        }
    values.update(
        {
            "rows": meta["shape"][0],
            "cols": meta["shape"][1],
            "resolution_x_deg": meta["resolution_x"],
            "resolution_y_deg": meta["resolution_y"],
            "unit_action": meta["unit_action"],
        }
    )
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rescan every GeoTIFF.")
    args = parser.parse_args()
    expected_dates = pd.date_range(DATE_START, DATE_END, freq="MS")
    summary_path = CACHE / "monthly_landtype_summary.csv"
    inventory_path = TABLES / "source_file_inventory.csv"
    if summary_path.exists() and inventory_path.exists() and not args.rebuild:
        print("Reusing existing raster summary cache. Use --rebuild to rescan.")
        summary = pd.read_csv(summary_path, parse_dates=["date"])
        inventory = pd.read_csv(inventory_path, parse_dates=["date", "modified"])
    else:
        summary_rows = []
        inventory_rows = []
        print(f"Using test data root: {TEST_ROOT}")
        for land_type in LAND_TYPES:
            for variable in VARIABLES:
                files = discover_files(land_type, variable)
                print(f"{land_type:16s} {variable:5s}: {len(files):3d} files")
                for date in expected_dates:
                    path = files.get(date)
                    if path is None:
                        summary_rows.append(
                            {
                                "date": date,
                                "year": date.year,
                                "month": date.month,
                                "land_type": land_type,
                                "variable": variable,
                                "path": "",
                                "source_period": "missing",
                                "mean": np.nan,
                                "median": np.nan,
                                "q25": np.nan,
                                "q75": np.nan,
                                "std": np.nan,
                                "valid_count": 0,
                                "rows": np.nan,
                                "cols": np.nan,
                                "resolution_x_deg": np.nan,
                                "resolution_y_deg": np.nan,
                                "unit_action": "",
                            }
                        )
                        continue
                    stats_row = summarize_raster(path, variable, date)
                    source_period = "test" if date.year >= 2022 else "train"
                    summary_rows.append(
                        {
                            "date": date,
                            "year": date.year,
                            "month": date.month,
                            "land_type": land_type,
                            "variable": variable,
                            "path": str(path),
                            "source_period": source_period,
                            **stats_row,
                        }
                    )
                    inventory_rows.append(
                        {
                            "land_type": land_type,
                            "variable": variable,
                            "date": date,
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "modified": pd.Timestamp(path.stat().st_mtime, unit="s"),
                        }
                    )
        summary = pd.DataFrame(summary_rows)
        inventory = pd.DataFrame(inventory_rows)
        summary.to_csv(summary_path, index=False)
        inventory.to_csv(inventory_path, index=False)

    coverage_source = summary.assign(
        available=summary["valid_count"].gt(0),
        available_date=summary["date"].where(summary["valid_count"].gt(0)),
    )
    coverage = (
        coverage_source
        .groupby(["land_type", "variable"], as_index=False)
        .agg(
            available_months=("available", "sum"),
            expected_months=("available", "size"),
            first_date=("available_date", "min"),
            last_date=("available_date", "max"),
            median_valid_pixels=("valid_count", "median"),
            median_resolution_x_deg=("resolution_x_deg", "median"),
        )
    )
    coverage["coverage_fraction"] = coverage["available_months"] / coverage["expected_months"]
    coverage.to_csv(TABLES / "data_coverage_by_landtype_variable.csv", index=False)

    native = pd.DataFrame(
        NATIVE_DATASET_TABLE,
        columns=[
            "variable",
            "source",
            "native_spatial_resolution",
            "native_temporal_resolution",
            "temporal_aggregation",
            "analysis_grid",
        ],
    )
    native["interpretation_note"] = ""
    native.loc[native["variable"] == "TWS", "interpretation_note"] = (
        "The 5 km grid is an analytical harmonization grid; effective TWS "
        "information remains at GRACE/GRACE-FO mascon scale."
    )
    native.to_csv(TABLES / "native_resolution_and_harmonization_table.csv", index=False)

    checksums = []
    for path in [CACHE / "monthly_landtype_summary.csv", TABLES / "source_file_inventory.csv"]:
        checksums.append({"file": str(path), "sha256": file_checksum(path)})
    pd.DataFrame(checksums).to_csv(CACHE / "cache_checksums.csv", index=False)

    with pd.ExcelWriter(TABLES / "data_inventory_revision.xlsx") as writer:
        coverage.to_excel(writer, sheet_name="coverage", index=False)
        native.to_excel(writer, sheet_name="native_resolution", index=False)
        inventory.to_excel(writer, sheet_name="files", index=False)

    print(f"Saved {len(summary):,} monthly summary rows.")


if __name__ == "__main__":
    main()
