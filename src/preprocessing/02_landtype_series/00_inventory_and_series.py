"""Inspect source rasters and build complete land-type monthly mean series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from detrending_utils import (
    DATE_INDEX,
    LAND_FOLDERS,
    ROOT,
    VARIABLES,
    configure_logging,
    discover_files,
    ensure_directories,
    monthwise_fill,
    read_raster_mean,
)


def main() -> None:
    ensure_directories()
    logger = configure_logging(ROOT / "logs" / "inventory_log.txt", "detrending_inventory")
    inventory_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        for variable in VARIABLES:
            files, file_warnings = discover_files(land_type, variable)
            for warning in file_warnings:
                logger.warning(warning)
            raw = pd.Series(index=DATE_INDEX, dtype=float)
            paths: dict[pd.Timestamp, str] = {}
            periods: dict[pd.Timestamp, str] = {}
            actions: dict[pd.Timestamp, str] = {}
            metadata: dict[str, object] = {}
            read_failures = 0
            for date, path in sorted(files.items()):
                try:
                    value, meta = read_raster_mean(path, variable, date)
                    raw.loc[date] = value
                    paths[date] = str(path)
                    periods[date] = "test" if "Test Data" in str(path) else "train"
                    actions[date] = str(meta["unit_action"])
                    if not metadata:
                        metadata = meta
                except Exception as exc:
                    read_failures += 1
                    logger.warning("Failed to read %s: %s", path, exc)

            missing_before = int(raw.isna().sum())
            if variable in {"TWS", "LST"}:
                analysis, fallback_count = monthwise_fill(raw)
                gapfill_method = (
                    "current_monthwise_cross_year: same-calendar-month interpolation "
                    "with monthly-mean boundary fallback"
                )
            else:
                analysis, fallback_count = raw.copy(), 0
                gapfill_method = "not required"

            for date in DATE_INDEX:
                monthly_rows.append(
                    {
                        "date": date,
                        "year": date.year,
                        "month": date.month,
                        "land_type": land_type,
                        "variable": variable,
                        "raw_value": raw.loc[date],
                        "analysis_value": analysis.loc[date],
                        "was_gap_filled": bool(pd.isna(raw.loc[date]) and pd.notna(analysis.loc[date])),
                        "gapfill_method": gapfill_method,
                        "source_period": periods.get(date, "missing"),
                        "source_path": paths.get(date, ""),
                        "unit_action": actions.get(date, ""),
                    }
                )

            observed = raw.dropna()
            inventory_rows.append(
                {
                    "land_type": land_type,
                    "variable": variable,
                    "expected_months": len(DATE_INDEX),
                    "raw_files_detected": len(files),
                    "observed_months": int(raw.notna().sum()),
                    "missing_months_before_gapfill": missing_before,
                    "missing_months_after_gapfill": int(analysis.isna().sum()),
                    "gapfilled_months": int((raw.isna() & analysis.notna()).sum()),
                    "boundary_fallback_months": fallback_count,
                    "detected_start": observed.index.min().strftime("%Y-%m") if len(observed) else "",
                    "detected_end": observed.index.max().strftime("%Y-%m") if len(observed) else "",
                    "crs": metadata.get("crs", ""),
                    "resolution_x": metadata.get("resolution_x", np.nan),
                    "resolution_y": metadata.get("resolution_y", np.nan),
                    "rows": metadata.get("rows", np.nan),
                    "cols": metadata.get("cols", np.nan),
                    "nodata": metadata.get("nodata", np.nan),
                    "unit_harmonization": (
                        "MOD17 low-magnitude files converted from kg C m-2 month-1 "
                        "to g C m-2 day-1"
                        if variable == "GPP"
                        else "none"
                    ),
                    "gapfill_method": gapfill_method,
                    "read_failures": read_failures,
                    "derivative_folders_excluded": True,
                }
            )
            logger.info(
                "%s | %s | observed=%d missing=%d gapfilled=%d",
                land_type,
                variable,
                int(raw.notna().sum()),
                missing_before,
                int((raw.isna() & analysis.notna()).sum()),
            )

    inventory = pd.DataFrame(inventory_rows)
    monthly = pd.DataFrame(monthly_rows)
    inventory.to_csv(ROOT / "tables" / "data_inventory.csv", index=False)
    monthly.to_csv(ROOT / "tables" / "monthly_landtype_series_raw.csv", index=False)
    monthly.to_csv(ROOT / "cache" / "monthly_landtype_series_snapshot.csv", index=False)
    logger.info("Wrote data inventory and %d monthly-series rows", len(monthly))
    logger.info("Original data were read only; derivative folders were excluded.")


if __name__ == "__main__":
    main()
