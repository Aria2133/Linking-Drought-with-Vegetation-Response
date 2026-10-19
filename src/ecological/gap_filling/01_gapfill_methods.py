"""Apply candidate gap-filling methods to raw TWS and LST land-type means."""

from __future__ import annotations

import pandas as pd

from gapfill_utils import (
    FOCUS_VARIABLES,
    LAND_FOLDERS,
    METHODS,
    METHOD_DESCRIPTIONS,
    ROOT,
    configure_logging,
    ensure_directories,
    fill_series,
    load_monthly_summary,
    series_for,
)


def main() -> None:
    ensure_directories()
    logger = configure_logging(ROOT / "logs" / "gapfill_methods_log.txt", "gapfill_methods")
    raw = load_monthly_summary()
    series_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        for variable in FOCUS_VARIABLES:
            original = series_for(raw, land_type, variable)
            for method in METHODS:
                filled, diagnostics = fill_series(original, method)
                diagnostics.update(
                    {
                        "land_type": land_type,
                        "variable": variable,
                        "description": METHOD_DESCRIPTIONS[method],
                        "degree": 2 if method == "polynomial_regression" else "",
                        "deterministic": True,
                    }
                )
                metadata_rows.append(diagnostics)
                for date in original.index:
                    series_rows.append(
                        {
                            "date": date,
                            "year": date.year,
                            "month": date.month,
                            "land_type": land_type,
                            "variable": variable,
                            "method": method,
                            "raw_value": original.loc[date],
                            "filled_value": filled.loc[date],
                            "was_missing": bool(pd.isna(original.loc[date])),
                            "value_source": "imputed" if pd.isna(original.loc[date]) else "observed",
                        }
                    )
                logger.info(
                    "%s | %s | %s | filled=%d remaining=%d unstable=%s warning=%s",
                    land_type,
                    variable,
                    method,
                    diagnostics["filled_count"],
                    diagnostics["remaining_missing"],
                    diagnostics["unstable"],
                    diagnostics["warning"],
                )

    pd.DataFrame(series_rows).to_csv(
        ROOT / "tables" / "gapfilled_monthly_series.csv", index=False
    )
    metadata = pd.DataFrame(metadata_rows)
    ordered = [
        "land_type",
        "variable",
        "method",
        "description",
        "observed_count",
        "missing_count",
        "filled_count",
        "remaining_missing",
        "fallback_count",
        "imputed_min",
        "imputed_max",
        "observed_min",
        "observed_max",
        "unstable",
        "unstable_count",
        "warning",
        "degree",
        "runtime_seconds",
        "deterministic",
    ]
    metadata[ordered].to_csv(ROOT / "tables" / "gapfill_method_metadata.csv", index=False)
    logger.info("Completed all %d method-land-variable combinations", len(metadata))


if __name__ == "__main__":
    main()
