"""Validate/copy prior detrending tables or rebuild them from thesis rasters."""

from __future__ import annotations

import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd

from fixed_utils import (
    PRIOR_ROOT,
    REQUIRED_INPUTS,
    ROOT,
    configure_logging,
    ensure_directories,
    sha256,
)


MINIMUM_ROWS = {
    "data_inventory.csv": 36,
    "monthly_landtype_series_raw.csv": 10368,
    "preprocessed_monthly_series.csv": 41472,
    "detrending_trend_removal_summary.csv": 144,
    "ewbi_detrending_sensitivity_metrics.csv": 32,
    "ewbi_detrending_monthly_series.csv": 4608,
    "state_detrending_sensitivity.csv": 16,
    "state_occupancy_by_preprocessing.csv": 256,
    "resilience_detrending_sensitivity.csv": 16,
    "forecasting_detrending_sensitivity_metrics.csv": 2304,
}


def valid_table(path, name: str) -> tuple[bool, int, str]:
    try:
        frame = pd.read_csv(path)
        rows = len(frame)
        valid = rows >= MINIMUM_ROWS[name] and len(frame.columns) > 1
        return valid, rows, "" if valid else "row/column validation failed"
    except Exception as exc:
        return False, 0, str(exc)


def rebuild(logger) -> None:
    scripts = [
        "00_inventory_and_series.py",
        "01_preprocessing_versions.py",
        "02_ewbi_detrending_sensitivity.py",
        "03_resilience_and_state_sensitivity.py",
        "04_forecasting_detrending_sensitivity.py",
    ]
    rebuild_root = ROOT / "scripts" / "rebuild_source"
    for script in scripts:
        logger.warning("Regenerating missing inputs with %s", script)
        result = subprocess.run(
            [sys.executable, str(rebuild_root / script)],
            cwd=rebuild_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        logger.info(result.stdout)
        if result.stderr:
            logger.warning(result.stderr)
        if result.returncode != 0:
            raise RuntimeError(f"Fallback rebuild failed at {script}")


def main() -> None:
    ensure_directories()
    logger = configure_logging(ROOT / "logs" / "prepare_inputs_log.txt", "prepare_inputs")
    manifest_rows: list[dict[str, object]] = []
    missing_or_invalid: list[str] = []

    for name in REQUIRED_INPUTS:
        destination = ROOT / "tables" / name
        source = PRIOR_ROOT / "tables" / name
        if not destination.exists() and source.exists():
            shutil.copy2(source, destination)
            logger.info("Copied verified prior table: %s", name)
        valid, rows, warning = valid_table(destination, name) if destination.exists() else (False, 0, "missing")
        if not valid:
            missing_or_invalid.append(name)
        manifest_rows.append(
            {
                "table": name,
                "source_path": str(source) if source.exists() else "fallback rebuild",
                "destination_path": str(destination),
                "validated": valid,
                "rows": rows,
                "sha256": sha256(destination) if destination.exists() else "",
                "copied_or_verified_at": datetime.now().isoformat(timespec="seconds"),
                "warning": warning,
            }
        )

    if missing_or_invalid:
        logger.warning("Missing or invalid inputs: %s", ", ".join(missing_or_invalid))
        rebuild(logger)
        manifest_rows = []
        for name in REQUIRED_INPUTS:
            destination = ROOT / "tables" / name
            valid, rows, warning = valid_table(destination, name)
            if not valid:
                raise RuntimeError(f"Required table remains invalid after rebuild: {name}")
            manifest_rows.append(
                {
                    "table": name,
                    "source_path": "local fallback rebuild from thesis rasters",
                    "destination_path": str(destination),
                    "validated": True,
                    "rows": rows,
                    "sha256": sha256(destination),
                    "copied_or_verified_at": datetime.now().isoformat(timespec="seconds"),
                    "warning": warning,
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(ROOT / "cache" / "input_table_manifest.csv", index=False)
    logger.info("Validated all %d input tables.", len(manifest))


if __name__ == "__main__":
    main()
