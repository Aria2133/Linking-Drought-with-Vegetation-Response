"""Run or verify the complete Reviewer #1 detrending sensitivity package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

import pandas as pd

from detrending_utils import ROOT, ensure_directories


PIPELINE = [
    ("Data inventory and series", "00_inventory_and_series.py", [
        "tables/data_inventory.csv", "tables/monthly_landtype_series_raw.csv"
    ]),
    ("Preprocessing versions", "01_preprocessing_versions.py", [
        "tables/preprocessed_monthly_series.csv", "tables/detrending_trend_removal_summary.csv",
        "figures/detrending_series_comparison.png", "figures/trend_removal_heatmap.png"
    ]),
    ("EWBI sensitivity", "02_ewbi_detrending_sensitivity.py", [
        "tables/ewbi_detrending_sensitivity_metrics.csv",
        "tables/ewbi_detrending_monthly_series.csv",
        "figures/ewbi_detrending_timeseries.png",
        "figures/ewbi_detrending_metric_heatmap.png",
        "figures/ewbi_period_separation_by_preprocessing.png",
    ]),
    ("Resilience and state sensitivity", "03_resilience_and_state_sensitivity.py", [
        "tables/state_detrending_sensitivity.csv",
        "tables/state_occupancy_by_preprocessing.csv",
        "tables/resilience_detrending_sensitivity.csv",
        "figures/state_detrending_agreement_barplot.png",
        "figures/state_occupancy_shift_heatmap.png",
    ]),
    ("Lightweight forecasting sensitivity", "04_forecasting_detrending_sensitivity.py", [
        "tables/forecasting_detrending_sensitivity_metrics.csv",
        "figures/forecasting_detrending_metric_heatmap.png",
        "figures/forecasting_detrending_delta_barplot.png",
    ]),
    ("Standalone report", "05_make_detrending_report.py", [
        "report/detrending_sensitivity_report.md",
        "report/detrending_sensitivity_report.pdf",
    ]),
]


def write_inventory(rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(ROOT / "tables" / "reproducibility_inventory.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Verify outputs from scripts already run in order and run only the report step.",
    )
    args = parser.parse_args()
    ensure_directories()
    log_path = ROOT / "logs" / "run_all_detrending_log.txt"
    rows: list[dict[str, object]] = []
    started = datetime.now()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Detrending sensitivity package started: {started.isoformat()}\n")
        log.write("Original rasters remain read only; no full ML/DL script is invoked.\n\n")
        for issue, script, outputs in PIPELINE:
            step_start = datetime.now()
            should_run = not args.verify_existing or script == "05_make_detrending_report.py"
            if should_run:
                log.write(f"START {script} | {step_start.isoformat()}\n")
                result = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / script)],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                log.write(result.stdout or "")
                log.write(result.stderr or "")
                return_code = result.returncode
                execution_note = "executed"
            else:
                return_code = 0
                execution_note = "verified existing output from prior ordered execution"
                log.write(f"VERIFY {script} | {step_start.isoformat()}\n")
            existing = [output for output in outputs if (ROOT / output).exists()]
            missing = [output for output in outputs if not (ROOT / output).exists()]
            status = "completed" if return_code == 0 and not missing else "failed"
            rows.append(
                {
                    "reviewer_issue": issue,
                    "script": script,
                    "status": status,
                    "outputs_created": "; ".join(existing),
                    "notes": (
                        f"{execution_note}; return code {return_code}; "
                        f"elapsed {(datetime.now() - step_start).total_seconds():.1f} seconds."
                    ),
                    "warnings": "Missing expected outputs: " + "; ".join(missing) if missing else "",
                }
            )
            write_inventory(rows)
            log.write(f"END {script} | status={status}\n\n")
            log.flush()
            if status == "failed":
                raise RuntimeError(f"{script} failed; see {log_path}")
        log.write(
            f"Package completed: {datetime.now().isoformat()} | "
            f"elapsed={(datetime.now() - started).total_seconds():.1f}s\n"
        )


if __name__ == "__main__":
    main()
