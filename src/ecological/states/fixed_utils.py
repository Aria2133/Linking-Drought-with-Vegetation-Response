"""Shared utilities for the corrected detrending sensitivity package."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\04_detrending_sensitivity_fixed"
)
PRIOR_ROOT = Path(r"<PROJECT_ROOT>\Detrending_Sensitivity_Revision")
THESIS_ROOT = Path(r"<DATA_ROOT>")
TABLES_DIR = ROOT / "tables"
FIGURES_DIR = ROOT / "figures"
REPORT_DIR = ROOT / "report"
LOGS_DIR = ROOT / "logs"
CACHE_DIR = ROOT / "cache"
LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
VERSIONS = [
    "raw_monthly",
    "deseasonalized_only",
    "month_specific_detrended",
    "ssa_detrended",
]
PERIODS = {
    "baseline": (pd.Timestamp("2001-01-01"), pd.Timestamp("2011-12-01")),
    "historic drought": (pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-01")),
    "recent": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-01")),
}
STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]
REQUIRED_INPUTS = [
    "data_inventory.csv",
    "monthly_landtype_series_raw.csv",
    "preprocessed_monthly_series.csv",
    "detrending_trend_removal_summary.csv",
    "ewbi_detrending_sensitivity_metrics.csv",
    "ewbi_detrending_monthly_series.csv",
    "state_detrending_sensitivity.csv",
    "state_occupancy_by_preprocessing.csv",
    "resilience_detrending_sensitivity.csv",
    "forecasting_detrending_sensitivity_metrics.csv",
]


def ensure_directories() -> None:
    for folder in ("scripts", "tables", "figures", "report", "logs", "cache"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)


def configure_logging(path: Path, name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(figure, stem: str | Path, dpi: int = 240) -> None:
    stem_path = Path(stem)
    if stem_path.is_absolute():
        output_stem = stem_path
    elif stem_path.parent == Path("."):
        output_stem = FIGURES_DIR / stem_path
    else:
        output_stem = ROOT / stem_path
    figure.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def classify_states(
    ewbi: pd.Series,
    nirv: pd.Series,
) -> tuple[pd.Series, float, float]:
    valid = ewbi.notna() & nirv.notna()
    ewbi_threshold = float(ewbi[valid].median())
    nirv_threshold = float(nirv[valid].median())
    labels = pd.Series(index=ewbi.index, dtype="object")
    labels.loc[valid & (ewbi >= ewbi_threshold) & (nirv >= nirv_threshold)] = (
        "Wet / Healthy"
    )
    labels.loc[valid & (ewbi >= ewbi_threshold) & (nirv < nirv_threshold)] = (
        "Wet / Suppressed"
    )
    labels.loc[valid & (ewbi < ewbi_threshold) & (nirv >= nirv_threshold)] = (
        "Dry / Resilient"
    )
    labels.loc[valid & (ewbi < ewbi_threshold) & (nirv < nirv_threshold)] = (
        "Dry / Stressed"
    )
    return labels, ewbi_threshold, nirv_threshold


def period_bounds(period: str, index: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
    if period == "all":
        return index.min(), index.max()
    return PERIODS[period]


def transition_rows(
    labels: pd.Series,
    land_type: str,
    period: str,
) -> list[dict[str, object]]:
    start, end = period_bounds(period, labels.index)
    subset = labels.loc[start:end]
    previous = subset.iloc[:-1].to_numpy()
    following = subset.iloc[1:].to_numpy()
    rows: list[dict[str, object]] = []
    for source in STATES:
        source_mask = previous == source
        denominator = int(source_mask.sum())
        for target in STATES:
            count = int(np.sum(source_mask & (following == target)))
            rows.append(
                {
                    "land_type": land_type,
                    "period": period,
                    "from_state": source,
                    "to_state": target,
                    "transition_count": count,
                    "source_state_transition_count": denominator,
                    "transition_probability": count / denominator
                    if denominator
                    else np.nan,
                    "preprocessing": "deseasonalized_only",
                }
            )
    return rows


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        rows.append(
            "| "
            + " | ".join(str(value).replace("|", "/") for value in row)
            + " |"
        )
    return "\n".join(rows)
