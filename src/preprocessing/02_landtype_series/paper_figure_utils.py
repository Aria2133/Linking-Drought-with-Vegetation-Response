"""Shared paths, labels, styling, and validation helpers."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\final_paper_figures_corrected"
)
MAJOR_ROOT = Path(r"<PROJECT_ROOT>\Docs\major_revision_analyses")
DETREND_ROOT = MAJOR_ROOT / "04_detrending_sensitivity_fixed"
GAPFILL_ROOT = Path(r"<PROJECT_ROOT>\GapFilling_Comparison_Revision")
TWS_ROOT = Path(r"<PROJECT_ROOT>\TWS_Harmonization_Uncertainty_Revision")
THESIS_ROOT = Path(
    r"<DATA_ROOT>"
)
SUBMITTED_MANUSCRIPT = Path(
    r"<DOWNLOADS>\Paper Final Draft-BZ-AA_Grammar Error free version.docx"
)

SCRIPTS = ROOT / "scripts"
PNG = ROOT / "figures_png"
PDF = ROOT / "figures_pdf"
TABLES = ROOT / "tables"
LOGS = ROOT / "logs"
REPORT = ROOT / "report"

LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
VARIABLES = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP", "VPD", "SPEI", "TWS", "LST"]
VEGETATION = ["NDVI", "GNDVI", "NIRv", "GOSIF", "GPP"]
FORCING = ["VPD", "SPEI", "TWS", "LST"]
LAND_COLORS = {
    "Grasslands": "#1F77B4",
    "Natural Forests": "#FF7F0E",
    "Savannas": "#2CA02C",
    "Shrublands": "#D62728",
}
STATE_ORDER = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]

PERIODS = {
    "baseline": (pd.Timestamp("2001-01-01"), pd.Timestamp("2011-12-01")),
    "historic": (pd.Timestamp("2012-01-01"), pd.Timestamp("2016-12-01")),
    "recent": (pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-01")),
}


def ensure_directories() -> None:
    for directory in (SCRIPTS, PNG, PDF, TABLES, LOGS, REPORT):
        directory.mkdir(parents=True, exist_ok=True)


def configure_logging(filename: str, name: str) -> logging.Logger:
    ensure_directories()
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(LOGS / filename, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(figure, filename_stem: str, dpi: int = 450) -> None:
    ensure_directories()
    figure.savefig(PNG / f"{filename_stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(PDF / f"{filename_stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    display = frame.copy()
    for column in display.select_dtypes(include="number").columns:
        if pd.api.types.is_integer_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(int(value))
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{digits}f}"
            )
    rows = [
        "| " + " | ".join(map(str, display.columns)) + " |",
        "| " + " | ".join(["---"] * len(display.columns)) + " |",
    ]
    for _, row in display.iterrows():
        rows.append(
            "| "
            + " | ".join(str(value).replace("|", "/") for value in row)
            + " |"
        )
    return "\n".join(rows)


def normalize_land_type(value: str) -> str:
    mapping = {
        "NaturalForests": "Natural Forests",
        "Natural Forests": "Natural Forests",
        "Grasslands": "Grasslands",
        "Savannas": "Savannas",
        "Shrublands": "Shrublands",
    }
    return mapping.get(value, value)
