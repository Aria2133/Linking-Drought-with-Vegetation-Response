"""Compare median-quadrant ecosystem states under each EWBI gap-filling method."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from gapfill_utils import (
    DATE_INDEX,
    LAND_FOLDERS,
    METHODS,
    ROOT,
    configure_logging,
    ensure_directories,
    load_monthly_summary,
    monthly_median_anomaly,
    series_for,
)


CURRENT = "current_monthwise_cross_year"
STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]


def classify(ewbi: pd.Series, nirv_anomaly: pd.Series) -> tuple[pd.Series, float, float]:
    ewbi_threshold = float(ewbi.median())
    nirv_threshold = float(nirv_anomaly.median())
    labels = pd.Series(index=ewbi.index, dtype="object")
    labels[(ewbi >= ewbi_threshold) & (nirv_anomaly >= nirv_threshold)] = "Wet / Healthy"
    labels[(ewbi >= ewbi_threshold) & (nirv_anomaly < nirv_threshold)] = "Wet / Suppressed"
    labels[(ewbi < ewbi_threshold) & (nirv_anomaly >= nirv_threshold)] = "Dry / Resilient"
    labels[(ewbi < ewbi_threshold) & (nirv_anomaly < nirv_threshold)] = "Dry / Stressed"
    return labels, ewbi_threshold, nirv_threshold


def persistence(labels: pd.Series, state: str = "Dry / Stressed") -> float:
    previous = labels.iloc[:-1].to_numpy()
    following = labels.iloc[1:].to_numpy()
    starts = previous == state
    if not starts.any():
        return np.nan
    return float(np.mean(following[starts] == state))


def create_figure(sensitivity: pd.DataFrame) -> None:
    alternatives = sensitivity[sensitivity["method"] != CURRENT]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    x = np.arange(len(LAND_FOLDERS))
    width = 0.15
    for offset, method in enumerate([method for method in METHODS if method != CURRENT]):
        values = [
            alternatives[
                (alternatives["land_type"] == land_type)
                & (alternatives["method"] == method)
            ]["state_agreement_percent"].iloc[0]
            for land_type in LAND_FOLDERS
        ]
        ax.bar(
            x + (offset - 2) * width,
            values,
            width,
            label=method.replace("_", " "),
        )
    ax.set_xticks(x, LAND_FOLDERS.keys())
    ax.set_ylim(0, 100)
    ax.axhline(90, color="black", linestyle="--", linewidth=0.9, label="90% stability threshold")
    ax.set_ylabel("Agreement with current classification (%)")
    ax.set_title("Ecosystem-state sensitivity to TWS/LST gap filling")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "state_gapfill_agreement_barplot.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_directories()
    logger = configure_logging(
        ROOT / "logs" / "state_gapfill_sensitivity_log.txt", "state_gapfill"
    )
    ewbi_frame = pd.read_csv(
        ROOT / "tables" / "ewbi_gapfill_monthly_series.csv", parse_dates=["date"]
    )
    raw = load_monthly_summary()
    occupancy_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        nirv = series_for(raw, land_type, "NIRv")
        nirv_anomaly = monthly_median_anomaly(nirv)
        classifications: dict[str, pd.Series] = {}
        thresholds: dict[str, tuple[float, float]] = {}
        for method in METHODS:
            ewbi = (
                ewbi_frame[
                    (ewbi_frame["land_type"] == land_type)
                    & (ewbi_frame["method"] == method)
                ]
                .set_index("date")["ewbi"]
                .reindex(DATE_INDEX)
            )
            labels, ewbi_threshold, nirv_threshold = classify(ewbi, nirv_anomaly)
            classifications[method] = labels
            thresholds[method] = (ewbi_threshold, nirv_threshold)
            for state in STATES:
                count = int((labels == state).sum())
                occupancy_rows.append(
                    {
                        "land_type": land_type,
                        "method": method,
                        "state": state,
                        "month_count": count,
                        "occupancy_percent": 100.0 * count / int(labels.notna().sum()),
                        "ewbi_median_threshold": ewbi_threshold,
                        "nirv_anomaly_median_threshold": nirv_threshold,
                        "dry_stressed_persistence_probability": persistence(labels),
                    }
                )

        reference = classifications[CURRENT]
        reference_occupancy = reference.value_counts(normalize=True) * 100.0
        reference_persistence = persistence(reference)
        for method in METHODS:
            comparison = classifications[method]
            valid = reference.notna() & comparison.notna()
            agreement = 100.0 * float(np.mean(reference[valid] == comparison[valid]))
            kappa = float(cohen_kappa_score(reference[valid], comparison[valid]))
            comparison_occupancy = comparison.value_counts(normalize=True) * 100.0
            occupancy_differences = {
                state: float(comparison_occupancy.get(state, 0.0) - reference_occupancy.get(state, 0.0))
                for state in STATES
            }
            comparison_persistence = persistence(comparison)
            max_occupancy_change = max(abs(value) for value in occupancy_differences.values())
            persistence_change = comparison_persistence - reference_persistence
            stable = (
                agreement >= 90.0
                and kappa >= 0.80
                and max_occupancy_change <= 5.0
                and abs(persistence_change) <= 0.10
            )
            sensitivity_rows.append(
                {
                    "land_type": land_type,
                    "reference_method": CURRENT,
                    "method": method,
                    "state_agreement_percent": agreement,
                    "cohen_kappa": kappa,
                    "valid_months": int(valid.sum()),
                    "max_absolute_occupancy_change_percentage_points": max_occupancy_change,
                    "wet_healthy_occupancy_change_percentage_points": occupancy_differences[
                        "Wet / Healthy"
                    ],
                    "wet_suppressed_occupancy_change_percentage_points": occupancy_differences[
                        "Wet / Suppressed"
                    ],
                    "dry_resilient_occupancy_change_percentage_points": occupancy_differences[
                        "Dry / Resilient"
                    ],
                    "dry_stressed_occupancy_change_percentage_points": occupancy_differences[
                        "Dry / Stressed"
                    ],
                    "reference_dry_stressed_persistence_probability": reference_persistence,
                    "comparison_dry_stressed_persistence_probability": comparison_persistence,
                    "dry_stressed_persistence_change_probability": persistence_change,
                    "stability_assessment": "stable" if stable else "potentially meaningful change",
                }
            )
        logger.info("Classified and compared ecosystem states for %s", land_type)

    occupancy = pd.DataFrame(occupancy_rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    occupancy.to_csv(
        ROOT / "tables" / "state_occupancy_by_gapfill_method.csv", index=False
    )
    sensitivity.to_csv(ROOT / "tables" / "state_gapfill_sensitivity.csv", index=False)
    create_figure(sensitivity)
    counts = sensitivity[sensitivity["method"] != CURRENT]["stability_assessment"].value_counts()
    logger.info("State-classification stability assessments: %s", counts.to_dict())
    logger.info(
        "NIRv resistance/recovery inputs were not gap-filled or changed in this analysis."
    )


if __name__ == "__main__":
    main()
