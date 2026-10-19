"""Evaluate NIRv resilience and EWBI-NIRv ecosystem-state sensitivity."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from detrending_utils import LAND_FOLDERS, PERIODS, ROOT, VERSIONS, configure_logging, ensure_directories


REFERENCE = "deseasonalized_only"
STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]


def classify(ewbi: pd.Series, nirv: pd.Series) -> pd.Series:
    valid = ewbi.notna() & nirv.notna()
    ewbi_cut = float(ewbi[valid].median())
    nirv_cut = float(nirv[valid].median())
    labels = pd.Series(index=ewbi.index, dtype="object")
    labels.loc[valid & (ewbi >= ewbi_cut) & (nirv >= nirv_cut)] = "Wet / Healthy"
    labels.loc[valid & (ewbi >= ewbi_cut) & (nirv < nirv_cut)] = "Wet / Suppressed"
    labels.loc[valid & (ewbi < ewbi_cut) & (nirv >= nirv_cut)] = "Dry / Resilient"
    labels.loc[valid & (ewbi < ewbi_cut) & (nirv < nirv_cut)] = "Dry / Stressed"
    return labels


def persistence(labels: pd.Series) -> float:
    previous = labels.iloc[:-1].to_numpy()
    following = labels.iloc[1:].to_numpy()
    starts = previous == "Dry / Stressed"
    return float(np.mean(following[starts] == "Dry / Stressed")) if starts.any() else np.nan


def period_mean(series: pd.Series, name: str) -> float:
    start, end = PERIODS[name]
    return float(series.loc[start:end].mean())


def create_agreement_figure(sensitivity: pd.DataFrame) -> None:
    figure, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(LAND_FOLDERS))
    width = 0.2
    for offset, version in enumerate(VERSIONS):
        values = [
            sensitivity[
                (sensitivity["land_type"] == land_type)
                & (sensitivity["preprocessing"] == version)
            ]["state_agreement_percent"].iloc[0]
            for land_type in LAND_FOLDERS
        ]
        ax.bar(x + (offset - 1.5) * width, values, width, label=version.replace("_", " "))
    ax.set_xticks(x, LAND_FOLDERS.keys())
    ax.set_ylim(0, 105)
    ax.axhline(90, color="black", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Agreement with deseasonalized-only states (%)")
    ax.set_title("Ecosystem-state sensitivity to detrending")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "state_detrending_agreement_barplot.png", dpi=220)
    plt.close(figure)


def create_occupancy_heatmap(sensitivity: pd.DataFrame) -> None:
    alternatives = sensitivity[sensitivity["preprocessing"] != REFERENCE].copy()
    alternatives["row"] = alternatives["land_type"].str.replace(
        "Natural Forests", "Forests", regex=False
    ) + "\n" + alternatives["preprocessing"].str.replace("_", " ", regex=False)
    columns = [
        "wet_healthy_change_percentage_points",
        "wet_suppressed_change_percentage_points",
        "dry_resilient_change_percentage_points",
        "dry_stressed_change_percentage_points",
    ]
    matrix = alternatives.set_index("row")[columns]
    limit = max(5.0, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float)))))
    figure, ax = plt.subplots(figsize=(9, 9))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(4), ["Wet/Healthy", "Wet/Suppressed", "Dry/Resilient", "Dry/Stressed"], rotation=25, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index, fontsize=8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iloc[row, col]
            ax.text(col, row, f"{value:+.1f}", ha="center", va="center", fontsize=7)
    ax.set_title("State-occupancy shift from deseasonalized-only reference")
    figure.colorbar(image, ax=ax, label="Change (percentage points)")
    figure.tight_layout()
    figure.savefig(ROOT / "figures" / "state_occupancy_shift_heatmap.png", dpi=220)
    plt.close(figure)


def main() -> None:
    ensure_directories()
    logger = configure_logging(ROOT / "logs" / "state_resilience_log.txt", "state_resilience")
    ewbi_frame = pd.read_csv(
        ROOT / "tables" / "ewbi_detrending_monthly_series.csv", parse_dates=["date"]
    )
    processed = pd.read_csv(
        ROOT / "tables" / "preprocessed_monthly_series.csv", parse_dates=["date"]
    )
    state_rows: list[dict[str, object]] = []
    occupancy_rows: list[dict[str, object]] = []
    resilience_rows: list[dict[str, object]] = []

    for land_type in LAND_FOLDERS:
        labels_by_version: dict[str, pd.Series] = {}
        nirv_versions: dict[str, pd.Series] = {}
        for version in VERSIONS:
            ewbi = (
                ewbi_frame[
                    (ewbi_frame["land_type"] == land_type)
                    & (ewbi_frame["preprocessing"] == version)
                ]
                .set_index("date")["ewbi"]
                .sort_index()
            )
            nirv = (
                processed[
                    (processed["land_type"] == land_type)
                    & (processed["variable"] == "NIRv")
                    & (processed["preprocessing"] == version)
                ]
                .set_index("date")["processed_value"]
                .sort_index()
            )
            labels = classify(ewbi, nirv)
            labels_by_version[version] = labels
            nirv_versions[version] = nirv
            for period, dates in {"all": (labels.index.min(), labels.index.max()), **PERIODS}.items():
                start, end = dates
                period_labels = labels.loc[start:end]
                valid_count = int(period_labels.notna().sum())
                for state in STATES:
                    count = int((period_labels == state).sum())
                    occupancy_rows.append(
                        {
                            "land_type": land_type,
                            "preprocessing": version,
                            "period": period,
                            "state": state,
                            "month_count": count,
                            "occupancy_percent": 100.0 * count / valid_count if valid_count else np.nan,
                            "valid_months": valid_count,
                            "dry_stressed_persistence_probability": persistence(labels)
                            if period == "all"
                            else persistence(period_labels),
                        }
                    )

        reference = labels_by_version[REFERENCE]
        reference_occupancy = reference.value_counts(normalize=True) * 100.0
        reference_persistence = persistence(reference)
        for version in VERSIONS:
            comparison = labels_by_version[version]
            valid = reference.notna() & comparison.notna()
            comparison_occupancy = comparison.value_counts(normalize=True) * 100.0
            changes = {
                state: float(comparison_occupancy.get(state, 0.0) - reference_occupancy.get(state, 0.0))
                for state in STATES
            }
            row = {
                "land_type": land_type,
                "reference_preprocessing": REFERENCE,
                "preprocessing": version,
                "state_agreement_percent": 100.0 * float(np.mean(reference[valid] == comparison[valid])),
                "cohen_kappa": float(cohen_kappa_score(reference[valid], comparison[valid])),
                "valid_months": int(valid.sum()),
                "wet_healthy_change_percentage_points": changes["Wet / Healthy"],
                "wet_suppressed_change_percentage_points": changes["Wet / Suppressed"],
                "dry_resilient_change_percentage_points": changes["Dry / Resilient"],
                "dry_stressed_change_percentage_points": changes["Dry / Stressed"],
                "max_absolute_occupancy_change_percentage_points": max(abs(v) for v in changes.values()),
                "reference_dry_stressed_persistence_probability": reference_persistence,
                "comparison_dry_stressed_persistence_probability": persistence(comparison),
                "dry_stressed_persistence_change_probability": persistence(comparison) - reference_persistence,
            }
            for period in PERIODS:
                start, end = PERIODS[period]
                row[f"{period}_dry_stressed_occupancy_percent"] = 100.0 * float(
                    (comparison.loc[start:end] == "Dry / Stressed").mean()
                )
            state_rows.append(row)

        raw_nirv = (
            processed[
                (processed["land_type"] == land_type)
                & (processed["variable"] == "NIRv")
                & (processed["preprocessing"] == "raw_monthly")
            ]
            .set_index("date")["original_value"]
            .sort_index()
        )
        raw_baseline = period_mean(raw_nirv, "baseline")
        raw_historic = period_mean(raw_nirv, "historic")
        raw_recent = period_mean(raw_nirv, "recent")
        raw_decline = raw_historic < raw_baseline
        raw_recovery = raw_recent > raw_historic
        for version in VERSIONS:
            nirv = nirv_versions[version]
            processed_baseline = period_mean(nirv, "baseline")
            processed_historic = period_mean(nirv, "historic")
            processed_recent = period_mean(nirv, "recent")
            processed_decline = processed_historic < processed_baseline
            processed_recovery = processed_recent > processed_historic
            resilience_rows.append(
                {
                    "land_type": land_type,
                    "preprocessing": version,
                    "raw_nirv_baseline_mean": raw_baseline,
                    "raw_nirv_historic_mean": raw_historic,
                    "raw_nirv_recent_mean": raw_recent,
                    "raw_historic_change_from_baseline": raw_historic - raw_baseline,
                    "raw_historic_change_percent": 100.0 * (raw_historic - raw_baseline) / raw_baseline,
                    "raw_recent_recovery_from_historic": raw_recent - raw_historic,
                    "raw_recent_recovery_percent": 100.0 * (raw_recent - raw_historic) / raw_historic,
                    "processed_nirv_baseline_mean": processed_baseline,
                    "processed_nirv_historic_mean": processed_historic,
                    "processed_nirv_recent_mean": processed_recent,
                    "raw_historic_decline_interpretation": raw_decline,
                    "raw_recent_recovery_interpretation": raw_recovery,
                    "processed_historic_decline_interpretation": processed_decline,
                    "processed_recent_recovery_interpretation": processed_recovery,
                    "stress_recovery_interpretation_matches_raw": (
                        raw_decline == processed_decline and raw_recovery == processed_recovery
                    ),
                    "raw_resistance_recovery_metrics_affected_by_detrending": False,
                    "note": (
                        "Primary NIRv resistance/recovery metrics use raw period changes and "
                        "therefore remain invariant; processed means are sensitivity diagnostics."
                    ),
                }
            )
        logger.info("Completed state and resilience sensitivity for %s", land_type)

    states = pd.DataFrame(state_rows)
    occupancy = pd.DataFrame(occupancy_rows)
    resilience = pd.DataFrame(resilience_rows)
    states.to_csv(ROOT / "tables" / "state_detrending_sensitivity.csv", index=False)
    occupancy.to_csv(ROOT / "tables" / "state_occupancy_by_preprocessing.csv", index=False)
    resilience.to_csv(ROOT / "tables" / "resilience_detrending_sensitivity.csv", index=False)
    create_agreement_figure(states)
    create_occupancy_heatmap(states)
    alternatives = states[states["preprocessing"] != REFERENCE]
    logger.info(
        "State sensitivity: minimum agreement=%.1f%%, minimum kappa=%.3f, maximum occupancy shift=%.1f pp",
        alternatives["state_agreement_percent"].min(),
        alternatives["cohen_kappa"].min(),
        alternatives["max_absolute_occupancy_change_percentage_points"].max(),
    )


if __name__ == "__main__":
    main()
