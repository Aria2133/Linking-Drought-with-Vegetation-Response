from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment

from config import FIGURES, LAND_LABELS, LAND_TYPES, TABLES
from revision_utils import (
    cohen_kappa,
    complete_landtype_frame,
)

STATES = ["Wet/Healthy", "Wet/Suppressed", "Dry/Resilient", "Dry/Stressed"]


def full_period_monthly_median_anomaly(series: pd.Series) -> pd.Series:
    """Match the state-anomaly definition reported in manuscript Section 3.2.3."""
    monthly_median = series.groupby(series.index.month).transform("median")
    return series - monthly_median


def time_zscore(series: pd.Series) -> pd.Series:
    standard_deviation = series.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / standard_deviation


def manuscript_state_inputs(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    forcing_anomalies = {
        variable: full_period_monthly_median_anomaly(frame[variable])
        for variable in ("SPEI", "TWS", "VPD", "LST")
    }
    ewbi = (
        time_zscore(forcing_anomalies["SPEI"])
        + time_zscore(forcing_anomalies["TWS"])
        - time_zscore(forcing_anomalies["VPD"])
        - time_zscore(forcing_anomalies["LST"])
    )
    nirv = full_period_monthly_median_anomaly(frame["NIRv"])
    return ewbi, nirv


def quadrant_label(ewbi: float, nirv: float, ewbi_cut: float = 0, nirv_cut: float = 0) -> str:
    if ewbi > ewbi_cut and nirv > nirv_cut:
        return "Wet/Healthy"
    if ewbi > ewbi_cut and nirv <= nirv_cut:
        return "Wet/Suppressed"
    if ewbi <= ewbi_cut and nirv > nirv_cut:
        return "Dry/Resilient"
    return "Dry/Stressed"


def median_labels(ewbi: pd.Series, nirv: pd.Series) -> pd.Series:
    valid = ewbi.notna() & nirv.notna()
    result = pd.Series(index=ewbi.index, dtype=object)
    e_cut, n_cut = ewbi[valid].median(), nirv[valid].median()
    result.loc[valid] = [
        quadrant_label(e, n, e_cut, n_cut)
        for e, n in zip(ewbi[valid].to_numpy(), nirv[valid].to_numpy())
    ]
    return result


def extreme_labels(ewbi: pd.Series, nirv: pd.Series) -> pd.Series:
    result = pd.Series(index=ewbi.index, dtype=object)
    e_low, e_high = ewbi.quantile([1 / 3, 2 / 3])
    n_low, n_high = nirv.quantile([1 / 3, 2 / 3])
    valid = ((ewbi <= e_low) | (ewbi >= e_high)) & ((nirv <= n_low) | (nirv >= n_high))
    result.loc[valid] = [
        quadrant_label(e, n, (e_low + e_high) / 2, (n_low + n_high) / 2)
        for e, n in zip(ewbi[valid].to_numpy(), nirv[valid].to_numpy())
    ]
    return result


def cluster_labels(
    ewbi: pd.Series,
    nirv: pd.Series,
    method: str,
) -> pd.Series:
    result = pd.Series(index=ewbi.index, dtype=object)
    valid = ewbi.notna() & nirv.notna()
    x = np.column_stack([ewbi[valid], nirv[valid]])
    if method == "kmeans":
        model = KMeans(n_clusters=4, n_init=50, random_state=42).fit(x)
        labels = model.labels_
        centers = model.cluster_centers_
    else:
        model = GaussianMixture(n_components=4, covariance_type="full", random_state=42, n_init=10).fit(x)
        labels = model.predict(x)
        centers = model.means_
    e_low, e_high = np.quantile(x[:, 0], [0.25, 0.75])
    n_low, n_high = np.quantile(x[:, 1], [0.25, 0.75])
    prototypes = np.asarray(
        [
            [e_high, n_high],
            [e_high, n_low],
            [e_low, n_high],
            [e_low, n_low],
        ]
    )
    cost = np.linalg.norm(centers[:, None, :] - prototypes[None, :, :], axis=2)
    cluster_indices, state_indices = linear_sum_assignment(cost)
    mapping = {
        int(cluster): STATES[int(state)]
        for cluster, state in zip(cluster_indices, state_indices)
    }
    result.loc[valid] = [mapping[label] for label in labels]
    return result


def transition_matrix(labels: pd.Series) -> tuple[np.ndarray, dict]:
    counts = np.zeros((4, 4), dtype=int)
    values = labels.to_numpy()
    for previous, current in zip(values[:-1], values[1:]):
        if previous in STATES and current in STATES:
            counts[STATES.index(previous), STATES.index(current)] += 1
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts, totals, where=totals > 0)
    occupancy_counts = Counter(value for value in values if value in STATES)
    occupancy_total = sum(occupancy_counts.values())
    occupancy = {
        state: occupancy_counts[state] / occupancy_total * 100 if occupancy_total else np.nan
        for state in STATES
    }
    return probabilities, occupancy


def main() -> None:
    agreement_rows = []
    matrix_rows = []
    occupancy_rows = []

    for land_type in LAND_TYPES:
        frame = complete_landtype_frame(land_type, "monthwise")
        ewbi, nirv = manuscript_state_inputs(frame)
        methods = {
            "median": median_labels(ewbi, nirv),
            "33_67_extremes": extreme_labels(ewbi, nirv),
            "kmeans": cluster_labels(ewbi, nirv, "kmeans"),
            "gmm": cluster_labels(ewbi, nirv, "gmm"),
        }
        reference = methods["median"]
        for method, labels in methods.items():
            common = reference.notna() & labels.notna()
            agreement_rows.append(
                {
                    "land_type": land_type,
                    "method": method,
                    "comparable_months": int(common.sum()),
                    "agreement_fraction": float(np.mean(reference[common] == labels[common])),
                    "cohen_kappa": cohen_kappa(
                        reference[common].tolist(), labels[common].tolist()
                    ),
                    "adjusted_rand_index": float(
                        adjusted_rand_score(reference[common], labels[common])
                    ),
                }
            )
            probabilities, occupancy = transition_matrix(labels)
            for i, source in enumerate(STATES):
                for j, destination in enumerate(STATES):
                    matrix_rows.append(
                        {
                            "land_type": land_type,
                            "method": method,
                            "from_state": source,
                            "to_state": destination,
                            "probability_percent": probabilities[i, j] * 100,
                        }
                    )
            for state, percentage in occupancy.items():
                occupancy_rows.append(
                    {
                        "land_type": land_type,
                        "method": method,
                        "state": state,
                        "occupancy_percent": percentage,
                    }
                )

    agreement = pd.DataFrame(agreement_rows)
    matrices = pd.DataFrame(matrix_rows)
    occupancy = pd.DataFrame(occupancy_rows)
    agreement.to_csv(TABLES / "state_classification_agreement.csv", index=False)
    matrices.to_csv(TABLES / "state_transition_matrices_all_methods.csv", index=False)
    occupancy.to_csv(TABLES / "state_occupancy_all_methods.csv", index=False)
    with pd.ExcelWriter(TABLES / "state_classification_sensitivity_revision.xlsx") as writer:
        agreement.to_excel(writer, sheet_name="agreement", index=False)
        matrices.to_excel(writer, sheet_name="transitions", index=False)
        occupancy.to_excel(writer, sheet_name="occupancy", index=False)

    figure, ax = plt.subplots(figsize=(10, 5.5))
    plot = agreement[agreement["method"] != "median"]
    sns.barplot(data=plot, x="land_type", y="cohen_kappa", hue="method", ax=ax)
    ax.set_ylim(-0.1, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Cohen's kappa vs median classification")
    ax.set_title("Ecosystem-state classification sensitivity")
    ax.legend(title="Alternative method")
    figure.tight_layout()
    figure.savefig(FIGURES / "state_classification_sensitivity.png", dpi=400)
    figure.savefig(FIGURES / "state_classification_sensitivity.pdf")
    plt.close(figure)
    print("State classification sensitivity outputs written.")


if __name__ == "__main__":
    main()
