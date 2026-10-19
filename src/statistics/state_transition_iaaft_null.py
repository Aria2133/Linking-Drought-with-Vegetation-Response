from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# exact Figure-9 inputs (copies of 04_detrending_sensitivity_fixed/tables/*) bundled with the repository
SOURCE = ROOT / "results" / "states" / "main_state_classification_deseasonalized.csv"
FINAL_TRANSITIONS = ROOT / "results" / "states" / "main_state_transition_probabilities_deseasonalized.csv"
OUT = ROOT / "results" / "states" / "iaaft_null"
STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]
N_SURROGATES = 500
RANDOM_SEED = 20261013
MAX_ITER = 100


def iaaft(x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, int, float]:
    """IAAFT surrogate preserving the sample distribution and approximating the spectrum."""
    x = np.asarray(x, dtype=float)
    sorted_x = np.sort(x)
    centered = x - x.mean()
    target_amplitude = np.abs(np.fft.rfft(centered))
    y = rng.permutation(x)
    previous_error = np.inf
    stable = 0
    error = np.inf
    for iteration in range(1, MAX_ITER + 1):
        spectrum = np.fft.rfft(y - y.mean())
        phases = np.exp(1j * np.angle(spectrum))
        adjusted = np.fft.irfft(target_amplitude * phases, n=len(x))
        order = np.argsort(adjusted, kind="mergesort")
        remapped = np.empty_like(adjusted)
        remapped[order] = sorted_x
        y = remapped
        current_amplitude = np.abs(np.fft.rfft(y - y.mean()))
        denominator = np.linalg.norm(target_amplitude)
        error = float(np.linalg.norm(current_amplitude - target_amplitude) / denominator) if denominator else 0.0
        if abs(previous_error - error) < 1e-10:
            stable += 1
            if stable >= 5:
                break
        else:
            stable = 0
        previous_error = error
    return y, iteration, error


def classify(ewbi: np.ndarray, nirv: np.ndarray, ewbi_threshold: float, nirv_threshold: float) -> np.ndarray:
    labels = np.empty(len(ewbi), dtype=object)
    wet = ewbi >= ewbi_threshold
    healthy = nirv >= nirv_threshold
    labels[wet & healthy] = STATES[0]
    labels[wet & ~healthy] = STATES[1]
    labels[~wet & healthy] = STATES[2]
    labels[~wet & ~healthy] = STATES[3]
    return labels


def diagonal_probabilities(labels: np.ndarray) -> dict[str, tuple[float, int, int]]:
    previous = labels[:-1]
    following = labels[1:]
    result = {}
    for state in STATES:
        source = previous == state
        denominator = int(source.sum())
        count = int(np.sum(source & (following == state)))
        result[state] = (count / denominator if denominator else np.nan, count, denominator)
    return result


def lag1(x: np.ndarray) -> float:
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    HERE.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SOURCE, parse_dates=["date"]).sort_values(["land_type", "date"])
    data.to_csv(OUT / "state_transition_null_exact_input_series.csv", index=False, date_format="%Y-%m-%d")
    final = pd.read_csv(FINAL_TRANSITIONS)
    final_diag = final[(final["period"].eq("all")) & (final["from_state"].eq(final["to_state"]))]

    rng = np.random.default_rng(RANDOM_SEED)
    distribution_rows = []
    observed_rows = []
    diagnostic_rows = []

    for land, subset in data.groupby("land_type", sort=False):
        subset = subset.sort_values("date")
        ewbi = subset["ewbi"].to_numpy(dtype=float)
        nirv = subset["nirv_deseasonalized_anomaly"].to_numpy(dtype=float)
        ewbi_threshold = float(subset["ewbi_median_threshold"].iloc[0])
        nirv_threshold = float(subset["nirv_median_threshold"].iloc[0])
        observed_labels = classify(ewbi, nirv, ewbi_threshold, nirv_threshold)
        observed = diagonal_probabilities(observed_labels)
        for state, (probability, count, denominator) in observed.items():
            expected = final_diag[(final_diag["land_type"].eq(land)) & (final_diag["from_state"].eq(state))]
            assert len(expected) == 1
            assert np.isclose(probability, float(expected["transition_probability"].iloc[0]), atol=1e-12)
            observed_rows.append(
                {
                    "land_type": land,
                    "state": state,
                    "observed_diagonal_probability": probability,
                    "observed_persistent_transitions": count,
                    "observed_source_state_transitions": denominator,
                    "ewbi_threshold": ewbi_threshold,
                    "nirv_threshold": nirv_threshold,
                }
            )

        for replicate in range(1, N_SURROGATES + 1):
            ewbi_s, ewbi_iterations, ewbi_error = iaaft(ewbi, rng)
            nirv_s, nirv_iterations, nirv_error = iaaft(nirv, rng)
            labels = classify(ewbi_s, nirv_s, ewbi_threshold, nirv_threshold)
            diagonal = diagonal_probabilities(labels)
            ewbi_lag1 = lag1(ewbi_s)
            nirv_lag1 = lag1(nirv_s)
            for state, (probability, count, denominator) in diagonal.items():
                distribution_rows.append(
                    {
                        "land_type": land,
                        "surrogate_id": replicate,
                        "state": state,
                        "diagonal_probability": probability,
                        "persistent_transitions": count,
                        "source_state_transitions": denominator,
                    }
                )
            diagnostic_rows.append(
                {
                    "land_type": land,
                    "surrogate_id": replicate,
                    "ewbi_iaaft_iterations": ewbi_iterations,
                    "nirv_iaaft_iterations": nirv_iterations,
                    "ewbi_relative_spectral_error": ewbi_error,
                    "nirv_relative_spectral_error": nirv_error,
                    "ewbi_lag1": ewbi_lag1,
                    "nirv_lag1": nirv_lag1,
                }
            )

    distributions = pd.DataFrame(distribution_rows)
    observed = pd.DataFrame(observed_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary_rows = []
    for row in observed.itertuples(index=False):
        null = distributions[
            distributions["land_type"].eq(row.land_type) & distributions["state"].eq(row.state)
        ]["diagonal_probability"].dropna().to_numpy()
        lower, upper = np.percentile(null, [2.5, 97.5])
        upper_p = (1 + np.sum(null >= row.observed_diagonal_probability)) / (len(null) + 1)
        lower_p = (1 + np.sum(null <= row.observed_diagonal_probability)) / (len(null) + 1)
        summary_rows.append(
            {
                **row._asdict(),
                "n_surrogates": len(null),
                "null_median": np.median(null),
                "null_ci95_low": lower,
                "null_ci95_high": upper,
                "empirical_p_upper_observed_greater": upper_p,
                "empirical_p_two_sided": min(1.0, 2 * min(upper_p, lower_p)),
                "observed_minus_null_median": row.observed_diagonal_probability - np.median(null),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "state_transition_null_summary.csv", index=False)
    distributions.to_csv(OUT / "state_transition_null_distributions.csv", index=False)
    diagnostics.to_csv(OUT / "state_transition_null_surrogate_diagnostics.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    short = {"Wet / Healthy": "W/H", "Wet / Suppressed": "W/S", "Dry / Resilient": "D/R", "Dry / Stressed": "D/S"}
    for ax, land in zip(axes.ravel(), data["land_type"].drop_duplicates()):
        sub = summary[summary["land_type"].eq(land)].set_index("state").loc[STATES].reset_index()
        x = np.arange(len(STATES))
        ax.vlines(x, sub["null_ci95_low"], sub["null_ci95_high"], color="#6B7280", linewidth=7, alpha=0.7, label="Null 95% interval")
        ax.scatter(x, sub["null_median"], color="white", edgecolor="#374151", s=45, zorder=3, label="Null median")
        ax.scatter(x, sub["observed_diagonal_probability"], color="#C62828", marker="D", s=58, zorder=4, label="Observed")
        ax.set_xticks(x, [short[s] for s in STATES])
        ax.set_ylim(0, 1)
        ax.set_title(land, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel("Diagonal transition probability")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=3, frameon=False)
    fig.suptitle("Observed state persistence versus autocorrelation-preserving IAAFT null", fontweight="bold", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT / "state_transition_null_figure.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    original_diag = data.groupby("land_type").agg(
        ewbi_lag1=("ewbi", lambda x: lag1(x.to_numpy())),
        nirv_lag1=("nirv_deseasonalized_anomaly", lambda x: lag1(x.to_numpy())),
    )
    surrogate_diag = diagnostics.groupby("land_type").agg(
        surrogate_ewbi_lag1_median=("ewbi_lag1", "median"),
        surrogate_nirv_lag1_median=("nirv_lag1", "median"),
        ewbi_spectral_error_median=("ewbi_relative_spectral_error", "median"),
        nirv_spectral_error_median=("nirv_relative_spectral_error", "median"),
    )
    diagnostic_summary = original_diag.join(surrogate_diag)

    significant = summary[summary["empirical_p_upper_observed_greater"] < 0.05]
    lines = [
        "# State-transition autocorrelation-aware null analysis",
        "",
        "## Design",
        "",
        f"The analysis used exactly the 288 monthly EWBI and NIRv values per land type stored for final Figure 9. "
        f"For each land type, {N_SURROGATES} independent EWBI/NIRv IAAFT surrogate pairs were generated with fixed seed {RANDOM_SEED}. "
        "IAAFT rank-remaps each surrogate to the observed marginal distribution while iteratively matching the Fourier amplitude spectrum, thereby approximately preserving temporal autocorrelation/power. EWBI and NIRv were randomized independently under the null.",
        "",
        "The original Figure-9 EWBI and NIRv median thresholds were held fixed; thresholds were not re-estimated for each surrogate. "
        "The empirical primary p-value is upper-tailed: the fraction of surrogate persistence probabilities at least as large as observed, with the plus-one correction. A two-sided empirical value is also supplied.",
        "",
        "## Results",
        "",
        f"{len(significant)} of {len(summary)} land-type/state persistence probabilities exceeded the autocorrelation-preserving null at upper-tail p < 0.05. "
        "The CSV contains observed persistence, null median, 2.5–97.5% interval, and empirical p-values for every state.",
        "",
        "## Dependence-preservation diagnostic",
        "",
        "```text",
        diagnostic_summary.to_string(float_format=lambda x: f"{x:.4f}"),
        "```",
        "",
        "## Interpretation limit",
        "",
        "This test asks whether diagonal persistence exceeds what is expected from independently generated autocorrelated EWBI and NIRv series with the observed marginal distributions. It does not validate a Markov mechanism or establish ecosystem-state dynamics.",
        "",
        f"Exact input: `{SOURCE}`",
    ]
    (OUT / "STATE_TRANSITION_NULL_ANALYSIS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nDiagnostic summary:")
    print(diagnostic_summary.to_string())


if __name__ == "__main__":
    main()
