from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[2]
MAJOR = PAPER / "Docs" / "major_revision_analyses"
EWBI_SOURCE = MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv"
STATE_SOURCE = MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "main_state_classification_deseasonalized.csv"
RESPONSE_SOURCE = MAJOR / "final_previous_GNDVI_with_test_rebuild" / "tables" / "Figure_2_previous_GNDVI_with_test_monthly_series.csv"

LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
RESPONSES = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
COMPONENTS = ["SPEI", "TWS", "VPD", "LST"]
SIGNED_COLUMNS = {
    "SPEI": "spei_signed_z",
    "TWS": "tws_signed_z",
    "VPD": "vpd_signed_z",
    "LST": "lst_signed_z",
}
PERIODS = [
    ("Baseline 2001–2011", 2001, 2011),
    ("Historic drought 2012–2016", 2012, 2016),
    ("Recent drought 2020–2024", 2020, 2024),
]
STATES = ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]
METHODS = ["median", "33_67_extremes", "kmeans", "gmm"]
METHOD_LABELS = {
    "median": "Median",
    "33_67_extremes": "33/67",
    "kmeans": "k-means",
    "gmm": "GMM",
}
SEED = 42
BOOTSTRAP_REPS = 2000
BLOCK_MONTHS = 12


def pearson_with_ci(x: pd.Series, y: pd.Series) -> tuple[int, float, float, float, float]:
    joined = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    n = len(joined)
    if n < 4 or joined["x"].nunique() < 2 or joined["y"].nunique() < 2:
        return n, np.nan, np.nan, np.nan, np.nan
    result = stats.pearsonr(joined["x"], joined["y"])
    ci = result.confidence_interval(confidence_level=0.95)
    return n, float(result.statistic), float(result.pvalue), float(ci.low), float(ci.high)


def paired_block_bootstrap_ci(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, 0
    block = min(BLOCK_MONTHS, n)
    starts = np.arange(max(1, n - block + 1))
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    blocks_needed = int(np.ceil(n / block))
    for _ in range(BOOTSTRAP_REPS):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in chosen])[:n]
        if np.std(x[idx]) > 0 and np.std(y[idx]) > 0:
            samples.append(float(np.corrcoef(x[idx], y[idx])[0, 1]))
    if not samples:
        return np.nan, np.nan, 0
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high), len(samples)


def moving_block_mean_ci(x: np.ndarray, seed: int) -> tuple[float, float, float, int]:
    x = np.asarray(x[np.isfinite(x)], dtype=float)
    n = len(x)
    if not n:
        return np.nan, np.nan, np.nan, 0
    block = min(BLOCK_MONTHS, n)
    starts = np.arange(max(1, n - block + 1))
    rng = np.random.default_rng(seed)
    blocks_needed = int(np.ceil(n / block))
    means = np.empty(BOOTSTRAP_REPS)
    for i in range(BOOTSTRAP_REPS):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        idx = np.concatenate([np.arange(s, min(s + block, n)) for s in chosen])[:n]
        means[i] = np.mean(x[idx])
    low, high = np.percentile(means, [2.5, 97.5])
    return float(np.mean(x)), float(low), float(high), n


def load_authoritative_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ewbi = pd.read_csv(EWBI_SOURCE, parse_dates=["date"])
    state = pd.read_csv(STATE_SOURCE, parse_dates=["date"])
    response_long = pd.read_csv(RESPONSE_SOURCE, parse_dates=["date"])
    response = (
        response_long.pivot_table(
            index=["date", "year", "month", "land_type"],
            columns="variable",
            values="standardized_anomaly",
            aggfunc="first",
        )
        .reset_index()
        .sort_values(["land_type", "date"])
    )
    return ewbi, state, response


def pca_ewbi(signed: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    valid = signed.dropna()
    x = StandardScaler().fit_transform(valid)
    pca = PCA(n_components=1, random_state=SEED)
    score = pca.fit_transform(x).ravel()
    equal = valid.mean(axis=1).to_numpy()
    if np.corrcoef(score, equal)[0, 1] < 0:
        score *= -1
        pca.components_[0] *= -1
    output = pd.Series(np.nan, index=signed.index, dtype=float)
    output.loc[valid.index] = score
    return output, pca.components_[0]


def regression_ewbi(signed: pd.DataFrame, vegetation: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    target = vegetation.mean(axis=1)
    joined = signed.join(target.rename("target")).dropna()
    model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    model.fit(joined[COMPONENTS], joined["target"])
    weights = model.coef_.copy()
    if weights.sum() < 0:
        weights *= -1
    weights = weights / np.sum(np.abs(weights))
    output = pd.Series(np.nan, index=signed.index, dtype=float)
    output.loc[joined.index] = joined[COMPONENTS].to_numpy() @ weights
    return output, weights


def build_s4(ewbi: pd.DataFrame, response: pd.DataFrame) -> dict[str, pd.DataFrame]:
    validation_rows: list[dict] = []
    component_rows: list[dict] = []
    loco_rows: list[dict] = []
    weight_rows: list[dict] = []
    period_rows: list[dict] = []
    period_test_rows: list[dict] = []
    exact_rows: list[pd.DataFrame] = []

    for li, land in enumerate(LANDS):
        e = ewbi[ewbi.land_type == land].set_index("date").sort_index()
        v = response[response.land_type == land].set_index("date").sort_index()
        idx = e.index.union(v.index).sort_values()
        signed = pd.DataFrame({name: e[col].reindex(idx) for name, col in SIGNED_COLUMNS.items()}, index=idx)
        veg = v[RESPONSES].reindex(idx)
        equal = signed.mean(axis=1)
        pca_score, pca_weights = pca_ewbi(signed)
        ridge_score, ridge_weights = regression_ewbi(signed, veg)
        indices = {"Equal": equal, "PCA": pca_score, "Regression": ridge_score}

        for scheme, weights in {
            "Equal": np.repeat(0.25, 4),
            "PCA": pca_weights / np.sum(np.abs(pca_weights)),
            "Regression": ridge_weights,
        }.items():
            for component, weight in zip(COMPONENTS, weights):
                weight_rows.append({"land_type": land, "weighting": scheme, "component": component, "signed_weight": float(weight)})

        for resp in RESPONSES:
            for scheme, series in indices.items():
                n, r, p, low, high = pearson_with_ci(series, veg[resp])
                validation_rows.append({"land_type": land, "response": resp, "index": scheme, "n": n, "pearson_r": r, "p_value": p, "fisher_ci95_low": low, "fisher_ci95_high": high})
            for component in COMPONENTS:
                n, r, p, low, high = pearson_with_ci(signed[component], veg[resp])
                component_rows.append({"land_type": land, "response": resp, "predictor": component, "n": n, "pearson_r": r, "p_value": p, "fisher_ci95_low": low, "fisher_ci95_high": high})
            for omitted in COMPONENTS:
                reduced = signed.drop(columns=omitted).mean(axis=1)
                n, r, p, low, high = pearson_with_ci(reduced, veg[resp])
                loco_rows.append({"land_type": land, "response": resp, "omitted_component": omitted, "n": n, "pearson_r": r, "p_value": p, "fisher_ci95_low": low, "fisher_ci95_high": high})

        groups = []
        for pi, (period, start, end) in enumerate(PERIODS):
            values = equal[(equal.index.year >= start) & (equal.index.year <= end)].dropna().to_numpy()
            groups.append(values)
            mean, low, high, n = moving_block_mean_ci(values, SEED + li * 10 + pi)
            period_rows.append({"land_type": land, "period": period, "start_year": start, "end_year": end, "n_months": n, "mean_ewbi": mean, "block_bootstrap_ci95_low": low, "block_bootstrap_ci95_high": high, "bootstrap_replicates": BOOTSTRAP_REPS, "block_length_months": BLOCK_MONTHS})
        anova = stats.f_oneway(*groups)
        kruskal = stats.kruskal(*groups)
        period_test_rows.append({"land_type": land, "anova_f": float(anova.statistic), "anova_p": float(anova.pvalue), "kruskal_h": float(kruskal.statistic), "kruskal_p": float(kruskal.pvalue)})

        exact = pd.DataFrame({"date": idx, "land_type": land, **{c: signed[c].values for c in COMPONENTS}, "EWBI_equal": equal.values, "EWBI_PCA": pca_score.values, "EWBI_regression": ridge_score.values, **{r: veg[r].values for r in RESPONSES}})
        exact_rows.append(exact)

    tables = {
        "validation": pd.DataFrame(validation_rows),
        "components": pd.DataFrame(component_rows),
        "loco": pd.DataFrame(loco_rows),
        "weights": pd.DataFrame(weight_rows),
        "periods": pd.DataFrame(period_rows),
        "period_tests": pd.DataFrame(period_test_rows),
        "exact": pd.concat(exact_rows, ignore_index=True),
    }
    names = {
        "validation": "Figure_S4_harmonized_validation_correlations.csv",
        "components": "Figure_S4_harmonized_component_correlations.csv",
        "loco": "Figure_S4_harmonized_leave_one_out.csv",
        "weights": "Figure_S4_harmonized_weights.csv",
        "periods": "Figure_S4_harmonized_period_summary.csv",
        "period_tests": "Figure_S4_harmonized_period_tests.csv",
        "exact": "Figure_S4_harmonized_exact_monthly_source.csv",
    }
    for key, table in tables.items():
        table.to_csv(HERE / names[key], index=False)

    sns.set_theme(style="white", font_scale=0.88)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.5))
    a = tables["validation"].groupby(["index", "response"], as_index=False).pearson_r.mean().pivot(index="index", columns="response", values="pearson_r").reindex(["Equal", "PCA", "Regression"], columns=RESPONSES)
    sns.heatmap(a, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar=False, ax=axes[0])
    axes[0].set_title("(a) EWBI weighting sensitivity")
    axes[0].set_xlabel("Response (mean across land types)")
    axes[0].set_ylabel("Weighting method")
    bframe = pd.concat([
        tables["components"].assign(test=lambda x: x.predictor),
        tables["loco"].assign(test=lambda x: "Without " + x.omitted_component)[["land_type", "response", "test", "pearson_r"]],
    ], ignore_index=True)
    border = COMPONENTS + [f"Without {c}" for c in COMPONENTS]
    b = bframe.groupby(["test", "response"], as_index=False).pearson_r.mean().pivot(index="test", columns="response", values="pearson_r").reindex(border, columns=RESPONSES)
    sns.heatmap(b, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar=False, ax=axes[1])
    axes[1].set_title("(b) Component and leave-one-out tests")
    axes[1].set_xlabel("Response (mean across land types)")
    axes[1].set_ylabel("")
    colors = sns.color_palette("colorblind", 4)
    x = np.arange(len(PERIODS))
    offsets = np.linspace(-0.27, 0.27, len(LANDS))
    for offset, color, land in zip(offsets, colors, LANDS):
        d = tables["periods"][tables["periods"].land_type == land].set_index("period").reindex([p[0] for p in PERIODS])
        y = d.mean_ewbi.to_numpy()
        err = np.vstack([y - d.block_bootstrap_ci95_low.to_numpy(), d.block_bootstrap_ci95_high.to_numpy() - y])
        axes[2].errorbar(x + offset, y, yerr=err, marker="o", capsize=3, lw=1.4, color=color, label=land)
    axes[2].axhline(0, color="0.3", lw=0.8)
    axes[2].set_xticks(x, ["2001–2011", "2012–2016", "2020–2024"], rotation=20, ha="right")
    axes[2].set_ylabel("Mean EWBI (95% moving-block CI)")
    axes[2].set_xlabel("Period")
    axes[2].set_title("(c) Temporal-period sensitivity")
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Figure S4. EWBI construction and component sensitivity (authoritative no-QA branch)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(HERE / "Figure_S4_harmonized.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return tables


def build_s5(ewbi: pd.DataFrame, response: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    exact_rows: list[pd.DataFrame] = []
    for li, land in enumerate(LANDS):
        e = ewbi[ewbi.land_type == land].set_index("date").sort_index()["ewbi"]
        v = response[response.land_type == land].set_index("date").sort_index()[RESPONSES]
        joined_all = pd.concat([e.rename("EWBI"), v], axis=1)
        exact_rows.append(joined_all.reset_index().assign(land_type=land))
        for pi, (period, start, end) in enumerate(PERIODS):
            mask = (joined_all.index.year >= start) & (joined_all.index.year <= end)
            for ri, resp in enumerate(RESPONSES):
                joined = joined_all.loc[mask, ["EWBI", resp]].dropna()
                n = len(joined)
                if n >= 4:
                    pr = stats.pearsonr(joined.EWBI, joined[resp])
                    sr = stats.spearmanr(joined.EWBI, joined[resp])
                    low, high, valid = paired_block_bootstrap_ci(joined.EWBI.to_numpy(), joined[resp].to_numpy(), SEED + li * 100 + pi * 10 + ri)
                else:
                    pr = sr = None
                    low = high = np.nan
                    valid = 0
                rows.append({
                    "land_type": land, "period": period, "period_start": f"{start}-01", "period_end": f"{end}-12", "response": resp, "n": n,
                    "pearson_r": float(pr.statistic) if pr else np.nan, "pearson_p_value": float(pr.pvalue) if pr else np.nan,
                    "spearman_rho": float(sr.statistic) if sr else np.nan, "spearman_p_value": float(sr.pvalue) if sr else np.nan,
                    "pearson_block_bootstrap_ci95_low": low, "pearson_block_bootstrap_ci95_high": high,
                    "significant_bootstrap_ci_excludes_zero": bool(low > 0 or high < 0) if np.isfinite(low) else False,
                    "bootstrap_replicates_requested": BOOTSTRAP_REPS, "bootstrap_valid_replicates": valid, "moving_block_length_months": min(BLOCK_MONTHS, max(n, 1)),
                    "significance_rule_used_in_figure": "95% paired moving-block bootstrap CI excludes zero",
                    "gndvi_branch": "previous/no-QA GNDVI",
                })
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "Figure_S5_harmonized_correlations.csv", index=False)
    pd.concat(exact_rows, ignore_index=True).to_csv(HERE / "Figure_S5_harmonized_exact_monthly_source.csv", index=False)

    sns.set_theme(style="white", font_scale=0.9)
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2), sharey=True)
    for ax, (period, _, _) in zip(axes, PERIODS):
        d = table[table.period == period]
        heat = d.pivot(index="land_type", columns="response", values="pearson_r").reindex(LANDS, columns=RESPONSES)
        annot = heat.copy().astype(object)
        sig = d.pivot(index="land_type", columns="response", values="significant_bootstrap_ci_excludes_zero").reindex(LANDS, columns=RESPONSES)
        for land in LANDS:
            for resp in RESPONSES:
                val = heat.loc[land, resp]
                annot.loc[land, resp] = "" if pd.isna(val) else f"{val:.2f}{'*' if bool(sig.loc[land, resp]) else ''}"
        sns.heatmap(heat, annot=annot, fmt="", cmap="RdBu_r", center=0, vmin=-1, vmax=1, cbar=ax is axes[-1], ax=ax)
        ax.set_title(period)
        ax.set_xlabel("Response")
        ax.set_ylabel("Land type" if ax is axes[0] else "")
    fig.suptitle("Figure S5. Period-stratified EWBI–response correlations\n* 95% paired moving-block bootstrap CI excludes zero", fontsize=14, y=1.03)
    fig.tight_layout()
    fig.savefig(HERE / "Figure_S5_harmonized.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return table


def quadrant(e: float, n: float, ecut: float, ncut: float) -> str:
    if e >= ecut and n >= ncut:
        return STATES[0]
    if e >= ecut and n < ncut:
        return STATES[1]
    if e < ecut and n >= ncut:
        return STATES[2]
    return STATES[3]


def extreme_labels(e: pd.Series, n: pd.Series) -> pd.Series:
    result = pd.Series(index=e.index, dtype=object)
    elow, ehigh = e.quantile([1 / 3, 2 / 3])
    nlow, nhigh = n.quantile([1 / 3, 2 / 3])
    valid = ((e <= elow) | (e >= ehigh)) & ((n <= nlow) | (n >= nhigh))
    result.loc[valid] = [quadrant(a, b, (elow + ehigh) / 2, (nlow + nhigh) / 2) for a, b in zip(e[valid], n[valid])]
    return result


def cluster_labels(e: pd.Series, n: pd.Series, method: str) -> pd.Series:
    result = pd.Series(index=e.index, dtype=object)
    valid = e.notna() & n.notna()
    x = np.column_stack([e[valid], n[valid]])
    if method == "kmeans":
        model = KMeans(n_clusters=4, n_init=50, random_state=SEED).fit(x)
        labels, centers = model.labels_, model.cluster_centers_
    else:
        model = GaussianMixture(n_components=4, covariance_type="full", n_init=10, random_state=SEED).fit(x)
        labels, centers = model.predict(x), model.means_
    elow, ehigh = np.quantile(x[:, 0], [0.25, 0.75])
    nlow, nhigh = np.quantile(x[:, 1], [0.25, 0.75])
    prototypes = np.asarray([[ehigh, nhigh], [ehigh, nlow], [elow, nhigh], [elow, nlow]])
    clusters, state_ids = linear_sum_assignment(np.linalg.norm(centers[:, None, :] - prototypes[None, :, :], axis=2))
    mapping = {int(c): STATES[int(s)] for c, s in zip(clusters, state_ids)}
    result.loc[valid] = [mapping[int(label)] for label in labels]
    return result


def transition_and_occupancy(labels: pd.Series) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    counts = np.zeros((4, 4), dtype=int)
    values = labels.to_numpy()
    for previous, current in zip(values[:-1], values[1:]):
        if previous in STATES and current in STATES:
            counts[STATES.index(previous), STATES.index(current)] += 1
    totals = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, totals, out=np.full((4, 4), np.nan), where=totals > 0)
    occupied = Counter(value for value in values if value in STATES)
    total = sum(occupied.values())
    occupancy = {state: 100 * occupied[state] / total if total else np.nan for state in STATES}
    return probs, occupancy, counts


def build_s6(state: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    agreement_rows: list[dict] = []
    transition_rows: list[dict] = []
    occupancy_rows: list[dict] = []
    exact_rows: list[pd.DataFrame] = []
    for land in LANDS:
        d = state[state.land_type == land].sort_values("date").set_index("date")
        e = d.ewbi
        n = d.nirv_deseasonalized_anomaly
        labels = {
            "median": d.ecosystem_state.copy(),
            "33_67_extremes": extreme_labels(e, n),
            "kmeans": cluster_labels(e, n, "kmeans"),
            "gmm": cluster_labels(e, n, "gmm"),
        }
        reference = labels["median"]
        exact = d.reset_index()[["date", "year", "month", "land_type", "ewbi", "nirv_deseasonalized_anomaly", "ewbi_median_threshold", "nirv_median_threshold", "ecosystem_state"]].copy()
        for method, lab in labels.items():
            exact[f"state_{method}"] = lab.to_numpy()
            common = reference.notna() & lab.notna()
            agreement_rows.append({
                "land_type": land, "method": method, "method_label": METHOD_LABELS[method], "reference_method": "Figure-9 median", "comparable_months": int(common.sum()),
                "agreement_fraction": float(np.mean(reference[common] == lab[common])), "cohen_kappa": float(cohen_kappa_score(reference[common], lab[common])),
                "adjusted_rand_index": float(adjusted_rand_score(reference[common], lab[common])),
            })
            probs, occupancy, counts = transition_and_occupancy(lab)
            for i, source in enumerate(STATES):
                for j, destination in enumerate(STATES):
                    transition_rows.append({"land_type": land, "method": method, "method_label": METHOD_LABELS[method], "from_state": source, "to_state": destination, "transition_count": int(counts[i, j]), "probability": probs[i, j], "probability_percent": probs[i, j] * 100})
            for state_name, percentage in occupancy.items():
                occupancy_rows.append({"land_type": land, "method": method, "method_label": METHOD_LABELS[method], "state": state_name, "occupancy_percent": percentage})
        exact_rows.append(exact)
    agreement = pd.DataFrame(agreement_rows)
    transitions = pd.DataFrame(transition_rows)
    occupancy = pd.DataFrame(occupancy_rows)
    agreement.to_csv(HERE / "Figure_S6_harmonized_agreement.csv", index=False)
    transitions.to_csv(HERE / "Figure_S6_harmonized_transition_matrices.csv", index=False)
    occupancy.to_csv(HERE / "Figure_S6_harmonized_occupancy.csv", index=False)
    pd.concat(exact_rows, ignore_index=True).to_csv(HERE / "Figure_S6_harmonized_exact_monthly_states.csv", index=False)

    sns.set_theme(style="white", font_scale=0.83)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    alt = ["33_67_extremes", "kmeans", "gmm"]
    a = agreement[agreement.method.isin(alt)].pivot(index="land_type", columns="method", values="cohen_kappa").reindex(LANDS, columns=alt)
    a.columns = [METHOD_LABELS[c] for c in a.columns]
    sns.heatmap(a, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, cbar=False, ax=axes[0])
    axes[0].set_title("(a) Classification agreement")
    axes[0].set_xlabel("Cohen's κ vs Figure-9 median")
    axes[0].set_ylabel("Land type")
    median_occ = occupancy[occupancy.method == "median"].set_index(["land_type", "state"]).occupancy_percent
    change_rows = []
    for land in LANDS:
        for method in alt:
            current = occupancy[(occupancy.land_type == land) & (occupancy.method == method)].set_index("state").occupancy_percent
            common_states = current.index.intersection(median_occ.loc[land].index)
            change_rows.append({"land_type": land, "method": method, "max_change": float((current.loc[common_states] - median_occ.loc[land, common_states]).abs().max())})
    change = pd.DataFrame(change_rows).pivot(index="land_type", columns="method", values="max_change").reindex(LANDS, columns=alt)
    change.columns = [METHOD_LABELS[c] for c in change.columns]
    sns.heatmap(change, annot=True, fmt=".1f", cmap="OrRd", vmin=0, cbar=False, ax=axes[1])
    axes[1].set_title("(b) Occupancy sensitivity")
    axes[1].set_xlabel("Maximum change vs median (percentage points)")
    axes[1].set_ylabel("")
    persistence = transitions[(transitions.from_state == STATES[3]) & (transitions.to_state == STATES[3])].pivot(index="land_type", columns="method", values="probability_percent").reindex(LANDS, columns=METHODS)
    persistence.columns = [METHOD_LABELS[c] for c in persistence.columns]
    sns.heatmap(persistence, annot=True, fmt=".1f", cmap="PuBu", vmin=0, vmax=100, cbar=False, ax=axes[2])
    axes[2].set_title("(c) Dry / Stressed persistence")
    axes[2].set_xlabel("Self-transition probability (%)")
    axes[2].set_ylabel("")
    fig.suptitle("Figure S6. Sensitivity of the exact Figure-9 state construction", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(HERE / "Figure_S6_harmonized.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return agreement, transitions, occupancy


def write_report(s4: dict[str, pd.DataFrame], s5: pd.DataFrame, s6: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> None:
    agreement, transitions, occupancy = s6
    equal = s4["validation"][s4["validation"]["index"] == "Equal"]
    pca = s4["validation"][s4["validation"]["index"] == "PCA"]
    ridge = s4["validation"][s4["validation"]["index"] == "Regression"]
    s5_sig = int(s5.significant_bootstrap_ci_excludes_zero.sum())
    alt_agreement = agreement[agreement.method != "median"]
    median_persistence = transitions[(transitions.method == "median") & (transitions.from_state == STATES[3]) & (transitions.to_state == STATES[3])]
    text = f"""# Supplementary harmonization

## Scope and authoritative branches

This rebuild changes only the inconsistent source branches identified by the completed forensic audits. S4 and S5 use the previous/no-QA GNDVI response series and the same signed, deseasonalized component anomalies used to construct the authoritative Figure-9 EWBI. S6 starts from the exact monthly EWBI, NIRv, thresholds, and median states used by Figure 9. No new weighting method or classifier was introduced.

## Figure S4

The existing equal, PCA, ridge-to-mean-vegetation, component, and leave-one-component-out methods were rerun on the authoritative inputs. Across the 20 land-type × response combinations, mean Pearson r was {equal.pearson_r.mean():.3f} for equal weighting, {pca.pearson_r.mean():.3f} for PCA weighting, and {ridge.pearson_r.mean():.3f} for the existing regression weighting. The regression result remains descriptive because its weights are fitted to the same vegetation responses used for validation; it is not independent predictive validation. Period uncertainty uses 2,000 moving-block bootstrap replicates with 12-month blocks.

## Figure S5

All 60 period × land-type × response correlations use previous/no-QA GNDVI. The figure uses one significance rule only: an asterisk means the 95% paired 12-month moving-block bootstrap interval for Pearson r excludes zero. Pearson and Spearman p-values remain in the source table for transparency but are not mixed with the plotted significance rule. {s5_sig} of 60 intervals exclude zero.

## Figure S6

The Figure-9 median classification is the reference. The same existing 33/67-percentile, k-means, and GMM alternatives were applied to those exact EWBI/NIRv inputs. Alternative-method Cohen κ versus the Figure-9 median ranges from {alt_agreement.cohen_kappa.min():.3f} to {alt_agreement.cohen_kappa.max():.3f}. The median-definition Dry / Stressed self-transition probabilities range from {median_persistence.probability_percent.min():.1f}% to {median_persistence.probability_percent.max():.1f}% across land types. These are sensitivity classifications, not validation of Markov ecosystem dynamics.

## Reproducibility and source tables

Every plotted value is available in the `Figure_S4_harmonized_*`, `Figure_S5_harmonized_*`, and `Figure_S6_harmonized_*` CSV files in this directory. The script fixes all numerical-library thread counts at one and uses fixed seed 42.
"""
    (HERE / "SUPPLEMENTARY_HARMONIZATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    ewbi, state, response = load_authoritative_inputs()
    s4 = build_s4(ewbi, response)
    s5 = build_s5(ewbi, response)
    s6 = build_s6(state)
    write_report(s4, s5, s6)
    print("Task 6 supplementary harmonization complete.")
    print(f"Output: {HERE}")


if __name__ == "__main__":
    main()
