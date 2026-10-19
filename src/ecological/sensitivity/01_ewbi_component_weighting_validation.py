from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from config import (
    BASELINE,
    FIGURES,
    FORCING,
    LAND_LABELS,
    LAND_TYPES,
    SUPPLEMENTARY,
    TABLES,
    VEGETATION,
)
from revision_utils import (
    baseline_monthly_anomaly,
    complete_landtype_frame,
    correlation_with_ci,
    moving_block_bootstrap,
    period_label,
    signed_forcing_anomalies,
)


def pca_ewbi(signed: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    valid = signed.dropna()
    scaler = StandardScaler()
    x = scaler.fit_transform(valid)
    pca = PCA(n_components=1, random_state=42)
    score = pca.fit_transform(x).ravel()
    equal = valid.mean(axis=1).to_numpy()
    if np.corrcoef(score, equal)[0, 1] < 0:
        score *= -1
        pca.components_[0] *= -1
    output = pd.Series(np.nan, index=signed.index)
    output.loc[valid.index] = score
    return output, pca.components_[0]


def regression_ewbi(signed: pd.DataFrame, vegetation: pd.DataFrame) -> tuple[pd.Series, np.ndarray]:
    target = vegetation.mean(axis=1)
    joined = signed.join(target.rename("target")).dropna()
    model = RidgeCV(alphas=np.logspace(-3, 3, 25))
    model.fit(joined[FORCING], joined["target"])
    weights = model.coef_
    if weights.sum() < 0:
        weights = -weights
    weights = weights / np.sum(np.abs(weights))
    output = pd.Series(np.nan, index=signed.index)
    output.loc[joined.index] = joined[FORCING].to_numpy() @ weights
    return output, weights


def main() -> None:
    correlation_rows = []
    component_rows = []
    loco_rows = []
    weight_rows = []
    period_rows = []
    ci_rows = []
    plotting = []

    for land_type in LAND_TYPES:
        frame = complete_landtype_frame(land_type, "monthwise")
        signed = signed_forcing_anomalies(frame)
        equal = signed.mean(axis=1)
        veg_anom = pd.DataFrame(
            {variable: baseline_monthly_anomaly(frame[variable]) for variable in VEGETATION}
        )
        pca_score, pca_weights = pca_ewbi(signed)
        regression_score, regression_weights = regression_ewbi(signed, veg_anom)

        for scheme, weights in {
            "equal": np.repeat(0.25, 4),
            "pca": pca_weights / np.sum(np.abs(pca_weights)),
            "ridge_to_mean_vegetation": regression_weights,
        }.items():
            for variable, weight in zip(FORCING, weights):
                weight_rows.append(
                    {
                        "land_type": land_type,
                        "weighting": scheme,
                        "component": variable,
                        "signed_weight": float(weight),
                    }
                )

        indices = {
            "EWBI_equal": equal,
            "EWBI_PCA": pca_score,
            "EWBI_regression": regression_score,
        }
        for response in VEGETATION:
            for name, index_series in indices.items():
                r, p, low, high = correlation_with_ci(index_series, veg_anom[response])
                correlation_rows.append(
                    {
                        "land_type": land_type,
                        "response": response,
                        "index": name,
                        "pearson_r": r,
                        "p_value": p,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
            for component in FORCING:
                r, p, low, high = correlation_with_ci(signed[component], veg_anom[response])
                component_rows.append(
                    {
                        "land_type": land_type,
                        "response": response,
                        "predictor": component,
                        "pearson_r": r,
                        "p_value": p,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
            for omitted in FORCING:
                reduced = signed.drop(columns=omitted).mean(axis=1)
                r, p, low, high = correlation_with_ci(reduced, veg_anom[response])
                loco_rows.append(
                    {
                        "land_type": land_type,
                        "response": response,
                        "omitted_component": omitted,
                        "pearson_r": r,
                        "p_value": p,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )

        labels = period_label(equal.index)
        groups = [
            equal[labels == label].dropna().to_numpy()
            for label in [
                "Baseline 2001-2011",
                "Historic drought 2012-2016",
                "Recent period 2020-2024",
            ]
        ]
        anova = stats.f_oneway(*groups)
        kruskal = stats.kruskal(*groups)
        period_rows.append(
            {
                "land_type": land_type,
                "anova_f": float(anova.statistic),
                "anova_p": float(anova.pvalue),
                "kruskal_h": float(kruskal.statistic),
                "kruskal_p": float(kruskal.pvalue),
            }
        )
        for label, group in zip(
            ["Baseline 2001-2011", "Historic drought 2012-2016", "Recent period 2020-2024"],
            groups,
        ):
            estimate, low, high = moving_block_bootstrap(group, block_length=12)
            ci_rows.append(
                {
                    "land_type": land_type,
                    "period": label,
                    "mean_ewbi": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_months": int(np.isfinite(group).sum()),
                }
            )
        plotting.append(
            pd.DataFrame(
                {
                    "date": equal.index,
                    "EWBI": equal.values,
                    "period": labels.values,
                    "land_type": LAND_LABELS[land_type],
                }
            )
        )

    correlations = pd.DataFrame(correlation_rows)
    components = pd.DataFrame(component_rows)
    loco = pd.DataFrame(loco_rows)
    weights = pd.DataFrame(weight_rows)
    tests = pd.DataFrame(period_rows)
    ci = pd.DataFrame(ci_rows)

    correlations.to_csv(TABLES / "ewbi_vegetation_validation.csv", index=False)
    components.to_csv(TABLES / "ewbi_component_comparison.csv", index=False)
    loco.to_csv(TABLES / "ewbi_leave_one_component_out.csv", index=False)
    weights.to_csv(TABLES / "ewbi_alternative_weights.csv", index=False)
    tests.to_csv(TABLES / "ewbi_period_tests.csv", index=False)
    ci.to_csv(TABLES / "ewbi_period_bootstrap_ci.csv", index=False)
    with pd.ExcelWriter(TABLES / "ewbi_validation_revision.xlsx") as writer:
        correlations.to_excel(writer, sheet_name="vegetation_validation", index=False)
        components.to_excel(writer, sheet_name="component_comparison", index=False)
        loco.to_excel(writer, sheet_name="leave_one_out", index=False)
        weights.to_excel(writer, sheet_name="weights", index=False)
        tests.to_excel(writer, sheet_name="period_tests", index=False)
        ci.to_excel(writer, sheet_name="period_CI", index=False)

    heat = (
        correlations[correlations["index"] == "EWBI_equal"]
        .pivot(index="land_type", columns="response", values="pearson_r")
        .reindex(LAND_TYPES)
    )
    plt.figure(figsize=(9, 5.5))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1)
    plt.title("EWBI association with vegetation and productivity anomalies", fontsize=14)
    plt.xlabel("Response index")
    plt.ylabel("Land type")
    plt.tight_layout()
    plt.savefig(FIGURES / "ewbi_validation_heatmap.png", dpi=400)
    plt.savefig(FIGURES / "ewbi_validation_heatmap.pdf")
    plt.close()

    plot_frame = pd.concat(plotting, ignore_index=True)
    period_order = [
        "Baseline 2001-2011",
        "Historic drought 2012-2016",
        "Recent period 2020-2024",
    ]
    plot_frame = plot_frame[plot_frame["period"].isin(period_order)]
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=plot_frame,
        x="land_type",
        y="EWBI",
        hue="period",
        hue_order=period_order,
        showfliers=False,
    )
    plt.axhline(0, color="0.25", linewidth=0.8)
    plt.title("EWBI separation among baseline and drought periods", fontsize=14)
    plt.xlabel("")
    plt.ylabel("EWBI standardized forcing score")
    plt.legend(title="", ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(FIGURES / "ewbi_period_separation.png", dpi=400, bbox_inches="tight")
    plt.savefig(FIGURES / "ewbi_period_separation.pdf", bbox_inches="tight")
    plt.close()

    weight_plot = weights.pivot_table(
        index=["land_type", "weighting"], columns="component", values="signed_weight"
    ).reset_index()
    weight_plot.to_csv(SUPPLEMENTARY / "ewbi_weighting_plot_data.csv", index=False)
    print("EWBI validation outputs written.")


if __name__ == "__main__":
    main()
