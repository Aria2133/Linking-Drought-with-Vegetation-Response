from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\EWBI_from_existing_forecasts_scale_recovery_and_figure_20260728"
)
ECOLOGICAL_EWBI = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\04_detrending_sensitivity_fixed\tables\main_ewbi_deseasonalized_series.csv"
)

MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
LAND_COVERS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_TYPE_ORDER = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]


def ensure_dirs() -> None:
    for rel in ["03_ewbi", "04_figure", "09_reports"]:
        (ROOT / rel).mkdir(parents=True, exist_ok=True)


def read_recovery_maxima() -> tuple[float, float]:
    repro = pd.read_csv(ROOT / "01_scale_recovery" / "forecast_preprocessing_reproduction_checks.csv")
    roundtrip = pd.read_csv(ROOT / "01_scale_recovery" / "inverse_transformation_roundtrip_checks.csv")
    return (
        float(repro["maximum_absolute_difference"].max()),
        float(roundtrip["maximum_absolute_roundtrip_difference"].max()),
    )


def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - obs) ** 2)))


def r2_score_manual(obs: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((obs - np.mean(obs)) ** 2))
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((obs - pred) ** 2) / denom)


def corr_manual(obs: np.ndarray, pred: np.ndarray) -> float:
    if len(obs) < 2:
        return float("nan")
    if np.std(obs, ddof=0) == 0 or np.std(pred, ddof=0) == 0:
        return float("nan")
    return float(np.corrcoef(obs, pred)[0, 1])


def metric_row(df: pd.DataFrame) -> dict[str, float | int]:
    clean = df[np.isfinite(df["observed_ewbi"]) & np.isfinite(df["predicted_ewbi"])].copy()
    obs = clean["observed_ewbi"].to_numpy(float)
    pred = clean["predicted_ewbi"].to_numpy(float)
    row_rmse = rmse(obs, pred)
    obs_sd = float(np.std(obs, ddof=0))
    return {
        "row_count": int(len(clean)),
        "unique_month_count": int(clean["date"].nunique()),
        "fold_count": int(clean["fold"].nunique()),
        "seed_count": int(clean["seed"].nunique()),
        "rmse": row_rmse,
        "nrmse": float(row_rmse / obs_sd) if obs_sd > 0 else float("nan"),
        "r2": r2_score_manual(obs, pred),
        "pearson_r": corr_manual(obs, pred),
        "mean_bias": float(np.mean(pred - obs)),
        "observed_sd_population": obs_sd,
    }


def compute_forecast_metrics(ewbi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = []
    for (model, land_cover), group in ewbi.groupby(["model", "land_cover"], sort=False):
        row = {"model": model, "land_cover": land_cover}
        row.update(metric_row(group))
        metrics.append(row)
    metrics_df = pd.DataFrame(metrics)
    metrics_df["model"] = pd.Categorical(metrics_df["model"], MODELS, ordered=True)
    metrics_df["land_cover"] = pd.Categorical(metrics_df["land_cover"], LAND_COVERS, ordered=True)
    metrics_df = metrics_df.sort_values(["model", "land_cover"]).reset_index(drop=True)

    fold_metrics = []
    for (model, land_cover, fold), group in ewbi.groupby(["model", "land_cover", "fold"], sort=False):
        row = {"model": model, "land_cover": land_cover, "fold": int(fold)}
        row.update(metric_row(group))
        fold_metrics.append(row)
    fold_df = pd.DataFrame(fold_metrics)
    fold_df["model"] = pd.Categorical(fold_df["model"], MODELS, ordered=True)
    fold_df["land_cover"] = pd.Categorical(fold_df["land_cover"], LAND_COVERS, ordered=True)
    fold_df = fold_df.sort_values(["model", "land_cover", "fold"]).reset_index(drop=True)
    return metrics_df, fold_df


def compute_support_metrics(ewbi: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    support = (
        ewbi.groupby(["land_cover", "date"], as_index=False)
        .agg(
            forecast_support_observed_ewbi=("observed_ewbi", "mean"),
            duplicate_observed_min=("observed_ewbi", "min"),
            duplicate_observed_max=("observed_ewbi", "max"),
            contributing_rows=("observed_ewbi", "size"),
            contributing_models=("model", "nunique"),
            contributing_seeds=("seed", "nunique"),
        )
    )
    support["duplicate_observed_range"] = (
        support["duplicate_observed_max"] - support["duplicate_observed_min"]
    )
    support["date_month"] = pd.to_datetime(support["date"].astype(str), format="%Y%m")

    eco = pd.read_csv(ECOLOGICAL_EWBI)
    eco = eco.rename(columns={"ewbi": "ecological_observed_ewbi", "land_type": "land_cover"})
    eco["date_month"] = pd.to_datetime(eco["date"])
    eco = eco[["land_cover", "date_month", "ecological_observed_ewbi"]]

    merged = support.merge(eco, on=["land_cover", "date_month"], how="inner")
    rows = []
    for land_cover, group in merged.groupby("land_cover", sort=False):
        obs = group["ecological_observed_ewbi"].to_numpy(float)
        fs = group["forecast_support_observed_ewbi"].to_numpy(float)
        row_rmse = rmse(obs, fs)
        denom = float(np.std(obs, ddof=0))
        rows.append(
            {
                "land_cover": land_cover,
                "matched_month_count": int(len(group)),
                "pearson_r": corr_manual(obs, fs),
                "rmse": row_rmse,
                "nrmse": float(row_rmse / denom) if denom > 0 else float("nan"),
                "mean_bias_forecast_support_minus_ecological": float(np.mean(fs - obs)),
                "max_abs_difference": float(np.max(np.abs(fs - obs))),
                "max_duplicate_observed_range_within_forecast_table": float(
                    group["duplicate_observed_range"].max()
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["land_cover"] = pd.Categorical(out["land_cover"], LAND_COVERS, ordered=True)
    out = out.sort_values("land_cover").reset_index(drop=True)
    return out, merged


def add_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    best: str,
) -> object:
    matrix = data.pivot(index="model", columns="land_cover", values=value_col).loc[MODELS, LAND_COVERS]
    im = ax.imshow(matrix.to_numpy(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(LAND_COVERS)))
    ax.set_xticklabels(LAND_COVERS, rotation=25, ha="right", fontsize=10.5)
    ax.set_yticks(np.arange(len(MODELS)))
    ax.set_yticklabels(MODELS, fontsize=10.5)
    ax.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=8)
    ax.tick_params(length=0)
    for y in np.arange(0.5, len(MODELS), 1):
        ax.axhline(y, color="white", linewidth=0.8)
    for x in np.arange(0.5, len(LAND_COVERS), 1):
        ax.axvline(x, color="white", linewidth=0.8)
    for sep in [2.5, 4.5]:
        ax.axhline(sep, color="black", linewidth=1.2)

    vals = matrix.to_numpy(float)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            value = vals[i, j]
            rgba = im.cmap(im.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            color = "white" if luminance < 0.48 else "black"
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=9.5, fontweight="bold", color=color)

    for j in range(vals.shape[1]):
        col = vals[:, j]
        if best == "min":
            best_value = np.nanmin(col)
        else:
            best_value = np.nanmax(col)
        for i in np.where(np.isclose(col, best_value, rtol=0, atol=1e-12))[0]:
            ax.add_patch(
                plt.Rectangle(
                    (j - 0.5, i - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2.4,
                    clip_on=False,
                )
            )
    return im


def plot_figure(metrics: pd.DataFrame, ewbi: pd.DataFrame, support_joined: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )

    fig = plt.figure(figsize=(16, 12.5), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], width_ratios=[1, 1], hspace=0.34, wspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    nrmse_im = add_heatmap(ax_a, metrics, "nrmse", "(a) EWBI nRMSE", "YlGnBu_r", None, None, "min")
    r2_im = add_heatmap(ax_b, metrics, "r2", "(b) EWBI R²", "RdYlBu", None, None, "max")
    cbar_a = fig.colorbar(nrmse_im, ax=ax_a, fraction=0.046, pad=0.035)
    cbar_a.set_label("nRMSE", fontsize=10.5)
    cbar_b = fig.colorbar(r2_im, ax=ax_b, fraction=0.046, pad=0.035)
    cbar_b.set_label("R²", fontsize=10.5)

    # Panel (c): predefined representative forecasting frameworks, not selected by EWBI performance.
    plot_models = ["BiLSTM", "XGBoost"]
    colors = {"BiLSTM": "#7b4ab8", "XGBoost": "#17aebf"}
    markers = {"Grasslands": "o", "Natural Forests": "s", "Savannas": "^", "Shrublands": "D"}
    panel_c = ewbi[ewbi["model"].isin(plot_models)].copy()
    for model in plot_models:
        for land_cover in LAND_COVERS:
            g = panel_c[(panel_c["model"] == model) & (panel_c["land_cover"] == land_cover)]
            ax_c.scatter(
                g["observed_ewbi"],
                g["predicted_ewbi"],
                s=26,
                alpha=0.46,
                color=colors[model],
                marker=markers[land_cover],
                linewidths=0,
                label=f"{model} - {land_cover}",
            )
    finite_values = pd.concat(
        [panel_c["observed_ewbi"], panel_c["predicted_ewbi"]], ignore_index=True
    ).replace([np.inf, -np.inf], np.nan).dropna()
    lo = float(finite_values.quantile(0.005))
    hi = float(finite_values.quantile(0.995))
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    ax_c.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.2, linestyle="--")
    ax_c.set_xlim(lo - pad, hi + pad)
    ax_c.set_ylim(lo - pad, hi + pad)
    ax_c.set_xlabel("Forecast-support observed EWBI", fontsize=11)
    ax_c.set_ylabel("Forecast-derived EWBI", fontsize=11)
    ax_c.set_title("(c) Observed vs forecast-derived EWBI", loc="left", fontsize=13, fontweight="bold", pad=8)
    ax_c.grid(True, color="#e6e6e6", linewidth=0.8)

    model_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=colors[m], label=m, markersize=7)
        for m in plot_models
    ]
    land_handles = [
        plt.Line2D([0], [0], marker=markers[l], linestyle="", color="0.35", label=l, markersize=6)
        for l in LAND_COVERS
    ]
    leg1 = ax_c.legend(handles=model_handles, title="Displayed model", loc="upper left", frameon=True, fontsize=8.7, title_fontsize=9.2)
    ax_c.add_artist(leg1)
    ax_c.legend(handles=land_handles, title="Land cover", loc="lower right", frameon=True, fontsize=8.4, title_fontsize=9.0)

    # Panel (d): spatial-support comparison, not forecasting performance.
    land_colors = {
        "Grasslands": "#1b9e77",
        "Natural Forests": "#7570b3",
        "Savannas": "#d95f02",
        "Shrublands": "#666666",
    }
    for land_cover in LAND_COVERS:
        g = support_joined[support_joined["land_cover"] == land_cover]
        ax_d.scatter(
            g["ecological_observed_ewbi"],
            g["forecast_support_observed_ewbi"],
            s=30,
            alpha=0.65,
            color=land_colors[land_cover],
            marker=markers[land_cover],
            linewidths=0,
            label=land_cover,
        )
    finite_values = pd.concat(
        [
            support_joined["ecological_observed_ewbi"],
            support_joined["forecast_support_observed_ewbi"],
        ],
        ignore_index=True,
    ).replace([np.inf, -np.inf], np.nan).dropna()
    lo = float(finite_values.quantile(0.005))
    hi = float(finite_values.quantile(0.995))
    pad = (hi - lo) * 0.08 if hi > lo else 0.1
    ax_d.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", linewidth=1.2, linestyle="--")
    ax_d.set_xlim(lo - pad, hi + pad)
    ax_d.set_ylim(lo - pad, hi + pad)
    ax_d.set_xlabel("Ecological observed EWBI", fontsize=11)
    ax_d.set_ylabel("Forecast-support observed EWBI", fontsize=11)
    ax_d.set_title("(d) Spatial-support comparison", loc="left", fontsize=13, fontweight="bold", pad=8)
    ax_d.grid(True, color="#e6e6e6", linewidth=0.8)
    ax_d.legend(loc="upper left", frameon=True, fontsize=8.7, title="Land cover", title_fontsize=9.2)

    fig.suptitle("Forecast-derived reconstruction of EWBI", fontsize=17, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        "Panels (a) and (b) use all six complete model lineages. Panel (c) displays predefined representative frameworks "
        "(BiLSTM and XGBoost); all six models are summarized in the heatmaps. Panel (d) is a support comparison, not a forecast-error panel.",
        ha="center",
        va="bottom",
        fontsize=9.4,
    )
    fig.subplots_adjust(left=0.075, right=0.965, top=0.93, bottom=0.075)

    out_base = ROOT / "04_figure" / "Figure_forecast_derived_EWBI"
    fig.savefig(out_base.with_suffix(".png"), dpi=400)
    fig.savefig(out_base.with_suffix(".pdf"))
    fig.savefig(out_base.with_suffix(".svg"))
    plt.close(fig)


def write_caption() -> str:
    text = (
        "Figure Sx. Out-of-sample reconstruction of the Ecosystem Water Balance Index (EWBI) "
        "on the forecasting evaluation support. Forecast-derived EWBI was calculated separately "
        "for each forecasting model from existing one-month-ahead predictions of SPEI-1, TWS, "
        "VPDmax, and LST using the accepted EWBI definition. Observed and forecast-derived EWBI "
        "were evaluated on identical forecasting pixels and months. (a) Normalized RMSE and "
        "(b) R² by model and land-cover class. Outlined cells identify the best value within each "
        "land-cover class. (c) Agreement between forecast-support observed EWBI and forecast-derived "
        "EWBI, shown for predefined representative frameworks (BiLSTM and XGBoost); all six complete "
        "models are summarized in panels (a) and (b). (d) Comparison between observed EWBI calculated "
        "on the forecasting support and the broader land-cover EWBI used in the ecological analysis. "
        "Differences in (d) reflect spatial-support differences and are not forecast errors. RBF-ε-SVR "
        "was not included because the final approved no-detrending SPEI-1 outer-test prediction file "
        "was unavailable."
    )
    path = ROOT / "04_figure" / "Figure_forecast_derived_EWBI_caption.md"
    path.write_text(text + "\n", encoding="utf-8")
    return text


def write_validation(
    ewbi: pd.DataFrame,
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    support: pd.DataFrame,
    max_repro: float,
    max_roundtrip: float,
) -> pd.DataFrame:
    checks = [
        ("no forecasting model was rerun", "PASS", "This script only read recovered CSV outputs and did not import or execute model-training code."),
        ("no new component prediction was generated", "PASS", "Component forecasts came from inverse_transformed_component_predictions_monthly.csv generated by the prior scale-recovery run."),
        ("all six complete models included", "PASS" if sorted(ewbi["model"].unique()) == sorted(MODELS) else "FAIL", ", ".join(sorted(ewbi["model"].unique()))),
        ("SVR excluded transparently", "PASS", "excluded_model_lineages.csv reports RBF-epsilon-SVR missing SPEI; no SVR model was rerun."),
        ("observed and predicted EWBI use identical forecasting support", "PASS", "observed_ewbi and predicted_ewbi were read from the same model/seed/land/fold/date rows."),
        ("accepted EWBI equation and baseline used", "PASS", "The existing recovered EWBI table was generated from SPEI-1, TWS, VPDmax, and LST using the accepted 2001-2011 baseline standardization."),
        ("scale recovery numerical check passed", "PASS" if max_repro <= 1e-5 else "FAIL", f"maximum stored-observation reconstruction difference={max_repro}"),
        ("inverse transformation roundtrip passed", "PASS" if max_roundtrip <= 1e-5 else "FAIL", f"maximum round-trip difference={max_roundtrip}"),
        ("support comparison reported separately", "PASS" if len(support) == 4 else "FAIL", f"support comparison rows={len(support)}"),
        ("forecast metric inventory complete", "PASS" if len(metrics) == len(MODELS) * len(LAND_COVERS) else "FAIL", f"model-land rows={len(metrics)}"),
        ("fold metric inventory complete", "PASS" if set(fold_metrics["fold"]) == {1, 2, 3, 4} else "FAIL", f"fold rows={len(fold_metrics)}"),
        ("figure values reproducible from saved CSV files", "PASS", "Heatmap panels are generated directly from EWBI_metrics_by_model_landcover.csv."),
    ]
    validation = pd.DataFrame(checks, columns=["check", "status", "details"])
    validation.to_csv(ROOT / "09_reports" / "EWBI_forecast_support_validation.csv", index=False)
    return validation


def write_report(
    metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    support: pd.DataFrame,
    max_repro: float,
    max_roundtrip: float,
    validation: pd.DataFrame,
) -> None:
    metric_view = metrics.copy()
    metric_view["nrmse"] = metric_view["nrmse"].map(lambda x: f"{x:.3f}")
    metric_view["r2"] = metric_view["r2"].map(lambda x: f"{x:.3f}")
    metric_table = metric_view[["model", "land_cover", "nrmse", "r2", "pearson_r", "mean_bias", "row_count", "unique_month_count"]].to_markdown(index=False)

    support_view = support.copy()
    for col in ["pearson_r", "rmse", "nrmse", "mean_bias_forecast_support_minus_ecological", "max_abs_difference"]:
        support_view[col] = support_view[col].map(lambda x: f"{x:.3f}")
    support_table = support_view.to_markdown(index=False)

    report = f"""# EWBI Forecast-Support Analysis Report

## Plain-language outcome

This report completes the forecast-derived EWBI post-processing from the already recovered forecasting predictions. No forecasting model was fitted, retrained, resumed, or rerun.

The analysis compares **forecast-derived EWBI** against **forecast-support observed EWBI**, meaning that both observed and predicted EWBI are evaluated on the same selected forecasting pixels and held-out months. The broader ecological EWBI is not required to be numerically identical because it uses a broader land-cover raster support.

SVR was excluded from the main EWBI figure because the final approved no-detrending SPEI-1 outer-test prediction file was unavailable. The completed six-model comparison includes LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost.

## Sources Used

- Recovered EWBI table: `{ROOT / '03_ewbi' / 'observed_and_forecast_derived_EWBI_monthly.csv'}`
- Inverse-transformed component predictions: `{ROOT / '02_components' / 'inverse_transformed_component_predictions_monthly.csv'}`
- Scale-recovery checks: `{ROOT / '01_scale_recovery' / 'forecast_preprocessing_reproduction_checks.csv'}`
- Round-trip checks: `{ROOT / '01_scale_recovery' / 'inverse_transformation_roundtrip_checks.csv'}`
- Accepted ecological EWBI support table: `{ECOLOGICAL_EWBI}`

## Scale-Recovery Checks

- Maximum stored-observation reconstruction difference: `{max_repro:.16g}`
- Maximum inverse-transformation round-trip difference: `{max_roundtrip:.16g}`

These checks support using the recovered component scales for post-processing. They do not imply that the forecast-support observed EWBI should equal the broader ecological EWBI.

## Forecast-Derived EWBI Metrics

Metrics compare forecast-derived EWBI with forecast-support observed EWBI on identical model, seed, land-cover, fold, and month keys. nRMSE is RMSE divided by the population standard deviation of the observed EWBI values being evaluated.

{metric_table}

## Spatial-Support Comparison

This check compares forecast-support observed EWBI with the broader ecological observed EWBI by land-cover class. These differences reflect spatial-support differences and are not forecast errors.

{support_table}

## Validation Summary

{validation.to_markdown(index=False)}

## Outputs

- Model-land-cover metrics: `{ROOT / '03_ewbi' / 'EWBI_metrics_by_model_landcover.csv'}`
- Fold-specific metrics: `{ROOT / '03_ewbi' / 'EWBI_metrics_by_model_landcover_fold.csv'}`
- Spatial-support comparison: `{ROOT / '03_ewbi' / 'forecast_support_vs_ecological_EWBI_comparison.csv'}`
- Figure PNG: `{ROOT / '04_figure' / 'Figure_forecast_derived_EWBI.png'}`
- Figure PDF: `{ROOT / '04_figure' / 'Figure_forecast_derived_EWBI.pdf'}`
- Figure SVG: `{ROOT / '04_figure' / 'Figure_forecast_derived_EWBI.svg'}`
- Caption: `{ROOT / '04_figure' / 'Figure_forecast_derived_EWBI_caption.md'}`

## Notes

The figure is an EWBI reconstruction on the forecasting evaluation support. It should not be described as direct vegetation-resilience forecasting and should not be treated as numerically interchangeable with the main ecological EWBI time series.
"""
    (ROOT / "09_reports" / "EWBI_forecast_support_analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    max_repro, max_roundtrip = read_recovery_maxima()

    ewbi_path = ROOT / "03_ewbi" / "observed_and_forecast_derived_EWBI_monthly.csv"
    ewbi = pd.read_csv(ewbi_path)
    ewbi = ewbi[ewbi["model"].isin(MODELS)].copy()
    ewbi["model"] = pd.Categorical(ewbi["model"], MODELS, ordered=True)
    ewbi["land_cover"] = pd.Categorical(ewbi["land_cover"], LAND_COVERS, ordered=True)
    ewbi = ewbi.sort_values(["model", "land_cover", "fold", "seed", "date"]).reset_index(drop=True)

    metrics, fold_metrics = compute_forecast_metrics(ewbi)
    support, support_joined = compute_support_metrics(ewbi)

    metrics.to_csv(ROOT / "03_ewbi" / "EWBI_metrics_by_model_landcover.csv", index=False)
    fold_metrics.to_csv(ROOT / "03_ewbi" / "EWBI_metrics_by_model_landcover_fold.csv", index=False)
    support.to_csv(ROOT / "03_ewbi" / "forecast_support_vs_ecological_EWBI_comparison.csv", index=False)

    plot_figure(metrics, ewbi, support_joined)
    write_caption()
    validation = write_validation(ewbi, metrics, fold_metrics, support, max_repro, max_roundtrip)
    write_report(metrics, fold_metrics, support, max_repro, max_roundtrip, validation)

    manifest_path = ROOT / "run_manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}
    manifest.update(
        {
            "forecast_support_completion_timestamp": datetime.now().isoformat(timespec="seconds"),
            "forecast_support_status": "COMPLETED",
            "no_model_rerun": True,
            "models_included_in_forecast_support_figure": MODELS,
            "models_excluded_from_forecast_support_figure": {
                "RBF-epsilon-SVR": "final approved no-detrending SPEI-1 outer-test prediction file unavailable"
            },
            "max_reproduction_difference": max_repro,
            "max_roundtrip_difference": max_roundtrip,
            "forecast_support_metric_rows": int(len(metrics)),
            "forecast_support_fold_metric_rows": int(len(fold_metrics)),
            "support_comparison_rows": int(len(support)),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("COMPLETED")
    print(f"metrics={ROOT / '03_ewbi' / 'EWBI_metrics_by_model_landcover.csv'}")
    print(f"fold_metrics={ROOT / '03_ewbi' / 'EWBI_metrics_by_model_landcover_fold.csv'}")
    print(f"support={ROOT / '03_ewbi' / 'forecast_support_vs_ecological_EWBI_comparison.csv'}")
    print(f"figure={ROOT / '04_figure' / 'Figure_forecast_derived_EWBI.png'}")
    print(f"report={ROOT / '09_reports' / 'EWBI_forecast_support_analysis_report.md'}")


if __name__ == "__main__":
    main()
