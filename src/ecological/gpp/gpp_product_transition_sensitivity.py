# HISTORICAL (kept as run). Its 'stitched 2001-2024' variant includes the duplicated 2022 MOD17 exports and two
# unconverted Natural-Forest months (2023-05/06); its 'FLUXCOM-only' variant equals the main Figure 5 analysis.
# Corrected Table S8 part a: src/ecological/gpp/gpp_table_s8_correction.py -> results/audit/gpp_table_s8/.
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr


HERE = Path(__file__).resolve().parent
DP = HERE.parents[2]
SERIES = (
    DP / "Docs" / "major_revision_analyses" / "final_paper_figures_corrected" / "tables" / "paper_standardized_anomaly_series.csv"
)
PREVIOUS_GNDVI = (
    DP / "Docs" / "major_revision_analyses" / "final_previous_GNDVI_with_test_rebuild" / "tables" / "Figure_2_previous_GNDVI_with_test_monthly_series.csv"
)
EWBI = (
    DP / "Docs" / "major_revision_analyses" / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv"
)
PREDICTION_ROOT = DP / "Docs" / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate" / "predictions"
CORRECTED_BILSTM = DP / "Docs" / "major_revision_analyses" / "corrected_true_BiLSTM_rerun_20260728" / "05_predictions"
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]


def hac_linear_trend(values: pd.Series) -> tuple[float, float, float, float]:
    values = values.dropna().sort_index()
    x = values.index.year + (values.index.month - 1) / 12.0
    x = x.to_numpy(dtype=float)
    design = sm.add_constant(x - x.min())
    fit = sm.OLS(values.to_numpy(dtype=float), design).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    slope = float(fit.params[1])
    low, high = fit.conf_int(alpha=0.05)[1]
    return slope, float(fit.pvalues[1]), float(low), float(high)


def ecological_sensitivity() -> pd.DataFrame:
    data = pd.read_csv(SERIES, parse_dates=["date"])
    previous = pd.read_csv(PREVIOUS_GNDVI, parse_dates=["date"])
    data = data[data["variable"].ne("GNDVI")]
    data = pd.concat([data, previous[previous["variable"].eq("GNDVI")]], ignore_index=True)
    data["land_type"] = data["land_type"].replace({"NaturalForests": "Natural Forests"})
    ewbi = pd.read_csv(EWBI, parse_dates=["date"])
    ewbi["land_type"] = ewbi["land_type"].replace({"NaturalForests": "Natural Forests"})

    rows = []
    datasets = {
        "stitched_2001_2024": lambda x: x,
        "fluxcom_only_2001_2021": lambda x: x[x.index.year <= 2021],
    }
    for land in sorted(data["land_type"].unique()):
        sub = data[data["land_type"].eq(land)]
        indexed = {
            variable: frame.drop_duplicates("date", keep="last").set_index("date").sort_index()
            for variable, frame in sub.groupby("variable")
        }
        gpp_raw = indexed["GPP"]["original_value"].rename("GPP")
        gpp_anomaly = indexed["GPP"]["standardized_anomaly"].rename("GPP")

        for comparator in ["NIRv", "NDVI", "GNDVI", "GOSIF"]:
            other = indexed[comparator]["original_value"].rename(comparator)
            paired_all = pd.concat([gpp_raw, other], axis=1, join="inner").dropna()
            estimates = {}
            for dataset, selector in datasets.items():
                paired = selector(paired_all)
                r, p = pearsonr(paired["GPP"], paired[comparator])
                estimates[dataset] = (float(r), float(p), len(paired), paired.index.min(), paired.index.max())
            full_r, full_p, _, _, _ = estimates["stitched_2001_2024"]
            flux_r, flux_p, _, _, _ = estimates["fluxcom_only_2001_2021"]
            changed = (full_p < 0.05) != (flux_p < 0.05)
            for dataset, (r, p, n, start, end) in estimates.items():
                rows.append(
                    {
                        "analysis_type": "raw_monthly_pearson_correlation",
                        "land_type": land,
                        "gpp_dataset": dataset,
                        "comparator": comparator,
                        "start_date": start.date().isoformat(),
                        "end_date": end.date().isoformat(),
                        "n_months": n,
                        "effect_measure": "pearson_r",
                        "estimate": r,
                        "p_value": p,
                        "ci95_low": np.nan,
                        "ci95_high": np.nan,
                        "significant_0_05": p < 0.05,
                        "stitched_estimate": full_r,
                        "fluxcom_only_estimate": flux_r,
                        "fluxcom_minus_stitched": flux_r - full_r,
                        "statistical_conclusion_changed": changed,
                    }
                )

        land_ewbi = ewbi[ewbi["land_type"].eq(land)].drop_duplicates("date").set_index("date")["ewbi"]
        paired_all = pd.concat([gpp_anomaly, land_ewbi.rename("EWBI")], axis=1, join="inner").dropna()
        estimates = {}
        for dataset, selector in datasets.items():
            paired = selector(paired_all)
            r, p = pearsonr(paired["GPP"], paired["EWBI"])
            estimates[dataset] = (float(r), float(p), len(paired), paired.index.min(), paired.index.max())
        full_r, full_p, *_ = estimates["stitched_2001_2024"]
        flux_r, flux_p, *_ = estimates["fluxcom_only_2001_2021"]
        for dataset, (r, p, n, start, end) in estimates.items():
            rows.append(
                {
                    "analysis_type": "deseasonalized_EWBI_GPP_correlation",
                    "land_type": land,
                    "gpp_dataset": dataset,
                    "comparator": "EWBI",
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "n_months": n,
                    "effect_measure": "pearson_r",
                    "estimate": r,
                    "p_value": p,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "significant_0_05": p < 0.05,
                    "stitched_estimate": full_r,
                    "fluxcom_only_estimate": flux_r,
                    "fluxcom_minus_stitched": flux_r - full_r,
                    "statistical_conclusion_changed": (full_p < 0.05) != (flux_p < 0.05),
                }
            )

        trend_estimates = {}
        for dataset, selector in datasets.items():
            series = selector(gpp_raw)
            slope, p, low, high = hac_linear_trend(series)
            trend_estimates[dataset] = (slope, p, low, high, len(series), series.index.min(), series.index.max())
        full_slope, full_p, *_ = trend_estimates["stitched_2001_2024"]
        flux_slope, flux_p, *_ = trend_estimates["fluxcom_only_2001_2021"]
        for dataset, (slope, p, low, high, n, start, end) in trend_estimates.items():
            rows.append(
                {
                    "analysis_type": "raw_GPP_linear_trend_HAC12",
                    "land_type": land,
                    "gpp_dataset": dataset,
                    "comparator": "time_year",
                    "start_date": start.date().isoformat(),
                    "end_date": end.date().isoformat(),
                    "n_months": n,
                    "effect_measure": "OLS_slope_per_year_HAC12",
                    "estimate": slope,
                    "p_value": p,
                    "ci95_low": low,
                    "ci95_high": high,
                    "significant_0_05": p < 0.05,
                    "stitched_estimate": full_slope,
                    "fluxcom_only_estimate": flux_slope,
                    "fluxcom_minus_stitched": flux_slope - full_slope,
                    "statistical_conclusion_changed": (full_p < 0.05) != (flux_p < 0.05),
                }
            )
    return pd.DataFrame(rows)


def load_gpp_predictions() -> pd.DataFrame:
    usecols = ["model", "fold", "land_type", "variable", "seed", "date", "observed", "predicted", "valid"]
    frames = []
    main_files = [
        PREDICTION_ROOT / "reviewer_safe_lstm_predictions.csv.gz",
        PREDICTION_ROOT / "reviewer_safe_remaining_models_predictions.csv.gz",
        PREDICTION_ROOT / "reviewer_safe_xgboost_predictions.csv.gz",
    ]
    for path in main_files:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
            chunk = chunk[chunk["variable"].eq("GPP") & chunk["valid"].astype(bool)]
            if path.name == "reviewer_safe_remaining_models_predictions.csv.gz":
                chunk = chunk[chunk["model"].ne("BiLSTM")]
            if len(chunk):
                frames.append(chunk)
    for path in sorted(CORRECTED_BILSTM.glob("GPP_*_corrected_predictions.csv.gz")):
        chunk = pd.read_csv(path, usecols=usecols)
        chunk = chunk[chunk["valid"].astype(bool)]
        frames.append(chunk)
    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"].astype(str), format="%Y%m")
    assert set(data["model"].unique()) == set(MODELS)
    return data


def metric_values(group: pd.DataFrame) -> tuple[float, float, int]:
    observed = group["observed"].to_numpy(dtype=float)
    predicted = group["predicted"].to_numpy(dtype=float)
    residual = observed - predicted
    denominator = np.sum((observed - observed.mean()) ** 2)
    r2 = 1.0 - np.sum(residual**2) / denominator if denominator > 0 else np.nan
    sd = np.std(observed, ddof=0)
    nrmse = np.sqrt(np.mean(residual**2)) / sd if sd > 0 else np.nan
    return float(r2), float(nrmse), len(group)


def forecast_sensitivity() -> pd.DataFrame:
    data = load_gpp_predictions()
    data["boundary_period"] = np.where(data["date"].dt.year <= 2021, "pre_boundary_le_2021", "post_boundary_ge_2022")
    seed_rows = []
    for keys, group in data.groupby(["model", "land_type", "seed", "boundary_period"], sort=False):
        r2, nrmse, n = metric_values(group)
        seed_rows.append(dict(zip(["model", "land_type", "seed", "boundary_period"], keys)) | {"r2": r2, "nrmse": nrmse, "n_predictions": n})
    seed_metrics = pd.DataFrame(seed_rows)
    summary = seed_metrics.groupby(["model", "land_type", "boundary_period"], as_index=False).agg(
        r2=("r2", "mean"),
        r2_sd_across_seeds=("r2", "std"),
        nrmse=("nrmse", "mean"),
        nrmse_sd_across_seeds=("nrmse", "std"),
        n_seeds=("seed", "nunique"),
        n_valid_predictions=("n_predictions", "sum"),
    )
    summary["r2_rank_within_land_period"] = summary.groupby(["land_type", "boundary_period"])["r2"].rank(ascending=False, method="min")
    summary["nrmse_rank_within_land_period"] = summary.groupby(["land_type", "boundary_period"])["nrmse"].rank(ascending=True, method="min")
    pre = summary[summary["boundary_period"].eq("pre_boundary_le_2021")]
    post = summary[summary["boundary_period"].eq("post_boundary_ge_2022")]
    changes = pre.merge(post, on=["model", "land_type"], suffixes=("_pre", "_post"))
    summary = summary.merge(
        changes[
            ["model", "land_type", "r2_pre", "r2_post", "nrmse_pre", "nrmse_post", "r2_rank_within_land_period_pre", "r2_rank_within_land_period_post", "nrmse_rank_within_land_period_pre", "nrmse_rank_within_land_period_post"]
        ],
        on=["model", "land_type"],
        how="left",
    )
    summary["r2_post_minus_pre"] = summary["r2_post"] - summary["r2_pre"]
    summary["nrmse_post_minus_pre"] = summary["nrmse_post"] - summary["nrmse_pre"]
    summary["r2_rank_change_post_minus_pre"] = summary["r2_rank_within_land_period_post"] - summary["r2_rank_within_land_period_pre"]
    summary["nrmse_rank_change_post_minus_pre"] = summary["nrmse_rank_within_land_period_post"] - summary["nrmse_rank_within_land_period_pre"]
    return summary.sort_values(["boundary_period", "land_type", "r2_rank_within_land_period"])


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    ecological = ecological_sensitivity()
    forecast = forecast_sensitivity()
    ecological.to_csv(HERE / "gpp_product_transition_sensitivity.csv", index=False)
    forecast.to_csv(HERE / "gpp_forecast_boundary_sensitivity.csv", index=False)

    paired = ecological[ecological["gpp_dataset"].eq("fluxcom_only_2001_2021")]
    changed = paired[paired["statistical_conclusion_changed"]]
    forecast_unique = forecast.drop_duplicates(["model", "land_type"])
    r2_stability = spearmanr(forecast_unique["r2_rank_within_land_period_pre"], forecast_unique["r2_rank_within_land_period_post"])
    nrmse_stability = spearmanr(forecast_unique["nrmse_rank_within_land_period_pre"], forecast_unique["nrmse_rank_within_land_period_post"])
    corr = paired[paired["analysis_type"].str.contains("correlation")]
    max_change = corr.loc[corr["fluxcom_minus_stitched"].abs().idxmax()]

    lines = [
        "# GPP product-transition sensitivity",
        "",
        "## Scope and data rule",
        "",
        "No FLUXCOM–MOD17 calibration was attempted or inferred. The stitched analysis uses the stored 2001–2024 GPP record exactly as analysed; the sensitivity truncates GPP and its paired dates at 2021-12, retaining FLUXCOM-era observations only.",
        "",
        "Manuscript-relevant uses identified were: raw GPP correlations supporting the NIRv-versus-NDVI comparison; EWBI–GPP response association; GPP temporal trend diagnostics; and GPP forecasting performance. Figure 5 itself labels GPP as 2001–2021, so it is already FLUXCOM-era for GPP; the stitched sensitivity here addresses the other 2001–2024 analyses.",
        "",
        "## Ecological/correlation results",
        "",
        f"Across {len(paired)} land-type/analysis comparisons, {len(changed)} changed nominal p<0.05 status after restricting to FLUXCOM-era data. "
        f"The largest absolute correlation change was {max_change['land_type']} {max_change['comparator']}: "
        f"r={max_change['stitched_estimate']:.3f} stitched versus {max_change['fluxcom_only_estimate']:.3f} FLUXCOM-only (Δr={max_change['fluxcom_minus_stitched']:+.3f}).",
        "",
        "Trend rows report the same linear slope-per-year estimand used by the detrending package, with HAC(12) standard errors/95% intervals added for serial dependence. These are sensitivity estimates, not a product calibration.",
        "",
        "## Forecast boundary results",
        "",
        "Saved processed-space out-of-sample predictions were used only; no model was retrained. Metrics were calculated within <=2021 and >=2022 subsets for each model × land type, first within seed and then averaged across seeds. Pixel rows were not treated as independent inferential replicates and no pre/post significance test was applied.",
        "",
        f"Model rank stability across the 24 model × land-type comparisons was Spearman ρ={r2_stability.statistic:.3f} for R² ranks and ρ={nrmse_stability.statistic:.3f} for nRMSE ranks. "
        "Because the post-boundary observations are a different product and occur in later rolling-origin folds, any change combines product, period, and fold effects; it cannot be attributed solely to the product transition.",
        "",
        "## Interpretation",
        "",
        "A correlation or trend conclusion is marked as changed only when nominal p<0.05 status differs between the stored stitched record and the FLUXCOM-era restriction. Direction and magnitude should be interpreted from the effect-size columns. No causal or calibration claim is made.",
    ]
    (HERE / "GPP_TRANSITION_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"ecological rows: {len(ecological)}; nominal conclusions changed: {len(changed)}")
    print(f"forecast rows: {len(forecast)}")
    print(f"rank stability R2={r2_stability.statistic:.3f}, nRMSE={nrmse_stability.statistic:.3f}")


if __name__ == "__main__":
    main()
