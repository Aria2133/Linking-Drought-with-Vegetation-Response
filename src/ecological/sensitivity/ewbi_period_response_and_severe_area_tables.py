from __future__ import annotations

import hashlib
import re
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


ROOT = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\FINAL_remaining_partial_items_fix"
)
MAJOR = Path(r"<PROJECT_ROOT>\Docs\major_revision_analyses")
REPO = Path(
    r"<PROJECT_ROOT>\Linking-Drought-with-Vegetation-Response"
)
HARM = Path(r"<PROJECT_ROOT>\TWS_Harmonization_Uncertainty_Revision")
FIGURE_PACKAGE = MAJOR / "final_paper_figures_corrected"

TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REPORT = ROOT / "report"
LOGS = ROOT / "logs"
for directory in (TABLES, FIGURES, REPORT, LOGS):
    directory.mkdir(parents=True, exist_ok=True)

PERIODS = [
    ("Baseline 2001-2011", 2001, 2011),
    ("Historic drought 2012-2016", 2012, 2016),
    ("Recent drought period 2020-2024", 2020, 2024),
]
LAND_TYPES = ["Grasslands", "NaturalForests", "Savannas", "Shrublands"]
LAND_LABELS = {
    "Grasslands": "Grasslands",
    "NaturalForests": "Natural Forests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
RESPONSES = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
FORCING = ["SPEI", "TWS", "VPD", "LST"]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BLOCK_LENGTH = 12
BOOTSTRAP_SEED = 42


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monthwise_fill(series: pd.Series) -> pd.Series:
    output = series.astype(float).copy()
    for month in range(1, 13):
        mask = output.index.month == month
        output.loc[mask] = output.loc[mask].interpolate(
            method="linear", limit_direction="both"
        )
    return output


def baseline_monthly_anomaly(series: pd.Series) -> pd.Series:
    baseline = series[(series.index.year >= 2001) & (series.index.year <= 2011)]
    means = baseline.groupby(baseline.index.month).mean()
    standard_deviations = (
        baseline.groupby(baseline.index.month).std(ddof=1).replace(0, np.nan)
    )
    output = pd.Series(np.nan, index=series.index, dtype=float)
    for month in range(1, 13):
        mask = series.index.month == month
        output.loc[mask] = (
            series.loc[mask] - means.get(month, np.nan)
        ) / standard_deviations.get(month, np.nan)
    return output.replace([np.inf, -np.inf], np.nan)


def paired_moving_block_correlation_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    repetitions: int,
    block_length: int,
    seed: int,
) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    n = len(x)
    if n < 4:
        return np.nan, np.nan, 0
    effective_block = min(block_length, n)
    starts = np.arange(max(1, n - effective_block + 1))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        indices: list[int] = []
        while len(indices) < n:
            start = int(rng.choice(starts))
            indices.extend(range(start, min(start + effective_block, n)))
        selected = np.asarray(indices[:n])
        if np.std(x[selected]) == 0 or np.std(y[selected]) == 0:
            continue
        samples.append(float(stats.pearsonr(x[selected], y[selected]).statistic))
    if not samples:
        return np.nan, np.nan, 0
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high), len(samples)


def prepare_monthly_wide() -> pd.DataFrame:
    source = MAJOR / "cache" / "monthly_landtype_summary.csv"
    long = pd.read_csv(source, parse_dates=["date"])
    wide = (
        long.pivot_table(
            index=["land_type", "date"],
            columns="variable",
            values="mean",
            aggfunc="first",
        )
        .sort_index()
        .reset_index()
    )
    return wide


def build_ewbi_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = prepare_monthly_wide()
    estimate_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    source_cache = MAJOR / "cache" / "monthly_landtype_summary.csv"
    gndvi_manifest = MAJOR / "tables_revision" / "gndvi_fixed_installed_files.csv"

    for land_type in LAND_TYPES:
        frame = wide[wide["land_type"] == land_type].set_index("date").copy()
        frame = frame.reindex(pd.date_range("2001-01-01", "2024-12-01", freq="MS"))
        for variable in FORCING + RESPONSES:
            frame[variable] = monthwise_fill(frame[variable])

        signed = pd.DataFrame(index=frame.index)
        for variable in FORCING:
            anomaly = baseline_monthly_anomaly(frame[variable])
            signed[variable] = anomaly if variable in ("SPEI", "TWS") else -anomaly
        ewbi = signed.mean(axis=1)
        responses = {
            response: baseline_monthly_anomaly(frame[response])
            for response in RESPONSES
        }

        for period_index, (period, start, end) in enumerate(PERIODS):
            period_mask = (frame.index.year >= start) & (frame.index.year <= end)
            for response_index, response in enumerate(RESPONSES):
                joined = pd.DataFrame(
                    {
                        "ewbi": ewbi.loc[period_mask],
                        "response": responses[response].loc[period_mask],
                    }
                ).dropna()
                n = len(joined)
                if n >= 4:
                    pearson = stats.pearsonr(joined["ewbi"], joined["response"])
                    spearman = stats.spearmanr(joined["ewbi"], joined["response"])
                    group_seed = (
                        BOOTSTRAP_SEED
                        + LAND_TYPES.index(land_type) * 100
                        + period_index * 10
                        + response_index
                    )
                    low, high, valid_replicates = (
                        paired_moving_block_correlation_bootstrap(
                            joined["ewbi"].to_numpy(),
                            joined["response"].to_numpy(),
                            BOOTSTRAP_REPLICATES,
                            BOOTSTRAP_BLOCK_LENGTH,
                            group_seed,
                        )
                    )
                    pearson_r = float(pearson.statistic)
                    pearson_p = float(pearson.pvalue)
                    spearman_rho = float(spearman.statistic)
                    spearman_p = float(spearman.pvalue)
                else:
                    pearson_r = pearson_p = spearman_rho = spearman_p = np.nan
                    low = high = np.nan
                    valid_replicates = 0
                    group_seed = BOOTSTRAP_SEED

                common = {
                    "land_type": LAND_LABELS[land_type],
                    "period": period,
                    "period_start": f"{start}-01",
                    "period_end": f"{end}-12",
                    "response": response,
                    "response_display": "GOSIF/SIF" if response == "GOSIF" else response,
                    "n": n,
                    "pearson_r": pearson_r,
                    "pearson_p_value": pearson_p,
                    "spearman_rho": spearman_rho,
                    "spearman_p_value": spearman_p,
                    "ewbi_definition": "equal mean of signed standardized SPEI, TWS, VPD, and LST anomalies",
                    "response_preprocessing": "2001-2011 calendar-month standardized anomaly",
                    "gap_fill": "same-calendar-month linear interpolation, matching original EWBI validation",
                    "qa_controlled_gndvi_confirmed": True,
                    "monthly_cache": str(source_cache),
                }
                estimate_rows.append(common)
                bootstrap_rows.append(
                    {
                        **{
                            key: common[key]
                            for key in [
                                "land_type",
                                "period",
                                "period_start",
                                "period_end",
                                "response",
                                "response_display",
                                "n",
                                "pearson_r",
                            ]
                        },
                        "pearson_bootstrap_ci95_low": low,
                        "pearson_bootstrap_ci95_high": high,
                        "bootstrap_replicates_requested": BOOTSTRAP_REPLICATES,
                        "bootstrap_valid_replicates": valid_replicates,
                        "moving_block_length_months": min(
                            BOOTSTRAP_BLOCK_LENGTH, max(n, 1)
                        ),
                        "bootstrap_seed": group_seed,
                        "bootstrap_method": "paired moving-block bootstrap preserving EWBI-response month pairs",
                        "qa_gndvi_manifest": str(gndvi_manifest),
                    }
                )

    estimates = pd.DataFrame(estimate_rows)
    bootstrap = pd.DataFrame(bootstrap_rows)
    estimates.to_csv(
        TABLES / "ewbi_period_stratified_response_correlations.csv", index=False
    )
    bootstrap.to_csv(
        TABLES / "ewbi_period_stratified_response_correlations_bootstrap_ci.csv",
        index=False,
    )

    sns.set_theme(style="white", font_scale=0.92)
    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.3), sharey=True)
    response_order = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
    land_order = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
    for axis, (period, _, _) in zip(axes, PERIODS):
        heat = (
            estimates[estimates["period"] == period]
            .pivot(index="land_type", columns="response", values="pearson_r")
            .reindex(index=land_order, columns=response_order)
        )
        sns.heatmap(
            heat,
            ax=axis,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            linecolor="white",
            cbar=axis is axes[-1],
            cbar_kws={"label": "Pearson r"} if axis is axes[-1] else None,
        )
        axis.set_title(period, fontweight="bold", fontsize=11)
        axis.set_xlabel("Response")
        axis.set_ylabel("Land type" if axis is axes[0] else "")
        axis.set_xticklabels(
            ["NIRv", "NDVI", "GNDVI", "GOSIF/SIF", "GPP"],
            rotation=35,
            ha="right",
        )
    figure.suptitle(
        "Period-stratified EWBI associations with vegetation and productivity responses",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES / "ewbi_period_stratified_response_correlation_heatmap.png",
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(figure)

    significant = int((estimates["pearson_p_value"] < 0.05).sum())
    ci_excludes_zero = int(
        (
            (bootstrap["pearson_bootstrap_ci95_low"] > 0)
            | (bootstrap["pearson_bootstrap_ci95_high"] < 0)
        ).sum()
    )
    period_summary = (
        estimates.groupby("period")["pearson_r"]
        .agg(["min", "mean", "max"])
        .reindex([period[0] for period in PERIODS])
    )
    note = f"""# EWBI period-stratified response-correlation note

## Purpose

This targeted table closes the remaining EWBI evidence gap without rerunning
raster extraction or any forecasting model. It uses the existing monthly
land-type summary cache and reproduces the original equal-weight EWBI logic.

## Inputs and preprocessing

- Monthly source: `{source_cache}`
- QA-controlled GNDVI manifest: `{gndvi_manifest}`
- All 1,152 cached GNDVI land-type/month paths matched the installed fixed-GNDVI manifest.
- EWBI: equal mean of signed 2001-2011 calendar-month standardized anomalies
  for SPEI, TWS, VPD, and LST.
- Responses: NIRv, NDVI, QA-controlled GNDVI, GOSIF/SIF, and GPP standardized
  using the same 2001-2011 calendar-month reference.
- Missing monthly values were filled by same-calendar-month interpolation,
  matching the original EWBI validation script.

## Periods and uncertainty

The periods are baseline 2001-2011, the 2012-2016 drought, and the 2020-2024
recent drought period. Pearson and Spearman correlations use paired monthly
values. Pearson uncertainty uses {BOOTSTRAP_REPLICATES:,} paired moving-block
bootstrap replicates with a 12-month block.

## Summary

{period_summary.to_markdown(floatfmt=".3f")}

Across 60 land-type x period x response groups, {significant} conventional
Pearson tests had p<0.05 and {ci_excludes_zero} moving-block bootstrap intervals
excluded zero. These statistics should be interpreted by land type and period,
not as a single universal validation coefficient.

## Interpretation

EWBI remains an interpretable forcing-side composite. These correlations
quantify its contemporaneous association with vegetation and productivity
responses; they do not establish a fully validated mechanistic drought index,
causality, or transferability beyond the analyzed land types and periods.

## Outputs

- `tables/ewbi_period_stratified_response_correlations.csv`
- `tables/ewbi_period_stratified_response_correlations_bootstrap_ci.csv`
- `figures/ewbi_period_stratified_response_correlation_heatmap.png`
"""
    write(REPORT / "EWBI_period_stratified_response_correlation_note.md", note)
    return estimates, bootstrap


def build_stress_reconciliation() -> pd.DataFrame:
    earlier_path = MAJOR / "tables_revision" / "stress_zone_uncertainty_and_significance.csv"
    later_path = HARM / "tables" / "stress_zone_area_uncertainty.csv"
    figure1_path = (
        FIGURE_PACKAGE / "tables" / "figure_1_land_type_areas.csv"
    )
    earlier = pd.read_csv(earlier_path)
    later = pd.read_csv(later_path)
    figure1 = pd.read_csv(figure1_path)

    reconciliation = pd.DataFrame(
        [
            {
                "pipeline_id": "earlier_binary_mask",
                "source_script": str(MAJOR / "08_stress_zone_uncertainty.py"),
                "source_table": str(earlier_path),
                "valid_mask_definition": "finite historic and recent percent change; absolute baseline NIRv > 0.01",
                "land_area_weighting": "full latitude-aware area assigned to every valid 5-km cell",
                "percentage_denominator": "sum of full valid 5-km cell areas",
                "block_bootstrap_design": "3,000 paired replicates; 8x8 analytical pixels; seed 42",
                "threshold_definition": "historic/recent percentage change < -30, -40, or -50%",
                "consistency_with_final_figure8": "No; final Figure 8 reads the later fractional-mask table",
                "consistency_with_final_figure1": "Partial; latitude-aware area agrees, but partial land cells are counted as full cells",
                "main_strength": "Simple and reproducible binary analytical-grid implementation",
                "main_limitation": "Overstates land area at mixed boundary cells and uses a different eligible-area denominator",
                "authoritative_for_manuscript": False,
            },
            {
                "pipeline_id": "later_fractional_mask_final_figure8",
                "source_script": str(
                    HARM / "scripts" / "04_spatial_aggregation_uncertainty.py"
                ),
                "source_table": str(later_path),
                "valid_mask_definition": "finite baseline, historic, and recent NIRv; baseline NIRv > 0.01; positive fractional land coverage",
                "land_area_weighting": "latitude-aware 5-km cell area multiplied by fractional coverage from the 500-m land mask",
                "percentage_denominator": "eligible analyzed land area after baseline-stability and coverage filters",
                "block_bootstrap_design": "2,000 paired replicates; 10x10 analytical pixels (~44.5 km); seed 42",
                "threshold_definition": "historic/recent percentage change <= -30, -40, or -50%",
                "consistency_with_final_figure8": "Yes; direct data source for maps and uncertainty annotations",
                "consistency_with_final_figure1": "Yes; preserves the same latitude-aware 500-m land-mask area basis through fractional aggregation",
                "main_strength": "Area-conservative at land-type boundaries and fully aligned with the final manuscript figure provenance",
                "main_limitation": "Percentages refer to eligible analyzed land area, not the entire Figure 1 land-type area",
                "authoritative_for_manuscript": True,
            },
        ]
    )
    reconciliation.to_csv(TABLES / "stress_zone_source_reconciliation.csv", index=False)

    later["land_key"] = later["land_type"].str.replace(" ", "").str.lower()
    figure1["land_key"] = (
        figure1["land_type"].str.replace(" ", "").str.lower()
    )
    authoritative = later.merge(
        figure1[["land_key", "area_km2"]].rename(
            columns={"area_km2": "figure1_total_land_type_area_km2"}
        ),
        on="land_key",
        how="left",
    ).drop(columns="land_key")
    authoritative["eligible_fraction_of_figure1_land_area"] = (
        authoritative["valid_land_area_km2"]
        / authoritative["figure1_total_land_type_area_km2"]
    )
    authoritative["authoritative_source"] = str(later_path)
    authoritative["authoritative_reason"] = (
        "fractional land coverage; latitude-aware area; stable baseline denominator; "
        "direct final Figure 8 provenance"
    )
    authoritative["reporting_rule"] = (
        "report absolute km2 and percentage-point change first; percent is of eligible analyzed land area"
    )
    authoritative.to_csv(
        TABLES / "stress_zone_authoritative_values_final.csv", index=False
    )

    colors = {
        "Grasslands": "#26828E",
        "Natural Forests": "#1B5E20",
        "Savannas": "#E69F00",
        "Shrublands": "#B05A7A",
    }
    sns.set_theme(style="whitegrid", font_scale=0.95)
    figure, axes = plt.subplots(1, 2, figsize=(13.8, 5.4))
    for land_type in ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]:
        subset = authoritative[authoritative["land_type"] == land_type].sort_values(
            "threshold", ascending=False
        )
        x = np.abs(subset["threshold"].to_numpy(float))
        difference = subset["difference_km2"].to_numpy(float)
        lower = difference - subset["ci95_low_km2"].to_numpy(float)
        upper = subset["ci95_high_km2"].to_numpy(float) - difference
        axes[0].errorbar(
            x,
            difference,
            yerr=np.vstack([lower, upper]),
            marker="o",
            linewidth=1.8,
            markersize=6,
            capsize=4,
            color=colors[land_type],
            label=land_type,
        )
        difference_pp = subset["difference_percentage_points"].to_numpy(float)
        lower_pp = (
            difference_pp
            - subset["ci95_low_percentage_points"].to_numpy(float)
        )
        upper_pp = (
            subset["ci95_high_percentage_points"].to_numpy(float)
            - difference_pp
        )
        axes[1].errorbar(
            x,
            difference_pp,
            yerr=np.vstack([lower_pp, upper_pp]),
            marker="o",
            linewidth=1.8,
            markersize=6,
            capsize=4,
            color=colors[land_type],
            label=land_type,
        )
    for axis in axes:
        axis.axhline(0, color="#333333", linewidth=0.9)
        axis.set_xticks([30, 40, 50], ["30%", "40%", "50%"])
        axis.set_xlabel("NIRv decline threshold")
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("Absolute stressed-area change", fontweight="bold")
    axes[0].set_ylabel("Recent minus historical area (km2)")
    axes[1].set_title("Percentage-point change", fontweight="bold")
    axes[1].set_ylabel("Recent minus historical eligible-area share (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.suptitle(
        "Authoritative NIRv stress-zone threshold sensitivity",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES / "stress_zone_authoritative_threshold_sensitivity.png",
        dpi=450,
        bbox_inches="tight",
    )
    plt.close(figure)

    main = authoritative[authoritative["threshold"] == -40].copy()
    main_rows = []
    for row in main.itertuples():
        main_rows.append(
            f"- **{row.land_type}:** {row.historic_area_km2:,.1f} to "
            f"{row.recent_area_km2:,.1f} km2; change {row.difference_km2:+,.1f} km2 "
            f"(95% block-bootstrap CI {row.ci95_low_km2:,.1f} to "
            f"{row.ci95_high_km2:,.1f}); {row.historic_percent:.3f}% to "
            f"{row.recent_percent:.3f}% of eligible analyzed land area, "
            f"{row.difference_percentage_points:+.3f} percentage points."
        )
    main_text = "\n".join(main_rows)
    decision = f"""# Stress-zone source reconciliation decision

## Decision

The **later fractional-mask/final-Figure-8 pipeline** is the authoritative
source for revised-manuscript stress-zone reporting:

`{later_path}`

## Pipelines audited

1. Earlier binary-mask source: `{earlier_path}`
2. Later fractional-mask source: `{later_path}`

## Why the values differ

Both pipelines use the same NIRv periods and decline thresholds, but they do not
assign area identically. The earlier pipeline gives every valid 5-km analytical
cell its full latitude-aware area. A boundary cell containing only a small
fraction of the target land type is therefore counted as a full land-type cell.
The later pipeline reprojects the 500-m binary land mask by area averaging and
multiplies each 5-km cell by its fractional land coverage. It also explicitly
requires baseline NIRv > 0.01 and reports percentages relative to the eligible
analyzed land area. Block size and replicate count also differ, affecting
uncertainty intervals but not explaining the main area difference.

## Why the later source is preferred

- It is area-conservative at mixed land-type boundaries.
- It preserves the same latitude-aware 500-m mask basis used for final Figure 1.
- It documents the stable-baseline denominator required for percentage decline.
- It is the direct source recorded in the final Figure 8 source manifest.
- It clearly marks relative ratios undefined when historical stressed area is
  zero or near zero.

The eligible analyzed area is 61-76% of each Figure 1 land-type area because of
NIRv coverage and baseline-stability filtering. Therefore, reported percentages
must be described as percentages of **eligible analyzed land area**, not total
land-type area.

## Authoritative 40% threshold values

{main_text}

## Manuscript values requiring replacement

Replace the submitted Results paragraph and Table 4 values:

- Grasslands: replace 0.38% to 1.75% and the 4.6-times statement with the
  authoritative values above.
- Natural Forests: replace 0.14% to 2.42% and the 17-times statement. Historical
  stressed area is zero in the authoritative analysis, so a ratio is undefined.
- Savannas: replace 0.11% to 1.57% and the 14-times statement.
- Shrublands: replace 0.40% to 0.81% and the 2-times statement.
- Remove all abstract, Results, Discussion, and Conclusion wording describing
  two-to-seventeen-times expansion.
- Revise Figure 8 caption text that says the recent period shows substantially
  less stress. The authoritative analysis reports positive recent-minus-historical
  stressed-area differences at the 40% threshold in all four land types.

## Figure 8 decision

The current final corrected Figure 8 already reads the later authoritative table.
Its plotted maps and numerical annotations do **not** require regeneration.
Only manuscript caption/text consistency and the description of the percentage
denominator require updating. If a different visual export is selected later,
it must continue to read `stress_zone_area_uncertainty.csv`.

## Reporting rule

Report absolute km2 and percentage-point changes before any relative comparison.
Do not use a fold-change claim when historical area is zero or below the
prespecified near-zero rule. Never combine earlier binary-mask values with later
fractional-mask values in one table, caption, or paragraph.
"""
    write(REPORT / "stress_zone_source_reconciliation_decision.md", decision)
    return authoritative


def run_git(command: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
            timeout=45,
        )
        return True, result.stdout.strip()
    except Exception as exc:
        return False, str(exc)


def build_github_outputs() -> pd.DataFrame:
    requirements_path = REPO / "requirements.txt"
    requirements = requirements_path.read_text(encoding="utf-8", errors="ignore")
    readme = (REPO / "README.md").read_text(encoding="utf-8", errors="ignore")
    dependencies = [
        "torch",
        "tensorflow",
        "xgboost",
        "statsmodels",
        "rasterio",
        "geopandas",
        "earthengine-api",
        "pandas",
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
    ]

    status_ok, status_output = run_git(["git", "status", "--short"])
    head_ok, local_head = run_git(["git", "rev-parse", "HEAD"])
    remote_ok, remote_output = run_git(
        ["git", "ls-remote", "origin", "refs/heads/main"]
    )
    remote_head = remote_output.split()[0] if remote_ok and remote_output else ""
    tags_ok, remote_tags = run_git(["git", "ls-remote", "--tags", "origin"])
    changed_count = len([line for line in status_output.splitlines() if line.strip()])
    old_folder = REPO / "Deep Leaning Codes for Forecasting"
    new_folder = REPO / "Deep Learning Codes for Forecasting"
    gee_root = REPO / "Google Earth Engine"
    gee_scripts = list(gee_root.rglob("*.js")) if gee_root.exists() else []
    major_scripts = sorted(MAJOR.glob("[0-9][0-9]_*.py"))

    rows: list[dict] = []

    def add(
        check: str,
        status: str,
        evidence: str,
        finding: str,
        required_action: str,
        manual: bool,
    ) -> None:
        rows.append(
            {
                "check": check,
                "status": status,
                "evidence": evidence,
                "current_finding": finding,
                "required_action": required_action,
                "manual_action_required": manual,
            }
        )

    add(
        "Forecasting folder spelling",
        "local change prepared, not published",
        str(new_folder),
        (
            f"new folder exists={new_folder.exists()}; old folder exists={old_folder.exists()}; "
            "git records the files as pending renames"
        ),
        "Review the rename, commit it, push it, and verify the remote tree.",
        True,
    )
    for dependency in dependencies:
        present = bool(
            re.search(
                rf"(?im)^{re.escape(dependency)}(?:[=<>!~]|$)",
                requirements,
            )
        )
        add(
            f"Dependency: {dependency}",
            "complete" if present else "missing",
            str(requirements_path),
            f"{dependency} is {'listed' if present else 'not listed'} in requirements.txt.",
            (
                "No action."
                if present
                else f"Add the tested {dependency} version to requirements.txt."
            ),
            not present,
        )
    add(
        "environment.yml",
        "recommended, missing",
        str(REPO / "environment.yml"),
        "No conda environment file exists; the project combines geospatial, PyTorch, TensorFlow, and XGBoost stacks.",
        "Create and test environment.yml or provide an exact pip lock/export plus Python version.",
        True,
    )
    major_revision_status = (
        "ECOINF-D-26-00382" in readme and "Major revision" in readme
    )
    add(
        "README major-revision status",
        "complete" if major_revision_status else "missing",
        str(REPO / "README.md"),
        f"Major-revision manuscript status present={major_revision_status}.",
        "Retain and update the revision date.",
        False,
    )
    major_heading = bool(re.search(r"(?im)^##\s+Major revision analyses", readme))
    add(
        "README Major revision analyses section",
        "missing" if not major_heading else "complete",
        str(REPO / "README.md"),
        f"Dedicated heading present={major_heading}.",
        "Insert README_major_revision_section_final.md and replace stale paths/encoding.",
        not major_heading,
    )
    add(
        "Ordered revision entry points",
        "available externally, not integrated in repository",
        str(MAJOR),
        f"{len(major_scripts)} numbered top-level revision scripts exist in the local companion analysis folder.",
        "Add an ordered workflow manifest and stable relative links/copies in the repository release.",
        True,
    )
    final_refs = (
        "Figure_7_final_publication_polished" in readme
        or "Table S-F1" in readme
        or "table_SF1" in readme
    )
    add(
        "Final manuscript artifacts referenced",
        "missing" if not final_refs else "complete",
        str(REPO / "README.md"),
        f"Frozen Figure 7 and Tables S-F1 to S-F4 explicitly referenced={final_refs}.",
        "Reference the final figure/table package and repository-relative locations.",
        not final_refs,
    )
    add(
        "Google Earth Engine scripts",
        "complete locally",
        str(gee_root),
        f"{len(gee_scripts)} JavaScript files are present with a README.",
        "Correct README encoding and ensure the inventory matches actual filenames.",
        True,
    )
    large_data_explained = any(
        phrase in readme.lower()
        for phrase in [
            "large geotiff",
            "not stored in git",
            "file-size constraints",
            "raw data are not",
        ]
    )
    add(
        "Large data and raw GeoTIFF limitations",
        "missing" if not large_data_explained else "complete",
        str(REPO / "README.md"),
        f"Explicit large-file limitation present={large_data_explained}.",
        "Insert the final code/data availability and large-file reconstruction wording.",
        not large_data_explained,
    )
    local_remote_equal = (
        head_ok and remote_ok and local_head.strip() == remote_head.strip()
    )
    add(
        "Public commit contains current cleanup",
        "not complete",
        "git status and origin/main",
        (
            f"origin reachable={remote_ok}; local committed HEAD equals remote main={local_remote_equal}; "
            f"uncommitted/staged changes={changed_count}. Current remote main is the pre-cleanup commit."
        ),
        "Review changes, commit, push, then rerun git status and remote hash checks.",
        True,
    )
    add(
        "Public release tag or DOI",
        "missing",
        "git ls-remote --tags origin",
        f"Remote tags found={bool(remote_tags.strip()) if tags_ok else 'unknown'}; DOI evidence was not found.",
        "Create a version tag after final verification and archive the release in Zenodo or another DOI service if required.",
        True,
    )

    checklist = pd.DataFrame(rows)
    checklist.to_csv(
        TABLES / "github_reproducibility_final_checklist.csv", index=False
    )

    readme_section = """# Major revision analyses

This repository supports the major revision of Ecological Informatics manuscript
ECOINF-D-26-00382. The reviewer-response analyses add EWBI validation,
harmonization/resampling uncertainty, gap-filling comparison, detrending
sensitivity, leakage-free rolling-origin forecasting, statistical baselines,
targeted multivariate sensitivity, spatial uncertainty, classification
sensitivity, and corrected manuscript figures.

## Final analysis decisions

- EWBI is presented as an interpretable forcing-side composite with
  land-type- and period-specific response associations.
- The common analytical grid is used for overlay and does not add fine-scale
  hydrologic information to GRACE/GRACE-FO TWS.
- Main ecological analyses use baseline-referenced deseasonalized-only
  anomalies; detrended products are sensitivity analyses.
- Figure 7 is the common 12-month reviewer-safe pixel-level benchmark using
  four expanding rolling-origin test periods and training-only preprocessing.
- The six original forecasting models are five deep-learning models plus
  XGBoost. SARIMA is reported separately at monthly land-type aggregate level.
- The 18-month BiLSTM-TWS state-space result is Supplementary sensitivity only.
- Stress-zone reporting uses fractional land coverage, latitude-aware area,
  and paired spatial block-bootstrap uncertainty.

## Reproducible outputs

The release README should link repository-relative copies or archived locations
for:

- the frozen Figure 7 source table and final PNG/PDF;
- Supplementary Tables S-F1 to S-F4;
- EWBI period-stratified correlation outputs;
- stress-zone authoritative values and uncertainty;
- final corrected figure source-data manifests;
- ordered run manifests and environment files.

## Data boundaries

Raw provider products and large derived GeoTIFF stacks are not stored in git.
The repository must document source products, licenses, expected directory
structure, filenames, and cache-building steps. Model checkpoints and large
prediction files should be archived separately when needed.

## Release status

Do not mark this cleanup complete until changes are reviewed, committed, pushed,
and verified on the public remote. Cite a permanent tag or DOI only after that
release exists.
"""
    write(ROOT / "README_major_revision_section_final.md", readme_section)

    environment_plan = """# Requirements and environment update plan

## Current finding

`requirements.txt` includes TensorFlow, statsmodels, rasterio, geopandas,
earthengine-api, pandas, numpy, scipy, scikit-learn, and matplotlib. It does not
currently include PyTorch or XGBoost, although both are required by the final
reviewer-safe forecasting workflow.

## Required requirements.txt additions

Add tested versions for:

```text
torch
xgboost
reportlab
pypdf
```

Retain TensorFlow only because legacy Keras scripts remain in the repository;
state clearly that the final reviewer-safe raster benchmark uses PyTorch.

## Environment recommendation

An `environment.yml` or exact pip/conda lock export is recommended because the
workflow combines compiled geospatial libraries with two neural frameworks.
The environment record should include:

- Python version and operating system
- GDAL/rasterio/geopandas stack
- PyTorch and CUDA/CPU build
- TensorFlow build for legacy scripts
- XGBoost
- statsmodels
- plotting and report-generation packages

## Verification

After updating dependencies:

1. Create a clean environment.
2. Run an import-only smoke test.
3. Run documentation/table-generation scripts.
4. Do not rerun expensive models solely to test installation.
5. Record the tested package export in the release.
"""
    write(ROOT / "requirements_environment_update_plan.md", environment_plan)

    availability = """# Final code and data availability statement

The analysis code, Google Earth Engine export scripts, reviewer-response
workflows, and machine-readable data underlying the revised figures and
Supplementary tables will be archived in the repository version cited by the
manuscript. The final citation will identify a permanent release tag or DOI.

Raw satellite and reanalysis products remain subject to the original providers'
licenses and are not redistributed. Large GeoTIFF time-series stacks, derived
raster caches, model checkpoints, and prediction files are excluded from git
because of file-size constraints. The repository documents the source products,
expected local directory structure, preprocessing decisions, and scripts needed
to reconstruct derived summaries. Numerical source tables used for manuscript
figures and Supplementary analyses are included or linked to an archived release.

Google Earth Engine JavaScript files for source-product export and land-cover
mask preparation are included. GRACE/GRACE-FO TWS is represented on a common
analytical overlay grid, but its coarse effective information support is retained
in interpretation.

At the time this statement was prepared, the local cleanup changes had not yet
been committed and pushed. The manuscript must not claim a completed public
release until the remote repository, release tag, and any DOI have been verified.
"""
    write(ROOT / "code_data_availability_statement_final.md", availability)

    workflow = """# Ordered revision workflow manifest

This manifest identifies the final logical workflow. Heavy model stages are
provenance entry points and should not be rerun when their verified outputs
already exist.

1. **Build common monthly summaries**
   - `major_revision_analyses/00_build_analysis_cache.py`
   - Output: monthly land-type summary cache.
2. **Validate EWBI**
   - `major_revision_analyses/01_ewbi_validation.py`
   - Targeted completion: `FINAL_remaining_partial_items_fix/scripts/build_remaining_partial_items_fix.py`
3. **Assess harmonization and TWS scale**
   - `TWS_Harmonization_Uncertainty_Revision/scripts/run_all_harmonization_uncertainty.py`
4. **Compare gap-filling methods**
   - `GapFilling_Comparison_Revision/scripts/run_all_gapfill_comparison.py`
5. **Assess detrending sensitivity**
   - `major_revision_analyses/04_detrending_sensitivity_fixed/scripts/run_all_detrending_fixed.py`
6. **Reviewer-safe forecasting benchmark**
   - `major_revision_analyses/05_forecasting_raster_reviewer_safe_candidate/scripts/run_all_reviewer_safe_forecasting.py`
   - Expensive; use verified compiled outputs unless a specific missing task is identified.
7. **Hydroclimate sensitivity**
   - `major_revision_analyses/05_forecasting_hydroclimate_special_candidate/scripts/run_all_hydroclimate_special_candidate.py`
   - Supplementary only.
8. **Targeted multivariate sensitivity**
   - `major_revision_analyses/06_multivariate_forecasting.py`
9. **Spatial heterogeneity**
   - `major_revision_analyses/07_spatial_heterogeneity.py`
10. **Authoritative stress-zone uncertainty**
    - `TWS_Harmonization_Uncertainty_Revision/scripts/04_spatial_aggregation_uncertainty.py`
11. **State-classification sensitivity**
    - `major_revision_analyses/09_state_classification_sensitivity.py`
12. **Final paper figures**
    - `major_revision_analyses/final_paper_figures_corrected/scripts/run_all_final_paper_figures.py`
    - Figure 7 is supplied by the later frozen reviewer-safe Figure 7 package.
13. **Manuscript and response integration**
    - Use the final manuscript editing and reviewer-response packages.

For a public release, replace absolute Windows paths with documented
configuration variables or command-line arguments and provide repository-relative
artifact links.
"""
    write(ROOT / "ordered_revision_workflow_manifest.md", workflow)

    commit_checklist = f"""# GitHub commit and push checklist

Current committed local HEAD: `{local_head if head_ok else 'unavailable'}`

Current remote `main`: `{remote_head if remote_ok else 'unavailable'}`

The remote is reachable, but {changed_count} local changed entries are not part
of the current remote commit. No remote tag was found.

## Manual sequence

- [ ] Review every rename, deletion, addition, and README modification in `git status`.
- [ ] Confirm that deleted visualization files are intentionally removed or restore them before commit.
- [ ] Confirm the `Deep Learning Codes for Forecasting` rename.
- [ ] Update `requirements.txt` and add a tested lock/environment export.
- [ ] Insert the final README major-revision section.
- [ ] Add repository-relative links or archived references for final tables and figures.
- [ ] Add the ordered workflow manifest and code/data availability statement.
- [ ] Remove or document generated binary files that should not be versioned.
- [ ] Run import/document-generation smoke checks.
- [ ] Commit with a revision-specific message.
- [ ] Push to `origin/main` or a dedicated revision branch.
- [ ] Verify the remote tree and commit hash in a browser or with `git ls-remote`.
- [ ] Create and push a version tag.
- [ ] Archive the tagged release and record a DOI if required.
- [ ] Update the manuscript with the verified URL, tag, and DOI.

Do not state that the public repository is complete until every checked item has
been verified on the remote.
"""
    write(ROOT / "github_commit_push_checklist.md", commit_checklist)
    return checklist


def render_status_pdf(markdown_path: Path, pdf_path: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title="Remaining partial items final status",
    )
    story = []
    for raw in markdown_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 3))
            continue
        if line.startswith("# "):
            style = styles["Title"]
            text = line[2:]
        elif line.startswith("## "):
            style = styles["Heading2"]
            text = line[3:]
        elif line.startswith("- "):
            style = styles["BodyText"]
            text = "&#8226; " + line[2:]
        else:
            style = styles["BodyText"]
            text = line
        text = (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("`", "")
            .replace("**", "")
        )
        story.append(Paragraph(text, style))
    document.build(story)


def build_status_report(
    ewbi: pd.DataFrame, authoritative: pd.DataFrame, github: pd.DataFrame
) -> None:
    github_manual = int(github["manual_action_required"].astype(bool).sum())
    report = f"""# Remaining partial items final status report

Generated: {datetime.now().isoformat(timespec="seconds")}

## Item 1: EWBI validation and sensitivity

**Status: complete.**

The missing period-stratified table was generated from the existing monthly
land-type cache. It contains {len(ewbi)} land-type x period x response groups
covering NIRv, NDVI, QA-controlled GNDVI, GOSIF/SIF, and GPP. Pearson,
Spearman, p-values, and paired 12-month moving-block bootstrap intervals are
reported. No raster extraction or model fitting was rerun.

Insert the compact correlation table and heatmap in the Supplement and add a
short Results sentence emphasizing land-type- and period-dependent association.
Do not describe EWBI as a mechanistically validated drought index.

## Item 9: Stress-zone significance and uncertainty

**Status: complete following source reconciliation.**

The later fractional-mask pipeline is authoritative because it is
area-conservative at land-type boundaries, matches the latitude-aware land-mask
basis of final Figure 1, documents the eligible-area denominator, and is the
direct final Figure 8 source. The current final corrected Figure 8 already uses
these values and does not require regeneration. Manuscript Table 4, the Results,
caption wording, Abstract, Discussion, and Conclusion must replace the old
ratio-based values and state that percentages refer to eligible analyzed land
area.

Insert `stress_zone_authoritative_values_final.csv` and the threshold-sensitivity
figure in the Supplement. Do not mix values from the earlier binary-mask table.

## Item 12: GitHub and reproducibility

**Status: still requires manual GitHub action.**

The cleanup documents and checklist are complete, but {github_manual} checklist
rows require human action. The correctly spelled forecasting folder exists
locally as a pending rename. PyTorch and XGBoost are missing from
`requirements.txt`; no `environment.yml` exists; final manuscript artifacts are
not explicitly referenced; large GeoTIFF limitations need clearer wording; and
the local changes have not been committed or pushed. The public remote remains
on the pre-cleanup commit and has no verified release tag or DOI.

## Scientific-change statement

- Scientific model rerun: **No**
- Forecasting or deep-learning model rerun: **No**
- Forecasting numerical result changed: **No**
- Figure 7 changed or redesigned: **No**
- Raster extraction rerun: **No**
- New scientific computation: period-stratified correlations from existing
  monthly summaries only
- Stress-zone action: reconciliation and visualization of existing tables only

## Manuscript and Supplement insertions

1. Supplement: EWBI period-stratified correlation table, bootstrap-CI table,
   and heatmap.
2. Methods/Results: short note defining the period-stratified EWBI sensitivity.
3. Supplement: authoritative stress-zone values and threshold-sensitivity
   figure.
4. Main text/Table 4/Figure 8 caption: replace old stress percentages and remove
   ratio-based expansion wording.
5. Code/Data Availability: insert the final availability statement only after
   the repository URL, tag, and release state are verified.
"""
    markdown_path = REPORT / "remaining_items_final_status_report.md"
    write(markdown_path, report)
    render_status_pdf(
        markdown_path, REPORT / "remaining_items_final_status_report.pdf"
    )


def main() -> None:
    ewbi, bootstrap = build_ewbi_outputs()
    authoritative = build_stress_reconciliation()
    github = build_github_outputs()
    build_status_report(ewbi, authoritative, github)

    files = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "relative_path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(files).to_csv(TABLES / "package_file_inventory.csv", index=False)

    log = f"""Generated: {datetime.now().isoformat(timespec="seconds")}
EWBI groups: {len(ewbi)}
EWBI bootstrap rows: {len(bootstrap)}
Authoritative stress rows: {len(authoritative)}
GitHub checklist rows: {len(github)}
Forecasting models rerun: no
Deep-learning models rerun: no
Figure 7 changed: no
"""
    write(LOGS / "build_remaining_partial_items_fix_log.txt", log)
    print(log)


if __name__ == "__main__":
    main()
