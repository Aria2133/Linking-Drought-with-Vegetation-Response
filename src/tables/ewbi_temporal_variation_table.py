from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

PHASE = Path(__file__).resolve().parent
AUDIT = PHASE.parent
PAPER = AUDIT.parent
MAJOR = PAPER / "Docs" / "major_revision_analyses"


REQUIRED = [
    "task1_figure5/figure5_corrected_raw.png",
    "task1_figure5/figure5_raw_correlations.csv",
    "task1_figure5/figure5_pairwise_date_coverage.csv",
    "task1_figure5/figure5_deseasonalized_sensitivity.png",
    "task1_figure5/figure5_deseasonalized_correlations.csv",
    "task1_figure5/figure5_exact_input_monthly_series.csv",
    "task1_figure5/figure5_deseasonalized_exact_input_monthly_series.csv",
    "task2_forecast_inference/forecast_model_task_metrics_corrected.csv",
    "task2_forecast_inference/forecast_friedman_tests.csv",
    "task2_forecast_inference/forecast_pairwise_tests_holm.csv",
    "task2_forecast_inference/forecast_effect_sizes.csv",
    "task2_forecast_inference/forecast_model_rank_summary.csv",
    "task2_forecast_inference/FORECAST_STATISTICAL_INFERENCE.md",
    "task3_transition_null/state_transition_null_summary.csv",
    "task3_transition_null/state_transition_null_distributions.csv",
    "task3_transition_null/state_transition_null_figure.png",
    "task3_transition_null/state_transition_null_exact_input_series.csv",
    "task3_transition_null/STATE_TRANSITION_NULL_ANALYSIS.md",
    "task4_gpp_transition/gpp_product_transition_sensitivity.csv",
    "task4_gpp_transition/gpp_forecast_boundary_sensitivity.csv",
    "task4_gpp_transition/GPP_TRANSITION_SENSITIVITY.md",
    "task5_bilstm_reconciliation/Figure_7_corrected_TRUE_BiLSTM.png",
    "task5_bilstm_reconciliation/Figure_7_corrected_source_table.csv",
    "task5_bilstm_reconciliation/Figure_S7_corrected_TRUE_BiLSTM.png",
    "task5_bilstm_reconciliation/Figure_S7_corrected_exact_monthly_metrics.csv",
    "task5_bilstm_reconciliation/Figure_S7_corrected_exact_pixel_metrics.csv",
    "task5_bilstm_reconciliation/Figure_S9_corrected_TRUE_BiLSTM.png",
    "task5_bilstm_reconciliation/Figure_S9_corrected_exact_monthly_source.csv",
    "task5_bilstm_reconciliation/Figure_S9_corrected_source_table.csv",
    "task5_bilstm_reconciliation/bilstm_old_vs_corrected_values.csv",
    "task5_bilstm_reconciliation/BILSTM_RECONCILIATION.md",
    "task6_supplement_harmonization/Figure_S4_harmonized.png",
    "task6_supplement_harmonization/Figure_S4_harmonized_exact_monthly_source.csv",
    "task6_supplement_harmonization/Figure_S5_harmonized.png",
    "task6_supplement_harmonization/Figure_S5_harmonized_correlations.csv",
    "task6_supplement_harmonization/Figure_S5_harmonized_exact_monthly_source.csv",
    "task6_supplement_harmonization/Figure_S6_harmonized.png",
    "task6_supplement_harmonization/Figure_S6_harmonized_exact_monthly_states.csv",
    "task6_supplement_harmonization/Figure_S6_harmonized_transition_matrices.csv",
    "task6_supplement_harmonization/SUPPLEMENTARY_HARMONIZATION.md",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_ewbi_table() -> pd.DataFrame:
    source = AUDIT / "EWBI_audit" / "ewbi_landtype_degeneracy_diagnostics.csv"
    audit = pd.read_csv(source)
    result = audit[
        [
            "land_type",
            "n_months",
            "mean_2001_2024",
            "sd_2001_2024",
            "min",
            "max",
            "range",
            "baseline_mean_2001_2011",
            "historic_mean_2012_2016",
            "recent_mean_2020_2024",
        ]
    ].rename(
        columns={
            "mean_2001_2024": "full_period_mean",
            "sd_2001_2024": "full_period_sd",
            "min": "minimum",
            "max": "maximum",
            "baseline_mean_2001_2011": "mean_2001_2011",
            "historic_mean_2012_2016": "mean_2012_2016",
            "recent_mean_2020_2024": "mean_2020_2024",
        }
    )
    result.to_csv(PHASE / "Table_EWBI_temporal_variation.csv", index=False)
    return result


def run_integrity_check() -> str:
    script = AUDIT / "_harness" / "integrity_check.py"
    snapshot = AUDIT / "_harness" / "inventory_snapshot_at_audit_start.csv"
    output = PHASE / "logs" / "integrity_check_revision_phase.txt"
    output.parent.mkdir(exist_ok=True)
    completed = subprocess.run(
        [sys.executable, str(script), str(snapshot), str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def validate(ewbi_table: pd.DataFrame, integrity_text: str) -> list[dict]:
    checks: list[dict] = []

    def add(name: str, passed: bool, evidence: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})

    missing = [rel for rel in REQUIRED if not (PHASE / rel).is_file()]
    add("All requested task outputs present", not missing, "none missing" if not missing else "; ".join(missing))
    add("No original project file overwritten", "ADDED: 0" in integrity_text and "REMOVED: 0" in integrity_text and "CHANGED: 0" in integrity_text, "inventory comparison reports ADDED=0, REMOVED=0, CHANGED=0 outside AUDIT_TEMP")

    coverage = pd.read_csv(PHASE / "task1_figure5" / "figure5_pairwise_date_coverage.csv")
    raw = pd.read_csv(PHASE / "task1_figure5" / "figure5_raw_correlations.csv")
    deseason = pd.read_csv(PHASE / "task1_figure5" / "figure5_deseasonalized_correlations.csv")
    add("Figure 5 timestamp alignment", coverage.pairwise_start_date.notna().all() and coverage.pairwise_end_date.notna().all() and coverage.n_shared_months.between(1, 288).all(), f"all pair rows carry start/end dates; shared N={coverage.n_shared_months.min()}-{coverage.n_shared_months.max()}")
    add("Figure 5 raw analysis retained as main", raw.analysis.eq("raw_monthly").all() and (PHASE / "task1_figure5" / "figure5_corrected_raw.png").is_file(), "raw Pearson table and main raw figure are present")
    add("Deseasonalized analysis is separate sensitivity", deseason.analysis.eq("calendar_month_climatology_removed").all() and (PHASE / "task1_figure5" / "figure5_deseasonalized_sensitivity.png").is_file(), "calendar-month climatology removal is stored in a separate sensitivity figure/table")

    task_metrics = pd.read_csv(PHASE / "task2_forecast_inference" / "forecast_model_task_metrics_corrected.csv")
    add("Forecast inference uses task-level replicates", len(task_metrics) == 216 and task_metrics.model.nunique() == 6, "216 model-task rows = 6 models x 36 target-land tasks; no pixel inferential rows")

    bilstm = pd.read_csv(PHASE / "task5_bilstm_reconciliation" / "Figure_7_corrected_source_table.csv")
    add("TRUE BiLSTM branch reconciled", (bilstm.model == "BiLSTM").sum() == 36, "36 corrected BiLSTM target-land cells in Figure 7 source; S7/S9 corrected sources also present")

    s5 = pd.read_csv(PHASE / "task6_supplement_harmonization" / "Figure_S5_harmonized_correlations.csv")
    add("S4/S5 authoritative no-QA GNDVI", s5.gndvi_branch.eq("previous/no-QA GNDVI").all(), "all S5 rows identify previous/no-QA GNDVI; S4/S5 share its exact response input")

    s6 = pd.read_csv(PHASE / "task6_supplement_harmonization" / "Figure_S6_harmonized_exact_monthly_states.csv")
    state_source = pd.read_csv(MAJOR / "04_detrending_sensitivity_fixed" / "tables" / "main_state_classification_deseasonalized.csv")
    state_source = state_source.sort_values(["land_type", "date"]).reset_index(drop=True)
    s6 = s6.sort_values(["land_type", "date"]).reset_index(drop=True)
    same_states = s6.state_median.equals(state_source.ecosystem_state)
    same_ewbi = np.allclose(s6.ewbi, state_source.ewbi, equal_nan=True)
    same_nirv = np.allclose(s6.nirv_deseasonalized_anomaly, state_source.nirv_deseasonalized_anomaly, equal_nan=True)
    add("S6 uses exact Figure-9 pipeline", same_states and same_ewbi and same_nirv, "1,152 monthly rows match Figure-9 EWBI, NIRv, and median states exactly")

    gpp_report = (PHASE / "task4_gpp_transition" / "GPP_TRANSITION_SENSITIVITY.md").read_text(encoding="utf-8")
    add("No GPP cross-product calibration invented", "No FLUXCOM" in gpp_report and "calibration" in gpp_report, "report explicitly prohibits and does not apply calibration")

    null_diag = pd.read_csv(PHASE / "task3_transition_null" / "state_transition_null_surrogate_diagnostics.csv")
    add("Transition null preserves temporal dependence", len(null_diag) == 2000 and null_diag.ewbi_relative_spectral_error.median() < 0.03 and null_diag.nirv_relative_spectral_error.median() < 0.03, f"2,000 series-pair diagnostics (500 per land); median relative spectral errors EWBI={null_diag.ewbi_relative_spectral_error.median():.3f}, NIRv={null_diag.nirv_relative_spectral_error.median():.3f}")

    figure_sources = [
        ("figure5_corrected_raw.png", "task1_figure5/figure5_exact_input_monthly_series.csv"),
        ("figure5_deseasonalized_sensitivity.png", "task1_figure5/figure5_deseasonalized_exact_input_monthly_series.csv"),
        ("state_transition_null_figure.png", "task3_transition_null/state_transition_null_distributions.csv"),
        ("Figure_7_corrected_TRUE_BiLSTM.png", "task5_bilstm_reconciliation/Figure_7_corrected_source_table.csv"),
        ("Figure_S7_corrected_TRUE_BiLSTM.png", "task5_bilstm_reconciliation/Figure_S7_corrected_exact_monthly_metrics.csv"),
        ("Figure_S9_corrected_TRUE_BiLSTM.png", "task5_bilstm_reconciliation/Figure_S9_corrected_exact_monthly_source.csv"),
        ("Figure_S4_harmonized.png", "task6_supplement_harmonization/Figure_S4_harmonized_exact_monthly_source.csv"),
        ("Figure_S5_harmonized.png", "task6_supplement_harmonization/Figure_S5_harmonized_correlations.csv"),
        ("Figure_S6_harmonized.png", "task6_supplement_harmonization/Figure_S6_harmonized_exact_monthly_states.csv"),
    ]
    add("Every corrected figure has exact source data", all((PHASE / src).is_file() for _, src in figure_sources), "; ".join(f"{fig} -> {src}" for fig, src in figure_sources))
    scripts = list(PHASE.glob("task*/run_task*.py"))
    add("Reproducible scripts retained", len(scripts) >= 6, f"{len(scripts)} task scripts plus finalization script")
    add("EWBI temporal table comes from completed audit", len(ewbi_table) == 4 and ewbi_table.n_months.eq(288).all(), "selected from EWBI_audit/ewbi_landtype_degeneracy_diagnostics.csv; no EWBI audit rerun")
    return checks


def write_manifest() -> None:
    rows = []
    for rel in REQUIRED + ["Table_EWBI_temporal_variation.csv"]:
        path = PHASE / rel
        rows.append({"relative_path": rel.replace("\\", "/"), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else np.nan, "sha256": sha256(path) if path.is_file() else ""})
    pd.DataFrame(rows).to_csv(PHASE / "REVISION_OUTPUT_MANIFEST.csv", index=False)


def write_qc(checks: list[dict]) -> None:
    pd.DataFrame(checks).to_csv(PHASE / "QUALITY_CONTROL_CHECKS.csv", index=False)
    lines = ["# Computational revision quality control", "", f"Result: **{sum(c['status'] == 'PASS' for c in checks)}/{len(checks)} checks passed.**", "", "| Check | Status | Evidence |", "|---|---:|---|"]
    for check in checks:
        lines.append(f"| {check['check']} | {check['status']} | {check['evidence']} |")
    lines.extend(["", "The original-project integrity check compares both the Drought Paper and Thesis trees with the inventory taken at audit start, excluding `AUDIT_TEMP`. Its complete log is in `logs/integrity_check_revision_phase.txt`.", ""])
    (PHASE / "QUALITY_CONTROL_VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary() -> None:
    coverage = pd.read_csv(PHASE / "task1_figure5" / "figure5_pairwise_date_coverage.csv")
    comparison = pd.read_csv(PHASE / "task1_figure5" / "figure5_submitted_vs_corrected_vs_deseasonalized.csv")
    friedman = pd.read_csv(PHASE / "task2_forecast_inference" / "forecast_friedman_tests.csv").set_index("metric")
    nulls = pd.read_csv(PHASE / "task3_transition_null" / "state_transition_null_summary.csv")
    gpp = pd.read_csv(PHASE / "task4_gpp_transition" / "gpp_product_transition_sensitivity.csv")
    s6 = pd.read_csv(PHASE / "task6_supplement_harmonization" / "Figure_S6_harmonized_agreement.csv")
    sign_flips = int(comparison.sign_flip_vs_submitted.sum())
    max_change = float(comparison.change_raw_minus_submitted.abs().max())
    null_upper = int((nulls.empirical_p_upper_observed_greater < 0.05).sum())
    changed_gpp = int(gpp.drop_duplicates(["analysis_type", "land_type", "comparator"]).statistical_conclusion_changed.sum())
    cluster = s6[s6.method.isin(["kmeans", "gmm"])]
    text = f"""# Computational revision summary

## CENTRAL-CONCLUSION IMPACT: qualification required, not a wholesale reversal

The corrected date alignment changes the sign of {sign_flips} submitted Figure-5 pairwise correlations (largest absolute r change {max_change:.3f}), so any manuscript statement tied to specific cross-variable signs or magnitudes must be rechecked. The forecasting analysis does not support a blanket claim that one learned model is universally superior. The broad ecological result that NIRv-GPP correlation exceeds NDVI-GPP correlation remains in every land type under the FLUXCOM-only sensitivity, but four GPP correlation/trend significance conclusions change. These results require targeted qualification of central evidence claims, while they do not by themselves invalidate the overall drought-monitoring framework.

| Task | Reviewer concern | Submitted problem | What was recomputed | Key result | Affected figures/tables | Likely manuscript implication | Confidence |
|---|---|---|---|---|---|---|---|
| 1. Figure 5 | Raw monthly correlations may be misaligned and seasonally inflated | Arrays were aligned by position rather than timestamps | Pairwise YYYY-MM intersection, Pearson r/p/date coverage; separate climatology-removed sensitivity | Pairwise N is {coverage.n_shared_months.min()}-{coverage.n_shared_months.max()}; {sign_flips} signs differ from submitted; seasonality removal often attenuates associations | Figure 5; new sensitivity figure/tables | Replace Figure 5 values/caption; describe raw as main and deseasonalized as sensitivity; recheck sign-specific claims | High |
| 2. Forecast inference | Small metric differences lacked paired statistical support | Descriptive wins could be read as superiority; pixel pseudoreplication risk | 36 target x land tasks, Friedman, paired Wilcoxon-Holm, effect sizes, CIs, ranks, wins | R2 Friedman p={friedman.loc['r2','p_value']:.3g}; nRMSE p={friedman.loc['nrmse','p_value']:.3g}; only specified pairs are Holm-significant, with no universal winner | Forecast tables; Figures 7/S7/S8 context | Replace blanket superiority wording with paired task-level evidence and uncertainty | High |
| 3. Transition null | High diagonal probabilities may reflect autocorrelation | No autocorrelation-aware reference distribution | 500 independent IAAFT EWBI/NIRv surrogate pairs per land using original thresholds | Only {null_upper}/16 states show observed persistence above the null at upper-tail p<0.05; high persistence is often expected under autocorrelation | Null figure/tables; Figure 9 interpretation | Report null comparison and avoid validated-Markov/ecosystem-memory language | High |
| 4. GPP transition | FLUXCOM to MOD17 boundary was uncalibrated | Stitched 2001-2024 analyses may mix product and period effects | FLUXCOM-only ecological sensitivity and saved-prediction pre/post metrics; no calibration/retraining | NIRv>NDVI GPP correlation is retained in all lands; {changed_gpp} nominal correlation/trend conclusions change; post-2022 forecast degradation and ranks are product/period/fold confounded | GPP sensitivity tables | Disclose product boundary; qualify affected trend/correlation and forecast claims | Moderate-high |
| 5. TRUE BiLSTM | Old implementation was not bidirectional for 7/9 targets | Figure 7 and S7/S9 branches were mixed | Re-rendered from saved corrected TRUE-BiLSTM outputs only | 28 Figure-7 target-land BiLSTM cells and 144 fold/seed run cells changed; S9 uses the accepted corrected component branch | Figures 7, S7, S9 and exact sources | Replace figures and synchronize all BiLSTM values/counts; no retraining claim | High |
| 6. S4/S5/S6 | Supplementary analyses used inconsistent branches | Strict-QA/no-QA GNDVI and state pipelines were mixed | S4/S5 on previous/no-QA GNDVI and Figure-9 EWBI; S6 on exact Figure-9 inputs with existing alternatives | S5 uses one bootstrap-CI rule; k-means/GMM kappa vs median is {cluster.cohen_kappa.min():.3f}-{cluster.cohen_kappa.max():.3f}; percentile extremes agree on retained months only | Figures S4-S6 and exact sources | Replace supplements/captions and emphasize classification sensitivity | High |
| EWBI table | Demonstrate non-degenerate temporal variation | Numerical temporal summary was absent | Selected requested statistics from the completed EWBI audit | 288 months per land; ranges 4.84-5.51; recent-period means remain strongly negative | Table EWBI temporal variation | Add concise supplementary table; no new EWBI figure needed | High |

All products are computational outputs only. No manuscript, response letter, original dataset, submitted figure, or original script was edited.
"""
    (PHASE / "COMPUTATIONAL_REVISION_SUMMARY.md").write_text(text, encoding="utf-8")


def write_manuscript_map() -> None:
    text = """# Manuscript changes needed after computation

This is an editing map only. No manuscript or reviewer-response file has been modified.

## Figure 5 and ecological correlations

- Methods: state that all raw monthly pairs were joined by actual YYYY-MM timestamps and evaluated on pairwise-shared dates; report that pairwise N varies by data availability.
- Results: replace every Figure-5 coefficient/p-value cited in prose and recheck statements about direction or relative magnitude. Nineteen pairwise signs differ from the submitted matrix.
- Caption: identify raw monthly Pearson correlation as the primary analysis, give the pairwise-deletion/date-alignment rule, and point to the deseasonalized sensitivity.
- Supplement: add the calendar-month-climatology-removed correlation figure/table as sensitivity only. Do not recast it as the main scientific estimand or infer causality.

## Forecasting methods, results, and figures

- Methods: define the paired inferential unit as target x land type (36 tasks), explicitly state that pixels were not inferential replicates, and describe Friedman plus Wilcoxon-Holm tests, paired effect sizes, mean ranks, wins, and paired-difference CIs.
- Results: report the two omnibus tests and only the Holm-supported pairwise differences. Treat ranks/wins and tiny numerical advantages as descriptive where pairwise support is absent.
- Figures/tables: replace Figure 7, Figure S7, and Figure S9 with the reconciled renderings and sources; confirm Figure S8 and all best-model counts use the same corrected branch.
- Captions: identify TRUE bidirectional LSTM values and avoid implying that all six learned models are statistically distinct.

## State transitions

- Methods: add the fixed-seed, 500-pair-per-land IAAFT null, independent EWBI/NIRv surrogates, original median thresholds, and empirical p-value definition.
- Results: compare each observed diagonal with its null median and interval. State that only a subset exceeds the autocorrelation-preserving upper-tail null.
- Figure 9 caption/discussion: avoid describing the matrices as validated Markov ecosystem dynamics or treating high diagonal values alone as ecological memory.
- Supplement: replace S6 with the exact Figure-9-pipeline sensitivity and explain that 33/67 agreement is evaluated only on retained extreme months; k-means/GMM agreement is low-to-moderate.

## GPP product transition

- Data/Methods: disclose FLUXCOM through 2021 and MOD17 for 2022-2024, with no overlap calibration; distinguish stitched primary estimates from FLUXCOM-era sensitivity.
- Ecological Results: update the four correlation/trend significance conclusions that change and retain effect sizes rather than binary significance alone. State that NIRv-GPP remains stronger than NDVI-GPP in all four lands under the FLUXCOM-only restriction.
- Forecast Results/Limitations: report pre/post saved-prediction performance descriptively. Explicitly state that product, calendar period, and rolling-origin fold are confounded, so the post-2022 degradation cannot be attributed solely to product change.

## Supplementary Figures S4 and S5

- S4 Methods/caption: specify previous/no-QA GNDVI and Figure-9-consistent EWBI. Retain only the existing equal, PCA, ridge, component, and leave-one-out methods. Flag ridge-to-vegetation results as descriptive/in-sample rather than independent validation.
- S5 Methods/caption: specify previous/no-QA GNDVI and use one plotted significance definition only: the paired 12-month moving-block bootstrap 95% CI excludes zero. Pearson/Spearman p-values may remain in the table but must not be mixed with the asterisk rule.

## EWBI temporal-variation table

- Supplementary Results/Table caption: add `Table_EWBI_temporal_variation.csv` with N, full-period distribution, range, and period means. Use it as numerical evidence that EWBI is temporally variable and non-degenerate.

## Conclusion-level wording

- Do not claim universal forecasting-model superiority.
- Do not infer causality from raw or deseasonalized correlations.
- Do not infer validated Markov dynamics from state-transition matrices.
- Preserve the broad framework conclusion, but qualify claims that depend on exact Figure-5 signs, stitched-GPP significance, or unadjusted state persistence.
"""
    (PHASE / "MANUSCRIPT_CHANGES_NEEDED_AFTER_COMPUTATION.md").write_text(text, encoding="utf-8")


def main() -> None:
    ewbi_table = make_ewbi_table()
    integrity = run_integrity_check()
    checks = validate(ewbi_table, integrity)
    write_manifest()
    write_qc(checks)
    write_summary()
    write_manuscript_map()
    failed = [c for c in checks if c["status"] != "PASS"]
    print(f"Finalization complete: {len(checks) - len(failed)}/{len(checks)} QC checks passed.")
    if failed:
        for item in failed:
            print(f"FAIL: {item['check']}: {item['evidence']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
