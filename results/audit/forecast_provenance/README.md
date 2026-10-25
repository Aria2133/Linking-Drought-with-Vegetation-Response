# Forecasting result provenance audit (October 2026)

Built by `src/tables/rebuild_main_benchmark_from_runs.py` from the saved per-run metric tables listed in
`config/forecast_result_registry.csv`. No model was trained. `manifest.json` records the SHA-256 hash of every input, the
code and the source Git commit; `before_after_published_vs_main.csv` lists every value that differs between the two
variants.

## Lineage verdicts

| Output | Source / generator | Configuration actually used | Coverage | Verdict |
|---|---|---|---|---|
| Figure 7, August-2026 manuscript (source `FIGURE_REFINEMENT_PHASE/source_tables/Figure_7_source.csv`, SHA-256 `f8a9cec9…`) | Codex `render_figure7_original_design_corrected.py` ← that source table | 212 cells from the main 12-month configurations; the **4 BiLSTM × TWS cells from the 18-month sensitivity** `bilstm_special_tws_state_space` (lookback 18, global min–max, training-only state-space gap filling, Adam, ≤ 30 epochs, patience 7, cap 6000) | 6 models × 9 targets × 4 land types × 4 folds; seeds 1 or 3 per model × target (registry); BiLSTM-TWS 3 seeds | **Verified**: all 216 cells rebuild from raw rows to < 1e-12 (variant `published_aug2026_18m_tws`) |
| Figure 7, current (`results/forecasting/figure7_final_task_metrics.csv`; FINALIZED `Figure_7.png` of 2026-10-20) | `src/tables/figure7_tws_common_benchmark_correction.py` → `src/figures/fig07_point_errorbar_best_counts.py` | all 216 cells main 12-month; BiLSTM × TWS = common run, seed 42 | as above; BiLSTM-TWS 1 seed × 4 folds | **Verified** (variant `main_common_12m`, identical to < 1e-12) |
| Manuscript Table 4 (model ranking; "Table 3" in older scripts) | `tables_4_5_from_corrected_figure7.py` / `verify_tables_2_4_5_6.py` logic from the Figure 7 source | manuscript (2026-10-23 copy) shows BiLSTM 0.669 / 0.532 = the **18-month** variant | 36 tasks, unweighted mean of task cells; wins on task means | **Verified – conflicting with the 12-month design**; main design gives 0.661 / 0.538 |
| Manuscript Table 5 (target predictability; "Table 4" in older scripts) | same | manuscript shows TWS 0.309 / 0.783 = **18-month** variant | 24 cells per target | **Verified – conflicting**; main design gives 0.296 / 0.792 (class and best models unchanged) |
| Supplementary statistical comparison (S14 in the 2026-10-25 Supplement; S12 earlier) | `forecast_model_comparison.py` / `forecast_inference_extended.py` / `forecast_pairwise_statistics_table.py` | Supplement 2026-10-25 shows nRMSE BiLSTM–N-HiTS W = 215, Holm p = 0.322 = **main 12-month** variant | 36 tasks as paired units | **Verified** (consistent with the main design) |
| Forecast-derived EWBI / Figure S9 | `forecast_derived_ewbi/recover_forecast_derived_ewbi.py` ← saved predictions | TWS component BiLSTM from `reviewer_safe_remaining_models_predictions.csv.gz` (lookback 12, `reviewer_mode_b`, seed 42); recomputed pooled R² reproduces all 16 metric rows exactly | 4 land types × 4 folds | **Verified**; never used the 18-month run, so no regeneration needed |

How the 18-month run entered Figure 7: the hydroclimate package (`05_forecasting_hydroclimate_special_candidate`)
compared its configurations with the common run **on the held-out test periods** and recommended "selective adoption" of
`bilstm_special_tws_state_space`; it was then used for all four land types (the common run scored higher for Grasslands).
This is test-period configuration selection and breaks the common-design comparison; the 18-month run is therefore kept
only as a labelled sensitivity.

## Aggregation (unchanged from the published workflow)
Each saved row is one model × target × land type × fold × seed run, with R² and nRMSE computed on all valid pixel-month
predictions of the run pooled together, in the standardized (calendar-month z) space. A task cell is the mean (SD, ddof = 1)
over its fold × seed rows; every fold of a cell has the same seeds, so this equals the fold-balanced mean (checked). Model
and target summaries are unweighted means over task cells, so cells with three seeds do not carry more weight than
one-seed cells. Pixels are never treated as replicates; the statistical tests use the 36 tasks as paired units.

## Values that change (published 18-month variant → main 12-month design)
| Quantity | Published | Main |
|---|---|---|
| BiLSTM mean R² / mean nRMSE (Table 4) | 0.669 / 0.532 | 0.661 / 0.538 |
| TWS mean R² / mean nRMSE (Table 5) | 0.309 / 0.783 | 0.296 / 0.792 |
| BiLSTM × TWS cells (R² mean ± SD): Grasslands | 0.756 ± 0.085 | 0.761 ± 0.065 |
| Natural Forests | 0.415 ± 0.274 | 0.367 ± 0.230 |
| Savannas | 0.685 ± 0.054 | 0.583 ± 0.089 |
| Shrublands | 0.607 ± 0.111 | 0.442 ± 0.259 |
| nRMSE BiLSTM vs N-HiTS Wilcoxon W / raw p / Holm p | 211 / 0.056 / 0.278 | 215 / 0.064 / 0.322 (not significant in either) |

Unchanged: best-model counts (XGBoost 17/18, BiLSTM 7/7, TCN 6/4, N-BEATS 2/4, LSTM 4/3, N-HiTS 0/0), predictability
classes and best model per target (BiLSTM still best for TWS), Friedman tests (R² χ²(5) = 37.08, p = 5.77 × 10⁻⁷;
nRMSE χ²(5) = 40.89, p = 9.88 × 10⁻⁸), and all nine Holm-significant pairs.

## Not regenerated
The published runs fitted preprocessing through the end of the internal validation window (see `docs/REPRODUCIBILITY.md`).
The code now defaults to fitting it on core-training dates only for new runs; the existing results were **not** rerun under
that boundary. Doing so requires rerunning every learned model for 36 tasks × 4 folds × their seeds.
