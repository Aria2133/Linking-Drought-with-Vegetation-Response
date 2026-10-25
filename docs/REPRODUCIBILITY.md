# Reproducibility

## Checklist (Reviewer 1, Comment 15)

| Requested item | Where |
|---|---|
| Environment / package specification | [`environment/`](../environment) — `requirements.txt`, `environment.yml`, `package_versions.txt` (Python 3.13.4) |
| Exact preprocessing scripts | [`src/preprocessing/`](../src/preprocessing) |
| Model-training scripts | [`src/forecasting/`](../src/forecasting) |
| Hyperparameter configurations | [`config/forecasting_hyperparameters.csv`](../config/forecasting_hyperparameters.csv) |
| Random seeds | [`config/random_seeds.yaml`](../config/random_seeds.yaml) |
| Fold definitions | [`config/rolling_origin_folds.csv`](../config/rolling_origin_folds.csv) |
| Pixel-selection criteria and counts | [`config/pixel_selection_criteria.md`](../config/pixel_selection_criteria.md), [`config/pixel_selection_counts.csv`](../config/pixel_selection_counts.csv) |
| Data-access instructions | [`data_access/`](../data_access) |
| Code for main figures and tables | [`src/figures/`](../src/figures), [`src/tables/`](../src/tables); map in the [README](../README.md) |
| Formal model comparison | [`src/statistics/forecast_model_comparison.py`](../src/statistics/forecast_model_comparison.py) (runs from bundled `results/`) |
| State-transition null model | [`src/statistics/state_transition_iaaft_null.py`](../src/statistics/state_transition_iaaft_null.py) (runs from bundled `results/`) |
| Earth Engine vs Python | [`gee/`](../gee) = Earth Engine (JavaScript); everything in [`src/`](../src) = Python |
| Provenance of every published script | [`FILE_PROVENANCE.md`](FILE_PROVENANCE.md) |

## Running the code
1. `pip install -r environment/requirements.txt` (or `conda env create -f environment/environment.yml`).
2. The two statistical scripts run directly from the bundled tables in `results/`:
   - `python src/statistics/forecast_model_comparison.py` reproduces the Friedman tests (R²: χ²(5) = 37.079, p = 5.77 × 10⁻⁷;
     nRMSE: χ²(5) = 40.889, p = 9.88 × 10⁻⁸) and the 15 Holm-corrected Wilcoxon comparisons per metric.
   - `python src/statistics/state_transition_iaaft_null.py` reproduces the IAAFT null (500 surrogate pairs per land type, seed
     20261013): 4 of 16 diagonal self-transition proportions exceed the null at upper-tail p < 0.05.
3. For everything else: obtain the source data (`data_access/README.md`), run the Earth Engine exports (`gee/`), replace the
   placeholders `<DATA_ROOT>` / `<PROJECT_ROOT>` in each script, and follow [`WORKFLOW.md`](WORKFLOW.md) in order.

The forecasting benchmark trains several thousand small models (9 targets × 4 land types × 4 folds × up to 3 seeds × 6
learned models, plus SVR and baselines); a CUDA GPU is recommended for the neural models.

## Configuration notes
- **Common benchmark.** All models use a 12-month input window, 1-month horizon, the four expanding rolling-origin folds
  and the same task-specific pixels. In each fold the model weights are fitted on the core training months, the last 24
  months of the fold's training period are the internal validation window (early stopping; SVR grid selection), and the
  outer test window is scored once.
- **Preprocessing boundary (published runs).** Calendar-month imputation, calendar-month linear detrending (where used) and
  calendar-month z-scaling were fitted on all dates up to the **end of the internal validation window**
  (`preprocessing_fit_end` in `config/rolling_origin_folds.csv`), i.e. core training plus internal validation. No outer-test
  observation enters any fitted quantity (no outer-test leakage), but internal-validation targets influence the
  preprocessing applied to the training inputs, so early stopping and the SVR grid selection used a mildly optimistic
  validation signal (internal-validation contamination). The GPP ×31.56 factor for 2022–2024 (MOD17 segment) is a fixed
  constant, not estimated from data. Pixel eligibility is based on data availability before 2022 and in 2024
  (retrospective availability-based selection; target values are not used).
- **Preprocessing boundary (new runs).** `src/forecasting/fold_boundaries.py` now defaults to `fit_subset`: preprocessing is
  fitted on core-training dates only and applied unchanged to validation and test inputs; the RBF-SVR final refit on
  training + validation refits preprocessing through the validation end for that refit only. Outputs are labelled
  `reviewer_mode_b_fitsubset`. Set `PREPROCESSING_BOUNDARY = "legacy_through_validation"` in `candidate_config.py` to
  reproduce the published runs (label `reviewer_mode_b`). The published results were **not** regenerated with the new
  boundary; doing so requires rerunning every learned model (36 tasks × 4 folds × seeds) — see "Pending reruns" below.
- **BiLSTM.** The first recurrent layer is bidirectional for all nine targets (128 units per direction for NDVI, GNDVI,
  NIRv, GOSIF/SIF, GPP, LST and VPDmax; 150 for TWS; 200 for SPEI-1), the second layer is unidirectional, dropout 0.20,
  head Linear(units→10)–ReLU–Linear(10→1), AdamW (lr 0.001, weight decay 0), batch 64, ≤ 8 epochs, patience 3,
  sample cap 4000, gradient clipping 1.0. Runs: seven targets in `corrected_bilstm/run_corrected_true_bilstm.py`;
  SPEI-1 in `gndvi_spei_no_detrend/`; TWS in the common benchmark (`benchmark/05_run_tcn_nbeats_nhits.py`).
  The BiLSTM function inside `benchmark/model_core.py` is not used for the final results of the seven targets above.
- **TWS × BiLSTM — verified history.** The Figure 7 source used for the August-2026 manuscript (and its Tables 4–5 and the
  statistical comparison) took its four BiLSTM-TWS cells from the 18-month hydroclimate run
  (`bilstm_special_tws_state_space`, 3 seeds × 4 folds). That run was adopted because the hydroclimate package compared
  configurations on the held-out test periods ("selective adoption"), and it differs from the common design in lookback,
  scaling, gap handling, optimizer, epochs, patience and sample cap. The common 12-month BiLSTM-TWS run already existed
  (seed 42 × 4 folds; `reviewer_safe_all_pixel_metrics.csv`; its saved predictions reproduce the metric rows exactly and also
  feed Figure S9). All 216 cells in this repository now use the common 12-month benchmark
  (`results/forecasting/figure7_final_task_metrics.csv`, column `selected_configuration`);
  `src/tables/figure7_tws_common_benchmark_correction.py` made the replacement and
  `src/tables/rebuild_main_benchmark_from_runs.py` rebuilds both variants from the raw run tables listed in
  `config/forecast_result_registry.csv` (outputs and SHA-256 manifest in `results/audit/forecast_provenance/`). No model was
  retrained.

  | land type | BiLSTM-TWS R² (mean ± SD) | nRMSE (mean ± SD) |
  |---|---|---|
  | Grasslands | 0.761 ± 0.065 | 0.485 ± 0.067 |
  | Natural Forests | 0.367 ± 0.230 | 0.786 ± 0.140 |
  | Savannas | 0.583 ± 0.089 | 0.643 ± 0.067 |
  | Shrublands | 0.442 ± 0.259 | 0.733 ± 0.167 |

  Consequences: BiLSTM mean R² 0.661, mean nRMSE 0.538 (Table 4); TWS target mean R² 0.296, nRMSE 0.792 (Table 5). Best-model
  counts, predictability classes, both Friedman tests and all nine Holm-significant pairs are unchanged; in the pairwise table
  only the non-significant nRMSE BiLSTM–N-HiTS row changed (W 215, p = 0.064, Holm p = 0.322, r = 0.354).
  These values were then updated by the GPP 2022 correction (section "MOD17 GPP 2022 (resolved)"): current values are in
  `results/forecasting/` and `results/gpp_2022_correction/`.
- **18-month TWS BiLSTM** (`tws_bilstm_special/`: 18-month lookback, Adam, ≤ 30 epochs, patience 7, global min–max scaling,
  training-only state-space gap filling, sample cap 6000, seeds 42/123/2026): supplementary hydroclimate sensitivity; not
  used in the main common 12-month benchmark. `config/forecast_result_registry.csv` marks it `sensitivity`, and
  `src/forecasting/provenance.py` refuses sensitivity or superseded rows in any main-benchmark aggregation (the statistics
  script checks its input table the same way).
- **Result provenance checks.** `src/forecasting/provenance.py` verifies, before aggregation: 12-month lookback and a single
  preprocessing/scaling family; one registered main configuration per model × target; no duplicate task/fold/seed records;
  complete fold × seed grids; balanced folds (so the run mean equals the fold-balanced mean and tasks are never weighted by
  their number of seeds); and a registered configuration label on every downstream task cell. Tests: `pytest tests`
  (set `DROUGHT_PROJECT_ROOT` to the project folder to include the lineage test on the saved runs).
- **Pending reruns.** (1) To report results under the corrected `fit_subset` preprocessing boundary, rerun all learned-model
  benchmark tasks (LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, XGBoost, RBF-SVR; 9 targets × 4 land types × 4 folds × their seeds).
  (2) No rerun is needed for the 12-month TWS BiLSTM: its saved run (seed 42) exists. A three-seed version would need two
  extra seeds (123, 2026) for 4 land types × 4 folds if seed parity with LSTM/XGBoost/N-BEATS is wanted.
- **Seeds.** Not every model was run with three seeds (`config/random_seeds.yaml`).
- **Paths.** Scripts are published as run, except that absolute local path roots were replaced by placeholders
  (checksums of the original and published files are in `FILE_PROVENANCE.csv`). Scripts written during the 2026 revision
  locate their inputs relative to the authors' revision workspace (`HERE.parents[...]`); their input files are named at the top
  of each script.
- **Manual post-processing.** Some figure files were cropped (titles/footers removed) after rendering; the scripts produce the
  uncropped versions. The final label-only edits of Figure 5 and two supplementary figures (e.g. "SPEI" → "SPEI-1") were applied
  without a saved script; no data changed.

## Code paths that are not part of the final results
These remain inside scripts that also contain final code; they are not used for any reported result.

| Where | What |
|---|---|
| `forecasting/benchmark/candidate_config.py`, `02_run_baselines.py`, `06_compile_benchmark.py`, `gndvi_spei_no_detrend/`, `svr/` | SARIMA baseline branches (dropped from the final paper) |
| `forecasting/*/candidate_config.py` (`MODE_A`), `01_define_rolling_origin_folds.py` | single 2024 hold-out design ("Mode A"); the final design is the four rolling-origin folds |
| `forecasting/tws_bilstm_special/` | 18-month TWS / 24-month SPEI hydroclimate configurations — supplementary hydroclimate sensitivity; not used in the main common 12-month Figure 7 benchmark |
| `figures/fig08_nirv_change_maps.py` | its Figure 7 function (only `render_figure8_no_boxes` is used) |
| `figures/figS02_S03_S04_S05_supplementary.py` | only `build_s4` (gap filling) and `build_s5` (detrending) are current; its S4/S5 builders are superseded by `ecological/sensitivity/harmonized_S4_S5_S6.py` |
| `preprocessing/02_landtype_series/02_prepare_paper_standardized_anomalies.py`; `ecological/sensitivity/ewbi_period_response_and_severe_area_tables.py` (S5 part) | GNDVI rows built from the QA-masked GNDVI product; the final analyses use the previous/no-QA GNDVI series (`03_previous_gndvi_series_and_figures_5_6.py`, `harmonized_S4_S5_S6.py`) |
| `ecological/correlations/cross_correlation_and_submitted_fig5_logic.py` | its Figure 5 function (position-truncated pairing) is superseded by `figure5_date_aligned_correlations.py`; `ccf_calculate` (Figure 6) is current |

## MOD17 GPP 2022 (resolved)
The original 2022 MOD17A2H monthly exports were duplicated (nine of twelve files byte-identical, two more a second pair;
`results/audit/gpp_table_s8/mod17_export_hashes_2022_2024.csv`), and the former "convert if median < 0.2" rule left
Natural Forests 2023-05/06 in kg C m⁻² month⁻¹. Both were corrected and every dependent result was rerun
(`results/gpp_2022_correction/README.md`):
1. 2022 rebuilt from Terra MOD17A2H v061 on Microsoft Planetary Computer with the processing of the Earth Engine script
   (`src/ecological/gpp/mod17_monthly_from_planetary_computer.py`; 8 x 8-pixel block mean = Earth Engine pyramid level 3);
   checked with `src/ecological/gpp/mod17_monthly.py verify` (12 distinct months, plausible values).
2. Land-type rasters and the forecasting cache rebuilt with the original masking (`src/preprocessing/01_landtype_masking/
   apply_rebuilt_mod17_2022.py`; masking reproduces the original files byte-for-byte); every MOD17 month converted with
   x 1000 / days (`src/preprocessing/02_landtype_series/apply_rebuilt_mod17_2022_series.py`; only the 50 GPP rows change).
3. GPP forecasting rerun for folds 3–4 (folds 1–2 never read 2022) with the published code, settings and seeds — all
   models, baselines and RBF-ε-SVR; published runs were first reproduced exactly (XGBoost, LSTM, BiLSTM, SVR fold 1).
   Merged by `src/tables/apply_gpp_2022_forecast_fix.py` (only GPP rows replaced; all reproduction checks pass);
   Tables 3–4 and the Friedman / Wilcoxon-Holm tests by `src/tables/gpp_2022_fix_tables_3_4_stats.py`.
4. Figure 3, Supplementary S4/S5 (Tables S5/S7), S7/S8 (Table S10), Table S8 (`src/ecological/gpp/gpp_table_s8_final.py`)
   and the statistical comparison were regenerated; `results/forecasting/*` hold the corrected values.

## Known gaps
- The script that masked gridMET SPEI-1 exports to land types was not preserved (see `WORKFLOW.md`). It was not reconstructed:
  the California-wide gridMET SPEI-1 exports it read (`SPEI1_GRIDMET_CA_YYYY_MM`, Earth Engine Drive folder `SPEI1_GRIDMET_TIFs`)
  are not kept locally, so a reconstruction could not be validated against the final land-type products.
- Figure 6 cross-correlations are computed on raw monthly land-type means (no detrending, deseasonalising or prewhitening);
  GNDVI enters through 2021 (with two forward-filled months), the other variables through 2024. They describe lagged
  co-variation only and are not causal.
