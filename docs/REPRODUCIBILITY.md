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
- **Common benchmark.** All models use a 12-month input window, 1-month horizon, the four expanding rolling-origin folds,
  the same task-specific pixels and training-only preprocessing (calendar-month z-scaling fitted through each fold's
  training end; the last 24 training months are the internal validation window).
- **BiLSTM.** The first recurrent layer is bidirectional for all nine targets (128 units per direction for NDVI, GNDVI,
  NIRv, GOSIF/SIF, GPP, LST and VPDmax; 150 for TWS; 200 for SPEI-1), the second layer is unidirectional, dropout 0.20,
  head Linear(units→10)–ReLU–Linear(10→1), AdamW (lr 0.001, weight decay 0), batch 64, ≤ 8 epochs, patience 3,
  sample cap 4000, gradient clipping 1.0. Runs: seven targets in `corrected_bilstm/run_corrected_true_bilstm.py`;
  SPEI-1 in `gndvi_spei_no_detrend/`; TWS in the common benchmark (`benchmark/05_run_tcn_nbeats_nhits.py`).
  The BiLSTM function inside `benchmark/model_core.py` is not used for the final results of the seven targets above.
- **TWS × BiLSTM — open provenance item.** The current Figure 7 source table
  (`results/forecasting/figure7_final_task_metrics.csv`, column `selected_configuration`) takes its four BiLSTM-TWS cells
  from `tws_bilstm_special/`, a *supplementary hydroclimate sensitivity* (18-month lookback, Adam, ≤ 30 epochs, patience 7,
  global min–max scaling, training-only state-space gap filling, 3 seeds). The common 12-month BiLSTM-TWS run also exists:

  | land type | R² in current table (18-month) | R² common 12-month run | nRMSE current | nRMSE 12-month |
  |---|---|---|---|---|
  | Grasslands | 0.756 | 0.761 | 0.487 | 0.486 |
  | Natural Forests | 0.415 | 0.367 | 0.748 | 0.786 |
  | Savannas | 0.685 | 0.583 | 0.559 | 0.643 |
  | Shrublands | 0.607 | 0.443 | 0.622 | 0.733 |

  Substituting the common run changes BiLSTM's mean R² from 0.669 to 0.661 and mean nRMSE from 0.532 to 0.538, and leaves the
  descriptive best-model counts, both Friedman tests and every Holm-significant pair unchanged. Which values the manuscript
  reports is an author decision; until then the repository reproduces the current table as published.
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
| `forecasting/tws_bilstm_special/` | 18-month TWS / 24-month SPEI hydroclimate configurations — supplementary sensitivity only |
| `figures/fig08_nirv_change_maps.py` | its Figure 7 function (only `render_figure8_no_boxes` is used) |
| `figures/figS02_S03_S04_S05_supplementary.py` | only `build_s4` (gap filling) and `build_s5` (detrending) are current; its S4/S5 builders are superseded by `ecological/sensitivity/harmonized_S4_S5_S6.py` |
| `preprocessing/02_landtype_series/02_prepare_paper_standardized_anomalies.py`; `ecological/sensitivity/ewbi_period_response_and_severe_area_tables.py` (S5 part) | GNDVI rows built from the QA-masked GNDVI product; the final analyses use the previous/no-QA GNDVI series (`03_previous_gndvi_series_and_figures_5_6.py`, `harmonized_S4_S5_S6.py`) |
| `ecological/correlations/cross_correlation_and_submitted_fig5_logic.py` | its Figure 5 function (position-truncated pairing) is superseded by `figure5_date_aligned_correlations.py`; `ccf_calculate` (Figure 6) is current |

## Known gaps
- The script that masked gridMET SPEI-1 exports to land types was not preserved (see `WORKFLOW.md`).
- Figure 6 cross-correlations are computed on raw monthly land-type means (no detrending, deseasonalising or prewhitening);
  GNDVI enters through 2021 (with two forward-filled months), the other variables through 2024. They describe lagged
  co-variation only and are not causal.
