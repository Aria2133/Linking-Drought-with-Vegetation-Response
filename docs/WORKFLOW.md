# Workflow (run order)

## 1. Google Earth Engine — `gee/scripts/`
Export monthly California GeoTIFFs (EPSG:4326) for NDVI, NIRv, GNDVI, LST, SPEI-1, VPDmax, TWS and MOD17A2H GPP, plus the
2021 MCD12Q1 land-type masks. GOSIF and FLUXCOM GPP are downloaded outside Earth Engine.

## 2. Land-type masking — `src/preprocessing/01_landtype_masking/`
| Script | Does |
|---|---|
| `mask_<land>_2001_2021.py` (4 files) | applies the 2021 land-type mask (nearest-neighbour onto each product grid) to the 2001–2021 exports |
| `mask_test_period_2022_2024.py` | same for the 2022–2024 exports |
| `mask_gndvi_2001_2021.py` | GNDVI masking |
| `fluxcom_gpp_netcdf_to_landtype_tifs.py` | FLUXCOM GPP NetCDF → monthly land-type GeoTIFFs |

The script that masked the gridMET SPEI-1 exports to land types was not preserved; the masked files follow the same
nearest-mask procedure (file names `SPEI1_GRIDMET_CA_YYYY_MM_<Land>.tif`).

## 3. Monthly land-type series — `src/preprocessing/02_landtype_series/`
1. `00_inventory_and_series.py` — unweighted mean of valid pixels per variable, land type and month (`detrending_utils.read_raster_mean`);
   TWS/LST gaps filled with same-calendar-month interpolation across years (`monthwise_fill`).
2. `01_preprocessing_versions.py` — removes the 2001–2011 calendar-month climatology (main analysis); detrended versions are sensitivity only.
3. `02_prepare_paper_standardized_anomalies.py` — baseline-referenced standardised anomalies (Figs 3–4).
4. `03_previous_gndvi_series_and_figures_5_6.py` — GNDVI land-type series used in the final analyses.

## 4. Ecological analyses — `src/ecological/`
Main ecological analyses use baseline-referenced, deseasonalised-only anomalies (2001–2011 calendar-month climatology removed;
no long-term detrending). Detrending and gap-filling alternatives are sensitivity analyses only.

| Analysis | Script |
|---|---|
| **EWBI** — equal-weight signed mean of land-type components (+SPEI-1, +TWS, −VPDmax, −LST), each deseasonalised and standardised over time with its 2001–2011 mean and SD (no spatial standardisation) | `ewbi/02_ewbi_by_preprocessing.py` (`compute_ewbi`) |
| EWBI temporal-variation diagnostics; recomputation check; component correlations; TWS/LST fill status | `ewbi/ewbi_temporal_variation_diagnostics.py`, `../tables/ewbi_temporal_variation_table.py` |
| EWBI standardisation variant and **TWS/LST observed-only baseline** sensitivity | `ewbi/ewbi_standardisation_and_observed_baseline_sensitivity.py` |
| EWBI components, leave-one-out, equal / PCA / ridge weighting; period-stratified responses; state-classification alternatives (33/67 percentile, k-means, GMM) — all on the Figure-9 EWBI/NIRv inputs and previous/no-QA GNDVI | `sensitivity/harmonized_S4_S5_S6.py` |
| Descriptive EWBI × NIRv classes (land-type medians) and empirical month-to-month transition proportions | `states/01_main_ewbi_states_transitions.py`, `states/fixed_utils.py` |
| Resistance / recovery and preprocessing sensitivity | `states/03_resilience_and_state_sensitivity.py` |
| NIRv % change maps; operational −40 % severe-response threshold with −30 / −50 % sensitivity; latitude-aware areas; paired spatial-block bootstrap | `nirv_response_and_tws_scale/04_nirv_change_maps_severe_area_bootstrap.py`, `sensitivity/ewbi_period_response_and_severe_area_tables.py` (severe-area table) |
| TWS coarse-support and resampling sensitivity | `nirv_response_and_tws_scale/02_*.py`, `03_*.py` |
| Gap-filling comparison | `gap_filling/` |
| Figure 5: raw monthly Pearson correlations on exact YYYY-MM pairs (primary); deseasonalised correlations (sensitivity) | `correlations/figure5_date_aligned_correlations.py` |
| Figure 6: cross-correlation functions on raw monthly means (no prewhitening) | `correlations/cross_correlation_and_submitted_fig5_logic.py` (`ccf_calculate`) |
| GPP: FLUXCOM-X 2001–2021 + MOD17A2H 2022–2024 as analysed; FLUXCOM-only (≤ 2021) sensitivity; no cross-product calibration | `gpp/gpp_product_transition_sensitivity.py` |

## 5. Forecasting — `src/forecasting/`
Target-specific, univariate: each of 9 targets × 4 land types is forecast one month ahead from its own previous 12 months.

| Step | Script |
|---|---|
| Pixel screening and caches, supervised windows | `benchmark/raster_pipeline.py` (rule: `config/pixel_selection_criteria.md`) |
| Rolling-origin folds | `benchmark/01_define_rolling_origin_folds.py` (`config/rolling_origin_folds.csv`) |
| Baselines (persistence, seasonal persistence, monthly climatology) | `benchmark/02_run_baselines.py` |
| XGBoost; LSTM; TCN / N-BEATS / N-HiTS and BiLSTM for TWS | `benchmark/03_*.py`; `04_*.py`; `05_*.py` |
| GNDVI and SPEI-1 without detrending (all learned models) | `gndvi_spei_no_detrend/run_no_detrend_gndvi_spei.py` |
| Corrected BiLSTM for NDVI, GNDVI, NIRv, GOSIF/SIF, GPP, LST, VPDmax | `corrected_bilstm/run_corrected_true_bilstm.py` |
| RBF-ε-SVR | `svr/run_svr_benchmark.py` |
| Combine results; replace superseded BiLSTM rows; corrected supplementary sources | `benchmark/06_compile_benchmark.py`, `corrected_bilstm/patch_corrected_bilstm_into_results.py`, `corrected_bilstm/reconcile_corrected_bilstm_figures.py` |
| Forecast-derived EWBI | `forecast_derived_ewbi/recover_forecast_derived_ewbi.py` |
| *Supplementary hydroclimate sensitivity; not used in the main common 12-month Figure 7 benchmark:* 18-month TWS / 24-month SPEI configurations | `tws_bilstm_special/` |

Fold windows and preprocessing boundaries are defined in `fold_boundaries.py`. Published runs fitted imputation, detrending
(where used) and calendar-month z-scaling on dates up to the end of the 24-month internal validation window (core training +
validation; never test months). New runs default to fitting preprocessing on core-training dates only (`fit_subset`); see
`REPRODUCIBILITY.md`. `config/forecast_result_registry.csv` names the saved run table that supplies each model × target of the
main benchmark and marks sensitivity/superseded runs, which `provenance.py` keeps out of every main-benchmark aggregation.

## 6. Statistics — `src/statistics/`
| Analysis | Script |
|---|---|
| Friedman test across the six learned models (36 paired tasks), then two-sided Wilcoxon signed-rank tests with Holm correction over the 15 pairs per metric | `forecast_model_comparison.py` |
| Extended tables (ranks, effect sizes, cluster bootstrap over targets, target-level sensitivity) | `forecast_inference_extended.py` |
| IAAFT null for diagonal self-transition proportions (500 independent EWBI/NIRv surrogate pairs per land type) | `state_transition_iaaft_null.py` |

## 7. Figures and tables — `src/figures/`, `src/tables/`
See the figure/table map in the main [README](../README.md).

Forecasting figures and tables are built in this order (no model is retrained):
0. `src/tables/rebuild_main_benchmark_from_runs.py --project-root <PROJECT_ROOT>` — rebuilds the 216 task cells, Tables 4–5 and the
   statistics from the raw run tables in the registry (main 12-month design and the historical published variant); writes
   `results/audit/forecast_provenance/` with a SHA-256 manifest.
1. `src/tables/figure7_tws_common_benchmark_correction.py` — final Figure 7 source: the accepted source table with the four
   BiLSTM × TWS cells taken from the common 12-month benchmark (mean and SD, ddof = 1, over the four rolling-origin fold rows);
   exported as `results/forecasting/figure7_final_task_metrics.csv`.
2. `src/figures/fig07_point_errorbar_best_counts.py` (Figure 7) and `src/figures/figS08_forecast_heatmaps.py` lineage (Figure S8) read that source.
3. `src/tables/tables_4_5_from_corrected_figure7.py` (Tables 4–5), `src/statistics/forecast_model_comparison.py`,
   `src/statistics/forecast_inference_extended.py` and `src/tables/forecast_pairwise_statistics_table.py` (statistical comparison).
