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
| Analysis | Script |
|---|---|
| **EWBI** — signed mean of temporally standardised (2001–2011 mean/SD) land-type components: +SPEI-1, +TWS, −VPDmax, −LST, equal weights | `ewbi/02_ewbi_by_preprocessing.py` (`compute_ewbi`) |
| Descriptive EWBI–NIRv states (median quadrants) and empirical month-to-month transition proportions | `states/01_main_ewbi_states_transitions.py`, `states/fixed_utils.py` |
| Resistance / recovery and preprocessing sensitivity | `states/03_resilience_and_state_sensitivity.py` |
| NIRv % change maps, severe-response area (−30/−40/−50 %), spatial-block bootstrap | `nirv_response_and_tws_scale/04_nirv_change_maps_severe_area_bootstrap.py` |
| TWS spatial-support and resampling sensitivity | `nirv_response_and_tws_scale/02_*.py`, `03_*.py` |
| Gap-filling comparison | `gap_filling/` |
| Cross-correlation functions (Fig 6) and Fig 5 correlations | `correlations/cross_correlation_and_submitted_fig5_logic.py` |
| EWBI component / weighting tests, period-stratified responses, state-classification alternatives (S4–S6) | `sensitivity/` |

## 5. Forecasting — `src/forecasting/`
Target-specific, univariate: each of 9 targets × 4 land types is forecast one month ahead from its own previous months.
| Step | Script |
|---|---|
| Pixel screening and caches, supervised windows | `benchmark/raster_pipeline.py` (rule: `config/pixel_selection_criteria.md`) |
| Rolling-origin folds | `benchmark/01_define_rolling_origin_folds.py` (`config/rolling_origin_folds.csv`) |
| Baselines | `benchmark/02_run_baselines.py` |
| XGBoost, LSTM, TCN / N-BEATS / N-HiTS | `benchmark/03_*.py`, `04_*.py`, `05_*.py` |
| GNDVI and SPEI-1 without detrending (all learned models) | `gndvi_spei_no_detrend/run_no_detrend_gndvi_spei.py` |
| Corrected BiLSTM (7 targets) | `corrected_bilstm/run_corrected_true_bilstm.py` |
| BiLSTM for TWS | `tws_bilstm_special/` |
| RBF-ε-SVR | `svr/run_svr_benchmark.py` |
| Combine results, replace superseded BiLSTM rows | `benchmark/06_compile_benchmark.py`, `corrected_bilstm/patch_corrected_bilstm_into_results.py`, `../tables/` |
| Forecast-derived EWBI | `forecast_derived_ewbi/recover_forecast_derived_ewbi.py` |

Preprocessing inside every fold (imputation, detrending where used, calendar-month z-scaling) is fitted on dates up to the fold's
training end only; the last 24 training months are the internal validation window; test months are never used for fitting.

## 6. Figures and tables — `src/figures/`, `src/tables/`
See the figure/table map in the main [README](../README.md).
