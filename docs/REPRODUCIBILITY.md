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
| Earth Engine vs Python | [`gee/`](../gee) = Earth Engine (JavaScript); everything in [`src/`](../src) = Python |
| Provenance of every published script | [`docs/FILE_PROVENANCE.md`](FILE_PROVENANCE.md) |

## Running the code
1. `pip install -r environment/requirements.txt` (or `conda env create -f environment/environment.yml`).
2. Obtain the source data (`data_access/README.md`) and run the Earth Engine exports (`gee/`).
3. Replace the path placeholders `<DATA_ROOT>` and `<PROJECT_ROOT>` at the top of each script with local folders.
4. Run the steps in [`WORKFLOW.md`](WORKFLOW.md) in order. Each script writes its own tables/figures under `<PROJECT_ROOT>`.

The forecasting benchmark trains several thousand small models (9 targets × 4 land types × 4 folds × up to 3 seeds × 6
learned models, plus SVR and baselines); a CUDA GPU is recommended for the neural models.

## Configuration notes (read before rerunning)
- **BiLSTM.** All final BiLSTM results use a bidirectional first recurrent layer.
  - NDVI, GNDVI, NIRv, GOSIF/SIF, GPP, LST and VPDmax: `corrected_bilstm/run_corrected_true_bilstm.py`.
  - SPEI-1: `gndvi_spei_no_detrend/` (200 units).
  - TWS: `tws_bilstm_special/`, a separate configuration — lookback 18, 150 units, Adam, ≤ 30 epochs, patience 7,
    global min–max scaling, training-only state-space gap filling (see `config/forecasting_hyperparameters.csv`).
  - The `BiLSTM` branch inside `benchmark/model_core.py` (bidirectional only for SPEI-1 and TWS) is the superseded
    implementation; its BiLSTM rows are replaced by `corrected_bilstm/patch_corrected_bilstm_into_results.py`.
- **Seeds.** Not every model was run with three seeds (`config/random_seeds.yaml`).
- **Paths.** Scripts are published as run, except that absolute local path roots were replaced by placeholders
  (checksums of the original and published files are in `FILE_PROVENANCE.csv`).
- **Manual post-processing.** Some final figure files were cropped (titles/footers removed) after rendering; the scripts
  produce the uncropped versions.
- **Figures 3–4 and S1** were drawn by code run in an interactive coding session; that code is restored here as standalone scripts.
- **Not preserved:** the script that masked gridMET SPEI-1 to land types (see `WORKFLOW.md`).
- **Statistical model comparison** (Friedman, Wilcoxon, Holm on 36 tasks) and the **IAAFT state-transition null**: scripts to be added.
