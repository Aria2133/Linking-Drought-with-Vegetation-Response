# File provenance

Every script in `src/` is a copy of the file that produced the reported results in the authors' project.
[`FILE_PROVENANCE.csv`](FILE_PROVENANCE.csv) lists, for each published file:

- `original_project_file` — path in the authors' project (or `<DATA_ROOT>` data workspace);
- `sha256_original` / `sha256_published` — checksums before and after publication;
- `status` — *final (submitted lineage)*, *final (corrected revision lineage)*, *final (October 2026 revision)*, or
  *submitted lineage; harmonised revision version pending* (now only the severe-area/S5 table script, whose S5 part is
  superseded by `ecological/sensitivity/harmonized_S4_S5_S6.py` while its severe-area table remains current);
- `path_check` — what was edited: normally path roots only. Three files carry one documented additional edit:
  - `src/figures/fig09_transition_matrices.py`: colorbar label "Transition probability (%)" → "Observed transition proportion (%)" (label only);
  - `src/figures/fig03_fig04_anomaly_timeseries.py`: caption text "QA-controlled GNDVI" → "GNDVI (previous/no-QA product)" (the plotted series is the previous/no-QA GNDVI);
  - `src/statistics/state_transition_iaaft_null.py`: input/output path block pointed at the bundled `results/states/` tables
    (verified to reproduce the original output exactly).
- `role` — what the script does.

Written for this repository (no project original): `src/statistics/forecast_model_comparison.py` (minimal, verified
re-implementation of the accepted Friedman/Wilcoxon/Holm analysis), the `results/` tables exported from the final source
tables, and all documentation.

Placeholders used for absolute paths: `<DATA_ROOT>`, `<PROJECT_ROOT>`, `<DOWNLOADS>`, `<WORKSPACE>`, `<HOME>`.

Earth Engine scripts in `gee/scripts/` are byte-identical to the scripts that were run (renamed for clarity; the original
names are kept in Git history).

**Removed from the current tree** (still in Git history): thesis-era exploratory scripts and Tonzi Ranch site analyses,
abandoned forecasting variants (GNN, CNN, random forest, ARIMA/SARIMA-only and multivariate runs, single-2024-test scripts),
intermediate audit tools, old output images/rasters/spreadsheets, and superseded revision renderings (polar Figure 7, the
strict-QA-based S4/S5 and old-pipeline S6, the mixed-branch S9 renderer).
