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
- Final TWS common-benchmark correction (October 2026): four scripts read the final Figure 7 source
  (`AUDIT_TEMP/FIGURE7_TWS_COMMON_CORRECTION/source_tables/Figure_7_source.csv`, built by
  `src/tables/figure7_tws_common_benchmark_correction.py`) instead of the earlier source whose four BiLSTM-TWS cells came from
  the 18-month sensitivity. Only the input path changed in `src/statistics/forecast_inference_extended.py`,
  `src/figures/figS08_forecast_heatmaps.py` and `src/tables/verify_tables_2_4_5_6.py`. In
  `src/tables/forecast_pairwise_statistics_table.py` the input folder changed, its hard-coded Table 4 check now reads the
  recomputed Table 4, and two report sentences quote the new values. `src/figures/fig07_point_errorbar_best_counts.py` is
  unchanged; it was rerun on the final source.
- `role` — what the script does.

Written during the final correction, in the revision workspace, and published verbatim:
`src/tables/figure7_tws_common_benchmark_correction.py` and `src/tables/tables_4_5_from_corrected_figure7.py`.

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
