# GPP 2022 correction (October 2026)

The 2022 MOD17A2H GPP exports used originally were duplicated (`../audit/gpp_table_s8/mod17_export_hashes_2022_2024.csv`)
and two Natural-Forest months of 2023 had not been unit-converted. 2022 was re-acquired and every GPP-dependent result
was recomputed. Steps and scripts: `docs/REPRODUCIBILITY.md`, section "MOD17 GPP 2022 (resolved)".

| File | Content |
|---|---|
| `mod17_2022_rebuilt_verification.csv` | the 12 rebuilt 2022 months: all distinct, plausible California means |
| `mod17_2023_partial_validation.csv`, `mod17_2022_block_size_validation.csv` | agreement of the rebuild with the Earth Engine processing (monthly means within ~1 %; 8 x 8 block mean fits best) |
| `gpp_series_changed_rows.csv` | the 50 GPP land-type months that changed (48 in 2022, Natural Forests 2023-05/06), old vs new |
| `table_s8_part_a*.csv` | Supplementary Table S8a: FLUXCOM 2001–2021 (main, = Figure 5) vs FLUXCOM + MOD17 2022–2024 |
| `table_s8_part_b.csv` | Supplementary Table S8b: forecast skill in test months ≤ 2021 vs ≥ 2022 |
| `figure7_gpp_cells_old_vs_corrected.csv` | the 24 Figure 7 GPP cells before and after |
| `table3_model_summary.csv`, `table4_target_summary.csv` | manuscript Tables 3–4 |
| `Figure_S7_panel_*_source.csv` | Supplementary Figure S7 / Table S10 |
| `forecast_fix_checks.csv` | reproduction checks (all pass) |

Statistical comparison: `../forecasting/forecast_friedman_tests.csv`, `../forecasting/forecast_pairwise_wilcoxon_holm.csv`.

Main changes: GPP target mean R² 0.677 → 0.739 and nRMSE 0.535 → 0.491 (best model XGBoost instead of TCN); XGBoost
best-model counts 17/18 → 19/20, BiLSTM 7/7 → 8/7, TCN 6/4 → 3/3; Friedman R² χ²(5) = 55.03 (p = 1.29 × 10⁻¹⁰),
nRMSE χ²(5) = 58.21 (p = 2.85 × 10⁻¹¹); 12 instead of 9 Holm-significant pairs (R²: LSTM–BiLSTM no longer significant;
BiLSTM–N-HiTS, TCN–N-BEATS, TCN–N-HiTS, TCN–XGBoost now significant). Table S8a: NIRv–GPP r 0.66–0.92 vs NDVI–GPP
0.27–0.48 with MOD17 2022–2024; three nominal GPP-trend significance statements differ from the FLUXCOM-only analysis.
