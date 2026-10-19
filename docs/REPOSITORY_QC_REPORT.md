# Repository QC report (final pre-submission consistency correction, 2026-10-19)

Scope: one numerical correction plus documentation. The four BiLSTM × TWS cells of the Figure 7 source now come from the
common 12-month benchmark; earlier they came from the 18-month supplementary sensitivity. The downstream tables, statistics and
documentation were propagated from that source. No model was retrained and no scientific dataset was added. All 23 automated
checks pass (`FINAL_CONSISTENCY_CHECKS.csv` in the revision workspace, `AUDIT_TEMP/FIGURE7_TWS_COMMON_CORRECTION/`).

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | All 216 Figure 7 cells use the common 12-month benchmark | **Pass.** 0 of 216 cells come from the 18-month sensitivity. The four BiLSTM-TWS cells are the mean and SD (ddof = 1) of the 16 existing common-run fold rows (4 land types × 4 folds, seed 42, lookback 12). The other 212 cells are byte-identical to the accepted source | `results/forecasting/figure7_final_task_metrics.csv`; `src/tables/figure7_tws_common_benchmark_correction.py` |
| 2 | 18-month TWS = sensitivity only | **Pass.** Labelled "Supplementary hydroclimate sensitivity; not used in the main common 12-month Figure 7 benchmark" in `config/forecasting_hyperparameters.csv`, `config/random_seeds.yaml`, `WORKFLOW.md` and `REPRODUCIBILITY.md`; no file presents it as main | — |
| 3 | True BiLSTM: first layer bidirectional for all 9 targets | **Pass.** Seven targets: `corrected_bilstm/run_corrected_true_bilstm.py` (`bidirectional=True`, 128 units). SPEI-1 (200) and TWS (150): `benchmark/model_core.py` (`bidirectional = variable in {"SPEI", "TWS"}`), used by the common TWS run and the SPEI-1 no-detrend run | code; hyperparameter table |
| 4 | Figure 7 source, plot, tables and supplementary shared cells agree | **Pass.** Repository table = source (max difference 0). Re-rendered Figure 7 plot values = source (0), and all 10 renderer checks pass. Figure S8 learned cells = source (432 cells, max difference 0; 8 BiLSTM-TWS cells changed). Figure S7 BiLSTM-TWS rows are the same 16 common 12-month rows. Figure S9 BiLSTM-TWS uses the common 12-month predictions. S7 and S9 are unchanged | consistency checks 7–12 |
| 5 | Inference reruns | **Pass.** `forecast_model_comparison.py`: R² χ²(5) = 37.079, p = 5.77 × 10⁻⁷; nRMSE χ²(5) = 40.889, p = 9.88 × 10⁻⁸ (file identical). The same 9 Holm-significant pairs (R²: LSTM–BiLSTM, LSTM–TCN, BiLSTM–TCN; nRMSE: these three plus TCN–N-BEATS, TCN–N-HiTS, TCN–XGBoost). One non-significant row changed: nRMSE BiLSTM–N-HiTS, W 211 → 215, p 0.056 → 0.064, Holm p 0.278 → 0.322 | `results/forecasting/forecast_*.csv` |
| 6 | No conclusions changed unexpectedly | **Pass.** Best-model counts unchanged (XGBoost 17/18, BiLSTM 7/7, TCN 6/4, N-BEATS 2/4, LSTM 4/3, N-HiTS 0/0). BiLSTM still wins all four TWS tasks. Predictability classes and best models per target unchanged; Friedman mean ranks unchanged | consistency checks 16–20 |
| 7 | Changed values (expected) | BiLSTM mean R² 0.669 → 0.661 and mean nRMSE 0.532 → 0.538 (Table 4); TWS target mean R² 0.309 → 0.296 and nRMSE 0.783 → 0.792 (Table 5) | `src/tables/tables_4_5_from_corrected_figure7.py` |
| 8 | No datasets added | **Pass.** Changes are text/CSV only; no raster, NetCDF, image, model or archive file | — |
| 9 | No unrelated analysis changed | **Pass.** Edits are limited to the Figure 7 source export, its four consumers (inference, S8, table verification, pairwise table) and documentation. No project, data or manuscript file outside the revision workspace changed (integrity check: 0 changed or removed; all added files are Git objects) | `FILE_PROVENANCE.csv` |
| 10 | IAAFT, pixel wording, GNDVI lineage, Figure 9 terminology, LST QA, GOSIF/FLUXCOM, GRACE | Unchanged from the previous pass (all pass) | previous QC (commit `bcad09d`) |
| — | License / data availability | MIT for code and documentation; datasets excluded. Five-category data-availability statement | `LICENSE`, `DATA_AVAILABILITY.md` |

## SPEI-1 land-type masking script
Not reconstructed. The gridMET Earth Engine export (`gee/scripts/gridmet_spei1_monthly_5km.js`) writes California-wide
`SPEI1_GRIDMET_CA_YYYY_MM` files to the Drive folder `SPEI1_GRIDMET_TIFs`, and those files are not stored locally. The only
local California-wide SPEI-1 files are CSIC products for 2022–2024. A reconstructed masking script therefore cannot be
validated against the final land-type products (dates, grid, masks, valid-pixel counts, values, hashes). The missing helper
stays documented in `WORKFLOW.md` and `REPRODUCIBILITY.md`.

## Manuscript values that must be updated (not changed here)
- **Table 4, BiLSTM:** mean R² 0.669 → **0.661**; mean nRMSE 0.532 → **0.538** (counts 7/7 unchanged).
- **Table 5, TWS:** mean R² 0.309 → **0.296**; mean nRMSE 0.783 → **0.792** (class and best models unchanged).
- **Results text (Table 5 paragraph):** "mean R² values of 0.124 and 0.309" → "0.124 and **0.296**".
- **Figure 7:** replace with the re-rendered figure. The BiLSTM-TWS cells, mean ± SD: Grasslands R² 0.761 ± 0.065 and nRMSE 0.485 ± 0.067; Natural Forests R² 0.367 ± 0.230 and nRMSE 0.786 ± 0.140; Savannas R² 0.583 ± 0.089 and nRMSE 0.643 ± 0.067; Shrublands R² 0.442 ± 0.259 and nRMSE 0.733 ± 0.167.
- **Figure S8:** re-export. The same 8 BiLSTM-TWS cells change.
- **Supplementary pairwise table:** nRMSE BiLSTM vs N-HiTS row: W 215, p = 0.064, Holm p = 0.322, rank-biserial r = 0.354 (still not significant).

## Earlier findings that still require manuscript corrections (not made here)
1. **LST quality control.** The code keeps pixels with `QC_Day` bits 0–1 < 2. It drops a daily scene when more than 30 % of
   California pixels with a valid `LST_Day` value fail that QA test. This is not a cloud-cover statistic. The analysed LST
   record is 2003-01 to 2024-12 (2009-10 missing).
2. **FLUXCOM.** FLUXCOM-X-BASE monthly GPP at 0.5°, not 0.05°.
3. **GOSIF.** GOSIF v2 monthly composites (0.05°), not 8-day.
4. **GNDVI period.** 2001–2024, not 2000–2025.
5. **GRACE.** Also state the ~3° effective mascon support and the 100 km export scale.

## Other open items
- The script that masked gridMET SPEI-1 to land types was not preserved (see above).
- The Figure-label cleanup of Figure 5 and two supplementary figures (Oct 2026) was applied without a saved script (labels only).
- Scripts written during the 2026 revision read their inputs relative to the authors' revision workspace. The two statistical
  scripts run from the bundled `results/` tables.
