# Repository QC report (pre-merge consistency pass, 2026-10-19)

Scope: documentation and provenance only. No model was retrained, no scientific dataset was added, and no reported
numerical value was changed. Verification scripts were run read-only against the authors' project tables.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Main TWS benchmark = common 12-month branch | **Open — author decision.** The common 12-month BiLSTM-TWS run exists and is documented as the main configuration, but the current Figure 7 source table takes the four BiLSTM-TWS cells from the 18-month supplementary configuration | `results/forecasting/figure7_final_task_metrics.csv` (`selected_configuration = bilstm_special_tws_state_space`, 4 rows); comparison table in `REPRODUCIBILITY.md` |
| 2 | 18-month TWS = sensitivity only | Labelled "Supplementary sensitivity" in `config/forecasting_hyperparameters.csv`, `WORKFLOW.md`, `REPRODUCIBILITY.md`; still feeds 4 Figure 7 cells (see 1) | as above |
| 3 | True BiLSTM: first layer bidirectional for all targets | Pass. Seven targets: `corrected_bilstm/run_corrected_true_bilstm.py` (`bidirectional=True`, 128 units); SPEI-1 (200) and TWS (150) runs use a bidirectional first layer. No public document presents the superseded implementation as final | code lines `CorrectedTrueBiLSTM.__init__`; hyperparameter table |
| 4 | Model-comparison script reproduces the accepted inference | Pass. `forecast_model_comparison.py`: R² χ²(5) = 37.079, p = 5.77 × 10⁻⁷; nRMSE χ²(5) = 40.889, p = 9.88 × 10⁻⁸; all 30 Wilcoxon statistics, raw and Holm p-values identical to the accepted analysis (max difference 0); 9 Holm-significant pairs | `results/forecasting/forecast_*.csv` |
| 5 | IAAFT script reproduces the accepted null outcome | Pass. 500 pairs per land type, seed 20261013: 4 of 16 diagonal proportions exceed the null at upper-tail p < 0.05; summary identical to the accepted output (max difference 0) | `results/states/iaaft_null/state_transition_null_summary.csv` |
| 6 | Floor-based pixel completeness wording | Pass. `max(1, floor(0.30 × N))`: 75 of 252 pre-2022 months, 3 of 12 months in 2024 | `config/pixel_selection_criteria.md`, `pixel_selection_counts.csv` |
| 7 | Accepted previous/no-QA GNDVI lineage | Pass. Final analyses and harmonised S4/S5 use previous/no-QA GNDVI; one stale caption string corrected (Figures 3–4); remaining QA-GNDVI code paths listed as not used | `REPRODUCIBILITY.md` table of unused code paths |
| 8 | Current Figure 7 renderer / source | Pass (with item 1). `src/figures/fig07_point_errorbar_best_counts.py` ← `results/forecasting/figure7_final_task_metrics.csv`; polar renderer removed | README figure map |
| 9 | Figure 9 proportion terminology | Pass in the repository: colorbar label is "Observed transition proportion (%)"; docs use "observed transition proportions" and state that matrices are not a fitted Markov model. The rendered Figure 9 file must be re-exported for the label to appear in the manuscript | `src/figures/fig09_transition_matrices.py` |
| 10 | LST QA | Resolved from code; manuscript wording needs correction (see below) | `gee/README.md`, `data_access/data_sources.csv` |
| 11 | GOSIF / FLUXCOM metadata | Resolved from the files; manuscript Table 1 needs correction (see below) | `data_access/data_sources.csv` |
| 12 | GRACE grid / export / effective support | Pass: source grid 0.5°, effective mascon support ~3°, export at 100 km (0.898° analysed grid) documented separately | `data_sources.csv`, `gee/README.md` |
| 13 | No scientific datasets added | Pass. Repository 2.1 MB; no raster, NetCDF, image, model or archive files; `.gitignore` blocks them | — |
| 14 | No final numerical manuscript values changed | Pass. Only documentation, labels, path placeholders and provenance were edited | `FILE_PROVENANCE.csv` (`path_check`) |
| — | License | MIT for code/documentation; datasets explicitly excluded | `LICENSE` |
| — | Data availability | Five-category statement (public source, raw source, processed intermediate, derived spatial, code) | `DATA_AVAILABILITY.md` |

## Findings that require manuscript corrections (not made here)
1. **LST quality control.** The code (`gee/scripts/modis_myd21c1_lst_monthly.js`) keeps pixels with `QC_Day` bits 0–1 < 2
   and drops a daily scene when more than 30 % of California pixels with a valid `LST_Day` value fail that QA test. This is not a
   cloud-cover statistic; the sentence "high-quality retrievals where cloud cover < 30 %" should be reworded accordingly.
   The analysed LST record is **2003-01 to 2024-12** (2009-10 missing); 2001–2002 are filled values in the ecological series only.
2. **FLUXCOM.** The analysed product is FLUXCOM-X-BASE monthly GPP at **0.5°** (`GPP_YYYY_050_monthly.nc`, created 2023-11-10),
   not 0.05° as stated in Table 1 and Section 2.2.2.
3. **GOSIF.** The analysed files are GOSIF v2 **monthly** composites (0.05°); Table 1 lists 8-day.
4. **GNDVI period.** Table 1 lists 2000–2025; the analyses use 2001–2024.
5. **GRACE.** The manuscript gives ~0.5° only; it should also state the ~3° effective mascon support and the 100 km export scale.

## Other open items
- TWS × BiLSTM provenance (item 1).
- The script that masked gridMET SPEI-1 to land types was not preserved.
- Figure-label cleanup of Figure 5 and two supplementary figures (Oct 2026) was applied without a saved script (labels only).
- Scripts written during the 2026 revision read inputs relative to the authors' revision workspace; the two statistical scripts
  were made runnable from the bundled `results/` tables.
