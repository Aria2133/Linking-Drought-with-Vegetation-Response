# Final pre-merge audit

| Item | Verdict |
|---|---|
| All 216 Figure 7 cells use the common 12-month benchmark | PASS |
| 0 Figure 7 cells come from the 18-month TWS sensitivity | PASS |
| 18-month TWS documented only as supplementary sensitivity | PASS |
| True BiLSTM: bidirectional first layer for all 9 targets | PASS |
| Figure 7 source, repository table, plotted values, Tables 4-5 and supplementary shared cells (S7, S8, S9) agree | PASS |
| Inference reruns successfully (Friedman R2 chi2(5) = 37.079, nRMSE chi2(5) = 40.889; same 9 Holm-significant pairs) | PASS |
| No conclusions changed unexpectedly (best-model counts, predictability classes, significance pattern unchanged) | PASS |
| No scientific datasets added | PASS |
| No unrelated analysis changed | PASS |
| IAAFT script reproduces accepted null outcome | PASS |
| Exact floor-based pixel completeness wording | PASS |
| Accepted previous/no-QA GNDVI lineage | PASS |
| Figure 9 proportion terminology | PASS |
| LST QA resolved or explicitly flagged | PASS |
| GOSIF/FLUXCOM metadata resolved | PASS |
| GRACE grid/export/effective-support distinction | PASS |
| SPEI-1 land-type masking script | NOT RECONSTRUCTED (inputs not available locally; documented as missing) |

Expected numerical changes for the manuscript: BiLSTM mean R2 0.669 -> 0.661 and mean nRMSE 0.532 -> 0.538 (Table 4);
TWS mean R2 0.309 -> 0.296 and mean nRMSE 0.783 -> 0.792 (Table 5 and Results text). Details:
[`REPOSITORY_QC_REPORT.md`](REPOSITORY_QC_REPORT.md).
