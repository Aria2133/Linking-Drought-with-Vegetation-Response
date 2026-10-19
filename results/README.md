# Compact result tables (no raster data)

| File | Content | Produced by |
|---|---|---|
| `forecasting/figure7_final_task_metrics.csv` | current Figure 7 source: mean and SD of out-of-sample R² and nRMSE for 36 target × land-type tasks × 6 learned models, with the run configuration of each cell | final corrected forecasting tables (exported without local paths) |
| `forecasting/forecast_friedman_tests.csv` | Friedman tests (R², nRMSE) across the six learned models | `src/statistics/forecast_model_comparison.py` |
| `forecasting/forecast_pairwise_wilcoxon_holm.csv` | two-sided Wilcoxon signed-rank tests for the 15 model pairs per metric, Holm-corrected | `src/statistics/forecast_model_comparison.py` |
| `states/main_state_classification_deseasonalized.csv` | exact Figure 9 inputs: monthly EWBI, NIRv anomaly, median thresholds and class per land type | `src/ecological/states/01_main_ewbi_states_transitions.py` |
| `states/main_state_transition_probabilities_deseasonalized.csv` | observed month-to-month transition proportions (Figure 9) | same |
| `states/iaaft_null/state_transition_null_summary.csv` | observed diagonal self-transition proportions vs the IAAFT null (median, 95 % interval, empirical p) | `src/statistics/state_transition_iaaft_null.py` |

Transition tables are descriptive empirical proportions, not estimates of a fitted Markov model.
