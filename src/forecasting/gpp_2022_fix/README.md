# GPP 2022 forecasting rerun drivers

These drivers rerun target GPP, folds 3–4, inside sandbox copies of the published run packages
(`Docs/major_revision_analyses/05_forecasting_raster_reviewer_safe_candidate/scripts`,
`corrected_true_BiLSTM_rerun_20260728/scripts/run_corrected_true_bilstm_rerun.py`,
`SVR_full_benchmark_supplementary_candidate_20260723_204013/scripts`). In each copy only the output root and the task cache
were redirected: the cache is the published one with the 12 GPP rows of 2022 replaced
(`src/preprocessing/01_landtype_masking/apply_rebuilt_mod17_2022.py`). Settings, seeds and the preprocessing boundary are the
published ones, so the reruns are directly comparable with the unchanged tasks.

| Driver | Placed next to | Runs |
|---|---|---|
| `run_gpp_2022_fix.py` | reviewer-safe `scripts/` | baselines, XGBoost, LSTM, N-BEATS (seeds 42/123/2026), TCN, N-HiTS (seed 42) |
| `run_bilstm_gpp_2022_fix.py` | copy of the corrected-BiLSTM script (`bilstm_module.py`) | corrected true BiLSTM, seed 42 |
| `run_svr_gpp_2022_fix.py` | SVR package root | RBF-ε-SVR validation grid, selection and refit (+ fold 1 Grasslands as a reproduction check) |

Reproduction before the rerun (original cache): XGBoost seed 123, LSTM seed 42, BiLSTM seed 42 (Grasslands fold 3) and SVR
fold 1 reproduce the published metrics exactly. Merging and checks: `src/tables/apply_gpp_2022_forecast_fix.py`.
N-BEATS seeds 123/2026 were rerun once after a CUDA out-of-memory failure caused by a concurrent job; the rerun of the one
task that had completed reproduced it exactly.
