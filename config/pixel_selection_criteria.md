# Forecasting pixel selection

Code: `src/forecasting/benchmark/raster_pipeline.py` (`_coverage_mask`; selection block using `numpy.random.default_rng(42)`);
constants in `src/forecasting/benchmark/candidate_config.py` (`MIN_VALID_FRACTION = 0.30`, `MAX_PIXELS = 100`).

1. **Unit.** Screening is done separately for each target × land-type task (9 targets × 4 land types = 36 tasks), on that
   target's land-type-masked monthly rasters.
2. **Completeness.** A nominal 30% completeness criterion was implemented as a minimum finite-observation count of
   floor(0.30 × N), with a minimum of one observation:
   `required = max(1, floor(0.30 × N))`. A pixel is eligible only if it meets this count in both windows:
   - pre-2022 source record (2001-01 to 2021-12): N = 252 months, so at least **75** finite months;
   - 2024: N = 12 months, so at least **3** finite months.
3. **Selection.** If more than 100 pixels are eligible, 100 are drawn without replacement with `numpy.random.default_rng(42)`;
   otherwise all eligible pixels are kept. Selection is random, not spatially stratified.
4. **Reuse.** The selected pixels of a task are the same for every model and every rolling-origin fold.
5. **Note.** The 2024 condition uses months inside the fold-4 test window. It controls data availability only (no target
   values enter model fitting), but it is not fold-specific.
6. **TWS.** GRACE/GRACE-FO cells are coarse, so only 5–14 TWS pixels are eligible per land type and all are retained.

Exact eligible/retained counts for all 36 tasks: [`pixel_selection_counts.csv`](pixel_selection_counts.csv).
