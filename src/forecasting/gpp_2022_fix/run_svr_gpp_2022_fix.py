"""GPP 2022 fix: RBF-epsilon-SVR for GPP with the published validation-only grid search, timeouts, selection rule and
refit, using the corrected cache. Tasks: folds 3-4 (all land types) plus fold 1 Grasslands as a reproduction check
(fold 1 never reads 2022, so it must equal the published row)."""
import importlib.util, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "scripts"))
import candidate_config as cfg
import run_helpers
run_helpers.ROLLING_FOLDS = [f for f in cfg.ROLLING_FOLDS if f["fold"] in (1, 3, 4)]
spec = importlib.util.spec_from_file_location("svrrun", HERE / "scripts" / "run_full_svr_benchmark_and_supplement.py")
R = importlib.util.module_from_spec(spec); sys.modules["svrrun"] = R; spec.loader.exec_module(R)
R.split_specs = run_helpers.split_specs
R.TARGET_ORDER = ["GPP"]
expected = R.expected_validation_candidates()
expected = expected[(expected.fold.isin([3, 4])) | ((expected.fold == 1) & (expected.land_type == "GRASSLANDS"))]
print("candidates", len(expected), flush=True)
results = R.run_validation_candidates(expected, max_workers=12)
selected = R.select_validation_configs(results)
final = R.run_final_refits(selected, max_workers=9)
print(final[["target", "land_type", "fold", "status", "outer_test_r2", "outer_test_nrmse", "C", "epsilon", "gamma_multiplier"]].to_string())
