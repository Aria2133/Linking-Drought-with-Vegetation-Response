"""Synthetic-array tests of the rolling-origin preprocessing boundaries (no data, no training)."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "forecasting"
sys.path.insert(0, str(SRC))
from fold_boundaries import FIT_SUBSET, LEGACY, rolling_fold_specs, window_usage  # noqa: E402

FOLDS = [
    {"fold": 1, "train_end": 201612, "test_start": 201701, "test_end": 201812},
    {"fold": 2, "train_end": 201812, "test_start": 201901, "test_end": 202012},
    {"fold": 3, "train_end": 202012, "test_start": 202101, "test_end": 202212},
    {"fold": 4, "train_end": 202212, "test_start": 202301, "test_end": 202412},
]


@pytest.fixture(scope="module")
def rp():
    """raster_pipeline with a stub candidate_config (the real one points at <PROJECT_ROOT> placeholders)."""
    stub = types.ModuleType("candidate_config")
    stub.CACHE_DIR = Path(".")
    stub.HISTORY_ROOT = Path(".")
    stub.TEST_ROOT = Path(".")
    stub.LAND_CONFIG = {}
    stub.MAIN_LOOKBACK = 12
    stub.MAX_PIXELS = 100
    stub.MIN_VALID_FRACTION = 0.30
    stub.TARGET_CLIP = 4.0
    saved = sys.modules.get("candidate_config")
    sys.modules["candidate_config"] = stub
    sys.path.insert(0, str(SRC / "benchmark"))
    sys.modules.pop("raster_pipeline", None)
    import raster_pipeline
    yield raster_pipeline
    sys.modules.pop("raster_pipeline", None)
    if saved is not None:
        sys.modules["candidate_config"] = saved
    else:
        sys.modules.pop("candidate_config", None)


def months(start=200101, end=202412):
    out, y, m = [], start // 100, start % 100
    while y * 100 + m <= end:
        out.append(y * 100 + m)
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return np.array(out, dtype=np.int32)


def test_fold_windows_published_and_fit_subset():
    legacy = rolling_fold_specs(FOLDS, LEGACY)
    strict = rolling_fold_specs(FOLDS, FIT_SUBSET)
    assert [s["train_end"] for s in legacy] == [201412, 201612, 201812, 202012]
    assert [s["validation_start"] for s in legacy] == [201501, 201701, 201901, 202101]
    assert [s["preprocessing_fit_end"] for s in legacy] == [201612, 201812, 202012, 202212]  # published behaviour
    assert [s["preprocessing_fit_end"] for s in strict] == [201412, 201612, 201812, 202012]
    assert all(s["refit_preprocessing_fit_end"] == s["validation_end"] for s in strict)
    assert all(s["preprocessing_fit_end"] < s["test_start"] for s in legacy + strict)
    assert legacy[0]["preprocessing_mode"] == "reviewer_mode_b"
    assert strict[0]["preprocessing_mode"] == "reviewer_mode_b_fitsubset"
    assert "never fitted" in window_usage(strict[0])["outer_test_scoring"]


def test_unknown_boundary_rejected():
    with pytest.raises(ValueError):
        rolling_fold_specs(FOLDS, "through_test")


def _pipeline(rp, values, dates, fit_end):
    v = rp.impute_calendar_month(values, dates, train_end=fit_end, legacy_all_periods=False)
    v, trends = rp.detrend_by_calendar_month(v, dates, fit_end)
    v, params = rp.scale_values(v, dates, fit_end, scaling="monthwise_z")
    return v, trends, params


def _synthetic(seed=0):
    dates = months()
    rng = np.random.default_rng(seed)
    t = np.arange(len(dates))
    values = (np.sin(2 * np.pi * (dates % 100) / 12)[:, None] + 0.01 * t[:, None] + rng.normal(0, 0.3, (len(dates), 5))).astype(np.float32)
    values[rng.random(values.shape) < 0.05] = np.nan
    return dates, values


@pytest.mark.parametrize("fold_index", [0, 3])
def test_fit_subset_ignores_validation_and_test_values(rp, fold_index):
    dates, values = _synthetic()
    spec = rolling_fold_specs(FOLDS, FIT_SUBSET)[fold_index]
    base, trends, params = _pipeline(rp, values, dates, spec["preprocessing_fit_end"])
    perturbed = values.copy()
    later = dates >= spec["validation_start"]          # internal validation + outer test + beyond
    perturbed[later] = perturbed[later] * 5 + 100
    out, trends2, params2 = _pipeline(rp, perturbed, dates, spec["preprocessing_fit_end"])
    assert trends == trends2 and params == params2
    train = dates <= spec["train_end"]
    np.testing.assert_array_equal(np.nan_to_num(base[train]), np.nan_to_num(out[train]))


def test_legacy_boundary_uses_validation_but_never_test(rp):
    dates, values = _synthetic(1)
    spec = rolling_fold_specs(FOLDS, LEGACY)[0]
    _, trends, params = _pipeline(rp, values, dates, spec["preprocessing_fit_end"])
    val = (dates >= spec["validation_start"]) & (dates <= spec["validation_end"])
    pv = values.copy()
    pv[val] += 50
    _, trends_v, params_v = _pipeline(rp, pv, dates, spec["preprocessing_fit_end"])
    assert params_v != params  # internal-validation contamination of the published runs
    pt = values.copy()
    test = dates >= spec["test_start"]
    pt[test] += 50
    _, trends_t, params_t = _pipeline(rp, pt, dates, spec["preprocessing_fit_end"])
    assert params_t == params and trends_t == trends  # no outer-test leakage


def test_gpp_unit_conversion_is_fixed_and_not_data_dependent(rp):
    dates, values = _synthetic(2)
    a, fa, _ = rp.apply_unit_correction(values, dates, "GPP", "reviewer_mode_b")
    b, fb, _ = rp.apply_unit_correction(values * 7 + 3, dates, "GPP", "reviewer_mode_b")
    assert fa == fb == 31.56
    late = dates >= 202201
    np.testing.assert_allclose(a[late], values[late] * 31.56)
    np.testing.assert_array_equal(a[~late], values[~late])


def test_legacy_boundary_matches_published_fold_table():
    import pandas as pd
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "config" / "rolling_origin_folds.csv")
    for spec, (_, row) in zip(rolling_fold_specs(FOLDS, LEGACY), table.iterrows()):
        assert spec["train_end"] == row["core_model_training_end"]
        assert spec["validation_start"] == row["internal_validation_start"]
        assert spec["validation_end"] == row["internal_validation_end"]
        assert spec["preprocessing_fit_end"] == row["preprocessing_fit_end"]
        assert (spec["test_start"], spec["test_end"]) == (row["test_start"], row["test_end"])
