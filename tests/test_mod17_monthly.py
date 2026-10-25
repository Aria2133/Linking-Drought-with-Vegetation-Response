"""MOD17 monthly conversion is deterministic and duplicated monthly exports are rejected."""
import importlib.util
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "src" / "ecological" / "gpp" / "mod17_monthly.py"
spec = importlib.util.spec_from_file_location("mod17", PATH)
mod17 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod17)


def test_conversion_ignores_magnitude():
    # Natural Forests 2023-05/06 values the old median<0.2 rule left unconverted
    assert np.isclose(mod17.kgc_month_to_gc_day(0.236101, 2023, 5), 7.616, atol=1e-3)
    assert np.isclose(mod17.kgc_month_to_gc_day(0.232394, 2023, 6), 7.746, atol=1e-3)
    assert np.isclose(mod17.kgc_month_to_gc_day(0.05, 2024, 2), 50 / 29)  # leap February
    assert np.isclose(mod17.kgc_month_to_gc_day(0.05, 2023, 2), 50 / 28)


def synthetic(dup=()):
    rng = np.random.default_rng(1)
    arrays = {m: 0.05 + 0.1 * np.sin(m / 2) ** 2 + rng.random((4, 5)) * 0.01 for m in range(1, 13)}
    hashes = {m: f"h{m}" for m in arrays}
    for m in dup:
        arrays[m], hashes[m] = arrays[1].copy(), hashes[1]
    return arrays, hashes


def test_valid_year_passes():
    report = mod17.check_months(*synthetic(), 2023)
    assert (report.problems == "").all() and len(report) == 12


def test_duplicated_months_fail():
    report = mod17.check_months(*synthetic(dup=(2, 3, 5)), 2022)
    bad = set(report.loc[report.problems != "", "month"])
    assert bad == {1, 2, 3, 5}


def test_missing_and_unconverted_scale_fail():
    arrays, hashes = synthetic()
    del arrays[12], hashes[12]
    arrays[4] = arrays[4] * 100  # values far outside a monthly kg C total
    report = mod17.check_months(arrays, hashes, 2022).set_index("month")
    assert report.loc[12, "problems"] == "missing" and "implausible" in report.loc[4, "problems"]
