"""Synthetic tests of the Table S8 correction logic (period selection, explicit joins, difference sign, flags)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "src" / "ecological" / "gpp" / "gpp_table_s8_correction.py"
spec = importlib.util.spec_from_file_location("gpp_s8", PATH)
gpp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpp)


def monthly(start="2001-01-01", end="2024-12-01"):
    return pd.date_range(start, end, freq="MS")


def test_period_selection():
    f = pd.DataFrame({"x": 1.0}, index=monthly())
    assert gpp.select(f, gpp.MAIN).index.max() == pd.Timestamp("2021-12-01") and len(gpp.select(f, gpp.MAIN)) == 252
    sens = gpp.select(f, gpp.SENS)
    assert len(sens) == 276 and not (sens.index.year == 2022).any() and sens.index.max() == pd.Timestamp("2024-12-01")
    assert len(gpp.select(f, gpp.OLD)) == 288


def test_unique_monthly_rejects_duplicates_and_mid_month_dates():
    d = pd.DataFrame({"date": [pd.Timestamp("2001-01-01"), pd.Timestamp("2001-01-01")], "v": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        gpp.unique_monthly(d, "v")
    with pytest.raises(ValueError, match="month starts"):
        gpp.unique_monthly(pd.DataFrame({"date": [pd.Timestamp("2001-01-15")], "v": [1]}), "v")


def synthetic_series(seed=0):
    rng = np.random.default_rng(seed)
    dates = monthly()
    rows = []
    base = np.sin(2 * np.pi * dates.month / 12)
    gppv = 2 + base + rng.normal(0, 0.2, len(dates))
    for var, vals in {"GPP": gppv, "NIRv": 0.3 + 0.1 * base + rng.normal(0, 0.03, len(dates)),
                      "NDVI": 0.5 + rng.normal(0, 0.05, len(dates)), "GNDVI": 0.4 + rng.normal(0, 0.05, len(dates)),
                      "GOSIF": gppv * 0.1 + rng.normal(0, 0.01, len(dates))}.items():
        for d, v in zip(dates, vals):
            rows.append({"date": d, "land_type": "Grasslands", "variable": var, "original_value": v, "standardized_anomaly": v - 2})
    s = pd.DataFrame(rows)
    nirv_dates = s[s.variable.eq("NIRv")].date
    s = s.drop(s[(s.variable.eq("NIRv")) & (s.date.isin(nirv_dates.iloc[[5, 6]]))].index)  # gaps -> explicit join
    ewbi = pd.DataFrame({"date": dates, "land_type": "Grasslands", "ewbi": rng.normal(0, 1, len(dates))})
    return s, ewbi


def test_compute_land_joins_and_counts():
    s, ewbi = synthetic_series()
    rows = pd.DataFrame(gpp.compute_land(s, ewbi, "Grasslands", (gpp.MAIN, gpp.SENS)))
    nirv = rows[rows.quantity.eq("GPP vs NIRv")].set_index("dataset")
    assert nirv.loc[gpp.MAIN, "n_months"] == 250 and nirv.loc[gpp.SENS, "n_months"] == 274  # two NIRv gaps dropped by the join
    assert (rows[rows.dataset == gpp.SENS]["n_2022_months"] == 0).all()
    main = rows[rows.dataset == gpp.MAIN].set_index("quantity")["estimate"]
    sens = rows[rows.dataset == gpp.SENS].set_index("quantity")["estimate"]
    diff = sens - main
    assert np.allclose(diff, rows[rows.dataset == gpp.SENS].set_index("quantity")["estimate"] - main)
    p = rows.pivot(index="quantity", columns="dataset", values="p_nominal")
    flags = (p[gpp.MAIN] < 0.05) != (p[gpp.SENS] < 0.05)
    assert flags.dtype == bool
