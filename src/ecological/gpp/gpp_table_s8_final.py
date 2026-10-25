"""Supplementary Table S8 part a with the re-acquired 2022 MOD17 GPP.

Main column: FLUXCOM-X-BASE GPP 2001-2021 (identical to Figure 5 for the four correlations). Sensitivity column: the same
series continued with MOD17A2H for 2022-2024 (2022 rebuilt from Planetary Computer, every MOD17 month converted from
kg C m-2 month-1 to g C m-2 d-1). Same series, joins and statistics as gpp_table_s8_correction.py (Pearson r with
nominal p; OLS trend with HAC-12 standard errors). Inputs are the corrected series written by
src/preprocessing/02_landtype_series/apply_rebuilt_mod17_2022_series.py.

Usage: python gpp_table_s8_final.py --series <corrected paper_standardized_anomaly_series.csv>
           --gndvi <corrected Figure_2_previous_GNDVI_with_test_monthly_series.csv> --ewbi <main_ewbi_deseasonalized_series.csv>
           --fig5 <figure5_raw_correlations.csv> --out <dir>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpp_table_s8_correction import COMPARATORS, DISPLAY, FULL, LANDS, MAIN, compute_land  # noqa: E402


def fmtp(p):
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def main():
    ap = argparse.ArgumentParser()
    for name in ("series", "gndvi", "ewbi", "fig5", "out"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    series = pd.read_csv(a.series, parse_dates=["date"])
    gnd = pd.read_csv(a.gndvi, parse_dates=["date"])
    series = pd.concat([series[series.variable.ne("GNDVI")], gnd[gnd.variable.eq("GNDVI")]], ignore_index=True)
    series["land_type"] = series["land_type"].replace({"NaturalForests": "Natural Forests"})
    ewbi = pd.read_csv(a.ewbi, parse_dates=["date"])
    ewbi["land_type"] = ewbi["land_type"].replace({"NaturalForests": "Natural Forests"})

    g = series[series.variable.eq("GPP")]
    g22 = g[g.date.dt.year == 2022]
    assert g22.groupby("land_type").original_value.nunique().eq(12).all(), "2022 GPP months must all differ"
    assert g[g.date.dt.year >= 2022].original_value.between(0.2, 12).all(), "MOD17 months must be in g C m-2 d-1"

    rows = []
    for land in LANDS:
        rows.extend(compute_land(series, ewbi, land, (MAIN, FULL)))
    res = pd.DataFrame(rows)
    res.to_csv(a.out / "table_s8_part_a_long.csv", index=False)
    assert (res[res.dataset == MAIN]["end"] == "2021-12").all() and (res[res.dataset == MAIN]["n_mod17_months"] == 0).all()
    assert (res[res.dataset == FULL]["n_2022_months"] == 12).all() and (res[res.dataset == FULL]["end"] == "2024-12").all()

    f5 = pd.read_csv(a.fig5)
    f5 = f5[(f5.variable_1.eq("GPP") ^ f5.variable_2.eq("GPP"))]
    f5["other"] = np.where(f5.variable_1.eq("GPP"), f5.variable_2, f5.variable_1)
    mr = res[(res.dataset == MAIN) & res.quantity.str.startswith("GPP vs")].copy()
    mr["other"] = mr.quantity.str.replace("GPP vs ", "").replace({"GOSIF/SIF": "GOSIF"})
    m = mr.merge(f5, on=["land_type", "other"], validate="one_to_one")
    assert len(m) == 16 and (m.estimate - m.pearson_r).abs().max() < 1e-12, "main column must equal Figure 5"

    order = [f"GPP vs {DISPLAY.get(c, c)}" for c in COMPARATORS] + ["Deseasonalized EWBI vs GPP anomaly",
                                                                    "GPP linear trend (gC m-2 d-1 per year, HAC-12)"]
    w = res.pivot_table(index=["land_type", "quantity"], columns="dataset", values=["estimate", "p_nominal", "n_months"],
                        aggfunc="first")
    tab, readable = [], []
    for land in LANDS:
        for q in order:
            r = w.loc[(land, q)]
            e1, e2, p1, p2 = r[("estimate", MAIN)], r[("estimate", FULL)], r[("p_nominal", MAIN)], r[("p_nominal", FULL)]
            changed = bool((p1 < 0.05) != (p2 < 0.05))
            tab.append({"land_type": land, "quantity": q, "fluxcom_only_2001_2021_estimate": e1, "fluxcom_only_p_nominal": p1,
                        "fluxcom_only_n": int(r[("n_months", MAIN)]), "with_mod17_2022_2024_estimate": e2,
                        "with_mod17_p_nominal": p2, "with_mod17_n": int(r[("n_months", FULL)]),
                        "difference_with_mod17_minus_fluxcom_only": e2 - e1, "nominal_p_lt_0_05_status_changed": changed})
            readable.append({"Land type": land, "Quantity": q, "FLUXCOM only, 2001–2021 (main)": f"{e1:.3f} (p = {fmtp(p1)})",
                             "FLUXCOM 2001–2021 + MOD17 2022–2024": f"{e2:.3f} (p = {fmtp(p2)})",
                             "Difference": f"{e2 - e1:+.3f}", "Nominal p < 0.05 status changed": "Yes" if changed else "No"})
    pd.DataFrame(tab).to_csv(a.out / "table_s8_part_a.csv", index=False)
    pd.DataFrame(readable).to_csv(a.out / "table_s8_part_a_readable.csv", index=False, encoding="utf-8-sig")
    t = pd.DataFrame(tab)
    corr = t[t.quantity.str.startswith("GPP vs")]
    print("changed flags:", t[t.nominal_p_lt_0_05_status_changed][["land_type", "quantity"]].values.tolist())
    for col in ("fluxcom_only_2001_2021_estimate", "with_mod17_2022_2024_estimate"):
        nirv = corr[corr.quantity == "GPP vs NIRv"][col]
        ndvi = corr[corr.quantity == "GPP vs NDVI"][col]
        print(f"{col}: NIRv {nirv.min():.3f}-{nirv.max():.3f}; NDVI {ndvi.min():.3f}-{ndvi.max():.3f}")


if __name__ == "__main__":
    main()
