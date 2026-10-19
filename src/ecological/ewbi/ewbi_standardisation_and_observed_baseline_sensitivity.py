"""Lightweight what-if checks on stored land-type tables (audit only; does not alter the pipeline).
V0 = submitted Fig 9 EWBI (deseasonalise by 2001-2011 calendar-month mean, then z with ONE baseline mean/SD).
V1 = manuscript Eq. (3) style: per-calendar-month baseline mean AND SD (temporal).
V2 = V0 but TWS/LST standardised on OBSERVED baseline months only (filled 2001-2002 months excluded).
R  = reviewer's hypothesis: spatial z across pixels of the same land type/month, then land-type mean.
     For any pixel field x_p, mean_p((x_p - mean_p x)/sd_p x) = 0 exactly, so R == 0 for every month by algebra.
"""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
FIX = r"<PROJECT_ROOT>\Docs\major_revision_analyses\04_detrending_sensitivity_fixed\tables"
B0, B1 = "2001-01-01", "2011-12-01"
LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
COMP = [("SPEI", 1), ("TWS", 1), ("VPD", -1), ("LST", -1)]
raw = pd.read_csv(os.path.join(FIX, "monthly_landtype_series_raw.csv"), parse_dates=["date"])
st = pd.read_csv(os.path.join(FIX, "main_state_classification_deseasonalized.csv"), parse_dates=["date"])
trans_stored = pd.read_csv(os.path.join(FIX, "main_state_transition_probabilities_deseasonalized.csv"))


def classify(e, n):
    et, nt = e.median(), n.median()
    lab = np.where(e >= et, np.where(n >= nt, "Wet / Healthy", "Wet / Suppressed"),
                   np.where(n >= nt, "Dry / Resilient", "Dry / Stressed"))
    return pd.Series(lab, index=e.index)


def diag_persist(lab):
    a, b = lab.iloc[:-1].to_numpy(), lab.iloc[1:].to_numpy()
    out = {}
    for s in ["Wet / Healthy", "Wet / Suppressed", "Dry / Resilient", "Dry / Stressed"]:
        m = a == s
        out[s] = 100 * float((b[m] == s).mean()) if m.any() else np.nan
    return out


rows, weights = [], []
for land in LANDS:
    r = raw[raw.land_type == land]
    w = r.pivot(index="date", columns="variable", values="analysis_value").sort_index()
    obs = r.pivot(index="date", columns="variable", values="raw_value").sort_index().notna()
    z0, z1, z2 = {}, {}, {}
    for v, s in COMP:
        x = w[v]; b = x.loc[B0:B1]
        clim = b.groupby(b.index.month).mean(); csd = b.groupby(b.index.month).std(ddof=1)
        d = x - x.index.month.map(clim).to_numpy()
        db = d.loc[B0:B1]
        z0[v] = s * (d - db.mean()) / db.std(ddof=1)
        z1[v] = s * d / x.index.month.map(csd).to_numpy()
        ob = obs[v].loc[B0:B1]
        if v in ("TWS", "LST"):
            bo = x.loc[B0:B1][ob]
            clim_o = bo.groupby(bo.index.month).mean()
            d2 = x - x.index.month.map(clim_o).to_numpy()
            d2b = d2.loc[B0:B1][ob]
            z2[v] = s * (d2 - d2b.mean()) / d2b.std(ddof=1)
        else:
            z2[v] = z0[v]
    Z0, Z1, Z2 = pd.DataFrame(z0), pd.DataFrame(z1), pd.DataFrame(z2)
    e0, e1, e2 = Z0.mean(axis=1), Z1.mean(axis=1), Z2.mean(axis=1)
    stored = st[st.land_type == land].set_index("date")
    nirv = stored["nirv_deseasonalized_anomaly"].reindex(e0.index)
    l0 = classify(e0, nirv)
    assert (l0 == stored["ecosystem_state"].reindex(e0.index)).all(), "V0 classification mismatch"
    for name, e in [("V1_eq3_calendar_month_sd", e1), ("V2_tws_lst_observed_baseline", e2)]:
        l = classify(e, nirv)
        rows.append({"land_type": land, "variant": name, "pearson_r_vs_submitted": float(np.corrcoef(e0, e)[0, 1]),
                     "mean_abs_diff": float((e - e0).abs().mean()),
                     "full_period_mean": float(e.mean()), "baseline_mean": float(e.loc[B0:B1].mean()),
                     "recent_2020_2024_mean": float(e.loc["2020":"2024"].mean()),
                     "state_agreement_pct": 100 * float((l == l0).mean()),
                     **{f"persist_{k.split(' / ')[0]}_{k.split(' / ')[1]}": v for k, v in diag_persist(l).items()}})
    rows.append({"land_type": land, "variant": "V0_submitted", "pearson_r_vs_submitted": 1.0, "mean_abs_diff": 0.0,
                 "full_period_mean": float(e0.mean()), "baseline_mean": float(e0.loc[B0:B1].mean()),
                 "recent_2020_2024_mean": float(e0.loc["2020":"2024"].mean()), "state_agreement_pct": 100.0,
                 **{f"persist_{k.split(' / ')[0]}_{k.split(' / ')[1]}": v for k, v in diag_persist(l0).items()}})
    # effective contribution of each component to EWBI variance (full period)
    cov = Z0.cov()
    tot = cov.to_numpy().sum()
    for v, _ in COMP:
        weights.append({"land_type": land, "component": v, "nominal_weight": 0.25,
                        "full_period_sd_of_signed_z": float(Z0[v].std(ddof=1)),
                        "share_of_EWBI_variance": float(cov[v].sum() / tot),
                        "lag1_autocorr": float(Z0[v].autocorr(1)), "lag12_autocorr": float(Z0[v].autocorr(12)),
                        "mean_2020_2024": float(Z0[v].loc["2020":"2024"].mean())})
res = pd.DataFrame(rows); wt = pd.DataFrame(weights)
res.to_csv(os.path.join(OUT, "ewbi_variant_sensitivity.csv"), index=False)
wt.to_csv(os.path.join(OUT, "ewbi_effective_component_contribution.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
txt = res.round(3).to_string() + "\n\n" + wt.round(3).to_string() + "\n\nReviewer hypothesis R: identically 0 (algebraic identity)."
open(os.path.join(OUT, "ewbi_sensitivity_checks_stdout.txt"), "w", encoding="utf-8").write(txt)
print(txt)
