"""EWBI degeneracy diagnostics on EXISTING compact tables only (no rasters). Writes only into this folder."""
import os
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
DOCS = r"<PROJECT_ROOT>\Docs"
FIX = os.path.join(DOCS, r"major_revision_analyses\04_detrending_sensitivity_fixed\tables")
B0, B1 = "2001-01-01", "2011-12-01"
LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]

main = pd.read_csv(os.path.join(FIX, "main_ewbi_deseasonalized_series.csv"), parse_dates=["date"])
pre = pd.read_csv(os.path.join(FIX, "preprocessed_monthly_series.csv"), parse_dates=["date"])
raw = pd.read_csv(os.path.join(FIX, "monthly_landtype_series_raw.csv"), parse_dates=["date"])
states = pd.read_csv(os.path.join(FIX, "main_state_classification_deseasonalized.csv"), parse_dates=["date"])
lines = [f"columns main_ewbi: {list(main.columns)}"]

# 1. Re-derive EWBI from the stored deseasonalized components with the code's own formula
rows, recon = [], []
for land in LANDS:
    p = pre[(pre.land_type == land) & (pre.preprocessing == "deseasonalized_only")]
    w = p.pivot(index="date", columns="variable", values="processed_value").sort_index()
    z = {}
    for v, sgn in [("SPEI", 1), ("TWS", 1), ("VPD", -1), ("LST", -1)]:
        b = w.loc[B0:B1, v]
        z[v] = sgn * (w[v] - b.mean()) / b.std(ddof=1)
    zdf = pd.DataFrame(z)
    e = zdf.mean(axis=1, skipna=False)
    m = main[main.land_type == land].set_index("date")["ewbi"].reindex(e.index)
    recon.append({"land_type": land, "max_abs_diff_recomputed_vs_stored": float((e - m).abs().max()),
                  "n_months": int(e.notna().sum())})
    s = m.dropna()
    base, hist, rec = s.loc[B0:B1], s.loc["2012-01-01":"2016-12-01"], s.loc["2020-01-01":"2024-12-01"]
    rows.append({
        "land_type": land, "n_months": len(s), "mean_2001_2024": s.mean(), "median_2001_2024": s.median(),
        "sd_2001_2024": s.std(ddof=1), "min": s.min(), "max": s.max(), "range": s.max() - s.min(),
        "temporal_variance": s.var(ddof=1),
        "n_unique_values_rounded_1e-6": int(s.round(6).nunique()),
        "n_distinct_at_0.01": int(s.round(2).nunique()),
        "baseline_mean_2001_2011": base.mean(), "baseline_sd_2001_2011": base.std(ddof=1),
        "historic_mean_2012_2016": hist.mean(), "recent_mean_2020_2024": rec.mean(),
        "lag1_autocorrelation": s.autocorr(1),
        "median_threshold_used_for_states": states[states.land_type == land]["ewbi_median_threshold"].iloc[0]
        if "ewbi_median_threshold" in states.columns else np.nan,
    })
    # component diagnostics
    cc = zdf.corr()
    cc.to_csv(os.path.join(OUT, f"component_signed_z_correlation_{land.replace(' ', '_')}.csv"))
    comp = pd.DataFrame({"baseline_mean": zdf.loc[B0:B1].mean(), "baseline_sd": zdf.loc[B0:B1].std(ddof=1),
                         "full_mean": zdf.mean(), "full_sd": zdf.std(ddof=1),
                         "lag1_autocorr": zdf.apply(lambda c: c.autocorr(1)),
                         "lag12_autocorr": zdf.apply(lambda c: c.autocorr(12)),
                         "corr_with_EWBI": zdf.corrwith(e)})
    comp.insert(0, "land_type", land)
    comp.to_csv(os.path.join(OUT, f"component_diagnostics_{land.replace(' ', '_')}.csv"))
    lines.append(f"\n== {land}: signed-z component correlations\n{cc.round(3)}\n{comp.round(3)}")

diag = pd.DataFrame(rows)
diag.to_csv(os.path.join(OUT, "ewbi_landtype_degeneracy_diagnostics.csv"), index=False)
pd.DataFrame(recon).to_csv(os.path.join(OUT, "ewbi_recomputation_check.csv"), index=False)

# 2. Calendar-month diagnostics (land type x calendar month)
mm = main.dropna(subset=["ewbi"]).assign(month=lambda d: d.date.dt.month)
cal = mm.groupby(["land_type", "month"])["ewbi"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
cal.to_csv(os.path.join(OUT, "ewbi_landtype_by_calendar_month.csv"), index=False)

# 3. Observation/fill status of the components (raw_value NaN = no raster observation -> filled)
fill = []
for land in LANDS:
    for v in ["SPEI", "TWS", "VPD", "LST"]:
        r = raw[(raw.land_type == land) & (raw.variable == v)].set_index("date").sort_index()
        obs = r["raw_value"].notna()
        fill.append({"land_type": land, "variable": v,
                     "first_observed": r.index[obs].min().date() if obs.any() else None,
                     "last_observed": r.index[obs].max().date() if obs.any() else None,
                     "filled_months_total": int((~obs).sum()),
                     "filled_months_in_2001_2011_baseline": int((~obs.loc[B0:B1]).sum()),
                     "baseline_months": int(len(obs.loc[B0:B1]))})
fill = pd.DataFrame(fill)
fill.to_csv(os.path.join(OUT, "component_observation_and_fill_status.csv"), index=False)

# 4. Effect of the TWS fill on its baseline standardisation (observed-only baseline 2002-04..2011-12)
eff = []
for land in LANDS:
    p = pre[(pre.land_type == land) & (pre.preprocessing == "deseasonalized_only") & (pre.variable == "TWS")].set_index("date")["processed_value"]
    r = raw[(raw.land_type == land) & (raw.variable == "TWS")].set_index("date")["raw_value"]
    obsb = p.loc[B0:B1][r.loc[B0:B1].notna()]
    eff.append({"land_type": land, "baseline_mean_all": p.loc[B0:B1].mean(), "baseline_sd_all": p.loc[B0:B1].std(ddof=1),
                "baseline_mean_observed_only": obsb.mean(), "baseline_sd_observed_only": obsb.std(ddof=1),
                "n_observed_baseline": len(obsb)})
pd.DataFrame(eff).to_csv(os.path.join(OUT, "tws_baseline_fill_effect.csv"), index=False)

pd.set_option("display.width", 220)
txt = "\n".join(lines) + "\n\n== recomputation check\n" + pd.DataFrame(recon).to_string() + \
      "\n\n== degeneracy diagnostics\n" + diag.round(4).T.to_string() + \
      "\n\n== fill status\n" + fill.to_string() + "\n\n== TWS baseline fill effect\n" + pd.DataFrame(eff).round(4).to_string()
open(os.path.join(OUT, "ewbi_diagnostics_stdout.txt"), "w", encoding="utf-8").write(txt)
print(txt)
