"""Correct the ecological part of Supplementary Table S8 (GPP product-transition sensitivity) from verified inputs.

Findings this script documents and checks (no model is run; nothing existing is overwritten):
  * Figure 5 (accepted) correlates GPP with other indices over 2001-01..2021-12 only (FLUXCOM era); its coefficients are
    exactly the old Table S8 "FLUXCOM-only 2001-2021" column, i.e. that column is the main analysis, not a sensitivity.
  * The old "stitched 2001-2024" column includes the 12 MOD17 months of 2022, but 9 of the 12 California-wide 2022
    MOD17 exports (and the land-type rasters derived from them) are byte-identical, and two more are a second identical
    pair: those months are not monthly observations. 2023 and 2024 exports are all distinct.
  * Corrected sensitivity: FLUXCOM 2001-2021 + MOD17 2023-2024, with 2022 excluded, compared with the FLUXCOM-only
    main analysis using the same series, units (gC m-2 d-1), spatial means, explicit YYYY-MM joins and statistics
    (Pearson r with nominal p; OLS trend with HAC-12 SE), as in the original task-4 script.

Usage: python src/ecological/gpp/gpp_table_s8_correction.py --project-root <PROJECT_ROOT> --data-root <DATA_ROOT> [--out DIR]
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_n, "1")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parents[3]
LANDS = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_DIR = {"Grasslands": "Grasslands", "Natural Forests": "NaturalForests", "Savannas": "Savannas", "Shrublands": "Shrublands"}
COMPARATORS = ["NIRv", "NDVI", "GNDVI", "GOSIF"]
DISPLAY = {"GOSIF": "GOSIF/SIF"}
MAIN = "fluxcom_only_2001_2021"
SENS = "fluxcom_2001_2021_plus_mod17_2023_2024"
OLD = "old_stitched_2001_2024_incl_2022"
FULL = "fluxcom_2001_2021_plus_mod17_2022_2024"  # with the re-acquired 2022 MOD17 (gpp_table_s8_final.py)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    git = REPO / ".git"
    gitdir = Path(git.read_text().split("gitdir:", 1)[1].strip()) if git.is_file() else git
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1]
    for base in (gitdir, gitdir.parent.parent):
        if (base / ref).is_file():
            return (base / ref).read_text().strip() + f" ({ref})"
    return f"unresolved ({ref})"


def select(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    y = frame.index.year
    if dataset == MAIN:
        return frame[y <= 2021]
    if dataset == SENS:
        return frame[(y <= 2021) | (y >= 2023)]
    if dataset in (OLD, FULL):
        return frame
    raise ValueError(dataset)


def hac_trend(values: pd.Series):
    values = values.dropna().sort_index()
    x = (values.index.year + (values.index.month - 1) / 12.0).to_numpy(float)
    fit = sm.OLS(values.to_numpy(float), sm.add_constant(x - x.min())).fit(cov_type="HAC", cov_kwds={"maxlags": 12})
    low, high = fit.conf_int(alpha=0.05)[1]
    return float(fit.params[1]), float(fit.pvalues[1]), float(low), float(high)


def unique_monthly(frame: pd.DataFrame, col: str) -> pd.Series:
    s = frame.set_index("date")[col].sort_index()
    if s.index.duplicated().any():
        raise ValueError(f"duplicate YYYY-MM records in {col}")
    if (s.index.day != 1).any():
        raise ValueError("dates must be month starts")
    return s


def compute_land(series: pd.DataFrame, ewbi: pd.DataFrame, land: str, datasets) -> list:
    """Correlations with GPP, EWBI-GPP anomaly correlation and GPP trend for one land type and the given variants.
    Every pair is joined explicitly on YYYY-MM (inner join, then complete-case)."""
    rows = []
    sub = series[series.land_type.eq(land)]
    idx = {v: f for v, f in sub.groupby("variable")}
    gpp_raw = unique_monthly(idx["GPP"], "original_value").rename("GPP")
    gpp_anom = unique_monthly(idx["GPP"], "standardized_anomaly").rename("GPP")
    ew = unique_monthly(ewbi[ewbi.land_type.eq(land)], "ewbi").rename("EWBI")
    quantities = [(f"GPP vs {DISPLAY.get(c, c)}", pd.concat([gpp_raw, unique_monthly(idx[c], "original_value").rename(c)], axis=1, join="inner").dropna(), c)
                  for c in COMPARATORS]
    quantities.append(("Deseasonalized EWBI vs GPP anomaly", pd.concat([gpp_anom, ew], axis=1, join="inner").dropna(), "EWBI"))
    for qname, paired_all, other in quantities:
        for ds in datasets:
            p = select(paired_all, ds)
            r, pv = pearsonr(p["GPP"], p[other])
            rows.append({"land_type": land, "quantity": qname, "measure": "pearson_r", "dataset": ds, "estimate": r, "p_nominal": pv,
                         "n_months": len(p), "start": p.index.min().strftime("%Y-%m"), "end": p.index.max().strftime("%Y-%m"),
                         "n_2022_months": int((p.index.year == 2022).sum()), "n_mod17_months": int((p.index.year >= 2022).sum())})
    for ds in datasets:
        s = select(gpp_raw.to_frame(), ds)["GPP"].dropna()
        slope, pv, lo, hi = hac_trend(s)
        rows.append({"land_type": land, "quantity": "GPP linear trend (gC m-2 d-1 per year, HAC-12)", "measure": "ols_slope", "dataset": ds,
                     "estimate": slope, "p_nominal": pv, "ci95_low": lo, "ci95_high": hi, "n_months": len(s),
                     "start": s.index.min().strftime("%Y-%m"), "end": s.index.max().strftime("%Y-%m"),
                     "n_2022_months": int((s.index.year == 2022).sum()), "n_mod17_months": int((s.index.year >= 2022).sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--data-root", required=True, help="folder containing 'Test Data' (MOD17 exports)")
    ap.add_argument("--out", default=str(REPO / "results" / "audit" / "gpp_table_s8"))
    a = ap.parse_args()
    root, data_root, out = Path(a.project_root), Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mra = root / "Docs" / "major_revision_analyses"
    src = {
        "series": mra / "final_paper_figures_corrected" / "tables" / "paper_standardized_anomaly_series.csv",
        "gndvi": mra / "final_previous_GNDVI_with_test_rebuild" / "tables" / "Figure_2_previous_GNDVI_with_test_monthly_series.csv",
        "ewbi": mra / "04_detrending_sensitivity_fixed" / "tables" / "main_ewbi_deseasonalized_series.csv",
        "raw": mra / "04_detrending_sensitivity_fixed" / "tables" / "monthly_landtype_series_raw.csv",
        "fig5_corr": root / "AUDIT_TEMP" / "REVISION_COMPUTATIONAL_PHASE" / "task1_figure5" / "figure5_raw_correlations.csv",
        "fig5_input": root / "AUDIT_TEMP" / "REVISION_COMPUTATIONAL_PHASE" / "task1_figure5" / "figure5_exact_input_monthly_series.csv",
        "fig5_deseason_input": root / "AUDIT_TEMP" / "REVISION_COMPUTATIONAL_PHASE" / "task1_figure5" / "figure5_deseasonalized_exact_input_monthly_series.csv",
        "old_s8": root / "AUDIT_TEMP" / "REVISION_COMPUTATIONAL_PHASE" / "task4_gpp_transition" / "gpp_product_transition_sensitivity.csv",
        "s5_input": root / "AUDIT_TEMP" / "FIGURE_REFINEMENT_PHASE" / "source_tables" / "Figure_S5_exact_monthly_input.csv",
        "fig5_displayed": REPO / "results" / "audit" / "gpp_table_s8" / "inputs" / "figure5_displayed_values_transcribed.csv",
    }
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": str(detail)})
        if not ok:
            raise AssertionError(f"{name}: {detail}")

    # ---------- 1. evidence on the 2022 MOD17 exports ----------
    ev = []
    for scope, folder, pattern in [("California-wide export", data_root / "Test Data" / "GPP", "GPP_MOD17A2H_CA_5km_{y}_{m:02d}.tif")] + \
            [(f"land type: {l}", data_root / "Test Data" / "Masked_LandTypes" / LAND_DIR[l] / "GPP", "GPP_MOD17A2H_CA_5km_{y}_{m:02d}_" + LAND_DIR[l] + ".tif") for l in LANDS]:
        for y in (2022, 2023, 2024):
            for m in range(1, 13):
                f = folder / pattern.format(y=y, m=m)
                ev.append({"scope": scope, "year": y, "month": m, "file": f.name, "exists": f.is_file(),
                           "sha256": sha256(f) if f.is_file() else ""})
    ev = pd.DataFrame(ev)
    ev["duplicate_of_other_month_same_year"] = ev.groupby(["scope", "year"])["sha256"].transform(lambda s: s.duplicated(keep=False))
    ev.to_csv(out / "mod17_export_hashes_2022_2024.csv", index=False)
    dup = ev.groupby(["scope", "year"]).agg(files=("file", "size"), distinct=("sha256", "nunique")).reset_index()
    check("2022 MOD17 exports are duplicated in every scope", (dup[dup.year == 2022]["distinct"] <= 3).all(), dup[dup.year == 2022].to_dict("records"))
    check("2023 and 2024 MOD17 exports are all distinct", (dup[dup.year != 2022]["distinct"] == 12).all(), "12 distinct per year and scope")

    # ---------- 2. load the analysed series (same as the original task-4 script) ----------
    series = pd.read_csv(src["series"], parse_dates=["date"])
    gnd = pd.read_csv(src["gndvi"], parse_dates=["date"])
    series = pd.concat([series[series.variable.ne("GNDVI")], gnd[gnd.variable.eq("GNDVI")]], ignore_index=True)
    series["land_type"] = series["land_type"].replace({"NaturalForests": "Natural Forests"})
    ewbi = pd.read_csv(src["ewbi"], parse_dates=["date"])
    ewbi["land_type"] = ewbi["land_type"].replace({"NaturalForests": "Natural Forests"})
    raw = pd.read_csv(src["raw"], parse_dates=["date"])
    raw["land_type"] = raw["land_type"].replace({"NaturalForests": "Natural Forests"})
    graw = raw[raw.variable.eq("GPP")]
    check("FLUXCOM GPP (2001-2021) stored in gC m-2 d-1 without conversion", set(graw[graw.date.dt.year <= 2021]["unit_action"]) == {"as stored"})
    check("GPP analysed values were not gap-filled", not graw["was_gap_filled"].astype(str).eq("True").any(), "was_gap_filled all False")
    # The conversion of MOD17 (kgC m-2 month-1 -> gC m-2 d-1) was applied only when a raster's median was < 0.2
    # (detrending_utils.py). Peak-season Natural-Forest months exceed that threshold and stayed unconverted.
    unconv = graw[(graw.date.dt.year >= 2022) & graw.unit_action.eq("as stored")][["date", "land_type", "raw_value", "source_path"]].copy()
    unconv["days"] = unconv.date.dt.days_in_month
    unconv["converted_gC_m2_d"] = unconv.raw_value * 1000.0 / unconv.days
    unconv.to_csv(out / "mod17_months_left_unconverted.csv", index=False)
    check("unconverted MOD17 months identified", len(unconv) == 2 and set(unconv.land_type) == {"Natural Forests"}
          and set(unconv.date.dt.strftime("%Y-%m")) == {"2023-05", "2023-06"}, unconv[["date", "land_type", "raw_value"]].to_dict("records"))

    # ---------- 3. compute the three variants ----------
    # Repair the two unconverted months for the corrected sensitivity only (deterministic unit conversion of the stored
    # land-type mean; the old stitched variant is reproduced exactly as published for comparison).
    repaired = series.copy()
    gser = repaired.variable.eq("GPP")
    for _, u in unconv.iterrows():
        sel = gser & repaired.land_type.eq(u.land_type) & repaired.date.eq(u.date)
        assert sel.sum() == 1 and np.isclose(repaired.loc[sel, "original_value"].iloc[0], u.raw_value)
        same_month = gser & repaired.land_type.eq(u.land_type) & (repaired.date.dt.month == u.date.month) & (repaired.date.dt.year <= 2011)
        clim = (repaired.loc[same_month, "original_value"] - repaired.loc[same_month, "processed_value"])
        assert clim.max() - clim.min() < 1e-9, "deseasonalized_only must subtract a fixed 2001-2011 calendar-month mean"
        row = repaired.loc[sel].iloc[0]
        new_orig = u.converted_gC_m2_d
        new_proc = new_orig - clim.iloc[0]
        repaired.loc[sel, ["original_value", "processed_value", "standardized_anomaly"]] = [
            new_orig, new_proc, (new_proc - row.baseline_anomaly_mean) / row.baseline_anomaly_sd]
    check("repaired months lie within the 2023-2024 MOD17 range of the same land type",
          all(repaired[gser & repaired.land_type.eq("Natural Forests") & (repaired.date.dt.year >= 2023)]["original_value"].between(1.0, 9.0)))

    rows = []
    for land in LANDS:
        for variant_series, datasets in ((series, (MAIN, OLD)), (repaired, (SENS,))):
            rows.extend(compute_land(variant_series, ewbi, land, datasets))
    res = pd.DataFrame(rows)
    res.to_csv(out / "all_variants_long.csv", index=False)
    cov = res[["land_type", "quantity", "dataset", "start", "end", "n_months", "n_2022_months", "n_mod17_months"]]
    cov.to_csv(out / "pairwise_date_coverage.csv", index=False)
    # period-selection checks
    check("main analysis ends 2021-12 with no MOD17 months", (res[res.dataset == MAIN]["end"] == "2021-12").all() and (res[res.dataset == MAIN]["n_mod17_months"] == 0).all())
    s = res[res.dataset == SENS]
    check("sensitivity excludes every 2022 month and includes 2023-2024", (s["n_2022_months"] == 0).all() and (s["end"] == "2024-12").all() and (s["n_mod17_months"] == 24).all())

    # ---------- 4. reproduce Figure 5 and the old Table S8 ----------
    f5 = pd.read_csv(src["fig5_corr"])
    f5 = f5[(f5.variable_1.eq("GPP") ^ f5.variable_2.eq("GPP"))]
    f5["other"] = np.where(f5.variable_1.eq("GPP"), f5.variable_2, f5.variable_1)
    mainraw = res[(res.dataset == MAIN) & res.quantity.str.startswith("GPP vs")].copy()
    mainraw["other"] = mainraw.quantity.str.replace("GPP vs ", "").replace({"GOSIF/SIF": "GOSIF"})
    m = mainraw.merge(f5, left_on=["land_type", "other"], right_on=["land_type", "other"], validate="one_to_one")
    check("main-analysis GPP correlations equal the accepted Figure 5 source (unrounded)", len(m) == 16 and (m.estimate - m.pearson_r).abs().max() < 1e-12,
          f"16 cells, max |diff| = {(m.estimate - m.pearson_r).abs().max():.2e}")
    check("main-analysis sample sizes equal Figure 5 pairwise N", (m.n_months == m.n_shared_months).all(), sorted(set(m.n_months)))
    old = pd.read_csv(src["old_s8"])
    ANAL = {"raw_monthly_pearson_correlation": None, "deseasonalized_EWBI_GPP_correlation": "Deseasonalized EWBI vs GPP anomaly",
            "raw_GPP_linear_trend_HAC12": "GPP linear trend (gC m-2 d-1 per year, HAC-12)"}
    old["quantity"] = [ANAL[t] or f"GPP vs {DISPLAY.get(c, c)}" for t, c in zip(old.analysis_type, old.comparator)]
    old["dataset"] = old.gpp_dataset.map({"stitched_2001_2024": OLD, "fluxcom_only_2001_2021": MAIN})
    mo = res.merge(old[["land_type", "quantity", "dataset", "estimate", "p_value", "n_months"]], on=["land_type", "quantity", "dataset"], suffixes=("", "_old"))
    check("old Table S8 reproduced exactly from the same inputs", len(mo) == 48 and (mo.estimate - mo.estimate_old).abs().max() < 1e-10 and (mo.n_months == mo.n_months_old).all(),
          f"48 rows, max |diff| = {(mo.estimate - mo.estimate_old).abs().max():.2e}")
    if src["fig5_displayed"].is_file():
        disp = pd.read_csv(src["fig5_displayed"])
        dm = disp.merge(f5, on=["land_type", "other"]) if "other" in disp else None
        full = pd.read_csv(src["fig5_corr"])
        full = full[full.variable_1 != full.variable_2]
        key = lambda a, b: "|".join(sorted([a, b]))
        full["pair"] = [key(a, b) for a, b in zip(full.variable_1, full.variable_2)]
        disp["pair"] = [key(a, b) for a, b in zip(disp.variable_1, disp.variable_2)]
        j = disp.merge(full[["land_type", "pair", "pearson_r"]], on=["land_type", "pair"], validate="one_to_one")
        bad = j[(j.displayed - j.pearson_r).abs() > 0.005 + 1e-9]
        check("accepted Figure 5 image agrees with its source at displayed precision (all 144 off-diagonal cells)",
              len(j) == 144 and bad.empty, f"{len(j)} cells; mismatches: {bad[['land_type', 'pair', 'displayed', 'pearson_r']].to_dict('records')}")

    # ---------- 5. corrected Table S8 part a ----------
    wide = res[res.dataset.isin([MAIN, SENS])].pivot_table(index=["land_type", "quantity"], columns="dataset",
                                                             values=["estimate", "p_nominal", "n_months"], aggfunc="first")
    tab = []
    order = [f"GPP vs {DISPLAY.get(c, c)}" for c in COMPARATORS] + ["Deseasonalized EWBI vs GPP anomaly", "GPP linear trend (gC m-2 d-1 per year, HAC-12)"]
    for land in LANDS:
        for q in order:
            w = wide.loc[(land, q)]
            e1, e2 = w[("estimate", MAIN)], w[("estimate", SENS)]
            p1, p2 = w[("p_nominal", MAIN)], w[("p_nominal", SENS)]
            diff = e2 - e1
            changed = (p1 < 0.05) != (p2 < 0.05)
            tab.append({"land_type": land, "quantity": q,
                        "fluxcom_only_2001_2021_estimate": e1, "fluxcom_only_p_nominal": p1, "fluxcom_only_n": int(w[("n_months", MAIN)]),
                        "with_mod17_2023_2024_estimate": e2, "with_mod17_p_nominal": p2, "with_mod17_n": int(w[("n_months", SENS)]),
                        "difference_with_mod17_minus_fluxcom_only": diff, "nominal_p_lt_0_05_status_changed": changed})
    tab = pd.DataFrame(tab)
    check("difference column = with-MOD17 minus FLUXCOM-only", np.allclose(tab.difference_with_mod17_minus_fluxcom_only, tab.with_mod17_2023_2024_estimate - tab.fluxcom_only_2001_2021_estimate))
    check("significance-change flags recomputed from nominal p", (tab.nominal_p_lt_0_05_status_changed == ((tab.fluxcom_only_p_nominal < 0.05) != (tab.with_mod17_p_nominal < 0.05))).all())
    tab.to_csv(out / "table_s8_part_a_corrected.csv", index=False)

    def fmtp(p):
        return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"

    read = pd.DataFrame({
        "Land type": tab.land_type, "Quantity": tab.quantity,
        "FLUXCOM only, 2001–2021 (main; = Figure 5 for correlations)": [f"{e:.3f} (p = {fmtp(p)}; N = {n})" for e, p, n in zip(tab.fluxcom_only_2001_2021_estimate, tab.fluxcom_only_p_nominal, tab.fluxcom_only_n)],
        "FLUXCOM 2001–2021 + MOD17 2023–2024 (2022 excluded)": [f"{e:.3f} (p = {fmtp(p)}; N = {n})" for e, p, n in zip(tab.with_mod17_2023_2024_estimate, tab.with_mod17_p_nominal, tab.with_mod17_n)],
        "Difference (with MOD17 − FLUXCOM only)": [f"{d:+.3f}" for d in tab.difference_with_mod17_minus_fluxcom_only],
        "Nominal p < 0.05 status changed": np.where(tab.nominal_p_lt_0_05_status_changed, "Yes", "No"),
    })
    read.to_csv(out / "table_s8_part_a_corrected_readable.csv", index=False, encoding="utf-8-sig")

    # ---------- 6. old vs corrected, cell by cell ----------
    oldw = res[res.dataset.isin([MAIN, OLD])].pivot_table(index=["land_type", "quantity"], columns="dataset", values=["estimate", "p_nominal", "n_months"], aggfunc="first")
    comp = []
    for _, r in tab.iterrows():
        o = oldw.loc[(r.land_type, r.quantity)]
        old_diff = o[("estimate", MAIN)] - o[("estimate", OLD)]
        old_flag = (o[("p_nominal", MAIN)] < 0.05) != (o[("p_nominal", OLD)] < 0.05)
        comp.append({"land_type": r.land_type, "quantity": r.quantity,
                     "old_column_fluxcom_only": o[("estimate", MAIN)], "corrected_fluxcom_only": r.fluxcom_only_2001_2021_estimate,
                     "old_column_stitched_incl_2022": o[("estimate", OLD)], "old_stitched_n": int(o[("n_months", OLD)]),
                     "corrected_with_mod17_excl_2022": r.with_mod17_2023_2024_estimate, "corrected_n": r.with_mod17_n,
                     "old_difference_fluxcom_minus_stitched": old_diff, "corrected_difference_with_mod17_minus_fluxcom": r.difference_with_mod17_minus_fluxcom_only,
                     "old_changed_flag": bool(old_flag), "corrected_changed_flag": bool(r.nominal_p_lt_0_05_status_changed),
                     "comparison_column_value_changed": not np.isclose(o[("estimate", OLD)], r.with_mod17_2023_2024_estimate, atol=5e-4),
                     "flag_changed": bool(old_flag) != bool(r.nominal_p_lt_0_05_status_changed)})
    comp = pd.DataFrame(comp)
    comp.to_csv(out / "old_vs_corrected_table_s8_part_a.csv", index=False)

    # ---------- 7. dependency screen ----------
    dep = []
    d5 = pd.read_csv(src["fig5_deseason_input"], parse_dates=["date"])
    dmax = d5[d5.variable.eq("GPP")]["date"].max()
    dep.append({"item": "Figure S11 (deseasonalized Figure-5 sensitivity)", "uses_2022_GPP": bool(dmax.year >= 2022),
                "evidence": f"GPP input ends {dmax:%Y-%m}", "action": "none needed" if dmax.year < 2022 else "affected"})
    if src["s5_input"].is_file():
        s5 = pd.read_csv(src["s5_input"], parse_dates=["date"])
        s5["land_type"] = s5["land_type"].replace({"NaturalForests": "Natural Forests"})
        for land in LANDS:
            g = s5[s5.land_type.eq(land)].set_index("date").sort_index()
            rec = g.loc["2020-01-01":"2024-12-01"]
            r_all = pearsonr(rec["EWBI"], rec["GPP"])[0]
            k = rec[rec.index.year != 2022]
            r_ex = pearsonr(k["EWBI"], k["GPP"])[0]
            dep.append({"item": f"EWBI-response association, recent 2020-2024 GPP cell, {land} (Figure S5 / Table S7 in the 2026-10-25 Supplement)",
                        "uses_2022_GPP": True, "evidence": f"r with 2022 = {r_all:.3f} (N = {len(rec)}); diagnostic r without 2022 = {r_ex:.3f} (N = {len(k)})",
                        "action": "affected; bootstrap cell should be recomputed without 2022 (not done here)"})
    dep.append({"item": "Forecasting: GPP target (Figure 7 / Table 4-5 GPP cells, Table S10/S7 fold summaries, statistical comparison, Table S8 part b)",
                "uses_2022_GPP": True, "evidence": "raster cache reads the same Test Data GPP rasters: fold-3 test targets 2021-2022 and fold-4 training/lag inputs include the duplicated 2022 months",
                "action": "affected; needs valid 2022 MOD17 exports and a GPP-only rerun (4 land types x 4 folds; folds 3-4 change)"})
    dep.append({"item": "Figure 3 (vegetation/productivity anomalies), GPP 2022 segment; S4 weighting/component correlations (GPP is one of five responses, 288 months)",
                "uses_2022_GPP": True, "evidence": "same paper anomaly series", "action": "affected (2022 GPP segment invalid); not changed here"})
    pd.DataFrame(dep).to_csv(out / "dependency_screen.csv", index=False)

    # ---------- 8. reproducibility and manifest ----------
    res2 = pd.read_csv(out / "all_variants_long.csv")
    check("coefficients and sample sizes reproducible (re-read)", np.allclose(res2.estimate, res.estimate) and (res2.n_months == res.n_months).all())
    pd.DataFrame(checks).to_csv(out / "validation_checks.csv", index=False)
    manifest = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source_git_commit": git_commit(),
                "inputs": [{"name": k, "path": str(p), "sha256": sha256(p)} for k, p in src.items() if p.is_file()],
                "mod17_export_hashes": "mod17_export_hashes_2022_2024.csv",
                "code": [{"path": str(Path(__file__).resolve().relative_to(REPO)).replace("\\", "/"), "sha256": sha256(Path(__file__))}],
                "outputs": [{"path": p.name, "sha256": sha256(p)} for p in sorted(out.glob("*.csv"))]}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(read.to_string(index=False))
    print(f"\n{len(checks)} checks passed; changed flags (corrected): {int(tab.nominal_p_lt_0_05_status_changed.sum())}; "
          f"old flags: {int(comp.old_changed_flag.sum())}")


if __name__ == "__main__":
    main()
