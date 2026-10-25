"""Merge the GPP 2022 rerun (target GPP, folds 3-4, all models, baselines and SVR) into the forecasting source tables.

Only rows with variable == GPP and fold in {3, 4} are replaced; every other row is copied unchanged. Before replacing,
each output is rebuilt from the unchanged inputs and compared with the accepted table (exact reproduction of the
aggregation). Aggregation conventions are those of the accepted tables:
  Figure 7 cell        mean and SD (ddof = 1) over the fold x seed pixel-metric rows of a model x target x land type
  S7/S10 panel a       per model: mean over seeds within task, then mean and SD over 144 land x target x fold tasks
                       (monthly land-type aggregate metrics, baselines and SVR included)
  S7/S10 panel b       per fold: mean over seeds within task, then mean and SD over the 216 learned-model tasks
  S8 part b            pooled predictions <= 2021 vs >= 2022 per model x land type x seed, then mean over seeds

Usage: python apply_gpp_2022_forecast_fix.py --audit <AUDIT_TEMP> --docs <Docs/major_revision_analyses> --fix <GPP2022_FIX/forecast> --out <dir>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

LEARNED = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
BASELINES = ["Persistence", "Seasonal persistence", "Monthly climatology"]
SVR = "RBF-epsilon-SVR"
KEY = ["model", "land_type", "variable", "fold", "seed"]


def calc(observed, predicted):
    observed, predicted = np.asarray(observed, float), np.asarray(predicted, float)
    res = observed - predicted
    den = np.sum((observed - observed.mean()) ** 2)
    sd = np.std(observed, ddof=0)
    return (1 - np.sum(res ** 2) / den if den > 0 else np.nan,
            np.sqrt(np.mean(res ** 2)) / sd if sd > 0 else np.nan)


def norm_seed(frame):
    frame = frame.copy()
    frame["seed"] = frame["seed"].astype(str).str.replace(r"\.0$", "", regex=True)
    return frame


def is_fixed(frame):
    return frame.variable.eq("GPP") & frame.fold.isin([3, 4])


def replace_rows(old, new, models):
    old, new = norm_seed(old), norm_seed(new)
    new = new[new.variable.eq("GPP") & new.fold.isin([3, 4]) & new.model.isin(models)]
    drop = is_fixed(old) & old.model.isin(models)
    a = old[drop].set_index(KEY).index.sort_values()
    b = new.set_index(KEY).index.sort_values()
    assert a.equals(b), f"replacement keys differ: {set(a) ^ set(b)}"
    out = pd.concat([old[~drop], new[[c for c in old.columns if c in new.columns]]], ignore_index=True)
    return out, int(drop.sum())


def panel_a(monthly):
    t = monthly.groupby(["model", "land_type", "variable", "fold"])[["r2", "normalized_rmse"]].mean().reset_index()
    return t.groupby("model").agg(n_tasks=("r2", "size"), mean_r2=("r2", "mean"), sd_r2=("r2", "std"),
                                  mean_normalized_rmse=("normalized_rmse", "mean"),
                                  sd_normalized_rmse=("normalized_rmse", "std")).reset_index()


def panel_b(pixel, periods):
    px = pixel[pixel.model.isin(LEARNED)]
    t = px.groupby(["model", "land_type", "variable", "fold"])[["r2", "normalized_rmse"]].mean().reset_index()
    out = t.groupby("fold").agg(mean_r2=("r2", "mean"), sd_r2=("r2", "std"), mean_nrmse=("normalized_rmse", "mean"),
                                sd_nrmse=("normalized_rmse", "std"), n_tasks=("r2", "size")).reset_index()
    return out.merge(periods, on="fold")


def fig7_cells(pixel, fig7):
    px = pixel[pixel.model.isin(LEARNED)]
    g = px.groupby(["land_type", "variable", "model"]).agg(
        mean_r2=("r2", "mean"), sd_r2=("r2", "std"), mean_normalized_rmse=("normalized_rmse", "mean"),
        sd_normalized_rmse=("normalized_rmse", "std"), available_rows=("r2", "size")).reset_index()
    return fig7.merge(g, left_on=["land_type", "target_internal", "model"], right_on=["land_type", "variable", "model"],
                      suffixes=("", "_new"), validate="one_to_one")


def svr_monthly_from_predictions(metrics_csv):
    rows, pix = [], []
    for _, r in pd.read_csv(metrics_csv).iterrows():
        z = np.load(r["prediction_path"], allow_pickle=False)
        f = pd.DataFrame({"date": z["date"], "observed": z["y_true"], "predicted": z["y_pred"]})
        f = f[np.isfinite(f.observed) & np.isfinite(f.predicted)]
        m = f.groupby("date", as_index=False)[["observed", "predicted"]].mean()
        r2m, nrm = calc(m.observed, m.predicted)
        r2p, nrp = calc(f.observed, f.predicted)
        base = {"model": SVR, "land_type": r["land_type"], "variable": r["target"], "fold": int(r["fold"]),
                "seed": "deterministic"}  # label used for SVR rows in the accepted table
        rows.append({**base, "r2": r2m, "normalized_rmse": nrm, "n_months": len(m)})
        pix.append({**base, "r2": r2p, "normalized_rmse": nrp, "outer_test_r2": r["outer_test_r2"]})
    return pd.DataFrame(rows), pd.DataFrame(pix)


def load_predictions(paths, usecols, keep):
    frames = []
    for p in paths:
        for chunk in pd.read_csv(p, usecols=usecols, chunksize=250_000):
            chunk = chunk[keep(chunk)]
            if len(chunk):
                frames.append(chunk)
    return pd.concat(frames, ignore_index=True)


def s8b(old_pred, new_pred):
    data = pd.concat([old_pred[old_pred.fold.isin([1, 2])], new_pred[new_pred.fold.isin([3, 4])]], ignore_index=True)
    data = data[data["valid"].astype(bool)]
    data["date"] = pd.to_datetime(data["date"].astype(str), format="%Y%m")
    data["period"] = np.where(data.date.dt.year <= 2021, "pre", "post")
    rows = []
    for (m, l, s, p), g in data.groupby(["model", "land_type", "seed", "period"]):
        r2, nr = calc(g.observed, g.predicted)
        rows.append({"model": m, "land_type": l, "seed": s, "period": p, "r2": r2, "nrmse": nr, "n": len(g)})
    t = pd.DataFrame(rows).groupby(["model", "land_type", "period"]).agg(r2=("r2", "mean"), nrmse=("nrmse", "mean"),
                                                                         n_seeds=("seed", "nunique"),
                                                                         n_predictions=("n", "sum")).reset_index()
    w = t.pivot_table(index=["model", "land_type"], columns="period", values=["r2", "nrmse", "n_predictions"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["r2_post_minus_pre"] = w.r2_post - w.r2_pre
    w["nrmse_post_minus_pre"] = w.nrmse_post - w.nrmse_pre
    return w


def main():
    ap = argparse.ArgumentParser()
    for n in ("audit", "docs", "fix", "out"):
        ap.add_argument(f"--{n}", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = a.audit / "FIGURE_REFINEMENT_PHASE" / "source_tables"
    fig7 = pd.read_csv(a.audit / "FIGURE7_TWS_COMMON_CORRECTION" / "source_tables" / "Figure_7_source.csv")
    pix = pd.read_csv(src / "Figure_S7_exact_pixel_metrics.csv", low_memory=False)
    mon = pd.read_csv(src / "Figure_S7_exact_monthly_metrics.csv", low_memory=False)
    pa_old, pb_old = pd.read_csv(src / "Figure_S7_panel_a_source.csv"), pd.read_csv(src / "Figure_S7_panel_b_source.csv")
    periods = pb_old[["fold", "test_period"]]
    checks = []

    def check(name, ok):
        checks.append({"check": name, "passed": bool(ok)})
        assert ok, name

    # reproduction of the accepted aggregates from the unchanged rows
    x = panel_a(mon).merge(pa_old, on="model", suffixes=("", "_acc"))
    check("panel a reproduced", np.allclose(x.mean_r2, x.mean_r2_acc, atol=1e-12) and np.allclose(x.sd_normalized_rmse, x.sd_normalized_rmse_acc, atol=1e-12))
    x = panel_b(pix, periods).merge(pb_old, on="fold", suffixes=("", "_acc"))
    check("panel b reproduced", np.allclose(x.mean_r2, x.mean_r2_acc, atol=1e-12) and np.allclose(x.sd_nrmse, x.sd_nrmse_acc, atol=1e-12))
    x = fig7_cells(pix, fig7)
    check("Figure 7 cells reproduced from the pixel rows (all 216)", len(x) == 216 and np.allclose(x.mean_r2, x.mean_r2_new, atol=1e-12)
          and np.allclose(x.sd_normalized_rmse, x.sd_normalized_rmse_new, atol=1e-12))

    # new rows
    rs = a.fix / "reviewer_safe_out" / "tables" / "fix"
    new_pix = pd.concat([pd.read_csv(p) for p in sorted(rs.glob("*_pixel_metrics.csv"))] +
                        [pd.read_csv(a.fix / "bilstm_out" / "fix" / "BiLSTM_pixel_metrics.csv")], ignore_index=True)
    new_mon = pd.concat([pd.read_csv(p) for p in sorted(rs.glob("*_monthly_metrics.csv"))] +
                        [pd.read_csv(a.fix / "bilstm_out" / "fix" / "BiLSTM_monthly_metrics.csv")], ignore_index=True)
    for f in (new_pix, new_mon):
        f["seed"] = f["seed"].astype(str)
    svr_mon, svr_pix = svr_monthly_from_predictions(a.fix / "svr_pkg" / "metrics" / "svr_outer_test_selected_refit_metrics.csv")
    acc = norm_seed(mon[(mon.model == SVR) & mon.variable.eq("GPP") & mon.fold.eq(1)])
    rep = svr_mon[svr_mon.fold == 1].merge(acc, on=["land_type", "variable", "fold"], suffixes=("", "_acc"))
    check("SVR fold 1 (never reads 2022) reproduces the accepted monthly metric",
          len(rep) >= 1 and np.allclose(rep.r2, rep.r2_acc, atol=1e-9))
    new_mon = pd.concat([new_mon, svr_mon[svr_mon.fold.isin([3, 4])]], ignore_index=True)

    pix_fixed, n_pix = replace_rows(pix, new_pix, LEARNED + BASELINES)
    mon_fixed, n_mon = replace_rows(mon, new_mon, LEARNED + BASELINES + [SVR])
    check("pixel rows replaced = GPP folds 3-4 only", n_pix == len(new_pix[new_pix.fold.isin([3, 4])]))
    pix_fixed.to_csv(a.out / "Figure_S7_exact_pixel_metrics.csv", index=False)
    mon_fixed.to_csv(a.out / "Figure_S7_exact_monthly_metrics.csv", index=False)
    pa_new = panel_a(mon_fixed)
    pa_new = pa_new.set_index("model").loc[pa_old.model].reset_index()
    pa_new.to_csv(a.out / "Figure_S7_panel_a_source.csv", index=False)
    pb_new = panel_b(pix_fixed, periods)
    pb_new[pb_old.columns].to_csv(a.out / "Figure_S7_panel_b_source.csv", index=False)

    cells = fig7_cells(pix_fixed, fig7)
    f7 = fig7.copy()
    for c in ("mean_r2", "sd_r2", "mean_normalized_rmse", "sd_normalized_rmse", "available_rows"):
        f7[c] = cells[f"{c}_new"].to_numpy()
    f7["r2_negative_flag"] = f7.mean_r2 < 0
    f7["r2_plot_radius"] = f7.mean_r2.clip(lower=0)
    f7["nrmse_plot_radius"] = f7.mean_normalized_rmse
    changed = ~(np.isclose(f7.mean_r2, fig7.mean_r2, atol=1e-12) & np.isclose(f7.mean_normalized_rmse, fig7.mean_normalized_rmse, atol=1e-12))
    check("only GPP Figure 7 cells change", set(f7.loc[changed, "target_internal"]) <= {"GPP"})
    f7.to_csv(a.out / "Figure_7_source.csv", index=False)
    pd.concat([fig7.loc[changed, ["land_type_display", "target_display", "model", "mean_r2", "mean_normalized_rmse"]].reset_index(drop=True),
               f7.loc[changed, ["mean_r2", "mean_normalized_rmse"]].reset_index(drop=True).add_suffix("_corrected")], axis=1
              ).to_csv(a.out / "figure7_gpp_cells_old_vs_corrected.csv", index=False)

    # Supplementary Table S8 part b
    usecols = ["model", "fold", "land_type", "variable", "seed", "date", "observed", "predicted", "valid"]
    pred_root = a.docs / "05_forecasting_raster_reviewer_safe_candidate" / "predictions"
    old_pred = load_predictions([pred_root / f"reviewer_safe_{n}_predictions.csv.gz" for n in ("lstm", "remaining_models", "xgboost")],
                                usecols, lambda c: c.variable.eq("GPP") & c.model.ne("BiLSTM"))
    old_bi = load_predictions(sorted((a.docs / "corrected_true_BiLSTM_rerun_20260728" / "05_predictions").glob("GPP_*_corrected_predictions.csv.gz")),
                              usecols, lambda c: c.variable.eq("GPP"))
    old_all = pd.concat([old_pred, old_bi], ignore_index=True)
    acc8 = pd.read_csv(a.audit / "REVISION_COMPUTATIONAL_PHASE" / "task4_gpp_transition" / "gpp_forecast_boundary_sensitivity.csv")
    acc8 = acc8[acc8.boundary_period.eq("pre_boundary_le_2021")]
    rep8 = s8b(old_all[old_all.fold.isin([1, 2])], old_all[old_all.fold.isin([3, 4])]).merge(acc8, on=["model", "land_type"], suffixes=("", "_acc"))
    check("accepted S8 part b reproduced from the original predictions", len(rep8) == 24 and np.allclose(rep8.r2_pre, rep8.r2_pre_acc, atol=1e-9)
          and np.allclose(rep8.nrmse_post, rep8.nrmse_post_acc, atol=1e-9))
    new_pred = load_predictions(sorted(rs.glob("*_predictions.csv.gz")) + sorted((a.fix / "bilstm_out" / "fix").glob("*_predictions.csv.gz")),
                                usecols, lambda c: c.variable.eq("GPP") & c.model.isin(LEARNED))
    s8 = s8b(old_all, new_pred)
    check("S8b covers 6 models x 4 land types", len(s8) == 24)
    s8.to_csv(a.out / "table_s8_part_b.csv", index=False)
    pd.DataFrame(checks).to_csv(a.out / "forecast_fix_checks.csv", index=False)
    print(pd.DataFrame(checks).to_string(index=False))


if __name__ == "__main__":
    main()
