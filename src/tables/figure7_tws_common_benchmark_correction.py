"""Replace the four BiLSTM x TWS Figure 7 cells (18-month state-space sensitivity) with the
common 12-month benchmark rows. No retraining: the fold x seed metric rows already exist.

Inputs (read-only):
  AUDIT_TEMP/FIGURE_REFINEMENT_PHASE/source_tables/Figure_7_source.csv   (accepted Figure 7 source)
  Docs/major_revision_analyses/corrected_true_BiLSTM_rerun_20260728/08_downstream/corrected_combined_pixel_metrics.csv
  Docs/major_revision_analyses/05_forecasting_raster_reviewer_safe_candidate/tables/reviewer_safe_all_pixel_metrics.csv
Output: source_tables/Figure_7_source.csv in this folder (same schema), plus a change log.
Aggregation convention (verified against all other cells): mean and SD (ddof=1) over fold x seed rows.
"""
import hashlib
import os
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
DP = AUDIT.parent
ORIG = AUDIT / "FIGURE_REFINEMENT_PHASE" / "source_tables" / "Figure_7_source.csv"
RUN = DP / "Docs" / "major_revision_analyses" / "corrected_true_BiLSTM_rerun_20260728" / "08_downstream" / "corrected_combined_pixel_metrics.csv"
RUN2 = DP / "Docs" / "major_revision_analyses" / "05_forecasting_raster_reviewer_safe_candidate" / "tables" / "reviewer_safe_all_pixel_metrics.csv"
OUT_DIR = HERE / "source_tables"
OUT = OUT_DIR / "Figure_7_source.csv"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    src = pd.read_csv(ORIG)
    rows = pd.read_csv(RUN)
    rows = rows[(rows.model == "BiLSTM") & (rows.variable == "TWS")].copy()
    alt = pd.read_csv(RUN2)
    alt = alt[(alt.model == "BiLSTM") & (alt.variable == "TWS")].copy()
    key = ["land_type", "fold", "seed"]
    a = rows.sort_values(key).reset_index(drop=True)
    b = alt.sort_values(key).reset_index(drop=True)
    assert len(a) == 16 and a[key].astype(str).equals(b[key].astype(str))
    assert np.array_equal(a.r2.values, b.r2.values) and np.array_equal(a.normalized_rmse.values, b.normalized_rmse.values)
    assert set(a.lookback) == {12} and set(a.preprocessing) == {"reviewer_mode_b"}
    assert set(a["mode"]) == {"reviewer_safe_rolling_origin"} and set(a.status) == {"success"}

    out = src.copy()
    log = []
    for lt, g in a.groupby("land_type"):
        assert sorted(g.fold) == [1, 2, 3, 4]
        idx = out.index[(out.model == "BiLSTM") & (out.target_internal == "TWS") & (out.land_type == lt)]
        assert len(idx) == 1
        i = idx[0]
        old = out.loc[i].copy()
        new = {
            "selected_configuration": "common_reviewer_safe",
            "mean_r2": g.r2.mean(),
            "sd_r2": g.r2.std(ddof=1),
            "mean_normalized_rmse": g.normalized_rmse.mean(),
            "sd_normalized_rmse": g.normalized_rmse.std(ddof=1),
            "available_folds": g.fold.nunique(),
            "available_rows": len(g),
            "available_seeds": g.seed.nunique(),
            "data_status": "replaced_with_common_12month_benchmark",
            "source_metric_table": str(RUN),
        }
        new["r2_negative_flag"] = bool(new["mean_r2"] < 0)
        new["r2_plot_radius"] = max(new["mean_r2"], 0.0)
        new["nrmse_plot_radius"] = new["mean_normalized_rmse"]
        for k, v in new.items():
            out.at[i, k] = v
        for k in ["selected_configuration", "mean_r2", "sd_r2", "mean_normalized_rmse", "sd_normalized_rmse", "available_rows", "available_seeds"]:
            log.append({"land_type": lt, "target": "TWS", "model": "BiLSTM", "field": k, "old": old[k], "new": new[k]})

    # every cell must now be on the common 12-month benchmark
    assert len(out) == 216
    assert not out.selected_configuration.eq("bilstm_special_tws_state_space").any()
    unchanged = out.drop(columns=[]).copy()
    mask = ~((out.model == "BiLSTM") & (out.target_internal == "TWS"))
    pd.testing.assert_frame_equal(out[mask].reset_index(drop=True), src[mask].reset_index(drop=True))

    OUT_DIR.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    pd.DataFrame(log).to_csv(HERE / "bilstm_tws_cell_changes.csv", index=False)
    a.to_csv(HERE / "bilstm_tws_common12_fold_seed_rows.csv", index=False)

    def model_means(d):
        return d.groupby("model")[["mean_r2", "mean_normalized_rmse"]].mean()
    mm = model_means(src).join(model_means(out), lsuffix="_old", rsuffix="_new")
    mm.to_csv(HERE / "model_level_means_old_vs_new.csv")
    tt = src.groupby("target_display")[["mean_r2", "mean_normalized_rmse"]].mean().join(
        out.groupby("target_display")[["mean_r2", "mean_normalized_rmse"]].mean(), lsuffix="_old", rsuffix="_new")
    tt.to_csv(HERE / "target_level_means_old_vs_new.csv")
    pd.DataFrame([
        {"file": "original Figure_7_source.csv", "path": str(ORIG), "sha256": sha(ORIG)},
        {"file": "BiLSTM-TWS 12-month metric rows", "path": str(RUN), "sha256": sha(RUN)},
        {"file": "cross-check (identical rows)", "path": str(RUN2), "sha256": sha(RUN2)},
        {"file": "corrected Figure_7_source.csv", "path": str(OUT), "sha256": sha(OUT)},
    ]).to_csv(HERE / "provenance.csv", index=False)
    print(pd.DataFrame(log).to_string())
    print(mm.round(4).to_string())
    print(tt.round(4).to_string())


if __name__ == "__main__":
    main()
