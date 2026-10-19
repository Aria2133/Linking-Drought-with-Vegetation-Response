"""Recompute Table 4 (model level) and Table 5 (target level) from the corrected Figure 7 source with the
same logic as FINAL_TABLE_CHECK/verify_tables_2_4_5_6.py (compute_table4_and_table5), and diff against the
previously verified tables."""
import os
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_n] = "1"

import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source_tables" / "Figure_7_source.csv"
OLD = HERE.parent / "FINAL_TABLE_CHECK"
TARGET_ORDER = ["NDVI", "GNDVI", "NIRv", "GOSIF/SIF", "GPP", "LST", "VPDmax", "SPEI-1", "TWS"]
MODEL_ORDER = ["XGBoost", "BiLSTM", "TCN", "N-BEATS", "LSTM", "N-HiTS"]

source = pd.read_csv(SOURCE)
task = ["land_type_display", "target_display"]
r2w = source.loc[source.groupby(task).mean_r2.idxmax()].model.value_counts()
nrw = source.loc[source.groupby(task).mean_normalized_rmse.idxmin()].model.value_counts()
mm = source.groupby("model").agg(r2=("mean_r2", "mean"), nr=("mean_normalized_rmse", "mean"))
t4 = pd.DataFrame([{
    "Model": m, "Model family": "Non-deep machine learning" if m == "XGBoost" else "Deep learning",
    "Best R² count": int(r2w.get(m, 0)), "Best nRMSE count": int(nrw.get(m, 0)),
    "Mean R²": f"{mm.loc[m, 'r2']:.3f}", "Mean nRMSE": f"{mm.loc[m, 'nr']:.3f}"} for m in MODEL_ORDER])

tm = source.groupby("target_display").agg(r2=("mean_r2", "mean"), nr=("mean_normalized_rmse", "mean"))
btm = source.groupby(["target_display", "model"]).agg(r2=("mean_r2", "mean"), nr=("mean_normalized_rmse", "mean")).reset_index()
br2 = btm.loc[btm.groupby("target_display").r2.idxmax()].set_index("target_display").model
bnr = btm.loc[btm.groupby("target_display").nr.idxmin()].set_index("target_display").model
cls = {**{t: "High" for t in TARGET_ORDER[:4]}, **{t: "Moderate" for t in ["GPP", "LST", "VPDmax"]}, "SPEI-1": "Low/variable", "TWS": "Low/variable"}
rows = []
for t in TARGET_ORDER:
    v = tm.loc[t, "r2"]
    assert ("High" if v >= 0.70 else "Moderate" if v >= 0.40 else "Low/variable") == cls[t], t
    rows.append({"Predictability class": cls[t], "Target": t, "Mean R²": f"{v:.3f}", "Mean nRMSE": f"{tm.loc[t, 'nr']:.3f}",
                 "Best-R² model": br2[t], "Best-nRMSE model": bnr[t]})
t5 = pd.DataFrame(rows)
t4.to_csv(HERE / "TABLE_4_CORRECTED.csv", index=False, encoding="utf-8-sig")
t5.to_csv(HERE / "TABLE_5_CORRECTED.csv", index=False, encoding="utf-8-sig")

diffs = []
for name, new in [("Table 4", t4), ("Table 5", t5)]:
    old = pd.read_csv(OLD / f"TABLE_{name[-1]}_VERIFIED.csv", encoding="utf-8-sig", dtype=str)
    new = new.astype(str)
    for i in range(len(new)):
        for c in new.columns:
            if old.iloc[i][c] != new.iloc[i][c]:
                diffs.append({"table": name, "row": new.iloc[i, 0] if name == "Table 4" else new.iloc[i]["Target"],
                              "column": c, "old": old.iloc[i][c], "new": new.iloc[i][c]})
pd.DataFrame(diffs).to_csv(HERE / "TABLE_4_5_VALUE_CHANGES.csv", index=False, encoding="utf-8-sig")
print(t4.to_string()); print(t5.to_string()); print(pd.DataFrame(diffs).to_string())
