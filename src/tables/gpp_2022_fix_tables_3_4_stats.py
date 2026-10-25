"""Tables 3-4 (model ranking, target predictability) and the Friedman / Wilcoxon-Holm comparison from a Figure 7 source
table, with the same summaries as rebuild_main_benchmark_from_runs.py. Run on the accepted source first (must reproduce
the accepted values) and then on the source with the corrected GPP cells.

Usage: python gpp_2022_fix_tables_3_4_stats.py --source <Figure_7_source.csv> --out <dir>
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rebuild_main_benchmark_from_runs import summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
import pandas as pd  # noqa: E402

src = pd.read_csv(a.source)
cells = src.rename(columns={"target_internal": "variable"})[["land_type", "variable", "model", "mean_r2", "mean_normalized_rmse"]]
t, table4, table5, friedman, pairs = summarize(cells)
table4.to_csv(a.out / "table3_model_summary.csv", index=False)
table5.to_csv(a.out / "table4_target_summary.csv", index=False)
friedman.to_csv(a.out / "friedman_tests.csv", index=False)
pairs.to_csv(a.out / "pairwise_wilcoxon_holm.csv", index=False)
print(table4.round(3).to_string(index=False)); print(table5.round(3).to_string(index=False))
print(friedman[["metric", "chi2", "p_value", "kendall_W"]].to_string(index=False))
