"""Rebuild the forecasting summaries from the saved per-run metric tables named in config/forecast_result_registry.csv.

Two variants are built from the same raw rows (no model is run):
  main_common_12m              every cell from a main common-benchmark configuration (12-month lookback); this is
                               the design described in the manuscript.
  published_aug2026_18m_tws    identical except the four BiLSTM x TWS cells come from the 18-month hydroclimate
                               sensitivity (sens_tws_bilstm_18m_state_space); reproduces the values that the
                               August-2026 Figure 7 / Tables 4-5 / statistical comparison were built from.

For each variant: 216 task cells (Figure 7 source), model means and win counts (Table 4), target means and best
models (Table 5), Friedman and Wilcoxon-Holm tests (Supplementary statistical-comparison table). Outputs go to a
new directory; nothing existing is overwritten. A manifest records SHA-256 hashes of every input, the code and the
source Git commit.

Usage:  python src/tables/rebuild_main_benchmark_from_runs.py --project-root "<PROJECT_ROOT>" [--out DIR]
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _n in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_n, "1")

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "forecasting"))
sys.path.insert(0, str(REPO / "src" / "statistics"))
from forecast_model_comparison import compare_models  # noqa: E402
from provenance import aggregate_cells, build_from_registry, load_registry  # noqa: E402

TARGET_DISPLAY = {"GNDVI": "GNDVI", "GOSIF": "GOSIF/SIF", "GPP": "GPP", "LST": "LST", "NDVI": "NDVI", "NIRv": "NIRv",
                  "SPEI": "SPEI-1", "TWS": "TWS", "VPD": "VPDmax"}
LAND_DISPLAY = {"GRASSLANDS": "Grasslands", "NATURAL_FORESTS": "Natural Forests", "SAVANNAS": "Savannas", "SHRUBLANDS": "Shrublands"}
MODEL_ORDER = ["XGBoost", "BiLSTM", "TCN", "N-BEATS", "LSTM", "N-HiTS"]
TARGET_ORDER = ["NDVI", "GNDVI", "NIRv", "GOSIF/SIF", "GPP", "LST", "VPDmax", "SPEI-1", "TWS"]
VARIANTS = {
    "main_common_12m": {},
    "published_aug2026_18m_tws": {"main_reviewer_safe_bilstm_tws": "sens_tws_bilstm_18m_state_space"},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(repo: Path) -> str:
    """Resolve HEAD from the .git metadata without running git."""
    git = repo / ".git"
    gitdir = Path(git.read_text().split("gitdir:", 1)[1].strip()) if git.is_file() else git
    head = (gitdir / "HEAD").read_text().strip()
    if not head.startswith("ref:"):
        return head
    ref = head.split(" ", 1)[1]
    for base in (gitdir, gitdir.parent.parent if gitdir.parent.name == "worktrees" else gitdir):
        if (base / ref).is_file():
            return (base / ref).read_text().strip() + f" ({ref})"
        packed = base / "packed-refs"
        if packed.is_file():
            for line in packed.read_text().splitlines():
                if line.endswith(" " + ref):
                    return line.split()[0] + f" ({ref})"
    return f"unresolved ({ref})"


def summarize(cells: pd.DataFrame):
    t = cells.assign(target=cells["variable"].map(TARGET_DISPLAY), land=cells["land_type"].map(LAND_DISPLAY))
    task = ["land", "target"]
    r2w = t.loc[t.groupby(task)["mean_r2"].idxmax(), "model"].value_counts()
    nrw = t.loc[t.groupby(task)["mean_normalized_rmse"].idxmin(), "model"].value_counts()
    mm = t.groupby("model")[["mean_r2", "mean_normalized_rmse"]].mean()
    table4 = pd.DataFrame([{"model": m, "best_r2_count": int(r2w.get(m, 0)), "best_nrmse_count": int(nrw.get(m, 0)),
                            "mean_r2": mm.loc[m, "mean_r2"], "mean_nrmse": mm.loc[m, "mean_normalized_rmse"]} for m in MODEL_ORDER])
    tm = t.groupby("target")[["mean_r2", "mean_normalized_rmse"]].mean()
    btm = t.groupby(["target", "model"])[["mean_r2", "mean_normalized_rmse"]].mean().reset_index()
    br2 = btm.loc[btm.groupby("target")["mean_r2"].idxmax()].set_index("target")["model"]
    bnr = btm.loc[btm.groupby("target")["mean_normalized_rmse"].idxmin()].set_index("target")["model"]
    table5 = pd.DataFrame([{"target": x, "mean_r2": tm.loc[x, "mean_r2"], "mean_nrmse": tm.loc[x, "mean_normalized_rmse"],
                            "best_r2_model": br2[x], "best_nrmse_model": bnr[x]} for x in TARGET_ORDER])
    stats_in = t.rename(columns={"land": "land_type_display"})[["target", "land_type_display", "model", "mean_r2", "mean_normalized_rmse"]]
    friedman, pairs = compare_models(stats_in.rename(columns={"land_type_display": "land_type"}))
    return t, table4, table5, friedman, pairs


def compare_reference(cells: pd.DataFrame, ref_path: Path) -> dict:
    if not ref_path.is_file():
        return {"reference": str(ref_path), "status": "unavailable"}
    ref = pd.read_csv(ref_path)
    if "target_internal" in ref.columns:
        ref = ref.rename(columns={"target_internal": "variable"})
    else:
        inv_t = {v: k for k, v in TARGET_DISPLAY.items()}
        inv_l = {v: k for k, v in LAND_DISPLAY.items()}
        ref = ref.assign(variable=ref["target"].map(inv_t), land_type=ref["land_type"].map(lambda x: inv_l.get(x, x)))
    m = cells.merge(ref, on=["model", "variable", "land_type"], suffixes=("", "_ref"), validate="one_to_one")
    diffs = {c: float((m[c] - m[c + "_ref"]).abs().max()) for c in ["mean_r2", "sd_r2", "mean_normalized_rmse", "sd_normalized_rmse"]}
    return {"reference": str(ref_path), "sha256": sha256(ref_path), "cells_matched": len(m), "max_abs_difference": diffs,
            "identical": len(m) == 216 and max(diffs.values()) < 1e-12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True, help="folder containing Docs/major_revision_analyses")
    ap.add_argument("--out", default=str(REPO / "results" / "audit" / "forecast_provenance"))
    args = ap.parse_args()
    root, out = Path(args.project_root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry_path = REPO / "config" / "forecast_result_registry.csv"
    registry = load_registry(registry_path)

    references = {
        "main_common_12m": REPO / "results" / "forecasting" / "figure7_final_task_metrics.csv",
        "published_aug2026_18m_tws": root / "AUDIT_TEMP" / "FIGURE_REFINEMENT_PHASE" / "source_tables" / "Figure_7_source.csv",
    }
    results, manifest_inputs = {}, {}
    for name, subs in VARIANTS.items():
        rows = build_from_registry(registry, root, substitutions=subs)
        cells = aggregate_cells(rows)
        t, table4, table5, friedman, pairs = summarize(cells)
        d = out / name
        d.mkdir(exist_ok=True)
        t.to_csv(d / "task_cells_216.csv", index=False)
        table4.to_csv(d / "model_summary_table4.csv", index=False)
        table5.to_csv(d / "target_summary_table5.csv", index=False)
        friedman.to_csv(d / "friedman_tests.csv", index=False)
        pairs.to_csv(d / "pairwise_wilcoxon_holm.csv", index=False)
        cov = rows.groupby(["configuration_id", "role", "lookback", "preprocessing", "scaling"]).agg(
            rows=("r2", "size"), models=("model", lambda s: ";".join(sorted(set(s)))),
            targets=("variable", lambda s: ";".join(sorted(set(s)))), seeds=("seed", lambda s: ";".join(map(str, sorted(set(map(int, s)))))),
            folds=("fold", lambda s: ";".join(map(str, sorted(set(map(int, s))))))).reset_index()
        cov.to_csv(d / "configuration_coverage.csv", index=False)
        results[name] = {"cells": t, "table4": table4, "table5": table5, "friedman": friedman, "pairs": pairs,
                         "reference_check": compare_reference(cells, references[name])}
        for cid, g in rows.groupby("configuration_id"):
            src = registry.set_index("configuration_id").loc[cid, "source_table"]
            manifest_inputs.setdefault(src, {"path": src, "sha256": sha256(root / src), "bytes": (root / src).stat().st_size, "used_rows": {}})
            manifest_inputs[src]["used_rows"][cid] = int(len(g))

    # before/after: published (18-month TWS cells) -> main (common 12-month)
    a, b = results["published_aug2026_18m_tws"], results["main_common_12m"]
    rows = []
    for col_tab, key in [("table4", "model"), ("table5", "target")]:
        m = a[col_tab].merge(b[col_tab], on=key, suffixes=("_published", "_main"))
        for _, r in m.iterrows():
            for c in [c[:-10] for c in m.columns if c.endswith("_published")]:
                old, new = r[c + "_published"], r[c + "_main"]
                changed = (old != new) if isinstance(old, str) else not np.isclose(old, new, rtol=0, atol=1e-12)
                if changed:
                    rows.append({"output": col_tab, "row": r[key], "quantity": c, "published_18m_tws": old, "main_12m": new,
                                 "published_3dp": old if isinstance(old, str) else round(old, 3), "main_3dp": new if isinstance(new, str) else round(new, 3)})
    cm = a["cells"].merge(b["cells"], on=["model", "variable", "land_type"], suffixes=("_published", "_main"))
    for _, r in cm.iterrows():
        for c in ["mean_r2", "sd_r2", "mean_normalized_rmse", "sd_normalized_rmse"]:
            if not np.isclose(r[c + "_published"], r[c + "_main"], rtol=0, atol=1e-12):
                rows.append({"output": "figure7_cell", "row": f"{r['model']} | {r['variable']} | {r['land_type']}", "quantity": c,
                             "published_18m_tws": r[c + "_published"], "main_12m": r[c + "_main"],
                             "published_3dp": round(r[c + "_published"], 3), "main_3dp": round(r[c + "_main"], 3)})
    for tab in ["friedman", "pairs"]:
        x, y = a[tab], b[tab]
        keys = ["metric"] if tab == "friedman" else ["metric", "model_a", "model_b"]
        m = x.merge(y, on=keys, suffixes=("_published", "_main"))
        for _, r in m.iterrows():
            for c in [c[:-10] for c in m.columns if c.endswith("_published")]:
                o, n = r[c + "_published"], r[c + "_main"]
                if isinstance(o, (bool, np.bool_)) or isinstance(o, str):
                    ch = o != n
                else:
                    ch = not np.isclose(o, n, rtol=0, atol=1e-12)
                if ch:
                    rows.append({"output": tab, "row": " | ".join(str(r[k]) for k in keys), "quantity": c,
                                 "published_18m_tws": o, "main_12m": n,
                                 "published_3dp": o if isinstance(o, (str, bool, np.bool_)) else round(float(o), 3),
                                 "main_3dp": n if isinstance(n, (str, bool, np.bool_)) else round(float(n), 3)})
    pd.DataFrame(rows).to_csv(out / "before_after_published_vs_main.csv", index=False)

    code = [REPO / "src" / "tables" / "rebuild_main_benchmark_from_runs.py", REPO / "src" / "forecasting" / "provenance.py",
            REPO / "src" / "statistics" / "forecast_model_comparison.py", registry_path]
    manifest = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_git_commit": git_commit(REPO),
        "note": "Working-tree files may differ from the commit; their SHA-256 hashes are listed under code.",
        "project_root": "<PROJECT_ROOT> (paths below are relative to it)",
        "inputs": list(manifest_inputs.values()),
        "code": [{"path": str(p.relative_to(REPO)).replace("\\", "/"), "sha256": sha256(p)} for p in code],
        "reference_checks": {k: v["reference_check"] for k, v in results.items()},
        "aggregation": "cell = mean (SD ddof=1) over fold x seed runs of the pixel-month-pooled metric; folds balanced; "
                       "model/target summaries = unweighted means over task cells",
        "outputs": [],
    }
    for p in sorted(out.rglob("*.csv")):
        manifest["outputs"].append({"path": str(p.relative_to(out)).replace("\\", "/"), "sha256": sha256(p)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for k, v in results.items():
        t4 = v["table4"].set_index("model")
        print(f"{k}: BiLSTM mean R2 {t4.loc['BiLSTM', 'mean_r2']:.4f}, nRMSE {t4.loc['BiLSTM', 'mean_nrmse']:.4f}; "
              f"TWS {v['table5'].set_index('target').loc['TWS', 'mean_r2']:.4f}/{v['table5'].set_index('target').loc['TWS', 'mean_nrmse']:.4f}; "
              f"Friedman R2 {v['friedman'].iloc[0]['chi2']:.3f}; Holm-sig {int(v['pairs']['significant_holm_0.05'].sum())}; "
              f"reference identical: {v['reference_check'].get('identical')}")


if __name__ == "__main__":
    main()
