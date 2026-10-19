"""Format the completed forecast-inference outputs without rerunning inference."""

from __future__ import annotations

import hashlib
import os
import csv
from datetime import datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
OUT = HERE.parent
PAPER = OUT.parents[1]
# Copy of FORECAST_STATISTICAL_TABLE_FINAL/scripts/build_forecast_pairwise_table.py. Changes: SOURCE_DIR points to the
# task-2 rerun on the corrected Figure 7 source, and the Table 4 gate reads the recomputed TABLE_4_CORRECTED.csv.
SOURCE_DIR = OUT.parent / "TASK2_INFERENCE"

FILES = [
    "forecast_pairwise_tests_holm.csv",
    "forecast_effect_sizes.csv",
    "forecast_friedman_tests.csv",
    "forecast_model_rank_summary.csv",
    "forecast_model_task_metrics_corrected.csv",
    "FORECAST_STATISTICAL_INFERENCE.md",
]
MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
METRIC_ORDER = {"r2": 0, "nrmse": 1}
METRIC_DISPLAY = {"r2": "R²", "nrmse": "nRMSE"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def format_p(value: float) -> str:
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.3f}"


def format_exact(value: float) -> str:
    return format(float(value), ".17g")


def direction_text(metric: str, favored_model: str) -> str:
    return f"{favored_model} higher R²" if metric == "r2" else f"{favored_model} lower nRMSE"


def main() -> None:
    missing = [name for name in FILES if not (SOURCE_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing authoritative file(s): " + ", ".join(missing))

    pairs = pd.read_csv(SOURCE_DIR / "forecast_pairwise_tests_holm.csv")
    effects = pd.read_csv(SOURCE_DIR / "forecast_effect_sizes.csv")
    friedman = pd.read_csv(SOURCE_DIR / "forecast_friedman_tests.csv")
    ranks = pd.read_csv(SOURCE_DIR / "forecast_model_rank_summary.csv")
    tasks = pd.read_csv(SOURCE_DIR / "forecast_model_task_metrics_corrected.csv")

    with (SOURCE_DIR / "forecast_pairwise_tests_holm.csv").open("r", encoding="utf-8", newline="") as handle:
        pairwise_raw_rows = list(csv.DictReader(handle))
    with (SOURCE_DIR / "forecast_effect_sizes.csv").open("r", encoding="utf-8", newline="") as handle:
        effect_raw_rows = list(csv.DictReader(handle))
    with (SOURCE_DIR / "forecast_friedman_tests.csv").open("r", encoding="utf-8", newline="") as handle:
        friedman_raw_rows = list(csv.DictReader(handle))
    pairwise_raw = {(row["metric"], row["model_a"], row["model_b"]): row for row in pairwise_raw_rows}
    effect_raw = {(row["metric"], row["model_a"], row["model_b"]): row for row in effect_raw_rows}
    friedman_raw = {row["metric"]: row for row in friedman_raw_rows}

    # Structural/provenance gates. These verify stored outputs; no test is recomputed.
    assert len(pairs) == 30
    assert pairs.groupby("metric").size().to_dict() == {"nrmse": 15, "r2": 15}
    assert len(effects) == 30
    assert len(friedman) == 2
    assert len(tasks) == 216
    assert tasks["task_id"].nunique() == 36
    assert tasks["model"].nunique() == 6
    assert set(tasks["model"]) == set(MODELS)
    assert not tasks.duplicated(["task_id", "model"]).any()
    assert tasks.groupby("task_id")["model"].nunique().eq(6).all()
    assert set(tasks["available_folds"]) == {4}
    assert pairs["n_paired_tasks"].eq(36).all()
    assert effects["n_paired_tasks"].eq(36).all()
    assert friedman["n_paired_tasks"].eq(36).all()
    assert friedman["n_models"].eq(6).all()
    assert pairs["significant_holm_0_05"].eq(pairs["p_value_holm"].lt(0.05)).all()

    joined = pairs.merge(
        effects,
        on=["metric", "model_a", "model_b", "n_paired_tasks"],
        how="left",
        validate="one_to_one",
        suffixes=("_pairwise", "_effect"),
    )
    embedded_effect = joined["rank_biserial_effect_positive_favors_a_pairwise"]
    joined_effect = joined["rank_biserial_effect_positive_favors_a_effect"]
    assert embedded_effect.equals(joined_effect)

    # Verify direction using the paired task values, not model-level means.
    direct_direction_checks = []
    for row in joined.itertuples(index=False):
        metric_column = "mean_r2" if row.metric == "r2" else "mean_nrmse"
        matrix = tasks.pivot(index="task_id", columns="model", values=metric_column)
        raw = matrix[row.model_a] - matrix[row.model_b]
        oriented = raw if row.metric == "r2" else -raw
        counts = (int((oriented > 0).sum()), int((oriented == 0).sum()), int((oriented < 0).sum()))
        expected = (int(row.tasks_favoring_a), int(row.tasks_tied), int(row.tasks_favoring_b))
        assert counts == expected
        effect = float(row.rank_biserial_effect_positive_favors_a_effect)
        favored = row.model_a if effect > 0 else row.model_b if effect < 0 else "Tie"
        direct_direction_checks.append(favored)
    joined["verified_favored_model"] = direct_direction_checks
    assert joined["verified_favored_model"].eq(joined["favored_model_by_rank_biserial"]).all()

    # Figure 7/Table 4 descriptive consistency gate.
    _t4 = pd.read_csv(OUT.parent / "TABLE_4_CORRECTED.csv", encoding="utf-8-sig")
    expected_table4 = {
        row["Model"]: (int(row["Best R² count"]), int(row["Best nRMSE count"]), float(row["Mean R²"]), float(row["Mean nRMSE"]))
        for _, row in _t4.iterrows()
    }
    table4_checks = []
    for model, expected in expected_table4.items():
        r2 = ranks[(ranks["metric"].eq("r2")) & (ranks["model"].eq(model))].iloc[0]
        nr = ranks[(ranks["metric"].eq("nrmse")) & (ranks["model"].eq(model))].iloc[0]
        observed = (
            int(r2["task_win_count"]),
            int(nr["task_win_count"]),
            round(float(r2["mean"]), 3),
            round(float(nr["mean"]), 3),
        )
        assert observed == expected
        table4_checks.append(
            {
                "model": model,
                "best_r2_count": observed[0],
                "best_nrmse_count": observed[1],
                "mean_r2_3dp": f"{observed[2]:.3f}",
                "mean_nrmse_3dp": f"{observed[3]:.3f}",
                "matches_final_figure7_table4": True,
            }
        )
    pd.DataFrame(table4_checks).to_csv(OUT / "FIGURE7_TABLE4_CONSISTENCY_CHECK.csv", index=False)

    model_order = {model: index for index, model in enumerate(MODELS)}
    joined["_metric_order"] = joined["metric"].map(METRIC_ORDER)
    joined["_model_a_order"] = joined["model_a"].map(model_order)
    joined["_model_b_order"] = joined["model_b"].map(model_order)
    joined = joined.sort_values(["_metric_order", "_model_a_order", "_model_b_order"]).reset_index(drop=True)

    joined["p_value_raw_exact_text"] = [
        pairwise_raw[(metric, model_a, model_b)]["p_value_raw"]
        for metric, model_a, model_b in zip(joined["metric"], joined["model_a"], joined["model_b"])
    ]
    joined["p_value_holm_exact_text"] = [
        pairwise_raw[(metric, model_a, model_b)]["p_value_holm"]
        for metric, model_a, model_b in zip(joined["metric"], joined["model_a"], joined["model_b"])
    ]
    joined["effect_exact_text"] = [
        effect_raw[(metric, model_a, model_b)]["rank_biserial_effect_positive_favors_a"]
        for metric, model_a, model_b in zip(joined["metric"], joined["model_a"], joined["model_b"])
    ]

    full = pd.DataFrame(
        {
            "Metric": joined["metric"].map(METRIC_DISPLAY),
            "Model 1": joined["model_a"],
            "Model 2": joined["model_b"],
            "Wilcoxon statistic": [
                pairwise_raw[(metric, model_a, model_b)]["wilcoxon_W"]
                for metric, model_a, model_b in zip(joined["metric"], joined["model_a"], joined["model_b"])
            ],
            "Unadjusted p": joined["p_value_raw_exact_text"],
            "Holm-adjusted p": joined["p_value_holm_exact_text"],
            "Significant after Holm correction": joined["significant_holm_0_05"],
            "Paired effect size": joined["effect_exact_text"],
        }
    )
    full.to_csv(
        OUT / "Supplementary_Table_Forecast_Pairwise_Statistical_Comparison_full_precision.csv",
        index=False,
    )

    display = full.copy()
    display["Wilcoxon statistic"] = display["Wilcoxon statistic"].map(lambda value: str(int(float(value))))
    display["Unadjusted p"] = display["Unadjusted p"].map(lambda value: format_p(float(value)))
    display["Holm-adjusted p"] = display["Holm-adjusted p"].map(lambda value: format_p(float(value)))
    display["Significant after Holm correction"] = display["Significant after Holm correction"].map({True: "Yes", False: "No"})
    display["Paired effect size"] = display["Paired effect size"].map(lambda value: f"{float(value):.3f}")
    display.to_csv(OUT / "Supplementary_Table_Forecast_Pairwise_Statistical_Comparison.csv", index=False)

    significant = joined[joined["significant_holm_0_05"]].copy()
    significant_output = pd.DataFrame(
        {
            "Metric": significant["metric"].map(METRIC_DISPLAY),
            "Model 1": significant["model_a"],
            "Model 2": significant["model_b"],
            "Holm-adjusted p": significant["p_value_holm_exact_text"],
            "Effect size": significant["effect_exact_text"],
            "Direction of better performance": [
                direction_text(metric, favored)
                for metric, favored in zip(significant["metric"], significant["verified_favored_model"])
            ],
        }
    )
    significant_output.to_csv(OUT / "forecast_holm_significant_pairs_only.csv", index=False)

    title = "Table Sx. Paired statistical comparison of forecasting models across 36 land-type × target tasks."
    note = (
        "Note: Pairwise comparisons were performed using two-sided Wilcoxon signed-rank tests across the 36 common "
        "land-type × target tasks. Holm correction was applied separately to the 15 learned-model comparisons within "
        "each metric. Individual pixels were not treated as independent inferential replicates. R² is higher-is-better, "
        "whereas nRMSE is lower-is-better. The paired effect size is the matched-pairs rank-biserial correlation, oriented "
        "so that positive values favor Model 1 and negative values favor Model 2."
    )
    table_lines = [title, "", "\t".join(display.columns)]
    table_lines.extend("\t".join(map(str, row)) for row in display.itertuples(index=False, name=None))
    table_lines.extend(["", note])
    (OUT / "Supplementary_Table_Forecast_Pairwise_Statistical_Comparison_manuscript.txt").write_text(
        "\n".join(table_lines) + "\n", encoding="utf-8"
    )

    fr2 = friedman[friedman["metric"].eq("r2")].iloc[0]
    fnr = friedman[friedman["metric"].eq("nrmse")].iloc[0]
    fr2_raw = friedman_raw["r2"]
    fnr_raw = friedman_raw["nrmse"]
    result_paragraph = (
        "Formal task-level comparison indicated overall differences among the six learned models for both R² "
        f"(Friedman test, χ²({fr2_raw['degrees_of_freedom']}) = {fr2_raw['friedman_chi_square']}, "
        f"p = {fr2_raw['p_value']}) and nRMSE "
        f"(χ²({fnr_raw['degrees_of_freedom']}) = {fnr_raw['friedman_chi_square']}, "
        f"p = {fnr_raw['p_value']}). Pairwise Wilcoxon signed-rank tests with Holm correction showed that, "
        "for R², BiLSTM exceeded LSTM and TCN, and LSTM exceeded TCN; for nRMSE, BiLSTM was lower than LSTM and TCN, "
        "LSTM was lower than TCN, and N-BEATS, N-HiTS, and XGBoost were each lower than TCN. No other learned-model "
        "contrast remained significant after Holm correction."
    )
    conservative = (
        "Although the omnibus tests indicated overall model differences, only a subset of pairwise contrasts remained "
        "significant after Holm correction; descriptive differences in mean skill and task-level win counts therefore "
        "should not be interpreted as universal model superiority."
    )
    (OUT / "MANUSCRIPT_READY_RESULTS_PARAGRAPH.txt").write_text(
        result_paragraph + "\n\n" + conservative + "\n", encoding="utf-8"
    )

    metadata_rows = []
    for name in FILES:
        path = SOURCE_DIR / name
        stat = path.stat()
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            columns = "; ".join(frame.columns)
            row_count = len(frame)
        else:
            columns = "Not applicable (Markdown report)"
            row_count = len(path.read_text(encoding="utf-8").splitlines())
        metadata_rows.append(
            {
                "file": name,
                "exact_path": str(path.resolve()),
                "bytes": stat.st_size,
                "modification_timestamp_filesystem_local": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                "columns": columns,
                "rows_or_text_lines": row_count,
                "sha256": sha256(path),
            }
        )
    metadata = pd.DataFrame(metadata_rows)
    metadata.to_csv(OUT / "AUTHORITATIVE_INPUT_FILE_MANIFEST.csv", index=False)

    sig_lines = []
    for row in significant_output.itertuples(index=False):
        sig_lines.append(
            f"- {row.Metric}: {row[1]} vs {row[2]}; Holm-adjusted p = {row[3]}; "
            f"rank-biserial effect = {row[4]}; {row[5]}."
        )

    file_table_lines = [
        "| File | Exact path | Bytes | Modified | Rows/lines | SHA-256 |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in metadata.itertuples(index=False):
        file_table_lines.append(
            f"| `{row.file}` | `{row.exact_path}` | {row.bytes} | {row.modification_timestamp_filesystem_local} | "
            f"{row.rows_or_text_lines} | `{row.sha256}` |"
        )

    schema_lines = []
    for row in metadata.itertuples(index=False):
        schema_lines.append(f"- `{row.file}`: {row.columns}")

    report = f"""# Forecast pairwise-table verification report

## 1. Authoritative files and provenance

The authoritative set is the internally consistent output bundle in `AUDIT_TEMP/REVISION_COMPUTATIONAL_PHASE/task2_forecast_inference`. Each requested CSV name occurs once within `REVISION_COMPUTATIONAL_PHASE`. A broader root-level Markdown report also exists, but it summarizes additional model sets and sensitivity analyses; it was not substituted for the task-specific six-learned-model output bundle.

The task-2 rerun reads `AUDIT_TEMP/FIGURE7_TWS_COMMON_CORRECTION/source_tables/Figure_7_source.csv` (the accepted Figure 7 source with the four BiLSTM x TWS cells replaced by the common 12-month benchmark rows), SHA-256 `a43983e5390f5c2b96a8e70ce170330e03a222d16f6aaa7bccdf51625e733884`.

{chr(10).join(file_table_lines)}

### Columns

{chr(10).join(schema_lines)}

## 2. Friedman omnibus results

- R²: χ²({fr2_raw['degrees_of_freedom']}) = {fr2_raw['friedman_chi_square']}; exact stored p = {fr2_raw['p_value']}; 36 paired tasks; 6 models; Kendall's W = {fr2_raw['kendalls_W']}.
- nRMSE: χ²({fnr_raw['degrees_of_freedom']}) = {fnr_raw['friedman_chi_square']}; exact stored p = {fnr_raw['p_value']}; 36 paired tasks; 6 models; Kendall's W = {fnr_raw['kendalls_W']}.

## 3. Pairwise comparisons

- Expected and observed: 15 model pairs × 2 metrics = 30 rows.
- R² Holm-significant comparisons: {int(significant['metric'].eq('r2').sum())}.
- nRMSE Holm-significant comparisons: {int(significant['metric'].eq('nrmse').sum())}.
- Nonsignificant comparisons: {int((~joined['significant_holm_0_05']).sum())} total ({int(((joined['metric'].eq('r2')) & (~joined['significant_holm_0_05'])).sum())} R²; {int(((joined['metric'].eq('nrmse')) & (~joined['significant_holm_0_05'])).sum())} nRMSE).

### Exact Holm-significant pairs

{chr(10).join(sig_lines)}

## 4. Effect-size definition and direction

The effect size is the matched-pairs rank-biserial correlation from the existing analysis. Differences were performance-oriented before ranking: Model 1 minus Model 2 for R², and the negative of Model 1 minus Model 2 for nRMSE. Therefore, for both metrics, a positive effect favors Model 1 and a negative effect favors Model 2. In raw metric terms, favored performance means higher R² but lower nRMSE. The sign was retained exactly; it was not reoriented during table formatting.

Effect sizes joined one-to-one across all 30 metric/model-pair rows. Direction labels were independently verified against the 36 paired task values and the stored counts of tasks favoring each model; model-level means were not used to assign direction.

## 5. Figure 7 and Table 4 consistency

All supplied final descriptive values match `forecast_model_rank_summary.csv`: XGBoost 17/18 wins and means 0.633/0.551; BiLSTM 7/7 and 0.661/0.538; TCN 6/4 and 0.559/0.608; N-BEATS 2/4 and 0.643/0.547; LSTM 4/3 and 0.618/0.567; N-HiTS 0/0 and 0.639/0.550. The task table contains 216 unique model-task rows, all six models on each of 36 tasks, and four folds per task. No inconsistency was found.

These descriptive summaries were used only as a consistency check and did not override the inferential results.

## 6. Exact manuscript-ready Results paragraph

{result_paragraph}

### Conservative alternative

{conservative}

## 7. Exact Supplementary table title and note

**{title}**

{note}

## 8. Interpretation

The results do not support a claim of universal model superiority. In particular, the stored comparisons do not distinguish XGBoost from corrected TRUE-BiLSTM after Holm correction for either metric. The supported pairwise statements are limited to the contrasts listed above.

## 9. Changes and safeguards

No model was retrained, no forecasting metric or inferential test was rerun, no Holm correction or effect-size definition was changed, no Figure 7 file was regenerated, and no existing analysis or manuscript file was modified. This package contains formatting and verification outputs only.
"""
    (OUT / "FORECAST_PAIRWISE_TABLE_VERIFICATION_REPORT.md").write_text(report, encoding="utf-8")

    output_rows = []
    for path in sorted(item for item in OUT.rglob("*") if item.is_file() and item.name != "OUTPUT_MANIFEST.csv"):
        output_rows.append(
            {
                "relative_path": path.relative_to(OUT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(output_rows).to_csv(OUT / "OUTPUT_MANIFEST.csv", index=False)
    print("Created verified forecast pairwise statistical table package.")


if __name__ == "__main__":
    main()
