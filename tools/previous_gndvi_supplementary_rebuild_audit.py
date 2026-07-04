from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


OUT_ROOT = Path(r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\previous_GNDVI_supplementary_figure_rebuild_audit")

SEPARATE_SUPP = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_supplementary_information_final_separate_figures")
FINAL_FIG_PACKAGE = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_revision_final_figures_package")
LEGACY_PACKAGE = Path(r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\legacy_gndvi_sensitivity_package")
COMPACT_POLISHED = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_compact_supplementary_figures_FINAL_POLISHED")


LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
PERIODS = ["baseline", "historic_drought", "recent_drought"]
RESPONSES = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
PERIOD_LABELS = {
    "baseline": "Baseline",
    "historic_drought": "2012-2016",
    "recent_drought": "2020-2024",
}
RESPONSE_LABELS = {"GOSIF": "GOSIF/SIF"}


def normalize_period(value: object) -> str:
    text = str(value).strip()
    low = text.lower()
    if "baseline" in low or "2001-2011" in low or "2001–2011" in low:
        return "baseline"
    if "historic" in low or "2012-2016" in low or "2012–2016" in low:
        return "historic_drought"
    if "recent" in low or "2020-2024" in low or "2020–2024" in low:
        return "recent_drought"
    return text


def ensure_dirs() -> dict[str, Path]:
    dirs = {
        "figures": OUT_ROOT / "figures",
        "tables": OUT_ROOT / "tables",
        "report": OUT_ROOT / "report",
        "source_mapping": OUT_ROOT / "source_mapping",
        "audit": OUT_ROOT / "audit",
        "copied_diagnostics": OUT_ROOT / "copied_diagnostics",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def normalize_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def copy_if_exists(src: Path, dst_dir: Path, new_name: str | None = None) -> dict[str, str]:
    if not src.exists():
        return {
            "source_path": str(src),
            "copied_path": "",
            "copied": "no",
            "sha256": "",
        }
    dst = dst_dir / (new_name or src.name)
    shutil.copy2(src, dst)
    return {
        "source_path": str(src),
        "copied_path": str(dst),
        "copied": "yes",
        "sha256": sha256(dst),
    }


def load_sources() -> dict[str, pd.DataFrame]:
    return {
        "separate_source_map": read_csv_if_exists(SEPARATE_SUPP / "source_mapping" / "separate_supplementary_figure_source_map.csv"),
        "compact_source_map": read_csv_if_exists(COMPACT_POLISHED / "source_mapping" / "final_compact_supplementary_source_map.csv"),
        "legacy_ewbi_values": read_csv_if_exists(LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_values.csv"),
        "legacy_ewbi_ci": read_csv_if_exists(LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_bootstrap_ci.csv"),
        "legacy_ewbi_change": read_csv_if_exists(LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_change_audit.csv"),
        "legacy_gndvi_vs_veg": read_csv_if_exists(LEGACY_PACKAGE / "tables" / "legacy_GNDVI_vs_vegetation_indices.csv"),
        "current_s1a": read_csv_if_exists(SEPARATE_SUPP / "source_mapping" / "S1a_period_response_panel_data.csv"),
        "current_s1b": read_csv_if_exists(SEPARATE_SUPP / "source_mapping" / "S1b_component_omission_panel_data.csv"),
        "current_s1c": read_csv_if_exists(SEPARATE_SUPP / "source_mapping" / "S1c_weighting_panel_data.csv"),
    }


def source_map_has_gndvi(df: pd.DataFrame, figure_filter: str | None = None) -> bool:
    if df.empty:
        return False
    work = df.copy()
    if figure_filter and "supplementary_figure" in work.columns:
        work = work[work["supplementary_figure"].astype(str).str.contains(figure_filter, case=False, na=False)]
    text = work.astype(str).agg(" ".join, axis=1).str.cat(sep=" ")
    return "GNDVI" in text or "gndvi" in text


def build_dependency_inventory(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    def add(
        figure_name: str,
        script_path: str,
        directness: str,
        output_path: str,
        status: str,
        action: str,
        requires_model_rerun: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "supplementary_figure_name": figure_name,
                "script_path": script_path,
                "uses_GNDVI_directly_or_indirectly": directness,
                "current_output_path": output_path,
                "rerun_status": status,
                "recommended_action": action,
                "requires_model_rerun": requires_model_rerun,
                "notes": notes,
            }
        )

    build_script = str(SEPARATE_SUPP / "scripts" / "build_separate_supplementary_information.py")
    compact_script = str(COMPACT_POLISHED / "scripts" / "build_final_polished_compact_figures.py")

    add(
        "Figure S1 - EWBI-response association",
        build_script,
        "direct: GNDVI is one response variable in the EWBI-response heatmap",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S1_EWBI_response_association.png"),
        "rebuilt from existing previous-GNDVI EWBI association table",
        "Use previous-GNDVI rebuilt S1 output in this package for sensitivity review",
        "no",
        "Only the GNDVI response rows change; NIRv, NDVI, GOSIF/SIF, and GPP rows are unchanged in the legacy audit table.",
    )
    add(
        "Figure S2 - EWBI component and weighting robustness, panel a",
        build_script,
        "indirect: component/leave-one-out summaries average over response variables including GNDVI",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S2_EWBI_component_weighting_robustness.png"),
        "not rebuilt",
        "Requires EWBI component and leave-one-out response tables recomputed with previous GNDVI before a true previous-GNDVI panel can be drawn",
        "no",
        "Existing previous-GNDVI package did not include previous-GNDVI component/leave-one-out tables. Replotting from current table would silently keep QA GNDVI.",
    )
    add(
        "Figure S2 - EWBI component and weighting robustness, panel b",
        build_script,
        "indirect: weighting summaries aggregate response correlations including GNDVI",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S2_EWBI_component_weighting_robustness.png"),
        "not rebuilt",
        "Requires alternative-weighting validation table recomputed with previous GNDVI before a true previous-GNDVI panel can be drawn",
        "no",
        "Existing previous-GNDVI package did not include previous-GNDVI weighting robustness table.",
    )
    add(
        "Figure S2 - EWBI drought-period separation, panel c",
        build_script,
        "no direct GNDVI dependency: forcing-side EWBI period separation",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S2_EWBI_component_weighting_robustness.png"),
        "unchanged",
        "Keep current panel",
        "no",
        "This panel separates EWBI forcing periods and does not use vegetation-response GNDVI.",
    )
    add(
        "Figure S3 - TWS scale and resampling sensitivity",
        build_script,
        "no GNDVI dependency identified",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S3_TWS_scale_resampling_sensitivity.png"),
        "unchanged",
        "No previous-GNDVI rerun needed",
        "no",
        "Source map points to TWS/harmonization/resampling tables.",
    )
    add(
        "Figure S4 - Gap-filling validation and downstream sensitivity",
        build_script,
        "no direct GNDVI dependency identified from source map",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S4_gap_filling_validation_downstream_sensitivity.png"),
        "unchanged",
        "No previous-GNDVI rerun needed unless downstream table is later proven to include GNDVI state metrics",
        "no",
        "Current source map points to TWS/LST gap-filling validation, EWBI gap-fill sensitivity, and state agreement summaries.",
    )
    add(
        "Figure S5 - Detrending and preprocessing sensitivity",
        build_script,
        "no direct GNDVI dependency identified from source map",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S5_detrending_preprocessing_sensitivity.png"),
        "unchanged",
        "No previous-GNDVI rerun needed unless downstream table is later proven to include GNDVI response metrics",
        "no",
        "Current source map points to TWS trend removal, EWBI agreement, and state agreement summaries.",
    )
    add(
        "Figure S6 - Forecasting baselines, panel a",
        build_script,
        "direct: aggregate forecasting baseline summaries include GNDVI target tasks",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S6_forecasting_baselines_rolling_origin.png"),
        "not rebuilt",
        "Do not reuse QA-GNDVI model outputs as previous-GNDVI results; a GNDVI-only forecasting/baseline rerun would be the minimum if previous GNDVI becomes final",
        "yes - GNDVI target only, not all targets",
        "This cannot be corrected by figure recomposition because trained/tested GNDVI target outputs depend on the target data source.",
    )
    add(
        "Figure S6 - Rolling-origin pixel metrics, panel b",
        build_script,
        "direct: fold-level pixel metrics are aggregated across targets including GNDVI",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S6_forecasting_baselines_rolling_origin.png"),
        "not rebuilt",
        "Do not change without rerunning GNDVI target model evaluations; non-GNDVI target tasks can remain unchanged",
        "yes - GNDVI target only, not all targets",
        "No forecasting/deep-learning/ML training was run in this package.",
    )
    add(
        "Figure S7 - Targeted multivariate forecasting sensitivity",
        build_script,
        "no GNDVI dependency: targets are GPP, NIRv, and SPEI",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL.png"),
        "unchanged",
        "No previous-GNDVI rerun needed",
        "no",
        "The plotted targets do not include GNDVI.",
    )
    add(
        "Figure S8 - NIRv stress-zone threshold sensitivity",
        build_script,
        "no GNDVI dependency: NIRv stress-zone analysis",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S8_NIRv_stress_zone_threshold_sensitivity.png"),
        "unchanged",
        "No previous-GNDVI rerun needed",
        "no",
        "Stress-zone source is NIRv, not GNDVI.",
    )
    add(
        "Figure S9 - Ecosystem-state classification sensitivity",
        build_script,
        "no GNDVI dependency identified from source map",
        str(SEPARATE_SUPP / "figures_png" / "Figure_S9_ecosystem_state_classification_sensitivity.png"),
        "unchanged",
        "No previous-GNDVI rerun needed",
        "no",
        "State-classification source map does not list GNDVI as a source variable.",
    )
    add(
        "Main Figure 5 diagnostic - Inter-index correlations",
        str(Path(r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\Clipping Tonzi Ranch\Inter index correlation.py")),
        "direct: raw inter-index correlations include GNDVI pairs",
        str(LEGACY_PACKAGE / "figures" / "Figure_5_legacy_GNDVI_corrected.png"),
        "already rebuilt in legacy package; copied as diagnostic, not a supplementary figure",
        "Use only as diagnostic context for GNDVI source choice",
        "no",
        "Existing legacy package reported zero unexpected non-GNDVI changes.",
    )
    add(
        "Main Figure 6 diagnostic - Cross-correlation heatmaps",
        str(Path(r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Codes\Visualizations\Advance Drought Visualizations.py")),
        "direct: CCF rows include GNDVI predictor/target pairs",
        str(LEGACY_PACKAGE / "figures" / "Figure_6_legacy_GNDVI_corrected.png"),
        "already rebuilt in legacy package; copied as diagnostic, not a supplementary figure",
        "Use only as diagnostic context for GNDVI source choice",
        "no",
        "Existing legacy package reported zero unexpected non-GNDVI changes.",
    )
    return pd.DataFrame(rows)


def pivot_s1_values(values: pd.DataFrame, ci: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if values.empty:
        raise FileNotFoundError("Missing legacy_EWBI_response_association_values.csv")
    df = values.copy()
    df["period"] = df["period"].map(normalize_period)
    df["response"] = df["response"].astype(str)
    df["land_type"] = df["land_type"].astype(str)
    df = df[df["response"].isin(RESPONSES)].copy()
    df["row_label"] = df["land_type"] + " | " + df["period"].map(PERIOD_LABELS).fillna(df["period"])
    df["response_label"] = df["response"].map(RESPONSE_LABELS).fillna(df["response"])

    matrix = df.pivot_table(index="row_label", columns="response_label", values="pearson_r", aggfunc="first")
    n_matrix = df.pivot_table(index="row_label", columns="response_label", values="n", aggfunc="first")

    row_order = [lt + " | " + PERIOD_LABELS[p] for lt in LAND_TYPES for p in PERIODS]
    col_order = [RESPONSE_LABELS.get(r, r) for r in RESPONSES]
    matrix = matrix.reindex(index=row_order, columns=col_order)
    n_matrix = n_matrix.reindex(index=row_order, columns=col_order)

    star_matrix = pd.DataFrame("", index=matrix.index, columns=matrix.columns)
    if not ci.empty:
        ci_work = ci.copy()
        for col in ["land_type", "period", "response"]:
            if col in ci_work.columns:
                ci_work[col] = ci_work[col].astype(str)
        if "period" in ci_work.columns:
            ci_work["period"] = ci_work["period"].map(normalize_period)
        lower_candidates = [c for c in ci_work.columns if c.lower() in {"ci_lower", "pearson_ci_low", "pearson_ci_lower", "bootstrap_ci_lower"}]
        upper_candidates = [c for c in ci_work.columns if c.lower() in {"ci_upper", "pearson_ci_high", "pearson_ci_upper", "bootstrap_ci_upper"}]
        exclude_col = next((c for c in ci_work.columns if "exclude" in c.lower() and "zero" in c.lower()), None)
        if lower_candidates and upper_candidates:
            lo, hi = lower_candidates[0], upper_candidates[0]
            ci_work["_excludes_zero"] = ((ci_work[lo] > 0) | (ci_work[hi] < 0))
        elif exclude_col:
            ci_work["_excludes_zero"] = normalize_bool(ci_work[exclude_col])
        else:
            ci_work["_excludes_zero"] = False
        for _, row in ci_work.iterrows():
            row_label = str(row.get("land_type", "")) + " | " + PERIOD_LABELS.get(str(row.get("period", "")), str(row.get("period", "")))
            response_label = RESPONSE_LABELS.get(str(row.get("response", "")), str(row.get("response", "")))
            if row_label in star_matrix.index and response_label in star_matrix.columns and bool(row.get("_excludes_zero", False)):
                star_matrix.loc[row_label, response_label] = "*"
    return matrix, n_matrix, star_matrix


def plot_s1_previous_gndvi(values: pd.DataFrame, ci: pd.DataFrame, out_figures: Path, out_tables: Path) -> None:
    matrix, n_matrix, star_matrix = pivot_s1_values(values, ci)
    values.to_csv(out_tables / "Figure_S1_previous_GNDVI_EWBI_response_values.csv", index=False)
    ci.to_csv(out_tables / "Figure_S1_previous_GNDVI_EWBI_response_bootstrap_ci.csv", index=False)

    annot = matrix.copy().astype(object)
    for idx in matrix.index:
        for col in matrix.columns:
            val = matrix.loc[idx, col]
            star = star_matrix.loc[idx, col]
            if pd.isna(val):
                annot.loc[idx, col] = ""
            else:
                n_val = n_matrix.loc[idx, col]
                n_suffix = "" if pd.isna(n_val) else f"\nn={int(n_val)}"
                annot.loc[idx, col] = f"{val:.2f}{star}{n_suffix}"

    sns.set_theme(style="white", context="paper")
    fig, ax = plt.subplots(figsize=(9.8, 7.0))
    cmap = sns.diverging_palette(240, 15, as_cmap=True)
    hm = sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.5,
        linecolor="white",
        annot=annot,
        fmt="",
        cbar_kws={"label": "Pearson r", "shrink": 0.82},
        annot_kws={"fontsize": 8.5, "fontweight": "bold"},
    )
    ax.set_xlabel("Vegetation/productivity response", fontsize=11, fontweight="bold")
    ax.set_ylabel("Land type and period", fontsize=11, fontweight="bold")
    ax.set_title("EWBI-response association using previous GNDVI", fontsize=12, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=9)
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Pearson r", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_figures / "Figure_S1_previous_GNDVI_EWBI_response_association.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_figures / "Figure_S1_previous_GNDVI_EWBI_response_association.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_gndvi_coherence(veg: pd.DataFrame, out_figures: Path, out_tables: Path) -> None:
    if veg.empty:
        return
    veg.to_csv(out_tables / "legacy_GNDVI_vs_vegetation_indices.csv", index=False)
    work = veg.copy()
    # Robust column normalization for legacy table variants.
    land_col = "land_type"
    response_col = "comparison_index" if "comparison_index" in work.columns else "response"
    strict_col = "strict_qa_pearson_r" if "strict_qa_pearson_r" in work.columns else "strictQA_pearson_r"
    legacy_col = "legacy_pearson_r" if "legacy_pearson_r" in work.columns else "previous_pearson_r"
    if response_col not in work.columns or strict_col not in work.columns or legacy_col not in work.columns:
        return
    work = work[work[response_col].isin(["NDVI", "NIRv", "GOSIF", "GPP"])].copy()
    long = work.melt(
        id_vars=[land_col, response_col],
        value_vars=[legacy_col, strict_col],
        var_name="source",
        value_name="pearson_r",
    )
    long["source"] = long["source"].map({legacy_col: "Previous GNDVI", strict_col: "Strict-QA GNDVI"}).fillna(long["source"])

    sns.set_theme(style="whitegrid", context="paper")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharey=True)
    axes = axes.ravel()
    palette = {"Previous GNDVI": "#4C78A8", "Strict-QA GNDVI": "#F58518"}
    for ax, lt in zip(axes, LAND_TYPES):
        sub = long[long[land_col] == lt]
        sns.barplot(data=sub, x=response_col, y="pearson_r", hue="source", palette=palette, ax=ax)
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_title(lt, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Pearson r" if ax in axes[::2] else "")
        ax.tick_params(axis="x", labelrotation=0, labelsize=9)
        if ax is not axes[0]:
            leg = ax.get_legend()
            if leg:
                leg.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend_.remove()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    fig.suptitle("Same-month GNDVI coherence with vegetation/productivity indices", y=1.08, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_figures / "diagnostic_previous_vs_strictQA_GNDVI_vegetation_coherence.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_figures / "diagnostic_previous_vs_strictQA_GNDVI_vegetation_coherence.pdf", bbox_inches="tight")
    plt.close(fig)


def build_rebuild_audit(inventory: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    s1_change = sources["legacy_ewbi_change"]

    def add(name: str, current: str, completed: str, non_gndvi: str, judgment: str, decision: str, notes: str) -> None:
        rows.append(
            {
                "figure_name": name,
                "old_or_current_figure_available": current,
                "previous_GNDVI_rerun_completed": completed,
                "non_GNDVI_changes_observed": non_gndvi,
                "qualitative_judgment": judgment,
                "keep_reject_inconclusive": decision,
                "notes": notes,
            }
        )

    non_gndvi_changed = "unknown"
    if not s1_change.empty:
        include_col = "response" if "response" in s1_change.columns else None
        if include_col:
            non_gndvi_changed = "no" if (s1_change[include_col].astype(str) == "GNDVI").all() else "yes"

    add(
        "Figure S1 - EWBI-response association",
        "yes",
        "yes",
        non_gndvi_changed,
        "Scientifically interpretable as a previous-GNDVI sensitivity; GNDVI rows changed, non-GNDVI response rows were not recomputed.",
        "usable as sensitivity, not enough alone for manuscript-wide GNDVI decision",
        "Previous GNDVI has shorter recent coverage in the legacy package, so recent-period GNDVI association is based on fewer months than strict-QA GNDVI.",
    )
    add(
        "Figure S2 - EWBI component/weighting robustness",
        "yes",
        "no",
        "not assessed",
        "Cannot be honestly redrawn from previous GNDVI without upstream component/weighting validation tables.",
        "inconclusive",
        "A true previous-GNDVI S2 panel would require recomputing component omission and weighting response summaries with previous GNDVI.",
    )
    add(
        "Figure S6 - Forecasting baselines and rolling-origin metrics",
        "yes",
        "no",
        "not assessed",
        "Cannot be converted to previous GNDVI by plotting only because GNDVI target model/baseline outputs depend on the GNDVI source.",
        "inconclusive",
        "Minimum scientific rerun, if previous GNDVI becomes final, is GNDVI-target forecasting/baseline evaluation only; non-GNDVI targets can remain unchanged.",
    )
    add(
        "Figures S3, S4, S5, S7, S8, S9",
        "yes",
        "not needed",
        "no",
        "No direct GNDVI dependency identified from the final source maps.",
        "keep current",
        "These figures should not be changed for a GNDVI-source sensitivity.",
    )
    add(
        "Main Figure 5 diagnostic",
        "yes",
        "already rebuilt in legacy package",
        "no",
        "Previous GNDVI recovered ecologically coherent raw inter-index behavior in the legacy sensitivity package.",
        "diagnostic support only",
        "Not a supplementary figure in this package; copied only for source-decision context.",
    )
    add(
        "Main Figure 6 diagnostic",
        "yes",
        "already rebuilt in legacy package",
        "no",
        "Previous GNDVI recovered ecologically coherent GNDVI-related CCF behavior in the legacy sensitivity package.",
        "diagnostic support only",
        "Not a supplementary figure in this package; copied only for source-decision context.",
    )
    return pd.DataFrame(rows)


def write_report(
    inventory: pd.DataFrame,
    audit: pd.DataFrame,
    sources: dict[str, pd.DataFrame],
    copied: list[dict[str, str]],
    out_report: Path,
) -> None:
    legacy_values = sources["legacy_ewbi_values"]
    legacy_change = sources["legacy_ewbi_change"]
    veg = sources["legacy_gndvi_vs_veg"]

    gndvi_rows = pd.DataFrame()
    if not legacy_values.empty and "response" in legacy_values.columns:
        gndvi_rows = legacy_values[legacy_values["response"].astype(str) == "GNDVI"].copy()

    recent_n_note = ""
    if not gndvi_rows.empty and "period" in gndvi_rows.columns and "n" in gndvi_rows.columns:
        gndvi_rows["_period_code"] = gndvi_rows["period"].map(normalize_period)
        recent = gndvi_rows[gndvi_rows["_period_code"] == "recent_drought"]
        if not recent.empty:
            recent_n_note = ", ".join(f"{r.land_type}: n={int(r.n)}" for r in recent.itertuples())

    veg_summary = "Not available."
    if not veg.empty:
        strict_col = "strict_qa_pearson_r" if "strict_qa_pearson_r" in veg.columns else "strictQA_pearson_r"
        legacy_col = "legacy_pearson_r" if "legacy_pearson_r" in veg.columns else "previous_pearson_r"
        if strict_col in veg.columns and legacy_col in veg.columns:
            tmp = veg.copy()
            tmp["delta_legacy_minus_strict"] = tmp[legacy_col] - tmp[strict_col]
            better = int((tmp["delta_legacy_minus_strict"] > 0).sum())
            total = int(tmp["delta_legacy_minus_strict"].notna().sum())
            mean_delta = tmp["delta_legacy_minus_strict"].mean()
            veg_summary = (
                f"Previous GNDVI had higher same-month correlations than strict-QA GNDVI in {better} of {total} "
                f"land-type/response comparisons; mean legacy-minus-strict delta r = {mean_delta:.3f}."
            )

    s1_change_summary = "Not available."
    if not legacy_change.empty:
        s1_change_summary = (
            f"The existing EWBI-response change audit contains {len(legacy_change)} changed rows. "
            "Those rows are expected to be the GNDVI response rows only; the package does not report unexpected non-GNDVI row changes."
        )

    copied_lines = "\n".join(
        f"- `{Path(row['copied_path']).name}` copied: {row['copied']} from `{row['source_path']}`"
        for row in copied
    )

    inv_md = inventory.to_markdown(index=False)
    audit_md = audit.to_markdown(index=False)

    report = f"""# Previous-GNDVI Supplementary Figure Rebuild Audit

## Scope

This package tests whether the previous/original GNDVI source can be retained for the manuscript by rebuilding or auditing supplementary figures that depend on GNDVI. No forecasting models, deep-learning models, ML training, raster extraction, or manuscript figure overwrites were performed.

The previous GNDVI source evaluated here is:

`C:\\Users\\LOQ\\OneDrive\\Desktop\\Previous GNDVI`

Existing previous-GNDVI sensitivity outputs were used from:

`{LEGACY_PACKAGE}`

## What Was Rebuilt

Only **Supplementary Figure S1 - EWBI-response association** could be rebuilt honestly from existing previous-GNDVI tables without rerunning scientific workflows. The rebuilt figure is saved as:

- `figures/Figure_S1_previous_GNDVI_EWBI_response_association.png`
- `figures/Figure_S1_previous_GNDVI_EWBI_response_association.pdf`

The source value tables copied into this package are:

- `tables/Figure_S1_previous_GNDVI_EWBI_response_values.csv`
- `tables/Figure_S1_previous_GNDVI_EWBI_response_bootstrap_ci.csv`

## What Was Not Rebuilt

Some final supplementary panels depend on GNDVI but cannot be converted by a plotting-only rerun:

- **Figure S2 component/weighting robustness**: GNDVI contributes indirectly to aggregate response summaries. The previous-GNDVI package does not contain the upstream component/leave-one-out or alternative-weighting response tables needed for a faithful redraw.
- **Figure S6 forecasting baselines and rolling-origin metrics**: GNDVI is a forecasting target in aggregate summaries. A true previous-GNDVI version would require rerunning GNDVI-target forecasting/baseline evaluations. This package did not do that.

## Dependency Inventory

{inv_md}

## Rebuild and Comparison Audit

{audit_md}

## Diagnostic Evidence

{veg_summary}

{s1_change_summary}

Recent-period previous-GNDVI EWBI-response sample sizes from the legacy package: {recent_n_note or "not available"}.

Copied diagnostic artifacts:

{copied_lines}

## Scientific Judgment

The previous GNDVI source is promising for raw inter-index and CCF diagnostics because the existing legacy package found more coherent behavior against vegetation/productivity indices and no unexpected non-GNDVI changes in Figure 5 or Figure 6 diagnostics. However, the supplementary evidence is not complete enough to select previous GNDVI as the final manuscript-wide source by itself.

The main reasons are:

1. The previous-GNDVI EWBI-response sensitivity can be shown for Figure S1, but recent-period GNDVI coverage is shorter in the existing legacy output.
2. Figure S2 component/weighting robustness needs upstream previous-GNDVI recomputation before it can be redrawn faithfully.
3. Figure S6 forecasting summaries cannot be changed to previous GNDVI without rerunning at least GNDVI-target forecasting/baseline evaluations.

## Final Decision

**INCONCLUSIVE - further investigation or targeted reruns are needed before previous GNDVI can be adopted as the single final manuscript source.**

The previous GNDVI is scientifically plausible and may be safer for raw Figure 5/Figure 6 behavior, but manuscript-wide adoption would require at minimum:

- a decision on how to handle the previous GNDVI coverage gap after 2021,
- recomputation of EWBI component/weighting response summaries with previous GNDVI if those panels remain in the Supplement,
- and a GNDVI-target-only forecasting/baseline rerun if Figure 7/S6 summaries must be fully consistent with previous GNDVI.

## Safety Checklist

- No forecasting models were rerun.
- No deep-learning or ML training was rerun.
- No raster extraction was rerun.
- No manuscript figure folders were overwritten.
- Non-GNDVI data sources were not changed.
- Current non-GNDVI supplementary figures should remain unchanged.
"""
    (out_report / "FINAL_previous_GNDVI_supplementary_figure_audit_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    dirs = ensure_dirs()
    sources = load_sources()

    inventory = build_dependency_inventory(sources)
    inventory.to_csv(dirs["tables"] / "GNDVI_supplementary_dependency_inventory.csv", index=False)

    # Copy source maps for traceability.
    copied: list[dict[str, str]] = []
    copied.append(copy_if_exists(SEPARATE_SUPP / "source_mapping" / "separate_supplementary_figure_source_map.csv", dirs["source_mapping"]))
    copied.append(copy_if_exists(COMPACT_POLISHED / "source_mapping" / "final_compact_supplementary_source_map.csv", dirs["source_mapping"]))
    copied.append(copy_if_exists(LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_change_audit.csv", dirs["tables"]))

    # Rebuild the direct GNDVI-dependent EWBI response figure from existing previous-GNDVI tables.
    plot_s1_previous_gndvi(sources["legacy_ewbi_values"], sources["legacy_ewbi_ci"], dirs["figures"], dirs["tables"])

    # Diagnostic comparison figure from existing previous-vs-strict table.
    plot_gndvi_coherence(sources["legacy_gndvi_vs_veg"], dirs["figures"], dirs["tables"])

    # Copy already rebuilt Figure 5/Figure 6 legacy diagnostics as context, not as supplementary figures.
    for src, name in [
        (LEGACY_PACKAGE / "figures" / "Figure_5_legacy_GNDVI_corrected.png", "diagnostic_Figure_5_legacy_GNDVI_corrected.png"),
        (LEGACY_PACKAGE / "figures" / "Figure_5_legacy_GNDVI_corrected.pdf", "diagnostic_Figure_5_legacy_GNDVI_corrected.pdf"),
        (LEGACY_PACKAGE / "figures" / "Figure_6_legacy_GNDVI_corrected.png", "diagnostic_Figure_6_legacy_GNDVI_corrected.png"),
        (LEGACY_PACKAGE / "figures" / "Figure_6_legacy_GNDVI_corrected.pdf", "diagnostic_Figure_6_legacy_GNDVI_corrected.pdf"),
        (LEGACY_PACKAGE / "tables" / "Figure_5_legacy_GNDVI_values.csv", "diagnostic_Figure_5_legacy_GNDVI_values.csv"),
        (LEGACY_PACKAGE / "tables" / "Figure_6_legacy_GNDVI_values.csv", "diagnostic_Figure_6_legacy_GNDVI_values.csv"),
    ]:
        copied.append(copy_if_exists(src, dirs["copied_diagnostics"], name))

    audit = build_rebuild_audit(inventory, sources)
    audit.to_csv(dirs["tables"] / "previous_GNDVI_supplementary_rebuild_audit.csv", index=False)

    write_report(inventory, audit, sources, copied, dirs["report"])

    # Simple manifest for the generated package.
    manifest_rows = []
    for path in OUT_ROOT.rglob("*"):
        if path.is_file():
            manifest_rows.append(
                {
                    "relative_path": str(path.relative_to(OUT_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(manifest_rows).sort_values("relative_path").to_csv(dirs["audit"] / "package_file_manifest.csv", index=False)

    print(f"Created package: {OUT_ROOT}")
    print(f"Inventory rows: {len(inventory)}")
    print(f"Audit rows: {len(audit)}")


if __name__ == "__main__":
    main()
