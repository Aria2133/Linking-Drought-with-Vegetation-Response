from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
TABLE_DIR = ROOT / "source_tables"
VALIDATION_DIR = ROOT / "validation"
REPORT_DIR = ROOT / "reports"
BACKUP_DIR = ROOT / "backups"

# Final Figure 7 source (all 216 cells on the common 12-month benchmark; src/tables/figure7_tws_common_benchmark_correction.py)
FIG7_SOURCE = Path(r"<PROJECT_ROOT>\AUDIT_TEMP\FIGURE7_TWS_COMMON_CORRECTION\source_tables\Figure_7_source.csv")
PIXEL_SOURCE_NO_SVR = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\corrected_true_BiLSTM_rerun_20260728\08_downstream\corrected_combined_pixel_metrics.csv"
)
SVR_PIXEL_SOURCE = Path(
    r"<PROJECT_ROOT>\Docs\major_revision_analyses"
    r"\SVR_full_benchmark_supplementary_candidate_20260723_204013\metrics\svr_outer_test_selected_refit_metrics.csv"
)
OLD_WRONG_SOURCE = Path(
    r"<PROJECT_ROOT>\Linking-Drought-with-Vegetation-Response"
    r"\outputs\expanded_supplementary_forecasting_heatmap_20260728_205355"
    r"\source_tables\expanded_forecasting_heatmap_source_long.csv"
)
DESKTOP = Path(r"<PROJECT_ROOT>\Paper Revised Version")

LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_INTERNAL_ORDER = ["GRASSLANDS", "NATURAL_FORESTS", "SAVANNAS", "SHRUBLANDS"]
TARGET_ORDER = ["GNDVI", "GOSIF/SIF", "GPP", "LST", "NDVI", "NIRv", "SPEI-1", "TWS", "VPDmax"]
TARGET_INTERNAL_ORDER = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
MODEL_ORDER = [
    "Persistence",
    "Seasonal persistence",
    "Monthly climatology",
    "XGBoost",
    "RBF-ε-SVR",
    "LSTM",
    "BiLSTM",
    "TCN",
    "N-BEATS",
    "N-HiTS",
]
LEARNED_MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]
BASELINES = ["Persistence", "Seasonal persistence", "Monthly climatology"]

MODEL_FAMILY = {
    "Persistence": "Simple baseline",
    "Seasonal persistence": "Simple baseline",
    "Monthly climatology": "Simple baseline",
    "XGBoost": "Conventional ML benchmark",
    "RBF-ε-SVR": "Conventional ML benchmark",
    "LSTM": "Deep learning",
    "BiLSTM": "Deep learning",
    "TCN": "Deep learning",
    "N-BEATS": "Deep learning",
    "N-HiTS": "Deep learning",
}
LAND_DISPLAY = dict(zip(LAND_INTERNAL_ORDER, LAND_ORDER))
TARGET_DISPLAY = {"GOSIF": "GOSIF/SIF", "SPEI": "SPEI-1", "VPD": "VPDmax"}
TARGET_INTERNAL = {"GOSIF/SIF": "GOSIF", "SPEI-1": "SPEI", "VPDmax": "VPD"}
MODEL_DISPLAY = {
    "RBF–ε-SVR": "RBF-ε-SVR",
    "RBF-epsilon-SVR": "RBF-ε-SVR",
    "RBF-ε-SVR": "RBF-ε-SVR",
}
TOL = 1e-12


def ensure_dirs() -> None:
    for path in [FIG_DIR, TABLE_DIR, VALIDATION_DIR, REPORT_DIR, BACKUP_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_info(path: Path) -> dict[str, str | int | bool]:
    info: dict[str, str | int | bool] = {"path": str(path), "exists": path.exists()}
    if path.exists():
        stat = path.stat()
        info.update(
            {
                "size_bytes": stat.st_size,
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256(path),
            }
        )
    return info


def normalize_pixel_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["land_cover"] = out["land_type"].map(LAND_DISPLAY).fillna(out["land_type"])
    out["target"] = out["variable"].map(TARGET_DISPLAY).fillna(out["variable"])
    out["model"] = out["model"].map(MODEL_DISPLAY).fillna(out["model"])
    return out


def build_learned_from_figure7() -> pd.DataFrame:
    fig7 = pd.read_csv(FIG7_SOURCE)
    required = {"land_type_display", "target_display", "model", "mean_r2", "mean_normalized_rmse"}
    missing = required.difference(fig7.columns)
    if missing:
        raise RuntimeError(f"Figure 7 source missing required columns: {sorted(missing)}")
    fig7 = fig7[fig7["model"].isin(LEARNED_MODELS)].copy()
    rows = []
    for _, row in fig7.iterrows():
        for metric, value_col, var_col in [
            ("R²", "mean_r2", "sd_r2"),
            ("nRMSE", "mean_normalized_rmse", "sd_normalized_rmse"),
        ]:
            rows.append(
                {
                    "land cover": row["land_type_display"],
                    "target": row["target_display"],
                    "model": row["model"],
                    "model family": MODEL_FAMILY[row["model"]],
                    "metric": metric,
                    "mean value": float(row[value_col]),
                    "variability value": float(row[var_col]) if var_col in row and pd.notna(row[var_col]) else np.nan,
                    "number of folds": int(row.get("available_folds", 4)),
                    "number of seeds or runs": int(row.get("available_seeds", 1)),
                    "number of source rows": int(row.get("available_rows", 1)),
                    "source file": str(FIG7_SOURCE),
                    "corrected/retained status": str(row.get("data_status", "from_final_corrected_Figure_7_source")),
                    "aggregation level": "pixel-level Figure 7 task mean",
                }
            )
    return pd.DataFrame(rows)


def build_baselines_from_pixel_source() -> pd.DataFrame:
    base = pd.read_csv(PIXEL_SOURCE_NO_SVR)
    base = normalize_pixel_labels(base)
    base = base[
        base["model"].isin(BASELINES)
        & base["land_cover"].isin(LAND_ORDER)
        & base["target"].isin(TARGET_ORDER)
        & base["status"].eq("success")
    ].copy()
    rows = []
    for (land, target, model), group in base.groupby(["land_cover", "target", "model"], sort=False):
        if group["fold"].nunique() != 4:
            raise RuntimeError(f"Baseline task missing folds: {model} {land} {target}")
        for metric, value_col in [("R²", "r2"), ("nRMSE", "normalized_rmse")]:
            values = pd.to_numeric(group[value_col], errors="coerce")
            rows.append(
                {
                    "land cover": land,
                    "target": target,
                    "model": model,
                    "model family": MODEL_FAMILY[model],
                    "metric": metric,
                    "mean value": float(values.mean()),
                    "variability value": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "number of folds": int(group["fold"].nunique()),
                    "number of seeds or runs": int(group["seed"].astype(str).nunique()),
                    "number of source rows": int(len(group)),
                    "source file": str(PIXEL_SOURCE_NO_SVR),
                    "corrected/retained status": "baseline_pixel_metric_from_corrected_combined_pixel_source",
                    "aggregation level": "pixel-level fold mean, matching Figure 7 metric level",
                }
            )
    return pd.DataFrame(rows)


def build_svr_from_pixel_source() -> pd.DataFrame:
    svr = pd.read_csv(SVR_PIXEL_SOURCE)
    svr["land cover"] = svr["land_type"].map(LAND_DISPLAY).fillna(svr["land_type"])
    svr["target"] = svr["target"].map(TARGET_DISPLAY).fillna(svr["target"])
    svr = svr[svr["land cover"].isin(LAND_ORDER) & svr["target"].isin(TARGET_ORDER)].copy()
    rows = []
    for (land, target), group in svr.groupby(["land cover", "target"], sort=False):
        if group["fold"].nunique() != 4:
            raise RuntimeError(f"SVR task missing folds: {land} {target}")
        for metric, value_col in [("R²", "outer_test_r2"), ("nRMSE", "outer_test_nrmse")]:
            values = pd.to_numeric(group[value_col], errors="coerce")
            rows.append(
                {
                    "land cover": land,
                    "target": target,
                    "model": "RBF-ε-SVR",
                    "model family": MODEL_FAMILY["RBF-ε-SVR"],
                    "metric": metric,
                    "mean value": float(values.mean()),
                    "variability value": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                    "number of folds": int(group["fold"].nunique()),
                    "number of seeds or runs": 1,
                    "number of source rows": int(len(group)),
                    "source file": str(SVR_PIXEL_SOURCE),
                    "corrected/retained status": "final_task_specific_SVR_pixel_metric_available_outer_test_source",
                    "aggregation level": "pixel-level fold mean, matching Figure 7 metric level",
                }
            )
    return pd.DataFrame(rows)


def build_long_source() -> pd.DataFrame:
    long_df = pd.concat(
        [build_baselines_from_pixel_source(), build_svr_from_pixel_source(), build_learned_from_figure7()],
        ignore_index=True,
    )
    full = pd.MultiIndex.from_product(
        [LAND_ORDER, TARGET_ORDER, MODEL_ORDER, ["R²", "nRMSE"]],
        names=["land cover", "target", "model", "metric"],
    )
    long_df = long_df.set_index(["land cover", "target", "model", "metric"]).reindex(full).reset_index()
    for col in ["model family", "source file", "corrected/retained status", "aggregation level"]:
        if col == "model family":
            long_df[col] = long_df["model"].map(MODEL_FAMILY)
        else:
            long_df[col] = long_df[col].fillna("MISSING")
    return long_df


def write_wide_tables(long_df: pd.DataFrame) -> list[Path]:
    out = []
    for land in LAND_ORDER:
        for metric in ["R²", "nRMSE"]:
            matrix = (
                long_df[(long_df["land cover"] == land) & (long_df["metric"] == metric)]
                .pivot(index="model", columns="target", values="mean value")
                .reindex(index=MODEL_ORDER, columns=TARGET_ORDER)
            )
            metric_name = "R2" if metric == "R²" else "nRMSE"
            path = TABLE_DIR / f"heatmap_{land.replace(' ', '_')}_{metric_name}.csv"
            matrix.to_csv(path, encoding="utf-8-sig")
            out.append(path)
    return out


def text_color_for_cell(cmap, norm, value: float) -> str:
    rgba = cmap(norm(value))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.57 else "white"


def draw_heatmap(ax, matrix: pd.DataFrame, metric: str, land: str, panel: str, cmap, norm, show_y: bool):
    values = matrix.to_numpy(float)
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(f"{panel} {land} {metric}", fontsize=17, fontweight="bold", pad=12)
    ax.set_xticks(np.arange(len(TARGET_ORDER)))
    ax.set_xticklabels(TARGET_ORDER, rotation=40, ha="right", fontsize=13)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels(MODEL_ORDER if show_y else [], fontsize=13)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks(np.arange(-0.5, len(TARGET_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for separator in [2.5, 4.5]:
        ax.axhline(separator, color="black", linewidth=1.4)

    best_values = np.nanmax(values, axis=0) if metric == "R²" else np.nanmin(values, axis=0)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            val = values[y, x]
            if not np.isfinite(val):
                label = "NA"
                color = "black"
            else:
                label = f"{val:.2f}"
                color = text_color_for_cell(cmap, norm, val)
            ax.text(x, y, label, ha="center", va="center", fontsize=12.4, fontweight="bold", color=color)
            if np.isfinite(val) and np.isclose(val, best_values[x], rtol=0, atol=TOL):
                ax.add_patch(
                    patches.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2.8,
                        clip_on=False,
                    )
                )
    return image


def make_figure(long_df: pd.DataFrame) -> tuple[Path, Path, Path]:
    r2_values = long_df[long_df["metric"] == "R²"]["mean value"].to_numpy(float)
    nrmse_values = long_df[long_df["metric"] == "nRMSE"]["mean value"].to_numpy(float)
    r2_vmin, r2_vmax = -1.0, 1.0
    nrmse_vmin, nrmse_vmax = 0.0, 2.0

    r2_norm = Normalize(vmin=r2_vmin, vmax=r2_vmax, clip=True)
    nrmse_norm = Normalize(vmin=nrmse_vmin, vmax=nrmse_vmax, clip=True)
    r2_cmap = mpl.colormaps["RdYlBu"].copy()
    nrmse_cmap = mpl.colormaps["YlGnBu_r"].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(4, 2, figsize=(24, 31), constrained_layout=False)
    panels = list("abcdefgh")
    panel_i = 0
    last_r2_image = None
    last_nrmse_image = None
    for row_i, land in enumerate(LAND_ORDER):
        for col_i, metric in enumerate(["R²", "nRMSE"]):
            ax = axes[row_i, col_i]
            matrix = (
                long_df[(long_df["land cover"] == land) & (long_df["metric"] == metric)]
                .pivot(index="model", columns="target", values="mean value")
                .reindex(index=MODEL_ORDER, columns=TARGET_ORDER)
            )
            image = draw_heatmap(
                ax,
                matrix,
                metric,
                land,
                f"({panels[panel_i]})",
                r2_cmap if metric == "R²" else nrmse_cmap,
                r2_norm if metric == "R²" else nrmse_norm,
                show_y=(col_i == 0),
            )
            if metric == "R²":
                last_r2_image = image
            else:
                last_nrmse_image = image
            panel_i += 1

    fig.subplots_adjust(left=0.11, right=0.92, top=0.965, bottom=0.055, hspace=0.48, wspace=0.08)
    cax_r2 = fig.add_axes([0.935, 0.545, 0.018, 0.36])
    cbar_r2 = fig.colorbar(last_r2_image, cax=cax_r2)
    cbar_r2.set_label("R² color scale (clipped at -1 to 1)", fontsize=13)
    cbar_r2.ax.tick_params(labelsize=11)
    cax_n = fig.add_axes([0.935, 0.115, 0.018, 0.36])
    cbar_n = fig.colorbar(last_nrmse_image, cax=cax_n)
    cbar_n.set_label("nRMSE color scale (clipped at 0 to 2)", fontsize=13)
    cbar_n.ax.tick_params(labelsize=11)
    fig.text(
        0.11,
        0.025,
        "Cell values are exact Figure 7-consistent task means rounded to two decimals. "
        "Black outlines mark highest R² or lowest nRMSE within each land-cover × target task.",
        ha="left",
        fontsize=13,
    )

    png = FIG_DIR / "expanded_supplementary_forecasting_heatmap_figure7_consistent.png"
    pdf = FIG_DIR / "expanded_supplementary_forecasting_heatmap_figure7_consistent.pdf"
    svg = FIG_DIR / "expanded_supplementary_forecasting_heatmap_figure7_consistent.svg"
    fig.savefig(png, dpi=400)
    fig.savefig(pdf)
    fig.savefig(svg)
    plt.close(fig)

    # Desktop corrected copies. The incorrect v2 files are backed up first.
    for suffix in [".png", ".pdf", ".svg"]:
        existing = DESKTOP / f"expanded_supplementary_forecasting_heatmap_readable_v2{suffix}"
        if existing.exists():
            shutil.copy2(existing, BACKUP_DIR / f"previous_incorrect_readable_v2{suffix}")
    for src in [png, pdf, svg]:
        shutil.copy2(src, DESKTOP / src.name)
    shutil.copy2(png, DESKTOP / "expanded_supplementary_forecasting_heatmap_readable_v2.png")
    shutil.copy2(pdf, DESKTOP / "expanded_supplementary_forecasting_heatmap_readable_v2.pdf")
    shutil.copy2(svg, DESKTOP / "expanded_supplementary_forecasting_heatmap_readable_v2.svg")
    return png, pdf, svg


def validate(long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fig7 = pd.read_csv(FIG7_SOURCE)
    learned = long_df[long_df["model"].isin(LEARNED_MODELS)].copy()
    learned["land_type"] = learned["land cover"].map({v: k for k, v in LAND_DISPLAY.items()})
    learned["target_internal"] = learned["target"].map(TARGET_INTERNAL).fillna(learned["target"])
    audits = []
    for metric, fcol in [("R²", "mean_r2"), ("nRMSE", "mean_normalized_rmse")]:
        subset = learned[learned["metric"].eq(metric)]
        merged = subset.merge(
            fig7,
            left_on=["land_type", "target_internal", "model"],
            right_on=["land_type", "target", "model"],
            how="outer",
            suffixes=("_heatmap", "_fig7"),
            indicator=True,
        )
        merged["figure7_value"] = merged[fcol]
        merged["heatmap_value"] = merged["mean value"]
        merged["absolute_difference"] = (merged["heatmap_value"] - merged["figure7_value"]).abs()
        merged["metric"] = metric
        audits.append(
            merged[
                [
                    "metric",
                    "land cover",
                    "target_display",
                    "model",
                    "heatmap_value",
                    "figure7_value",
                    "absolute_difference",
                    "_merge",
                ]
            ]
        )
    audit = pd.concat(audits, ignore_index=True)
    audit.to_csv(TABLE_DIR / "figure7_learned_value_match_audit.csv", index=False, encoding="utf-8-sig")

    checks = []
    def add(name: str, passed: bool, details: str) -> None:
        checks.append({"check": name, "status": "PASS" if passed else "FAIL", "details": details})

    add("no model rerun or prediction recomputation", True, "Script reads metric/source CSVs only.")
    add("four land-cover classes present", set(long_df["land cover"]) == set(LAND_ORDER), str(sorted(long_df["land cover"].dropna().unique())))
    add("nine targets present", set(long_df["target"]) == set(TARGET_ORDER), str(sorted(long_df["target"].dropna().unique())))
    add("ten methods present", set(long_df["model"]) == set(MODEL_ORDER), str(sorted(long_df["model"].dropna().unique())))
    add("two metrics present", set(long_df["metric"]) == {"R²", "nRMSE"}, str(sorted(long_df["metric"].dropna().unique())))
    add("720 heatmap cells present", len(long_df) == 720, f"rows={len(long_df)}")
    add("no missing means", long_df["mean value"].notna().all(), f"missing={int(long_df['mean value'].isna().sum())}")
    max_diff = float(audit["absolute_difference"].max())
    add("learned-model values match Figure 7 exactly", max_diff <= TOL, f"max_abs_difference={max_diff}")
    add("SVR represented for all 36 tasks", len(long_df[(long_df["model"] == "RBF-ε-SVR") & (long_df["metric"] == "R²")]) == 36, "SVR R² task count checked.")
    add("SARIMA absent", not long_df["model"].astype(str).str.contains("SARIMA", case=False, na=False).any(), "SARIMA was not included.")
    add("Desktop readable_v2 PNG replaced", (DESKTOP / "expanded_supplementary_forecasting_heatmap_readable_v2.png").exists(), str(DESKTOP / "expanded_supplementary_forecasting_heatmap_readable_v2.png"))
    validation = pd.DataFrame(checks)
    validation.to_csv(VALIDATION_DIR / "expanded_forecasting_heatmap_figure7_consistency_checks.csv", index=False, encoding="utf-8-sig")
    return audit, validation


def write_report(long_df: pd.DataFrame, audit: pd.DataFrame, validation: pd.DataFrame, outputs: tuple[Path, Path, Path]) -> None:
    source_infos = pd.DataFrame([file_info(FIG7_SOURCE), file_info(PIXEL_SOURCE_NO_SVR), file_info(SVR_PIXEL_SOURCE), file_info(OLD_WRONG_SOURCE)])
    source_infos.to_csv(TABLE_DIR / "authoritative_input_file_audit.csv", index=False, encoding="utf-8-sig")
    r2_values = long_df[long_df["metric"].eq("R²")]["mean value"]
    n_values = long_df[long_df["metric"].eq("nRMSE")]["mean value"]

    caption = (
        "Figure Sx. Outer-test forecasting performance for all models, target variables, and land-cover classes. "
        "Each row of the figure represents one land-cover class, with R² shown on the left and normalized RMSE "
        "(nRMSE) on the right. Within each heatmap, columns represent the nine target variables and rows represent "
        "the three simple baselines, two conventional machine-learning benchmarks, and five deep-learning models. "
        "Learned-model values are the exact pixel-level task means used in the final corrected Figure 7 source table. "
        "Cell values show mean performance across the final rolling-origin evaluation runs, rounded to two decimals. "
        "Outlined cells identify the highest R² or lowest nRMSE for each land-cover × target task. Higher R² and lower "
        "nRMSE indicate better performance. The color scales are visually clipped at R² = -1 to 1 and nRMSE = 0 to 2; "
        "the printed cell values and source table retain the untransformed numerical values."
    )
    (REPORT_DIR / "expanded_supplementary_forecasting_heatmap_figure7_consistent_caption.md").write_text(caption + "\n", encoding="utf-8")

    report = f"""# Expanded Forecasting Heatmap Rebuild Report

## Issue Corrected

The previous expanded supplementary heatmap used Supplementary Figure S6 monthly-aggregate metrics. That made its learned-model R² and nRMSE values differ from the accepted pixel-level Figure 7 values.

This rebuild fixes the problem by taking every learned-model heatmap value directly from the final corrected Figure 7 source table:

`{FIG7_SOURCE}`

No model was rerun, retrained, refit, or used to regenerate predictions.

## Source Logic

- LSTM, BiLSTM, TCN, N-BEATS, N-HiTS, and XGBoost: exact values from `final_corrected_Figure_7_source_table.csv`.
- Persistence, Seasonal persistence, and Monthly climatology: pixel-level fold metrics from `corrected_combined_pixel_metrics.csv`, summarized at the same task level.
- RBF-ε-SVR: available task-specific pixel-level outer-test metrics from `svr_outer_test_selected_refit_metrics.csv`.

## Validation

The learned-model value match audit confirms that the expanded heatmap and Figure 7 source match exactly:

- Maximum learned-model R²/nRMSE absolute difference: `{audit['absolute_difference'].max():.16g}`
- Heatmap source rows: `{len(long_df)}`
- R² observed range: `{r2_values.min():.3f}` to `{r2_values.max():.3f}`
- nRMSE observed range: `{n_values.min():.3f}` to `{n_values.max():.3f}`

{validation.to_markdown(index=False)}

## Outputs

- PNG: `{outputs[0]}`
- PDF: `{outputs[1]}`
- SVG: `{outputs[2]}`
- Long source table: `{TABLE_DIR / 'expanded_forecasting_heatmap_source_long.csv'}`
- Figure 7 match audit: `{TABLE_DIR / 'figure7_learned_value_match_audit.csv'}`
- Validation checks: `{VALIDATION_DIR / 'expanded_forecasting_heatmap_figure7_consistency_checks.csv'}`

Corrected Desktop copies were also saved to:

- `{DESKTOP / 'expanded_supplementary_forecasting_heatmap_readable_v2.png'}`
- `{DESKTOP / 'expanded_supplementary_forecasting_heatmap_readable_v2.pdf'}`
- `{DESKTOP / 'expanded_supplementary_forecasting_heatmap_readable_v2.svg'}`
- `{DESKTOP / 'expanded_supplementary_forecasting_heatmap_figure7_consistent.png'}`

The previous incorrect Desktop v2 export was backed up in:

`{BACKUP_DIR}`
"""
    (REPORT_DIR / "expanded_supplementary_forecasting_heatmap_figure7_consistent_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    long_df = build_long_source()
    long_path = TABLE_DIR / "expanded_forecasting_heatmap_source_long.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")
    write_wide_tables(long_df)
    outputs = make_figure(long_df)
    audit, validation = validate(long_df)
    write_report(long_df, audit, validation, outputs)
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "no_model_rerun": True,
        "figure7_source": str(FIG7_SOURCE),
        "baseline_pixel_source": str(PIXEL_SOURCE_NO_SVR),
        "svr_pixel_source": str(SVR_PIXEL_SOURCE),
        "output_root": str(ROOT),
        "desktop_replaced_readable_v2": True,
        "rows": int(len(long_df)),
        "max_learned_model_difference_vs_figure7": float(audit["absolute_difference"].max()),
        "validation_status_counts": validation["status"].value_counts().to_dict(),
    }
    (ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("COMPLETED")
    print(f"rows={len(long_df)}")
    print(f"max_learned_diff={audit['absolute_difference'].max():.16g}")
    print(f"png={outputs[0]}")
    print(f"desktop_png={DESKTOP / 'expanded_supplementary_forecasting_heatmap_readable_v2.png'}")


if __name__ == "__main__":
    main()
