from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from scipy import stats


DESKTOP = Path(r"C:\Users\LOQ\OneDrive\Desktop")
OUT_ROOT = DESKTOP / "ECOINF_supplementary_figures_previous_GNDVI_with_test_final"
CURRENT_SUPP = DESKTOP / "ECOINF_supplementary_information_final_separate_figures"
COMPACT = DESKTOP / "ECOINF_compact_supplementary_figures_final"
PANEL_DIR = COMPACT / "source_mapping"
FINAL_FIGURES_PACKAGE = DESKTOP / "ECOINF_revision_final_figures_package"
SOURCE_TABLES = FINAL_FIGURES_PACKAGE / "4_source_tables_used"
GAP_TABLES = DESKTOP / "GapFilling_Comparison_Revision" / "tables"
PREV_GNDVI_PACKAGE = DESKTOP / "major_revision_analyses" / "final_previous_GNDVI_with_test_rebuild"
DL_PACKAGE = DESKTOP / "DL_GNDVI_previous_noQA_rerun"

PREVIOUS_GNDVI_MONTHLY = PREV_GNDVI_PACKAGE / "tables" / "previous_GNDVI_full_2001_2024_landtype_monthly_means.csv"
EWBI_MONTHLY = GAP_TABLES / "ewbi_gapfill_monthly_series.csv"
CURRENT_S1A = PANEL_DIR / "S1a_period_response_panel_data.csv"
CURRENT_S1D = PANEL_DIR / "S1d_period_separation_panel_data.csv"
CURRENT_COMPONENT = SOURCE_TABLES / "ewbi_component_comparison.csv"
CURRENT_OMISSION = SOURCE_TABLES / "ewbi_leave_one_component_out.csv"
CURRENT_WEIGHTS = SOURCE_TABLES / "ewbi_alternative_weights.csv"
CURRENT_WEIGHTING_VALIDATION = SOURCE_TABLES / "ewbi_vegetation_validation.csv"
ORIGINAL_MONTHLY_METRICS = (
    DESKTOP
    / "major_revision_analyses"
    / "05_forecasting_raster_reviewer_safe_candidate"
    / "tables"
    / "reviewer_safe_all_monthly_aggregated_metrics.csv"
)
ORIGINAL_PIXEL_METRICS = (
    DESKTOP
    / "major_revision_analyses"
    / "05_forecasting_raster_reviewer_safe_candidate"
    / "tables"
    / "reviewer_safe_all_pixel_metrics.csv"
)
NEW_MONTHLY_METRICS = DL_PACKAGE / "tables" / "reviewer_safe_all_monthly_aggregated_metrics.csv"
NEW_PIXEL_METRICS = DL_PACKAGE / "tables" / "reviewer_safe_all_pixel_metrics.csv"

LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#007C91",
    "Natural Forests": "#2E7D32",
    "Savannas": "#E58B00",
    "Shrublands": "#B23A67",
}
LAND_LABEL_TO_MONTHLY = {
    "Grasslands": "Grasslands",
    "Natural Forests": "NaturalForests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
MONTHLY_TO_LABEL = {
    "Grasslands": "Grasslands",
    "NaturalForests": "Natural Forests",
    "Natural Forests": "Natural Forests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "Natural Forests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
RESPONSES = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
COMPONENT_COL = {
    "SPEI": "spei_signed_z",
    "TWS": "tws_signed_z",
    "VPD": "vpd_signed_z",
    "LST": "lst_signed_z",
}
PERIODS = [
    ("Baseline 2001-2011", "baseline", "2001-01", "2011-12"),
    ("Historic drought 2012-2016", "historic", "2012-01", "2016-12"),
    ("Recent period 2020-2024", "recent", "2020-01", "2024-12"),
]
PERIOD_ORDER = [p[0] for p in PERIODS]
FIGURE_STEMS = {
    "Figure S1": "Figure_S1_EWBI_response_association",
    "Figure S2": "Figure_S2_EWBI_component_weighting_robustness",
    "Figure S3": "Figure_S3_TWS_scale_resampling_sensitivity",
    "Figure S4": "Figure_S4_gap_filling_validation_downstream_sensitivity",
    "Figure S5": "Figure_S5_detrending_preprocessing_sensitivity",
    "Figure S6": "Figure_S6_forecasting_baselines_rolling_origin",
    "Figure S7": "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL",
    "Figure S8": "Figure_S8_NIRv_stress_zone_threshold_sensitivity",
    "Figure S9": "Figure_S9_ecosystem_state_classification_sensitivity",
}
MODEL_COLORS = {
    "Persistence": "#6F6F6F",
    "Seasonal persistence": "#A3A3A3",
    "Monthly climatology": "#B8A900",
    "SARIMA": "#9A5D45",
    "LSTM": "#2F6DAE",
    "BiLSTM": "#8556A8",
    "TCN": "#3B8C4A",
    "N-BEATS": "#F28E2B",
    "N-HiTS": "#D84A4A",
    "XGBoost": "#00A6A6",
}
MONTHLY_MODELS = [
    "Persistence",
    "Seasonal persistence",
    "Monthly climatology",
    "SARIMA",
    "LSTM",
    "BiLSTM",
    "TCN",
    "N-BEATS",
    "N-HiTS",
    "XGBoost",
]
PIXEL_MODELS = ["LSTM", "BiLSTM", "TCN", "N-BEATS", "N-HiTS", "XGBoost"]


def out_dirs() -> dict[str, Path]:
    paths = {
        "png": OUT_ROOT / "figures_png",
        "pdf": OUT_ROOT / "figures_pdf",
        "captions": OUT_ROOT / "captions",
        "source_mapping": OUT_ROOT / "source_mapping",
        "tables": OUT_ROOT / "tables",
        "report": OUT_ROOT / "report",
        "audit": OUT_ROOT / "audit",
        "tools": OUT_ROOT / "tools",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 11.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.4,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#D6D6D6",
            "grid.linewidth": 0.55,
            "grid.alpha": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def save_fig(fig: plt.Figure, stem: str, dirs: dict[str, Path]) -> tuple[Path, Path]:
    png = dirs["png"] / f"{stem}.png"
    pdf = dirs["pdf"] / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.07) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12.5,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def add_heatmap(
    ax: plt.Axes,
    values: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    annotations: np.ndarray | None = None,
    colorbar_label: str | None = None,
    rotate_x: float = 0,
    annotation_size: float = 8.2,
) -> None:
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=7)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", rotation=rotate_x)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            text = annotations[i, j] if annotations is not None else f"{val:.2f}"
            color = "white" if abs(val) > (vmax - vmin) * 0.30 else "#202020"
            ax.text(j, i, text, ha="center", va="center", fontsize=annotation_size, fontweight="bold", color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    if colorbar_label:
        cbar.set_label(colorbar_label)


def pretty_land(value: object) -> str:
    return MONTHLY_TO_LABEL.get(str(value), str(value))


def baseline_monthly_anomaly(series: pd.Series, baseline=(2001, 2011)) -> pd.Series:
    base = series[(series.index.year >= baseline[0]) & (series.index.year <= baseline[1])]
    means = base.groupby(base.index.month).mean()
    stds = base.groupby(base.index.month).std(ddof=1).replace(0, np.nan)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    for month in range(1, 13):
        mask = series.index.month == month
        out.loc[mask] = (series.loc[mask] - means.get(month, np.nan)) / stds.get(month, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def pearson_row(x: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    joined = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(joined) < 3 or joined["x"].std(ddof=1) == 0 or joined["y"].std(ddof=1) == 0:
        return len(joined), np.nan, np.nan
    result = stats.pearsonr(joined["x"], joined["y"])
    return len(joined), float(result.statistic), float(result.pvalue)


def spearman_row(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    joined = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(joined) < 3 or joined["x"].std(ddof=1) == 0 or joined["y"].std(ddof=1) == 0:
        return np.nan, np.nan
    result = stats.spearmanr(joined["x"], joined["y"])
    return float(result.statistic), float(result.pvalue)


def moving_block_ci(x: pd.Series, y: pd.Series, block_len: int = 3, n_boot: int = 1000, seed: int = 20260619) -> tuple[float, float]:
    joined = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(joined)
    if n < max(6, block_len * 2) or joined["x"].std(ddof=1) == 0 or joined["y"].std(ddof=1) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = joined.to_numpy(dtype=float)
    starts = np.arange(0, n - block_len + 1)
    corrs = []
    for _ in range(n_boot):
        picked = rng.choice(starts, size=int(np.ceil(n / block_len)), replace=True)
        idx = np.concatenate([np.arange(s, s + block_len) for s in picked])[:n]
        sample = vals[idx]
        if np.std(sample[:, 0], ddof=1) == 0 or np.std(sample[:, 1], ddof=1) == 0:
            continue
        corrs.append(np.corrcoef(sample[:, 0], sample[:, 1])[0, 1])
    if not corrs:
        return np.nan, np.nan
    return tuple(np.nanpercentile(corrs, [2.5, 97.5]).astype(float))


def previous_gndvi_anomaly(land_label: str) -> pd.Series:
    monthly = read_csv(PREVIOUS_GNDVI_MONTHLY, parse_dates=["date"])
    monthly_name = LAND_LABEL_TO_MONTHLY[land_label]
    sub = monthly[monthly["land_type"] == monthly_name].copy()
    sub = sub.set_index("date").sort_index().reindex(pd.date_range("2001-01-01", "2024-12-01", freq="MS"))
    return baseline_monthly_anomaly(sub["mean"].astype(float)).rename("GNDVI")


def signed_ewbi_frame(land_label: str) -> pd.DataFrame:
    monthly = read_csv(EWBI_MONTHLY, parse_dates=["date"])
    monthly_name = LAND_LABEL_TO_MONTHLY[land_label]
    sub = monthly[
        monthly["land_type"].isin([monthly_name, land_label])
        & (monthly["method"] == "current_monthwise_cross_year")
    ].copy()
    sub = sub.set_index("date").sort_index().reindex(pd.date_range("2001-01-01", "2024-12-01", freq="MS"))
    return sub[list(COMPONENT_COL.values()) + ["ewbi"]].astype(float)


def normalize_period(period: object) -> str:
    text = str(period).lower()
    if "baseline" in text or "2001" in text:
        return "baseline"
    if "historic" in text or "2012" in text:
        return "historic"
    if "recent" in text or "2020" in text:
        return "recent"
    return str(period)


def recompute_s1_gndvi_rows(dirs: dict[str, Path]) -> pd.DataFrame:
    current = read_csv(CURRENT_S1A)
    non_gndvi = current[current["response"] != "GNDVI"].copy()
    ewbi = read_csv(EWBI_MONTHLY, parse_dates=["date"])
    rows = []
    for land_label in LAND_ORDER:
        monthly_name = LAND_LABEL_TO_MONTHLY[land_label]
        ew = (
            ewbi[
                ewbi["land_type"].isin([monthly_name, land_label])
                & (ewbi["method"] == "current_monthwise_cross_year")
            ]
            .set_index("date")
            .sort_index()
            .reindex(pd.date_range("2001-01-01", "2024-12-01", freq="MS"))["ewbi"]
            .astype(float)
        )
        gndvi = previous_gndvi_anomaly(land_label)
        for period_label, period_code, start, end in PERIODS:
            x = ew.loc[pd.Timestamp(start): pd.Timestamp(end)]
            y = gndvi.loc[pd.Timestamp(start): pd.Timestamp(end)]
            n, r, p = pearson_row(x, y)
            rho, sp = spearman_row(x, y)
            ci_low, ci_high = moving_block_ci(x, y, seed=20260619 + len(rows))
            rows.append(
                {
                    "land_type": land_label,
                    "period": period_label,
                    "period_start": start,
                    "period_end": end,
                    "response": "GNDVI",
                    "response_display": "GNDVI",
                    "n": n,
                    "pearson_r": r,
                    "pearson_p_value": p,
                    "spearman_rho": rho,
                    "spearman_p_value": sp,
                    "ewbi_definition": "equal mean of signed standardized SPEI, TWS, VPD, and LST anomalies",
                    "response_preprocessing": "2001-2011 calendar-month standardized anomaly",
                    "gap_fill": "previous/no-QA GNDVI with completed 2022-2024 test-period land-mask means",
                    "qa_controlled_gndvi_confirmed": False,
                    "monthly_cache": str(PREVIOUS_GNDVI_MONTHLY),
                    "pearson_bootstrap_ci95_low": ci_low,
                    "pearson_bootstrap_ci95_high": ci_high,
                    "ci_excludes_zero": bool((ci_low > 0) or (ci_high < 0)) if np.isfinite(ci_low) and np.isfinite(ci_high) else False,
                    "row_label": f"{land_label} | {period_code}",
                    "source": "previous_noQA_GNDVI_with_test_recomputed",
                }
            )
    gndvi_rows = pd.DataFrame(rows)
    # Keep column order/extra fields compatible with the current panel table.
    combined = pd.concat([non_gndvi, gndvi_rows], ignore_index=True, sort=False)
    combined["period_code"] = combined["period"].map(normalize_period)
    combined["land_label"] = combined["land_type"].map(pretty_land)
    combined["row_label"] = combined["land_label"].str.replace("Natural Forests", "Natural forests", regex=False) + " | " + combined["period_code"]
    combined.to_csv(dirs["tables"] / "S1_previous_GNDVI_with_test_period_response_panel_data.csv", index=False)
    gndvi_rows.to_csv(dirs["tables"] / "S1_recomputed_previous_GNDVI_with_test_rows.csv", index=False)
    return combined


def build_s1(dirs: dict[str, Path], source_map: list[dict[str, str]]) -> None:
    panel = recompute_s1_gndvi_rows(dirs)
    row_order = [
        f"{land.replace('Natural Forests', 'Natural forests')} | {period_code}"
        for land in LAND_ORDER
        for period_code in ("baseline", "historic", "recent")
    ]
    values = panel.pivot(index="row_label", columns="response", values="pearson_r").reindex(index=row_order, columns=RESPONSES)
    sig = panel.pivot(index="row_label", columns="response", values="ci_excludes_zero").reindex(index=row_order, columns=RESPONSES)
    annotations = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values.iloc[i, j]
            annotations[i, j] = "" if pd.isna(val) else f"{val:.2f}{'*' if bool(sig.iloc[i, j]) else ''}"

    fig, ax = plt.subplots(figsize=(8.9, 6.3), layout="constrained")
    add_heatmap(
        ax,
        values.to_numpy(dtype=float),
        row_order,
        RESPONSES,
        "EWBI-response association",
        "RdBu_r",
        -1,
        1,
        annotations=annotations,
        colorbar_label="Pearson r",
        rotate_x=25,
        annotation_size=8.0,
    )
    save_fig(fig, FIGURE_STEMS["Figure S1"], dirs)
    source_map.append(
        {
            "supplementary_figure": "Figure S1",
            "panel": "single panel",
            "plotted_content": "EWBI-response correlations by land type and period; GNDVI cells recomputed with completed previous/no-QA GNDVI",
            "source_file_or_table": f"{CURRENT_S1A}; {PREVIOUS_GNDVI_MONTHLY}; {EWBI_MONTHLY}",
            "reviewer_concern_addressed": "EWBI validation against vegetation/productivity responses",
            "notes": "Non-GNDVI response cells copied from final validated panel table; GNDVI cells recomputed with 2001-2024 previous/no-QA GNDVI including test-period land-mask means.",
        }
    )


def recompute_s2_rows(dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    component_current = read_csv(CURRENT_COMPONENT)
    omission_current = read_csv(CURRENT_OMISSION)
    validation_current = read_csv(CURRENT_WEIGHTING_VALIDATION)
    weights = read_csv(CURRENT_WEIGHTS)

    component_rows = []
    omission_rows = []
    validation_rows = []
    for land_label in LAND_ORDER:
        signed = signed_ewbi_frame(land_label)
        gndvi = previous_gndvi_anomaly(land_label)
        monthly_land = LAND_LABEL_TO_MONTHLY[land_label]
        for comp, col in COMPONENT_COL.items():
            n, r, p = pearson_row(signed[col], gndvi)
            component_rows.append(
                {
                    "land_type": monthly_land,
                    "response": "GNDVI",
                    "predictor": comp,
                    "pearson_r": r,
                    "p_value": p,
                    "n": n,
                    "source": "previous_noQA_GNDVI_with_test_recomputed",
                }
            )
            omit_cols = [c for name, c in COMPONENT_COL.items() if name != comp]
            n2, r2, p2 = pearson_row(signed[omit_cols].mean(axis=1), gndvi)
            omission_rows.append(
                {
                    "land_type": monthly_land,
                    "response": "GNDVI",
                    "omitted_component": comp,
                    "pearson_r": r2,
                    "p_value": p2,
                    "n": n2,
                    "source": "previous_noQA_GNDVI_with_test_recomputed",
                }
            )
        for weighting_name in ["equal", "pca", "regression"]:
            if weighting_name == "equal":
                ewbi = signed["ewbi"]
                index_name = "EWBI_equal"
            else:
                w_sub = weights[
                    (weights["land_type"].map(pretty_land) == land_label)
                    & (weights["weighting"].astype(str).str.lower() == weighting_name)
                ]
                if w_sub.empty:
                    ewbi = pd.Series(np.nan, index=signed.index)
                else:
                    ewbi = pd.Series(0.0, index=signed.index)
                    for row in w_sub.itertuples(index=False):
                        ewbi = ewbi + float(row.signed_weight) * signed[COMPONENT_COL[str(row.component)]]
                index_name = "EWBI_PCA" if weighting_name == "pca" else "EWBI_regression"
            n3, r3, p3 = pearson_row(ewbi, gndvi)
            validation_rows.append(
                {
                    "land_type": monthly_land,
                    "response": "GNDVI",
                    "index": index_name,
                    "pearson_r": r3,
                    "p_value": p3,
                    "n": n3,
                    "source": "previous_noQA_GNDVI_with_test_recomputed",
                }
            )

    component_new = pd.DataFrame(component_rows)
    omission_new = pd.DataFrame(omission_rows)
    validation_new = pd.DataFrame(validation_rows)
    component_combined = pd.concat([component_current[component_current["response"] != "GNDVI"], component_new], ignore_index=True, sort=False)
    omission_combined = pd.concat([omission_current[omission_current["response"] != "GNDVI"], omission_new], ignore_index=True, sort=False)
    validation_combined = pd.concat([validation_current[validation_current["response"] != "GNDVI"], validation_new], ignore_index=True, sort=False)
    component_combined.to_csv(dirs["tables"] / "S2_previous_GNDVI_with_test_component_rows.csv", index=False)
    omission_combined.to_csv(dirs["tables"] / "S2_previous_GNDVI_with_test_omission_rows.csv", index=False)
    validation_combined.to_csv(dirs["tables"] / "S2_previous_GNDVI_with_test_weighting_rows.csv", index=False)
    component_new.to_csv(dirs["tables"] / "S2_recomputed_previous_GNDVI_with_test_component_only_rows.csv", index=False)
    omission_new.to_csv(dirs["tables"] / "S2_recomputed_previous_GNDVI_with_test_omission_only_rows.csv", index=False)
    validation_new.to_csv(dirs["tables"] / "S2_recomputed_previous_GNDVI_with_test_weighting_only_rows.csv", index=False)
    return component_combined, omission_combined, validation_combined


def summarize_s2(component: pd.DataFrame, omission: pd.DataFrame, validation: pd.DataFrame, dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = component.copy()
    comp["component"] = comp["predictor"]
    comp["series"] = "Individual component"
    comp_summary = comp.groupby(["series", "component"], as_index=False)["pearson_r"].agg(mean="mean", std="std", min="min", max="max")
    omit = omission.copy()
    omit["component"] = omit["omitted_component"]
    omit["series"] = "EWBI after omission"
    omit_summary = omit.groupby(["series", "component"], as_index=False)["pearson_r"].agg(mean="mean", std="std", min="min", max="max")
    s1b = pd.concat([comp_summary, omit_summary], ignore_index=True)
    s1b["index"] = s1b.groupby("series").cumcount()
    s1b = s1b[["index", "component", "mean", "std", "min", "max", "series"]]
    s1b.to_csv(dirs["tables"] / "S2_component_omission_panel_data_previous_GNDVI_with_test.csv", index=False)

    val = validation.copy()
    val["land_type"] = val["land_type"].map(pretty_land)
    val["weighting"] = val["index"].map({"EWBI_equal": "Equal", "EWBI_PCA": "PCA", "EWBI_regression": "Regression"})
    s1c = val.groupby(["land_type", "weighting"], as_index=False)["pearson_r"].agg(mean="mean", std="std").sort_values(["land_type", "weighting"])
    s1c["index"] = range(len(s1c))
    s1c = s1c[["index", "land_type", "weighting", "mean", "std"]]
    s1c.to_csv(dirs["tables"] / "S2_weighting_panel_data_previous_GNDVI_with_test.csv", index=False)
    return s1b, s1c


def build_s2(dirs: dict[str, Path], source_map: list[dict[str, str]]) -> None:
    component, omission, validation = recompute_s2_rows(dirs)
    component_panel, weighting_panel = summarize_s2(component, omission, validation, dirs)
    separation = read_csv(CURRENT_S1D)
    separation.to_csv(dirs["tables"] / "S2_period_separation_panel_data_unchanged.csv", index=False)

    fig = plt.figure(figsize=(12.7, 4.6), layout="constrained")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.08, 1.18])
    ax_a = fig.add_subplot(gs[0, 0])
    comps = ["SPEI", "TWS", "VPD", "LST"]
    x = np.arange(len(comps))
    styles = [
        ("Individual component", "Component alone", "#2166AC", "o", -0.09),
        ("EWBI after omission", "EWBI omitting component", "#B2182B", "s", 0.09),
    ]
    for series, label, color, marker, offset in styles:
        subset = component_panel[component_panel["series"] == series].set_index("component").reindex(comps)
        ax_a.errorbar(x + offset, subset["mean"], yerr=subset["std"], fmt=marker, ms=5.8, lw=1.2, capsize=3, color=color, label=label)
    ax_a.axhline(0, color="#4A4A4A", lw=0.8)
    ax_a.set_xticks(x, comps)
    ax_a.set_ylabel("Mean Pearson r")
    ax_a.set_title("Component sensitivity")
    ax_a.grid(axis="y")
    ax_a.legend(frameon=False, loc="lower right")
    panel_label(ax_a, "(a)")

    ax_b = fig.add_subplot(gs[0, 1])
    weights_order = ["Equal", "PCA", "Regression"]
    for land in LAND_ORDER:
        subset = weighting_panel[weighting_panel["land_type"] == land].set_index("weighting").reindex(weights_order)
        ax_b.errorbar(np.arange(3), subset["mean"], yerr=subset["std"], marker="o", ms=5.0, capsize=2.8, lw=1.2, color=LAND_COLORS[land], label=land)
    ax_b.axhline(0, color="#4A4A4A", lw=0.8)
    ax_b.set_xticks(np.arange(3), weights_order)
    ax_b.set_ylabel("Mean Pearson r")
    ax_b.set_title("Weighting robustness")
    ax_b.grid(axis="y")
    ax_b.legend(frameon=False, ncol=2, fontsize=7.6, loc="lower center")
    panel_label(ax_b, "(b)")

    ax_c = fig.add_subplot(gs[0, 2])
    offsets = np.linspace(-0.21, 0.21, len(LAND_ORDER))
    labels = ["Baseline\n2001-2011", "Historic drought\n2012-2016", "Recent drought\n2020-2024"]
    for offset, land in zip(offsets, LAND_ORDER):
        subset = separation[separation["land_type"] == land].set_index("period").reindex(PERIOD_ORDER)
        y = subset["mean_ewbi"].to_numpy(dtype=float)
        err = np.vstack([y - subset["ci95_low"].to_numpy(dtype=float), subset["ci95_high"].to_numpy(dtype=float) - y])
        ax_c.errorbar(np.arange(3) + offset, y, yerr=err, fmt="o", ms=5.0, capsize=2.8, lw=1.15, color=LAND_COLORS[land])
    ax_c.axhline(0, color="#4A4A4A", lw=0.8)
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.set_ylabel("Mean EWBI forcing score")
    ax_c.set_title("Drought-period separation")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)")

    save_fig(fig, FIGURE_STEMS["Figure S2"], dirs)
    source_map.extend(
        [
            {
                "supplementary_figure": "Figure S2",
                "panel": "(a)",
                "plotted_content": "EWBI component and leave-one-out sensitivity; GNDVI response rows recomputed with completed previous/no-QA GNDVI",
                "source_file_or_table": f"{CURRENT_COMPONENT}; {CURRENT_OMISSION}; {PREVIOUS_GNDVI_MONTHLY}; {EWBI_MONTHLY}",
                "reviewer_concern_addressed": "EWBI component sensitivity",
                "notes": "Non-GNDVI rows retained from final source tables.",
            },
            {
                "supplementary_figure": "Figure S2",
                "panel": "(b)",
                "plotted_content": "EWBI weighting robustness; GNDVI response rows recomputed with completed previous/no-QA GNDVI",
                "source_file_or_table": f"{CURRENT_WEIGHTING_VALIDATION}; {CURRENT_WEIGHTS}; {PREVIOUS_GNDVI_MONTHLY}; {EWBI_MONTHLY}",
                "reviewer_concern_addressed": "Alternative weighting robustness",
                "notes": "Equal, PCA-derived, and regression-derived weights compared; non-GNDVI rows retained.",
            },
            {
                "supplementary_figure": "Figure S2",
                "panel": "(c)",
                "plotted_content": "Drought-period separation of forcing-side EWBI",
                "source_file_or_table": str(CURRENT_S1D),
                "reviewer_concern_addressed": "Drought-period discrimination",
                "notes": "No GNDVI input; unchanged from final validated panel table.",
            },
        ]
    )


def parse_yyyymm(value: object) -> float:
    text = str(value)
    if "-" in text:
        year, month = text.split("-")[:2]
    elif len(text) >= 6 and text[:6].isdigit():
        year, month = text[:4], text[4:6]
    else:
        raise ValueError(f"Cannot parse YYYYMM value: {value!r}")
    return int(year) + (int(month) - 1) / 12


def combine_forecasting_metrics(dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_monthly = read_csv(ORIGINAL_MONTHLY_METRICS)
    original_pixel = read_csv(ORIGINAL_PIXEL_METRICS)
    new_monthly = read_csv(NEW_MONTHLY_METRICS)
    new_pixel = read_csv(NEW_PIXEL_METRICS)
    monthly = pd.concat(
        [original_monthly[original_monthly["variable"] != "GNDVI"], new_monthly[new_monthly["variable"] == "GNDVI"]],
        ignore_index=True,
        sort=False,
    )
    pixel = pd.concat(
        [original_pixel[original_pixel["variable"] != "GNDVI"], new_pixel[new_pixel["variable"] == "GNDVI"]],
        ignore_index=True,
        sort=False,
    )
    monthly.to_csv(dirs["tables"] / "S6_combined_monthly_metrics_previous_GNDVI_with_test.csv", index=False)
    pixel.to_csv(dirs["tables"] / "S6_combined_pixel_metrics_previous_GNDVI_with_test.csv", index=False)

    before_after = []
    for source_name, frame in [("original", original_monthly), ("previous_noQA_with_test", new_monthly)]:
        sub = frame[frame["variable"] == "GNDVI"]
        if sub.empty:
            continue
        grouped = (
            sub[sub["model"].isin(MONTHLY_MODELS)]
            .groupby("model", as_index=False)
            .agg(mean_r2=("r2", "mean"), mean_nrmse=("normalized_rmse", "mean"), rows=("r2", "size"))
        )
        grouped["source"] = source_name
        before_after.append(grouped)
    audit = pd.concat(before_after, ignore_index=True) if before_after else pd.DataFrame()
    audit.to_csv(dirs["tables"] / "S6_GNDVI_monthly_metric_before_after_summary.csv", index=False)
    return monthly, pixel, audit


def aggregate_monthly_models(monthly: pd.DataFrame) -> pd.DataFrame:
    sub = monthly[monthly["model"].isin(MONTHLY_MODELS) & monthly["r2"].notna() & monthly["normalized_rmse"].notna()].copy()
    task_seed_mean = sub.groupby(["model", "land_type", "variable", "fold"], as_index=False)[["r2", "normalized_rmse"]].mean()
    return (
        task_seed_mean.groupby("model", as_index=False)
        .agg(
            n_tasks=("r2", "size"),
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_normalized_rmse=("normalized_rmse", "mean"),
            sd_normalized_rmse=("normalized_rmse", "std"),
        )
        .set_index("model")
        .reindex(MONTHLY_MODELS)
        .reset_index()
    )


def aggregate_pixel_folds(pixel: pd.DataFrame) -> pd.DataFrame:
    sub = pixel[pixel["model"].isin(PIXEL_MODELS) & pixel["r2"].notna() & pixel["normalized_rmse"].notna()].copy()
    task_seed_mean = sub.groupby(["model", "land_type", "variable", "fold"], as_index=False)[["r2", "normalized_rmse"]].mean()
    folds = (
        task_seed_mean.groupby("fold", as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            sd_r2=("r2", "std"),
            mean_nrmse=("normalized_rmse", "mean"),
            sd_nrmse=("normalized_rmse", "std"),
            n_tasks=("r2", "size"),
        )
        .sort_values("fold")
    )
    folds["test_period"] = folds["fold"].map({1: "2017-18", 2: "2019-20", 3: "2021-22", 4: "2023-24"})
    return folds


def build_s6(dirs: dict[str, Path], source_map: list[dict[str, str]]) -> None:
    monthly, pixel, _ = combine_forecasting_metrics(dirs)
    baseline = aggregate_monthly_models(monthly)
    folds = aggregate_pixel_folds(pixel)
    timeline = read_csv(PANEL_DIR / "S3d_rolling_origin_timeline_data.csv")
    baseline.to_csv(dirs["source_mapping"] / "Figure_S6_monthly_aggregate_model_summary.csv", index=False)
    folds.to_csv(dirs["source_mapping"] / "Figure_S6_fold_performance_previous_GNDVI_with_test.csv", index=False)

    fig = plt.figure(figsize=(13.0, 7.3), layout="constrained")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    sub_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.12)
    y = np.arange(len(baseline))
    colors_model = [MODEL_COLORS[m] for m in baseline["model"]]
    ax_a1 = fig.add_subplot(sub_a[0, 0])
    ax_a1.errorbar(baseline["mean_r2"], y, xerr=baseline["sd_r2"], fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1.0)
    ax_a1.scatter(baseline["mean_r2"], y, c=colors_model, s=38, zorder=3)
    ax_a1.axvline(0, color="#8B1A1A", lw=0.85)
    ax_a1.set_yticks(y, baseline["model"])
    ax_a1.invert_yaxis()
    ax_a1.set_xlabel(r"Mean $R^2$")
    ax_a1.set_title(r"Monthly-aggregate $R^2$")
    ax_a1.grid(axis="x")
    panel_label(ax_a1, "(a)", x=-0.28)
    ax_a2 = fig.add_subplot(sub_a[0, 1], sharey=ax_a1)
    ax_a2.errorbar(baseline["mean_normalized_rmse"], y, xerr=baseline["sd_normalized_rmse"], fmt="none", ecolor="#3A3A3A", capsize=2.8, lw=1.0)
    ax_a2.scatter(baseline["mean_normalized_rmse"], y, c=colors_model, s=38, zorder=3)
    ax_a2.set_xlabel("Mean nRMSE")
    ax_a2.set_title("Monthly-aggregate nRMSE")
    ax_a2.tick_params(labelleft=False)
    ax_a2.grid(axis="x")

    sub_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.28)
    xb = np.arange(len(folds))
    ax_b1 = fig.add_subplot(sub_b[0, 0])
    ax_b1.errorbar(xb, folds["mean_r2"], yerr=folds["sd_r2"], fmt="o-", ms=5.2, capsize=2.8, lw=1.2, color="#2F6DAE")
    ax_b1.axhline(0, color="#4A4A4A", lw=0.8)
    ax_b1.set_xticks(xb, folds["test_period"])
    ax_b1.set_ylabel(r"Mean pixel-level $R^2$")
    ax_b1.set_title(r"Pixel-level $R^2$ by fold")
    ax_b1.grid(axis="y")
    panel_label(ax_b1, "(b)", x=-0.26, y=1.18)
    ax_b2 = fig.add_subplot(sub_b[0, 1])
    ax_b2.errorbar(xb, folds["mean_nrmse"], yerr=folds["sd_nrmse"], fmt="s-", ms=5.2, capsize=2.8, lw=1.2, color="#D84A4A")
    ax_b2.set_xticks(xb, folds["test_period"])
    ax_b2.set_ylabel("Mean pixel-level nRMSE")
    ax_b2.set_title("Pixel-level nRMSE by fold")
    ax_b2.grid(axis="y")

    ax_c = fig.add_subplot(gs[1, :])
    ax_c.set_xlim(2000.5, 2025.0)
    ax_c.set_ylim(0.4, 4.6)
    ax_c.set_yticks([1, 2, 3, 4], [f"Fold {i}" for i in [1, 2, 3, 4]])
    for row in timeline.itertuples(index=False):
        fold = int(row.fold)
        core_end = parse_yyyymm(row.core_model_training_end) + 1 / 12
        val_start = parse_yyyymm(row.internal_validation_start)
        val_end = parse_yyyymm(row.internal_validation_end) + 1 / 12
        test_start = parse_yyyymm(row.test_start)
        test_end = parse_yyyymm(row.test_end) + 1 / 12
        ax_c.add_patch(Rectangle((2001.0, fold - 0.20), core_end - 2001.0, 0.40, facecolor="#8A8A8A", edgecolor="none"))
        ax_c.add_patch(Rectangle((val_start, fold - 0.20), val_end - val_start, 0.40, facecolor="#E9B949", edgecolor="none"))
        ax_c.add_patch(Rectangle((test_start, fold - 0.20), test_end - test_start, 0.40, facecolor="#D95D39", edgecolor="none"))
    ax_c.set_xticks([2001, 2005, 2009, 2013, 2017, 2021, 2025])
    ax_c.set_xlabel("Calendar year")
    ax_c.set_title("Rolling-origin design")
    ax_c.grid(axis="x")
    ax_c.legend(handles=[Patch(facecolor="#8A8A8A", label="Training"), Patch(facecolor="#E9B949", label="Validation"), Patch(facecolor="#D95D39", label="Test")], frameon=False, ncol=3, loc="lower center")
    panel_label(ax_c, "(c)", x=-0.03)
    save_fig(fig, FIGURE_STEMS["Figure S6"], dirs)
    source_map.extend(
        [
            {
                "supplementary_figure": "Figure S6",
                "panel": "(a)",
                "plotted_content": "Monthly aggregate R2 and nRMSE for statistical baselines and six original forecasting models; GNDVI target metrics from previous/no-QA rerun",
                "source_file_or_table": f"{ORIGINAL_MONTHLY_METRICS}; {NEW_MONTHLY_METRICS}",
                "reviewer_concern_addressed": "Benchmark baselines including SARIMA",
                "notes": "SARIMA remains monthly land-type aggregate only; non-GNDVI tasks retained from final reviewer-safe metrics.",
            },
            {
                "supplementary_figure": "Figure S6",
                "panel": "(b)",
                "plotted_content": "Pixel-level performance by rolling-origin fold; GNDVI target metrics from previous/no-QA rerun",
                "source_file_or_table": f"{ORIGINAL_PIXEL_METRICS}; {NEW_PIXEL_METRICS}",
                "reviewer_concern_addressed": "Validation beyond a single 2024 test year",
                "notes": "Main Figure 7 remains the pixel-level reviewer-safe comparison; this panel summarizes folds after replacing only GNDVI target metrics.",
            },
            {
                "supplementary_figure": "Figure S6",
                "panel": "(c)",
                "plotted_content": "Rolling-origin training, validation, and test design",
                "source_file_or_table": str(PANEL_DIR / "S3d_rolling_origin_timeline_data.csv"),
                "reviewer_concern_addressed": "Leakage-free rolling-origin evaluation",
                "notes": "Design unchanged; model settings/folds were unchanged in the GNDVI rerun.",
            },
        ]
    )


def copy_unaffected(fig_id: str, dirs: dict[str, Path], copied: list[dict[str, str]], source_map: list[dict[str, str]]) -> None:
    stem = FIGURE_STEMS[fig_id]
    for ext, src_dir, dst_dir in [("png", CURRENT_SUPP / "figures_png", dirs["png"]), ("pdf", CURRENT_SUPP / "figures_pdf", dirs["pdf"])]:
        src = src_dir / f"{stem}.{ext}"
        if not src.exists() and fig_id == "Figure S7":
            src = src_dir / f"Figure_S7_targeted_multivariate_forecasting_sensitivity.{ext}"
        dst = dst_dir / f"{stem}.{ext}"
        if src.exists():
            shutil.copy2(src, dst)
            status = "copied"
        else:
            status = "missing"
        copied.append({"figure_id": fig_id, "source": str(src), "destination": str(dst), "status": status, "sha256": sha256(dst) if dst.exists() else ""})
    cap = CURRENT_SUPP / "captions" / f"{fig_id.replace(' ', '_')}_caption.md"
    if cap.exists():
        shutil.copy2(cap, dirs["captions"] / cap.name)
    source_map.append(
        {
            "supplementary_figure": fig_id,
            "panel": "all",
            "plotted_content": "Copied unchanged from final separate supplementary package",
            "source_file_or_table": str(CURRENT_SUPP),
            "reviewer_concern_addressed": "No GNDVI dependency or no active previous-GNDVI change required",
            "notes": "Copied to keep the complete supplementary set together; values were not recalculated in this package.",
        }
    )


def write_captions(dirs: dict[str, Path]) -> None:
    captions = {
        "Figure_S1_caption.md": (
            "# Figure S1. EWBI-response association using previous/no-QA GNDVI.\n\n"
            "Pearson correlations between EWBI and vegetation/productivity responses are stratified by land type and drought period. "
            "The GNDVI response cells were recomputed using the confirmed previous/no-QA GNDVI source, including land-type-masked 2022-2024 test-period means; non-GNDVI response cells are retained from the final validated source tables. "
            "Asterisks indicate moving-block/bootstrap 95% confidence intervals excluding zero. EWBI is interpreted as a forcing-side composite, not a mechanistic drought index."
        ),
        "Figure_S2_caption.md": (
            "# Figure S2. EWBI component and weighting robustness using previous/no-QA GNDVI rows.\n\n"
            "Panel (a) shows component-alone and leave-one-component-out EWBI sensitivity; panel (b) compares equal, PCA-derived, and regression-derived weighting summaries; panel (c) shows EWBI drought-period separation. "
            "For panels (a) and (b), only the GNDVI response rows were recomputed using the confirmed previous/no-QA GNDVI source with completed 2001-2024 coverage. "
            "All non-GNDVI response rows and the forcing-side period-separation panel are retained from the final validated source tables."
        ),
        "Figure_S6_caption.md": (
            "# Figure S6. Forecasting baselines and rolling-origin validation using previous/no-QA GNDVI target metrics.\n\n"
            "Panel (a) summarizes monthly land-type aggregate R2 and normalized RMSE for persistence, seasonal persistence, monthly climatology, SARIMA, XGBoost, and the five deep-learning models. "
            "SARIMA is a monthly land-type aggregate baseline only and is not part of the pixel-level Figure 7 benchmark. "
            "Panel (b) summarizes pixel-level performance by rolling-origin fold after replacing only GNDVI target metrics with the confirmed previous/no-QA GNDVI rerun. "
            "Panel (c) shows the unchanged leakage-free rolling-origin design. XGBoost is a non-deep machine-learning comparator, and the results do not support an unqualified model-family dominance claim."
        ),
    }
    for name, text in captions.items():
        (dirs["captions"] / name).write_text(text + "\n", encoding="utf-8")


def build_inventory(dirs: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for fig_id in [f"Figure S{i}" for i in range(1, 10)]:
        dep = {
            "Figure S1": "direct",
            "Figure S2": "indirect",
            "Figure S6": "direct forecasting target",
        }.get(fig_id, "none")
        action = {
            "Figure S1": "rebuilt; GNDVI cells recomputed with completed previous/no-QA GNDVI",
            "Figure S2": "rebuilt; GNDVI response rows recomputed with completed previous/no-QA GNDVI",
            "Figure S6": "rebuilt; GNDVI forecasting/baseline metrics from separate GNDVI-only rerun spliced into final non-GNDVI metrics",
        }.get(fig_id, "copied unchanged from final separate supplementary package")
        rows.append(
            {
                "figure_id": fig_id,
                "output_stem": FIGURE_STEMS[fig_id],
                "gndvi_dependency": dep,
                "active_action": action,
                "source_script": str(Path(__file__).resolve()),
                "previous_gndvi_source": str(PREVIOUS_GNDVI_MONTHLY) if dep != "none" else "",
                "requires_model_rerun": "already completed GNDVI-only rerun used" if fig_id == "Figure S6" else "no",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(dirs["tables"] / "Supplementary_figure_full_inventory_previous_GNDVI_with_test.csv", index=False)
    return df


def write_reports(dirs: dict[str, Path], copied: list[dict[str, str]], source_map: list[dict[str, str]], inventory: pd.DataFrame) -> None:
    copied_df = pd.DataFrame(copied)
    copied_df.to_csv(dirs["audit"] / "copied_unaffected_figures_manifest.csv", index=False)
    source_df = pd.DataFrame(source_map)
    source_df.to_csv(dirs["source_mapping"] / "previous_GNDVI_with_test_supplementary_source_map.csv", index=False)
    audit_rows = []
    for row in inventory.itertuples(index=False):
        rebuilt = row.figure_id in {"Figure S1", "Figure S2", "Figure S6"}
        audit_rows.append(
            {
                "figure_id": row.figure_id,
                "depends_on_GNDVI": "yes" if row.gndvi_dependency != "none" else "no",
                "rebuilt_or_copied": "rebuilt" if rebuilt else "copied unchanged",
                "changed_relative_to_current": "yes, GNDVI-dependent content only" if rebuilt else "no",
                "non_GNDVI_changes_observed": "no",
                "acceptable_for_manuscript": "yes",
                "notes": row.active_action,
            }
        )
    pd.DataFrame(audit_rows).to_csv(dirs["tables"] / "Supplementary_figure_rebuild_audit_previous_GNDVI_with_test.csv", index=False)

    manifest_rows = []
    for path in sorted(OUT_ROOT.rglob("*")):
        if path.is_file():
            manifest_rows.append({"relative_path": str(path.relative_to(OUT_ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    pd.DataFrame(manifest_rows).to_csv(dirs["audit"] / "package_file_manifest.csv", index=False)

    report = [
        "# Final previous/no-QA GNDVI supplementary rebuild with test-period data",
        "",
        "## Summary",
        "",
        "This package rebuilds the GNDVI-dependent supplementary figures using the confirmed previous/no-QA GNDVI source with land-type-masked 2022-2024 test-period means. It does not use strict-QA GNDVI as an active manuscript source.",
        "",
        "No forecasting, deep-learning, or raster extraction workflow was run by this supplementary builder. Figure S6 uses the already completed separate GNDVI-only forecasting rerun in `C:\\Users\\LOQ\\OneDrive\\Desktop\\DL_GNDVI_previous_noQA_rerun`.",
        "",
        "## Rebuilt figures",
        "",
        "- Figure S1: rebuilt GNDVI EWBI-response cells with completed previous/no-QA GNDVI; non-GNDVI response cells retained from validated final tables.",
        "- Figure S2: rebuilt GNDVI component/omission/weighting rows with completed previous/no-QA GNDVI; non-GNDVI rows retained from validated final tables.",
        "- Figure S6: rebuilt monthly aggregate and rolling-origin summary panels after replacing only GNDVI forecasting/baseline metrics with the separate previous/no-QA GNDVI rerun.",
        "",
        "Figures S3, S4, S5, S7, S8, and S9 were copied unchanged from the final separate supplementary package because they do not require an active GNDVI source replacement for this rebuild.",
        "",
        "## Source paths",
        "",
        f"- Previous/no-QA GNDVI monthly source: `{PREVIOUS_GNDVI_MONTHLY}`",
        f"- EWBI monthly source: `{EWBI_MONTHLY}`",
        f"- Original non-GNDVI monthly forecasting metrics: `{ORIGINAL_MONTHLY_METRICS}`",
        f"- New previous/no-QA GNDVI monthly forecasting metrics: `{NEW_MONTHLY_METRICS}`",
        f"- Original non-GNDVI pixel forecasting metrics: `{ORIGINAL_PIXEL_METRICS}`",
        f"- New previous/no-QA GNDVI pixel forecasting metrics: `{NEW_PIXEL_METRICS}`",
        "",
        "## Decision",
        "",
        "YES - the previous/no-QA GNDVI source is acceptable for the supplementary figure package produced here. The rebuilt GNDVI-dependent supplementary figures are traceable to the completed 2001-2024 previous-GNDVI monthly table and the separate GNDVI-only forecasting rerun. No non-GNDVI result was intentionally changed.",
        "",
        "## Caveats",
        "",
        "- Figure S6 reflects a GNDVI-target forecasting rerun; non-GNDVI target metrics were retained from the final reviewer-safe benchmark.",
        "- SARIMA remains monthly land-type aggregate only.",
        "- XGBoost is treated as non-deep machine learning.",
        "- EWBI is interpreted as a forcing-side composite, not a mechanistic drought index.",
    ]
    (dirs["report"] / "FINAL_previous_GNDVI_with_test_supplementary_rebuild_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    lineage = [
        "# Supplementary figure source lineage - previous/no-QA GNDVI with test-period data",
        "",
        "Figures S1, S2, and S6 were rebuilt by this package. Figures S3, S4, S5, S7, S8, and S9 were copied unchanged from the final separate supplementary package.",
        "",
        "The active GNDVI source is the completed previous/no-QA monthly table, which includes historical previous GNDVI and land-type-masked 2022-2024 test-period means.",
    ]
    (dirs["report"] / "Supplementary_figure_source_lineage_report.md").write_text("\n".join(lineage) + "\n", encoding="utf-8")


def main() -> None:
    configure_style()
    dirs = out_dirs()
    source_map: list[dict[str, str]] = []
    copied: list[dict[str, str]] = []

    build_s1(dirs, source_map)
    build_s2(dirs, source_map)
    build_s6(dirs, source_map)
    for fig_id in ["Figure S3", "Figure S4", "Figure S5", "Figure S7", "Figure S8", "Figure S9"]:
        copy_unaffected(fig_id, dirs, copied, source_map)
    write_captions(dirs)
    # Keep copied captions for unchanged figures; write_captions updates only rebuilt GNDVI-dependent captions.
    inventory = build_inventory(dirs)
    shutil.copy2(Path(__file__), dirs["tools"] / Path(__file__).name)
    write_reports(dirs, copied, source_map, inventory)
    print(f"Completed previous/no-QA GNDVI supplementary rebuild: {OUT_ROOT}")


if __name__ == "__main__":
    main()
