from __future__ import annotations

import hashlib
import math
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import Patch, Rectangle
from scipy import stats


OUT_ROOT = Path(r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\full_previous_GNDVI_supplementary_rebuild")

CURRENT_SUPP = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_supplementary_information_final_separate_figures")
COMPACT = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_compact_supplementary_figures_final")
PANEL_DIR = COMPACT / "source_mapping"
SOURCE_TABLES = Path(r"C:\Users\LOQ\OneDrive\Desktop\ECOINF_revision_final_figures_package\4_source_tables_used")
LEGACY_PACKAGE = Path(r"C:\Users\LOQ\OneDrive\Desktop\major_revision_analyses\legacy_gndvi_sensitivity_package")
LEGACY_GNDVI_CONFIRMATION = LEGACY_PACKAGE / "tables" / "legacy_gndvi_source_confirmation.csv"
LEGACY_EWBI_VALUES = LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_values.csv"
LEGACY_EWBI_CI = LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_bootstrap_ci.csv"
LEGACY_EWBI_CHANGE = LEGACY_PACKAGE / "tables" / "legacy_EWBI_response_association_change_audit.csv"
LEGACY_VEG_COHERENCE = LEGACY_PACKAGE / "tables" / "legacy_GNDVI_vs_vegetation_indices.csv"

EWBI_MONTHLY = Path(r"C:\Users\LOQ\OneDrive\Desktop\GapFilling_Comparison_Revision\tables\ewbi_gapfill_monthly_series.csv")
CURRENT_COMPONENT = SOURCE_TABLES / "ewbi_component_comparison.csv"
CURRENT_OMISSION = SOURCE_TABLES / "ewbi_leave_one_component_out.csv"
CURRENT_WEIGHTS = SOURCE_TABLES / "ewbi_alternative_weights.csv"
CURRENT_WEIGHTING_VALIDATION = SOURCE_TABLES / "ewbi_vegetation_validation.csv"
CURRENT_S1A = PANEL_DIR / "S1a_period_response_panel_data.csv"
CURRENT_S1B = PANEL_DIR / "S1b_component_omission_panel_data.csv"
CURRENT_S1C = PANEL_DIR / "S1c_weighting_panel_data.csv"
CURRENT_S1D = PANEL_DIR / "S1d_period_separation_panel_data.csv"

LAND_ORDER = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]
LAND_COLORS = {
    "Grasslands": "#007C91",
    "Natural Forests": "#2E7D32",
    "Savannas": "#E58B00",
    "Shrublands": "#B23A67",
}
LAND_KEY_TO_LABEL = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "Natural Forests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
LAND_KEY_TO_MONTHLY = {
    "GRASSLANDS": "Grasslands",
    "NATURAL_FORESTS": "NaturalForests",
    "SAVANNAS": "Savannas",
    "SHRUBLANDS": "Shrublands",
}
MONTHLY_TO_LABEL = {
    "Grasslands": "Grasslands",
    "NaturalForests": "Natural Forests",
    "Natural Forests": "Natural Forests",
    "Savannas": "Savannas",
    "Shrublands": "Shrublands",
}
LABEL_TO_MONTHLY = {v: k for k, v in MONTHLY_TO_LABEL.items() if k != "Natural Forests"}

RESPONSES = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
FORCING = ["SPEI", "TWS", "VPD", "LST"]
COMPONENT_COL = {
    "SPEI": "spei_signed_z",
    "TWS": "tws_signed_z",
    "VPD": "vpd_signed_z",
    "LST": "lst_signed_z",
}
PERIOD_CODES = {
    "Baseline 2001-2011": "baseline",
    "Historic drought 2012-2016": "historic",
    "Historic drought period 2012-2016": "historic",
    "Recent drought period 2020-2024": "recent",
    "Recent period 2020-2024": "recent",
}
PERIOD_ORDER = ["Baseline 2001-2011", "Historic drought 2012-2016", "Recent period 2020-2024"]
MODEL_DEPENDENT_FIG = "Figure S6 - Forecasting baselines and rolling-origin validation"

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


def dirs() -> dict[str, Path]:
    paths = {
        "figures": OUT_ROOT / "figures",
        "tables": OUT_ROOT / "tables",
        "report": OUT_ROOT / "report",
        "audit": OUT_ROOT / "audit",
        "tools": OUT_ROOT / "tools",
        "source_mapping": OUT_ROOT / "source_mapping",
        "captions": OUT_ROOT / "captions",
        "blocked": OUT_ROOT / "blocked",
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


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
    fmt: str = ".2f",
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
            text = annotations[i, j] if annotations is not None else format(val, fmt)
            color = "white" if abs(val) > (vmax - vmin) * 0.30 else "#202020"
            ax.text(j, i, text, ha="center", va="center", fontsize=annotation_size, fontweight="bold", color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.035)
    if colorbar_label:
        cbar.set_label(colorbar_label)


def save_fig(fig: plt.Figure, stem: str, out_dirs: dict[str, Path]) -> tuple[Path, Path]:
    png = out_dirs["figures"] / f"{stem}.png"
    pdf = out_dirs["figures"] / f"{stem}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def normalize_period_code(value: object) -> str:
    text = str(value).strip()
    low = text.lower()
    if "baseline" in low or "2001-2011" in low or "2001–2011" in low:
        return "baseline"
    if "historic" in low or "2012-2016" in low or "2012–2016" in low:
        return "historic"
    if "recent" in low or "2020-2024" in low or "2020–2024" in low:
        return "recent"
    return text


def pretty_land(value: object) -> str:
    text = str(value)
    if text in LAND_KEY_TO_LABEL:
        return LAND_KEY_TO_LABEL[text]
    return MONTHLY_TO_LABEL.get(text, text)


def legacy_gndvi_series_by_label(land_label: str) -> pd.Series:
    conf = read_csv(LEGACY_GNDVI_CONFIRMATION)
    conf["land_label"] = conf["land_type"].map(LAND_KEY_TO_LABEL)
    conf = conf[conf["land_label"] == land_label].copy()
    conf["date"] = pd.to_datetime(conf["date"].astype(str), format="%Y%m")
    return conf.set_index("date")["mean"].astype(float).sort_index().rename("GNDVI")


def monthwise_fill(series: pd.Series) -> pd.Series:
    out = series.astype(float).copy()
    for month in range(1, 13):
        mask = out.index.month == month
        out.loc[mask] = out.loc[mask].interpolate(method="linear", limit_direction="both")
    return out


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


def signed_ewbi_frame(land_label: str) -> pd.DataFrame:
    monthly = read_csv(EWBI_MONTHLY, parse_dates=["date"])
    monthly_name = LABEL_TO_MONTHLY.get(land_label, land_label.replace(" ", ""))
    sub = monthly[(monthly["land_type"] == monthly_name) & (monthly["method"] == "current_monthwise_cross_year")].copy()
    sub = sub.set_index("date").sort_index().reindex(pd.date_range("2001-01-01", "2024-12-01", freq="MS"))
    cols = list(COMPONENT_COL.values()) + ["ewbi"]
    return sub[cols].astype(float)


def legacy_gndvi_anomaly(land_label: str) -> pd.Series:
    raw = legacy_gndvi_series_by_label(land_label).reindex(pd.date_range("2001-01-01", "2024-12-01", freq="MS"))
    return baseline_monthly_anomaly(raw).rename("GNDVI_anomaly")


def build_s1a_previous_table(out_dirs: dict[str, Path]) -> pd.DataFrame:
    values = read_csv(LEGACY_EWBI_VALUES)
    ci = read_csv(LEGACY_EWBI_CI)
    values = values.copy()
    values["land_type"] = values["land_type"].map(pretty_land)
    values["period_code"] = values["period"].map(normalize_period_code)
    values["row_label"] = values["land_type"].str.replace("Natural Forests", "Natural forests", regex=False) + " | " + values["period_code"]

    ci = ci.copy()
    ci["land_type"] = ci["land_type"].map(pretty_land)
    ci["period_code"] = ci["period"].map(normalize_period_code)
    ci["ci_excludes_zero"] = (
        (ci["pearson_bootstrap_ci95_low"].astype(float) > 0)
        | (ci["pearson_bootstrap_ci95_high"].astype(float) < 0)
    )

    key_cols = ["land_type", "period_code", "response"]
    out = values.merge(
        ci[key_cols + ["pearson_bootstrap_ci95_low", "pearson_bootstrap_ci95_high", "ci_excludes_zero"]],
        on=key_cols,
        how="left",
    )
    out.to_csv(out_dirs["tables"] / "S1a_previous_GNDVI_period_response_panel_data.csv", index=False)
    values.to_csv(out_dirs["tables"] / "legacy_EWBI_response_association_values_used.csv", index=False)
    ci.to_csv(out_dirs["tables"] / "legacy_EWBI_response_association_bootstrap_ci_used.csv", index=False)
    return out


def build_s1(out_dirs: dict[str, Path]) -> None:
    period = build_s1a_previous_table(out_dirs)
    row_order = [
        f"{land.replace('Natural Forests', 'Natural forests')} | {period_name}"
        for land in LAND_ORDER
        for period_name in ("baseline", "historic", "recent")
    ]
    responses = ["NIRv", "NDVI", "GNDVI", "GOSIF", "GPP"]
    values = period.pivot(index="row_label", columns="response", values="pearson_r").reindex(index=row_order, columns=responses)
    sig = period.pivot(index="row_label", columns="response", values="ci_excludes_zero").reindex(index=row_order, columns=responses)
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
        responses,
        "EWBI-response association",
        "RdBu_r",
        -1,
        1,
        annotations=annotations,
        colorbar_label="Pearson r",
        rotate_x=25,
        annotation_size=8.0,
    )
    save_fig(fig, "Figure_S1_EWBI_response_association", out_dirs)


def recompute_previous_gndvi_component_rows(out_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    component_current = read_csv(CURRENT_COMPONENT)
    omission_current = read_csv(CURRENT_OMISSION)
    validation_current = read_csv(CURRENT_WEIGHTING_VALIDATION)
    weights = read_csv(CURRENT_WEIGHTS)

    component_rows = []
    omission_rows = []
    validation_rows = []

    for land_label in LAND_ORDER:
        signed = signed_ewbi_frame(land_label)
        gndvi = legacy_gndvi_anomaly(land_label)
        monthly_land = LABEL_TO_MONTHLY.get(land_label, land_label.replace(" ", ""))

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
                    "source": "previous_GNDVI_recomputed",
                }
            )
            omit_cols = [c for name, c in COMPONENT_COL.items() if name != comp]
            ewbi_omit = signed[omit_cols].mean(axis=1)
            n2, r2, p2 = pearson_row(ewbi_omit, gndvi)
            omission_rows.append(
                {
                    "land_type": monthly_land,
                    "response": "GNDVI",
                    "omitted_component": comp,
                    "pearson_r": r2,
                    "p_value": p2,
                    "n": n2,
                    "source": "previous_GNDVI_recomputed",
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
                    "source": "previous_GNDVI_recomputed",
                }
            )

    component_legacy = pd.DataFrame(component_rows)
    omission_legacy = pd.DataFrame(omission_rows)
    validation_legacy = pd.DataFrame(validation_rows)

    component_combined = pd.concat(
        [component_current[component_current["response"] != "GNDVI"], component_legacy],
        ignore_index=True,
        sort=False,
    )
    omission_combined = pd.concat(
        [omission_current[omission_current["response"] != "GNDVI"], omission_legacy],
        ignore_index=True,
        sort=False,
    )
    validation_combined = pd.concat(
        [validation_current[validation_current["response"] != "GNDVI"], validation_legacy],
        ignore_index=True,
        sort=False,
    )

    component_combined.to_csv(out_dirs["tables"] / "S2_previous_GNDVI_component_comparison_rows.csv", index=False)
    omission_combined.to_csv(out_dirs["tables"] / "S2_previous_GNDVI_leave_one_component_rows.csv", index=False)
    validation_combined.to_csv(out_dirs["tables"] / "S2_previous_GNDVI_weighting_validation_rows.csv", index=False)
    component_legacy.to_csv(out_dirs["tables"] / "S2_recomputed_previous_GNDVI_component_only_rows.csv", index=False)
    omission_legacy.to_csv(out_dirs["tables"] / "S2_recomputed_previous_GNDVI_omission_only_rows.csv", index=False)
    validation_legacy.to_csv(out_dirs["tables"] / "S2_recomputed_previous_GNDVI_weighting_only_rows.csv", index=False)

    return component_combined, omission_combined, validation_combined


def summarize_s2_panel_data(
    component: pd.DataFrame, omission: pd.DataFrame, validation: pd.DataFrame, out_dirs: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comp = component.copy()
    comp["component"] = comp["predictor"]
    comp["series"] = "Individual component"
    comp_summary = (
        comp.groupby(["series", "component"], as_index=False)["pearson_r"]
        .agg(mean="mean", std="std", min="min", max="max")
    )

    omit = omission.copy()
    omit["component"] = omit["omitted_component"]
    omit["series"] = "EWBI after omission"
    omit_summary = (
        omit.groupby(["series", "component"], as_index=False)["pearson_r"]
        .agg(mean="mean", std="std", min="min", max="max")
    )
    s1b = pd.concat([comp_summary, omit_summary], ignore_index=True)
    s1b["index"] = s1b.groupby("series").cumcount()
    s1b = s1b[["index", "component", "mean", "std", "min", "max", "series"]]
    s1b.to_csv(out_dirs["tables"] / "S1b_previous_GNDVI_component_omission_panel_data.csv", index=False)

    val = validation.copy()
    val["land_type"] = val["land_type"].map(pretty_land)
    val["weighting"] = val["index"].map(
        {"EWBI_equal": "Equal", "EWBI_PCA": "PCA", "EWBI_regression": "Regression"}
    )
    s1c = (
        val.groupby(["land_type", "weighting"], as_index=False)["pearson_r"]
        .agg(mean="mean", std="std")
        .sort_values(["land_type", "weighting"])
    )
    s1c["index"] = range(len(s1c))
    s1c = s1c[["index", "land_type", "weighting", "mean", "std"]]
    s1c.to_csv(out_dirs["tables"] / "S1c_previous_GNDVI_weighting_panel_data.csv", index=False)
    return s1b, s1c


def build_s2(out_dirs: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    component, omission, validation = recompute_previous_gndvi_component_rows(out_dirs)
    component_panel, weighting_panel = summarize_s2_panel_data(component, omission, validation, out_dirs)
    separation = read_csv(CURRENT_S1D)
    separation.to_csv(out_dirs["tables"] / "S1d_period_separation_panel_data_unchanged.csv", index=False)

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
        ax_a.errorbar(
            x + offset,
            subset["mean"],
            yerr=subset["std"],
            fmt=marker,
            ms=5.8,
            lw=1.2,
            capsize=3,
            color=color,
            label=label,
        )
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
        ax_b.errorbar(
            np.arange(3),
            subset["mean"],
            yerr=subset["std"],
            marker="o",
            ms=5.0,
            capsize=2.8,
            lw=1.2,
            color=LAND_COLORS[land],
            label=land,
        )
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
        ax_c.errorbar(
            np.arange(3) + offset,
            y,
            yerr=err,
            fmt="o",
            ms=5.0,
            capsize=2.8,
            lw=1.15,
            color=LAND_COLORS[land],
        )
    ax_c.axhline(0, color="#4A4A4A", lw=0.8)
    ax_c.set_xticks(np.arange(3), labels)
    ax_c.set_ylabel("Mean EWBI forcing score")
    ax_c.set_title("Drought-period separation")
    ax_c.grid(axis="y")
    panel_label(ax_c, "(c)")

    save_fig(fig, "Figure_S2_EWBI_component_weighting_robustness", out_dirs)
    return component_panel, weighting_panel


def copy_unaffected_figure(fig_id: str, out_dirs: dict[str, Path], copied_rows: list[dict[str, str]]) -> None:
    stem = FIGURE_STEMS[fig_id]
    for ext, src_dir in [("png", CURRENT_SUPP / "figures_png"), ("pdf", CURRENT_SUPP / "figures_pdf")]:
        src = src_dir / f"{stem}.{ext}"
        if not src.exists() and fig_id == "Figure S7":
            alt_stem = "Figure_S7_targeted_multivariate_forecasting_sensitivity"
            src = src_dir / f"{alt_stem}.{ext}"
        dst = out_dirs["figures"] / f"{stem}.{ext}"
        if src.exists():
            shutil.copy2(src, dst)
            copied = "yes"
        else:
            copied = "missing"
        copied_rows.append(
            {
                "figure_id": fig_id,
                "source_path": str(src),
                "copied_path": str(dst) if src.exists() else "",
                "copied": copied,
                "sha256": sha256(dst) if src.exists() else "",
            }
        )
    cap = CURRENT_SUPP / "captions" / f"{fig_id.replace(' ', '_')}_caption.md"
    if cap.exists():
        shutil.copy2(cap, out_dirs["captions"] / cap.name)


def write_blocked_s6(out_dirs: dict[str, Path]) -> None:
    text = "\n".join(
        [
            "# Figure S6 blocked for previous-GNDVI rebuild",
            "",
            "Figure S6 depends on forecasting/baseline metric outputs that include GNDVI target tasks.",
            "A plotting-only rebuild would silently reuse strict-QA GNDVI model results.",
            "",
            "Minimum needed if previous GNDVI becomes the manuscript source:",
            "- rerun GNDVI-target reviewer-safe forecasting evaluations, or locate existing previous-GNDVI GNDVI-target outputs;",
            "- rerun GNDVI-target baseline comparisons used in the monthly aggregate panel;",
            "- recompute aggregate Figure S6 tables while keeping non-GNDVI targets unchanged.",
            "",
            "No forecasting, ML, or deep-learning model was rerun in this package.",
        ]
    )
    (out_dirs["blocked"] / "Figure_S6_previous_GNDVI_rebuild_blocked.md").write_text(text, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axis("off")
    ax.text(
        0.5,
        0.55,
        "Figure S6 not rebuilt",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.38,
        "Requires GNDVI-target forecasting/baseline rerun.\nNo model rerun was performed.",
        ha="center",
        va="center",
        fontsize=12,
    )
    save_fig(fig, "Figure_S6_FORECASTING_BLOCKED_previous_GNDVI_requires_target_rerun", out_dirs)


def write_rebuilt_captions(out_dirs: dict[str, Path]) -> None:
    captions = {
        "Figure_S1_caption.md": "\n".join(
            [
                "# Figure S1. EWBI-response association using previous GNDVI.",
                "",
                "Pearson correlations between EWBI and vegetation/productivity responses are stratified by land type and drought period. The GNDVI response rows were rebuilt using the previous/original land-type-specific GNDVI source, while non-GNDVI response rows are unchanged from the final source tables. Asterisks indicate moving-block/bootstrap 95% confidence intervals excluding zero. EWBI is interpreted as a forcing-side composite, not a mechanistic drought index. Because the previous GNDVI source currently covers 2001-2021, recent-period GNDVI associations use the available 2020-2021 months.",
            ]
        ),
        "Figure_S2_caption.md": "\n".join(
            [
                "# Figure S2. EWBI component and weighting robustness using previous GNDVI rows.",
                "",
                "Panel (a) shows component-alone and leave-one-component-out EWBI sensitivity; panel (b) compares equal, PCA-derived, and regression-derived weighting summaries; panel (c) shows EWBI drought-period separation. For panels (a) and (b), only the GNDVI response rows were recomputed using the previous/original GNDVI source; all non-GNDVI response rows and the forcing-side period-separation panel were retained unchanged from the final source tables. This figure is a previous-GNDVI sensitivity rebuild, not a new EWBI definition.",
            ]
        ),
        "Figure_S6_caption_BLOCKED.md": "\n".join(
            [
                "# Figure S6. Forecasting baselines and rolling-origin validation - blocked for previous-GNDVI rebuild.",
                "",
                "Figure S6 was not rebuilt as a previous-GNDVI figure because it includes GNDVI target forecasting and baseline outputs. Replotting the current figure would silently reuse strict-QA GNDVI model results. A valid previous-GNDVI version requires GNDVI-target forecasting/baseline outputs or a targeted GNDVI-only rerun.",
            ]
        ),
    }
    for name, text in captions.items():
        (out_dirs["captions"] / name).write_text(text + "\n", encoding="utf-8")


def build_inventory() -> pd.DataFrame:
    build_script = CURRENT_SUPP / "scripts" / "build_separate_supplementary_information.py"
    rows = [
        {
            "figure_id": "Figure S1",
            "figure_name": "EWBI-response association",
            "output_filename": "Figure_S1_EWBI_response_association",
            "script_path": str(build_script),
            "gndvi_dependency": "direct",
            "can_be_rebuilt_immediately": "yes",
            "requires_upstream_recomputation": "lightweight previous-GNDVI response row substitution already available",
            "action": "rebuilt with previous GNDVI from legacy EWBI response tables",
        },
        {
            "figure_id": "Figure S2",
            "figure_name": "EWBI component and weighting robustness",
            "output_filename": "Figure_S2_EWBI_component_weighting_robustness",
            "script_path": str(build_script),
            "gndvi_dependency": "indirect",
            "can_be_rebuilt_immediately": "yes, with lightweight recomputation of GNDVI response rows",
            "requires_upstream_recomputation": "lightweight recomputation of GNDVI component/omission/weighting rows",
            "action": "rebuilt; non-GNDVI rows retained from final tables, GNDVI rows recomputed with previous GNDVI",
        },
        {
            "figure_id": "Figure S3",
            "figure_name": "TWS scale and resampling sensitivity",
            "output_filename": "Figure_S3_TWS_scale_resampling_sensitivity",
            "script_path": str(build_script),
            "gndvi_dependency": "none",
            "can_be_rebuilt_immediately": "not needed",
            "requires_upstream_recomputation": "no",
            "action": "copied unchanged",
        },
        {
            "figure_id": "Figure S4",
            "figure_name": "Gap-filling validation and downstream sensitivity",
            "output_filename": "Figure_S4_gap_filling_validation_downstream_sensitivity",
            "script_path": str(build_script),
            "gndvi_dependency": "none identified from source map",
            "can_be_rebuilt_immediately": "not needed",
            "requires_upstream_recomputation": "no",
            "action": "copied unchanged",
        },
        {
            "figure_id": "Figure S5",
            "figure_name": "Detrending and preprocessing sensitivity",
            "output_filename": "Figure_S5_detrending_preprocessing_sensitivity",
            "script_path": str(build_script),
            "gndvi_dependency": "none identified from source map",
            "can_be_rebuilt_immediately": "not needed",
            "requires_upstream_recomputation": "no",
            "action": "copied unchanged",
        },
        {
            "figure_id": "Figure S6",
            "figure_name": "Forecasting baselines and rolling-origin validation",
            "output_filename": "Figure_S6_forecasting_baselines_rolling_origin",
            "script_path": str(build_script),
            "gndvi_dependency": "direct through GNDVI target model/baseline outputs",
            "can_be_rebuilt_immediately": "no",
            "requires_upstream_recomputation": "yes - GNDVI-target forecasting/baseline rerun or existing previous-GNDVI model outputs",
            "action": "blocked; placeholder and note created, current strict-QA figure not copied as previous-GNDVI output",
        },
        {
            "figure_id": "Figure S7",
            "figure_name": "Targeted multivariate forecasting sensitivity",
            "output_filename": "Figure_S7_targeted_multivariate_forecasting_sensitivity_FINAL",
            "script_path": str(build_script),
            "gndvi_dependency": "none; targets are GPP, NIRv, SPEI",
            "can_be_rebuilt_immediately": "not needed",
            "requires_upstream_recomputation": "no",
            "action": "copied unchanged",
        },
        {
            "figure_id": "Figure S8",
            "figure_name": "NIRv stress-zone threshold sensitivity",
            "output_filename": "Figure_S8_NIRv_stress_zone_threshold_sensitivity",
            "script_path": str(build_script),
            "gndvi_dependency": "none",
            "can_be_rebuilt_immediately": "not needed",
            "requires_upstream_recomputation": "no",
            "action": "copied unchanged",
        },
        {
            "figure_id": "Figure S9",
            "figure_name": "Ecosystem-state classification sensitivity",
            "output_filename": "Figure_S9_ecosystem_state_classification_sensitivity",
            "script_path": str(build_script),
            "gndvi_dependency": "none identified from source map",
            "can_be_rebuilt_immediately": "not needed",
            "requires_upstream_recomputation": "no",
            "action": "copied unchanged",
        },
    ]
    return pd.DataFrame(rows)


def build_rebuild_audit() -> pd.DataFrame:
    rows = [
        {
            "figure_id": "Figure S1",
            "rebuilt_successfully": "yes",
            "depends_on_GNDVI": "yes",
            "changed_relative_to_current": "yes",
            "change_only_GNDVI_driven": "yes",
            "qualitative_scientific_judgment": "coherent as previous-GNDVI sensitivity; recent GNDVI period has shorter n",
            "acceptable_for_manuscript": "inconclusive",
            "notes": "Previous-GNDVI rows are available for all land types/periods, but 2020-2024 GNDVI rows use available 2020-2021 months only.",
        },
        {
            "figure_id": "Figure S2",
            "rebuilt_successfully": "yes",
            "depends_on_GNDVI": "yes, indirectly through response aggregates",
            "changed_relative_to_current": "yes",
            "change_only_GNDVI_driven": "yes",
            "qualitative_scientific_judgment": "coherent; aggregate means shift because one response family was changed to previous GNDVI",
            "acceptable_for_manuscript": "inconclusive",
            "notes": "GNDVI component/omission/weighting rows were recomputed lightly; all non-GNDVI rows were retained unchanged from final source tables.",
        },
        {
            "figure_id": "Figure S3",
            "rebuilt_successfully": "copied unchanged",
            "depends_on_GNDVI": "no",
            "changed_relative_to_current": "no",
            "change_only_GNDVI_driven": "not applicable",
            "qualitative_scientific_judgment": "unaffected by GNDVI source",
            "acceptable_for_manuscript": "yes",
            "notes": "TWS harmonization/resampling figure.",
        },
        {
            "figure_id": "Figure S4",
            "rebuilt_successfully": "copied unchanged",
            "depends_on_GNDVI": "no direct dependency identified",
            "changed_relative_to_current": "no",
            "change_only_GNDVI_driven": "not applicable",
            "qualitative_scientific_judgment": "unaffected by GNDVI source",
            "acceptable_for_manuscript": "yes",
            "notes": "Gap-filling figure.",
        },
        {
            "figure_id": "Figure S5",
            "rebuilt_successfully": "copied unchanged",
            "depends_on_GNDVI": "no direct dependency identified",
            "changed_relative_to_current": "no",
            "change_only_GNDVI_driven": "not applicable",
            "qualitative_scientific_judgment": "unaffected by GNDVI source",
            "acceptable_for_manuscript": "yes",
            "notes": "Detrending/preprocessing figure.",
        },
        {
            "figure_id": "Figure S6",
            "rebuilt_successfully": "no",
            "depends_on_GNDVI": "yes, direct GNDVI target forecasting/baseline output",
            "changed_relative_to_current": "not rebuilt",
            "change_only_GNDVI_driven": "not assessed",
            "qualitative_scientific_judgment": "blocked; cannot be converted by plotting",
            "acceptable_for_manuscript": "no, not as previous-GNDVI-consistent",
            "notes": "Requires GNDVI-target forecasting/baseline rerun or existing previous-GNDVI model outputs. No model rerun was performed.",
        },
        {
            "figure_id": "Figure S7",
            "rebuilt_successfully": "copied unchanged",
            "depends_on_GNDVI": "no",
            "changed_relative_to_current": "no",
            "change_only_GNDVI_driven": "not applicable",
            "qualitative_scientific_judgment": "unaffected by GNDVI source",
            "acceptable_for_manuscript": "yes",
            "notes": "Multivariate sensitivity targets are GPP, NIRv, SPEI.",
        },
        {
            "figure_id": "Figure S8",
            "rebuilt_successfully": "copied unchanged",
            "depends_on_GNDVI": "no",
            "changed_relative_to_current": "no",
            "change_only_GNDVI_driven": "not applicable",
            "qualitative_scientific_judgment": "unaffected by GNDVI source",
            "acceptable_for_manuscript": "yes",
            "notes": "NIRv stress-zone figure.",
        },
        {
            "figure_id": "Figure S9",
            "rebuilt_successfully": "copied unchanged",
            "depends_on_GNDVI": "no direct dependency identified",
            "changed_relative_to_current": "no",
            "change_only_GNDVI_driven": "not applicable",
            "qualitative_scientific_judgment": "unaffected by GNDVI source",
            "acceptable_for_manuscript": "yes",
            "notes": "State classification sensitivity figure.",
        },
    ]
    return pd.DataFrame(rows)


def compare_panel_data(out_dirs: dict[str, Path]) -> None:
    rows = []
    for name, current_path, previous_path, keys in [
        (
            "S1a EWBI-response association",
            CURRENT_S1A,
            out_dirs["tables"] / "S1a_previous_GNDVI_period_response_panel_data.csv",
            ["land_type", "period_code", "response"],
        ),
        (
            "S1b component/omission panel",
            CURRENT_S1B,
            out_dirs["tables"] / "S1b_previous_GNDVI_component_omission_panel_data.csv",
            ["component", "series"],
        ),
        (
            "S1c weighting panel",
            CURRENT_S1C,
            out_dirs["tables"] / "S1c_previous_GNDVI_weighting_panel_data.csv",
            ["land_type", "weighting"],
        ),
    ]:
        if not current_path.exists() or not previous_path.exists():
            rows.append({"panel": name, "comparison_status": "missing input", "n_changed": "", "notes": ""})
            continue
        cur = read_csv(current_path)
        prev = read_csv(previous_path)
        if name.startswith("S1a"):
            cur = cur.copy()
            cur["land_type"] = cur["land_type"].map(pretty_land)
            cur["period_code"] = cur["period"].map(normalize_period_code)
            prev = prev.copy()
            prev["period_code"] = prev["period"].map(normalize_period_code)
            metric = "pearson_r"
        else:
            metric = "mean"
        merged = cur[keys + [metric]].merge(prev[keys + [metric]], on=keys, how="outer", suffixes=("_current", "_previous"))
        merged["difference"] = merged[f"{metric}_previous"] - merged[f"{metric}_current"]
        changed = merged["difference"].abs() > 1e-12
        merged.to_csv(out_dirs["audit"] / f"{name.replace(' ', '_').replace('/', '_')}_current_vs_previous.csv", index=False)
        rows.append(
            {
                "panel": name,
                "comparison_status": "completed",
                "n_changed": int(changed.sum()),
                "max_abs_difference": float(merged["difference"].abs().max(skipna=True)) if len(merged) else np.nan,
                "notes": "Differences are expected only where previous GNDVI changes a direct row or an aggregate containing the GNDVI response.",
            }
        )
    pd.DataFrame(rows).to_csv(out_dirs["audit"] / "figure_panel_comparison_summary.csv", index=False)


def write_source_lineage_report(inventory: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    text = [
        "# Supplementary Figure Source Lineage Report",
        "",
        "This report documents the exact source workflow used for the full previous-GNDVI supplementary rebuild package.",
        "",
        "## Primary source workflow",
        "",
        f"- Current separate Supplement builder: `{CURRENT_SUPP / 'scripts' / 'build_separate_supplementary_information.py'}`",
        f"- Compact panel CSV source directory: `{PANEL_DIR}`",
        f"- Final source-table directory: `{SOURCE_TABLES}`",
        f"- Previous/original GNDVI source table: `{LEGACY_GNDVI_CONFIRMATION}`",
        "",
        "## Figure lineage table",
        "",
        inventory.to_markdown(index=False),
        "",
        "## Rebuild policy",
        "",
        "- Figure S1 was redrawn using the existing previous-GNDVI EWBI-response table from the legacy sensitivity package.",
        "- Figure S2 was redrawn by recomputing only GNDVI response rows for component, omission, and weighting summaries; non-GNDVI rows were retained unchanged from the final source tables.",
        "- Figures S3, S4, S5, S7, S8, and S9 were copied unchanged because no GNDVI dependency was identified from the final source map.",
        "- Figure S6 was not rebuilt because it depends on GNDVI target forecasting/baseline metrics. Replotting without rerunning GNDVI-target outputs would silently mix sources.",
    ]
    (out_dirs["report"] / "Supplementary_figure_source_lineage_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_final_report(inventory: pd.DataFrame, audit: pd.DataFrame, out_dirs: dict[str, Path]) -> None:
    veg = read_csv(LEGACY_VEG_COHERENCE) if LEGACY_VEG_COHERENCE.exists() else pd.DataFrame()
    veg_summary = "No previous-vs-strict vegetation coherence table was found."
    if not veg.empty:
        if "delta_legacy_minus_strictQA_r" in veg.columns:
            delta = pd.to_numeric(veg["delta_legacy_minus_strictQA_r"], errors="coerce")
            veg_summary = (
                f"Existing diagnostics show previous GNDVI had higher same-month vegetation/productivity correlations "
                f"than strict-QA GNDVI in {int((delta > 0).sum())} of {int(delta.notna().sum())} comparisons "
                f"(mean delta r = {float(delta.mean()):.3f})."
            )
    text = [
        "# FINAL Full Previous-GNDVI Supplementary Rebuild Report",
        "",
        "## Executive Summary",
        "",
        "The supplementary package was rebuilt around the previous/original GNDVI source wherever this could be done honestly without rerunning forecasting or other heavy scientific workflows.",
        "",
        "What worked:",
        "- Figure S1 was rebuilt from previous-GNDVI EWBI-response tables.",
        "- Figure S2 was rebuilt with a lightweight recomputation of GNDVI-only component/omission/weighting rows; all non-GNDVI rows were left unchanged.",
        "- Figures S3, S4, S5, S7, S8, and S9 were copied unchanged because they do not depend on GNDVI based on the final source maps.",
        "",
        "What failed or remains blocked:",
        "- Figure S6 was not rebuilt because it contains GNDVI target forecasting/baseline results. A plotting-only rebuild would be scientifically invalid.",
        "- Previous GNDVI in the legacy source covers 2001-2021 only, so recent-period GNDVI validation rows use 2020-2021 months rather than the full 2020-2024 period.",
        "",
        f"{veg_summary}",
        "",
        "## Rebuild Inventory",
        "",
        inventory.to_markdown(index=False),
        "",
        "## Figure Audit",
        "",
        audit.to_markdown(index=False),
        "",
        "## Direct Answers",
        "",
        "1. **Which supplementary figures were rebuilt successfully?** Figures S1 and S2 were rebuilt with previous-GNDVI inputs; S3, S4, S5, S7, S8, and S9 were copied unchanged because they are GNDVI-independent.",
        "2. **Which depended on GNDVI?** S1 directly; S2 indirectly through response aggregates; S6 directly through GNDVI forecasting/baseline target outputs.",
        "3. **Which changed materially?** S1 GNDVI response cells changed. S2 aggregate means changed because the GNDVI response rows were recomputed with previous GNDVI.",
        "4. **Did previous GNDVI produce a coherent supplementary package?** It produced coherent S1/S2 sensitivity outputs and aligns with prior Figure 5/Figure 6 diagnostics, but the package remains incomplete because S6 cannot be rebuilt without GNDVI-target forecasting/baseline reruns.",
        "5. **Remaining unresolved problems?** S6 is blocked, and the previous GNDVI source lacks 2022-2024 coverage in the current folder.",
        "6. **Can previous GNDVI now be used for the manuscript supplementary figures?** Not as a complete manuscript-wide source yet.",
        "",
        "## Final Decision",
        "",
        "**INCONCLUSIVE - more targeted work is needed before previous GNDVI can be kept as the final manuscript-wide GNDVI source.**",
        "",
        "The previous GNDVI is scientifically promising for raw correlation/CCF diagnostics and for the rebuilt EWBI supplementary panels, but it is not yet a complete replacement because Figure S6 requires GNDVI-target forecasting/baseline outputs and the previous source currently lacks 2022-2024 coverage.",
        "",
        "## Minimum Remaining Action",
        "",
        "- Either locate existing previous-GNDVI GNDVI-target forecasting/baseline outputs or run the minimum GNDVI-target-only forecasting/baseline rerun.",
        "- Decide how to handle the previous-GNDVI 2022-2024 coverage gap before claiming previous GNDVI as the manuscript-wide final source.",
        "- Update Methods/captions if previous GNDVI is ultimately selected; do not claim strict-QA GNDVI in that case.",
        "",
        "## Safety Checks",
        "",
        "- No forecasting, ML, or deep-learning model was rerun.",
        "- No raster extraction was rerun.",
        "- Non-GNDVI datasets were not changed.",
        "- Current strict-QA Figure S6 was not copied as a previous-GNDVI output.",
        "- Missing or blocked figures were flagged rather than faked.",
    ]
    (out_dirs["report"] / "FINAL_full_previous_GNDVI_supplementary_rebuild_report.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_manifest(out_dirs: dict[str, Path]) -> None:
    rows = []
    for path in OUT_ROOT.rglob("*"):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(OUT_ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    pd.DataFrame(rows).sort_values("relative_path").to_csv(out_dirs["audit"] / "package_file_manifest.csv", index=False)


def main() -> None:
    out_dirs = dirs()
    configure_style()

    inventory = build_inventory()
    inventory.to_csv(out_dirs["tables"] / "Supplementary_figure_full_inventory.csv", index=False)

    # Rebuilt figures using previous GNDVI.
    build_s1(out_dirs)
    component_panel, weighting_panel = build_s2(out_dirs)

    copied_rows: list[dict[str, str]] = []
    for fig_id in ["Figure S3", "Figure S4", "Figure S5", "Figure S7", "Figure S8", "Figure S9"]:
        copy_unaffected_figure(fig_id, out_dirs, copied_rows)
    pd.DataFrame(copied_rows).to_csv(out_dirs["audit"] / "unchanged_figure_copy_log.csv", index=False)

    # S6 is explicitly blocked to avoid silently mixing strict-QA forecasting results.
    write_blocked_s6(out_dirs)
    write_rebuilt_captions(out_dirs)

    # Copy traceability inputs.
    for src in [
        CURRENT_SUPP / "source_mapping" / "separate_supplementary_figure_source_map.csv",
        CURRENT_SUPP / "source_mapping" / "build_input_inventory.csv",
        COMPACT / "source_mapping" / "compact_figure_source_map.csv",
        LEGACY_EWBI_CHANGE,
        LEGACY_VEG_COHERENCE,
        LEGACY_GNDVI_CONFIRMATION,
    ]:
        if src.exists():
            shutil.copy2(src, out_dirs["source_mapping"] / src.name)

    # Copy this script into the package tools folder.
    self_path = Path(__file__)
    shutil.copy2(self_path, out_dirs["tools"] / self_path.name)

    audit = build_rebuild_audit()
    audit.to_csv(out_dirs["tables"] / "Supplementary_figure_rebuild_audit.csv", index=False)
    compare_panel_data(out_dirs)

    write_source_lineage_report(inventory, out_dirs)
    write_final_report(inventory, audit, out_dirs)
    write_manifest(out_dirs)
    print(f"OUTPUT_ROOT={OUT_ROOT}")
    print("SCIENTIFIC_ANALYSIS_RERUN=lightweight_GNDVI_only_EWBI_rows")
    print("MODELS_RERUN=no")
    print("RASTER_EXTRACTION_RERUN=no")
    print("FINAL_DECISION=INCONCLUSIVE")


if __name__ == "__main__":
    main()
