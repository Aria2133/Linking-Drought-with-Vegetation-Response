from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr


HERE = Path(__file__).resolve().parent
DP = HERE.parents[2]
RAW_TABLE = (
    DP
    / "Docs"
    / "major_revision_analyses"
    / "04_detrending_sensitivity_fixed"
    / "tables"
    / "monthly_landtype_series_raw.csv"
)
PREVIOUS_GNDVI_TABLE = (
    DP
    / "Docs"
    / "major_revision_analyses"
    / "final_previous_GNDVI_with_test_rebuild"
    / "tables"
    / "Figure_2_previous_GNDVI_with_test_monthly_series.csv"
)

VARIABLES = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
LAND_TYPES = ["Grasslands", "Natural Forests", "Savannas", "Shrublands"]


def load_source_series() -> pd.DataFrame:
    """Load compact raw monthly values and replace GNDVI with the no-QA branch."""
    raw = pd.read_csv(RAW_TABLE, parse_dates=["date"])
    raw = raw.loc[
        raw["variable"].ne("GNDVI"),
        ["date", "land_type", "variable", "raw_value", "source_period", "source_path", "unit_action"],
    ].rename(columns={"raw_value": "value"})

    previous = pd.read_csv(PREVIOUS_GNDVI_TABLE, parse_dates=["date"])
    previous = previous.loc[
        previous["variable"].eq("GNDVI"),
        ["date", "land_type", "variable", "original_value"],
    ].rename(columns={"original_value": "value"})
    previous["source_period"] = "previous/no-QA GNDVI branch"
    previous["source_path"] = str(PREVIOUS_GNDVI_TABLE)
    previous["unit_action"] = "none"

    data = pd.concat([raw, previous], ignore_index=True)
    data["land_type"] = data["land_type"].replace({"NaturalForests": "Natural Forests"})
    data = data[data["variable"].isin(VARIABLES)].copy()

    # Preserve the submitted Figure-5 scientific scope: the old figure explicitly
    # labelled GPP as 2001-2021 only. The stitched GPP sensitivity is Task 4.
    data = data[(data["variable"].ne("GPP")) | (data["date"].dt.year <= 2021)]
    data = data.dropna(subset=["value"])
    data["date"] = data["date"].dt.to_period("M").dt.to_timestamp()
    data = data.drop_duplicates(["land_type", "variable", "date"], keep="last")
    return data.sort_values(["land_type", "variable", "date"]).reset_index(drop=True)


def deseasonalize(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    out["calendar_month"] = out["date"].dt.month
    climatology = out.groupby(["land_type", "variable", "calendar_month"])["value"].transform("mean")
    out["value"] = out["value"] - climatology
    return out


def pairwise_correlations(data: pd.DataFrame, analysis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    corr_rows: list[dict] = []
    coverage_rows: list[dict] = []
    for land in LAND_TYPES:
        sub = data[data["land_type"].eq(land)]
        series = {
            variable: (
                sub[sub["variable"].eq(variable)][["date", "value"]]
                .dropna()
                .drop_duplicates("date", keep="last")
                .set_index("date")["value"]
                .sort_index()
            )
            for variable in VARIABLES
        }
        for i, variable_1 in enumerate(VARIABLES):
            for variable_2 in VARIABLES[: i + 1]:
                paired = pd.concat(
                    [series[variable_1].rename("x"), series[variable_2].rename("y")],
                    axis=1,
                    join="inner",
                ).dropna()
                if len(paired) < 3:
                    r = p = np.nan
                elif variable_1 == variable_2:
                    r, p = 1.0, 0.0
                else:
                    r, p = pearsonr(paired["x"].to_numpy(), paired["y"].to_numpy())
                common = {
                    "analysis": analysis,
                    "land_type": land,
                    "variable_1": variable_1,
                    "variable_2": variable_2,
                    "pairwise_start_date": paired.index.min().date().isoformat() if len(paired) else "",
                    "pairwise_end_date": paired.index.max().date().isoformat() if len(paired) else "",
                    "n_shared_months": int(len(paired)),
                }
                corr_rows.append({**common, "pearson_r": float(r), "pearson_p": float(p)})
                coverage_rows.append(common)
    return pd.DataFrame(corr_rows), pd.DataFrame(coverage_rows)


def render(correlations: pd.DataFrame, output: Path, title_suffix: str) -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({"font.family": "Arial", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 2, figsize=(24, 22))
    for ax, land in zip(axes.ravel(), LAND_TYPES):
        sub = correlations[correlations["land_type"].eq(land)]
        matrix = pd.DataFrame(np.nan, index=VARIABLES, columns=VARIABLES)
        for row in sub.itertuples(index=False):
            matrix.loc[row.variable_1, row.variable_2] = row.pearson_r
            matrix.loc[row.variable_2, row.variable_1] = row.pearson_r
        mask = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
        sns.heatmap(
            matrix,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="magma_r",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=1.0,
            cbar_kws={"shrink": 0.8, "pad": 0.02},
            annot_kws={"size": 16, "weight": "bold", "color": "white"},
            ax=ax,
        )
        ax.set_title(land.upper(), fontsize=18, pad=15)
        ax.set_xlabel("Variables", fontsize=14, fontweight="bold")
        ax.set_ylabel("Variables", fontsize=14, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=13)
        ax.tick_params(axis="y", rotation=0, labelsize=13)
        ax.collections[0].colorbar.ax.tick_params(labelsize=12)
    fig.suptitle(title_suffix, fontsize=20, fontweight="bold", y=0.995)
    fig.subplots_adjust(hspace=0.35, wspace=0.25)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    data = load_source_series()
    data.to_csv(HERE / "figure5_exact_input_monthly_series.csv", index=False, date_format="%Y-%m-%d")

    raw_corr, coverage = pairwise_correlations(data, "raw_monthly")
    raw_corr.to_csv(HERE / "figure5_raw_correlations.csv", index=False)
    coverage.to_csv(HERE / "figure5_pairwise_date_coverage.csv", index=False)
    render(
        raw_corr,
        HERE / "figure5_corrected_raw.png",
        "Inter-Index Correlation Matrix (date-aligned raw monthly values)\nGPP: 2001-2021 only",
    )

    sensitivity = deseasonalize(data)
    sensitivity.to_csv(
        HERE / "figure5_deseasonalized_exact_input_monthly_series.csv",
        index=False,
        date_format="%Y-%m-%d",
    )
    deseason_corr, _ = pairwise_correlations(sensitivity, "calendar_month_climatology_removed")
    deseason_corr.to_csv(HERE / "figure5_deseasonalized_correlations.csv", index=False)
    render(
        deseason_corr,
        HERE / "figure5_deseasonalized_sensitivity.png",
        "Inter-Index Correlation Sensitivity\nCalendar-month climatology removed; pairwise date alignment",
    )

    print(f"source rows: {len(data)}")
    print(f"raw correlation rows: {len(raw_corr)}")
    print(f"date coverage: {coverage.n_shared_months.min()}-{coverage.n_shared_months.max()} shared months")
    print(f"outputs: {HERE}")


if __name__ == "__main__":
    main()
