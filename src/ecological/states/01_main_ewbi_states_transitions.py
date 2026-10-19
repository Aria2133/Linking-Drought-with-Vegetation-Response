"""Build corrected main ecological outputs from deseasonalized-only anomalies."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fixed_utils import (
    LAND_TYPES,
    PERIODS,
    ROOT,
    STATES,
    classify_states,
    configure_logging,
    ensure_directories,
    period_bounds,
    save_figure,
    transition_rows,
)


STATE_COLORS = {
    "Wet / Healthy": "#2E8B57",
    "Wet / Suppressed": "#5DADE2",
    "Dry / Resilient": "#F5B041",
    "Dry / Stressed": "#C0392B",
}


def make_ewbi_timeseries(series: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey=True)
    for ax, land_type in zip(axes.flat, LAND_TYPES):
        land = series[series["land_type"] == land_type].set_index("date")
        ax.plot(land.index, land["ewbi"], color="#6BAED6", linewidth=0.7, alpha=0.45)
        ax.plot(
            land.index,
            land["ewbi"].rolling(3, center=True, min_periods=1).mean(),
            color="#08519C",
            linewidth=1.5,
            label="3-month mean",
        )
        ax.axvspan("2012-01-01", "2016-12-31", color="#F4A261", alpha=0.13)
        ax.axvspan("2020-01-01", "2024-12-31", color="#E76F51", alpha=0.09)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title(land_type)
        ax.grid(alpha=0.2)
    figure.suptitle(
        "Corrected main EWBI: baseline-referenced deseasonalized-only anomalies"
    )
    figure.supxlabel("Year", y=0.045)
    figure.supylabel("Equal-weight EWBI")
    figure.text(
        0.5,
        0.012,
        "Shading: 2012-2016 historic drought and 2020-2024 recent period",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.96))
    save_figure(figure, "main_ewbi_timeseries_deseasonalized")
    plt.close(figure)


def make_period_figure(period_means: pd.DataFrame) -> None:
    periods = list(PERIODS)
    x = np.arange(len(LAND_TYPES))
    width = 0.24
    colors = ["#9ECAE1", "#F4A261", "#E76F51"]
    figure, ax = plt.subplots(figsize=(12, 6.5))
    for offset, (period, color) in enumerate(zip(periods, colors)):
        frame = period_means[period_means["period"] == period].set_index("land_type")
        values = [frame.loc[land, "ewbi_mean"] for land in LAND_TYPES]
        errors = [frame.loc[land, "ewbi_standard_error"] for land in LAND_TYPES]
        ax.bar(
            x + (offset - 1) * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=period,
            color=color,
        )
    ax.set_xticks(x, LAND_TYPES)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_ylabel("Period mean EWBI (mean +/- SE)")
    ax.set_title("Corrected EWBI period separation: deseasonalized-only reference")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    figure.tight_layout()
    save_figure(figure, "main_ewbi_period_separation_deseasonalized")
    plt.close(figure)


def make_transition_figure(transitions: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 11.5))
    for ax, land_type in zip(axes.flat, LAND_TYPES):
        frame = transitions[
            (transitions["land_type"] == land_type) & (transitions["period"] == "all")
        ]
        matrix = (
            frame.pivot(
                index="from_state",
                columns="to_state",
                values="transition_probability",
            )
            .reindex(index=STATES, columns=STATES)
            .to_numpy(dtype=float)
        )
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(4), [value.replace(" / ", "/\n") for value in STATES], fontsize=8)
        ax.set_yticks(range(4), STATES, fontsize=8)
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.set_title(land_type)
        for row in range(4):
            for col in range(4):
                value = matrix[row, col]
                if np.isfinite(value):
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color="white" if value > 0.55 else "black",
                    )
    figure.subplots_adjust(
        left=0.10,
        right=0.86,
        bottom=0.08,
        top=0.86,
        wspace=0.38,
        hspace=0.46,
    )
    colorbar_axis = figure.add_axes([0.90, 0.20, 0.022, 0.58])
    figure.colorbar(image, cax=colorbar_axis, label="Transition probability")
    figure.suptitle(
        "Corrected ecosystem-state transition probabilities\n"
        "Deseasonalized-only EWBI and NIRv anomalies",
        y=0.96,
    )
    save_figure(figure, "main_state_transition_probabilities_deseasonalized")
    plt.close(figure)


def make_occupancy_figure(occupancy: pd.DataFrame) -> None:
    periods = ["baseline", "historic drought", "recent"]
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for ax, land_type in zip(axes.flat, LAND_TYPES):
        frame = occupancy[
            (occupancy["land_type"] == land_type)
            & (occupancy["period"].isin(periods))
        ]
        bottom = np.zeros(len(periods))
        for state in STATES:
            state_frame = frame[frame["state"] == state].set_index("period")
            values = np.array(
                [state_frame.loc[period, "occupancy_percent"] for period in periods]
            )
            ax.bar(
                periods,
                values,
                bottom=bottom,
                label=state,
                color=STATE_COLORS[state],
            )
            bottom += values
        ax.set_ylim(0, 100)
        ax.set_title(land_type)
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4)
    figure.suptitle(
        "Corrected ecosystem-state occupancy by period\n"
        "Deseasonalized-only EWBI and NIRv anomalies"
    )
    figure.supylabel("Monthly occupancy (%)")
    figure.tight_layout(rect=(0.02, 0.08, 1, 0.95))
    save_figure(figure, "main_state_occupancy_deseasonalized")
    plt.close(figure)


def main() -> None:
    ensure_directories()
    logger = configure_logging(
        ROOT / "logs" / "main_ecological_outputs_log.txt", "main_ecological_outputs"
    )
    ewbi = pd.read_csv(
        ROOT / "tables" / "ewbi_detrending_monthly_series.csv",
        parse_dates=["date"],
    )
    processed = pd.read_csv(
        ROOT / "tables" / "preprocessed_monthly_series.csv",
        parse_dates=["date"],
    )
    main_ewbi = ewbi[ewbi["preprocessing"] == "deseasonalized_only"].copy()
    main_ewbi["ecological_reference"] = "deseasonalized_only"
    main_ewbi["seasonal_reference"] = "2001-2011 calendar-month mean"
    main_ewbi.to_csv(
        ROOT / "tables" / "main_ewbi_deseasonalized_series.csv", index=False
    )

    period_rows = []
    for land_type in LAND_TYPES:
        series = main_ewbi[main_ewbi["land_type"] == land_type].set_index("date")
        for period, (start, end) in PERIODS.items():
            subset = series.loc[start:end]
            period_rows.append(
                {
                    "land_type": land_type,
                    "period": period,
                    "period_start": start,
                    "period_end": end,
                    "valid_months": int(subset["ewbi"].notna().sum()),
                    "ewbi_mean": float(subset["ewbi"].mean()),
                    "ewbi_standard_deviation": float(subset["ewbi"].std(ddof=1)),
                    "ewbi_standard_error": float(
                        subset["ewbi"].std(ddof=1)
                        / np.sqrt(subset["ewbi"].notna().sum())
                    ),
                    "spei_signed_z_mean": float(subset["spei_signed_z"].mean()),
                    "tws_signed_z_mean": float(subset["tws_signed_z"].mean()),
                    "vpd_signed_z_mean": float(subset["vpd_signed_z"].mean()),
                    "lst_signed_z_mean": float(subset["lst_signed_z"].mean()),
                    "preprocessing": "deseasonalized_only",
                }
            )
    period_means = pd.DataFrame(period_rows)
    period_means.to_csv(
        ROOT / "tables" / "main_ewbi_period_means_deseasonalized.csv", index=False
    )

    classification_rows = []
    transition_output = []
    occupancy_rows = []
    for land_type in LAND_TYPES:
        ewbi_series = (
            main_ewbi[main_ewbi["land_type"] == land_type]
            .set_index("date")["ewbi"]
            .sort_index()
        )
        nirv = (
            processed[
                (processed["land_type"] == land_type)
                & (processed["variable"] == "NIRv")
                & (processed["preprocessing"] == "deseasonalized_only")
            ]
            .set_index("date")["processed_value"]
            .sort_index()
        )
        labels, ewbi_threshold, nirv_threshold = classify_states(ewbi_series, nirv)
        for date in labels.index:
            classification_rows.append(
                {
                    "date": date,
                    "year": date.year,
                    "month": date.month,
                    "land_type": land_type,
                    "ewbi": ewbi_series.loc[date],
                    "nirv_deseasonalized_anomaly": nirv.loc[date],
                    "ewbi_median_threshold": ewbi_threshold,
                    "nirv_median_threshold": nirv_threshold,
                    "ecosystem_state": labels.loc[date],
                    "preprocessing": "deseasonalized_only",
                    "classification_logic": "median quadrant",
                }
            )
        for period in ["all", *PERIODS.keys()]:
            transition_output.extend(transition_rows(labels, land_type, period))
            start, end = period_bounds(period, labels.index)
            period_labels = labels.loc[start:end]
            valid_count = int(period_labels.notna().sum())
            for state in STATES:
                count = int((period_labels == state).sum())
                occupancy_rows.append(
                    {
                        "land_type": land_type,
                        "period": period,
                        "state": state,
                        "month_count": count,
                        "valid_months": valid_count,
                        "occupancy_percent": 100.0 * count / valid_count,
                        "preprocessing": "deseasonalized_only",
                    }
                )

    classification = pd.DataFrame(classification_rows)
    transitions = pd.DataFrame(transition_output)
    occupancy = pd.DataFrame(occupancy_rows)
    classification.to_csv(
        ROOT / "tables" / "main_state_classification_deseasonalized.csv",
        index=False,
    )
    transitions.to_csv(
        ROOT
        / "tables"
        / "main_state_transition_probabilities_deseasonalized.csv",
        index=False,
    )
    occupancy.to_csv(
        ROOT / "tables" / "main_state_occupancy_deseasonalized.csv",
        index=False,
    )

    make_ewbi_timeseries(main_ewbi)
    make_period_figure(period_means)
    make_transition_figure(transitions)
    make_occupancy_figure(occupancy)
    logger.info(
        "Created corrected main outputs: %d EWBI rows, %d state rows, %d transitions.",
        len(main_ewbi),
        len(classification),
        len(transitions),
    )


if __name__ == "__main__":
    main()
