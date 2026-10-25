"""Write corrected copies of the land-type monthly series with valid MOD17 GPP for 2022 and one MOD17 unit rule.

For every land type, GPP months 2022-01..2022-12 are replaced by the unweighted mean of valid pixels of the corrected
land-type rasters (apply_rebuilt_mod17_2022.py) converted from kg C m-2 month-1 to g C m-2 d-1 (x 1000 / days in
month) - the procedure that produced the 2023-2024 values (checked: the published 2023/2024 means are reproduced).
The same conversion is applied to every MOD17 month: Natural Forests 2023-05 and 2023-06, which the former
"convert if median < 0.2" rule left in kg C m-2 month-1, are converted from their unchanged rasters.

Derived columns are recomputed with the unchanged references: processed = value - 2001-2011 calendar-month mean
(recovered from the file as original - processed), standardized = (processed - baseline mean) / baseline SD.
All other rows are copied unchanged. Supported layouts: standardized series (original_value / processed_value /
standardized_anomaly) and the raw series (raw_value / analysis_value).

Usage: python apply_rebuilt_mod17_2022_series.py --rasters <Masked_LandTypes root with corrected 2022>
           --original-rasters <Test Data/Masked_LandTypes> --out <dir> <series.csv> [<series.csv> ...]
"""
import argparse
import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

FOLDER = {"Grasslands": "Grasslands", "Natural Forests": "NaturalForests", "Savannas": "Savannas",
          "Shrublands": "Shrublands"}


def month_mean_gc_day(path: Path, year: int, month: int) -> float:
    with rasterio.open(path) as src:
        a = src.read(1).astype("float64")
    return float(np.nanmean(a)) * 1000.0 / calendar.monthrange(year, month)[1]


def corrected_values(rasters: Path, original_rasters: Path) -> dict:
    """{(land_type, 'YYYY-MM-01'): g C m-2 d-1} for every month that changes."""
    out = {}
    for land, folder in FOLDER.items():
        for m in range(1, 13):
            p = rasters / folder / "GPP" / f"GPP_MOD17A2H_CA_5km_2022_{m:02d}_{folder}.tif"
            out[land, f"2022-{m:02d}-01"] = month_mean_gc_day(p, 2022, m)
    for m in (5, 6):  # left unconverted by the former median rule
        p = original_rasters / "NaturalForests" / "GPP" / f"GPP_MOD17A2H_CA_5km_2023_{m:02d}_NaturalForests.tif"
        out["Natural Forests", f"2023-{m:02d}-01"] = month_mean_gc_day(p, 2023, m)
    return out


def correct(frame: pd.DataFrame, new: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    land = frame["land_type"].replace({"NaturalForests": "Natural Forests"})
    key = list(zip(land, frame["date"].dt.strftime("%Y-%m-%d")))
    hit = (frame["variable"] == "GPP").to_numpy() & np.array([k in new for k in key])
    log = []
    if "original_value" in frame:
        gpp = frame["variable"] == "GPP"
        seasonal = (frame["original_value"] - frame["processed_value"])[gpp]
        seasonal_ref = seasonal.groupby([land[gpp], frame.loc[gpp, "date"].dt.month]).agg(["mean", "std"])
        assert (seasonal_ref["std"].fillna(0) < 1e-9).all(), "seasonal reference is not constant per calendar month"
        for i in np.where(hit)[0]:
            idx = frame.index[i]
            l, d = key[i]
            value = new[l, d]
            proc = value - seasonal_ref.loc[(l, frame.at[idx, "date"].month), "mean"]
            std = (proc - frame.at[idx, "baseline_anomaly_mean"]) / frame.at[idx, "baseline_anomaly_sd"]
            log.append({"land_type": l, "date": d, "old_value": frame.at[idx, "original_value"], "new_value": value,
                        "old_standardized": frame.at[idx, "standardized_anomaly"], "new_standardized": std})
            frame.at[idx, "original_value"], frame.at[idx, "processed_value"] = value, proc
            frame.at[idx, "standardized_anomaly"] = std
    elif "raw_value" in frame:
        for i in np.where(hit)[0]:
            idx = frame.index[i]
            l, d = key[i]
            log.append({"land_type": l, "date": d, "old_value": frame.at[idx, "raw_value"], "new_value": new[l, d]})
            frame.at[idx, "raw_value"] = frame.at[idx, "analysis_value"] = new[l, d]
            if "unit_action" in frame:
                frame.at[idx, "unit_action"] = "MOD17 kgC m-2 month-1 to gC m-2 day-1 (all MOD17 months)"
    else:
        raise ValueError("unsupported series layout")
    assert len(log) == 50, f"expected 4 x 12 + 2 corrected GPP rows, found {len(log)}"
    return frame, pd.DataFrame(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rasters", required=True, type=Path)
    ap.add_argument("--original-rasters", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("series", nargs="+", type=Path)
    args = ap.parse_args()
    new = corrected_values(args.rasters, args.original_rasters)
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.series:
        frame = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")  # exact copy of unchanged rows
        fixed, log = correct(frame, new)
        fixed["date"] = fixed["date"].dt.strftime("%Y-%m-%d")
        fixed.to_csv(args.out / path.name, index=False)
        log.to_csv(args.out / f"{path.stem}__changed_rows.csv", index=False)
        print(f"{path.name}: {len(log)} GPP rows corrected")


if __name__ == "__main__":
    main()
