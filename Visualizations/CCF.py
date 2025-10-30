# -*- coding: utf-8 -*-
"""
Fixed pipeline:
- Streaming pixel-mean (block-wise) to avoid huge full-band reads (faster, low RAM)
- Removed pyGAM to avoid NumPy np.int issue; trend step uses LOWESS, then quadratic fallback
- More robust error handling and logging
- Modified outputs: cleaned Excel per land type, Cross-Correlation Function plots, VAR IRFs

Outputs (in ./_Out):
  - cleaned_timeseries_<Landtype>.xlsx
  - cross_correlation_function_<Target>.png (2 plots with 4 subplots each)
  - var_irf_<Landtype>_<Target>.png
  - logs.txt
"""

import os
import re
import sys
import math
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from statsmodels.tsa.api import VAR
from statsmodels.nonparametric.smoothers_lowess import lowess

import rasterio
from rasterio.windows import Window

# =========================
#  USER CONFIG (edit here)
# =========================
ROOTS: Dict[str, str] = {
    "Grasslands": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS",
    "NaturalForests": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS",
    "Savannas": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS",
    "Shrublands": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS",
}
INDICES: List[str] = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
TARGETS: List[str] = ["GPP", "GOSIF"]  # responses (Y)
PREDICTORS: List[str] = ["GNDVI", "NDVI", "NIRv", "LST", "TWS", "VPD", "SPEI"]  # predictors (X)
IMPUTE_THESE: List[str] = ["SPEI", "LST", "TWS"]  # apply 3-step imputation
DATE_START = "2001-01"
DATE_END = "2021-12"
MAX_LAG = 12  # months for Cross-Correlation Function and VAR order search
OUTDIR = Path("_Out")
OUTDIR.mkdir(exist_ok=True)

# =========================
#  Logging helpers
# =========================
LOGF = OUTDIR / "logs.txt"


def log(msg: str):
    print(msg)
    try:
        with open(LOGF, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")
    except Exception:
        pass


# =========================
#  Filename → date parser
# =========================
DATE_PATS = [
    re.compile(r"(?P<y>20\d{2})[_-](?P<m>0[1-9]|1[0-2])"),  # YYYY_MM
    re.compile(r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])"),  # YYYYMM
]


def parse_period_from_name(name: str) -> Optional[pd.Period]:
    for pat in DATE_PATS:
        m = pat.search(name)
        if m:
            y, mo = int(m.group("y")), int(m.group("m"))
            return pd.Period(f"{y}-{mo:02d}", freq="M")
    return None


# =========================
#  Raster → pixel-mean (streaming)
# =========================
def mean_over_pixels_stream(path: Path) -> Optional[float]:
    """
    Stream over blocks to compute mean of valid pixels in band 1.
    Skips NoData/NaN. Returns None if unreadable or no valid pixels.
    """
    try:
        with rasterio.Env():
            with rasterio.open(path) as ds:
                nodata = ds.nodata
                total = 0.0
                count = 0
                for ji, window in ds.block_windows(1):
                    arr = ds.read(1, window=window, masked=True)
                    # If mask exists, it already masks nodata
                    if hasattr(arr, "mask"):
                        valid = ~arr.mask
                        if np.any(valid):
                            vals = arr[valid].astype("float64")
                            total += float(vals.sum())
                            count += int(vals.size)
                    else:
                        vals = arr.astype("float64")
                        if nodata is not None:
                            vals = vals[vals != nodata]
                        vals = vals[np.isfinite(vals)]
                        if vals.size:
                            total += float(vals.sum())
                            count += int(vals.size)
                if count == 0:
                    return None
                return total / count
    except KeyboardInterrupt:
        raise
    except Exception as e:
        log(f"[ERROR] mean_over_pixels_stream failed for {path}: {e}")
        log(traceback.format_exc())
        return None


# =========================
#  Scan a land type folder
# =========================
def collect_timeseries_for_landtype(landtype: str, root: str) -> pd.DataFrame:
    """
    For a given land type, traverse each index subfolder, compute monthly pixel-mean.
    Returns a DataFrame (index = Period M 2001-01..2021-12, columns = INDICES).
    """
    log(f"\n=== Collecting pixel means for land type: {landtype} ===")
    months = pd.period_range(DATE_START, DATE_END, freq="M")
    df = pd.DataFrame(index=months, columns=INDICES, dtype=float)

    root_path = Path(root)
    if not root_path.exists():
        log(f"[WARN] Root path not found for {landtype}: {root}")
        return df

    for idx in INDICES:
        sub = root_path / idx
        if not sub.exists():
            log(f"[WARN] Index folder missing for {landtype}/{idx}: {sub}")
            continue

        files = sorted([p for p in sub.rglob("*") if p.suffix.lower() in (".tif", ".tiff")])
        if not files:
            log(f"[WARN] No GeoTIFFs in {sub}")
            continue

        tmp = {}
        for fp in files:
            per = parse_period_from_name(fp.name)
            if per is None:
                log(f"[WARN] Could not parse date from filename: {fp.name}")
                continue
            if per < months.min() or per > months.max():
                continue
            val = mean_over_pixels_stream(fp)
            if val is None or (isinstance(val, float) and (math.isnan(val) or not math.isfinite(val))):
                log(f"[INFO] All-masked/invalid mean for {fp} → skipping")
                continue
            tmp.setdefault(per, []).append(val)

        if tmp:
            s = pd.Series({k: np.mean(v) for k, v in tmp.items()}, dtype=float)
            df.loc[s.index, idx] = s.values
        else:
            log(f"[WARN] No valid pixel means for {landtype}/{idx}")

    return df


# =========================
#  Imputation (3-step)
# =========================
def _neighbor_year_mean(month_vals: pd.Series, year: int) -> Optional[float]:
    """Step 1: mean of same-month values at y-1 and y+1 if both exist."""
    v1 = month_vals.get(year - 1, np.nan)
    v2 = month_vals.get(year + 1, np.nan)
    if pd.notna(v1) and pd.notna(v2):
        return float((v1 + v2) / 2.0)
    return None


def _month_trend_predict_lowess(month_vals: pd.Series, year: int) -> Optional[float]:
    """
    Step 2a: LOWESS smooth over years for this calendar month.
    """
    known = month_vals.dropna()
    if len(known) < 3:
        return None
    x = known.index.values.astype(float)
    y = known.values.astype(float)
    try:
        frac = min(0.8, max(0.4, 3.0 / len(known)))  # heuristic
        yhat = lowess(y, x, frac=frac, it=1, return_sorted=False)
        # interpolate/extrapolate to the requested year
        years_sorted = np.sort(x)
        # exact year?
        if year in x:
            return float(yhat[np.where(x == year)[0][0]])
        # extrapolate using the two nearest ends if outside
        if year < years_sorted.min() and len(years_sorted) >= 2:
            x1, x2 = years_sorted[0], years_sorted[1]
            y1 = yhat[np.where(x == x1)[0][0]]
            y2 = yhat[np.where(x == x2)[0][0]]
            if x2 != x1:
                return float(y1 + (y2 - y1) * (year - x1) / (x2 - x1))
        if year > years_sorted.max() and len(years_sorted) >= 2:
            x1, x2 = years_sorted[-2], years_sorted[-1]
            y1 = yhat[np.where(x == x1)[0][0]]
            y2 = yhat[np.where(x == x2)[0][0]]
            if x2 != x1:
                return float(y1 + (y2 - y1) * (year - x1) / (x2 - x1))
        # within-range interpolation
        x_all = np.r_[x, [year]]
        y_all = np.r_[yhat, [np.nan]]
        order = np.argsort(x_all)
        x_sorted, y_sorted = x_all[order], y_all[order]
        i = np.where(x_sorted == year)[0][0]
        if 0 < i < len(x_sorted) - 1:
            x1, x2 = x_sorted[i - 1], x_sorted[i + 1]
            y1, y2 = y_sorted[i - 1], y_sorted[i + 1]
            if np.isfinite(y1) and np.isfinite(y2) and x2 != x1:
                return float(y1 + (y2 - y1) * (year - x1) / (x2 - x1))
        return None
    except Exception as e:
        log(f"[WARN] LOWESS trend predict failed (month={month_vals.name} year={year}): {e}")
        return None


def _month_trend_predict_poly(month_vals: pd.Series, year: int) -> Optional[float]:
    """
    Step 2b: quadratic regression over years (fallback if LOWESS fails).
    """
    known = month_vals.dropna()
    if len(known) < 3:
        return None
    x = known.index.values.astype(float)
    y = known.values.astype(float)
    try:
        # center years to improve conditioning
        x0 = x - x.mean()
        coef = np.polyfit(x0, y, deg=2)
        yhat = np.polyval(coef, year - x.mean())
        return float(yhat)
    except Exception as e:
        log(f"[WARN] Poly trend failed (month={month_vals.name} year={year}): {e}")
        return None


def _month_climatology_mean(month_vals: pd.Series) -> Optional[float]:
    """Step 3: mean across all years for that calendar month."""
    if month_vals.dropna().empty:
        return None
    return float(month_vals.dropna().mean())


def impute_series_3step(s: pd.Series, name: str) -> pd.Series:
    """
    Apply the 3-step month-wise imputation to a monthly PeriodIndex series:
      1) neighbor year mean, 2) month-wise trend (LOWESS → quadratic fallback), 3) month climatology.
    Returns a new Series with NaNs filled where possible.
    """
    if not isinstance(s.index, pd.PeriodIndex):
        raise ValueError("impute_series_3step expects a PeriodIndex (freq='M').")
    out = s.copy()
    years = sorted(set(s.index.year))
    # per calendar month
    for m in range(1, 13):
        mask_m = (s.index.month == m)
        vals = s[mask_m]
        vals_by_year = pd.Series(vals.values, index=vals.index.year, dtype=float)
        vals_by_year.name = m  # label month for logging
        for yr in years:
            if pd.isna(vals_by_year.get(yr, np.nan)):
                # Step 1
                v = _neighbor_year_mean(vals_by_year, yr)
                # Step 2
                if v is None:
                    v = _month_trend_predict_lowess(vals_by_year, yr)
                    if v is None:
                        v = _month_trend_predict_poly(vals_by_year, yr)
                # Step 3
                if v is None:
                    v = _month_climatology_mean(vals_by_year)
                if v is not None and np.isfinite(v):
                    idx = pd.Period(f"{yr}-{m:02d}", freq="M")
                    out.loc[idx] = float(v)
                else:
                    log(f"[INFO] Could not impute {name} at {yr}-{m:02d} (insufficient data)")
    return out


# =========================
#  Preprocessing (z-scores)
# =========================
def monthly_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """Deseasonalize by month: for each column and calendar month, (x - mean_m) / std_m."""
    z = df.copy()
    for col in z.columns:
        for m in range(1, 13):
            mask = (z.index.month == m)
            mu = z.loc[mask, col].mean(skipna=True)
            sd = z.loc[mask, col].std(skipna=True)
            if pd.isna(sd) or sd == 0:
                z.loc[mask, col] = z.loc[mask, col] - mu
            else:
                z.loc[mask, col] = (z.loc[mask, col] - mu) / sd
    return z


# =========================
#  Cross-Correlation Function computation & plot
# =========================
def cross_correlation_function_series(x: pd.Series, y: pd.Series, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cross-correlation function for lags -max_lag..+max_lag.
    Positive lag k means x leads y by k months (x at t-k correlates with y at t).
    """
    df = pd.concat([x, y], axis=1).dropna()
    lags = np.arange(-max_lag, max_lag + 1)
    if df.shape[0] < max_lag + 5:
        return lags, np.full(lags.size, np.nan)
    x0 = df.iloc[:, 0].values
    y0 = df.iloc[:, 1].values
    x0 = x0 - np.nanmean(x0)
    y0 = y0 - np.nanmean(y0)
    denom = np.nanstd(x0) * np.nanstd(y0)
    if denom == 0 or not np.isfinite(denom):
        return lags, np.full(lags.size, np.nan)
    corrs = []
    for k in lags:
        if k < 0:
            a, b = x0[-k:], y0[:len(y0) + k]
        elif k > 0:
            a, b = x0[:len(x0) - k], y0[k:]
        else:
            a, b = x0, y0
        if len(a) > 2 and len(b) > 2:
            c = np.corrcoef(a, b)[0, 1]
        else:
            c = np.nan
        corrs.append(c)
    return lags, np.array(corrs, dtype=float)


def plot_cross_correlation_function_combined(all_landtype_data: Dict[str, pd.DataFrame], predictors: List[str],
                                             target: str, max_lag: int, outdir: Path):
    """
    Create a single plot with 4 subplots (one per land type) for the given target.
    """
    try:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        landtypes = list(all_landtype_data.keys())
        lags = np.arange(-max_lag, max_lag + 1)

        # Store the last imshow object for colorbar
        im = None

        for i, landtype in enumerate(landtypes):
            if i >= 4:  # Safety check
                break

            ax = axes[i]
            zdf = all_landtype_data[landtype]

            # Compute Cross-Correlation Function matrix for this landtype
            H = np.zeros((len(predictors), len(lags))) * np.nan
            for j, pred in enumerate(predictors):
                if pred in zdf.columns and target in zdf.columns:
                    lg, cc = cross_correlation_function_series(zdf[pred], zdf[target], max_lag)
                    H[j, :] = cc

            # Plot heatmap
            im = ax.imshow(H, aspect='auto', origin='lower',
                           extent=[lags.min() - 0.5, lags.max() + 0.5, -0.5, len(predictors) - 0.5])
            ax.set_yticks(range(len(predictors)))
            ax.set_yticklabels(predictors)
            ax.set_xticks(lags[::2])  # Show every other lag to avoid crowding
            ax.set_xlabel("Lag (months) — positive = predictor leads target")
            ax.set_title(f"{landtype}")

        # Adjust subplot spacing to make room for colorbar
        fig.subplots_adjust(right=0.85)

        # Add colorbar positioned to the right of all subplots
        cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Correlation")

        fig.suptitle(f"Cross-Correlation Function: predictors vs {target}", fontsize=16)

        outp = outdir / f"cross_correlation_function_{target}.png"
        fig.savefig(outp, dpi=180, bbox_inches='tight')
        plt.close(fig)
        log(f"[OK] Saved Cross-Correlation Function plot: {outp}")

    except Exception as e:
        log(f"[ERROR] Failed Cross-Correlation Function plot for {target}: {e}")
        log(traceback.format_exc())


# =========================
#  VAR + IRF plotting
# =========================
def fit_var_and_plot_irf(zdf: pd.DataFrame, targets: List[str], predictors: List[str],
                         landtype: str, max_lag: int, outdir: Path):
    cols = targets + predictors
    data = zdf[cols].dropna()
    if data.shape[0] < (max_lag * 3 + 20):
        log(f"[WARN] Not enough data for robust VAR in {landtype} (rows={data.shape[0]}). Skipping VAR.")
        return
    try:
        model = VAR(data)
        sel = model.select_order(maxlags=max_lag)
        # prefer AIC, then BIC/HQIC/FPE
        p = None
        for crit in ("aic", "bic", "hqic", "fpe"):
            v = getattr(sel, crit, None)
            if v is not None and np.isfinite(v):
                p = int(v)
                break
        if p is None or p < 1:
            p = min(2, max_lag)
        res = model.fit(p)
        irf = res.irf(24)  # 24-month horizon

        for tgt in targets:
            fig, ax = plt.subplots(figsize=(12, max(5, 0.8 * len(predictors))))
            t_idx = cols.index(tgt)
            T = np.arange(0, 25)
            for pred in predictors:
                j_idx = cols.index(pred)
                resp = irf.irfs[:, t_idx, j_idx]
                ax.plot(T, resp, label=f"shock: {pred}")
            ax.axhline(0, lw=0.8, ls="--", color="k")
            ax.set_xlabel("Months after shock")
            ax.set_ylabel(f"Response in {tgt} (z-units)")
            ax.set_title(f"VAR IRF: responses of {tgt} to predictor shocks  |  {landtype}  (lag p={res.k_ar})")
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            outp = outdir / f"var_irf_{landtype}_{tgt}.png"
            fig.savefig(outp, dpi=180)
            plt.close(fig)
            log(f"[OK] Saved VAR IRF: {outp}")
    except Exception as e:
        log(f"[ERROR] VAR/IRF failed for {landtype}: {e}")
        log(traceback.format_exc())


# =========================
#  Main driver
# =========================
def main():
    # fresh log
    try:
        if LOGF.exists():
            LOGF.unlink()
    except Exception:
        pass
    log("Starting pipeline (RAM-only, streaming raster means)…")

    all_landtype_tables = {}

    # 1) Aggregate means for each land type
    for lt, root in ROOTS.items():
        try:
            df = collect_timeseries_for_landtype(lt, root)
        except KeyboardInterrupt:
            log("[FATAL] Interrupted by user during raster aggregation.")
            raise
        except Exception as e:
            log(f"[ERROR] Failed collecting timeseries for {lt}: {e}")
            log(traceback.format_exc())
            continue

        # 2) Impute gaps for configured variables (SPEI/LST/TWS)
        for col in IMPUTE_THESE:
            if col in df.columns:
                before_nans = int(df[col].isna().sum())
                if before_nans > 0:
                    try:
                        df[col] = impute_series_3step(df[col], name=f"{lt}/{col}")
                        after_nans = int(df[col].isna().sum())
                        log(f"[OK] Imputed {lt}/{col}: NaNs {before_nans} → {after_nans}")
                    except KeyboardInterrupt:
                        log("[FATAL] Interrupted by user during imputation.")
                        raise
                    except Exception as e:
                        log(f"[ERROR] Imputation failed for {lt}/{col}: {e}")
                        log(traceback.format_exc())

        # 3) Save cleaned monthly table to Excel (one per land type)
        try:
            out_xlsx = OUTDIR / f"cleaned_timeseries_{lt}.xlsx"
            with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as xw:
                df.to_excel(xw, sheet_name="monthly_mean")
                # simple imputation summary
                imp_counts = []
                for col in df.columns:
                    imp_counts.append({
                        "variable": col,
                        "n_missing_after": int(df[col].isna().sum()),
                        "imputed": (col in IMPUTE_THESE)
                    })
                pd.DataFrame(imp_counts).to_excel(xw, sheet_name="imputation_summary", index=False)
            log(f"[OK] Saved cleaned monthly Excel: {out_xlsx}")
        except Exception as e:
            log(f"[ERROR] Could not save Excel for {lt}: {e}")
            log(traceback.format_exc())

        all_landtype_tables[lt] = df

    # 4) Build anomalies (z-scores by month) and prepare for Cross-Correlation Function analysis
    landtype_z_data = {}
    for lt, df in all_landtype_tables.items():
        missing_cols = [c for c in (TARGETS + PREDICTORS) if c not in df.columns]
        if missing_cols:
            log(f"[WARN] Missing columns for {lt}: {missing_cols} — skipping analyses for this land type.")
            continue

        sub = df[TARGETS + PREDICTORS].copy()
        sub = sub.loc[~sub.isna().all(axis=1)]
        if sub.empty:
            log(f"[WARN] {lt}: no data after dropping all-NaN rows.")
            continue

        z = monthly_zscore(sub)
        landtype_z_data[lt] = z

        # 4b) VAR + IRFs (responses of each target to shocks in predictors)
        fit_var_and_plot_irf(z, TARGETS, PREDICTORS, lt, MAX_LAG, OUTDIR)

    # 4a) Cross-Correlation Function plots - 2 plots with 4 subplots each
    if landtype_z_data:
        for tgt in TARGETS:
            plot_cross_correlation_function_combined(landtype_z_data, PREDICTORS, tgt, MAX_LAG, OUTDIR)

    log("\nAll done. See the _Out folder for results and logs.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        # make sure we log a nice line even if Ctrl+C happens
        log("[FATAL] KeyboardInterrupt — exiting cleanly.")
        sys.exit(130)
    except Exception as e:
        log(f"[FATAL] Uncaught exception: {e}")
        log(traceback.format_exc())
        sys.exit(1)