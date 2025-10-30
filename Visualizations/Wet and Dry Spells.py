# -*- coding: utf-8 -*-
"""
Spatial correlation maps with on-the-fly reprojection to a common reference grid.

Fixes:
- Reproject/resample every monthly raster to a single reference profile per Water Year
  to avoid shape mismatches across indices/landtypes.
- Robust logging & error handling.

Outputs:
  _Out/Spatial/<MODE>/<WY>/corr_WY<wy>_<TARGET>_vs_<PRED>.png
  _Out/Spatial/spatial_logs.txt
"""

import os, re, sys, math, traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import rasterio
from rasterio.warp import reproject, Resampling

# -------------------- USER CONFIG --------------------
ROOTS: Dict[str, str] = {
    "Grasslands": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS",
    "NaturalForests": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS",
    "Savannas": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS",
    "Shrublands": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS",
}
INDICES_ALL: List[str] = ["GNDVI", "GOSIF", "GPP", "LST", "NDVI", "NIRv", "SPEI", "TWS", "VPD"]
TARGETS: List[str]   = ["GPP", "GOSIF"]
PREDICTORS: List[str]= ["GNDVI", "NDVI", "NIRv", "LST", "TWS", "VPD", "SPEI"]

# Which regimes to render: "dry" / "wet" / "both"
MODE = "dry"

# I/O
OUTDIR = Path("_Out")
CLEAN_XLSX_TMPL = OUTDIR / "cleaned_timeseries_{lt}.xlsx"
SPATIAL_OUT = OUTDIR / "Spatial"
LOGF = SPATIAL_OUT / "spatial_logs.txt"

# Correlation needs at least N valid month-pairs
MIN_VALID_PAIRS = 8

# ----------------------------------------------------

DATE_PATS = [
    re.compile(r"(?P<y>20\d{2})[_-](?P<m>0[1-9]|1[0-2])"),
    re.compile(r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])"),
]

def log(msg: str):
    print(msg)
    try:
        SPATIAL_OUT.mkdir(exist_ok=True, parents=True)
        with open(LOGF, "a", encoding="utf-8") as f:
            f.write(str(msg).rstrip() + "\n")
    except Exception:
        pass

def parse_period_from_name(name: str) -> Optional[pd.Period]:
    for pat in DATE_PATS:
        m = pat.search(name)
        if m:
            y, mo = int(m.group("y")), int(m.group("m"))
            return pd.Period(f"{y}-{mo:02d}", freq="M")
    return None

def monthly_zscore(df: pd.DataFrame) -> pd.DataFrame:
    z = df.copy()
    if isinstance(z.index, pd.DatetimeIndex):
        z.index = z.index.to_period("M")
    elif not isinstance(z.index, pd.PeriodIndex):
        z.index = pd.PeriodIndex(z.index, freq="M")
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

def build_cmi(zdf: pd.DataFrame) -> pd.Series:
    miss = [x for x in ["SPEI","TWS","VPD","LST"] if x not in zdf.columns]
    if miss:
        raise KeyError(f"Missing columns for CMI: {miss}")
    M = zdf["SPEI"] + zdf["TWS"] - zdf["VPD"] - zdf["LST"]
    M.name = "CMI"
    return M

def water_year_of_period(p: pd.Period) -> int:
    return p.year + 1 if p.month >= 10 else p.year

def pick_top3_years_by_cmi() -> Tuple[List[int], List[int]]:
    wy_cmis = []
    for lt in ROOTS.keys():
        xlsx = CLEAN_XLSX_TMPL.with_name(f"cleaned_timeseries_{lt}.xlsx")
        if not xlsx.exists():
            log(f"[WARN] Missing cleaned table for {lt}: {xlsx}")
            continue
        try:
            df = pd.read_excel(xlsx, sheet_name="monthly_mean", index_col=0)
            if not isinstance(df.index, (pd.PeriodIndex, pd.DatetimeIndex)):
                df.index = pd.to_datetime(df.index)
            if isinstance(df.index, pd.DatetimeIndex):
                df.index = df.index.to_period("M")
            z = monthly_zscore(df)
            M = build_cmi(z)
            wy = pd.Series([water_year_of_period(p) for p in M.index], index=M.index)
            cmi_wy = pd.DataFrame({"wy": wy.values, "M": M.values}).groupby("wy")["M"].mean()
            wy_cmis.append(cmi_wy)
        except Exception as e:
            log(f"[ERROR] Failed CMI for {lt}: {e}")
            log(traceback.format_exc())

    if not wy_cmis:
        raise RuntimeError("No cleaned tables found to compute CMI.")
    df_all = pd.concat(wy_cmis, axis=1).mean(axis=1).dropna()
    if df_all.shape[0] < 6:
        log("[WARN] Few WYs available; proceeding anyway.")
    driest  = df_all.nsmallest(3).index.tolist()
    wettest = df_all.nlargest(3).index.tolist()
    log(f"[OK] 3 driest WYs:  {driest}")
    log(f"[OK] 3 wettest WYs: {wettest}")
    return driest, wettest

def find_month_files_for_index(landtype: str, index_name: str, year: int, month: int) -> List[Path]:
    root = Path(ROOTS[landtype]) / index_name
    if not root.exists():
        return []
    ym1 = f"{year}_{month:02d}"
    ym2 = f"{year}{month:02d}"
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in (".tif",".tiff"):
            continue
        name = p.name
        if ym1 in name or ym2 in name:
            files.append(p)
    return sorted(files)

def wy_month_to_ym(wy: int, slot: int) -> Tuple[int, int]:
    # slot: 0..11 for Oct..Sep
    mlist = [10,11,12,1,2,3,4,5,6,7,8,9]
    m = mlist[slot]
    y = wy-1 if m>=10 else wy
    return y, m

# ----------- NEW: reference profile & reprojection -----------
def choose_reference_profile(wy: int) -> Optional[dict]:
    """
    Pick a robust statewide reference grid for this WY.
    Priority: indices with statewide coverage (GRIDMET VPD, MODIS NDVI 5km),
    then GPP/GOSIF, then others. Also validate by checking that at least one
    other index overlaps (reproject a sample and ensure > threshold finite px).
    """
    preferred = [
        "VPD",   # GRIDMET, سراسری
        "NDVI",  # MOD13Q1 5km
        "GPP",
        "GOSIF",
    ] + [x for x in INDICES_ALL if x not in ("VPD","NDVI","GPP","GOSIF")]

    # جمع‌آوری کاندیدها (profile + تعداد پیکسل)
    candidates = []
    for idx in preferred:
        for slot in range(12):
            y, m = wy_month_to_ym(wy, slot)
            for lt in ROOTS.keys():
                try:
                    cand = find_month_files_for_index(lt, idx, y, m)
                    if not cand:
                        continue
                    fp = cand[0]
                    with rasterio.open(fp) as ds:
                        prof = ds.profile.copy()
                        if not all(k in prof for k in ("height","width","transform","crs")):
                            continue
                        prof.update(dtype="float32")
                        area = int(prof.get("height",0)) * int(prof.get("width",0))
                        if area <= 0:
                            continue
                        candidates.append((idx, fp, prof, area))
                except Exception:
                    continue

    if not candidates:
        return None

    # ابتدا فقط کاندیدهای سریِ preferred را نگه داریم (همان ترتیب)
    filtered = []
    seen = set()
    for p in preferred:
        for (idx, fp, prof, area) in sorted(candidates, key=lambda x: -x[3]):
            if idx == p and (idx, fp) not in seen:
                filtered.append((idx, fp, prof, area))
                seen.add((idx, fp))

    # اعتبارسنجی: روی هر کاندید یک شاخص دیگر را تست کنیم که حداقل 5% پیکسل finite بده
    def overlaps_ok(prof: dict) -> bool:
        test_pairs = [("GPP","NDVI"), ("GOSIF","NDVI"), ("GPP","VPD"), ("GOSIF","VPD")]
        for a,b in test_pairs:
            # یکی از این دو را پیدا کنیم در یکی از ماه‌ها
            for slot in range(12):
                y, m = wy_month_to_ym(wy, slot)
                for lt in ROOTS.keys():
                    fa = find_month_files_for_index(lt, a, y, m)
                    if not fa:
                        continue
                    try:
                        arr = read_and_reproject_to_ref(fa[0], prof)
                        if arr is None:
                            continue
                        frac = np.isfinite(arr).mean()
                        if frac >= 0.05:  # حداقل ۵٪ پوشش
                            return True
                    except Exception:
                        continue
        return False

    for (idx, fp, prof, area) in filtered:
        try:
            if overlaps_ok(prof):
                return prof
        except Exception:
            continue

    # اگر هیچ‌کدام overlap پاس نداد، بزرگ‌ترین فریم را برگردانیم (آخرین علاج)
    best = max(filtered, key=lambda x: x[3])[2]
    return best

def read_and_reproject_to_ref(fp: Path, ref_prof: dict) -> Optional[np.ndarray]:
    """
    Read band1 and reproject/resample to ref_prof (crs, transform, width, height).
    Output float32 with NaN for nodata.
    """
    try:
        with rasterio.open(fp) as src:
            arr = src.read(1)
            src_nodata = src.nodata
            # prepare destination
            H, W = ref_prof["height"], ref_prof["width"]
            dst = np.full((H, W), np.nan, dtype="float32")

            reproject(
                source=arr,
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src_nodata,
                dst_transform=ref_prof["transform"],
                dst_crs=ref_prof["crs"],
                dst_nodata=np.nan,
                resampling=Resampling.bilinear
            )
            # set non-finite to NaN
            dst = dst.astype("float32")
            dst[~np.isfinite(dst)] = np.nan
            return dst
    except Exception as e:
        log(f"[ERROR] reproject failed for {fp}: {e}")
        log(traceback.format_exc())
        return None

def mosaic_month_raster(index_name: str, wy: int, slot: int, ref_prof: dict) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Build statewide array for given index_name and WY slot (0..11) on ref_prof grid.
    """
    y, m = wy_month_to_ym(wy, slot)
    base = None
    for lt in ROOTS.keys():
        try:
            cand = find_month_files_for_index(lt, index_name, y, m)
            if not cand:
                continue
            fp = cand[0]
            arr = read_and_reproject_to_ref(fp, ref_prof)
            if arr is None:
                continue
            if base is None:
                base = np.full(arr.shape, np.nan, dtype="float32")
            if arr.shape != base.shape:
                log(f"[WARN] After reprojection, shape mismatch for {index_name} {y}-{m:02d} ({lt}). Skipping this tile.")
                continue
            valid = np.isfinite(arr)
            base[valid] = arr[valid]  # overwrite OK
        except Exception as e:
            log(f"[ERROR] mosaic month failed {index_name} {wy} {y}-{m:02d} ({lt}): {e}")
            log(traceback.format_exc())
    if base is None:
        return None, None
    return base, ref_prof

def stack_year(index_name: str, wy: int, ref_prof: dict) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    """
    Stack 12 reprojected monthly arrays to shape (12,H,W).
    """
    arrs = []
    for slot in range(12):
        a, _ = mosaic_month_raster(index_name, wy, slot, ref_prof)
        arrs.append(a)
    if all(a is None for a in arrs):
        return None, None
    # infer shape
    shape = None
    for a in arrs:
        if a is not None:
            shape = a.shape
            break
    if shape is None:
        return None, None
    H,W = shape
    out = np.full((12,H,W), np.nan, dtype="float32")
    for i,a in enumerate(arrs):
        if a is not None:
            if a.shape != (H,W):
                log(f"[WARN] Year stack shape mismatch at {index_name} WY{wy} month slot {i+1}; skipping this month.")
                continue
            out[i] = a
    return out, ref_prof

# ---------------- correlation & plotting ----------------
def corr_map(target_stack: np.ndarray, pred_stack: np.ndarray, min_pairs: int = MIN_VALID_PAIRS) -> np.ndarray:
    """
    Compute Pearson r per pixel with pairwise NaN handling and a minimum number of
    valid month-pairs. Avoids runtime warnings by masking columns with insufficient data
    before moments are computed.
    """
    if target_stack is None or pred_stack is None:
        return None
    if target_stack.shape != pred_stack.shape:
        raise ValueError("Stacks must have same shape (T,H,W).")

    T, H, W = target_stack.shape
    X = target_stack.reshape(T, -1)
    Y = pred_stack.reshape(T, -1)
    valid = np.isfinite(X) & np.isfinite(Y)
    n = valid.sum(axis=0)

    # ستون‌هایی که زوجِ معتبر کافی ندارند → NaN نهایی
    bad = n < min_pairs
    if bad.all():
        return np.full((H, W), np.nan, dtype="float32")

    # فقط روی ستون‌های خوب آمار بگیریم تا هشدار نگیریم
    good_idx = np.where(~bad)[0]
    Xg = X[:, good_idx]
    Yg = Y[:, good_idx]
    Vg = valid[:, good_idx]

    with np.errstate(invalid="ignore", divide="ignore"):
        Xg = np.where(Vg, Xg, np.nan)
        Yg = np.where(Vg, Yg, np.nan)
        mx = np.nanmean(Xg, axis=0)
        my = np.nanmean(Yg, axis=0)
        Xc = Xg - mx
        Yc = Yg - my
        num = np.nanmean(Xc * Yc, axis=0)
        sx = np.nanstd(Xc, axis=0)
        sy = np.nanstd(Yc, axis=0)
        denom = sx * sy
        rg = np.divide(num, denom, out=np.full_like(num, np.nan), where=denom > 0)

    r = np.full(X.shape[1], np.nan, dtype="float32")
    r[good_idx] = rg.astype("float32")
    return r.reshape(H, W)


def save_corr_png(rmap: np.ndarray, title: str, outpng: Path):
    try:
        outpng.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7,6))
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("rwblue", [(1,0,0),(1,1,1),(0,0,1)], N=256)
        im = ax.imshow(rmap, cmap=cmap, vmin=-1, vmax=1)
        ax.set_axis_off()
        cb = plt.colorbar(im, ax=ax, shrink=0.8)
        cb.set_label("Pearson r  (blue=positive, red=negative)")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outpng, dpi=180)
        plt.close(fig)
        log(f"[OK] Saved: {outpng}")
        finite_frac = float(np.isfinite(rmap).mean()) if rmap is not None else 0.0
        log(f"[COVER] {title}: finite pixels = {finite_frac * 100:.1f}% (min_pairs={MIN_VALID_PAIRS})")

    except Exception as e:
        log(f"[ERROR] save_corr_png failed: {e}")
        log(traceback.format_exc())

# ---------------- main ----------------
def main():
    # fresh log
    try:
        if LOGF.exists(): LOGF.unlink()
    except Exception: pass
    log("Starting spatial correlation pipeline with reprojection…")

    # 1) pick years (from cleaned tables)
    try:
        driest3, wettest3 = pick_top3_years_by_cmi()
    except Exception as e:
        log(f"[FATAL] Could not pick top-3 years: {e}")
        log(traceback.format_exc())
        return 1

    years_sets = []
    if MODE.lower() in ("dry","both"): years_sets.append(("dry", driest3))
    if MODE.lower() in ("wet","both"): years_sets.append(("wet", wettest3))
    if not years_sets:
        log("[FATAL] MODE must be one of: dry / wet / both")
        return 1

    for label, years in years_sets:
        for wy in years:
            log(f"\n=== Working on WY {wy} ({label}) ===")
            # 2) choose reference grid for this WY
            ref_prof = choose_reference_profile(wy)
            if ref_prof is None:
                log(f"[WARN] No reference grid found for WY{wy}; skipping this year.")
                continue

            # Ensure required keys exist (height/width/transform/crs)
            if not all(k in ref_prof for k in ("height","width","transform","crs")):
                log(f"[WARN] Reference profile incomplete for WY{wy}; skipping.")
                continue

            # 3) cache target stacks
            target_cache: Dict[str, Tuple[np.ndarray, dict]] = {}
            for tgt in TARGETS:
                try:
                    t_stack, _ = stack_year(tgt, wy, ref_prof)
                    if t_stack is None:
                        log(f"[WARN] No stack for {tgt} WY{wy}; skipping this target/year.")
                        continue
                    target_cache[tgt] = (t_stack, ref_prof)

                    # 4) for each predictor
                    for pred in PREDICTORS:
                        try:
                            p_stack, _ = stack_year(pred, wy, ref_prof)
                            if p_stack is None:
                                log(f"[WARN] No stack for {pred} WY{wy}; skipping this correlation.")
                                continue
                            if p_stack.shape != t_stack.shape:
                                log(f"[WARN] Shape mismatch {tgt} vs {pred} at WY{wy} AFTER reprojection; skipping.")
                                continue
                            rmap = corr_map(t_stack, p_stack, MIN_VALID_PAIRS)
                            if rmap is None:
                                log(f"[WARN] rmap None for {tgt}-{pred} WY{wy}")
                                continue
                            outpng = SPATIAL_OUT / label / f"WY{wy}" / f"corr_WY{wy}_{tgt}_vs_{pred}.png"
                            title = f"Pearson r: {tgt} vs {pred} | WY {wy} ({label})"
                            save_corr_png(rmap, title, outpng)
                        except Exception as e:
                            log(f"[ERROR] correlation {tgt}~{pred} WY{wy} failed: {e}")
                            log(traceback.format_exc())
                except Exception as e:
                    log(f"[ERROR] target stack failed for {tgt} WY{wy}: {e}")
                    log(traceback.format_exc())

    log("\nAll done. See _Out/Spatial/")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("[FATAL] Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        log(f"[FATAL] Uncaught: {e}")
        log(traceback.format_exc())
        sys.exit(1)
