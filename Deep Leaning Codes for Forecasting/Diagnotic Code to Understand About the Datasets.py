# ============================== DIAGNOSTICS SUITE ==============================
# Purpose: Inspect your raster stacks per ecosystem, check shapes/CRS/nodata,
#          reproject one sample per variable to a common grid, and report
#          valid-pixel overlaps + value ranges so we can see what's breaking.
# Outputs (per ecosystem):
#   01_scan_summary.csv              -> shapes/CRS/nodata/value ranges (raw)
#   02_overlap_pairwise.csv          -> pairwise valid-pixel overlaps (after reproj)
#   03_overlap_all.txt               -> count of pixels valid in ALL vars
#   04_footprints_preview.png        -> quick masks preview
#   05_value_ranges_after_reproj.csv -> value ranges after reprojection
# ===============================================================================
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

# headless plotting (no GUI needed)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import rasterio
from rasterio.warp import reproject
from rasterio.enums import Resampling as RioResampling

# Common fill values frequently used in Earth observation rasters
COMMON_FILL_VALUES = set([
    -9999, -9999.0, -32768, -3.4028235e+38, 3.4028235e+38, 9.96921e36, 1e20, 1e30
])

# ------------------------------- helpers --------------------------------
def _pixel_size_from_transform(transform):
    try:
        return (abs(transform.a), abs(transform.e))
    except Exception:
        return (np.nan, np.nan)

def _read_band_masked(src):
    """
    Read band 1 as a masked array (respects src.nodata if present),
    convert to float32 with NaN for nodata/fills, and return (data, mask).
    mask=True means valid data.
    """
    arr = src.read(1, masked=True)
    data = arr.filled(np.nan).astype(np.float32)
    mask = ~np.isnan(data)

    # Extra masking for classic fills or absurd values if nodata tag is missing/insufficient
    if mask.any():
        extra_mask = np.zeros_like(mask, dtype=bool)
        for fv in COMMON_FILL_VALUES:
            extra_mask |= (data == fv)
        # mask “impossible” ranges (huge positive/negative)
        extra_mask |= np.abs(data) > 1e10
        if extra_mask.any():
            data[extra_mask] = np.nan
            mask = ~np.isnan(data)

    return data, mask

def _summarize_array(data, mask):
    valid = data[mask]
    n = valid.size
    if n == 0:
        return dict(valid=0, min=np.nan, p5=np.nan, median=np.nan, p95=np.nan, max=np.nan, mean=np.nan)
    return dict(
        valid=int(n),
        min=float(np.nanmin(valid)),
        p5=float(np.nanpercentile(valid, 5)),
        median=float(np.nanmedian(valid)),
        p95=float(np.nanpercentile(valid, 95)),
        max=float(np.nanmax(valid)),
        mean=float(np.nanmean(valid)),
    )

def _choose_reference_grid(eco_path: Path,
                           preferred=("NDVI","NIRv","SPEI","LST","GNDVI","GOSIF","GPP","VPD","TWS"),
                           max_files_per_var=6):
    """
    Pick a reference raster among preferred variables by the *largest* grid
    (height*width). Returns (area, (h,w), profile, path, var) or None.
    """
    best = None
    for var in preferred:
        d = eco_path / var
        if not d.exists():
            continue
        # accept both .tif and .tiff
        tifs = list(d.glob("*.tif")) + list(d.glob("*.tiff"))
        for fp in tifs[:max_files_per_var]:
            try:
                with rasterio.open(fp) as src:
                    h, w = src.height, src.width
                    area = h * w
                    if (best is None) or (area > best[0]):
                        best = (area, (h, w), src.profile.copy(), fp, var)
            except Exception:
                pass
    return best

def _reproject_to_ref(src_data, src_transform, src_crs, dst_profile, resampling=RioResampling.bilinear):
    """
    Reproject a 2D array to the destination profile. Returns (dst_data, dst_mask).
    dst_mask=True where dst_data is finite.
    """
    dst_h, dst_w = dst_profile["height"], dst_profile["width"]
    dst_transform = dst_profile["transform"]
    dst_crs = dst_profile["crs"]

    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    reproject(
        source=src_data,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling
    )
    mask = np.isfinite(dst)
    return dst, mask

# --------------------------- core diagnostics ---------------------------
def diagnose_ecosystem(base_path: str, ecosystem: str, max_files_per_var=6, sample_pixels=100000):
    eco_path = Path(base_path) / ecosystem
    out_dir = Path(base_path) / "SEM_RESULTS" / "_DIAGNOSTICS" / ecosystem.replace("CALIFORNIA ","")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🔎 DIAGNOSTICS for {ecosystem}")
    if not eco_path.exists():
        print(f"  ❌ Missing folder: {eco_path}")
        return

    variables = ["SPEI","TWS","VPD","LST","NDVI","GNDVI","GOSIF","GPP","NIRv"]
    records = []

    # 1) Per-variable quick summaries on first few files
    print("  📂 Scanning first files per variable...")
    per_var_samples = {}
    for var in variables:
        vdir = eco_path / var
        if not vdir.exists():
            records.append(dict(var=var, status="missing_dir"))
            print(f"    ⚠️ {var}: folder not found")
            continue

        tifs = list(vdir.glob("*.tif")) + list(vdir.glob("*.tiff"))
        if not tifs:
            records.append(dict(var=var, status="no_tifs"))
            print(f"    ⚠️ {var}: no tif/tiff files")
            continue

        used = 0
        for fp in tifs[:max_files_per_var]:
            try:
                with rasterio.open(fp) as src:
                    data, mask = _read_band_masked(src)
                    h, w = data.shape
                    px = _pixel_size_from_transform(src.transform)
                    nod = src.nodata
                    crs_str = str(src.crs) if src.crs else "None"

                    summ = _summarize_array(data, mask)
                    rec = dict(
                        var=var, file=fp.name, status="ok",
                        shape=f"{h}x{w}", dtype=str(src.dtypes[0]),
                        crs=crs_str, px_x=px[0], px_y=px[1],
                        nodata=str(nod),
                        **summ
                    )
                    records.append(rec)

                    # Keep one sample per var for later reproject test
                    if var not in per_var_samples:
                        per_var_samples[var] = (data, mask, src.transform, src.crs, fp.name)
                    used += 1
            except Exception as e:
                records.append(dict(var=var, file=fp.name, status=f"read_error: {e}"))

        if used == 0:
            print(f"    ⚠️ {var}: unable to read any files")

    df_scan = pd.DataFrame(records)
    df_scan.to_csv(out_dir / "01_scan_summary.csv", index=False)
    print(f"  💾 Saved scan summary -> {out_dir / '01_scan_summary.csv'}")

    # 2) Choose reference grid
    ref = _choose_reference_grid(eco_path, max_files_per_var=max_files_per_var)
    if ref is None:
        print("  ❌ Could not choose a reference grid (no readable rasters).")
        return
    area, (rh, rw), ref_profile, ref_path, ref_var = ref
    print(f"  🎯 Reference grid: {rh}x{rw} from {ref_path.name} (var={ref_var})")
    ref_profile["dtype"] = "float32"  # ensure float destination

    # Limit destination to at most sample_pixels for speed
    total = rh * rw
    if total > sample_pixels:
        factor = math.sqrt(total / sample_pixels)
        new_h = max(1, int(rh / factor))
        new_w = max(1, int(rw / factor))
        t = ref_profile["transform"]
        # scale transform so that geolocation stays consistent with new pixel size
        scale_y = ref_profile["height"] / new_h
        scale_x = ref_profile["width"] / new_w
        new_transform = rasterio.Affine(t.a*scale_x, t.b, t.c, t.d, t.e*scale_y, t.f)
        ref_profile["height"] = new_h
        ref_profile["width"]  = new_w
        ref_profile["transform"] = new_transform
        print(f"  ⚖️  Downsampled ref grid to {new_h}x{new_w} for fast diagnostics")

    # 3) Reproject 1 sample per variable to ref grid; compute valid px and overlaps
    repro = {}
    for var in variables:
        if var not in per_var_samples:
            continue
        data, mask, trf, crs, fname = per_var_samples[var]
        try:
            if crs is None:
                print(f"    ⚠️ {var}: missing CRS in {fname}, skipping reprojection test")
                continue
            # Use bilinear for continuous variables (all your vars are continuous)
            dst, dst_mask = _reproject_to_ref(
                data, trf, crs, ref_profile, resampling=RioResampling.bilinear
            )
            repro[var] = dict(arr=dst, mask=dst_mask, from_file=fname)
            print(f"    ↪ {var}: valid after reprojection -> {int(dst_mask.sum()):,} / {dst_mask.size:,}")
        except Exception as e:
            print(f"    ❌ {var}: reprojection failed: {e}")

    vars_present = sorted(repro.keys())
    if not vars_present:
        print("  ❌ No variables reprojected successfully.")
        return

    # Pairwise overlap matrix
    overlap = np.zeros((len(vars_present), len(vars_present)), dtype=int)
    for i, vi in enumerate(vars_present):
        mi = repro[vi]["mask"]
        for j, vj in enumerate(vars_present):
            mj = repro[vj]["mask"]
            overlap[i, j] = int((mi & mj).sum())

    df_ov = pd.DataFrame(overlap, index=vars_present, columns=vars_present)
    df_ov.to_csv(out_dir / "02_overlap_pairwise.csv")
    print(f"  💾 Saved pairwise overlap -> {out_dir / '02_overlap_pairwise.csv'}")

    # All-variables overlap (how many pixels valid in ALL variables)
    all_mask = None
    for v in vars_present:
        all_mask = repro[v]["mask"] if all_mask is None else (all_mask & repro[v]["mask"])
    all_valid = int(all_mask.sum()) if all_mask is not None else 0
    with open(out_dir / "03_overlap_all.txt", "w") as f:
        f.write(f"Valid pixels in ALL variables reprojected together: {all_valid}\n")
        for v in vars_present:
            f.write(f"{v}: {int(repro[v]['mask'].sum())}\n")
    print(f"  💾 Saved ALL-overlap count -> {out_dir / '03_overlap_all.txt'}")
    print(f"  📊 ALL-variables valid pixels: {all_valid:,} ({ref_profile['height']}x{ref_profile['width']} grid)")

    # 4) Quick footprints preview PNG
    fig, axes = plt.subplots(1, min(len(vars_present), 5), figsize=(4*min(len(vars_present),5), 4))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, v in zip(axes, vars_present[:5]):
        ax.imshow(repro[v]["mask"], origin="upper")
        ax.set_title(f"{v}\nvalid={int(repro[v]['mask'].sum())}")
        ax.axis("off")
    plt.tight_layout()
    png = out_dir / "04_footprints_preview.png"
    plt.savefig(png, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  🖼️ Saved footprints preview -> {png}")

    # 5) Value ranges after reprojection (sanity check)
    post_stats = []
    for v in vars_present:
        arr = repro[v]["arr"]
        msk = repro[v]["mask"]
        s = _summarize_array(arr, msk)
        s.update(dict(var=v))
        post_stats.append(s)
    pd.DataFrame(post_stats).to_csv(out_dir / "05_value_ranges_after_reproj.csv", index=False)
    print(f"  💾 Saved post-reprojection value ranges -> {out_dir / '05_value_ranges_after_reproj.csv'}")

def diagnose_all(base_path: str, ecosystems: list, max_files_per_var=6, sample_pixels=100000):
    print("\n======================= GLOBAL DIAGNOSTICS =======================")
    for eco in ecosystems:
        diagnose_ecosystem(base_path, eco, max_files_per_var=max_files_per_var, sample_pixels=sample_pixels)
    print("===================== END GLOBAL DIAGNOSTICS =====================")

# ============================== MAIN (YOUR PATHS) ==============================
if __name__ == "__main__":
    BASE_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
    ECOSYSTEMS = [
        "CALIFORNIA GRASSLANDS",
        "CALIFORNIA NATURAL FORESTS",
        "CALIFORNIA SAVANNAS",
        "CALIFORNIA SHRUBLANDS",
    ]

    # tweak these for speed/coverage if you like
    diagnose_all(
        base_path=BASE_PATH,
        ecosystems=ECOSYSTEMS,
        max_files_per_var=6,     # scan up to 6 tifs per variable
        sample_pixels=200000     # downsample the reference grid to ~200k cells for speed
    )
