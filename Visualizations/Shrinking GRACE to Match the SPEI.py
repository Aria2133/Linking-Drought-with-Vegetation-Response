#!/usr/bin/env python
# Batch resample GRACE rasters to SPEI grid using area-weighted average

import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import glob

# ─── Paths ─────────────────────────────────────────────────────
SPEI_REF   = r"C:\Users\LOQ\Downloads\SPEI_2001_01.tif"  # Reference grid
GRACE_DIR  = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\GRACE_TWS_Regridded_Tonzi CORRECT"
OUT_DIR    = r"C:\Users\LOQ\Downloads\GRACE_on_SPEI_Grid"

os.makedirs(OUT_DIR, exist_ok=True)

# ─── Read SPEI grid properties ────────────────────────────────
with rasterio.open(SPEI_REF) as ref:
    ref_tr    = ref.transform
    ref_crs   = ref.crs
    ref_h     = ref.height
    ref_w     = ref.width
    ref_dtype = ref.dtypes[0]

print(f"📐 Target grid: {ref_w} × {ref_h} px  •  CRS {ref_crs}")

# ─── Process each GRACE file ──────────────────────────────────
grace_files = glob.glob(os.path.join(GRACE_DIR, "*.tif"))

for grace_path in sorted(grace_files):
    filename = os.path.basename(grace_path)
    out_path = os.path.join(OUT_DIR, filename)

    with rasterio.open(grace_path) as src:
        src_data = src.read(1)
        dst_data = np.full((ref_h, ref_w), np.nan, dtype=ref_dtype)

        reproject(
            source        = src_data,
            destination   = dst_data,
            src_transform = src.transform,
            src_crs       = src.crs,
            dst_transform = ref_tr,
            dst_crs       = ref_crs,
            resampling    = Resampling.average,
            src_nodata    = src.nodata,
            dst_nodata    = np.nan
        )

    out_meta = {
        "driver"   : "GTiff",
        "height"   : ref_h,
        "width"    : ref_w,
        "count"    : 1,
        "dtype"    : ref_dtype,
        "crs"      : ref_crs,
        "transform": ref_tr,
        "nodata"   : np.nan,
        "compress" : "lzw"
    }

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(dst_data, 1)

    print(f"✅  Saved: {filename}")

print("🎉 All GRACE files processed and saved to:")
print("📁", OUT_DIR)
