#!/usr/bin/env python
"""
Mask monthly CALIFORNIA GOSIF rasters with cropland masks
--------------------------------------------------------

Cropland rules
--------------
• 2001–2012  : 1  (or NaN)  = cropland  → blank out
               (2008–2012 re-use the 2007 mask)
• 2013–2021  : 1            = cropland  → blank out
--------------------------------------------------------
Outputs:  CALIFORNIA_GOSIF_MASKED_YYYYMM.tif  in OUT_DIR
"""

import os
import re
import glob
import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.errors import NotGeoreferencedWarning

# ── USER SETTINGS ───────────────────────────────────────────────
GOSIF_DIR = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\GOSIF_California_Clipped"
MASK_DIR  = r"C:\Users\LOQ\OneDrive\Desktop\California_Cropland_Masks"
OUT_DIR   = r"C:\Users\LOQ\OneDrive\Desktop\GOSIF_CA_CLEAN"
# ────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DATE_RE = re.compile(r"(\d{6})")   # grabs YYYYMM anywhere in filename

# 2008-2012 use the 2007 cropland mask
OVERRIDE_MASK_YEAR = {yr: 2007 for yr in range(2008, 2013)}   # 2008-2012 inclusive


def crop_bool(mask_array: np.ndarray, mask_year: int) -> np.ndarray:
    """
    Return a boolean array where True marks cropland pixels to be set to NaN.
    • 2001-2012 (binary masks, 1 = cropland)
    • 2013-2021 (USDA cultivated, 1 = cultivated cropland)
    """
    return (mask_array == 1) | np.isnan(mask_array)


def open_mask_aligned(mask_path: str, template_ds: rasterio.DatasetReader) -> np.ndarray:
    """
    Read the annual cropland mask; if needed, reproject to match the GOSIF grid.
    """
    with rasterio.open(mask_path) as src:
        if (src.transform == template_ds.transform and
            src.crs == template_ds.crs and
            src.shape == template_ds.shape):
            return src.read(1)

        dest = np.empty(template_ds.shape, dtype=src.dtypes[0])
        reproject(
            rasterio.band(src, 1),
            dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_ds.transform,
            dst_crs=template_ds.crs,
            resampling=Resampling.nearest,
        )
        return dest


def process_one(gosif_path: str):
    basename = os.path.basename(gosif_path)
    m = DATE_RE.search(basename)
    if not m:
        print(f"⚠️  Skipping {basename}: can’t find YYYYMM in name")
        return

    yyyymm = m.group(1)
    year   = int(yyyymm[:4])

    mask_year = OVERRIDE_MASK_YEAR.get(year, year)
    mask_tif  = os.path.join(MASK_DIR, f"CA_CroplandMask_{mask_year}.tif")
    if not os.path.exists(mask_tif):
        print(f"⚠️  Skipping {basename}: mask file missing → {mask_tif}")
        return

    with rasterio.open(gosif_path) as gosif_ds:
        gosif = gosif_ds.read(1).astype("float32")               # allow NaN
        mask  = open_mask_aligned(mask_tif, gosif_ds)
        cropland = crop_bool(mask, mask_year)                    # apply rule

        gosif[cropland] = np.nan                                 # blank out cropland

        meta = gosif_ds.meta.copy()
        meta.update(dtype="float32", nodata=np.nan, compress="lzw")

        out_name = f"CALIFORNIA_GOSIF_MASKED_{yyyymm}.tif"
        out_path = os.path.join(OUT_DIR, out_name)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(gosif, 1)

    print(f"✅  {out_name} saved.")


def main():
    tifs = sorted(glob.glob(os.path.join(GOSIF_DIR, "*.tif")))
    if not tifs:
        print("❌  No *.tif files found in GOSIF_DIR.")
        return
    for tif in tifs:
        process_one(tif)


if __name__ == "__main__":
    main()
