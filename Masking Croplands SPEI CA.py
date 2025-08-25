#!/usr/bin/env python
"""
Mask monthly SPEI rasters (CSIC) with California cropland masks
---------------------------------------------------------------

Cropland rules (same as for GOSIF)
----------------------------------
• 2001–2012  : 1  (or NaN)  = cropland  → blank out
               (2008–2012 re-use the 2007 mask)
• 2013–2021  : 1            = cropland  → blank out
---------------------------------------------------------------
Input filenames expected:
    SPEI1_CSIC_CA_YYYY_MM.tif
Output filenames:
    SPEI1_CSIC_CA_YYYY_MM_masked.tif
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
SPEI_DIR = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\SPEI1_CSIC_TIFs"
MASK_DIR = r"C:\Users\LOQ\OneDrive\Desktop\California_Cropland_Masks"
OUT_DIR  = r"C:\Users\LOQ\OneDrive\Desktop\SPEI_Masked_Output"
# ────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# capture YYYY_MM anywhere in the filename
DATE_RE = re.compile(r"(\d{4})[_\-]?(\d{2})")

# years 2008-2012 use the 2007 cropland mask
OVERRIDE_MASK_YEAR = {yr: 2007 for yr in range(2008, 2013)}   # 2008-2012 inclusive


def crop_bool(mask_array: np.ndarray) -> np.ndarray:
    """True where pixel is cropland (mask value 1 or NaN)."""
    return (mask_array == 1) | np.isnan(mask_array)


def open_mask_aligned(mask_path: str, template_ds: rasterio.DatasetReader) -> np.ndarray:
    """Read mask and reproject to match SPEI raster if needed."""
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


def process_one(spei_path: str):
    basename = os.path.basename(spei_path)
    m = DATE_RE.search(basename)
    if not m:
        print(f"⚠️  Skipping {basename}: can’t find YYYY_MM in name")
        return

    year  = int(m.group(1))
    month = m.group(2)               # keep 2-digit month for output

    mask_year = OVERRIDE_MASK_YEAR.get(year, year)
    mask_tif  = os.path.join(MASK_DIR, f"CA_CroplandMask_{mask_year}.tif")
    if not os.path.exists(mask_tif):
        print(f"⚠️  Skipping {basename}: cropland mask missing → {mask_tif}")
        return

    with rasterio.open(spei_path) as spei_ds:
        spei = spei_ds.read(1).astype("float32")             # enable NaN
        mask = open_mask_aligned(mask_tif, spei_ds)
        spei[crop_bool(mask)] = np.nan                       # blank cropland

        meta = spei_ds.meta.copy()
        meta.update(dtype="float32", nodata=np.nan, compress="lzw")

        out_name = f"SPEI1_CSIC_CA_{year}_{month}_masked.tif"
        out_path = os.path.join(OUT_DIR, out_name)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(spei, 1)

    print(f"✅  {out_name} saved.")


def main():
    tifs = sorted(glob.glob(os.path.join(SPEI_DIR, "*.tif")))
    if not tifs:
        print("❌  No *.tif files found in SPEI_DIR.")
        return
    for tif in tifs:
        process_one(tif)


if __name__ == "__main__":
    main()
