#!/usr/bin/env python
"""
Mask monthly GRACE TWS rasters with California cropland masks
-------------------------------------------------------------

Cropland rules
--------------
• 2001–2012  : mask value 1 (or NaN) = cropland → set to NaN
               (2008–2012 reuse the 2007 mask)
• 2013–2021  : mask value 1          = cropland → set to NaN
-------------------------------------------------------------
Input  : TWS_GRACE_CA_YYYY_MM.tif
Output : TWS_GRACE_CA_YYYY_MM_masked.tif
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
TWS_DIR  = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\TWS_GRACE_TIFs"
MASK_DIR = r"C:\Users\LOQ\OneDrive\Desktop\California_Cropland_Masks"
OUT_DIR  = r"C:\Users\LOQ\OneDrive\Desktop\TWS_Masked_Output"
# ────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

DATE_RE = re.compile(r"(\d{4})[_\-]?(\d{2})")   # finds YYYY_MM in filename

# 2008-2012 → use the 2007 cropland mask
OVERRIDE_MASK_YEAR = {yr: 2007 for yr in range(2008, 2013)}


def crop_bool(mask_arr: np.ndarray) -> np.ndarray:
    """True where pixel is cropland (mask value 1 or NaN)."""
    return (mask_arr == 1) | np.isnan(mask_arr)


def open_mask_aligned(mask_path: str, template_ds: rasterio.DatasetReader) -> np.ndarray:
    """Read mask and re-project to match the GRACE grid if needed."""
    with rasterio.open(mask_path) as src:
        if (src.transform == template_ds.transform and
            src.crs == template_ds.crs and
            src.shape == template_ds.shape):
            return src.read(1)

        dest = np.empty(template_ds.shape, dtype=src.dtypes[0])
        reproject(
            rasterio.band(src, 1), dest,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=template_ds.transform, dst_crs=template_ds.crs,
            resampling=Resampling.nearest,
        )
        return dest


def process_one(tws_path: str):
    fname = os.path.basename(tws_path)
    m = DATE_RE.search(fname)
    if not m:
        print(f"⚠️  Skipping {fname}: can’t find YYYY_MM in name")
        return

    year  = int(m.group(1))
    month = m.group(2)                      # already two digits

    mask_year = OVERRIDE_MASK_YEAR.get(year, year)
    mask_tif  = os.path.join(MASK_DIR, f"CA_CroplandMask_{mask_year}.tif")
    if not os.path.exists(mask_tif):
        print(f"⚠️  Skipping {fname}: cropland mask missing → {mask_tif}")
        return

    with rasterio.open(tws_path) as tws_ds:
        tws = tws_ds.read(1).astype("float64")          # GRACE is float64
        mask = open_mask_aligned(mask_tif, tws_ds)
        tws[crop_bool(mask)] = np.nan                   # blank cropland

        meta = tws_ds.meta.copy()
        meta.update(dtype="float64", nodata=np.nan, compress="lzw")

        out_name = f"TWS_GRACE_CA_{year}_{month}_masked.tif"
        out_path = os.path.join(OUT_DIR, out_name)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(tws, 1)

    print(f"✅  {out_name} saved.")


def main():
    tifs = sorted(glob.glob(os.path.join(TWS_DIR, "*.tif")))
    if not tifs:
        print("❌  No *.tif files found in TWS_DIR.")
        return
    for tif in tifs:
        process_one(tif)


if __name__ == "__main__":
    main()
