#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean -9999 pixels from an entire folder of GeoTIFFs.

• Keeps your originals untouched.
• Writes cleaned copies (LZW-compressed) into a sibling folder.
• Drops the troublesome 'tiled' flag that caused “Cannot handle zero number of tiles”.

REQUIRES:
    pip install rasterio numpy tqdm
"""

from pathlib import Path
import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

# ────────────────────────────────────────────────────────────────
# SETTINGS ── change these if you need something different
# ────────────────────────────────────────────────────────────────
SRC_DIR       = Path(r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\GOSIF_CA_MASKED")
DST_DIR       = SRC_DIR.parent / "GOSIF_CA_CLEAN"      # output folder
BAD_VAL       = -9999                                 # value to remove
REPLACEMENT   = np.nan                                # what to write instead
WRITE_NODATA  = True                                   # put a nodata tag in output?
COMPRESSION   = "LZW"                                  # keep sizes sensible
OVERWRITE_OK  = False                                  # safety: refuse to overwrite

# ────────────────────────────────────────────────────────────────
def clean_tif(src_path: Path, dst_path: Path):
    """Replace BAD_VAL pixels in one GeoTIFF and save to dst_path."""
    if dst_path.exists() and not OVERWRITE_OK:
        print(f"⚠ {dst_path.name} already exists – skipped (set OVERWRITE_OK=True to overwrite)")
        return

    with rasterio.open(src_path) as src:
        profile = src.profile.copy()

        # Read full raster (works fine for moderate-sized rasters; for huge rasters use windows).
        data = src.read()                                # shape: (bands, rows, cols)

        # Identify bad pixels
        mask = data == BAD_VAL
        if mask.any():
            # Cast to float32 if we're about to introduce NaNs
            if REPLACEMENT is np.nan and not np.issubdtype(data.dtype, np.floating):
                data = data.astype("float32", copy=False)

            # Replace
            if REPLACEMENT is np.nan:
                data = np.where(mask, np.nan, data)
            else:
                data[mask] = REPLACEMENT

        # Update metadata ─ remove 'tiled' settings that caused your error
        for k in ("tiled", "blockxsize", "blockysize"):
            profile.pop(k, None)

        profile.update(
            compress=COMPRESSION,
            nodata=(BAD_VAL if REPLACEMENT is np.nan else REPLACEMENT) if WRITE_NODATA else None,
            dtype=data.dtype,
        )

        # Write cleaned raster
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

            # Rebuild overviews (optional but nice for fast zooming in QGIS)
            if src.overviews(1):
                factors = src.overviews(1)
                dst.build_overviews(factors, Resampling.nearest)
                dst.update_tags(OVR_RESAMPLING="nearest")


def main():
    if not SRC_DIR.exists():
        raise RuntimeError(f"Source directory not found: {SRC_DIR}")

    DST_DIR.mkdir(exist_ok=True)

    tifs = sorted(SRC_DIR.glob("*.tif"))
    if not tifs:
        raise RuntimeError(f"No .tif files found in {SRC_DIR}")

    print(f"Cleaning {len(tifs)} GeoTIFFs …")
    for tif in tqdm(tifs, unit="file"):
        clean_tif(tif, DST_DIR / tif.name)

    print(f"✔ All done! Clean files are in: {DST_DIR}")


if __name__ == "__main__":
    main()
