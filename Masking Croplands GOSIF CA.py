import os
import rasterio
import numpy as np
from glob import glob
from rasterio.warp import reproject, Resampling

# Define paths
gosif_dir = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\GOSIF_California_Clipped"
cropland_dir = r"C:\Users\LOQ\OneDrive\Desktop\California_Cropland_Masks"
output_dir = r"C:\Users\LOQ\OneDrive\Desktop\GOSIF_CA_MASKED"

os.makedirs(output_dir, exist_ok=True)

# Get all GOSIF files
gosif_files = sorted(glob(os.path.join(gosif_dir, "CALIFORNIA_GOSIF_*.tif")))

for gosif_path in gosif_files:
    # Extract year and month from filename
    basename = os.path.basename(gosif_path)
    year_month = basename.split("_")[-1][:6]  # Extracts YYYYMM
    year = year_month[:4]
    month = year_month[4:]

    cropland_path = os.path.join(cropland_dir, f"CA_CroplandMask_{year}.tif")

    if not os.path.exists(cropland_path):
        print(f"Cropland file missing for year {year}, skipping.")
        continue

    with rasterio.open(gosif_path) as gosif_src, rasterio.open(cropland_path) as crop_src:
        gosif = gosif_src.read(1)
        gosif_meta = gosif_src.meta.copy()

        # Prepare empty array for reprojected cropland mask
        crop_resampled = np.empty(gosif.shape, dtype=np.uint8)

        # Reproject cropland to match GOSIF
        reproject(
            source=rasterio.band(crop_src, 1),
            destination=crop_resampled,
            src_transform=crop_src.transform,
            src_crs=crop_src.crs,
            dst_transform=gosif_src.transform,
            dst_crs=gosif_src.crs,
            resampling=Resampling.nearest
        )

        # Mask croplands: set to nan where cropland == 1
        masked_gosif = np.where(crop_resampled == 1, np.nan, gosif)

        # Save masked output
        gosif_meta.update(dtype=rasterio.float32, nodata=np.nan)
        out_path = os.path.join(output_dir, f"CALIFORNIA_GOSIF_MASKED_{year_month}.tif")
        with rasterio.open(out_path, 'w', **gosif_meta) as dst:
            dst.write(masked_gosif.astype(np.float32), 1)

        print(f"Saved masked GOSIF for {year_month} to {out_path}")
