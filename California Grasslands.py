import os
import rasterio
import numpy as np
from glob import glob
from rasterio.warp import reproject, Resampling

# -------------------------------------------------------------------
# 1) Paths
# -------------------------------------------------------------------
datasets = {
    "GOSIF": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\GOSIF_California_Clipped",
    "LST":   r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\MODIS_CA_Monthly_LST",
    "NDVI":  r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\NDVI_MOD13Q1_5km_TIFs",
    "NIRv":  r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\NIRv_MOD13Q1_5km_TIFs",
    "SPEI":  r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\SPEI1_CSIC_TIFs",
    "TWS":   r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\TWS_GRACE_TIFs",
    "VPD":   r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA INDICES\VPDmax_GRIDMET_CA_TIFs"
}

mask_path = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Grasslands_2021.tif"
output_root = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS"

# -------------------------------------------------------------------
# 2) Function to apply mask
# -------------------------------------------------------------------
def apply_mask(input_path, mask_path, output_path):
    with rasterio.open(input_path) as src, rasterio.open(mask_path) as mask_src:
        data = src.read(1)
        meta = src.meta.copy()

        # Reproject mask to match raster grid
        mask_resampled = np.empty(data.shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=mask_resampled,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest
        )

        # Keep only grassland pixels (assumed 1 = grassland, 0/NaN = not grassland)
        masked = np.where(mask_resampled == 1, data, np.nan)

        # Update metadata
        meta.update(dtype="float32", nodata=np.nan, compress="LZW")

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(masked.astype(np.float32), 1)

# -------------------------------------------------------------------
# 3) Process all datasets
# -------------------------------------------------------------------
for name, folder in datasets.items():
    tif_files = sorted(glob(os.path.join(folder, "*.tif")))
    if not tif_files:
        print(f"⚠ No files found in {folder}")
        continue

    # Create sub-folder for this index
    out_dir = os.path.join(output_root, name)
    os.makedirs(out_dir, exist_ok=True)

    for f in tif_files:
        base = os.path.basename(f)
        out_name = f"{name}_Grasslands_{base}"
        out_path = os.path.join(out_dir, out_name)

        try:
            apply_mask(f, mask_path, out_path)
            print(f"✔ Saved: {out_path}")
        except Exception as e:
            print(f"❌ Failed: {base} – {e}")

print("✅ Finished masking all datasets to Grasslands only.")
