import os
import rasterio
import numpy as np
from glob import glob
from rasterio.warp import reproject, Resampling

# -------------------------------------------------------------------
# 1) Paths
# -------------------------------------------------------------------
gndvi_folder = r"C:\Users\LOQ\OneDrive\Desktop\GNDVI_MOD09A1_5km_TIFs"
land_masks = {
    "Grasslands":     r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Grasslands_2021.tif",
    "NaturalForests": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
    "Savannas":       r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Savannas_2021.tif",
    "Shrublands":     r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Shrublands_2021.tif"
}

output_root = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\GNDVI_LandTypes"

# -------------------------------------------------------------------
# 2) Function to apply mask
# -------------------------------------------------------------------
def apply_mask(input_path, mask_path, output_path):
    with rasterio.open(input_path) as src, rasterio.open(mask_path) as mask_src:
        data = src.read(1)
        meta = src.meta.copy()

        # Reproject mask to match GNDVI raster grid
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

        # Mask out non-target pixels (keep only value==1 in mask)
        masked = np.where(mask_resampled == 1, data, np.nan)

        # Update metadata
        meta.update(dtype="float32", nodata=np.nan, compress="LZW")

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(masked.astype(np.float32), 1)

# -------------------------------------------------------------------
# 3) Process all GNDVI files for each land type
# -------------------------------------------------------------------
gndvi_files = sorted(glob(os.path.join(gndvi_folder, "*.tif")))

if not gndvi_files:
    print(f"⚠ No GNDVI files found in {gndvi_folder}")
else:
    for land_type, mask_path in land_masks.items():
        out_dir = os.path.join(output_root, land_type)
        os.makedirs(out_dir, exist_ok=True)

        for f in gndvi_files:
            base = os.path.basename(f)
            out_name = f"GNDVI_{land_type}_{base}"
            out_path = os.path.join(out_dir, out_name)

            try:
                apply_mask(f, mask_path, out_path)
                print(f"✔ {land_type}: {out_name}")
            except Exception as e:
                print(f"❌ Failed {land_type}, {base} – {e}")

print("✅ Finished masking GNDVI for all land types.")
