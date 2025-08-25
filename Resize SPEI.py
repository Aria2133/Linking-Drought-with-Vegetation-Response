import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os

# Template file path (GOSIF or GRACE raster with 0.005°)
template_path = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\GOSIF_Tonzi_Clipped CORRECT\GOSIF_200104_Enhanced_Tonzi.tif"
# SPEI file you want to fix
spei_path = r"C:\Users\LOQ\OneDrive\Desktop\SPEI_2001_01_Corrected\SPEI_2001_01_cleaned.tif"
# Output
output_path = r"C:\Users\LOQ\OneDrive\Desktop\RESAMPLED_SPEI.tif"

# Open template to get shape and transform
with rasterio.open(template_path) as ref:
    template_meta = ref.meta.copy()
    template_data = ref.read(1)
    dst_shape = template_data.shape
    dst_transform = ref.transform
    dst_crs = ref.crs

# Open SPEI
with rasterio.open(spei_path) as src:
    src_data = src.read(1)
    src_transform = src.transform
    src_crs = src.crs

    # Prepare output array
    dst_array = np.full(dst_shape, -9999, dtype=np.float32)

    reproject(
        source=src_data,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

# Save the aligned SPEI
template_meta.update(dtype=rasterio.float32, nodata=-9999)
with rasterio.open(output_path, "w", **template_meta) as dst:
    dst.write(dst_array, 1)
