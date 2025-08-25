import rasterio
import numpy as np
import os

# ───── Set your input/output folders ─────
input_dir = r"C:\Users\LOQ\OneDrive\Desktop\SPEI Tif NEEDS NaN VALUES REMOVED"  # folder with 252 original SPEI files
output_dir = r"C:\Users\LOQ\OneDrive\Desktop\SPEI_tif_Cleaned"  # where cleaned files go

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# ───── Loop through all .tif files ─────
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename.replace(".tif", "_cleaned.tif"))

        with rasterio.open(input_path) as src:
            data = src.read(1)
            profile = src.profile

            # Detect NaNs or NoData values
            if src.nodata is not None:
                nodata = src.nodata
                mask_array = (data == nodata)
            else:
                mask_array = np.isnan(data)

            # Replace bad pixels with 0
            cleaned_data = np.where(mask_array, 0, data)

            # Update metadata
            profile.update(dtype=rasterio.float32, nodata=0)

            # Save cleaned raster
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(cleaned_data.astype(rasterio.float32), 1)

        print(f"✅ Cleaned: {filename}")

print("🎉 All SPEI files cleaned and saved!")
