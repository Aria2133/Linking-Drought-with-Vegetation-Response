#!/usr/bin/env python
import os, glob, warnings
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping

warnings.filterwarnings("ignore", category=UserWarning)

# ───── USER SETTINGS ─────
KML_PATH = r"C:\Users\LOQ\OneDrive\Desktop\Tonzi.kml"
GOSIF_TEMPLATE = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\GOSIF_Tonzi_Clipped CORRECT\GOSIF_200101_Enhanced_Tonzi.tif"
INPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\GEE Exports\Rasters\TWS Monthly"
OUTPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Desktop\GRACE_TWS_Regridded_Tonzi"
FILE_PATTERN = "*.tif"
REGRID_METHOD = Resampling.bilinear
NODATA_VALUE = -9999
UPSAMPLE_FACTOR = 10  # Match GOSIF resolution
# ─────────────────────────

if not os.path.exists(GOSIF_TEMPLATE):
    raise FileNotFoundError(f"GOSIF template not found:\n{GOSIF_TEMPLATE}")

with rasterio.open(GOSIF_TEMPLATE) as gosif:
    tpl_transform = gosif.transform
    tpl_crs = gosif.crs
    tpl_width = gosif.width
    tpl_height = gosif.height
    tpl_dtype = gosif.dtypes[0]
    px_size_deg = tpl_transform.a

print("🔹 GOSIF grid reference loaded")
print(f"   Resolution: {px_size_deg:.6f}°")
print(f"   Grid: {tpl_width} × {tpl_height}")

gdf = gpd.read_file(KML_PATH, driver="KML").dissolve()
if gdf.crs is None:
    gdf.set_crs("EPSG:4326", inplace=True)
gdf = gdf.to_crs(tpl_crs)
tonzi_geom = [mapping(gdf.iloc[0].geometry)]

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

grace_files = sorted(glob.glob(os.path.join(INPUT_FOLDER, FILE_PATTERN)))
print(f"\n📂 Found {len(grace_files)} GRACE raster(s)\n")

succ = fail = 0
for idx, in_path in enumerate(grace_files, 1):
    base = os.path.basename(in_path)
    out_path = os.path.join(
        OUTPUT_FOLDER,
        base.replace(".tif", "_Upsampled_Tonzi.tif")
    )
    print(f"{idx}/{len(grace_files)}  {base}")

    try:
        with rasterio.open(in_path) as src:
            # Prepare destination array
            dst_array = np.full((1, tpl_height, tpl_width), NODATA_VALUE, dtype=tpl_dtype)

            reproject(
                source=src.read(),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=tpl_transform,
                dst_crs=tpl_crs,
                resampling=REGRID_METHOD,
                src_nodata=src.nodata,
                dst_nodata=NODATA_VALUE
            )

            mem_profile = {
                "driver": "GTiff",
                "height": tpl_height,
                "width": tpl_width,
                "count": 1,
                "dtype": tpl_dtype,
                "crs": tpl_crs,
                "transform": tpl_transform,
                "nodata": NODATA_VALUE
            }

            with rasterio.io.MemoryFile() as mem:
                with mem.open(**mem_profile) as tmp:
                    tmp.write(dst_array)
                    clipped, clipped_tr = mask(
                        tmp, tonzi_geom, crop=True,
                        nodata=NODATA_VALUE,
                        filled=True,
                        all_touched=True
                    )

            out_profile = mem_profile.copy()
            out_profile.update({
                "height": clipped.shape[1],
                "width": clipped.shape[2],
                "transform": clipped_tr,
                "compress": "lzw"
            })

            with rasterio.open(out_path, "w", **out_profile) as dst:
                dst.write(clipped)

            vals = clipped[clipped != NODATA_VALUE]
            print(f"   ✅ saved ({vals.size} px, min {vals.min():.2f}, max {vals.max():.2f})" if vals.size else "   ⚠️ all nodata")
            succ += 1

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        fail += 1

print("\n📊 SUMMARY")
print("=" * 50)
print(f"Processed : {succ + fail}")
print(f"Success   : {succ}")
print(f"Failed    : {fail}")
print(f"Output dir: {OUTPUT_FOLDER}")
print(f"\n🎯 Now GRACE data is upsampled to match GOSIF grid ({px_size_deg:.4f}°) and clipped to Tonzi Ranch.")
