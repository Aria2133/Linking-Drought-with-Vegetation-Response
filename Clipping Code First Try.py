#!/usr/bin/env python
# -------------------------------------------------------------
#  Clip GOSIF to Tonzi Ranch ‑‑ preserve original values
# -------------------------------------------------------------
import os, glob, warnings
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
from rasterio.warp import reproject
import geopandas as gpd
from shapely.geometry import mapping
warnings.filterwarnings("ignore")

# ────────── SETTINGS ─────────────────────────────────────────
KML_PATH      = r"C:\Users\LOQ\OneDrive\Desktop\Tonzi.kml"
INPUT_FOLDER  = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\Extracted GOSIF"
OUTPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Desktop\GOSIF_Tonzi_Clipped"
RASTER_PATTERN = "*.tif"
NODATA_VALUE   = -9999

UPSAMPLE_FACTOR       = 10                     # 0.05° → 0.005°
INTERPOLATION_METHOD  = Resampling.nearest     # <-- key change
BUFFER_DISTANCE       = 0.01                  # ~1 km buffer
# ─────────────────────────────────────────────────────────────

# ---------- load Tonzi boundary (as before) -----------------
gdf = gpd.read_file(KML_PATH, driver="KML").dissolve()
gdf = gdf.to_crs("EPSG:4326")
gdf_buf  = gdf.copy()
gdf_buf["geometry"] = gdf_buf.geometry.buffer(BUFFER_DISTANCE)
tonzi_geom         = [mapping(g) for g in gdf.geometry]
tonzi_buf_geom     = [mapping(g) for g in gdf_buf.geometry]

# ---------- helper to up‑sample with NN ---------------------
def upsample_raster(src, factor=10, method=Resampling.nearest):
    new_w, new_h = src.width*factor, src.height*factor
    tr = src.transform
    new_tr = rasterio.Affine(tr.a/factor, tr.b, tr.c,
                             tr.d,        tr.e/factor, tr.f)
    arr = np.empty((src.count, new_h, new_w), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, list(range(1, src.count+1))),
        destination=arr,
        src_transform=tr,      src_crs=src.crs,
        dst_transform=new_tr,  dst_crs=src.crs,
        resampling=method,
        src_nodata=src.nodata, dst_nodata=NODATA_VALUE
    )
    return arr, new_tr

# ---------- main clip routine (identical except NN) ---------
def clip_one(in_path, out_path):
    with rasterio.open(in_path) as src:
        # 1) clip to buffered extent
        buf, buf_tr = mask(src, tonzi_buf_geom, crop=True,
                           nodata=NODATA_VALUE, filled=True)
        # 2) up‑sample 10× with **nearest**
        buf_meta = src.meta.copy()
        buf_meta.update(height=buf.shape[1], width=buf.shape[2],
                        transform=buf_tr, nodata=NODATA_VALUE)
        with rasterio.io.MemoryFile() as m:
            with m.open(**buf_meta) as tmp:
                tmp.write(buf)
                up, up_tr = upsample_raster(tmp,
                                            factor=UPSAMPLE_FACTOR,
                                            method=INTERPOLATION_METHOD)
        # 3) final clip to exact Tonzi polygon
        up_meta = buf_meta.copy()
        up_meta.update(height=up.shape[1], width=up.shape[2],
                       transform=up_tr)
        with rasterio.io.MemoryFile() as m:
            with m.open(**up_meta) as tmp:
                tmp.write(up)
                fin, fin_tr = mask(tmp, tonzi_geom, crop=True,
                                   nodata=NODATA_VALUE, filled=True)
        # 4) write out
        out_meta = src.meta.copy()
        out_meta.update(driver="GTiff", height=fin.shape[1],
                        width=fin.shape[2], transform=fin_tr,
                        nodata=NODATA_VALUE, compress="lzw")
        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(fin)

# ---------- batch -------------------------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
files = sorted(glob.glob(os.path.join(INPUT_FOLDER, RASTER_PATTERN)))
print(f"Found {len(files)} rasters")

for i, f in enumerate(files, 1):
    out = os.path.join(OUTPUT_FOLDER,
                       os.path.basename(f).replace(".tif",
                                                   "_Tonzi_NN.tif"))
    print(f"{i}/{len(files)} {os.path.basename(f)} → {os.path.basename(out)}")
    clip_one(f, out)

print("✅ Finished. 0.05° values copied 1‑for‑1 into 0.005° grid (nearest‑neighbour).")
