import os
import glob
import rasterio
from rasterio.mask import mask
from rasterio.enums import Resampling
import geopandas as gpd
from shapely.geometry import mapping
import numpy as np
from rasterio.warp import transform_bounds, reproject, calculate_default_transform
from pyproj import Transformer
import warnings

warnings.filterwarnings('ignore')

# --------- Settings ---------
KML_PATH = r"C:\Users\LOQ\OneDrive\Desktop\Tonzi.kml"
INPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\Extracted GOSIF"
OUTPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Desktop\GOSIF_Tonzi_Clipped"
RASTER_PATTERN = "*.tif"
RASTER_BAND = 1
NODATA_VALUE = -9999

# NEW: Enhanced processing options
UPSAMPLE_FACTOR = 10  # Increase resolution by this factor
INTERPOLATION_METHOD = Resampling.bilinear  # or cubic for smoother results
BUFFER_DISTANCE = 0.01  # Buffer around polygon in degrees (~1km)


# --------- Helper Functions ---------
def calculate_polygon_area_km2(gdf):
    """Calculate polygon area in km²"""
    bounds = gdf.bounds.iloc[0]
    center_lon = (bounds['minx'] + bounds['maxx']) / 2
    center_lat = (bounds['miny'] + bounds['maxy']) / 2

    utm_zone = int((center_lon + 180) / 6) + 1

    if center_lat >= 0:
        utm_crs = f"EPSG:{32600 + utm_zone}"
    else:
        utm_crs = f"EPSG:{32700 + utm_zone}"

    try:
        gdf_proj = gdf.to_crs(utm_crs)
        area_m2 = gdf_proj.geometry.area.sum()
        area_km2 = area_m2 / 1e6
        return area_km2
    except:
        width_deg = bounds['maxx'] - bounds['minx']
        height_deg = bounds['maxy'] - bounds['miny']
        width_km = width_deg * 111.0 * np.cos(np.radians(center_lat))
        height_km = height_deg * 111.0
        return width_km * height_km


def degrees_to_km(degrees_lat, degrees_lon, latitude):
    """Convert degrees to kilometers at given latitude"""
    lat_km = degrees_lat * 111.0
    lon_km = degrees_lon * 111.0 * np.cos(np.radians(latitude))
    return lat_km, lon_km


def upsample_raster(src, factor=10, method=Resampling.bilinear):
    """Upsample raster to higher resolution"""
    # Calculate new dimensions
    new_width = src.width * factor
    new_height = src.height * factor

    # Calculate new transform
    transform = src.transform
    new_transform = rasterio.Affine(
        transform.a / factor,  # new pixel width
        transform.b,
        transform.c,
        transform.d,
        transform.e / factor,  # new pixel height
        transform.f
    )

    # Create output array
    upsampled = np.empty((src.count, new_height, new_width), dtype=src.dtypes[0])

    # Resample
    reproject(
        source=rasterio.band(src, list(range(1, src.count + 1))),
        destination=upsampled,
        src_transform=transform,
        src_crs=src.crs,
        dst_transform=new_transform,
        dst_crs=src.crs,
        resampling=method
    )

    return upsampled, new_transform


# --------- Load KML Polygon ---------
print("🔹 Loading Tonzi polygon...")
try:
    tonzi_gdf = gpd.read_file(KML_PATH, driver="KML")
    print(f"   Loaded {len(tonzi_gdf)} geometries")

    # Ensure WGS84
    if tonzi_gdf.crs is None:
        tonzi_gdf = tonzi_gdf.set_crs("EPSG:4326")
    elif tonzi_gdf.crs != "EPSG:4326":
        tonzi_gdf = tonzi_gdf.to_crs("EPSG:4326")

    # Dissolve if multiple geometries
    if len(tonzi_gdf) > 1:
        tonzi_gdf = tonzi_gdf.dissolve()

    # Add buffer to capture surrounding pixels
    tonzi_buffered = tonzi_gdf.copy()
    tonzi_buffered['geometry'] = tonzi_buffered.geometry.buffer(BUFFER_DISTANCE)

    # Calculate area and dimensions
    polygon_area_km2 = calculate_polygon_area_km2(tonzi_gdf)
    bounds = tonzi_gdf.bounds.iloc[0]
    buffered_bounds = tonzi_buffered.bounds.iloc[0]

    center_lat = (bounds['miny'] + bounds['maxy']) / 2
    width_deg = bounds['maxx'] - bounds['minx']
    height_deg = bounds['maxy'] - bounds['miny']
    width_km, height_km = degrees_to_km(height_deg, width_deg, center_lat)

    print(f"   📐 Polygon area: {polygon_area_km2:.2f} km²")
    print(f"   📏 Polygon dimensions: {width_km:.2f} km × {height_km:.2f} km")
    print(
        f"   🌍 Original bounds: {bounds['minx']:.6f}, {bounds['miny']:.6f}, {bounds['maxx']:.6f}, {bounds['maxy']:.6f}")
    print(
        f"   🔄 Buffered bounds: {buffered_bounds['minx']:.6f}, {buffered_bounds['miny']:.6f}, {buffered_bounds['maxx']:.6f}, {buffered_bounds['maxy']:.6f}")

    # GOSIF pixel size analysis
    gosif_pixel_size_deg = 0.05
    gosif_pixel_km, _ = degrees_to_km(gosif_pixel_size_deg, gosif_pixel_size_deg, center_lat)

    print(f"\n📊 GOSIF Analysis:")
    print(f"   📏 Original GOSIF pixel size: {gosif_pixel_size_deg}° ≈ {gosif_pixel_km:.2f} km")
    print(
        f"   🔍 After {UPSAMPLE_FACTOR}x upsampling: {gosif_pixel_size_deg / UPSAMPLE_FACTOR:.4f}° ≈ {gosif_pixel_km / UPSAMPLE_FACTOR:.2f} km")
    print(f"   🔢 Expected pixels covering polygon (upsampled):")
    print(f"      - Width: {width_km / (gosif_pixel_km / UPSAMPLE_FACTOR):.1f} pixels")
    print(f"      - Height: {height_km / (gosif_pixel_km / UPSAMPLE_FACTOR):.1f} pixels")

    # Prepare geometries for clipping
    tonzi_geom = [mapping(geom) for geom in tonzi_gdf.geometry]
    tonzi_buffered_geom = [mapping(geom) for geom in tonzi_buffered.geometry]

except Exception as e:
    print(f"❌ Error loading KML: {e}")
    exit(1)

# --------- Create output folder ---------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --------- Enhanced Clip Function ---------
def clip_raster_enhanced(input_path, output_path):
    """Enhanced clipping with upsampling and proper nodata handling"""
    try:
        with rasterio.open(input_path) as src:
            print(f"      📁 Input file: {os.path.basename(input_path)}")
            print(f"      🗺️  Raster CRS: {src.crs}")
            print(f"      📏 Original shape: {src.shape}")
            print(f"      🎯 Original bounds: {src.bounds}")

            # Read original data
            original_data = src.read(1)
            original_nodata = src.nodata if src.nodata is not None else -9999

            # Check for actual data values
            if original_nodata is not None:
                valid_mask = original_data != original_nodata
                if np.sum(valid_mask) == 0:
                    print(f"      ⚠️  No valid data found in original raster!")
                    return False
                valid_data = original_data[valid_mask]
                print(f"      📊 Original data range: {valid_data.min():.6f} to {valid_data.max():.6f}")
            else:
                valid_data = original_data.flatten()
                print(f"      📊 Original data range: {valid_data.min():.6f} to {valid_data.max():.6f}")

            # Step 1: Clip to buffered area first
            print(f"      🔄 Step 1: Clipping to buffered area...")
            geom_for_clip = tonzi_buffered_geom
            if src.crs != tonzi_gdf.crs:
                geom_reproj = tonzi_buffered.to_crs(src.crs)
                geom_for_clip = [mapping(geom) for geom in geom_reproj.geometry]

            # Clip with buffered geometry
            clipped_data, clipped_transform = mask(
                src,
                geom_for_clip,
                crop=True,
                nodata=original_nodata,
                filled=True,
                all_touched=True,
                invert=False
            )

            print(f"      📏 Clipped shape: {clipped_data.shape}")

            # Step 2: Upsample the clipped data
            print(f"      🔍 Step 2: Upsampling by factor {UPSAMPLE_FACTOR}...")

            # Create temporary raster for upsampling
            temp_meta = src.meta.copy()
            temp_meta.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': clipped_transform,
                'nodata': original_nodata
            })

            # Create in-memory dataset
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**temp_meta) as temp_src:
                    temp_src.write(clipped_data)

                    # Upsample
                    upsampled_data, upsampled_transform = upsample_raster(
                        temp_src,
                        factor=UPSAMPLE_FACTOR,
                        method=INTERPOLATION_METHOD
                    )

            print(f"      📏 Upsampled shape: {upsampled_data.shape}")

            # Step 3: Final clip with original polygon
            print(f"      ✂️  Step 3: Final clipping to study area...")

            # Create metadata for upsampled raster
            upsampled_meta = temp_meta.copy()
            upsampled_meta.update({
                'height': upsampled_data.shape[1],
                'width': upsampled_data.shape[2],
                'transform': upsampled_transform,
                'nodata': NODATA_VALUE  # Use our custom nodata value
            })

            # Final clip with original polygon
            with rasterio.io.MemoryFile() as memfile:
                with memfile.open(**upsampled_meta) as upsampled_src:
                    upsampled_src.write(upsampled_data)

                    # Prepare geometry for final clip
                    final_geom = tonzi_geom
                    if upsampled_src.crs != tonzi_gdf.crs:
                        geom_reproj = tonzi_gdf.to_crs(upsampled_src.crs)
                        final_geom = [mapping(geom) for geom in geom_reproj.geometry]

                    # Final mask
                    final_data, final_transform = mask(
                        upsampled_src,
                        final_geom,
                        crop=True,
                        nodata=NODATA_VALUE,
                        filled=True,
                        all_touched=True,
                        invert=False
                    )

            print(f"      📏 Final shape: {final_data.shape}")

            # Analyze final results
            final_array = final_data[0]
            valid_mask = final_array != NODATA_VALUE
            valid_pixels = np.sum(valid_mask)
            total_pixels = final_array.size

            print(f"      📊 Final results:")
            print(f"         - Valid pixels: {valid_pixels}/{total_pixels} ({100 * valid_pixels / total_pixels:.1f}%)")

            if valid_pixels > 0:
                valid_values = final_array[valid_mask]
                unique_values = np.unique(valid_values)
                print(f"         - Unique values: {len(unique_values)}")
                print(f"         - Data range: {valid_values.min():.6f} to {valid_values.max():.6f}")
                print(f"         - Mean value: {valid_values.mean():.6f}")

                # Show some unique values
                if len(unique_values) <= 10:
                    print(f"         - All values: {unique_values}")
                else:
                    print(f"         - Sample values: {unique_values[:10]}")
            else:
                print(f"         - ❌ No valid pixels found after final clipping!")
                return False

            # Write output with proper metadata
            out_meta = {
                "driver": "GTiff",
                "height": final_data.shape[1],
                "width": final_data.shape[2],
                "count": 1,
                "dtype": final_data.dtype,
                "crs": src.crs,
                "transform": final_transform,
                "nodata": NODATA_VALUE,
                "compress": "lzw"
            }

            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(final_data)

            print(f"      ✅ Successfully created enhanced output!")
            return True

    except Exception as e:
        print(f"      ❌ Error in enhanced clipping: {e}")
        import traceback
        traceback.print_exc()
        return False


# --------- Process Files ---------
input_files = glob.glob(os.path.join(INPUT_FOLDER, RASTER_PATTERN))
print(f"\n🚀 Found {len(input_files)} raster files")

if len(input_files) == 0:
    print("❌ No raster files found!")
    exit(1)

# Process first file as test
print(f"\n🧪 Testing enhanced processing with: {os.path.basename(input_files[0])}")
test_output = os.path.join(OUTPUT_FOLDER, f"ENHANCED_{os.path.basename(input_files[0])}")

if clip_raster_enhanced(input_files[0], test_output):
    print(f"\n✅ Enhanced test file created: {test_output}")
    print(f"\n💡 IMPROVEMENTS MADE:")
    print(
        f"   🔍 Upsampled resolution by {UPSAMPLE_FACTOR}x (from ~{gosif_pixel_km:.1f}km to ~{gosif_pixel_km / UPSAMPLE_FACTOR:.1f}km pixels)")
    print(f"   🎯 Added buffer to capture surrounding context")
    print(f"   🔄 Used {INTERPOLATION_METHOD.name} interpolation for smoother results")
    print(f"   ✂️  Applied precise clipping to study area")
    print(f"   📊 Proper nodata handling ({NODATA_VALUE})")
    print(f"\n   This should now show more detail and smoother transitions!")

    # Ask if user wants to continue
    response = input(f"\n❓ Process all {len(input_files)} files with enhanced method? (y/n): ").strip().lower()
    if response == 'y':
        successful = 0
        for i, in_file in enumerate(input_files, 1):
            base = os.path.basename(in_file)
            out_file = os.path.join(OUTPUT_FOLDER, base.replace(".tif", "_Enhanced_Tonzi.tif"))
            print(f"\n{i}/{len(input_files)} Processing: {base}")
            if clip_raster_enhanced(in_file, out_file):
                successful += 1
        print(f"\n🎉 Completed! {successful}/{len(input_files)} files processed with enhanced method")
else:
    print("\n❌ Enhanced processing failed!")

print(f"\n📋 SUMMARY:")
print(f"   The enhanced method addresses the black rectangle issue by:")
print(f"   1. Upsampling coarse GOSIF pixels to higher resolution")
print(f"   2. Using interpolation to create smoother transitions")
print(f"   3. Proper buffering to capture surrounding context")
print(f"   4. Correct nodata value handling")
print(f"   5. Multi-step clipping for better results")
print(f"\n   Your output should now look much more like the satellite imagery!")