import os
import glob
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import numpy as np
import warnings

# --- User Inputs ---
SHAPEFILE_PATH = r"C:\Users\LOQ\OneDrive\Desktop\SHAPEFILE OF CA\CA_State.shp"
INPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\Extracted GOSIF"
OUTPUT_FOLDER = r"C:\Users\LOQ\OneDrive\Desktop\GOSIF_California_Clipped"
RASTER_PATTERN = "*.tif"
NODATA_VALUE = -9999
BUFFER_DISTANCE = 0  # Set to 0 for exact California boundary, or 5000 for 5km buffer


def load_california_shapefile():
    """Load and validate the California shapefile"""
    print(f"🔹 Loading California shapefile from: {SHAPEFILE_PATH}")

    # Check if shapefile exists
    if not os.path.exists(SHAPEFILE_PATH):
        print(f"❌ Shapefile not found at: {SHAPEFILE_PATH}")
        return None

    try:
        # Load the shapefile
        gdf = gpd.read_file(SHAPEFILE_PATH)

        print(f"✅ Successfully loaded shapefile")
        print(f"   📊 Number of features: {len(gdf)}")
        print(f"   🗺️  Original CRS: {gdf.crs}")
        print(f"   📐 Bounds: {gdf.total_bounds}")

        # Print column information
        print(f"   📋 Columns: {list(gdf.columns)}")

        # Print first few rows of attribute data (excluding geometry)
        if len(gdf.columns) > 1:
            print("   📄 Sample data:")
            for col in gdf.columns:
                if col != 'geometry':
                    print(f"      {col}: {gdf[col].tolist()}")

        # Ensure we have valid geometries
        if gdf.geometry.isna().all():
            print("❌ No valid geometries found in shapefile")
            return None

        # Check for invalid geometries
        invalid_geoms = ~gdf.geometry.is_valid
        if invalid_geoms.any():
            print(f"⚠️  Found {invalid_geoms.sum()} invalid geometries, attempting to fix...")
            gdf.loc[invalid_geoms, 'geometry'] = gdf.loc[invalid_geoms, 'geometry'].buffer(0)

        return gdf

    except Exception as e:
        print(f"❌ Error loading shapefile: {e}")
        print(f"   This could be due to:")
        print(f"   - Corrupted shapefile")
        print(f"   - Missing .shx, .dbf, or .prj files")
        print(f"   - Unsupported projection")
        print(f"   - File permission issues")
        return None


def apply_buffer_properly(gdf, buffer_meters):
    """Apply buffer in a projected CRS for accurate distance calculation"""
    if buffer_meters <= 0:
        return gdf

    original_crs = gdf.crs

    # Check if already in a projected CRS
    if original_crs.is_geographic:
        # Use appropriate UTM zone for California (UTM Zone 10N or 11N)
        # California spans zones 10N and 11N, using 10N as it covers most of the state
        projected_crs = "EPSG:32610"  # UTM Zone 10N
        print(f"   🔄 Temporarily converting to {projected_crs} for accurate buffering...")

        # Reproject to projected CRS
        gdf_proj = gdf.to_crs(projected_crs)

        # Apply buffer in meters
        gdf_buffered = gdf_proj.copy()
        gdf_buffered['geometry'] = gdf_proj.geometry.buffer(buffer_meters)

        # Convert back to original CRS
        gdf_buffered = gdf_buffered.to_crs(original_crs)

        print(f"   📏 Applied {buffer_meters}m buffer using projected coordinates")
        return gdf_buffered
    else:
        # Already in projected CRS
        print(f"   📏 Applying {buffer_meters}m buffer (already in projected CRS)")
        gdf_buffered = gdf.copy()
        gdf_buffered['geometry'] = gdf.geometry.buffer(buffer_meters)
        return gdf_buffered


def clip_raster_to_california(input_path, output_path, california_gdf):
    """Clip a single raster file to California boundaries"""
    try:
        with rasterio.open(input_path) as src:
            print(f"📂 Processing: {os.path.basename(input_path)}")
            print(f"   🗺️  Raster CRS: {src.crs}")
            print(f"   📐 Raster bounds: {src.bounds}")
            print(f"   📏 Raster size: {src.width} x {src.height}")

            # Check if raster has any data
            if src.width == 0 or src.height == 0:
                print(f"⚠️  Skipping empty raster")
                return False

            # Reproject California geometry to match raster CRS
            if california_gdf.crs != src.crs:
                print(f"   🔄 Reprojecting California boundary from {california_gdf.crs} to {src.crs}")
                gdf_proj = california_gdf.to_crs(src.crs)
            else:
                print(f"   ✅ CRS already matches")
                gdf_proj = california_gdf.copy()

            # Apply buffer if specified (now using proper projection)
            if BUFFER_DISTANCE > 0:
                gdf_proj_buffered = apply_buffer_properly(gdf_proj, BUFFER_DISTANCE)
                geoms = [mapping(geom) for geom in gdf_proj_buffered.geometry]
            else:
                geoms = [mapping(geom) for geom in gdf_proj.geometry]

            print(f"   📐 California bounds in raster CRS: {gdf_proj.total_bounds}")

            # Check if geometries intersect with raster bounds
            raster_bounds = src.bounds
            california_bounds = gdf_proj.total_bounds

            # Check for overlap
            if (california_bounds[2] < raster_bounds[0] or  # California max_x < raster min_x
                    california_bounds[0] > raster_bounds[2] or  # California min_x > raster max_x
                    california_bounds[3] < raster_bounds[1] or  # California max_y < raster min_y
                    california_bounds[1] > raster_bounds[3]):  # California min_y > raster max_y
                print(f"⚠️  No spatial overlap between raster and California boundary")
                return False

            print(f"   ✅ Spatial overlap detected, proceeding with clipping...")

            # Suppress specific warnings during clipping
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS.*")

                # Clip the raster and properly mask areas outside the polygon
                # Using all_touched=False to be more precise with polygon boundaries
                clipped_data, clipped_transform = mask(
                    src,
                    geoms,
                    crop=True,
                    nodata=NODATA_VALUE,
                    filled=True,
                    invert=False,
                    all_touched=False
                )

            # Additional step: ensure areas outside polygon are set to nodata
            # The mask function should handle this, but let's be explicit
            print(f"   📊 Clipped data shape: {clipped_data.shape}")
            print(f"   📊 Data range: {clipped_data.min()} to {clipped_data.max()}")

            # Create a temporary raster to verify masking worked correctly
            temp_profile = src.profile.copy()
            temp_profile.update({
                'height': clipped_data.shape[1],
                'width': clipped_data.shape[2],
                'transform': clipped_transform
            })

            # Double-check: create another mask to ensure areas outside polygon are nodata
            from rasterio.features import geometry_mask
            mask_array = geometry_mask(
                geoms,
                transform=clipped_transform,
                invert=True,  # True means inside polygon = True
                out_shape=(clipped_data.shape[1], clipped_data.shape[2])
            )

            # Apply the mask: set areas outside polygon to nodata
            for band in range(clipped_data.shape[0]):
                clipped_data[band][~mask_array] = NODATA_VALUE

            # Check if clipped data has any valid values
            if clipped_data.size == 0:
                print(f"⚠️  No data after clipping")
                return False

            # Count valid pixels
            if src.nodata is not None:
                valid_pixels = np.sum(clipped_data != src.nodata)
                clipped_data[clipped_data == src.nodata] = NODATA_VALUE
            else:
                valid_pixels = np.sum(clipped_data != NODATA_VALUE)

            print(f"   📊 Valid pixels after clipping: {valid_pixels}")

            if valid_pixels == 0:
                print(f"⚠️  No valid pixels after clipping")
                return False

            # Update metadata
            out_meta = src.meta.copy()
            out_meta.update({
                "height": clipped_data.shape[1],
                "width": clipped_data.shape[2],
                "transform": clipped_transform,
                "nodata": NODATA_VALUE
            })

            # Save clipped raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(clipped_data)

            print(f"✅ Successfully saved: {os.path.basename(output_path)}")
            return True

    except Exception as e:
        print(f"❌ Error processing {os.path.basename(input_path)}: {e}")
        import traceback
        print(f"   Detailed error: {traceback.format_exc()}")
        return False


def main():
    """Main processing function"""

    print("🚀 Starting GOSIF California clipping process...")
    print(f"📁 Input folder: {INPUT_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"🗺️  Shapefile: {SHAPEFILE_PATH}")
    print(f"📏 Buffer distance: {BUFFER_DISTANCE}m")

    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"✅ Output folder ready")

    # Load California boundary
    california_gdf = load_california_shapefile()
    if california_gdf is None:
        print("❌ Cannot proceed without California boundary data")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if all shapefile components exist (.shp, .shx, .dbf, .prj)")
        print("   2. Try opening the shapefile in QGIS or ArcGIS to verify it's valid")
        print("   3. Check file permissions")
        print("   4. Ensure the shapefile path is correct")
        return

    # Find all raster files
    print(f"\n🔍 Searching for raster files...")
    raster_files = glob.glob(os.path.join(INPUT_FOLDER, RASTER_PATTERN))
    print(f"Found {len(raster_files)} raster files")

    if len(raster_files) == 0:
        print(f"❌ No .tif files found in {INPUT_FOLDER}")
        print("   Please check the input folder path and ensure it contains .tif files")
        return

    # Show first few filenames
    print("   Sample files:")
    for i, file in enumerate(raster_files[:5]):
        print(f"      {i + 1}. {os.path.basename(file)}")
    if len(raster_files) > 5:
        print(f"      ... and {len(raster_files) - 5} more files")

    # Process each raster file
    print(f"\n📊 Starting processing...")
    successful = 0
    failed = 0

    for i, raster_path in enumerate(raster_files, 1):
        filename = os.path.basename(raster_path)
        output_path = os.path.join(OUTPUT_FOLDER, f"CALIFORNIA_{filename}")

        print(f"\n{'=' * 60}")
        print(f"Processing {i}/{len(raster_files)}: {filename}")
        print(f"{'=' * 60}")

        if clip_raster_to_california(raster_path, output_path, california_gdf):
            successful += 1
        else:
            failed += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"🎯 PROCESSING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"✅ Successfully processed: {successful} files")
    print(f"❌ Failed: {failed} files")
    print(f"📁 All clipped files saved to: {OUTPUT_FOLDER}")

    if failed > 0:
        print(f"\n⚠️  Some files failed to process. Common reasons:")
        print(f"   - Raster doesn't overlap with California")
        print(f"   - Corrupted raster files")
        print(f"   - Projection issues")
        print(f"   - No valid data in the clipped area")


if __name__ == "__main__":
    main()