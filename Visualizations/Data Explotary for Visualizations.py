import rasterio
from rasterio.transform import array_bounds
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from collections import defaultdict

# Base directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

# California approximate bounds
CA_BOUNDS = {
    'lon_min': -124.5,
    'lon_max': -114.1,
    'lat_min': 32.5,
    'lat_max': 42.1
}


def check_spatial_properties(index, n_samples=3):
    """Comprehensive spatial analysis"""
    print("\n" + "=" * 80)
    print(f"SPATIAL ANALYSIS: {index}")
    print("=" * 80)

    # Get sample files
    path = os.path.join(train_base, "CALIFORNIA GRASSLANDS", index)
    files = glob.glob(os.path.join(path, "*.tif"))[:n_samples]

    if not files:
        print(f"No files found")
        return

    spatial_stats = []

    for i, filepath in enumerate(files, 1):
        print(f"\n--- File {i}: {os.path.basename(filepath)} ---")

        try:
            with rasterio.open(filepath) as src:
                # Basic info
                bounds = src.bounds
                print(f"Shape: {src.shape} (height x width)")
                print(f"Resolution: {src.res} (x_res, y_res)")
                print(f"CRS: {src.crs}")
                print(f"Transform: {src.transform}")

                # Bounds
                print(f"\nBounds:")
                print(f"  Longitude: [{bounds.left:.4f}, {bounds.right:.4f}]")
                print(f"  Latitude:  [{bounds.bottom:.4f}, {bounds.top:.4f}]")

                # Check if covers California
                covers_ca = (bounds.left <= CA_BOUNDS['lon_min'] and
                             bounds.right >= CA_BOUNDS['lon_max'] and
                             bounds.bottom <= CA_BOUNDS['lat_min'] and
                             bounds.top >= CA_BOUNDS['lat_max'])

                is_california_only = (abs(bounds.left - CA_BOUNDS['lon_min']) < 1 and
                                      abs(bounds.right - CA_BOUNDS['lon_max']) < 1 and
                                      abs(bounds.bottom - CA_BOUNDS['lat_min']) < 1 and
                                      abs(bounds.top - CA_BOUNDS['lat_max']) < 1)

                if is_california_only:
                    print(f"  ✓ CALIFORNIA-SPECIFIC extent")
                elif covers_ca:
                    print(f"  ⚠ COVERS California but extends beyond")
                else:
                    print(f"  ✗ DOES NOT COVER California properly!")

                # Calculate pixel size
                width_deg = bounds.right - bounds.left
                height_deg = bounds.top - bounds.bottom
                pixel_size_lon = width_deg / src.width
                pixel_size_lat = height_deg / src.height

                # Convert to km (approximate)
                km_per_deg_lon = 111.0 * np.cos(np.radians((bounds.top + bounds.bottom) / 2))
                km_per_deg_lat = 111.0
                pixel_size_km_lon = pixel_size_lon * km_per_deg_lon
                pixel_size_km_lat = pixel_size_lat * km_per_deg_lat

                print(f"\nPixel size:")
                print(f"  Degrees: {pixel_size_lon:.6f}° x {pixel_size_lat:.6f}°")
                print(f"  Kilometers: {pixel_size_km_lon:.2f} km x {pixel_size_km_lat:.2f} km")
                print(f"  Approximate resolution: ~{(pixel_size_km_lon + pixel_size_km_lat) / 2:.1f} km")

                # Read data and check valid pixels
                data = src.read(1)
                total_pixels = data.size

                # Check for nodata
                nodata_count = 0
                if src.nodata is not None:
                    nodata_count = np.sum(data == src.nodata)
                nan_count = np.sum(np.isnan(data))

                valid_pixels = total_pixels - nodata_count - nan_count

                print(f"\nData coverage:")
                print(f"  Total pixels: {total_pixels:,}")
                print(f"  Valid pixels: {valid_pixels:,} ({valid_pixels / total_pixels * 100:.1f}%)")
                print(
                    f"  NoData/NaN:   {nodata_count + nan_count:,} ({(nodata_count + nan_count) / total_pixels * 100:.1f}%)")

                # Value ranges (excluding nodata)
                if valid_pixels > 0:
                    valid_data = data[~np.isnan(data)]
                    if src.nodata is not None:
                        valid_data = valid_data[valid_data != src.nodata]

                    if len(valid_data) > 0:
                        print(f"\nValue statistics (valid pixels only):")
                        print(f"  Min:    {np.min(valid_data):.4f}")
                        print(f"  Max:    {np.max(valid_data):.4f}")
                        print(f"  Mean:   {np.mean(valid_data):.4f}")
                        print(f"  Median: {np.median(valid_data):.4f}")
                        print(f"  Std:    {np.std(valid_data):.4f}")

                # Store for comparison
                spatial_stats.append({
                    'shape': src.shape,
                    'bounds': bounds,
                    'resolution_km': (pixel_size_km_lon + pixel_size_km_lat) / 2,
                    'valid_pct': valid_pixels / total_pixels * 100,
                    'ca_specific': is_california_only
                })

        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Summary comparison
    if len(spatial_stats) > 1:
        print(f"\n{'=' * 80}")
        print("CONSISTENCY CHECK:")
        print(f"{'=' * 80}")

        shapes = [s['shape'] for s in spatial_stats]
        resolutions = [s['resolution_km'] for s in spatial_stats]

        if len(set(shapes)) == 1:
            print("✓ All files have SAME shape")
        else:
            print(f"✗ DIFFERENT shapes detected: {set(shapes)}")

        if max(resolutions) - min(resolutions) < 0.1:
            print(f"✓ Consistent resolution (~{np.mean(resolutions):.1f} km)")
        else:
            print(f"✗ VARYING resolutions: {min(resolutions):.1f} to {max(resolutions):.1f} km")


# Run comprehensive analysis
print("\n" + "=" * 80)
print("COMPREHENSIVE SPATIAL DATA ANALYSIS")
print("=" * 80)
print(f"\nCalifornia Reference Bounds:")
print(f"  Longitude: [{CA_BOUNDS['lon_min']}, {CA_BOUNDS['lon_max']}]")
print(f"  Latitude:  [{CA_BOUNDS['lat_min']}, {CA_BOUNDS['lat_max']}]")

for index in indices:
    check_spatial_properties(index, n_samples=2)

print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)
print("""
Based on spatial analysis:

1. RESOLUTION ISSUES:
   - TWS is very coarse (11x12 pixels) - may need exclusion or special handling
   - Most indices are ~5km resolution, but some differ

2. EXTENT ISSUES:
   - GPP appears to be GLOBAL extent - needs clipping to California
   - Check if all indices properly cover California region

3. RESAMPLING STRATEGY:
   - Use highest resolution as reference (likely 5km indices)
   - Exclude or separately handle very coarse data (TWS)
   - Clip global data (GPP) to California bounds before processing

4. DATA QUALITY:
   - Some indices have <15% valid pixels in California region
   - May need to aggregate only land-type-specific pixels rather than full grid
""")