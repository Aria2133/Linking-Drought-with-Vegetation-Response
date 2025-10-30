import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from rasterio.mask import mask
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import glob
from shapely.geometry import box

# Define directories and indices
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

# Base directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

# California bounds for clipping
CA_BOUNDS = {
    'west': -124.5,
    'east': -114.1,
    'south': 32.5,
    'north': 42.1
}

# Create custom red-green colormap
colors_rg = ['#8B0000', '#CD5C5C', '#F5DEB3', '#90EE90', '#228B22']
n_bins = 100
cmap_rg = LinearSegmentedColormap.from_list('red_green', colors_rg, N=n_bins)

# For diverging indices like SPEI
colors_div = ['#8B0000', '#CD5C5C', '#FFFFFF', '#90EE90', '#228B22']
cmap_div = LinearSegmentedColormap.from_list('red_white_green', colors_div, N=n_bins)

print("Starting data loading and processing...")


def get_training_files(land_type, index):
    """Get all training files for a specific land type and index"""
    folder_map = {
        'Grasslands': 'CALIFORNIA GRASSLANDS',
        'NaturalForests': 'CALIFORNIA NATURAL FORESTS',
        'Savannas': 'CALIFORNIA SAVANNAS',
        'Shrublands': 'CALIFORNIA SHRUBLANDS'
    }

    folder = folder_map[land_type]
    path = os.path.join(train_base, folder, index)

    if not os.path.exists(path):
        return []

    files = glob.glob(os.path.join(path, "*.tif"))
    return sorted(files)


def get_test_files(land_type, index):
    """Get all test files for a specific land type and index"""
    path = os.path.join(test_base, land_type, index)

    if not os.path.exists(path):
        return []

    files = glob.glob(os.path.join(path, "*.tif"))
    return sorted(files)


def clean_data(data, index):
    """Apply index-specific data cleaning and masking"""
    # Convert to float for proper NaN handling
    data = data.astype(np.float64)

    # First, mask all common nodata values - be VERY aggressive
    nodata_values = [-9999, -3.4e+38, 32767, -32768, 65535, -32767, 255, 254]
    for nodata in nodata_values:
        # Use a more lenient tolerance for matching
        data[np.abs(data - nodata) < 1.0] = np.nan

    # Also mask any suspiciously large values that look like nodata
    data[np.abs(data) > 10000] = np.nan

    # Index-specific cleaning with reasonable ranges
    if index == 'GOSIF':
        # GOSIF: typical range 0-2000 (W/m²/μm/sr)
        data[(data < -100) | (data > 3000)] = np.nan

    elif index == 'GPP':
        # GPP: typical range 0-15 gC/m²/day
        # Be strict: only keep reasonable values
        data[(data < 0) | (data > 20)] = np.nan
        # Also check if values are suspiciously clustered around weird numbers
        if np.nanmax(data) > 100:  # Something is wrong
            print(f"  WARNING: GPP has suspiciously high values (max={np.nanmax(data):.1f}), aggressive cleaning...")
            data[(data < 0) | (data > 15)] = np.nan

    elif index == 'LST':
        # LST: in Celsius, reasonable range -30 to 60
        data[(data < -40) | (data > 70)] = np.nan

    elif index in ['GNDVI', 'NDVI']:
        # Vegetation indices: typically -0.2 to 1
        data[(data < -1) | (data > 1.5)] = np.nan

    elif index == 'NIRv':
        # NIRv: typically 0 to 0.4
        data[(data < -0.1) | (data > 0.6)] = np.nan

    elif index == 'SPEI':
        # SPEI: typically -3 to 3
        data[(data < -4) | (data > 4)] = np.nan

    elif index == 'VPD':
        # VPD: in kPa, typically 0 to 8
        data[(data < 0) | (data > 10)] = np.nan

    elif index == 'TWS':
        # TWS: can be negative or positive (cm)
        data[(data < -100) | (data > 100)] = np.nan

    return data


# Determine reference grid from highest resolution data
reference_file = None
reference_transform = None
reference_crs = None
reference_shape = None
reference_bounds = None

print("\nDetermining reference grid from highest resolution data...")

# Try to find reference from GNDVI (typically ~4.5km resolution)
for land_type in land_types:
    train_files = get_training_files(land_type, 'GNDVI')
    if train_files:
        try:
            with rasterio.open(train_files[0]) as src:
                reference_transform = src.transform
                reference_crs = src.crs
                reference_shape = (src.height, src.width)
                reference_bounds = src.bounds
                if hasattr(reference_bounds, 'left'):
                    reference_bounds = (reference_bounds.left, reference_bounds.bottom,
                                        reference_bounds.right, reference_bounds.top)
                print(f"Reference grid: {reference_shape[0]}x{reference_shape[1]} pixels")
                print(f"Resolution: ~{reference_transform.a:.4f}° ({abs(reference_transform.a) * 111:.1f} km)")
                print(f"Bounds: Lon [{reference_bounds[0]:.2f}, {reference_bounds[2]:.2f}], "
                      f"Lat [{reference_bounds[1]:.2f}, {reference_bounds[3]:.2f}]")
                break
        except Exception as e:
            print(f"Error reading {train_files[0]}: {e}")
            continue

if reference_transform is None:
    print("ERROR: Could not determine reference grid!")
    exit(1)


def is_global_extent(bounds):
    """Check if bounds cover global extent"""
    lon_range = bounds[2] - bounds[0]
    lat_range = bounds[3] - bounds[1]
    return lon_range > 300 or lat_range > 150


def read_and_resample(filepath, ref_transform, ref_crs, ref_shape, index):
    """Read a file, clean it, clip if needed, and resample to reference grid"""
    try:
        with rasterio.open(filepath) as src:
            # Read data
            data = src.read(1).astype(np.float64)
            src_transform = src.transform
            src_crs = src.crs
            src_bounds = src.bounds
            src_nodata = src.nodata

            # If the file has a defined nodata value, mask it first
            if src_nodata is not None:
                data[data == src_nodata] = np.nan

            # CRITICAL: Clean data FIRST before any spatial operations
            data = clean_data(data, index)

            # Check if this is global data that needs clipping
            if is_global_extent(src_bounds):
                # Calculate which pixels correspond to California for global data
                min_col = int((CA_BOUNDS['west'] - src_bounds[0]) / src_transform.a)
                max_col = int((CA_BOUNDS['east'] - src_bounds[0]) / src_transform.a)
                max_row = int((src_bounds[3] - CA_BOUNDS['north']) / abs(src_transform.e))
                min_row = int((src_bounds[3] - CA_BOUNDS['south']) / abs(src_transform.e))

                # Ensure indices are within bounds
                min_col = max(0, min(src.width - 1, min_col))
                max_col = max(0, min(src.width, max_col))
                min_row = max(0, min(src.height - 1, min_row))
                max_row = max(0, min(src.height, max_row))

                if max_col <= min_col or max_row <= min_row:
                    print(f"  Warning: Invalid clip bounds for {os.path.basename(filepath)}")
                        return None

                # Clip to California region
                data = data[min_row:max_row, min_col:max_col]

                # Update transform for clipped data
                src_transform = rasterio.Affine(
                    src_transform.a, src_transform.b,
                    src_bounds[0] + min_col * src_transform.a,
                    src_transform.d, src_transform.e,
                    src_bounds[3] + min_row * src_transform.e
                )

            # Check if resampling is needed
            if data.shape == ref_shape and src_transform.almost_equals(ref_transform):
                return data

            # Resample to reference grid
            output = np.empty(ref_shape, dtype=np.float64)
            output[:] = np.nan

            reproject(
                source=data,
                destination=output,
                src_transform=src_transform,
                src_crs=src_crs if src_crs else ref_crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear,
                src_nodata=np.nan,
                dst_nodata=np.nan
            )

            return output

    except Exception as e:
        print(f"  Error processing {os.path.basename(filepath)}: {e}")
        return None


def calculate_index_mean(index):
    """Calculate temporal mean across all land types and years"""
    print(f"\nProcessing {index}...")

    all_data = []
    file_count = 0

    # Process each land type
    for land_type in land_types:
        train_files = get_training_files(land_type, index)
        test_files = get_test_files(land_type, index)
        all_files = train_files + test_files

        # Read and accumulate data
        for filepath in all_files:
            data = read_and_resample(filepath, reference_transform, reference_crs, reference_shape, index)
            if data is not None:
                valid_count = np.sum(~np.isnan(data))
                if valid_count > 0:
                    # Debug output for GPP
                    if index == 'GPP' and file_count < 3:
                        print(f"    File {file_count + 1}: {valid_count} valid pixels, "
                              f"range=[{np.nanmin(data):.3f}, {np.nanmax(data):.3f}]")
                    all_data.append(data)
                    file_count += 1

    if len(all_data) == 0:
        print(f"  WARNING: No valid data found for {index}")
        return None

    # Stack and calculate mean
    stacked = np.stack(all_data, axis=0)
    mean_data = np.nanmean(stacked, axis=0)

    valid_pixels = np.sum(~np.isnan(mean_data))
    total_pixels = mean_data.size

    print(f"  Files processed: {file_count}")
    print(f"  Valid pixels: {valid_pixels:,} ({valid_pixels / total_pixels * 100:.1f}%)")

    # Only show range if there's valid data
    if valid_pixels > 0:
        print(f"  Value range: [{np.nanmin(mean_data):.4f}, {np.nanmax(mean_data):.4f}]")
        print(f"  Mean: {np.nanmean(mean_data):.4f}, Std: {np.nanstd(mean_data):.4f}")

    return mean_data


# Calculate means for all indices
results = {}
for index in indices:
    mean_data = calculate_index_mean(index)
    if mean_data is not None and np.sum(~np.isnan(mean_data)) > 0:
        results[index] = mean_data

# Create 3x3 visualization
fig, axes = plt.subplots(3, 3, figsize=(20, 22))
axes = axes.flatten()

# Plot each index
for idx, index in enumerate(indices):
    ax = axes[idx]

    if index not in results:
        ax.text(0.5, 0.5, f'{index}\nNo Data Available',
                ha='center', va='center', fontsize=14, fontweight='bold',
                transform=ax.transAxes)
        ax.axis('off')
        continue

    data = results[index]

    # Choose colormap and scaling based on index
    if index == 'SPEI':
        cmap = cmap_div
        vmin, vmax = -2, 2
    elif index == 'VPD':
        # VPD: higher = more stress = red
        cmap = cmap_rg.reversed()
        vmin, vmax = np.nanpercentile(data, [5, 95])
    elif index == 'LST':
        # LST: higher temps = red
        cmap = cmap_rg.reversed()
        vmin, vmax = np.nanpercentile(data, [5, 95])
    elif index in ['GNDVI', 'GOSIF', 'GPP', 'NDVI', 'NIRv', 'TWS']:
        # Higher = better/healthier = green
        cmap = cmap_rg
        vmin, vmax = np.nanpercentile(data, [5, 95])
    else:
        cmap = cmap_rg
        vmin, vmax = np.nanpercentile(data, [5, 95])

    # Plot
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=reference_bounds, interpolation='nearest', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    # Add units
    units = {
        'LST': '°C', 'VPD': 'kPa', 'GPP': 'gC/m²/day',
        'GOSIF': 'W/m²/μm/sr', 'TWS': 'cm'
    }
    if index in units:
        cbar.set_label(units[index], fontsize=11, fontweight='bold')

    # Formatting
    ax.set_xlabel('Longitude (°E)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=11, fontweight='bold')
    ax.set_title(f'{index}\n(Mean 2001-2024)', fontsize=13, fontweight='bold', pad=10)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.suptitle('Spatial Distribution of Drought Indices in California (2001-2024 Average)',
             fontsize=17, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('california_9panel_indices_fixed.png', dpi=300, bbox_inches='tight')
plt.savefig('california_9panel_indices_fixed.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved: california_9panel_indices_fixed.png/pdf")
plt.show()

# Print summary statistics
print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)
for index in indices:
    if index in results:
        data = results[index]
        valid_mask = ~np.isnan(data)
        if np.sum(valid_mask) > 0:
            print(f"\n{index}:")
            print(f"  Mean:      {np.nanmean(data):10.4f}")
            print(f"  Median:    {np.nanmedian(data):10.4f}")
            print(f"  Std:       {np.nanstd(data):10.4f}")
            print(f"  Min:       {np.nanmin(data):10.4f}")
            print(f"  Max:       {np.nanmax(data):10.4f}")
            print(f"  P[5,95]:   [{np.nanpercentile(data, 5):10.4f}, {np.nanpercentile(data, 95):10.4f}]")
            print(f"  Valid:     {np.sum(valid_mask):,} pixels ({np.sum(valid_mask) / data.size * 100:.1f}%)")

print("\n" + "=" * 70)
print("Processing complete!")
print("=" * 70)