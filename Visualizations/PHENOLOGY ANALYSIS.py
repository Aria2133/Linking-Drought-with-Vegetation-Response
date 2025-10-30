import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

"""
DROUGHT PHENOLOGY ANALYSIS - VEGETATION PRODUCTIVITY INDICES
Analyzes stress patterns for NDVI, NIRv, GPP, GOSIF
Low values = vegetation stress/drought
Extracts: onset, peak stress, duration, recovery
"""

# Configuration - ONLY PRODUCTIVITY INDICES
indices = ['NDVI', 'NIRv', 'GPP', 'GOSIF']
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']

# Stress thresholds (below these = stress)
STRESS_THRESHOLDS = {
    'NDVI': 0.3,  # NDVI < 0.3 = stressed
    'NIRv': 0.05,  # NIRv < 0.05 = stressed
    'GPP': 2.0,  # GPP < 2 gC/m²/day = stressed
    'GOSIF': 200  # GOSIF < 200 = stressed
}

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("Creating Vegetation Stress Phenology Maps...")


def get_files(land_type, index):
    folder_map = {
        'Grasslands': 'CALIFORNIA GRASSLANDS',
        'NaturalForests': 'CALIFORNIA NATURAL FORESTS',
        'Savannas': 'CALIFORNIA SAVANNAS',
        'Shrublands': 'CALIFORNIA SHRUBLANDS'
    }
    folder = folder_map[land_type]
    train_path = os.path.join(train_base, folder, index)
    test_path = os.path.join(test_base, land_type, index)

    files = []
    if os.path.exists(train_path):
        files.extend(glob.glob(os.path.join(train_path, "*.tif")))
    if os.path.exists(test_path):
        files.extend(glob.glob(os.path.join(test_path, "*.tif")))

    return sorted(list(set(files)))


def clean_data(data, index):
    data = data.astype(np.float64)
    nodata_values = [-9999, -3.4e+38, 32767, -32768, 65535, -32767]
    for nodata in nodata_values:
        data[np.abs(data - nodata) < 1.0] = np.nan
    data[np.abs(data) > 10000] = np.nan

    if index in ['NDVI']:
        data[(data < -1) | (data > 1.5)] = np.nan
    elif index == 'NIRv':
        data[(data < -0.1) | (data > 0.6)] = np.nan
    elif index == 'GPP':
        data[(data < 0) | (data > 20)] = np.nan
    elif index == 'GOSIF':
        data[(data < -100) | (data > 3000)] = np.nan

    return data


# Load reference grid
print("Loading reference grid...")
ref_transform = None
for land_type in land_types:
    for index in indices:
        files = get_files(land_type, index)
        if files:
            with rasterio.open(files[0]) as src:
                ref_transform = src.transform
                ref_crs = src.crs
                ref_shape = (src.height, src.width)
                ref_bounds = src.bounds
                if hasattr(ref_bounds, 'left'):
                    ref_bounds = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
                break
        if ref_transform:
            break
    if ref_transform:
        break

print(f"Reference shape: {ref_shape}")

# Process each index
all_phenology = {}

for index in indices:
    print(f"\n{'=' * 60}")
    print(f"Processing {index}...")
    print('=' * 60)

    threshold = STRESS_THRESHOLDS[index]

    # Load time series - COMBINE all land types for each month
    print(f"  Organizing files by date...")

    # First, collect all files and extract their dates
    all_files_with_dates = []
    for land_type in land_types:
        files = get_files(land_type, index)
        print(f"  {land_type}: {len(files)} files")

        for filepath in files:
            filename = os.path.basename(filepath)
            try:
                # Try standard format first (year and month in separate segments)
                year = int(filename.split('_')[-2])
                month = int(filename.split('_')[-1].split('.')[0])
                all_files_with_dates.append((year, month, filepath, land_type))
            except:
                # Try GOSIF format (YYYYMM in last segment before .tif)
                try:
                    date_str = filename.split('_')[-1].split('.')[0]
                    if len(date_str) == 6:  # YYYYMM format
                        year = int(date_str[:4])
                        month = int(date_str[4:6])
                        all_files_with_dates.append((year, month, filepath, land_type))
                except:
                    continue

    # Group files by (year, month)
    from collections import defaultdict

    files_by_date = defaultdict(list)
    for year, month, filepath, land_type in all_files_with_dates:
        files_by_date[(year, month)].append(filepath)

    # Sort dates
    unique_dates = sorted(files_by_date.keys())
    print(f"  Found {len(unique_dates)} unique months")

    if len(unique_dates) == 0:
        print(f"  WARNING: No dates found for {index}")
        continue

    # Load and combine data for each month
    time_series_data = []
    dates = []

    for year, month in unique_dates:
        # Load all land type files for this month
        monthly_grids = []

        for filepath in files_by_date[(year, month)]:
            try:
                with rasterio.open(filepath) as src:
                    data = src.read(1)

                    if src.shape != ref_shape:
                        output = np.empty(ref_shape, dtype=np.float32)
                        reproject(
                            source=data,
                            destination=output,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan,
                            dst_nodata=np.nan
                        )
                        data = output

                    data = clean_data(data, index)
                    monthly_grids.append(data)
            except:
                continue

        if len(monthly_grids) == 0:
            continue

        # Combine all land types: since masks don't overlap, just fill in NaNs
        # Start with all NaN
        monthly_map = np.full(ref_shape, np.nan)

        # Fill in each land type's data (non-overlapping masks)
        for grid in monthly_grids:
            valid_mask = ~np.isnan(grid)
            monthly_map[valid_mask] = grid[valid_mask]

        # Only keep if there's valid data
        if np.sum(~np.isnan(monthly_map)) > 0:
            time_series_data.append(monthly_map)
            dates.append((year, month))

    print(f"  Loaded {len(time_series_data)} combined monthly maps")

    # Stack into cube
    data_cube = np.stack(time_series_data, axis=0)
    n_times, n_rows, n_cols = data_cube.shape

    # Show data range for debugging
    all_valid = data_cube[~np.isnan(data_cube)]
    if len(all_valid) > 0:
        print(f"  Data range: [{np.min(all_valid):.3f}, {np.max(all_valid):.3f}]")
        print(f"  Percentiles [5,25,50,75,95]: {np.percentile(all_valid, [5, 25, 50, 75, 95])}")

    # Calculate phenology metrics using ANOMALY DETECTION
    print(f"  Calculating phenology using anomalies...")
    print(f"  Checking data cube validity...")

    # Count how many pixels have any data at all
    pixels_with_any_data = 0
    for i in range(n_rows):
        for j in range(n_cols):
            ts = data_cube[:, i, j]
            if np.sum(~np.isnan(ts)) > 0:
                pixels_with_any_data += 1

    print(f"  Pixels with ANY data: {pixels_with_any_data} out of {n_rows * n_cols}")

    # Check a sample pixel
    sample_found = False
    for i in range(n_rows):
        for j in range(n_cols):
            ts = data_cube[:, i, j]
            valid_count = np.sum(~np.isnan(ts))
            if valid_count > 10:
                print(
                    f"  Sample pixel [{i},{j}]: {valid_count} valid values, range=[{np.nanmin(ts):.3f}, {np.nanmax(ts):.3f}]")
                sample_found = True
                break
        if sample_found:
            break

    if pixels_with_any_data == 0:
        print(f"  ERROR: No pixels have data! Data cube may be all NaN.")
        continue

    onset_map = np.full(ref_shape, np.nan)
    peak_map = np.full(ref_shape, np.nan)
    duration_map = np.full(ref_shape, np.nan)
    recovery_map = np.full(ref_shape, np.nan)
    frequency_map = np.full(ref_shape, np.nan)

    valid_pixels = 0
    total_checked = 0

    for i in range(n_rows):
        for j in range(n_cols):
            ts = data_cube[:, i, j]

            # Need at least 20% valid data (reduced from 50%)
            valid_count = np.sum(~np.isnan(ts))
            if valid_count < n_times * 0.2:  # Changed from 0.5
                continue

            total_checked += 1

            # Simple interpolation for missing values
            valid_mask = ~np.isnan(ts)
            valid_idx = np.where(valid_mask)[0]
            valid_vals = ts[valid_mask]

            if len(valid_idx) < 3:
                continue

            ts_filled = np.interp(np.arange(n_times), valid_idx, valid_vals)

            # ANOMALY DETECTION: stress = below 25th percentile of pixel's own history
            pixel_threshold = np.percentile(ts_filled, 25)
            in_stress = ts_filled < pixel_threshold

            if not np.any(in_stress):
                continue

            valid_pixels += 1

            # Onset: first stress event
            stress_starts = np.where(np.diff(in_stress.astype(int)) == 1)[0]
            if len(stress_starts) > 0:
                onset_map[i, j] = stress_starts[0]

            # Peak: minimum value (worst stress)
            peak_map[i, j] = np.min(ts_filled[in_stress])

            # Duration: total months in stress
            duration_map[i, j] = np.sum(in_stress)

            # Recovery: last exit from stress
            stress_ends = np.where(np.diff(in_stress.astype(int)) == -1)[0]
            if len(stress_ends) > 0:
                recovery_map[i, j] = stress_ends[-1]

            # Frequency: count discrete stress events
            events = 0
            in_event = False
            for val in in_stress:
                if val and not in_event:
                    events += 1
                    in_event = True
                elif not val:
                    in_event = False
            frequency_map[i, j] = events

        if (i + 1) % 20 == 0:
            print(f"    Row {i + 1}/{n_rows}, checked: {total_checked}, valid: {valid_pixels}")

    print(f"  Total pixels checked: {total_checked}")
    print(f"  Total valid pixels: {valid_pixels}")

    all_phenology[index] = {
        'onset': onset_map,
        'peak': peak_map,
        'duration': duration_map,
        'recovery': recovery_map,
        'frequency': frequency_map,
        'dates': dates,
        'threshold': threshold
    }

# Create visualization - one figure per index
print("\n" + "=" * 60)
print("Creating visualizations...")
print("=" * 60)

for index in indices:
    if index not in all_phenology:
        continue

    data = all_phenology[index]

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # 1. Onset
    im1 = axes[0].imshow(data['onset'], cmap='coolwarm', extent=ref_bounds, aspect='auto')
    axes[0].set_title(f'{index} Stress Onset\n(First Occurrence)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Longitude (°E)', fontweight='bold')
    axes[0].set_ylabel('Latitude (°N)', fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046)
    cbar1.set_label('Time Index', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 2. Peak Stress
    im2 = axes[1].imshow(data['peak'], cmap='RdYlGn', extent=ref_bounds, aspect='auto')
    axes[1].set_title(f'{index} Peak Stress\n(Minimum Value)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Longitude (°E)', fontweight='bold')
    axes[1].set_ylabel('Latitude (°N)', fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046)
    cbar2.set_label(f'{index} Value', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # 3. Duration
    im3 = axes[2].imshow(data['duration'], cmap='YlOrRd', extent=ref_bounds, aspect='auto')
    axes[2].set_title(f'{index} Stress Duration\n(Total Months)', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('Longitude (°E)', fontweight='bold')
    axes[2].set_ylabel('Latitude (°N)', fontweight='bold')
    cbar3 = plt.colorbar(im3, ax=axes[2], fraction=0.046)
    cbar3.set_label('Months', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # 4. Recovery
    im4 = axes[3].imshow(data['recovery'], cmap='coolwarm', extent=ref_bounds, aspect='auto')
    axes[3].set_title(f'{index} Recovery Time\n(Last Exit from Stress)', fontsize=13, fontweight='bold')
    axes[3].set_xlabel('Longitude (°E)', fontweight='bold')
    axes[3].set_ylabel('Latitude (°N)', fontweight='bold')
    cbar4 = plt.colorbar(im4, ax=axes[3], fraction=0.046)
    cbar4.set_label('Time Index', fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # 5. Frequency
    im5 = axes[4].imshow(data['frequency'], cmap='plasma', extent=ref_bounds, aspect='auto')
    axes[4].set_title(f'{index} Stress Frequency\n(Number of Events)', fontsize=13, fontweight='bold')
    axes[4].set_xlabel('Longitude (°E)', fontweight='bold')
    axes[4].set_ylabel('Latitude (°N)', fontweight='bold')
    cbar5 = plt.colorbar(im5, ax=axes[4], fraction=0.046)
    cbar5.set_label('Events', fontsize=10)
    axes[4].grid(True, alpha=0.3)

    # 6. Summary
    axes[5].axis('off')

    dates = data['dates']
    n_years = (dates[-1][0] - dates[0][0]) + 1
    stats_text = f"""{index} STRESS PHENOLOGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Method: Anomaly Detection
Stress = Below 25th percentile
Period: {dates[0][0]}-{dates[-1][0]}
Total Months: {len(dates)} ({n_years} years)

SPATIAL STATISTICS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Onset (median): {np.nanmedian(data['onset']):.0f} months
Peak Stress (mean): {np.nanmean(data['peak']):.3f}
Duration (mean): {np.nanmean(data['duration']):.0f} months
Recovery (median): {np.nanmedian(data['recovery']):.0f} months
Frequency (mean): {np.nanmean(data['frequency']):.1f} events

Valid Pixels: {np.sum(~np.isnan(data['onset'])):,}

INTERPRETATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stress = when {index} drops below its
own historical 25th percentile
(pixel-specific thresholds)

ALL LAND TYPES COMBINED:
This analysis combines Grasslands,
Natural Forests, Savannas, and
Shrublands into one map per month.

• Onset: First anomaly (months)
• Peak: Worst {index} value
• Duration: Total months stressed
• Recovery: Last recovery (months)
• Frequency: Number of stress events

Reveals relative stress patterns
across all California vegetation.
"""

    axes[5].text(0.05, 0.95, stats_text, transform=axes[5].transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f'Vegetation Stress Phenology: {index} in California',
                 fontsize=18, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'phenology_{index}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'phenology_{index}.pdf', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: phenology_{index}.png/pdf")
    plt.close()

print("\n" + "=" * 60)
print("PHENOLOGY ANALYSIS COMPLETE")
print("=" * 60)
print(f"Created {len(all_phenology)} phenology maps:")
for index in all_phenology.keys():
    print(f"  • phenology_{index}.png/pdf")
print("\nThese maps reveal HOW vegetation stress manifests:")
print("  - Temporal patterns (when stress occurs)")
print("  - Severity patterns (how bad it gets)")
print("  - Duration patterns (how long it lasts)")
print("  - Recovery patterns (resilience)")
print("  - Frequency patterns (chronic vs episodic)")
print("=" * 60)