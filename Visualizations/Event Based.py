import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

"""
EVENT-BASED DROUGHT VULNERABILITY ANALYSIS
Compares NIRv during 2012-2016 California drought vs baseline period
Shows which areas declined most = most vulnerable to drought
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']

# Define periods
BASELINE_YEARS = (2001, 2011)  # Pre-drought baseline
DROUGHT_YEARS = (2012, 2016)  # Major California drought
RECOVERY_YEARS = (2017, 2021)  # Post-drought recovery

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("=" * 70)
print("EVENT-BASED DROUGHT VULNERABILITY ANALYSIS")
print("=" * 70)
print(f"Baseline period: {BASELINE_YEARS[0]}-{BASELINE_YEARS[1]}")
print(f"Drought period: {DROUGHT_YEARS[0]}-{DROUGHT_YEARS[1]}")
print(f"Recovery period: {RECOVERY_YEARS[0]}-{RECOVERY_YEARS[1]}")


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


def clean_data(data):
    data = data.astype(np.float64)
    nodata_values = [-9999, -3.4e+38, 32767, -32768, 65535, -32767]
    for nodata in nodata_values:
        data[np.abs(data - nodata) < 1.0] = np.nan
    data[np.abs(data) > 10000] = np.nan
    data[(data < -0.1) | (data > 0.6)] = np.nan
    return data


# Load reference grid
print("\nLoading reference grid...")
ref_transform = None
for land_type in land_types:
    files = get_files(land_type, 'NIRv')
    if files:
        with rasterio.open(files[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = (src.height, src.width)
            ref_bounds = src.bounds
            if hasattr(ref_bounds, 'left'):
                ref_bounds = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
            print(f"Reference: {ref_shape[0]}x{ref_shape[1]} pixels")
            break

# Process each land type
vulnerability_results = {}

for land_type in land_types:
    print(f"\n{'=' * 70}")
    print(f"Processing {land_type}...")
    print('=' * 70)

    # Load all NIRv files with dates
    files = get_files(land_type, 'NIRv')

    baseline_data = []
    drought_data = []
    recovery_data = []

    for filepath in files:
        try:
            # Extract date
            filename = os.path.basename(filepath)
            year = int(filename.split('_')[-2])
            month = int(filename.split('_')[-1].split('.')[0])

            # Read data
            with rasterio.open(filepath) as src:
                data = src.read(1)

                if src.shape != ref_shape:
                    output = np.empty(ref_shape, dtype=np.float32)
                    reproject(
                        source=data, destination=output,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=ref_transform, dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=np.nan, dst_nodata=np.nan
                    )
                    data = output

                data = clean_data(data)

                # Categorize by period
                if BASELINE_YEARS[0] <= year <= BASELINE_YEARS[1]:
                    baseline_data.append(data)
                elif DROUGHT_YEARS[0] <= year <= DROUGHT_YEARS[1]:
                    drought_data.append(data)
                elif RECOVERY_YEARS[0] <= year <= RECOVERY_YEARS[1]:
                    recovery_data.append(data)
        except:
            continue

    print(f"  Baseline: {len(baseline_data)} time points")
    print(f"  Drought: {len(drought_data)} time points")
    print(f"  Recovery: {len(recovery_data)} time points")

    if len(baseline_data) == 0 or len(drought_data) == 0:
        print(f"  WARNING: Insufficient data for {land_type}")
        continue

    # Calculate means for each period
    baseline_mean = np.nanmean(np.stack(baseline_data, axis=0), axis=0)
    drought_mean = np.nanmean(np.stack(drought_data, axis=0), axis=0)

    if len(recovery_data) > 0:
        recovery_mean = np.nanmean(np.stack(recovery_data, axis=0), axis=0)
    else:
        recovery_mean = None

    # Calculate anomalies
    drought_anomaly = drought_mean - baseline_mean

    if recovery_mean is not None:
        recovery_anomaly = recovery_mean - baseline_mean
    else:
        recovery_anomaly = None

    # Calculate percent change
    percent_change = (drought_anomaly / baseline_mean) * 100

    # Store results
    vulnerability_results[land_type] = {
        'baseline_mean': baseline_mean,
        'drought_mean': drought_mean,
        'recovery_mean': recovery_mean,
        'drought_anomaly': drought_anomaly,
        'recovery_anomaly': recovery_anomaly,
        'percent_change': percent_change
    }

    print(f"  Mean NIRv - Baseline: {np.nanmean(baseline_mean):.4f}")
    print(f"  Mean NIRv - Drought: {np.nanmean(drought_mean):.4f}")
    print(f"  Mean anomaly: {np.nanmean(drought_anomaly):.4f}")
    print(f"  Mean % change: {np.nanmean(percent_change):.1f}%")

# Create visualization
print("\n" + "=" * 70)
print("Creating vulnerability maps...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
axes = axes.flatten()

for idx, land_type in enumerate(land_types):
    ax = axes[idx]

    if land_type not in vulnerability_results:
        ax.text(0.5, 0.5, f'{land_type}\nNo Data',
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        continue

    # Use percent change for visualization
    pct_change = vulnerability_results[land_type]['percent_change']

    # Plot
    im = ax.imshow(pct_change, cmap='RdYlGn', vmin=-50, vmax=50,
                   extent=ref_bounds, aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('NIRv Change (%)', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_xlabel('Longitude (°E)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=12, fontweight='bold')
    ax.set_title(f'{land_type}\nDrought Impact (2012-2016 vs 2001-2011)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Statistics
    valid_pct = pct_change[~np.isnan(pct_change)]
    if len(valid_pct) > 0:
        mean_pct = np.mean(valid_pct)
        median_pct = np.median(valid_pct)
        severe_decline = np.sum(valid_pct < -30)
        moderate_decline = np.sum((valid_pct >= -30) & (valid_pct < -10))

        stats_text = f'Mean: {mean_pct:.1f}%\nMedian: {median_pct:.1f}%\n'
        stats_text += f'Severe decline (<-30%): {severe_decline:,} px\n'
        stats_text += f'Moderate decline (-30 to -10%): {moderate_decline:,} px'

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Add interpretation
plt.figtext(0.5, 0.01,
            'Red = Severe decline during drought (VULNERABLE) • Yellow = Moderate change • Green = Increase or stable (RESILIENT)\n' +
            f'Compares mean NIRv during drought ({DROUGHT_YEARS[0]}-{DROUGHT_YEARS[1]}) vs baseline ({BASELINE_YEARS[0]}-{BASELINE_YEARS[1]})',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

plt.suptitle('Drought Vulnerability: Vegetation Response to 2012-2016 California Drought',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.04, 1, 0.96])
plt.savefig('drought_vulnerability_event_based.png', dpi=300, bbox_inches='tight')
plt.savefig('drought_vulnerability_event_based.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Saved: drought_vulnerability_event_based.png/pdf")
plt.show()

# Print summary
print("\n" + "=" * 70)
print("SUMMARY BY LAND TYPE")
print("=" * 70)
for land_type in land_types:
    if land_type in vulnerability_results:
        pct = vulnerability_results[land_type]['percent_change']
        valid_pct = pct[~np.isnan(pct)]
        if len(valid_pct) > 0:
            print(f"\n{land_type}:")
            print(f"  Mean change: {np.mean(valid_pct):.1f}%")
            print(f"  Median change: {np.median(valid_pct):.1f}%")
            print(f"  Range: [{np.min(valid_pct):.1f}%, {np.max(valid_pct):.1f}%]")
            print(f"  Pixels with >30% decline: {np.sum(valid_pct < -30):,}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)