import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

"""
OPTION C: DROUGHT EVENT COMPARISON
Compares 2012-2016 historic drought vs 2020-2024 recent crisis
Side-by-side maps showing which land types were hit harder in each event
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
SELECTED_INDEX = 'NIRv'  # Change to GNDVI, GPP, GOSIF if needed

# Define periods
BASELINE = (2001, 2011)
DROUGHT_2012 = (2012, 2016)
CRISIS_2020 = (2020, 2024)

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("=" * 70)
print(f"DROUGHT EVENT COMPARISON: {SELECTED_INDEX}")
print("=" * 70)
print(f"Baseline: {BASELINE[0]}-{BASELINE[1]}")
print(f"2012-2016 Drought: {DROUGHT_2012[0]}-{DROUGHT_2012[1]}")
print(f"2020-2024 Crisis: {CRISIS_2020[0]}-{CRISIS_2020[1]}")


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

    if index in ['NIRv']:
        data[(data < -0.1) | (data > 0.6)] = np.nan
    elif index in ['GNDVI', 'NDVI']:
        data[(data < -1) | (data > 1.5)] = np.nan
    elif index == 'GPP':
        data[(data < 0) | (data > 20)] = np.nan
    elif index == 'GOSIF':
        data[(data < -100) | (data > 3000)] = np.nan

    return data


# Load reference grid
print("\nLoading reference grid...")
ref_transform = None
for land_type in land_types:
    files = get_files(land_type, SELECTED_INDEX)
    if files:
        with rasterio.open(files[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = (src.height, src.width)
            ref_bounds = src.bounds
            if hasattr(ref_bounds, 'left'):
                ref_bounds = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
            break

# Process each land type
results = {}

for land_type in land_types:
    print(f"\n{'=' * 70}")
    print(f"Processing {land_type}...")
    print('=' * 70)

    files = get_files(land_type, SELECTED_INDEX)

    baseline_data = []
    drought_2012_data = []
    crisis_2020_data = []

    for filepath in files:
        try:
            filename = os.path.basename(filepath)
            year = int(filename.split('_')[-2])

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

                data = clean_data(data, SELECTED_INDEX)

                if BASELINE[0] <= year <= BASELINE[1]:
                    baseline_data.append(data)
                elif DROUGHT_2012[0] <= year <= DROUGHT_2012[1]:
                    drought_2012_data.append(data)
                elif CRISIS_2020[0] <= year <= CRISIS_2020[1]:
                    crisis_2020_data.append(data)
        except:
            continue

    print(f"  Baseline: {len(baseline_data)} files")
    print(f"  2012-2016: {len(drought_2012_data)} files")
    print(f"  2020-2024: {len(crisis_2020_data)} files")

    if len(baseline_data) == 0:
        print(f"  WARNING: No baseline data")
        continue

    baseline_mean = np.nanmean(np.stack(baseline_data, axis=0), axis=0)

    # 2012-2016 anomaly
    if len(drought_2012_data) > 0:
        drought_2012_mean = np.nanmean(np.stack(drought_2012_data, axis=0), axis=0)
        drought_2012_anomaly = ((drought_2012_mean - baseline_mean) / baseline_mean) * 100
    else:
        drought_2012_anomaly = None

    # 2020-2024 anomaly
    if len(crisis_2020_data) > 0:
        crisis_2020_mean = np.nanmean(np.stack(crisis_2020_data, axis=0), axis=0)
        crisis_2020_anomaly = ((crisis_2020_mean - baseline_mean) / baseline_mean) * 100
    else:
        crisis_2020_anomaly = None

    results[land_type] = {
        'drought_2012': drought_2012_anomaly,
        'crisis_2020': crisis_2020_anomaly
    }

    if drought_2012_anomaly is not None:
        print(f"  2012-2016 mean change: {np.nanmean(drought_2012_anomaly):.1f}%")
    if crisis_2020_anomaly is not None:
        print(f"  2020-2024 mean change: {np.nanmean(crisis_2020_anomaly):.1f}%")

# Create visualization - 4 rows × 2 columns
print("\n" + "=" * 70)
print("Creating comparison maps...")
print("=" * 70)

fig, axes = plt.subplots(4, 2, figsize=(18, 24))

for idx, land_type in enumerate(land_types):
    # Left column: 2012-2016 drought
    ax_left = axes[idx, 0]
    # Right column: 2020-2024 crisis
    ax_right = axes[idx, 1]

    if land_type not in results:
        for ax in [ax_left, ax_right]:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        continue

    # Plot 2012-2016
    if results[land_type]['drought_2012'] is not None:
        data_2012 = results[land_type]['drought_2012']
        im_left = ax_left.imshow(data_2012, cmap='RdYlGn', vmin=-60, vmax=30,
                                 extent=ref_bounds, aspect='auto', interpolation='nearest')
        cbar_left = plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
        cbar_left.set_label('Change (%)', fontsize=10, fontweight='bold')

        # Stats
        valid = data_2012[~np.isnan(data_2012)]
        if len(valid) > 0:
            stats = f'Mean: {np.mean(valid):.1f}%\nSevere (<-40%): {np.sum(valid < -40):,} px'
            ax_left.text(0.02, 0.98, stats, transform=ax_left.transAxes,
                         fontsize=9, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax_left.text(0.5, 0.5, 'No Data', ha='center', va='center',
                     transform=ax_left.transAxes, fontsize=12)

    ax_left.set_title(f'{land_type}\n2012-2016 Historic Drought',
                      fontsize=13, fontweight='bold', pad=10)
    ax_left.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_left.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax_left.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Plot 2020-2024
    if results[land_type]['crisis_2020'] is not None:
        data_2020 = results[land_type]['crisis_2020']
        im_right = ax_right.imshow(data_2020, cmap='RdYlGn', vmin=-60, vmax=30,
                                   extent=ref_bounds, aspect='auto', interpolation='nearest')
        cbar_right = plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
        cbar_right.set_label('Change (%)', fontsize=10, fontweight='bold')

        # Stats
        valid = data_2020[~np.isnan(data_2020)]
        if len(valid) > 0:
            stats = f'Mean: {np.mean(valid):.1f}%\nSevere (<-40%): {np.sum(valid < -40):,} px'
            ax_right.text(0.02, 0.98, stats, transform=ax_right.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    else:
        ax_right.text(0.5, 0.5, 'No Data', ha='center', va='center',
                      transform=ax_right.transAxes, fontsize=12)

    ax_right.set_title(f'{land_type}\n2020-2024 Recent Crisis',
                       fontsize=13, fontweight='bold', pad=10)
    ax_right.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_right.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax_right.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.figtext(0.5, 0.01,
            f'Red = Decline • Green = Increase/Stable • Both periods compared to 2001-2011 baseline\n' +
            f'{SELECTED_INDEX} anomalies show spatial patterns of vegetation stress during major drought events',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

plt.suptitle(f'Drought Event Comparison: {SELECTED_INDEX} Response\n' +
             'Historic Drought (2012-2016) vs Recent Crisis (2020-2024)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.99])
plt.savefig(f'drought_comparison_{SELECTED_INDEX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'drought_comparison_{SELECTED_INDEX}.pdf', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: drought_comparison_{SELECTED_INDEX}.png/pdf")
plt.show()

# Comparison summary
print("\n" + "=" * 70)
print("EVENT COMPARISON SUMMARY")
print("=" * 70)
for land_type in land_types:
    if land_type in results:
        print(f"\n{land_type}:")

        if results[land_type]['drought_2012'] is not None:
            d2012 = results[land_type]['drought_2012'][~np.isnan(results[land_type]['drought_2012'])]
            print(f"  2012-2016: Mean={np.mean(d2012):.1f}%, Severe pixels={np.sum(d2012 < -40):,}")

        if results[land_type]['crisis_2020'] is not None:
            d2020 = results[land_type]['crisis_2020'][~np.isnan(results[land_type]['crisis_2020'])]
            print(f"  2020-2024: Mean={np.mean(d2020):.1f}%, Severe pixels={np.sum(d2020 < -40):,}")

        # Compare
        if results[land_type]['drought_2012'] is not None and results[land_type]['crisis_2020'] is not None:
            diff = np.mean(d2020) - np.mean(d2012)
            if diff < -5:
                print(f"  → 2020-2024 was WORSE by {abs(diff):.1f}%")
            elif diff > 5:
                print(f"  → 2012-2016 was WORSE by {diff:.1f}%")
            else:
                print(f"  → Similar impact")

print("\n" + "=" * 70)