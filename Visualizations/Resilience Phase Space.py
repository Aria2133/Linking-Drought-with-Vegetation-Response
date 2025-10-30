import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os
import glob

"""
RESILIENCE PHASE SPACE
X-axis = SPEI (water stress)
Y-axis = NIRv (vegetation response)
Each trajectory = one land type moving through drought cycles
Reveals hysteresis, stability, and dynamic responses
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

# Colors for each land type
colors = {'Grasslands': '#2ecc71', 'NaturalForests': '#27ae60',
          'Savannas': '#f39c12', 'Shrublands': '#e67e22'}

print("=" * 70)
print("RESILIENCE PHASE SPACE ANALYSIS")
print("=" * 70)


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

    if index == 'SPEI':
        data[(data < -4) | (data > 4)] = np.nan
    elif index == 'NIRv':
        data[(data < -0.1) | (data > 0.6)] = np.nan

    return data


# Load reference grid
ref_transform = None
for land_type in land_types:
    files = get_files(land_type, 'NIRv')
    if files:
        with rasterio.open(files[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = (src.height, src.width)
            break

# Extract time series for each land type
trajectories = {}

for land_type in land_types:
    print(f"\nProcessing {land_type}...")

    # Load SPEI
    spei_files = get_files(land_type, 'SPEI')
    spei_data = []
    dates = []

    for filepath in spei_files:
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                if src.shape != ref_shape:
                    output = np.empty(ref_shape, dtype=np.float32)
                    reproject(source=data, destination=output,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=ref_transform, dst_crs=ref_crs,
                              resampling=Resampling.bilinear,
                              src_nodata=np.nan, dst_nodata=np.nan)
                    data = output
                data = clean_data(data, 'SPEI')
                spei_data.append(np.nanmean(data))

                filename = os.path.basename(filepath)
                year = int(filename.split('_')[-2].split('.')[0]) if 'GRIDMET' in filename else int(
                    filename.split('_')[-2])
                month = int(filename.split('_')[-1].split('.')[0]) if 'GRIDMET' not in filename else int(
                    filename.split('_')[-1].split('_')[0])
                dates.append((year, month))
        except:
            continue

    # Load NIRv
    nirv_files = get_files(land_type, 'NIRv')
    nirv_data = []

    for filepath in nirv_files:
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                if src.shape != ref_shape:
                    output = np.empty(ref_shape, dtype=np.float32)
                    reproject(source=data, destination=output,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=ref_transform, dst_crs=ref_crs,
                              resampling=Resampling.bilinear,
                              src_nodata=np.nan, dst_nodata=np.nan)
                    data = output
                data = clean_data(data, 'NIRv')
                nirv_data.append(np.nanmean(data))
        except:
            continue

    # Match lengths
    min_len = min(len(spei_data), len(nirv_data))
    spei_data = spei_data[:min_len]
    nirv_data = nirv_data[:min_len]
    dates = dates[:min_len]

    print(f"  Collected {len(spei_data)} matched time points")

    trajectories[land_type] = {
        'spei': np.array(spei_data),
        'nirv': np.array(nirv_data),
        'dates': dates
    }

# Create phase space visualization
print("\nCreating phase space diagram...")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for idx, land_type in enumerate(land_types):
    ax = axes[idx]

    if land_type not in trajectories:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        continue

    spei = trajectories[land_type]['spei']
    nirv = trajectories[land_type]['nirv']
    dates = trajectories[land_type]['dates']

    # Create segments for colored trajectory
    points = np.array([spei, nirv]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Color by time (gradient from early to late)
    time_colors = np.linspace(0, 1, len(spei))

    # Plot trajectory with gradient
    lc = LineCollection(segments, array=time_colors, cmap='viridis',
                        linewidth=2, alpha=0.7)
    ax.add_collection(lc)

    # Mark start and end
    ax.scatter(spei[0], nirv[0], s=150, c='green', marker='o',
               edgecolors='black', linewidth=2, label='Start (2001)', zorder=5)
    ax.scatter(spei[-1], nirv[-1], s=150, c='red', marker='s',
               edgecolors='black', linewidth=2, label='End (2024)', zorder=5)

    # Mark 2012-2016 drought period
    drought_mask = [(2012 <= d[0] <= 2016) for d in dates]
    if np.any(drought_mask):
        drought_spei = spei[drought_mask]
        drought_nirv = nirv[drought_mask]
        ax.scatter(drought_spei, drought_nirv, s=50, c='orange',
                   alpha=0.5, label='2012-2016 Drought')

    # Equilibrium reference (mean state)
    mean_spei = np.mean(spei)
    mean_nirv = np.mean(nirv)
    ax.axvline(mean_spei, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(mean_nirv, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.plot(mean_spei, mean_nirv, 'k+', markersize=20, markeredgewidth=3,
            label='Mean State')

    # Formatting
    ax.set_xlabel('SPEI (Water Availability)', fontsize=12, fontweight='bold')
    ax.set_ylabel('NIRv (Vegetation Productivity)', fontsize=12, fontweight='bold')
    ax.set_title(f'{land_type}\nDrought Response Trajectory',
                 fontsize=13, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # Set consistent axes limits
    ax.set_xlim(-2.5, 1.5)
    ax.set_ylim(np.nanmin(nirv) * 0.95, np.nanmax(nirv) * 1.05)

    # Add colorbar for time
    cbar = plt.colorbar(lc, ax=ax, orientation='horizontal',
                        fraction=0.046, pad=0.08)
    cbar.set_label('Time (2001 → 2024)', fontsize=10)

plt.figtext(0.5, 0.02,
            'Phase space shows vegetation-water stress dynamics • Tight loops = resilient • Wide loops = slow recovery\n' +
            'Arrows point from past to future • Orange = 2012-2016 drought period • Gray crosshairs = mean equilibrium state',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Resilience Phase Space: Vegetation Response to Water Stress Cycles',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])
plt.savefig('resilience_phase_space.png', dpi=300, bbox_inches='tight')
plt.savefig('resilience_phase_space.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Saved: resilience_phase_space.png/pdf")
plt.show()

print("\n" + "=" * 70)
print("PHASE SPACE INTERPRETATION")
print("=" * 70)
print("Tight trajectories near center = High resilience")
print("Large loops = System takes time to recover (hysteresis)")
print("Trajectories returning to mean = Stable equilibrium")
print("Drift away from mean = System state shift")
print("=" * 70)