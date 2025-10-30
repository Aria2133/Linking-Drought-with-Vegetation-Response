import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import os
import glob

"""
CIRCULAR HEATMAPS - Seasonal Patterns
Each ring = one year (2001-2024)
360° = 12 months
Color = drought index value
Shows seasonal cycles and multi-year patterns
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
SELECTED_INDEX = 'SPEI'  # Change to: NDVI, GPP, VPD, etc.

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print(f"Creating Circular Heatmaps for {SELECTED_INDEX}...")


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
    elif index in ['GNDVI', 'NDVI']:
        data[(data < -1) | (data > 1.5)] = np.nan
    elif index == 'NIRv':
        data[(data < -0.1) | (data > 0.6)] = np.nan
    elif index == 'VPD':
        data[(data < 0) | (data > 10)] = np.nan
    elif index == 'GPP':
        data[(data < 0) | (data > 20)] = np.nan
    elif index == 'LST':
        data[(data < -40) | (data > 70)] = np.nan
    elif index == 'GOSIF':
        data[(data < -100) | (data > 3000)] = np.nan
    elif index == 'TWS':
        data[(data < -100) | (data > 100)] = np.nan

    return data


def extract_time_series(land_type, index):
    """Extract monthly time series"""
    files = get_files(land_type, index)

    if not files:
        return {}

    time_series = {}  # {(year, month): mean_value}

    for filepath in files:
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                data = clean_data(data, index)

                mean_val = np.nanmean(data)
                if np.isnan(mean_val):
                    continue

                # Extract date from filename
                filename = os.path.basename(filepath)
                try:
                    year = int(filename.split('_')[-2])
                    month = int(filename.split('_')[-1].split('.')[0])
                    time_series[(year, month)] = mean_val
                except:
                    continue
        except:
            continue

    return time_series


# Collect data for all land types
print("\nExtracting time series...")
all_data = {}

for land_type in land_types:
    ts = extract_time_series(land_type, SELECTED_INDEX)
    if ts:
        all_data[land_type] = ts
        print(f"  {land_type}: {len(ts)} time points")

if not all_data:
    print("ERROR: No data found!")
    exit(1)

# Organize data by year and month
years = sorted(list(set([y for lt_data in all_data.values() for (y, m) in lt_data.keys()])))
print(f"Years: {years[0]}-{years[-1]}")

# Create circular heatmaps
fig, axes = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
axes = axes.flatten()

for idx, land_type in enumerate(land_types):
    ax = axes[idx]

    if land_type not in all_data:
        ax.text(0, 0, f'{land_type}\nNo Data', ha='center', va='center')
        continue

    ts = all_data[land_type]

    # Create matrix: rows=years, cols=months
    year_month_matrix = np.full((len(years), 12), np.nan)

    for (year, month), value in ts.items():
        if year in years:
            year_idx = years.index(year)
            year_month_matrix[year_idx, month - 1] = value

    # Determine color scale
    if SELECTED_INDEX == 'SPEI':
        vmin, vmax = -2, 2
        cmap = 'RdYlGn'
    elif SELECTED_INDEX == 'VPD':
        vmin, vmax = np.nanpercentile(year_month_matrix, [5, 95])
        cmap = 'RdYlGn_r'
    elif SELECTED_INDEX in ['GNDVI', 'NDVI', 'GPP']:
        vmin, vmax = np.nanpercentile(year_month_matrix, [5, 95])
        cmap = 'RdYlGn'
    else:
        vmin, vmax = np.nanpercentile(year_month_matrix, [5, 95])
        cmap = 'viridis'

    # Plot each month as a wedge
    theta = np.linspace(0, 2 * np.pi, 13)  # 13 points for 12 months

    for year_idx, year in enumerate(years):
        r_inner = year_idx
        r_outer = year_idx + 1

        for month in range(12):
            value = year_month_matrix[year_idx, month]

            if np.isnan(value):
                continue

            # Normalize color
            color_val = (value - vmin) / (vmax - vmin)
            color_val = np.clip(color_val, 0, 1)

            # Get color from colormap
            from matplotlib import cm

            cmap_obj = cm.get_cmap(cmap)
            color = cmap_obj(color_val)

            # Create wedge
            theta1 = theta[month]
            theta2 = theta[month + 1]

            wedge = Wedge((0, 0), r_outer, np.degrees(theta1), np.degrees(theta2),
                          width=1, facecolor=color, edgecolor='white', linewidth=0.5)
            ax.add_patch(wedge)

    # Format polar plot
    ax.set_ylim(0, len(years))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(month_labels, fontsize=10)

    # Year labels (every 5 years)
    year_ticks = list(range(0, len(years), 5))
    ax.set_yticks(year_ticks)
    ax.set_yticklabels([years[i] for i in year_ticks], fontsize=8)

    ax.set_title(f'{land_type}\n{SELECTED_INDEX}', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

# Add colorbar
from matplotlib import cm

cmap_obj = cm.get_cmap(cmap if SELECTED_INDEX != 'VPD' else 'RdYlGn_r')
sm = plt.cm.ScalarMappable(cmap=cmap_obj,
                           norm=plt.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                    fraction=0.05, pad=0.05, aspect=40)
cbar.set_label(SELECTED_INDEX, fontsize=12, fontweight='bold')

plt.suptitle(f'Seasonal Patterns: {SELECTED_INDEX} Across California Land Types (2001-2024)',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(f'circular_heatmap_{SELECTED_INDEX}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'circular_heatmap_{SELECTED_INDEX}.pdf', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: circular_heatmap_{SELECTED_INDEX}.png/pdf")
plt.show()

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("• Each ring = one year (inner = earlier, outer = recent)")
print("• 360° circle = 12 months")
print("• Color = drought intensity")
print("• Look for:")
print("  - Seasonal patterns (repeating colors at same angle)")
print("  - Multi-year droughts (persistent red across rings)")
print("  - Recovery periods (transition from red to green)")
print("=" * 70)