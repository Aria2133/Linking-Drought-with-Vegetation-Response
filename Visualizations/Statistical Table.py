import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy import stats
import os
import glob

"""
COLORFUL STATISTICAL SUMMARY TABLE
Quantitative comparison of resilience metrics across land types
Includes significance testing and color-coded values
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
SELECTED_INDEX = 'NIRv'

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("=" * 70)
print("STATISTICAL SUMMARY TABLE")
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

    if index in ['NIRv']:
        data[(data < -0.1) | (data > 0.6)] = np.nan

    return data


# Load reference grid
ref_transform = None
for land_type in land_types:
    files = get_files(land_type, SELECTED_INDEX)
    if files:
        with rasterio.open(files[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = (src.height, src.width)
            break

# Calculate statistics for each land type
BASELINE = (2001, 2011)
DROUGHT = (2012, 2016)

stats_summary = {}

for land_type in land_types:
    print(f"\nProcessing {land_type}...")

    files = get_files(land_type, SELECTED_INDEX)

    baseline_data = []
    drought_data = []
    all_data = []

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
                all_data.append(data)

                if BASELINE[0] <= year <= BASELINE[1]:
                    baseline_data.append(data)
                elif DROUGHT[0] <= year <= DROUGHT[1]:
                    drought_data.append(data)
        except:
            continue

    if len(baseline_data) == 0 or len(drought_data) == 0:
        continue

    # Calculate metrics
    baseline_mean = np.nanmean(np.stack(baseline_data, axis=0), axis=0)
    drought_mean = np.nanmean(np.stack(drought_data, axis=0), axis=0)

    # Resistance
    resistance = 100 - np.abs((drought_mean - baseline_mean) / baseline_mean * 100)
    resistance = np.clip(resistance, 0, 100)

    # Stability (coefficient of variation)
    all_stacked = np.stack(all_data, axis=0)
    temporal_mean = np.nanmean(all_stacked, axis=0)
    temporal_std = np.nanstd(all_stacked, axis=0)
    cv = (temporal_std / temporal_mean) * 100
    stability = 100 - np.clip(cv, 0, 100)

    # Overall resilience
    overall = (resistance + stability) / 2

    # Flatten for statistics
    res_flat = resistance[~np.isnan(resistance)]
    stab_flat = stability[~np.isnan(stability)]
    overall_flat = overall[~np.isnan(overall)]

    # Calculate percentages
    high_res = np.sum(overall_flat > 80) / len(overall_flat) * 100
    mod_res = np.sum((overall_flat >= 60) & (overall_flat <= 80)) / len(overall_flat) * 100
    low_res = np.sum(overall_flat < 60) / len(overall_flat) * 100

    stats_summary[land_type] = {
        'mean_resilience': np.mean(overall_flat),
        'std_resilience': np.std(overall_flat),
        'mean_resistance': np.mean(res_flat),
        'mean_stability': np.mean(stab_flat),
        'high_percent': high_res,
        'moderate_percent': mod_res,
        'low_percent': low_res,
        'n_pixels': len(overall_flat),
        'resilience_values': overall_flat  # For ANOVA
    }

    print(f"  Mean resilience: {stats_summary[land_type]['mean_resilience']:.1f}%")

# Perform ANOVA
print("\n" + "=" * 70)
print("STATISTICAL SIGNIFICANCE TEST")
print("=" * 70)

groups = [stats_summary[lt]['resilience_values'] for lt in land_types if lt in stats_summary]
f_stat, p_value = stats.f_oneway(*groups)

print(f"One-way ANOVA:")
print(f"  F-statistic: {f_stat:.3f}")
print(f"  P-value: {p_value:.6f}")
if p_value < 0.001:
    print(f"  Result: HIGHLY SIGNIFICANT (p < 0.001) ***")
elif p_value < 0.01:
    print(f"  Result: VERY SIGNIFICANT (p < 0.01) **")
elif p_value < 0.05:
    print(f"  Result: SIGNIFICANT (p < 0.05) *")
else:
    print(f"  Result: NOT SIGNIFICANT (p >= 0.05)")

# Create colorful table
print("\nCreating colorful summary table...")

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.axis('off')

# Table data
columns = ['Land Type', 'Mean\nResilience\n(%)', 'Resistance\n(%)',
           'Stability\n(%)', 'High\nResilience\n(>80%)', 'Moderate\n(60-80%)',
           'Low\n(<60%)', 'Sample\nSize']

# Prepare data matrix
data = []
for lt in land_types:
    if lt in stats_summary:
        s = stats_summary[lt]
        data.append([
            lt,
            f"{s['mean_resilience']:.1f} ± {s['std_resilience']:.1f}",
            f"{s['mean_resistance']:.1f}",
            f"{s['mean_stability']:.1f}",
            f"{s['high_percent']:.1f}%",
            f"{s['moderate_percent']:.1f}%",
            f"{s['low_percent']:.1f}%",
            f"{s['n_pixels']:,}"
        ])

# Create table structure
n_rows = len(data) + 1  # +1 for header
n_cols = len(columns)
cell_height = 0.12
cell_width = 0.12


# Color functions
def get_resilience_color(value_str):
    """Color based on resilience value"""
    try:
        value = float(value_str.split('±')[0])
        if value >= 80:
            return '#d4edda'  # Light green
        elif value >= 70:
            return '#fff3cd'  # Light yellow
        else:
            return '#f8d7da'  # Light red
    except:
        return 'white'


def get_percentage_color(value_str):
    """Color based on percentage"""
    try:
        value = float(value_str.strip('%'))
        if value >= 60:
            return '#d4edda'  # Light green
        elif value >= 30:
            return '#fff3cd'  # Light yellow
        else:
            return '#f8d7da'  # Light red
    except:
        return 'white'


# Draw header
for col_idx, col_name in enumerate(columns):
    x = col_idx * cell_width
    y = 1 - cell_height

    # Header background
    rect = Rectangle((x, y), cell_width, cell_height,
                     facecolor='#4a90e2', edgecolor='black', linewidth=2)
    ax.add_patch(rect)

    # Header text
    ax.text(x + cell_width / 2, y + cell_height / 2, col_name,
            ha='center', va='center', fontsize=10, fontweight='bold',
            color='white', wrap=True)

# Draw data rows
for row_idx, row_data in enumerate(data):
    for col_idx, cell_value in enumerate(row_data):
        x = col_idx * cell_width
        y = 1 - (row_idx + 2) * cell_height

        # Determine cell color
        if col_idx == 0:  # Land type
            color = 'white'
        elif col_idx == 1:  # Mean resilience
            color = get_resilience_color(cell_value)
        elif col_idx in [2, 3]:  # Resistance, Stability
            color = get_resilience_color(cell_value)
        elif col_idx in [4, 5, 6]:  # Percentages
            color = get_percentage_color(cell_value)
        else:
            color = 'white'

        # Cell background
        rect = Rectangle((x, y), cell_width, cell_height,
                         facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(rect)

        # Cell text
        text_color = 'black' if col_idx != 0 else '#2c3e50'
        fontweight = 'bold' if col_idx == 0 else 'normal'

        ax.text(x + cell_width / 2, y + cell_height / 2, cell_value,
                ha='center', va='center', fontsize=10,
                fontweight=fontweight, color=text_color)

# Add significance note
sig_text = f"\n\nStatistical Test: One-way ANOVA\nF = {f_stat:.3f}, "
if p_value < 0.001:
    sig_text += f"p < 0.001 ***\nDifferences between land types are HIGHLY SIGNIFICANT"
elif p_value < 0.01:
    sig_text += f"p = {p_value:.4f} **\nDifferences between land types are VERY SIGNIFICANT"
elif p_value < 0.05:
    sig_text += f"p = {p_value:.4f} *\nDifferences between land types are SIGNIFICANT"
else:
    sig_text += f"p = {p_value:.4f}\nDifferences between land types are NOT significant"

ax.text(0.5, -0.15, sig_text, ha='center', fontsize=11,
        fontweight='bold', transform=ax.transAxes,
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Add legend for colors
legend_text = "Color Guide:\nGreen = High/Good  |  Yellow = Moderate  |  Red = Low/Concerning"
ax.text(0.5, 1.15, legend_text, ha='center', fontsize=10,
        transform=ax.transAxes, style='italic',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

# Set limits
ax.set_xlim(0, n_cols * cell_width)
ax.set_ylim(0, 1)

plt.title(f'Vegetation Resilience Summary Statistics by Land Type\n{SELECTED_INDEX}-based Analysis',
          fontsize=16, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('resilience_summary_table.png', dpi=300, bbox_inches='tight')
plt.savefig('resilience_summary_table.pdf', dpi=300, bbox_inches='tight')
print("\n✓ Saved: resilience_summary_table.png/pdf")
plt.show()

print("\n" + "=" * 70)
print("TABLE COMPLETE")
print("=" * 70)
print("Green cells = high values (good)")
print("Yellow cells = moderate values")
print("Red cells = low values (concerning)")
print("=" * 70)