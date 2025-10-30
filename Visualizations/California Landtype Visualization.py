import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patheffects
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# File paths
files = {
    'Grasslands': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Grasslands_2021.tif",
    'Natural Forests': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
    'Savannas': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Savannas_2021.tif",
    'Shrublands': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Shrublands_2021.tif"
}

# Define distinct colors for each land type (publication quality)
colors = {
    'Grasslands': '#8FBC8F',  # Medium sea green
    'Natural Forests': '#2D5016',  # Dark forest green
    'Savannas': '#DAA520',  # Goldenrod
    'Shrublands': '#CD853F'  # Peru/sandy brown
}

# Color indices
color_indices = {
    'Grasslands': 1,
    'Natural Forests': 2,
    'Savannas': 3,
    'Shrublands': 4
}

# Read the first file to get dimensions and geotransform
with rasterio.open(files['Grasslands']) as src:
    bounds = src.bounds
    # Handle bounds as either BoundingBox object or tuple
    if hasattr(bounds, 'left'):
        left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    else:
        left, bottom, right, top = bounds

    transform = src.transform
    crs = src.crs
    height = src.height
    width = src.width

    # Create coordinate arrays
    x_coords = np.linspace(left, right, width)
    y_coords = np.linspace(bottom, top, height)

print(f"Map bounds: {left:.2f}°E to {right:.2f}°E, {bottom:.2f}°N to {top:.2f}°N")
print(f"Dimensions: {width} x {height} pixels")

# Initialize combined array
combined = np.zeros((height, width), dtype=np.uint8)

# Read and combine all land type masks
print("\nReading land type masks...")
for land_type, filepath in files.items():
    with rasterio.open(filepath) as src:
        data = src.read(1)
        # Where this land type exists (value = 1), assign its color index
        mask = data == 1
        combined[mask] = color_indices[land_type]
        print(f"{land_type}: {np.sum(mask)} pixels")

# Create figure with high quality
fig, ax = plt.subplots(figsize=(12, 14), dpi=150)

# Create custom colormap
from matplotlib.colors import ListedColormap, BoundaryNorm

colors_list = ['white'] + [colors[lt] for lt in ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']]
cmap = ListedColormap(colors_list)
bounds_norm = [0, 1, 2, 3, 4, 5]
norm = BoundaryNorm(bounds_norm, cmap.N)

# Plot the land types
im = ax.imshow(combined, cmap=cmap, norm=norm,
               extent=[left, right, bottom, top],
               interpolation='nearest', aspect='auto')

# Format axes with coordinates
ax.set_xlabel('Longitude (°E)', fontsize=13, fontweight='bold')
ax.set_ylabel('Latitude (°N)', fontsize=13, fontweight='bold')
ax.tick_params(labelsize=11)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Create legend with patches
legend_patches = []
for land_type in ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']:
    patch = mpatches.Patch(facecolor=colors[land_type],
                           edgecolor='black',
                           linewidth=1.5,
                           label=land_type)
    legend_patches.append(patch)

legend = ax.legend(handles=legend_patches,
                   loc='upper right',
                   fontsize=12,
                   frameon=True,
                   fancybox=True,
                   shadow=True,
                   framealpha=0.95,
                   edgecolor='black',
                   title='Land Types',
                   title_fontsize=13)
legend.get_title().set_fontweight('bold')

# Add north arrow
arrow_x = 0.92
arrow_y = 0.55
ax.annotate('N', xy=(arrow_x, arrow_y), xytext=(arrow_x, arrow_y - 0.05),
            xycoords='axes fraction',
            fontsize=16, fontweight='bold',
            ha='center', va='bottom',
            arrowprops=dict(arrowstyle='->', lw=2.5, color='black'))

# Add scale bar (approximate)
# Calculate scale: distance per pixel
x_range = right - left
y_range = top - bottom
km_per_degree_lon = 111.0 * np.cos(np.radians((top + bottom) / 2))
km_per_degree_lat = 111.0

# Add scale bar manually
scale_length_km = 200  # 200 km scale bar
scale_length_deg = scale_length_km / km_per_degree_lon

scale_x_start = left + (right - left) * 0.05
scale_y = bottom + (top - bottom) * 0.05

ax.plot([scale_x_start, scale_x_start + scale_length_deg],
        [scale_y, scale_y],
        'k-', linewidth=4, solid_capstyle='butt')
ax.plot([scale_x_start, scale_x_start],
        [scale_y - 0.1, scale_y + 0.1],
        'k-', linewidth=2)
ax.plot([scale_x_start + scale_length_deg, scale_x_start + scale_length_deg],
        [scale_y - 0.1, scale_y + 0.1],
        'k-', linewidth=2)

# Add scale bar text
ax.text(scale_x_start + scale_length_deg / 2, scale_y + 0.3,
        f'{scale_length_km} km',
        ha='center', va='bottom',
        fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='black', alpha=0.9))

# Calculate areas
pixel_area_km2 = (x_range / width * km_per_degree_lon) * (y_range / height * km_per_degree_lat)
total_pixels = (combined > 0).sum()
total_area_km2 = total_pixels * pixel_area_km2

# Add statistics text box - positioned below the legend
stats_text = f"Total Area: {total_area_km2:,.0f} km²\n"
for land_type in ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']:
    pixels = (combined == color_indices[land_type]).sum()
    area = pixels * pixel_area_km2
    percentage = (pixels / total_pixels * 100) if total_pixels > 0 else 0
    stats_text += f"{land_type}: {area:,.0f} km² ({percentage:.1f}%)\n"

ax.text(0.98, 0.73, stats_text.strip(),
        transform=ax.transAxes,
        fontsize=10,
        horizontalalignment='right',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.8',
                  facecolor='white',
                  edgecolor='black',
                  alpha=0.95,
                  linewidth=1.5))

plt.title('Spatial Distribution of Land Types in California (2021)',
          fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('california_landtypes_map.png', dpi=300, bbox_inches='tight')
plt.savefig('california_landtypes_map.pdf', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("SPATIAL DISTRIBUTION SUMMARY")
print("=" * 70)
print(f"\nTotal mapped area: {total_area_km2:,.0f} km²")
print("\nBreakdown by land type:")
for land_type in ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']:
    pixels = (combined == color_indices[land_type]).sum()
    area = pixels * pixel_area_km2
    percentage = (pixels / total_pixels * 100) if total_pixels > 0 else 0
    print(f"  {land_type:20s}: {area:10,.0f} km² ({percentage:5.1f}%)")

print("\nMap saved as 'california_landtypes_map.png' and '.pdf'")