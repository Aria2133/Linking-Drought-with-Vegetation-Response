import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import glob

"""
PCA BIPLOT MAP
Shows principal drought gradients across California
PC1 vs PC2 mapped geographically
Arrows show which indices drive patterns
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("Creating PCA Biplot Map...")


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


# Load reference grid
print("Loading reference grid...")
ref_transform = None
for land_type in land_types:
    files = get_files(land_type, 'GNDVI')
    if files:
        with rasterio.open(files[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = (src.height, src.width)
            ref_bounds = src.bounds
            if hasattr(ref_bounds, 'left'):
                ref_bounds = (ref_bounds.left, ref_bounds.bottom, ref_bounds.right, ref_bounds.top)
            break

# Load temporal means for all indices
print("\nLoading data for all indices...")
index_data = {}

for index in indices:
    print(f"  {index}...")
    all_data = []

    for land_type in land_types:
        files = get_files(land_type, index)

        for filepath in files[:30]:  # Use first 30 files
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
                    all_data.append(data)
            except:
                continue

    if all_data:
        stacked = np.stack(all_data, axis=0)
        mean_data = np.nanmean(stacked, axis=0)
        index_data[index] = mean_data

if len(index_data) < 3:
    print("ERROR: Not enough indices with data!")
    exit(1)

print(f"Loaded {len(index_data)} indices")

# Prepare data matrix for PCA
print("\nPreparing data for PCA...")
n_rows, n_cols = ref_shape
n_pixels = n_rows * n_cols

# Stack all indices into matrix: (n_pixels, n_indices)
data_matrix = []
pixel_coords = []

for i in range(n_rows):
    for j in range(n_cols):
        pixel_values = []
        valid = True

        for index in indices:
            if index in index_data:
                val = index_data[index][i, j]
                if np.isnan(val):
                    valid = False
                    break
                pixel_values.append(val)
            else:
                valid = False
                break

        if valid:
            data_matrix.append(pixel_values)
            pixel_coords.append((i, j))

data_matrix = np.array(data_matrix)
print(f"Valid pixels: {len(data_matrix)}")

# Standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_matrix)

# Perform PCA
print("Running PCA...")
pca = PCA(n_components=2)
pca_scores = pca.fit_transform(data_scaled)

print(
    f"Variance explained: PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, PC2={pca.explained_variance_ratio_[1] * 100:.1f}%")

# Map PCA scores back to spatial grid
pc1_map = np.full(ref_shape, np.nan)
pc2_map = np.full(ref_shape, np.nan)

for idx, (i, j) in enumerate(pixel_coords):
    pc1_map[i, j] = pca_scores[idx, 0]
    pc2_map[i, j] = pca_scores[idx, 1]

# Create visualization
print("Creating PCA biplot map...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# PC1 Map
im1 = axes[0].imshow(pc1_map, cmap='RdBu_r', extent=ref_bounds, aspect='auto')
axes[0].set_title(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
axes[0].set_xlabel('Longitude (°E)', fontweight='bold')
axes[0].set_ylabel('Latitude (°N)', fontweight='bold')
plt.colorbar(im1, ax=axes[0], label='PC1 Score')

# PC2 Map
im2 = axes[1].imshow(pc2_map, cmap='RdBu_r', extent=ref_bounds, aspect='auto')
axes[1].set_title(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)',
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel('Longitude (°E)', fontweight='bold')
axes[1].set_ylabel('Latitude (°N)', fontweight='bold')
plt.colorbar(im2, ax=axes[1], label='PC2 Score')

# Biplot (loadings)
ax = axes[2]
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

# Plot arrows for each index
for idx, index in enumerate([i for i in indices if i in index_data]):
    ax.arrow(0, 0, loadings[idx, 0], loadings[idx, 1],
             head_width=0.08, head_length=0.08, fc='red', ec='red', linewidth=2)

    # Label
    ax.text(loadings[idx, 0] * 1.15, loadings[idx, 1] * 1.15, index,
            fontsize=11, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)',
              fontsize=12, fontweight='bold')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)',
              fontsize=12, fontweight='bold')
ax.set_title('PCA Loadings Biplot', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Set limits
max_load = np.max(np.abs(loadings)) * 1.3
ax.set_xlim(-max_load, max_load)
ax.set_ylim(-max_load, max_load)

plt.suptitle('PCA Analysis: Principal Drought Gradients in California',
             fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig('pca_biplot_map.png', dpi=300, bbox_inches='tight')
plt.savefig('pca_biplot_map.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: pca_biplot_map.png/pdf")
plt.show()

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("PC1 & PC2 Maps:")
print("  • Show principal drought gradients across California")
print("  • Red/Blue = different drought regimes")
print("  • Spatial patterns reveal dominant drought drivers")
print("\nLoadings Biplot:")
print("  • Arrows = contribution of each index to PCs")
print("  • Arrow length = importance")
print("  • Arrow direction = correlation pattern")
print("  • Arrows pointing same direction = correlated indices")
print("  • Arrows at 180° = negatively correlated")
print("=" * 70)