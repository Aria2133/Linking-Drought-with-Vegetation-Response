import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import rasterio
from rasterio.warp import reproject, Resampling

### Read TIFF files using rasterio
with rasterio.open(r"C:\Users\LOQ\OneDrive\Desktop\Zahra_NEW\albedo_python.tif") as src_albedo:
    albedo = src_albedo.read(1)
    albedo_meta = src_albedo.meta

with rasterio.open(r"C:\Users\LOQ\OneDrive\Desktop\Zahra_NEW\LST_PYTHON_CORRECT.tif") as src_lst:
    lst = src_lst.read(1)
    lst_meta = src_lst.meta

    # Resize LST to match Albedo
    lst_resized = np.empty_like(albedo)
    reproject(
        source=lst,
        destination=lst_resized,
        src_transform=src_lst.transform,
        src_crs=src_lst.crs,
        dst_transform=src_albedo.transform,
        dst_crs=src_albedo.crs,
        resampling=Resampling.bilinear
    )

### Flatten arrays
albedo_flat = albedo.flatten()
lst_flat = lst_resized.flatten()

### Remove NaNs
mask = ~np.isnan(albedo_flat) & ~np.isnan(lst_flat)
albedo_clean = albedo_flat[mask]
lst_clean = lst_flat[mask]

### Outlier Removal
mean_lst = np.mean(lst_clean)
std_lst = np.std(lst_clean)
outlier_mask = (lst_clean >= mean_lst - 3 * std_lst) & (lst_clean <= mean_lst + 3 * std_lst)
albedo_clean = albedo_clean[outlier_mask]
lst_clean = lst_clean[outlier_mask]

### Bin-based dry/wet edge computation
n = 50
albedo_bins = np.linspace(albedo_clean.min(), albedo_clean.max(), n)
dry_edge = []
wet_edge = []

for i in range(n - 1):
    bin_mask = (albedo_clean >= albedo_bins[i]) & (albedo_clean < albedo_bins[i + 1])
    if np.sum(bin_mask) > 10:
        dry_edge.append([albedo_bins[i], np.percentile(lst_clean[bin_mask], 95)])
        wet_edge.append([albedo_bins[i], np.percentile(lst_clean[bin_mask], 5)])

dry_edge = np.array(dry_edge)
wet_edge = np.array(wet_edge)

### Linear regression
dry_model = LinearRegression().fit(dry_edge[:, 0].reshape(-1, 1), dry_edge[:, 1])
wet_model = LinearRegression().fit(wet_edge[:, 0].reshape(-1, 1), wet_edge[:, 1])

### Plotting
x_range = np.linspace(albedo_clean.min(), albedo_clean.max())
dry_line = dry_model.predict(x_range.reshape(-1, 1))
wet_line = wet_model.predict(x_range.reshape(-1, 1))

plt.figure(figsize=(14, 8))
plt.scatter(albedo_clean, lst_clean, c='grey', alpha=0.3)
plt.scatter(dry_edge[:, 0], dry_edge[:, 1], c='red', s=80, marker='X')
plt.scatter(wet_edge[:, 0], wet_edge[:, 1], c='blue', s=80, marker='o', label='Wet point (5th)')
plt.plot(x_range, dry_line, 'r--', lw=3, label='Dry edge')
plt.plot(x_range, wet_line, 'b--', lw=3, label='Wet edge')

plt.xlabel('Albedo', fontsize=20)
plt.ylabel('Land Surface Temperature (K)', fontsize=20)
plt.text(dry_edge[-1, 0] + 0.01, dry_edge[-1, 1], 'Dry Edge', color='red', fontsize=20)
plt.text(wet_edge[-1, 0] + 0.01, wet_edge[-1, 1], 'Wet Edge', color='blue', fontsize=20)
plt.text(0.05, 0.1,
         f'Dry edge: LST = {dry_model.coef_[0]:.2f} × Albedo + {dry_model.intercept_:.2f}\n'
         f'Wet edge: LST = {wet_model.coef_[0]:.2f} × Albedo + {wet_model.intercept_:.2f}',
         transform=plt.gca().transAxes,
         fontsize=18,
         bbox=dict(facecolor='white'))
plt.legend()
plt.show()

### Print to console
print(f"Dry edge equation: LST = {dry_model.coef_[0]:.2f} × Albedo + {dry_model.intercept_:.2f}")
print(f"Wet edge equation: LST = {wet_model.coef_[0]:.2f} × Albedo + {wet_model.intercept_:.2f}")
