import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
import itertools
from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings('ignore')

# Dataset paths
data_paths = {
    'GOSIF': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\GOSIF_CA_CLEAN",
    'LST': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\MODIS_CA_Monthly_LST",
    'NDVI': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\NDVI_California_NaturalOnly",
    'NIRv': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\NIRv_MOD13Q1_5km_TIFs",
    'SPEI': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\SPEI_Masked_Output",
    'TWS': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\TWS_Masked_Output"
}


def extract_date_from_filename(filename, dataset_name):
    """Extract date from filename based on dataset naming convention"""
    basename = os.path.basename(filename)

    if dataset_name == 'GOSIF':
        date_str = basename.split('_')[-1].split('.')[0]
        if len(date_str) == 6:
            year = int(date_str[:4])
            month = int(date_str[4:6])
        else:
            return None

    elif dataset_name == 'LST':
        parts = basename.split('_')
        if len(parts) >= 5:
            year = int(parts[-2])
            month = int(parts[-1].split('.')[0])
        else:
            return None

    elif dataset_name in ['NDVI', 'NIRv']:
        parts = basename.split('_')
        if len(parts) >= 6:
            year = int(parts[-2])
            month = int(parts[-1].split('.')[0])
        else:
            return None

    elif dataset_name == 'SPEI':
        parts = basename.split('_')
        if len(parts) >= 5:
            year = int(parts[-3])
            month = int(parts[-2])
        else:
            return None

    elif dataset_name == 'TWS':
        parts = basename.split('_')
        if len(parts) >= 5:
            year = int(parts[-3])
            month = int(parts[-2])
        else:
            return None

    return datetime(year, month, 1)


def load_dataset_as_array(dataset_name, data_path):
    """Load all TIF files for a dataset into a 3D array (time, height, width)"""
    print(f"Loading {dataset_name}...")

    # Get all TIF files
    tif_files = glob.glob(os.path.join(data_path, "*.tif*"))

    if not tif_files:
        print(f"No TIF files found in {data_path}")
        return None, None, None, None

    # Filter files by date and sort
    valid_files = []
    for tif_file in tif_files:
        date = extract_date_from_filename(tif_file, dataset_name)
        if date and 2001 <= date.year <= 2021:
            valid_files.append((date, tif_file))

    valid_files.sort(key=lambda x: x[0])
    print(f"Found {len(valid_files)} valid files for {dataset_name}")

    if not valid_files:
        return None, None, None, None

    # Read first file to get dimensions and geospatial info
    with rasterio.open(valid_files[0][1]) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs = src.crs
        bounds = src.bounds

        # Get coordinate arrays
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows, cols)
        lon_array = np.array(xs)
        lat_array = np.array(ys)

    # Initialize 3D array
    data_array = np.full((len(valid_files), height, width), np.nan)
    dates = []

    # Load all files
    for i, (date, filepath) in enumerate(valid_files):
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1).astype(np.float64)
                # Replace zeros with NaN for correlation calculation
                data[data == 0] = np.nan
                data_array[i, :, :] = data
                dates.append(date)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    print(f"Loaded {len(dates)} files for {dataset_name}")
    return data_array, dates, (lon_array, lat_array), (transform, crs, bounds)


def calculate_pixel_correlation(array1, array2):
    """Calculate correlation between two time series arrays pixel by pixel"""
    print("Calculating pixel-wise correlations...")

    # Get dimensions
    n_times, height, width = array1.shape

    # Initialize correlation array
    correlation_array = np.full((height, width), np.nan)
    p_value_array = np.full((height, width), np.nan)

    # Calculate correlation for each pixel
    valid_pixels = 0
    total_pixels = height * width

    for i in range(height):
        for j in range(width):
            # Get time series for this pixel from both datasets
            ts1 = array1[:, i, j]
            ts2 = array2[:, i, j]

            # Remove NaN values
            valid_mask = ~(np.isnan(ts1) | np.isnan(ts2))

            if np.sum(valid_mask) >= 10:  # Need at least 10 valid observations
                ts1_clean = ts1[valid_mask]
                ts2_clean = ts2[valid_mask]

                try:
                    corr, p_val = stats.pearsonr(ts1_clean, ts2_clean)
                    correlation_array[i, j] = corr
                    p_value_array[i, j] = p_val
                    valid_pixels += 1
                except:
                    continue

        # Progress indicator
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{height} rows...")

    print(f"Valid correlations calculated for {valid_pixels}/{total_pixels} pixels")
    return correlation_array, p_value_array


def create_correlation_map(corr_array, p_array, coords, dataset1, dataset2, geo_info, common_shape):
    """Create a correlation heatmap with geographic coordinates"""
    lon_array, lat_array = coords
    transform, crs, bounds = geo_info

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Custom colormap for correlations (blue-white-red)
    colors_corr = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
                   '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
    n_bins = 100
    cmap_corr = LinearSegmentedColormap.from_list('correlation', colors_corr, N=n_bins)

    # Plot 1: Correlation values
    im1 = ax1.imshow(corr_array, cmap=cmap_corr, vmin=-1, vmax=1,
                     extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                     origin='upper', interpolation='bilinear')
    ax1.set_title(f'Correlation: {dataset1} vs {dataset2}', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Longitude (°)', fontsize=10)
    ax1.set_ylabel('Latitude (°)', fontsize=10)

    # Add colorbar for correlations
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
    cbar1.set_label('Correlation Coefficient', fontsize=10)

    # Plot significance contours using extent-based approach
    try:
        significance_mask = (p_array < 0.05).astype(float)
        significance_mask[np.isnan(p_array)] = np.nan

        # Only add contours if there are significant pixels
        if np.nansum(significance_mask) > 0:
            # Create contour using imshow extent
            CS = ax1.contour(significance_mask, levels=[0.5],
                             extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                             origin='upper', colors='black', linewidths=1.5, alpha=0.8)

            # Add contour labels
            if hasattr(CS, 'levels') and len(CS.levels) > 0:
                ax1.clabel(CS, CS.levels, inline=True, fontsize=8, fmt='p<0.05')

    except Exception as e:
        print(f"  Warning: Could not add significance contours: {e}")

    # Plot 2: P-values
    im2 = ax2.imshow(p_array, cmap='viridis_r', vmin=0, vmax=1,
                     extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                     origin='upper', interpolation='bilinear')
    ax2.set_title(f'P-values: {dataset1} vs {dataset2}', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Longitude (°)', fontsize=10)
    ax2.set_ylabel('Latitude (°)', fontsize=10)

    # Add colorbar for p-values
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
    cbar2.set_label('P-value', fontsize=10)

    # Add significance contour for p-values
    try:
        CS2 = ax2.contour(p_array, levels=[0.05],
                          extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                          origin='upper', colors='red', linewidths=2, alpha=0.8)

        if hasattr(CS2, 'levels') and len(CS2.levels) > 0:
            ax2.clabel(CS2, CS2.levels, inline=True, fontsize=8, fmt='p=0.05')

    except Exception as e:
        print(f"  Warning: Could not add p-value contours: {e}")

    # Calculate and display statistics
    valid_corr = corr_array[~np.isnan(corr_array)]
    if len(valid_corr) > 0:
        mean_corr = np.mean(valid_corr)
        median_corr = np.median(valid_corr)
        significant_pixels = np.sum(p_array < 0.05)
        total_pixels = np.sum(~np.isnan(p_array))
        sig_percentage = (significant_pixels / total_pixels * 100) if total_pixels > 0 else 0

        stats_text = f'Mean Corr: {mean_corr:.3f}\nMedian Corr: {median_corr:.3f}\nSignificant: {sig_percentage:.1f}%\nValid pixels: {total_pixels:,}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=9)

    plt.tight_layout()

    # Save individual correlation map
    filename = f'correlation_map_{dataset1}_vs_{dataset2}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    return corr_array, p_array


def main():
    # Load all datasets
    all_datasets = {}
    geo_infos = {}
    coordinate_arrays = {}

    print("Loading all datasets...")
    for dataset_name, path in data_paths.items():
        data_array, dates, coords, geo_info = load_dataset_as_array(dataset_name, path)
        if data_array is not None:
            all_datasets[dataset_name] = {
                'data': data_array,
                'dates': dates,
                'coords': coords,
                'geo_info': geo_info
            }
            geo_infos[dataset_name] = geo_info
            coordinate_arrays[dataset_name] = coords
            print(f"{dataset_name}: {data_array.shape} (time, height, width)")

    if len(all_datasets) < 2:
        print("Need at least 2 datasets for correlation analysis!")
        return

    # Generate all possible pairs of datasets
    dataset_names = list(all_datasets.keys())
    dataset_pairs = list(itertools.combinations(dataset_names, 2))

    print(f"\nGenerating {len(dataset_pairs)} correlation maps...")

    correlation_results = {}

    # Create correlation maps for all pairs
    for i, (dataset1, dataset2) in enumerate(dataset_pairs):
        print(f"\n{i + 1}/{len(dataset_pairs)}: Analyzing {dataset1} vs {dataset2}")

        # Get data arrays
        array1 = all_datasets[dataset1]['data']
        array2 = all_datasets[dataset2]['data']

        # Ensure arrays have same temporal coverage
        dates1 = all_datasets[dataset1]['dates']
        dates2 = all_datasets[dataset2]['dates']

        # Find common dates
        common_dates = sorted(set(dates1) & set(dates2))
        print(f"  Common time points: {len(common_dates)}")

        if len(common_dates) < 10:
            print(f"  Insufficient common dates for {dataset1} vs {dataset2}")
            continue

        # Extract data for common dates
        idx1 = [dates1.index(date) for date in common_dates if date in dates1]
        idx2 = [dates2.index(date) for date in common_dates if date in dates2]

        common_array1 = array1[idx1, :, :]
        common_array2 = array2[idx2, :, :]

        # Handle spatial dimension mismatch
        if common_array1.shape[1:] != common_array2.shape[1:]:
            print(f"  Spatial dimension mismatch: {common_array1.shape[1:]} vs {common_array2.shape[1:]}")
            # Find overlapping dimensions
            min_height = min(common_array1.shape[1], common_array2.shape[1])
            min_width = min(common_array1.shape[2], common_array2.shape[2])
            common_array1 = common_array1[:, :min_height, :min_width]
            common_array2 = common_array2[:, :min_height, :min_width]
            print(f"  Resized to common dimensions: {common_array1.shape[1:]}")
            common_shape = (min_height, min_width)
        else:
            common_shape = common_array1.shape[1:]

        # Calculate pixel-wise correlations
        corr_array, p_array = calculate_pixel_correlation(common_array1, common_array2)

        # Use geo info from first dataset and adjust for common shape
        geo_info1 = geo_infos[dataset1]
        coords1 = coordinate_arrays[dataset1]

        # Store results
        correlation_results[f"{dataset1}_vs_{dataset2}"] = {
            'correlation': corr_array,
            'p_values': p_array,
            'coords': coords1,
            'geo_info': geo_info1,
            'common_shape': common_shape
        }

        # Create correlation map
        create_correlation_map(corr_array, p_array, coords1,
                               dataset1, dataset2, geo_info1, common_shape)

    # Create summary correlation matrix
    print("\nCreating summary correlation matrix...")
    create_summary_correlation_matrix(correlation_results, dataset_names)

    # Save correlation results to files
    print("\nSaving correlation arrays...")
    save_correlation_results(correlation_results)

    print(f"\nAnalysis complete! Generated {len(correlation_results)} correlation maps.")
    return correlation_results


def create_summary_correlation_matrix(correlation_results, dataset_names):
    """Create a summary matrix showing mean correlations between all datasets"""
    n_datasets = len(dataset_names)
    summary_matrix = np.full((n_datasets, n_datasets), np.nan)

    # Fill diagonal with 1s
    np.fill_diagonal(summary_matrix, 1.0)

    # Fill off-diagonal with mean correlations
    for key, results in correlation_results.items():
        dataset1, dataset2 = key.split('_vs_')
        idx1 = dataset_names.index(dataset1)
        idx2 = dataset_names.index(dataset2)

        mean_corr = np.nanmean(results['correlation'])
        summary_matrix[idx1, idx2] = mean_corr
        summary_matrix[idx2, idx1] = mean_corr  # Symmetric

    # Plot summary matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(summary_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    # Add text annotations
    for i in range(n_datasets):
        for j in range(n_datasets):
            if not np.isnan(summary_matrix[i, j]):
                text = ax.text(j, i, f'{summary_matrix[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')

    # Customize plot
    ax.set_xticks(range(n_datasets))
    ax.set_yticks(range(n_datasets))
    ax.set_xticklabels(dataset_names, rotation=45, ha='right')
    ax.set_yticklabels(dataset_names)
    ax.set_title('Mean Spatial Correlation Matrix\n(Average correlation across all pixels)',
                 fontsize=14, fontweight='bold', pad=20)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
    cbar.set_label('Mean Correlation Coefficient', fontsize=12)

    plt.tight_layout()
    plt.savefig('summary_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


def save_correlation_results(correlation_results):
    """Save correlation arrays as GeoTIFF files"""
    for pair_name, results in correlation_results.items():
        # Save correlation array
        corr_filename = f'correlation_{pair_name}.tif'

        # Save p-value array
        pval_filename = f'pvalues_{pair_name}.tif'

        transform, crs, bounds = results['geo_info']

        # Create a new transform for the common shape if dimensions changed
        height, width = results['correlation'].shape
        if 'common_shape' in results:
            # Adjust transform for the new dimensions
            x_res = (bounds.right - bounds.left) / width
            y_res = (bounds.top - bounds.bottom) / height
            from rasterio.transform import from_bounds
            transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, width, height)

        # Save correlation as GeoTIFF
        try:
            with rasterio.open(
                    corr_filename, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=results['correlation'].dtype,
                    crs=crs,
                    transform=transform,
            ) as dst:
                dst.write(results['correlation'], 1)

            # Save p-values as GeoTIFF
            with rasterio.open(
                    pval_filename, 'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=1,
                    dtype=results['p_values'].dtype,
                    crs=crs,
                    transform=transform,
            ) as dst:
                dst.write(results['p_values'], 1)

            print(f"  Saved: {corr_filename} and {pval_filename}")

        except Exception as e:
            print(f"  Error saving {pair_name}: {e}")

    print(f"Completed saving GeoTIFF files")


if __name__ == "__main__":
    results = main()