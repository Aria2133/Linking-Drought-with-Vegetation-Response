import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy import stats
from scipy.interpolate import griddata
import xarray as xr
from rasterio.plot import show
from matplotlib.colors import LinearSegmentedColormap
import os
from datetime import datetime
from pygam import LinearGAM, s
from itertools import combinations
import re
from tqdm import tqdm
from scipy.ndimage import zoom

warnings.filterwarnings('ignore')


class DroughtAnalysis:
    def __init__(self):
        self.datasets = {
            'GOSIF': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\GOSIF_California_Clipped",
            'LST': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\MODIS_CA_Monthly_LST",
            'NDVI': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\NDVI_MOD13Q1_5km_TIFs",
            'NIRv': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\NIRv_MOD13Q1_5km_TIFs",
            'SPEI': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\SPEI1_CSIC_TIFs",
            'TWS': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\TWS_GRACE_TIFs"
        }

        self.data_arrays = {}
        self.filled_data = {}
        self.aligned_data = {}

    def load_datasets(self):
        """Load all TIF files for each dataset"""
        print("Loading datasets...")

        for dataset_name, path in self.datasets.items():
            print(f"Loading {dataset_name}...")

            # Get all TIF files in the directory
            tif_files = glob.glob(os.path.join(path, "*.tif"))

            if not tif_files:
                print(f"No TIF files found in {path}")
                continue

            # Sort files chronologically
            tif_files.sort()

            data_list = []
            dates = []
            reference_profile = None

            for file_path in tif_files:
                try:
                    with rasterio.open(file_path) as src:
                        data = src.read(1)
                        # Store reference profile for later use
                        if reference_profile is None:
                            reference_profile = src.profile

                        # Convert no-data values to NaN
                        data = data.astype(np.float64)
                        if src.nodata is not None:
                            data[data == src.nodata] = np.nan
                        data_list.append(data)

                        # Extract date from filename
                        filename = os.path.basename(file_path)
                        date = self.extract_date_from_filename(filename, dataset_name)
                        if date:
                            dates.append(date)
                        else:
                            # Skip this file if date extraction failed
                            data_list.pop()
                            continue

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    continue

            if data_list:
                # Stack all arrays along time dimension
                self.data_arrays[dataset_name] = {
                    'data': np.stack(data_list, axis=0),
                    'dates': dates,
                    'shape': data_list[0].shape,
                    'profile': reference_profile
                }
                print(f"{dataset_name}: Loaded {len(data_list)} time steps, Shape: {data_list[0].shape}")

    def extract_date_from_filename(self, filename, dataset_name):
        """Extract date from filename based on naming pattern"""
        try:
            if dataset_name == 'GOSIF':
                # Format: CALIFORNIA_GOSIF_200101.tif
                match = re.search(r'(\d{6})\.tif$', filename)
                if match:
                    date_str = match.group(1)
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    return datetime(year, month, 1)

            elif dataset_name == 'LST':
                # Format: MYD21C1_CA_LST_2003_01.tif
                match = re.search(r'(\d{4})_(\d{2})\.tif$', filename)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    return datetime(year, month, 1)

            elif dataset_name in ['NDVI', 'NIRv']:
                # Format: NDVI_MOD13Q1_CA_5km_2001_01.tif or NIRv_MOD13Q1_CA_5km_2001_01.tif
                match = re.search(r'(\d{4})_(\d{2})\.tif$', filename)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    return datetime(year, month, 1)

            elif dataset_name == 'SPEI':
                # Format: SPEI1_CSIC_CA_2001_01.tif
                match = re.search(r'(\d{4})_(\d{2})\.tif$', filename)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    return datetime(year, month, 1)

            elif dataset_name == 'TWS':
                # Format: TWS_GRACE_CA_2002_04.tif
                match = re.search(r'(\d{4})_(\d{2})\.tif$', filename)
                if match:
                    year = int(match.group(1))
                    month = int(match.group(2))
                    return datetime(year, month, 1)

            return None

        except Exception as e:
            print(f"Error extracting date from {filename}: {e}")
            return None

    def fill_missing_data_vectorized(self):
        """Fill missing data using optimized vectorized approach"""
        print("Filling missing data using optimized method...")

        for dataset_name, dataset in self.data_arrays.items():
            print(f"Processing {dataset_name}...")

            data = dataset['data'].copy()
            dates = pd.DatetimeIndex(dataset['dates'])
            n_time, n_rows, n_cols = data.shape

            # Create a 3D mask for missing values
            missing_mask = np.isnan(data)

            # If no missing data, skip processing
            if not missing_mask.any():
                self.filled_data[dataset_name] = dataset.copy()
                self.filled_data[dataset_name]['data'] = data
                print(f"{dataset_name}: No missing data to fill")
                continue

            filled_data = data.copy()

            # Convert to DataFrame for easier date operations
            df_dates = pd.DataFrame({
                'date': dates,
                'year': dates.year,
                'month': dates.month,
                'time_idx': np.arange(len(dates))
            })

            # Process each time step that has missing data
            missing_time_steps = np.where(missing_mask.any(axis=(1, 2)))[0]

            print(f"Processing {len(missing_time_steps)} time steps with missing data...")

            for t_idx in tqdm(missing_time_steps, desc=f"Filling {dataset_name}"):
                if not missing_mask[t_idx].any():
                    continue

                target_year = df_dates.iloc[t_idx]['year']
                target_month = df_dates.iloc[t_idx]['month']

                # Step 1: Adjacent years average (vectorized)
                prev_year_idx = df_dates[(df_dates['year'] == target_year - 1) &
                                         (df_dates['month'] == target_month)].index
                next_year_idx = df_dates[(df_dates['year'] == target_year + 1) &
                                         (df_dates['month'] == target_month)].index

                if len(prev_year_idx) > 0 and len(next_year_idx) > 0:
                    prev_data = filled_data[prev_year_idx[0]]
                    next_data = filled_data[next_year_idx[0]]

                    # Fill where both adjacent years have valid data
                    valid_adjacent = ~(np.isnan(prev_data) | np.isnan(next_data))
                    fill_mask = missing_mask[t_idx] & valid_adjacent

                    if fill_mask.any():
                        filled_data[t_idx][fill_mask] = (prev_data[fill_mask] + next_data[fill_mask]) / 2

                # Step 2: Monthly climatology for remaining missing pixels
                still_missing = missing_mask[t_idx] & np.isnan(filled_data[t_idx])
                if still_missing.any():
                    # Get all data for this month across all years (vectorized)
                    same_month_indices = df_dates[df_dates['month'] == target_month]['time_idx'].values
                    same_month_data = filled_data[same_month_indices]

                    # Calculate climatology (mean across years for each pixel)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        monthly_climatology = np.nanmean(same_month_data, axis=0)

                    # Fill remaining missing values with climatology
                    valid_climatology = ~np.isnan(monthly_climatology)
                    final_fill_mask = still_missing & valid_climatology

                    if final_fill_mask.any():
                        filled_data[t_idx][final_fill_mask] = monthly_climatology[final_fill_mask]

            # Final check: fill any remaining NaN with overall mean for each pixel (vectorized)
            remaining_nan_mask = np.isnan(filled_data)
            if remaining_nan_mask.any():
                print("Filling remaining NaN values with pixel means...")

                # Calculate pixel-wise means (ignoring NaN)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    pixel_means = np.nanmean(filled_data, axis=0)

                # For each pixel that still has NaN values
                pixels_with_nan = remaining_nan_mask.any(axis=0)

                for i in range(n_rows):
                    for j in range(n_cols):
                        if pixels_with_nan[i, j] and not np.isnan(pixel_means[i, j]):
                            nan_times = remaining_nan_mask[:, i, j]
                            filled_data[nan_times, i, j] = pixel_means[i, j]

            self.filled_data[dataset_name] = dataset.copy()
            self.filled_data[dataset_name]['data'] = filled_data

            # Print statistics
            original_missing = np.sum(missing_mask)
            final_missing = np.sum(np.isnan(filled_data))
            filled_count = original_missing - final_missing
            print(f"{dataset_name}: Filled {filled_count:,} missing values "
                  f"({original_missing:,} -> {final_missing:,})")

    def create_common_time_index(self):
        """Create a common time index for all datasets (2001-01 to 2021-12)"""
        print("Creating common time index...")

        # Create monthly time index from 2001-01 to 2021-12
        start_date = datetime(2001, 1, 1)
        end_date = datetime(2021, 12, 1)
        common_dates = pd.date_range(start_date, end_date, freq='MS')

        print(
            f"Common time index: {len(common_dates)} months from {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}")

        return common_dates

    def align_temporal_data(self):
        """Align all datasets to common time index using optimized approach"""
        print("Aligning temporal data...")

        common_dates = self.create_common_time_index()

        for dataset_name, dataset in self.filled_data.items():
            print(f"Aligning {dataset_name}...")

            data = dataset['data']
            dates = pd.DatetimeIndex(dataset['dates'])
            n_rows, n_cols = dataset['shape']

            # Create aligned data array
            aligned_data = np.full((len(common_dates), n_rows, n_cols), np.nan)

            # Create mapping from dates to common_dates indices
            date_to_idx = {date: idx for idx, date in enumerate(common_dates)}

            # Map existing data to common time index (vectorized)
            for i, date in enumerate(dates):
                if date in date_to_idx:
                    common_idx = date_to_idx[date]
                    aligned_data[common_idx] = data[i]

            # Fill missing time steps using simplified approach
            missing_time_indices = np.where(np.all(np.isnan(aligned_data), axis=(1, 2)))[0]

            if len(missing_time_indices) > 0:
                print(f"Filling {len(missing_time_indices)} missing time steps...")

                for common_idx in tqdm(missing_time_indices, desc="Filling time steps"):
                    target_date = common_dates[common_idx]
                    target_year = target_date.year
                    target_month = target_date.month

                    # Find all same-month data that exists
                    same_month_indices = []
                    for idx, date in enumerate(common_dates):
                        if (date.month == target_month and
                                not np.all(np.isnan(aligned_data[idx]))):
                            same_month_indices.append(idx)

                    if same_month_indices:
                        # Use mean of all available same-month data (vectorized)
                        same_month_data = aligned_data[same_month_indices]
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            monthly_mean = np.nanmean(same_month_data, axis=0)

                        # Only fill where we have valid monthly mean
                        valid_mask = ~np.isnan(monthly_mean)
                        aligned_data[common_idx][valid_mask] = monthly_mean[valid_mask]

            self.aligned_data[dataset_name] = {
                'data': aligned_data,
                'dates': common_dates.tolist(),
                'shape': dataset['shape'],
                'profile': dataset['profile']
            }

            # Count filled values
            original_valid = np.sum(~np.isnan(data))
            final_valid = np.sum(~np.isnan(aligned_data))
            print(f"{dataset_name}: Aligned to {len(common_dates)} time steps "
                  f"(valid values: {original_valid:,} -> {final_valid:,})")

    def resample_to_common_spatial_grid(self):
        """Resample all datasets to the FINEST spatial resolution (highest detail)"""
        print("Resampling to common spatial grid (maintaining high resolution)...")

        # Find the dataset with the FINEST resolution (most pixels)
        max_pixels = 0
        reference_dataset = None

        print("Dataset spatial information:")
        for name, dataset in self.aligned_data.items():
            n_pixels = np.prod(dataset['shape'])
            print(f"{name}: {dataset['shape']} = {n_pixels:,} pixels")
            if n_pixels > max_pixels:
                max_pixels = n_pixels
                reference_dataset = name

        print(
            f"Using {reference_dataset} as reference grid ({self.aligned_data[reference_dataset]['shape']}) - FINEST resolution")
        reference_shape = self.aligned_data[reference_dataset]['shape']
        reference_profile = self.aligned_data[reference_dataset]['profile']

        # Resample all OTHER datasets to the finest resolution
        for dataset_name, dataset in self.aligned_data.items():
            if dataset_name == reference_dataset:
                continue

            print(f"Upsampling {dataset_name} from {dataset['shape']} to {reference_shape}")

            data = dataset['data']
            n_time = data.shape[0]
            current_shape = dataset['shape']

            # Calculate zoom factors for each dimension
            zoom_factors = (
                1.0,  # Keep time dimension unchanged
                reference_shape[0] / current_shape[0],  # Height zoom factor
                reference_shape[1] / current_shape[1]  # Width zoom factor
            )

            print(
                f"  Zoom factors: time={zoom_factors[0]:.2f}, height={zoom_factors[1]:.2f}, width={zoom_factors[2]:.2f}")

            # Use scipy.ndimage.zoom for high-quality resampling
            resampled_data = zoom(data, zoom_factors, order=1, prefilter=False)

            # Ensure exact target shape (crop or pad if necessary due to rounding)
            target_shape = (n_time, reference_shape[0], reference_shape[1])
            if resampled_data.shape != target_shape:
                print(f"  Adjusting shape from {resampled_data.shape} to {target_shape}")
                # Create array with exact target shape
                final_data = np.full(target_shape, np.nan)

                # Copy data with appropriate bounds checking
                min_t = min(resampled_data.shape[0], target_shape[0])
                min_h = min(resampled_data.shape[1], target_shape[1])
                min_w = min(resampled_data.shape[2], target_shape[2])

                final_data[:min_t, :min_h, :min_w] = resampled_data[:min_t, :min_h, :min_w]
                resampled_data = final_data

            # Update the dataset
            self.aligned_data[dataset_name]['data'] = resampled_data
            self.aligned_data[dataset_name]['shape'] = reference_shape
            self.aligned_data[dataset_name]['profile'] = reference_profile.copy()

            print(f"  Final shape: {resampled_data.shape}")

        return reference_dataset

    def calculate_resilience_recovery(self):
        """Calculate resilience and recovery metrics for each pair of indices"""
        print("Calculating resilience and recovery metrics...")

        # Get all dataset combinations
        dataset_names = list(self.aligned_data.keys())
        combinations_list = list(combinations(dataset_names, 2))

        self.resilience_maps = {}
        self.recovery_maps = {}
        self.correlation_maps = {}  # Add correlation maps for summary matrix
        self.dataset_names = dataset_names  # Store for summary matrix

        for combo in combinations_list:
            dataset1, dataset2 = combo
            combo_name = f"{dataset1}_{dataset2}"

            print(f"Processing combination: {combo_name}")

            # Get aligned data
            data1 = self.aligned_data[dataset1]['data']
            data2 = self.aligned_data[dataset2]['data']

            # Both datasets should now have the same shape
            assert data1.shape == data2.shape, f"Shape mismatch: {data1.shape} vs {data2.shape}"

            print(f"  Processing {data1.shape[1:]} pixels ({np.prod(data1.shape[1:]):,} total pixels)")

            # Calculate resilience, recovery, and correlation
            resilience_map = self.calculate_resilience_vectorized(data1, data2)
            recovery_map = self.calculate_recovery_vectorized(data1, data2)
            correlation_map = self.calculate_correlation_vectorized(data1, data2)

            self.resilience_maps[combo_name] = {
                'data': resilience_map,
                'shape': data1.shape[1:]
            }
            self.recovery_maps[combo_name] = {
                'data': recovery_map,
                'shape': data1.shape[1:]
            }
            self.correlation_maps[combo_name] = {
                'data': correlation_map,
                'shape': data1.shape[1:],
                'dataset1': dataset1,
                'dataset2': dataset2
            }

    def calculate_resilience_vectorized(self, data1, data2):
        """Calculate resilience metric using correlation coefficient"""
        n_time, n_rows, n_cols = data1.shape
        resilience_map = np.full((n_rows, n_cols), np.nan)

        print(f"    Calculating resilience for {n_rows}x{n_cols} pixels...")

        # Calculate correlation for each pixel with progress tracking
        processed = 0
        total_pixels = n_rows * n_cols

        for i in range(n_rows):
            for j in range(n_cols):
                series1 = data1[:, i, j]
                series2 = data2[:, i, j]

                # Skip if too many NaN values
                valid_mask = ~(np.isnan(series1) | np.isnan(series2))
                if np.sum(valid_mask) < 24:  # Need at least 2 years of data
                    continue

                # Calculate correlation coefficient
                try:
                    corr, p_value = stats.pearsonr(series1[valid_mask], series2[valid_mask])
                    # Use absolute correlation as resilience measure
                    resilience_map[i, j] = abs(corr) if not np.isnan(corr) else np.nan
                except:
                    continue

                processed += 1

            # Progress update every 10 rows
            if (i + 1) % 10 == 0:
                progress = ((i + 1) * n_cols) / total_pixels * 100
                print(f"      Progress: {progress:.1f}% ({i + 1}/{n_rows} rows)")

    def calculate_correlation_vectorized(self, data1, data2):
        """Calculate correlation coefficient map between two datasets"""
        n_time, n_rows, n_cols = data1.shape
        correlation_map = np.full((n_rows, n_cols), np.nan)

        print(f"    Calculating correlation for {n_rows}x{n_cols} pixels...")

        # Calculate correlation for each pixel with progress tracking
        processed = 0
        total_pixels = n_rows * n_cols

        for i in range(n_rows):
            for j in range(n_cols):
                series1 = data1[:, i, j]
                series2 = data2[:, i, j]

                # Skip if too many NaN values
                valid_mask = ~(np.isnan(series1) | np.isnan(series2))
                if np.sum(valid_mask) < 24:  # Need at least 2 years of data
                    continue

                # Calculate correlation coefficient
                try:
                    corr, p_value = stats.pearsonr(series1[valid_mask], series2[valid_mask])
                    correlation_map[i, j] = corr if not np.isnan(corr) else np.nan
                except:
                    continue

                processed += 1

            # Progress update every 10 rows
            if (i + 1) % 10 == 0:
                progress = ((i + 1) * n_cols) / total_pixels * 100
                print(f"      Progress: {progress:.1f}% ({i + 1}/{n_rows} rows)")

        print(f"    Completed: {processed:,}/{total_pixels:,} pixels processed")
        return correlation_map

    def calculate_recovery_vectorized(self, data1, data2):
        """Calculate recovery metric using trend correlation"""
        n_time, n_rows, n_cols = data1.shape
        recovery_map = np.full((n_rows, n_cols), np.nan)

        print(f"    Calculating recovery for {n_rows}x{n_cols} pixels...")

        # Calculate trend correlation for each pixel
        time_indices = np.arange(n_time)
        processed = 0
        total_pixels = n_rows * n_cols

        for i in range(n_rows):
            for j in range(n_cols):
                series1 = data1[:, i, j]
                series2 = data2[:, i, j]

                # Skip if too many NaN values
                valid_mask = ~(np.isnan(series1) | np.isnan(series2))
                if np.sum(valid_mask) < 24:  # Need at least 2 years of data
                    continue

                try:
                    # Calculate linear trends
                    valid_time = time_indices[valid_mask]
                    valid_series1 = series1[valid_mask]
                    valid_series2 = series2[valid_mask]

                    slope1, _, r1, _, _ = stats.linregress(valid_time, valid_series1)
                    slope2, _, r2, _, _ = stats.linregress(valid_time, valid_series2)

                    # Recovery metric: correlation between the two trend directions
                    recovery_map[i, j] = slope1 * slope2 * abs(r1 * r2)

                    processed += 1
                except:
                    continue

            # Progress update every 10 rows
            if (i + 1) % 10 == 0:
                progress = ((i + 1) * n_cols) / total_pixels * 100
                print(f"      Progress: {progress:.1f}% ({i + 1}/{n_rows} rows)")

        print(f"    Completed: {processed:,}/{total_pixels:,} pixels processed")
        return recovery_map

    def create_heatmaps(self):
        """Create spatial heatmaps for resilience, recovery, and individual correlations"""
        print("Creating high-resolution heatmaps...")

        # Create individual correlation heatmaps (15 separate maps)
        self.plot_individual_correlation_heatmaps()

        # Create resilience heatmaps
        self.plot_resilience_heatmaps()

        # Create recovery heatmaps
        self.plot_recovery_heatmaps()

        # Create summary correlation matrix
        self.create_summary_correlation_matrix()

    def plot_individual_correlation_heatmaps(self):
        """Create 15 separate correlation heatmaps"""
        print("Creating individual correlation heatmaps...")

        if not hasattr(self, 'correlation_maps') or not self.correlation_maps:
            print("No correlation maps to plot")
            return

        # Create custom colormap (blue-white-red) like in your reference
        colors = ['#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7',
                  '#fddbc7', '#f4a582', '#d6604d', '#b2182b']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('correlation', colors, N=n_bins)

        # Create individual heatmap for each correlation pair
        for combo_name, corr_data in self.correlation_maps.items():
            correlation_map = corr_data['data']
            dataset1 = corr_data['dataset1']
            dataset2 = corr_data['dataset2']

            print(f"Creating correlation heatmap: {dataset1} vs {dataset2}")

            # Create individual figure
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))

            # Handle NaN values
            masked_data = np.ma.masked_invalid(correlation_map)

            # Plot correlation map
            im = ax.imshow(masked_data, cmap=cmap, vmin=-1, vmax=1,
                           aspect='auto', interpolation='bilinear')

            # Calculate statistics for title
            valid_data = correlation_map[~np.isnan(correlation_map)]
            if len(valid_data) > 0:
                mean_corr = np.mean(valid_data)
                std_corr = np.std(valid_data)
                min_corr = np.min(valid_data)
                max_corr = np.max(valid_data)

                title = f'Spatial Correlation: {dataset1} vs {dataset2}\n' + \
                        f'Mean: {mean_corr:.3f}, Std: {std_corr:.3f}, ' + \
                        f'Range: [{min_corr:.3f}, {max_corr:.3f}]'
            else:
                title = f'Spatial Correlation: {dataset1} vs {dataset2}'

            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.axis('off')

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
            cbar.set_label('Correlation Coefficient', fontsize=12)

            # Add statistics text box
            if len(valid_data) > 0:
                stats_text = f'Valid pixels: {len(valid_data):,}\n' + \
                             f'Total pixels: {np.prod(correlation_map.shape):,}\n' + \
                             f'Coverage: {len(valid_data) / np.prod(correlation_map.shape) * 100:.1f}%'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                        fontsize=10)

            plt.tight_layout()

            # Save individual correlation map
            filename = f'correlation_heatmap_{dataset1}_vs_{dataset2}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"  Saved: {filename}")

    def create_summary_correlation_matrix(self):
        """Create summary matrix showing mean correlations between all datasets"""
        print("Creating summary correlation matrix...")

        if not hasattr(self, 'dataset_names') or not hasattr(self, 'correlation_maps'):
            print("Missing data for summary matrix")
            return

        dataset_names = self.dataset_names
        n_datasets = len(dataset_names)

        # Initialize summary matrix
        summary_matrix = np.full((n_datasets, n_datasets), np.nan)

        # Fill diagonal with 1s
        np.fill_diagonal(summary_matrix, 1.0)

        # Fill off-diagonal with mean correlations
        for combo_name, corr_data in self.correlation_maps.items():
            dataset1 = corr_data['dataset1']
            dataset2 = corr_data['dataset2']
            correlation_map = corr_data['data']

            # Calculate mean correlation (ignoring NaN)
            valid_correlations = correlation_map[~np.isnan(correlation_map)]
            if len(valid_correlations) > 0:
                mean_corr = np.mean(valid_correlations)

                # Find indices
                idx1 = dataset_names.index(dataset1)
                idx2 = dataset_names.index(dataset2)

                # Fill symmetric matrix
                summary_matrix[idx1, idx2] = mean_corr
                summary_matrix[idx2, idx1] = mean_corr

        # Create the summary plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Use the same colormap as your reference image
        im = ax.imshow(summary_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

        # Add text annotations with correlation values
        for i in range(n_datasets):
            for j in range(n_datasets):
                if not np.isnan(summary_matrix[i, j]):
                    # Choose text color based on background
                    text_color = 'white' if abs(summary_matrix[i, j]) > 0.5 else 'black'
                    text = ax.text(j, i, f'{summary_matrix[i, j]:.2f}',
                                   ha="center", va="center",
                                   color=text_color, fontsize=14, fontweight='bold')

        # Customize plot to match your reference
        ax.set_xticks(range(n_datasets))
        ax.set_yticks(range(n_datasets))
        ax.set_xticklabels(dataset_names, rotation=45, ha='right', fontsize=12)
        ax.set_yticklabels(dataset_names, fontsize=12)

        # Add title matching your reference
        ax.set_title('Mean Spatial Correlations Between Drought Indices\n(2001-2021)',
                     fontsize=16, fontweight='bold', pad=20)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Mean Spatial Correlation', fontsize=12)

        # Add grid for better readability
        ax.set_xlim(-0.5, n_datasets - 0.5)
        ax.set_ylim(-0.5, n_datasets - 0.5)

        plt.tight_layout()

        # Save summary matrix
        plt.savefig('mean_spatial_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("Summary correlation matrix saved as 'mean_spatial_correlation_matrix.png'")

        # Print correlation matrix as text for verification
        print("\nMean Spatial Correlation Matrix:")
        print("=" * 60)
        correlation_df = pd.DataFrame(summary_matrix,
                                      index=dataset_names,
                                      columns=dataset_names)
        print(correlation_df.round(3))

        return summary_matrix

    def plot_resilience_heatmaps(self):
        """Plot resilience heatmaps with blue = good resilience, red = poor resilience"""
        n_combinations = len(self.resilience_maps)

        if n_combinations == 0:
            print("No resilience maps to plot")
            return

        # Calculate number of rows and columns for subplots
        n_cols = 3
        n_rows = (n_combinations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Create custom colormap (red to blue)
        colors = ['red', 'yellow', 'blue']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('resilience', colors, N=n_bins)

        for idx, (combo_name, resilience_data) in enumerate(self.resilience_maps.items()):
            row = idx // n_cols
            col = idx % n_cols

            resilience_map = resilience_data['data']

            print(f"Plotting {combo_name}: {resilience_map.shape} pixels")

            # Handle NaN values
            masked_data = np.ma.masked_invalid(resilience_map)

            if n_rows == 1 and n_cols == 1:
                ax = axes
            else:
                ax = axes[row, col] if n_rows > 1 else axes[col]

            # Use high DPI and interpolation for smooth display
            im = ax.imshow(masked_data, cmap=cmap, vmin=0, vmax=1,
                           aspect='auto', interpolation='bilinear')
            ax.set_title(f'Resilience: {combo_name.replace("_", " vs ")}',
                         fontsize=10, fontweight='bold')
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

        # Hide unused subplots
        for idx in range(n_combinations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
            elif n_cols > 1:
                axes[col].axis('off')

        plt.tight_layout()
        plt.suptitle(
            'California Drought Resilience Analysis - High Resolution\n(Blue = High Resilience, Red = Low Resilience)',
            fontsize=14, fontweight='bold', y=0.98)
        plt.savefig('california_resilience_heatmaps_highres.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_recovery_heatmaps(self):
        """Plot recovery heatmaps with green = good recovery, red = poor recovery"""
        n_combinations = len(self.recovery_maps)

        if n_combinations == 0:
            print("No recovery maps to plot")
            return

        # Calculate number of rows and columns for subplots
        n_cols = 3
        n_rows = (n_combinations + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Create custom colormap (red to green)
        colors = ['red', 'yellow', 'green']
        n_bins = 256
        cmap = LinearSegmentedColormap.from_list('recovery', colors, N=n_bins)

        # Calculate global min/max for consistent scaling
        all_values = []
        for recovery_data in self.recovery_maps.values():
            recovery_map = recovery_data['data']
            valid_values = recovery_map[~np.isnan(recovery_map)]
            if len(valid_values) > 0:
                all_values.extend(valid_values)

        if all_values:
            vmin, vmax = np.percentile(all_values, [5, 95])
            # Make colorbar symmetric around zero
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax
        else:
            vmin, vmax = -1, 1

        for idx, (combo_name, recovery_data) in enumerate(self.recovery_maps.items()):
            row = idx // n_cols
            col = idx % n_cols

            recovery_map = recovery_data['data']

            print(f"Plotting {combo_name}: {recovery_map.shape} pixels")

            # Handle NaN values
            masked_data = np.ma.masked_invalid(recovery_map)

            if n_rows == 1 and n_cols == 1:
                ax = axes
            else:
                ax = axes[row, col] if n_rows > 1 else axes[col]

            # Use high DPI and interpolation for smooth display
            im = ax.imshow(masked_data, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect='auto', interpolation='bilinear')
            ax.set_title(f'Recovery: {combo_name.replace("_", " vs ")}',
                         fontsize=10, fontweight='bold')
            ax.axis('off')

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)

        # Hide unused subplots
        for idx in range(n_combinations, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            if n_rows > 1:
                axes[row, col].axis('off')
            elif n_cols > 1:
                axes[col].axis('off')

        plt.tight_layout()
        plt.suptitle(
            'California Drought Recovery Analysis - High Resolution\n(Green = High Recovery, Red = Low Recovery)',
            fontsize=14, fontweight='bold', y=0.98)
        plt.savefig('california_recovery_heatmaps_highres.png', dpi=300, bbox_inches='tight')
        plt.show()

    def save_results_to_excel(self):
        """Save results to Excel file"""
        print("Saving results to Excel...")

        try:
            with pd.ExcelWriter('california_drought_analysis_results_highres.xlsx', engine='openpyxl') as writer:

                # Save resilience statistics
                resilience_stats = []
                for combo_name, resilience_data in self.resilience_maps.items():
                    resilience_map = resilience_data['data']
                    valid_data = resilience_map[~np.isnan(resilience_map)]
                    if len(valid_data) > 0:
                        stats_dict = {
                            'Combination': combo_name,
                            'Mean_Resilience': np.mean(valid_data),
                            'Median_Resilience': np.median(valid_data),
                            'Std_Resilience': np.std(valid_data),
                            'Min_Resilience': np.min(valid_data),
                            'Max_Resilience': np.max(valid_data),
                            'Q25_Resilience': np.percentile(valid_data, 25),
                            'Q75_Resilience': np.percentile(valid_data, 75),
                            'Valid_Pixels': len(valid_data),
                            'Spatial_Resolution': f"{resilience_data['shape'][0]}x{resilience_data['shape'][1]}"
                        }
                        resilience_stats.append(stats_dict)

                if resilience_stats:
                    resilience_df = pd.DataFrame(resilience_stats)
                    resilience_df.to_excel(writer, sheet_name='Resilience_Statistics', index=False)

                # Save recovery statistics
                recovery_stats = []
                for combo_name, recovery_data in self.recovery_maps.items():
                    recovery_map = recovery_data['data']
                    valid_data = recovery_map[~np.isnan(recovery_map)]
                    if len(valid_data) > 0:
                        stats_dict = {
                            'Combination': combo_name,
                            'Mean_Recovery': np.mean(valid_data),
                            'Median_Recovery': np.median(valid_data),
                            'Std_Recovery': np.std(valid_data),
                            'Min_Recovery': np.min(valid_data),
                            'Max_Recovery': np.max(valid_data),
                            'Q25_Recovery': np.percentile(valid_data, 25),
                            'Q75_Recovery': np.percentile(valid_data, 75),
                            'Valid_Pixels': len(valid_data),
                            'Spatial_Resolution': f"{recovery_data['shape'][0]}x{recovery_data['shape'][1]}"
                        }
                        recovery_stats.append(stats_dict)

                if recovery_stats:
                    recovery_df = pd.DataFrame(recovery_stats)
                    recovery_df.to_excel(writer, sheet_name='Recovery_Statistics', index=False)

                # Save dataset information
                dataset_info = []
                for name, data in self.aligned_data.items():
                    info_dict = {
                        'Dataset': name,
                        'Time_Steps': len(data['dates']),
                        'Spatial_Shape': f"{data['shape'][0]} x {data['shape'][1]}",
                        'Total_Pixels': np.prod(data['shape']),
                        'Date_Range': f"{min(data['dates']).strftime('%Y-%m')} to {max(data['dates']).strftime('%Y-%m')}",
                        'Pixel_Resolution': 'High Resolution (upsampled to finest available)'
                    }
                    dataset_info.append(info_dict)

                if dataset_info:
                    info_df = pd.DataFrame(dataset_info)
                    info_df.to_excel(writer, sheet_name='Dataset_Info', index=False)

            print("Results saved to 'california_drought_analysis_results_highres.xlsx'")

        except Exception as e:
            print(f"Error saving to Excel: {e}")

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting California Drought Analysis (High Resolution)...")
        print("=" * 60)

        try:
            # Step 1: Load datasets
            self.load_datasets()

            if not self.data_arrays:
                print("No datasets loaded successfully. Exiting.")
                return

            # Step 2: Fill missing data using the specified three-step method
            self.fill_missing_data_vectorized()

            # Step 3: Align temporal data to common time index
            self.align_temporal_data()

            # Step 4: Resample to common spatial grid (FINEST resolution)
            self.resample_to_common_spatial_grid()

            # Step 5: Calculate resilience and recovery
            self.calculate_resilience_recovery()

            # Step 6: Create heatmaps
            self.create_heatmaps()

            # Step 7: Save results
            self.save_results_to_excel()

            print("=" * 60)
            print("High-resolution analysis completed successfully!")
            if hasattr(self, 'resilience_maps'):
                total_pixels = sum(np.prod(data['shape']) for data in self.resilience_maps.values())
                print(f"Generated {len(self.resilience_maps)} resilience heatmaps")
                print(f"Total pixels processed: {total_pixels:,}")
            if hasattr(self, 'recovery_maps'):
                print(f"Generated {len(self.recovery_maps)} recovery heatmaps")

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()


# Run the analysis
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = DroughtAnalysis()

    # Run complete analysis
    analyzer.run_analysis()

    # Print summary statistics
    print("\nHIGH-RESOLUTION SUMMARY STATISTICS:")
    print("=" * 60)

    if hasattr(analyzer, 'resilience_maps') and analyzer.resilience_maps:
        print("\nResilience Analysis:")
        for combo_name, resilience_data in analyzer.resilience_maps.items():
            resilience_map = resilience_data['data']
            valid_data = resilience_map[~np.isnan(resilience_map)]
            shape_info = resilience_data['shape']
            if len(valid_data) > 0:
                print(f"{combo_name}: Mean = {np.mean(valid_data):.3f}, "
                      f"Std = {np.std(valid_data):.3f}, "
                      f"Valid pixels = {len(valid_data):,} "
                      f"(Resolution: {shape_info[0]}x{shape_info[1]})")

    if hasattr(analyzer, 'recovery_maps') and analyzer.recovery_maps:
        print("\nRecovery Analysis:")
        for combo_name, recovery_data in analyzer.recovery_maps.items():
            recovery_map = recovery_data['data']
            valid_data = recovery_map[~np.isnan(recovery_map)]
            shape_info = recovery_data['shape']
            if len(valid_data) > 0:
                print(f"{combo_name}: Mean = {np.mean(valid_data):.3f}, "
                      f"Std = {np.std(valid_data):.3f}, "
                      f"Valid pixels = {len(valid_data):,} "
                      f"(Resolution: {shape_info[0]}x{shape_info[1]})")

    if hasattr(analyzer, 'correlation_maps') and analyzer.correlation_maps:
        print("\nCorrelation Analysis:")
        for combo_name, corr_data in analyzer.correlation_maps.items():
            correlation_map = corr_data['data']
            valid_data = correlation_map[~np.isnan(correlation_map)]
            shape_info = corr_data['shape']
            if len(valid_data) > 0:
                print(f"{combo_name}: Mean = {np.mean(valid_data):.3f}, "
                      f"Std = {np.std(valid_data):.3f}, "
                      f"Valid pixels = {len(valid_data):,} "
                      f"(Resolution: {shape_info[0]}x{shape_info[1]})")

    if hasattr(analyzer, 'aligned_data'):
        print("\nAligned Dataset Summary (High Resolution):")
        for name, data in analyzer.aligned_data.items():
            total_pixels = np.prod(data['shape'])
            print(f"{name}: {len(data['dates'])} time steps, "
                  f"Shape: {data['shape']}, "
                  f"Total pixels: {total_pixels:,}, "
                  f"Date range: 2001-2021")