    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from datetime import datetime
    import warnings
    from sklearn.linear_model import LinearRegression
    from scipy.signal import find_peaks
    from scipy.ndimage import uniform_filter1d
    import traceback
    
    warnings.filterwarnings('ignore')
    
    
    class DroughtCorrelationAnalyzer:
        def __init__(self, base_path):
            """Initialize the correlation analyzer"""
            self.base_path = base_path
            self.indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPDmax']
            self.land_types = ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']
            self.land_type_mapping = {
                'GRASSLANDS': 'Grasslands',
                'NATURAL FORESTS': 'NaturalForests',
                'SAVANNAS': 'Savannas',
                'SHRUBLANDS': 'Shrublands'
            }
    
            # File patterns and folder mapping
            self.file_patterns = {
                'GNDVI': 'GNDVI_{}_GNDVI_MOD09A1_CA_5km_{}',
                'GOSIF': 'GOSIF_{}_CALIFORNIA_GOSIF_{}',
                'GPP': '{}_GPP_{}',
                'LST': 'LST_{}_MYD21C1_CA_LST_{}',
                'NDVI': 'NDVI_{}_NDVI_MOD13Q1_CA_5km_{}',
                'NIRv': 'NIRv_{}_NIRv_MOD13Q1_CA_5km_{}',
                'SPEI': 'SPEI_{}_SPEI1_CSIC_CA_{}',
                'TWS': 'TWS_{}_TWS_GRACE_CA_{}',
                'VPDmax': 'VPD_{}_VPDmax_GRIDMET_CA_5km_{}'
            }
    
            self.folder_mapping = {
                'GNDVI': 'GNDVI', 'GOSIF': 'GOSIF', 'GPP': 'GPP', 'LST': 'LST',
                'NDVI': 'NDVI', 'NIRv': 'NIRv', 'SPEI': 'SPEI', 'TWS': 'TWS', 'VPDmax': 'VPD'
            }
    
            self.gosif_paths = {
                'GRASSLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\GOSIF",
                'NATURAL FORESTS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\GOSIF",
                'SAVANNAS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\GOSIF",
                'SHRUBLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\GOSIF"
            }
    
            self.start_years = {
                'GNDVI': 2001, 'GOSIF': 2001, 'GPP': 2001, 'LST': 2003,
                'NDVI': 2001, 'NIRv': 2001, 'SPEI': 2001, 'TWS': 2002, 'VPDmax': 2001
            }
    
            # Debug flags
            self.debug_mode = False
            self.missing_files_log = []
    
        def set_debug_mode(self, debug=True):
            """Enable/disable debug mode for verbose output"""
            self.debug_mode = debug
    
        def generate_date_range(self, index_name, start_year=2001, end_year=2021):
            """Generate date range for time series extraction"""
            try:
                actual_start = max(start_year, self.start_years.get(index_name, start_year))
                actual_end = end_year
    
                dates = []
                for year in range(actual_start, actual_end + 1):
                    for month in range(1, 13):
                        # Handle special cases
                        if index_name == 'TWS' and year == 2002 and month < 4:
                            continue
                        if index_name == 'GPP' and year > 2011:
                            continue
                        dates.append(f"{year:04d}_{month:02d}")
    
                if self.debug_mode:
                    print(f"Generated {len(dates)} dates for {index_name} from {actual_start} to {actual_end}")
    
                return dates
            except Exception as e:
                print(f"Error generating date range for {index_name}: {e}")
                return []
    
        def load_raster_data(self, file_path):
            """Load raster data from TIFF file with error handling"""
            try:
                # Check if rasterio is available
                try:
                    import rasterio
                except ImportError:
                    print("Error: rasterio library not found. Install with: pip install rasterio")
                    return None
    
                if not os.path.exists(file_path):
                    if self.debug_mode:
                        print(f"File not found: {file_path}")
                    return None
    
                with rasterio.open(file_path) as src:
                    data = src.read(1)
                    data = data.astype(float)
    
                    # Handle NoData values
                    if src.nodata is not None:
                        data[data == src.nodata] = np.nan
    
                    # Additional checks for common NoData values
                    data[data == -9999] = np.nan
                    data[data == -3.4e38] = np.nan
                    data[np.isinf(data)] = np.nan
    
                    if self.debug_mode and np.all(np.isnan(data)):
                        print(f"Warning: All data is NaN in file: {file_path}")
    
                    return data
            except Exception as e:
                if self.debug_mode:
                    print(f"Error loading raster data from {file_path}: {e}")
                return None
    
        def find_actual_filename(self, folder_path, expected_pattern, date_str, index_name):
            """Find the actual filename that matches the pattern with improved error handling"""
            try:
                if not os.path.exists(folder_path):
                    if self.debug_mode:
                        print(f"Folder not found: {folder_path}")
                    self.missing_files_log.append(f"Missing folder: {folder_path}")
                    return None
    
                files_in_folder = os.listdir(folder_path)
    
                # Try exact match first
                exact_file = expected_pattern + '.tif'
                if exact_file in files_in_folder:
                    return os.path.join(folder_path, exact_file)
    
                # Special handling for GOSIF
                if index_name == 'GOSIF':
                    gosif_date = date_str.replace('_', '')
                    for file in files_in_folder:
                        if file.endswith('.tif') and gosif_date in file:
                            return os.path.join(folder_path, file)
    
                # Try various date patterns
                year = date_str[:4]
                month = date_str[5:7]
                year_month_patterns = [
                    f"{year}_{month}", f"{year}{month}", f"{year}-{month}",
                    f"{year}.{month}", f"{month}_{year}", f"{month}{year}"
                ]
    
                for file in files_in_folder:
                    if file.endswith('.tif'):
                        file_lower = file.lower()
                        for pattern in year_month_patterns:
                            if pattern in file_lower:
                                return os.path.join(folder_path, file)
    
                # Log missing file for debugging
                self.missing_files_log.append(f"Missing file: {expected_pattern} in {folder_path}")
                return None
    
            except Exception as e:
                if self.debug_mode:
                    print(f"Error finding file in {folder_path}: {e}")
                return None
    
        def get_land_type_for_filename(self, land_type, index_name):
            """Get the correct land type name for filename"""
            try:
                if index_name == 'GOSIF':
                    if land_type == 'SAVANNAS':
                        return 'Savanna'
                    elif land_type == 'NATURAL FORESTS':
                        return 'NaturalForests'
                    else:
                        return self.land_type_mapping.get(land_type, land_type)
                else:
                    return self.land_type_mapping.get(land_type, land_type)
            except Exception as e:
                print(f"Error getting land type for filename: {e}")
                return land_type
    
        def extract_time_series_all_indices(self):
            """Extract time series for all indices and land types with improved error handling"""
            print("Extracting time series for all indices and land types...")
    
            try:
                # Create a master date range (2001-2021)
                master_dates = []
                for year in range(2001, 2022):
                    for month in range(1, 13):
                        master_dates.append(datetime(year, month, 1))
    
                # Initialize data storage
                time_series_data = {}
                extraction_stats = {}
    
                for land_type in self.land_types:
                    time_series_data[land_type] = {}
                    extraction_stats[land_type] = {}
                    clean_land_name = self.land_type_mapping[land_type]
                    print(f"Processing {clean_land_name}...")
    
                    for index_name in self.indices:
                        print(f"  Extracting {index_name}...")
                        dates = self.generate_date_range(index_name)
                        data_values = []
                        files_found = 0
                        files_processed = 0
    
                        for date_str in dates:
                            try:
                                # Build file path
                                if index_name == 'GOSIF':
                                    folder_path = self.gosif_paths[land_type]
                                else:
                                    land_folder = f"CALIFORNIA {land_type}"
                                    index_folder = self.folder_mapping[index_name]
                                    folder_path = os.path.join(self.base_path, land_folder, index_folder)
    
                                # Generate filename
                                land_type_clean = self.get_land_type_for_filename(land_type, index_name)
                                expected_filename = self.file_patterns[index_name].format(land_type_clean, date_str)
    
                                # Find and load file
                                file_path = self.find_actual_filename(folder_path, expected_filename, date_str, index_name)
    
                                if file_path:
                                    files_found += 1
                                    data = self.load_raster_data(file_path)
                                    if data is not None and not np.all(np.isnan(data)):
                                        avg_value = np.nanmean(data)
                                        if not np.isnan(avg_value):
                                            data_values.append(avg_value)
                                            files_processed += 1
                                        else:
                                            data_values.append(np.nan)
                                    else:
                                        data_values.append(np.nan)
                                else:
                                    data_values.append(np.nan)
    
                            except Exception as e:
                                if self.debug_mode:
                                    print(f"    Error processing {date_str}: {e}")
                                data_values.append(np.nan)
    
                        extraction_stats[land_type][index_name] = {
                            'total_expected': len(dates),
                            'files_found': files_found,
                            'files_processed': files_processed,
                            'success_rate': files_processed / len(dates) if len(dates) > 0 else 0
                        }
    
                        # Create full time series aligned with master dates
                        full_series = np.full(len(master_dates), np.nan)
    
                        # Fill in available data
                        data_idx = 0
                        for i, master_date in enumerate(master_dates):
                            if data_idx < len(dates):
                                try:
                                    date_str = dates[data_idx]
                                    expected_date = datetime(int(date_str[:4]), int(date_str[5:7]), 1)
    
                                    if master_date == expected_date:
                                        if data_idx < len(data_values):
                                            full_series[i] = data_values[data_idx]
                                        data_idx += 1
                                    elif master_date > expected_date:
                                        data_idx += 1
                                        if data_idx < len(dates):
                                            date_str = dates[data_idx]
                                            expected_date = datetime(int(date_str[:4]), int(date_str[5:7]), 1)
                                            if master_date == expected_date and data_idx < len(data_values):
                                                full_series[i] = data_values[data_idx]
                                                data_idx += 1
                                except Exception as e:
                                    if self.debug_mode:
                                        print(f"    Error aligning date {date_str}: {e}")
    
                        time_series_data[land_type][index_name] = full_series
    
                        # Print extraction statistics
                        stats = extraction_stats[land_type][index_name]
                        print(
                            f"    {index_name}: {stats['files_processed']}/{stats['total_expected']} files processed ({stats['success_rate']:.1%})")
    
                # Print overall extraction summary
                self.print_extraction_summary(extraction_stats)
    
                return time_series_data, master_dates
    
            except Exception as e:
                print(f"Critical error in extract_time_series_all_indices: {e}")
                traceback.print_exc()
                return {}, []
    
        def print_extraction_summary(self, extraction_stats):
            """Print summary of data extraction process"""
            print("\n" + "=" * 60)
            print("DATA EXTRACTION SUMMARY")
            print("=" * 60)
    
            for land_type in self.land_types:
                land_name = self.land_type_mapping[land_type]
                print(f"\n{land_name}:")
                for index_name in self.indices:
                    stats = extraction_stats[land_type][index_name]
                    print(
                        f"  {index_name:8}: {stats['files_processed']:3d}/{stats['total_expected']:3d} ({stats['success_rate']:5.1%})")
    
        def calculate_residuals_and_slope_changes(self, data, window_size=12):
            """
            Calculate residuals from trend and detect slope changes in residuals
            Enhanced with better error handling
            """
            try:
                if len(data) < window_size * 2:
                    if self.debug_mode:
                        print(f"Insufficient data length: {len(data)} < {window_size * 2}")
                    return np.full_like(data, np.nan), np.full_like(data, np.nan)
    
                # Remove NaN values for trend calculation
                valid_idx = ~np.isnan(data)
                valid_count = np.sum(valid_idx)
    
                if valid_count < window_size:
                    if self.debug_mode:
                        print(f"Insufficient valid data: {valid_count} < {window_size}")
                    return np.full_like(data, np.nan), np.full_like(data, np.nan)
    
                # Calculate long-term trend
                x_trend = np.arange(len(data))[valid_idx]
                y_trend = data[valid_idx]
    
                if len(y_trend) < 3:
                    return np.full_like(data, np.nan), np.full_like(data, np.nan)
    
                # Fit linear trend with error handling
                try:
                    slope, intercept = np.polyfit(x_trend, y_trend, 1)
                    trend_line = slope * np.arange(len(data)) + intercept
                except np.linalg.LinAlgError:
                    if self.debug_mode:
                        print("Linear algebra error in trend fitting")
                    return np.full_like(data, np.nan), np.full_like(data, np.nan)
    
                # Calculate residuals
                residuals = data - trend_line
    
                # Calculate slope changes in residuals using moving window
                slope_changes = np.full_like(data, np.nan)
    
                for i in range(window_size, len(residuals) - window_size):
                    try:
                        window_data = residuals[i - window_size:i + window_size]
                        valid_window_count = np.sum(~np.isnan(window_data))
    
                        if valid_window_count >= window_size:
                            # Calculate slope before and after point i
                            before_window = residuals[i - window_size:i]
                            after_window = residuals[i:i + window_size]
    
                            # Remove NaN values
                            before_valid = before_window[~np.isnan(before_window)]
                            after_valid = after_window[~np.isnan(after_window)]
    
                            if len(before_valid) >= 3 and len(after_valid) >= 3:
                                try:
                                    slope_before = np.polyfit(np.arange(len(before_valid)), before_valid, 1)[0]
                                    slope_after = np.polyfit(np.arange(len(after_valid)), after_valid, 1)[0]
                                    slope_changes[i] = slope_after - slope_before
                                except np.linalg.LinAlgError:
                                    continue
                    except Exception as e:
                        if self.debug_mode:
                            print(f"Error in slope calculation at index {i}: {e}")
                        continue
    
                return residuals, slope_changes
    
            except Exception as e:
                print(f"Error in calculate_residuals_and_slope_changes: {e}")
                return np.full_like(data, np.nan), np.full_like(data, np.nan)
    
        def calculate_optimal_lags_slope_based(self, time_series_data, max_lag=12):
            """
            Calculate optimal lags based on correlation between slope changes in residuals
            Enhanced with better error handling
            """
            print("Calculating optimal lags based on slope changes in residuals...")
    
            optimal_lags = {}
    
            for land_type in self.land_types:
                print(f"Processing {self.land_type_mapping[land_type]}...")
                optimal_lags[land_type] = np.zeros((len(self.indices), len(self.indices)))
    
                try:
                    # Calculate slope changes for all indices
                    slope_changes_data = {}
                    for index_name in self.indices:
                        data = time_series_data[land_type][index_name]
                        residuals, slope_changes = self.calculate_residuals_and_slope_changes(data)
                        slope_changes_data[index_name] = slope_changes
    
                    # Calculate cross-correlations between slope changes
                    for i, index1 in enumerate(self.indices):
                        for j, index2 in enumerate(self.indices):
                            if i == j:
                                optimal_lags[land_type][i, j] = 0
                                continue
    
                            try:
                                slope1 = slope_changes_data[index1]
                                slope2 = slope_changes_data[index2]
    
                                # Find valid overlapping period
                                valid_mask = ~(np.isnan(slope1) | np.isnan(slope2))
                                valid_count = np.sum(valid_mask)
    
                                if valid_count < 10:  # Need at least 10 valid points
                                    optimal_lags[land_type][i, j] = 0
                                    continue
    
                                valid_slope1 = slope1[valid_mask]
                                valid_slope2 = slope2[valid_mask]
    
                                # Calculate cross-correlation at different lags
                                max_corr = 0
                                best_lag = 0
    
                                for lag in range(-max_lag, max_lag + 1):
                                    try:
                                        if lag == 0:
                                            if len(valid_slope1) > 1 and len(valid_slope2) > 1:
                                                corr_matrix = np.corrcoef(valid_slope1, valid_slope2)
                                                if corr_matrix.shape == (2, 2):
                                                    corr_coef = corr_matrix[0, 1]
                                                else:
                                                    continue
                                            else:
                                                continue
                                        elif lag > 0:
                                            if len(valid_slope1) > lag and len(valid_slope2) > lag:
                                                corr_matrix = np.corrcoef(valid_slope1[:-lag], valid_slope2[lag:])
                                                if corr_matrix.shape == (2, 2):
                                                    corr_coef = corr_matrix[0, 1]
                                                else:
                                                    continue
                                            else:
                                                continue
                                        else:  # lag < 0
                                            if len(valid_slope2) > abs(lag) and len(valid_slope1) > abs(lag):
                                                corr_matrix = np.corrcoef(valid_slope1[abs(lag):], valid_slope2[:lag])
                                                if corr_matrix.shape == (2, 2):
                                                    corr_coef = corr_matrix[0, 1]
                                                else:
                                                    continue
                                            else:
                                                continue
    
                                        if not np.isnan(corr_coef) and abs(corr_coef) > abs(max_corr):
                                            max_corr = corr_coef
                                            best_lag = lag
    
                                    except Exception as e:
                                        if self.debug_mode:
                                            print(f"Error calculating correlation for lag {lag}: {e}")
                                        continue
    
                                optimal_lags[land_type][i, j] = best_lag
    
                            except Exception as e:
                                if self.debug_mode:
                                    print(f"Error calculating lag for {index1}-{index2}: {e}")
                                optimal_lags[land_type][i, j] = 0
    
                except Exception as e:
                    print(f"Error processing {land_type}: {e}")
                    optimal_lags[land_type] = np.zeros((len(self.indices), len(self.indices)))
    
            return optimal_lags
    
        def calculate_correlations(self, time_series_data):
            """Calculate correlation matrices for raw data with enhanced error handling"""
            print("Calculating correlation matrices for raw data...")
    
            correlations = {}
    
            for land_type in self.land_types:
                print(f"Processing {self.land_type_mapping[land_type]}...")
    
                try:
                    # Create data matrix
                    data_matrix = []
                    valid_indices = []
    
                    for idx, index_name in enumerate(self.indices):
                        data = time_series_data[land_type][index_name]
                        if not np.all(np.isnan(data)):
                            data_matrix.append(data)
                            valid_indices.append(idx)
    
                    if len(data_matrix) < 2:
                        print(f"Warning: Insufficient valid data for {land_type}, using identity matrix")
                        correlations[land_type] = np.eye(len(self.indices))
                        continue
    
                    data_matrix = np.array(data_matrix)
    
                    # Calculate correlation matrix with error handling
                    try:
                        corr_matrix_partial = np.corrcoef(data_matrix)
    
                        # Create full correlation matrix
                        corr_matrix_full = np.eye(len(self.indices))
                        for i, idx_i in enumerate(valid_indices):
                            for j, idx_j in enumerate(valid_indices):
                                if corr_matrix_partial.ndim == 2:
                                    corr_matrix_full[idx_i, idx_j] = corr_matrix_partial[i, j]
                                elif len(valid_indices) == 1:
                                    corr_matrix_full[idx_i, idx_j] = 1.0 if i == j else 0.0
    
                        correlations[land_type] = corr_matrix_full
    
                    except Exception as e:
                        print(f"Error calculating correlations for {land_type}: {e}")
                        correlations[land_type] = np.eye(len(self.indices))
    
                except Exception as e:
                    print(f"Critical error processing {land_type}: {e}")
                    correlations[land_type] = np.eye(len(self.indices))
    
            return correlations
    
        def create_correlation_heatmaps(self, correlations, save_path='correlation_heatmaps_9indices.png'):
            """Create correlation heatmaps for all land types with error handling"""
            try:
                fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                fig.suptitle('Inter-Index Correlations by Land Type', fontsize=16, fontweight='bold')
    
                land_type_names = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
                positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
                for idx, (land_type, (row, col)) in enumerate(zip(self.land_types, positions)):
                    ax = axes[row, col]
    
                    try:
                        # Create heatmap
                        mask = np.triu(np.ones_like(correlations[land_type], dtype=bool), k=1)
    
                        sns.heatmap(correlations[land_type],
                                    annot=True,
                                    mask=mask,
                                    cmap='RdYlBu_r',
                                    center=0,
                                    square=True,
                                    xticklabels=self.indices,
                                    yticklabels=self.indices,
                                    cbar_kws={"shrink": 0.8},
                                    fmt='.2f',
                                    ax=ax,
                                    vmin=-1, vmax=1)
    
                        ax.set_title(land_type_names[idx], fontsize=12, fontweight='bold')
                        ax.set_xlabel('')
                        ax.set_ylabel('')
    
                    except Exception as e:
                        print(f"Error creating heatmap for {land_type}: {e}")
                        ax.text(0.5, 0.5, f'Error creating\nheatmap for\n{land_type_names[idx]}',
                                ha='center', va='center', transform=ax.transAxes)
    
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Correlation heatmaps saved as {save_path}")
    
            except Exception as e:
                print(f"Error creating correlation heatmaps: {e}")
    
        def create_optimal_lags_heatmaps(self, optimal_lags, save_path='optimal_lags_9indices.png'):
            """Create optimal lags heatmaps for all land types with error handling"""
            try:
                fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                fig.suptitle(
                    'Optimal Lag Times Between Drought Indicators Across California Land Types\n(Based on Slope Changes in Residuals)',
                    fontsize=16, fontweight='bold')
    
                land_type_names = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
                positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
                for idx, (land_type, (row, col)) in enumerate(zip(self.land_types, positions)):
                    ax = axes[row, col]
    
                    try:
                        # Create heatmap with custom colormap
                        sns.heatmap(optimal_lags[land_type],
                                    annot=True,
                                    cmap='RdYlBu_r',
                                    center=0,
                                    square=True,
                                    xticklabels=self.indices,
                                    yticklabels=self.indices,
                                    cbar_kws={"shrink": 0.8, "label": "Optimal Lag (months)"},
                                    fmt='.0f',
                                    ax=ax)
    
                        ax.set_title(land_type_names[idx], fontsize=12, fontweight='bold')
                        ax.set_xlabel('')
                        ax.set_ylabel('')
    
                    except Exception as e:
                        print(f"Error creating lag heatmap for {land_type}: {e}")
                        ax.text(0.5, 0.5, f'Error creating\nlag heatmap for\n{land_type_names[idx]}',
                                ha='center', va='center', transform=ax.transAxes)
    
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                print(f"Optimal lags heatmaps saved as {save_path}")
    
            except Exception as e:
                print(f"Error creating optimal lags heatmaps: {e}")
    
        def print_analysis_summary(self, correlations, optimal_lags):
            """Print summary of analysis results with error handling"""
            try:
                print("\n" + "=" * 80)
                print("ANALYSIS SUMMARY")
                print("=" * 80)
    
                print("\nCORRELATION ANALYSIS:")
                print("-" * 40)
                for land_type in self.land_types:
                    land_name = self.land_type_mapping[land_type]
                    corr_matrix = correlations[land_type]
    
                    try:
                        # Find highest correlations (excluding diagonal)
                        mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
                        masked_corr = corr_matrix.copy()
                        masked_corr[~mask] = 0  # Set diagonal to 0
    
                        if np.any(~np.isnan(masked_corr[mask])):
                            max_corr_idx = np.unravel_index(np.nanargmax(np.abs(masked_corr)), corr_matrix.shape)
                            max_corr_value = corr_matrix[max_corr_idx]
    
                            print(f"{land_name}:")
                            print(
                                f"  Highest correlation: {self.indices[max_corr_idx[0]]} - {self.indices[max_corr_idx[1]]}: {max_corr_value:.3f}")
                        else:
                            print(f"{land_name}: No valid correlations found")
    
                    except Exception as e:
                        print(f"{land_name}: Error in correlation analysis - {e}")
    
                print("\nOPTIMAL LAG ANALYSIS (based on slope changes):")
                print("-" * 40)
                for land_type in self.land_types:
                    land_name = self.land_type_mapping[land_type]
    
                    try:
                        lag_matrix = optimal_lags[land_type]
    
                        # Find most significant lags
                        non_zero_lags = lag_matrix[lag_matrix != 0]
                        if len(non_zero_lags) > 0:
                            max_lag_idx = np.unravel_index(np.argmax(np.abs(lag_matrix)), lag_matrix.shape)
                            max_lag_value = lag_matrix[max_lag_idx]
    
                            print(f"{land_name}:")
                            print(
                                f"  Largest lag: {self.indices[max_lag_idx[0]]} → {self.indices[max_lag_idx[1]]}: {max_lag_value:.0f} months")
                            print(f"  Average lag magnitude: {np.mean(np.abs(non_zero_lags)):.1f} months")
                        else:
                            print(f"{land_name}: No significant lags found")
    
                    except Exception as e:
                        print(f"{land_name}: Error in lag analysis - {e}")
    
            except Exception as e:
                print(f"Error in analysis summary: {e}")
    
        def validate_data_integrity(self, time_series_data):
            """Validate the integrity of extracted time series data"""
            print("\n" + "=" * 60)
            print("DATA INTEGRITY VALIDATION")
            print("=" * 60)
    
            for land_type in self.land_types:
                land_name = self.land_type_mapping[land_type]
                print(f"\n{land_name}:")
    
                for index_name in self.indices:
                    data = time_series_data[land_type][index_name]
                    valid_count = np.sum(~np.isnan(data))
                    total_count = len(data)
    
                    if valid_count == 0:
                        status = "❌ NO DATA"
                    elif valid_count < total_count * 0.1:
                        status = "⚠️  VERY LOW"
                    elif valid_count < total_count * 0.3:
                        status = "⚠️  LOW"
                    else:
                        status = "✅ GOOD"
    
                    print(
                        f"  {index_name:8}: {valid_count:3d}/{total_count:3d} ({valid_count / total_count:5.1%}) {status}")
    
        def run_complete_analysis(self):
            """Run complete correlation and lag analysis with comprehensive error handling"""
            print("Starting comprehensive correlation and lag analysis...")
            print("=" * 80)
    
            try:
                # 1. Extract time series data
                print("\n🔄 STEP 1: Extracting time series data...")
                time_series_data, dates = self.extract_time_series_all_indices()
    
                if not time_series_data:
                    print("❌ Failed to extract time series data. Check file paths and data availability.")
                    return None, None, None
    
                # 2. Validate data integrity
                self.validate_data_integrity(time_series_data)
    
                # 3. Calculate correlations (raw data)
                print("\n🔄 STEP 2: Calculating correlations...")
                correlations = self.calculate_correlations(time_series_data)
    
                # 4. Calculate optimal lags (based on slope changes in residuals)
                print("\n🔄 STEP 3: Calculating optimal lags...")
                optimal_lags = self.calculate_optimal_lags_slope_based(time_series_data)
    
                # 5. Create visualizations
                print("\n🔄 STEP 4: Creating visualizations...")
                try:
                    print("Creating correlation heatmaps...")
                    self.create_correlation_heatmaps(correlations)
                except Exception as e:
                    print(f"Error creating correlation heatmaps: {e}")
    
                try:
                    print("Creating optimal lags heatmaps...")
                    self.create_optimal_lags_heatmaps(optimal_lags)
                except Exception as e:
                    print(f"Error creating lag heatmaps: {e}")
    
                # 6. Print analysis summary
                print("\n🔄 STEP 5: Generating analysis summary...")
                self.print_analysis_summary(correlations, optimal_lags)
    
                # 7. Print missing files summary if debug mode is on
                if self.debug_mode and self.missing_files_log:
                    print("\n" + "=" * 60)
                    print("MISSING FILES LOG")
                    print("=" * 60)
                    for missing_file in self.missing_files_log[:20]:  # Show first 20
                        print(f"  {missing_file}")
                    if len(self.missing_files_log) > 20:
                        print(f"  ... and {len(self.missing_files_log) - 20} more")
    
                print("\n" + "=" * 80)
                print("✅ ANALYSIS COMPLETE!")
                print("=" * 80)
                print("\nKey Features:")
                print("✓ Enhanced error handling and debugging capabilities")
                print("✓ Data integrity validation")
                print("✓ Correlation analysis using raw time series data")
                print("✓ Optimal lag analysis based on slope changes in residuals")
                print("✓ Detection of unusual changes and their propagation effects")
                print("✓ Comprehensive visualization for all 9 drought indices")
                print("✓ Analysis across all 4 California land types")
                print("✓ Robust handling of missing data and file path issues")
    
                return correlations, optimal_lags, time_series_data
    
            except Exception as e:
                print(f"\n❌ CRITICAL ERROR in run_complete_analysis: {e}")
                traceback.print_exc()
                return None, None, None
    
        def diagnose_issues(self):
            """Diagnose common issues with data and file paths"""
            print("\n" + "=" * 60)
            print("DIAGNOSTIC MODE")
            print("=" * 60)
    
            # Check base path
            print(f"\n1. Checking base path: {self.base_path}")
            if os.path.exists(self.base_path):
                print("   ✅ Base path exists")
            else:
                print("   ❌ Base path does not exist!")
                return
    
            # Check land type folders
            print("\n2. Checking land type folders:")
            for land_type in self.land_types:
                land_folder = f"CALIFORNIA {land_type}"
                folder_path = os.path.join(self.base_path, land_folder)
                if os.path.exists(folder_path):
                    print(f"   ✅ {land_folder}")
                else:
                    print(f"   ❌ {land_folder}")
    
            # Check GOSIF paths
            print("\n3. Checking GOSIF paths:")
            for land_type, path in self.gosif_paths.items():
                if os.path.exists(path):
                    print(f"   ✅ {land_type}")
                else:
                    print(f"   ❌ {land_type}: {path}")
    
            # Check index folders for first land type
            print(f"\n4. Checking index folders for {self.land_types[0]}:")
            land_folder = f"CALIFORNIA {self.land_types[0]}"
            for index_name in self.indices:
                if index_name == 'GOSIF':
                    continue  # Skip GOSIF as it has special path
                index_folder = self.folder_mapping[index_name]
                folder_path = os.path.join(self.base_path, land_folder, index_folder)
                if os.path.exists(folder_path):
                    file_count = len([f for f in os.listdir(folder_path) if f.endswith('.tif')])
                    print(f"   ✅ {index_name}: {file_count} TIFF files")
                else:
                    print(f"   ❌ {index_name}: {folder_path}")
    
            # Check required libraries
            print("\n5. Checking required libraries:")
            libraries = ['rasterio', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'sklearn']
            for lib in libraries:
                try:
                    __import__(lib)
                    print(f"   ✅ {lib}")
                except ImportError:
                    print(f"   ❌ {lib} - not installed")
    
    
    # Usage example with enhanced debugging
    if __name__ == "__main__":
        # Set your base path
        BASE_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
    
        print("🚀 Starting Drought Correlation Analysis with Debug Features")
        print("=" * 80)
    
        # Initialize analyzer
        analyzer = DroughtCorrelationAnalyzer(BASE_PATH)
    
        # Enable debug mode for detailed output
        analyzer.set_debug_mode(True)
    
        # Run diagnostics first to check for common issues
        print("\n🔍 Running diagnostics...")
        analyzer.diagnose_issues()
    
        # Ask user if they want to proceed
        proceed = input("\nDo you want to proceed with the analysis? (y/n): ").lower().strip()
        if proceed != 'y':
            print("Analysis cancelled by user.")
            exit()
    
        # Run complete analysis
        try:
            correlations, optimal_lags, time_series_data = analyzer.run_complete_analysis()
    
            if correlations is not None:
                print(f"\n🎉 Analysis completed successfully!")
                print(f"📊 Results available in correlations, optimal_lags, and time_series_data variables")
            else:
                print(f"\n❌ Analysis failed. Check the diagnostic output above.")
    
        except KeyboardInterrupt:
            print("\n\n⏹️  Analysis interrupted by user.")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            traceback.print_exc()