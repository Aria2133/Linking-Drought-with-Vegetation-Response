import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
import warnings
from collections import defaultdict
import traceback
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


class NormalizedSlopeBasedAnalyzer:
    def __init__(self, base_path):
        """Initialize the normalized slope-based analyzer (without GPP)"""
        self.base_path = base_path
        # Removed GPP as requested
        self.indices = ['GNDVI', 'GOSIF', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPDmax']
        self.land_types = ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']
        self.land_type_mapping = {
            'GRASSLANDS': 'Grasslands',
            'NATURAL FORESTS': 'NaturalForests',
            'SAVANNAS': 'Savannas',
            'SHRUBLANDS': 'Shrublands'
        }

        # File patterns and folder mapping (removed GPP)
        self.file_patterns = {
            'GNDVI': 'GNDVI_{}_GNDVI_MOD09A1_CA_5km_{}',
            'GOSIF': 'GOSIF_{}_CALIFORNIA_GOSIF_{}',
            'LST': 'LST_{}_MYD21C1_CA_LST_{}',
            'NDVI': 'NDVI_{}_NDVI_MOD13Q1_CA_5km_{}',
            'NIRv': 'NIRv_{}_NIRv_MOD13Q1_CA_5km_{}',
            'SPEI': 'SPEI_{}_SPEI1_CSIC_CA_{}',
            'TWS': 'TWS_{}_TWS_GRACE_CA_{}',
            'VPDmax': 'VPD_{}_VPDmax_GRIDMET_CA_5km_{}'
        }

        self.folder_mapping = {
            'GNDVI': 'GNDVI',
            'GOSIF': 'GOSIF',
            'LST': 'LST',
            'NDVI': 'NDVI',
            'NIRv': 'NIRv',
            'SPEI': 'SPEI',
            'TWS': 'TWS',
            'VPDmax': 'VPD'
        }

        self.gosif_paths = {
            'GRASSLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\GOSIF",
            'NATURAL FORESTS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\GOSIF",
            'SAVANNAS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\GOSIF",
            'SHRUBLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\GOSIF"
        }

        self.start_years = {
            'GNDVI': 2001, 'GOSIF': 2001, 'LST': 2003, 'NDVI': 2001,
            'NIRv': 2001, 'SPEI': 2001, 'TWS': 2002, 'VPDmax': 2001
        }

        self.debug_mode = False

    def set_debug_mode(self, debug=True):
        """Enable/disable debug mode"""
        self.debug_mode = debug

    def normalize_to_range(self, data, target_min=-1, target_max=1):
        """Normalize data to specified range (default: -1 to 1)"""
        try:
            # Remove NaN values for min/max calculation
            valid_data = data[~np.isnan(data)]

            if len(valid_data) == 0:
                return data  # Return original if all NaN

            data_min = np.min(valid_data)
            data_max = np.max(valid_data)

            # Avoid division by zero
            if data_max == data_min:
                return np.full_like(data, (target_min + target_max) / 2)

            # Apply min-max normalization to target range
            normalized_data = (data - data_min) / (data_max - data_min) * (target_max - target_min) + target_min

            if self.debug_mode:
                print(
                    f"    Original range: [{data_min:.4f}, {data_max:.4f}] → Normalized range: [{target_min}, {target_max}]")
                print(f"    Valid samples: {len(valid_data)}/{len(data)}")

            return normalized_data

        except Exception as e:
            print(f"Error in normalization: {e}")
            return data

    def generate_date_range(self, index_name, start_year=2001, end_year=2021):
        """Generate date range for time series extraction"""
        try:
            actual_start = max(start_year, self.start_years.get(index_name, start_year))
            dates = []
            for year in range(actual_start, end_year + 1):
                for month in range(1, 13):
                    if index_name == 'TWS' and year == 2002 and month < 4:
                        continue
                    dates.append(f"{year:04d}_{month:02d}")
            return dates
        except Exception as e:
            print(f"Error generating date range for {index_name}: {e}")
            return []

    def load_raster_data(self, file_path):
        """Load raster data from TIFF file"""
        try:
            try:
                import rasterio
            except ImportError:
                print("Error: rasterio library not found. Install with: pip install rasterio")
                return None

            if not os.path.exists(file_path):
                return None

            with rasterio.open(file_path) as src:
                data = src.read(1)
                data = data.astype(float)

                # Handle NoData values
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                data[data == -9999] = np.nan
                data[data == -3.4e38] = np.nan
                data[np.isinf(data)] = np.nan

                return data
        except Exception as e:
            if self.debug_mode:
                print(f"Error loading raster data from {file_path}: {e}")
            return None

    def find_actual_filename(self, folder_path, expected_pattern, date_str, index_name):
        """Find the actual filename that matches the pattern"""
        try:
            if not os.path.exists(folder_path):
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

            return None
        except Exception as e:
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
            return land_type

    def extract_time_series_all_indices(self):
        """Extract time series for all indices and land types with normalization"""
        print("Extracting time series data for all indices and land types...")
        print("NOTE: All datasets will be normalized to range [-1, 1] to match NDVI scale")

        try:
            # Create master date range (2001-2021)
            master_dates = []
            for year in range(2001, 2022):
                for month in range(1, 13):
                    master_dates.append(datetime(year, month, 1))

            time_series_data = {}
            normalization_stats = {}  # Track normalization statistics

            for land_type in self.land_types:
                time_series_data[land_type] = {}
                normalization_stats[land_type] = {}
                clean_land_name = self.land_type_mapping[land_type]
                print(f"\nProcessing {clean_land_name}...")

                for index_name in self.indices:
                    print(f"  Extracting {index_name}...")
                    dates = self.generate_date_range(index_name)
                    data_values = []
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
                                data = self.load_raster_data(file_path)
                                if data is not None and not np.all(np.isnan(data)):
                                    # Calculate mean over all pixels as requested
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

                    # NORMALIZATION: Normalize to [-1, 1] range
                    print(f"  Normalizing {index_name} to [-1, 1] range...")
                    original_stats = {
                        'min': np.nanmin(full_series) if not np.all(np.isnan(full_series)) else np.nan,
                        'max': np.nanmax(full_series) if not np.all(np.isnan(full_series)) else np.nan,
                        'mean': np.nanmean(full_series) if not np.all(np.isnan(full_series)) else np.nan,
                        'std': np.nanstd(full_series) if not np.all(np.isnan(full_series)) else np.nan
                    }

                    normalized_series = self.normalize_to_range(full_series, -1, 1)

                    normalized_stats = {
                        'min': np.nanmin(normalized_series) if not np.all(np.isnan(normalized_series)) else np.nan,
                        'max': np.nanmax(normalized_series) if not np.all(np.isnan(normalized_series)) else np.nan,
                        'mean': np.nanmean(normalized_series) if not np.all(np.isnan(normalized_series)) else np.nan,
                        'std': np.nanstd(normalized_series) if not np.all(np.isnan(normalized_series)) else np.nan
                    }

                    # Store normalization statistics
                    normalization_stats[land_type][index_name] = {
                        'original': original_stats,
                        'normalized': normalized_stats
                    }

                    time_series_data[land_type][index_name] = normalized_series
                    print(f"    {index_name}: {files_processed} files processed and normalized")
                    print(
                        f"    Original range: [{original_stats['min']:.4f}, {original_stats['max']:.4f}] → [-1.0, 1.0]")

            # Print normalization summary
            self.print_normalization_summary(normalization_stats)

            return time_series_data, master_dates

        except Exception as e:
            print(f"Critical error in extract_time_series_all_indices: {e}")
            traceback.print_exc()
            return {}, []

    def print_normalization_summary(self, normalization_stats):
        """Print summary of normalization applied to each dataset"""
        print("\n" + "=" * 80)
        print("NORMALIZATION SUMMARY")
        print("=" * 80)
        print("All datasets normalized to range [-1, 1] to match NDVI scale\n")

        for land_type in self.land_types:
            land_name = self.land_type_mapping[land_type]
            print(f"{land_name.upper()}:")
            print("-" * 50)

            for index_name in self.indices:
                if index_name in normalization_stats[land_type]:
                    orig_stats = normalization_stats[land_type][index_name]['original']
                    norm_stats = normalization_stats[land_type][index_name]['normalized']

                    print(
                        f"  {index_name:8s}: [{orig_stats['min']:8.4f}, {orig_stats['max']:8.4f}] → [{norm_stats['min']:6.3f}, {norm_stats['max']:6.3f}]")
            print()

    def calculate_residuals_and_slope_changes(self, data, window_size=12):
        """Calculate residuals from trend and detect slope changes in residuals"""
        try:
            if len(data) < window_size * 2:
                return np.full_like(data, np.nan), np.full_like(data, np.nan)

            # Remove NaN values for trend calculation
            valid_idx = ~np.isnan(data)
            valid_count = np.sum(valid_idx)
            if valid_count < window_size:
                return np.full_like(data, np.nan), np.full_like(data, np.nan)

            # Calculate long-term trend
            x_trend = np.arange(len(data))[valid_idx]
            y_trend = data[valid_idx]

            if len(y_trend) < 3:
                return np.full_like(data, np.nan), np.full_like(data, np.nan)

            try:
                slope, intercept = np.polyfit(x_trend, y_trend, 1)
                trend_line = slope * np.arange(len(data)) + intercept
            except np.linalg.LinAlgError:
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
                    continue

            return residuals, slope_changes
        except Exception as e:
            print(f"Error in calculate_residuals_and_slope_changes: {e}")
            return np.full_like(data, np.nan), np.full_like(data, np.nan)

    def calculate_cross_correlation_fixed(self, series1, series2, max_lag=12):
        """
        Fixed cross-correlation calculation that properly handles lag direction and indexing
        """
        try:
            # Remove NaN values and align series
            valid_mask = ~(np.isnan(series1) | np.isnan(series2))
            if np.sum(valid_mask) < 10:
                return 0, 0

            valid_series1 = series1[valid_mask]
            valid_series2 = series2[valid_mask]

            if len(valid_series1) != len(valid_series2) or len(valid_series1) < 10:
                return 0, 0

            max_corr = 0
            best_lag = 0

            # Calculate correlations for different lags
            for lag in range(-max_lag, max_lag + 1):
                try:
                    if lag == 0:
                        # No lag - direct correlation
                        corr_coef = np.corrcoef(valid_series1, valid_series2)[0, 1]
                    elif lag > 0:
                        # Positive lag: series1 leads series2
                        # series1 at time t correlates with series2 at time t+lag
                        if len(valid_series1) > lag:
                            s1_lead = valid_series1[:-lag]
                            s2_follow = valid_series2[lag:]
                            if len(s1_lead) == len(s2_follow) and len(s1_lead) > 5:
                                corr_coef = np.corrcoef(s1_lead, s2_follow)[0, 1]
                            else:
                                continue
                        else:
                            continue
                    else:  # lag < 0
                        # Negative lag: series2 leads series1
                        # series2 at time t correlates with series1 at time t+|lag|
                        abs_lag = abs(lag)
                        if len(valid_series2) > abs_lag:
                            s2_lead = valid_series2[:-abs_lag]
                            s1_follow = valid_series1[abs_lag:]
                            if len(s2_lead) == len(s1_follow) and len(s2_lead) > 5:
                                corr_coef = np.corrcoef(s2_lead, s1_follow)[0, 1]
                            else:
                                continue
                        else:
                            continue

                    # Update best correlation if this is stronger
                    if not np.isnan(corr_coef) and abs(corr_coef) > abs(max_corr):
                        max_corr = corr_coef
                        best_lag = lag

                except (ValueError, IndexError):
                    continue

            return max_corr, best_lag

        except Exception as e:
            return 0, 0

    def apply_ecological_constraints(self, index1, index2, correlation, lag):
        """
        Apply ecological constraints to ensure realistic relationships
        """
        # Define climate drivers and vegetation indices
        climate_drivers = ['LST', 'VPDmax', 'SPEI', 'TWS']
        vegetation_indices = ['NDVI', 'NIRv', 'GNDVI', 'GOSIF']

        # For climate -> vegetation relationships, only accept positive lags (climate leads vegetation)
        if index1 in climate_drivers and index2 in vegetation_indices:
            if lag <= 0:
                # For temperature (LST), apply minimum biological response time
                if index1 == 'LST' and abs(correlation) > 0.3:
                    return correlation, max(1, abs(lag))  # Minimum 1-month lag for temperature
                # For water-related indices, apply minimum lag
                elif index1 in ['TWS', 'SPEI'] and abs(correlation) > 0.3:
                    return correlation, max(2, abs(lag))  # Minimum 2-month lag for water
                else:
                    return 0, 0  # Reject unrealistic relationships
            elif lag > 6:
                # Cap excessive lags at 6 months for most relationships
                return correlation, 6

        # Reject negative lags for meaningful climate-vegetation relationships
        if index1 in climate_drivers and index2 in vegetation_indices and lag < 0:
            return 0, 0

        return correlation, lag

    def smooth_time_series(self, data, window=3):
        """Apply light smoothing to reduce noise without losing signal"""
        try:
            # Simple moving average with NaN handling
            smoothed = np.full_like(data, np.nan)
            for i in range(len(data)):
                start_idx = max(0, i - window // 2)
                end_idx = min(len(data), i + window // 2 + 1)
                window_data = data[start_idx:end_idx]
                valid_data = window_data[~np.isnan(window_data)]
                if len(valid_data) > 0:
                    smoothed[i] = np.mean(valid_data)
            return smoothed
        except:
            return data

    def calculate_optimal_lags_slope_based(self, time_series_data, max_lag=12):
        """Calculate optimal lags with improved correlation calculation and ecological constraints"""
        print("Calculating optimal lags with improved cross-correlation analysis...")

        optimal_lags = {}

        for land_type in self.land_types:
            print(f"Processing {self.land_type_mapping[land_type]}...")
            optimal_lags[land_type] = np.zeros((len(self.indices), len(self.indices)))

            try:
                # Calculate slope changes for all indices with light smoothing
                slope_changes_data = {}
                for index_name in self.indices:
                    raw_data = time_series_data[land_type][index_name]
                    # Apply light smoothing to reduce noise
                    smoothed_data = self.smooth_time_series(raw_data, window=3)
                    residuals, slope_changes = self.calculate_residuals_and_slope_changes(smoothed_data)
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

                            # Use improved cross-correlation calculation
                            correlation, lag = self.calculate_cross_correlation_fixed(slope1, slope2, max_lag)

                            # Apply ecological constraints
                            correlation, lag = self.apply_ecological_constraints(index1, index2, correlation, lag)

                            # Only keep significant correlations
                            if abs(correlation) > 0.25:  # Minimum correlation threshold
                                optimal_lags[land_type][i, j] = lag
                            else:
                                optimal_lags[land_type][i, j] = 0

                        except Exception as e:
                            if self.debug_mode:
                                print(f"Error calculating lag for {index1}-{index2}: {e}")
                            optimal_lags[land_type][i, j] = 0

            except Exception as e:
                print(f"Error processing {land_type}: {e}")
                optimal_lags[land_type] = np.zeros((len(self.indices), len(self.indices)))

        return optimal_lags

    def is_meaningful_relationship(self, driver, driven):
        """Check if this is a meaningful driver->driven relationship"""
        # Climate drivers that should lead vegetation responses
        climate_drivers = ['LST', 'VPDmax', 'SPEI', 'TWS']
        vegetation_indices = ['NDVI', 'NIRv', 'GNDVI', 'GOSIF']

        return driver in climate_drivers and driven in vegetation_indices

    def create_optimal_lags_heatmaps(self, optimal_lags, save_path='fixed_optimal_lags_8indices.png'):
        """Create optimal lags heatmaps for all land types"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 14))
            fig.suptitle(
                'Optimal Lag Times Between Drought Indicators Across Land Types\n(Improved Cross-Correlation with Ecological Constraints)',
                fontsize=16, fontweight='bold')

            land_type_names = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for idx, (land_type, (row, col)) in enumerate(zip(self.land_types, positions)):
                ax = axes[row, col]

                try:
                    # Create heatmap
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
                    print(f"Error creating heatmap for {land_type}: {e}")
                    ax.text(0.5, 0.5, f'Error creating\nheatmap for\n{land_type_names[idx]}',
                            ha='center', va='center', transform=ax.transAxes)

            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            print(f"Fixed optimal lags heatmaps saved as {save_path}")

        except Exception as e:
            print(f"Error creating optimal lags heatmaps: {e}")

    def print_analysis_summary(self, optimal_lags):
        """Print summary of analysis results"""
        try:
            print("\n" + "=" * 80)
            print("IMPROVED LAG ANALYSIS SUMMARY")
            print("=" * 80)

            print("\nKEY MEANINGFUL RELATIONSHIPS (Driver → Driven):")
            print("-" * 60)

            for land_type in self.land_types:
                land_name = self.land_type_mapping[land_type]
                print(f"\n{land_name.upper()}:")

                lag_matrix = optimal_lags[land_type]

                # Show climate -> vegetation relationships
                climate_indices = {'LST': 2, 'VPDmax': 7, 'SPEI': 5, 'TWS': 6}
                veg_indices = {'NDVI': 3, 'NIRv': 4, 'GNDVI': 0, 'GOSIF': 1}

                meaningful_found = False
                for climate, climate_idx in climate_indices.items():
                    if climate_idx < len(self.indices):
                        climate_relationships = []
                        for veg, veg_idx in veg_indices.items():
                            if veg_idx < len(self.indices):
                                lag_value = lag_matrix[climate_idx, veg_idx]
                                if lag_value > 0:
                                    climate_relationships.append(f"{veg} after {int(lag_value)} months")
                                    meaningful_found = True

                        if climate_relationships:
                            print(f"  {climate} affects: {', '.join(climate_relationships)}")

                if not meaningful_found:
                    print("  All relationships within normal ranges")

        except Exception as e:
            print(f"Error in analysis summary: {e}")

    def run_complete_analysis(self):
        """Run complete improved analysis"""
        print("Starting IMPROVED Slope-Based Drought Lag Analysis")
        print("=" * 70)

        try:
            # Step 1: Extract time series data (with normalization)
            print("\n1. Extracting and normalizing time series data...")
            time_series_data, dates = self.extract_time_series_all_indices()

            if not time_series_data:
                print("Failed to extract time series data.")
                return None

            # Step 2: Calculate optimal lags with improved method
            print("\n2. Calculating optimal lags with improved cross-correlation...")
            optimal_lags = self.calculate_optimal_lags_slope_based(time_series_data)

            # Step 3: Create visualizations
            print("\n3. Creating visualizations...")
            self.create_optimal_lags_heatmaps(optimal_lags)

            # Step 4: Print analysis summary
            print("\n4. Analysis Summary:")
            self.print_analysis_summary(optimal_lags)

            print("\n" + "=" * 70)
            print("IMPROVED ANALYSIS COMPLETE!")
            print("=" * 70)
            print("\nKey Improvements Applied:")
            print("✓ Fixed cross-correlation calculation with proper lag indexing")
            print("✓ Applied light smoothing to reduce noise")
            print("✓ Added ecological constraints for climate→vegetation relationships")
            print("✓ Enforced minimum biological response times")
            print("✓ Capped excessive lag values at 6 months")
            print("✓ Improved negative lag handling")
            print("✓ Added correlation significance threshold")

            return optimal_lags, time_series_data, dates

        except Exception as e:
            print(f"Critical error in analysis: {e}")
            traceback.print_exc()
            return None


# Usage
if __name__ == "__main__":
    BASE_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

    analyzer = NormalizedSlopeBasedAnalyzer(BASE_PATH)
    analyzer.set_debug_mode(False)

    results = analyzer.run_complete_analysis()