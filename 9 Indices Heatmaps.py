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

        # File patterns and folder mapping (same as before)
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

    def generate_date_range(self, index_name, start_year=2001, end_year=2021):
        """Generate date range for time series extraction"""
        actual_start = max(start_year, self.start_years[index_name])
        actual_end = end_year

        dates = []
        for year in range(actual_start, actual_end + 1):
            for month in range(1, 13):
                if index_name == 'TWS' and year == 2002 and month < 4:
                    continue
                if index_name == 'GPP' and year > 2011:
                    continue
                dates.append(f"{year:04d}_{month:02d}")
        return dates

    def load_raster_data(self, file_path):
        """Load raster data from TIFF file"""
        try:
            import rasterio
            if not os.path.exists(file_path):
                return None
            with rasterio.open(file_path) as src:
                data = src.read(1)
                data = data.astype(float)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                return data
        except Exception as e:
            return None

    def find_actual_filename(self, folder_path, expected_pattern, date_str, index_name):
        """Find the actual filename that matches the pattern"""
        if not os.path.exists(folder_path):
            return None

        files_in_folder = os.listdir(folder_path)
        exact_file = expected_pattern + '.tif'
        if exact_file in files_in_folder:
            return os.path.join(folder_path, exact_file)

        if index_name == 'GOSIF':
            gosif_date = date_str.replace('_', '')
            for file in files_in_folder:
                if file.endswith('.tif') and gosif_date in file:
                    return os.path.join(folder_path, file)

        year = date_str[:4]
        month = date_str[5:7]
        year_month_patterns = [f"{year}_{month}", f"{year}{month}", f"{year}-{month}"]

        for file in files_in_folder:
            if file.endswith('.tif'):
                for pattern in year_month_patterns:
                    if pattern in file:
                        return os.path.join(folder_path, file)
        return None

    def get_land_type_for_filename(self, land_type, index_name):
        """Get the correct land type name for filename"""
        if index_name == 'GOSIF':
            if land_type == 'SAVANNAS':
                return 'Savanna'
            elif land_type == 'NATURAL FORESTS':
                return 'NaturalForests'
            else:
                return self.land_type_mapping[land_type]
        else:
            return self.land_type_mapping[land_type]

    def extract_time_series_all_indices(self):
        """Extract time series for all indices and land types"""
        print("Extracting time series for all indices and land types...")

        # Create a master date range (2001-2021)
        master_dates = []
        for year in range(2001, 2022):
            for month in range(1, 13):
                master_dates.append(datetime(year, month, 1))

        # Initialize data storage
        time_series_data = {}

        for land_type in self.land_types:
            time_series_data[land_type] = {}
            clean_land_name = self.land_type_mapping[land_type]
            print(f"Processing {clean_land_name}...")

            for index_name in self.indices:
                print(f"  Extracting {index_name}...")
                dates = self.generate_date_range(index_name)
                data_values = []

                for date_str in dates:
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
                        if data is not None:
                            avg_value = np.nanmean(data)
                            data_values.append(avg_value)
                        else:
                            data_values.append(np.nan)
                    else:
                        data_values.append(np.nan)

                # Create full time series aligned with master dates
                full_series = np.full(len(master_dates), np.nan)

                # Fill in available data
                data_idx = 0
                for i, master_date in enumerate(master_dates):
                    if data_idx < len(dates):
                        date_str = dates[data_idx]
                        expected_date = datetime(int(date_str[:4]), int(date_str[5:7]), 1)

                        if master_date == expected_date:
                            full_series[i] = data_values[data_idx]
                            data_idx += 1
                        elif master_date > expected_date:
                            data_idx += 1
                            if data_idx < len(dates):
                                date_str = dates[data_idx]
                                expected_date = datetime(int(date_str[:4]), int(date_str[5:7]), 1)
                                if master_date == expected_date:
                                    full_series[i] = data_values[data_idx]
                                    data_idx += 1

                time_series_data[land_type][index_name] = full_series

        return time_series_data, master_dates

    def calculate_residuals_and_slope_changes(self, data, window_size=12):
        """
        Calculate residuals from trend and detect slope changes in residuals
        This is the key function for detecting unusual changes in datasets
        """
        if len(data) < window_size * 2:
            return np.full_like(data, np.nan), np.full_like(data, np.nan)

        # Remove NaN values for trend calculation
        valid_idx = ~np.isnan(data)
        if np.sum(valid_idx) < window_size:
            return np.full_like(data, np.nan), np.full_like(data, np.nan)

        # Calculate long-term trend
        x_trend = np.arange(len(data))[valid_idx]
        y_trend = data[valid_idx]

        if len(y_trend) < 3:
            return np.full_like(data, np.nan), np.full_like(data, np.nan)

        # Fit linear trend
        slope, intercept = np.polyfit(x_trend, y_trend, 1)
        trend_line = slope * np.arange(len(data)) + intercept

        # Calculate residuals
        residuals = data - trend_line

        # Calculate slope changes in residuals using moving window
        slope_changes = np.full_like(data, np.nan)

        for i in range(window_size, len(residuals) - window_size):
            if np.sum(~np.isnan(residuals[i - window_size:i + window_size])) >= window_size:
                # Calculate slope before and after point i
                before_window = residuals[i - window_size:i]
                after_window = residuals[i:i + window_size]

                # Remove NaN values
                before_valid = before_window[~np.isnan(before_window)]
                after_valid = after_window[~np.isnan(after_window)]

                if len(before_valid) >= 3 and len(after_valid) >= 3:
                    slope_before = np.polyfit(np.arange(len(before_valid)), before_valid, 1)[0]
                    slope_after = np.polyfit(np.arange(len(after_valid)), after_valid, 1)[0]

                    # Calculate change in slope (this detects unusual changes)
                    slope_changes[i] = slope_after - slope_before

        return residuals, slope_changes

    def calculate_optimal_lags_slope_based(self, time_series_data, max_lag=12):
        """
        Calculate optimal lags based on correlation between slope changes in residuals
        This identifies when changes in one dataset affect another dataset
        """
        print("Calculating optimal lags based on slope changes in residuals...")

        optimal_lags = {}

        for land_type in self.land_types:
            print(f"Processing {self.land_type_mapping[land_type]}...")
            optimal_lags[land_type] = np.zeros((len(self.indices), len(self.indices)))

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

                    slope1 = slope_changes_data[index1]
                    slope2 = slope_changes_data[index2]

                    # Find valid overlapping period
                    valid_mask = ~(np.isnan(slope1) | np.isnan(slope2))

                    if np.sum(valid_mask) < 10:  # Need at least 10 valid points
                        optimal_lags[land_type][i, j] = 0
                        continue

                    valid_slope1 = slope1[valid_mask]
                    valid_slope2 = slope2[valid_mask]

                    # Calculate cross-correlation at different lags
                    max_corr = 0
                    best_lag = 0

                    for lag in range(-max_lag, max_lag + 1):
                        if lag == 0:
                            corr_coef = np.corrcoef(valid_slope1, valid_slope2)[0, 1]
                        elif lag > 0:
                            if len(valid_slope1) > lag:
                                corr_coef = np.corrcoef(valid_slope1[:-lag], valid_slope2[lag:])[0, 1]
                            else:
                                continue
                        else:  # lag < 0
                            if len(valid_slope2) > abs(lag):
                                corr_coef = np.corrcoef(valid_slope1[abs(lag):], valid_slope2[:lag])[0, 1]
                            else:
                                continue

                        if not np.isnan(corr_coef) and abs(corr_coef) > abs(max_corr):
                            max_corr = corr_coef
                            best_lag = lag

                    optimal_lags[land_type][i, j] = best_lag

        return optimal_lags

    def calculate_correlations(self, time_series_data):
        """Calculate correlation matrices for raw data"""
        print("Calculating correlation matrices for raw data...")

        correlations = {}

        for land_type in self.land_types:
            print(f"Processing {self.land_type_mapping[land_type]}...")

            # Create data matrix
            data_matrix = []
            for index_name in self.indices:
                data_matrix.append(time_series_data[land_type][index_name])

            data_matrix = np.array(data_matrix)

            # Calculate correlation matrix
            corr_matrix = np.corrcoef(data_matrix)
            correlations[land_type] = corr_matrix

        return correlations

    def create_correlation_heatmaps(self, correlations, save_path='correlation_heatmaps_9indices.png'):
        """Create correlation heatmaps for all land types"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Inter-Index Correlations by Land Type', fontsize=16, fontweight='bold')

        land_type_names = ['Grasslands', 'Natural Forests', 'Savanna', 'Shrublands']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for idx, (land_type, (row, col)) in enumerate(zip(self.land_types, positions)):
            ax = axes[row, col]

            # Create heatmap
            mask = np.triu(np.ones_like(correlations[land_type]), k=1)  # Mask upper triangle

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
                        ax=ax)

            ax.set_title(land_type_names[idx], fontsize=12, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Correlation heatmaps saved as {save_path}")

    def create_optimal_lags_heatmaps(self, optimal_lags, save_path='optimal_lags_9indices.png'):
        """Create optimal lags heatmaps for all land types"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle(
            'Optimal Lag Times Between Drought Indicators Across California Land Types\n(Based on Slope Changes in Residuals)',
            fontsize=16, fontweight='bold')

        land_type_names = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

        for idx, (land_type, (row, col)) in enumerate(zip(self.land_types, positions)):
            ax = axes[row, col]

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

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Optimal lags heatmaps saved as {save_path}")

    def print_analysis_summary(self, correlations, optimal_lags):
        """Print summary of analysis results"""
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)

        print("\nCORRELATION ANALYSIS:")
        print("-" * 40)
        for land_type in self.land_types:
            land_name = self.land_type_mapping[land_type]
            corr_matrix = correlations[land_type]

            # Find highest correlations (excluding diagonal)
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            max_corr_idx = np.unravel_index(np.argmax(np.abs(corr_matrix[mask])), corr_matrix.shape)
            max_corr_value = corr_matrix[max_corr_idx]

            print(f"{land_name}:")
            print(
                f"  Highest correlation: {self.indices[max_corr_idx[0]]} - {self.indices[max_corr_idx[1]]}: {max_corr_value:.3f}")

        print("\nOPTIMAL LAG ANALYSIS (based on slope changes):")
        print("-" * 40)
        for land_type in self.land_types:
            land_name = self.land_type_mapping[land_type]
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

    def run_complete_analysis(self):
        """Run complete correlation and lag analysis"""
        print("Starting comprehensive correlation and lag analysis...")
        print("=" * 80)

        # 1. Extract time series data
        time_series_data, dates = self.extract_time_series_all_indices()

        # 2. Calculate correlations (raw data)
        correlations = self.calculate_correlations(time_series_data)

        # 3. Calculate optimal lags (based on slope changes in residuals)
        optimal_lags = self.calculate_optimal_lags_slope_based(time_series_data)

        # 4. Create visualizations
        print("\nCreating correlation heatmaps...")
        self.create_correlation_heatmaps(correlations)

        print("\nCreating optimal lags heatmaps...")
        self.create_optimal_lags_heatmaps(optimal_lags)

        # 5. Print analysis summary
        self.print_analysis_summary(correlations, optimal_lags)

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nKey Features:")
        print("✓ Correlation analysis using raw time series data")
        print("✓ Optimal lag analysis based on slope changes in residuals")
        print("✓ Detection of unusual changes and their propagation effects")
        print("✓ Comprehensive visualization for all 9 drought indices")
        print("✓ Analysis across all 4 California land types")

        return correlations, optimal_lags, time_series_data


# Usage example
if __name__ == "__main__":
    # Set your base path
    BASE_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

    # Initialize analyzer
    analyzer = DroughtCorrelationAnalyzer(BASE_PATH)

    # Run complete analysis
    correlations, optimal_lags, time_series_data = analyzer.run_complete_analysis()OPtimal