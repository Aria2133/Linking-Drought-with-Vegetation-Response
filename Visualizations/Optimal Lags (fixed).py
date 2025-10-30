import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import rasterio
from scipy import stats
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings('ignore')


class DroughtLagAnalysis:
    def __init__(self, base_paths):
        """
        Initialize the drought lag analysis

        Parameters:
        base_paths: dict with land types as keys and their base paths as values
        """
        self.base_paths = base_paths
        self.land_types = list(base_paths.keys())
        self.indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']
        self.max_lag = 12  # -12 to +12 months

    def get_file_pattern(self, land_type, index):
        """Get the file pattern for a specific land type and index"""
        patterns = {
            'GRASSLAND': {
                'GNDVI': 'GNDVI_Grasslands_GNDVI_MOD09A1_CA_5km_',
                'GOSIF': 'GOSIF_Grasslands_CALIFORNIA_GOSIF_',
                'GPP': 'Grasslands_GPP_',
                'LST': 'LST_Grasslands_MYD21C1_CA_LST_',
                'NDVI': 'NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_',
                'NIRv': 'NIRv_Grasslands_NIRv_MOD13Q1_CA_5km_',
                'SPEI': 'SPEI_Grasslands_SPEI1_CSIC_CA_',
                'TWS': 'TWS_Grasslands_TWS_GRACE_CA_',
                'VPD': 'VPD_Grasslands_VPDmax_GRIDMET_CA_5km_'
            },
            'NATURAL FORESTS': {
                'GNDVI': 'GNDVI_NaturalForests_GNDVI_MOD09A1_CA_5km_',
                'GOSIF': 'GOSIF_NaturalForests_CALIFORNIA_GOSIF_',
                'GPP': 'NaturalForests_GPP_',
                'LST': 'LST_NaturalForests_MYD21C1_CA_LST_',
                'NDVI': 'NDVI_NaturalForests_NDVI_MOD13Q1_CA_5km_',
                'NIRv': 'NIRv_NaturalForests_NIRv_MOD13Q1_CA_5km_',
                'SPEI': 'SPEI_NaturalForests_SPEI1_CSIC_CA_',
                'TWS': 'TWS_NaturalForests_TWS_GRACE_CA_',
                'VPD': 'VPD_NaturalForests_VPDmax_GRIDMET_CA_5km_'
            },
            'SAVANNAS': {
                'GNDVI': 'GNDVI_Savannas_GNDVI_MOD09A1_CA_5km_',
                'GOSIF': 'GOSIF_Savanna_CALIFORNIA_GOSIF_',
                'GPP': 'Savannas_GPP_',
                'LST': 'LST_Savanna_MYD21C1_CA_LST_',
                'NDVI': 'NDVI_Savanna_NDVI_MOD13Q1_CA_5km_',
                'NIRv': 'NIRv_Savanna_NIRv_MOD13Q1_CA_5km_',
                'SPEI': 'SPEI_Savanna_SPEI1_CSIC_CA_',
                'TWS': 'TWS_Savanna_TWS_GRACE_CA_',
                'VPD': 'VPD_Savanna_VPDmax_GRIDMET_CA_5km_'
            },
            'SHRUBLANDS': {
                'GNDVI': 'GNDVI_Shrublands_GNDVI_MOD09A1_CA_5km_',
                'GOSIF': 'GOSIF_Shrublands_CALIFORNIA_GOSIF_',
                'GPP': 'Shrublands_GPP_',
                'LST': 'LST_Shrublands_MYD21C1_CA_LST_',
                'NDVI': 'NDVI_Shrublands_NDVI_MOD13Q1_CA_5km_',
                'NIRv': 'NIRv_Shrublands_NIRv_MOD13Q1_CA_5km_',
                'SPEI': 'SPEI_Shrublands_SPEI1_CSIC_CA_',
                'TWS': 'TWS_Shrublands_TWS_GRACE_CA_',
                'VPD': 'VPD_Shrublands_VPDmax_GRIDMET_CA_5km_'
            }
        }
        return patterns[land_type][index]

    def read_tif_files(self, land_type, index):
        """Read all TIF files for a specific land type and index"""
        index_path = os.path.join(self.base_paths[land_type], index)
        pattern = self.get_file_pattern(land_type, index)

        # Get all TIF files matching the pattern
        files = glob(os.path.join(index_path, f"{pattern}*.tif"))
        files.sort()

        data = []
        dates = []

        print(f"    Processing {len(files)} files for {index}...")

        for file in files:
            try:
                # Extract date from filename
                filename = os.path.basename(file)
                if 'GOSIF' in filename:
                    # GOSIF format: YYYYMM
                    date_part = filename.split('_')[-1].replace('.tif', '')
                    year = int(date_part[:4])
                    month = int(date_part[4:6])
                else:
                    # Standard format: YYYY_MM
                    parts = filename.split('_')
                    year = int(parts[-2])
                    month = int(parts[-1].replace('.tif', ''))

                date = pd.Timestamp(year=year, month=month, day=1)

                # Read raster and calculate spatial mean
                with rasterio.open(file) as src:
                    raster_data = src.read(1)
                    # Handle nodata values more carefully
                    if src.nodata is not None:
                        raster_data = raster_data.astype(float)
                        raster_data[raster_data == src.nodata] = np.nan

                    # Calculate spatial mean, handling NaN values
                    if not np.all(np.isnan(raster_data)):
                        spatial_mean = np.nanmean(raster_data)
                        if not np.isnan(spatial_mean) and not np.isinf(spatial_mean):
                            data.append(spatial_mean)
                            dates.append(date)
                        else:
                            print(f"      Warning: Invalid spatial mean for {filename}")
                    else:
                        print(f"      Warning: All NaN values in {filename}")

            except Exception as e:
                print(f"      Error reading {file}: {e}")
                continue

        if len(data) == 0:
            print(f"    No valid data found for {index}")
            return pd.Series(dtype=float)

        # Create time series
        ts = pd.Series(data, index=dates)
        ts = ts.sort_index()
        ts = ts.dropna()

        print(f"    Final time series: {len(ts)} valid data points")
        return ts

    def remove_seasonality(self, ts):
        """Remove seasonality from time series"""
        try:
            # Ensure we have enough data points
            if len(ts) < 24:  # Need at least 2 years for seasonal decomposition
                print(f"      Using detrending for short series ({len(ts)} points)")
                return pd.Series(detrend(ts.values), index=ts.index)

            # Resample to monthly frequency and interpolate missing values
            ts_monthly = ts.resample('M').mean()

            # Fill any remaining gaps with interpolation
            if ts_monthly.isna().sum() > 0:
                ts_monthly = ts_monthly.interpolate(method='linear', limit_direction='both')

            # Perform seasonal decomposition
            decomposition = seasonal_decompose(ts_monthly, model='additive', period=12, extrapolate_trend='freq')
            residuals = decomposition.resid

            # Return residuals aligned with original time series
            return residuals.reindex(ts.index, method='nearest').dropna()

        except Exception as e:
            print(f"      Error in seasonality removal: {e}")
            # Fallback to simple detrending
            return pd.Series(detrend(ts.values), index=ts.index)

    def calculate_lagged_correlation(self, ts1, ts2, max_lag=12):
        """
        Calculate correlation between two time series at different lags
        Returns the lag with maximum absolute correlation
        """
        if len(ts1) == 0 or len(ts2) == 0:
            return 0, 0, {}

        correlations = {}

        # Find overlapping period
        common_dates = ts1.index.intersection(ts2.index)
        if len(common_dates) < 12:  # Need sufficient overlap
            return 0, 0, {}

        ts1_aligned = ts1.loc[common_dates]
        ts2_aligned = ts2.loc[common_dates]

        for lag in range(-max_lag, max_lag + 1):
            try:
                if lag == 0:
                    corr = ts1_aligned.corr(ts2_aligned)
                elif lag > 0:
                    # ts1 leads ts2 by lag months
                    if len(ts1_aligned) > lag:
                        ts1_shifted = ts1_aligned.iloc[:-lag]
                        ts2_shifted = ts2_aligned.iloc[lag:]
                        if len(ts1_shifted) > 6:  # Need minimum data points
                            corr = ts1_shifted.corr(ts2_shifted)
                        else:
                            continue
                    else:
                        continue
                else:
                    # ts2 leads ts1 by abs(lag) months
                    abs_lag = abs(lag)
                    if len(ts2_aligned) > abs_lag:
                        ts1_shifted = ts1_aligned.iloc[abs_lag:]
                        ts2_shifted = ts2_aligned.iloc[:-abs_lag]
                        if len(ts1_shifted) > 6:  # Need minimum data points
                            corr = ts1_shifted.corr(ts2_shifted)
                        else:
                            continue
                    else:
                        continue

                if not np.isnan(corr):
                    correlations[lag] = corr
            except Exception as e:
                continue

        if correlations:
            # Find lag with maximum absolute correlation
            optimal_lag = max(correlations.keys(), key=lambda k: abs(correlations[k]))
            max_corr = correlations[optimal_lag]
            return optimal_lag, max_corr, correlations
        else:
            return 0, 0, {}

    def load_all_data(self, land_type):
        """Load all time series data for a specific land type"""
        data = {}

        print(f"Loading data for {land_type}...")
        for index in self.indices:
            try:
                ts = self.read_tif_files(land_type, index)
                if len(ts) > 0:
                    data[index] = ts
                    print(f"  {index}: {len(ts)} data points from {ts.index.min()} to {ts.index.max()}")
                else:
                    print(f"  {index}: No valid data found")
            except Exception as e:
                print(f"  {index}: Error loading data - {e}")

        return data

    def calculate_optimal_lags(self, data, remove_seasonality=False):
        """Calculate optimal lag matrix for all index pairs"""
        available_indices = list(data.keys())
        n_indices = len(available_indices)

        # Prepare data
        processed_data = {}
        for index in available_indices:
            ts = data[index].copy()
            if remove_seasonality:
                print(f"    Removing seasonality from {index}...")
                ts = self.remove_seasonality(ts)
            processed_data[index] = ts

        # Initialize matrix with NaN
        lag_matrix = np.full((n_indices, n_indices), np.nan)

        for i, idx1 in enumerate(available_indices):
            for j, idx2 in enumerate(available_indices):
                if i == j:
                    lag_matrix[i, j] = 0  # Same index, no lag
                else:
                    print(f"    Calculating lag: {idx1} -> {idx2}")
                    ts1 = processed_data[idx1]
                    ts2 = processed_data[idx2]

                    if len(ts1) > 0 and len(ts2) > 0:
                        optimal_lag, max_corr, _ = self.calculate_lagged_correlation(ts1, ts2, self.max_lag)
                        lag_matrix[i, j] = optimal_lag
                        print(f"      Optimal lag: {optimal_lag} months (corr: {max_corr:.3f})")
                    else:
                        print(f"      No data available for this pair")

        return lag_matrix, available_indices

    def plot_combined_heatmaps(self, results, save_path=None):
        """Plot combined heatmaps for all land types"""

        # Create figure for raw data
        fig_raw, axes_raw = plt.subplots(2, 2, figsize=(20, 16))
        fig_raw.suptitle('Optimal Lag Heatmaps - Raw Data', fontsize=20, y=0.95)

        # Create figure for deseasonalized data
        fig_seas, axes_seas = plt.subplots(2, 2, figsize=(20, 16))
        fig_seas.suptitle('Optimal Lag Heatmaps - Deseasonalized Data', fontsize=20, y=0.95)

        # Flatten axes arrays for easier indexing
        axes_raw = axes_raw.flatten()
        axes_seas = axes_seas.flatten()

        for idx, (land_type, result) in enumerate(results.items()):
            if idx >= 4:  # Only handle first 4 land types
                break

            # Raw data heatmap
            lag_matrix_raw = result['raw']['lag_matrix']
            indices_raw = result['raw']['indices']

            # Create mask for NaN values
            mask_raw = np.isnan(lag_matrix_raw)

            sns.heatmap(lag_matrix_raw,
                        annot=True,
                        fmt='.0f',
                        cmap='RdBu_r',
                        center=0,
                        vmin=-12,
                        vmax=12,
                        xticklabels=indices_raw,
                        yticklabels=indices_raw,
                        mask=mask_raw,
                        cbar_kws={'label': 'Optimal Lag (months)'},
                        ax=axes_raw[idx])

            axes_raw[idx].set_title(f'{land_type}', fontsize=14, pad=10)
            axes_raw[idx].set_xlabel('Target Index')
            axes_raw[idx].set_ylabel('Source Index')

            # Deseasonalized data heatmap
            lag_matrix_seas = result['deseasonalized']['lag_matrix']
            indices_seas = result['deseasonalized']['indices']

            # Create mask for NaN values
            mask_seas = np.isnan(lag_matrix_seas)

            sns.heatmap(lag_matrix_seas,
                        annot=True,
                        fmt='.0f',
                        cmap='RdBu_r',
                        center=0,
                        vmin=-12,
                        vmax=12,
                        xticklabels=indices_seas,
                        yticklabels=indices_seas,
                        mask=mask_seas,
                        cbar_kws={'label': 'Optimal Lag (months)'},
                        ax=axes_seas[idx])

            axes_seas[idx].set_title(f'{land_type}', fontsize=14, pad=10)
            axes_seas[idx].set_xlabel('Target Index')
            axes_seas[idx].set_ylabel('Source Index')

        # Hide unused subplots if less than 4 land types
        for idx in range(len(results), 4):
            axes_raw[idx].axis('off')
            axes_seas[idx].axis('off')

        plt.tight_layout()

        if save_path:
            os.makedirs(save_path, exist_ok=True)

            # Save raw data plot
            raw_path = os.path.join(save_path, 'optimal_lags_raw_data.png')
            fig_raw.savefig(raw_path, dpi=300, bbox_inches='tight')
            print(f"Saved raw data plot: {raw_path}")

            # Save deseasonalized data plot
            seas_path = os.path.join(save_path, 'optimal_lags_deseasonalized.png')
            fig_seas.savefig(seas_path, dpi=300, bbox_inches='tight')
            print(f"Saved deseasonalized plot: {seas_path}")

        plt.show()
        return fig_raw, fig_seas

    def analyze_all_land_types(self, save_path=None):
        """Perform complete analysis for all land types"""
        results = {}

        # Create save directory if specified
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            print(f"Results will be saved to: {save_path}")

        for land_type in self.land_types:
            print(f"\n{'=' * 50}")
            print(f"Analyzing {land_type}")
            print(f"{'=' * 50}")

            # Load data
            data = self.load_all_data(land_type)

            if len(data) < 2:
                print(f"Insufficient data for {land_type}")
                continue

            # Analysis with raw data
            print("\nCalculating optimal lags for raw data...")
            lag_matrix_raw, indices_raw = self.calculate_optimal_lags(data, remove_seasonality=False)

            # Analysis with deseasonalized data
            print("\nCalculating optimal lags for deseasonalized data...")
            lag_matrix_seas, indices_seas = self.calculate_optimal_lags(data, remove_seasonality=True)

            # Store results
            results[land_type] = {
                'raw': {
                    'lag_matrix': lag_matrix_raw,
                    'indices': indices_raw
                },
                'deseasonalized': {
                    'lag_matrix': lag_matrix_seas,
                    'indices': indices_seas
                }
            }

            print(f"Completed analysis for {land_type}")

        # Create combined plots
        print("\n" + "=" * 50)
        print("Creating combined visualizations...")
        print("=" * 50)

        if results:
            self.plot_combined_heatmaps(results, save_path)

        return results

    def print_summary_statistics(self, results):
        """Print summary statistics for all results"""
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)

        for land_type, result in results.items():
            print(f"\n{land_type}:")

            # Raw data statistics
            lag_matrix_raw = result['raw']['lag_matrix']
            valid_lags_raw = lag_matrix_raw[~np.isnan(lag_matrix_raw) & (lag_matrix_raw != 0)]

            if len(valid_lags_raw) > 0:
                print(f"  Raw data:")
                print(f"    Average lag magnitude: {np.abs(valid_lags_raw).mean():.1f} months")
                print(f"    Max lag: {np.abs(valid_lags_raw).max():.0f} months")
                print(f"    Most common lag: {stats.mode(valid_lags_raw.astype(int), keepdims=True)[0][0]} months")
            else:
                print(f"  Raw data: No valid lag relationships found")

            # Deseasonalized data statistics
            lag_matrix_seas = result['deseasonalized']['lag_matrix']
            valid_lags_seas = lag_matrix_seas[~np.isnan(lag_matrix_seas) & (lag_matrix_seas != 0)]

            if len(valid_lags_seas) > 0:
                print(f"  Deseasonalized data:")
                print(f"    Average lag magnitude: {np.abs(valid_lags_seas).mean():.1f} months")
                print(f"    Max lag: {np.abs(valid_lags_seas).max():.0f} months")
                print(f"    Most common lag: {stats.mode(valid_lags_seas.astype(int), keepdims=True)[0][0]} months")
            else:
                print(f"  Deseasonalized data: No valid lag relationships found")


# Usage example
if __name__ == "__main__":
    # Define your data paths
    base_paths = {
        'GRASSLAND': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS",
        'NATURAL FORESTS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS",
        'SAVANNAS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS",
        'SHRUBLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS"
    }

    # Initialize analyzer
    analyzer = DroughtLagAnalysis(base_paths)

    # Run complete analysis
    results = analyzer.analyze_all_land_types(save_path=r"C:\Users\LOQ\OneDrive\Documents\Results")

    # Print summary statistics
    analyzer.print_summary_statistics(results)