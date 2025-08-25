import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import rasterio
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class CaliforniaLandTypeAnalysis:
    def __init__(self):
        self.land_types = ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']
        self.variables = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']
        self.lag_variables = ['GNDVI', 'GOSIF', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS',
                              'VPD']  # Exclude GPP from lag analysis
        self.base_paths = {
            'GRASSLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS",
            'NATURAL FORESTS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS",
            'SAVANNAS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS",
            'SHRUBLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS"
        }

    def get_file_pattern(self, land_type, variable):
        """Generate file naming patterns based on the provided examples"""
        patterns = {
            'GRASSLANDS': {
                'GNDVI': 'GNDVI_Grasslands_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}',
                'GOSIF': 'GOSIF_Grasslands_CALIFORNIA_GOSIF_{year}{month:02d}',
                'GPP': 'Grasslands_GPP_{year}_{month:02d}',
                'LST': 'LST_Grasslands_MYD21C1_CA_LST_{year}_{month:02d}',
                'NDVI': 'NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'NIRv': 'NIRv_Grasslands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SPEI': 'SPEI_Grasslands_SPEI1_CSIC_CA_{year}_{month:02d}',
                'TWS': 'TWS_Grasslands_TWS_GRACE_CA_{year}_{month:02d}',
                'VPD': 'VPD_Grasslands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}'
            },
            'NATURAL FORESTS': {
                'GNDVI': 'GNDVI_NaturalForests_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}',
                'GOSIF': 'GOSIF_NaturalForests_CALIFORNIA_GOSIF_{year}{month:02d}',
                'GPP': 'NaturalForests_GPP_{year}_{month:02d}',
                'LST': 'LST_NaturalForests_MYD21C1_CA_LST_{year}_{month:02d}',
                'NDVI': 'NDVI_NaturalForests_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'NIRv': 'NIRv_NaturalForests_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SPEI': 'SPEI_NaturalForests_SPEI1_CSIC_CA_{year}_{month:02d}',
                'TWS': 'TWS_NaturalForests_TWS_GRACE_CA_{year}_{month:02d}',
                'VPD': 'VPD_NaturalForests_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}'
            },
            'SAVANNAS': {
                'GNDVI': 'GNDVI_Savannas_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}',
                'GOSIF': 'GOSIF_Savanna_CALIFORNIA_GOSIF_{year}{month:02d}',
                'GPP': 'Savannas_GPP_{year}_{month:02d}',
                'LST': 'LST_Savanna_MYD21C1_CA_LST_{year}_{month:02d}',
                'NDVI': 'NDVI_Savanna_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'NIRv': 'NIRv_Savanna_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SPEI': 'SPEI_Savanna_SPEI1_CSIC_CA_{year}_{month:02d}',
                'TWS': 'TWS_Savanna_TWS_GRACE_CA_{year}_{month:02d}',
                'VPD': 'VPD_Savanna_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}'
            },
            'SHRUBLANDS': {
                'GNDVI': 'GNDVI_Shrublands_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}',
                'GOSIF': 'GOSIF_Shrublands_CALIFORNIA_GOSIF_{year}{month:02d}',
                'GPP': 'Shrublands_GPP_{year}_{month:02d}',
                'LST': 'LST_Shrublands_MYD21C1_CA_LST_{year}_{month:02d}',
                'NDVI': 'NDVI_Shrublands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'NIRv': 'NIRv_Shrublands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SPEI': 'SPEI_Shrublands_SPEI1_CSIC_CA_{year}_{month:02d}',
                'TWS': 'TWS_Shrublands_TWS_GRACE_CA_{year}_{month:02d}',
                'VPD': 'VPD_Shrublands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}'
            }
        }
        return patterns[land_type][variable]

    def load_tif_data(self, filepath):
        """Load and process TIFF file data"""
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                # Handle nodata values
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, np.nan, data)
                return np.nanmean(data)  # Return spatial average
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return np.nan

    def load_time_series(self, land_type, variable):
        """Load time series data for a specific land type and variable"""
        data_path = os.path.join(self.base_paths[land_type], variable)

        # Determine time range based on variable
        start_year = 2001  # Default start year
        end_year = 2021  # Default end year

        if variable == 'GPP':
            start_year = 2001
            end_year = 2011
        elif variable == 'LST':
            start_year = 2003  # LST starts from 2003
            end_year = 2021
        elif variable == 'TWS':
            start_year = 2002  # TWS starts from 2002-04
            end_year = 2021

        time_series = []
        dates = []

        pattern = self.get_file_pattern(land_type, variable)

        for year in range(start_year, end_year + 1):
            start_month = 4 if variable == 'TWS' and year == 2002 else 1
            end_month = 12 if variable != 'GPP' or year <= 2011 else 12

            for month in range(start_month, end_month + 1):
                filename = pattern.format(year=year, month=month) + '.tif'
                filepath = os.path.join(data_path, filename)

                if os.path.exists(filepath):
                    value = self.load_tif_data(filepath)
                    time_series.append(value)
                    dates.append(datetime(year, month, 1))
                else:
                    # Handle missing files - skip rather than add NaN
                    print(f"Missing file: {filename}")
                    continue

        return pd.Series(time_series, index=dates)

    def calculate_cross_correlation_lag(self, series1, series2, max_lag=12):
        """Calculate optimal lag using cross-correlation method"""
        # Remove NaN values and align series
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        if len(df) < 24:  # Need at least 2 years of data
            return np.nan

        s1 = df['s1'].values
        s2 = df['s2'].values

        # Normalize series
        s1 = (s1 - np.mean(s1)) / np.std(s1)
        s2 = (s2 - np.mean(s2)) / np.std(s2)

        best_lag = 0
        best_correlation = 0

        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(s1, s2)[0, 1]
            elif lag > 0:
                if len(s1) > lag:
                    corr = np.corrcoef(s1[:-lag], s2[lag:])[0, 1]
                else:
                    continue
            else:  # lag < 0
                if len(s2) > abs(lag):
                    corr = np.corrcoef(s1[abs(lag):], s2[:lag])[0, 1]
                else:
                    continue

            if not np.isnan(corr) and abs(corr) > abs(best_correlation):
                best_correlation = corr
                best_lag = lag

        return best_lag if abs(best_correlation) > 0.1 else np.nan

    def detect_slope_changes(self, series, window=6):
        """Detect significant slope changes in time series"""
        if len(series) < window * 2:
            return []

        # Remove NaN values and ensure we have enough data
        clean_series = series.dropna()
        if len(clean_series) < window * 2:
            return []

        # Calculate rolling slopes
        slopes = []
        slope_indices = []

        for i in range(window, len(clean_series) - window):
            y = clean_series.iloc[i - window // 2:i + window // 2].values
            if len(y) >= 3 and not np.isnan(y).all():  # Need at least 3 points for regression
                x = np.arange(len(y))
                valid_mask = ~np.isnan(y)
                if np.sum(valid_mask) >= 3:  # Need at least 3 valid points
                    try:
                        slope, _, _, _, _ = stats.linregress(x[valid_mask], y[valid_mask])
                        slopes.append(slope)
                        slope_indices.append(clean_series.index[i])
                    except:
                        slopes.append(np.nan)
                        slope_indices.append(clean_series.index[i])
                else:
                    slopes.append(np.nan)
                    slope_indices.append(clean_series.index[i])
            else:
                slopes.append(np.nan)
                slope_indices.append(clean_series.index[i])

        if not slopes:
            return []

        slopes = pd.Series(slopes, index=slope_indices)

        # Find significant slope changes (peaks and troughs in slope)
        slope_changes = []
        valid_slopes = slopes.dropna()

        if len(valid_slopes) < 3:
            return []

        slope_std = np.nanstd(valid_slopes)
        if slope_std == 0:
            return []

        for i in range(1, len(valid_slopes) - 1):
            prev_slope = valid_slopes.iloc[i - 1]
            curr_slope = valid_slopes.iloc[i]
            next_slope = valid_slopes.iloc[i + 1]

            # Check for significant change in slope direction
            if (prev_slope < curr_slope > next_slope) or (prev_slope > curr_slope < next_slope):
                if abs(curr_slope - prev_slope) > slope_std * 0.5:
                    slope_changes.append((valid_slopes.index[i], curr_slope))

        return slope_changes

    def calculate_correlations(self):
        """Calculate inter-index correlations for each land type"""
        correlation_matrices = {}

        for land_type in self.land_types:
            print(f"Processing correlations for {land_type}...")

            # Load all time series for this land type
            data_dict = {}
            for variable in self.variables:
                try:
                    ts = self.load_time_series(land_type, variable)
                    if not ts.empty and not ts.isna().all():
                        data_dict[variable] = ts
                        print(f"  Loaded {variable}: {len(ts)} data points")
                    else:
                        print(f"  {variable}: No valid data found")
                except Exception as e:
                    print(f"  Error loading {variable} for {land_type}: {e}")

            # Create DataFrame and calculate correlations
            if data_dict:
                df = pd.DataFrame(data_dict)
                # Align time series to common time period
                df = df.dropna(how='all')
                correlation_matrix = df.corr()
                correlation_matrices[land_type] = correlation_matrix

        return correlation_matrices

    def calculate_optimal_lags(self):
        """Calculate optimal lag times between drought indicators"""
        lag_matrices = {}

        for land_type in self.land_types:
            print(f"Processing lag analysis for {land_type}...")

            # Load time series (exclude GPP for lag analysis)
            data_dict = {}
            for variable in self.lag_variables:
                try:
                    ts = self.load_time_series(land_type, variable)
                    if not ts.empty and not ts.isna().all():
                        data_dict[variable] = ts
                        print(f"  Loaded {variable}: {len(ts)} data points")
                    else:
                        print(f"  {variable}: No valid data found")
                except Exception as e:
                    print(f"  Error loading {variable} for {land_type}: {e}")

            if len(data_dict) < 2:
                continue

            # Initialize lag matrix with NaN values
            variables_list = list(data_dict.keys())
            lag_matrix = pd.DataFrame(index=variables_list, columns=variables_list, dtype=float)
            lag_matrix.fillna(np.nan, inplace=True)

            for var1 in variables_list:
                for var2 in variables_list:
                    if var1 == var2:
                        lag_matrix.loc[var1, var2] = 0
                        continue

                    # Method 1: Cross-correlation based lag
                    cross_corr_lag = self.calculate_cross_correlation_lag(data_dict[var1], data_dict[var2])

                    # Method 2: Slope change based lag (original method)
                    slope_lag = np.nan
                    try:
                        changes1 = self.detect_slope_changes(data_dict[var1])
                        changes2 = self.detect_slope_changes(data_dict[var2])

                        if changes1 and changes2:
                            best_lag = 0
                            best_correlation = 0

                            for lag in range(-12, 13):
                                matched_changes = 0
                                total_changes = 0

                                for date1, slope1 in changes1:
                                    target_date = date1 + pd.DateOffset(months=lag)

                                    for date2, slope2 in changes2:
                                        if abs((date2 - target_date).days) <= 30:
                                            total_changes += 1
                                            if (slope1 > 0 and slope2 > 0) or (slope1 < 0 and slope2 < 0):
                                                matched_changes += 1
                                            break

                                if total_changes > 0:
                                    correlation = matched_changes / total_changes
                                    if correlation > best_correlation:
                                        best_correlation = correlation
                                        best_lag = lag

                            if best_correlation > 0.3:
                                slope_lag = best_lag
                    except:
                        pass

                    # Use cross-correlation lag if available, otherwise use slope-based lag
                    if not np.isnan(cross_corr_lag):
                        lag_matrix.loc[var1, var2] = float(cross_corr_lag)
                    elif not np.isnan(slope_lag):
                        lag_matrix.loc[var1, var2] = float(slope_lag)
                    else:
                        # If both methods fail, use a simple correlation-based approach
                        # Find lag with highest absolute correlation
                        best_lag = 0
                        best_corr = 0
                        try:
                            for test_lag in range(-6, 7):  # Test smaller range
                                df = pd.DataFrame({'v1': data_dict[var1], 'v2': data_dict[var2]}).dropna()
                                if len(df) > abs(test_lag) + 12:
                                    if test_lag == 0:
                                        corr = df['v1'].corr(df['v2'])
                                    elif test_lag > 0:
                                        corr = df['v1'].iloc[:-test_lag].corr(df['v2'].iloc[test_lag:])
                                    else:
                                        corr = df['v1'].iloc[abs(test_lag):].corr(df['v2'].iloc[:test_lag])

                                    if not np.isnan(corr) and abs(corr) > abs(best_corr):
                                        best_corr = corr
                                        best_lag = test_lag

                            lag_matrix.loc[var1, var2] = float(best_lag) if abs(best_corr) > 0.2 else 0
                        except:
                            lag_matrix.loc[var1, var2] = 0

            lag_matrices[land_type] = lag_matrix

        return lag_matrices

    def create_correlation_heatmap(self, correlation_matrices):
        """Create correlation heatmap visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        land_type_names = {
            'GRASSLANDS': 'Grasslands',
            'NATURAL FORESTS': 'Natural Forests',
            'SAVANNAS': 'Savanna',
            'SHRUBLANDS': 'Shrublands'
        }

        for idx, land_type in enumerate(self.land_types):
            ax = axes[idx]

            if land_type in lags:
                lag_matrix = lags[land_type]
            # Convert to numeric, handling any non-numeric values
            try:
                lag_numeric = pd.to_numeric(lag_matrix.values.flatten(), errors='coerce')
                valid_lags = lag_numeric[~np.isnan(lag_numeric)]
                if len(valid_lags) > 0:
                    print(f"  Mean lag time: {valid_lags.mean():.1f} months")
                    print(f"  Lag time range: {valid_lags.min():.0f} to {valid_lags.max():.0f} months")
                else:
                    print(f"  No valid lag times calculated")
            except Exception as e:
                print(f"  Error processing lag matrix: {e}")
                corr_matrix = correlation_matrices[land_type]

                # Create heatmap
                sns.heatmap(corr_matrix,
                            annot=True,
                            fmt='.2f',
                            cmap='RdBu_r',
                            center=0,
                            vmin=-1,
                            vmax=1,
                            square=True,
                            cbar_kws={'shrink': 0.8},
                            ax=ax)

                ax.set_title(land_type_names[land_type], fontsize=14, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                ax.text(0.5, 0.5, f'No data for\n{land_type_names[land_type]}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle('Inter-Index Correlations by Land Type', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def create_lag_heatmap(self, lag_matrices):
        """Create optimal lag time heatmap visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        land_type_names = {
            'GRASSLANDS': 'GRASSLANDS',
            'NATURAL FORESTS': 'NATURAL FORESTS',
            'SAVANNAS': 'SAVANNAS',
            'SHRUBLANDS': 'SHRUBLANDS'
        }

        for idx, land_type in enumerate(self.land_types):
            ax = axes[idx]

            if land_type in lag_matrices:
                lag_matrix = lag_matrices[land_type]

                # Convert to numeric and handle NaN values
                lag_numeric = lag_matrix.astype(float)

                # Create heatmap
                sns.heatmap(lag_numeric,
                            annot=True,
                            fmt='.0f',
                            cmap='RdBu_r',
                            center=0,
                            vmin=-12,
                            vmax=12,
                            square=True,
                            cbar_kws={'shrink': 0.8, 'label': 'Optimal Lag (months)'},
                            ax=ax)

                ax.set_title(land_type_names[land_type], fontsize=14, fontweight='bold')
                ax.set_xlabel('')
                ax.set_ylabel('')
            else:
                ax.text(0.5, 0.5, f'No data for\n{land_type_names[land_type]}',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle('Optimal Lag Times Between Drought Indicators Across California Land Types',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    def run_analysis(self):
        """Run the complete analysis and generate visualizations"""
        print("Starting California Land Type Analysis...")

        # Calculate correlations
        print("\n1. Calculating inter-index correlations...")
        correlation_matrices = self.calculate_correlations()

        # Calculate optimal lags
        print("\n2. Calculating optimal lag times...")
        lag_matrices = self.calculate_optimal_lags()

        # Create visualizations
        print("\n3. Creating visualizations...")

        # Correlation heatmap
        corr_fig = self.create_correlation_heatmap(correlation_matrices)
        corr_fig.savefig('california_correlations_heatmap.png', dpi=300, bbox_inches='tight')

        # Lag heatmap
        lag_fig = self.create_lag_heatmap(lag_matrices)
        lag_fig.savefig('california_lag_times_heatmap.png', dpi=300, bbox_inches='tight')

        plt.show()

        print("\nAnalysis complete! Heatmaps saved as PNG files.")

        return correlation_matrices, lag_matrices


# Usage example
if __name__ == "__main__":
    # Initialize analysis
    analyzer = CaliforniaLandTypeAnalysis()

    # Run complete analysis
    correlations, lags = analyzer.run_analysis()

    # Print summary statistics
    print("\nSummary of results:")
    for land_type in analyzer.land_types:
        if land_type in correlations:
            print(f"\n{land_type}:")
            corr_matrix = correlations[land_type]
            print(f"  Variables analyzed: {list(corr_matrix.columns)}")
            print(f"  Mean correlation: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.3f}")
