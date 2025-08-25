import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.linalg import svd
import warnings
import seaborn as sns
from matplotlib.patches import Rectangle

warnings.filterwarnings('ignore')

# Updated dataset paths for California data
data_paths = {
    'GOSIF': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\GOSIF_CA_CLEAN",
    'LST': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\MODIS_CA_Monthly_LST",
    'NDVI': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\NDVI_California_NaturalOnly",
    'NIRv': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\NIRv_MOD13Q1_5km_TIFs",
    'SPEI': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\SPEI_Masked_Output",
    'TWS': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\TWS_Masked_Output"
}

# Known California drought periods
DROUGHT_PERIODS = [
    (2007, 2009, 'Moderate Drought'),
    (2012, 2016, 'Severe Drought'),  # The "Great Drought"
    (2020, 2021, 'Flash Drought')
]


def extract_date_from_filename(filename, dataset_name):
    """Extract date from filename based on dataset naming convention"""
    basename = os.path.basename(filename)

    try:
        if dataset_name == 'GOSIF':
            # CALIFORNIA_GOSIF_MASKED_200101.tif
            parts = basename.split('_')
            if len(parts) >= 4:
                date_str = parts[3].split('.')[0]  # Remove file extension: 200101
                if len(date_str) == 6:  # YYYYMM
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                else:
                    print(f"Invalid date format in {basename}: expected YYYYMM, got {date_str}")
                    return None
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'LST':
            # MYD21C1_CA_LST_masked_2002_07.tif
            parts = basename.split('_')
            if len(parts) >= 6:
                year = int(parts[4])
                month_str = parts[5].split('.')[0]  # Remove file extension
                month = int(month_str)
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'NDVI':
            # NDVI_CA_natland_5km_2001_01.tif
            parts = basename.split('_')
            if len(parts) >= 6:
                year = int(parts[4])
                month = int(parts[5].split('.')[0])  # Remove file extension
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'NIRv':
            # NIRv_CA_natland_5km_2001_01.tif
            parts = basename.split('_')
            if len(parts) >= 6:
                year = int(parts[4])
                month = int(parts[5].split('.')[0])  # Remove file extension
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'SPEI':
            # Note: Your example shows NIRv format for SPEI, but I'll assume it follows SPEI pattern
            # If it's actually NIRv format, change this to match NIRv parsing above
            parts = basename.split('_')
            if 'SPEI' in basename.upper():
                # Assuming format like: SPEI_CA_masked_2001_01.tif or similar
                # Look for year and month in the parts
                year_found = False
                for i, part in enumerate(parts):
                    if part.isdigit() and len(part) == 4 and 1900 <= int(part) <= 2100:
                        year = int(part)
                        if i + 1 < len(parts):
                            month_part = parts[i + 1].split('.')[0]
                            if month_part.isdigit():
                                month = int(month_part)
                                year_found = True
                                break

                if not year_found:
                    print(f"Could not find year/month in {basename}")
                    return None
            else:
                # If it follows NIRv format: NIRv_CA_natland_5km_2001_01.tif
                if len(parts) >= 6:
                    year = int(parts[4])
                    month = int(parts[5].split('.')[0])
                else:
                    print(f"Insufficient parts in {basename}: {parts}")
                    return None

        elif dataset_name == 'TWS':
            # TWS_GRACE_CA_2002_04_masked.tif
            parts = basename.split('_')
            if len(parts) >= 5:
                year = int(parts[3])
                month = int(parts[4])
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        # Validate year and month ranges
        if year < 1900 or year > 2100:
            print(f"Invalid year in {basename}: {year}")
            return None

        if month < 1 or month > 12:
            print(f"Invalid month in {basename}: {month} (should be 1-12)")
            return None

        return datetime(year, month, 1)

    except (ValueError, IndexError) as e:
        print(f"Error parsing {basename}: {e}")
        return None


def read_tif_and_calculate_mean(filepath):
    """Read TIF file and calculate mean, excluding zeros and NaNs"""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float64)
            valid_data = data[(data != 0) & (~np.isnan(data)) & (data != src.nodata)]

            if len(valid_data) > 0:
                return np.mean(valid_data)
            else:
                return np.nan
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.nan


class SSADecomposer:
    """Singular Spectrum Analysis for time series decomposition"""

    def __init__(self, window_length=None):
        self.window_length = window_length
        self.components = None
        self.reconstructed_series = None

    def decompose(self, series, n_components=None):
        """Decompose time series using SSA"""
        series = np.array(series)
        n = len(series)

        # Default window length (common choice: N/2)
        if self.window_length is None:
            self.window_length = n // 2

        # Create trajectory matrix
        K = n - self.window_length + 1
        X = np.zeros((self.window_length, K))

        for i in range(K):
            X[:, i] = series[i:i + self.window_length]

        # SVD decomposition
        U, sigma, Vt = svd(X, full_matrices=False)

        # Store components
        self.components = []
        for i in range(len(sigma)):
            Xi = sigma[i] * np.outer(U[:, i], Vt[i, :])
            component = self._diagonal_averaging(Xi)
            self.components.append(component)

        return self.components

    def _diagonal_averaging(self, X):
        """Diagonal averaging to convert matrix back to time series"""
        L, K = X.shape
        n = L + K - 1
        series = np.zeros(n)

        for k in range(n):
            if k < L - 1:
                indices = [(i, k - i) for i in range(k + 1)]
            elif k < K:
                indices = [(i, k - i) for i in range(L)]
            else:
                indices = [(i, k - i) for i in range(k - K + 1, L)]

            series[k] = np.mean([X[i, j] for i, j in indices])

        return series

    def reconstruct(self, component_indices):
        """Reconstruct series using selected components"""
        if self.components is None:
            raise ValueError("Must decompose series first")

        reconstructed = np.zeros_like(self.components[0])
        for i in component_indices:
            if i < len(self.components):
                reconstructed += self.components[i]

        return reconstructed

    def get_trend_seasonal_residual(self, series):
        """Get trend, seasonal, and residual components"""
        components = self.decompose(series)

        # Trend: usually first 1-2 components
        trend = components[0]
        if len(components) > 1:
            trend += components[1]

        # Seasonal: components with periodic patterns (typically 2-6)
        seasonal = np.zeros_like(components[0])
        for i in range(2, min(7, len(components))):
            seasonal += components[i]

        # Residual: remaining components (noise + irregular patterns)
        residual = np.zeros_like(components[0])
        for i in range(7, len(components)):
            residual += components[i]

        return trend, seasonal, residual


def process_dataset_monthly(dataset_name, data_path):
    """Process individual dataset for monthly data"""
    print(f"Processing {dataset_name}...")

    tif_files = glob.glob(os.path.join(data_path, "*.tif*"))
    if not tif_files:
        print(f"No TIF files found in {data_path}")
        return pd.DataFrame()

    data_records = []
    skipped_files = []

    for tif_file in tif_files:
        date = extract_date_from_filename(tif_file, dataset_name)
        if date is None:
            skipped_files.append(os.path.basename(tif_file))
            continue

        if date.year < 2001 or date.year > 2021:
            skipped_files.append(f"{os.path.basename(tif_file)} (outside date range: {date.year})")
            continue

        mean_value = read_tif_and_calculate_mean(tif_file)
        data_records.append({
            'year': date.year,
            'month': date.month,
            'date': date,
            'value': mean_value,
            'filename': os.path.basename(tif_file)
        })

    if skipped_files:
        print(f"Skipped {len(skipped_files)} files")
        if len(skipped_files) <= 10:  # Show first 10 skipped files for debugging
            for file in skipped_files[:10]:
                print(f"  - {file}")

    if not data_records:
        print(f"No valid data found for {dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(data_records)
    df = df.sort_values(['year', 'month']).reset_index(drop=True)
    print(f"{dataset_name}: Found {len(df)} monthly records")

    return df[['year', 'month', 'date', 'value']]


def fill_missing_data_interpolation(df, dataset_name):
    """Fill missing data using linear interpolation"""
    df_filled = df.copy()
    df_filled = df_filled.sort_values(['year', 'month']).reset_index(drop=True)

    # Linear interpolation for missing values
    df_filled[dataset_name] = df_filled[dataset_name].interpolate(method='linear')

    # Forward/backward fill for any remaining NaN values at edges
    df_filled[dataset_name] = df_filled[dataset_name].fillna(method='ffill').fillna(method='bfill')

    return df_filled


def create_complete_monthly_timeline():
    """Create complete monthly timeline from 2001 to 2021"""
    dates = []
    for year in range(2001, 2022):
        for month in range(1, 13):
            dates.append({
                'year': year,
                'month': month,
                'date': datetime(year, month, 1)
            })
    return pd.DataFrame(dates)


def detect_anomalies(series, threshold=2.0):
    """Detect anomalies using z-score threshold"""
    z_scores = np.abs(stats.zscore(series))
    return z_scores > threshold


def main():
    # Process all datasets
    all_data = {}

    for dataset_name, path in data_paths.items():
        df = process_dataset_monthly(dataset_name, path)
        if not df.empty:
            all_data[dataset_name] = df

    if not all_data:
        print("No data found for any dataset!")
        return

    # Create complete timeline and merge datasets
    complete_timeline = create_complete_monthly_timeline()
    merged_df = complete_timeline.copy()

    for dataset_name, df in all_data.items():
        df_temp = df[['year', 'month', 'value']].copy()
        df_temp = df_temp.rename(columns={'value': dataset_name})
        merged_df = merged_df.merge(df_temp, on=['year', 'month'], how='left')

    print(f"\nMerged dataframe shape: {merged_df.shape}")
    print("Missing values per dataset:")
    for col in merged_df.columns:
        if col not in ['year', 'month', 'date']:
            missing = merged_df[col].isna().sum()
            print(f"{col}: {missing}/{len(merged_df)} missing ({missing / len(merged_df) * 100:.1f}%)")

    # Fill missing data
    print("\nFilling missing data with interpolation...")
    for dataset_name in all_data.keys():
        merged_df = fill_missing_data_interpolation(merged_df, dataset_name)

    # Apply SSA decomposition to extract residuals (noise + anomalies)
    print("\nApplying SSA decomposition to remove seasonality...")
    ssa_results = {}
    residuals_df = merged_df[['year', 'month', 'date']].copy()

    for dataset_name in all_data.keys():
        if dataset_name in merged_df.columns:
            series = merged_df[dataset_name].values

            # Apply SSA
            ssa = SSADecomposer(window_length=24)  # 2-year window for monthly data
            trend, seasonal, residual = ssa.get_trend_seasonal_residual(series)

            ssa_results[dataset_name] = {
                'original': series,
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual
            }

            # Store residuals for analysis
            residuals_df[dataset_name] = residual
            residuals_df[f'{dataset_name}_zscore'] = stats.zscore(residual)

    # Aggregate to yearly data for visualization
    print("\nAggregating to yearly data...")
    yearly_df = merged_df.groupby('year').agg({
        col: 'mean' for col in all_data.keys()
    }).reset_index()

    yearly_residuals = residuals_df.groupby('year').agg({
        col: 'mean' for col in all_data.keys() if col in residuals_df.columns
    }).reset_index()

    # Add z-scores for yearly residuals
    for dataset_name in all_data.keys():
        if dataset_name in yearly_residuals.columns:
            yearly_residuals[f'{dataset_name}_zscore'] = stats.zscore(yearly_residuals[dataset_name])

    # Create comprehensive visualization
    print("\nCreating visualizations...")

    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Figure 1: SSA Decomposition for one example dataset
    if 'SPEI' in ssa_results:
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        example_dataset = 'SPEI'
        time_axis = merged_df['year'] + (merged_df['month'] - 1) / 12

        # Original
        axes[0].plot(time_axis, ssa_results[example_dataset]['original'], 'b-', linewidth=1)
        axes[0].set_title(f'{example_dataset} - Original Time Series (California)')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(time_axis, ssa_results[example_dataset]['trend'], 'r-', linewidth=2)
        axes[1].set_title('Trend Component (SSA)')
        axes[1].set_ylabel('Value')
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(time_axis, ssa_results[example_dataset]['seasonal'], 'g-', linewidth=1)
        axes[2].set_title('Seasonal Component (SSA)')
        axes[2].set_ylabel('Value')
        axes[2].grid(True, alpha=0.3)

        # Residual (Noise + Anomalies)
        axes[3].plot(time_axis, ssa_results[example_dataset]['residual'], 'purple', linewidth=1)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('Residual Component (Noise + Drought Signals)')
        axes[3].set_ylabel('Value')
        axes[3].set_xlabel('Year')
        axes[3].grid(True, alpha=0.3)

        # Highlight drought periods
        for start_year, end_year, label in DROUGHT_PERIODS:
            for ax in axes:
                ax.axvspan(start_year, end_year, alpha=0.2, color='red', label=label if ax == axes[0] else "")

        axes[0].legend()
        plt.tight_layout()
        plt.savefig('california_ssa_decomposition_example.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Figure 2: Yearly Drought Signal Analysis
    fig, axes = plt.subplots(len(all_data), 1, figsize=(15, 3 * len(all_data)), sharex=True)
    if len(all_data) == 1:
        axes = [axes]

    for i, dataset_name in enumerate(all_data.keys()):
        if dataset_name in yearly_residuals.columns:
            # Plot residuals (drought signals)
            bars = axes[i].bar(yearly_residuals['year'], yearly_residuals[dataset_name],
                               color=colors[i % len(colors)], alpha=0.7, width=0.8)

            # Color bars based on anomaly strength
            for j, bar in enumerate(bars):
                year = yearly_residuals['year'].iloc[j]
                value = yearly_residuals[dataset_name].iloc[j]
                z_score = yearly_residuals[f'{dataset_name}_zscore'].iloc[j]

                if abs(z_score) > 2:  # Strong anomaly
                    bar.set_color('red' if z_score < 0 else 'darkgreen')
                    bar.set_alpha(0.9)
                elif abs(z_score) > 1:  # Moderate anomaly
                    bar.set_color('orange' if z_score < 0 else 'lightgreen')
                    bar.set_alpha(0.8)

            axes[i].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.7)
            axes[i].set_ylabel(f'{dataset_name}\nDrought Signal', fontsize=10)
            axes[i].grid(True, alpha=0.3, axis='y')

            # Highlight drought periods
            for start_year, end_year, label in DROUGHT_PERIODS:
                axes[i].axvspan(start_year, end_year, alpha=0.15, color='red')
                if i == 0:  # Only label on first subplot
                    axes[i].text((start_year + end_year) / 2, axes[i].get_ylim()[1] * 0.8,
                                 label, ha='center', fontsize=8, rotation=0)

            # Add statistics
            std_val = yearly_residuals[dataset_name].std()
            mean_val = yearly_residuals[dataset_name].mean()
            axes[i].text(0.02, 0.95, f'Std: {std_val:.3f}',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=8)

    axes[-1].set_xlabel('Year', fontsize=12)
    axes[-1].set_xticks(range(2001, 2022, 2))

    plt.suptitle('California Drought Signal Analysis - Deseasonalized Anomalies (2001-2021)\n' +
                 'Red/Orange: Drought Conditions | Green: Wet Conditions | Gray: Normal',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig('california_drought_signals_yearly.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 3: Anomaly Detection Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create anomaly matrix
    anomaly_data = yearly_residuals[
        [col + '_zscore' for col in all_data.keys() if col + '_zscore' in yearly_residuals.columns]]
    anomaly_data.index = yearly_residuals['year']
    anomaly_data.columns = [col.replace('_zscore', '') for col in anomaly_data.columns]

    # Create heatmap
    sns.heatmap(anomaly_data.T, cmap='RdBu_r', center=0,
                annot=True, fmt='.1f', cbar_kws={'label': 'Z-Score'},
                xticklabels=True, yticklabels=True, ax=ax)

    ax.set_title('California Drought Anomaly Heatmap - Z-Scores of Deseasonalized Residuals\n' +
                 'Red: Drought Stress | Blue: Above Normal Conditions', fontsize=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Climate Indicators', fontsize=12)

    # Add drought period indicators
    for start_year, end_year, label in DROUGHT_PERIODS:
        start_idx = start_year - yearly_residuals['year'].min()
        end_idx = end_year - yearly_residuals['year'].min()
        ax.add_patch(Rectangle((start_idx, -0.5), end_idx - start_idx + 1, len(all_data),
                               fill=False, edgecolor='black', linewidth=2))

    plt.tight_layout()
    plt.savefig('california_anomaly_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save processed data
    print("\nSaving processed data...")

    # Monthly data with SSA components
    monthly_output = merged_df.copy()
    for dataset_name in all_data.keys():
        if dataset_name in ssa_results:
            monthly_output[f'{dataset_name}_trend'] = ssa_results[dataset_name]['trend']
            monthly_output[f'{dataset_name}_seasonal'] = ssa_results[dataset_name]['seasonal']
            monthly_output[f'{dataset_name}_residual'] = ssa_results[dataset_name]['residual']

    monthly_output['date_string'] = (monthly_output['year'].astype(str) + '-' +
                                     monthly_output['month'].astype(str).str.zfill(2))

    monthly_output.to_excel('california_drought_analysis_monthly_ssa.xlsx', index=False)

    # Yearly drought signals
    yearly_output = yearly_residuals.copy()
    yearly_output.to_excel('california_drought_signals_yearly.xlsx', index=False)

    print("Data saved to:")
    print("- california_drought_analysis_monthly_ssa.xlsx (monthly data with SSA components)")
    print("- california_drought_signals_yearly.xlsx (yearly drought signals)")

    # Summary statistics
    print(f"\nCalifornia Drought Signal Analysis Summary:")
    print(f"Time period: {merged_df['year'].min()} to {merged_df['year'].max()}")

    print("\nStrongest drought signals (|Z-score| > 2):")
    for dataset_name in all_data.keys():
        if f'{dataset_name}_zscore' in yearly_residuals.columns:
            strong_anomalies = yearly_residuals[abs(yearly_residuals[f'{dataset_name}_zscore']) > 2]
            if not strong_anomalies.empty:
                print(f"\n{dataset_name}:")
                for _, row in strong_anomalies.iterrows():
                    condition = "Severe Drought" if row[f'{dataset_name}_zscore'] < -2 else "Very Wet"
                    print(f"  {int(row['year'])}: {condition} (Z={row[f'{dataset_name}_zscore']:.2f})")

    return merged_df, yearly_residuals, ssa_results


if __name__ == "__main__":
    main()