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

    # Create comprehensive and easy-to-understand visualizations
    print("\nCreating improved visualizations...")

    # Set up the plot style
    plt.style.use('default')  # Use cleaner default style
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # Define clear color scheme for drought conditions
    drought_colors = {
        'severe_drought': '#8B0000',  # Dark red
        'moderate_drought': '#FF6B6B',  # Light red
        'normal': '#4ECDC4',  # Teal
        'wet': '#45B7D1',  # Blue
        'very_wet': '#0066CC'  # Dark blue
    }

    # Figure 1: Drought Timeline - Single clear overview
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create drought severity matrix
    years = yearly_residuals['year'].values
    datasets = [col for col in all_data.keys() if col in yearly_residuals.columns]

    drought_matrix = np.zeros((len(datasets), len(years)))
    drought_labels = np.full((len(datasets), len(years)), '', dtype='<U10')

    for i, dataset in enumerate(datasets):
        z_scores = yearly_residuals[f'{dataset}_zscore'].values
        for j, (year, z_score) in enumerate(zip(years, z_scores)):
            drought_matrix[i, j] = z_score

            # Create clear labels
            if z_score <= -2:
                drought_labels[i, j] = 'SEVERE'
            elif z_score <= -1:
                drought_labels[i, j] = 'MODERATE'
            elif z_score <= -0.5:
                drought_labels[i, j] = 'MILD'
            elif z_score >= 2:
                drought_labels[i, j] = 'V.WET'
            elif z_score >= 1:
                drought_labels[i, j] = 'WET'
            else:
                drought_labels[i, j] = 'NORMAL'

    # Create the heatmap with clear drought categories
    im = ax.imshow(drought_matrix, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)

    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(years)):
            color = 'white' if abs(drought_matrix[i, j]) > 1.5 else 'black'
            ax.text(j, i, drought_labels[i, j], ha='center', va='center',
                    fontsize=8, fontweight='bold', color=color)

    # Customize the plot
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45)
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels(datasets)
    ax.set_xlabel('Year', fontsize=14, fontweight='bold')
    ax.set_ylabel('Climate Indicators', fontsize=14, fontweight='bold')
    ax.set_title('California Drought Conditions Timeline (2001-2021)\n' +
                 'SEVERE/MODERATE = Drought | WET/V.WET = Above Normal | NORMAL = Typical',
                 fontsize=16, fontweight='bold', pad=20)

    # Add colorbar with clear labels
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Drought Severity (Z-Score)', fontsize=12, fontweight='bold')
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(['Extreme\nDrought', 'Severe\nDrought', 'Moderate\nDrought',
                         'Normal', 'Moderate\nWet', 'Very\nWet', 'Extreme\nWet'])

    # Highlight known drought periods with boxes
    for start_year, end_year, label in DROUGHT_PERIODS:
        start_idx = start_year - years.min()
        end_idx = end_year - years.min()
        rect = Rectangle((start_idx - 0.5, -0.5), end_idx - start_idx + 1, len(datasets),
                         fill=False, edgecolor='black', linewidth=3)
        ax.add_patch(rect)

        # Add drought period label
        ax.text((start_idx + end_idx) / 2, len(datasets), label,
                ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('california_drought_timeline_clear.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 2: Drought Summary Dashboard
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Main drought timeline (simplified)
    ax_main = fig.add_subplot(gs[0, :])

    # Calculate overall drought index (average of all indicators)
    drought_indicators = ['SPEI', 'GOSIF', 'NDVI', 'NIRv']  # Drought-sensitive indicators
    available_indicators = [ind for ind in drought_indicators if ind in yearly_residuals.columns]

    if available_indicators:
        overall_drought = yearly_residuals[[f'{ind}_zscore' for ind in available_indicators]].mean(axis=1)

        # Plot as filled area with clear color coding
        colors_timeline = []
        for score in overall_drought:
            if score <= -2:
                colors_timeline.append(drought_colors['severe_drought'])
            elif score <= -1:
                colors_timeline.append(drought_colors['moderate_drought'])
            elif score >= 1:
                colors_timeline.append(drought_colors['wet'])
            else:
                colors_timeline.append(drought_colors['normal'])

        bars = ax_main.bar(years, overall_drought, color=colors_timeline, alpha=0.8, width=0.8)

        # Add value labels on bars
        for bar, score in zip(bars, overall_drought):
            height = bar.get_height()
            if abs(height) > 0.5:  # Only label significant values
                condition = ''
                if height <= -2:
                    condition = 'SEVERE DROUGHT'
                elif height <= -1:
                    condition = 'DROUGHT'
                elif height >= 1:
                    condition = 'WET'

                ax_main.text(bar.get_x() + bar.get_width() / 2., height + (0.1 if height > 0 else -0.1),
                             condition, ha='center', va='bottom' if height > 0 else 'top',
                             fontsize=8, fontweight='bold', rotation=90)

    ax_main.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax_main.axhline(y=-1, color='orange', linestyle='--', alpha=0.7, label='Drought Threshold')
    ax_main.axhline(y=-2, color='red', linestyle='--', alpha=0.7, label='Severe Drought')
    ax_main.set_ylabel('Overall Drought Index', fontsize=12, fontweight='bold')
    ax_main.set_title('California Overall Drought Conditions', fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    ax_main.legend()

    # Individual indicator plots
    plot_positions = [(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)]

    for idx, dataset_name in enumerate(datasets[:6]):  # Limit to 6 indicators
        if idx < len(plot_positions):
            ax = fig.add_subplot(gs[plot_positions[idx]])

            values = yearly_residuals[f'{dataset_name}_zscore'].values
            colors_ind = []
            for val in values:
                if val <= -2:
                    colors_ind.append(drought_colors['severe_drought'])
                elif val <= -1:
                    colors_ind.append(drought_colors['moderate_drought'])
                elif val >= 1:
                    colors_ind.append(drought_colors['wet'])
                else:
                    colors_ind.append(drought_colors['normal'])

            bars = ax.bar(years, values, color=colors_ind, alpha=0.8, width=0.6)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
            ax.axhline(y=-2, color='red', linestyle='--', alpha=0.5)

            ax.set_title(f'{dataset_name}', fontsize=11, fontweight='bold')
            ax.set_ylabel('Drought Score', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)

    plt.suptitle('California Drought Analysis Dashboard (2001-2021)\n' +
                 'Red = Drought | Orange = Mild Drought | Teal = Normal | Blue = Wet',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('california_drought_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Figure 3: Simple Drought Years Summary
    fig, ax = plt.subplots(figsize=(14, 8))

    # Count drought conditions per year
    drought_counts = []
    year_labels = []

    for year in years:
        year_idx = list(years).index(year)
        severe_count = sum(1 for dataset in datasets if yearly_residuals[f'{dataset}_zscore'].iloc[year_idx] <= -2)
        moderate_count = sum(
            1 for dataset in datasets if -2 < yearly_residuals[f'{dataset}_zscore'].iloc[year_idx] <= -1)

        drought_counts.append({
            'year': year,
            'severe': severe_count,
            'moderate': moderate_count,
            'total_indicators': len(datasets)
        })

    # Plot stacked bars
    severe_counts = [d['severe'] for d in drought_counts]
    moderate_counts = [d['moderate'] for d in drought_counts]

    p1 = ax.bar(years, severe_counts, color=drought_colors['severe_drought'],
                label='Severe Drought Indicators', alpha=0.9)
    p2 = ax.bar(years, moderate_counts, bottom=severe_counts,
                color=drought_colors['moderate_drought'],
                label='Moderate Drought Indicators', alpha=0.9)

    # Add percentage labels
    for i, (year, severe, moderate) in enumerate(zip(years, severe_counts, moderate_counts)):
        total_drought = severe + moderate
        if total_drought > 0:
            percentage = (total_drought / len(datasets)) * 100
            ax.text(year, total_drought + 0.1, f'{percentage:.0f}%',
                    ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Number of Drought Indicators', fontsize=12, fontweight='bold')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_title('Drought Indicator Summary by Year\n' +
                 'Shows how many climate indicators detected drought conditions each year',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Highlight major drought years
    for start_year, end_year, label in DROUGHT_PERIODS:
        ax.axvspan(start_year, end_year, alpha=0.1, color='red')
        ax.text((start_year + end_year) / 2, max(severe_counts) + max(moderate_counts) + 0.5,
                label, ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('california_drought_summary.png', dpi=300, bbox_inches='tight')
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