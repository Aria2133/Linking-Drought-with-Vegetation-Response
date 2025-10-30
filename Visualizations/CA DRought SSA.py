import os
import glob
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.linalg import svd
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
        # CALIFORNIA_GOSIF_MASKED_200101.tif
        date_str = basename.split('_')[-1].split('.')[0]  # Remove extension
        if len(date_str) == 6:  # YYYYMM
            year = int(date_str[:4])
            month = int(date_str[4:6])
        else:
            return None

    elif dataset_name == 'LST':
        # MYD21C1_CA_LST_masked_2002_07
        parts = basename.split('_')
        if len(parts) >= 5:
            year = int(parts[-2])
            month = int(parts[-1].split('.')[0])  # Remove file extension
        else:
            return None

    elif dataset_name in ['NDVI', 'NIRv']:
        # NDVI_CA_natland_5km_2001_01 or NIRv_CA_natland_5km_2001_01
        parts = basename.split('_')
        if len(parts) >= 6:
            year = int(parts[-2])
            month = int(parts[-1].split('.')[0])  # Remove file extension
        else:
            return None

    elif dataset_name == 'SPEI':
        # SPEI1_CSIC_CA_2001_01_masked
        parts = basename.split('_')
        if len(parts) >= 5:
            year = int(parts[-3])
            month = int(parts[-2])
        else:
            return None

    elif dataset_name == 'TWS':
        # TWS_GRACE_CA_2002_04_masked
        parts = basename.split('_')
        if len(parts) >= 5:
            year = int(parts[-3])
            month = int(parts[-2])
        else:
            return None

    return datetime(year, month, 1)


def read_tif_and_calculate_mean(filepath):
    """Read TIF file and calculate mean, excluding zeros and NaNs"""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1).astype(np.float64)

            # Remove zeros and NaNs for this specific file
            valid_data = data[(data != 0) & (~np.isnan(data))]

            if len(valid_data) > 0:
                return np.mean(valid_data)
            else:
                return np.nan
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.nan


def simple_gam_trend(dates, values, target_month):
    """Simple GAM-like trend estimation using polynomial regression"""
    if len(dates) < 3:
        return np.nan

    # Convert dates to numerical values (years since first date)
    numeric_dates = np.array([(d.year + d.month / 12.0) for d in dates])

    # Use polynomial regression (degree 2) as simple GAM substitute
    try:
        coeffs = np.polyfit(numeric_dates, values, deg=min(2, len(dates) - 1))
        target_numeric = target_month.year + target_month.month / 12.0
        return np.polyval(coeffs, target_numeric)
    except:
        return np.nan


def ssa_decomposition(ts, window_length=None, n_components=None):
    """
    Singular Spectrum Analysis (SSA) decomposition

    Parameters:
    - ts: time series array
    - window_length: embedding window length (default: N/4 where N is series length)
    - n_components: number of components to reconstruct (default: all)

    Returns:
    - components: dictionary with trend, seasonal, and residual components
    - reconstructed: reconstructed series
    """
    ts = np.array(ts)
    N = len(ts)

    if window_length is None:
        window_length = N // 4

    # Step 1: Embedding
    K = N - window_length + 1
    X = np.zeros((window_length, K))

    for i in range(K):
        X[:, i] = ts[i:i + window_length]

    # Step 2: SVD
    U, s, Vt = svd(X, full_matrices=False)
    V = Vt.T

    # Step 3: Grouping (reconstruct components)
    if n_components is None:
        n_components = len(s)

    # Reconstruct individual components
    components_list = []
    for i in range(n_components):
        X_i = s[i] * np.outer(U[:, i], V[:, i])
        # Diagonal averaging (Hankelization)
        ts_i = np.zeros(N)

        for k in range(N):
            if k < window_length:
                indices = [(j, k - j) for j in range(k + 1) if j < window_length and k - j < K]
            elif k < K:
                indices = [(j, k - j) for j in range(window_length) if k - j < K]
            else:
                indices = [(j, k - j) for j in range(k - K + 1, window_length) if k - j < K]

            ts_i[k] = np.mean([X_i[j, l] for j, l in indices])

        components_list.append(ts_i)

    # Group components into trend, seasonal, and residual
    # This is a heuristic grouping - you might need to adjust based on your data
    trend = components_list[0]  # First component is usually trend

    # Components 2-6 are typically seasonal (adjust as needed)
    seasonal = np.zeros(N)
    seasonal_end = min(6, len(components_list))
    for i in range(1, seasonal_end):
        seasonal += components_list[i]

    # Remaining components are residual
    residual = np.zeros(N)
    for i in range(seasonal_end, len(components_list)):
        residual += components_list[i]

    reconstructed = trend + seasonal + residual

    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
        'components': components_list,
        'singular_values': s
    }, reconstructed


def apply_ssa_detrending(df, dataset_names, window_length=None):
    """Apply SSA detrending to remove seasonality from all datasets"""
    df_detrended = df.copy()
    ssa_results = {}

    print("\nApplying SSA detrending to remove seasonality...")

    for dataset_name in dataset_names:
        if dataset_name in df.columns:
            print(f"Detrending {dataset_name}...")

            # Get valid data
            valid_data = df[dataset_name].dropna()
            if len(valid_data) < 24:  # Need at least 2 years of data
                print(f"Warning: {dataset_name} has insufficient data for SSA")
                continue

            # Apply SSA
            ssa_result, reconstructed = ssa_decomposition(
                valid_data.values,
                window_length=window_length
            )

            # Store results
            ssa_results[dataset_name] = ssa_result

            # Create detrended series (trend + residual, removing seasonal)
            detrended = ssa_result['trend'] + ssa_result['residual']

            # Map back to original dataframe indices
            valid_indices = df[dataset_name].notna()
            df_detrended.loc[valid_indices, f'{dataset_name}_detrended'] = detrended

            # Also store components for analysis
            df_detrended.loc[valid_indices, f'{dataset_name}_trend'] = ssa_result['trend']
            df_detrended.loc[valid_indices, f'{dataset_name}_seasonal'] = ssa_result['seasonal']
            df_detrended.loc[valid_indices, f'{dataset_name}_residual'] = ssa_result['residual']

    return df_detrended, ssa_results


def fill_missing_data_3step(df, dataset_name):
    """3-step gap filling method"""
    df_filled = df.copy()

    for idx in df_filled.index:
        if pd.isna(df_filled.loc[idx, dataset_name]):
            date = df_filled.loc[idx, 'date']
            month = date.month
            year = date.year

            # Step 1: Average of same month in adjacent years
            prev_year_date = datetime(year - 1, month, 1)
            next_year_date = datetime(year + 1, month, 1)

            prev_val = df_filled[df_filled['date'] == prev_year_date][dataset_name]
            next_val = df_filled[df_filled['date'] == next_year_date][dataset_name]

            prev_val = prev_val.iloc[0] if len(prev_val) > 0 and not pd.isna(prev_val.iloc[0]) else np.nan
            next_val = next_val.iloc[0] if len(next_val) > 0 and not pd.isna(next_val.iloc[0]) else np.nan

            if not pd.isna(prev_val) and not pd.isna(next_val):
                df_filled.loc[idx, dataset_name] = (prev_val + next_val) / 2
                continue

            # Step 2: GAM trend estimation for same month
            same_month_data = df_filled[df_filled['date'].dt.month == month]
            valid_same_month = same_month_data.dropna(subset=[dataset_name])

            if len(valid_same_month) >= 2:
                trend_val = simple_gam_trend(valid_same_month['date'].tolist(),
                                             valid_same_month[dataset_name].tolist(),
                                             date)
                if not pd.isna(trend_val):
                    df_filled.loc[idx, dataset_name] = trend_val
                    continue

            # Step 3: Overall monthly average
            same_month_mean = df_filled[df_filled['date'].dt.month == month][dataset_name].mean()
            if not pd.isna(same_month_mean):
                df_filled.loc[idx, dataset_name] = same_month_mean

    return df_filled


def process_dataset(dataset_name, data_path):
    """Process individual dataset"""
    print(f"Processing {dataset_name}...")

    # Get all TIF files
    tif_files = glob.glob(os.path.join(data_path, "*.tif*"))

    if not tif_files:
        print(f"No TIF files found in {data_path}")
        return pd.DataFrame()

    data_records = []

    for tif_file in tif_files:
        date = extract_date_from_filename(tif_file, dataset_name)
        if date is None:
            print(f"Could not parse date from {tif_file}")
            continue

        # Skip if outside 2001-2021 range
        if date.year < 2001 or date.year > 2021:
            continue

        mean_value = read_tif_and_calculate_mean(tif_file)
        data_records.append({
            'date': date,
            'value': mean_value,
            'filename': os.path.basename(tif_file)
        })

    if not data_records:
        return pd.DataFrame()

    df = pd.DataFrame(data_records)
    df = df.sort_values('date').reset_index(drop=True)

    print(f"{dataset_name}: Found {len(df)} files, {df['value'].isna().sum()} with NaN values")
    return df


def create_complete_timeline():
    """Create complete monthly timeline from 2001 to 2021"""
    dates = []
    for year in range(2001, 2022):
        for month in range(1, 13):
            dates.append(datetime(year, month, 1))
    return dates


def main():
    # Process all datasets
    all_data = {}

    for dataset_name, path in data_paths.items():
        df = process_dataset(dataset_name, path)
        if not df.empty:
            all_data[dataset_name] = df

    if not all_data:
        print("No data found for any dataset!")
        return

    # Create complete timeline
    complete_dates = create_complete_timeline()

    # Merge all datasets
    merged_df = pd.DataFrame({'date': complete_dates})

    for dataset_name, df in all_data.items():
        # Rename value column to dataset name
        df_temp = df[['date', 'value']].copy()
        df_temp = df_temp.rename(columns={'value': dataset_name})
        merged_df = merged_df.merge(df_temp, on='date', how='left')

    print(f"\nMerged dataframe shape: {merged_df.shape}")
    print("Missing values per dataset:")
    for col in merged_df.columns:
        if col != 'date':
            missing = merged_df[col].isna().sum()
            print(f"{col}: {missing} missing values")

    # Apply 3-step gap filling for each dataset
    print("\nApplying 3-step gap filling method...")
    for dataset_name in all_data.keys():
        merged_df = fill_missing_data_3step(merged_df, dataset_name)

    print("\nAfter gap filling - Missing values per dataset:")
    for col in merged_df.columns:
        if col != 'date':
            missing = merged_df[col].isna().sum()
            print(f"{col}: {missing} missing values")

    # Apply SSA detrending
    print("\nApplying SSA detrending...")
    merged_df_detrended, ssa_results = apply_ssa_detrending(
        merged_df,
        list(all_data.keys()),
        window_length=24  # 2 years window for monthly data
    )

    # Calculate Z-scores for both original and detrended data
    print("\nCalculating Z-scores...")
    z_score_df = merged_df_detrended.copy()

    for dataset_name in all_data.keys():
        if dataset_name in z_score_df.columns:
            # Original data z-scores
            values = z_score_df[dataset_name].dropna()
            if len(values) > 1:
                z_scores = stats.zscore(values)
                valid_indices = z_score_df[dataset_name].notna()
                z_score_df.loc[valid_indices, f'{dataset_name}_zscore'] = z_scores

            # Detrended data z-scores
            if f'{dataset_name}_detrended' in z_score_df.columns:
                detrended_values = z_score_df[f'{dataset_name}_detrended'].dropna()
                if len(detrended_values) > 1:
                    z_scores_detrended = stats.zscore(detrended_values)
                    valid_indices_detrended = z_score_df[f'{dataset_name}_detrended'].notna()
                    z_score_df.loc[valid_indices_detrended, f'{dataset_name}_detrended_zscore'] = z_scores_detrended

    # Create visualization with both original and detrended data
    print("\nCreating visualization...")
    fig, axes = plt.subplots(len(all_data) * 2, 1, figsize=(15, 4 * len(all_data)), sharex=True)
    if len(all_data) == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, dataset_name in enumerate(all_data.keys()):
        # Original data plot
        if f'{dataset_name}_zscore' in z_score_df.columns:
            valid_data = z_score_df.dropna(subset=[f'{dataset_name}_zscore'])
            axes[i * 2].plot(valid_data['date'], valid_data[f'{dataset_name}_zscore'],
                             color=colors[i % len(colors)], linewidth=1, alpha=0.8, label='Original')
            axes[i * 2].fill_between(valid_data['date'], valid_data[f'{dataset_name}_zscore'],
                                     alpha=0.3, color=colors[i % len(colors)])

            axes[i * 2].set_ylabel(f'{dataset_name}\nOriginal (Z-score)', fontsize=10)
            axes[i * 2].grid(True, alpha=0.3)
            axes[i * 2].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)

            # Add statistics
            mean_val = merged_df[dataset_name].mean()
            std_val = merged_df[dataset_name].std()
            axes[i * 2].text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                             transform=axes[i * 2].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                             fontsize=8)

        # Detrended data plot
        if f'{dataset_name}_detrended_zscore' in z_score_df.columns:
            valid_data_detrended = z_score_df.dropna(subset=[f'{dataset_name}_detrended_zscore'])
            axes[i * 2 + 1].plot(valid_data_detrended['date'], valid_data_detrended[f'{dataset_name}_detrended_zscore'],
                                 color=colors[i % len(colors)], linewidth=1, alpha=0.8, label='Detrended')
            axes[i * 2 + 1].fill_between(valid_data_detrended['date'],
                                         valid_data_detrended[f'{dataset_name}_detrended_zscore'],
                                         alpha=0.3, color=colors[i % len(colors)])

            axes[i * 2 + 1].set_ylabel(f'{dataset_name}\nDetrended (Z-score)', fontsize=10)
            axes[i * 2 + 1].grid(True, alpha=0.3)
            axes[i * 2 + 1].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)

            # Add statistics for detrended data
            if f'{dataset_name}_detrended' in merged_df_detrended.columns:
                mean_val_detrended = merged_df_detrended[f'{dataset_name}_detrended'].mean()
                std_val_detrended = merged_df_detrended[f'{dataset_name}_detrended'].std()
                axes[i * 2 + 1].text(0.02, 0.95, f'Mean: {mean_val_detrended:.3f}\nStd: {std_val_detrended:.3f}',
                                     transform=axes[i * 2 + 1].transAxes, verticalalignment='top',
                                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                     fontsize=8)

    axes[-1].set_xlabel('Time', fontsize=12)
    plt.suptitle(
        'Temporal Trends of Climate-Eco Indicators (2001-2021)\nCalifornia Drought Indices - Original vs SSA Detrended',
        fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Save the plot
    plt.savefig('california_drought_indices_ssa_detrended.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create SSA decomposition plots for each dataset
    print("\nCreating SSA decomposition plots...")
    for dataset_name, ssa_result in ssa_results.items():
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)

        # Get corresponding dates
        valid_indices = merged_df[dataset_name].notna()
        dates = merged_df.loc[valid_indices, 'date']

        # Original series
        axes[0].plot(dates, merged_df.loc[valid_indices, dataset_name], 'b-', linewidth=1)
        axes[0].set_ylabel('Original', fontsize=10)
        axes[0].set_title(f'SSA Decomposition of {dataset_name}', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Trend
        axes[1].plot(dates, ssa_result['trend'], 'r-', linewidth=1)
        axes[1].set_ylabel('Trend', fontsize=10)
        axes[1].grid(True, alpha=0.3)

        # Seasonal
        axes[2].plot(dates, ssa_result['seasonal'], 'g-', linewidth=1)
        axes[2].set_ylabel('Seasonal', fontsize=10)
        axes[2].grid(True, alpha=0.3)

        # Residual
        axes[3].plot(dates, ssa_result['residual'], 'orange', linewidth=1)
        axes[3].set_ylabel('Residual', fontsize=10)
        axes[3].set_xlabel('Time', fontsize=12)
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'ssa_decomposition_{dataset_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()

    # Plot singular values for each dataset
    print("\nCreating singular values plots...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (dataset_name, ssa_result) in enumerate(ssa_results.items()):
        if i < len(axes):
            axes[i].plot(range(1, len(ssa_result['singular_values']) + 1),
                         ssa_result['singular_values'], 'bo-', markersize=4)
            axes[i].set_xlabel('Component')
            axes[i].set_ylabel('Singular Value')
            axes[i].set_title(f'{dataset_name} - Singular Values')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_yscale('log')

    # Hide unused subplots
    for i in range(len(ssa_results), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('ssa_singular_values.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save processed data to Excel
    print("\nSaving processed data to Excel...")

    # Prepare data for Excel export
    excel_data = merged_df_detrended.copy()

    # Add z-score columns for both original and detrended data
    for dataset_name in all_data.keys():
        if f'{dataset_name}_zscore' in z_score_df.columns:
            excel_data[f'{dataset_name}_zscore'] = z_score_df[f'{dataset_name}_zscore']
        if f'{dataset_name}_detrended_zscore' in z_score_df.columns:
            excel_data[f'{dataset_name}_detrended_zscore'] = z_score_df[f'{dataset_name}_detrended_zscore']

    excel_data.to_excel('california_drought_indices_ssa_processed.xlsx', index=False)
    print("Data saved to 'california_drought_indices_ssa_processed.xlsx'")

    print(f"\nFinal dataset summary:")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Total months: {len(merged_df)}")
    print("\nDataset completeness:")
    for dataset_name in all_data.keys():
        if dataset_name in merged_df.columns:
            completeness = (1 - merged_df[dataset_name].isna().sum() / len(merged_df)) * 100
            print(f"{dataset_name}: {completeness:.1f}% complete")

    print("\nSSA Detrending Summary:")
    for dataset_name, ssa_result in ssa_results.items():
        print(f"{dataset_name}:")
        print(f"  - Number of components: {len(ssa_result['components'])}")
        print(
            f"  - Trend variance explained: {(ssa_result['singular_values'][0] ** 2 / np.sum(ssa_result['singular_values'] ** 2) * 100):.1f}%")
        print(
            f"  - Top 3 components variance: {(np.sum(ssa_result['singular_values'][:3] ** 2) / np.sum(ssa_result['singular_values'] ** 2) * 100):.1f}%")


if __name__ == "__main__":
    main()