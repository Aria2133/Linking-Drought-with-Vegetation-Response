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

    # Calculate Z-scores for each dataset independently
    print("\nCalculating Z-scores...")
    z_score_df = merged_df.copy()

    for dataset_name in all_data.keys():
        if dataset_name in z_score_df.columns:
            values = z_score_df[dataset_name].dropna()
            if len(values) > 1:
                z_scores = stats.zscore(values)
                # Map z-scores back to original positions
                valid_indices = z_score_df[dataset_name].notna()
                z_score_df.loc[valid_indices, f'{dataset_name}_zscore'] = z_scores

    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(len(all_data), 1, figsize=(15, 3 * len(all_data)), sharex=True)
    if len(all_data) == 1:
        axes = [axes]

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, dataset_name in enumerate(all_data.keys()):
        if f'{dataset_name}_zscore' in z_score_df.columns:
            # Plot z-score values
            valid_data = z_score_df.dropna(subset=[f'{dataset_name}_zscore'])
            axes[i].plot(valid_data['date'], valid_data[f'{dataset_name}_zscore'],
                         color=colors[i % len(colors)], linewidth=1, alpha=0.8)
            axes[i].fill_between(valid_data['date'], valid_data[f'{dataset_name}_zscore'],
                                 alpha=0.3, color=colors[i % len(colors)])

            axes[i].set_ylabel(f'{dataset_name}\n(Z-score)', fontsize=10)
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.5)

            # Add statistics to the plot
            mean_val = merged_df[dataset_name].mean()
            std_val = merged_df[dataset_name].std()
            axes[i].text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}',
                         transform=axes[i].transAxes, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                         fontsize=8)

    axes[-1].set_xlabel('Time', fontsize=12)
    plt.suptitle(
        'Temporal Trends of Climate-Eco Indicators (2001-2021)\nCalifornia Drought Indices - Z-score Normalized',
        fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Save the plot
    plt.savefig('california_drought_indices_zscore.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save processed data to Excel
    print("\nSaving processed data to Excel...")

    # Prepare data for Excel export
    excel_data = merged_df.copy()

    # Add z-score columns
    for dataset_name in all_data.keys():
        if f'{dataset_name}_zscore' in z_score_df.columns:
            excel_data[f'{dataset_name}_zscore'] = z_score_df[f'{dataset_name}_zscore']

    excel_data.to_excel('california_drought_indices_processed.xlsx', index=False)
    print("Data saved to 'california_drought_indices_processed.xlsx'")

    print(f"\nFinal dataset summary:")
    print(f"Date range: {merged_df['date'].min()} to {merged_df['date'].max()}")
    print(f"Total months: {len(merged_df)}")
    print("\nDataset completeness:")
    for dataset_name in all_data.keys():
        if dataset_name in merged_df.columns:
            completeness = (1 - merged_df[dataset_name].isna().sum() / len(merged_df)) * 100
            print(f"{dataset_name}: {completeness:.1f}% complete")


if __name__ == "__main__":
    main()