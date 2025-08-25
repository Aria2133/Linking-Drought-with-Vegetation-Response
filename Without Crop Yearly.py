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
    'GOSIF': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GOSIF_on_SPEI_Grid",
    'LST': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\Tonzi_LST",
    'NDVI': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\Tonzi_NDVI",
    'NIRv': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\Tonzi_NIRv",
    'SPEI': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\SPEI_tif_Cleaned",
    'TWS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GRACE_on_SPEI_Grid"
}


def extract_date_from_filename(filename, dataset_name):
    """Extract date from filename based on dataset naming convention"""
    basename = os.path.basename(filename)

    try:
        if dataset_name == 'GOSIF':
            # GOSIF_200101_Enhanced_Tonzi.tif
            parts = basename.split('_')
            if len(parts) >= 2:
                date_str = parts[1]  # 200101
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
            # MYD21C1_Tonzi_LST_2002_07.tif
            parts = basename.split('_')
            if len(parts) >= 5:
                year = int(parts[-2])
                month_str = parts[-1].split('.')[0]  # Remove file extension
                month = int(month_str)
                print(f"LST parsing: {basename} -> year={year}, month={month}")
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'NDVI':
            # NDVI_MOD13Q1_Tonzi_2001_01.tif
            parts = basename.split('_')
            if len(parts) >= 5:
                year = int(parts[-2])
                month = int(parts[-1].split('.')[0])  # Remove file extension
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'NIRv':
            # NIRv_MOD13Q1_Tonzi_2001_02.tif
            parts = basename.split('_')
            if len(parts) >= 5:
                year = int(parts[-2])
                month = int(parts[-1].split('.')[0])  # Remove file extension
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'SPEI':
            # SPEI_2001_01_cleaned.tif
            parts = basename.split('_')
            if len(parts) >= 3:
                year = int(parts[1])
                month = int(parts[2])
            else:
                print(f"Insufficient parts in {basename}: {parts}")
                return None

        elif dataset_name == 'TWS':
            # GRACE_TWS_Tonzi_2002_04_Tonzi_Regridded_Tonzi.tif
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

            # Remove zeros and NaNs for this specific file
            valid_data = data[(data != 0) & (~np.isnan(data))]

            if len(valid_data) > 0:
                return np.mean(valid_data)
            else:
                return np.nan
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return np.nan


def simple_gam_trend(dates, values, target_date):
    """Simple GAM-like trend estimation using polynomial regression"""
    if len(dates) < 3:
        return np.nan

    # Convert dates to numerical values (months since start)
    start_date = min(dates)
    numeric_dates = np.array([(d.year - start_date.year) * 12 + (d.month - start_date.month)
                              for d in dates])
    target_numeric = (target_date.year - start_date.year) * 12 + (target_date.month - start_date.month)

    # Use polynomial regression (degree 2) as simple GAM substitute
    try:
        coeffs = np.polyfit(numeric_dates, values, deg=min(2, len(dates) - 1))
        return np.polyval(coeffs, target_numeric)
    except:
        return np.nan


def fill_missing_data_3step_monthly(df, dataset_name):
    """3-step gap filling method for monthly data"""
    df_filled = df.copy()
    df_filled = df_filled.sort_values(['year', 'month']).reset_index(drop=True)

    for idx in df_filled.index:
        if pd.isna(df_filled.loc[idx, dataset_name]):
            year = df_filled.loc[idx, 'year']
            month = df_filled.loc[idx, 'month']
            current_date = datetime(year, month, 1)

            # Step 1: Average of adjacent months
            prev_month_idx = idx - 1 if idx > 0 else None
            next_month_idx = idx + 1 if idx < len(df_filled) - 1 else None

            prev_val = np.nan
            next_val = np.nan

            if prev_month_idx is not None and not pd.isna(df_filled.loc[prev_month_idx, dataset_name]):
                prev_val = df_filled.loc[prev_month_idx, dataset_name]

            if next_month_idx is not None and not pd.isna(df_filled.loc[next_month_idx, dataset_name]):
                next_val = df_filled.loc[next_month_idx, dataset_name]

            if not pd.isna(prev_val) and not pd.isna(next_val):
                df_filled.loc[idx, dataset_name] = (prev_val + next_val) / 2
                continue

            # Step 2: GAM trend estimation
            valid_data = df_filled.dropna(subset=[dataset_name])

            if len(valid_data) >= 2:
                valid_dates = [datetime(row['year'], row['month'], 1) for _, row in valid_data.iterrows()]
                trend_val = simple_gam_trend(valid_dates,
                                             valid_data[dataset_name].tolist(),
                                             current_date)
                if not pd.isna(trend_val):
                    df_filled.loc[idx, dataset_name] = trend_val
                    continue

            # Step 3: Seasonal mean (same month across years)
            same_month_data = df_filled[df_filled['month'] == month][dataset_name]
            seasonal_mean = same_month_data.mean()

            if not pd.isna(seasonal_mean):
                df_filled.loc[idx, dataset_name] = seasonal_mean
                continue

            # Step 4: Overall mean as last resort
            overall_mean = df_filled[dataset_name].mean()
            if not pd.isna(overall_mean):
                df_filled.loc[idx, dataset_name] = overall_mean

    return df_filled


def process_dataset_monthly(dataset_name, data_path):
    """Process individual dataset for monthly data"""
    print(f"Processing {dataset_name}...")

    # Get all TIF files
    tif_files = glob.glob(os.path.join(data_path, "*.tif*"))

    if not tif_files:
        print(f"No TIF files found in {data_path}")
        return pd.DataFrame()

    print(f"Found {len(tif_files)} TIF files in {data_path}")

    data_records = []
    skipped_files = []

    for tif_file in tif_files:
        date = extract_date_from_filename(tif_file, dataset_name)
        if date is None:
            skipped_files.append(os.path.basename(tif_file))
            continue

        # Skip if outside 2001-2021 range
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
        print(f"Skipped {len(skipped_files)} files:")
        for skipped in skipped_files[:10]:  # Show first 10 skipped files
            print(f"  - {skipped}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")

    if not data_records:
        print(f"No valid data found for {dataset_name}")
        return pd.DataFrame()

    df = pd.DataFrame(data_records)
    df = df.sort_values(['year', 'month']).reset_index(drop=True)

    print(f"{dataset_name}: Found {len(df)} monthly records")
    return df[['year', 'month', 'date', 'value']]


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


def main():
    # Process all datasets with monthly data
    all_data = {}

    for dataset_name, path in data_paths.items():
        df = process_dataset_monthly(dataset_name, path)
        if not df.empty:
            all_data[dataset_name] = df

    if not all_data:
        print("No data found for any dataset!")
        return

    # Create complete monthly timeline
    complete_timeline = create_complete_monthly_timeline()

    # Merge all datasets
    merged_df = complete_timeline.copy()

    for dataset_name, df in all_data.items():
        # Rename value column to dataset name
        df_temp = df[['year', 'month', 'value']].copy()
        df_temp = df_temp.rename(columns={'value': dataset_name})
        merged_df = merged_df.merge(df_temp, on=['year', 'month'], how='left')

    print(f"\nMerged dataframe shape: {merged_df.shape}")
    print(
        f"Time period: {merged_df['year'].min()}-{merged_df['month'].min()} to {merged_df['year'].max()}-{merged_df['month'].max()}")
    print("Missing values per dataset:")
    for col in merged_df.columns:
        if col not in ['year', 'month', 'date']:
            missing = merged_df[col].isna().sum()
            total = len(merged_df)
            print(f"{col}: {missing}/{total} missing ({missing / total * 100:.1f}%)")

    # Apply 3-step gap filling for each dataset
    print("\nApplying 3-step gap filling method...")
    for dataset_name in all_data.keys():
        merged_df = fill_missing_data_3step_monthly(merged_df, dataset_name)

    print("\nAfter gap filling - Missing values per dataset:")
    for col in merged_df.columns:
        if col not in ['year', 'month', 'date']:
            missing = merged_df[col].isna().sum()
            total = len(merged_df)
            print(f"{col}: {missing}/{total} missing ({missing / total * 100:.1f}%)")

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
    fig, axes = plt.subplots(len(all_data), 1, figsize=(20, 4 * len(all_data)), sharex=True)
    if len(all_data) == 1:
        axes = [axes]

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    for i, dataset_name in enumerate(all_data.keys()):
        if f'{dataset_name}_zscore' in z_score_df.columns:
            # Create date strings for x-axis
            z_score_df['date_str'] = z_score_df['year'].astype(str) + '-' + z_score_df['month'].astype(str).str.zfill(2)

            # Plot z-score values
            valid_data = z_score_df.dropna(subset=[f'{dataset_name}_zscore'])

            # Convert to datetime for better plotting
            valid_data_plot = valid_data.copy()
            valid_data_plot['plot_date'] = pd.to_datetime(valid_data_plot[['year', 'month']].assign(day=1))

            axes[i].plot(valid_data_plot['plot_date'], valid_data_plot[f'{dataset_name}_zscore'],
                         color=colors[i % len(colors)], linewidth=1.5, alpha=0.8, marker='.', markersize=3)
            axes[i].fill_between(valid_data_plot['plot_date'], valid_data_plot[f'{dataset_name}_zscore'],
                                 alpha=0.2, color=colors[i % len(colors)])

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

            # Format x-axis to show years
            import matplotlib.dates as mdates
            axes[i].xaxis.set_major_locator(mdates.YearLocator())
            axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axes[i].xaxis.set_minor_locator(mdates.MonthLocator())

    axes[-1].set_xlabel('Year', fontsize=12)
    plt.suptitle('Monthly Trends of Climate-Eco Indicators (2001-2021)\nTonzi Ranch - Z-score Normalized',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    # Rotate x-axis labels for better readability
    for ax in axes:
        ax.tick_params(axis='x', rotation=45)

    # Save the plot
    plt.savefig('tonzi_climate_indices_monthly_zscore.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save processed data to Excel
    print("\nSaving processed data to Excel...")

    # Prepare data for Excel export
    excel_data = merged_df.copy()

    # Add z-score columns
    for dataset_name in all_data.keys():
        if f'{dataset_name}_zscore' in z_score_df.columns:
            excel_data[f'{dataset_name}_zscore'] = z_score_df[f'{dataset_name}_zscore']

    # Add date string for easier reading
    excel_data['date_string'] = excel_data['year'].astype(str) + '-' + excel_data['month'].astype(str).str.zfill(2)

    # Reorder columns for better organization
    date_cols = ['year', 'month', 'date_string']
    value_cols = [col for col in excel_data.columns if
                  col not in date_cols and not col.endswith('_zscore') and col != 'date']
    zscore_cols = [col for col in excel_data.columns if col.endswith('_zscore')]

    excel_data = excel_data[date_cols + value_cols + zscore_cols]
    excel_data.to_excel('tonzi_climate_indices_monthly_processed.xlsx', index=False)
    print("Data saved to 'tonzi_climate_indices_monthly_processed.xlsx'")

    # Display summary statistics
    print(f"\nFinal dataset summary:")
    print(
        f"Time period: {merged_df['year'].min()}-{merged_df['month'].min():02d} to {merged_df['year'].max()}-{merged_df['month'].max():02d}")
    print(f"Total months: {len(merged_df)}")
    print("\nDataset completeness:")
    for dataset_name in all_data.keys():
        if dataset_name in merged_df.columns:
            completeness = (1 - merged_df[dataset_name].isna().sum() / len(merged_df)) * 100
            print(f"{dataset_name}: {completeness:.1f}% complete")

    # Display correlation matrix
    print("\nCorrelation Matrix (Monthly Values):")
    numeric_cols = [col for col in merged_df.columns if col not in ['year', 'month', 'date']]
    correlation_matrix = merged_df[numeric_cols].corr()
    print(correlation_matrix.round(3))

    return merged_df, z_score_df


if __name__ == "__main__":
    main()