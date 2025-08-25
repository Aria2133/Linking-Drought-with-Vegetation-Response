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
from statsmodels.tsa.seasonal import seasonal_decompose
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


def simple_gam_trend(dates, values, target_date):
    """Simple GAM-like trend estimation using polynomial regression"""
    if len(dates) < 3:
        return np.nan

    # Convert dates to numerical values (years since first date)
    if isinstance(dates[0], datetime):
        numeric_dates = np.array([(d.year + d.month / 12.0) for d in dates])
        target_numeric = target_date.year + target_date.month / 12.0
    else:
        numeric_dates = np.array(dates)
        target_numeric = target_date

    # Use polynomial regression (degree 2) as simple GAM substitute
    try:
        coeffs = np.polyfit(numeric_dates, values, deg=min(2, len(dates) - 1))
        return np.polyval(coeffs, target_numeric)
    except:
        return np.nan


def fill_missing_data_3step(df, dataset_name):
    """3-step gap filling method for monthly data"""
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


def remove_seasonality(df, dataset_name):
    """Remove seasonality from monthly time series data - FIXED VERSION"""
    print(f"    Attempting seasonality removal for {dataset_name}...")

    # Check if dataset column exists
    if dataset_name not in df.columns:
        print(f"    ERROR: Column {dataset_name} not found in dataframe")
        df[f'{dataset_name}_deseason'] = np.nan
        return df

    # Create complete monthly time series using original dataframe dates
    df_temp = df[['date', dataset_name]].copy()
    df_temp = df_temp.dropna(subset=[dataset_name])
    df_temp = df_temp.set_index('date')

    # Resample to ensure monthly frequency and forward fill small gaps
    monthly_data = df_temp[dataset_name].resample('M').mean()

    print(f"    Monthly data points: {len(monthly_data)}, Valid: {monthly_data.count()}")

    # Need at least 24 months for seasonal decomposition
    if monthly_data.count() < 24:
        print(f"    Insufficient data for seasonal decomposition of {dataset_name} ({monthly_data.count()} < 24)")
        # If we can't deseasonalize, use original data as deseasonalized
        df[f'{dataset_name}_deseason'] = df[dataset_name]
        return df

    try:
        # Fill any remaining NaN values with interpolation for decomposition
        monthly_data_filled = monthly_data.interpolate(method='linear', limit=3)

        if monthly_data_filled.count() < 24:
            print(f"    Still insufficient data after interpolation")
            df[f'{dataset_name}_deseason'] = df[dataset_name]
            return df

        print(f"    Performing seasonal decomposition...")
        decomposition = seasonal_decompose(monthly_data_filled,
                                           model='additive',
                                           period=12,  # 12 months seasonality
                                           extrapolate_trend='freq')

        # Extract trend and residual (removing seasonal component)
        deseasonalized = decomposition.trend + decomposition.resid

        # Initialize deseason column
        df[f'{dataset_name}_deseason'] = np.nan

        # Map deseasonalized values back to original dataframe
        mapped_count = 0
        for idx, row in df.iterrows():
            # Convert to end of month for matching resampled data
            month_end = pd.Timestamp(row['date'].year, row['date'].month, 1) + pd.offsets.MonthEnd(0)

            if month_end in deseasonalized.index and not pd.isna(deseasonalized[month_end]):
                df.loc[idx, f'{dataset_name}_deseason'] = deseasonalized[month_end]
                mapped_count += 1

        print(f"    Successfully mapped {mapped_count} deseasonalized values")

        # If mapping failed, try direct date matching
        if mapped_count == 0:
            print(f"    Trying direct date matching...")
            for idx, row in df.iterrows():
                # Try multiple date formats
                possible_dates = [
                    row['date'],
                    pd.Timestamp(row['date'].year, row['date'].month, 1),
                    pd.Timestamp(row['date'].year, row['date'].month, 1) + pd.offsets.MonthEnd(0)
                ]

                for test_date in possible_dates:
                    if test_date in deseasonalized.index and not pd.isna(deseasonalized[test_date]):
                        df.loc[idx, f'{dataset_name}_deseason'] = deseasonalized[test_date]
                        mapped_count += 1
                        break

            print(f"    Direct matching mapped {mapped_count} values")

        # If still no mapping, use interpolation approach
        if mapped_count == 0:
            print(f"    Using interpolation approach for mapping...")
            deseason_values = []
            for idx, row in df.iterrows():
                target_date = pd.Timestamp(row['date'])
                # Find closest dates in deseasonalized data
                time_diffs = abs(deseasonalized.index - target_date)
                closest_idx = time_diffs.argmin()

                if time_diffs.iloc[closest_idx].days <= 31:  # Within a month
                    deseason_values.append(deseasonalized.iloc[closest_idx])
                    mapped_count += 1
                else:
                    deseason_values.append(np.nan)

            df[f'{dataset_name}_deseason'] = deseason_values
            print(f"    Interpolation approach mapped {mapped_count} values")

        return df

    except Exception as e:
        print(f"    Error in seasonal decomposition for {dataset_name}: {e}")
        # If decomposition fails, use original data
        df[f'{dataset_name}_deseason'] = df[dataset_name]
        return df


def process_dataset(dataset_name, data_path):
    """Process individual dataset - monthly processing first"""
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


def aggregate_to_yearly(df, dataset_name):
    """Aggregate monthly data to yearly, handling both original and deseasonalized data"""
    yearly_data = []

    original_col = dataset_name
    deseason_col = f'{dataset_name}_deseason'

    print(f"    Aggregating {dataset_name}...")

    for year in range(2001, 2022):
        year_data = df[df['date'].dt.year == year]

        if len(year_data) >= 6:  # At least 6 months of data
            # Original data
            original_mean = year_data[original_col].mean() if original_col in df.columns else np.nan

            # Deseasonalized data (if available)
            deseason_mean = year_data[deseason_col].mean() if deseason_col in df.columns else np.nan

            yearly_data.append({
                'year': year,
                f'{dataset_name}_original': original_mean,
                f'{dataset_name}_deseason': deseason_mean,
                f'{dataset_name}_months_available': len(year_data.dropna(subset=[original_col]))
            })

            if not np.isnan(deseason_mean):
                print(f"      Year {year}: original={original_mean:.3f}, deseason={deseason_mean:.3f}")

    result_df = pd.DataFrame(yearly_data)
    print(f"    Created yearly dataframe with {len(result_df)} years")
    return result_df


def calculate_trend_stats(years, values):
    """Calculate trend statistics and p-value"""
    # Remove NaN values
    data_df = pd.DataFrame({'year': years, 'value': values}).dropna()
    if len(data_df) < 3:
        return np.nan, np.nan, np.nan

    years_clean = data_df['year'].values
    values_clean = data_df['value'].values

    # Linear regression
    X = years_clean.reshape(-1, 1)
    y = values_clean

    reg = LinearRegression()
    reg.fit(X, y)

    # Calculate R-squared
    y_pred = reg.predict(X)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Calculate p-value using correlation test
    correlation, p_value = stats.pearsonr(years_clean, values_clean)

    return reg.coef_[0], r_squared, p_value


def main():
    # Process all datasets
    all_monthly_data = {}

    for dataset_name, path in data_paths.items():
        df = process_dataset(dataset_name, path)
        if not df.empty:
            all_monthly_data[dataset_name] = df

    if not all_monthly_data:
        print("No data found for any dataset!")
        return

    # Create complete monthly timeline and merge datasets
    complete_dates = create_complete_timeline()
    merged_monthly_df = pd.DataFrame({'date': complete_dates})

    for dataset_name, df in all_monthly_data.items():
        df_temp = df[['date', 'value']].copy()
        df_temp = df_temp.rename(columns={'value': dataset_name})
        merged_monthly_df = merged_monthly_df.merge(df_temp, on='date', how='left')

    print(f"\nMonthly merged dataframe shape: {merged_monthly_df.shape}")

    # Apply gap filling to monthly data
    print("Applying gap filling to monthly data...")
    for dataset_name in all_monthly_data.keys():
        merged_monthly_df = fill_missing_data_3step(merged_monthly_df, dataset_name)

    # Remove seasonality from each dataset
    print("Removing seasonality from datasets...")
    for dataset_name in all_monthly_data.keys():
        print(f"  Deseasonalizing {dataset_name}...")
        merged_monthly_df = remove_seasonality(merged_monthly_df, dataset_name)
        # Check if deseasonalization worked
        deseason_col = f'{dataset_name}_deseason'
        if deseason_col in merged_monthly_df.columns:
            non_nan_count = merged_monthly_df[deseason_col].count()
            print(f"    {dataset_name} deseasonalized: {non_nan_count} valid values")
        else:
            print(f"    WARNING: {dataset_name} deseasonalization failed!")

    # Aggregate to yearly data (both original and deseasonalized)
    print("Aggregating to yearly data...")

    # Create base yearly dataframe
    years = list(range(2001, 2022))
    merged_yearly_df = pd.DataFrame({'year': years})

    # Process each dataset and merge
    for dataset_name in all_monthly_data.keys():
        print(f"  Aggregating {dataset_name} to yearly...")
        yearly_df = aggregate_to_yearly(merged_monthly_df, dataset_name)
        if not yearly_df.empty:
            merged_yearly_df = merged_yearly_df.merge(yearly_df, on='year', how='left')
            print(f"    {dataset_name}: {len(yearly_df)} years with data")
        else:
            print(f"    WARNING: No yearly data for {dataset_name}")

    print(f"\nYearly merged dataframe shape: {merged_yearly_df.shape}")

    # Check data availability
    for col in merged_yearly_df.columns:
        if '_deseason' in col:
            valid_count = merged_yearly_df[col].count()
            print(f"{col}: {valid_count} valid years")

    # Calculate Z-scores for deseasonalized data
    print("\nCalculating Z-scores...")
    z_score_df = merged_yearly_df.copy()

    for dataset_name in all_monthly_data.keys():
        deseason_col = f'{dataset_name}_deseason'
        if deseason_col in z_score_df.columns:
            values = z_score_df[deseason_col].dropna()
            if len(values) > 1:
                z_scores = stats.zscore(values)
                valid_indices = z_score_df[deseason_col].notna()
                z_score_df.loc[valid_indices, f'{dataset_name}_zscore'] = z_scores
                print(f"  {dataset_name} Z-scores calculated: {len(z_scores)} values")
            else:
                print(f"  WARNING: Insufficient data for {dataset_name} Z-scores")

    # Create comprehensive visualization
    print("Creating visualization...")
    fig, axes = plt.subplots(len(all_monthly_data), 1, figsize=(15, 4 * len(all_monthly_data)), sharex=True)
    if len(all_monthly_data) == 1:
        axes = [axes]

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

    trend_summary = []

    for i, dataset_name in enumerate(all_monthly_data.keys()):
        zscore_col = f'{dataset_name}_zscore'
        deseason_col = f'{dataset_name}_deseason'

        print(f"  Plotting {dataset_name}...")

        if zscore_col in z_score_df.columns:
            valid_data = z_score_df.dropna(subset=[zscore_col])
            print(f"    Valid data points for {dataset_name}: {len(valid_data)}")

            if len(valid_data) > 0:
                # Plot the data points and filled area
                axes[i].plot(valid_data['year'], valid_data[zscore_col],
                             color=colors[i % len(colors)], linewidth=2, alpha=0.8,
                             marker='o', markersize=5, label='Deseasonalized Data')
                axes[i].fill_between(valid_data['year'], valid_data[zscore_col],
                                     alpha=0.3, color=colors[i % len(colors)])

                # Calculate and plot trend line
                slope, r_squared, p_value = calculate_trend_stats(
                    valid_data['year'].values, valid_data[zscore_col].values
                )

                print(f"    Trend stats for {dataset_name}: slope={slope:.4f}, R²={r_squared:.3f}, p={p_value:.4f}")

                if not np.isnan(slope):
                    # Plot trend line
                    trend_years = np.array([valid_data['year'].min(), valid_data['year'].max()])
                    trend_values = slope * (trend_years - trend_years.mean()) + valid_data[zscore_col].mean()

                    trend_color = 'darkred' if p_value < 0.05 else 'gray'
                    trend_style = '-' if p_value < 0.05 else '--'

                    axes[i].plot(trend_years, trend_values,
                                 color=trend_color, linewidth=2.5,
                                 linestyle=trend_style, alpha=0.8,
                                 label=f'Trend (p={p_value:.3f})')

                    trend_summary.append({
                        'Dataset': dataset_name,
                        'Slope': slope,
                        'R²': r_squared,
                        'P-value': p_value,
                        'Significance': 'Significant' if p_value < 0.05 else 'Not Significant'
                    })

                # Add statistics box
                if deseason_col in merged_yearly_df.columns:
                    mean_val = merged_yearly_df[deseason_col].mean()
                    std_val = merged_yearly_df[deseason_col].std()
                    axes[i].text(0.02, 0.95,
                                 f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nSlope: {slope:.4f}/yr',
                                 transform=axes[i].transAxes, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                 fontsize=9)

                axes[i].legend(loc='upper right', fontsize=9)
            else:
                print(f"    WARNING: No valid data to plot for {dataset_name}")
                axes[i].text(0.5, 0.5, f'No data available for {dataset_name}',
                             transform=axes[i].transAxes, ha='center', va='center',
                             fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        else:
            print(f"    WARNING: No Z-score column found for {dataset_name}")
            axes[i].text(0.5, 0.5, f'Z-score calculation failed for {dataset_name}',
                         transform=axes[i].transAxes, ha='center', va='center',
                         fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        axes[i].set_ylabel(f'{dataset_name}\n(Z-score)', fontsize=11)
        axes[i].grid(True, alpha=0.3)
        axes[i].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        axes[i].set_xticks(range(2001, 2022, 2))
        axes[i].set_xlim(2000.5, 2021.5)

    axes[-1].set_xlabel('Year', fontsize=12)
    plt.suptitle('Annual Trends of Deseasonalized Climate-Eco Indicators (2001-2021)\n' +
                 'California Drought Indices - Z-score Normalized with Trend Analysis',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Save the plot
    plt.savefig('california_drought_indices_deseasonalized_trends.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print trend summary
    print("\n" + "=" * 70)
    print("TREND ANALYSIS SUMMARY")
    print("=" * 70)

    if trend_summary:
        trend_df = pd.DataFrame(trend_summary)
        for _, row in trend_df.iterrows():
            print(f"{row['Dataset']:>8}: Slope = {row['Slope']:>8.4f}/yr, R² = {row['R²']:>6.3f}, "
                  f"p-value = {row['P-value']:>7.4f} ({row['Significance']})")
    else:
        print("No trend analysis could be performed (insufficient data)")
        trend_df = pd.DataFrame(columns=['Dataset', 'Slope', 'R²', 'P-value', 'Significance'])

    # Save processed data to Excel
    print("\nSaving processed data to Excel...")

    # Prepare comprehensive Excel output
    with pd.ExcelWriter('california_drought_indices_comprehensive_analysis.xlsx') as writer:
        # Monthly data with deseasonalized values
        merged_monthly_df.to_excel(writer, sheet_name='Monthly_Data', index=False)

        # Yearly aggregated data
        merged_yearly_df.to_excel(writer, sheet_name='Yearly_Data', index=False)

        # Z-scores
        z_score_df.to_excel(writer, sheet_name='Yearly_ZScores', index=False)

        # Trend analysis
        if not trend_df.empty:
            trend_df.to_excel(writer, sheet_name='Trend_Analysis', index=False)

    print("Comprehensive analysis saved to 'california_drought_indices_comprehensive_analysis.xlsx'")

    # Final summary
    print(f"\nFINAL ANALYSIS SUMMARY:")
    print(f"Time period: 2001-2021 ({len(merged_yearly_df)} years)")
    print(f"Datasets analyzed: {len(all_monthly_data)}")

    if not trend_df.empty and 'P-value' in trend_df.columns:
        significant_trends = trend_df[trend_df['P-value'] < 0.05]
        if len(significant_trends) > 0:
            print(f"\nSignificant trends detected in {len(significant_trends)} datasets:")
            for _, row in significant_trends.iterrows():
                direction = "increasing" if row['Slope'] > 0 else "decreasing"
                print(f"  - {row['Dataset']}: {direction} trend ({row['Slope']:.4f}/yr)")
        else:
            print("\nNo statistically significant trends detected in any dataset.")
    else:
        print("\nTrend analysis could not be completed.")

    return merged_monthly_df, merged_yearly_df, z_score_df, trend_df


if __name__ == "__main__":
    monthly_df, yearly_df, zscore_df, trends_df = main()