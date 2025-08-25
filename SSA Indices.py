import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
from scipy import stats
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore')


def read_tif_and_calculate_average(file_path):
    """
    Read a TIFF file and calculate the average of all pixels
    """
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)  # Read first band
            # Handle nodata values
            if src.nodata is not None:
                data = np.ma.masked_equal(data, src.nodata)
            # Calculate average, ignoring NaN/nodata values
            avg_value = np.nanmean(data)
            return avg_value
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
    except KeyboardInterrupt:
        print(f"Process interrupted while reading {file_path}")
        raise


def extract_date_from_filename(filename, dataset_type):
    """
    Extract date from filename pattern based on dataset type
    """
    try:
        # Remove the .tif extension first
        filename_no_ext = filename.replace('.tif', '')

        if dataset_type == 'GOSIF':
            # GOSIF_Grasslands_CALIFORNIA_GOSIF_200101
            date_str = filename_no_ext.split('_')[-1]
            year = int(date_str[:4])
            month = int(date_str[4:6])
        elif dataset_type == 'LST':
            # LST_Grasslands_MYD21C1_CA_LST_2003_01
            parts = filename_no_ext.split('_')
            year = int(parts[-2])
            month = int(parts[-1])
        elif dataset_type in ['NDVI', 'NIRv', 'VPD']:
            # NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_2001_01
            # NIRv_Shrublands_NIRv_MOD13Q1_CA_5km_2001_01
            parts = filename_no_ext.split('_')
            year = int(parts[-2])
            month = int(parts[-1])
        elif dataset_type == 'TWS':
            # TWS_Grasslands_TWS_GRACE_CA_2002_04
            parts = filename_no_ext.split('_')
            year = int(parts[-2])
            month = int(parts[-1])
        elif dataset_type == 'SPEI':
            # Assuming similar pattern to others
            parts = filename_no_ext.split('_')
            year = int(parts[-2])
            month = int(parts[-1])

        return datetime(year, month, 1)
    except Exception as e:
        print(f"Error extracting date from {filename}: {e}")
        return None


def fill_missing_data_gam_approach(dates, values, dataset_name, land_type):
    """
    Fill missing data using the 3-step approach:
    1. Average of same month in adjacent years
    2. GAM-like trend estimation using polynomial regression
    3. Overall monthly average as fallback
    """
    # Convert to pandas for easier manipulation
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'year': [d.year for d in dates],
        'month': [d.month for d in dates]
    })

    # Create complete date range
    min_date = min(dates)
    max_date = max(dates)
    all_dates = pd.date_range(start=min_date, end=max_date, freq='MS')

    # Create complete dataframe
    complete_df = pd.DataFrame({
        'date': all_dates,
        'year': all_dates.year,
        'month': all_dates.month
    })

    # Merge with existing data
    complete_df = complete_df.merge(df, on=['date', 'year', 'month'], how='left')

    # Fill missing values
    filled_count = 0

    for idx, row in complete_df.iterrows():
        if pd.isna(row['value']):
            current_year = row['year']
            current_month = row['month']

            # Method 1: Average of same month in adjacent years
            prev_year_value = complete_df[(complete_df['year'] == current_year - 1) &
                                          (complete_df['month'] == current_month)]['value'].values
            next_year_value = complete_df[(complete_df['year'] == current_year + 1) &
                                          (complete_df['month'] == current_month)]['value'].values

            if len(prev_year_value) > 0 and len(next_year_value) > 0:
                if not pd.isna(prev_year_value[0]) and not pd.isna(next_year_value[0]):
                    complete_df.at[idx, 'value'] = (prev_year_value[0] + next_year_value[0]) / 2
                    filled_count += 1
                    continue

            # Method 2: GAM-like trend estimation using polynomial regression
            month_data = complete_df[complete_df['month'] == current_month].copy()
            month_data = month_data.dropna(subset=['value'])

            if len(month_data) >= 3:  # Need at least 3 points for trend
                try:
                    # Create polynomial features for non-linear trend
                    X = month_data['year'].values.reshape(-1, 1)
                    y = month_data['value'].values

                    # Use polynomial regression as GAM approximation
                    degree = min(2, len(month_data) - 1)  # Adaptive degree
                    poly_reg = Pipeline([
                        ('poly', PolynomialFeatures(degree=degree)),
                        ('linear', LinearRegression())
                    ])

                    poly_reg.fit(X, y)
                    predicted_value = poly_reg.predict([[current_year]])[0]

                    complete_df.at[idx, 'value'] = predicted_value
                    filled_count += 1
                    continue
                except:
                    pass

            # Method 3: Monthly average fallback
            monthly_avg = complete_df[complete_df['month'] == current_month]['value'].mean()
            if not pd.isna(monthly_avg):
                complete_df.at[idx, 'value'] = monthly_avg
                filled_count += 1

    if filled_count > 0:
        print(f"  {dataset_name} - {land_type}: Filled {filled_count} missing values")

    return complete_df['date'].tolist(), complete_df['value'].tolist()


def seasonal_trend_decomposition(dates, values):
    """
    Advanced seasonal-trend decomposition to identify anomalies

    Returns:
    trend: Long-term trend
    expected_seasonal: Expected seasonal pattern
    anomalies: Deviations from expected seasonal behavior (what you want!)
    """
    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'value': values,
        'year': [d.year for d in dates],
        'month': [d.month for d in dates]
    })
    df = df.sort_values('date').reset_index(drop=True)

    # Step 1: Calculate long-term trend using LOWESS-like approach
    # Use rolling mean with larger window for trend
    window_size = max(12, len(values) // 8)  # At least 1 year, adaptive
    trend = df['value'].rolling(window=window_size, center=True, min_periods=1).mean()

    # Step 2: Detrend the data
    detrended = df['value'] - trend

    # Step 3: Calculate expected seasonal pattern for each month
    seasonal_df = pd.DataFrame({
        'month': df['month'],
        'detrended': detrended
    })

    # Calculate robust seasonal averages (median to avoid outlier influence)
    monthly_patterns = seasonal_df.groupby('month')['detrended'].median().to_dict()

    # Create expected seasonal pattern
    expected_seasonal = df['month'].map(monthly_patterns)

    # Step 4: Calculate anomalies (deviations from expected seasonal behavior)
    anomalies = detrended - expected_seasonal

    # Step 5: Smooth the anomalies to highlight significant events
    # Use smaller window to preserve drought/disturbance signals
    anomaly_window = max(3, len(anomalies) // 24)  # ~3 months smoothing
    smoothed_anomalies = pd.Series(anomalies).rolling(
        window=anomaly_window, center=True, min_periods=1
    ).mean()

    return trend.values, expected_seasonal.values, smoothed_anomalies.values


def classify_anomaly_severity(anomalies, dataset_type):
    """
    Classify anomaly severity based on standard deviations
    """
    std = np.std(anomalies)

    # Define thresholds based on dataset characteristics
    if dataset_type in ['GOSIF', 'NDVI', 'NIRv']:  # Vegetation indices
        severe_threshold = 2.0 * std
        moderate_threshold = 1.0 * std
    elif dataset_type == 'SPEI':  # Drought index
        severe_threshold = 1.5 * std
        moderate_threshold = 0.8 * std
    else:  # Other indices
        severe_threshold = 2.5 * std
        moderate_threshold = 1.5 * std

    return severe_threshold, moderate_threshold


def process_land_type_data(folder_path, land_type_pattern, dataset_type):
    """
    Process all TIFF files in a folder for a specific land type
    """
    # Get all TIFF files matching the pattern
    pattern = os.path.join(folder_path, f"*{land_type_pattern}*.tif")
    tif_files = glob.glob(pattern)

    if not tif_files:
        print(f"No TIFF files found for pattern: {pattern}")
        return None, None, None, None

    print(f"  Found {len(tif_files)} files for {land_type_pattern}")

    dates = []
    averages = []

    for tif_file in sorted(tif_files):
        try:
            filename = os.path.basename(tif_file)
            date = extract_date_from_filename(filename, dataset_type)

            if date is not None:
                avg_value = read_tif_and_calculate_average(tif_file)
                if avg_value is not None:
                    dates.append(date)
                    averages.append(avg_value)
        except KeyboardInterrupt:
            print(f"Process interrupted while processing {tif_file}")
            raise
        except Exception as e:
            print(f"Error processing {tif_file}: {e}")
            continue

    # Apply missing data filling for LST, TWS, and SPEI
    if dataset_type in ['LST', 'TWS', 'SPEI'] and dates and averages:
        dates, averages = fill_missing_data_gam_approach(dates, averages, dataset_type, land_type_pattern)

    # Apply seasonal-trend decomposition if we have enough data
    if len(averages) >= 24:  # Need at least 2 years of monthly data
        try:
            trend, expected_seasonal, anomalies = seasonal_trend_decomposition(dates, averages)
            return dates, trend, expected_seasonal, anomalies
        except Exception as e:
            print(f"  Seasonal decomposition failed for {land_type_pattern}: {e}")
            return dates, None, None, None
    else:
        print(f"  Not enough data for seasonal decomposition for {land_type_pattern}")
        return dates, None, None, None


def create_dataset_visualization(dataset_config):
    """
    Create visualization showing trend and anomalies for drought/disturbance detection
    """
    dataset_name = dataset_config['name']
    base_paths = dataset_config['paths']
    land_types = dataset_config['land_types']
    land_type_labels = dataset_config['labels']

    print(f"\nProcessing {dataset_name} dataset...")

    # Create figure with 4 subplots (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(15, 16))
    fig.suptitle(f'{dataset_name} Anomaly Detection for California Land Types\n(Trend + Seasonal Anomalies)',
                 fontsize=16, fontweight='bold')

    # Define colors for different datasets
    colors = {
        'GOSIF': '#228B22',  # Forest green
        'LST': '#FF4500',  # Orange red
        'NDVI': '#006400',  # Dark green
        'NIRv': '#32CD32',  # Lime green
        'SPEI': '#1E90FF',  # Dodger blue
        'TWS': '#4169E1',  # Royal blue
        'VPD': '#8B008B'  # Dark magenta
    }

    trend_color = colors.get(dataset_name, '#228B22')
    anomaly_color = '#DC143C'  # Crimson for anomalies
    severe_color = '#8B0000'  # Dark red for severe anomalies

    # First pass: collect all anomalies for consistent scaling
    all_anomalies = []
    all_data = []

    for i, (path, land_type, label) in enumerate(zip(base_paths, land_types, land_type_labels)):
        dates, trend, expected_seasonal, anomalies = process_land_type_data(path, land_type, dataset_name)
        all_data.append((dates, trend, expected_seasonal, anomalies, label))
        if anomalies is not None:
            all_anomalies.extend(anomalies)

    # Find consistent scaling for anomalies
    if all_anomalies:
        anomaly_std = np.std(all_anomalies)
        anomaly_mean = np.mean(all_anomalies)
        anomaly_max = max(abs(np.min(all_anomalies)), abs(np.max(all_anomalies)))
        y_lim = max(3 * anomaly_std, anomaly_max * 1.1)
    else:
        y_lim = 1

    # Second pass: create plots
    for i, (dates, trend, expected_seasonal, anomalies, label) in enumerate(all_data):
        if dates and trend is not None:
            # Calculate severity thresholds
            severe_threshold, moderate_threshold = classify_anomaly_severity(anomalies, dataset_name)

            # Create the plot
            ax1 = axes[i]
            ax2 = ax1.twinx()

            # Plot trend line (left axis)
            line1 = ax1.plot(dates, trend, color=trend_color, linewidth=3,
                             label=f'{dataset_name} Trend', alpha=0.8)

            # Plot anomalies (right axis)
            line2 = ax2.plot(dates, anomalies, color=anomaly_color, linewidth=2,
                             label='Seasonal Anomalies', alpha=0.7)

            # Highlight severe anomalies
            severe_mask = np.abs(anomalies) > severe_threshold
            if np.any(severe_mask):
                severe_dates = [d for d, mask in zip(dates, severe_mask) if mask]
                severe_values = [v for v, mask in zip(anomalies, severe_mask) if mask]
                ax2.scatter(severe_dates, severe_values, color=severe_color, s=50,
                            zorder=5, label='Severe Anomalies', alpha=0.8)

            # Add reference lines for anomalies
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.axhline(y=severe_threshold, color=severe_color, linestyle='--', alpha=0.5)
            ax2.axhline(y=-severe_threshold, color=severe_color, linestyle='--', alpha=0.5)
            ax2.axhline(y=moderate_threshold, color=anomaly_color, linestyle=':', alpha=0.5)
            ax2.axhline(y=-moderate_threshold, color=anomaly_color, linestyle=':', alpha=0.5)

            # Fill severe anomaly areas
            ax2.fill_between(dates, -severe_threshold, -y_lim, alpha=0.1, color=severe_color)
            ax2.fill_between(dates, severe_threshold, y_lim, alpha=0.1, color=severe_color)

            # Set axis limits and labels
            ax2.set_ylim(-y_lim, y_lim)
            ax1.set_ylabel(f'{dataset_name} Trend', color=trend_color, fontweight='bold')
            ax2.set_ylabel('Anomalies', color=anomaly_color, fontweight='bold')
            ax1.tick_params(axis='y', labelcolor=trend_color)
            ax2.tick_params(axis='y', labelcolor=anomaly_color)

            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

            # Count significant anomalies
            severe_count = np.sum(np.abs(anomalies) > severe_threshold)
            moderate_count = np.sum((np.abs(anomalies) > moderate_threshold) &
                                    (np.abs(anomalies) <= severe_threshold))

            print(f"  {label}: {severe_count} severe anomalies, {moderate_count} moderate anomalies")

            # Add title with anomaly info
            axes[i].set_title(f'{label} - {dataset_name}\n({severe_count} Severe, {moderate_count} Moderate Anomalies)',
                              fontweight='bold')

        else:
            axes[i].text(0.5, 0.5, f'No data or insufficient data for {label}',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{label} - No Data Available')

        # Common formatting
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)

    # Set x-label only for the bottom subplot
    axes[-1].set_xlabel('Date', fontweight='bold')

    # Add interpretation note
    fig.text(0.02, 0.02,
             f'Interpretation: Negative {dataset_name} anomalies in spring/summer may indicate drought stress.\n' +
             'Severe anomalies (dark red) represent significant departures from normal seasonal patterns.',
             fontsize=9, style='italic', wrap=True)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    print(f"  Anomaly analysis complete for {dataset_name}")

    # Show plot
    plt.show()

    return fig


def main():
    """
    Main function to process all datasets with anomaly detection
    """
    # Define all dataset configurations
    datasets = {
        'GOSIF': {
            'name': 'GOSIF',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\GOSIF",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\GOSIF",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\GOSIF",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\GOSIF"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        },
        'LST': {
            'name': 'LST',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\LST",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\LST",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\LST",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\LST"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        },
        'NDVI': {
            'name': 'NDVI',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\NDVI",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\NDVI",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\NDVI",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\NDVI"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        },
        'NIRv': {
            'name': 'NIRv',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\NIRv",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\NIRv",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\NIRv",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\NIRv"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        },
        'SPEI': {
            'name': 'SPEI',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\SPEI",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\SPEI",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\SPEI",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\SPEI"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        },
        'TWS': {
            'name': 'TWS',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\TWS",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\TWS",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\TWS",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\TWS"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        },
        'VPD': {
            'name': 'VPD',
            'paths': [
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\VPD",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\VPD",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\VPD",
                r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\VPD"
            ],
            'land_types': ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands'],
            'labels': ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']
        }
    }

    # Check if required libraries are installed
    required_libraries = ['rasterio', 'matplotlib', 'numpy', 'scipy', 'pandas', 'sklearn']
    missing_libraries = []

    for lib in required_libraries:
        try:
            __import__(lib)
        except ImportError:
            missing_libraries.append(lib)

    if missing_libraries:
        print("Missing required libraries:")
        for lib in missing_libraries:
            print(f"  - {lib}")
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing_libraries)}")
        return

    # Process all datasets
    figures = {}

    print("=" * 80)
    print("ANOMALY DETECTION TOOL FOR DROUGHT AND DISTURBANCE ANALYSIS")
    print("=" * 80)
    print("This tool identifies unusual changes in ecological indicators:")
    print("• TREND LINE: Shows long-term changes")
    print("• SEASONAL ANOMALIES: Deviations from expected seasonal patterns")
    print("• SEVERE ANOMALIES: Significant drought/disturbance events (dark red points)")
    print("• DROUGHT DETECTION: Negative vegetation anomalies in growing season")
    print("=" * 80)

    for dataset_name, config in datasets.items():
        try:
            fig = create_dataset_visualization(config)
            figures[dataset_name] = fig
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("ANOMALY DETECTION ANALYSIS COMPLETE!")
    print(f"Successfully processed {len(figures)} out of {len(datasets)} datasets")
    print("\nLook for:")
    print("• Negative GOSIF/NDVI/NIRv anomalies during spring/summer = Drought stress")
    print("• Positive LST/VPD anomalies = Heat stress periods")
    print("• Negative SPEI anomalies = Meteorological drought")
    print("• Negative TWS anomalies = Groundwater/soil moisture deficit")
    print("=" * 80)

    return figures


# Main execution
if __name__ == "__main__":
    figures = main()