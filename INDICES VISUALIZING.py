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


def process_land_type_data(folder_path, land_type_pattern, dataset_type):
    """
    Process all TIFF files in a folder for a specific land type
    """
    # Get all TIFF files matching the pattern
    pattern = os.path.join(folder_path, f"*{land_type_pattern}*.tif")
    tif_files = glob.glob(pattern)

    if not tif_files:
        print(f"No TIFF files found for pattern: {pattern}")
        return None, None

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

    return dates, averages


def create_dataset_visualization(dataset_config):
    """
    Create visualization for a specific dataset with 4 subplots
    """
    dataset_name = dataset_config['name']
    base_paths = dataset_config['paths']
    land_types = dataset_config['land_types']
    land_type_labels = dataset_config['labels']

    print(f"\nProcessing {dataset_name} dataset...")

    # Create figure with 4 subplots (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle(f'{dataset_name} Time Series Analysis for California Land Types',
                 fontsize=16, fontweight='bold')

    # Define colors for different datasets
    colors = {
        'GOSIF': '#32CD32',
        'LST': '#FF4500',
        'NDVI': '#228B22',
        'NIRv': '#006400',
        'SPEI': '#1E90FF',
        'TWS': '#4169E1',
        'VPD': '#8B008B'
    }

    main_color = colors.get(dataset_name, '#32CD32')
    trend_color = '#8B0000'  # Dark red for trend lines

    # First pass: collect all data to find global max/min for y-axis
    all_data = []
    all_values = []

    for i, (path, land_type, label) in enumerate(zip(base_paths, land_types, land_type_labels)):
        dates, averages = process_land_type_data(path, land_type, dataset_name)
        all_data.append((dates, averages, label))
        if averages:
            all_values.extend(averages)

    # Find global range for consistent y-axis scaling
    if all_values:
        global_min = min(all_values)
        global_max = max(all_values)
        y_range = global_max - global_min
        y_min = global_min - y_range * 0.05
        y_max = global_max + y_range * 0.05
    else:
        y_min, y_max = 0, 1

    # Second pass: create plots with consistent y-axis
    for i, (dates, averages, label) in enumerate(all_data):
        if dates and averages:
            # Convert dates to numerical format for trend calculation
            date_nums = [d.toordinal() for d in dates]

            # Calculate trend line
            slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, averages)
            trend_line = [slope * x + intercept for x in date_nums]

            # Plot continuous line
            axes[i].plot(dates, averages, color=main_color, linewidth=2,
                         marker='o', markersize=3, label=f'{dataset_name} Values')

            # Plot trend line
            axes[i].plot(dates, trend_line, color=trend_color, linewidth=2,
                         linestyle='--', label=f'Trend (slope: {slope:.2e})')

            # Set consistent y-axis limits
            axes[i].set_ylim(y_min, y_max)

            # Customize subplot
            axes[i].set_title(f'{label} - {dataset_name} Average Values', fontweight='bold')
            axes[i].set_ylabel(f'Average {dataset_name} Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)

            print(
                f"  {label}: Processed {len(dates)} files, Range: {min(averages):.4f} to {max(averages):.4f}, Trend slope: {slope:.2e}")
        else:
            axes[i].text(0.5, 0.5, f'No data found for {label}',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{label} - No Data Available')
            axes[i].set_ylim(y_min, y_max)

    # Set x-label only for the bottom subplot
    axes[-1].set_xlabel('Date')

    # Adjust layout
    plt.tight_layout()

    print(f"  Global value range: {global_min:.4f} to {global_max:.4f}")
    print(f"  All plots have consistent y-axis scaling for easy comparison.")

    # Show plot
    plt.show()

    return fig


def main():
    """
    Main function to process all datasets
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
    print("MULTI-DATASET VISUALIZATION TOOL FOR CALIFORNIA LAND TYPES")
    print("=" * 80)
    print("This tool will process 7 datasets (GOSIF, LST, NDVI, NIRv, SPEI, TWS, VPD)")
    print("Each dataset will generate a visualization with 4 subplots (one per land type)")
    print("Missing data in LST, TWS, and SPEI will be filled using GAM-like approach")
    print("=" * 80)

    for dataset_name, config in datasets.items():
        try:
            fig = create_dataset_visualization(config)
            figures[dataset_name] = fig
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE!")
    print(f"Successfully processed {len(figures)} out of {len(datasets)} datasets")
    print("=" * 80)

    return figures


# Main execution
if __name__ == "__main__":
    figures = main()