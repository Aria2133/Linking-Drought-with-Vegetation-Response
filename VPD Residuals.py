import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
from scipy import stats
import pandas as pd
from sklearn.linear_model import LinearRegression
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
            return avg_value if not np.isnan(avg_value) else None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def extract_date_from_filename(filename):
    """
    Extract date from filename pattern like VPD_Grasslands_VPDmax_GRIDMET_CA_5km_2001_01
    """
    try:
        print(f"        Trying to extract date from: {filename}")

        # Remove file extension
        name_without_ext = filename.replace('.tif', '').replace('.TIF', '')

        # Split by underscore and look for year and month patterns
        parts = name_without_ext.split('_')
        print(f"        Parts: {parts}")

        # Look for year-month pattern (YYYY_MM) at the end
        if len(parts) >= 2:
            try:
                year = int(parts[-2])
                month = int(parts[-1])
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    result = datetime(year, month, 1)
                    print(f"        Extracted date: {result}")
                    return result
            except ValueError:
                pass

        # Look for combined pattern (YYYYMM) at the end
        if len(parts) >= 1:
            try:
                date_part = parts[-1]
                if len(date_part) == 6:  # YYYYMM
                    year = int(date_part[:4])
                    month = int(date_part[4:6])
                    if 1900 <= year <= 2100 and 1 <= month <= 12:
                        result = datetime(year, month, 1)
                        print(f"        Extracted date: {result}")
                        return result
            except ValueError:
                pass

        # Look anywhere in the filename for date patterns
        import re
        # Pattern for YYYY_MM or YYYY-MM
        date_match = re.search(r'(\d{4})[_-](\d{1,2})', filename)
        if date_match:
            year = int(date_match.group(1))
            month = int(date_match.group(2))
            if 1900 <= year <= 2100 and 1 <= month <= 12:
                result = datetime(year, month, 1)
                print(f"        Extracted date with regex: {result}")
                return result

        # Pattern for YYYYMM
        date_match = re.search(r'(\d{6})', filename)
        if date_match:
            date_str = date_match.group(1)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            if 1900 <= year <= 2100 and 1 <= month <= 12:
                result = datetime(year, month, 1)
                print(f"        Extracted date with regex (YYYYMM): {result}")
                return result

        print(f"        Could not extract date from filename")
        return None

    except Exception as e:
        print(f"        Error extracting date: {e}")
        return None


def process_land_type_data(folder_path, land_type_pattern):
    """
    Process all TIFF files in a folder for a specific land type
    """
    print(f"    Looking in folder: {folder_path}")

    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"    ERROR: Folder does not exist!")
        return None, None

    # Try multiple patterns to find files
    patterns = [
        f"*{land_type_pattern}*.tif",
        f"*{land_type_pattern}*.TIF",
        f"VPD_{land_type_pattern}*.tif",
        f"VPD_{land_type_pattern}*.TIF",
        "*.tif",
        "*.TIF"
    ]

    tif_files = []
    for pattern in patterns:
        full_pattern = os.path.join(folder_path, pattern)
        found_files = glob.glob(full_pattern)
        if found_files:
            print(f"    Found {len(found_files)} files with pattern: {pattern}")
            tif_files = found_files
            break
        else:
            print(f"    No files found with pattern: {pattern}")

    if not tif_files:
        # List all files in directory for debugging
        try:
            all_files = os.listdir(folder_path)
            print(f"    All files in directory ({len(all_files)} total):")
            for file in all_files[:10]:  # Show first 10 files
                print(f"      {file}")
            if len(all_files) > 10:
                print(f"      ... and {len(all_files) - 10} more files")
        except:
            print(f"    Cannot list directory contents")
        return None, None

    # Collect all available data
    dates = []
    averages = []
    successful_files = 0

    print(f"    Processing {len(tif_files)} files...")
    for tif_file in sorted(tif_files):  # Sort to ensure chronological order
        filename = os.path.basename(tif_file)
        print(f"      Processing: {filename}")

        date = extract_date_from_filename(filename)
        if date is None:
            print(f"        Could not extract date from filename")
            continue

        avg_value = read_tif_and_calculate_average(tif_file)
        if avg_value is None:
            print(f"        Could not read TIFF data")
            continue

        dates.append(date)
        averages.append(avg_value)
        successful_files += 1
        print(f"        Success: {date.strftime('%Y-%m')} -> {avg_value:.4f}")

    print(f"    Successfully processed: {successful_files} files")

    if not dates or not averages:
        print(f"    No valid data extracted from files")
        return None, None

    # Sort by date to ensure chronological order
    sorted_data = sorted(zip(dates, averages), key=lambda x: x[0])
    sorted_dates, sorted_averages = zip(*sorted_data)

    print(f"    Date range: {sorted_dates[0].strftime('%Y-%m')} to {sorted_dates[-1].strftime('%Y-%m')}")
    print(f"    Total data points: {len(sorted_dates)}")

    return list(sorted_dates), list(sorted_averages)


def fit_trend_and_seasonal(dates, values):
    """
    Fit linear trend and sinusoidal seasonal components to the data
    Returns fitted values and model parameters
    """
    if len(dates) < 12:
        print(f"    Insufficient data for trend/seasonal fitting")
        return np.zeros(len(values)), {'trend_slope': 0, 'seasonal_amplitude': 0, 'seasonal_phase': 0}

    # Convert dates to numerical format (days since first date)
    start_date = dates[0]
    time_numeric = np.array([(d - start_date).days for d in dates])

    # Create design matrix for regression
    # Linear trend + annual cycle (sin and cos terms)
    X = np.column_stack([
        np.ones(len(time_numeric)),  # Intercept
        time_numeric,  # Linear trend
        np.sin(2 * np.pi * time_numeric / 365.25),  # Annual sine component
        np.cos(2 * np.pi * time_numeric / 365.25),  # Annual cosine component
    ])

    # Add semi-annual cycle for more complex seasonality
    if len(dates) >= 24:  # Need at least 2 years for semi-annual
        X = np.column_stack([
            X,
            np.sin(4 * np.pi * time_numeric / 365.25),  # Semi-annual sine
            np.cos(4 * np.pi * time_numeric / 365.25),  # Semi-annual cosine
        ])

    # Fit linear regression
    try:
        model = LinearRegression()
        model.fit(X, values)
        fitted_values = model.predict(X)

        # Extract parameters
        params = {
            'intercept': model.intercept_,
            'trend_slope': model.coef_[1],  # Slope in units per day
            'annual_sin': model.coef_[2] if len(model.coef_) > 2 else 0,
            'annual_cos': model.coef_[3] if len(model.coef_) > 3 else 0,
            'r_squared': model.score(X, values)
        }

        # Calculate seasonal amplitude and phase
        if len(model.coef_) > 3:
            seasonal_amplitude = np.sqrt(model.coef_[2] ** 2 + model.coef_[3] ** 2)
            seasonal_phase = np.arctan2(model.coef_[3], model.coef_[2])
        else:
            seasonal_amplitude = 0
            seasonal_phase = 0

        params['seasonal_amplitude'] = seasonal_amplitude
        params['seasonal_phase'] = seasonal_phase

        print(f"    Fitted model: R² = {params['r_squared']:.3f}, "
              f"Trend = {params['trend_slope'] * 365.25:.4f} units/year, "
              f"Seasonal amplitude = {seasonal_amplitude:.4f}")

        return fitted_values, params

    except Exception as e:
        print(f"    Error fitting model: {e}")
        return np.zeros(len(values)), {'trend_slope': 0, 'seasonal_amplitude': 0, 'seasonal_phase': 0}


def calculate_residuals(values, fitted_values):
    """
    Calculate residuals by subtracting fitted trend and seasonal components
    """
    residuals = np.array(values) - np.array(fitted_values)
    return residuals


def create_visualization():
    """
    Create the main visualization with 4 subplots showing VPD residuals
    """
    # Define paths and land types for VPD data
    base_paths = [
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\VPD",
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\VPD",
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\VPD",
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\VPD"
    ]

    land_types = ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands']
    land_type_labels = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']

    # Create figure with 4 subplots (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('VPD Residuals (Detrended & Deseasonalized) for California Land Types', fontsize=16, fontweight='bold')

    # Define red/orange color for VPD (appropriate for atmospheric dryness/heat stress)
    bright_orange = '#FF6600'

    # First pass: collect all residual data to find global range for y-axis
    all_residual_data = []
    all_residual_ranges = []

    print("Processing VPD data with traditional trend/seasonal fitting:")

    for i, (path, land_type, label) in enumerate(zip(base_paths, land_types, land_type_labels)):
        print(f"\n{label}:")
        dates, averages = process_land_type_data(path, land_type)

        if dates is not None and averages is not None and len(averages) > 6:  # Need some minimum data
            print(f"  Data range: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')} ({len(dates)} points)")

            # Fit trend and seasonal components
            fitted_values, model_params = fit_trend_and_seasonal(dates, averages)

            # Calculate residuals
            residuals = calculate_residuals(averages, fitted_values)

            all_residual_data.append((dates, residuals, fitted_values, model_params, label))
            all_residual_ranges.extend([min(residuals), max(residuals)])
        else:
            print(f"  Insufficient data for fitting (need >6 points)")
            all_residual_data.append((dates, averages if averages is not None else [], [], {}, label))
            if averages is not None:
                all_residual_ranges.extend([min(averages), max(averages)])

    # Find global range for consistent y-axis scaling
    if all_residual_ranges:
        global_min = min(all_residual_ranges)
        global_max = max(all_residual_ranges)
        y_range = global_max - global_min
        y_min = global_min - 0.05 * y_range
        y_max = global_max + 0.05 * y_range
    else:
        y_min, y_max = -1, 1

    # Second pass: create plots with consistent y-axis
    print(f"\nCreating visualization with y-axis range: [{y_min:.4f}, {y_max:.4f}]")

    for i, (dates, residuals, fitted_values, model_params, label) in enumerate(all_residual_data):
        if dates is not None and len(residuals) > 0:
            # Convert dates to numerical format for trend calculation
            date_nums = [d.toordinal() for d in dates]

            # Calculate trend line for residuals (should be close to zero)
            if len(residuals) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, residuals)
                trend_line = [slope * x + intercept for x in date_nums]
            else:
                slope = 0
                trend_line = [0] * len(dates)

            # Plot continuous line for residuals
            axes[i].plot(dates, residuals, color=bright_orange, linewidth=2,
                         marker='o', markersize=3, label='VPD Residuals')

            # Plot trend line for residuals
            axes[i].plot(dates, trend_line, color='darkorange', linewidth=2,
                         linestyle='--', label=f'Residual Trend (slope: {slope:.2e})')

            # Add zero reference line
            axes[i].axhline(y=0, color='gray', linestyle=':', alpha=0.7, label='Zero Reference')

            # Set consistent y-axis limits
            axes[i].set_ylim(y_min, y_max)

            # Customize subplot with model fit info
            model_info = ""
            if model_params:
                r2 = model_params.get('r_squared', 0)
                trend_yearly = model_params.get('trend_slope', 0) * 365.25
                seasonal_amp = model_params.get('seasonal_amplitude', 0)
                model_info = f" (R²={r2:.3f}, Trend={trend_yearly:.3f}/yr, Seasonal={seasonal_amp:.3f})"

            axes[i].set_title(f'{label} - VPD Residuals{model_info}', fontweight='bold')
            axes[i].set_ylabel('VPD Residuals (kPa)')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)

            residual_std = np.std(residuals) if len(residuals) > 1 else 0
            print(f"{label}: Residual range [{min(residuals):.4f}, {max(residuals):.4f}], "
                  f"Std: {residual_std:.4f}, Residual trend slope: {slope:.2e}")

        else:
            axes[i].text(0.5, 0.5, f'Insufficient data for {label}',
                         ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'{label} - Insufficient Data')
            axes[i].set_ylim(y_min, y_max)

    # Set x-label only for the bottom subplot
    axes[-1].set_xlabel('Date')

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

    return fig


# Main execution
if __name__ == "__main__":
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
    else:
        print("Creating VPD residuals visualization with traditional detrending...")
        fig = create_visualization()
        print("\nVisualization complete!")
        print("\nNote: Residuals calculated by subtracting fitted linear trend + sinusoidal seasonal components.")
        print("The residuals show variations not explained by the simple trend/seasonal model.")