import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio
from scipy import stats
import pandas as pd
from scipy.linalg import svd


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


def extract_date_from_filename(filename):
    """
    Extract date from filename pattern like GOSIF_Grasslands_CALIFORNIA_GOSIF_200101
    """
    try:
        # Extract the date part (last 6 digits)
        date_str = filename.split('_')[-1]
        # Convert YYYYMM to datetime
        year = int(date_str[:4])
        month = int(date_str[4:6])
        return datetime(year, month, 1)
    except:
        return None


def process_land_type_data(folder_path, land_type_pattern):
    """
    Process all TIFF files in a folder for a specific land type
    """
    # Get all TIFF files matching the pattern
    pattern = os.path.join(folder_path, f"*{land_type_pattern}*.tif")
    tif_files = glob.glob(pattern)

    if not tif_files:
        print(f"No TIFF files found for pattern: {pattern}")
        return None, None

    dates = []
    averages = []

    for tif_file in sorted(tif_files):
        filename = os.path.basename(tif_file)
        date = extract_date_from_filename(filename)
        avg_value = read_tif_and_calculate_average(tif_file)

        if date is not None and avg_value is not None:
            dates.append(date)
            averages.append(avg_value)

    return dates, averages


def create_trajectory_matrix(time_series, window_length):
    """
    Create trajectory matrix for SSA
    """
    N = len(time_series)
    K = N - window_length + 1

    # Create trajectory matrix
    X = np.zeros((window_length, K))
    for i in range(K):
        X[:, i] = time_series[i:i + window_length]

    return X


def ssa_decomposition(time_series, window_length=None, n_components=None):
    """
    Perform Singular Spectrum Analysis (SSA) decomposition
    """
    N = len(time_series)

    # Default window length (typically N/3 to N/2)
    if window_length is None:
        window_length = min(N // 3, 50)  # Cap at 50 for computational efficiency

    # Create trajectory matrix
    X = create_trajectory_matrix(time_series, window_length)

    # Singular Value Decomposition
    U, sigma, Vt = svd(X, full_matrices=False)

    # Number of components to use
    if n_components is None:
        n_components = len(sigma)

    # Reconstruct components
    components = []
    for i in range(min(n_components, len(sigma))):
        Xi = sigma[i] * np.outer(U[:, i], Vt[i, :])

        # Diagonal averaging (Hankelization)
        reconstructed = np.zeros(N)
        for k in range(N):
            indices = []
            for j in range(window_length):
                if 0 <= k - j < N - window_length + 1:
                    indices.append((j, k - j))

            if indices:
                reconstructed[k] = np.mean([Xi[idx[0], idx[1]] for idx in indices])

        components.append(reconstructed)

    return components, sigma


def identify_seasonal_components(components, sigma, seasonal_period=12):
    """
    Identify which components represent seasonal patterns
    Returns indices of trend, seasonal, and noise components
    """
    # Usually the first component is trend
    trend_idx = [0] if len(components) > 0 else []

    # Seasonal components are typically components 2-4 for monthly data
    # We'll use a simple heuristic: components with high variance that aren't the first
    seasonal_idx = []
    noise_idx = []

    # Calculate variance explained by each component
    total_variance = np.sum(sigma ** 2)
    variance_explained = (sigma ** 2) / total_variance

    # Components explaining more than 1% variance (except first) might be seasonal
    for i in range(1, min(len(components), seasonal_period)):
        if variance_explained[i] > 0.01:
            seasonal_idx.append(i)
        else:
            noise_idx.append(i)

    # Remaining components are noise
    for i in range(max(seasonal_idx + [0]) + 1, len(components)):
        noise_idx.append(i)

    return trend_idx, seasonal_idx, noise_idx


def extract_noise_component(time_series, window_length=None):
    """
    Extract noise component using SSA by removing trend and seasonal components
    """
    if len(time_series) < 12:  # Too short for meaningful SSA
        return time_series - np.mean(time_series)

    # Perform SSA decomposition
    components, sigma = ssa_decomposition(time_series, window_length)

    # Identify component types
    trend_idx, seasonal_idx, noise_idx = identify_seasonal_components(components, sigma)

    # Reconstruct noise by summing noise components
    if noise_idx:
        noise = np.sum([components[i] for i in noise_idx], axis=0)
    else:
        # If no clear noise components, use residual after removing trend and seasonal
        trend_seasonal = np.sum([components[i] for i in trend_idx + seasonal_idx], axis=0)
        noise = time_series - trend_seasonal

    print(f"  SSA: {len(trend_idx)} trend, {len(seasonal_idx)} seasonal, {len(noise_idx)} noise components")

    return noise


def create_visualization():
    """
    Create the main visualization with 4 subplots showing noise components
    """
    # Define paths and land types
    base_paths = [
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\GOSIF",
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\GOSIF",
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\GOSIF",
        r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\GOSIF"
    ]

    land_types = ['Grasslands', 'NaturalForests', 'Savanna', 'Shrublands']
    land_type_labels = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']

    # Create figure with 4 subplots (4 rows, 1 column)
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    fig.suptitle('GOSIF Noise Components (SSA Deseasonalized) for California Land Types', fontsize=16,
                 fontweight='bold')

    # Define bright green color
    bright_green = '#32CD32'

    # First pass: collect all noise data to find global range for y-axis
    all_noise_data = []
    all_noise_ranges = []

    print("Performing SSA decomposition for each land type:")

    for i, (path, land_type, label) in enumerate(zip(base_paths, land_types, land_type_labels)):
        dates, averages = process_land_type_data(path, land_type)

        if dates and averages and len(averages) > 12:  # Need sufficient data for SSA
            print(f"\n{label}:")
            # Extract noise component using SSA
            noise_component = extract_noise_component(np.array(averages))
            all_noise_data.append((dates, noise_component, label))
            all_noise_ranges.extend([min(noise_component), max(noise_component)])
        else:
            print(f"\n{label}: Insufficient data for SSA (need >12 points)")
            all_noise_data.append((dates, averages if averages else [], label))
            if averages:
                all_noise_ranges.extend([min(averages), max(averages)])

    # Find global range for consistent y-axis scaling
    if all_noise_ranges:
        global_min = min(all_noise_ranges)
        global_max = max(all_noise_ranges)
        y_range = global_max - global_min
        y_min = global_min - 0.05 * y_range
        y_max = global_max + 0.05 * y_range
    else:
        y_min, y_max = -1, 1

    # Second pass: create plots with consistent y-axis
    print(f"\nCreating visualization with y-axis range: [{y_min:.4f}, {y_max:.4f}]")

    for i, (dates, noise_data, label) in enumerate(all_noise_data):
        if dates and len(noise_data) > 0:
            # Convert dates to numerical format for trend calculation
            date_nums = [d.toordinal() for d in dates]

            # Calculate trend line for noise component
            if len(noise_data) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(date_nums, noise_data)
                trend_line = [slope * x + intercept for x in date_nums]
            else:
                slope = 0
                trend_line = [0] * len(dates)

            # Plot continuous line for noise component
            axes[i].plot(dates, noise_data, color=bright_green, linewidth=2,
                         marker='o', markersize=3, label='Noise Component')

            # Plot trend line
            axes[i].plot(dates, trend_line, color='darkgreen', linewidth=2,
                         linestyle='--', label=f'Trend (slope: {slope:.2e})')

            # Add zero reference line
            axes[i].axhline(y=0, color='gray', linestyle=':', alpha=0.7, label='Zero Reference')

            # Set consistent y-axis limits
            axes[i].set_ylim(y_min, y_max)

            # Customize subplot
            axes[i].set_title(f'{label} - GOSIF Noise Component (Deseasonalized)', fontweight='bold')
            axes[i].set_ylabel('Noise Component Value')
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()

            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)

            print(f"{label}: Noise range [{min(noise_data):.4f}, {max(noise_data):.4f}], Trend slope: {slope:.2e}")
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
    required_libraries = ['rasterio', 'matplotlib', 'numpy', 'scipy', 'pandas']
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
        print("Creating GOSIF noise component visualization using SSA...")
        fig = create_visualization()
        print("\nVisualization complete!")
        print("\nNote: The noise components show the irregular, non-seasonal variations")
        print("in GOSIF values after removing trend and seasonal patterns using SSA.")