import os
import glob
import numpy as np
import pandas as pd
import rasterio
from scipy.signal import correlate
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def extract_date_from_filename(filename):
    """
    Extract date from various filename patterns
    """
    filename_clean = filename.replace('.tif', '')

    # Pattern 1: GNDVI_Grasslands_GNDVI_MOD09A1_CA_5km_2001_01
    # Pattern 2: NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_2001_01
    # Pattern 3: NIRv_Grasslands_NIRv_MOD13Q1_CA_5km_2001_01
    # Pattern 4: LST_Grasslands_MYD21C1_CA_LST_2003_01
    # Pattern 5: SPEI_Grasslands_SPEI1_CSIC_CA_2001_01
    # Pattern 6: TWS_Grasslands_TWS_GRACE_CA_2002_04
    # Pattern 7: VPD_Grasslands_VPDmax_GRIDMET_CA_5km_2001_01

    if filename_clean.endswith('_' + filename_clean.split('_')[-2] + '_' + filename_clean.split('_')[-1]):
        try:
            year = int(filename_clean.split('_')[-2])
            month = int(filename_clean.split('_')[-1])
            if year >= 2001 and 1 <= month <= 12:
                return pd.Timestamp(year=year, month=month, day=1)
        except:
            pass

    # Pattern 8: GOSIF_Grasslands_CALIFORNIA_GOSIF_200101
    if 'GOSIF' in filename_clean:
        parts = filename_clean.split('_')
        for part in parts:
            if len(part) == 6 and part.isdigit():
                try:
                    year = int(part[:4])
                    month = int(part[4:])
                    if year >= 2001 and 1 <= month <= 12:
                        return pd.Timestamp(year=year, month=month, day=1)
                except:
                    pass

    # Pattern 9: Grasslands_GPP_2001_01
    if 'GPP' in filename_clean:
        parts = filename_clean.split('_')
        try:
            year = int(parts[-2])
            month = int(parts[-1])
            if year >= 2001 and 1 <= month <= 12:
                return pd.Timestamp(year=year, month=month, day=1)
        except:
            pass

    return None


def read_tif_files(folder_path, pattern):
    """
    Read all TIFF files matching the pattern and extract spatial means
    """
    files = glob.glob(os.path.join(folder_path, "*.tif"))
    files.sort()  # Sort to ensure chronological order

    data = []
    dates = []

    print(f"    Found {len(files)} TIFF files in {folder_path}")

    for file in files:
        try:
            with rasterio.open(file) as src:
                # Read the data
                raster_data = src.read(1)

                # Calculate spatial mean (excluding nodata values)
                if src.nodata is not None:
                    raster_data = np.where(raster_data == src.nodata, np.nan, raster_data)

                spatial_mean = np.nanmean(raster_data)

                if not np.isnan(spatial_mean):
                    # Extract date from filename
                    filename = os.path.basename(file)
                    date = extract_date_from_filename(filename)

                    if date is not None:
                        data.append(spatial_mean)
                        dates.append(date)
                    else:
                        print(f"    Could not extract date from {filename}")

        except Exception as e:
            print(f"    Error reading {file}: {e}")

    print(f"    Successfully processed {len(data)} files with valid dates")

    if len(data) == 0:
        return pd.Series(dtype=float)

    return pd.Series(data, index=dates).sort_index()


def detrend_and_deseasonalize(ts, freq=12):
    """
    Remove trend and seasonal components from time series
    """
    if len(ts) < 2 * freq:  # Need at least 2 years of data
        # Simple linear detrend for short series
        x = np.arange(len(ts))
        coeffs = np.polyfit(x, ts.values, 1)
        trend = np.polyval(coeffs, x)
        return pd.Series(ts.values - trend, index=ts.index)

    try:
        # Use seasonal decomposition
        decomposition = seasonal_decompose(ts, model='additive', period=freq)
        residuals = decomposition.resid.dropna()
        return residuals
    except:
        # Fallback to linear detrend
        x = np.arange(len(ts))
        coeffs = np.polyfit(x, ts.values, 1)
        trend = np.polyval(coeffs, x)
        return pd.Series(ts.values - trend, index=ts.index)


def find_optimal_lag(x, y, max_lag=12):
    """
    Find optimal lag between two time series using cross-correlation
    """
    # Ensure both series have the same length and aligned dates
    common_dates = x.index.intersection(y.index)
    if len(common_dates) < 24:  # Need at least 2 years of data
        return np.nan

    x_aligned = x.loc[common_dates]
    y_aligned = y.loc[common_dates]

    # Remove any remaining NaN values
    valid_idx = ~(np.isnan(x_aligned) | np.isnan(y_aligned))
    if np.sum(valid_idx) < 12:  # Need at least 1 year of valid data
        return np.nan

    x_clean = x_aligned[valid_idx].values
    y_clean = y_aligned[valid_idx].values

    # Calculate cross-correlation
    correlation = correlate(y_clean, x_clean, mode='full')

    # Get lag range
    lags = np.arange(-len(x_clean) + 1, len(x_clean))

    # Limit to desired lag range
    valid_lags = (lags >= -max_lag) & (lags <= max_lag)
    correlation = correlation[valid_lags]
    lags = lags[valid_lags]

    # Find lag with maximum absolute correlation
    optimal_lag_idx = np.argmax(np.abs(correlation))
    optimal_lag = lags[optimal_lag_idx]

    return optimal_lag


def process_land_type(base_paths, land_type_name, indices):
    """
    Process all indices for a specific land type using raw data
    """
    print(f"Processing {land_type_name}...")

    # Read all indices (raw data, no detrending)
    processed_data = {}

    for idx_name, folder_name in indices.items():
        folder_path = base_paths[folder_name]
        print(f"  Reading {idx_name} data...")

        # Read the time series (raw data)
        ts = read_tif_files(folder_path, idx_name)

        if len(ts) > 0:
            # Use raw data without detrending
            processed_data[idx_name] = ts
            print(f"    {idx_name}: {len(ts)} valid data points")
        else:
            print(f"    {idx_name}: No valid data found")

    # Calculate optimal lags between all pairs using raw data
    lag_matrix = pd.DataFrame(index=list(processed_data.keys()),
                              columns=list(processed_data.keys()))

    for idx1 in processed_data.keys():
        for idx2 in processed_data.keys():
            if idx1 == idx2:
                lag_matrix.loc[idx1, idx2] = 0
            else:
                lag = find_optimal_lag(processed_data[idx1], processed_data[idx2])
                lag_matrix.loc[idx1, idx2] = lag

    return lag_matrix.astype(float)


# Define the data structure
land_types = {
    'Grasslands': {
        'GNDVI': 'GNDVI',
        'GOSIF': 'GOSIF',
        'GPP': 'GPP',
        'LST': 'LST',
        'NDVI': 'NDVI',
        'NIRv': 'NIRv',
        'SPEI': 'SPEI',
        'TWS': 'TWS',
        'VPD': 'VPD'
    },
    'Natural Forests': {
        'GNDVI': 'GNDVI',
        'GOSIF': 'GOSIF',
        'GPP': 'GPP',
        'LST': 'LST',
        'NDVI': 'NDVI',
        'NIRv': 'NIRv',
        'SPEI': 'SPEI',
        'TWS': 'TWS',
        'VPD': 'VPD'
    },
    'Savannas': {
        'GNDVI': 'GNDVI',
        'GOSIF': 'GOSIF',
        'GPP': 'GPP',
        'LST': 'LST',
        'NDVI': 'NDVI',
        'NIRv': 'NIRv',
        'SPEI': 'SPEI',
        'TWS': 'TWS',
        'VPD': 'VPD'
    },
    'Shrublands': {
        'GNDVI': 'GNDVI',
        'GOSIF': 'GOSIF',
        'GPP': 'GPP',
        'LST': 'LST',
        'NDVI': 'NDVI',
        'NIRv': 'NIRv',
        'SPEI': 'SPEI',
        'TWS': 'TWS',
        'VPD': 'VPD'
    }
}

# Define base paths - UPDATE THESE PATHS TO YOUR ACTUAL DATA LOCATIONS
base_data_path = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

# Define base paths for each land type
grasslands_paths = {
    'GNDVI': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "GNDVI"),
    'GOSIF': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "GOSIF"),
    'GPP': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "GPP"),
    'LST': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "LST"),
    'NDVI': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "NDVI"),
    'NIRv': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "NIRv"),
    'SPEI': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "SPEI"),
    'TWS': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "TWS"),
    'VPD': os.path.join(base_data_path, "CALIFORNIA GRASSLANDS", "VPD")
}

natural_forests_paths = {
    'GNDVI': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "GNDVI"),
    'GOSIF': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "GOSIF"),
    'GPP': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "GPP"),
    'LST': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "LST"),
    'NDVI': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "NDVI"),
    'NIRv': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "NIRv"),
    'SPEI': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "SPEI"),
    'TWS': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "TWS"),
    'VPD': os.path.join(base_data_path, "CALIFORNIA NATURAL FORESTS", "VPD")
}

savannas_paths = {
    'GNDVI': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "GNDVI"),
    'GOSIF': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "GOSIF"),
    'GPP': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "GPP"),
    'LST': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "LST"),
    'NDVI': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "NDVI"),
    'NIRv': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "NIRv"),
    'SPEI': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "SPEI"),
    'TWS': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "TWS"),
    'VPD': os.path.join(base_data_path, "CALIFORNIA SAVANNAS", "VPD")
}

shrublands_paths = {
    'GNDVI': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "GNDVI"),
    'GOSIF': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "GOSIF"),
    'GPP': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "GPP"),
    'LST': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "LST"),
    'NDVI': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "NDVI"),
    'NIRv': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "NIRv"),
    'SPEI': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "SPEI"),
    'TWS': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "TWS"),
    'VPD': os.path.join(base_data_path, "CALIFORNIA SHRUBLANDS", "VPD")
}

# Process all land types
all_paths = {
    'Grasslands': grasslands_paths,
    'Natural Forests': natural_forests_paths,
    'Savannas': savannas_paths,
    'Shrublands': shrublands_paths
}

lag_matrices = {}

for land_type_name, paths in all_paths.items():
    indices = land_types[land_type_name]
    lag_matrix = process_land_type(paths, land_type_name, indices)
    lag_matrices[land_type_name] = lag_matrix

# Create the visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Optimal Lag Times by Land Type (months) - Raw Data Analysis', fontsize=16, fontweight='bold')

# Flatten axes for easy iteration
axes_flat = axes.flatten()

# Land type positions to match your reference image
positions = ['Grasslands', 'Natural Forests', 'Savannas', 'Shrublands']

for idx, land_type in enumerate(positions):
    ax = axes_flat[idx]

    if land_type in lag_matrices:
        lag_matrix = lag_matrices[land_type]

        # Create heatmap
        sns.heatmap(lag_matrix,
                    annot=True,
                    fmt='.0f',
                    cmap='viridis',
                    center=0,
                    vmin=-12,
                    vmax=12,
                    cbar_kws={'label': 'Lag (months)'},
                    ax=ax,
                    annot_kws={'size': 8})

        ax.set_title(land_type, fontweight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel('')

        # Rotate x-axis labels
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    else:
        ax.text(0.5, 0.5, f'No data available\nfor {land_type}',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(land_type, fontweight='bold', fontsize=12)
        ax.axis('off')

plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "=" * 50)
print("SUMMARY STATISTICS")
print("=" * 50)

for land_type, lag_matrix in lag_matrices.items():
    print(f"\n{land_type}:")
    print(f"  Mean absolute lag: {np.nanmean(np.abs(lag_matrix.values)):.2f} months")
    print(f"  Max lag: {np.nanmax(lag_matrix.values):.0f} months")
    print(f"  Min lag: {np.nanmin(lag_matrix.values):.0f} months")
    print(f"  Missing values: {np.sum(np.isnan(lag_matrix.values))} out of {lag_matrix.size}")