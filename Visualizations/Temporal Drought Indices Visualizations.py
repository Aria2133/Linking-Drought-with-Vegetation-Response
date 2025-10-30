import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import rasterio
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# Define base directories for training data (2001-2021)
base_dirs = {
    'Grasslands': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS",
    'NaturalForests': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS",
    'Savannas': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS",
    'Shrublands': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS"
}

# Define base directories for test data (2022-2024)
test_dirs = {
    'Grasslands': r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes\Grasslands",
    'NaturalForests': r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes\NaturalForests",
    'Savannas': r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes\Savannas",
    'Shrublands': r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes\Shrublands"
}

# Define colors for each land type
colors = {
    'Grasslands': '#1f77b4',
    'NaturalForests': '#ff7f0e',
    'Savannas': '#2ca02c',
    'Shrublands': '#d62728'
}

# Define indices with their units
indices_with_units = {
    'GNDVI': 'GNDVI',
    'GOSIF': 'GOSIF (W m⁻² μm⁻¹ sr⁻¹)',
    'GPP': 'GPP (µmol CO₂ m⁻² s⁻¹)',
    'LST': 'LST (°C)',
    'NDVI': 'NDVI',
    'NIRv': 'NIRv',
    'SPEI': 'SPEI',
    'TWS': 'TWS (cm)',
    'VPD': 'VPDmax (kPa)'
}

# Define which indices go in which figure
vegetation_indices = ['NDVI', 'NIRv', 'GNDVI', 'GOSIF', 'GPP']
environmental_indices = ['VPD', 'SPEI', 'TWS', 'LST']

indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

# Define naming patterns for training data (2001-2021)
naming_patterns = {
    'Grasslands': {
        'GNDVI': 'GNDVI_Grasslands_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif',
        'GOSIF': 'GOSIF_Grasslands_CALIFORNIA_GOSIF_{year}{month:02d}.tif',
        'GPP': 'Grasslands_GPP_{year}_{month:02d}.tif',
        'LST': 'LST_Grasslands_MYD21C1_CA_LST_{year}_{month:02d}.tif',
        'NDVI': 'NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'NIRv': 'NIRv_Grasslands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Grasslands.tif',
        'TWS': 'TWS_Grasslands_TWS_GRACE_CA_{year}_{month:02d}.tif',
        'VPD': 'VPD_Grasslands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif'
    },
    'NaturalForests': {
        'GNDVI': 'GNDVI_NaturalForests_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif',
        'GOSIF': 'GOSIF_NaturalForests_CALIFORNIA_GOSIF_{year}{month:02d}.tif',
        'GPP': 'NaturalForests_GPP_{year}_{month:02d}.tif',
        'LST': 'LST_NaturalForests_MYD21C1_CA_LST_{year}_{month:02d}.tif',
        'NDVI': 'NDVI_NaturalForests_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'NIRv': 'NIRv_NaturalForests_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_NaturalForests.tif',
        'TWS': 'TWS_NaturalForests_TWS_GRACE_CA_{year}_{month:02d}.tif',
        'VPD': 'VPD_NaturalForests_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif'
    },
    'Savannas': {
        'GNDVI': 'GNDVI_Savannas_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif',
        'GOSIF': 'GOSIF_Savanna_CALIFORNIA_GOSIF_{year}{month:02d}.tif',
        'GPP': 'Savannas_GPP_{year}_{month:02d}.tif',
        'LST': 'LST_Savanna_MYD21C1_CA_LST_{year}_{month:02d}.tif',
        'NDVI': 'NDVI_Savanna_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'NIRv': 'NIRv_Savanna_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Savannas.tif',
        'TWS': 'TWS_Savanna_TWS_GRACE_CA_{year}_{month:02d}.tif',
        'VPD': 'VPD_Savanna_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif'
    },
    'Shrublands': {
        'GNDVI': 'GNDVI_Shrublands_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif',
        'GOSIF': 'GOSIF_Shrublands_CALIFORNIA_GOSIF_{year}{month:02d}.tif',
        'GPP': 'Shrublands_GPP_{year}_{month:02d}.tif',
        'LST': 'LST_Shrublands_MYD21C1_CA_LST_{year}_{month:02d}.tif',
        'NDVI': 'NDVI_Shrublands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'NIRv': 'NIRv_Shrublands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Shrublands.tif',
        'TWS': 'TWS_Shrublands_TWS_GRACE_CA_{year}_{month:02d}.tif',
        'VPD': 'VPD_Shrublands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif'
    }
}

# Define naming patterns for test data (2022-2024)
test_naming_patterns = {
    'Grasslands': {
        'GNDVI': 'GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_Grasslands.tif',
        'GOSIF': 'GOSIF_{year}.M{month:02d}_Grasslands.tif',
        'GPP': 'GPP_MOD17A2H_CA_5km_{year}_{month:02d}_Grasslands.tif',
        'LST': 'MYD21C1_CA_LST_{year}_{month:02d}_Grasslands.tif',
        'NDVI': 'NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_Grasslands.tif',
        'NIRv': 'NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_Grasslands.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Grasslands.tif',
        'TWS': 'TWS_GRACE_CA_{year}_{month:02d}_Grasslands.tif',
        'VPD': 'VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_Grasslands.tif'
    },
    'NaturalForests': {
        'GNDVI': 'GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_NaturalForests.tif',
        'GOSIF': 'GOSIF_{year}.M{month:02d}_NaturalForests.tif',
        'GPP': 'GPP_MOD17A2H_CA_5km_{year}_{month:02d}_NaturalForests.tif',
        'LST': 'MYD21C1_CA_LST_{year}_{month:02d}_NaturalForests.tif',
        'NDVI': 'NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_NaturalForests.tif',
        'NIRv': 'NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_NaturalForests.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_NaturalForests.tif',
        'TWS': 'TWS_GRACE_CA_{year}_{month:02d}_NaturalForests.tif',
        'VPD': 'VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_NaturalForests.tif'
    },
    'Savannas': {
        'GNDVI': 'GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_Savannas.tif',
        'GOSIF': 'GOSIF_{year}.M{month:02d}_Savannas.tif',
        'GPP': 'GPP_MOD17A2H_CA_5km_{year}_{month:02d}_Savannas.tif',
        'LST': 'MYD21C1_CA_LST_{year}_{month:02d}_Savannas.tif',
        'NDVI': 'NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_Savannas.tif',
        'NIRv': 'NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_Savannas.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Savannas.tif',
        'TWS': 'TWS_GRACE_CA_{year}_{month:02d}_Savannas.tif',
        'VPD': 'VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_Savannas.tif'
    },
    'Shrublands': {
        'GNDVI': 'GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_Shrublands.tif',
        'GOSIF': 'GOSIF_{year}.M{month:02d}_Shrublands.tif',
        'GPP': 'GPP_MOD17A2H_CA_5km_{year}_{month:02d}_Shrublands.tif',
        'LST': 'MYD21C1_CA_LST_{year}_{month:02d}_Shrublands.tif',
        'NDVI': 'NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_Shrublands.tif',
        'NIRv': 'NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_Shrublands.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Shrublands.tif',
        'TWS': 'TWS_GRACE_CA_{year}_{month:02d}_Shrublands.tif',
        'VPD': 'VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_Shrublands.tif'
    }
}


def read_raster_mean(filepath):
    """Read a raster file and return the mean value."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)
            mean_val = np.nanmean(data)
            return mean_val
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None


def fill_gaps_hierarchical(dates, values, target_start_year=2001, target_end_year=2024):
    """Fill gaps using three-step hierarchical method."""
    if len(dates) == 0:
        return dates, values

    full_date_range = pd.date_range(
        start=pd.Timestamp(year=target_start_year, month=1, day=15),
        end=pd.Timestamp(year=target_end_year, month=12, day=15),
        freq='MS'
    )

    df = pd.DataFrame({'date': dates, 'value': values})
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    full_df = pd.DataFrame({'date': full_date_range})
    full_df['year'] = full_df['date'].dt.year
    full_df['month'] = full_df['date'].dt.month
    full_df = full_df[full_df['year'] <= target_end_year]
    full_df = full_df.merge(df, on=['date', 'year', 'month'], how='left')

    # Step 1: Fill using average of proximate years
    for idx, row in full_df[full_df['value'].isna()].iterrows():
        year = row['year']
        month = row['month']

        prev_val = df[(df['year'] == year - 1) & (df['month'] == month)]
        next_val = df[(df['year'] == year + 1) & (df['month'] == month)]

        proximate_vals = []
        if not prev_val.empty:
            proximate_vals.append(prev_val['value'].values[0])
        if not next_val.empty:
            proximate_vals.append(next_val['value'].values[0])

        if len(proximate_vals) > 0:
            full_df.loc[idx, 'value'] = np.mean(proximate_vals)

    # Step 2: Fill using polynomial trend
    still_missing = full_df['value'].isna()
    if still_missing.any() and not full_df['value'].isna().all():
        for month in range(1, 13):
            month_data = full_df[full_df['month'] == month].copy()

            if month_data['value'].isna().any() and not month_data['value'].isna().all():
                train_data = month_data[~month_data['value'].isna()]

                if len(train_data) >= 3:
                    train_data.loc[:, 'time_idx'] = train_data['year'] + (train_data['month'] - 1) / 12
                    X_train = train_data['time_idx'].values.reshape(-1, 1)
                    y_train = train_data['value'].values

                    poly = PolynomialFeatures(degree=2)
                    X_poly = poly.fit_transform(X_train)

                    model = LinearRegression()
                    model.fit(X_poly, y_train)

                    test_data = month_data[month_data['value'].isna()].copy()
                    if len(test_data) > 0:
                        test_data.loc[:, 'time_idx'] = test_data['year'] + (test_data['month'] - 1) / 12
                        X_test = test_data['time_idx'].values.reshape(-1, 1)
                        X_test_poly = poly.transform(X_test)
                        predictions = model.predict(X_test_poly)

                        full_df.loc[test_data.index, 'value'] = predictions

    # Step 3: Fill with climatological mean
    still_missing = full_df['value'].isna()
    if still_missing.any():
        for month in range(1, 13):
            month_mean = full_df[full_df['month'] == month]['value'].mean()
            if not np.isnan(month_mean):
                full_df.loc[(full_df['month'] == month) & (full_df['value'].isna()), 'value'] = month_mean

    full_df = full_df[full_df['year'] <= target_end_year]

    return full_df['date'].values, full_df['value'].values


def load_time_series(land_type, index, start_year=2001, end_year=2024):
    """Load time series data for a specific land type and index."""
    time_series = []
    dates = []

    # Start years for different indices
    actual_data_start_years = {
        'LST': 2003,
        'TWS': 2002
    }

    actual_start_year = actual_data_start_years.get(index, start_year)

    files_found = 0
    test_files_found = 0

    # Load training data (2001-2021)
    base_dir = base_dirs[land_type]
    index_dir = os.path.join(base_dir, index)

    if os.path.exists(index_dir):
        for year in range(actual_start_year, min(2022, end_year + 1)):
            for month in range(1, 13):
                base_pattern = naming_patterns[land_type][index].format(year=year, month=month)

                for ext in ['.tif', '.tiff', '.TIF', '.TIFF', '']:
                    if ext:
                        filepath = os.path.join(index_dir, base_pattern + ext)
                    else:
                        filepath = os.path.join(index_dir, base_pattern)

                    if os.path.exists(filepath):
                        mean_val = read_raster_mean(filepath)
                        if mean_val is not None and not np.isnan(mean_val):
                            time_series.append(mean_val)
                            dates.append(pd.Timestamp(year=year, month=month, day=15))
                            files_found += 1
                        break

    # Load test data (2022-2024) - SKIP FOR GPP
    if index != 'GPP':
        test_base_dir = test_dirs[land_type]
        test_index_dir = os.path.join(test_base_dir, index)

        if os.path.exists(test_index_dir):
            for year in range(2022, end_year + 1):
                for month in range(1, 13):
                    base_pattern = test_naming_patterns[land_type][index].format(year=year, month=month)

                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF', '']:
                        if ext:
                            filepath = os.path.join(test_index_dir, base_pattern + ext)
                        else:
                            filepath = os.path.join(test_index_dir, base_pattern)

                        if os.path.exists(filepath):
                            mean_val = read_raster_mean(filepath)
                            if mean_val is not None and not np.isnan(mean_val):
                                time_series.append(mean_val)
                                dates.append(pd.Timestamp(year=year, month=month, day=15))
                                test_files_found += 1
                            break

    total_files = files_found + test_files_found

    if total_files == 0:
        print(f"  WARNING: No files found for {land_type} - {index}")
        return np.array(dates), np.array(time_series)
    else:
        print(f"  Found {total_files} files for {land_type} - {index} (train: {files_found}, test: {test_files_found})")

        # Apply gap-filling for LST and TWS
        if index in ['LST', 'TWS'] and len(dates) > 0:
            dates_array = np.array(dates)
            values_array = np.array(time_series)

            dates_filled, values_filled = fill_gaps_hierarchical(
                dates_array, values_array,
                target_start_year=start_year,
                target_end_year=end_year
            )

            min_actual_year = min([pd.Timestamp(d).year for d in dates])
            filled_years = f"{start_year}-{min_actual_year - 1}" if min_actual_year > start_year else "none"
            print(f"  Gap-filled early years: {filled_years}")
            return dates_filled, values_filled

    return np.array(dates), np.array(time_series)


def calculate_residuals(dates, values, baseline_start=2001, baseline_end=2011):
    """
    Calculate deseasonalized residuals (anomalies) in ORIGINAL UNITS.
    Simply subtracts monthly climatological mean - NO standardization!

    This preserves physical meaning:
    - NDVI residual in dimensionless units (e.g., -0.15)
    - VPD residual in kPa (e.g., +1.2 kPa)
    - TWS residual in cm (e.g., -35 cm)
    """
    if len(dates) == 0:
        return dates, values

    # Convert to DataFrame
    df = pd.DataFrame({'date': dates, 'value': values})
    df['year'] = pd.to_datetime(df['date']).dt.year
    df['month'] = pd.to_datetime(df['date']).dt.month

    # Calculate monthly climatology (baseline period mean for each month)
    baseline_mask = (df['year'] >= baseline_start) & (df['year'] <= baseline_end)
    monthly_climatology = df[baseline_mask].groupby('month')['value'].mean()

    # Calculate residuals (simply subtract the climatological mean)
    residuals = []
    for idx, row in df.iterrows():
        month = row['month']
        climatology_value = monthly_climatology[month]
        residual = row['value'] - climatology_value  # NO division by SD!
        residuals.append(residual)

    return dates, np.array(residuals)


def moving_average(values, window=3):
    """Calculate moving average with specified window."""
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window=window, center=True, min_periods=1).mean().values


def plot_figure(indices_list, figure_title, filename, fig_num):
    """Create a figure with deseasonalized residuals in original units."""

    # Define appropriate y-axis ranges for each index (in original units!)
    y_ranges = {
        'NDVI': (-0.20, 0.20),
        'NIRv': (-0.06, 0.06),
        'GNDVI': (-0.20, 0.20),
        'GOSIF': (-600, 600),  # W m⁻² μm⁻¹ sr⁻¹
        'GPP': (-2.5, 2.5),  # µmol CO₂ m⁻² s⁻¹
        'VPD': (-1.2, 1.2),  # kPa
        'SPEI': (-2.5, 2.5),  # dimensionless (already z-scores, so residuals are similar)
        'TWS': (-35, 35),  # cm
        'LST': (-8, 8)  # °C
    }

    # Y-axis labels with proper units
    y_labels = {
        'NDVI': 'NDVI Anomaly',
        'NIRv': 'NIRv Anomaly',
        'GNDVI': 'GNDVI Anomaly',
        'GOSIF': 'SIF Anomaly\n(W m⁻² μm⁻¹ sr⁻¹)',
        'GPP': 'GPP Anomaly\n(µmol CO₂ m⁻² s⁻¹)',
        'VPD': 'VPDmax Anomaly (kPa)',
        'SPEI': 'SPEI-1 Anomaly',
        'TWS': 'TWS Anomaly (cm)',
        'LST': 'LST Anomaly (°C)'
    }

    # Create figure
    n_panels = len(indices_list)
    fig, axes = plt.subplots(n_panels, 1, figsize=(15, 3.5 * n_panels))

    if n_panels == 1:
        axes = [axes]

    fig.suptitle(figure_title, fontsize=16, fontweight='bold', y=0.995)

    # Create legend
    legend_elements = [
        Line2D([0], [0], color=colors['Grasslands'], lw=2, label='Grasslands'),
        Line2D([0], [0], color=colors['NaturalForests'], lw=2, label='Natural Forests'),
        Line2D([0], [0], color=colors['Savannas'], lw=2, label='Savannas'),
        Line2D([0], [0], color=colors['Shrublands'], lw=2, label='Shrublands'),
        Line2D([0], [0], color='gray', lw=1, alpha=0.3, label='Raw data'),
        Line2D([0], [0], color='black', lw=2, label='3-month MA')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=6,
               bbox_to_anchor=(0.5, 0.985), framealpha=0.9, fontsize=10)

    # Plot each index
    for idx, (ax, index) in enumerate(zip(axes, indices_list)):
        print(f"\nProcessing {index}...")

        has_data = False
        for land_type in ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']:
            dates, values = load_time_series(land_type, index)

            if len(dates) > 0:
                has_data = True

                # Convert dates to numeric
                if isinstance(dates[0], pd.Timestamp):
                    x = np.array([d.year + (d.month - 1) / 12 for d in dates])
                else:
                    dates_pd = pd.to_datetime(dates)
                    x = np.array([d.year + (d.month - 1) / 12 for d in dates_pd])

                # For GPP, only show data until 2021
                if index == 'GPP':
                    mask = x <= 2021 + 11 / 12
                    x = x[mask]
                    values = values[mask]
                    dates = dates[mask]

                # Calculate RESIDUALS (not z-scores!)
                dates_resid, residuals = calculate_residuals(dates, values)

                # Recalculate x after residual calculation
                if isinstance(dates_resid[0], pd.Timestamp):
                    x = np.array([d.year + (d.month - 1) / 12 for d in dates_resid])
                else:
                    dates_pd = pd.to_datetime(dates_resid)
                    x = np.array([d.year + (d.month - 1) / 12 for d in dates_pd])

                # Ensure data doesn't go beyond 2024
                mask = x <= 2024 + 11 / 12
                x = x[mask]
                residuals = residuals[mask]

                # Plot raw residuals (thin, transparent)
                ax.plot(x, residuals, color=colors[land_type], linewidth=1, alpha=0.3)

                # Calculate and plot 3-month moving average
                if len(residuals) >= 3:
                    residuals_ma = moving_average(residuals, window=3)
                    ax.plot(x, residuals_ma, color=colors[land_type], linewidth=2.5, alpha=0.9)

        # Add drought period shading
        ax.axvspan(2012, 2016, alpha=0.15, color='red', label='2012-2016 Drought')
        ax.axvspan(2020, 2024, alpha=0.15, color='orange', label='2020-2024 Drought')

        # Add horizontal reference line at zero (climatological normal)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.7)

        # Set labels
        ax.set_title(f'{index} - Deseasonalized Anomaly',
                     fontsize=12, fontweight='bold', loc='left')
        ax.set_ylabel(y_labels[index], fontsize=10, fontweight='bold')
        ax.set_xlabel('Year' if idx == n_panels - 1 else '')
        ax.grid(True, alpha=0.3, linestyle='--')

        if not has_data:
            ax.text(0.5, 0.5, 'No data found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')

        # Set x-axis
        ax.set_xlim(2001, 2024)
        ax.set_xticks([2005, 2010, 2015, 2020, 2024])
        ax.set_xticklabels(['2005', '2010', '2015', '2020', '2024'])

        # Set y-axis limits based on index (in original units!)
        ax.set_ylim(y_ranges[index])

        # Add annotation for severe values (optional)
        y_min, y_max = y_ranges[index]
        severe_threshold = y_min * 0.7  # 70% of range
        if index in ['TWS', 'NDVI', 'NIRv', 'GNDVI', 'GOSIF', 'GPP']:
            ax.axhline(y=severe_threshold, color='red', linestyle=':', linewidth=0.8,
                       alpha=0.4, label=f'Severe ({severe_threshold:.2f})')

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.975])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nFigure {fig_num} created successfully!")
    print(f"Output saved as: {filename}")


# Create Figure 3: Vegetation and Productivity Response
print("\n" + "=" * 80)
print("CREATING FIGURE 3: VEGETATION AND PRODUCTIVITY RESPONSE (RESIDUALS)")
print("=" * 80)
plot_figure(
    vegetation_indices,
    'Figure 3: Vegetation and Productivity Response - Deseasonalized Anomalies (2001-2024)',
    'figure3_vegetation_productivity_residuals.png',
    3
)

# Create Figure 4: Meteorological and Hydrological Forcing
print("\n" + "=" * 80)
print("CREATING FIGURE 4: METEOROLOGICAL AND HYDROLOGICAL FORCING (RESIDUALS)")
print("=" * 80)
plot_figure(
    environmental_indices,
    'Figure 4: Meteorological and Hydrological Forcing - Deseasonalized Anomalies (2001-2024)',
    'figure4_meteorological_hydrological_residuals.png',
    4
)

print("\n" + "=" * 80)
print("ALL FIGURES CREATED SUCCESSFULLY!")
print("=" * 80)
print("\nNOTE: All anomalies shown in ORIGINAL UNITS:")
print("  - NDVI, NIRv, GNDVI: dimensionless")
print("  - GOSIF: W m⁻² μm⁻¹ sr⁻¹")
print("  - GPP: µmol CO₂ m⁻² s⁻¹")
print("  - VPDmax: kPa")
print("  - SPEI: dimensionless")
print("  - TWS: cm")
print("  - LST: °C")
print("\nResiduals represent deviations from monthly climatology (2001-2011).")
print("Seasonal cycles removed; physical units preserved.")
print("=" * 80)