import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import warnings

warnings.filterwarnings('ignore')

print("Starting California Land Type Analysis...")
print("Analyzing period: 2001-2024 (GPP: 2001-2021 only)")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

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

# Land type mapping
LAND_TYPE_MAP = {
    'GRASSLANDS': 'Grasslands',
    'NATURAL FORESTS': 'NaturalForests',
    'SAVANNAS': 'Savannas',
    'SHRUBLANDS': 'Shrublands'
}

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
        'LST': 'MYD21C1_CA_LST_{year}_{month:02d}_Shrublands.tif',
        'NDVI': 'NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_Shrublands.tif',
        'NIRv': 'NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_Shrublands.tif',
        'SPEI': 'SPEI1_GRIDMET_CA_{year}_{month:02d}_Shrublands.tif',
        'TWS': 'TWS_GRACE_CA_{year}_{month:02d}_Shrublands.tif',
        'VPD': 'VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_Shrublands.tif'
    }
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def read_raster_mean(filepath):
    """Read a raster file and return the mean value."""
    try:
        with rasterio.open(filepath) as src:
            data = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                data = np.ma.masked_equal(data, nodata)
            mean_val = np.nanmean(data)
            return mean_val if not np.isnan(mean_val) else None
    except Exception as e:
        return None


def load_variable_data(variable, land_type_key, start_year=2001, end_year=2024):
    """Load data for a specific variable and land type."""
    land_type = LAND_TYPE_MAP[land_type_key]
    data = []

    # Start years for different indices
    actual_data_start_years = {
        'LST': 2003,
        'TWS': 2002
    }
    actual_start_year = actual_data_start_years.get(variable, start_year)

    files_found = 0
    test_files_found = 0

    # Load training data (2001-2021)
    base_dir = base_dirs[land_type]
    index_dir = os.path.join(base_dir, variable)

    if os.path.exists(index_dir):
        for year in range(actual_start_year, min(2022, end_year + 1)):
            for month in range(1, 13):
                base_pattern = naming_patterns[land_type][variable].format(year=year, month=month)

                # Try different file extensions
                for ext in ['.tif', '.tiff', '.TIF', '.TIFF', '']:
                    if ext:
                        filepath = os.path.join(index_dir, base_pattern + ext)
                    else:
                        filepath = os.path.join(index_dir, base_pattern)

                    if os.path.exists(filepath):
                        mean_val = read_raster_mean(filepath)
                        if mean_val is not None:
                            data.append(mean_val)
                            files_found += 1
                        break

    # Load test data (2022-2024) - SKIP FOR GPP
    if variable != 'GPP':
        test_base_dir = test_dirs[land_type]
        test_index_dir = os.path.join(test_base_dir, variable)

        if os.path.exists(test_index_dir):
            for year in range(2022, end_year + 1):
                for month in range(1, 13):
                    base_pattern = test_naming_patterns[land_type][variable].format(year=year, month=month)

                    # Try different file extensions
                    for ext in ['.tif', '.tiff', '.TIF', '.TIFF', '']:
                        if ext:
                            filepath = os.path.join(test_index_dir, base_pattern + ext)
                        else:
                            filepath = os.path.join(test_index_dir, base_pattern)

                        if os.path.exists(filepath):
                            mean_val = read_raster_mean(filepath)
                            if mean_val is not None:
                                data.append(mean_val)
                                test_files_found += 1
                            break

    total_files = files_found + test_files_found
    print(f"  Loaded {variable}: {total_files} data points (train: {files_found}, test: {test_files_found})")

    return np.array(data)


def load_all_data(land_type_key):
    """Load all variable data for a land type."""
    all_data = {}
    variables = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

    for var in variables:
        data = load_variable_data(var, land_type_key)
        if len(data) > 0:
            all_data[var] = data
        else:
            print(f"  {var}: No valid data found")

    return all_data


def calculate_correlation_matrix(data_dict):
    """Calculate correlation matrix from data dictionary."""
    # Get variables that have data
    valid_vars = [k for k, v in data_dict.items() if len(v) > 0]

    if len(valid_vars) < 2:
        print("  Error: Need at least 2 variables with data")
        return None, []

    # Find minimum length to align data
    min_length = min(len(data_dict[var]) for var in valid_vars)
    print(f"  Minimum data length: {min_length}")

    # Create aligned data matrix
    data_matrix = np.array([data_dict[var][:min_length] for var in valid_vars]).T

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(data_matrix.T)

    return corr_matrix, valid_vars


def create_combined_correlation_heatmap(all_results, output_path):
    """Create combined correlation heatmap with 4 subplots."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    axes = axes.flatten()

    land_types = ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']

    for idx, (ax, land_type) in enumerate(zip(axes, land_types)):
        if land_type in all_results:
            corr_matrix = all_results[land_type]['matrix']
            variables = all_results[land_type]['variables']

            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

            # Create heatmap with reversed magma colormap
            sns.heatmap(corr_matrix,
                        mask=mask,
                        annot=True,
                        fmt='.2f',
                        cmap='magma_r',
                        vmin=-1,
                        vmax=1,
                        center=0,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": 0.8},
                        xticklabels=variables,
                        yticklabels=variables,
                        annot_kws={'size': 8},
                        ax=ax)

            ax.set_title(f'{land_type}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xlabel('Variables', fontsize=11, fontweight='bold')
            ax.set_ylabel('Variables', fontsize=11, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=9)
            ax.tick_params(axis='y', rotation=0, labelsize=9)
        else:
            ax.text(0.5, 0.5, f'No data for {land_type}',
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_xticks([])
            ax.set_yticks([])

    # Add main title
    fig.suptitle('Inter-Index Correlation Matrix (2001-2024)\nGPP: 2001-2021 only',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved combined correlation heatmap: {output_path}")


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

print("\nCalculating inter-index correlations...")

all_results = {}

for land_type_key in ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']:
    print(f"\nProcessing correlations for {land_type_key}...")

    # Load all data
    data = load_all_data(land_type_key)

    # Calculate correlation matrix
    corr_matrix, variables = calculate_correlation_matrix(data)

    if corr_matrix is not None:
        # Store results
        all_results[land_type_key] = {
            'matrix': corr_matrix,
            'variables': variables,
            'mean_correlation': np.mean(np.abs(corr_matrix[~np.eye(len(variables), dtype=bool)])),
            'matrix_shape': corr_matrix.shape
        }

        print(f"  Correlation matrix shape: {corr_matrix.shape}")
        print(f"  Mean absolute correlation: {all_results[land_type_key]['mean_correlation']:.3f}")
    else:
        print(f"  Could not create correlation matrix for {land_type_key}")

# Create combined heatmap with all land types
if len(all_results) > 0:
    output_file = "combined_correlation_heatmap_2001_2024.png"
    create_combined_correlation_heatmap(all_results, output_file)

    print("\n" + "=" * 80)
    print("Analysis complete! Combined correlation heatmap saved.")
    print("\nSummary of results:\n")

    for land_type, results in all_results.items():
        print(f"{land_type}:")
        print(f"  Variables analyzed: {results['variables']}")
        print(f"  Mean absolute correlation: {results['mean_correlation']:.3f}")
        print()
else:
    print("\n" + "=" * 80)
    print("ERROR: No correlation matrices could be calculated!")

print("Note: GPP data spans 2001-2021, all other variables span 2001-2024")