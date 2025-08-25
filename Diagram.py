import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from scipy import stats
from scipy.ndimage import gaussian_filter
import os
import glob
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class CaliforniaDroughtAnalyzer:
    def __init__(self, base_path):
        """
        Initialize the California Drought Analyzer

        Parameters:
        base_path (str): Base path to the data directory
        """
        self.base_path = base_path
        self.land_types = ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']
        self.indicators = ['GOSIF', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPDmax']
        self.data = {}
        self.residuals = {}
        self.trends = {}
        self.seasonal = {}

        # Define data paths and naming patterns - FIXED PATHS
        self.data_paths = {
            'GOSIF': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\GOSIF',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\GOSIF',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\GOSIF',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\GOSIF'  # Fixed: Added GOSIF subfolder
            },
            'LST': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\LST',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\LST',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\LST',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\LST'
            },
            'NDVI': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\NDVI',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\NDVI',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\NDVI',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\NDVI'
            },
            'NIRv': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\NIRv',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\NIRv',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\NIRv',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\NIRv'
            },
            'SPEI': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\SPEI',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\SPEI',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\SPEI',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\SPEI'
            },
            'TWS': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\TWS',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\TWS',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\TWS',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\TWS'
            },
            'VPDmax': {
                'GRASSLANDS': 'CALIFORNIA GRASSLANDS\\VPD',
                'NATURAL FORESTS': 'CALIFORNIA NATURAL FORESTS\\VPD',
                'SAVANNAS': 'CALIFORNIA SAVANNAS\\VPD',
                'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS\\VPD'
            }
        }

        # Define file naming patterns
        self.naming_patterns = {
            'GOSIF': {
                'GRASSLANDS': 'GOSIF_Grasslands_CALIFORNIA_GOSIF_{year}{month:02d}',
                'NATURAL FORESTS': 'GOSIF_NaturalForests_CALIFORNIA_GOSIF_{year}{month:02d}',
                'SAVANNAS': 'GOSIF_Savanna_CALIFORNIA_GOSIF_{year}{month:02d}',
                'SHRUBLANDS': 'GOSIF_Shrublands_CALIFORNIA_GOSIF_{year}{month:02d}'
            },
            'LST': {
                'GRASSLANDS': 'LST_Grasslands_MYD21C1_CA_LST_{year}_{month:02d}',
                'NATURAL FORESTS': 'LST_NaturalForests_MYD21C1_CA_LST_{year}_{month:02d}',
                'SAVANNAS': 'LST_Savanna_MYD21C1_CA_LST_{year}_{month:02d}',
                'SHRUBLANDS': 'LST_Shrublands_MYD21C1_CA_LST_{year}_{month:02d}'
            },
            'NDVI': {
                'GRASSLANDS': 'NDVI_Grasslands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'NATURAL FORESTS': 'NDVI_NaturalForests_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SAVANNAS': 'NDVI_Savanna_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SHRUBLANDS': 'NDVI_Shrublands_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}'
            },
            'NIRv': {
                'GRASSLANDS': 'NIRv_Grasslands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'NATURAL FORESTS': 'NIRv_NaturalForests_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SAVANNAS': 'NIRv_Savanna_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}',
                'SHRUBLANDS': 'NIRv_Shrublands_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}'
            },
            'SPEI': {
                'GRASSLANDS': 'SPEI_Grasslands_SPEI1_CSIC_CA_{year}_{month:02d}',
                'NATURAL FORESTS': 'SPEI_NaturalForests_SPEI1_CSIC_CA_{year}_{month:02d}',
                'SAVANNAS': 'SPEI_Savanna_SPEI1_CSIC_CA_{year}_{month:02d}',
                'SHRUBLANDS': 'SPEI_Shrublands_SPEI1_CSIC_CA_{year}_{month:02d}'
            },
            'TWS': {
                'GRASSLANDS': 'TWS_Grasslands_TWS_GRACE_CA_{year}_{month:02d}',
                'NATURAL FORESTS': 'TWS_NaturalForests_TWS_GRACE_CA_{year}_{month:02d}',
                'SAVANNAS': 'TWS_Savanna_TWS_GRACE_CA_{year}_{month:02d}',
                'SHRUBLANDS': 'TWS_Shrublands_TWS_GRACE_CA_{year}_{month:02d}'
            },
            'VPDmax': {
                'GRASSLANDS': 'VPD_Grasslands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}',
                'NATURAL FORESTS': 'VPD_NaturalForests_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}',
                'SAVANNAS': 'VPD_Savanna_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}',
                'SHRUBLANDS': 'VPD_Shrublands_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}'
            }
        }

    def load_raster_data(self, file_path):
        """Load and process raster data from TIFF files"""
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                # Handle different nodata values
                if src.nodata is not None:
                    data = np.where(data == src.nodata, np.nan, data)
                # Also handle extreme values that might be nodata
                data = np.where(np.abs(data) > 1e10, np.nan, data)
                return data, src.transform, src.crs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None, None

    def check_data_paths(self):
        """Check and report which data paths exist"""
        print("Checking data paths...")
        missing_paths = []

        for indicator in self.indicators:
            for land_type in self.land_types:
                folder_path = os.path.join(self.base_path, self.data_paths[indicator][land_type])
                if not os.path.exists(folder_path):
                    missing_paths.append(f"{indicator} - {land_type}: {folder_path}")
                    print(f"❌ Missing: {folder_path}")
                else:
                    # Count files in the directory
                    files = glob.glob(os.path.join(folder_path, "*.tif"))
                    print(f"✅ Found: {folder_path} ({len(files)} .tif files)")

        if missing_paths:
            print(f"\n⚠️  Found {len(missing_paths)} missing paths. Analysis will continue with available data.")
        return missing_paths

    def fill_missing_data(self, time_series, dates):
        """
        Fill missing data using the three-step method described in the document:
        1. Average of same month in adjacent years
        2. GAM trend estimation for the same month
        3. Overall monthly average
        """
        filled_series = time_series.copy()

        for i, (date, value) in enumerate(zip(dates, time_series)):
            if np.isnan(value):
                year = date.year
                month = date.month

                # Step 1: Average of same month in adjacent years
                prev_year_idx = None
                next_year_idx = None

                for j, d in enumerate(dates):
                    if d.year == year - 1 and d.month == month:
                        prev_year_idx = j
                    elif d.year == year + 1 and d.month == month:
                        next_year_idx = j

                if (prev_year_idx is not None and next_year_idx is not None and
                        not np.isnan(time_series[prev_year_idx]) and
                        not np.isnan(time_series[next_year_idx])):
                    filled_series[i] = (time_series[prev_year_idx] + time_series[next_year_idx]) / 2
                    continue

                # Step 2: Monthly average for that specific month across all years
                same_month_values = []
                for j, d in enumerate(dates):
                    if d.month == month and not np.isnan(time_series[j]):
                        same_month_values.append(time_series[j])

                if same_month_values:
                    filled_series[i] = np.mean(same_month_values)
                else:
                    # Step 3: Overall mean as last resort
                    valid_values = time_series[~np.isnan(time_series)]
                    if len(valid_values) > 0:
                        filled_series[i] = np.mean(valid_values)

        return filled_series

    def load_all_data(self):
        """Load all datasets for all indicators and land types"""
        print("Loading all drought indicator datasets...")

        # First check which paths exist
        self.check_data_paths()

        for indicator in self.indicators:
            self.data[indicator] = {}
            for land_type in self.land_types:
                print(f"Loading {indicator} data for {land_type}...")

                # Construct full path
                folder_path = os.path.join(self.base_path, self.data_paths[indicator][land_type])

                if not os.path.exists(folder_path):
                    print(f"⚠️  Warning: Path does not exist: {folder_path}")
                    continue

                # Get all TIFF files and sort them
                pattern = self.naming_patterns[indicator][land_type]
                files = []
                dates = []

                # Generate expected file names for 2001-2021
                start_year = 2002 if indicator == 'TWS' else 2003 if indicator == 'LST' else 2001
                end_year = 2021

                for year in range(start_year, end_year + 1):
                    for month in range(1, 13):
                        if indicator == 'TWS' and year == 2002 and month < 4:
                            continue  # TWS starts from April 2002

                        filename = pattern.format(year=year, month=month) + '.tif'
                        file_path = os.path.join(folder_path, filename)

                        if os.path.exists(file_path):
                            files.append(file_path)
                            dates.append(datetime(year, month, 1))

                if not files:
                    print(f"❌ No files found for {indicator} - {land_type}")
                    continue

                # Load the first file to get spatial dimensions
                first_data, transform, crs = self.load_raster_data(files[0])
                if first_data is None:
                    continue

                # Initialize arrays
                time_series_stack = np.full((len(files), *first_data.shape), np.nan)

                # Load all files
                loaded_count = 0
                for i, file_path in enumerate(files):
                    data, _, _ = self.load_raster_data(file_path)
                    if data is not None:
                        time_series_stack[i] = data
                        loaded_count += 1

                # Store the data
                self.data[indicator][land_type] = {
                    'data': time_series_stack,
                    'dates': dates,
                    'transform': transform,
                    'crs': crs,
                    'shape': first_data.shape
                }

                print(f"✅ Loaded {loaded_count}/{len(files)} files for {indicator} - {land_type}")

    def perform_stl_decomposition(self):
        """Perform STL decomposition on all datasets"""
        print("Performing STL decomposition...")

        for indicator in self.indicators:
            if indicator not in self.data:
                continue

            self.residuals[indicator] = {}
            self.trends[indicator] = {}
            self.seasonal[indicator] = {}

            for land_type in self.land_types:
                if land_type not in self.data[indicator]:
                    continue

                print(f"Decomposing {indicator} - {land_type}")

                data_stack = self.data[indicator][land_type]['data']
                dates = self.data[indicator][land_type]['dates']

                # Initialize output arrays
                residuals_stack = np.full_like(data_stack, np.nan)
                trends_stack = np.full_like(data_stack, np.nan)
                seasonal_stack = np.full_like(data_stack, np.nan)

                # Process each pixel (sample a subset for testing)
                rows, cols = data_stack.shape[1], data_stack.shape[2]

                # For faster processing, sample every 5th pixel
                step_size = 5
                processed_pixels = 0
                total_pixels = (rows // step_size) * (cols // step_size)

                for i in range(0, rows, step_size):
                    for j in range(0, cols, step_size):
                        pixel_series = data_stack[:, i, j]

                        # Skip if too many missing values
                        valid_mask = ~np.isnan(pixel_series)
                        if np.sum(valid_mask) < len(pixel_series) * 0.5:  # Lowered threshold
                            continue

                        # Fill missing data
                        filled_series = self.fill_missing_data(pixel_series, dates)

                        if np.sum(~np.isnan(filled_series)) < 24:  # Need at least 2 years
                            continue

                        try:
                            # Create pandas series for STL
                            date_index = pd.DatetimeIndex(dates)
                            ts = pd.Series(filled_series, index=date_index)
                            ts = ts.dropna()

                            if len(ts) < 24:
                                continue

                            # Perform STL decomposition with adjusted parameters
                            stl = STL(ts, seasonal=13, period=12, robust=True)
                            decomposition = stl.fit()

                            # Store results back to original time indices
                            for k, date in enumerate(dates):
                                if date in decomposition.resid.index:
                                    # Fill neighboring pixels with the same values for visualization
                                    for di in range(step_size):
                                        for dj in range(step_size):
                                            if i + di < rows and j + dj < cols:
                                                residuals_stack[k, i + di, j + dj] = decomposition.resid[date]
                                                trends_stack[k, i + di, j + dj] = decomposition.trend[date]
                                                seasonal_stack[k, i + di, j + dj] = decomposition.seasonal[date]

                            processed_pixels += 1

                        except Exception as e:
                            continue

                print(f"Processed {processed_pixels} pixels for {indicator} - {land_type}")

                # Store decomposed data
                self.residuals[indicator][land_type] = {
                    'data': residuals_stack,
                    'dates': dates,
                    'transform': self.data[indicator][land_type]['transform'],
                    'crs': self.data[indicator][land_type]['crs']
                }

                self.trends[indicator][land_type] = {
                    'data': trends_stack,
                    'dates': dates,
                    'transform': self.data[indicator][land_type]['transform'],
                    'crs': self.data[indicator][land_type]['crs']
                }

                self.seasonal[indicator][land_type] = {
                    'data': seasonal_stack,
                    'dates': dates,
                    'transform': self.data[indicator][land_type]['transform'],
                    'crs': self.data[indicator][land_type]['crs']
                }

    def create_static_trend_maps(self, output_dir='trend_maps'):
        """Create static maps showing long-term trends - FIXED VERSION"""
        print("Creating static trend maps...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for indicator in self.indicators:
            if indicator not in self.trends:
                print(f"No trend data available for {indicator}")
                continue

            # Check if we have any data for this indicator
            has_data = False
            for land_type in self.land_types:
                if land_type in self.trends[indicator]:
                    has_data = True
                    break

            if not has_data:
                print(f"No land type data available for {indicator}")
                continue

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.flatten()

            for i, land_type in enumerate(self.land_types):
                ax = axes[i]

                if land_type not in self.trends[indicator]:
                    ax.text(0.5, 0.5, 'No Data Available',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{land_type}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Calculate mean trend over all time periods
                trend_data = self.trends[indicator][land_type]['data']

                # Check if trend_data has valid values
                if trend_data.size == 0 or np.all(np.isnan(trend_data)):
                    ax.text(0.5, 0.5, 'No Valid Trend Data',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{land_type}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Calculate temporal mean trend
                mean_trend = np.nanmean(trend_data, axis=0)

                # Check if mean_trend has valid values
                if np.all(np.isnan(mean_trend)):
                    ax.text(0.5, 0.5, 'No Valid Mean Trend',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{land_type}')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue

                # Apply smoothing
                try:
                    smoothed_trend = gaussian_filter(mean_trend, sigma=1.0)

                    # Calculate robust percentiles for color scaling
                    valid_data = smoothed_trend[~np.isnan(smoothed_trend)]
                    if len(valid_data) == 0:
                        ax.text(0.5, 0.5, 'No Valid Smoothed Data',
                                horizontalalignment='center', verticalalignment='center',
                                transform=ax.transAxes, fontsize=12)
                        ax.set_title(f'{land_type}')
                        ax.set_xticks([])
                        ax.set_yticks([])
                        continue

                    # Use more robust percentiles
                    vmin, vmax = np.percentile(valid_data, [10, 90])

                    # Ensure we have a valid range
                    if vmin == vmax:
                        vmin = np.min(valid_data)
                        vmax = np.max(valid_data)
                        if vmin == vmax:  # Still the same, use symmetric range
                            vmin = -abs(vmin) if vmin != 0 else -1
                            vmax = abs(vmax) if vmax != 0 else 1

                    # Determine color scheme based on indicator
                    if indicator in ['LST', 'VPDmax']:
                        cmap = plt.cm.Reds  # Higher values = more drought stress
                    elif indicator in ['GOSIF', 'NDVI', 'NIRv', 'TWS', 'SPEI']:
                        cmap = plt.cm.RdBu  # Lower values = more drought stress
                    else:
                        cmap = plt.cm.RdBu_r  # Default

                    im = ax.imshow(smoothed_trend, cmap=cmap,
                                   vmin=vmin, vmax=vmax, aspect='auto')
                    ax.set_title(f'{land_type}', fontweight='bold')
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Add colorbar for each subplot
                    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    cbar.set_label('Trend', fontsize=8)

                except Exception as e:
                    print(f"Error processing {indicator} - {land_type}: {e}")
                    ax.text(0.5, 0.5, f'Processing Error:\n{str(e)[:50]}...',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'{land_type}')
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.suptitle(f'{indicator} Long-term Trends (2001-2021)',
                         fontsize=16, fontweight='bold')
            plt.tight_layout()

            # Save with error handling
            try:
                plt.savefig(os.path.join(output_dir, f'{indicator}_trends.png'),
                            dpi=300, bbox_inches='tight')
                print(f"✅ Saved trend map for {indicator}")
            except Exception as e:
                print(f"❌ Error saving trend map for {indicator}: {e}")

            plt.close()

    def create_simple_time_series_plot(self, output_dir='simple_plots'):
        """Create simple time series plots to verify data loading"""
        print("Creating simple time series plots...")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for indicator in self.indicators:
            if indicator not in self.data:
                continue

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()

            for i, land_type in enumerate(self.land_types):
                ax = axes[i]

                if land_type not in self.data[indicator]:
                    ax.text(0.5, 0.5, 'No Data Available',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    ax.set_title(f'{land_type}')
                    continue

                data_stack = self.data[indicator][land_type]['data']
                dates = self.data[indicator][land_type]['dates']

                # Calculate spatial mean for each time step
                spatial_means = []
                valid_dates = []

                for t, date in enumerate(dates):
                    mean_val = np.nanmean(data_stack[t])
                    if not np.isnan(mean_val):
                        spatial_means.append(mean_val)
                        valid_dates.append(date)

                if spatial_means:
                    ax.plot(valid_dates, spatial_means, linewidth=1.5, alpha=0.8)
                    ax.set_title(f'{land_type}', fontweight='bold')
                    ax.set_ylabel(f'{indicator}')
                    ax.grid(True, alpha=0.3)

                    # Rotate x-axis labels
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(45)
                else:
                    ax.text(0.5, 0.5, 'No Valid Data',
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax.transAxes)
                    ax.set_title(f'{land_type}')

            plt.suptitle(f'{indicator} Time Series (Spatial Mean)',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{indicator}_timeseries.png'),
                        dpi=300, bbox_inches='tight')
            plt.close()

            print(f"✅ Created time series plot for {indicator}")

    def run_basic_analysis(self, output_dir='basic_drought_analysis'):
        """Run basic analysis with error handling"""
        print("Starting basic California drought analysis...")

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            # Step 1: Load all data
            print("\n" + "=" * 50)
            print("STEP 1: Loading Data")
            print("=" * 50)
            self.load_all_data()

            # Step 2: Create simple time series plots first
            print("\n" + "=" * 50)
            print("STEP 2: Creating Time Series Plots")
            print("=" * 50)
            self.create_simple_time_series_plot(os.path.join(output_dir, 'time_series'))

            # Step 3: Perform STL decomposition
            print("\n" + "=" * 50)
            print("STEP 3: STL Decomposition")
            print("=" * 50)
            self.perform_stl_decomposition()

            # Step 4: Create trend maps
            print("\n" + "=" * 50)
            print("STEP 4: Creating Trend Maps")
            print("=" * 50)
            self.create_static_trend_maps(os.path.join(output_dir, 'trend_maps'))

            print(f"\n✅ Basic analysis completed! Results saved to: {output_dir}")

        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()

    def quick_data_check(self):
        """Quick check of what data is actually available"""
        print("Quick Data Availability Check:")
        print("=" * 40)

        total_datasets = 0
        available_datasets = 0

        for indicator in self.indicators:
            print(f"\n{indicator}:")
            for land_type in self.land_types:
                folder_path = os.path.join(self.base_path, self.data_paths[indicator][land_type])
                total_datasets += 1

                if os.path.exists(folder_path):
                    files = glob.glob(os.path.join(folder_path, "*.tif"))
                    if files:
                        print(f"  ✅ {land_type}: {len(files)} files")
                        available_datasets += 1
                    else:
                        print(f"  ⚠️  {land_type}: Folder exists but no .tif files")
                else:
                    print(f"  ❌ {land_type}: Folder not found")

        print(f"\nSummary: {available_datasets}/{total_datasets} datasets available")
        return available_datasets > 0


# Example usage with better error handling
if __name__ == "__main__":
    # Set your base data path
    BASE_DATA_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

    # Initialize analyzer
    analyzer = CaliforniaDroughtAnalyzer(BASE_DATA_PATH)

    # First, do a quick check of data availability
    if analyzer.quick_data_check():
        print("\n" + "=" * 60)
        print("STARTING ANALYSIS")
        print("=" * 60)

        # Run basic analysis (safer than full analysis)
        analyzer.run_basic_analysis()

        # If basic analysis works, you can optionally run additional components
        print("\n" + "=" * 60)
        print("ADDITIONAL ANALYSIS OPTIONS")
        print("=" * 60)
        print("If the basic analysis worked well, you can uncomment and run:")
        print("# analyzer.create_correlation_analysis()")
        print("# analyzer.create_phenology_analysis()")
        print("# analyzer.create_animated_visualization()")

    else:
        print("\n❌ No data found. Please check your file paths and folder structure.")
        print("\nExpected folder structure:")
        print("Data/")
        print("├── CALIFORNIA GRASSLANDS/")
        print("│   ├── GOSIF/")
        print("│   ├── LST/")
        print("│   ├── NDVI/")
        print("│   └── ...")
        print("├── CALIFORNIA NATURAL FORESTS/")
        print("│   ├── GOSIF/")
        print("│   └── ...")
        print("├── CALIFORNIA SAVANNAS/")
        print("│   ├── GOSIF/")
        print("│   └── ...")
        print("└── CALIFORNIA SHRUBLANDS/")
        print("    ├── GOSIF/  ← Make sure this folder exists!")
        print("    └── ...")


def create_folder_structure_fixer(base_path):
    """Helper function to create missing folder structure"""
    print("Creating missing folder structure...")

    indicators_folders = {
        'GOSIF': 'GOSIF',
        'LST': 'LST',
        'NDVI': 'NDVI',
        'NIRv': 'NIRv',
        'SPEI': 'SPEI',
        'TWS': 'TWS',
        'VPDmax': 'VPD'
    }

    land_types = ['CALIFORNIA GRASSLANDS', 'CALIFORNIA NATURAL FORESTS',
                  'CALIFORNIA SAVANNAS', 'CALIFORNIA SHRUBLANDS']

    created_folders = []

    for land_type in land_types:
        land_path = os.path.join(base_path, land_type)
        if not os.path.exists(land_path):
            os.makedirs(land_path)
            created_folders.append(land_path)

        for indicator, folder_name in indicators_folders.items():
            indicator_path = os.path.join(land_path, folder_name)
            if not os.path.exists(indicator_path):
                os.makedirs(indicator_path)
                created_folders.append(indicator_path)
                print(f"Created: {indicator_path}")

    if created_folders:
        print(f"\n✅ Created {len(created_folders)} missing folders.")
        print("Now move your data files to the appropriate folders.")
    else:
        print("All required folders already exist.")


def diagnose_gosif_shrublands_issue(base_path):
    """Specific function to diagnose the GOSIF Shrublands issue"""
    print("Diagnosing GOSIF Shrublands issue...")
    print("=" * 40)

    shrublands_base = os.path.join(base_path, "CALIFORNIA SHRUBLANDS")

    if not os.path.exists(shrublands_base):
        print(f"❌ Base folder doesn't exist: {shrublands_base}")
        return

    print(f"✅ Base folder exists: {shrublands_base}")

    # List what's actually in the shrublands folder
    try:
        contents = os.listdir(shrublands_base)
        print(f"Contents of CALIFORNIA SHRUBLANDS folder:")
        for item in contents:
            item_path = os.path.join(shrublands_base, item)
            if os.path.isdir(item_path):
                file_count = len(glob.glob(os.path.join(item_path, "*.tif")))
                print(f"  📁 {item}/ ({file_count} .tif files)")
            else:
                print(f"  📄 {item}")
    except Exception as e:
        print(f"❌ Error reading folder contents: {e}")
        return

    # Check specifically for GOSIF folder
    gosif_path = os.path.join(shrublands_base, "GOSIF")
    if os.path.exists(gosif_path):
        print(f"✅ GOSIF folder exists: {gosif_path}")
        tif_files = glob.glob(os.path.join(gosif_path, "*.tif"))
        print(f"   Found {len(tif_files)} .tif files")
        if len(tif_files) > 0:
            print(f"   Sample files:")
            for i, file in enumerate(tif_files[:3]):
                print(f"     - {os.path.basename(file)}")
            if len(tif_files) > 3:
                print(f"     - ... and {len(tif_files) - 3} more")
    else:
        print(f"❌ GOSIF folder missing: {gosif_path}")
        print("Solution: Create the GOSIF folder and move your GOSIF shrublands data there.")

        # Check if there are GOSIF files in the base shrublands folder
        base_tif_files = glob.glob(os.path.join(shrublands_base, "*GOSIF*.tif"))
        if base_tif_files:
            print(f"💡 Found {len(base_tif_files)} GOSIF files in the base folder.")
            print("   These should be moved to a GOSIF subfolder.")


def fix_common_issues():
    """Function to fix common data structure issues"""
    print("Common Issue Fixes:")
    print("=" * 20)
    print("1. Missing GOSIF folder in SHRUBLANDS:")
    print("   - Create: CALIFORNIA SHRUBLANDS/GOSIF/")
    print("   - Move GOSIF_Shrublands_*.tif files there")
    print()
    print("2. Inconsistent folder naming:")
    print("   - VPD vs VPDmax folder names")
    print("   - Check your actual folder names match the code")
    print()
    print("3. File naming patterns:")
    print("   - GOSIF: GOSIF_Shrublands_CALIFORNIA_GOSIF_YYYYMM.tif")
    print("   - Make sure your files follow this pattern")


# Additional utility functions for troubleshooting
def check_single_file_loading(file_path):
    """Test loading a single file to check for issues"""
    print(f"Testing file: {os.path.basename(file_path)}")

    try:
        with rasterio.open(file_path) as src:
            print(f"  ✅ File opens successfully")
            print(f"  Shape: {src.shape}")
            print(f"  CRS: {src.crs}")
            print(f"  NoData: {src.nodata}")

            # Read first few pixels
            data = src.read(1)
            print(f"  Data type: {data.dtype}")
            print(f"  Min value: {np.nanmin(data)}")
            print(f"  Max value: {np.nanmax(data)}")
            print(f"  NaN count: {np.sum(np.isnan(data))}")

            return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


if __name__ == "__main__":
    BASE_DATA_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

    print("CALIFORNIA DROUGHT ANALYZER - DIAGNOSTIC MODE")
    print("=" * 60)

    # Step 1: Diagnose the specific GOSIF Shrublands issue
    diagnose_gosif_shrublands_issue(BASE_DATA_PATH)

    print("\n" + "=" * 60)

    # Step 2: Show common fixes
    fix_common_issues()

    print("\n" + "=" * 60)

    # Step 3: Create folder structure if needed
    create_missing = input("\nCreate missing folder structure? (y/n): ").lower().strip()
    if create_missing == 'y':
        create_folder_structure_fixer(BASE_DATA_PATH)

    print("\n" + "=" * 60)

    # Step 4: Run the analysis
    run_analysis = input("Run analysis with current data? (y/n): ").lower().strip()
    if run_analysis == 'y':
        analyzer = CaliforniaDroughtAnalyzer(BASE_DATA_PATH)
        if analyzer.quick_data_check():
            analyzer.run_basic_analysis()
        else:
            print("❌ Please fix data structure issues first.")

    print("\n" + "=" * 60)
    print("TROUBLESHOOTING TIPS:")
    print("1. Check that CALIFORNIA SHRUBLANDS/GOSIF/ folder exists")
    print("2. Verify file naming patterns match expected format")
    print("3. Run the basic analysis first before trying advanced features")
    print("4. Check file paths are correct for your system")
    print("=" * 60)