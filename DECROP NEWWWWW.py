import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import zoom
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
import glob
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class DroughtCorrelationAnalysis:
    def __init__(self):
        self.datasets = {
            'GOSIF': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\GOSIF_CA_CLEAN",
            'LST': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\MODIS_CA_Monthly_LST",
            'NDVI': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\NDVI_California_NaturalOnly",
            'NIRv': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\NIRv_MOD13Q1_5km_TIFs",
            'SPEI': r"C:\Users\LOQ\OneDrive\Desktop\California MASKED\SPEI_Masked_Output",
            'TWS': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\TWS_GRACE_TIFs"
        }

        self.file_patterns = {
            'GOSIF': 'CALIFORNIA_GOSIF_MASKED_*',
            'LST': 'MYD21C1_CA_LST_masked_*',
            'NDVI': 'NDVI_CA_natland_5km_*',
            'NIRv': 'NIRv_CA_natland_5km_*',
            'SPEI': 'SPEI1_CSIC_CA_*_masked*',
            'TWS': 'TWS_GRACE_CA_*'
        }

        self.target_resolution = 5000  # 5km in meters to match GEE export
        self.years = range(2001, 2022)
        self.months = range(1, 13)

    def get_file_list(self, dataset_name):
        """Get list of files for a specific dataset"""
        folder_path = self.datasets[dataset_name]
        pattern = self.file_patterns[dataset_name]
        file_pattern = os.path.join(folder_path, pattern)
        files = glob.glob(file_pattern)
        return sorted(files)

    def parse_date_from_filename(self, filename, dataset_name):
        """Extract date from filename based on dataset naming convention"""
        basename = os.path.basename(filename)

        try:
            if dataset_name == 'GOSIF':
                # CALIFORNIA_GOSIF_MASKED_200101.tif
                date_str = basename.split('_')[-1].split('.')[0]  # Remove .tif extension
                return datetime.strptime(date_str, '%Y%m')
            elif dataset_name == 'LST':
                # MYD21C1_CA_LST_masked_2002_07
                parts = basename.split('_')
                year = parts[-2]
                month = parts[-1].split('.')[0]
                return datetime.strptime(f"{year}{month.zfill(2)}", '%Y%m')
            elif dataset_name in ['NDVI', 'NIRv']:
                # NDVI_CA_natland_5km_2001_01
                parts = basename.split('_')
                year = parts[-2]
                month = parts[-1].split('.')[0]
                return datetime.strptime(f"{year}{month.zfill(2)}", '%Y%m')
            elif dataset_name == 'SPEI':
                # SPEI1_CSIC_CA_2001_01_masked
                parts = basename.split('_')
                year = parts[3]
                month = parts[4]
                return datetime.strptime(f"{year}{month.zfill(2)}", '%Y%m')
            elif dataset_name == 'TWS':
                # TWS_GRACE_CA_2002_04
                parts = basename.split('_')
                year = parts[-2]
                month = parts[-1].split('.')[0]
                return datetime.strptime(f"{year}{month.zfill(2)}", '%Y%m')
        except:
            print(f"Could not parse date from {basename}")
            return None

    def get_reference_grid(self):
        """Use NDVI or NIRv data as reference grid (already at 5km resolution matching GEE export)"""
        # Use NDVI as reference since it's already at 5km from GEE
        ndvi_files = self.get_file_list('NDVI')
        if not ndvi_files:
            nirv_files = self.get_file_list('NIRv')
            if nirv_files:
                ref_file = nirv_files[0]
            else:
                raise ValueError("No NDVI or NIRv files found for reference grid")
        else:
            ref_file = ndvi_files[0]

        with rasterio.open(ref_file) as src:
            self.reference_transform = src.transform
            self.reference_crs = src.crs
            self.reference_shape = src.shape
            self.reference_bounds = src.bounds

        print(f"Reference 5km grid (matching GEE export): {self.reference_shape}, bounds: {self.reference_bounds}")
        print(f"Transform: {self.reference_transform}")
        return self.reference_transform, self.reference_crs, self.reference_shape

    def regrid_to_reference(self, input_file, output_array):
        """Regrid a raster to the 5km reference grid (matching GEE export)"""
        try:
            with rasterio.open(input_file) as src:
                # Read the data
                data = src.read(1)

                # If already same shape and transform, just return the data
                if (data.shape == self.reference_shape and
                        src.transform == self.reference_transform):
                    return data

                # Reproject to reference grid
                reproject(
                    source=data,
                    destination=output_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=self.reference_transform,
                    dst_crs=self.reference_crs,
                    resampling=Resampling.bilinear
                )

                return output_array
        except Exception as e:
            print(f"Error regridding {input_file}: {e}")
            return None

    def load_dataset_timeseries(self, dataset_name):
        """Load all files for a dataset and create time series"""
        files = self.get_file_list(dataset_name)
        print(f"Loading {dataset_name}: found {len(files)} files")

        if not files:
            print(f"No files found for {dataset_name}")
            return None, None

        # Initialize output array for regridding
        output_array = np.zeros(self.reference_shape, dtype=np.float32)

        data_dict = {}
        dates = []

        for file_path in files:
            date = self.parse_date_from_filename(file_path, dataset_name)
            if date is None:
                continue

            # Only process data within our time range
            if date.year < 2001 or date.year > 2021:
                continue

            try:
                regridded_data = self.regrid_to_reference(file_path, output_array.copy())
                if regridded_data is not None:
                    # Mask invalid values (preserve the exact masking from GEE)
                    regridded_data = np.where(
                        (regridded_data == -9999) |
                        (regridded_data == -32768) |
                        (regridded_data == -3.4028235e+38) |  # Common GEE nodata value
                        np.isnan(regridded_data) |
                        (regridded_data == 0),  # Only mask explicit 0, not near-zero values
                        np.nan,
                        regridded_data
                    )

                    data_dict[date] = regridded_data
                    dates.append(date)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue

        if not data_dict:
            print(f"No valid data loaded for {dataset_name}")
            return None, None

        # Convert to 3D array (time, y, x)
        dates = sorted(dates)
        data_array = np.array([data_dict[date] for date in dates])

        print(f"{dataset_name}: loaded {len(dates)} time steps, shape: {data_array.shape}")
        return data_array, dates

    def create_common_mask(self, data_arrays):
        """Create a common valid pixel mask exactly matching the GEE agricultural masking"""
        print("Creating common mask (matching GEE agricultural masking)...")

        # Use NDVI or NIRv as the baseline mask since they have the exact agricultural masking from GEE
        baseline_mask = None
        for dataset_name in ['NDVI', 'NIRv']:
            if dataset_name in data_arrays and data_arrays[dataset_name][0] is not None:
                data_array, dates = data_arrays[dataset_name]
                # Create mask where at least some time steps have valid data (non-NaN, non-zero)
                baseline_mask = ~np.isnan(data_array).all(axis=0) & ~(data_array == 0).all(axis=0)
                print(f"Using {dataset_name} as baseline mask: {np.sum(baseline_mask)} pixels")
                break

        if baseline_mask is None:
            print("Warning: No NDVI/NIRv data found for baseline mask, using general approach")
            baseline_mask = np.ones(self.reference_shape, dtype=bool)

        # Apply baseline mask to all other datasets
        common_mask = baseline_mask.copy()

        for dataset_name, (data_array, dates) in data_arrays.items():
            if data_array is None:
                continue

            # Create mask where at least some time steps have valid data
            dataset_mask = ~np.isnan(data_array).all(axis=0)

            # For datasets other than NDVI/NIRv, intersect with baseline mask
            if dataset_name not in ['NDVI', 'NIRv']:
                dataset_mask = dataset_mask & baseline_mask

            common_mask = common_mask & dataset_mask

            print(f"{dataset_name}: {np.sum(dataset_mask)} valid pixels")

        total_valid_pixels = np.sum(common_mask)
        print(f"Final common mask (natural lands only): {total_valid_pixels} pixels")

        return common_mask

    def calculate_pixel_correlations(self, data1, data2, common_mask):
        """Calculate correlation for each pixel between two datasets"""
        correlation_map = np.full(self.reference_shape, np.nan)
        p_value_map = np.full(self.reference_shape, np.nan)

        valid_pixels = np.where(common_mask)

        for i, j in zip(valid_pixels[0], valid_pixels[1]):
            # Get time series for this pixel
            ts1 = data1[:, i, j]
            ts2 = data2[:, i, j]

            # Remove NaN values
            valid_idx = ~(np.isnan(ts1) | np.isnan(ts2))

            if np.sum(valid_idx) < 10:  # Need at least 10 valid observations for robust correlation
                continue

            ts1_clean = ts1[valid_idx]
            ts2_clean = ts2[valid_idx]

            # Calculate correlation
            if len(ts1_clean) > 3 and np.var(ts1_clean) > 0 and np.var(ts2_clean) > 0:
                corr_coef, p_val = stats.pearsonr(ts1_clean, ts2_clean)
                correlation_map[i, j] = corr_coef
                p_value_map[i, j] = p_val

        return correlation_map, p_value_map

    def get_coordinates(self):
        """Get longitude and latitude coordinates"""
        # Create coordinate arrays
        cols, rows = np.meshgrid(np.arange(self.reference_shape[1]),
                                 np.arange(self.reference_shape[0]))

        # Transform to geographic coordinates
        xs, ys = rasterio.transform.xy(self.reference_transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)

        return lons, lats

    def plot_correlation_heatmap(self, correlation_map, dataset1, dataset2, lons, lats, common_mask):
        """Create correlation heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Mask correlation map
        masked_corr = np.where(common_mask, correlation_map, np.nan)

        # Create the heatmap
        im = ax.imshow(masked_corr,
                       extent=[lons.min(), lons.max(), lats.min(), lats.max()],
                       cmap='RdBu_r',
                       vmin=-1, vmax=1,
                       aspect='auto',
                       origin='upper')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient', fontsize=12)

        # Set labels and title
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        ax.set_title(f'{dataset1} vs {dataset2} Correlation (5km resolution)\n(2001-2021, Natural Lands Only)',
                     fontsize=14, fontweight='bold')

        # Add statistics
        valid_corrs = masked_corr[~np.isnan(masked_corr)]
        if len(valid_corrs) > 0:
            mean_corr = np.mean(valid_corrs)
            std_corr = np.std(valid_corrs)
            ax.text(0.02, 0.98, f'Mean: {mean_corr:.3f}\nStd: {std_corr:.3f}\nPixels: {len(valid_corrs)}',
                    transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        # Save the plot
        output_filename = f'correlation_{dataset1}_{dataset2}.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_filename}")

        return fig

    def run_analysis(self):
        """Run the complete correlation analysis"""
        print("Starting California Drought Correlation Analysis...")

        # Get reference grid
        self.get_reference_grid()

        # Load all datasets
        print("\nLoading datasets...")
        data_arrays = {}
        for dataset_name in self.datasets.keys():
            data_array, dates = self.load_dataset_timeseries(dataset_name)
            data_arrays[dataset_name] = (data_array, dates)

        # Remove datasets that failed to load
        data_arrays = {k: v for k, v in data_arrays.items() if v[0] is not None}

        if len(data_arrays) < 2:
            print("Error: Need at least 2 datasets for correlation analysis")
            return

        # Create common mask
        common_mask = self.create_common_mask(data_arrays)

        # Get coordinates
        lons, lats = self.get_coordinates()

        # Calculate correlations for all pairs
        dataset_names = list(data_arrays.keys())
        correlation_results = {}

        print(f"\nCalculating correlations for {len(dataset_names)} datasets...")

        for i in range(len(dataset_names)):
            for j in range(i + 1, len(dataset_names)):
                name1, name2 = dataset_names[i], dataset_names[j]
                print(f"Calculating correlation: {name1} vs {name2}")

                data1, dates1 = data_arrays[name1]
                data2, dates2 = data_arrays[name2]

                # Find common dates
                common_dates = sorted(set(dates1) & set(dates2))
                if len(common_dates) < 10:
                    print(f"Warning: Only {len(common_dates)} common dates for {name1} vs {name2}")
                    continue

                # Align data to common dates
                idx1 = [dates1.index(d) for d in common_dates if d in dates1]
                idx2 = [dates2.index(d) for d in common_dates if d in dates2]

                aligned_data1 = data1[idx1]
                aligned_data2 = data2[idx2]

                # Calculate correlation
                corr_map, p_map = self.calculate_pixel_correlations(
                    aligned_data1, aligned_data2, common_mask
                )

                correlation_results[(name1, name2)] = (corr_map, p_map)

                # Plot heatmap
                self.plot_correlation_heatmap(
                    corr_map, name1, name2, lons, lats, common_mask
                )

                plt.show()

        print(f"\nAnalysis complete! Generated {len(correlation_results)} correlation maps.")
        return correlation_results, common_mask, lons, lats


# Run the analysis
if __name__ == "__main__":
    analysis = DroughtCorrelationAnalysis()
    results, mask, lons, lats = analysis.run_analysis()