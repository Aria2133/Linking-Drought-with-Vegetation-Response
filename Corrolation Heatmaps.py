import os
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.ndimage import zoom
import seaborn as sns
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


class SpatialCorrelationAnalyzer:
    def __init__(self):
        self.data_paths = {
            'GOSIF': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\GOSIF_California_Clipped",
            'LST': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\MODIS_CA_Monthly_LST",
            'NDVI': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\NDVI_MOD13Q1_5km_TIFs",
            'NIRv': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\NIRv_MOD13Q1_5km_TIFs",
            'SPEI': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\SPEI1_CSIC_TIFs",
            'TWS': r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\TWS_GRACE_TIFs"
        }

        self.file_patterns = {
            'GOSIF': r'CALIFORNIA_GOSIF_(\d{4})(\d{2})',
            'LST': r'MYD21C1_CA_LST_(\d{4})_(\d{2})',
            'NDVI': r'NDVI_MOD13Q1_CA_5km_(\d{4})_(\d{2})',
            'NIRv': r'NIRv_MOD13Q1_CA_5km_(\d{4})_(\d{2})',
            'SPEI': r'SPEI1_CSIC_CA_(\d{4})_(\d{2})',
            'TWS': r'TWS_GRACE_CA_(\d{4})_(\d{2})'
        }

        self.valid_pixels_mask = None
        self.processed_data = {}
        self.spatial_data = {}
        self.output_folder = "Correlation_Heatmaps"

    def extract_date_from_filename(self, filename, pattern):
        """Extract year and month from filename using regex pattern"""
        match = re.search(pattern, filename)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            return datetime(year, month, 1)
        return None

    def read_tif_files(self, folder_path, pattern):
        """Read all TIF files from folder and organize by date"""
        data_dict = {}

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            return data_dict

        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.tif'):
                date = self.extract_date_from_filename(filename, pattern)
                if date:
                    filepath = os.path.join(folder_path, filename)
                    try:
                        with rasterio.open(filepath) as src:
                            data = src.read(1).astype(np.float32)
                            # Replace no-data values with NaN
                            data[data == src.nodata] = np.nan
                            data_dict[date] = data
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")

        return dict(sorted(data_dict.items()))

    def resample_to_common_grid(self, data, target_shape):
        """Resample data to target shape using nearest neighbor interpolation"""
        if data.shape == target_shape:
            return data

        # Calculate zoom factors
        zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])

        # Resample using nearest neighbor to preserve data characteristics
        resampled = zoom(data, zoom_factors, order=0, prefilter=False)

        return resampled

    def find_common_dimensions(self, all_data):
        """Find the most common dimensions across all datasets"""
        dimension_counts = {}

        for index_name, index_data in all_data.items():
            for date, data in index_data.items():
                shape = data.shape
                if shape in dimension_counts:
                    dimension_counts[shape] += 1
                else:
                    dimension_counts[shape] = 1

        # Find the most common dimension
        most_common_shape = max(dimension_counts.keys(), key=dimension_counts.get)
        print(f"Most common dimensions: {most_common_shape} (appears {dimension_counts[most_common_shape]} times)")

        return most_common_shape

    def load_and_preprocess_data(self):
        """Load and preprocess all spatial data"""
        print("Loading and preprocessing spatial data...")

        all_raw_data = {}
        for index_name, folder_path in self.data_paths.items():
            print(f"Reading {index_name} data...")
            pattern = self.file_patterns[index_name]
            all_raw_data[index_name] = self.read_tif_files(folder_path, pattern)

        if not any(all_raw_data.values()):
            raise ValueError("No valid data found in any folder!")

        # Find common dimensions and resample
        target_shape = self.find_common_dimensions(all_raw_data)

        print(f"Resampling all data to common grid: {target_shape}")
        resampled_data = {}

        for index_name, index_data in all_raw_data.items():
            if not index_data:
                continue

            resampled_data[index_name] = {}

            for date, data in index_data.items():
                if data.shape != target_shape:
                    resampled_data[index_name][date] = self.resample_to_common_grid(data, target_shape)
                else:
                    resampled_data[index_name][date] = data

        # Fill missing data and create complete time series for each pixel
        self.create_spatial_time_series(resampled_data, target_shape)

    def fill_missing_data_spatial(self, data_dict, index_name):
        """Fill missing data using the three-step method for spatial data"""
        # Create complete date range
        start_date = datetime(2001, 1, 1)
        end_date = datetime(2021, 12, 1)
        complete_dates = pd.date_range(start_date, end_date, freq='MS')

        filled_data = {}

        for target_date in complete_dates:
            if target_date in data_dict:
                filled_data[target_date] = data_dict[target_date].copy()
            else:
                print(f"Filling missing data for {index_name} - {target_date.strftime('%Y-%m')}")
                filled_data[target_date] = self.estimate_missing_data_spatial(
                    data_dict, target_date, complete_dates
                )

        return filled_data

    def estimate_missing_data_spatial(self, data_dict, target_date, complete_dates):
        """Estimate missing spatial data using the three-step approach"""
        target_year = target_date.year
        target_month = target_date.month

        # Get sample shape from existing data
        sample_shape = None
        for data in data_dict.values():
            sample_shape = data.shape
            break

        if sample_shape is None:
            return np.full(sample_shape, np.nan, dtype=np.float32)

        # Step 1: Try adjacent years average
        prev_year_date = datetime(target_year - 1, target_month, 1)
        next_year_date = datetime(target_year + 1, target_month, 1)

        if prev_year_date in data_dict and next_year_date in data_dict:
            estimated = (data_dict[prev_year_date] + data_dict[next_year_date]) / 2
            return estimated

        # Step 2: Try trend-based estimation (simple approach)
        same_month_data = []
        same_month_years = []

        for date, data in data_dict.items():
            if date.month == target_month:
                same_month_data.append(data)
                same_month_years.append(date.year)

        if len(same_month_data) >= 2:
            # Use closest available year
            closest_year_idx = np.argmin(np.abs(np.array(same_month_years) - target_year))
            return same_month_data[closest_year_idx].copy()

        # Step 3: Use any available data for the month
        if same_month_data:
            return same_month_data[0].copy()

        # Fallback: return NaN array
        return np.full(sample_shape, np.nan, dtype=np.float32)

    def create_spatial_time_series(self, resampled_data, target_shape):
        """Create spatial time series for each index"""
        print("Creating spatial time series...")

        # Create complete date range
        start_date = datetime(2001, 1, 1)
        end_date = datetime(2021, 12, 1)
        complete_dates = pd.date_range(start_date, end_date, freq='MS')

        self.spatial_data = {}

        for index_name in self.data_paths.keys():
            if index_name not in resampled_data or not resampled_data[index_name]:
                print(f"No data found for {index_name}")
                continue

            print(f"Processing spatial data for {index_name}...")

            # Fill missing data
            filled_data = self.fill_missing_data_spatial(resampled_data[index_name], index_name)

            # Create 3D array: (time, height, width)
            time_series_3d = np.zeros((len(complete_dates), target_shape[0], target_shape[1]), dtype=np.float32)

            for i, date in enumerate(complete_dates):
                if date in filled_data:
                    time_series_3d[i, :, :] = filled_data[date]
                else:
                    time_series_3d[i, :, :] = np.nan

            self.spatial_data[index_name] = time_series_3d
            print(f"{index_name}: {time_series_3d.shape}")

    def calculate_spatial_correlation(self, index1_name, index2_name):
        """Calculate pixel-wise correlation between two indices"""
        print(f"Calculating correlation between {index1_name} and {index2_name}...")

        data1 = self.spatial_data[index1_name]
        data2 = self.spatial_data[index2_name]

        height, width = data1.shape[1], data1.shape[2]
        correlation_map = np.full((height, width), np.nan, dtype=np.float32)
        p_value_map = np.full((height, width), np.nan, dtype=np.float32)

        # Calculate correlation for each pixel
        for i in range(height):
            for j in range(width):
                # Get time series for this pixel
                ts1 = data1[:, i, j]
                ts2 = data2[:, i, j]

                # Remove NaN values
                valid_mask = ~(np.isnan(ts1) | np.isnan(ts2))

                if np.sum(valid_mask) >= 10:  # Need at least 10 valid observations
                    valid_ts1 = ts1[valid_mask]
                    valid_ts2 = ts2[valid_mask]

                    # Calculate Pearson correlation
                    corr, p_val = stats.pearsonr(valid_ts1, valid_ts2)
                    correlation_map[i, j] = corr
                    p_value_map[i, j] = p_val

        return correlation_map, p_value_map

    def create_correlation_heatmap(self, correlation_map, index1_name, index2_name, save_path):
        """Create and save correlation heatmap"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        # Create heatmap
        im = ax.imshow(correlation_map, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson Correlation Coefficient', fontsize=12, rotation=270, labelpad=20)

        # Set title and labels
        ax.set_title(f'Spatial Correlation: {index1_name} vs {index2_name}\n(2001-2021)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude (pixels)', fontsize=12)
        ax.set_ylabel('Latitude (pixels)', fontsize=12)

        # Add statistics text
        valid_corr = correlation_map[~np.isnan(correlation_map)]
        if len(valid_corr) > 0:
            mean_corr = np.mean(valid_corr)
            std_corr = np.std(valid_corr)
            min_corr = np.min(valid_corr)
            max_corr = np.max(valid_corr)

            stats_text = f'Mean: {mean_corr:.3f}\\nStd: {std_corr:.3f}\\nMin: {min_corr:.3f}\\nMax: {max_corr:.3f}\\nValid pixels: {len(valid_corr)}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved: {save_path}")

        return valid_corr

    def create_summary_correlation_matrix(self, correlation_stats):
        """Create summary correlation matrix showing mean correlations"""
        indices = list(self.data_paths.keys())
        n_indices = len(indices)

        # Create correlation matrix
        corr_matrix = np.ones((n_indices, n_indices))

        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                if i != j:
                    pair_key = tuple(sorted([idx1, idx2]))
                    if pair_key in correlation_stats:
                        mean_corr = correlation_stats[pair_key]['mean']
                        corr_matrix[i, j] = mean_corr

        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Mean Spatial Correlation', fontsize=12, rotation=270, labelpad=20)

        # Set ticks and labels
        ax.set_xticks(range(n_indices))
        ax.set_yticks(range(n_indices))
        ax.set_xticklabels(indices, rotation=45, ha='right')
        ax.set_yticklabels(indices)

        # Add correlation values as text
        for i in range(n_indices):
            for j in range(n_indices):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white",
                               fontweight='bold')

        ax.set_title('Mean Spatial Correlations Between Drought Indices\\n(2001-2021)',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()

        # Save summary matrix
        summary_path = os.path.join(self.output_folder, "Summary_Correlation_Matrix.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved summary matrix: {summary_path}")

        return corr_matrix

    def run_correlation_analysis(self):
        """Run complete spatial correlation analysis"""
        print("Starting Spatial Correlation Analysis...")
        print("=" * 60)

        try:
            # Create output folder
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
                print(f"Created output folder: {self.output_folder}")

            # Load and preprocess data
            self.load_and_preprocess_data()

            # Get all possible pairs of indices
            indices = [name for name in self.data_paths.keys() if name in self.spatial_data]
            index_pairs = list(combinations(indices, 2))

            print(f"\\nCalculating correlations for {len(index_pairs)} index pairs...")

            correlation_stats = {}

            # Calculate correlation for each pair
            for i, (index1, index2) in enumerate(index_pairs, 1):
                print(f"\\nPair {i}/{len(index_pairs)}: {index1} vs {index2}")

                # Calculate spatial correlation
                correlation_map, p_value_map = self.calculate_spatial_correlation(index1, index2)

                # Create and save heatmap
                filename = f"Correlation_{index1}_vs_{index2}.png"
                save_path = os.path.join(self.output_folder, filename)

                valid_corr = self.create_correlation_heatmap(correlation_map, index1, index2, save_path)

                # Store statistics
                if len(valid_corr) > 0:
                    correlation_stats[(index1, index2)] = {
                        'mean': np.mean(valid_corr),
                        'std': np.std(valid_corr),
                        'min': np.min(valid_corr),
                        'max': np.max(valid_corr),
                        'valid_pixels': len(valid_corr)
                    }

            # Create summary correlation matrix
            print("\\nCreating summary correlation matrix...")
            summary_matrix = self.create_summary_correlation_matrix(correlation_stats)

            # Print summary statistics
            print("\\n" + "=" * 60)
            print("CORRELATION ANALYSIS SUMMARY")
            print("=" * 60)

            for (idx1, idx2), stats in correlation_stats.items():
                print(f"\\n{idx1} vs {idx2}:")
                print(f"  Mean correlation: {stats['mean']:.4f}")
                print(f"  Std deviation: {stats['std']:.4f}")
                print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
                print(f"  Valid pixels: {stats['valid_pixels']}")

            print(f"\\nAll {len(index_pairs)} correlation heatmaps saved to: {self.output_folder}")

            return correlation_stats, summary_matrix

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# Usage
if __name__ == "__main__":
    # Initialize and run the analyzer
    analyzer = SpatialCorrelationAnalyzer()
    correlation_stats, summary_matrix = analyzer.run_correlation_analysis()

    if correlation_stats:
        print("\\nAnalysis completed successfully!")
        print(f"Generated {len(correlation_stats)} individual correlation heatmaps")
        print("Plus 1 summary correlation matrix")
        print("\\nCheck the 'Correlation_Heatmaps' folder for all output files.")