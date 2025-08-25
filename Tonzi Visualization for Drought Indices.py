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
import warnings

warnings.filterwarnings('ignore')


class TonziDroughtProcessor:
    def __init__(self):
        self.data_paths = {
            'GOSIF': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GOSIF_on_SPEI_Grid",
            'LST': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\Tonzi_LST",
            'NDVI': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\Tonzi_NDVI",
            'NIRv': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\Tonzi_NIRv",
            'SPEI': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\SPEI_tif_Cleaned",
            'TWS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\FINAL GOSIF AND GRACE AND SPEI\GRACE_on_SPEI_Grid"
        }

        self.file_patterns = {
            'GOSIF': r'GOSIF_(\d{4})(\d{2})_Enhanced_Tonzi',
            'LST': r'MYD21C1_Tonzi_LST_(\d{4})_(\d{2})',
            'NDVI': r'NDVI_MOD13Q1_Tonzi_(\d{4})_(\d{2})',
            'NIRv': r'NIRv_MOD13Q1_Tonzi_(\d{4})_(\d{2})',
            'SPEI': r'SPEI_(\d{4})_(\d{2})_cleaned',
            'TWS': r'GRACE_TWS_Tonzi_(\d{4})_(\d{2}).*'  # Made more flexible to handle the long suffix
        }

        self.valid_pixels_mask = None
        self.processed_data = {}

    def extract_date_from_filename(self, filename, pattern):
        """Extract year and month from filename using regex pattern"""
        match = re.search(pattern, filename)
        if match:
            year, month = int(match.group(1)), int(match.group(2))
            # Debug print to identify problematic files
            print(f"  Parsing {filename}: Year={year}, Month={month}")

            # Validate month range
            if month < 1 or month > 12:
                print(f"  Warning: Invalid month {month} in file {filename}, skipping...")
                return None

            return datetime(year, month, 1)
        else:
            print(f"  No match for pattern in file: {filename}")
        return None

    def read_tif_files(self, folder_path, pattern):
        """Read all TIF files from folder and organize by date"""
        data_dict = {}

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist!")
            return data_dict

        print(f"  Scanning folder: {folder_path}")
        tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.tif')]
        print(f"  Found {len(tif_files)} TIF files")

        for filename in tif_files:
            print(f"  Processing: {filename}")
            date = self.extract_date_from_filename(filename, pattern)
            if date:
                filepath = os.path.join(folder_path, filename)
                try:
                    with rasterio.open(filepath) as src:
                        data = src.read(1).astype(np.float32)
                        # Replace no-data values with NaN
                        if src.nodata is not None:
                            data[data == src.nodata] = np.nan
                        data_dict[date] = data
                        print(f"    Successfully loaded data for {date.strftime('%Y-%m')}")
                except Exception as e:
                    print(f"    Error reading {filename}: {e}")
            else:
                print(f"    Could not extract date from {filename}")

        print(f"  Total files loaded: {len(data_dict)}")
        return dict(sorted(data_dict.items()))

    def resample_to_common_grid(self, data, target_shape):
        """Resample data to target shape using nearest neighbor interpolation"""
        from scipy.ndimage import zoom

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

        # Print all found dimensions
        print("All dimensions found:")
        for shape, count in sorted(dimension_counts.items()):
            print(f"  {shape}: {count} files")

        return most_common_shape

    def create_valid_pixels_mask(self):
        """Create a mask of pixels that are valid across all indices and time periods"""
        print("Creating valid pixels mask...")

        all_data = {}
        for index_name, folder_path in self.data_paths.items():
            print(f"Reading {index_name} data...")
            pattern = self.file_patterns[index_name]
            all_data[index_name] = self.read_tif_files(folder_path, pattern)

        if not any(all_data.values()):
            raise ValueError("No valid data found in any folder!")

        # Find common dimensions
        target_shape = self.find_common_dimensions(all_data)

        # Resample all data to common grid
        print(f"\nResampling all data to common grid: {target_shape}")
        resampled_data = {}

        for index_name, index_data in all_data.items():
            if not index_data:
                continue

            resampled_data[index_name] = {}
            resample_count = 0

            for date, data in index_data.items():
                if data.shape != target_shape:
                    resampled_data[index_name][date] = self.resample_to_common_grid(data, target_shape)
                    resample_count += 1
                else:
                    resampled_data[index_name][date] = data

            if resample_count > 0:
                print(f"  {index_name}: Resampled {resample_count} files")
            else:
                print(f"  {index_name}: No resampling needed")

        # Initialize mask with True values
        self.valid_pixels_mask = np.ones(target_shape, dtype=bool)

        # For each index, check all time periods
        for index_name, index_data in resampled_data.items():
            if not index_data:
                continue

            # Create a mask for this index (pixels that are never zero or NaN)
            index_mask = np.ones(target_shape, dtype=bool)

            for date, data in index_data.items():
                # Mark pixels as invalid if they are NaN, zero, or extreme values
                invalid_pixels = (np.isnan(data) |
                                  (data == 0) |
                                  (np.abs(data) > 1e10))  # Remove extreme values
                index_mask &= ~invalid_pixels

            # Update global mask
            self.valid_pixels_mask &= index_mask
            print(f"{index_name}: {np.sum(index_mask)} valid pixels")

        print(f"Final valid pixels: {np.sum(self.valid_pixels_mask)} out of {self.valid_pixels_mask.size}")
        return resampled_data

    def fill_missing_data(self, data_dict, index_name):
        """Fill missing data using the three-step method"""
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
                filled_data[target_date] = self.estimate_missing_data(
                    data_dict, target_date, complete_dates
                )

        return filled_data

    def estimate_missing_data(self, data_dict, target_date, complete_dates):
        """Estimate missing data using the three-step approach"""
        target_year = target_date.year
        target_month = target_date.month

        # Get sample shape from existing data
        sample_shape = None
        for data in data_dict.values():
            sample_shape = data.shape
            break

        if sample_shape is None:
            return np.full(self.valid_pixels_mask.shape, np.nan, dtype=np.float32)

        # Step 1: Try adjacent years average
        prev_year_date = datetime(target_year - 1, target_month, 1)
        next_year_date = datetime(target_year + 1, target_month, 1)

        if prev_year_date in data_dict and next_year_date in data_dict:
            estimated = (data_dict[prev_year_date] + data_dict[next_year_date]) / 2
            print(f"  Using adjacent years average")
            return estimated

        # Step 2: Try trend-based estimation (simple linear trend)
        same_month_data = []
        same_month_years = []

        for date, data in data_dict.items():
            if date.month == target_month:
                same_month_data.append(np.nanmean(data[self.valid_pixels_mask]))
                same_month_years.append(date.year)

        if len(same_month_data) >= 3:  # Need at least 3 points for trend
            # Fit linear trend
            X = np.array(same_month_years).reshape(-1, 1)
            y = np.array(same_month_data)

            valid_idx = ~np.isnan(y)
            if np.sum(valid_idx) >= 2:
                model = LinearRegression()
                model.fit(X[valid_idx], y[valid_idx])
                predicted_mean = model.predict([[target_year]])[0]

                # Use the spatial pattern from the closest available year
                closest_year = min(same_month_years, key=lambda x: abs(x - target_year))
                closest_date = datetime(closest_year, target_month, 1)
                spatial_pattern = data_dict[closest_date]

                # Scale the spatial pattern to match predicted mean
                current_mean = np.nanmean(spatial_pattern[self.valid_pixels_mask])
                if current_mean != 0 and not np.isnan(current_mean):
                    scale_factor = predicted_mean / current_mean
                    estimated = spatial_pattern * scale_factor
                    print(f"  Using trend-based estimation")
                    return estimated

        # Step 3: Use long-term average for the month
        if same_month_data:
            monthly_mean = np.nanmean(same_month_data)
            # Use spatial pattern from closest available data
            if data_dict:
                closest_date = min(data_dict.keys(), key=lambda x: abs((x - target_date).days))
                spatial_pattern = data_dict[closest_date]
                current_mean = np.nanmean(spatial_pattern[self.valid_pixels_mask])

                if current_mean != 0 and not np.isnan(current_mean):
                    scale_factor = monthly_mean / current_mean
                    estimated = spatial_pattern * scale_factor
                    print(f"  Using monthly average")
                    return estimated

        # Fallback: return NaN array
        print(f"  No estimation possible, using NaN")
        return np.full(self.valid_pixels_mask.shape, np.nan, dtype=np.float32)

    def process_all_indices(self):
        """Process all indices and calculate time series"""
        print("Processing all indices...")

        # First, create the valid pixels mask
        all_raw_data = self.create_valid_pixels_mask()

        # Process each index
        for index_name in self.data_paths.keys():
            print(f"\nProcessing {index_name}...")

            if index_name not in all_raw_data or not all_raw_data[index_name]:
                print(f"No data found for {index_name}")
                continue

            # Fill missing data
            filled_data = self.fill_missing_data(all_raw_data[index_name], index_name)

            # Calculate time series (mean of valid pixels)
            time_series = []
            dates = sorted(filled_data.keys())

            for date in dates:
                data = filled_data[date]
                # Calculate mean only from valid pixels
                valid_data = data[self.valid_pixels_mask]
                mean_value = np.nanmean(valid_data)
                time_series.append(mean_value)

            # Store processed data
            self.processed_data[index_name] = {
                'dates': dates,
                'values': np.array(time_series),
                'raw_data': filled_data
            }

            print(f"{index_name}: {len(time_series)} time points processed")

    def calculate_z_scores(self):
        """Calculate z-scores for each index separately"""
        print("\nCalculating z-scores...")

        for index_name, data in self.processed_data.items():
            values = data['values']
            # Remove any remaining NaN values for z-score calculation
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                z_scores = stats.zscore(values, nan_policy='omit')
                data['z_scores'] = z_scores
                print(f"{index_name}: Mean={np.nanmean(values):.3f}, Std={np.nanstd(values):.3f}")
            else:
                data['z_scores'] = np.full_like(values, np.nan)
                print(f"{index_name}: No valid data for z-score calculation")

    def create_visualization(self):
        """Create the temporal trends visualization"""
        print("\nCreating visualization...")

        # Set up the plot
        fig, axes = plt.subplots(len(self.processed_data), 1, figsize=(15, 3 * len(self.processed_data)))
        if len(self.processed_data) == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, (index_name, data) in enumerate(self.processed_data.items()):
            ax = axes[i]
            dates = data['dates']
            z_scores = data['z_scores']

            # Plot the time series
            ax.plot(dates, z_scores, color=colors[i % len(colors)], linewidth=1, alpha=0.8)
            ax.fill_between(dates, z_scores, alpha=0.3, color=colors[i % len(colors)])
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

            # Set labels and title
            ax.set_ylabel(f'{index_name}\n(Z-score)', fontsize=10)
            ax.set_title(f'{index_name} - Mean: {np.nanmean(z_scores):.2f}, Std Dev: {np.nanstd(z_scores):.2f}',
                         fontsize=10, loc='left')

            # Format x-axis
            ax.set_xlim(dates[0], dates[-1])

            # Add text box with statistics in upper left
            textstr = f'Mean: {np.nanmean(z_scores):.2f}\nStd Dev: {np.nanstd(z_scores):.2f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
                    verticalalignment='top', bbox=props)

        # Set overall title and labels
        plt.suptitle('Temporal Trends of Climate-Eco Indicators at Tonzi Ranch (2001-2021)', fontsize=16, y=0.98)
        axes[-1].set_xlabel('Time', fontsize=12)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)

        return fig

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Tonzi Ranch Drought Indices Analysis...")
        print("=" * 50)

        try:
            # Process all data
            self.process_all_indices()

            # Calculate z-scores
            self.calculate_z_scores()

            # Create visualization
            fig = self.create_visualization()

            # Show results
            print("\nAnalysis completed successfully!")
            print(f"Processed {len(self.processed_data)} indices")
            print(f"Valid pixels used: {np.sum(self.valid_pixels_mask)}")

            plt.show()

            return fig, self.processed_data

        except Exception as e:
            print(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return None, None


# Usage
if __name__ == "__main__":
    # Initialize and run the processor
    processor = TonziDroughtProcessor()
    fig, processed_data = processor.run_analysis()

    # Print summary statistics
    if processed_data:
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)

        for index_name, data in processed_data.items():
            values = data['values']
            z_scores = data['z_scores']
            dates = data['dates']

            print(f"\n{index_name}:")
            print(f"  Time period: {dates[0].strftime('%Y-%m')} to {dates[-1].strftime('%Y-%m')}")
            print(f"  Data points: {len(values)}")
            print(f"  Raw data - Mean: {np.nanmean(values):.4f}, Std: {np.nanstd(values):.4f}")
            print(f"  Z-scores - Mean: {np.nanmean(z_scores):.4f}, Std: {np.nanstd(z_scores):.4f}")
            print(f"  Min Z-score: {np.nanmin(z_scores):.2f}, Max Z-score: {np.nanmax(z_scores):.2f}")