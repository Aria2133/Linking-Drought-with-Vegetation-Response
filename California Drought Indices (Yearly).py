import os
import re
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from scipy.ndimage import zoom
import warnings

warnings.filterwarnings('ignore')


class CaliforniaDroughtProcessor:
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

    def aggregate_to_yearly(self, monthly_data):
        """Aggregate monthly data to yearly data"""
        # Group data by year
        yearly_data = {}
        yearly_values = {}

        # Group monthly data by year
        for date, value in monthly_data.items():
            year = date.year
            if year not in yearly_values:
                yearly_values[year] = []
            yearly_values[year].append(value)

        # Calculate yearly means
        for year, values in yearly_values.items():
            if values:  # Check if list is not empty
                yearly_mean = np.nanmean(values)
                yearly_data[datetime(year, 1, 1)] = yearly_mean

        return yearly_data

    def process_all_indices(self):
        """Process all indices and calculate time series with yearly aggregation"""
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

            # Calculate monthly time series (mean of valid pixels)
            monthly_time_series = {}
            dates = sorted(filled_data.keys())

            for date in dates:
                data = filled_data[date]
                # Calculate mean only from valid pixels
                valid_data = data[self.valid_pixels_mask]
                mean_value = np.nanmean(valid_data)
                monthly_time_series[date] = mean_value

            # Aggregate to yearly data
            yearly_data = self.aggregate_to_yearly(monthly_time_series)
            yearly_dates = sorted(yearly_data.keys())
            yearly_values = [yearly_data[date] for date in yearly_dates]

            # Store processed data
            self.processed_data[index_name] = {
                'dates': yearly_dates,
                'values': np.array(yearly_values),
                'monthly_data': monthly_time_series
            }

            print(f"{index_name}: {len(yearly_values)} yearly points processed")

    def calculate_z_scores_and_trends(self):
        """Calculate z-scores and trend analysis for each index separately"""
        print("\nCalculating z-scores and trends...")

        for index_name, data in self.processed_data.items():
            values = data['values']
            dates = data['dates']

            # Remove any remaining NaN values for z-score calculation
            valid_values = values[~np.isnan(values)]

            if len(valid_values) > 0:
                # Calculate z-scores
                z_scores = stats.zscore(values, nan_policy='omit')
                data['z_scores'] = z_scores

                # Calculate trend and R² for z-scores
                trend_line, r2, slope = self.calculate_trend_and_r2(dates, z_scores)
                data['trend_line'] = trend_line
                data['r2'] = r2
                data['slope'] = slope

                print(f"{index_name}: Mean={np.nanmean(values):.3f}, Std={np.nanstd(values):.3f}, R²={r2:.3f}")
            else:
                data['z_scores'] = np.full_like(values, np.nan)
                data['trend_line'] = np.full_like(values, np.nan)
                data['r2'] = np.nan
                data['slope'] = np.nan
                print(f"{index_name}: No valid data for z-score calculation")

    def calculate_trend_and_r2(self, dates, values):
        """Calculate linear trend and R² value for time series"""
        # Convert dates to numeric values (years since first date)
        valid_idx = ~np.isnan(values)
        if np.sum(valid_idx) < 3:
            return np.full_like(values, np.nan), np.nan, np.nan

        valid_dates = np.array(dates)[valid_idx]
        valid_values = values[valid_idx]

        # Convert dates to years since start
        start_year = valid_dates[0].year
        x_numeric = np.array([d.year - start_year for d in valid_dates])

        # Fit linear regression
        model = LinearRegression()
        model.fit(x_numeric.reshape(-1, 1), valid_values)

        # Calculate trend line for all dates
        all_x_numeric = np.array([d.year - start_year for d in dates])
        trend_line = model.predict(all_x_numeric.reshape(-1, 1))

        # Calculate R²
        predicted_valid = model.predict(x_numeric.reshape(-1, 1))
        r2 = r2_score(valid_values, predicted_valid)
        slope = model.coef_[0]

        return trend_line, r2, slope

    def create_visualization(self):
        """Create the temporal trends visualization with yearly data"""
        print("\nCreating visualization...")

        # Filter out indices with no valid data
        valid_indices = {name: data for name, data in self.processed_data.items()
                         if not np.all(np.isnan(data['z_scores']))}

        if not valid_indices:
            print("No valid data to plot!")
            return None

        # Set up the plot - 6 separate subplots
        n_indices = len(valid_indices)
        fig, axes = plt.subplots(n_indices, 1, figsize=(15, 3 * n_indices), sharex=True)

        # If only one subplot, make axes a list for consistency
        if n_indices == 1:
            axes = [axes]

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, (index_name, data) in enumerate(valid_indices.items()):
            ax = axes[i]
            dates = data['dates']
            z_scores = data['z_scores']
            trend_line = data['trend_line']
            r2 = data['r2']
            slope = data['slope']

            color = colors[i % len(colors)]

            # Plot the z-scores as filled area with line
            ax.plot(dates, z_scores, color=color, linewidth=2, alpha=0.8, marker='o', markersize=4)
            ax.fill_between(dates, z_scores, 0, color=color, alpha=0.3)

            # Plot the trend line
            ax.plot(dates, trend_line, color='black', linewidth=2, linestyle='--', alpha=0.8)

            # Add horizontal line at zero
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.5)
            ax.grid(True, alpha=0.3)

            # Calculate statistics for display
            mean_z = np.nanmean(z_scores)
            std_z = np.nanstd(z_scores)

            # Set title for each subplot with statistics
            ax.set_title(f'{index_name} - Mean: {mean_z:.2f}, Std Dev: {std_z:.2f}, R²: {r2:.3f}',
                         fontsize=12, pad=10)

            # Set y-label
            ax.set_ylabel(f'{index_name}\n(Z-score)', fontsize=10)

            # Add statistics box in top left
            stats_text = f'Mean: {mean_z:.2f}\nStd Dev: {std_z:.2f}\nR²: {r2:.3f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=props)

            # Format y-axis to show reasonable range
            y_min, y_max = np.nanmin(z_scores), np.nanmax(z_scores)
            y_range = y_max - y_min
            if y_range > 0:
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        # Set main title
        fig.suptitle('Temporal Trends of Climate-Eco Indicators (2001-2021)\nYearly Aggregated Data',
                     fontsize=16, y=0.98)

        # Set x-label only on bottom subplot
        axes[-1].set_xlabel('Year', fontsize=12)

        # Format x-axis
        if valid_indices:
            first_data = next(iter(valid_indices.values()))
            axes[-1].set_xlim(first_data['dates'][0], first_data['dates'][-1])

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for suptitle

        return fig

    def run_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting California Drought Indices Analysis (Yearly Aggregation)...")
        print("=" * 60)

        try:
            # Process all data
            self.process_all_indices()

            # Calculate z-scores and trends
            self.calculate_z_scores_and_trends()

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
    processor = CaliforniaDroughtProcessor()
    fig, processed_data = processor.run_analysis()

    # Print summary statistics
    if processed_data:
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS (YEARLY AGGREGATION)")
        print("=" * 60)

        for index_name, data in processed_data.items():
            values = data['values']
            z_scores = data['z_scores']
            dates = data['dates']
            r2 = data['r2']
            slope = data['slope']

            print(f"\n{index_name}:")
            print(f"  Time period: {dates[0].year} to {dates[-1].year}")
            print(f"  Data points: {len(values)} years")
            print(f"  Raw data - Mean: {np.nanmean(values):.4f}, Std: {np.nanstd(values):.4f}")
            print(f"  Z-scores - Mean: {np.nanmean(z_scores):.4f}, Std: {np.nanstd(z_scores):.4f}")
            print(f"  Trend - R²: {r2:.3f}, Slope: {slope:.4f}")
            print(f"  Min Z-score: {np.nanmin(z_scores):.2f}, Max Z-score: {np.nanmax(z_scores):.2f}")