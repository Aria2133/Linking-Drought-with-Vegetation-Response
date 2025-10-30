import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import rasterio
import warnings

warnings.filterwarnings('ignore')


class DroughtIndicesBoxplotAnalysis:
    def __init__(self, base_path):
        """Initialize the boxplot analysis"""
        self.base_path = base_path
        self.indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPDmax']
        self.land_types = ['GRASSLANDS', 'NATURAL FORESTS', 'SAVANNAS', 'SHRUBLANDS']
        self.land_type_mapping = {
            'GRASSLANDS': 'Grasslands',
            'NATURAL FORESTS': 'Natural Forests',
            'SAVANNAS': 'Savannas',
            'SHRUBLANDS': 'Shrublands'
        }

        # Updated colors for each land type - more vibrant and distinct
        self.colors = {
            'Grasslands': '#4CAF50',  # Green
            'Natural Forests': '#2196F3',  # Blue
            'Savannas': '#FF9800',  # Orange
            'Shrublands': '#9C27B0'  # Purple
        }

        # File patterns for each index
        self.file_patterns = {
            'GNDVI': 'GNDVI_{}_GNDVI_MOD09A1_CA_5km_{}',
            'GOSIF': 'GOSIF_{}_CALIFORNIA_GOSIF_{}',
            'GPP': '{}_GPP_{}',
            'LST': 'LST_{}_MYD21C1_CA_LST_{}',
            'NDVI': 'NDVI_{}_NDVI_MOD13Q1_CA_5km_{}',
            'NIRv': 'NIRv_{}_NIRv_MOD13Q1_CA_5km_{}',
            'SPEI': 'SPEI_{}_SPEI1_CSIC_CA_{}',
            'TWS': 'TWS_{}_TWS_GRACE_CA_{}',
            'VPDmax': 'VPD_{}_VPDmax_GRIDMET_CA_5km_{}'
        }

        # Folder mapping
        self.folder_mapping = {
            'GNDVI': 'GNDVI', 'GOSIF': 'GOSIF', 'GPP': 'GPP', 'LST': 'LST',
            'NDVI': 'NDVI', 'NIRv': 'NIRv', 'SPEI': 'SPEI', 'TWS': 'TWS', 'VPDmax': 'VPD'
        }

        # Special paths for GOSIF
        self.gosif_paths = {
            'GRASSLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA GRASSLANDS\GOSIF",
            'NATURAL FORESTS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA NATURAL FORESTS\GOSIF",
            'SAVANNAS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SAVANNAS\GOSIF",
            'SHRUBLANDS': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CALIFORNIA SHRUBLANDS\GOSIF"
        }

        # Start years for each index
        self.start_years = {
            'GNDVI': 2001, 'GOSIF': 2001, 'GPP': 2001, 'LST': 2003,
            'NDVI': 2001, 'NIRv': 2001, 'SPEI': 2001, 'TWS': 2002, 'VPDmax': 2001
        }

        # Units for each index
        self.units = {
            'GNDVI': 'GNDVI',
            'GOSIF': 'SIF (mW m⁻² nm⁻¹ sr⁻¹)',
            'GPP': 'gC m⁻² d⁻¹',
            'LST': 'Temperature (°C)',
            'NDVI': 'NDVI',
            'NIRv': 'NIRv',
            'SPEI': 'SPEI',
            'TWS': 'Water Equivalent (cm)',
            'VPDmax': 'VPD (kPa)'
        }

    def load_raster_data(self, file_path):
        """Load raster data from TIFF file"""
        try:
            if not os.path.exists(file_path):
                return None
            with rasterio.open(file_path) as src:
                data = src.read(1)
                data = data.astype(float)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                return np.nanmean(data)
        except Exception as e:
            return None

    def find_actual_filename(self, folder_path, expected_pattern, date_str, index_name):
        """Find the actual filename that matches the pattern"""
        if not os.path.exists(folder_path):
            return None

        files_in_folder = os.listdir(folder_path)
        exact_file = expected_pattern + '.tif'
        if exact_file in files_in_folder:
            return os.path.join(folder_path, exact_file)

        # Special handling for different indices
        if index_name == 'GOSIF':
            gosif_date = date_str.replace('_', '')
            for file in files_in_folder:
                if file.endswith('.tif') and gosif_date in file:
                    return os.path.join(folder_path, file)

        if index_name == 'GPP':
            year = date_str[:4]
            month = date_str[5:7]
            for file in files_in_folder:
                if (file.endswith('.tif') and 'GPP' in file and year in file and month in file):
                    return os.path.join(folder_path, file)

        # General pattern matching
        year = date_str[:4]
        month = date_str[5:7]
        year_month_patterns = [f"{year}_{month}", f"{year}{month}", f"{year}-{month}"]

        for file in files_in_folder:
            if file.endswith('.tif'):
                for pattern in year_month_patterns:
                    if pattern in file:
                        return os.path.join(folder_path, file)
        return None

    def get_land_type_for_filename(self, land_type, index_name):
        """Get the correct land type name for filename"""
        if index_name == 'GOSIF':
            if land_type == 'SAVANNAS':
                return 'Savanna'
            elif land_type == 'NATURAL FORESTS':
                return 'NaturalForests'
            else:
                return {
                    'GRASSLANDS': 'Grasslands',
                    'SHRUBLANDS': 'Shrublands'
                }[land_type]
        else:
            return {
                'GRASSLANDS': 'Grasslands',
                'NATURAL FORESTS': 'NaturalForests',
                'SAVANNAS': 'Savannas',
                'SHRUBLANDS': 'Shrublands'
            }[land_type]

    def generate_date_range(self, index_name, start_year=2001, end_year=2021):
        """Generate date range for time series extraction"""
        actual_start = max(start_year, self.start_years[index_name])
        actual_end = end_year

        dates = []
        for year in range(actual_start, actual_end + 1):
            for month in range(1, 13):
                if index_name == 'TWS' and year == 2002 and month < 4:
                    continue
                dates.append(f"{year:04d}_{month:02d}")
        return dates

    def extract_data_for_boxplots(self):
        """Extract all time series data for boxplot analysis"""
        print("Extracting data for boxplot analysis...")

        all_data = {}

        for index_name in self.indices:
            print(f"\nProcessing {index_name}...")

            index_data = {}

            for land_type in self.land_types:
                land_type_clean = self.land_type_mapping[land_type]
                print(f"  Extracting {land_type_clean}...")

                # Generate dates for this index
                date_strings = self.generate_date_range(index_name)

                values = []
                successful_extractions = 0

                for date_str in date_strings:
                    # Build file path
                    if index_name == 'GOSIF':
                        folder_path = self.gosif_paths[land_type]
                    else:
                        land_folder = f"CALIFORNIA {land_type}"
                        index_folder = self.folder_mapping[index_name]
                        folder_path = os.path.join(self.base_path, land_folder, index_folder)

                    # Generate filename
                    land_type_for_file = self.get_land_type_for_filename(land_type, index_name)
                    expected_filename = self.file_patterns[index_name].format(land_type_for_file, date_str)

                    # Find and load file
                    file_path = self.find_actual_filename(folder_path, expected_filename, date_str, index_name)

                    if file_path:
                        avg_value = self.load_raster_data(file_path)
                        if avg_value is not None and not np.isnan(avg_value):
                            values.append(avg_value)
                            successful_extractions += 1

                print(f"    Successfully extracted {successful_extractions} data points")
                index_data[land_type_clean] = values

            all_data[index_name] = index_data

        return all_data

    def calculate_statistics(self, data):
        """Calculate comprehensive statistics for boxplot data"""
        if not data:
            return {}

        data_array = np.array(data)
        stats = {
            'n': len(data_array),
            'mean': np.mean(data_array),
            'median': np.median(data_array),
            'std': np.std(data_array),
            'min': np.min(data_array),
            'max': np.max(data_array),
            'q25': np.percentile(data_array, 25),
            'q75': np.percentile(data_array, 75),
            'iqr': np.percentile(data_array, 75) - np.percentile(data_array, 25),
            'cv': np.std(data_array) / np.mean(data_array) if np.mean(data_array) != 0 else np.inf
        }
        return stats

    def create_boxplot_grid(self, all_data, save_path='drought_indices_boxplots.png'):
        """Create 3x3 grid of boxplots for all indices with individual CV values"""

        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        axes = axes.flatten()

        fig.suptitle('Drought Indices Distribution Variations Across California Land Types\n(2001-2021 Monthly Data)',
                     fontsize=16, fontweight='bold', y=0.98)

        # Color palette for land types
        land_type_colors = [self.colors[lt] for lt in self.land_type_mapping.values()]

        # Process each index
        for i, index_name in enumerate(self.indices):
            ax = axes[i]

            if index_name not in all_data or not any(all_data[index_name].values()):
                ax.text(0.5, 0.5, f'No data available\nfor {index_name}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgray'))
                ax.set_title(f'{index_name}', fontweight='bold', fontsize=14)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            # Prepare data for boxplot
            plot_data = []
            plot_labels = []
            valid_land_types = []
            cv_values = []

            for land_type in self.land_type_mapping.values():
                if land_type in all_data[index_name] and len(all_data[index_name][land_type]) > 0:
                    data = all_data[index_name][land_type]
                    plot_data.append(data)

                    # Calculate CV for this land type
                    stats = self.calculate_statistics(data)
                    cv_percent = stats['cv'] * 100 if stats['cv'] != np.inf else 0
                    cv_values.append(cv_percent)

                    # Create label with CV
                    short_name = land_type.replace(' ', '\n')
                    label_with_cv = f"{short_name}\nCV: {cv_percent:.1f}%"
                    plot_labels.append(label_with_cv)
                    valid_land_types.append(land_type)

            if not plot_data:
                ax.text(0.5, 0.5, f'No valid data\nfor {index_name}',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=12)
                ax.set_title(f'{index_name}', fontweight='bold', fontsize=14)
                continue

            # Create boxplot with custom colors
            valid_colors = [self.colors[lt] for lt in valid_land_types]

            bp = ax.boxplot(plot_data,
                            labels=plot_labels,
                            patch_artist=True,
                            notch=True,  # Add notches for median confidence intervals
                            showmeans=True,  # Show means as well as medians
                            meanprops={'marker': 'D', 'markerfacecolor': 'red', 'markeredgecolor': 'red',
                                       'markersize': 4})

            # Color the boxes
            for patch, color in zip(bp['boxes'], valid_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Customize other elements
            for element in ['whiskers', 'caps', 'medians']:
                for item in bp[element]:
                    item.set_color('black')
                    item.set_linewidth(1.5)

            # Set title with index name
            ax.set_title(f'{index_name}', fontweight='bold', fontsize=14)

            # Format axes
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=9)

            # Add sample size info in top-right corner
            total_points = sum(len(data) for data in plot_data)
            ax.text(0.98, 0.98, f'n={total_points}',
                    transform=ax.transAxes, fontsize=8,
                    horizontalalignment='right', verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.8))

            # Add CV range in top-left corner if there are multiple land types
            if len(cv_values) > 1:
                cv_range = f'CV range: {min(cv_values):.1f}-{max(cv_values):.1f}%'
                ax.text(0.02, 0.98, cv_range,
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.93, bottom=0.08, hspace=0.4, wspace=0.3)

        # Add legend
        legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=land_type)
                           for land_type, color in self.colors.items()]
        fig.legend(handles=legend_elements, loc='lower center', ncol=4,
                   bbox_to_anchor=(0.5, 0.04), fontsize=11)

        # Add interpretation note
        fig.text(0.02, 0.01,
                 'Box: IQR (25th-75th percentiles), Line: Median, Diamond: Mean, Whiskers: 1.5×IQR, Dots: Outliers\n'
                 'CV: Coefficient of Variation (%) for each land type, n: Total monthly observations',
                 fontsize=9, style='italic')

        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Boxplot grid saved as {save_path}")

    def print_detailed_statistics(self, all_data):
        """Print comprehensive statistics for each index and land type"""
        print("\n" + "=" * 100)
        print("DETAILED STATISTICS SUMMARY")
        print("=" * 100)

        for index_name in self.indices:
            if index_name not in all_data:
                continue

            print(f"\n{index_name} ({self.units.get(index_name, '')}):")
            print("-" * 80)

            # Calculate statistics for each land type
            land_type_stats = {}
            for land_type, data in all_data[index_name].items():
                if len(data) > 0:
                    stats = self.calculate_statistics(data)
                    land_type_stats[land_type] = stats

                    print(f"\n  {land_type}:")
                    print(f"    Sample size: {stats['n']}")
                    print(f"    Mean ± SD: {stats['mean']:.4f} ± {stats['std']:.4f}")
                    print(f"    Median [IQR]: {stats['median']:.4f} [{stats['q25']:.4f}-{stats['q75']:.4f}]")
                    print(f"    Range: {stats['min']:.4f} to {stats['max']:.4f}")
                    print(f"    Coefficient of Variation: {stats['cv'] * 100:.2f}%")

            # Compare across land types
            if len(land_type_stats) > 1:
                print(f"\n  Cross-Land Type Comparison:")
                means = [stats['mean'] for stats in land_type_stats.values()]
                cvs = [stats['cv'] * 100 for stats in land_type_stats.values()]

                highest_mean = max(land_type_stats.items(), key=lambda x: x[1]['mean'])
                lowest_mean = min(land_type_stats.items(), key=lambda x: x[1]['mean'])
                most_variable = max(land_type_stats.items(), key=lambda x: x[1]['cv'])
                least_variable = min(land_type_stats.items(), key=lambda x: x[1]['cv'])

                print(f"    Highest mean: {highest_mean[0]} ({highest_mean[1]['mean']:.4f})")
                print(f"    Lowest mean: {lowest_mean[0]} ({lowest_mean[1]['mean']:.4f})")
                print(f"    Most variable: {most_variable[0]} (CV = {most_variable[1]['cv'] * 100:.2f}%)")
                print(f"    Least variable: {least_variable[0]} (CV = {least_variable[1]['cv'] * 100:.2f}%)")
                print(f"    Mean coefficient of variation across land types: {np.mean(cvs):.2f}%")

                # Calculate ratio of highest to lowest mean
                ratio = highest_mean[1]['mean'] / lowest_mean[1]['mean'] if lowest_mean[1]['mean'] != 0 else float(
                    'inf')
                print(f"    Range ratio (highest/lowest mean): {ratio:.2f}")

    def run_analysis(self):
        """Run complete boxplot analysis"""
        print("=" * 80)
        print("DROUGHT INDICES BOXPLOT VARIATION ANALYSIS")
        print("=" * 80)
        print("This analysis creates boxplots showing distribution variations")
        print("for all 9 drought indices across 4 California land types.")
        print("Individual CV values are shown for each land type.")
        print("=" * 80)

        # Extract all data
        all_data = self.extract_data_for_boxplots()

        # Create boxplot grid
        self.create_boxplot_grid(all_data)

        # Print detailed statistics
        self.print_detailed_statistics(all_data)

        print("\n" + "=" * 80)
        print("BOXPLOT ANALYSIS COMPLETE!")
        print("=" * 80)
        print("Generated visualization shows:")
        print("• Distribution shapes and central tendencies for each drought index")
        print("• Individual CV values for each land type within each index")
        print("• Variability differences across California land types")
        print("• Outlier identification and frequency")
        print("• Statistical summaries including medians, quartiles, and means")
        print("• Sample sizes and data completeness metrics")
        print("=" * 80)

        return all_data


# Usage
if __name__ == "__main__":
    # Set your base path
    BASE_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

    # Initialize analyzer
    analyzer = DroughtIndicesBoxplotAnalysis(BASE_PATH)

    # Run analysis
    results = analyzer.run_analysis()