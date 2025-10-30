import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import rasterio
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')


class EnvironmentalDendrogramAnalysis:
    def __init__(self, base_data_path):
        """Initialize dendrogram analysis with raw TIFF data"""
        self.base_data_path = base_data_path
        self.land_types = ['CALIFORNIA GRASSLANDS', 'CALIFORNIA NATURAL FORESTS', 'CALIFORNIA SAVANNAS',
                           'CALIFORNIA SHRUBLANDS']
        self.land_type_clean = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
        self.indices = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

        # Color mapping for land types
        self.colors = {
            'Grasslands': '#4CAF50',
            'NaturalForests': '#2196F3',
            'Savannas': '#FF9800',
            'Shrublands': '#9C27B0'
        }

        # Distance metrics to test
        self.distance_metrics = ['euclidean', 'correlation', 'cosine', 'manhattan']

        # Linkage methods to test
        self.linkage_methods = ['ward', 'complete', 'average', 'single']

        self.data = {}
        self.standardized_data = None

    def parse_period_from_name(self, name: str):
        """Parse date from filename"""
        patterns = [
            re.compile(r"(?P<y>20\d{2})[_-](?P<m>0[1-9]|1[0-2])"),
            re.compile(r"(?P<y>20\d{2})(?P<m>0[1-9]|1[0-2])"),
        ]
        for pat in patterns:
            m = pat.search(name)
            if m:
                y, mo = int(m.group("y")), int(m.group("m"))
                return pd.Period(f"{y}-{mo:02d}", freq="M")
        return None

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
        except Exception:
            return None

    def collect_timeseries_data(self):
        """Collect time series data from raw TIFF files"""
        print("Collecting time series data from TIFF files...")

        # Date range
        months = pd.period_range("2001-01", "2021-12", freq="M")

        for i, land_type_folder in enumerate(self.land_types):
            land_type_clean = self.land_type_clean[i]
            print(f"\nProcessing {land_type_clean}...")

            land_data = pd.DataFrame(index=months, columns=self.indices, dtype=float)

            for index_name in self.indices:
                index_path = os.path.join(self.base_data_path, land_type_folder, index_name)

                if not os.path.exists(index_path):
                    print(f"  Warning: {index_name} folder not found for {land_type_clean}")
                    continue

                # Get all TIFF files
                tiff_files = [f for f in os.listdir(index_path) if f.endswith('.tif')]
                successful_extractions = 0

                for tiff_file in tiff_files:
                    period = self.parse_period_from_name(tiff_file)
                    if period is None or period not in months:
                        continue

                    file_path = os.path.join(index_path, tiff_file)
                    value = self.load_raster_data(file_path)

                    if value is not None and np.isfinite(value):
                        land_data.loc[period, index_name] = value
                        successful_extractions += 1

                print(f"    {index_name}: {successful_extractions} files processed")

            # Store the data
            self.data[land_type_clean] = land_data
            print(f"  Completed {land_type_clean}: {land_data.shape[0]} months × {land_data.shape[1]} indices")

        print(f"\nSuccessfully collected data for {len(self.data)} land types")

    def prepare_data_for_clustering(self, method='correlation'):
        """Prepare data for clustering analysis"""
        print(f"\nPreparing data for clustering using {method} approach...")

        if method == 'correlation':
            # Method 1: Use correlation patterns between indices
            feature_vectors = []
            land_type_labels = []

            for land_type, df in self.data.items():
                # Remove columns with all NaN
                df_clean = df.dropna(axis=1, how='all')

                if df_clean.shape[1] < 3:  # Need at least 3 indices for meaningful correlations
                    print(f"  Warning: {land_type} has too few valid indices")
                    continue

                # Calculate correlation matrix
                corr_matrix = df_clean.corr()

                # Extract upper triangle (excluding diagonal)
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
                features = corr_matrix.where(mask).stack().values

                # Remove NaN values
                features = features[~np.isnan(features)]

                feature_vectors.append(features)
                land_type_labels.append(land_type)

            # Pad feature vectors to same length (in case different numbers of valid indices)
            max_len = max(len(fv) for fv in feature_vectors)
            padded_vectors = []
            for fv in feature_vectors:
                padded = np.full(max_len, np.nan)
                padded[:len(fv)] = fv
                padded_vectors.append(padded)

            feature_names = [f"corr_feature_{i}" for i in range(max_len)]
            self.clustering_data = pd.DataFrame(padded_vectors,
                                                index=land_type_labels,
                                                columns=feature_names)

        elif method == 'statistical':
            # Method 2: Use statistical summaries
            from scipy.stats import skew, kurtosis

            feature_vectors = []
            land_type_labels = []
            feature_names = []

            for land_type, df in self.data.items():
                features = []

                for col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 10:  # Ensure enough data points
                        features.extend([
                            series.mean(),
                            series.std(),
                            series.median(),
                            series.quantile(0.25),
                            series.quantile(0.75),
                            skew(series),
                            kurtosis(series)
                        ])

                        if land_type == list(self.data.keys())[0]:  # Create feature names once
                            feature_names.extend([
                                f"{col}_mean", f"{col}_std", f"{col}_median",
                                f"{col}_q25", f"{col}_q75", f"{col}_skew", f"{col}_kurt"
                            ])

                feature_vectors.append(features)
                land_type_labels.append(land_type)

            self.clustering_data = pd.DataFrame(feature_vectors,
                                                index=land_type_labels,
                                                columns=feature_names)

        elif method == 'temporal':
            # Method 3: Use temporal patterns
            feature_vectors = []
            land_type_labels = []

            for land_type, df in self.data.items():
                features = []

                for col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 24:  # Need at least 2 years
                        # Monthly climatology
                        monthly_means = series.groupby(series.index.month).mean()
                        features.extend(monthly_means.values)

                        # Annual trend
                        annual_means = series.groupby(series.index.year).mean()
                        if len(annual_means) > 2:
                            annual_trend = np.polyfit(range(len(annual_means)), annual_means.values, 1)[0]
                            features.append(annual_trend)
                        else:
                            features.append(0)

                feature_vectors.append(features)
                land_type_labels.append(land_type)

            # Create feature names
            feature_names = []
            for col in list(self.data.values())[0].columns:
                feature_names.extend([f"{col}_month_{i + 1}" for i in range(12)])
                feature_names.append(f"{col}_trend")

            # Pad to same length if needed
            max_len = max(len(fv) for fv in feature_vectors)
            padded_vectors = []
            for fv in feature_vectors:
                padded = np.zeros(max_len)
                padded[:len(fv)] = fv
                padded_vectors.append(padded)

            self.clustering_data = pd.DataFrame(padded_vectors,
                                                index=land_type_labels,
                                                columns=feature_names[:max_len])

        # Remove columns with all NaN or constant values
        self.clustering_data = self.clustering_data.loc[:, self.clustering_data.var() > 1e-10]

        # Standardize the data
        scaler = StandardScaler()
        self.standardized_data = pd.DataFrame(
            scaler.fit_transform(self.clustering_data.fillna(0)),  # Fill NaN with 0 after standardization
            index=self.clustering_data.index,
            columns=self.clustering_data.columns
        )

        print(
            f"Prepared clustering data: {self.clustering_data.shape[0]} samples × {self.clustering_data.shape[1]} features")
        return self.clustering_data

    def create_dendrograms(self, save_path='dendrograms_analysis.png'):
        """Create comprehensive dendrogram analysis"""

        # Test different combinations of distance metrics and linkage methods
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hierarchical Clustering Analysis: California Land Type Similarities\n' +
                     'Based on Environmental Index Patterns (2001-2021)',
                     fontsize=14, fontweight='bold')

        best_combinations = []

        # Test key combinations
        test_combinations = [
            ('euclidean', 'ward'),
            ('correlation', 'complete'),
            ('euclidean', 'complete'),
            ('cosine', 'average')
        ]

        for idx, (distance_metric, linkage_method) in enumerate(test_combinations):
            ax = axes.flatten()[idx]

            try:
                # Calculate distance matrix
                if distance_metric == 'correlation':
                    # For correlation distance, use 1 - correlation
                    corr_matrix = self.standardized_data.T.corr()
                    distance_matrix = 1 - corr_matrix
                    distances = squareform(distance_matrix, checks=False)
                else:
                    distances = pdist(self.standardized_data, metric=distance_metric)

                # Perform hierarchical clustering
                linkage_matrix = linkage(distances, method=linkage_method)

                # Create dendrogram
                dendrogram(linkage_matrix,
                           labels=self.standardized_data.index,
                           ax=ax,
                           color_threshold=0,
                           above_threshold_color='gray',
                           leaf_rotation=45)

                # Customize appearance
                ax.set_title(f'{distance_metric.title()} + {linkage_method.title()}',
                             fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=10)
                ax.tick_params(axis='y', labelsize=10)
                ax.grid(True, alpha=0.3)

                # Color the labels
                xlbls = ax.get_xmajorticklabels()
                for lbl in xlbls:
                    land_type = lbl.get_text()
                    if land_type in self.colors:
                        lbl.set_color(self.colors[land_type])
                        lbl.set_fontweight('bold')

                # Calculate silhouette score
                try:
                    clusters = fcluster(linkage_matrix, t=2, criterion='maxclust')
                    if len(set(clusters)) > 1:
                        sil_score = silhouette_score(self.standardized_data, clusters)
                        best_combinations.append((distance_metric, linkage_method, sil_score))

                        ax.text(0.02, 0.98, f'Silhouette: {sil_score:.3f}',
                                transform=ax.transAxes, fontsize=9,
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                verticalalignment='top')
                except:
                    pass

            except Exception as e:
                ax.text(0.5, 0.5, f'Error:\n{str(e)[:50]}...',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral'))
                ax.set_title(f'{distance_metric.title()} + {linkage_method.title()}',
                             fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Print best combinations
        if best_combinations:
            best_combinations.sort(key=lambda x: x[2], reverse=True)
            print(f"\nClustering results (by silhouette score):")
            for i, (dist, link, score) in enumerate(best_combinations):
                print(f"  {i + 1}. {dist} + {link}: {score:.3f}")

        return best_combinations

    def create_optimal_dendrogram(self, distance_metric='euclidean', linkage_method='ward',
                                  save_path='optimal_dendrogram.png'):
        """Create detailed dendrogram with optimal parameters"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Calculate distances and linkage
        if distance_metric == 'correlation':
            corr_matrix = self.standardized_data.T.corr()
            distance_matrix = 1 - corr_matrix
            distances = squareform(distance_matrix, checks=False)
        else:
            distances = pdist(self.standardized_data, metric=distance_metric)

        linkage_matrix = linkage(distances, method=linkage_method)

        # Main dendrogram
        dendrogram_plot = dendrogram(linkage_matrix,
                                     labels=self.standardized_data.index,
                                     ax=ax1,
                                     color_threshold=None,
                                     above_threshold_color='black',
                                     leaf_rotation=45,
                                     leaf_font_size=12)

        ax1.set_title(
            f'California Land Types Hierarchical Clustering\n{distance_metric.title()} Distance + {linkage_method.title()} Linkage',
            fontsize=14, fontweight='bold')
        ax1.set_xlabel('Land Types', fontsize=12)
        ax1.set_ylabel('Distance', fontsize=12)
        ax1.grid(True, alpha=0.3)

        # Color the labels
        xlbls = ax1.get_xmajorticklabels()
        for lbl in xlbls:
            land_type = lbl.get_text()
            if land_type in self.colors:
                lbl.set_color(self.colors[land_type])
                lbl.set_fontweight('bold')

        # Distance matrix heatmap
        if distance_metric == 'correlation':
            distance_df = pd.DataFrame(distance_matrix,
                                       index=self.standardized_data.index,
                                       columns=self.standardized_data.index)
        else:
            distance_matrix_full = squareform(distances)
            distance_df = pd.DataFrame(distance_matrix_full,
                                       index=self.standardized_data.index,
                                       columns=self.standardized_data.index)

        sns.heatmap(distance_df, annot=True, cmap='viridis',
                    square=True, ax=ax2, fmt='.3f',
                    cbar_kws={'label': f'{distance_metric.title()} Distance'})
        ax2.set_title('Pairwise Distance Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # Analysis of clusters
        print(f"\nDendrogram Analysis Results:")
        print(f"Distance Metric: {distance_metric}")
        print(f"Linkage Method: {linkage_method}")

        # Try different numbers of clusters
        for n_clusters in [2, 3, 4]:
            clusters = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')

            print(f"\nWith {n_clusters} clusters:")
            cluster_dict = {}
            for land_type, cluster in zip(self.standardized_data.index, clusters):
                if cluster not in cluster_dict:
                    cluster_dict[cluster] = []
                cluster_dict[cluster].append(land_type)

            for cluster_id, members in cluster_dict.items():
                print(f"  Cluster {cluster_id}: {', '.join(members)}")

            if len(set(clusters)) > 1 and len(set(clusters)) < len(self.standardized_data):
                sil_score = silhouette_score(self.standardized_data, clusters)
                print(f"  Silhouette Score: {sil_score:.3f}")
            else:
                print(f"  Silhouette Score: Not applicable (each point is its own cluster)")

    def run_comprehensive_analysis(self, clustering_method='correlation'):
        """Run complete dendrogram analysis"""
        print("=" * 80)
        print("ENVIRONMENTAL DENDROGRAM ANALYSIS")
        print("=" * 80)

        # Collect raw data
        self.collect_timeseries_data()

        # Prepare data for clustering
        self.prepare_data_for_clustering(method=clustering_method)

        if self.standardized_data is None or self.standardized_data.empty:
            print("Error: No data available for clustering analysis")
            return

        # Create comprehensive dendrogram comparison
        print(f"\nCreating dendrogram analysis...")
        best_combinations = self.create_dendrograms()

        # Create optimal dendrogram
        if best_combinations:
            best_dist, best_link, best_score = best_combinations[0]
            print(f"\nCreating optimal dendrogram with {best_dist} + {best_link}...")
            self.create_optimal_dendrogram(best_dist, best_link)
        else:
            print(f"\nCreating dendrogram with default parameters...")
            self.create_optimal_dendrogram()

        print("\n" + "=" * 80)
        print("DENDROGRAM ANALYSIS COMPLETE!")
        print("=" * 80)
        print("Generated:")
        print("• Hierarchical clustering comparison across methods")
        print("• Optimal dendrogram with best-performing parameters")
        print("• Distance matrix heatmap showing pairwise similarities")
        print("• Cluster validation using silhouette scores")
        print("=" * 80)


# Usage
if __name__ == "__main__":
    # Set path to your data directory
    BASE_DATA_PATH = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"

    # Initialize analyzer
    analyzer = EnvironmentalDendrogramAnalysis(BASE_DATA_PATH)

    # Run comprehensive analysis
    # Try different clustering methods: 'correlation', 'statistical', 'temporal'
    analyzer.run_comprehensive_analysis(clustering_method='correlation')