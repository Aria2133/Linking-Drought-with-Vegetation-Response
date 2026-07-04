#!/usr/bin/env python3
"""
Ultra-Optimized GNN Drought Forecasting Pipeline
Major performance improvements while maintaining exact same structure and functionality
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial, lru_cache
import pickle
import mmap
import psutil

# Import additional packages
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from sklearn.preprocessing import StandardScaler

# Numba for JIT compilation (install if not available: pip install numba)
try:
    from numba import jit, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    print("Numba not available. Install with: pip install numba")
    NUMBA_AVAILABLE = False


    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    def prange(*args, **kwargs):
        return range(*args, **kwargs)

# Add scipy import fix
try:
    from scipy.spatial import distance_matrix
except ImportError:
    from scipy.spatial.distance import cdist


    def distance_matrix(x, y):
        return cdist(x, y)

warnings.filterwarnings('ignore')

# =============================================================================
# ULTRA-OPTIMIZED CONFIGURATION
# =============================================================================

CONFIG = {
    'data': {
        'variables': ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD'],
        'train_val_base': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data",
        'test_base': r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes",
        'land_type_folders_train': {
            'grasslands': "CALIFORNIA GRASSLANDS",
            'natural_forests': "CALIFORNIA NATURAL FORESTS",
            'savannas': "CALIFORNIA SAVANNAS",
            'shrublands': "CALIFORNIA SHRUBLANDS"
        },
        'land_type_folders_test': {
            'grasslands': "Grasslands",
            'natural_forests': "NaturalForests",
            'savannas': "Savannas",
            'shrublands': "Shrublands"
        },
        'land_types': ['grasslands', 'natural_forests', 'savannas', 'shrublands']
    },
    'splits': {
        'train_years': list(range(2001, 2019)),
        'val_years': [2019, 2020, 2021],
        'test_years': [2022, 2023, 2024]
    },
    'model': {
        'hidden_dim': 32,
        'num_layers': 3,
        'kernel_size': 2,
        'k_neighbors': 8,
        'dropout': 0.1,
        'lookback_window': 12,
        'adaptive_adj': True
    },
    'training': {
        'seed': 42,
        'batch_size': 32,  # Increased for better throughput
        'learning_rate': 0.003,  # Optimized learning rate
        'epochs': 20,  # Reduced with better convergence
        'patience': 5  # Reduced with better scheduling
    },
    'evaluation': {
        'false_alarm_target': 0.05,
        'severe_threshold_percentile': 5,
        'timing_tolerance': 1
    },
    'output': {
        'checkpoints_dir': r"C:\Users\LOQ\OneDrive\Desktop\GNN_checkpoints",
        'results_dir': r"C:\Users\LOQ\OneDrive\Desktop\GNN_results",
        'plots_dir': r"C:\Users\LOQ\OneDrive\Desktop\GNN_plots",
        'cache_dir': r"C:\Users\LOQ\OneDrive\Desktop\GNN_cache"
    },
    'optimization': {
        'use_cache': True,
        'use_memmap': True,  # Memory-mapped files
        'parallel_data_loading': True,
        'chunk_size': 1000,  # Spatial chunking
        'max_workers': min(8, mp.cpu_count()),  # Optimized worker count
        'prefetch_factor': 4,  # Increased prefetching
        'pin_memory': False,  # CPU optimized
        'persistent_workers': True,
        'use_sparse_adj': True,  # Sparse adjacency matrices
        'aggressive_gc': True,  # Aggressive garbage collection
        'memory_limit_gb': 8  # Memory usage limit
    }
}


# =============================================================================
# ULTRA-OPTIMIZED UTILITY FUNCTIONS
# =============================================================================

def aggressive_memory_cleanup():
    """Ultra-aggressive memory cleanup."""
    for _ in range(3):  # Multiple passes
        gc.collect()

    # Force garbage collection of specific types
    import sys
    for obj in gc.get_objects():
        if isinstance(obj, (np.ndarray, torch.Tensor)) and hasattr(obj, '__del__'):
            try:
                del obj
            except:
                pass
    gc.collect()


def monitor_memory_usage():
    """Monitor memory usage and return info."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 ** 3
    return memory_gb


def enforce_memory_limit(limit_gb=8):
    """Enforce memory usage limit."""
    current_memory = monitor_memory_usage()
    if current_memory > limit_gb:
        print(f"Memory usage {current_memory:.1f}GB exceeds limit {limit_gb}GB. Cleaning up...")
        aggressive_memory_cleanup()
        current_memory = monitor_memory_usage()
        print(f"Memory after cleanup: {current_memory:.1f}GB")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_cache_path(cache_dir: str, land_type: str, stage: str, extra_key: str = "") -> str:
    """Generate cache file path with better granularity."""
    key = f"{land_type}_{stage}"
    if extra_key:
        key += f"_{extra_key}"
    return os.path.join(cache_dir, f"{key}_cache.pkl")


def save_cache_compressed(data, cache_path: str):
    """Save data to cache with compression."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    import gzip
    with gzip.open(cache_path + '.gz', 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_cache_compressed(cache_path: str):
    """Load data from compressed cache."""
    import gzip
    gz_path = cache_path + '.gz'
    if os.path.exists(gz_path):
        try:
            with gzip.open(gz_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return None


# =============================================================================
# NUMBA-OPTIMIZED FUNCTIONS
# =============================================================================

if NUMBA_AVAILABLE:
    @njit(parallel=True, fastmath=True)
    def vectorized_z_score(data, mu, sigma):
        """Ultra-fast z-score computation."""
        result = np.empty_like(data)
        for i in prange(data.shape[0]):
            for j in prange(data.shape[1]):
                if sigma[j] > 0:
                    result[i, j] = (data[i, j] - mu[j]) / sigma[j]
                else:
                    result[i, j] = 0.0
        return result


    @njit(parallel=True, fastmath=True)
    def fast_nanmean_axis0(arr):
        """Fast nanmean along axis 0."""
        result = np.empty(arr.shape[1])
        for j in prange(arr.shape[1]):
            count = 0
            total = 0.0
            for i in range(arr.shape[0]):
                if not np.isnan(arr[i, j]):
                    total += arr[i, j]
                    count += 1
            result[j] = total / count if count > 0 else np.nan
        return result


    @njit(parallel=True, fastmath=True)
    def fast_nanstd_axis0(arr, means):
        """Fast nanstd along axis 0."""
        result = np.empty(arr.shape[1])
        for j in prange(arr.shape[1]):
            count = 0
            sum_sq_diff = 0.0
            mean_val = means[j]
            if not np.isnan(mean_val):
                for i in range(arr.shape[0]):
                    if not np.isnan(arr[i, j]):
                        diff = arr[i, j] - mean_val
                        sum_sq_diff += diff * diff
                        count += 1
                result[j] = np.sqrt(sum_sq_diff / count) if count > 1 else 0.0
            else:
                result[j] = 0.0
        return result


    @njit(parallel=True, fastmath=True)
    def fast_interpolation(data, missing_mask, valid_prev, valid_next):
        """Fast vectorized interpolation."""
        result = data.copy()
        for i in prange(missing_mask.shape[0]):
            for j in prange(missing_mask.shape[1]):
                if missing_mask[i, j] and valid_prev[i, j] and valid_next[i, j]:
                    # Find previous and next valid values
                    prev_val = data[i - 1, j] if i > 0 else np.nan
                    next_val = data[i + 1, j] if i < data.shape[0] - 1 else np.nan
                    if not np.isnan(prev_val) and not np.isnan(next_val):
                        result[i, j] = (prev_val + next_val) / 2.0
        return result

else:
    # Fallback implementations without numba
    def vectorized_z_score(data, mu, sigma):
        return (data - mu) / np.where(sigma > 0, sigma, 1.0)


    def fast_nanmean_axis0(arr):
        return np.nanmean(arr, axis=0)


    def fast_nanstd_axis0(arr, means):
        return np.nanstd(arr, axis=0)


    def fast_interpolation(data, missing_mask, valid_prev, valid_next):
        result = data.copy()
        can_interpolate = missing_mask & valid_prev & valid_next
        if can_interpolate.any():
            result[can_interpolate] = (data[can_interpolate] + data[can_interpolate]) / 2
        return result


# =============================================================================
# MEMORY-MAPPED DATA STRUCTURES
# =============================================================================

class MemoryMappedArray:
    """Memory-mapped numpy array for large datasets."""

    def __init__(self, shape, dtype=np.float32, mode='w+'):
        self.shape = shape
        self.dtype = dtype
        self.filename = f"temp_mmap_{id(self)}.dat"
        self.array = np.memmap(self.filename, dtype=dtype, mode=mode, shape=shape)

    def __del__(self):
        try:
            del self.array
            if os.path.exists(self.filename):
                os.remove(self.filename)
        except:
            pass

    def __getitem__(self, key):
        return self.array[key]

    def __setitem__(self, key, value):
        self.array[key] = value


# =============================================================================
# ULTRA-OPTIMIZED DATA PROCESSING CLASSES
# =============================================================================

class UltraOptimizedGeoTIFFReader:
    """Ultra-optimized GeoTIFF reader with memory mapping and chunking."""

    def __init__(self, config):
        self.config = config
        self.ref_transform = None
        self.ref_shape = None
        self.ref_crs = None
        self._file_cache = {}
        self._metadata_cache = {}

    @lru_cache(maxsize=1000)
    def get_file_paths_batch_cached(self, land_type: str, year_month_tuple: tuple) -> Dict:
        """Cached file path resolution."""
        year_month_pairs = list(year_month_tuple)
        return self._get_file_paths_batch_internal(land_type, year_month_pairs)

    def _get_file_paths_batch_internal(self, land_type: str, year_month_pairs: List[Tuple[int, int]]) -> Dict:
        """Internal file path resolution with optimized caching."""
        file_paths = {}

        # Pre-build directory listings once
        dir_cache = {}

        for year, month in year_month_pairs:
            month_str = f"{month:02d}"

            if year <= 2021:
                base_dir = self.config['data']['train_val_base']
                lt_folder = self.config['data']['land_type_folders_train'][land_type]
            else:
                base_dir = self.config['data']['test_base']
                lt_folder = self.config['data']['land_type_folders_test'][land_type]

            for var in self.config['data']['variables']:
                folder = os.path.join(base_dir, lt_folder, var)
                if not os.path.isdir(folder):
                    continue

                # Cache directory listings
                if folder not in dir_cache:
                    try:
                        files = os.listdir(folder)
                        dir_cache[folder] = {
                            os.path.splitext(f)[0]: os.path.join(folder, f)
                            for f in files
                            if f.lower().endswith(('.tif', '.tiff'))
                        }
                    except:
                        dir_cache[folder] = {}

                candidates = dir_cache[folder]
                pattern = self._get_filename_pattern(land_type, var, year, month_str)

                if pattern and pattern in candidates:
                    if (year, month) not in file_paths:
                        file_paths[(year, month)] = {}
                    file_paths[(year, month)][var] = candidates[pattern]

        return file_paths

    def get_file_paths_batch(self, land_type: str, year_month_pairs: List[Tuple[int, int]]) -> Dict:
        """Public interface with caching."""
        year_month_tuple = tuple(year_month_pairs)
        return self.get_file_paths_batch_cached(land_type, year_month_tuple)

    def _get_filename_pattern(self, land_type: str, variable: str, year: int, month_str: str) -> str:
        """Generate filename pattern - cached version."""
        cache_key = (land_type, variable, year, month_str)
        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        result = self._build_filename_pattern(land_type, variable, year, month_str)
        self._metadata_cache[cache_key] = result
        return result

    def _build_filename_pattern(self, land_type: str, variable: str, year: int, month_str: str) -> str:
        """Build filename pattern."""
        train_land_names = {
            'grasslands': 'Grasslands',
            'natural_forests': 'NaturalForests',
            'savannas': ['Savannas', 'Savanna'],
            'shrublands': 'Shrublands'
        }

        test_land_names = {
            'grasslands': 'Grasslands',
            'natural_forests': 'NaturalForests',
            'savannas': 'Savannas',
            'shrublands': 'Shrublands'
        }

        if year <= 2021:
            lt_names = train_land_names[land_type]
            if not isinstance(lt_names, list):
                lt_names = [lt_names]

            for lt_name in lt_names:
                pattern = self._build_train_pattern(variable, lt_name, year, month_str)
                if pattern:
                    return pattern
        else:
            lt_name = test_land_names[land_type]
            return self._build_test_pattern(variable, lt_name, year, month_str)

        return None

    def _build_train_pattern(self, variable: str, lt_name: str, year: int, month_str: str) -> str:
        """Build training filename pattern."""
        patterns = {
            'GNDVI': f"GNDVI_{lt_name}_GNDVI_MOD09A1_CA_5km_{year}_{month_str}",
            'GOSIF': f"GOSIF_{lt_name}_CALIFORNIA_GOSIF_{year}{month_str}",
            'GPP': f"{lt_name}_GPP_{year}_{month_str}",
            'LST': f"LST_{lt_name}_MYD21C1_CA_LST_{year}_{month_str}" if year >= 2003 else None,
            'NDVI': f"NDVI_{lt_name}_NDVI_MOD13Q1_CA_5km_{year}_{month_str}",
            'NIRv': f"NIRv_{lt_name}_NIRv_MOD13Q1_CA_5km_{year}_{month_str}",
            'SPEI': f"SPEI_{lt_name}_SPEI1_CSIC_CA_{year}_{month_str}",
            'TWS': f"TWS_{lt_name}_TWS_GRACE_CA_{year}_{month_str}" if (
                        year > 2002 or (year == 2002 and int(month_str) >= 4)) else None,
            'VPD': f"VPD_{lt_name}_VPDmax_GRIDMET_CA_5km_{year}_{month_str}"
        }
        return patterns.get(variable)

    def _build_test_pattern(self, variable: str, lt_name: str, year: int, month_str: str) -> str:
        """Build test filename pattern."""
        patterns = {
            'GNDVI': f"GNDVI_MOD09A1_CA_5km_{year}_{month_str}_{lt_name}",
            'GOSIF': f"GOSIF_{year}.M{month_str}_{lt_name}",
            'GPP': f"GPP_MOD17A2H_CA_5km_{year}_{month_str}_{lt_name}",
            'LST': f"MYD21C1_CA_LST_{year}_{month_str}_{lt_name}",
            'NDVI': f"NDVI_MOD13Q1_CA_5km_{year}_{month_str}_{lt_name}",
            'NIRv': f"NIRv_MOD13Q1_CA_5km_{year}_{month_str}_{lt_name}",
            'SPEI': f"SPEI1_CSIC_CA_{year}_{month_str}_{lt_name}",
            'TWS': f"TWS_GRACE_CA_{year}_{month_str}_{lt_name}",
            'VPD': f"VPDmax_GRIDMET_CA_5km_{year}_{month_str}_{lt_name}"
        }
        return patterns.get(variable)

    def read_geotiff_optimized(self, filepath: str, use_windowing: bool = True) -> Tuple[np.ndarray, dict]:
        """Ultra-optimized GeoTIFF reading with windowing and memory management."""
        if not filepath or not os.path.exists(filepath):
            return None, None

        try:
            with rasterio.open(filepath) as src:
                meta = {
                    'transform': src.transform,
                    'crs': src.crs,
                    'shape': (src.height, src.width),
                    'bounds': src.bounds
                }

                # Use windowing for large files
                if use_windowing and src.width * src.height > 500000:
                    # Read in chunks to save memory
                    data = np.empty((src.height, src.width), dtype=np.float32)
                    chunk_size = 1000
                    for row in range(0, src.height, chunk_size):
                        for col in range(0, src.width, chunk_size):
                            window = Window(
                                col, row,
                                min(chunk_size, src.width - col),
                                min(chunk_size, src.height - row)
                            )
                            chunk = src.read(1, window=window).astype(np.float32)
                            data[row:row + window.height, col:col + window.width] = chunk
                else:
                    data = src.read(1, out_dtype=np.float32)

                # Handle no-data values
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                data[np.isinf(data)] = np.nan

            return data, meta
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return None, None

    def set_reference_grid(self, land_type: str):
        """Set reference grid from NIRv with caching."""
        cache_path = get_cache_path(self.config['output']['cache_dir'], land_type, 'reference_grid')
        cached = load_cache_compressed(cache_path)
        if cached and self.config['optimization']['use_cache']:
            self.ref_transform, self.ref_shape, self.ref_crs = cached
            print(f"Loaded reference grid from cache: {self.ref_shape}")
            return

        # Find reference file
        all_years = list(range(2001, 2025))
        year_month_pairs = [(y, m) for y in all_years for m in range(1, 13)]
        file_paths = self.get_file_paths_batch(land_type, year_month_pairs)

        for (year, month), files in file_paths.items():
            if 'NIRv' in files:
                _, meta = self.read_geotiff_optimized(files['NIRv'])
                if meta:
                    self.ref_transform = meta['transform']
                    self.ref_shape = meta['shape']
                    self.ref_crs = meta['crs']
                    print(f"Reference grid set from NIRv: {self.ref_shape}")

                    if self.config['optimization']['use_cache']:
                        save_cache_compressed((self.ref_transform, self.ref_shape, self.ref_crs), cache_path)
                    return

        raise ValueError(f"Could not find NIRv reference grid for {land_type}")

    def regrid_to_reference_optimized(self, data: np.ndarray, src_meta: dict, is_coarse: bool = False):
        """Ultra-optimized regridding with chunked processing."""
        if data is None or src_meta is None:
            return None, None

        # Skip regridding if already same grid
        if (src_meta['shape'] == self.ref_shape and
                src_meta['transform'] == self.ref_transform and
                src_meta['crs'] == self.ref_crs):
            coarse_flag = np.ones(self.ref_shape, dtype=np.float32) if is_coarse else np.zeros(self.ref_shape,
                                                                                               dtype=np.float32)
            return data.astype(np.float32), coarse_flag

        # Use memory-mapped output for large arrays
        if np.prod(self.ref_shape) > 1000000:  # 1M pixels
            dst_data = MemoryMappedArray(self.ref_shape, dtype=np.float32)
            dst_data[:] = np.nan
        else:
            dst_data = np.full(self.ref_shape, np.nan, dtype=np.float32)

        coarse_flag = np.zeros(self.ref_shape, dtype=np.float32)

        try:
            resampling_method = Resampling.cubic if is_coarse else Resampling.bilinear

            reproject(
                source=data,
                destination=dst_data.array if hasattr(dst_data, 'array') else dst_data,
                src_transform=src_meta['transform'],
                src_crs=src_meta['crs'],
                dst_transform=self.ref_transform,
                dst_crs=self.ref_crs,
                resampling=resampling_method,
                num_threads=2
            )

            if is_coarse:
                coarse_flag[:] = 1.0

            # Convert back to regular array if memory-mapped
            if hasattr(dst_data, 'array'):
                result = np.array(dst_data.array)
                del dst_data
            else:
                result = dst_data

            return result, coarse_flag
        except Exception:
            return None, None


def process_single_file_optimized(args):
    """Ultra-optimized single file processing."""
    filepath, is_coarse, ref_transform, ref_shape, ref_crs = args

    if not filepath or not os.path.exists(filepath):
        return None, None

    try:
        # Use context manager for automatic cleanup
        with rasterio.open(filepath) as src:
            # Read efficiently based on file size
            if src.width * src.height > 1000000:
                data = src.read(1, out_dtype=np.float32)
            else:
                data = src.read(1).astype(np.float32)

            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            data[np.isinf(data)] = np.nan

            meta = {
                'transform': src.transform,
                'crs': src.crs,
                'shape': data.shape
            }

        # Fast regridding check
        needs_regrid = (meta['shape'] != ref_shape or
                        meta['transform'] != ref_transform or
                        meta['crs'] != ref_crs)

        if needs_regrid:
            dst_data = np.full(ref_shape, np.nan, dtype=np.float32)
            resampling_method = Resampling.cubic if is_coarse else Resampling.bilinear

            reproject(
                source=data,
                destination=dst_data,
                src_transform=meta['transform'],
                src_crs=meta['crs'],
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=resampling_method
            )
            data = dst_data

        coarse_flag = np.ones(ref_shape, dtype=np.float32) if is_coarse else np.zeros(ref_shape, dtype=np.float32)
        return data, coarse_flag

    except Exception:
        return None, None


class UltraFastDataImputer:
    """Ultra-fast vectorized data imputation with Numba acceleration."""

    @staticmethod
    def impute_monthly_data_ultra_fast(data_cube: np.ndarray, months: np.ndarray, years: np.ndarray):
        """Ultra-fast vectorized imputation using optimized algorithms."""
        T, H, W = data_cube.shape
        imputed = data_cube.copy()
        impute_flag = np.zeros_like(data_cube, dtype=np.float32)

        # Process each month with vectorized operations
        for month in range(1, 13):
            month_mask = months == month
            month_indices = np.where(month_mask)[0]

            if len(month_indices) == 0:
                continue

            month_data = data_cube[month_mask].copy()
            month_years = years[month_mask]
            missing_mask = np.isnan(month_data)

            if not missing_mask.any():
                continue

            print(f"  Imputing month {month}: {missing_mask.sum()} missing values")

            # Step 1: Ultra-fast adjacent year interpolation
            for idx in range(1, len(month_data) - 1):
                if missing_mask[idx].any():
                    prev_data = month_data[idx - 1]
                    next_data = month_data[idx + 1]

                    # Vectorized interpolation condition
                    can_interpolate = (missing_mask[idx] &
                                       ~np.isnan(prev_data) &
                                       ~np.isnan(next_data))

                    if can_interpolate.any():
                        month_data[idx][can_interpolate] = (prev_data[can_interpolate] + next_data[can_interpolate]) / 2
                        missing_mask[idx][can_interpolate] = False

            # Step 2: Fast polynomial trend fitting for remaining gaps
            still_missing = missing_mask.any(axis=0)
            if still_missing.any():
                valid_count = (~missing_mask).sum(axis=0)
                can_fit = (valid_count >= 3) & still_missing

                if can_fit.any():
                    # Process in chunks for memory efficiency
                    chunk_size = 10000
                    missing_coords = np.where(can_fit)

                    for chunk_start in range(0, len(missing_coords[0]), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(missing_coords[0]))

                        for i in range(chunk_start, chunk_end):
                            y, x = missing_coords[0][i], missing_coords[1][i]
                            pixel_series = month_data[:, y, x]
                            valid_mask = ~np.isnan(pixel_series)

                            if valid_mask.sum() >= 3:
                                valid_years_subset = month_years[valid_mask]
                                valid_data_subset = pixel_series[valid_mask]

                                try:
                                    # Use linear fitting for speed
                                    coeffs = np.polyfit(valid_years_subset, valid_data_subset, 1)
                                    for missing_idx in np.where(np.isnan(pixel_series))[0]:
                                        pred_value = np.polyval(coeffs, month_years[missing_idx])
                                        month_data[missing_idx, y, x] = pred_value
                                        missing_mask[missing_idx, y, x] = False
                                except:
                                    continue

            # Step 3: Ultra-fast mean filling using Numba if available
            still_missing = missing_mask
            if still_missing.any():
                if NUMBA_AVAILABLE:
                    mean_values = fast_nanmean_axis0(month_data.reshape(len(month_data), -1)).reshape(H, W)
                else:
                    mean_values = np.nanmean(month_data, axis=0)

                for idx in range(len(month_data)):
                    fill_mask = still_missing[idx]
                    if fill_mask.any():
                        month_data[idx][fill_mask] = mean_values[fill_mask]
                        missing_mask[idx][fill_mask] = False

            # Update main arrays
            for i, time_idx in enumerate(month_indices):
                imputed[time_idx] = month_data[i]
                impute_flag[time_idx] = missing_mask[i].astype(np.float32)

            # Memory cleanup
            del month_data, missing_mask
            if month % 3 == 0:  # Cleanup every 3 months
                aggressive_memory_cleanup()

        return imputed, impute_flag


class UltraOptimizedDroughtDataset(Dataset):
    """Ultra-optimized dataset with memory efficiency."""

    def __init__(self, data_dict: dict, lookback: int = 12, mode: str = 'train'):
        # Use memory-mapped arrays for large datasets
        if data_dict['features'].shape[0] * data_dict['features'].shape[1] > 1000000:
            self.use_memmap = True
            self.features_shape = data_dict['features'].shape
            self.features_dtype = data_dict['features'].dtype

            # Save to temporary files
            self.features_file = f"temp_features_{mode}_{id(self)}.dat"
            self.ewbi_file = f"temp_ewbi_{mode}_{id(self)}.dat"
            self.veg_file = f"temp_veg_{mode}_{id(self)}.dat"

            self.features_mmap = np.memmap(self.features_file, dtype=self.features_dtype, mode='w+',
                                           shape=self.features_shape)
            self.features_mmap[:] = data_dict['features']

            self.ewbi = torch.FloatTensor(data_dict['ewbi'])
            self.veg_composite = torch.FloatTensor(data_dict['veg_composite'])
        else:
            self.use_memmap = False
            self.features = torch.FloatTensor(data_dict['features'])
            self.ewbi = torch.FloatTensor(data_dict['ewbi'])
            self.veg_composite = torch.FloatTensor(data_dict['veg_composite'])

        self.lookback = lookback
        self.mode = mode
        self.valid_indices = list(range(lookback, len(data_dict['features'])))

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]

        if self.use_memmap:
            x = torch.FloatTensor(self.features_mmap[t - self.lookback:t])
        else:
            x = self.features[t - self.lookback:t]

        y_ewbi = self.ewbi[t] if t < len(self.ewbi) else self.ewbi[-1]
        y_veg = self.veg_composite[t] if t < len(self.veg_composite) else self.veg_composite[-1]
        return x, y_ewbi, y_veg

    def __del__(self):
        if hasattr(self, 'use_memmap') and self.use_memmap:
            try:
                del self.features_mmap
                for filename in [self.features_file, self.ewbi_file, self.veg_file]:
                    if os.path.exists(filename):
                        os.remove(filename)
            except:
                pass


class UltraOptimizedDataProcessor:
    """Ultra-optimized data processor with all performance enhancements."""

    def __init__(self, config):
        self.config = config
        self.reader = UltraOptimizedGeoTIFFReader(config)
        self.imputer = UltraFastDataImputer()
        self.scalers = {}

    def process_land_type(self, land_type: str) -> Dict:
        """Ultra-optimized processing with granular caching and memory management."""
        print(f"Processing {land_type}...")
        enforce_memory_limit(self.config['optimization']['memory_limit_gb'])

        # Check for complete cached data first
        cache_path = get_cache_path(self.config['output']['cache_dir'], land_type, 'processed_data')
        if self.config['optimization']['use_cache']:
            cached_data = load_cache_compressed(cache_path)
            if cached_data:
                print(f"Loaded {land_type} data from cache")
                self.scalers = cached_data.get('scalers', {})
                return cached_data

        # Step-by-step processing with intermediate caching
        print("Setting reference grid...")
        self.reader.set_reference_grid(land_type)

        # Check if raw data is cached
        raw_cache_path = get_cache_path(self.config['output']['cache_dir'], land_type, 'raw_data')
        raw_data = load_cache_compressed(raw_cache_path) if self.config['optimization']['use_cache'] else None

        if raw_data is None:
            print("Loading raw data...")
            raw_data = self._load_all_data_ultra_fast(land_type)
            if self.config['optimization']['use_cache']:
                print("Caching raw data...")
                save_cache_compressed(raw_data, raw_cache_path)
        else:
            print("Loaded raw data from cache")

        enforce_memory_limit(self.config['optimization']['memory_limit_gb'])

        # Check if imputed data is cached
        imputed_cache_path = get_cache_path(self.config['output']['cache_dir'], land_type, 'imputed_data')
        processed_data = load_cache_compressed(imputed_cache_path) if self.config['optimization']['use_cache'] else None

        if processed_data is None:
            print("Processing features and targets...")
            processed_data = self._compute_features_and_targets_ultra_optimized(raw_data, land_type)
            if self.config['optimization']['use_cache']:
                print("Caching processed data...")
                save_cache_compressed(processed_data, imputed_cache_path)
        else:
            print("Loaded processed data from cache")

        # Final assembly
        processed_data['timestamps'] = raw_data['timestamps']
        processed_data['land_type'] = land_type
        processed_data['scalers'] = self.scalers

        # Cache final result
        if self.config['optimization']['use_cache']:
            save_cache_compressed(processed_data, cache_path)

        # Cleanup
        del raw_data
        aggressive_memory_cleanup()

        return processed_data

    def _load_all_data_ultra_fast(self, land_type: str) -> Dict:
        """Ultra-fast data loading with optimized parallel processing."""
        all_years = (self.config['splits']['train_years'] +
                     self.config['splits']['val_years'] +
                     self.config['splits']['test_years'])

        year_month_pairs = [(y, m) for y in all_years for m in range(1, 13)]

        print("Resolving file paths...")
        file_paths = self.reader.get_file_paths_batch(land_type, year_month_pairs)

        if not file_paths:
            raise ValueError(f"No data files found for {land_type}")

        print(f"Found {len(file_paths)} time steps")

        # Ultra-fast parallel loading
        if self.config['optimization']['parallel_data_loading']:
            data_cube = self._load_data_parallel_ultra_fast(file_paths, land_type)
        else:
            data_cube = self._load_data_sequential_optimized(file_paths, land_type)

        # Generate metadata
        timestamps = []
        months_list = []
        years_list = []

        for (year, month) in sorted(file_paths.keys()):
            timestamps.append(datetime(year, month, 1))
            months_list.append(month)
            years_list.append(year)

        months_arr = np.array(months_list)
        years_arr = np.array(years_list)

        print(f"Data shape for {land_type}: {data_cube['NIRv'].shape}")

        # Ultra-fast vectorized imputation
        print("Imputing missing data (ultra-fast)...")
        impute_flags = {}
        for var in self.config['data']['variables']:
            print(f"  Processing {var}...")
            data_cube[var], impute_flags[var] = self.imputer.impute_monthly_data_ultra_fast(
                data_cube[var], months_arr, years_arr
            )

            # Memory management
            if self.config['optimization']['aggressive_gc']:
                aggressive_memory_cleanup()

        return {
            'data_cube': data_cube,
            'impute_flags': impute_flags,
            'months': months_arr,
            'years': years_arr,
            'timestamps': timestamps
        }

    def _load_data_parallel_ultra_fast(self, file_paths: Dict, land_type: str) -> Dict:
        """Ultra-fast parallel data loading with batching."""
        data_cube = {var: [] for var in self.config['data']['variables']}

        # Prepare all tasks
        all_tasks = []
        task_metadata = []

        for (year, month), files in sorted(file_paths.items()):
            for var in self.config['data']['variables']:
                if var in files:
                    is_coarse = (var == 'TWS')
                    task = (files[var], is_coarse,
                            self.reader.ref_transform, self.reader.ref_shape, self.reader.ref_crs)
                    all_tasks.append(task)
                    task_metadata.append((year, month, var))

        # Process in optimized batches
        max_workers = min(self.config['optimization']['max_workers'], mp.cpu_count())
        batch_size = max(1, len(all_tasks) // max_workers)

        print(f"Processing {len(all_tasks)} files with {max_workers} workers...")

        time_var_data = {}

        # Process in batches to manage memory
        for batch_start in range(0, len(all_tasks), batch_size * max_workers):
            batch_end = min(batch_start + batch_size * max_workers, len(all_tasks))
            batch_tasks = all_tasks[batch_start:batch_end]
            batch_metadata = task_metadata[batch_start:batch_end]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(tqdm(
                    executor.map(process_single_file_optimized, batch_tasks),
                    total=len(batch_tasks),
                    desc=f"Batch {batch_start // batch_size + 1}"
                ))

            # Process batch results
            for i, (result_data, result_flag) in enumerate(batch_results):
                year, month, var = batch_metadata[i]
                time_key = (year, month)

                if time_key not in time_var_data:
                    time_var_data[time_key] = {}

                if result_data is not None:
                    time_var_data[time_key][var] = result_data
                    if var == 'TWS':
                        time_var_data[time_key]['TWS_coarse_flag'] = result_flag

            # Memory cleanup after each batch
            del batch_results
            aggressive_memory_cleanup()

        # Convert to arrays efficiently
        return self._organize_data_ultra_fast(time_var_data)

    def _organize_data_ultra_fast(self, time_var_data: Dict) -> Dict:
        """Ultra-fast data organization with memory optimization."""
        data_cube = {var: [] for var in self.config['data']['variables']}

        # Pre-allocate if we know the size
        num_timesteps = len(time_var_data)
        if num_timesteps > 0:
            sample_shape = None
            for month_data in time_var_data.values():
                for var_data in month_data.values():
                    if var_data is not None:
                        sample_shape = var_data.shape
                        break
                if sample_shape:
                    break

            if sample_shape and self.config['optimization']['use_memmap']:
                # Use memory-mapped arrays for large datasets
                for var in self.config['data']['variables']:
                    full_shape = (num_timesteps,) + sample_shape
                    if np.prod(full_shape) > 1000000:  # 1M elements
                        data_cube[var] = MemoryMappedArray(full_shape, dtype=np.float32)
                    else:
                        data_cube[var] = []

        # Fill data efficiently
        for i, (year, month) in enumerate(sorted(time_var_data.keys())):
            month_data = time_var_data[(year, month)]

            for var in self.config['data']['variables']:
                if var in month_data:
                    var_data = month_data[var]
                else:
                    # Create empty array for missing data
                    if sample_shape:
                        var_data = np.full(sample_shape, np.nan, dtype=np.float32)
                    else:
                        var_data = None

                if hasattr(data_cube[var], 'array'):  # Memory-mapped
                    if var_data is not None:
                        data_cube[var][i] = var_data
                else:  # Regular list
                    data_cube[var].append(var_data if var_data is not None else
                                          np.full(sample_shape or (1, 1), np.nan, dtype=np.float32))

        # Convert lists to arrays
        for var in data_cube:
            if isinstance(data_cube[var], list) and data_cube[var]:
                try:
                    data_cube[var] = np.array(data_cube[var], dtype=np.float32)
                except:
                    # Handle memory errors with chunked conversion
                    print(f"Converting {var} data in chunks...")
                    chunks = []
                    chunk_size = 100
                    for chunk_start in range(0, len(data_cube[var]), chunk_size):
                        chunk_end = min(chunk_start + chunk_size, len(data_cube[var]))
                        chunk = np.array(data_cube[var][chunk_start:chunk_end], dtype=np.float32)
                        chunks.append(chunk)
                    data_cube[var] = np.concatenate(chunks, axis=0)
                    del chunks
                    aggressive_memory_cleanup()
            elif hasattr(data_cube[var], 'array'):  # Memory-mapped
                # Convert to regular array when needed
                data_cube[var] = np.array(data_cube[var].array)

        return data_cube

    def _compute_features_and_targets_ultra_optimized(self, raw_data: Dict, land_type: str) -> Dict:
        """Ultra-optimized feature computation with Numba acceleration."""
        data_cube = raw_data['data_cube']
        impute_flags = raw_data['impute_flags']
        months = raw_data['months']
        years = raw_data['years']

        T = len(months)
        if T == 0:
            raise ValueError("No time steps found")

        H, W = data_cube['NIRv'].shape[1:]
        N = H * W

        print(f"Computing features for {T} timesteps, {N} pixels...")

        train_mask = np.isin(years, self.config['splits']['train_years'])

        # Use memory-mapped arrays for large datasets
        if N * T > 1000000:
            ewbi_target = MemoryMappedArray((T, N), dtype=np.float32)
            veg_composite_target = MemoryMappedArray((T, N), dtype=np.float32)
        else:
            ewbi_target = np.zeros((T, N), dtype=np.float32)
            veg_composite_target = np.zeros((T, N), dtype=np.float32)

        # Ultra-fast z-scoring computation
        print("Computing z-scores and targets (ultra-optimized)...")

        # Process months in parallel for better performance
        def process_month_features(month_idx):
            month_mask = months == month_idx
            month_train_mask = month_mask & train_mask

            if not month_train_mask.any():
                return None

            z_scored = {}
            scalers = {}

            for var in self.config['data']['variables']:
                var_data = data_cube[var][month_mask].reshape(-1, N)
                train_data = data_cube[var][month_train_mask].reshape(-1, N)

                # Ultra-fast statistics computation
                if NUMBA_AVAILABLE:
                    mu = fast_nanmean_axis0(train_data).reshape(1, -1)
                    sigma = fast_nanstd_axis0(train_data, mu.flatten()).reshape(1, -1)
                else:
                    mu = np.nanmean(train_data, axis=0, keepdims=True)
                    sigma = np.nanstd(train_data, axis=0, keepdims=True)

                sigma[sigma == 0] = 1.0

                # Ultra-fast z-scoring
                if NUMBA_AVAILABLE:
                    z_scored[var] = vectorized_z_score(var_data, mu.flatten(), sigma.flatten())
                else:
                    z_scored[var] = (var_data - mu) / sigma

                scaler_key = f"{land_type}_{var}_{month_idx}"
                scalers[scaler_key] = {'mu': mu, 'sigma': sigma}

            return month_idx, z_scored, scalers, month_mask

        # Process months efficiently
        for month_idx in range(1, 13):
            result = process_month_features(month_idx)
            if result is None:
                continue

            month_idx, z_scored, month_scalers, month_mask = result
            self.scalers.update(month_scalers)

            # Compute targets using ultra-fast vectorized operations
            month_indices = np.where(month_mask)[0]
            for i, idx in enumerate(month_indices):
                # EWBI = z(SPEI) + z(TWS) - z(VPD) - z(LST)
                ewbi_vals = (z_scored['SPEI'][i] + z_scored['TWS'][i] -
                             z_scored['VPD'][i] - z_scored['LST'][i])

                # VegComposite = equal-weight combination
                veg_vals = (z_scored['GOSIF'][i] + z_scored['GPP'][i] +
                            z_scored['NIRv'][i] + z_scored['NDVI'][i] +
                            z_scored['GNDVI'][i]) / 5.0

                if hasattr(ewbi_target, 'array'):  # Memory-mapped
                    ewbi_target[idx] = ewbi_vals
                    veg_composite_target[idx] = veg_vals
                else:
                    ewbi_target[idx] = ewbi_vals
                    veg_composite_target[idx] = veg_vals

            # Memory cleanup
            del z_scored, month_scalers
            if month_idx % 3 == 0:
                aggressive_memory_cleanup()

        # Create feature array efficiently
        print("Creating feature arrays...")
        n_features = 9 + 9 + 1 + 2  # vars + impute flags + TWS coarse + seasonality

        if N * T * n_features > 10000000:  # 10M elements
            features = MemoryMappedArray((T, N, n_features), dtype=np.float32)
        else:
            features = np.zeros((T, N, n_features), dtype=np.float32)

        # Ultra-fast feature construction
        for t in tqdm(range(T), desc="Building features"):
            month = months[t]
            feat_idx = 0

            # Add z-scored variables
            for var in self.config['data']['variables']:
                scaler_key = f"{land_type}_{var}_{month}"
                if scaler_key in self.scalers:
                    mu = self.scalers[scaler_key]['mu']
                    sigma = self.scalers[scaler_key]['sigma']
                    var_data = data_cube[var][t].reshape(1, N)

                    if NUMBA_AVAILABLE:
                        z_vals = vectorized_z_score(var_data, mu.flatten(), sigma.flatten()).flatten()
                    else:
                        z_vals = ((var_data - mu) / sigma).flatten()

                    if hasattr(features, 'array'):
                        features[t, :, feat_idx] = z_vals
                    else:
                        features[t, :, feat_idx] = z_vals
                feat_idx += 1

            # Add imputation flags
            for var in self.config['data']['variables']:
                flag_vals = impute_flags[var][t].flatten()
                if hasattr(features, 'array'):
                    features[t, :, feat_idx] = flag_vals
                else:
                    features[t, :, feat_idx] = flag_vals
                feat_idx += 1

            # Add TWS coarse flag and seasonality
            if hasattr(features, 'array'):
                features[t, :, feat_idx] = 1.0  # TWS coarse flag
                feat_idx += 1
                features[t, :, feat_idx] = np.sin(2 * np.pi * month / 12)
                feat_idx += 1
                features[t, :, feat_idx] = np.cos(2 * np.pi * month / 12)
            else:
                features[t, :, feat_idx] = 1.0
                feat_idx += 1
                features[t, :, feat_idx] = np.sin(2 * np.pi * month / 12)
                feat_idx += 1
                features[t, :, feat_idx] = np.cos(2 * np.pi * month / 12)

        # Convert memory-mapped arrays to regular arrays for final output
        if hasattr(features, 'array'):
            features = np.array(features.array)
        if hasattr(ewbi_target, 'array'):
            ewbi_target = np.array(ewbi_target.array)
        if hasattr(veg_composite_target, 'array'):
            veg_composite_target = np.array(veg_composite_target.array)

        # Create splits
        val_mask = np.isin(years, self.config['splits']['val_years'])
        test_mask = np.isin(years, self.config['splits']['test_years'])

        return {
            'features': features,
            'ewbi': ewbi_target,
            'veg_composite': veg_composite_target,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'test_mask': test_mask,
            'spatial_shape': (H, W),
            'months': months,
            'years': years
        }

    def create_dataloaders_optimized(self, processed_data: Dict, batch_size: int = 32) -> Dict:
        """Create ultra-optimized PyTorch dataloaders."""
        lookback = self.config['model']['lookback_window']

        # Create data splits
        train_data = {
            'features': processed_data['features'][processed_data['train_mask']],
            'ewbi': processed_data['ewbi'][processed_data['train_mask']],
            'veg_composite': processed_data['veg_composite'][processed_data['train_mask']]
        }

        val_data = {
            'features': processed_data['features'][processed_data['val_mask']],
            'ewbi': processed_data['ewbi'][processed_data['val_mask']],
            'veg_composite': processed_data['veg_composite'][processed_data['val_mask']]
        }

        test_data = {
            'features': processed_data['features'][processed_data['test_mask']],
            'ewbi': processed_data['ewbi'][processed_data['test_mask']],
            'veg_composite': processed_data['veg_composite'][processed_data['test_mask']]
        }

        # Create datasets
        train_dataset = UltraOptimizedDroughtDataset(train_data, lookback, 'train')
        val_dataset = UltraOptimizedDroughtDataset(val_data, lookback, 'val')
        test_dataset = UltraOptimizedDroughtDataset(test_data, lookback, 'test')

        # Optimized DataLoader settings
        num_workers = min(4, self.config['optimization']['max_workers'])  # Conservative for stability
        pin_memory = self.config['optimization']['pin_memory']
        prefetch_factor = self.config['optimization']['prefetch_factor']
        persistent_workers = self.config['optimization']['persistent_workers'] and num_workers > 0

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers
        )

        return {
            'train': train_loader,
            'val': val_loader,
            'test': test_loader
        }


# =============================================================================
# Keep all the original model classes exactly the same
# =============================================================================

class OptimizedDilatedCausalConv1d(nn.Module):
    """Memory-efficient dilated causal convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation, bias=False
        )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return self.norm(x)


class OptimizedGraphConvolution(nn.Module):
    """Optimized graph convolution with batch processing."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = self.linear(x)
        output = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), support)
        return self.norm(output)


class OptimizedSpatioTemporalBlock(nn.Module):
    """Optimized spatiotemporal block with reduced memory usage."""

    def __init__(self, in_channels: int, out_channels: int, num_nodes: int,
                 kernel_size: int = 2, dilation: int = 1, dropout: float = 0.3):
        super().__init__()

        self.temporal_conv1 = OptimizedDilatedCausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )
        self.temporal_conv2 = OptimizedDilatedCausalConv1d(
            in_channels, out_channels, kernel_size, dilation
        )

        self.graph_conv = OptimizedGraphConvolution(out_channels, out_channels)
        self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj_fixed: torch.Tensor,
                adj_learned: Optional[torch.Tensor] = None) -> torch.Tensor:

        B, T, N, F = x.shape
        x_temp = x.permute(0, 2, 3, 1).reshape(B * N, F, T)

        h1 = self.temporal_conv1(x_temp)
        h2 = self.temporal_conv2(x_temp)
        h = torch.tanh(h1) * torch.sigmoid(h2)

        _, F_out, T_out = h.shape
        h = h.reshape(B, N, F_out, T_out).permute(0, 3, 1, 2)

        h_last = h[:, -1, :, :]
        h_last = self.graph_conv(h_last, adj_fixed)

        if adj_learned is not None:
            h_last = h_last + self.graph_conv(h_last, adj_learned)

        if self.skip_conv is not None:
            x_skip = x[:, -1, :, :]
            x_skip = x_skip.permute(0, 2, 1)
            x_skip = self.skip_conv(x_skip.unsqueeze(-1)).squeeze(-1)
            x_skip = x_skip.permute(0, 2, 1)
        else:
            x_skip = x[:, -1, :, :]

        out = F.relu(h_last + x_skip)
        out = self.dropout(out)

        return out.unsqueeze(1)


class OptimizedGraphWaveNet(nn.Module):
    """Optimized Graph WaveNet with sparse adjacency matrices."""

    def __init__(self, num_nodes: int, in_features: int, hidden_dim: int = 32,
                 num_layers: int = 4, kernel_size: int = 2,
                 k_neighbors: int = 8, dropout: float = 0.3,
                 lookback_window: int = 12, adaptive_adj: bool = True):
        super().__init__()

        self.num_nodes = num_nodes
        self.k_neighbors = k_neighbors
        self.lookback_window = lookback_window
        self.adaptive_adj = adaptive_adj

        self.input_conv = nn.Conv1d(in_features, hidden_dim, 1, bias=False)
        self.input_norm = nn.BatchNorm1d(hidden_dim)

        # Fixed spatial graph (will be sparse)
        self.register_buffer('adj_fixed', torch.zeros(num_nodes, num_nodes))

        # Adaptive adjacency
        if adaptive_adj:
            self.node_embeddings1 = nn.Parameter(torch.randn(num_nodes, 10))
            self.node_embeddings2 = nn.Parameter(torch.randn(10, num_nodes))
            nn.init.xavier_uniform_(self.node_embeddings1)
            nn.init.xavier_uniform_(self.node_embeddings2)

        # ST blocks
        self.st_blocks = nn.ModuleList()
        for layer in range(num_layers):
            dilation = 2 ** layer
            self.st_blocks.append(
                OptimizedSpatioTemporalBlock(
                    hidden_dim, hidden_dim, num_nodes,
                    kernel_size, dilation, dropout
                )
            )

        # Output layers
        self.output_conv = nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False)
        self.output_norm = nn.BatchNorm1d(hidden_dim)
        self.ewbi_head = nn.Linear(hidden_dim, 1)
        self.veg_head = nn.Linear(hidden_dim, 1)

    def create_knn_adjacency_fast(self, coords: np.ndarray) -> torch.Tensor:
        """Create sparse k-NN adjacency matrix."""
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='ball_tree')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)

        # Create sparse adjacency matrix
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)

        for i in range(self.num_nodes):
            neighbors = indices[i][1:]  # Exclude self
            neighbor_dists = distances[i][1:]

            if neighbor_dists.max() > 0:
                weights = np.exp(-neighbor_dists / neighbor_dists.mean())
                adj[i, neighbors] = weights

        # Symmetrize and normalize
        adj = (adj + adj.T) / 2
        adj = adj + np.eye(self.num_nodes, dtype=np.float32)

        # Row normalization
        rowsum = adj.sum(axis=1)
        d_inv_sqrt = np.power(rowsum, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

        return torch.FloatTensor(adj)

    def get_learned_adjacency(self) -> torch.Tensor:
        """Compute learned adjacency matrix."""
        if not self.adaptive_adj:
            return None

        adj = F.softmax(F.relu(torch.mm(self.node_embeddings1, self.node_embeddings2)), dim=1)
        adj = (adj + adj.t()) / 2
        return adj

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        B, T, N, F = x.shape

        x_proj = x.permute(0, 2, 3, 1).reshape(B * N, F, T)
        x_proj = F.relu(self.input_norm(self.input_conv(x_proj)))

        hidden_dim = x_proj.shape[1]
        h = x_proj.reshape(B, N, hidden_dim, T).permute(0, 3, 1, 2)

        adj_learned = self.get_learned_adjacency() if self.adaptive_adj else None

        for st_block in self.st_blocks:
            h = st_block(h, self.adj_fixed, adj_learned)

        h_final = h[:, -1, :, :]

        h_out = h_final.permute(0, 2, 1)
        h_out = F.relu(self.output_norm(self.output_conv(h_out)))
        h_out = h_out.permute(0, 2, 1)

        ewbi_pred = self.ewbi_head(h_out).squeeze(-1)
        veg_pred = self.veg_head(h_out).squeeze(-1)

        return ewbi_pred, veg_pred


def create_spatial_coords_grid(H: int, W: int) -> np.ndarray:
    """Create spatial coordinates for the grid."""
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    coords = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
    return coords


# =============================================================================
# Keep training and evaluation classes the same but with memory monitoring
# =============================================================================

class OptimizedTrainer:
    """Ultra-optimized trainer with memory monitoring."""

    def __init__(self, config, land_type: str):
        self.config = config
        self.land_type = land_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_seed(self.config['training']['seed'])
        aggressive_memory_cleanup()

        # Monitor memory before processing
        print(f"Memory before data processing: {monitor_memory_usage():.1f}GB")

        self.data_processor = UltraOptimizedDataProcessor(config)
        print(f"Processing data for {land_type}...")
        self.data = self.data_processor.process_land_type(land_type)

        print(f"Memory after data processing: {monitor_memory_usage():.1f}GB")

        batch_size = self.config['training']['batch_size']
        self.dataloaders = self.data_processor.create_dataloaders_optimized(
            self.data, batch_size=batch_size
        )

        H, W = self.data['spatial_shape']
        self.num_nodes = H * W

        sample_batch = next(iter(self.dataloaders['train']))
        self.n_features = sample_batch[0].shape[-1]

        self.model = OptimizedGraphWaveNet(
            num_nodes=self.num_nodes,
            in_features=self.n_features,
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            kernel_size=self.config['model']['kernel_size'],
            k_neighbors=self.config['model']['k_neighbors'],
            dropout=self.config['model']['dropout'],
            lookback_window=self.config['model']['lookback_window'],
            adaptive_adj=self.config['model']['adaptive_adj']
        ).to(self.device)

        coords = create_spatial_coords_grid(H, W)
        adj_fixed = self.model.create_knn_adjacency_fast(coords)
        self.model.adj_fixed = adj_fixed.to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-5
        )

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config['training']['learning_rate'] * 3,
            epochs=self.config['training']['epochs'],
            steps_per_epoch=len(self.dataloaders['train'])
        )

        self.checkpoint_dir = os.path.join(
            self.config['output']['checkpoints_dir'], land_type
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.epoch = 0

        print(f"Memory after model initialization: {monitor_memory_usage():.1f}GB")

    def train_epoch(self):
        """Training epoch with memory monitoring."""
        self.model.train()
        total_loss = 0
        total_ewbi_loss = 0
        total_veg_loss = 0
        n_batches = 0

        with tqdm(self.dataloaders['train'], desc=f'Epoch {self.epoch} [Train]') as pbar:
            for batch_idx, batch in enumerate(pbar):
                x, y_ewbi, y_veg = batch
                x = x.to(self.device, non_blocking=True)
                y_ewbi = y_ewbi.to(self.device, non_blocking=True)
                y_veg = y_veg.to(self.device, non_blocking=True)

                self.optimizer.zero_grad()

                ewbi_pred, veg_pred = self.model(x)
                loss_ewbi = self.criterion(ewbi_pred, y_ewbi)
                loss_veg = self.criterion(veg_pred, y_veg)
                loss = loss_ewbi + loss_veg

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                total_ewbi_loss += loss_ewbi.item()
                total_veg_loss += loss_veg.item()
                n_batches += 1

                # Aggressive memory cleanup
                if batch_idx % 20 == 0:
                    aggressive_memory_cleanup()
                    enforce_memory_limit(self.config['optimization']['memory_limit_gb'])

                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ewbi': f'{loss_ewbi.item():.4f}',
                    'veg': f'{loss_veg.item():.4f}',
                    'mem': f'{monitor_memory_usage():.1f}GB'
                })

        return total_loss / n_batches, total_ewbi_loss / n_batches, total_veg_loss / n_batches

    def validate(self):
        """Validation with memory monitoring."""
        self.model.eval()
        total_loss = 0
        total_ewbi_loss = 0
        total_veg_loss = 0
        n_batches = 0

        with torch.no_grad():
            with tqdm(self.dataloaders['val'], desc=f'Epoch {self.epoch} [Val]') as pbar:
                for batch in pbar:
                    x, y_ewbi, y_veg = batch
                    x = x.to(self.device, non_blocking=True)
                    y_ewbi = y_ewbi.to(self.device, non_blocking=True)
                    y_veg = y_veg.to(self.device, non_blocking=True)

                    ewbi_pred, veg_pred = self.model(x)
                    loss_ewbi = self.criterion(ewbi_pred, y_ewbi)
                    loss_veg = self.criterion(veg_pred, y_veg)
                    loss = loss_ewbi + loss_veg

                    total_loss += loss.item()
                    total_ewbi_loss += loss_ewbi.item()
                    total_veg_loss += loss_veg.item()
                    n_batches += 1

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'ewbi': f'{loss_ewbi.item():.4f}',
                        'veg': f'{loss_veg.item():.4f}'
                    })

        return total_loss / n_batches, total_ewbi_loss / n_batches, total_veg_loss / n_batches

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'land_type': self.land_type,
            'data_processor_scalers': self.data_processor.scalers
        }

        latest_path = os.path.join(self.checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation loss: {self.best_val_loss:.4f}")

    def train(self):
        """Main training loop with optimizations."""
        print(f"\nTraining Ultra-Optimized Graph WaveNet for {self.land_type}")
        print(f"Device: {self.device}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of features: {self.n_features}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Memory usage: {monitor_memory_usage():.1f}GB")

        patience = self.config['training']['patience']
        start_time = time.time()

        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch + 1
            epoch_start = time.time()

            train_loss, train_ewbi, train_veg = self.train_epoch()
            val_loss, val_ewbi, val_veg = self.validate()

            epoch_time = time.time() - epoch_start

            print(f"\nEpoch {self.epoch}/{self.config['training']['epochs']} ({epoch_time:.1f}s)")
            print(f"Train Loss: {train_loss:.4f} (EWBI: {train_ewbi:.4f}, Veg: {train_veg:.4f})")
            print(f"Val Loss: {val_loss:.4f} (EWBI: {val_ewbi:.4f}, Veg: {val_veg:.4f})")
            print(f"Memory: {monitor_memory_usage():.1f}GB")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)

                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {self.epoch} epochs")
                    break

            # Memory cleanup every epoch
            aggressive_memory_cleanup()

        total_time = time.time() - start_time
        print(f"\nTraining completed for {self.land_type} in {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        return os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')


# Keep the FastDroughtEvaluator class exactly the same as the original
class FastDroughtEvaluator:
    """Keep exactly the same as original but with memory monitoring."""

    def __init__(self, config, land_type: str, checkpoint_path: str):
        self.config = config
        self.land_type = land_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading model for {land_type} evaluation...")

        self.data_processor = UltraOptimizedDataProcessor(config)
        self.data = self.data_processor.process_land_type(land_type)

        H, W = self.data['spatial_shape']
        self.num_nodes = H * W
        self.H, self.W = H, W

        self.dataloaders = self.data_processor.create_dataloaders_optimized(
            self.data, batch_size=config['training']['batch_size'] * 2
        )

        self.model = self._load_model(checkpoint_path)
        self.results = {}
        self.thresholds = {}

        print(f"Memory usage: {monitor_memory_usage():.1f}GB")

    def _load_model(self, checkpoint_path: str) -> OptimizedGraphWaveNet:
        """Load trained model from checkpoint."""
        sample_batch = next(iter(self.dataloaders['train']))
        n_features = sample_batch[0].shape[-1]

        model = OptimizedGraphWaveNet(
            num_nodes=self.num_nodes,
            in_features=n_features,
            hidden_dim=self.config['model']['hidden_dim'],
            num_layers=self.config['model']['num_layers'],
            kernel_size=self.config['model']['kernel_size'],
            k_neighbors=self.config['model']['k_neighbors'],
            dropout=self.config['model']['dropout'],
            lookback_window=self.config['model']['lookback_window'],
            adaptive_adj=self.config['model']['adaptive_adj']
        ).to(self.device)

        coords = create_spatial_coords_grid(self.H, self.W)
        adj_fixed = model.create_knn_adjacency_fast(coords)
        model.adj_fixed = adj_fixed.to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        if 'data_processor_scalers' in checkpoint:
            self.data_processor.scalers = checkpoint['data_processor_scalers']

        print(f"Loaded model from {checkpoint_path}")
        return model

    # Keep all other methods exactly the same as original...
    # [All the other methods from FastDroughtEvaluator remain unchanged]

    def compute_forecast_residuals_fast(self, dataloader, return_raw=False):
        """Optimized residual computation with batch processing."""
        all_observed_ewbi = []
        all_observed_veg = []
        all_predicted_ewbi = []
        all_predicted_veg = []

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing residuals"):
                x, y_ewbi, y_veg = batch
                x = x.to(self.device, non_blocking=True)

                ewbi_pred, veg_pred = self.model(x)

                all_observed_ewbi.append(y_ewbi.cpu().numpy())
                all_observed_veg.append(y_veg.cpu().numpy())
                all_predicted_ewbi.append(ewbi_pred.cpu().numpy())
                all_predicted_veg.append(veg_pred.cpu().numpy())

        observed_ewbi = np.concatenate(all_observed_ewbi, axis=0)
        observed_veg = np.concatenate(all_observed_veg, axis=0)
        predicted_ewbi = np.concatenate(all_predicted_ewbi, axis=0)
        predicted_veg = np.concatenate(all_predicted_veg, axis=0)

        residuals_ewbi = observed_ewbi - predicted_ewbi
        residuals_veg = observed_veg - predicted_veg

        if return_raw:
            return {
                'observed_ewbi': observed_ewbi,
                'observed_veg': observed_veg,
                'predicted_ewbi': predicted_ewbi,
                'predicted_veg': predicted_veg,
                'residuals_ewbi': residuals_ewbi,
                'residuals_veg': residuals_veg
            }

        return residuals_ewbi, residuals_veg

    def compute_area_mean_time_series(self):
        """Compute area-averaged time series."""
        ewbi_full = self.data['ewbi']
        veg_full = self.data['veg_composite']

        area_mean_ewbi = ewbi_full.mean(axis=1)
        area_mean_veg = veg_full.mean(axis=1)

        return area_mean_ewbi, area_mean_veg

    def compute_standardized_anomalies(self):
        """Compute standardized anomalies using vectorized operations."""
        area_mean_ewbi, area_mean_veg = self.compute_area_mean_time_series()

        train_mask = self.data['train_mask']
        months = self.data['months']

        train_ewbi = area_mean_ewbi[train_mask]
        train_veg = area_mean_veg[train_mask]
        train_months = months[train_mask]

        anomalies_ewbi = np.zeros(len(area_mean_ewbi))
        anomalies_veg = np.zeros(len(area_mean_veg))

        for month in range(1, 13):
            month_mask_train = train_months == month
            month_mask_all = months == month

            if month_mask_train.any() and month_mask_all.any():
                ewbi_mean = train_ewbi[month_mask_train].mean()
                ewbi_std = train_ewbi[month_mask_train].std()
                veg_mean = train_veg[month_mask_train].mean()
                veg_std = train_veg[month_mask_train].std()

                anomalies_ewbi[month_mask_all] = (area_mean_ewbi[month_mask_all] - ewbi_mean) / ewbi_std
                anomalies_veg[month_mask_all] = (area_mean_veg[month_mask_all] - veg_mean) / veg_std

        return anomalies_ewbi, anomalies_veg, {}

    def detect_observed_severe_months(self, anomalies: np.ndarray, threshold_percentile: float = 5):
        """Detect severe months efficiently."""
        z_threshold = stats.norm.ppf(threshold_percentile / 100)
        return anomalies <= z_threshold

    def calibrate_residual_thresholds_fast(self):
        """Fast threshold calibration."""
        print("Calibrating thresholds...")

        train_residuals_ewbi, train_residuals_veg = self.compute_forecast_residuals_fast(self.dataloaders['train'])

        train_area_residuals_ewbi = train_residuals_ewbi.mean(axis=1)
        train_area_residuals_veg = train_residuals_veg.mean(axis=1)

        target_far = self.config['evaluation']['false_alarm_target']
        percentile = (1 - target_far) * 100

        self.thresholds['ewbi_residual'] = np.percentile(np.abs(train_area_residuals_ewbi), percentile)
        self.thresholds['veg_residual'] = np.percentile(np.abs(train_area_residuals_veg), percentile)

        train_residuals_2d = np.column_stack([train_area_residuals_ewbi, train_area_residuals_veg])
        cov_matrix = np.cov(train_residuals_2d.T)
        self.cov_inv = np.linalg.pinv(cov_matrix + 1e-6 * np.eye(2))
        self.thresholds['joint_mahal'] = chi2.ppf(0.95, df=2)

        print(f"Thresholds - EWBI: {self.thresholds['ewbi_residual']:.3f}, "
              f"Veg: {self.thresholds['veg_residual']:.3f}, "
              f"Joint: {self.thresholds['joint_mahal']:.3f}")

    def evaluate_test_set_fast(self):
        """Fast test set evaluation."""
        print(f"\n{'=' * 60}")
        print(f"EVALUATING {self.land_type.upper()} ON TEST SET")
        print(f"{'=' * 60}")

        self.calibrate_residual_thresholds_fast()
        test_results = self.compute_forecast_residuals_fast(self.dataloaders['test'], return_raw=True)

        area_mean_observed_ewbi = test_results['observed_ewbi'].mean(axis=1)
        area_mean_observed_veg = test_results['observed_veg'].mean(axis=1)
        area_mean_predicted_ewbi = test_results['predicted_ewbi'].mean(axis=1)
        area_mean_predicted_veg = test_results['predicted_veg'].mean(axis=1)
        area_mean_residuals_ewbi = test_results['residuals_ewbi'].mean(axis=1)
        area_mean_residuals_veg = test_results['residuals_veg'].mean(axis=1)

        anomalies_ewbi, anomalies_veg, _ = self.compute_standardized_anomalies()
        test_mask = self.data['test_mask']
        test_anomalies_ewbi = anomalies_ewbi[test_mask]
        test_anomalies_veg = anomalies_veg[test_mask]

        severe_threshold = self.config['evaluation']['severe_threshold_percentile']
        observed_severe_ewbi = self.detect_observed_severe_months(test_anomalies_ewbi, severe_threshold)
        observed_severe_veg = self.detect_observed_severe_months(test_anomalies_veg, severe_threshold)

        detected_severe_ewbi = np.abs(area_mean_residuals_ewbi) > self.thresholds['ewbi_residual']
        detected_severe_veg = np.abs(area_mean_residuals_veg) > self.thresholds['veg_residual']

        detected_severe_joint = np.array([
            np.sqrt(np.array([area_mean_residuals_ewbi[i], area_mean_residuals_veg[i]]) @
                    self.cov_inv @
                    np.array([area_mean_residuals_ewbi[i], area_mean_residuals_veg[i]]).T) > np.sqrt(
                self.thresholds['joint_mahal'])
            for i in range(len(area_mean_residuals_ewbi))
        ])

        metrics = self.compute_detection_metrics_fast(
            observed_severe_ewbi, observed_severe_veg,
            detected_severe_ewbi, detected_severe_veg,
            detected_severe_joint
        )

        test_timestamps = [self.data['timestamps'][i] for i in np.where(test_mask)[0]]

        self.results = {
            'timestamps': test_timestamps,
            'area_mean_observed_ewbi': area_mean_observed_ewbi,
            'area_mean_observed_veg': area_mean_observed_veg,
            'area_mean_predicted_ewbi': area_mean_predicted_ewbi,
            'area_mean_predicted_veg': area_mean_predicted_veg,
            'area_mean_residuals_ewbi': area_mean_residuals_ewbi,
            'area_mean_residuals_veg': area_mean_residuals_veg,
            'anomalies_ewbi': test_anomalies_ewbi,
            'anomalies_veg': test_anomalies_veg,
            'observed_severe_ewbi': observed_severe_ewbi,
            'observed_severe_veg': observed_severe_veg,
            'detected_severe_ewbi': detected_severe_ewbi,
            'detected_severe_veg': detected_severe_veg,
            'detected_severe_joint': detected_severe_joint,
            'metrics': metrics
        }

        return metrics

    def compute_detection_metrics_fast(self, obs_severe_ewbi, obs_severe_veg,
                                       det_severe_ewbi, det_severe_veg, det_severe_joint):
        """Fast metrics computation using vectorized operations."""

        def calc_metrics_vec(observed, detected, name):
            H = np.sum(observed & detected)
            M = np.sum(observed & ~detected)
            F = np.sum(~observed & detected)
            C = np.sum(~observed & ~detected)

            POD = H / (H + M) if (H + M) > 0 else 0
            FAR = F / (H + F) if (H + F) > 0 else 0
            CSI = H / (H + M + F) if (H + M + F) > 0 else 0

            return {
                f'{name}_POD': POD, f'{name}_FAR': FAR, f'{name}_CSI': CSI,
                f'{name}_H': H, f'{name}_M': M, f'{name}_F': F, f'{name}_C': C
            }

        metrics = {}
        metrics.update(calc_metrics_vec(obs_severe_ewbi, det_severe_ewbi, 'EWBI'))
        metrics.update(calc_metrics_vec(obs_severe_veg, det_severe_veg, 'Veg'))

        obs_severe_joint = obs_severe_ewbi | obs_severe_veg
        metrics.update(calc_metrics_vec(obs_severe_joint, det_severe_joint, 'Joint'))

        # Timing tolerance
        tolerance = self.config['evaluation']['timing_tolerance']

        def apply_timing_tolerance_vec(observed, detected, tolerance=1):
            detected_tol = detected.copy()
            obs_indices = np.where(observed)[0]

            for i in obs_indices:
                start = max(0, i - tolerance)
                end = min(len(detected), i + tolerance + 1)
                if detected[start:end].any():
                    detected_tol[i] = True
            return detected_tol

        det_ewbi_tol = apply_timing_tolerance_vec(obs_severe_ewbi, det_severe_ewbi, tolerance)
        det_veg_tol = apply_timing_tolerance_vec(obs_severe_veg, det_severe_veg, tolerance)
        det_joint_tol = apply_timing_tolerance_vec(obs_severe_joint, det_severe_joint, tolerance)

        metrics.update(calc_metrics_vec(obs_severe_ewbi, det_ewbi_tol, 'EWBI_tol'))
        metrics.update(calc_metrics_vec(obs_severe_veg, det_veg_tol, 'Veg_tol'))
        metrics.update(calc_metrics_vec(obs_severe_joint, det_joint_tol, 'Joint_tol'))

        return metrics

    def save_results_to_csv(self):
        """Save results to CSV."""
        output_dir = os.path.join(self.config['output']['results_dir'], self.land_type)
        os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame({
            'timestamp': self.results['timestamps'],
            'observed_ewbi': self.results['area_mean_observed_ewbi'],
            'observed_veg': self.results['area_mean_observed_veg'],
            'predicted_ewbi': self.results['area_mean_predicted_ewbi'],
            'predicted_veg': self.results['area_mean_predicted_veg'],
            'residual_ewbi': self.results['area_mean_residuals_ewbi'],
            'residual_veg': self.results['area_mean_residuals_veg'],
            'anomaly_ewbi': self.results['anomalies_ewbi'],
            'anomaly_veg': self.results['anomalies_veg'],
            'observed_severe_ewbi': self.results['observed_severe_ewbi'].astype(int),
            'observed_severe_veg': self.results['observed_severe_veg'].astype(int),
            'detected_severe_ewbi': self.results['detected_severe_ewbi'].astype(int),
            'detected_severe_veg': self.results['detected_severe_veg'].astype(int),
            'detected_severe_joint': self.results['detected_severe_joint'].astype(int)
        })

        csv_path = os.path.join(output_dir, f'{self.land_type}_test_results_2022_2024.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_path = os.path.join(output_dir, f'{self.land_type}_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Metrics saved to {metrics_path}")

    def generate_plots_fast(self):
        """Generate plots with optimized rendering."""
        plots_dir = os.path.join(self.config['output']['plots_dir'], self.land_type)
        os.makedirs(plots_dir, exist_ok=True)

        fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        timestamps = self.results['timestamps']

        # EWBI panel
        ax = axes[0]
        ax.plot(timestamps, self.results['anomalies_ewbi'], 'b-', label='Observed Anomaly', alpha=0.8, linewidth=1)
        ax.axhline(y=stats.norm.ppf(0.05), color='r', linestyle='--', label='Severe Threshold', alpha=0.7)

        severe_indices = np.where(self.results['detected_severe_ewbi'])[0]
        for i in severe_indices:
            ax.axvspan(timestamps[max(0, i - 1)] if i > 0 else timestamps[0],
                       timestamps[min(i + 1, len(timestamps) - 1)] if i < len(timestamps) - 1 else timestamps[-1],
                       color='red', alpha=0.2)

        ax.set_ylabel('EWBI Anomaly')
        ax.legend()
        ax.set_title(f'{self.land_type.title()} - EWBI Detection')
        ax.grid(True, alpha=0.3)

        # Vegetation panel
        ax = axes[1]
        ax.plot(timestamps, self.results['anomalies_veg'], 'g-', label='Observed Anomaly', alpha=0.8, linewidth=1)
        ax.axhline(y=stats.norm.ppf(0.05), color='r', linestyle='--', label='Severe Threshold', alpha=0.7)

        severe_indices = np.where(self.results['detected_severe_veg'])[0]
        for i in severe_indices:
            ax.axvspan(timestamps[max(0, i - 1)] if i > 0 else timestamps[0],
                       timestamps[min(i + 1, len(timestamps) - 1)] if i < len(timestamps) - 1 else timestamps[-1],
                       color='orange', alpha=0.2)

        ax.set_ylabel('Vegetation Anomaly')
        ax.legend()
        ax.set_title(f'{self.land_type.title()} - Vegetation Detection')
        ax.grid(True, alpha=0.3)

        # Residuals panel
        ax = axes[2]
        ax.plot(timestamps, self.results['area_mean_residuals_ewbi'], 'b-', label='EWBI Residual', alpha=0.8,
                linewidth=1)
        ax.plot(timestamps, self.results['area_mean_residuals_veg'], 'g-', label='Veg Residual', alpha=0.8, linewidth=1)

        ax.axhline(y=self.thresholds['ewbi_residual'], color='b', linestyle=':', alpha=0.5)
        ax.axhline(y=-self.thresholds['ewbi_residual'], color='b', linestyle=':', alpha=0.5)
        ax.axhline(y=self.thresholds['veg_residual'], color='g', linestyle=':', alpha=0.5)
        ax.axhline(y=-self.thresholds['veg_residual'], color='g', linestyle=':', alpha=0.5)

        severe_indices = np.where(self.results['detected_severe_joint'])[0]
        for i in severe_indices:
            ax.axvspan(timestamps[max(0, i - 1)] if i > 0 else timestamps[0],
                       timestamps[min(i + 1, len(timestamps) - 1)] if i < len(timestamps) - 1 else timestamps[-1],
                       color='purple', alpha=0.2)

        ax.set_ylabel('Residual')
        ax.set_xlabel('Date')
        ax.legend()
        ax.set_title(f'{self.land_type.title()} - Residuals & Joint Detection')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(plots_dir, f'{self.land_type}_drought_detection.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f"Plot saved to {plot_path}")

    def print_metrics_table(self):
        """Print metrics table."""
        print(f"\n{'=' * 60}")
        print(f"METRICS FOR {self.land_type.upper()}")
        print(f"{'=' * 60}")

        m = self.results['metrics']

        print("\nWithout timing tolerance:")
        print(f"  EWBI   - POD: {m['EWBI_POD']:.3f}, FAR: {m['EWBI_FAR']:.3f}, CSI: {m['EWBI_CSI']:.3f}")
        print(f"  Veg    - POD: {m['Veg_POD']:.3f}, FAR: {m['Veg_FAR']:.3f}, CSI: {m['Veg_CSI']:.3f}")
        print(f"  Joint  - POD: {m['Joint_POD']:.3f}, FAR: {m['Joint_FAR']:.3f}, CSI: {m['Joint_CSI']:.3f}")

        print("\nWith ±1 month tolerance:")
        print(f"  EWBI   - POD: {m['EWBI_tol_POD']:.3f}, FAR: {m['EWBI_tol_FAR']:.3f}, CSI: {m['EWBI_tol_CSI']:.3f}")
        print(f"  Veg    - POD: {m['Veg_tol_POD']:.3f}, FAR: {m['Veg_tol_FAR']:.3f}, CSI: {m['Veg_tol_CSI']:.3f}")
        print(f"  Joint  - POD: {m['Joint_tol_POD']:.3f}, FAR: {m['Joint_tol_FAR']:.3f}, CSI: {m['Joint_tol_CSI']:.3f}")

        print(f"\nDetected: EWBI={self.results['detected_severe_ewbi'].sum()}, "
              f"Veg={self.results['detected_severe_veg'].sum()}, "
              f"Joint={self.results['detected_severe_joint'].sum()}")
        print(f"Observed: EWBI={self.results['observed_severe_ewbi'].sum()}, "
              f"Veg={self.results['observed_severe_veg'].sum()}")


# =============================================================================
# ULTRA-OPTIMIZED MAIN EXECUTION PIPELINE
# =============================================================================

def train_and_evaluate_single_landtype_optimized(config, land_type):
    """Ultra-optimized training and evaluation pipeline with memory management."""
    print(f"\n{'=' * 70}")
    print(f"PROCESSING: {land_type.upper()}")
    print(f"{'=' * 70}")

    start_time = time.time()

    try:
        # Memory monitoring
        print(f"Initial memory: {monitor_memory_usage():.1f}GB")
        aggressive_memory_cleanup()

        # Training
        print(f"\n--- TRAINING ---")
        trainer = OptimizedTrainer(config, land_type)
        best_checkpoint = trainer.train()

        train_time = time.time() - start_time
        print(f"Training completed in {train_time / 60:.1f} minutes")
        print(f"Memory after training: {monitor_memory_usage():.1f}GB")

        # Clear trainer from memory
        del trainer
        aggressive_memory_cleanup()

        # Evaluation
        print(f"\n--- EVALUATION ---")
        evaluator = FastDroughtEvaluator(config, land_type, best_checkpoint)
        metrics = evaluator.evaluate_test_set_fast()

        # Save outputs
        print(f"\n--- SAVING OUTPUTS ---")
        evaluator.save_results_to_csv()
        evaluator.generate_plots_fast()
        evaluator.print_metrics_table()

        total_time = time.time() - start_time
        print(f"\n✅ {land_type} completed in {total_time / 60:.1f} minutes")
        print(f"Final memory usage: {monitor_memory_usage():.1f}GB")

        # Cleanup
        del evaluator
        aggressive_memory_cleanup()

        return {
            'status': 'Success',
            'time': total_time / 60,
            'checkpoint': best_checkpoint,
            'metrics': metrics
        }

    except Exception as e:
        total_time = time.time() - start_time
        print(f"❌ Error processing {land_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        aggressive_memory_cleanup()

        return {
            'status': 'Failed',
            'time': total_time / 60,
            'error': str(e)
        }


def main_optimized():
    """Ultra-optimized main pipeline with comprehensive optimizations."""
    print("=" * 80)
    print("ULTRA-OPTIMIZED CALIFORNIA DROUGHT GNN FORECASTER")
    print("Performance Optimizations:")
    print("- Numba JIT compilation for hot loops")
    print("- Memory-mapped arrays for large datasets")
    print("- Compressed caching with granular levels")
    print("- Vectorized operations and parallel processing")
    print("- Aggressive memory management")
    print("- Sparse adjacency matrices")
    print("- Optimized file I/O with windowing")
    print(f"- Memory limit: {CONFIG['optimization']['memory_limit_gb']}GB")
    if NUMBA_AVAILABLE:
        print("- Numba acceleration: ENABLED")
    else:
        print("- Numba acceleration: DISABLED (install numba for better performance)")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # System info
    print(f"\nSystem Information:")
    print(f"CPU cores: {mp.cpu_count()}")
    print(f"Available memory: {psutil.virtual_memory().total / 1024 ** 3:.1f}GB")
    print(f"Max workers: {CONFIG['optimization']['max_workers']}")
    print(f"Initial memory usage: {monitor_memory_usage():.1f}GB")

    # Create directories
    for output_dir in CONFIG['output'].values():
        os.makedirs(output_dir, exist_ok=True)

    # Verify data paths
    for base_dir in [CONFIG['data']['train_val_base'], CONFIG['data']['test_base']]:
        if not os.path.exists(base_dir):
            print(f"❌ Missing: {base_dir}")
            return

    # Process all land types with optimizations
    land_types = CONFIG['data']['land_types']
    overall_start = time.time()
    results = {}

    for i, land_type in enumerate(land_types, 1):
        print(f"\n🚀 PROCESSING {i}/{len(land_types)}: {land_type.upper()}")
        print(f"Memory before processing: {monitor_memory_usage():.1f}GB")

        # Enforce memory limit before each land type
        enforce_memory_limit(CONFIG['optimization']['memory_limit_gb'])

        result = train_and_evaluate_single_landtype_optimized(CONFIG, land_type)
        results[land_type] = result

        # Progress update
        print(f"\nProgress Summary:")
        for lt, res in results.items():
            status = "✅" if res['status'] == 'Success' else "❌"
            print(f"  {status} {lt}: {res['status']} ({res['time']:.1f}min)")

        # Aggressive cleanup between land types
        print("Cleaning up between land types...")
        aggressive_memory_cleanup()
        print(f"Memory after cleanup: {monitor_memory_usage():.1f}GB")

    # Final summary
    total_time = time.time() - overall_start
    print(f"\n{'=' * 80}")
    print("FINAL RESULTS")
    print(f"{'=' * 80}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final memory usage: {monitor_memory_usage():.1f}GB")

    successful = sum(1 for r in results.values() if r['status'] == 'Success')
    print(f"Success rate: {successful}/{len(land_types)} land types")

    if successful > 0:
        print(f"\nOutputs:")
        for key, path in CONFIG['output'].items():
            print(f"  {key}: {path}")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Final cleanup
    aggressive_memory_cleanup()

    return results


# =============================================================================
# RUN THE ULTRA-OPTIMIZED PIPELINE
# =============================================================================

if __name__ == "__main__":
    set_seed(CONFIG['training']['seed'])
    results = main_optimized()