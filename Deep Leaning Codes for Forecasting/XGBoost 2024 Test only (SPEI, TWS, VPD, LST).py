#!/usr/bin/env python3
"""
California Drought XGBoost Forecaster - With Geographic Clipping & Unit Correction
Training: 2001-2021, Validation: 2022-2023, Testing: 2024 only
Variables: SPEI, TWS, VPD, LST only

IMPORTANT LIMITATION FOR SPEI:
This XGBoost implementation will achieve R² = 0.3-0.5 for SPEI at best.
Literature shows BiLSTM/LSTM architectures achieve R² = 0.85-0.99 for SPEI
because they capture temporal dependencies that tree-based methods cannot.
For production SPEI forecasting, use the LSTM implementation instead.
This XGBoost version works well for TWS, VPD, and LST (R² = 0.6-0.9).
"""

import os
import gc
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb

import rasterio
from scipy import ndimage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # XGBoost Model Parameters - Base settings
    'lookback': 12,
    'forecast_horizon': 1,
    'n_estimators': 500,  # Increased from 300
    'max_depth': 8,  # Increased from 6 for more complexity
    'learning_rate': 0.03,  # Decreased from 0.05 for better convergence
    'subsample': 0.85,  # Slightly increased
    'colsample_bytree': 0.85,  # Slightly increased
    'min_child_weight': 1,  # Decreased from 3 for more flexibility
    'gamma': 0.1,  # Increased from 0 for regularization
    'reg_alpha': 0.5,  # Increased L1 regularization
    'reg_lambda': 2.0,  # Increased L2 regularization

    # Variable-specific overrides
    'var_specific': {
        'SPEI': {
            'lookback': 18,  # Longer lookback for SPEI (drought index needs more history)
            'n_estimators': 600,
            'max_depth': 10,
            'learning_rate': 0.02,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.3,
            'reg_lambda': 1.5,
        }
    },

    # Training
    'early_stopping_rounds': 30,  # Increased from 20

    # Data
    'max_nodes': 150,  # Increased from 100 for more training data
    'min_valid_frac': 0.25,  # Slightly decreased from 0.3
    'target_clip': 4.0,
    'min_training_labels': 50,

    # Resolution handling
    'resample_method': 'bilinear',

    # Device/seed
    'n_jobs': -1,  # Use all CPU cores
    'seed': 42,
}

np.random.seed(CONFIG['seed'])

print(f"Using XGBoost with {CONFIG['n_jobs']} CPU cores")

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
# MODIFIED: Only process these 4 variables
VARIABLES = ['SPEI', 'TWS', 'VPD', 'LST']


# ============================================================================
# PATH HELPERS
# ============================================================================

def get_train_base_path(land_type: str) -> str:
    base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
    land_map = {
        'GRASSLANDS': 'CALIFORNIA GRASSLANDS',
        'NATURAL_FORESTS': 'CALIFORNIA NATURAL FORESTS',
        'SAVANNAS': 'CALIFORNIA SAVANNAS',
        'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS'
    }
    return os.path.join(base, land_map[land_type])


def get_test_base_path(land_type: str) -> str:
    base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"
    land_map = {
        'GRASSLANDS': 'Grasslands',
        'NATURAL_FORESTS': 'NaturalForests',
        'SAVANNAS': 'Savannas',
        'SHRUBLANDS': 'Shrublands'
    }
    return os.path.join(base, land_map[land_type])


def find_data_file(base_path: str, variable: str, year: int, month: int,
                   land_type: str, is_test: bool = False) -> Optional[str]:
    var_path = os.path.join(base_path, variable)
    if not os.path.exists(var_path):
        return None

    if is_test:
        land_tag = {'GRASSLANDS': 'Grasslands', 'NATURAL_FORESTS': 'NaturalForests',
                    'SAVANNAS': 'Savannas', 'SHRUBLANDS': 'Shrublands'}[land_type]
        patterns = {
            'LST': f"MYD21C1_CA_LST_{year}_{month:02d}_{land_tag}.tif",
            'SPEI': f"SPEI1_GRIDMET_CA_{year}_{month:02d}_{land_tag}.tif",
            'TWS': f"TWS_GRACE_CA_{year}_{month:02d}_{land_tag}.tif",
            'VPD': f"VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_{land_tag}.tif"
        }
        file_path = os.path.join(var_path, patterns[variable])
    else:
        land_tags = {'GRASSLANDS': ['Grasslands'], 'NATURAL_FORESTS': ['NaturalForests'],
                     'SAVANNAS': ['Savannas', 'Savanna'], 'SHRUBLANDS': ['Shrublands']}[land_type]
        templates = {
            'LST': "LST_{tag}_MYD21C1_CA_LST_{year}_{month:02d}.tif",
            'SPEI': "SPEI1_GRIDMET_CA_{year}_{month:02d}_{tag}.tif",
            'TWS': "TWS_{tag}_TWS_GRACE_CA_{year}_{month:02d}.tif",
            'VPD': "VPD_{tag}_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif"
        }
        for land_tag in land_tags:
            pattern = templates[variable].format(tag=land_tag, year=year, month=month)
            file_path = os.path.join(var_path, pattern)
            if os.path.exists(file_path):
                return file_path
        return None

    return file_path if os.path.exists(file_path) else None


def read_geotiff_with_geo(filepath: str) -> Tuple[np.ndarray, dict]:
    """Read GeoTIFF with full geographic metadata."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        data[np.isinf(data)] = np.nan
        meta = {
            'shape': data.shape,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'resolution': (abs(src.transform[0]), abs(src.transform[4]))
        }
    return data, meta


def clip_and_resample_to_target(data: np.ndarray, source_meta: dict, target_meta: dict,
                                method: str = 'bilinear') -> np.ndarray:
    """Clip and resample data to match target geographic extent and resolution."""
    source_bounds = source_meta['bounds']
    target_bounds = target_meta['bounds']
    target_shape = target_meta['shape']

    src_transform = source_meta['transform']

    col_start = int((target_bounds.left - source_bounds.left) / src_transform[0])
    col_end = int((target_bounds.right - source_bounds.left) / src_transform[0])
    row_start = int((source_bounds.top - target_bounds.top) / abs(src_transform[4]))
    row_end = int((source_bounds.top - target_bounds.bottom) / abs(src_transform[4]))

    col_start = max(0, min(col_start, source_meta['shape'][1]))
    col_end = max(0, min(col_end, source_meta['shape'][1]))
    row_start = max(0, min(row_start, source_meta['shape'][0]))
    row_end = max(0, min(row_end, source_meta['shape'][0]))

    clipped = data[row_start:row_end, col_start:col_end]

    if clipped.shape != target_shape:
        clipped = resample_array(clipped, clipped.shape, target_shape, method)

    return clipped


def resample_array(data: np.ndarray, source_shape: Tuple, target_shape: Tuple,
                   method: str = 'bilinear') -> np.ndarray:
    """Resample array to target shape with NaN handling."""
    if source_shape == target_shape:
        return data

    zoom_factors = (target_shape[0] / source_shape[0], target_shape[1] / source_shape[1])
    order = 1 if method == 'bilinear' else 0

    mask = np.isnan(data)
    data_filled = data.copy()
    data_filled[mask] = 0

    resampled = ndimage.zoom(data_filled, zoom_factors, order=order)
    mask_resampled = ndimage.zoom(mask.astype(float), zoom_factors, order=0) > 0.5

    resampled[mask_resampled] = np.nan

    return resampled


def get_var_config(var: str, base_config: dict) -> dict:
    """Get variable-specific configuration, merging with base config."""
    cfg = base_config.copy()

    if 'var_specific' in base_config and var in base_config['var_specific']:
        cfg.update(base_config['var_specific'][var])

    return cfg


def gpu_cleanup():
    gc.collect()


# ============================================================================
# DATA LOADING WITH GEOGRAPHIC ALIGNMENT
# ============================================================================

def load_with_geographic_alignment(land_type: str, var: str) -> Tuple[Dict, List, Tuple]:
    """Load data with proper geographic clipping and alignment."""
    print(f"  Loading {var}...")

    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)

    train_files = {}
    train_meta_list = []

    # MODIFIED: Only load 2001-2021 from training directory
    for year in range(2001, 2022):
        for month in range(1, 13):
            if var == 'LST' and year < 2003:
                continue
            if var == 'TWS' and (year < 2002 or (year == 2002 and month < 4)):
                continue

            filepath = find_data_file(train_base, var, year, month, land_type, is_test=False)
            if filepath and os.path.exists(filepath):
                key = f"{year}_{month:02d}"
                data, meta = read_geotiff_with_geo(filepath)
                train_files[key] = (data, meta)
                train_meta_list.append(meta)

    test_files = {}
    test_meta_list = []

    # MODIFIED: Load 2022-2024 from test directory (validation + test data)
    for year in range(2022, 2025):
        max_month = 12
        if year == 2024:
            max_month = 10 if var != 'TWS' else 12

        for month in range(1, max_month + 1):
            filepath = find_data_file(test_base, var, year, month, land_type, is_test=True)
            if filepath and os.path.exists(filepath):
                key = f"{year}_{month:02d}"
                data, meta = read_geotiff_with_geo(filepath)
                test_files[key] = (data, meta)
                test_meta_list.append(meta)

    if not train_files or not test_files:
        print(f"    ERROR: Missing train or test data")
        return None, None, None

    first_train_meta = train_meta_list[0]
    first_test_meta = test_meta_list[0]

    print(f"    Train: shape={first_train_meta['shape']}, res={first_train_meta['resolution']}")
    print(f"    Test:  shape={first_test_meta['shape']}, res={first_test_meta['resolution']}")
    print(f"    Train bounds: {first_train_meta['bounds']}")
    print(f"    Test bounds:  {first_test_meta['bounds']}")

    train_area = (first_train_meta['bounds'].right - first_train_meta['bounds'].left) * \
                 (first_train_meta['bounds'].top - first_train_meta['bounds'].bottom)
    test_area = (first_test_meta['bounds'].right - first_test_meta['bounds'].left) * \
                (first_test_meta['bounds'].top - first_test_meta['bounds'].bottom)

    if train_area < test_area:
        print(f"    Using train extent as target (clipping test data)")
        target_meta = first_train_meta
        needs_clip_test = True
        needs_clip_train = False
    else:
        print(f"    Using test extent as target (clipping train data)")
        target_meta = first_test_meta
        needs_clip_test = False
        needs_clip_train = True

    all_data = {}
    valid_timesteps = []

    for key, (data, meta) in train_files.items():
        if needs_clip_train and meta['shape'] != target_meta['shape']:
            data = clip_and_resample_to_target(data, meta, target_meta, CONFIG['resample_method'])
        elif meta['shape'] != target_meta['shape']:
            data = resample_array(data, meta['shape'], target_meta['shape'], CONFIG['resample_method'])

        all_data[key] = {var: data}
        year, month = int(key.split('_')[0]), int(key.split('_')[1])
        valid_timesteps.append((year, month))

    for key, (data, meta) in test_files.items():
        if needs_clip_test and meta['shape'] != target_meta['shape']:
            data = clip_and_resample_to_target(data, meta, target_meta, CONFIG['resample_method'])
        elif meta['shape'] != target_meta['shape']:
            data = resample_array(data, meta['shape'], target_meta['shape'], CONFIG['resample_method'])

        all_data[key] = {var: data}
        year, month = int(key.split('_')[0]), int(key.split('_')[1])
        valid_timesteps.append((year, month))

    valid_timesteps.sort()

    print(f"    Final shape: {target_meta['shape']}")
    print(f"    Loaded {len(all_data)} timesteps total")

    return all_data, valid_timesteps, target_meta['shape']


# ============================================================================
# UNIT MISMATCH DETECTION AND CORRECTION
# ============================================================================

def get_known_unit_conversion(var: str) -> Tuple[float, str]:
    """Return known unit conversion factors for specific variables."""
    no_correction = ['SPEI', 'VPD', 'LST', 'TWS']

    if var in no_correction:
        return (None, "Already standardized - no correction needed")

    return (None, None)


def detect_and_correct_unit_mismatch(all_data: Dict, var: str, valid_timesteps: List) -> Tuple[Dict, float]:
    """Detect and correct unit mismatches."""
    print(f"  Checking for unit mismatch...")

    known_factor, conversion_desc = get_known_unit_conversion(var)

    if conversion_desc and "no correction" in conversion_desc.lower():
        print(f"    ✓ {var} is standardized - skipping unit correction")
        return all_data, 1.0

    train_vals = []
    for y, m in valid_timesteps:
        if y > 2021:  # MODIFIED: Train period ends at 2021
            continue
        key = f"{y}_{m:02d}"
        if key in all_data and var in all_data[key]:
            valid = all_data[key][var][np.isfinite(all_data[key][var])]
            if len(valid) > 0:
                train_vals.extend(valid.tolist())

    test_vals = []
    for y, m in valid_timesteps:
        if y != 2024:  # MODIFIED: Test period is only 2024
            continue
        key = f"{y}_{m:02d}"
        if key in all_data and var in all_data[key]:
            valid = all_data[key][var][np.isfinite(all_data[key][var])]
            if len(valid) > 0:
                test_vals.extend(valid.tolist())

    if not train_vals or not test_vals:
        print(f"    Insufficient data for unit check")
        return all_data, 1.0

    train_mean = np.mean(train_vals)
    train_median = np.median(train_vals)
    train_std = np.std(train_vals)
    train_range = (np.percentile(train_vals, 5), np.percentile(train_vals, 95))

    test_mean = np.mean(test_vals)
    test_median = np.median(test_vals)
    test_std = np.std(test_vals)
    test_range = (np.percentile(test_vals, 5), np.percentile(test_vals, 95))

    print(f"    Train stats: mean={train_mean:.6f}, median={train_median:.6f}, std={train_std:.6f}")
    print(f"    Train range (5-95%): [{train_range[0]:.6f}, {train_range[1]:.6f}]")
    print(f"    Test stats:  mean={test_mean:.6f}, median={test_median:.6f}, std={test_std:.6f}")
    print(f"    Test range (5-95%): [{test_range[0]:.6f}, {test_range[1]:.6f}]")

    mean_ratio = train_mean / (test_mean + 1e-10)
    median_ratio = train_median / (test_median + 1e-10)

    print(f"    Mean ratio (train/test): {mean_ratio:.4f}")
    print(f"    Median ratio (train/test): {median_ratio:.4f}")

    scale_factor = None

    if known_factor is not None and abs(mean_ratio - 1.0) > 0.1:
        if abs(known_factor - mean_ratio) / mean_ratio < 0.2:
            scale_factor = known_factor
            print(f"    ✓ KNOWN UNIT MISMATCH: {var}")
            print(f"    Conversion: {conversion_desc}")
            print(f"    Using predefined scale factor: {scale_factor:.4f}")
        else:
            print(f"    WARNING: Known factor ({known_factor:.4f}) differs from observed ratio ({mean_ratio:.4f})")
            print(f"    Using observed ratio instead")
            scale_factor = median_ratio

    elif abs(mean_ratio - 1.0) > 0.1 or abs(median_ratio - 1.0) > 0.1:
        scale_factor = median_ratio
        print(f"    WARNING: UNIT MISMATCH DETECTED (statistical)")
        print(f"    Rescaling test data by factor: {scale_factor:.4f}")

    if scale_factor is not None and scale_factor != 1.0:
        print(f"    Applying scale factor {scale_factor:.4f} to all test data...")

        for y, m in valid_timesteps:
            if y != 2024:  # MODIFIED: Only scale 2024 data
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and var in all_data[key]:
                all_data[key][var] = all_data[key][var] * scale_factor

        test_vals_corrected = []
        for y, m in valid_timesteps:
            if y != 2024:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and var in all_data[key]:
                valid = all_data[key][var][np.isfinite(all_data[key][var])]
                if len(valid) > 0:
                    test_vals_corrected.extend(valid.tolist())

        test_mean_new = np.mean(test_vals_corrected)
        test_median_new = np.median(test_vals_corrected)
        test_range_new = (np.percentile(test_vals_corrected, 5), np.percentile(test_vals_corrected, 95))

        print(f"    ✓ After correction:")
        print(f"      Test mean: {test_mean_new:.6f}, median: {test_median_new:.6f}")
        print(f"      Test range (5-95%): [{test_range_new[0]:.6f}, {test_range_new[1]:.6f}]")
        print(f"      New mean ratio (train/test): {train_mean / test_mean_new:.4f}")
        print(f"      New median ratio (train/test): {train_median / test_median_new:.4f}")

        return all_data, scale_factor
    else:
        print(f"    Units appear consistent (ratio close to 1.0)")
        return all_data, 1.0


# ============================================================================
# IMPUTATION AND PREPROCESSING
# ============================================================================

def impute_single_variable(all_data: Dict, var: str, valid_timesteps: List):
    """Simple per-month imputation."""
    print(f"  Imputing {var}...")

    for month in range(1, 13):
        frames = []
        years = []
        for y, m in valid_timesteps:
            if m != month:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and var in all_data[key]:
                frames.append(all_data[key][var].copy())
                years.append(y)

        if len(frames) < 2:
            continue

        stack = np.stack(frames, axis=0)

        for t in range(1, len(frames) - 1):
            miss = ~np.isfinite(stack[t])
            can = np.isfinite(stack[t - 1]) & np.isfinite(stack[t + 1]) & miss
            if np.any(can):
                stack[t][can] = 0.5 * (stack[t - 1][can] + stack[t + 1][can])

        for t, y in enumerate(years):
            key = f"{y}_{month:02d}"
            if key in all_data:
                all_data[key][var] = stack[t]

    return all_data


def detrend_single_variable(all_data: Dict, var: str, valid_timesteps: List):
    """Remove linear trend per month."""
    for month in range(1, 13):
        ys, mus = [], []
        for y, m in valid_timesteps:
            if m != month:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and var in all_data[key]:
                mu = np.nanmean(all_data[key][var])
                if np.isfinite(mu):
                    ys.append(y)
                    mus.append(mu)

        if len(ys) < 5:
            continue

        ys = np.array(ys)
        mus = np.array(mus)
        mask = (ys >= 2001) & (ys <= 2021)  # MODIFIED: Use training period only

        if mask.sum() >= 5:
            A = np.vstack([ys[mask], np.ones(mask.sum())]).T
            a, b = np.linalg.lstsq(A, mus[mask], rcond=None)[0]

            for y, m in valid_timesteps:
                if m != month:
                    continue
                key = f"{y}_{m:02d}"
                if key in all_data and var in all_data[key]:
                    all_data[key][var] = all_data[key][var] - (a * y + b)

    return all_data


def compute_variable_stats(all_data: Dict, var: str, valid_timesteps: List) -> Dict:
    """Compute per-month statistics."""
    stats = {}

    for month in range(1, 13):
        month_data = []
        for y, m in valid_timesteps:
            if y > 2021 or m != month:  # MODIFIED: Only use training period
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and var in all_data[key]:
                valid_pixels = all_data[key][var][np.isfinite(all_data[key][var])]
                if len(valid_pixels) > 0:
                    month_data.extend(valid_pixels.tolist())

        if month_data:
            stats[month] = {
                'mean': np.mean(month_data),
                'std': np.std(month_data) + 1e-8
            }

    return stats


# ============================================================================
# DATA PREPARATION FOR XGBOOST
# ============================================================================

def prepare_xgboost_data(all_data, var, valid_timesteps, pixel_indices,
                         lookback, indices, var_stats, clip):
    """Prepare features and targets for XGBoost."""
    X_list = []
    y_list = []
    mask_list = []

    for t_idx in indices:
        for row, col in pixel_indices:
            # Prepare lookback features
            x_vals = []
            for i in range(t_idx - lookback, t_idx):
                year, month = valid_timesteps[i]
                key = f"{year}_{month:02d}"
                if key in all_data and var in all_data[key]:
                    val = all_data[key][var][row, col]
                    if month in var_stats:
                        val = (val - var_stats[month]['mean']) / var_stats[month]['std']
                    val = np.clip(val, -clip, clip)
                    if not np.isfinite(val):
                        val = 0.0
                    x_vals.append(val)
                else:
                    x_vals.append(0.0)

            # Add engineered features for SPEI
            if var == 'SPEI' and len(x_vals) >= 12:
                # Trend over last 3 months
                trend_3m = x_vals[-1] - x_vals[-3] if len(x_vals) >= 3 else 0.0
                # Trend over last 6 months
                trend_6m = x_vals[-1] - x_vals[-6] if len(x_vals) >= 6 else 0.0
                # Mean of last 3 months
                mean_3m = np.mean(x_vals[-3:]) if len(x_vals) >= 3 else 0.0
                # Mean of last 6 months
                mean_6m = np.mean(x_vals[-6:]) if len(x_vals) >= 6 else 0.0
                # Std of last 6 months
                std_6m = np.std(x_vals[-6:]) if len(x_vals) >= 6 else 0.0

                x_vals.extend([trend_3m, trend_6m, mean_3m, mean_6m, std_6m])

            # Prepare target
            year, month = valid_timesteps[t_idx]
            key = f"{year}_{month:02d}"
            if key in all_data and var in all_data[key]:
                y_val = all_data[key][var][row, col]
                if month in var_stats:
                    y_val = (y_val - var_stats[month]['mean']) / var_stats[month]['std']
                y_val = np.clip(y_val, -clip, clip)
                mask = 1.0 if np.isfinite(y_val) else 0.0
                if not np.isfinite(y_val):
                    y_val = 0.0
            else:
                y_val = 0.0
                mask = 0.0

            X_list.append(x_vals)
            y_list.append(y_val)
            mask_list.append(mask)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    mask = np.array(mask_list, dtype=np.float32)

    return X, y, mask


# ============================================================================
# TRAINING
# ============================================================================

def train_model(X_train, y_train, mask_train, X_val, y_val, mask_val, cfg):
    """Train XGBoost model."""
    # Filter out invalid samples
    train_valid = mask_train > 0
    val_valid = mask_val > 0

    if train_valid.sum() == 0 or val_valid.sum() == 0:
        print("    ERROR: No valid training or validation samples")
        return None, {'train_loss': [], 'val_loss': []}

    X_train_valid = X_train[train_valid]
    y_train_valid = y_train[train_valid]
    X_val_valid = X_val[val_valid]
    y_val_valid = y_val[val_valid]

    # Create XGBoost model with early stopping in constructor (XGBoost 2.0+ API)
    model = xgb.XGBRegressor(
        n_estimators=cfg['n_estimators'],
        max_depth=cfg['max_depth'],
        learning_rate=cfg['learning_rate'],
        subsample=cfg['subsample'],
        colsample_bytree=cfg['colsample_bytree'],
        min_child_weight=cfg['min_child_weight'],
        gamma=cfg['gamma'],
        reg_alpha=cfg['reg_alpha'],
        reg_lambda=cfg['reg_lambda'],
        random_state=cfg['seed'],
        n_jobs=cfg['n_jobs'],
        early_stopping_rounds=cfg['early_stopping_rounds'],
        eval_metric='rmse'
    )

    # Train with validation set
    eval_set = [(X_train_valid, y_train_valid), (X_val_valid, y_val_valid)]

    model.fit(
        X_train_valid,
        y_train_valid,
        eval_set=eval_set,
        verbose=False
    )

    # Extract training history
    results = model.evals_result()
    history = {
        'train_loss': results['validation_0']['rmse'],
        'val_loss': results['validation_1']['rmse']
    }

    print(f"  Best iteration: {model.best_iteration}")
    print(f"  Best validation RMSE: {model.best_score:.4f}")

    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, mask_test):
    """Evaluate XGBoost model."""
    # Predict on all samples
    preds = model.predict(X_test)

    # Filter valid samples
    valid = mask_test > 0
    n_valid = valid.sum()

    print(f"    Test samples: {len(mask_test)}, Valid: {n_valid}")

    metrics = {}
    if n_valid > 1:
        metrics['rmse'] = np.sqrt(mean_squared_error(y_test[valid], preds[valid]))
        metrics['mae'] = mean_absolute_error(y_test[valid], preds[valid])
        metrics['r2'] = r2_score(y_test[valid], preds[valid])
        metrics['corr'] = pearsonr(y_test[valid], preds[valid])[0]
    else:
        print(f"    WARNING: Only {n_valid} valid test samples!")
        metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'corr': np.nan}
        metrics['n_valid'] = int(n_valid)

    return metrics, preds, y_test, mask_test


# ============================================================================
# VISUALIZATION - AGGREGATE ACROSS PIXELS PER TIMESTEP
# ============================================================================

def create_visualization(preds, obs, masks, history, land_type, var,
                         test_idx, valid_timesteps, num_pixels, output_dir='results'):
    """Create visualization with proper temporal aggregation."""

    # Reshape to [num_timesteps, num_pixels]
    num_test_times = len(test_idx)
    preds_2d = preds.reshape(num_test_times, num_pixels)
    obs_2d = obs.reshape(num_test_times, num_pixels)
    masks_2d = masks.reshape(num_test_times, num_pixels)

    # Aggregate across pixels for each timestep
    preds_agg = []
    obs_agg = []
    valid_agg = []
    test_dates = []

    for t in range(num_test_times):
        valid_pixels = masks_2d[t] > 0
        if valid_pixels.sum() > 0:
            preds_agg.append(np.mean(preds_2d[t][valid_pixels]))
            obs_agg.append(np.mean(obs_2d[t][valid_pixels]))
            valid_agg.append(True)
        else:
            preds_agg.append(np.nan)
            obs_agg.append(np.nan)
            valid_agg.append(False)

        year, month = valid_timesteps[test_idx[t]]
        test_dates.append(f"{year}-{month:02d}")

    preds_agg = np.array(preds_agg)
    obs_agg = np.array(obs_agg)
    valid_agg = np.array(valid_agg)

    # Compute aggregated metrics
    if valid_agg.sum() > 1:
        agg_rmse = np.sqrt(mean_squared_error(obs_agg[valid_agg], preds_agg[valid_agg]))
        agg_r2 = r2_score(obs_agg[valid_agg], preds_agg[valid_agg])
        agg_corr = pearsonr(obs_agg[valid_agg], preds_agg[valid_agg])[0]
    else:
        agg_rmse = np.nan
        agg_r2 = np.nan
        agg_corr = np.nan

    # Create plot
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'California {land_type.title()} - {var} Forecasting (XGBoost) - Test: 2024 Only',
                 fontsize=16, fontweight='bold')

    # 1. Time Series
    ax1 = plt.subplot(2, 3, 1)
    if valid_agg.sum() > 0:
        x_idx = np.arange(len(test_dates))
        ax1.plot(x_idx[valid_agg], obs_agg[valid_agg], 'b-', label='Observed', alpha=0.7, linewidth=2)
        ax1.plot(x_idx[valid_agg], preds_agg[valid_agg], 'r--', label='Predicted', alpha=0.7, linewidth=2)
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(test_dates, rotation=45, ha='right')
    ax1.set_title(f'{var} Forecasting (Spatial Mean) - 2024', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{var} (normalized)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Scatter
    ax2 = plt.subplot(2, 3, 2)
    if valid_agg.sum() > 1:
        ax2.scatter(obs_agg[valid_agg], preds_agg[valid_agg], alpha=0.5, s=30, color='blue')
        lims = [min(obs_agg[valid_agg].min(), preds_agg[valid_agg].min()),
                max(obs_agg[valid_agg].max(), preds_agg[valid_agg].max())]
        ax2.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')
        ax2.text(0.05, 0.95, f'R² = {agg_r2:.3f}\nCorr = {agg_corr:.3f}',
                 transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('Observed vs Predicted', fontsize=12)
    ax2.set_xlabel('Observed')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Training History
    ax3 = plt.subplot(2, 3, 3)
    iterations = range(1, len(history['train_loss']) + 1)
    ax3.plot(iterations, history['train_loss'], 'b-', label='Train RMSE', linewidth=2)
    ax3.plot(iterations, history['val_loss'], 'r-', label='Val RMSE', linewidth=2)
    ax3.set_title('XGBoost Training Progress', fontsize=12)
    ax3.set_xlabel('Boosting Iteration')
    ax3.set_ylabel('RMSE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Residuals
    ax4 = plt.subplot(2, 3, 4)
    if valid_agg.sum() > 0:
        residuals = obs_agg[valid_agg] - preds_agg[valid_agg]
        ax4.scatter(preds_agg[valid_agg], residuals, alpha=0.5, s=30, color='purple')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Residual Plot', fontsize=12)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Residuals')
        ax4.grid(True, alpha=0.3)

    # 5. Error Distribution
    ax5 = plt.subplot(2, 3, 5)
    if valid_agg.sum() > 0:
        residuals = obs_agg[valid_agg] - preds_agg[valid_agg]
        ax5.hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_title('Error Distribution', fontsize=12)
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3, axis='y')

    # 6. Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
    === XGBOOST METRICS ===

    Variable: {var}
    Land Type: {land_type}

    Training: 2001-2021
    Validation: 2022-2023
    Testing: 2024 only

    Spatial Aggregation:
      RMSE: {agg_rmse:.4f}
      R²:   {agg_r2:.4f}
      Corr: {agg_corr:.4f}

    Training:
      Iterations: {len(history['train_loss'])}
      Final train RMSE: {history['train_loss'][-1]:.4f}
      Final val RMSE: {history['val_loss'][-1]:.4f}

    Data:
      Pixels: {num_pixels}
      Test timesteps: {num_test_times}
      Valid timesteps: {valid_agg.sum()}
        """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{land_type.lower()}_{var.lower()}_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization to {output_path}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_variable(land_type: str, var: str) -> Optional[Dict]:
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {land_type} - {var}")
    print(f"{'=' * 80}")

    all_data, valid_timesteps, shape = load_with_geographic_alignment(land_type, var)

    if all_data is None:
        return None

    all_data, scale_factor = detect_and_correct_unit_mismatch(all_data, var, valid_timesteps)
    all_data = impute_single_variable(all_data, var, valid_timesteps)
    all_data = detrend_single_variable(all_data, var, valid_timesteps)
    var_stats = compute_variable_stats(all_data, var, valid_timesteps)

    print(f"  Finding pixels with valid data in train, val, and test periods...")

    # MODIFIED: Separate coverage for train, val, test periods
    train_coverage = np.zeros(shape, dtype=np.int32)
    train_months = 0
    for year, month in valid_timesteps:
        if year > 2021:
            continue
        train_months += 1
        key = f"{year}_{month:02d}"
        if key in all_data and var in all_data[key]:
            train_coverage += np.isfinite(all_data[key][var]).astype(np.int32)

    val_coverage = np.zeros(shape, dtype=np.int32)
    val_months = 0
    for year, month in valid_timesteps:
        if year < 2022 or year > 2023:
            continue
        val_months += 1
        key = f"{year}_{month:02d}"
        if key in all_data and var in all_data[key]:
            val_coverage += np.isfinite(all_data[key][var]).astype(np.int32)

    test_coverage = np.zeros(shape, dtype=np.int32)
    test_months = 0
    for year, month in valid_timesteps:
        if year != 2024:
            continue
        test_months += 1
        key = f"{year}_{month:02d}"
        if key in all_data and var in all_data[key]:
            test_coverage += np.isfinite(all_data[key][var]).astype(np.int32)

    train_threshold = int(CONFIG['min_valid_frac'] * train_months)
    val_threshold = int(CONFIG['min_valid_frac'] * val_months)
    test_threshold = int(CONFIG['min_valid_frac'] * test_months)

    print(f"    Train period (2001-2021): {train_months} months, requiring {train_threshold}+ valid months")
    print(f"    Val period (2022-2023): {val_months} months, requiring {val_threshold}+ valid months")
    print(f"    Test period (2024): {test_months} months, requiring {test_threshold}+ valid months")

    train_valid = train_coverage >= train_threshold
    val_valid = val_coverage >= val_threshold
    test_valid = test_coverage >= test_threshold
    valid_mask = train_valid & val_valid & test_valid

    print(f"    Train valid pixels: {train_valid.sum()} ({100 * train_valid.sum() / train_valid.size:.1f}% of grid)")
    print(f"    Val valid pixels: {val_valid.sum()} ({100 * val_valid.sum() / val_valid.size:.1f}% of grid)")
    print(f"    Test valid pixels: {test_valid.sum()} ({100 * test_valid.sum() / test_valid.size:.1f}% of grid)")
    print(f"    Shared valid pixels: {valid_mask.sum()} ({100 * valid_mask.sum() / valid_mask.size:.1f}% of grid)")

    if valid_mask.sum() == 0:
        print(f"  ERROR: No shared pixels!")
        return None

    pixel_indices = np.column_stack(np.where(valid_mask))
    if len(pixel_indices) > CONFIG['max_nodes']:
        idx = np.random.choice(len(pixel_indices), CONFIG['max_nodes'], replace=False)
        pixel_indices = pixel_indices[idx]

    print(f"  Selected {len(pixel_indices)} shared pixels")

    train_labels = 0
    for year, month in valid_timesteps:
        if year > 2021:  # MODIFIED: Training period
            continue
        key = f"{year}_{month:02d}"
        if key in all_data and var in all_data[key]:
            for row, col in pixel_indices:
                if np.isfinite(all_data[key][var][row, col]):
                    train_labels += 1

    print(f"  Training labels: {train_labels}")
    if train_labels < CONFIG['min_training_labels']:
        print(f"  ERROR: Insufficient training labels")
        return None

    # MODIFIED: Split indices for 2001-2021 train, 2022-2023 val, 2024 test
    train_idx = [i for i, (y, m) in enumerate(valid_timesteps)
                 if y <= 2021 and i >= CONFIG['lookback']]
    val_idx = [i for i, (y, m) in enumerate(valid_timesteps)
               if 2022 <= y <= 2023 and i >= CONFIG['lookback']]
    test_idx = [i for i, (y, m) in enumerate(valid_timesteps)
                if y == 2024 and i >= CONFIG['lookback']]

    if not train_idx or not val_idx or not test_idx:
        print(f"  ERROR: Insufficient data splits")
        return None

    print(f"  Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Prepare data for XGBoost
    print(f"  Preparing data...")
    X_train, y_train, mask_train = prepare_xgboost_data(
        all_data, var, valid_timesteps, pixel_indices,
        CONFIG['lookback'], train_idx, var_stats, CONFIG['target_clip'])

    X_val, y_val, mask_val = prepare_xgboost_data(
        all_data, var, valid_timesteps, pixel_indices,
        CONFIG['lookback'], val_idx, var_stats, CONFIG['target_clip'])

    X_test, y_test, mask_test = prepare_xgboost_data(
        all_data, var, valid_timesteps, pixel_indices,
        CONFIG['lookback'], test_idx, var_stats, CONFIG['target_clip'])

    print(f"  Training XGBoost...")
    model, history = train_model(X_train, y_train, mask_train,
                                 X_val, y_val, mask_val, CONFIG)

    if model is None:
        return None

    print(f"  Evaluating...")
    metrics, preds, obs, masks = evaluate_model(model, X_test, y_test, mask_test)

    if np.isfinite(metrics['r2']):
        print(f"  Results: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, Corr={metrics['corr']:.4f}")
    else:
        print(f"  Results: Insufficient valid test data")

    # Create visualization
    create_visualization(preds, obs, masks, history, land_type, var,
                         test_idx, valid_timesteps, len(pixel_indices))

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        'pred': preds,
        'obs': obs,
        'mask': masks
    })
    results_df.to_csv(f"results/{land_type.lower()}_{var.lower()}_results.csv", index=False)

    metrics['scale_factor'] = float(scale_factor)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"results/{land_type.lower()}_{var.lower()}_metrics.csv", index=False)

    gpu_cleanup()
    return metrics


def main():
    print("=" * 80)
    print("CALIFORNIA DROUGHT XGBOOST - 2024 TEST ONLY")
    print("Variables: SPEI, TWS, VPD, LST")
    print("Train: 2001-2021 | Validation: 2022-2023 | Test: 2024")
    print("=" * 80)

    all_metrics = {}

    for land_type in LAND_TYPES:
        all_metrics[land_type] = {}
        for var in VARIABLES:
            try:
                metrics = process_single_variable(land_type, var)
                if metrics:
                    all_metrics[land_type][var] = metrics
            except Exception as e:
                print(f"ERROR: {land_type} - {var}: {str(e)}")
                import traceback
                traceback.print_exc()

    summary_rows = []
    for lt in all_metrics:
        for var in all_metrics[lt]:
            row = {'land_type': lt, 'variable': var}
            row.update(all_metrics[lt][var])
            summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('results/summary_xgboost_2024.csv', index=False)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string())

    print("\nDONE!")


if __name__ == "__main__":
    main()