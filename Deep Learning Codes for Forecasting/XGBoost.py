#!/usr/bin/env python3
"""
California Drought XGBoost Forecaster
Complete implementation with robust file handling, proper t+1 forecasting,
independent evaluation masks, and comprehensive metrics using XGBoost instead of ST-GNN.
"""

import os
import gc
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import rasterio
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # XGBoost Model Parameters
    'max_depth': 8,
    'learning_rate': 0.1,
    'n_estimators': 750,  # Increased from 500 to match the higher epoch count
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 30,  # Increased from 20 to match higher patience
    'eval_metric': 'rmse',

    # Training (adapted from GNN config)
    'lookback': 12,  # From GNN config
    'patience': 15,  # From GNN config

    # XGBoost specific training parameters
    'num_boost_rounds': 150,  # Maps to num_epochs from GNN
    'validation_fraction': 0.2,  # For internal validation split if needed

    # Spatial features (adapted from graph config)
    'k_neighbors': 3,  # From GNN config

    # Data (from GNN config)
    'downsample_factor': 4,
    'max_nodes': 1000,
    'min_valid_frac': 0.4,  # From GNN config
    'target_clip': 4.0,  # From GNN config

    # Seed
    'seed': 42,

    # Coverage thresholds (from GNN config)
    "min_ewbi_drivers": 2,
    "min_pgsi_vars": 3,

    # Optional flags (carried over from GNN)
    'use_ewbi': False,
    'use_drought_proxy': True,
    'drought_vars': ['VPD', 'LST'],
}

# Set seeds for reproducibility
np.random.seed(CONFIG['seed'])
rng = np.random.RandomState(CONFIG['seed'])

print(f"Using XGBoost with random_state: {CONFIG['seed']}")

# ============================================================================
# DATA PATHS AND PATTERNS (Same as original)
# ============================================================================

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
VARIABLES = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']


def get_train_base_path(land_type: str) -> str:
    """Get training data base path for a land type."""
    base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
    land_map = {
        'GRASSLANDS': 'CALIFORNIA GRASSLANDS',
        'NATURAL_FORESTS': 'CALIFORNIA NATURAL FORESTS',
        'SAVANNAS': 'CALIFORNIA SAVANNAS',
        'SHRUBLANDS': 'CALIFORNIA SHRUBLANDS'
    }
    return os.path.join(base, land_map[land_type])


def get_test_base_path(land_type: str) -> str:
    """Get test data base path for a land type."""
    base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"
    land_map = {
        'GRASSLANDS': 'Grasslands',
        'NATURAL_FORESTS': 'NaturalForests',
        'SAVANNAS': 'Savannas',
        'SHRUBLANDS': 'Shrublands'
    }
    return os.path.join(base, land_map[land_type])


# ============================================================================
# GPU MEMORY MANAGEMENT (Adapted for XGBoost)
# ============================================================================

def cleanup():
    """Clean up memory."""
    gc.collect()


# ============================================================================
# FILE FINDING WITH ROBUST PATTERNS (Same as original)
# ============================================================================

def find_data_file(base_path: str, variable: str, year: int, month: int,
                   land_type: str, is_test: bool = False) -> Optional[str]:
    """
    Find data file with robust pattern matching.
    Handles Savanna/Savannas inconsistency and various naming patterns.
    """
    var_path = os.path.join(base_path, variable)

    if not os.path.exists(var_path):
        print(f"Warning: Variable subfolder does not exist: {var_path}")
        return None

    # Map land types for file patterns
    if is_test:
        # Test data patterns
        land_tag = {
            'GRASSLANDS': 'Grasslands',
            'NATURAL_FORESTS': 'NaturalForests',
            'SAVANNAS': 'Savannas',
            'SHRUBLANDS': 'Shrublands'
        }[land_type]

        if variable == 'GOSIF':
            pattern = f"GOSIF_{year}.M{month:02d}_{land_tag}.tif"
        elif variable == 'GNDVI':
            pattern = f"GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'GPP':
            pattern = f"GPP_MOD17A2H_CA_5km_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'LST':
            pattern = f"MYD21C1_CA_LST_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'NDVI':
            pattern = f"NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'NIRv':
            pattern = f"NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'SPEI':
            pattern = f"SPEI1_CSIC_CA_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'TWS':
            pattern = f"TWS_GRACE_CA_{year}_{month:02d}_{land_tag}.tif"
        elif variable == 'VPD':
            pattern = f"VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_{land_tag}.tif"

    else:
        # Training data patterns - handle Savanna/Savannas inconsistency
        land_tags = {
            'GRASSLANDS': ['Grasslands'],
            'NATURAL_FORESTS': ['NaturalForests'],
            'SAVANNAS': ['Savannas', 'Savanna'],  # Try both
            'SHRUBLANDS': ['Shrublands']
        }[land_type]

        patterns = []
        for land_tag in land_tags:
            if variable == 'GOSIF':
                patterns.append(f"GOSIF_{land_tag}_CALIFORNIA_GOSIF_{year}{month:02d}.tif")
            elif variable == 'GNDVI':
                patterns.append(f"GNDVI_{land_tag}_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif")
            elif variable == 'GPP':
                patterns.append(f"{land_tag}_GPP_{year}_{month:02d}.tif")
            elif variable == 'LST':
                patterns.append(f"LST_{land_tag}_MYD21C1_CA_LST_{year}_{month:02d}.tif")
            elif variable == 'NDVI':
                patterns.append(f"NDVI_{land_tag}_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif")
            elif variable == 'NIRv':
                patterns.append(f"NIRv_{land_tag}_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif")
            elif variable == 'SPEI':
                patterns.append(f"SPEI_{land_tag}_SPEI1_CSIC_CA_{year}_{month:02d}.tif")
            elif variable == 'TWS':
                patterns.append(f"TWS_{land_tag}_TWS_GRACE_CA_{year}_{month:02d}.tif")
            elif variable == 'VPD':
                patterns.append(f"VPD_{land_tag}_VPDmax_GRIDMET_CA_5km_{year}_{month:02d}.tif")

        # Try each pattern
        for pattern in patterns:
            file_path = os.path.join(var_path, pattern)
            if os.path.exists(file_path):
                return file_path

        # If no exact match, debug
        print(f"Debug: No match for {variable} {year}-{month:02d} {land_type}")
        print(f"  Tried patterns: {patterns}")
        files = glob.glob(os.path.join(var_path, "*.tif"))[:3]
        if files:
            print(f"  Sample files in folder: {[os.path.basename(f) for f in files]}")
        return None

    file_path = os.path.join(var_path, pattern)
    if os.path.exists(file_path):
        return file_path

    print(f"Debug: No match for {variable} {year}-{month:02d} {land_type}")
    print(f"  Search pattern: {pattern}")
    files = glob.glob(os.path.join(var_path, "*.tif"))[:3]
    if files:
        print(f"  Sample files in folder: {[os.path.basename(f) for f in files]}")
    return None


# ============================================================================
# RASTER I/O AND PROCESSING (Same as original)
# ============================================================================

def read_geotiff(filepath: str, downsample: int = 1) -> Tuple[np.ndarray, dict]:
    """Read GeoTIFF and return data + metadata."""
    with rasterio.open(filepath) as src:
        data = src.read(1).astype(np.float32)

        # Replace nodata with NaN
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        # Scrub infinities
        data[np.isinf(data)] = np.nan

        # Downsample if requested
        if downsample > 1:
            data = data[::downsample, ::downsample]

        meta = {
            'shape': data.shape,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds
        }

    return data, meta


def regrid_to_reference(data: np.ndarray, ref_shape: Tuple[int, int]) -> np.ndarray:
    """Regrid data to reference shape using bilinear interpolation."""
    if data.shape == ref_shape:
        return data

    zoom_factors = (ref_shape[0] / data.shape[0], ref_shape[1] / data.shape[1])
    return ndimage.zoom(data, zoom_factors, order=1)


# ============================================================================
# IMPUTATION FUNCTIONS (Same as original)
# ============================================================================

def impute_all_vars_monthwise(all_data, valid_timesteps, reference_shape,
                              train_years=range(2001, 2019),
                              all_years=range(2001, 2025),
                              smooth_factor=0.5):
    """
    Split-safe wrapper: fit imputation on train_years, apply to all_years.
    """
    vars_to_fill = ['SPEI', 'TWS', 'VPD', 'LST', 'GOSIF', 'GPP', 'NDVI', 'NIRv', 'GNDVI']
    for v in vars_to_fill:
        all_data = impute_monthwise_trend_split(
            all_data=all_data,
            var=v,
            valid_timesteps=valid_timesteps,
            reference_shape=reference_shape,
            fit_years=list(train_years),
            apply_years=list(all_years),
            smooth_factor=smooth_factor
        )
    return all_data


from scipy.interpolate import UnivariateSpline


def impute_monthwise_trend_split(all_data, var, valid_timesteps, reference_shape,
                                 fit_years, apply_years, smooth_factor=0.5):
    """
    Fit per-month smoothing (per pixel) on fit_years ONLY, then predict/fill for all apply_years.
    No information from val/test years enters the fit.
    """
    months = range(1, 13)

    # helpers
    def get_arr(y, m):
        key = f"{y}_{m:02d}"
        if key in all_data and var in all_data[key]:
            arr = all_data[key][var]
            if arr.shape != reference_shape:
                arr = regrid_to_reference(arr, reference_shape)
            arr = arr.copy()
            arr[~np.isfinite(arr)] = np.nan
            return arr
        return np.full(reference_shape, np.nan, dtype=np.float32)

    for m in months:
        # stacks for fit and for application
        fit_frames = [get_arr(y, m) for y in fit_years]  # [Tf,H,W]
        apply_frames = [get_arr(y, m) for y in apply_years]  # [Ta,H,W]
        fit_stack = np.stack(fit_frames, axis=0)
        out_stack = np.stack(apply_frames, axis=0)  # we will fill into this

        Tf, H, W = fit_stack.shape
        Ta = out_stack.shape[0]

        # Step 1: adjacent-year mean fill (FIT set only)
        for t in range(1, Tf - 1):
            miss = ~np.isfinite(fit_stack[t])
            can = np.isfinite(fit_stack[t - 1]) & np.isfinite(fit_stack[t + 1]) & miss
            if np.any(can):
                fit_stack[t][can] = 0.5 * (fit_stack[t - 1][can] + fit_stack[t + 1][can])

        # Step 2: per-pixel spline on FIT, applied to ALL years
        x_fit = np.arange(Tf, dtype=float)
        x_app = np.linspace(0, Tf - 1, Ta)  # map apply_years onto fit index space (monotone)
        for r in range(H):
            for c in range(W):
                y_fit = fit_stack[:, r, c]
                ok = np.isfinite(y_fit)
                if ok.sum() >= 4:
                    try:
                        s_val = smooth_factor * ok.sum()
                        spl = UnivariateSpline(x_fit[ok], y_fit[ok], s=s_val, k=3)
                        pred = spl(x_app)
                        # fill missing in out_stack with spline predictions
                        ts = out_stack[:, r, c]
                        nan_mask = ~np.isfinite(ts)
                        ts[nan_mask] = pred[nan_mask]
                        out_stack[:, r, c] = ts
                    except Exception:
                        pass  # fall through to step 3

        # Step 3: month mean over FIT if anything still missing in OUT
        month_mean_fit = np.nanmean(fit_stack, axis=0)  # [H,W]
        remaining = ~np.isfinite(out_stack)
        if np.any(remaining):
            out_stack[remaining] = np.broadcast_to(month_mean_fit, out_stack.shape)[remaining]

        # Write back
        for ti, y in enumerate(apply_years):
            key = f"{y}_{m:02d}"
            if key not in all_data:
                all_data[key] = {}
            all_data[key][var] = out_stack[ti].astype(np.float32)

    return all_data


def detrend_per_month(all_data, valid_timesteps, variables, years=(2001, 2018)):
    # fit per-month linear trend over training years and subtract from all years
    months = range(1, 13)
    for var in variables:
        for m in months:
            ys, mus = [], []
            for (y, mm) in valid_timesteps:
                if mm != m: continue
                key = f"{y}_{mm:02d}"
                if key in all_data and var in all_data[key]:
                    mu = np.nanmean(all_data[key][var])
                    if np.isfinite(mu):
                        ys.append(y);
                        mus.append(mu)
            ys = np.array(ys);
            mus = np.array(mus)
            mask = (ys >= years[0]) & (ys <= years[1])
            if mask.sum() >= 5:
                A = np.vstack([ys[mask], np.ones(mask.sum())]).T
                a, b = np.linalg.lstsq(A, mus[mask], rcond=None)[0]
                # subtract (a*y + b - mean over train) so training mean remains
                for (y, mm) in valid_timesteps:
                    if mm != m: continue
                    k = f"{y}_{mm:02d}"
                    if k in all_data and var in all_data[k]:
                        adj = a * y + b
                        all_data[k][var] = all_data[k][var] - adj
    return all_data


def compute_training_stats(all_data: Dict, valid_timesteps: List,
                           variables: List[str]) -> Dict:
    """Compute per-pixel, per-month statistics from training data only (<=2018)."""
    print("Computing training statistics (2001-2018 only)...")
    train_stats = {}

    for var in variables:
        train_stats[var] = {}

        for month in range(1, 13):
            month_data = []

            for year, m in valid_timesteps:
                if year <= 2018 and m == month:  # Training only
                    key = f"{year}_{m:02d}"
                    if key in all_data and var in all_data[key]:
                        data = all_data[key][var]
                        valid_pixels = data[np.isfinite(data)]
                        if len(valid_pixels) > 0:
                            month_data.extend(valid_pixels.tolist())

            if month_data:
                train_stats[var][month] = {
                    'mean': np.mean(month_data),
                    'std': np.std(month_data) + 1e-8
                }

    return train_stats


def robust_z_score_normalize(data: np.ndarray, var: str, month: int,
                             train_stats: Dict, clip: float = 4.0) -> np.ndarray:
    """Apply robust z-score normalization using training statistics."""
    if var in train_stats and month in train_stats[var]:
        stats = train_stats[var][month]
        z = (data - stats['mean']) / stats['std']
    else:
        # Fallback to robust statistics
        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            median = np.median(valid)
            mad = np.median(np.abs(valid - median))
            std = mad * 1.4826 + 1e-8
            z = (data - median) / std
        else:
            z = np.zeros_like(data)

    # Replace non-finite with 0
    z[~np.isfinite(z)] = 0.0

    # Clip
    z = np.clip(z, -clip, clip)

    return z


# ============================================================================
# SEASONAL CLIMATOLOGY BASELINE (Same as original)
# ============================================================================

def compute_seasonal_climatology(all_data: Dict, valid_timesteps: List,
                                 train_stats: Dict, clip: float = 4.0) -> Dict:
    """Compute monthly climatology from training data only (<=2018)."""
    print("Computing seasonal climatology baseline (2001-2018)...")

    climatology = {'ewbi': {}, 'pgsi': {}}

    for month in range(1, 13):
        ewbi_values = []
        pgsi_values = []

        for year, m in valid_timesteps:
            if year <= 2018 and m == month:  # Training only
                key = f"{year}_{m:02d}"
                if key not in all_data:
                    continue

                # Get standardized data
                month_data = all_data[key]

                # EWBI components
                ewbi_components = []
                for var, sign in [('SPEI', 1), ('TWS', 1), ('VPD', -1), ('LST', -1)]:
                    if var in month_data:
                        z_data = robust_z_score_normalize(
                            month_data[var], var, month, train_stats, clip
                        )
                        valid_z = z_data[np.isfinite(z_data)]
                        if len(valid_z) > 0:
                            ewbi_components.append(sign * np.mean(valid_z))

                if len(ewbi_components) >= 3:
                    ewbi_values.append(np.mean(ewbi_components))

                # PGSI components
                pgsi_components = []
                for var in ['GOSIF', 'GPP', 'NIRv', 'NDVI', 'GNDVI']:
                    if var in month_data:
                        z_data = robust_z_score_normalize(
                            month_data[var], var, month, train_stats, clip
                        )
                        valid_z = z_data[np.isfinite(z_data)]
                        if len(valid_z) > 0:
                            pgsi_components.append(np.mean(valid_z))

                if len(pgsi_components) >= 3:
                    pgsi_values.append(np.mean(pgsi_components))

        if ewbi_values:
            climatology['ewbi'][month] = np.mean(ewbi_values)
        if pgsi_values:
            climatology['pgsi'][month] = np.mean(pgsi_values)

    return climatology


# ============================================================================
# PIXEL SELECTION (Same as original)
# ============================================================================

def select_stable_pixels(all_data: Dict, valid_timesteps: List,
                         max_nodes: int = 1500, min_valid_frac: float = 0.7) -> np.ndarray:
    """Select stable pixels using train+val only (<=2021) with ≥3-of-4 (EWBI) and ≥3-of-5 (veg) per-month rules."""
    print("Selecting stable pixels (using train+val only, <=2021)...")

    # 1) Find the reference shape from existing arrays (<=2021)
    shapes = []
    for year, month in valid_timesteps:
        if year > 2021:
            continue
        key = f"{year}_{month:02d}"
        if key in all_data and all_data[key]:
            any_arr = next(iter(all_data[key].values()))
            shapes.append(any_arr.shape)
    from collections import Counter
    if not shapes:
        raise RuntimeError("No arrays found for train/val to infer reference shape.")
    reference_shape = Counter(shapes).most_common(1)[0][0]
    print(f"  Using reference shape: {reference_shape}")

    def get_var_arr(month_data: Dict, var: str) -> np.ndarray:
        """Return array for var regridded to reference_shape, or NaN array if missing."""
        if var in month_data:
            arr = month_data[var]
            if arr.shape != reference_shape:
                arr = regrid_to_reference(arr, reference_shape)
            return arr
        return np.full(reference_shape, np.nan, dtype=np.float32)

    # 2) Coverage counters (per-month success, not per-variable sum)
    ewbi_coverage = np.zeros(reference_shape, dtype=np.int32)  # months with ≥3 of 4 drivers
    veg_coverage = np.zeros(reference_shape, dtype=np.int32)  # months with ≥3 of 5 veg proxies

    total_months = 0
    for year, month in valid_timesteps:
        if year > 2021:
            continue
        key = f"{year}_{month:02d}"
        if key not in all_data:
            continue
        md = all_data[key]

        # EWBI drivers
        spei = get_var_arr(md, 'SPEI')
        tws = get_var_arr(md, 'TWS')
        vpd = get_var_arr(md, 'VPD')
        lst = get_var_arr(md, 'LST')
        drv_stack = np.stack([spei, tws, vpd, lst], axis=0)
        min_e = CONFIG.get("min_ewbi_drivers", 3)
        ewbi_coverage += (np.isfinite(drv_stack).sum(axis=0) >= min_e).astype(np.int32)

        # Vegetation proxies
        gosif = get_var_arr(md, 'GOSIF')
        gpp = get_var_arr(md, 'GPP')
        nirv = get_var_arr(md, 'NIRv')
        ndvi = get_var_arr(md, 'NDVI')
        gndvi = get_var_arr(md, 'GNDVI')
        veg_stack = np.stack([gosif, gpp, nirv, ndvi, gndvi], axis=0)
        min_v = CONFIG.get("min_pgsi_vars", 3)
        veg_coverage += (np.isfinite(veg_stack).sum(axis=0) >= min_v).astype(np.int32)

        total_months += 1

    print(f"  Months considered for selection: {total_months}")
    if total_months == 0:
        raise RuntimeError("No train/val months were available for pixel selection.")

    # 3) Thresholds are fractions of months
    ewbi_threshold = int(np.ceil(min_valid_frac * total_months))
    veg_threshold = int(np.ceil(min_valid_frac * total_months))
    valid_mask = (ewbi_coverage >= ewbi_threshold) & (veg_coverage >= veg_threshold)

    # 4) Relax if too strict
    if valid_mask.sum() < 300:
        print("  Fallback to looser criteria (0.3x thresholds)...")
        ewbi_threshold = max(1, int(np.floor(0.3 * ewbi_threshold)))
        veg_threshold = max(1, int(np.floor(0.3 * veg_threshold)))
        valid_mask = (ewbi_coverage >= ewbi_threshold) & (veg_coverage >= veg_threshold)

    # 5) If still bad, rank primarily by EWBI coverage (so EWBI labels exist), then veg coverage
    if valid_mask.sum() == 0:
        print("  No pixels met thresholds; ranking by EWBI coverage, then vegetation coverage.")
        ewbi_flat = ewbi_coverage.ravel()
        veg_flat = veg_coverage.ravel()
        # Rank: ewbi first, then veg
        order = np.lexsort((-veg_flat, -ewbi_flat))  # negative for descending
        r, c = np.unravel_index(order, reference_shape)
        valid_indices = np.column_stack([r, c])
        # Keep only pixels that have at least 1 month with EWBI coverage
        valid_indices = valid_indices[ewbi_flat[order] > 0]
        if len(valid_indices) == 0:
            # As a last resort, just take the top by combined coverage
            score = ewbi_flat + veg_flat
            order = np.argsort(score)[::-1]
            r, c = np.unravel_index(order, reference_shape)
            valid_indices = np.column_stack([r, c])
    else:
        valid_indices = np.column_stack(np.where(valid_mask))

    # 6) Subsample if needed
    if len(valid_indices) > max_nodes:
        global rng
        idx = rng.choice(len(valid_indices), max_nodes, replace=False)
        valid_indices = valid_indices[idx]

    print(f"  Selected {len(valid_indices)} pixels")
    return valid_indices


# ============================================================================
# FEATURE ENGINEERING (Same as original)
# ============================================================================

def create_features_and_targets(all_data: Dict, valid_timesteps: List,
                                pixel_indices: np.ndarray, train_stats: Dict,
                                lookback: int = 6, clip: float = 4.0) -> Dict:
    """Create features and targets with proper t+1 forecasting + per-var resolution confidence."""
    print("Creating features and targets...")

    n_pixels = len(pixel_indices)
    n_times = len(valid_timesteps)

    # --- per-variable features: value z + missing flag + resolution confidence => 3*n_vars ---
    n_vars = 9
    # other features you already had: +5 temporal, +2 positional, +4 deltas = +11
    n_features = n_vars * 3 + 11

    all_features = np.zeros((n_times, n_pixels, n_features), dtype=np.float32)
    ewbi_targets = np.full((n_times, n_pixels), np.nan, dtype=np.float32)
    pgsi_targets = np.full((n_times, n_pixels), np.nan, dtype=np.float32)
    ewbi_masks = np.zeros((n_times, n_pixels), dtype=np.float32)
    pgsi_masks = np.zeros((n_times, n_pixels), dtype=np.float32)

    # reference grid (H,W) to compute confidence
    ref_shape = train_stats.get('_ref_shape', None)

    for t_idx, (year, month) in enumerate(valid_timesteps):
        key = f"{year}_{month:02d}"
        if key not in all_data:
            continue

        month_data = all_data[key]

        for p_idx, (row, col) in enumerate(pixel_indices):
            feat_idx = 0
            raw_values = {}

            # ---------- per-variable features ----------
            for var in VARIABLES:
                # 1) z-value
                if var in month_data:
                    raw_val = month_data[var][row, col]
                    raw_values[var] = raw_val
                    z_val = robust_z_score_normalize(
                        np.array([raw_val]), var, month, train_stats, clip
                    )[0]
                    all_features[t_idx, p_idx, feat_idx] = z_val
                    missing_flag = 0 if np.isfinite(raw_val) else 1
                else:
                    all_features[t_idx, p_idx, feat_idx] = 0.0
                    missing_flag = 1

                # 2) missing flag
                all_features[t_idx, p_idx, feat_idx + n_vars] = missing_flag

                # 3) resolution confidence
                conf = 0.0
                if ref_shape is not None:
                    src_shape = month_data.get(var + '_SRC_SHAPE', None)
                    if src_shape is not None and src_shape[0] > 0 and src_shape[1] > 0:
                        # sqrt(area ratio) gives a nice 0..1 scale
                        area_ratio = (src_shape[0] * src_shape[1]) / (ref_shape[0] * ref_shape[1])
                        conf = float(np.clip(np.sqrt(area_ratio), 0.0, 1.0))
                all_features[t_idx, p_idx, feat_idx + 2 * n_vars] = conf

                # advance to next variable slot
                feat_idx += 1

            # after loop, feat_idx == n_vars
            # next two blocks (missing-flag & confidence) were written with offsets +n_vars and +2*n_vars

            # ---------- temporal features (5) ----------
            feat_base = n_vars * 3
            # TWS availability
            all_features[t_idx, p_idx, feat_base] = 1 if 'TWS' in month_data else 0
            # month sin/cos
            all_features[t_idx, p_idx, feat_base + 1] = np.sin(2 * np.pi * month / 12)
            all_features[t_idx, p_idx, feat_base + 2] = np.cos(2 * np.pi * month / 12)
            # normalized year/month
            all_features[t_idx, p_idx, feat_base + 3] = (year - 2001) / 23.0
            all_features[t_idx, p_idx, feat_base + 4] = month / 12.0

            # ---------- positional features (2) ----------
            pos_base = feat_base + 5
            all_features[t_idx, p_idx, pos_base] = row / max(1, pixel_indices[:, 0].max())
            all_features[t_idx, p_idx, pos_base + 1] = col / max(1, pixel_indices[:, 1].max())

            # ---------- first differences (4) ----------
            delta_base = pos_base + 2
            if t_idx > 0:
                prev_y, prev_m = valid_timesteps[t_idx - 1]
                prev_key = f"{prev_y}_{prev_m:02d}"
                if prev_key in all_data:
                    for d_i, vname in enumerate(['SPEI', 'VPD', 'LST', 'TWS']):
                        if vname in month_data and vname in all_data[prev_key]:
                            curr = month_data[vname][row, col]
                            prev = all_data[prev_key][vname][row, col]
                            if np.isfinite(curr) and np.isfinite(prev):
                                curr_z = robust_z_score_normalize(np.array([curr]), vname, month, train_stats, clip)[0]
                                prev_z = robust_z_score_normalize(np.array([prev]), vname, prev_m, train_stats, clip)[0]
                                all_features[t_idx, p_idx, delta_base + d_i] = curr_z - prev_z

            # ---------- targets & masks ----------
            min_e = CONFIG.get("min_ewbi_drivers", 3)
            min_v = CONFIG.get("min_pgsi_vars", 3)

            ewbi_components = []
            for var, sign in [('SPEI', 1), ('TWS', 1), ('VPD', -1), ('LST', -1)]:
                if var in raw_values and np.isfinite(raw_values[var]):
                    z_val = robust_z_score_normalize(
                        np.array([raw_values[var]]), var, month, train_stats, clip
                    )[0]
                    ewbi_components.append(sign * z_val)
            if len(ewbi_components) >= min_e:
                ewbi_targets[t_idx, p_idx] = np.clip(np.mean(ewbi_components), -clip, clip)
                ewbi_masks[t_idx, p_idx] = 1.0

            pgsi_components = []
            for var in ['GOSIF', 'GPP', 'NIRv', 'NDVI', 'GNDVI']:
                if var in raw_values and np.isfinite(raw_values[var]):
                    z_val = robust_z_score_normalize(
                        np.array([raw_values[var]]), var, month, train_stats, clip
                    )[0]
                    pgsi_components.append(z_val)
            if len(pgsi_components) >= min_v:
                pgsi_targets[t_idx, p_idx] = np.clip(np.mean(pgsi_components), -clip, clip)
                pgsi_masks[t_idx, p_idx] = 1.0

    print(f"  Total valid EWBI labels: {np.sum(ewbi_masks):.0f}")
    print(f"  Total valid PGSI labels: {np.sum(pgsi_masks):.0f}")

    valid_indices = list(range(lookback, len(valid_timesteps)))

    return {
        'features': all_features,
        'ewbi_targets': ewbi_targets,
        'pgsi_targets': pgsi_targets,
        'ewbi_masks': ewbi_masks,
        'pgsi_masks': pgsi_masks,
        'valid_indices': valid_indices,
        'timesteps': valid_timesteps
    }


# ============================================================================
# SPATIAL FEATURES (Adapted from graph construction)
# ============================================================================

def build_spatial_features(pixel_indices: np.ndarray, k: int = 8) -> np.ndarray:
    """Build spatial features for each pixel based on KNN distances."""
    n_nodes = len(pixel_indices)
    if n_nodes == 0:
        return np.zeros((0, k))
    if n_nodes == 1:
        return np.zeros((1, k))

    k = min(k, n_nodes - 1)
    print(f"Building spatial features (k={k})...")

    nbrs = NearestNeighbors(n_neighbors=k + 1)
    nbrs.fit(pixel_indices)
    distances, indices = nbrs.kneighbors(pixel_indices)

    # Create spatial features: mean distance, std distance, etc.
    spatial_features = np.zeros((n_nodes, 4), dtype=np.float32)

    for i in range(n_nodes):
        neighbor_dists = distances[i, 1:]  # exclude self (distance 0)
        if len(neighbor_dists) > 0:
            spatial_features[i, 0] = np.mean(neighbor_dists)  # mean neighbor distance
            spatial_features[i, 1] = np.std(neighbor_dists)  # std neighbor distance
            spatial_features[i, 2] = np.min(neighbor_dists)  # closest neighbor distance
            spatial_features[i, 3] = np.max(neighbor_dists)  # furthest neighbor distance

    return spatial_features


def compute_training_thresholds(all_data: Dict, valid_timesteps: List,
                                train_stats: Dict, pixel_indices: np.ndarray,
                                clip: float = 4.0) -> Tuple[float, float, float]:
    """
    Compute 5th percentile thresholds from TRAINING years (<=2018),
    using UNCLIPPED monthly means for the percentile calculation.
    """
    print("Computing training thresholds (2001-2018)...")

    ewbi_values = []  # UNCLIPPED per-month means over selected pixels
    pgsi_values = []  # UNCLIPPED per-month means over selected pixels
    joint_values = []  # min(EWBI, PGSI) using UNCLIPPED monthly means

    min_e = CONFIG.get("min_ewbi_drivers", 3)  # match your label rule
    min_v = CONFIG.get("min_pgsi_vars", 3)

    for year, month in valid_timesteps:
        if year > 2018:  # training only
            continue

        key = f"{year}_{month:02d}"
        if key not in all_data:
            continue
        md = all_data[key]

        # accumulate UNCLIPPED per-pixel EWBI/PGSI for this month
        month_ewbi_vals = []
        month_pgsi_vals = []

        for row, col in pixel_indices:
            # EWBI components (signed)
            ewbi_components = []
            for var, sign in [('SPEI', 1), ('TWS', 1), ('VPD', -1), ('LST', -1)]:
                if var in md:
                    raw = md[var][row, col]
                    if np.isfinite(raw):
                        z = robust_z_score_normalize(
                            np.array([raw]), var, month, train_stats, clip
                        )[0]
                        ewbi_components.append(sign * z)

            if len(ewbi_components) >= min_e:
                # UNCLIPPED mean for threshold stats
                month_ewbi_vals.append(np.mean(ewbi_components))

            # PGSI components (unsigned mean)
            pgsi_components = []
            for var in ['GOSIF', 'GPP', 'NIRv', 'NDVI', 'GNDVI']:
                if var in md:
                    raw = md[var][row, col]
                    if np.isfinite(raw):
                        z = robust_z_score_normalize(
                            np.array([raw]), var, month, train_stats, clip
                        )[0]
                        pgsi_components.append(z)

            if len(pgsi_components) >= min_v:
                month_pgsi_vals.append(np.mean(pgsi_components))

        # finalize this month: take spatial mean over selected pixels (if any)
        if len(month_ewbi_vals) > 0:
            ewbi_values.append(float(np.mean(month_ewbi_vals)))
        if len(month_pgsi_vals) > 0:
            pgsi_values.append(float(np.mean(month_pgsi_vals)))
        if len(month_ewbi_vals) > 0 and len(month_pgsi_vals) > 0:
            joint_values.append(float(min(np.mean(month_ewbi_vals), np.mean(month_pgsi_vals))))

    # Percentiles from UNCLIPPED lists
    ewbi_5th = np.percentile(ewbi_values, 5) if len(ewbi_values) else -2.0
    pgsi_5th = np.percentile(pgsi_values, 5) if len(pgsi_values) else -2.0
    joint_5th = np.percentile(joint_values, 5) if len(joint_values) else -2.0

    print(f"  Thresholds: EWBI={ewbi_5th:.3f}, PGSI={pgsi_5th:.3f}, Joint={joint_5th:.3f}")
    return ewbi_5th, pgsi_5th, joint_5th


# ============================================================================
# XGBOOST DATA PREPARATION
# ============================================================================

def prepare_xgboost_data(features_dict: Dict, lookback: int = 24) -> Dict:
    """
    Convert the 3D features array to 2D format suitable for XGBoost.
    For each valid timestep, create a feature vector that includes lookback window.
    """
    print(f"Preparing XGBoost data with lookback={lookback}...")

    all_features = features_dict['features']  # [T, N, F]
    ewbi_targets = features_dict['ewbi_targets']  # [T, N]
    pgsi_targets = features_dict['pgsi_targets']  # [T, N]
    ewbi_masks = features_dict['ewbi_masks']  # [T, N]
    pgsi_masks = features_dict['pgsi_masks']  # [T, N]
    valid_indices = features_dict['valid_indices']

    T, N, F = all_features.shape

    # For XGBoost, we'll flatten the lookback window into the feature vector
    # Each sample will be: [pixel_features_t-lookback, ..., pixel_features_t-1, spatial_features]
    xgb_features = []
    ewbi_y = []
    pgsi_y = []
    ewbi_m = []
    pgsi_m = []
    sample_indices = []  # track which (t, pixel) each sample corresponds to

    for t_idx in valid_indices:
        for p_idx in range(N):
            # Create lookback window: [lookback, F]
            lookback_window = all_features[t_idx - lookback:t_idx, p_idx, :]  # [lookback, F]

            # Flatten the lookback window
            feature_vector = lookback_window.flatten()  # [lookback * F]

            xgb_features.append(feature_vector)
            ewbi_y.append(ewbi_targets[t_idx, p_idx])
            pgsi_y.append(pgsi_targets[t_idx, p_idx])
            ewbi_m.append(ewbi_masks[t_idx, p_idx])
            pgsi_m.append(pgsi_masks[t_idx, p_idx])
            sample_indices.append((t_idx, p_idx))

    return {
        'features': np.array(xgb_features, dtype=np.float32),
        'ewbi_targets': np.array(ewbi_y, dtype=np.float32),
        'pgsi_targets': np.array(pgsi_y, dtype=np.float32),
        'ewbi_masks': np.array(ewbi_m, dtype=np.float32),
        'pgsi_masks': np.array(pgsi_m, dtype=np.float32),
        'sample_indices': sample_indices,
        'timesteps': features_dict['timesteps']
    }


# ============================================================================
# XGBOOST MODEL TRAINING
# ============================================================================

class DroughtXGBoostModel:
    """XGBoost-based drought forecasting model."""

    def __init__(self, config):
        self.config = config
        self.ewbi_model = None
        self.pgsi_model = None
        self.ewbi_calibration = (1.0, 0.0)  # (a, b) for a*pred + b
        self.pgsi_calibration = (1.0, 0.0)

    def train(self, X_train, y_ewbi_train, y_pgsi_train,
              mask_ewbi_train, mask_pgsi_train,
              X_val, y_ewbi_val, y_pgsi_val,
              mask_ewbi_val, mask_pgsi_val,
              train_thresholds=None):
        """Train separate XGBoost models for EWBI and PGSI."""
        print("Training XGBoost models...")

        # Get drought upweighting thresholds
        ewbi_threshold = train_thresholds[0] if train_thresholds else -2.0
        pgsi_threshold = train_thresholds[1] if train_thresholds else -2.0

        # Train EWBI model
        print("  Training EWBI model...")
        ewbi_train_mask = mask_ewbi_train > 0
        if ewbi_train_mask.sum() > 0:
            X_ewbi = X_train[ewbi_train_mask]
            y_ewbi = y_ewbi_train[ewbi_train_mask]

            # Create sample weights (upweight drought conditions)
            sample_weights = np.ones(len(y_ewbi))
            drought_mask = y_ewbi <= ewbi_threshold
            sample_weights[drought_mask] *= 4.0  # 4x weight for drought samples

            # Validation data
            ewbi_val_mask = mask_ewbi_val > 0
            if ewbi_val_mask.sum() > 0:
                X_ewbi_val = X_val[ewbi_val_mask]
                y_ewbi_val_masked = y_ewbi_val[ewbi_val_mask]
                eval_set = [(X_ewbi_val, y_ewbi_val_masked)]
            else:
                eval_set = None

            self.ewbi_model = xgb.XGBRegressor(
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                n_estimators=self.config['n_estimators'],
                subsample=self.config['subsample'],
                colsample_bytree=self.config['colsample_bytree'],
                reg_alpha=self.config['reg_alpha'],
                reg_lambda=self.config['reg_lambda'],
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs'],
                eval_metric=self.config['eval_metric']
            )

            # For compatibility with different XGBoost versions
            try:
                # Try newer callback syntax first
                if eval_set is not None:
                    self.ewbi_model.fit(
                        X_ewbi, y_ewbi,
                        sample_weight=sample_weights,
                        eval_set=eval_set,
                        callbacks=[xgb.callback.EarlyStopping(rounds=self.config['early_stopping_rounds'])],
                        verbose=False
                    )
                else:
                    self.ewbi_model.fit(
                        X_ewbi, y_ewbi,
                        sample_weight=sample_weights,
                        verbose=False
                    )
            except TypeError:
                # Fallback for older XGBoost versions
                if eval_set is not None:
                    # Use validation monitoring without early stopping for older versions
                    self.ewbi_model.fit(
                        X_ewbi, y_ewbi,
                        sample_weight=sample_weights,
                        eval_set=eval_set,
                        verbose=False
                    )
                else:
                    self.ewbi_model.fit(
                        X_ewbi, y_ewbi,
                        sample_weight=sample_weights,
                        verbose=False
                    )

        # Train PGSI model
        print("  Training PGSI model...")
        pgsi_train_mask = mask_pgsi_train > 0
        if pgsi_train_mask.sum() > 0:
            X_pgsi = X_train[pgsi_train_mask]
            y_pgsi = y_pgsi_train[pgsi_train_mask]

            # Create sample weights (upweight drought conditions)
            sample_weights = np.ones(len(y_pgsi))
            drought_mask = y_pgsi <= pgsi_threshold
            sample_weights[drought_mask] *= 4.0  # 4x weight for drought samples

            # Validation data
            pgsi_val_mask = mask_pgsi_val > 0
            if pgsi_val_mask.sum() > 0:
                X_pgsi_val = X_val[pgsi_val_mask]
                y_pgsi_val_masked = y_pgsi_val[pgsi_val_mask]
                eval_set = [(X_pgsi_val, y_pgsi_val_masked)]
            else:
                eval_set = None

            self.pgsi_model = xgb.XGBRegressor(
                max_depth=self.config['max_depth'],
                learning_rate=self.config['learning_rate'],
                n_estimators=self.config['n_estimators'],
                subsample=self.config['subsample'],
                colsample_bytree=self.config['colsample_bytree'],
                reg_alpha=self.config['reg_alpha'],
                reg_lambda=self.config['reg_lambda'],
                random_state=self.config['random_state'],
                n_jobs=self.config['n_jobs'],
                eval_metric=self.config['eval_metric']
            )

            # For compatibility with different XGBoost versions
            try:
                # Try newer callback syntax first
                if eval_set is not None:
                    self.pgsi_model.fit(
                        X_pgsi, y_pgsi,
                        sample_weight=sample_weights,
                        eval_set=eval_set,
                        callbacks=[xgb.callback.EarlyStopping(rounds=self.config['early_stopping_rounds'])],
                        verbose=False
                    )
                else:
                    self.pgsi_model.fit(
                        X_pgsi, y_pgsi,
                        sample_weight=sample_weights,
                        verbose=False
                    )
            except TypeError:
                # Fallback for older XGBoost versions
                if eval_set is not None:
                    # Use validation monitoring without early stopping for older versions
                    self.pgsi_model.fit(
                        X_pgsi, y_pgsi,
                        sample_weight=sample_weights,
                        eval_set=eval_set,
                        verbose=False
                    )
                else:
                    self.pgsi_model.fit(
                        X_pgsi, y_pgsi,
                        sample_weight=sample_weights,
                        verbose=False
                    )

    def predict(self, X):
        """Make predictions with both models."""
        ewbi_pred = np.full(len(X), np.nan, dtype=np.float32)
        pgsi_pred = np.full(len(X), np.nan, dtype=np.float32)

        if self.ewbi_model is not None:
            ewbi_pred = self.ewbi_model.predict(X).astype(np.float32)

        if self.pgsi_model is not None:
            pgsi_pred = self.pgsi_model.predict(X).astype(np.float32)

        return ewbi_pred, pgsi_pred

    def calibrate(self, X_val, y_ewbi_val, y_pgsi_val, mask_ewbi_val, mask_pgsi_val):
        """Calibrate predictions using validation data."""
        print("Calibrating models...")

        ewbi_pred, pgsi_pred = self.predict(X_val)

        # EWBI calibration
        ewbi_mask = mask_ewbi_val > 0
        if ewbi_mask.sum() > 8:  # need sufficient samples
            pred_vals = ewbi_pred[ewbi_mask]
            true_vals = y_ewbi_val[ewbi_mask]

            # Ridge regression for calibration
            X_cal = np.vstack([pred_vals, np.ones(len(pred_vals))]).T
            XtX = X_cal.T @ X_cal
            XtX[0, 0] += 1e-2  # L2 regularization
            XtX[1, 1] += 1e-2
            Xty = X_cal.T @ true_vals

            try:
                a, b = np.linalg.solve(XtX, Xty)
                a = float(np.clip(a, -3.0, 3.0))
                b = float(np.clip(b, -3.0, 3.0))
                self.ewbi_calibration = (a, b)
            except:
                self.ewbi_calibration = (1.0, 0.0)

        # PGSI calibration
        pgsi_mask = mask_pgsi_val > 0
        if pgsi_mask.sum() > 8:  # need sufficient samples
            pred_vals = pgsi_pred[pgsi_mask]
            true_vals = y_pgsi_val[pgsi_mask]

            # Ridge regression for calibration
            X_cal = np.vstack([pred_vals, np.ones(len(pred_vals))]).T
            XtX = X_cal.T @ X_cal
            XtX[0, 0] += 1e-2  # L2 regularization
            XtX[1, 1] += 1e-2
            Xty = X_cal.T @ true_vals

            try:
                a, b = np.linalg.solve(XtX, Xty)
                a = float(np.clip(a, -3.0, 3.0))
                b = float(np.clip(b, -3.0, 3.0))
                self.pgsi_calibration = (a, b)
            except:
                self.pgsi_calibration = (1.0, 0.0)

        print(f"  EWBI calibration: a={self.ewbi_calibration[0]:.3f}, b={self.ewbi_calibration[1]:.3f}")
        print(f"  PGSI calibration: a={self.pgsi_calibration[0]:.3f}, b={self.pgsi_calibration[1]:.3f}")

    def predict_calibrated(self, X):
        """Make calibrated predictions."""
        ewbi_pred, pgsi_pred = self.predict(X)

        # Apply calibration
        a_e, b_e = self.ewbi_calibration
        a_p, b_p = self.pgsi_calibration

        ewbi_pred_cal = a_e * ewbi_pred + b_e
        pgsi_pred_cal = a_p * pgsi_pred + b_p

        return ewbi_pred_cal, pgsi_pred_cal


# ============================================================================
# EVALUATION (Adapted for XGBoost)
# ============================================================================

def identify_drought_events(worst_mask: np.ndarray, dates: List[str]) -> List[Dict]:
    """Identify consecutive drought events."""
    events = []
    if worst_mask.sum() == 0:
        return events

    in_event = False
    start_idx = None

    for i, is_drought in enumerate(worst_mask):
        if is_drought and not in_event:
            start_idx = i
            in_event = True
        elif not is_drought and in_event:
            events.append({
                'start': dates[start_idx],
                'end': dates[i - 1],
                'duration': i - start_idx
            })
            in_event = False

    # Handle ongoing event
    if in_event:
        events.append({
            'start': dates[start_idx],
            'end': dates[-1],
            'duration': len(dates) - start_idx
        })

    return events


def evaluate_xgboost_model(model, X_test, y_ewbi_test, y_pgsi_test,
                           mask_ewbi_test, mask_pgsi_test, sample_indices,
                           train_thresholds, timesteps, climatology,
                           pixel_indices):
    """Evaluate XGBoost model with independent masks and comprehensive metrics."""
    print("Evaluating XGBoost model...")

    # Get predictions
    ewbi_pred, pgsi_pred = model.predict_calibrated(X_test)

    # Group predictions by timestep for spatial averaging
    timestep_predictions = {}
    for i, (t_idx, p_idx) in enumerate(sample_indices):
        if t_idx not in timestep_predictions:
            timestep_predictions[t_idx] = {
                'ewbi_pred': [], 'ewbi_obs': [], 'ewbi_mask': [],
                'pgsi_pred': [], 'pgsi_obs': [], 'pgsi_mask': []
            }

        timestep_predictions[t_idx]['ewbi_pred'].append(ewbi_pred[i])
        timestep_predictions[t_idx]['ewbi_obs'].append(y_ewbi_test[i])
        timestep_predictions[t_idx]['ewbi_mask'].append(mask_ewbi_test[i])
        timestep_predictions[t_idx]['pgsi_pred'].append(pgsi_pred[i])
        timestep_predictions[t_idx]['pgsi_obs'].append(y_pgsi_test[i])
        timestep_predictions[t_idx]['pgsi_mask'].append(mask_pgsi_test[i])

    # Compute spatial averages for each timestep
    ewbi_pred_all = []
    ewbi_obs_all = []
    pgsi_pred_all = []
    pgsi_obs_all = []
    dates = []

    for t_idx in sorted(timestep_predictions.keys()):
        if t_idx < len(timesteps):
            year, month = timesteps[t_idx]
            dates.append(f"{year}-{month:02d}")

            tp = timestep_predictions[t_idx]

            # EWBI spatial average (use EWBI mask)
            ewbi_masks = np.array(tp['ewbi_mask'])
            if ewbi_masks.sum() > 0:
                ewbi_pred_vals = np.array(tp['ewbi_pred'])[ewbi_masks > 0]
                ewbi_obs_vals = np.array(tp['ewbi_obs'])[ewbi_masks > 0]
                ewbi_pred_all.append(np.mean(ewbi_pred_vals))
                ewbi_obs_all.append(np.mean(ewbi_obs_vals))
            else:
                ewbi_pred_all.append(np.nan)
                ewbi_obs_all.append(np.nan)

            # PGSI spatial average (use PGSI mask)
            pgsi_masks = np.array(tp['pgsi_mask'])
            if pgsi_masks.sum() > 0:
                pgsi_pred_vals = np.array(tp['pgsi_pred'])[pgsi_masks > 0]
                pgsi_obs_vals = np.array(tp['pgsi_obs'])[pgsi_masks > 0]
                pgsi_pred_all.append(np.mean(pgsi_pred_vals))
                pgsi_obs_all.append(np.mean(pgsi_obs_vals))
            else:
                pgsi_pred_all.append(np.nan)
                pgsi_obs_all.append(np.nan)

    # Convert to arrays
    ewbi_obs = np.array(ewbi_obs_all)
    ewbi_pred = np.array(ewbi_pred_all)
    pgsi_obs = np.array(pgsi_obs_all)
    pgsi_pred = np.array(pgsi_pred_all)

    # Independent validity masks
    valid_e = np.isfinite(ewbi_obs) & np.isfinite(ewbi_pred)
    valid_p = np.isfinite(pgsi_obs) & np.isfinite(pgsi_pred)

    # Count valid samples
    n_ewbi = np.sum(valid_e)
    n_pgsi = np.sum(valid_p)

    print(f"Evaluation counts: n_ewbi={n_ewbi}, n_pgsi={n_pgsi}")

    # Initialize metrics
    metrics = {}

    # EWBI metrics (only on valid_e points)
    if n_ewbi > 1:
        metrics['ewbi_rmse'] = np.sqrt(mean_squared_error(ewbi_obs[valid_e], ewbi_pred[valid_e]))
        metrics['ewbi_mae'] = mean_absolute_error(ewbi_obs[valid_e], ewbi_pred[valid_e])
        metrics['ewbi_r2'] = r2_score(ewbi_obs[valid_e], ewbi_pred[valid_e])
        metrics['ewbi_corr'] = pearsonr(ewbi_obs[valid_e], ewbi_pred[valid_e])[0]
    else:
        metrics['ewbi_rmse'] = np.nan
        metrics['ewbi_mae'] = np.nan
        metrics['ewbi_r2'] = np.nan
        metrics['ewbi_corr'] = np.nan
    metrics['n_ewbi'] = n_ewbi

    # PGSI metrics (only on valid_p points)
    if n_pgsi > 1:
        metrics['pgsi_rmse'] = np.sqrt(mean_squared_error(pgsi_obs[valid_p], pgsi_pred[valid_p]))
        metrics['pgsi_mae'] = mean_absolute_error(pgsi_obs[valid_p], pgsi_pred[valid_p])
        metrics['pgsi_r2'] = r2_score(pgsi_obs[valid_p], pgsi_pred[valid_p])
        metrics['pgsi_corr'] = pearsonr(pgsi_obs[valid_p], pgsi_pred[valid_p])[0]
    else:
        metrics['pgsi_rmse'] = np.nan
        metrics['pgsi_mae'] = np.nan
        metrics['pgsi_r2'] = np.nan
        metrics['pgsi_corr'] = np.nan
    metrics['n_pgsi'] = n_pgsi

    # Worst-case analysis
    ewbi_5th, pgsi_5th, joint_5th = train_thresholds

    # EWBI worst-5% (on EWBI-valid points)
    ewbi_worst_mask = np.zeros(len(ewbi_obs), dtype=bool)
    if n_ewbi > 0:
        ewbi_worst_mask[valid_e] = ewbi_obs[valid_e] <= ewbi_5th
        ewbi_worst_idx = np.where(valid_e & ewbi_worst_mask)[0]
        if len(ewbi_worst_idx) > 0:
            metrics['ewbi_worst5_rmse'] = np.sqrt(mean_squared_error(
                ewbi_obs[ewbi_worst_idx], ewbi_pred[ewbi_worst_idx]
            ))
        else:
            metrics['ewbi_worst5_rmse'] = np.nan
    else:
        metrics['ewbi_worst5_rmse'] = np.nan

    # PGSI worst-5% (on PGSI-valid points)
    pgsi_worst_mask = np.zeros(len(pgsi_obs), dtype=bool)
    if n_pgsi > 0:
        pgsi_worst_mask[valid_p] = pgsi_obs[valid_p] <= pgsi_5th
        pgsi_worst_idx = np.where(valid_p & pgsi_worst_mask)[0]
        if len(pgsi_worst_idx) > 0:
            metrics['pgsi_worst5_rmse'] = np.sqrt(mean_squared_error(
                pgsi_obs[pgsi_worst_idx], pgsi_pred[pgsi_worst_idx]
            ))
        else:
            metrics['pgsi_worst5_rmse'] = np.nan
    else:
        metrics['pgsi_worst5_rmse'] = np.nan

    # Joint analysis (both valid)
    valid_joint = valid_e & valid_p
    n_joint = np.sum(valid_joint)

    if n_joint > 0:
        joint_obs = np.minimum(ewbi_obs[valid_joint], pgsi_obs[valid_joint])
        joint_worst_mask = joint_obs <= joint_5th

        if joint_worst_mask.sum() > 0:
            joint_worst_idx = np.where(valid_joint)[0][joint_worst_mask]
            metrics['joint_worst5_rmse'] = np.sqrt(mean_squared_error(
                joint_obs[joint_worst_mask],
                np.minimum(ewbi_pred[joint_worst_idx], pgsi_pred[joint_worst_idx])
            ))
        else:
            metrics['joint_worst5_rmse'] = np.nan

        # Drought events
        events = identify_drought_events(joint_worst_mask,
                                         [dates[i] for i in np.where(valid_joint)[0]])
    else:
        metrics['joint_worst5_rmse'] = np.nan
        events = []

    # Baselines
    # Persistence baseline
    if n_ewbi > 1:
        ewbi_persist = np.full_like(ewbi_obs, np.nan)
        valid_e_idx = np.where(valid_e)[0]
        if len(valid_e_idx) > 0:
            ewbi_persist[valid_e_idx[0]] = ewbi_obs[valid_e_idx[0]]  # First copies itself
            for i in range(1, len(valid_e_idx)):
                ewbi_persist[valid_e_idx[i]] = ewbi_obs[valid_e_idx[i - 1]]
            metrics['ewbi_persist_rmse'] = np.sqrt(mean_squared_error(
                ewbi_obs[valid_e], ewbi_persist[valid_e]
            ))
        else:
            metrics['ewbi_persist_rmse'] = np.nan
    else:
        metrics['ewbi_persist_rmse'] = np.nan

    if n_pgsi > 1:
        pgsi_persist = np.full_like(pgsi_obs, np.nan)
        valid_p_idx = np.where(valid_p)[0]
        if len(valid_p_idx) > 0:
            pgsi_persist[valid_p_idx[0]] = pgsi_obs[valid_p_idx[0]]
            for i in range(1, len(valid_p_idx)):
                pgsi_persist[valid_p_idx[i]] = pgsi_obs[valid_p_idx[i - 1]]
            metrics['pgsi_persist_rmse'] = np.sqrt(mean_squared_error(
                pgsi_obs[valid_p], pgsi_persist[valid_p]
            ))
        else:
            metrics['pgsi_persist_rmse'] = np.nan
    else:
        metrics['pgsi_persist_rmse'] = np.nan

    # Seasonal climatology baseline
    if n_ewbi > 0 and climatology and 'ewbi' in climatology:
        ewbi_seasonal = np.array([
            climatology['ewbi'].get(int(d.split('-')[1]), np.nan)
            for d in dates
        ])
        if np.isfinite(ewbi_seasonal[valid_e]).sum() > 0:
            metrics['ewbi_seasonal_rmse'] = np.sqrt(mean_squared_error(
                ewbi_obs[valid_e], ewbi_seasonal[valid_e]
            ))
        else:
            metrics['ewbi_seasonal_rmse'] = np.nan
    else:
        metrics['ewbi_seasonal_rmse'] = np.nan

    if n_pgsi > 0 and climatology and 'pgsi' in climatology:
        pgsi_seasonal = np.array([
            climatology['pgsi'].get(int(d.split('-')[1]), np.nan)
            for d in dates
        ])
        if np.isfinite(pgsi_seasonal[valid_p]).sum() > 0:
            metrics['pgsi_seasonal_rmse'] = np.sqrt(mean_squared_error(
                pgsi_obs[valid_p], pgsi_seasonal[valid_p]
            ))
        else:
            metrics['pgsi_seasonal_rmse'] = np.nan
    else:
        metrics['pgsi_seasonal_rmse'] = np.nan

    # Check if model beats baselines
    beats_persistence = (
            (not np.isnan(metrics['ewbi_rmse']) and
             not np.isnan(metrics['ewbi_persist_rmse']) and
             metrics['ewbi_rmse'] < metrics['ewbi_persist_rmse']) and
            (not np.isnan(metrics['pgsi_rmse']) and
             not np.isnan(metrics['pgsi_persist_rmse']) and
             metrics['pgsi_rmse'] < metrics['pgsi_persist_rmse'])
    )

    metrics['beats_persistence'] = beats_persistence

    return {
        'metrics': metrics,
        'dates': dates,
        'ewbi_obs': ewbi_obs,
        'ewbi_pred': ewbi_pred,
        'pgsi_obs': pgsi_obs,
        'pgsi_pred': pgsi_pred,
        'ewbi_worst_mask': ewbi_worst_mask,
        'pgsi_worst_mask': pgsi_worst_mask,
        'drought_events': events
    }


# ============================================================================
# VISUALIZATION (Adapted for XGBoost)
# ============================================================================

def create_comprehensive_plots(results: Dict, training_history: Dict, land_type: str,
                               output_dir: str = 'results'):
    """Create comprehensive visualization with all required plots."""

    # Extract data
    metrics = results['metrics']
    dates = results['dates']
    ewbi_obs = results['ewbi_obs']
    ewbi_pred = results['ewbi_pred']
    pgsi_obs = results['pgsi_obs']
    pgsi_pred = results['pgsi_pred']
    ewbi_worst = results['ewbi_worst_mask']
    pgsi_worst = results['pgsi_worst_mask']
    events = results['drought_events']

    # Create figure
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'California {land_type.title()} - Drought XGBoost Analysis',
                 fontsize=16, fontweight='bold')

    # 1. EWBI Time Series
    ax1 = plt.subplot(3, 3, 1)
    valid_e = np.isfinite(ewbi_obs) & np.isfinite(ewbi_pred)
    if valid_e.sum() > 0:
        x_dates = np.arange(len(dates))
        ax1.plot(x_dates[valid_e], ewbi_obs[valid_e], 'b-', label='Observed', alpha=0.7)
        ax1.plot(x_dates[valid_e], ewbi_pred[valid_e], 'r--', label='Predicted', alpha=0.7)

        # Mark worst 5%
        worst_idx = np.where(valid_e & ewbi_worst)[0]
        if len(worst_idx) > 0:
            ax1.scatter(worst_idx, ewbi_obs[worst_idx], c='red', s=20,
                        zorder=5, label='Worst 5%')

        # Threshold line
        if 'ewbi_5th' in results:
            ax1.axhline(y=results['ewbi_5th'], color='orange', linestyle=':',
                        label='5th percentile')

    ax1.set_title('EWBI Forecasting (XGBoost)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('EWBI')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. PGSI Time Series
    ax2 = plt.subplot(3, 3, 2)
    valid_p = np.isfinite(pgsi_obs) & np.isfinite(pgsi_pred)
    if valid_p.sum() > 0:
        x_dates = np.arange(len(dates))
        ax2.plot(x_dates[valid_p], pgsi_obs[valid_p], 'g-', label='Observed', alpha=0.7)
        ax2.plot(x_dates[valid_p], pgsi_pred[valid_p], 'r--', label='Predicted', alpha=0.7)

        # Mark worst 5%
        worst_idx = np.where(valid_p & pgsi_worst)[0]
        if len(worst_idx) > 0:
            ax2.scatter(worst_idx, pgsi_obs[worst_idx], c='red', s=20,
                        zorder=5, label='Worst 5%')

        # Threshold line
        if 'pgsi_5th' in results:
            ax2.axhline(y=results['pgsi_5th'], color='orange', linestyle=':',
                        label='5th percentile')

    ax2.set_title('PGSI Forecasting (XGBoost)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('PGSI')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Training History (Feature Importance)
    ax3 = plt.subplot(3, 3, 3)
    if 'feature_importance' in training_history and training_history['feature_importance'] is not None:
        importance = training_history['feature_importance']
        # Show top 20 features
        n_show = min(20, len(importance))
        top_indices = np.argsort(importance)[-n_show:]
        ax3.barh(range(n_show), importance[top_indices])
        ax3.set_yticks(range(n_show))
        ax3.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax3.set_xlabel('Importance')
        ax3.set_title('Top Feature Importances')
    else:
        ax3.text(0.5, 0.5, 'XGBoost Model\nTrained Successfully',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Model Training Status')
    ax3.grid(True, alpha=0.3)

    # 4. EWBI Scatter
    ax4 = plt.subplot(3, 3, 4)
    if valid_e.sum() > 1:
        ax4.scatter(ewbi_obs[valid_e], ewbi_pred[valid_e], alpha=0.5, s=10)

        # Add 1:1 line
        lims = [np.min([ewbi_obs[valid_e].min(), ewbi_pred[valid_e].min()]),
                np.max([ewbi_obs[valid_e].max(), ewbi_pred[valid_e].max()])]
        ax4.plot(lims, lims, 'k--', alpha=0.5, label='1:1')

        # Add R²
        if not np.isnan(metrics['ewbi_r2']):
            ax4.text(0.05, 0.95, f'R² = {metrics["ewbi_r2"]:.3f}',
                     transform=ax4.transAxes, fontsize=10,
                     verticalalignment='top')

    ax4.set_title('EWBI: Observed vs Predicted')
    ax4.set_xlabel('Observed')
    ax4.set_ylabel('Predicted')
    ax4.grid(True, alpha=0.3)

    # 5. PGSI Scatter
    ax5 = plt.subplot(3, 3, 5)
    if valid_p.sum() > 1:
        ax5.scatter(pgsi_obs[valid_p], pgsi_pred[valid_p], alpha=0.5, s=10, c='green')

        # Add 1:1 line
        lims = [np.min([pgsi_obs[valid_p].min(), pgsi_pred[valid_p].min()]),
                np.max([pgsi_obs[valid_p].max(), pgsi_pred[valid_p].max()])]
        ax5.plot(lims, lims, 'k--', alpha=0.5, label='1:1')

        # Add R²
        if not np.isnan(metrics['pgsi_r2']):
            ax5.text(0.05, 0.95, f'R² = {metrics["pgsi_r2"]:.3f}',
                     transform=ax5.transAxes, fontsize=10,
                     verticalalignment='top')

    ax5.set_title('PGSI: Observed vs Predicted')
    ax5.set_xlabel('Observed')
    ax5.set_ylabel('Predicted')
    ax5.grid(True, alpha=0.3)

    # 6. Model Parameters
    ax6 = plt.subplot(3, 3, 6)
    param_text = f"""
XGBoost Parameters:
    Max Depth: {CONFIG['max_depth']}
    Learning Rate: {CONFIG['learning_rate']}
    N Estimators: {CONFIG['n_estimators']}
    Subsample: {CONFIG['subsample']}
    Colsample: {CONFIG['colsample_bytree']}
    Reg Alpha: {CONFIG['reg_alpha']}
    Reg Lambda: {CONFIG['reg_lambda']}
    """
    ax6.text(0.1, 0.9, param_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', family='monospace')
    ax6.set_title('Model Configuration')
    ax6.axis('off')

    # 7. Comparison with Baselines
    ax7 = plt.subplot(3, 3, 7)
    models = ['XGBoost', 'Persistence', 'Seasonal']

    ewbi_rmses = [
        metrics['ewbi_rmse'],
        metrics.get('ewbi_persist_rmse', np.nan),
        metrics.get('ewbi_seasonal_rmse', np.nan)
    ]
    pgsi_rmses = [
        metrics['pgsi_rmse'],
        metrics.get('pgsi_persist_rmse', np.nan),
        metrics.get('pgsi_seasonal_rmse', np.nan)
    ]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax7.bar(x - width / 2, ewbi_rmses, width, label='EWBI', color='blue', alpha=0.7)
    bars2 = ax7.bar(x + width / 2, pgsi_rmses, width, label='PGSI', color='green', alpha=0.7)

    # Add value labels on bars (only if finite)
    for bar, val in zip(bars1, ewbi_rmses):
        if np.isfinite(val):
            ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    for bar, val in zip(bars2, pgsi_rmses):
        if np.isfinite(val):
            ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax7.set_title('Model vs Baselines (RMSE)')
    ax7.set_xticks(x)
    ax7.set_xticklabels(models)
    ax7.set_ylabel('RMSE')
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # 8. Drought Events Timeline
    ax8 = plt.subplot(3, 3, 8)
    if events:
        for i, event in enumerate(events):
            start_idx = dates.index(event['start'])
            ax8.barh(i, event['duration'], left=start_idx, height=0.8,
                     color='darkred', alpha=0.7)
            ax8.text(start_idx + event['duration'] / 2, i,
                     f"{event['duration']} months",
                     ha='center', va='center', fontsize=8, color='white')

        ax8.set_yticks(range(len(events)))
        ax8.set_yticklabels([f"Event {i + 1}" for i in range(len(events))])
        ax8.set_xlabel('Time Index')
        ax8.set_title('Drought Events (Consecutive Worst-5% Months)')
    else:
        ax8.text(0.5, 0.5, 'No drought events detected',
                 ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Drought Events')

    ax8.grid(True, alpha=0.3, axis='x')

    # 9. Metrics Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
    === XGBOOST METRICS SUMMARY ===

    EWBI:
      RMSE: {metrics['ewbi_rmse']:.4f}
      MAE: {metrics['ewbi_mae']:.4f}
      R²: {metrics['ewbi_r2']:.4f}
      Corr: {metrics['ewbi_corr']:.4f}
      n: {metrics['n_ewbi']:.0f}
      Worst-5% RMSE: {metrics.get('ewbi_worst5_rmse', np.nan):.4f}

    PGSI:
      RMSE: {metrics['pgsi_rmse']:.4f}
      MAE: {metrics['pgsi_mae']:.4f}
      R²: {metrics['pgsi_r2']:.4f}
      Corr: {metrics['pgsi_corr']:.4f}
      n: {metrics['n_pgsi']:.0f}
      Worst-5% RMSE: {metrics.get('pgsi_worst5_rmse', np.nan):.4f}

    Baselines:
      Beats Persistence: {metrics.get('beats_persistence', False)}
      Drought Events: {len(events)}
    """

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
             fontsize=10, verticalalignment='top', family='monospace')

    plt.tight_layout()

    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{land_type.lower()}_xgboost_analysis.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot to {output_path}")


# ============================================================================
# SAVE RESULTS (Same as original)
# ============================================================================

def save_results(results: Dict, training_history: Dict, land_type: str,
                 output_dir: str = 'results'):
    """Save all results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Detailed results
    detailed_df = pd.DataFrame({
        'date': results['dates'],
        'ewbi_obs': results['ewbi_obs'],
        'ewbi_pred': results['ewbi_pred'],
        'pgsi_obs': results['pgsi_obs'],
        'pgsi_pred': results['pgsi_pred'],
        'ewbi_worst': results['ewbi_worst_mask'].astype(int),
        'pgsi_worst': results['pgsi_worst_mask'].astype(int)
    })
    detailed_path = os.path.join(output_dir, f'{land_type.lower()}_detailed_results.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  Saved detailed results to {detailed_path}")

    # Drought events
    if results['drought_events']:
        events_df = pd.DataFrame(results['drought_events'])
        events_path = os.path.join(output_dir, f'{land_type.lower()}_drought_events.csv')
        events_df.to_csv(events_path, index=False)
        print(f"  Saved drought events to {events_path}")

    # Training history (XGBoost specific)
    if training_history.get('feature_importance') is not None:
        importance_df = pd.DataFrame({
            'feature_idx': range(len(training_history['feature_importance'])),
            'importance': training_history['feature_importance']
        })
        importance_path = os.path.join(output_dir, f'{land_type.lower()}_feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"  Saved feature importance to {importance_path}")

    # Metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_path = os.path.join(output_dir, f'{land_type.lower()}_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics to {metrics_path}")


# ============================================================================
# DATA LOADING FUNCTION (Same as original)
# ============================================================================

def load_data_for_land_type(land_type: str, downsample: int = 4) -> Dict:
    """
    Load all rasters for a land type, define a reference grid from any NIRv file,
    regrid all inputs to that grid, and record each raster's ORIGINAL (post-downsample)
    shape for 'resolution confidence' features later.
    """
    print(f"\n{'=' * 60}")
    print(f"Loading data for {land_type}")
    print(f"{'=' * 60}")

    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)

    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    valid_timesteps: List[Tuple[int, int]] = []
    reference_meta: Optional[Dict[str, Any]] = None

    # Find a reference grid from any NIRv file (train preferred)
    print("Finding reference grid...")
    # Try training years first
    for year in range(2001, 2022):
        if reference_meta: break
        for month in range(1, 13):
            filepath = find_data_file(train_base, 'NIRv', year, month, land_type, is_test=False)
            if filepath:
                _, reference_meta = read_geotiff(filepath, downsample)
                print(f"Reference grid set from training NIRv: {os.path.basename(filepath)}")
                print(f"  Shape: {reference_meta['shape']}")
                break

    # Fallback: any test NIRv
    if not reference_meta:
        for year in range(2022, 2025):
            if reference_meta: break
            for month in range(1, 13):
                filepath = find_data_file(test_base, 'NIRv', year, month, land_type, is_test=True)
                if filepath:
                    _, reference_meta = read_geotiff(filepath, downsample)
                    print(f"Reference grid set from test NIRv: {os.path.basename(filepath)}")
                    print(f"  Shape: {reference_meta['shape']}")
                    break

    if not reference_meta:
        print("ERROR: Could not find any NIRv file to set reference grid!")
        return {'all_data': {}, 'valid_timesteps': [], 'reference_meta': None}

    ref_shape = reference_meta['shape']

    # Small helper to load one raster, regrid, and store src shape
    def _ingest(var_folder_base: str, var: str, year: int, month: int, is_test: bool) -> Tuple[
        Optional[np.ndarray], Optional[Tuple[int, int]]]:
        fp = find_data_file(var_folder_base, var, year, month, land_type, is_test=is_test)
        if not fp:
            return None, None
        arr, meta = read_geotiff(fp, downsample)
        src_shape = meta['shape']  # shape BEFORE regrid (after downsample)
        if arr.shape != ref_shape:
            print(f"  Regridding {var} {year}-{month:02d} from {arr.shape} to {ref_shape}")
            arr = regrid_to_reference(arr, ref_shape)
        return arr, src_shape

    # Load TRAIN/VAL (2001–2021)
    print("Loading training/validation data (2001-2021)...")
    for year in range(2001, 2022):
        for month in range(1, 13):
            key = f"{year}_{month:02d}"
            month_data: Dict[str, Any] = {}

            for var in VARIABLES:
                # Physical availability guards
                if var == 'LST' and year < 2003:
                    continue
                if var == 'TWS' and (year < 2002 or (year == 2002 and month < 4)):
                    continue

                arr, src_shape = _ingest(train_base, var, year, month, is_test=False)
                if arr is not None:
                    month_data[var] = arr
                    month_data[var + '_SRC_SHAPE'] = src_shape

            if month_data:
                all_data[key] = month_data
                valid_timesteps.append((year, month))

    print(f"Loaded {len([k for k in all_data if int(k.split('_')[0]) <= 2021])} months of train/val data")

    # Load TEST (2022–2024)
    print("Loading test data (2022-2024)...")
    for year in range(2022, 2025):
        for month in range(1, 13):
            if year == 2024 and month > 10:  # adjust if needed
                break
            key = f"{year}_{month:02d}"
            month_data: Dict[str, Any] = {}

            for var in VARIABLES:
                arr, src_shape = _ingest(test_base, var, year, month, is_test=True)
                if arr is not None:
                    month_data[var] = arr
                    month_data[var + '_SRC_SHAPE'] = src_shape

            if month_data:
                all_data[key] = month_data
                valid_timesteps.append((year, month))

    print(f"Loaded {len([k for k in all_data if int(k.split('_')[0]) >= 2022])} months of test data")
    print(f"Total timesteps: {len(valid_timesteps)}")

    return {
        'all_data': all_data,
        'valid_timesteps': sorted(valid_timesteps),
        'reference_meta': reference_meta
    }


# ============================================================================
# MAIN PIPELINE (Adapted for XGBoost)
# ============================================================================

def process_land_type(land_type: str) -> Dict:
    """Process a single land type using XGBoost."""
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {land_type} WITH XGBOOST")
    print(f"{'=' * 80}")

    # Load data (same as original)
    data = load_data_for_land_type(land_type, CONFIG['downsample_factor'])
    all_data = data['all_data']
    valid_timesteps = data['valid_timesteps']
    ref_shape = data['reference_meta']['shape']
    train_years = range(2001, 2019)
    all_years = range(2001, 2024)
    all_data = impute_all_vars_monthwise(all_data, valid_timesteps, ref_shape,
                                         train_years=train_years, all_years=all_years,
                                         smooth_factor=0.5)

    # Impute all vars and detrend (same as original)
    all_data = detrend_per_month(all_data, valid_timesteps, VARIABLES, years=(2001, 2018))

    # Update data reference
    data['all_data'] = all_data

    # Compute normalization stats on UPDATED data
    train_stats = compute_training_stats(data['all_data'], data['valid_timesteps'], VARIABLES)
    train_stats['_ref_shape'] = ref_shape

    # Compute seasonal climatology
    climatology = compute_seasonal_climatology(
        data['all_data'], data['valid_timesteps'], train_stats, CONFIG['target_clip']
    )

    if not all_data:
        print(f"ERROR: No data loaded for {land_type}")
        return None

    def log_driver_coverage(all_data, timesteps, pixels, name="driver_coverage"):
        counts = {'SPEI': 0, 'TWS': 0, 'VPD': 0, 'LST': 0}
        total = 0
        for (yy, mm) in timesteps:
            key = f"{yy}_{mm:02d}"
            if key not in all_data: continue
            md = all_data[key]
            for (r, c) in pixels:
                total += 1
                for v in counts:
                    if v in md and np.isfinite(md[v][r, c]):
                        counts[v] += 1
        print(f"[{name}] totals over selected pixels x months = {total}")
        for v, k in counts.items():
            pct = (k / total) * 100 if total > 0 else 0
            print(f"  {v}: {k} ({pct:.1f}%)")

    # Select pixels
    pixel_indices = select_stable_pixels(
        data['all_data'], data['valid_timesteps'],
        CONFIG['max_nodes'], CONFIG['min_valid_frac']
    )

    # If we end up with no EWBI labels, try a gentler selection once.
    tmp_features = create_features_and_targets(
        data['all_data'], data['valid_timesteps'], pixel_indices,
        train_stats, CONFIG['lookback'], CONFIG['target_clip']
    )
    if np.sum(tmp_features['ewbi_masks']) == 0:
        print("No EWBI labels with current selection; retrying pixel selection with min_valid_frac=0.4")
        pixel_indices = select_stable_pixels(
            data['all_data'], data['valid_timesteps'],
            CONFIG['max_nodes'], 0.4
        )
        log_driver_coverage(data['all_data'], data['valid_timesteps'], pixel_indices, name=f"{land_type}_drivers")

    # Create features and targets
    features = create_features_and_targets(
        data['all_data'], data['valid_timesteps'], pixel_indices,
        train_stats, CONFIG['lookback'], CONFIG['target_clip']
    )

    # Compute training thresholds
    train_thresholds = compute_training_thresholds(
        data['all_data'], data['valid_timesteps'], train_stats,
        pixel_indices, CONFIG['target_clip']
    )

    # Prepare data for XGBoost
    xgb_data = prepare_xgboost_data(features, CONFIG['lookback'])

    # Split data
    train_years = set(range(2001, 2019))
    val_years = set(range(2019, 2022))
    test_years = set(range(2022, 2025))

    # Create masks for train/val/test based on timesteps
    train_mask = []
    val_mask = []
    test_mask = []

    for i, (t_idx, p_idx) in enumerate(xgb_data['sample_indices']):
        year = xgb_data['timesteps'][t_idx][0]
        if year in train_years:
            train_mask.append(i)
        elif year in val_years:
            val_mask.append(i)
        elif year in test_years:
            test_mask.append(i)

    train_mask = np.array(train_mask)
    val_mask = np.array(val_mask)
    test_mask = np.array(test_mask)

    print(f"\nXGBoost data splits:")
    print(f"  Train: {len(train_mask)} samples")
    print(f"  Val: {len(val_mask)} samples")
    print(f"  Test: {len(test_mask)} samples")

    # Extract training, validation, and test data
    X_train = xgb_data['features'][train_mask]
    y_ewbi_train = xgb_data['ewbi_targets'][train_mask]
    y_pgsi_train = xgb_data['pgsi_targets'][train_mask]
    mask_ewbi_train = xgb_data['ewbi_masks'][train_mask]
    mask_pgsi_train = xgb_data['pgsi_masks'][train_mask]

    X_val = xgb_data['features'][val_mask]
    y_ewbi_val = xgb_data['ewbi_targets'][val_mask]
    y_pgsi_val = xgb_data['pgsi_targets'][val_mask]
    mask_ewbi_val = xgb_data['ewbi_masks'][val_mask]
    mask_pgsi_val = xgb_data['pgsi_masks'][val_mask]

    X_test = xgb_data['features'][test_mask]
    y_ewbi_test = xgb_data['ewbi_targets'][test_mask]
    y_pgsi_test = xgb_data['pgsi_targets'][test_mask]
    mask_ewbi_test = xgb_data['ewbi_masks'][test_mask]
    mask_pgsi_test = xgb_data['pgsi_masks'][test_mask]
    test_sample_indices = [xgb_data['sample_indices'][i] for i in test_mask]

    # Initialize and train XGBoost model
    print(f"\nInitializing XGBoost model with {X_train.shape[1]} features...")
    model = DroughtXGBoostModel(CONFIG)

    # Train the model
    model.train(
        X_train, y_ewbi_train, y_pgsi_train, mask_ewbi_train, mask_pgsi_train,
        X_val, y_ewbi_val, y_pgsi_val, mask_ewbi_val, mask_pgsi_val,
        train_thresholds=train_thresholds
    )

    # Calibrate the model
    model.calibrate(X_val, y_ewbi_val, y_pgsi_val, mask_ewbi_val, mask_pgsi_val)

    # Create training history for visualization
    training_history = {}

    # Get feature importance if available
    if model.ewbi_model is not None and hasattr(model.ewbi_model, 'feature_importances_'):
        ewbi_importance = model.ewbi_model.feature_importances_
    else:
        ewbi_importance = None

    if model.pgsi_model is not None and hasattr(model.pgsi_model, 'feature_importances_'):
        pgsi_importance = model.pgsi_model.feature_importances_
    else:
        pgsi_importance = None

    # Combine feature importances (average if both available)
    if ewbi_importance is not None and pgsi_importance is not None:
        combined_importance = (ewbi_importance + pgsi_importance) / 2
    elif ewbi_importance is not None:
        combined_importance = ewbi_importance
    elif pgsi_importance is not None:
        combined_importance = pgsi_importance
    else:
        combined_importance = None

    training_history['feature_importance'] = combined_importance

    print(f"\nModel training completed:")
    if model.ewbi_model is not None:
        print(f"  EWBI model: {model.ewbi_model.n_estimators} trees")
    if model.pgsi_model is not None:
        print(f"  PGSI model: {model.pgsi_model.n_estimators} trees")

    # Evaluate on test set
    results = evaluate_xgboost_model(
        model, X_test, y_ewbi_test, y_pgsi_test,
        mask_ewbi_test, mask_pgsi_test, test_sample_indices,
        train_thresholds, xgb_data['timesteps'], climatology, pixel_indices
    )

    # Add thresholds to results for plotting
    results['ewbi_5th'] = train_thresholds[0]
    results['pgsi_5th'] = train_thresholds[1]
    results['joint_5th'] = train_thresholds[2]

    # Create visualizations
    create_comprehensive_plots(results, training_history, land_type)

    # Save results
    save_results(results, training_history, land_type)

    # Cleanup
    cleanup()

    return results['metrics']


def main():
    """Main entry point for XGBoost drought forecasting."""
    print("=" * 80)
    print("CALIFORNIA DROUGHT XGBOOST FORECASTER")
    print("=" * 80)

    all_metrics = {}

    # Process each land type
    for land_type in LAND_TYPES:
        try:
            metrics = process_land_type(land_type)
            if metrics:
                all_metrics[land_type] = metrics
        except Exception as e:
            print(f"ERROR processing {land_type}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save summary
    if all_metrics:
        summary_df = pd.DataFrame(all_metrics).T
        summary_df.index.name = 'land_type'
        summary_path = os.path.join('results', 'summary_metrics.csv')
        os.makedirs('results', exist_ok=True)
        summary_df.to_csv(summary_path)
        print(f"\nSaved summary metrics to {summary_path}")

        print("\n" + "=" * 80)
        print("XGBOOST SUMMARY RESULTS")
        print("=" * 80)
        print(summary_df)

    print("\nXGBoost processing complete!")


if __name__ == "__main__":
    main()