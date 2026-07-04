#!/usr/bin/env python3
"""
California Drought CNN Forecaster
Complete implementation with CNN-based spatial modeling instead of graph networks,
maintaining the same data processing pipeline and forecasting approach.
"""

import os
import gc
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

import rasterio
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

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
    # Model
    'hidden_dim': 192,
    'num_cnn_layers': 4,
    'kernel_size': 3,
    'dropout': 0.2,
    'use_attention': True,

    # Training
    'lookback': 12,
    'batch_size': 8,
    'num_epochs': 150,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'patience': 15,

    # Data
    'downsample_factor': 4,
    'max_grid_size': (64, 64),  # CNN works with spatial grids
    'min_valid_frac': 0.4,
    'target_clip': 4.0,

    # Device/seed
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,

    # Coverage thresholds
    "min_ewbi_drivers": 2,
    "min_pgsi_vars": 3,
}

# Set seeds for reproducibility
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Device: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# DATA PATHS AND PATTERNS
# ============================================================================

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
VARIABLES = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']


# ============================================================================
# PATH HELPER FUNCTIONS (SAME AS ORIGINAL)
# ============================================================================

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


def gpu_cleanup():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# FILE FINDING WITH ROBUST PATTERNS (SAME AS ORIGINAL)
# ============================================================================

def find_data_file(base_path: str, variable: str, year: int, month: int,
                   land_type: str, is_test: bool = False) -> Optional[str]:
    """Find data file with robust pattern matching."""
    var_path = os.path.join(base_path, variable)

    if not os.path.exists(var_path):
        print(f"Warning: Variable subfolder does not exist: {var_path}")
        return None

    # Map land types for file patterns
    if is_test:
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
# RASTER I/O AND PROCESSING (MOSTLY SAME AS ORIGINAL)
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


# [KEEPING ALL IMPUTATION AND PROCESSING FUNCTIONS FROM ORIGINAL]

def impute_all_vars_monthwise(all_data, valid_timesteps, reference_shape,
                              train_years=range(2001, 2019),
                              all_years=range(2001, 2025),
                              smooth_factor=0.5):
    """Split-safe wrapper: fit imputation on train_years, apply to all_years."""
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
    """Fit per-month smoothing (per pixel) on fit_years ONLY, then predict/fill for all apply_years."""
    months = range(1, 13)

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
        fit_frames = [get_arr(y, m) for y in fit_years]
        apply_frames = [get_arr(y, m) for y in apply_years]
        fit_stack = np.stack(fit_frames, axis=0)
        out_stack = np.stack(apply_frames, axis=0)

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
        x_app = np.linspace(0, Tf - 1, Ta)
        for r in range(H):
            for c in range(W):
                y_fit = fit_stack[:, r, c]
                ok = np.isfinite(y_fit)
                if ok.sum() >= 4:
                    try:
                        s_val = smooth_factor * ok.sum()
                        spl = UnivariateSpline(x_fit[ok], y_fit[ok], s=s_val, k=3)
                        pred = spl(x_app)
                        ts = out_stack[:, r, c]
                        nan_mask = ~np.isfinite(ts)
                        ts[nan_mask] = pred[nan_mask]
                        out_stack[:, r, c] = ts
                    except Exception:
                        pass

        # Step 3: month mean over FIT if anything still missing in OUT
        month_mean_fit = np.nanmean(fit_stack, axis=0)
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
    """Fit per-month linear trend over training years and subtract from all years."""
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
                if year <= 2018 and m == month:
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
        valid = data[np.isfinite(data)]
        if len(valid) > 0:
            median = np.median(valid)
            mad = np.median(np.abs(valid - median))
            std = mad * 1.4826 + 1e-8
            z = (data - median) / std
        else:
            z = np.zeros_like(data)

    z[~np.isfinite(z)] = 0.0
    z = np.clip(z, -clip, clip)
    return z


def compute_seasonal_climatology(all_data: Dict, valid_timesteps: List,
                                 train_stats: Dict, clip: float = 4.0) -> Dict:
    """Compute monthly climatology from training data only (<=2018)."""
    print("Computing seasonal climatology baseline (2001-2018)...")

    climatology = {'ewbi': {}, 'pgsi': {}}

    for month in range(1, 13):
        ewbi_values = []
        pgsi_values = []

        for year, m in valid_timesteps:
            if year <= 2018 and m == month:
                key = f"{year}_{m:02d}"
                if key not in all_data:
                    continue

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
# SPATIAL GRID PROCESSING FOR CNN
# ============================================================================

def create_spatial_grids(all_data: Dict, valid_timesteps: List,
                         train_stats: Dict, lookback: int = 6,
                         clip: float = 4.0, target_size: Tuple[int, int] = (64, 64)) -> Dict:
    """
    Create spatial grid features and targets for CNN.
    Returns grids of shape [T, H, W, C] where C is number of channels.
    """
    print("Creating spatial grids for CNN...")

    n_times = len(valid_timesteps)
    H, W = target_size

    # Channel layout: 9 variables * 3 features (z-score, missing_flag, confidence) + temporal features
    n_var_channels = 9 * 3  # z-score, missing flag, confidence per variable
    n_temporal_channels = 5  # TWS availability, month sin/cos, normalized year/month
    n_channels = n_var_channels + n_temporal_channels

    # Initialize arrays
    all_features = np.zeros((n_times, H, W, n_channels), dtype=np.float32)
    ewbi_targets = np.full((n_times, H, W), np.nan, dtype=np.float32)
    pgsi_targets = np.full((n_times, H, W), np.nan, dtype=np.float32)
    ewbi_masks = np.zeros((n_times, H, W), dtype=np.float32)
    pgsi_masks = np.zeros((n_times, H, W), dtype=np.float32)

    for t_idx, (year, month) in enumerate(valid_timesteps):
        key = f"{year}_{month:02d}"
        if key not in all_data:
            continue

        month_data = all_data[key]

        # Process each variable
        channel_idx = 0
        for var_idx, var in enumerate(VARIABLES):
            if var in month_data:
                # Resize to target grid size
                data = month_data[var]
                if data.shape != (H, W):
                    data = regrid_to_reference(data, (H, W))

                # Z-score normalization
                z_data = robust_z_score_normalize(data, var, month, train_stats, clip)
                all_features[t_idx, :, :, channel_idx] = z_data

                # Missing flag (1 if missing, 0 if present)
                missing_flag = (~np.isfinite(data)).astype(np.float32)
                all_features[t_idx, :, :, channel_idx + 9] = missing_flag

                # Resolution confidence (simplified to 1.0 for now)
                all_features[t_idx, :, :, channel_idx + 18] = 1.0

            else:
                # Variable completely missing
                all_features[t_idx, :, :, channel_idx + 9] = 1.0  # missing flag

            channel_idx += 1

        # Temporal features (broadcast to all spatial locations)
        temporal_base = n_var_channels

        # TWS availability
        tws_available = 1.0 if 'TWS' in month_data else 0.0
        all_features[t_idx, :, :, temporal_base] = tws_available

        # Month sin/cos
        all_features[t_idx, :, :, temporal_base + 1] = np.sin(2 * np.pi * month / 12)
        all_features[t_idx, :, :, temporal_base + 2] = np.cos(2 * np.pi * month / 12)

        # Normalized year/month
        all_features[t_idx, :, :, temporal_base + 3] = (year - 2001) / 23.0
        all_features[t_idx, :, :, temporal_base + 4] = month / 12.0

        # Compute targets pixel-wise
        min_e = CONFIG.get("min_ewbi_drivers", 3)
        min_v = CONFIG.get("min_pgsi_vars", 3)

        for r in range(H):
            for c in range(W):
                # EWBI components
                ewbi_components = []
                for var, sign in [('SPEI', 1), ('TWS', 1), ('VPD', -1), ('LST', -1)]:
                    if var in month_data:
                        data = month_data[var]
                        if data.shape != (H, W):
                            data = regrid_to_reference(data, (H, W))
                        val = data[r, c]
                        if np.isfinite(val):
                            z_val = robust_z_score_normalize(
                                np.array([val]), var, month, train_stats, clip
                            )[0]
                            ewbi_components.append(sign * z_val)

                if len(ewbi_components) >= min_e:
                    month_ewbi_vals.append(np.mean(ewbi_components))

                # PGSI components
                pgsi_components = []
                for var in ['GOSIF', 'GPP', 'NIRv', 'NDVI', 'GNDVI']:
                    if var in md:
                        data = md[var]
                        if data.shape != (H, W):
                            data = regrid_to_reference(data, (H, W))
                        val = data[r, c]
                        if np.isfinite(val):
                            z = robust_z_score_normalize(
                                np.array([val]), var, month, train_stats, clip
                            )[0]
                            pgsi_components.append(z)

                if len(pgsi_components) >= min_v:
                    month_pgsi_vals.append(np.mean(pgsi_components))

        # Take spatial mean over valid pixels for this month
        if len(month_ewbi_vals) > 0:
            ewbi_values.append(float(np.mean(month_ewbi_vals)))
        if len(month_pgsi_vals) > 0:
            pgsi_values.append(float(np.mean(month_pgsi_vals)))
        if len(month_ewbi_vals) > 0 and len(month_pgsi_vals) > 0:
            joint_values.append(float(min(np.mean(month_ewbi_vals), np.mean(month_pgsi_vals))))

    # Compute percentiles
    ewbi_5th = np.percentile(ewbi_values, 5) if len(ewbi_values) else -2.0
    pgsi_5th = np.percentile(pgsi_values, 5) if len(pgsi_values) else -2.0
    joint_5th = np.percentile(joint_values, 5) if len(joint_values) else -2.0

    print(f"  Thresholds: EWBI={ewbi_5th:.3f}, PGSI={pgsi_5th:.3f}, Joint={joint_5th:.3f}")
    return ewbi_5th, pgsi_5th, joint_5th


# ============================================================================
# CNN DATASET
# ============================================================================

class DroughtCNNDataset(Dataset):
    """
    CNN Dataset that returns spatial grids instead of node features.
    Returns:
      x:        FloatTensor [C, L, H, W]  (channels, lookback, height, width)
      ewbi_y:   FloatTensor [H, W]
      pgsi_y:   FloatTensor [H, W]
      ewbi_m:   FloatTensor [H, W]
      pgsi_m:   FloatTensor [H, W]
    """

    def __init__(self,
                 features: np.ndarray,  # [T, H, W, C]
                 ewbi_targets: np.ndarray,  # [T, H, W]
                 pgsi_targets: np.ndarray,  # [T, H, W]
                 ewbi_masks: np.ndarray,  # [T, H, W]
                 pgsi_masks: np.ndarray,  # [T, H, W]
                 indices: List[int],  # list of t where target exists
                 lookback: int):
        self.X = features  # [T, H, W, C]
        self.ey = ewbi_targets
        self.py = pgsi_targets
        self.em = ewbi_masks
        self.pm = pgsi_masks
        self.idx = indices
        self.L = lookback

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        t = self.idx[i]
        # Window: [t-L, ..., t-1]
        x_win = self.X[t - self.L:t]  # [L, H, W, C]

        # Transpose to [C, L, H, W] for CNN processing
        x_win = np.transpose(x_win, (3, 0, 1, 2))

        ewbi_y = self.ey[t]  # [H, W]
        pgsi_y = self.py[t]  # [H, W]
        ewbi_m = self.em[t]  # [H, W]
        pgsi_m = self.pm[t]  # [H, W]

        return (torch.from_numpy(x_win).float(),
                torch.from_numpy(ewbi_y).float(),
                torch.from_numpy(pgsi_y).float(),
                torch.from_numpy(ewbi_m).float(),
                torch.from_numpy(pgsi_m).float())


# ============================================================================
# CNN MODEL ARCHITECTURE
# ============================================================================

class SpatialTemporalCNN(nn.Module):
    """
    CNN-based drought forecasting model.
    Processes spatial-temporal data using 3D convolutions and attention mechanisms.
    """

    def __init__(self, input_channels, hidden_dim=192, num_cnn_layers=4,
                 kernel_size=3, dropout=0.2, use_attention=True):
        super().__init__()
        self.hidden = hidden_dim
        self.use_attention = use_attention
        self.dropout = nn.Dropout(dropout)

        # 3D CNN for spatial-temporal feature extraction
        self.conv3d_layers = nn.ModuleList()
        self.bn3d_layers = nn.ModuleList()

        # First layer: input_channels -> hidden_dim
        self.conv3d_layers.append(
            nn.Conv3d(input_channels, hidden_dim,
                      kernel_size=(3, kernel_size, kernel_size),
                      padding=(1, kernel_size // 2, kernel_size // 2))
        )
        self.bn3d_layers.append(nn.BatchNorm3d(hidden_dim))

        # Additional layers
        for i in range(num_cnn_layers - 1):
            self.conv3d_layers.append(
                nn.Conv3d(hidden_dim, hidden_dim,
                          kernel_size=(3, kernel_size, kernel_size),
                          padding=(1, kernel_size // 2, kernel_size // 2))
            )
            self.bn3d_layers.append(nn.BatchNorm3d(hidden_dim))

        # Temporal attention mechanism
        if self.use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim, num_heads=8, batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)

        # 2D CNN for spatial refinement
        self.conv2d_layers = nn.ModuleList()
        self.bn2d_layers = nn.ModuleList()

        for i in range(2):  # Two 2D conv layers
            self.conv2d_layers.append(
                nn.Conv2d(hidden_dim, hidden_dim,
                          kernel_size=kernel_size,
                          padding=kernel_size // 2)
            )
            self.bn2d_layers.append(nn.BatchNorm2d(hidden_dim))

        # Task-specific heads
        self.ewbi_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

        self.pgsi_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: [B, C, L, H, W] - batch, channels, lookback, height, width
        returns: (ewbi_pred, pgsi_pred) with shape [B, H, W]
        """
        B, C, L, H, W = x.shape

        # 3D CNN processing
        for i, (conv3d, bn3d) in enumerate(zip(self.conv3d_layers, self.bn3d_layers)):
            x = conv3d(x)
            x = bn3d(x)
            x = F.relu(x)
            if i < len(self.conv3d_layers) - 1:  # Don't dropout on last layer
                x = self.dropout(x)

        # x shape: [B, hidden_dim, L, H, W]

        # Temporal attention (optional)
        if self.use_attention:
            # Reshape for attention: [B*H*W, L, hidden_dim]
            x_reshaped = x.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, L, hidden_dim]
            x_reshaped = x_reshaped.view(B * H * W, L, self.hidden)

            # Apply attention
            attn_out, _ = self.temporal_attention(x_reshaped, x_reshaped, x_reshaped)
            attn_out = self.attention_norm(attn_out + x_reshaped)

            # Reshape back and take last timestep
            attn_out = attn_out.view(B, H, W, L, self.hidden)
            x = attn_out[:, :, :, -1, :].permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]
        else:
            # Just take the last timestep
            x = x[:, :, -1, :, :]  # [B, hidden_dim, H, W]

        # 2D CNN for spatial refinement
        for conv2d, bn2d in zip(self.conv2d_layers, self.bn2d_layers):
            residual = x
            x = conv2d(x)
            x = bn2d(x)
            x = F.relu(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection

        # Task heads
        ewbi_pred = self.ewbi_head(x).squeeze(1)  # [B, H, W]
        pgsi_pred = self.pgsi_head(x).squeeze(1)  # [B, H, W]

        return ewbi_pred, pgsi_pred


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_cnn_model(model, train_loader, val_loader, cfg, train_thresholds=None):
    """Train CNN model with spatial loss functions."""
    device = cfg['device']
    model = model.to(device)

    opt = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['num_epochs'])

    import copy
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    patience = cfg['patience']
    no_improve = 0

    history = {'train_loss': [], 'val_loss': [], 'lr_epoch': []}

    def _spatial_masked_mse(pred, target, mask):
        """Compute MSE over spatial mask."""
        if mask.sum() == 0:
            return (pred * 0.0).sum()
        diff = (pred - target) * mask
        return (diff ** 2).sum() / (mask.sum() + 1e-8)

    def _spatial_quantile_loss(pred, target, mask, q=0.1):
        """Spatial quantile loss for emphasizing lower tail."""
        if mask.sum() == 0:
            return (pred * 0.0).sum()
        e = (target - pred) * mask
        return torch.mean(torch.maximum(q * e, (q - 1) * e))

    def _spatial_focal_loss(pred, target, mask, alpha=0.25, gamma=2.0):
        """Focal loss adapted for regression to focus on hard examples."""
        if mask.sum() == 0:
            return (pred * 0.0).sum()

        diff = torch.abs(pred - target) * mask
        # Use sigmoid to convert differences to probabilities
        p = torch.sigmoid(-diff)  # Higher for smaller errors
        focal_weight = alpha * (1 - p) ** gamma
        loss = focal_weight * (diff ** 2)
        return loss.sum() / (mask.sum() + 1e-8)

    # Loss weights
    lam_quantile = 0.3
    lam_focal = 0.2

    for epoch in range(cfg['num_epochs']):
        # ======================= TRAIN =======================
        model.train()
        tr_losses = []

        for batch in train_loader:
            x, ewbi_y, pgsi_y, ewbi_m, pgsi_m = [b.to(device) for b in batch]
            opt.zero_grad()

            ewbi_pred, pgsi_pred = model(x)

            # Task reweighting by valid pixel counts
            n_e = ewbi_m.sum().clamp_min(1.0)
            n_p = pgsi_m.sum().clamp_min(1.0)
            w_e = (n_p / (n_e + n_p)).detach()
            w_p = (n_e / (n_e + n_p)).detach()

            # Multiple loss components
            mse = (w_e * _spatial_masked_mse(ewbi_pred, ewbi_y, ewbi_m) +
                   w_p * _spatial_masked_mse(pgsi_pred, pgsi_y, pgsi_m))

            quantile = (w_e * _spatial_quantile_loss(ewbi_pred, ewbi_y, ewbi_m, q=0.1) +
                        w_p * _spatial_quantile_loss(pgsi_pred, pgsi_y, pgsi_m, q=0.1))

            focal = (w_e * _spatial_focal_loss(ewbi_pred, ewbi_y, ewbi_m) +
                     w_p * _spatial_focal_loss(pgsi_pred, pgsi_y, pgsi_m))

            loss = mse + lam_quantile * quantile + lam_focal * focal
            loss.backward()

            if cfg['grad_clip'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])

            opt.step()
            tr_losses.append(loss.detach().item())

        train_loss = float(np.mean(tr_losses)) if tr_losses else np.nan
        history['train_loss'].append(train_loss)

        # ======================= VALIDATION =======================
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, ewbi_y, pgsi_y, ewbi_m, pgsi_m = [b.to(device) for b in batch]
                ewbi_pred, pgsi_pred = model(x)

                n_e = ewbi_m.sum().clamp_min(1.0)
                n_p = pgsi_m.sum().clamp_min(1.0)
                w_e = (n_p / (n_e + n_p)).detach()
                w_p = (n_e / (n_e + n_p)).detach()

                mse = (w_e * _spatial_masked_mse(ewbi_pred, ewbi_y, ewbi_m) +
                       w_p * _spatial_masked_mse(pgsi_pred, pgsi_y, pgsi_m))

                quantile = (w_e * _spatial_quantile_loss(ewbi_pred, ewbi_y, ewbi_m, q=0.1) +
                            w_p * _spatial_quantile_loss(pgsi_pred, pgsi_y, pgsi_m, q=0.1))

                focal = (w_e * _spatial_focal_loss(ewbi_pred, ewbi_y, ewbi_m) +
                         w_p * _spatial_focal_loss(pgsi_pred, pgsi_y, pgsi_m))

                val_losses.append((mse + lam_quantile * quantile + lam_focal * focal).item())

        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        history['val_loss'].append(val_loss)
        history['lr_epoch'].append(opt.param_groups[0]['lr'])
        sched.step()

        # Early stopping
        if np.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{cfg['num_epochs']}: "
                  f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history


# ============================================================================
# CALIBRATION FOR CNN
# ============================================================================

def compute_cnn_calibration(model, val_loader, device, l2=1e-2, clip_a=3.0, clip_b=3.0):
    """Compute calibration for CNN model using spatial means."""
    model.eval()
    e_pred, e_true, p_pred, p_true = [], [], [], []

    with torch.no_grad():
        for batch in val_loader:
            x, ewbi_y, pgsi_y, ewbi_m, pgsi_m = [b.to(device) for b in batch]
            yhat_e, yhat_p = model(x)

            for j in range(x.shape[0]):
                # EWBI spatial mean
                if ewbi_m[j].sum() > 0:
                    e_pred.append(((yhat_e[j] * ewbi_m[j]).sum() /
                                   (ewbi_m[j].sum() + 1e-8)).item())
                    e_true.append(((ewbi_y[j] * ewbi_m[j]).sum() /
                                   (ewbi_m[j].sum() + 1e-8)).item())

                # PGSI spatial mean
                if pgsi_m[j].sum() > 0:
                    p_pred.append(((yhat_p[j] * pgsi_m[j]).sum() /
                                   (pgsi_m[j].sum() + 1e-8)).item())
                    p_true.append(((pgsi_y[j] * pgsi_m[j]).sum() /
                                   (pgsi_m[j].sum() + 1e-8)).item())

    def _ridge(pred, true):
        if len(pred) < 8:
            return 1.0, 0.0
        X = np.vstack([pred, np.ones(len(pred))]).T
        XtX = X.T @ X
        XtX[0, 0] += l2
        XtX[1, 1] += l2
        Xty = X.T @ np.array(true)
        a, b = np.linalg.solve(XtX, Xty)
        a = float(np.clip(a, -clip_a, clip_a))
        b = float(np.clip(b, -clip_b, clip_b))
        return a, b

    a_e, b_e = _ridge(e_pred, e_true)
    a_p, b_p = _ridge(p_pred, p_true)
    return {"ewbi": (a_e, b_e), "pgsi": (a_p, b_p)}


# ============================================================================
# EVALUATION FUNCTIONS
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

    if in_event:
        events.append({
            'start': dates[start_idx],
            'end': dates[-1],
            'duration': len(dates) - start_idx
        })

    return events


def evaluate_cnn_model(model, test_loader, device, calibration,
                       train_thresholds, timesteps, test_indices, climatology):
    """Evaluate CNN model with spatial predictions."""
    model.eval()

    # Storage for predictions (spatial means)
    ewbi_preds_all = []
    ewbi_obs_all = []
    pgsi_preds_all = []
    pgsi_obs_all = []
    dates = []

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            x, ewbi_y, pgsi_y, ewbi_m, pgsi_m = [b.to(device) for b in batch]

            # Get predictions
            ewbi_pred, pgsi_pred = model(x)

            # Apply calibration
            a_e, b_e = calibration['ewbi']
            a_p, b_p = calibration['pgsi']
            ewbi_pred_cal = a_e * ewbi_pred + b_e
            pgsi_pred_cal = a_p * pgsi_pred + b_p

            # Compute spatial means for evaluation
            batch_idx = test_indices[i * test_loader.batch_size:(i + 1) * test_loader.batch_size]

            for j, idx in enumerate(batch_idx):
                if idx < len(timesteps):
                    year, month = timesteps[idx]
                    dates.append(f"{year}-{month:02d}")

                    # EWBI spatial mean
                    if ewbi_m[j].sum() > 0:
                        ewbi_pred_mean = (ewbi_pred_cal[j] * ewbi_m[j]).sum() / ewbi_m[j].sum()
                        ewbi_obs_mean = (ewbi_y[j] * ewbi_m[j]).sum() / ewbi_m[j].sum()
                        ewbi_preds_all.append(ewbi_pred_mean.cpu().numpy())
                        ewbi_obs_all.append(ewbi_obs_mean.cpu().numpy())
                    else:
                        ewbi_preds_all.append(np.nan)
                        ewbi_obs_all.append(np.nan)

                    # PGSI spatial mean
                    if pgsi_m[j].sum() > 0:
                        pgsi_pred_mean = (pgsi_pred_cal[j] * pgsi_m[j]).sum() / pgsi_m[j].sum()
                        pgsi_obs_mean = (pgsi_y[j] * pgsi_m[j]).sum() / pgsi_m[j].sum()
                        pgsi_preds_all.append(pgsi_pred_mean.cpu().numpy())
                        pgsi_obs_all.append(pgsi_obs_mean.cpu().numpy())
                    else:
                        pgsi_preds_all.append(np.nan)
                        pgsi_obs_all.append(np.nan)

    # Convert to arrays
    ewbi_obs = np.array(ewbi_obs_all)
    ewbi_pred = np.array(ewbi_preds_all)
    pgsi_obs = np.array(pgsi_obs_all)
    pgsi_pred = np.array(pgsi_preds_all)

    # Independent validity masks
    valid_e = np.isfinite(ewbi_obs) & np.isfinite(ewbi_pred)
    valid_p = np.isfinite(pgsi_obs) & np.isfinite(pgsi_pred)

    # Count valid samples
    n_ewbi = np.sum(valid_e)
    n_pgsi = np.sum(valid_p)

    print(f"\nEvaluation counts: n_ewbi={n_ewbi}, n_pgsi={n_pgsi}")

    # Initialize metrics
    metrics = {}

    # EWBI metrics
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

    # PGSI metrics
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

    # EWBI worst-5%
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

    # PGSI worst-5%
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

    # Joint analysis
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

        events = identify_drought_events(joint_worst_mask,
                                         [dates[i] for i in np.where(valid_joint)[0]])
    else:
        metrics['joint_worst5_rmse'] = np.nan
        events = []

    # Baselines (Persistence)
    if n_ewbi > 1:
        ewbi_persist = np.full_like(ewbi_obs, np.nan)
        valid_e_idx = np.where(valid_e)[0]
        ewbi_persist[valid_e_idx[0]] = ewbi_obs[valid_e_idx[0]]
        for i in range(1, len(valid_e_idx)):
            ewbi_persist[valid_e_idx[i]] = ewbi_obs[valid_e_idx[i - 1]]
        metrics['ewbi_persist_rmse'] = np.sqrt(mean_squared_error(
            ewbi_obs[valid_e], ewbi_persist[valid_e]
        ))
    else:
        metrics['ewbi_persist_rmse'] = np.nan

    if n_pgsi > 1:
        pgsi_persist = np.full_like(pgsi_obs, np.nan)
        valid_p_idx = np.where(valid_p)[0]
        pgsi_persist[valid_p_idx[0]] = pgsi_obs[valid_p_idx[0]]
        for i in range(1, len(valid_p_idx)):
            pgsi_persist[valid_p_idx[i]] = pgsi_obs[valid_p_idx[i - 1]]
        metrics['pgsi_persist_rmse'] = np.sqrt(mean_squared_error(
            pgsi_obs[valid_p], pgsi_persist[valid_p]
        ))
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
# DATA LOADING FOR CNN
# ============================================================================

def load_data_for_land_type(land_type: str, downsample: int = 4) -> Dict:
    """
    Load all rasters for a land type and create spatial grids.
    Returns dictionary with loaded data for CNN processing.
    """
    print(f"\n{'=' * 60}")
    print(f"Loading data for {land_type}")
    print(f"{'=' * 60}")

    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)

    all_data: Dict[str, Dict[str, np.ndarray]] = {}
    valid_timesteps: List[Tuple[int, int]] = []
    reference_meta: Optional[Dict[str, Any]] = None

    # Find reference grid from NIRv files
    print("Finding reference grid...")
    for year in range(2001, 2022):
        if reference_meta:
            break
        for month in range(1, 13):
            filepath = find_data_file(train_base, 'NIRv', year, month, land_type, is_test=False)
            if filepath:
                _, reference_meta = read_geotiff(filepath, downsample)
                print(f"Reference grid set from training NIRv: {os.path.basename(filepath)}")
                print(f"  Shape: {reference_meta['shape']}")
                break

    if not reference_meta:
        for year in range(2022, 2025):
            if reference_meta:
                break
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

    def _ingest(var_folder_base: str, var: str, year: int, month: int, is_test: bool) -> Tuple[
        Optional[np.ndarray], Optional[Tuple[int, int]]]:
        fp = find_data_file(var_folder_base, var, year, month, land_type, is_test=is_test)
        if not fp:
            return None, None
        arr, meta = read_geotiff(fp, downsample)
        src_shape = meta['shape']
        if arr.shape != ref_shape:
            arr = regrid_to_reference(arr, ref_shape)
        return arr, src_shape

    # Load training/validation data (2001-2021)
    print("Loading training/validation data (2001-2021)...")
    for year in range(2001, 2022):
        for month in range(1, 13):
            key = f"{year}_{month:02d}"
            month_data: Dict[str, Any] = {}

            for var in VARIABLES:
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

    # Load test data (2022-2024)
    print("Loading test data (2022-2024)...")
    for year in range(2022, 2025):
        for month in range(1, 13):
            if year == 2024 and month > 10:
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
# VISUALIZATION (ADAPTED FOR CNN)
# ============================================================================

def create_comprehensive_plots(results: Dict, history: Dict, land_type: str,
                               output_dir: str = 'results'):
    """Create comprehensive visualization adapted for CNN results."""

    metrics = results['metrics']
    dates = results['dates']
    ewbi_obs = results['ewbi_obs']
    ewbi_pred = results['ewbi_pred']
    pgsi_obs = results['pgsi_obs']
    pgsi_pred = results['pgsi_pred']
    ewbi_worst = results['ewbi_worst_mask']
    pgsi_worst = results['pgsi_worst_mask']
    events = results['drought_events']

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(f'California {land_type.title()} - Drought CNN Analysis',
                 fontsize=16, fontweight='bold')

    # 1. EWBI Time Series
    ax1 = plt.subplot(3, 3, 1)
    valid_e = np.isfinite(ewbi_obs) & np.isfinite(ewbi_pred)
    if valid_e.sum() > 0:
        x_dates = np.arange(len(dates))
        ax1.plot(x_dates[valid_e], ewbi_obs[valid_e], 'b-', label='Observed', alpha=0.7)
        ax1.plot(x_dates[valid_e], ewbi_pred[valid_e], 'r--', label='Predicted', alpha=0.7)

        worst_idx = np.where(valid_e & ewbi_worst)[0]
        if len(worst_idx) > 0:
            ax1.scatter(worst_idx, ewbi_obs[worst_idx], c='red', s=20,
                        zorder=5, label='Worst 5%')

        if 'ewbi_5th' in results:
            ax1.axhline(y=results['ewbi_5th'], color='orange', linestyle=':',
                        label='5th percentile')

    ax1.set_title('EWBI Forecasting (CNN)')
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

        worst_idx = np.where(valid_p & pgsi_worst)[0]
        if len(worst_idx) > 0:
            ax2.scatter(worst_idx, pgsi_obs[worst_idx], c='red', s=20,
                        zorder=5, label='Worst 5%')

        if 'pgsi_5th' in results:
            ax2.axhline(y=results['pgsi_5th'], color='orange', linestyle=':',
                        label='5th percentile')

    ax2.set_title('PGSI Forecasting (CNN)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('PGSI')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. Training History
    ax3 = plt.subplot(3, 3, 3)
    epochs = range(1, len(history['train_loss']) + 1)
    ax3.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax3.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax3.set_title('CNN Training Progress')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. EWBI Scatter
    ax4 = plt.subplot(3, 3, 4)
    if valid_e.sum() > 1:
        ax4.scatter(ewbi_obs[valid_e], ewbi_pred[valid_e], alpha=0.5, s=10)

        lims = [np.min([ewbi_obs[valid_e].min(), ewbi_pred[valid_e].min()]),
                np.max([ewbi_obs[valid_e].max(), ewbi_pred[valid_e].max()])]
        ax4.plot(lims, lims, 'k--', alpha=0.5, label='1:1')

        if not np.isnan(metrics['ewbi_r2']):
            ax4.text(0.05, 0.95, f'R² = {metrics["ewbi_r2"]:.3f}',
                     transform=ax4.transAxes, fontsize=10,
                     verticalalignment='top')

    ax4.set_title('EWBI: Observed vs Predicted (CNN)')
    ax4.set_xlabel('Observed')
    ax4.set_ylabel('Predicted')
    ax4.grid(True, alpha=0.3)

    # 5. PGSI Scatter
    ax5 = plt.subplot(3, 3, 5)
    if valid_p.sum() > 1:
        ax5.scatter(pgsi_obs[valid_p], pgsi_pred[valid_p], alpha=0.5, s=10, c='green')

        lims = [np.min([pgsi_obs[valid_p].min(), pgsi_pred[valid_p].min()]),
                np.max([pgsi_obs[valid_p].max(), pgsi_pred[valid_p].max()])]
        ax5.plot(lims, lims, 'k--', alpha=0.5, label='1:1')

        if not np.isnan(metrics['pgsi_r2']):
            ax5.text(0.05, 0.95, f'R² = {metrics["pgsi_r2"]:.3f}',
                     transform=ax5.transAxes, fontsize=10,
                     verticalalignment='top')

    ax5.set_title('PGSI: Observed vs Predicted (CNN)')
    ax5.set_xlabel('Observed')
    ax5.set_ylabel('Predicted')
    ax5.grid(True, alpha=0.3)

    # 6. Learning Rate Schedule
    ax6 = plt.subplot(3, 3, 6)
    if 'lr_epoch' in history and len(history['lr_epoch']) == len(history.get('train_loss', [])):
        ax6.plot(range(1, len(history['lr_epoch']) + 1), history['lr_epoch'], 'g-')
    elif 'lr' in history and len(history['lr']) > 0:
        ax6.plot(range(1, len(history['lr']) + 1), history['lr'], 'g-')
    else:
        ax6.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax6.transAxes)

    ax6.set_title('Learning Rate Schedule')
    ax6.set_xlabel('Epoch')
    ax6.set_ylabel('Learning Rate')
    ax6.set_yscale('log')
    ax6.grid(True, alpha=0.3)

    # 7. Comparison with Baselines
    ax7 = plt.subplot(3, 3, 7)
    models = ['CNN', 'Persistence', 'Seasonal']

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

    for bar, val in zip(bars1, ewbi_rmses):
        if np.isfinite(val):
            ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    for bar, val in zip(bars2, pgsi_rmses):
        if np.isfinite(val):
            ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax7.set_title('CNN vs Baselines (RMSE)')
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
        ax8.set_title('Drought Events (CNN)')
    else:
        ax8.text(0.5, 0.5, 'No drought events detected',
                 ha='center', va='center', transform=ax8.transAxes)
        ax8.set_title('Drought Events (CNN)')

    ax8.grid(True, alpha=0.3, axis='x')

    # 9. Metrics Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_text = f"""
    === CNN METRICS SUMMARY ===

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
    output_path = os.path.join(output_dir, f'{land_type.lower()}_cnn_analysis.png')
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"  Saved CNN plot to {output_path}")


# ============================================================================
# SAVE RESULTS (SAME AS ORIGINAL)
# ============================================================================

def save_results(results: Dict, history: Dict, land_type: str,
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
    detailed_path = os.path.join(output_dir, f'{land_type.lower()}_cnn_detailed_results.csv')
    detailed_df.to_csv(detailed_path, index=False)
    print(f"  Saved detailed results to {detailed_path}")

    # Drought events
    if results['drought_events']:
        events_df = pd.DataFrame(results['drought_events'])
        events_path = os.path.join(output_dir, f'{land_type.lower()}_cnn_drought_events.csv')
        events_df.to_csv(events_path, index=False)
        print(f"  Saved drought events to {events_path}")

    # Training history
    epochs = list(range(1, len(history.get('train_loss', [])) + 1))
    hist_dict = {
        'epoch': epochs,
        'train_loss': history.get('train_loss', []),
        'val_loss': history.get('val_loss', []),
    }
    if len(history.get('lr_epoch', [])) == len(epochs):
        hist_dict['lr_epoch'] = history['lr_epoch']

    history_df = pd.DataFrame(hist_dict)
    history_path = os.path.join(output_dir, f'{land_type.lower()}_cnn_training_history.csv')
    history_df.to_csv(history_path, index=False)
    print(f"  Saved training history to {history_path}")

    # Metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_path = os.path.join(output_dir, f'{land_type.lower()}_cnn_metrics.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Saved metrics to {metrics_path}")


# ============================================================================
# MAIN CNN PIPELINE
# ============================================================================

def process_land_type_cnn(land_type: str) -> Dict:
    """Process a single land type using CNN architecture."""
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {land_type} WITH CNN")
    print(f"{'=' * 80}")

    # Load data
    data = load_data_for_land_type(land_type, CONFIG['downsample_factor'])
    all_data = data['all_data']
    valid_timesteps = data['valid_timesteps']
    ref_shape = data['reference_meta']['shape']

    # Imputation and preprocessing
    train_years = range(2001, 2019)
    all_years = range(2001, 2025)
    all_data = impute_all_vars_monthwise(all_data, valid_timesteps, ref_shape,
                                         train_years=train_years, all_years=all_years,
                                         smooth_factor=0.5)

    # Detrend
    all_data = detrend_per_month(all_data, valid_timesteps, VARIABLES, years=(2001, 2018))
    data['all_data'] = all_data

    # Compute training statistics
    train_stats = compute_training_stats(data['all_data'], data['valid_timesteps'], VARIABLES)
    train_stats['_ref_shape'] = ref_shape

    # Compute seasonal climatology
    climatology = compute_seasonal_climatology(
        data['all_data'], data['valid_timesteps'], train_stats, CONFIG['target_clip']
    )

    if not all_data:
        print(f"ERROR: No data loaded for {land_type}")
        return None

    # Create spatial grids for CNN
    features = create_spatial_grids(
        data['all_data'], data['valid_timesteps'], train_stats,
        CONFIG['lookback'], CONFIG['target_clip'], CONFIG['max_grid_size']
    )

    # Compute training thresholds
    train_thresholds = compute_training_thresholds_spatial(
        data['all_data'], data['valid_timesteps'], train_stats,
        CONFIG['target_clip'], CONFIG['max_grid_size']
    )

    # Split data
    train_years = set(range(2001, 2019))
    val_years = set(range(2019, 2022))
    test_years = set(range(2022, 2025))

    train_idx = [i for i in features['valid_indices']
                 if features['timesteps'][i][0] in train_years]
    val_idx = [i for i in features['valid_indices']
               if features['timesteps'][i][0] in val_years]
    test_idx = [i for i in features['valid_indices']
                if features['timesteps'][i][0] in test_years]

    print(f"\nData splits:")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    print(f"  Test: {len(test_idx)} samples")

    # Create CNN datasets
    train_dataset = DroughtCNNDataset(
        features['features'], features['ewbi_targets'], features['pgsi_targets'],
        features['ewbi_masks'], features['pgsi_masks'], train_idx, CONFIG['lookback']
    )
    val_dataset = DroughtCNNDataset(
        features['features'], features['ewbi_targets'], features['pgsi_targets'],
        features['ewbi_masks'], features['pgsi_masks'], val_idx, CONFIG['lookback']
    )
    test_dataset = DroughtCNNDataset(
        features['features'], features['ewbi_targets'], features['pgsi_targets'],
        features['ewbi_masks'], features['pgsi_masks'], test_idx, CONFIG['lookback']
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Initialize CNN model
    n_channels = features['features'].shape[-1]
    model = SpatialTemporalCNN(
        input_channels=n_channels,
        hidden_dim=CONFIG['hidden_dim'],
        num_cnn_layers=CONFIG['num_cnn_layers'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout'],
        use_attention=CONFIG['use_attention']
    )

    print(f"\nCNN Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Train model
    model, history = train_cnn_model(model, train_loader, val_loader, CONFIG)

    # Compute calibration
    calibration = compute_cnn_calibration(model, val_loader, CONFIG['device'])
    print(f"\nCalibration coefficients:")
    print(f"  EWBI: a={calibration['ewbi'][0]:.3f}, b={calibration['ewbi'][1]:.3f}")
    print(f"  PGSI: a={calibration['pgsi'][0]:.3f}, b={calibration['pgsi'][1]:.3f}")

    # Evaluate on test set
    results = evaluate_cnn_model(
        model, test_loader, CONFIG['device'], calibration,
        train_thresholds, features['timesteps'], test_idx, climatology
    )

    # Add thresholds to results for plotting
    results['ewbi_5th'] = train_thresholds[0]
    results['pgsi_5th'] = train_thresholds[1]
    results['joint_5th'] = train_thresholds[2]

    # Create visualizations
    create_comprehensive_plots(results, history, land_type)

    # Save results
    save_results(results, history, land_type)

    # Cleanup
    gpu_cleanup()

    return results['metrics']


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for CNN drought forecasting."""
    print("=" * 80)
    print("CALIFORNIA DROUGHT CNN FORECASTER")
    print("=" * 80)

    all_metrics = {}

    # Process each land type
    for land_type in LAND_TYPES:
        try:
            metrics = process_land_type_cnn(land_type)
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
        summary_path = os.path.join('results', 'cnn_summary_metrics.csv')
        summary_df.to_csv(summary_path)
        print(f"\nSaved CNN summary metrics to {summary_path}")

        print("\n" + "=" * 80)
        print("CNN SUMMARY RESULTS")
        print("=" * 80)
        print(summary_df)

    print("\nCNN Processing complete!")


if __name__ == "__main__":
    main()
    ewbi_targets[t_idx, r, c] = np.clip(np.mean(ewbi_components), -clip, clip)
    ewbi_masks[t_idx, r, c] = 1.0

    # PGSI components
pgsi_components = []
for var in ['GOSIF', 'GPP', 'NIRv', 'NDVI', 'GNDVI']:
    if var in month_data:
        data = month_data[var]
        if data.shape != (H, W):
            data = regrid_to_reference(data, (H, W))
        val = data[r, c]
        if np.isfinite(val):
            z_val = robust_z_score_normalize(
                np.array([val]), var, month, train_stats, clip
            )[0]
            pgsi_components.append(z_val)

if len(pgsi_components) >= min_v:
    pgsi_targets[t_idx, r, c] = np.clip(np.mean(pgsi_components), -clip, clip)
    pgsi_masks[t_idx, r, c] = 1.0

print(f"  Grid shape: {all_features.shape}")
print(f"  Total valid EWBI pixels: {np.sum(ewbi_masks):.0f}")
print(f"  Total valid PGSI pixels: {np.sum(pgsi_masks):.0f}")

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


def compute_training_thresholds_spatial(all_data: Dict, valid_timesteps: List,
                                        train_stats: Dict, clip: float = 4.0,
                                        target_size: Tuple[int, int] = (64, 64)) -> Tuple[float, float, float]:
    """Compute 5th percentile thresholds from TRAINING years using spatial grids."""
    print("Computing training thresholds (2001-2018)...")

    H, W = target_size
    ewbi_values = []
    pgsi_values = []
    joint_values = []

    min_e = CONFIG.get("min_ewbi_drivers", 3)
    min_v = CONFIG.get("min_pgsi_vars", 3)

    for year, month in valid_timesteps:
        if year > 2018:
            continue

        key = f"{year}_{month:02d}"
        if key not in all_data:
            continue
        md = all_data[key]

        month_ewbi_vals = []
        month_pgsi_vals = []

        for r in range(H):
            for c in range(W):
                # EWBI components
                ewbi_components = []
                for var, sign in [('SPEI', 1), ('TWS', 1), ('VPD', -1), ('LST', -1)]:
                    if var in md:
                        data = md[var]
                        if data.shape != (H, W):
                            data = regrid_to_reference(data, (H, W))
                        val = data[r, c]
                        if np.isfinite(val):
                            z = robust_z_score_normalize(
                                np.array([val]), var, month, train_stats, clip
                            )[0]
                            ewbi_components.append(sign * z)

                if