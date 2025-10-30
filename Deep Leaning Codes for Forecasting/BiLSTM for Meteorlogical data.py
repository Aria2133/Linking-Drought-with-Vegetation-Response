#!/usr/bin/env python3
"""
California Drought BiLSTM Forecaster - GPU-Optimized Implementation
Training: 2001-2021, Validation: 2022-2023, Testing: 2024 only
Variables: SPEI, TWS, VPD, LST

GPU-OPTIMIZED VERSION:
Based on 2020-2025 literature achieving R² = 0.85-0.99 for SPEI using BiLSTM.
- Full 200 LSTM units per layer (literature standard)
- 500 epochs with early stopping
- Bidirectional LSTM for SPEI and TWS
- GPU training: ~2-5 minutes per variable

Requires: CUDA-compatible GPU with TensorFlow GPU support
"""

import os
import gc
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# GPU configuration - ensures TensorFlow uses GPU if available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to prevent TF from allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Optional: Limit GPU memory usage (uncomment if needed)
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)]  # 4GB limit
        # )

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Using GPU: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print(f"GPU Device: {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("WARNING: No GPU found. Training will be VERY SLOW on CPU!")
    print("For GPU support, install: pip install tensorflow[and-cuda]")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'max_nodes': 150,  # Full dataset for GPU training
    'min_valid_frac': 0.25,
    'target_clip': 4.0,
    'min_training_labels': 50,
    'resample_method': 'bilinear',
    'seed': 42,

    # Variable-specific LSTM configurations based on literature
    'var_configs': {
        'SPEI': {
            'lookback': 24,  # 24-36 months optimal for SPEI
            'lstm_units': 200,  # Literature standard for R² ≥ 0.85
            'lstm_layers': 2,
            'dropout': 0.2,
            'batch_size': 64,  # Optimal for GPU
            'epochs': 500,  # Full training with early stopping
            'learning_rate': 0.001,
            'patience': 10,
            'bidirectional': True,
        },
        'TWS': {
            'lookback': 18,
            'lstm_units': 150,
            'lstm_layers': 2,
            'dropout': 0.2,
            'batch_size': 64,
            'epochs': 300,
            'learning_rate': 0.001,
            'patience': 10,
            'bidirectional': True,
        },
        'VPD': {
            'lookback': 12,
            'lstm_units': 128,
            'lstm_layers': 2,
            'dropout': 0.2,
            'batch_size': 64,
            'epochs': 300,
            'learning_rate': 0.001,
            'patience': 10,
            'bidirectional': False,
        },
        'LST': {
            'lookback': 12,
            'lstm_units': 128,
            'lstm_layers': 2,
            'dropout': 0.2,
            'batch_size': 64,
            'epochs': 300,
            'learning_rate': 0.001,
            'patience': 10,
            'bidirectional': False,
        }
    }
}

np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
VARIABLES = ['SPEI', 'TWS', 'VPD', 'LST']


# ============================================================================
# PATH HELPERS (Same as XGBoost version)
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


# ============================================================================
# DATA LOADING (Same as XGBoost version)
# ============================================================================

def load_with_geographic_alignment(land_type: str, var: str) -> Tuple[Dict, List, Tuple]:
    """Load data with proper geographic clipping and alignment."""
    print(f"  Loading {var}...")

    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)

    train_files = {}
    train_meta_list = []

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

    train_area = (first_train_meta['bounds'].right - first_train_meta['bounds'].left) * \
                 (first_train_meta['bounds'].top - first_train_meta['bounds'].bottom)
    test_area = (first_test_meta['bounds'].right - first_test_meta['bounds'].left) * \
                (first_test_meta['bounds'].top - first_test_meta['bounds'].bottom)

    if train_area < test_area:
        target_meta = first_train_meta
        needs_clip_test = True
        needs_clip_train = False
    else:
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
# PREPROCESSING (Adapted for LSTM)
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
        mask = (ys >= 2001) & (ys <= 2021)

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


# ============================================================================
# LSTM DATA PREPARATION
# ============================================================================

def prepare_lstm_sequences(all_data, var, valid_timesteps, pixel_indices,
                           lookback, indices, scaler):
    """
    Prepare sequences for LSTM: (samples, timesteps, features=1)
    CRITICAL: Use scaler fitted on training data only!
    """
    X_list = []
    y_list = []
    mask_list = []

    for t_idx in indices:
        for row, col in pixel_indices:
            # Extract lookback sequence
            sequence = []
            for i in range(t_idx - lookback, t_idx):
                year, month = valid_timesteps[i]
                key = f"{year}_{month:02d}"
                if key in all_data and var in all_data[key]:
                    val = all_data[key][var][row, col]
                    if not np.isfinite(val):
                        val = 0.0
                    sequence.append(val)
                else:
                    sequence.append(0.0)

            # Target value
            year, month = valid_timesteps[t_idx]
            key = f"{year}_{month:02d}"
            if key in all_data and var in all_data[key]:
                y_val = all_data[key][var][row, col]
                mask = 1.0 if np.isfinite(y_val) else 0.0
                if not np.isfinite(y_val):
                    y_val = 0.0
            else:
                y_val = 0.0
                mask = 0.0

            X_list.append(sequence)
            y_list.append(y_val)
            mask_list.append(mask)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    mask = np.array(mask_list, dtype=np.float32)

    # Apply scaling
    n_samples = X.shape[0]
    X_reshaped = X.reshape(-1, 1)  # Flatten for scaling
    X_scaled = scaler.transform(X_reshaped)
    X_scaled = X_scaled.reshape(n_samples, lookback, 1)  # Back to 3D

    y_scaled = scaler.transform(y.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled, mask


# ============================================================================
# LSTM MODEL BUILDING
# ============================================================================

def build_lstm_model(var_config, lookback):
    """Build BiLSTM or LSTM model based on literature recommendations."""
    model = Sequential()

    if var_config['bidirectional']:
        # First Bidirectional LSTM layer
        model.add(Bidirectional(
            LSTM(var_config['lstm_units'], activation='relu', return_sequences=True),
            input_shape=(lookback, 1)
        ))
        model.add(Dropout(var_config['dropout']))

        # Second LSTM layer
        model.add(LSTM(var_config['lstm_units'], activation='relu'))
        model.add(Dropout(var_config['dropout']))
    else:
        # Standard LSTM layers
        model.add(LSTM(var_config['lstm_units'], activation='relu',
                       return_sequences=True, input_shape=(lookback, 1)))
        model.add(Dropout(var_config['dropout']))

        model.add(LSTM(var_config['lstm_units'], activation='relu'))
        model.add(Dropout(var_config['dropout']))

    # Output layers
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=var_config['learning_rate'])
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_lstm_model(model, X_train, y_train, mask_train, X_val, y_val, mask_val, var_config):
    """Train LSTM model with early stopping."""
    # Filter valid samples
    train_valid = mask_train > 0
    val_valid = mask_val > 0

    if train_valid.sum() == 0 or val_valid.sum() == 0:
        print("    ERROR: No valid training or validation samples")
        return None

    X_train_valid = X_train[train_valid]
    y_train_valid = y_train[train_valid]
    X_val_valid = X_val[val_valid]
    y_val_valid = y_val[val_valid]

    print(f"  Training samples: {len(X_train_valid)}, Validation samples: {len(X_val_valid)}")

    # Early stopping
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=var_config['patience'],
        restore_best_weights=True,
        verbose=1
    )

    # Train with progress bar
    history = model.fit(
        X_train_valid, y_train_valid,
        epochs=var_config['epochs'],
        batch_size=var_config['batch_size'],
        validation_data=(X_val_valid, y_val_valid),
        callbacks=[early_stop],
        verbose=1  # Show full progress bar on GPU
    )

    print(f"  Training completed at epoch {len(history.history['loss'])}")
    print(f"  Best val_loss: {min(history.history['val_loss']):.4f}")

    return history


def evaluate_lstm_model(model, X_test, y_test, mask_test):
    """Evaluate LSTM model."""
    valid = mask_test > 0
    n_valid = valid.sum()

    print(f"    Test samples: {len(mask_test)}, Valid: {n_valid}")

    if n_valid < 2:
        print(f"    WARNING: Only {n_valid} valid test samples!")
        return {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'corr': np.nan, 'n_valid': int(n_valid)}, None

    # Predict
    preds = model.predict(X_test[valid], verbose=0).flatten()
    obs = y_test[valid]

    # Metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(obs, preds)),
        'mae': mean_absolute_error(obs, preds),
        'r2': r2_score(obs, preds),
        'corr': pearsonr(obs, preds)[0],
        'n_valid': int(n_valid)
    }

    # Full predictions array for visualization
    preds_full = np.zeros_like(y_test)
    preds_full[valid] = preds

    return metrics, preds_full


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_lstm_visualization(preds, obs, masks, history, land_type, var,
                              test_idx, valid_timesteps, num_pixels, output_dir='results'):
    """Create visualization for LSTM results."""
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
    fig.suptitle(f'California {land_type.title()} - {var} Forecasting (BiLSTM) - Test: 2024 Only',
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
    if history is not None:
        ax3.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax3.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax3.set_title('BiLSTM Training Progress', fontsize=12)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss (MSE)')
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

    epochs_trained = len(history.history['loss']) if history else 0
    final_train_loss = history.history['loss'][-1] if history else 0
    final_val_loss = history.history['val_loss'][-1] if history else 0

    summary_text = f"""
=== BiLSTM METRICS ===

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
  Epochs: {epochs_trained}
  Final train loss: {final_train_loss:.4f}
  Final val loss: {final_val_loss:.4f}

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
    output_path = os.path.join(output_dir, f'{land_type.lower()}_{var.lower()}_lstm_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization to {output_path}")


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_single_variable_lstm(land_type: str, var: str) -> Optional[Dict]:
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {land_type} - {var} with BiLSTM")
    print(f"{'=' * 80}")

    var_config = CONFIG['var_configs'][var]
    lookback = var_config['lookback']

    print(f"  Using lookback={lookback}, BiLSTM={var_config['bidirectional']}")

    # Load data
    all_data, valid_timesteps, shape = load_with_geographic_alignment(land_type, var)

    if all_data is None:
        return None

    # Preprocessing
    all_data = impute_single_variable(all_data, var, valid_timesteps)
    all_data = detrend_single_variable(all_data, var, valid_timesteps)

    print(f"  Finding valid pixels...")

    # Pixel selection
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

    train_valid = train_coverage >= train_threshold
    val_valid = val_coverage >= val_threshold
    test_valid = test_coverage >= test_threshold
    valid_mask = train_valid & val_valid & test_valid

    print(f"    Shared valid pixels: {valid_mask.sum()} ({100 * valid_mask.sum() / valid_mask.size:.1f}% of grid)")

    if valid_mask.sum() == 0:
        print(f"  ERROR: No shared pixels!")
        return None

    pixel_indices = np.column_stack(np.where(valid_mask))
    if len(pixel_indices) > CONFIG['max_nodes']:
        idx = np.random.choice(len(pixel_indices), CONFIG['max_nodes'], replace=False)
        pixel_indices = pixel_indices[idx]

    print(f"  Selected {len(pixel_indices)} shared pixels")

    # Create indices
    train_idx = [i for i, (y, m) in enumerate(valid_timesteps)
                 if y <= 2021 and i >= lookback]
    val_idx = [i for i, (y, m) in enumerate(valid_timesteps)
               if 2022 <= y <= 2023 and i >= lookback]
    test_idx = [i for i, (y, m) in enumerate(valid_timesteps)
                if y == 2024 and i >= lookback]

    if not train_idx or not val_idx or not test_idx:
        print(f"  ERROR: Insufficient data splits")
        return None

    print(f"  Data split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # CRITICAL: Fit scaler on TRAINING data only (literature requirement)
    print(f"  Fitting scaler on training data...")
    train_vals = []
    for year, month in valid_timesteps:
        if year > 2021:
            continue
        key = f"{year}_{month:02d}"
        if key in all_data and var in all_data[key]:
            valid_pixels = all_data[key][var][np.isfinite(all_data[key][var])]
            if len(valid_pixels) > 0:
                train_vals.extend(valid_pixels.tolist())

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array(train_vals).reshape(-1, 1))

    # Prepare sequences
    print(f"  Preparing LSTM sequences...")
    X_train, y_train, mask_train = prepare_lstm_sequences(
        all_data, var, valid_timesteps, pixel_indices, lookback, train_idx, scaler)

    X_val, y_val, mask_val = prepare_lstm_sequences(
        all_data, var, valid_timesteps, pixel_indices, lookback, val_idx, scaler)

    X_test, y_test, mask_test = prepare_lstm_sequences(
        all_data, var, valid_timesteps, pixel_indices, lookback, test_idx, scaler)

    # Build and train model
    print(f"  Building BiLSTM model...")
    model = build_lstm_model(var_config, lookback)
    model.summary()

    print(f"  Training BiLSTM...")
    history = train_lstm_model(model, X_train, y_train, mask_train,
                               X_val, y_val, mask_val, var_config)

    if history is None:
        return None

    # Evaluate
    print(f"  Evaluating...")
    metrics, preds = evaluate_lstm_model(model, X_test, y_test, mask_test)

    if np.isfinite(metrics['r2']):
        print(f"  Results: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, Corr={metrics['corr']:.4f}")
    else:
        print(f"  Results: Insufficient valid test data")

    # Create visualization
    create_lstm_visualization(preds, y_test, mask_test, history, land_type, var,
                              test_idx, valid_timesteps, len(pixel_indices))

    # Save results
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        'pred': preds,
        'obs': y_test,
        'mask': mask_test
    })
    results_df.to_csv(f"results/{land_type.lower()}_{var.lower()}_lstm_results.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"results/{land_type.lower()}_{var.lower()}_lstm_metrics.csv", index=False)

    # Save model
    model.save(f"results/{land_type.lower()}_{var.lower()}_lstm_model.keras")

    # Cleanup
    tf.keras.backend.clear_session()
    gc.collect()

    return metrics


def main():
    print("=" * 80)
    print("CALIFORNIA DROUGHT BiLSTM FORECASTER - 2024 TEST ONLY")
    print("Variables: SPEI, TWS, VPD, LST")
    print("Train: 2001-2021 | Validation: 2022-2023 | Test: 2024")
    print("Literature-based implementation for R² ≥ 0.85 SPEI performance")
    print("=" * 80)

    all_metrics = {}

    for land_type in LAND_TYPES:
        all_metrics[land_type] = {}
        for var in VARIABLES:
            try:
                metrics = process_single_variable_lstm(land_type, var)
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
        summary_df.to_csv('results/summary_lstm_2024.csv', index=False)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string())x

    print("\nDONE!")


if __name__ == "__main__":
    main()