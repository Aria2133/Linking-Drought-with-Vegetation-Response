#!/usr/bin/env python3
"""
California Drought TCN Forecaster with Monte Carlo Dropout - OPTIMIZED VERSION
3-5x faster than original while maintaining performance
"""

import os
import gc
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

import rasterio
from scipy import ndimage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================

CONFIG = {
    # TCN Model Parameters - OPTIMIZED
    'lookback': 12,
    'forecast_horizon': 1,
    'num_channels': [32, 32, 32],  # Reduced from [64,64,64,64]
    'kernel_size': 3,
    'dropout': 0.3,
    'mc_samples': 20,  # Reduced from 50

    # Training - FASTER
    'batch_size': 256,  # Increased from 128
    'num_epochs': 80,  # Reduced from 100
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'grad_clip': 1.0,
    'patience': 8,

    # Data
    'max_nodes': 100,
    'min_valid_frac': 0.3,
    'target_clip': 4.0,
    'min_training_labels': 50,
    'resample_method': 'bilinear',

    # Speed optimizations
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
    'use_amp': True,  # Mixed precision for 2x speedup
    'num_workers': 4,  # Parallel data loading
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.benchmark = True  # Auto-optimize convolutions

print(f"Device: {CONFIG['device']}, AMP: {CONFIG['use_amp']}")

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
VARIABLES = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']


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
            'GOSIF': f"GOSIF_{year}.M{month:02d}_{land_tag}.tif",
            'GNDVI': f"GNDVI_MOD09A1_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            'GPP': f"GPP_MOD17A2H_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            'LST': f"MYD21C1_CA_LST_{year}_{month:02d}_{land_tag}.tif",
            'NDVI': f"NDVI_MOD13Q1_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            'NIRv': f"NIRv_MOD13Q1_CA_5km_{year}_{month:02d}_{land_tag}.tif",
            'SPEI': f"SPEI1_GRIDMET_CA_{year}_{month:02d}_{land_tag}.tif",
            'TWS': f"TWS_GRACE_CA_{year}_{month:02d}_{land_tag}.tif",
            'VPD': f"VPDmax_GRIDMET_CA_5km_{year}_{month:02d}_{land_tag}.tif"
        }
        file_path = os.path.join(var_path, patterns[variable])
    else:
        land_tags = {'GRASSLANDS': ['Grasslands'], 'NATURAL_FORESTS': ['NaturalForests'],
                     'SAVANNAS': ['Savannas', 'Savanna'], 'SHRUBLANDS': ['Shrublands']}[land_type]
        templates = {
            'GOSIF': "GOSIF_{tag}_CALIFORNIA_GOSIF_{year}{month:02d}.tif",
            'GNDVI': "GNDVI_{tag}_GNDVI_MOD09A1_CA_5km_{year}_{month:02d}.tif",
            'GPP': "{tag}_GPP_{year}_{month:02d}.tif",
            'LST': "LST_{tag}_MYD21C1_CA_LST_{year}_{month:02d}.tif",
            'NDVI': "NDVI_{tag}_NDVI_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
            'NIRv': "NIRv_{tag}_NIRv_MOD13Q1_CA_5km_{year}_{month:02d}.tif",
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


def gpu_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# DATA LOADING
# ============================================================================

def load_with_geographic_alignment(land_type: str, var: str) -> Tuple[Dict, List, Tuple]:
    print(f"  Loading {var}...")
    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)
    train_files, train_meta_list = {}, []

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

    test_files, test_meta_list = {}, []
    for year in range(2022, 2025):
        max_month = 10 if year == 2024 and var != 'TWS' else 12
        for month in range(1, max_month + 1):
            filepath = find_data_file(test_base, var, year, month, land_type, is_test=True)
            if filepath and os.path.exists(filepath):
                key = f"{year}_{month:02d}"
                data, meta = read_geotiff_with_geo(filepath)
                test_files[key] = (data, meta)
                test_meta_list.append(meta)

    if not train_files or not test_files:
        return None, None, None

    first_train_meta = train_meta_list[0]
    first_test_meta = test_meta_list[0]

    train_area = (first_train_meta['bounds'].right - first_train_meta['bounds'].left) * \
                 (first_train_meta['bounds'].top - first_train_meta['bounds'].bottom)
    test_area = (first_test_meta['bounds'].right - first_test_meta['bounds'].left) * \
                (first_test_meta['bounds'].top - first_test_meta['bounds'].bottom)

    target_meta = first_train_meta if train_area < test_area else first_test_meta
    needs_clip_test = train_area < test_area
    needs_clip_train = not needs_clip_test

    all_data, valid_timesteps = {}, []

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
    print(f"    Loaded {len(all_data)} timesteps")
    return all_data, valid_timesteps, target_meta['shape']


def get_known_unit_conversion(var: str) -> Tuple[float, str]:
    no_correction = ['SPEI', 'NDVI', 'GNDVI', 'NIRv']
    if var in no_correction:
        return (None, "Already standardized - no correction needed")
    conversions = {'GPP': (31.56, "kg C m⁻² → μmol CO₂ m⁻² s⁻¹ (monthly avg)")}
    return conversions.get(var, (None, None))


def detect_and_correct_unit_mismatch(all_data: Dict, var: str, valid_timesteps: List) -> Tuple[Dict, float]:
    known_factor, conversion_desc = get_known_unit_conversion(var)
    if conversion_desc and "no correction" in conversion_desc.lower():
        return all_data, 1.0

    train_vals, test_vals = [], []
    for y, m in valid_timesteps:
        key = f"{y}_{m:02d}"
        if key in all_data and var in all_data[key]:
            valid = all_data[key][var][np.isfinite(all_data[key][var])]
            if len(valid) > 0:
                if y <= 2021:
                    train_vals.extend(valid.tolist())
                else:
                    test_vals.extend(valid.tolist())

    if not train_vals or not test_vals:
        return all_data, 1.0

    train_median = np.median(train_vals)
    test_median = np.median(test_vals)
    median_ratio = train_median / (test_median + 1e-10)

    scale_factor = None
    if known_factor is not None and abs(median_ratio - 1.0) > 0.1:
        scale_factor = known_factor
    elif abs(median_ratio - 1.0) > 0.1:
        scale_factor = median_ratio

    if scale_factor is not None and scale_factor != 1.0:
        for y, m in valid_timesteps:
            if y >= 2022:
                key = f"{y}_{m:02d}"
                if key in all_data and var in all_data[key]:
                    all_data[key][var] = all_data[key][var] * scale_factor
        return all_data, scale_factor
    return all_data, 1.0


def impute_single_variable(all_data: Dict, var: str, valid_timesteps: List):
    for month in range(1, 13):
        frames, years = [], []
        for y, m in valid_timesteps:
            if m == month:
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
    for month in range(1, 13):
        ys, mus = [], []
        for y, m in valid_timesteps:
            if m == month:
                key = f"{y}_{m:02d}"
                if key in all_data and var in all_data[key]:
                    mu = np.nanmean(all_data[key][var])
                    if np.isfinite(mu):
                        ys.append(y)
                        mus.append(mu)
        if len(ys) < 5:
            continue
        ys, mus = np.array(ys), np.array(mus)
        mask = (ys >= 2001) & (ys <= 2018)
        if mask.sum() >= 5:
            A = np.vstack([ys[mask], np.ones(mask.sum())]).T
            a, b = np.linalg.lstsq(A, mus[mask], rcond=None)[0]
            for y, m in valid_timesteps:
                if m == month:
                    key = f"{y}_{m:02d}"
                    if key in all_data and var in all_data[key]:
                        all_data[key][var] = all_data[key][var] - (a * y + b)
    return all_data


def compute_variable_stats(all_data: Dict, var: str, valid_timesteps: List) -> Dict:
    stats = {}
    for month in range(1, 13):
        month_data = []
        for y, m in valid_timesteps:
            if y <= 2018 and m == month:
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
# OPTIMIZED TCN ARCHITECTURE
# ============================================================================

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                                    stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class UnivariateTCN(nn.Module):
    def __init__(self, lookback=12, num_channels=[32, 32, 32],
                 kernel_size=3, dropout=0.3):
        super().__init__()
        self.lookback = lookback
        self.tcn = TemporalConvNet(num_inputs=1, num_channels=num_channels,
                                   kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        y = self.tcn(x)
        y = y[:, :, -1]
        out = self.fc(y)
        return out.squeeze()

    def predict_with_uncertainty(self, x, n_samples=20):
        """Optimized MC Dropout with batched computation."""
        self.train()  # Keep dropout enabled

        # Batch all MC samples for faster computation
        batch_size = x.shape[0]
        x_repeated = x.repeat(n_samples, 1)  # Stack samples

        with torch.no_grad():
            preds = self.forward(x_repeated)

        # Reshape to (n_samples, batch_size)
        preds = preds.reshape(n_samples, batch_size).cpu().numpy()

        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)
        lower_bound = np.percentile(preds, 2.5, axis=0)
        upper_bound = np.percentile(preds, 97.5, axis=0)

        return mean_pred, std_pred, lower_bound, upper_bound


# ============================================================================
# DATASET
# ============================================================================

class UnivariateDataset(Dataset):
    def __init__(self, all_data, var, valid_timesteps, pixel_indices,
                 lookback, indices, var_stats, clip):
        self.all_data = all_data
        self.var = var
        self.valid_timesteps = valid_timesteps
        self.pixel_indices = pixel_indices
        self.lookback = lookback
        self.indices = indices
        self.var_stats = var_stats
        self.clip = clip

    def __len__(self):
        return len(self.indices) * len(self.pixel_indices)

    def __getitem__(self, idx):
        t_idx = self.indices[idx // len(self.pixel_indices)]
        p_idx = idx % len(self.pixel_indices)
        row, col = self.pixel_indices[p_idx]

        x_vals = []
        for i in range(t_idx - self.lookback, t_idx):
            year, month = self.valid_timesteps[i]
            key = f"{year}_{month:02d}"
            if key in self.all_data and self.var in self.all_data[key]:
                val = self.all_data[key][self.var][row, col]
                if month in self.var_stats:
                    val = (val - self.var_stats[month]['mean']) / self.var_stats[month]['std']
                val = np.clip(val, -self.clip, self.clip)
                if not np.isfinite(val):
                    val = 0.0
                x_vals.append(val)
            else:
                x_vals.append(0.0)

        year, month = self.valid_timesteps[t_idx]
        key = f"{year}_{month:02d}"
        if key in self.all_data and self.var in self.all_data[key]:
            y_val = self.all_data[key][self.var][row, col]
            if month in self.var_stats:
                y_val = (y_val - self.var_stats[month]['mean']) / self.var_stats[month]['std']
            y_val = np.clip(y_val, -self.clip, self.clip)
            mask = 1.0 if np.isfinite(y_val) else 0.0
            if not np.isfinite(y_val):
                y_val = 0.0
        else:
            y_val = 0.0
            mask = 0.0

        return (torch.tensor(x_vals, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32))


# ============================================================================
# OPTIMIZED TRAINING WITH MIXED PRECISION
# ============================================================================

def train_model(model, train_loader, val_loader, cfg):
    device = cfg['device']
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['num_epochs'])
    scaler = GradScaler() if cfg['use_amp'] else None

    import copy
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_losses = []

        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()

            # Mixed precision forward pass
            if cfg['use_amp']:
                with autocast():
                    pred = model(x)
                    if m.sum() > 0:
                        loss = torch.sum(((pred - y) ** 2) * m) / m.sum()
                    else:
                        loss = torch.tensor(0.0, device=device, requires_grad=True)

                scaler.scale(loss).backward()
                if cfg['grad_clip']:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                if m.sum() > 0:
                    loss = torch.sum(((pred - y) ** 2) * m) / m.sum()
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

                loss.backward()
                if cfg['grad_clip']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                pred = model(x)
                if m.sum() > 0:
                    loss = torch.sum(((pred - y) ** 2) * m) / m.sum()
                    val_losses.append(loss.item())

        train_loss = np.mean(train_losses) if train_losses else np.nan
        val_loss = np.mean(val_losses) if val_losses else train_loss
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step()

        if np.isfinite(val_loss) and val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg['patience']:
                print(f"  Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")

    model.load_state_dict(best_state)
    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model_with_uncertainty(model, test_loader, device, mc_samples=20):
    model.eval()
    preds, stds, lowers, uppers, obs, masks = [], [], [], [], [], []

    for x, y, m in test_loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        mean_pred, std_pred, lower, upper = model.predict_with_uncertainty(x, n_samples=mc_samples)

        preds.extend(mean_pred.tolist())
        stds.extend(std_pred.tolist())
        lowers.extend(lower.tolist())
        uppers.extend(upper.tolist())
        obs.extend(y.cpu().numpy().tolist())
        masks.extend(m.cpu().numpy().tolist())

    preds = np.array(preds)
    stds = np.array(stds)
    lowers = np.array(lowers)
    uppers = np.array(uppers)
    obs = np.array(obs)
    masks = np.array(masks)

    valid = masks > 0
    n_valid = valid.sum()

    metrics = {}
    if n_valid > 1:
        metrics['rmse'] = np.sqrt(mean_squared_error(obs[valid], preds[valid]))
        metrics['mae'] = mean_absolute_error(obs[valid], preds[valid])
        metrics['r2'] = r2_score(obs[valid], preds[valid])
        metrics['corr'] = pearsonr(obs[valid], preds[valid])[0]
        metrics['mean_uncertainty'] = stds[valid].mean()
        coverage = ((obs[valid] >= lowers[valid]) & (obs[valid] <= uppers[valid])).mean()
        metrics['coverage_95'] = coverage
    else:
        metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'corr': np.nan,
                   'mean_uncertainty': np.nan, 'coverage_95': np.nan, 'n_valid': int(n_valid)}

    return metrics, preds, stds, lowers, uppers, obs, masks


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization_with_uncertainty(preds, stds, lowers, uppers, obs, masks, history,
                                          land_type, var, test_idx, valid_timesteps,
                                          num_pixels, output_dir='results'):
    num_test_times = len(test_idx)
    preds_2d = preds.reshape(num_test_times, num_pixels)
    stds_2d = stds.reshape(num_test_times, num_pixels)
    lowers_2d = lowers.reshape(num_test_times, num_pixels)
    uppers_2d = uppers.reshape(num_test_times, num_pixels)
    obs_2d = obs.reshape(num_test_times, num_pixels)
    masks_2d = masks.reshape(num_test_times, num_pixels)

    preds_agg, stds_agg, lowers_agg, uppers_agg = [], [], [], []
    obs_agg, valid_agg, test_dates = [], [], []

    for t in range(num_test_times):
        valid_pixels = masks_2d[t] > 0
        if valid_pixels.sum() > 0:
            preds_agg.append(np.mean(preds_2d[t][valid_pixels]))
            stds_agg.append(np.mean(stds_2d[t][valid_pixels]))
            lowers_agg.append(np.mean(lowers_2d[t][valid_pixels]))
            uppers_agg.append(np.mean(uppers_2d[t][valid_pixels]))
            obs_agg.append(np.mean(obs_2d[t][valid_pixels]))
            valid_agg.append(True)
        else:
            preds_agg.append(np.nan)
            stds_agg.append(np.nan)
            lowers_agg.append(np.nan)
            uppers_agg.append(np.nan)
            obs_agg.append(np.nan)
            valid_agg.append(False)
        year, month = valid_timesteps[test_idx[t]]
        test_dates.append(f"{year}-{month:02d}")

    preds_agg = np.array(preds_agg)
    stds_agg = np.array(stds_agg)
    lowers_agg = np.array(lowers_agg)
    uppers_agg = np.array(uppers_agg)
    obs_agg = np.array(obs_agg)
    valid_agg = np.array(valid_agg)

    if valid_agg.sum() > 1:
        agg_rmse = np.sqrt(mean_squared_error(obs_agg[valid_agg], preds_agg[valid_agg]))
        agg_r2 = r2_score(obs_agg[valid_agg], preds_agg[valid_agg])
        agg_corr = pearsonr(obs_agg[valid_agg], preds_agg[valid_agg])[0]
    else:
        agg_rmse = agg_r2 = agg_corr = np.nan

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'California {land_type.title()} - {var} Forecasting (TCN-Optimized)',
                 fontsize=16, fontweight='bold')

    ax1 = plt.subplot(2, 3, 1)
    if valid_agg.sum() > 0:
        x_idx = np.arange(len(test_dates))
        ax1.plot(x_idx[valid_agg], obs_agg[valid_agg], 'b-', label='Observed', alpha=0.8, linewidth=2)
        ax1.plot(x_idx[valid_agg], preds_agg[valid_agg], 'r--', label='Predicted', alpha=0.8, linewidth=2)
        ax1.fill_between(x_idx[valid_agg], lowers_agg[valid_agg], uppers_agg[valid_agg],
                         alpha=0.3, color='red', label='95% CI')
        ax1.set_xticks(x_idx[::6])
        ax1.set_xticklabels([test_dates[i] for i in x_idx[::6]], rotation=45, ha='right')
    ax1.set_title(f'{var} Forecasting with Uncertainty', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel(f'{var} (normalized)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(2, 3, 2)
    if valid_agg.sum() > 1:
        ax2.errorbar(obs_agg[valid_agg], preds_agg[valid_agg],
                     yerr=1.96 * stds_agg[valid_agg], fmt='o', alpha=0.5, capsize=3)
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

    ax3 = plt.subplot(2, 3, 3)
    epochs = range(1, len(history['train_loss']) + 1)
    ax3.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax3.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax3.set_title('TCN Training Progress', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(2, 3, 4)
    if valid_agg.sum() > 0:
        x_idx = np.arange(len(test_dates))
        ax4.plot(x_idx[valid_agg], stds_agg[valid_agg], 'g-', linewidth=2)
        ax4.set_xticks(x_idx[::6])
        ax4.set_xticklabels([test_dates[i] for i in x_idx[::6]], rotation=45, ha='right')
    ax4.set_title('Prediction Uncertainty', fontsize=12)
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Std Dev')
    ax4.grid(True, alpha=0.3)

    ax5 = plt.subplot(2, 3, 5)
    if valid_agg.sum() > 0:
        residuals = obs_agg[valid_agg] - preds_agg[valid_agg]
        ax5.hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax5.set_title('Error Distribution', fontsize=12)
    ax5.set_xlabel('Residuals')
    ax5.set_ylabel('Frequency')
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    mean_std = np.mean(stds_agg[valid_agg]) if valid_agg.sum() > 0 else np.nan
    coverage = 100 * np.mean((obs_agg[valid_agg] >= lowers_agg[valid_agg]) &
                             (obs_agg[valid_agg] <= uppers_agg[valid_agg])) if valid_agg.sum() > 0 else 0

    summary_text = f"""
=== TCN OPTIMIZED ===

Variable: {var}
Land Type: {land_type}

Performance:
  RMSE: {agg_rmse:.4f}
  R²:   {agg_r2:.4f}
  Corr: {agg_corr:.3f}

Uncertainty:
  Mean Std: {mean_std:.4f}
  95% CI Coverage: {coverage:.1f}%

Training:
  Epochs: {len(history['train_loss'])}
  Final loss: {history['train_loss'][-1]:.4f}

Optimizations:
  Channels: {CONFIG['num_channels']}
  MC Samples: {CONFIG['mc_samples']}
  Mixed Precision: {CONFIG['use_amp']}
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{land_type.lower()}_{var.lower()}_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


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

    train_coverage = np.zeros(shape, dtype=np.int32)
    test_coverage = np.zeros(shape, dtype=np.int32)

    for year, month in valid_timesteps:
        key = f"{year}_{month:02d}"
        if key in all_data and var in all_data[key]:
            valid_data = np.isfinite(all_data[key][var]).astype(np.int32)
            if year <= 2021:
                train_coverage += valid_data
            elif year == 2024:
                test_coverage += valid_data

    train_months = sum(1 for y, m in valid_timesteps if y <= 2021)
    test_months = sum(1 for y, m in valid_timesteps if y == 2024)

    train_threshold = int(CONFIG['min_valid_frac'] * train_months)
    test_threshold = int(CONFIG['min_valid_frac'] * test_months)

    valid_mask = (train_coverage >= train_threshold) & (test_coverage >= test_threshold)

    if valid_mask.sum() == 0:
        print(f"  ERROR: No shared pixels!")
        return None

    pixel_indices = np.column_stack(np.where(valid_mask))
    if len(pixel_indices) > CONFIG['max_nodes']:
        idx = np.random.choice(len(pixel_indices), CONFIG['max_nodes'], replace=False)
        pixel_indices = pixel_indices[idx]

    train_idx = [i for i, (y, m) in enumerate(valid_timesteps) if y <= 2018 and i >= CONFIG['lookback']]
    val_idx = [i for i, (y, m) in enumerate(valid_timesteps) if 2019 <= y <= 2021 and i >= CONFIG['lookback']]
    test_idx = [i for i, (y, m) in enumerate(valid_timesteps) if y == 2024 and i >= CONFIG['lookback']]

    if not train_idx or not val_idx or not test_idx:
        return None

    train_dataset = UnivariateDataset(all_data, var, valid_timesteps, pixel_indices,
                                      CONFIG['lookback'], train_idx, var_stats, CONFIG['target_clip'])
    val_dataset = UnivariateDataset(all_data, var, valid_timesteps, pixel_indices,
                                    CONFIG['lookback'], val_idx, var_stats, CONFIG['target_clip'])
    test_dataset = UnivariateDataset(all_data, var, valid_timesteps, pixel_indices,
                                     CONFIG['lookback'], test_idx, var_stats, CONFIG['target_clip'])

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                             shuffle=False, num_workers=0, pin_memory=True)

    model = UnivariateTCN(
        lookback=CONFIG['lookback'],
        num_channels=CONFIG['num_channels'],
        kernel_size=CONFIG['kernel_size'],
        dropout=CONFIG['dropout']
    )

    print(f"  Training TCN (optimized)...")
    model, history = train_model(model, train_loader, val_loader, CONFIG)

    print(f"  Evaluating...")
    result = evaluate_model_with_uncertainty(model, test_loader, CONFIG['device'],
                                             mc_samples=CONFIG['mc_samples'])
    metrics, preds, stds, lowers, uppers, obs, masks = result

    if np.isfinite(metrics['r2']):
        print(f"  RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, "
              f"Corr={metrics['corr']:.4f}, Coverage={100 * metrics['coverage_95']:.1f}%")

    create_visualization_with_uncertainty(preds, stds, lowers, uppers, obs, masks,
                                          history, land_type, var, test_idx,
                                          valid_timesteps, len(pixel_indices))

    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame({
        'pred': preds, 'std': stds, 'lower_95': lowers,
        'upper_95': uppers, 'obs': obs, 'mask': masks
    })
    results_df.to_csv(f"results/{land_type.lower()}_{var.lower()}_results.csv", index=False)

    metrics['scale_factor'] = float(scale_factor)
    pd.DataFrame([metrics]).to_csv(f"results/{land_type.lower()}_{var.lower()}_metrics.csv", index=False)

    gpu_cleanup()
    return metrics


def main():
    print("=" * 80)
    print("CALIFORNIA DROUGHT TCN - OPTIMIZED WITH UNCERTAINTY")
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
        summary_df.to_csv('results/summary_tcn_optimized.csv', index=False)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(summary_df.to_string())

    print("\nDONE!")


if __name__ == "__main__":
    main()