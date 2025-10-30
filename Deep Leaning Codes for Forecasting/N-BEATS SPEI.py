#!/usr/bin/env python3
"""
California Drought N-BEATS Forecaster - SPEI Only Version
Focused on SPEI variable across all land types for architecture experimentation
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
import rasterio
from scipy import ndimage
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans

# ============================================================================
# CONFIGURATION - MODIFY THESE TO EXPERIMENT WITH ARCHITECTURE
# ============================================================================

CONFIG = {
    # N-BEATS Model Parameters - EXPERIMENT WITH THESE
    'lookback':6,  # Try: 6, 12, 18, 24
    'forecast_horizon': 1,
    'stack_types': ['trend', 'seasonality'],  # Try different combinations
    'nb_blocks_per_stack': 3,  # Try: 1, 2, 3, 4
    'thetas_dim': [4, 8],  # IMPORTANT: First value must be >= lookback for generic basis
    'hidden_layer_units': 64,  # Try: 32, 64, 128, 256
    'nb_harmonics': 3,  # Try: 2, 4, 6, 8

    # Training
    'batch_size': 32,
    'num_epochs': 40,
    'lr': 0.001,
    'weight_decay': 0.001,
    'dropout': 0.4,
    'grad_clip': 1.0,
    'patience': 10,
    'loss': 'huber',
    'ema_decay': 0.995,
    'use_ema': True,

    # Data
    'max_nodes': 50,
    'min_valid_frac': 0.3,
    'target_clip': 3.0,
    'min_training_labels': 50,
    'resample_method': 'cubic',

    # Device/seed
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seed': 42,
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print(f"Device: {CONFIG['device']}")

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
VARIABLE = 'SPEI'


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


def find_data_file(base_path: str, year: int, month: int, land_type: str, is_test: bool = False) -> Optional[str]:
    var_path = os.path.join(base_path, VARIABLE)
    if not os.path.exists(var_path):
        return None

    if is_test:
        land_tag = {'GRASSLANDS': 'Grasslands', 'NATURAL_FORESTS': 'NaturalForests',
                    'SAVANNAS': 'Savannas', 'SHRUBLANDS': 'Shrublands'}[land_type]
        pattern = f"SPEI1_GRIDMET_CA_{year}_{month:02d}_{land_tag}.tif"
        file_path = os.path.join(var_path, pattern)
    else:
        land_tags = {'GRASSLANDS': ['Grasslands'], 'NATURAL_FORESTS': ['NaturalForests'],
                     'SAVANNAS': ['Savannas', 'Savanna'], 'SHRUBLANDS': ['Shrublands']}[land_type]
        template = "SPEI1_GRIDMET_CA_{year}_{month:02d}_{tag}.tif"
        for land_tag in land_tags:
            pattern = template.format(tag=land_tag, year=year, month=month)
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


def resample_array(data: np.ndarray, source_shape: Tuple, target_shape: Tuple, method: str = 'bilinear') -> np.ndarray:
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

def load_spei_data(land_type: str) -> Tuple[Dict, List, Tuple]:
    print(f"  Loading SPEI for {land_type}...")

    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)

    train_files = {}
    train_meta_list = []

    for year in range(2001, 2022):
        for month in range(1, 13):
            filepath = find_data_file(train_base, year, month, land_type, is_test=False)
            if filepath and os.path.exists(filepath):
                key = f"{year}_{month:02d}"
                data, meta = read_geotiff_with_geo(filepath)
                train_files[key] = (data, meta)
                train_meta_list.append(meta)

    test_files = {}
    test_meta_list = []

    for year in range(2022, 2025):
        for month in range(1, 13):
            filepath = find_data_file(test_base, year, month, land_type, is_test=True)
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

        all_data[key] = {VARIABLE: data}
        year, month = int(key.split('_')[0]), int(key.split('_')[1])
        valid_timesteps.append((year, month))

    for key, (data, meta) in test_files.items():
        if needs_clip_test and meta['shape'] != target_meta['shape']:
            data = clip_and_resample_to_target(data, meta, target_meta, CONFIG['resample_method'])
        elif meta['shape'] != target_meta['shape']:
            data = resample_array(data, meta['shape'], target_meta['shape'], CONFIG['resample_method'])

        all_data[key] = {VARIABLE: data}
        year, month = int(key.split('_')[0]), int(key.split('_')[1])
        valid_timesteps.append((year, month))

    valid_timesteps.sort()
    print(f"    Loaded {len(all_data)} timesteps, shape: {target_meta['shape']}")

    return all_data, valid_timesteps, target_meta['shape']


# ============================================================================
# PREPROCESSING
# ============================================================================

def impute_spei(all_data: Dict, valid_timesteps: List):
    print(f"  Imputing SPEI...")
    for month in range(1, 13):
        frames = []
        years = []
        for y, m in valid_timesteps:
            if m != month:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and VARIABLE in all_data[key]:
                frames.append(all_data[key][VARIABLE].copy())
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
                all_data[key][VARIABLE] = stack[t]

    return all_data


def spatio_temporal_impute(all_data: Dict, valid_timesteps: List, max_spatial_iters: int = 3):
    """
    Build a time x H x W stack, do per-pixel temporal interpolation (linear),
    then simple iterative spatial mean fill for remaining NaNs.
    Writes values back into all_data.
    """
    keys = [f"{y}_{m:02d}" for (y, m) in valid_timesteps]
    # build stack T x H x W
    sample = next(iter(all_data.values()))
    H, W = sample[VARIABLE].shape
    T = len(keys)
    stack = np.full((T, H, W), np.nan, dtype=np.float32)
    for t, key in enumerate(keys):
        if key in all_data and VARIABLE in all_data[key]:
            stack[t] = all_data[key][VARIABLE]

    # temporal interp per pixel
    idxs = np.arange(T)
    for i in range(H):
        for j in range(W):
            series = stack[:, i, j]
            valid = np.isfinite(series)
            if valid.sum() == 0:
                continue
            if valid.sum() == 1:
                series[~valid] = series[valid][0]
            else:
                series[~valid] = np.interp(idxs[~valid], idxs[valid], series[valid])
            stack[:, i, j] = series

    # spatial iterative mean fill for any remaining NaNs
    for it in range(max_spatial_iters):
        nan_any = False
        for t in range(T):
            arr = stack[t]
            mask = ~np.isfinite(arr)
            if not mask.any():
                continue
            nan_any = True
            # local mean via generic filter; ignores NaNs by replacing them with 0 and counting neighbors
            filled = ndimage.generic_filter(np.nan_to_num(arr, nan=0.0), np.nanmean, size=3, mode='nearest')
            arr[mask] = filled[mask]
            stack[t] = arr
        if not nan_any:
            break

    # write back
    for t, key in enumerate(keys):
        if key in all_data and VARIABLE in all_data[key]:
            all_data[key][VARIABLE] = stack[t]

    return all_data


def detrend_spei(all_data: Dict, valid_timesteps: List):
    for month in range(1, 13):
        ys, mus = [], []
        for y, m in valid_timesteps:
            if m != month:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and VARIABLE in all_data[key]:
                mu = np.nanmean(all_data[key][VARIABLE])
                if np.isfinite(mu):
                    ys.append(y)
                    mus.append(mu)

        if len(ys) < 5:
            continue

        ys = np.array(ys)
        mus = np.array(mus)
        mask = (ys >= 2001) & (ys <= 2018)

        if mask.sum() >= 5:
            A = np.vstack([ys[mask], np.ones(mask.sum())]).T
            a, b = np.linalg.lstsq(A, mus[mask], rcond=None)[0]

            for y, m in valid_timesteps:
                if m != month:
                    continue
                key = f"{y}_{m:02d}"
                if key in all_data and VARIABLE in all_data[key]:
                    all_data[key][VARIABLE] = all_data[key][VARIABLE] - (a * y + b)

    return all_data


def compute_spei_stats(all_data: Dict, valid_timesteps: List) -> Dict:
    stats = {}
    # collect train values (years <= 2018)
    all_train_vals = []
    for y, m in valid_timesteps:
        if y > 2018:
            continue
        key = f"{y}_{m:02d}"
        if key in all_data and VARIABLE in all_data[key]:
            vals = all_data[key][VARIABLE][np.isfinite(all_data[key][VARIABLE])]
            if vals.size > 0:
                all_train_vals.extend(vals.tolist())
    if len(all_train_vals) == 0:
        global_median = 0.0
        global_iqr = 1.0
    else:
        global_median = np.median(all_train_vals)
        q75, q25 = np.percentile(all_train_vals, [75, 25])
        global_iqr = max(1e-8, (q75 - q25) / 1.349)  # convert IQR to robust std approx

    for month in range(1, 13):
        month_vals = []
        for y, m in valid_timesteps:
            if y > 2018 or m != month:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and VARIABLE in all_data[key]:
                vals = all_data[key][VARIABLE][np.isfinite(all_data[key][VARIABLE])]
                if vals.size > 0:
                    month_vals.extend(vals.tolist())
        if len(month_vals) > 0:
            median = np.median(month_vals)
            q75, q25 = np.percentile(month_vals, [75, 25])
            iqr_std = max(1e-8, (q75 - q25) / 1.349)
            stats[month] = {
                'mean': float(median),
                'std': float(iqr_std),
                'global_mean': float(global_median),
                'global_std': float(global_iqr)
            }
    return stats


# ============================================================================
# N-BEATS ARCHITECTURE
# ============================================================================

class NBeatsBlock(nn.Module):
    def __init__(self, input_size, theta_size, basis_function, layers, layer_size):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.layers = nn.ModuleList([nn.Linear(input_size if i == 0 else layer_size, layer_size)
                                     for i in range(layers)])
        self.backcast_linear = nn.Linear(layer_size, theta_size)
        self.forecast_linear = nn.Linear(layer_size, theta_size)

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        backcast_theta = self.backcast_linear(x)
        forecast_theta = self.forecast_linear(x)
        backcast = self.basis_function(backcast_theta, self.input_size, is_forecast=False)
        forecast = self.basis_function(forecast_theta, 1, is_forecast=True)
        return backcast, forecast


class GenericBasis(nn.Module):
    def __init__(self, backcast_size, forecast_size):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta, size, is_forecast):
        if is_forecast:
            return theta[:, :min(self.forecast_size, theta.shape[1])]
        return theta[:, :min(self.backcast_size, theta.shape[1])]


class TrendBasis(nn.Module):
    def __init__(self, degree=2):
        super().__init__()
        self.degree = degree

    def forward(self, theta, size, is_forecast):
        batch_size = theta.shape[0]
        theta_trend = theta[:, :self.degree + 1]
        t = torch.arange(size if not is_forecast else 2 * size, dtype=torch.float32, device=theta.device)
        if is_forecast:
            t = t[size:]
        t = t.unsqueeze(0).repeat(batch_size, 1) / size
        basis = torch.stack([t ** i for i in range(self.degree + 1)], dim=-1)
        return torch.sum(basis * theta_trend.unsqueeze(1), dim=-1)


class SeasonalityBasis(nn.Module):
    def __init__(self, harmonics):
        super().__init__()
        self.harmonics = harmonics

    def forward(self, theta, size, is_forecast):
        batch_size = theta.shape[0]
        theta_seasonal = theta[:, :2 * self.harmonics]
        t = torch.arange(size if not is_forecast else 2 * size, dtype=torch.float32, device=theta.device)
        if is_forecast:
            t = t[size:]
        t = t.unsqueeze(0).repeat(batch_size, 1) / 12.0
        cos_terms = [torch.cos(2 * np.pi * i * t) for i in range(1, self.harmonics + 1)]
        sin_terms = [torch.sin(2 * np.pi * i * t) for i in range(1, self.harmonics + 1)]
        basis = torch.stack(cos_terms + sin_terms, dim=-1)
        return torch.sum(basis * theta_seasonal.unsqueeze(1), dim=-1)


class UnivariateNBeats(nn.Module):
    def __init__(self, lookback=12, stack_types=['generic', 'trend', 'seasonality'],
                 nb_blocks_per_stack=3, thetas_dim=[12, 4, 4],
                 hidden_layer_units=128, nb_harmonics=2):
        super().__init__()
        self.lookback = lookback

        self.basis_functions = {}
        for i, stack_type in enumerate(stack_types):
            if stack_type == 'generic':
                self.basis_functions[stack_type] = GenericBasis(lookback, 1)
            elif stack_type == 'trend':
                self.basis_functions[stack_type] = TrendBasis(degree=thetas_dim[i] - 1)
            elif stack_type == 'seasonality':
                self.basis_functions[stack_type] = SeasonalityBasis(nb_harmonics)

        self.stacks = nn.ModuleList()
        for i, stack_type in enumerate(stack_types):
            stack = nn.ModuleList()
            for _ in range(nb_blocks_per_stack):
                block = NBeatsBlock(lookback, thetas_dim[i], self.basis_functions[stack_type],
                                    4, hidden_layer_units)
                stack.append(block)
            self.stacks.append(stack)

    def forward(self, x):
        residual = x
        forecast = torch.zeros(x.shape[0], 1, device=x.device)

        for stack in self.stacks:
            for block in stack:
                backcast, block_forecast = block(residual)
                residual = residual - backcast
                forecast = forecast + block_forecast

        return forecast.squeeze()


# ============================================================================
# DATASET
# ============================================================================

class SPEIDataset(Dataset):
    def __init__(self, all_data, valid_timesteps, pixel_indices,
                 lookback, indices, var_stats, clip, augment=False, aug_std_frac=0.02):
        self.all_data = all_data
        self.valid_timesteps = valid_timesteps
        self.pixel_indices = pixel_indices
        self.lookback = lookback
        self.indices = indices
        self.var_stats = var_stats
        self.clip = clip
        self.augment = augment
        self.aug_std_frac = aug_std_frac
        self.rng = np.random.RandomState(CONFIG['seed'])

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
            if key in self.all_data and VARIABLE in self.all_data[key]:
                val = self.all_data[key][VARIABLE][row, col]
                if month in self.var_stats:
                    val = (val - self.var_stats[month]['mean']) / self.var_stats[month]['std']
                val = np.clip(val, -self.clip, self.clip)
                if not np.isfinite(val):
                    val = 0.0
                x_vals.append(val)
            else:
                x_vals.append(0.0)

        x = np.array(x_vals, dtype=np.float32)
        if self.augment:
            # relative to monthly/global std
            avg_std = np.nanmean([self.var_stats[m]['std'] for (y,m) in self.valid_timesteps if m in self.var_stats]) if self.var_stats else 1.0
            noise = self.rng.normal(scale=self.aug_std_frac * avg_std, size=x.shape).astype(np.float32)
            x = x + noise

        year, month = self.valid_timesteps[t_idx]
        key = f"{year}_{month:02d}"
        if key in self.all_data and VARIABLE in self.all_data[key]:
            y_val = self.all_data[key][VARIABLE][row, col]
            if month in self.var_stats:
                y_val = (y_val - self.var_stats[month]['mean']) / self.var_stats[month]['std']
            y_val = np.clip(y_val, -self.clip, self.clip)
            mask = 1.0 if np.isfinite(y_val) else 0.0
            if not np.isfinite(y_val):
                y_val = 0.0
        else:
            y_val = 0.0
            mask = 0.0

        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
                torch.tensor(mask, dtype=torch.float32))


# ============================================================================
# TRAINING
# ============================================================================

def train_model(model, train_loader, val_loader, cfg):
    device = cfg['device']
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    # Add warmup + cosine schedule
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (cfg['num_epochs'] - warmup_epochs)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ... rest of training code

    import copy
    best_state = copy.deepcopy(model.state_dict())
    best_val = float('inf')
    no_improve = 0

    history = {'train_loss': [], 'val_loss': []}

    ema_decay = cfg.get('ema_decay', 0.999)
    use_ema = cfg.get('use_ema', True)
    ema_model = copy.deepcopy(model) if use_ema else None
    if ema_model is not None:
        for p in ema_model.parameters():
            p.requires_grad_(False)

    loss_type = cfg.get('loss', 'mse')

    for epoch in range(cfg['num_epochs']):
        model.train()
        train_losses = []

        for x, y, m in train_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            optimizer.zero_grad()
            pred = model(x)

            if m.sum() > 0:
                if loss_type == 'huber':
                    # Smooth L1 per element, then mask and average
                    elem_loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction='none')
                    loss = torch.sum(elem_loss * m) / m.sum()
                else:
                    loss = torch.sum(((pred - y) ** 2) * m) / m.sum()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            if cfg['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
            optimizer.step()
            train_losses.append(loss.item())
            if ema_model is not None:
                with torch.no_grad():
                    for p, ema_p in zip(model.parameters(), ema_model.parameters()):
                        ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, m in val_loader:
                x, y, m = x.to(device), y.to(device), m.to(device)
                pred = model(x)
                if m.sum() > 0:
                    if loss_type == 'huber':
                        elem_loss = torch.nn.functional.smooth_l1_loss(pred, y, reduction='none')
                        loss = torch.sum(elem_loss * m) / m.sum()
                    else:
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
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}: train={train_loss:.4f}, val={val_loss:.4f}")

    if ema_model is not None:
        model.load_state_dict(ema_model.state_dict())
    else:
        model.load_state_dict(best_state)
    return model, history


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    model.eval()
    preds, obs, masks = [], [], []

    with torch.no_grad():
        for x, y, m in test_loader:
            x, y, m = x.to(device), y.to(device), m.to(device)
            pred = model(x)

            if len(pred.shape) > 1:
                pred = pred.squeeze()

            preds.extend(pred.cpu().numpy().tolist())
            obs.extend(y.cpu().numpy().tolist())
            masks.extend(m.cpu().numpy().tolist())

    preds = np.array(preds)
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
    else:
        metrics = {'rmse': np.nan, 'mae': np.nan, 'r2': np.nan, 'corr': np.nan}
        metrics['n_valid'] = int(n_valid)

    return metrics, preds, obs, masks


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(preds, obs, masks, history, land_type, test_idx,
                         valid_timesteps, num_pixels, output_dir='results_spei'):
    num_test_times = len(test_idx)
    preds_2d = preds.reshape(num_test_times, num_pixels)
    obs_2d = obs.reshape(num_test_times, num_pixels)
    masks_2d = masks.reshape(num_test_times, num_pixels)

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

    if valid_agg.sum() > 1:
        agg_rmse = np.sqrt(mean_squared_error(obs_agg[valid_agg], preds_agg[valid_agg]))
        agg_r2 = r2_score(obs_agg[valid_agg], preds_agg[valid_agg])
        agg_corr = pearsonr(obs_agg[valid_agg], preds_agg[valid_agg])[0]
    else:
        agg_rmse = np.nan
        agg_r2 = np.nan
        agg_corr = np.nan

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'California {land_type.title()} - SPEI Forecasting (N-BEATS)',
                 fontsize=16, fontweight='bold')

    # Time Series
    ax1 = plt.subplot(2, 3, 1)
    if valid_agg.sum() > 0:
        x_idx = np.arange(len(test_dates))
        ax1.plot(x_idx[valid_agg], obs_agg[valid_agg], 'b-', label='Observed', alpha=0.7, linewidth=2)
        ax1.plot(x_idx[valid_agg], preds_agg[valid_agg], 'r--', label='Predicted', alpha=0.7, linewidth=2)
        ax1.set_xticks(x_idx[::6])
        ax1.set_xticklabels([test_dates[i] for i in x_idx[::6]], rotation=45, ha='right')
    ax1.set_title('SPEI Forecasting (Spatial Mean)', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('SPEI (normalized)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Scatter Plot
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

    # Training History
    ax3 = plt.subplot(2, 3, 3)
    epochs = range(1, len(history['train_loss']) + 1)
    ax3.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax3.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax3.set_title('Training Progress', fontsize=12)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss (MSE)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Residuals
    ax4 = plt.subplot(2, 3, 4)
    if valid_agg.sum() > 0:
        residuals = obs_agg[valid_agg] - preds_agg[valid_agg]
        ax4.scatter(preds_agg[valid_agg], residuals, alpha=0.5, s=30, color='purple')
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.set_title('Residual Plot', fontsize=12)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Residuals')
        ax4.grid(True, alpha=0.3)

    # Error Distribution
    ax5 = plt.subplot(2, 3, 5)
    if valid_agg.sum() > 0:
        residuals = obs_agg[valid_agg] - preds_agg[valid_agg]
        ax5.hist(residuals, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_title('Error Distribution', fontsize=12)
        ax5.set_xlabel('Residuals')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3, axis='y')

    # Metrics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')

    summary_text = f"""
=== SPEI N-BEATS METRICS ===

Land Type: {land_type}

Spatial Aggregation:
  RMSE: {agg_rmse:.4f}
  R²:   {agg_r2:.4f}
  Corr: {agg_corr:.4f}

Training:
  Epochs: {len(history['train_loss'])}
  Final train: {history['train_loss'][-1]:.4f}
  Final val: {history['val_loss'][-1]:.4f}

Data:
  Pixels: {num_pixels}
  Test timesteps: {num_test_times}
  Valid timesteps: {valid_agg.sum()}

Model Architecture:
  Lookback: {CONFIG['lookback']}
  Stacks: {CONFIG['stack_types']}
  Blocks/stack: {CONFIG['nb_blocks_per_stack']}
  Hidden units: {CONFIG['hidden_layer_units']}
    """

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{land_type.lower()}_spei_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved visualization to {output_path}")


def stratified_pixel_sampling(all_data, valid_timesteps, valid_mask, n_nodes=100, n_clusters=6, seed=42):
    # compute per-pixel training mean (2001-2018)
    H, W = valid_mask.shape
    pixels = np.column_stack(np.where(valid_mask))
    feats = []
    for (r, c) in pixels:
        vals = []
        for y, m in valid_timesteps:
            if y > 2018:
                continue
            key = f"{y}_{m:02d}"
            if key in all_data and VARIABLE in all_data[key]:
                v = all_data[key][VARIABLE][r, c]
                if np.isfinite(v):
                    vals.append(v)
        if len(vals) == 0:
            feats.append([0.0, 0.0])
        else:
            feats.append([np.nanmean(vals), np.nanstd(vals)])
    feats = np.array(feats, dtype=np.float32)
    k = min(n_clusters, len(pixels))
    kmeans = KMeans(n_clusters=k, random_state=seed).fit(feats)
    sel_idx = []
    per_cluster = max(1, n_nodes // k)
    for cl in range(k):
        ids = np.where(kmeans.labels_ == cl)[0]
        if len(ids) == 0:
            continue
        choose = ids if len(ids) <= per_cluster else np.random.choice(ids, per_cluster, replace=False)
        sel_idx.extend(choose.tolist())
    # if not enough, fill randomly
    if len(sel_idx) < n_nodes:
        remaining = list(set(range(len(pixels))) - set(sel_idx))
        add = min(n_nodes - len(sel_idx), len(remaining))
        if add > 0:
            sel_idx.extend(list(np.random.choice(remaining, add, replace=False)))
    chosen_pixels = pixels[sel_idx]
    return chosen_pixels


# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_spei_for_land_type(land_type: str) -> Optional[Dict]:
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {land_type} - SPEI")
    print(f"{'=' * 80}")

    all_data, valid_timesteps, shape = load_spei_data(land_type)

    if all_data is None:
        return None

    all_data = spatio_temporal_impute(all_data, valid_timesteps)
    all_data = detrend_spei(all_data, valid_timesteps)
    var_stats = compute_spei_stats(all_data, valid_timesteps)

    print(f"  Finding valid pixels...")

    train_coverage = np.zeros(shape, dtype=np.int32)
    train_months = 0
    for year, month in valid_timesteps:
        if year > 2021:
            continue
        train_months += 1
        key = f"{year}_{month:02d}"
        if key in all_data and VARIABLE in all_data[key]:
            train_coverage += np.isfinite(all_data[key][VARIABLE]).astype(np.int32)

    test_coverage = np.zeros(shape, dtype=np.int32)
    test_months = 0
    for year, month in valid_timesteps:
        if year < 2022:
            continue
        test_months += 1
        key = f"{year}_{month:02d}"
        if key in all_data and VARIABLE in all_data[key]:
            test_coverage += np.isfinite(all_data[key][VARIABLE]).astype(np.int32)

    train_threshold = int(CONFIG['min_valid_frac'] * train_months)
    test_threshold = int(CONFIG['min_valid_frac'] * test_months)

    train_valid = train_coverage >= train_threshold
    test_valid = test_coverage >= test_threshold
    valid_mask = train_valid & test_valid

    print(f"    Shared valid pixels: {valid_mask.sum()}")

    if valid_mask.sum() == 0:
        print(f"  ERROR: No shared pixels!")
        return None

    pixel_indices = stratified_pixel_sampling(all_data, valid_timesteps, valid_mask, n_nodes=CONFIG['max_nodes'], n_clusters=6, seed=CONFIG['seed'])

    print(f"  Selected {len(pixel_indices)} pixels")

    train_idx = [i for i, (y, m) in enumerate(valid_timesteps)
                 if y <= 2018 and i >= CONFIG['lookback']]
    val_idx = [i for i, (y, m) in enumerate(valid_timesteps)
               if 2019 <= y <= 2021 and i >= CONFIG['lookback']]
    test_idx = [i for i, (y, m) in enumerate(valid_timesteps)
                if y >= 2022 and i >= CONFIG['lookback']]

    if not train_idx or not val_idx or not test_idx:
        print(f"  ERROR: Insufficient data splits")
        return None

    train_dataset = SPEIDataset(all_data, valid_timesteps, pixel_indices,
                                CONFIG['lookback'], train_idx, var_stats, CONFIG['target_clip'], augment=True, aug_std_frac=0.02)
    val_dataset = SPEIDataset(all_data, valid_timesteps, pixel_indices,
                              CONFIG['lookback'], val_idx, var_stats, CONFIG['target_clip'], augment=False)
    test_dataset = SPEIDataset(all_data, valid_timesteps, pixel_indices,
                               CONFIG['lookback'], test_idx, var_stats, CONFIG['target_clip'], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    model = UnivariateNBeats(
        lookback=CONFIG['lookback'],
        stack_types=CONFIG['stack_types'],
        nb_blocks_per_stack=CONFIG['nb_blocks_per_stack'],
        thetas_dim=CONFIG['thetas_dim'],
        hidden_layer_units=CONFIG['hidden_layer_units'],
        nb_harmonics=CONFIG['nb_harmonics']
    )

    print(f"  Training model...")
    model, history = train_model(model, train_loader, val_loader, CONFIG)

    print(f"  Evaluating...")
    metrics, preds, obs, masks = evaluate_model(model, test_loader, CONFIG['device'])

    if np.isfinite(metrics['r2']):
        print(f"  Results: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}, Corr={metrics['corr']:.4f}")
    else:
        print(f"  Results: Insufficient valid test data")

    create_visualization(preds, obs, masks, history, land_type,
                         test_idx, valid_timesteps, len(pixel_indices))

    os.makedirs('results_spei', exist_ok=True)
    results_df = pd.DataFrame({
        'pred': preds,
        'obs': obs,
        'mask': masks
    })
    results_df.to_csv(f"results_spei/{land_type.lower()}_spei_results.csv", index=False)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"results_spei/{land_type.lower()}_spei_metrics.csv", index=False)

    gpu_cleanup()
    return metrics


def main():
    print("=" * 80)
    print("CALIFORNIA DROUGHT N-BEATS - SPEI ONLY")
    print("=" * 80)
    print(f"\nMODEL CONFIGURATION:")
    print(f"  Lookback: {CONFIG['lookback']}")
    print(f"  Stack types: {CONFIG['stack_types']}")
    print(f"  Blocks per stack: {CONFIG['nb_blocks_per_stack']}")
    print(f"  Thetas dimensions: {CONFIG['thetas_dim']}")
    print(f"  Hidden units: {CONFIG['hidden_layer_units']}")
    print(f"  Harmonics: {CONFIG['nb_harmonics']}")
    print(f"  Learning rate: {CONFIG['lr']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    print()

    all_metrics = {}

    for land_type in LAND_TYPES:
        try:
            metrics = process_spei_for_land_type(land_type)
            if metrics:
                all_metrics[land_type] = metrics
        except Exception as e:
            print(f"ERROR: {land_type}: {str(e)}")
            import traceback
            traceback.print_exc()

    summary_rows = []
    for lt in all_metrics:
        row = {'land_type': lt}
        row.update(all_metrics[lt])
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv('results_spei/summary_spei.csv', index=False)
        print("\n" + "=" * 80)
        print("SUMMARY - SPEI ACROSS ALL LAND TYPES")
        print("=" * 80)
        print(summary_df.to_string())

    print("\nDONE!")


if __name__ == "__main__":
    main()