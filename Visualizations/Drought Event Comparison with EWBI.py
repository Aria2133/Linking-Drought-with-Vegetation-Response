import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import glob
import re
import pandas as pd

"""
ENHANCED EWBI vs VEGETATION RESPONSE COMPARISON
Figure 1: EWBI (Ecohydrological Water Balance Index) changes in SD units
Figure 2: Vegetation (GOSIF or NIRv) response in percentage
Compares 2012-2016 vs 2020-2024 drought events

ENHANCEMENTS:
1. Statistical significance testing (t-test)
2. Quantitative spatial pattern analysis
3. Persistent vulnerability tracking
4. Statistics boxes in upper RIGHT corner
"""

# Configuration
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
VEGETATION_INDEX = 'NIRv'  # Change to 'GOSIF' for SIF analysis

# Define periods
BASELINE = (2001, 2011)
DROUGHT_2012 = (2012, 2016)
CRISIS_2020 = (2020, 2024)

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("=" * 70)
print(f"ENHANCED EWBI vs {VEGETATION_INDEX} COMPARISON")
print("=" * 70)
print(f"EWBI = z(SPEI) + z(TWS) - z(VPD) - z(LST)")
print(f"Vegetation Response: {VEGETATION_INDEX}")
print(f"Baseline: {BASELINE[0]}-{BASELINE[1]}")
print(f"2012-2016 Drought: {DROUGHT_2012[0]}-{DROUGHT_2012[1]}")
print(f"2020-2024 Crisis: {CRISIS_2020[0]}-{CRISIS_2020[1]}")


def standardize(data):
    """Convert to z-scores (mean=0, std=1)"""
    if data is None or np.all(np.isnan(data)):
        return None
    mean = np.nanmean(data)
    std = np.nanstd(data)
    if std == 0 or np.isnan(std):
        return np.zeros_like(data)
    return (data - mean) / std


def extract_year_from_filename(filename, index_name):
    """Extract year from different naming patterns"""
    basename = os.path.basename(filename)

    # Pattern 1: GOSIF_2022.M01_Grasslands or GOSIF_200101
    if index_name == 'GOSIF':
        match = re.search(r'(\d{4})[.\-_]?M?(\d{2})', basename)
        if match:
            return int(match.group(1))

    # Pattern 2: Standard format with _YYYY_MM or _YYYY_
    match = re.search(r'_(\d{4})_(\d{2})', basename)
    if match:
        return int(match.group(1))

    # Pattern 3: Year with month like 200101
    match = re.search(r'(\d{4})(\d{2})', basename)
    if match:
        year = int(match.group(1))
        if 1900 < year < 2100:  # Valid year range
            return year

    return None


def get_files(land_type, index):
    """Get files for a given land type and index from both training and test directories"""
    folder_map = {
        'Grasslands': 'CALIFORNIA GRASSLANDS',
        'NaturalForests': 'CALIFORNIA NATURAL FORESTS',
        'Savannas': 'CALIFORNIA SAVANNAS',
        'Shrublands': 'CALIFORNIA SHRUBLANDS'
    }

    # Handle VPDmax folder naming (folder is "VPD" but files contain "VPDmax")
    folder_name = 'VPD' if index == 'VPDmax' else index

    folder = folder_map[land_type]
    train_path = os.path.join(train_base, folder, folder_name)
    test_path = os.path.join(test_base, land_type, folder_name)

    files = []

    if os.path.exists(train_path):
        files.extend(glob.glob(os.path.join(train_path, "*.tif")))

    if os.path.exists(test_path):
        files.extend(glob.glob(os.path.join(test_path, "*.tif")))

    return sorted(list(set(files)))


def clean_data(data, index):
    """Clean data by removing invalid values"""
    data = data.astype(np.float64)
    nodata_values = [-9999, -3.4e+38, 32767, -32768, 65535, -32767]

    for nodata in nodata_values:
        data[np.abs(data - nodata) < 1.0] = np.nan

    data[np.abs(data) > 10000] = np.nan

    if index in ['NIRv']:
        data[(data < -0.1) | (data > 0.6)] = np.nan
    elif index in ['GNDVI', 'NDVI']:
        data[(data < -1) | (data > 1.5)] = np.nan
    elif index == 'GPP':
        data[(data < 0) | (data > 20)] = np.nan
    elif index == 'GOSIF':
        data[(data < -100) | (data > 3000)] = np.nan
    elif index == 'SPEI':
        data[(data < -5) | (data > 5)] = np.nan
    elif index == 'TWS':
        data[(data < -100) | (data > 100)] = np.nan
    elif index in ['VPDmax', 'VPD']:
        data[(data < 0) | (data > 10)] = np.nan
    elif index == 'LST':
        data[(data < -50) | (data > 60)] = np.nan

    return data


def load_and_process_index(land_type, index, year_start, year_end, ref_shape, ref_transform, ref_crs,
                           return_stack=False):
    """
    Load index data for a period and return mean - ALL REGRIDDED TO 5KM

    Parameters:
    -----------
    return_stack : bool
        If True, return the full monthly stack instead of mean (for significance testing)
    """
    files = get_files(land_type, index)

    period_data = []
    valid_files = 0

    for filepath in files:
        try:
            year = extract_year_from_filename(filepath, index)

            if year is None:
                continue

            if year_start <= year <= year_end:
                with rasterio.open(filepath) as src:
                    data = src.read(1)

                    # ALWAYS REGRID TO 5KM REFERENCE GRID
                    if src.shape != ref_shape:
                        output = np.empty(ref_shape, dtype=np.float32)
                        reproject(
                            source=data, destination=output,
                            src_transform=src.transform, src_crs=src.crs,
                            dst_transform=ref_transform, dst_crs=ref_crs,
                            resampling=Resampling.bilinear,
                            src_nodata=np.nan, dst_nodata=np.nan
                        )
                        data = output

                    data = clean_data(data, index)
                    period_data.append(data)
                    valid_files += 1
        except Exception as e:
            continue

    if len(period_data) == 0:
        return (None, 0) if not return_stack else (None, None, 0)

    period_stack = np.stack(period_data, axis=0)

    if return_stack:
        return period_stack, np.nanmean(period_stack, axis=0), valid_files
    else:
        return np.nanmean(period_stack, axis=0), valid_files


def calculate_significance(baseline_stack, drought_stack, alpha=0.05):
    """
    ENHANCEMENT 1: Test if drought period is significantly different from baseline

    Parameters:
    -----------
    baseline_stack : ndarray (n_months, rows, cols)
    drought_stack : ndarray (n_months, rows, cols)
    alpha : float
        Significance level

    Returns:
    --------
    p_values : ndarray (rows, cols)
    significant : ndarray (rows, cols) - boolean mask
    """
    if baseline_stack is None or drought_stack is None:
        return None, None

    rows, cols = baseline_stack.shape[1], baseline_stack.shape[2]
    p_values = np.ones((rows, cols)) * np.nan

    print("  Calculating pixel-wise significance (this may take a few minutes)...")

    for i in range(rows):
        if i % 50 == 0:
            print(f"    Processing row {i}/{rows}...")
        for j in range(cols):
            baseline_ts = baseline_stack[:, i, j]
            drought_ts = drought_stack[:, i, j]

            # Remove NaNs
            baseline_valid = baseline_ts[~np.isnan(baseline_ts)]
            drought_valid = drought_ts[~np.isnan(drought_ts)]

            # Need at least 10 baseline points and 5 drought points
            if len(baseline_valid) >= 10 and len(drought_valid) >= 5:
                # Two-sample t-test
                _, p = stats.ttest_ind(baseline_valid, drought_valid)
                p_values[i, j] = p

    significant = p_values < alpha

    return p_values, significant


def analyze_spatial_patterns(change_map, thresholds=[-40, -20, 20]):
    """
    ENHANCEMENT 2: Quantify spatial patterns in % change

    Parameters:
    -----------
    change_map : ndarray
        Map of percentage changes
    thresholds : list
        [severe_decline, moderate_decline, increase] thresholds

    Returns:
    --------
    dict with statistics
    """
    valid = change_map[~np.isnan(change_map)]

    if len(valid) == 0:
        return None

    stats_dict = {
        'mean_change': np.mean(valid),
        'median_change': np.median(valid),
        'std_change': np.std(valid),
        'severe_decline_pct': (np.sum(valid < thresholds[0]) / len(valid)) * 100,
        'severe_decline_npx': np.sum(valid < thresholds[0]),
        'moderate_decline_pct': (np.sum((valid >= thresholds[0]) & (valid < thresholds[1])) / len(valid)) * 100,
        'moderate_decline_npx': np.sum((valid >= thresholds[0]) & (valid < thresholds[1])),
        'stable_pct': (np.sum((valid >= thresholds[1]) & (valid <= thresholds[2])) / len(valid)) * 100,
        'stable_npx': np.sum((valid >= thresholds[1]) & (valid <= thresholds[2])),
        'increase_pct': (np.sum(valid > thresholds[2]) / len(valid)) * 100,
        'increase_npx': np.sum(valid > thresholds[2]),
        'min_change': np.min(valid),
        'max_change': np.max(valid),
        'q05': np.percentile(valid, 5),
        'q25': np.percentile(valid, 25),
        'q75': np.percentile(valid, 75),
        'q95': np.percentile(valid, 95),
        'total_pixels': len(valid)
    }

    return stats_dict


def identify_persistent_vulnerability(change_2012, change_2020, severe_threshold=-40):
    """
    ENHANCEMENT 3: Find pixels that were severely affected in BOTH droughts

    Parameters:
    -----------
    change_2012 : ndarray
        % change in 2012-2016
    change_2020 : ndarray
        % change in 2020-2024
    severe_threshold : float
        Threshold for "severe" decline

    Returns:
    --------
    persistence_map : ndarray
        Categorical map (0=never severe, 1=only 2020, 2=only 2012, 3=both)
    stats_dict : dict
    """
    if change_2012 is None or change_2020 is None:
        return None, None

    # Pixels severely affected in each period
    severe_2012 = change_2012 < severe_threshold
    severe_2020 = change_2020 < severe_threshold

    # Persistent vulnerability (both periods)
    persistent = severe_2012 & severe_2020

    # Period-specific vulnerability
    only_2012 = severe_2012 & ~severe_2020
    only_2020 = severe_2020 & ~severe_2012

    # Resilient pixels (never severe)
    resilient = (change_2012 >= severe_threshold) & (change_2020 >= severe_threshold)

    # Calculate statistics
    valid_pixels = np.sum(~np.isnan(change_2012))

    stats_dict = {
        'persistent_vulnerable_npx': np.sum(persistent),
        'persistent_vulnerable_pct': (np.sum(persistent) / valid_pixels) * 100,
        'only_2012_npx': np.sum(only_2012),
        'only_2012_pct': (np.sum(only_2012) / valid_pixels) * 100,
        'only_2020_npx': np.sum(only_2020),
        'only_2020_pct': (np.sum(only_2020) / valid_pixels) * 100,
        'resilient_npx': np.sum(resilient),
        'resilient_pct': (np.sum(resilient) / valid_pixels) * 100,
        'total_pixels': valid_pixels
    }

    # Create categorical map
    persistence_map = np.full_like(change_2012, np.nan)
    persistence_map[resilient] = 0  # Never severe
    persistence_map[only_2020] = 1  # Only 2020-2024
    persistence_map[only_2012] = 2  # Only 2012-2016
    persistence_map[persistent] = 3  # Both periods

    return persistence_map, stats_dict


def calculate_ewbi(spei_z, tws_z, vpd_z, lst_z):
    """Calculate EWBI with proper None handling"""
    components = []

    if spei_z is not None:
        components.append(spei_z)

    if tws_z is not None:
        components.append(tws_z)

    if vpd_z is not None:
        components.append(-vpd_z)  # Negative contribution

    if lst_z is not None:
        components.append(-lst_z)  # Negative contribution

    # Need at least 2 components
    if len(components) < 2:
        return None

    return np.sum(components, axis=0)


# ===== LOAD REFERENCE GRID =====
print("\n" + "=" * 70)
print("Loading reference grid...")
print("=" * 70)

ref_shape = None
ref_transform = None
ref_crs = None
ref_bounds = None

for land_type in land_types:
    files = get_files(land_type, VEGETATION_INDEX)
    if len(files) > 0:
        with rasterio.open(files[0]) as src:
            ref_shape = src.shape
            ref_transform = src.transform
            ref_crs = src.crs
            ref_bounds = src.bounds
            ref_bounds = [ref_bounds[0], ref_bounds[2], ref_bounds[1], ref_bounds[3]]
        print(f"✓ Reference grid loaded from {land_type}")
        print(f"  Shape: {ref_shape}")
        print(f"  CRS: {ref_crs}")
        print(f"  Bounds: {ref_bounds}")
        break

if ref_shape is None:
    raise RuntimeError("Could not load reference grid!")

# ===== PROCESS DATA FOR ALL LAND TYPES =====
print("\n" + "=" * 70)
print("Processing data for all land types...")
print("=" * 70)

results_ewbi = {}
results_veg = {}
significance_results = {}
spatial_stats = {}
persistence_results = {}

for land_type in land_types:
    print(f"\n{'=' * 70}")
    print(f"Processing {land_type}...")
    print(f"{'=' * 70}")

    results_ewbi[land_type] = {'drought_2012': None, 'crisis_2020': None}
    results_veg[land_type] = {'drought_2012': None, 'crisis_2020': None}
    spatial_stats[land_type] = {}

    # Load EWBI components for each period
    print("\n1. Loading EWBI components...")

    # Baseline
    spei_base, _ = load_and_process_index(land_type, 'SPEI', *BASELINE, ref_shape, ref_transform, ref_crs)
    tws_base, _ = load_and_process_index(land_type, 'TWS', *BASELINE, ref_shape, ref_transform, ref_crs)
    vpd_base, _ = load_and_process_index(land_type, 'VPDmax', *BASELINE, ref_shape, ref_transform, ref_crs)
    lst_base, _ = load_and_process_index(land_type, 'LST', *BASELINE, ref_shape, ref_transform, ref_crs)

    # 2012-2016 drought
    spei_2012, _ = load_and_process_index(land_type, 'SPEI', *DROUGHT_2012, ref_shape, ref_transform, ref_crs)
    tws_2012, _ = load_and_process_index(land_type, 'TWS', *DROUGHT_2012, ref_shape, ref_transform, ref_crs)
    vpd_2012, _ = load_and_process_index(land_type, 'VPDmax', *DROUGHT_2012, ref_shape, ref_transform, ref_crs)
    lst_2012, _ = load_and_process_index(land_type, 'LST', *DROUGHT_2012, ref_shape, ref_transform, ref_crs)

    # 2020-2024 crisis
    spei_2020, _ = load_and_process_index(land_type, 'SPEI', *CRISIS_2020, ref_shape, ref_transform, ref_crs)
    tws_2020, _ = load_and_process_index(land_type, 'TWS', *CRISIS_2020, ref_shape, ref_transform, ref_crs)
    vpd_2020, _ = load_and_process_index(land_type, 'VPDmax', *CRISIS_2020, ref_shape, ref_transform, ref_crs)
    lst_2020, _ = load_and_process_index(land_type, 'LST', *CRISIS_2020, ref_shape, ref_transform, ref_crs)

    # Standardize EWBI components spatially
    print("  Standardizing EWBI components...")
    spei_base_z = standardize(spei_base)
    tws_base_z = standardize(tws_base)
    vpd_base_z = standardize(vpd_base)
    lst_base_z = standardize(lst_base)

    spei_2012_z = standardize(spei_2012)
    tws_2012_z = standardize(tws_2012)
    vpd_2012_z = standardize(vpd_2012)
    lst_2012_z = standardize(lst_2012)

    spei_2020_z = standardize(spei_2020)
    tws_2020_z = standardize(tws_2020)
    vpd_2020_z = standardize(vpd_2020)
    lst_2020_z = standardize(lst_2020)

    # Calculate EWBI for each period
    ewbi_base = calculate_ewbi(spei_base_z, tws_base_z, vpd_base_z, lst_base_z)
    ewbi_2012 = calculate_ewbi(spei_2012_z, tws_2012_z, vpd_2012_z, lst_2012_z)
    ewbi_2020 = calculate_ewbi(spei_2020_z, tws_2020_z, vpd_2020_z, lst_2020_z)

    # Calculate EWBI changes (difference from baseline)
    if ewbi_base is not None and ewbi_2012 is not None:
        results_ewbi[land_type]['drought_2012'] = ewbi_2012 - ewbi_base
        print(f"  ✓ EWBI 2012-2016 calculated")

    if ewbi_base is not None and ewbi_2020 is not None:
        results_ewbi[land_type]['crisis_2020'] = ewbi_2020 - ewbi_base
        print(f"  ✓ EWBI 2020-2024 calculated")

    # Load vegetation data (with stacks for significance testing)
    print(f"\n2. Loading {VEGETATION_INDEX} data (with monthly stacks)...")

    veg_base_stack, veg_base, n_base = load_and_process_index(
        land_type, VEGETATION_INDEX, *BASELINE, ref_shape, ref_transform, ref_crs, return_stack=True
    )
    veg_2012_stack, veg_2012, n_2012 = load_and_process_index(
        land_type, VEGETATION_INDEX, *DROUGHT_2012, ref_shape, ref_transform, ref_crs, return_stack=True
    )
    veg_2020_stack, veg_2020, n_2020 = load_and_process_index(
        land_type, VEGETATION_INDEX, *CRISIS_2020, ref_shape, ref_transform, ref_crs, return_stack=True
    )

    print(f"  Loaded {n_base} baseline files, {n_2012} drought files, {n_2020} crisis files")

    # Calculate percentage changes
    if veg_base is not None and veg_2012 is not None:
        results_veg[land_type]['drought_2012'] = ((veg_2012 - veg_base) / veg_base) * 100
        print(f"  ✓ {VEGETATION_INDEX} 2012-2016 change calculated")

    if veg_base is not None and veg_2020 is not None:
        results_veg[land_type]['crisis_2020'] = ((veg_2020 - veg_base) / veg_base) * 100
        print(f"  ✓ {VEGETATION_INDEX} 2020-2024 change calculated")

    # ENHANCEMENT 1: Calculate significance
    print(f"\n3. ENHANCEMENT 1: Calculating statistical significance...")
    significance_results[land_type] = {}

    if veg_base_stack is not None and veg_2012_stack is not None:
        p_2012, sig_2012 = calculate_significance(veg_base_stack, veg_2012_stack)
        significance_results[land_type]['p_2012'] = p_2012
        significance_results[land_type]['sig_2012'] = sig_2012
        if sig_2012 is not None:
            sig_count = np.sum(sig_2012)
            sig_pct = (sig_count / np.sum(~np.isnan(sig_2012))) * 100
            print(f"  ✓ 2012-2016: {sig_count:,} pixels ({sig_pct:.1f}%) show significant change (p<0.05)")

    if veg_base_stack is not None and veg_2020_stack is not None:
        p_2020, sig_2020 = calculate_significance(veg_base_stack, veg_2020_stack)
        significance_results[land_type]['p_2020'] = p_2020
        significance_results[land_type]['sig_2020'] = sig_2020
        if sig_2020 is not None:
            sig_count = np.sum(sig_2020)
            sig_pct = (sig_count / np.sum(~np.isnan(sig_2020))) * 100
            print(f"  ✓ 2020-2024: {sig_count:,} pixels ({sig_pct:.1f}%) show significant change (p<0.05)")

    # ENHANCEMENT 2: Spatial pattern analysis
    print(f"\n4. ENHANCEMENT 2: Analyzing spatial patterns...")

    if results_veg[land_type]['drought_2012'] is not None:
        stats_2012 = analyze_spatial_patterns(results_veg[land_type]['drought_2012'])
        spatial_stats[land_type]['2012'] = stats_2012
        if stats_2012:
            print(f"  ✓ 2012-2016 statistics:")
            print(f"    Mean change: {stats_2012['mean_change']:.1f}%")
            print(
                f"    Severe decline (<-40%): {stats_2012['severe_decline_npx']:,} px ({stats_2012['severe_decline_pct']:.1f}%)")
            print(f"    Stable (-20% to +20%): {stats_2012['stable_npx']:,} px ({stats_2012['stable_pct']:.1f}%)")

    if results_veg[land_type]['crisis_2020'] is not None:
        stats_2020 = analyze_spatial_patterns(results_veg[land_type]['crisis_2020'])
        spatial_stats[land_type]['2020'] = stats_2020
        if stats_2020:
            print(f"  ✓ 2020-2024 statistics:")
            print(f"    Mean change: {stats_2020['mean_change']:.1f}%")
            print(
                f"    Severe decline (<-40%): {stats_2020['severe_decline_npx']:,} px ({stats_2020['severe_decline_pct']:.1f}%)")
            print(f"    Stable (-20% to +20%): {stats_2020['stable_npx']:,} px ({stats_2020['stable_pct']:.1f}%)")

    # ENHANCEMENT 3: Persistent vulnerability
    print(f"\n5. ENHANCEMENT 3: Tracking persistent vulnerability...")

    persist_map, persist_stats = identify_persistent_vulnerability(
        results_veg[land_type]['drought_2012'],
        results_veg[land_type]['crisis_2020']
    )

    if persist_stats:
        persistence_results[land_type] = {
            'map': persist_map,
            'stats': persist_stats
        }
        print(f"  ✓ Persistent vulnerability analysis:")
        print(
            f"    Both droughts severe: {persist_stats['persistent_vulnerable_npx']:,} px ({persist_stats['persistent_vulnerable_pct']:.1f}%)")
        print(f"    Only 2012-2016: {persist_stats['only_2012_npx']:,} px ({persist_stats['only_2012_pct']:.1f}%)")
        print(f"    Only 2020-2024: {persist_stats['only_2020_npx']:,} px ({persist_stats['only_2020_pct']:.1f}%)")
        print(
            f"    Resilient (never severe): {persist_stats['resilient_npx']:,} px ({persist_stats['resilient_pct']:.1f}%)")

# ===== FIGURE 1: EWBI COMPARISON =====
print("\n" + "=" * 70)
print("Creating EWBI comparison figure...")
print("=" * 70)

fig1, axes1 = plt.subplots(4, 2, figsize=(18, 24))

for idx, land_type in enumerate(land_types):
    ax_left = axes1[idx, 0]
    ax_right = axes1[idx, 1]

    if land_type not in results_ewbi or results_ewbi[land_type]['drought_2012'] is None:
        for ax in [ax_left, ax_right]:
            ax.text(0.5, 0.5, 'No Data\n(Missing EWBI components)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.axis('off')
        continue

    # Plot 2012-2016 EWBI (in SD units)
    if results_ewbi[land_type]['drought_2012'] is not None:
        data_2012 = results_ewbi[land_type]['drought_2012']
        im_left = ax_left.imshow(data_2012, cmap='RdYlGn', vmin=-3, vmax=3,
                                 extent=ref_bounds, aspect='auto', interpolation='nearest')
        cbar_left = plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
        cbar_left.set_label('Change (SD)', fontsize=10, fontweight='bold')

        valid = data_2012[~np.isnan(data_2012)]
        if len(valid) > 0:
            stats = f'Mean: {np.mean(valid):.2f} SD\nSevere (<-1.5): {np.sum(valid < -1.5):,} px'
            # MOVED TO UPPER RIGHT
            ax_left.text(0.98, 0.98, stats, transform=ax_left.transAxes,
                         fontsize=9, verticalalignment='top', horizontalalignment='right',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_left.set_title(f'{land_type}\n2012-2016 Historic Drought',
                      fontsize=13, fontweight='bold', pad=10)
    ax_left.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_left.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax_left.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Plot 2020-2024 EWBI
    if results_ewbi[land_type]['crisis_2020'] is not None:
        data_2020 = results_ewbi[land_type]['crisis_2020']
        im_right = ax_right.imshow(data_2020, cmap='RdYlGn', vmin=-3, vmax=3,
                                   extent=ref_bounds, aspect='auto', interpolation='nearest')
        cbar_right = plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
        cbar_right.set_label('Change (SD)', fontsize=10, fontweight='bold')

        valid = data_2020[~np.isnan(data_2020)]
        if len(valid) > 0:
            stats = f'Mean: {np.mean(valid):.2f} SD\nSevere (<-1.5): {np.sum(valid < -1.5):,} px'
            # MOVED TO UPPER RIGHT
            ax_right.text(0.98, 0.98, stats, transform=ax_right.transAxes,
                          fontsize=9, verticalalignment='top', horizontalalignment='right',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_right.set_title(f'{land_type}\n2020-2024 Recent Crisis',
                       fontsize=13, fontweight='bold', pad=10)
    ax_right.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_right.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax_right.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.figtext(0.5, 0.01,
            'Red = Worse Ecohydrological Stress • Green = Better Conditions • Values in Standard Deviations (SD)\n' +
            'EWBI = z(SPEI) + z(TWS) - z(VPD) - z(LST) | All indices regridded to 5km resolution',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

plt.suptitle('Ecohydrological Water Balance Index (EWBI) Changes\n' +
             'Historic Drought (2012-2016) vs Recent Crisis (2020-2024)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.99])
plt.savefig('EWBI_drought_comparison_enhanced.png', dpi=300, bbox_inches='tight')
plt.savefig('EWBI_drought_comparison_enhanced.pdf', dpi=300, bbox_inches='tight')
print("✓ Saved: EWBI_drought_comparison_enhanced.png/pdf")

# ===== FIGURE 2: VEGETATION COMPARISON (percentage) =====
print("\n" + "=" * 70)
print(f"Creating {VEGETATION_INDEX} comparison figure...")
print("=" * 70)

fig2, axes2 = plt.subplots(4, 2, figsize=(18, 24))

for idx, land_type in enumerate(land_types):
    ax_left = axes2[idx, 0]
    ax_right = axes2[idx, 1]

    if land_type not in results_veg or results_veg[land_type]['drought_2012'] is None:
        for ax in [ax_left, ax_right]:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        continue

    # Plot 2012-2016 Vegetation
    if results_veg[land_type]['drought_2012'] is not None:
        data_2012 = results_veg[land_type]['drought_2012']
        im_left = ax_left.imshow(data_2012, cmap='RdYlGn', vmin=-60, vmax=30,
                                 extent=ref_bounds, aspect='auto', interpolation='nearest')
        cbar_left = plt.colorbar(im_left, ax=ax_left, fraction=0.046, pad=0.04)
        cbar_left.set_label('Change (%)', fontsize=10, fontweight='bold')

        valid = data_2012[~np.isnan(data_2012)]
        if len(valid) > 0:
            stats = f'Mean: {np.mean(valid):.1f}%\nSevere (<-40%): {np.sum(valid < -40):,} px'
            # MOVED TO UPPER RIGHT
            ax_left.text(0.98, 0.98, stats, transform=ax_left.transAxes,
                         fontsize=9, verticalalignment='top', horizontalalignment='right',
                         fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_left.set_title(f'{land_type}\n2012-2016 Historic Drought',
                      fontsize=13, fontweight='bold', pad=10)
    ax_left.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_left.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax_left.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Plot 2020-2024 Vegetation
    if results_veg[land_type]['crisis_2020'] is not None:
        data_2020 = results_veg[land_type]['crisis_2020']
        im_right = ax_right.imshow(data_2020, cmap='RdYlGn', vmin=-60, vmax=30,
                                   extent=ref_bounds, aspect='auto', interpolation='nearest')
        cbar_right = plt.colorbar(im_right, ax=ax_right, fraction=0.046, pad=0.04)
        cbar_right.set_label('Change (%)', fontsize=10, fontweight='bold')

        valid = data_2020[~np.isnan(data_2020)]
        if len(valid) > 0:
            stats = f'Mean: {np.mean(valid):.1f}%\nSevere (<-40%): {np.sum(valid < -40):,} px'
            # MOVED TO UPPER RIGHT
            ax_right.text(0.98, 0.98, stats, transform=ax_right.transAxes,
                          fontsize=9, verticalalignment='top', horizontalalignment='right',
                          fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax_right.set_title(f'{land_type}\n2020-2024 Recent Crisis',
                       fontsize=13, fontweight='bold', pad=10)
    ax_right.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax_right.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax_right.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.figtext(0.5, 0.01,
            f'Red = Decline • Green = Increase/Stable • Percentage change from 2001-2011 baseline\n' +
            f'{VEGETATION_INDEX} shows vegetation response to ecohydrological stress',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

plt.suptitle(f'Vegetation Response: {VEGETATION_INDEX} Changes\n' +
             'Historic Drought (2012-2016) vs Recent Crisis (2020-2024)',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.99])
plt.savefig(f'{VEGETATION_INDEX}_drought_comparison_enhanced.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{VEGETATION_INDEX}_drought_comparison_enhanced.pdf', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VEGETATION_INDEX}_drought_comparison_enhanced.png/pdf")

# ===== FIGURE 3: PERSISTENT VULNERABILITY MAP =====
print("\n" + "=" * 70)
print("Creating persistent vulnerability comparison figure...")
print("=" * 70)

fig3, axes3 = plt.subplots(2, 2, figsize=(18, 16))
axes3 = axes3.flatten()

for idx, land_type in enumerate(land_types):
    ax = axes3[idx]

    if land_type not in persistence_results or persistence_results[land_type]['map'] is None:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.axis('off')
        continue

    persist_map = persistence_results[land_type]['map']
    persist_stats = persistence_results[land_type]['stats']

    # Create custom colormap
    from matplotlib.colors import ListedColormap

    colors = ['#2d7f2d', '#ffeb99', '#ff9966', '#cc3333']  # Green, Light yellow, Orange, Dark red
    cmap = ListedColormap(colors)

    im = ax.imshow(persist_map, cmap=cmap, vmin=0, vmax=3,
                   extent=ref_bounds, aspect='auto', interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(['Never Severe', 'Only 2020-2024', 'Only 2012-2016', 'Both Periods'],
                            fontsize=9)

    # Statistics in upper right
    stats_text = (
        f"Persistent: {persist_stats['persistent_vulnerable_npx']:,} px ({persist_stats['persistent_vulnerable_pct']:.1f}%)\n"
        f"Only 2012: {persist_stats['only_2012_npx']:,} px ({persist_stats['only_2012_pct']:.1f}%)\n"
        f"Only 2020: {persist_stats['only_2020_npx']:,} px ({persist_stats['only_2020_pct']:.1f}%)\n"
        f"Resilient: {persist_stats['resilient_npx']:,} px ({persist_stats['resilient_pct']:.1f}%)")

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_title(f'{land_type}\nPersistent Vulnerability Pattern',
                 fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Longitude (°E)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Latitude (°N)', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.figtext(0.5, 0.01,
            'Persistent vulnerability: Pixels showing severe decline (<-40%) in BOTH drought periods\n' +
            'These pixels likely represent species or sites with inherently low drought resilience',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

plt.suptitle(f'Persistent Drought Vulnerability Analysis: {VEGETATION_INDEX}\n' +
             'Comparing 2012-2016 and 2020-2024 Drought Events',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.99])
plt.savefig(f'{VEGETATION_INDEX}_persistent_vulnerability.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{VEGETATION_INDEX}_persistent_vulnerability.pdf', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {VEGETATION_INDEX}_persistent_vulnerability.png/pdf")

plt.show()

# ===== COMPREHENSIVE SUMMARY =====
print("\n" + "=" * 70)
print("ENHANCED COMPARISON SUMMARY")
print("=" * 70)

# Create summary DataFrame
summary_data = []

for land_type in land_types:
    row = {'Land Type': land_type}

    # Spatial statistics
    if land_type in spatial_stats and '2012' in spatial_stats[land_type]:
        stats_2012 = spatial_stats[land_type]['2012']
        if stats_2012:
            row['Mean Change 2012-2016 (%)'] = f"{stats_2012['mean_change']:.1f}"
            row['Severe Decline 2012-2016 (%)'] = f"{stats_2012['severe_decline_pct']:.1f}"
            row['Stable 2012-2016 (%)'] = f"{stats_2012['stable_pct']:.1f}"

    if land_type in spatial_stats and '2020' in spatial_stats[land_type]:
        stats_2020 = spatial_stats[land_type]['2020']
        if stats_2020:
            row['Mean Change 2020-2024 (%)'] = f"{stats_2020['mean_change']:.1f}"
            row['Severe Decline 2020-2024 (%)'] = f"{stats_2020['severe_decline_pct']:.1f}"
            row['Stable 2020-2024 (%)'] = f"{stats_2020['stable_pct']:.1f}"

    # Persistence statistics
    if land_type in persistence_results and persistence_results[land_type]['stats']:
        persist_stats = persistence_results[land_type]['stats']
        row['Persistent Vulnerability (%)'] = f"{persist_stats['persistent_vulnerable_pct']:.1f}"
        row['Resilient (%)'] = f"{persist_stats['resilient_pct']:.1f}"

    summary_data.append(row)

df_summary = pd.DataFrame(summary_data)
print("\n" + df_summary.to_string(index=False))

# Save summary to CSV
df_summary.to_csv(f'{VEGETATION_INDEX}_drought_comparison_summary.csv', index=False)
print(f"\n✓ Saved summary table to: {VEGETATION_INDEX}_drought_comparison_summary.csv")

print("\n" + "=" * 70)
print("KEY FINDINGS:")
print("=" * 70)

for land_type in land_types:
    print(f"\n{land_type}:")

    if land_type in spatial_stats and '2012' in spatial_stats[land_type] and '2020' in spatial_stats[land_type]:
        stats_2012 = spatial_stats[land_type]['2012']
        stats_2020 = spatial_stats[land_type]['2020']

        if stats_2012 and stats_2020:
            # Compare severity
            severity_change = stats_2020['severe_decline_pct'] - stats_2012['severe_decline_pct']
            mean_change = stats_2020['mean_change'] - stats_2012['mean_change']

            print(f"  • Mean vegetation change improved by {mean_change:.1f}% points")
            print(f"  • Severe decline area changed by {severity_change:.1f}% points")

            if severity_change < 0:
                print(f"    → LESS severe in 2020-2024 despite persistent TWS deficit")
            else:
                print(f"    → MORE severe in 2020-2024")

    if land_type in persistence_results and persistence_results[land_type]['stats']:
        persist_stats = persistence_results[land_type]['stats']
        print(f"  • {persist_stats['persistent_vulnerable_pct']:.1f}% of pixels show PERSISTENT vulnerability")
        print(f"  • {persist_stats['resilient_pct']:.1f}% remained resilient in both droughts")

print("\n" + "=" * 70)
print("NOTES:")
print("  • All data regridded to 5km resolution (bilinear resampling)")
print("  • Statistical significance tested using two-sample t-test (α=0.05)")
print("  • Severe decline threshold: < -40% from baseline")
print("  • Persistent vulnerability: severe in BOTH drought periods")
print("  • Three figures generated:")
print("    1. EWBI comparison (forcing)")
print("    2. Vegetation response comparison")
print("    3. Persistent vulnerability map")
print("=" * 70)