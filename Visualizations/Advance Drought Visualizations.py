"""
Advanced Multi-Sensor Drought Analysis Visualizations - FIXED CCF & PCA
Fixed methodology + Crystal-clear interpretable figures
Author: Aria Ahmadi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import rasterio
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import ccf, grangercausalitytests
import warnings
import re
warnings.filterwarnings('ignore')

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.weight'] = 'normal'

# ============================================================================
# CONFIGURATION
# ============================================================================

LAND_TYPES = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']
INDICES = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']

# Data directories
BASE_DIR = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
TEST_DIR = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

TRAIN_DIRS = {
    'Grasslands': Path(BASE_DIR) / "CALIFORNIA GRASSLANDS",
    'NaturalForests': Path(BASE_DIR) / "CALIFORNIA NATURAL FORESTS",
    'Savannas': Path(BASE_DIR) / "CALIFORNIA SAVANNAS",
    'Shrublands': Path(BASE_DIR) / "CALIFORNIA SHRUBLANDS"
}

TEST_DIRS = {
    'Grasslands': Path(TEST_DIR) / "Grasslands",
    'NaturalForests': Path(TEST_DIR) / "NaturalForests",
    'Savannas': Path(TEST_DIR) / "Savannas",
    'Shrublands': Path(TEST_DIR) / "Shrublands"
}

OUTPUT_DIR = Path(r"C:\Users\LOQ\OneDrive\Desktop\Advanced_Figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# DATA LOADING
# ============================================================================

def parse_date_from_filename(filename):
    """Robust date parser for all filename patterns"""

    pattern1 = r'(\d{4})\.M(\d{2})'
    match = re.search(pattern1, filename)
    if match:
        return int(match.group(1)), int(match.group(2))

    pattern2 = r'(\d{4})_(\d{2})'
    match = re.search(pattern2, filename)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if 2000 <= year <= 2025 and 1 <= month <= 12:
            return year, month

    pattern3 = r'(\d{6})(?:_|$)'
    match = re.search(pattern3, filename)
    if match:
        date_str = match.group(1)
        year = int(date_str[:4])
        month = int(date_str[4:6])
        if 2000 <= year <= 2025 and 1 <= month <= 12:
            return year, month

    return None, None

def clean_raster_data(data, index):
    """Clean raster data - ONLY REMOVE NODATA VALUES"""
    data = data.astype(np.float32)

    nodata_values = [-9999, -3.4e+38, 32767, -32768, 65535, -32767]
    for nodata in nodata_values:
        data[np.abs(data - nodata) < 1.0] = np.nan

    data[np.abs(data) > 1e10] = np.nan

    return data

def load_timeseries_data(land_type, index):
    """Load spatially-averaged time series"""

    all_files = []

    train_dir = TRAIN_DIRS[land_type] / index
    if train_dir.exists():
        train_files = list(train_dir.glob("*.tif"))
        all_files.extend(train_files)

    if index == 'LST':
        train_filled_dir = TRAIN_DIRS[land_type] / f"{index}_filled"
        if train_filled_dir.exists():
            filled_files = list(train_filled_dir.glob("*.tif"))
            all_files.extend(filled_files)

    test_dir = TEST_DIRS[land_type] / index
    if test_dir.exists():
        test_files = list(test_dir.glob("*.tif"))
        all_files.extend(test_files)

    if not all_files:
        return None

    all_files = sorted(all_files)
    data_list = []

    MIN_VALID_PIXELS = {
        'GOSIF': 5,
        'GPP': 5,
        'TWS': 1,
        'default': 10
    }

    min_pixels = MIN_VALID_PIXELS.get(index, MIN_VALID_PIXELS['default'])

    for file in all_files:
        try:
            filename = file.stem
            year, month = parse_date_from_filename(filename)

            if year is None or month is None:
                continue

            with rasterio.open(file) as src:
                data = src.read(1)
                data = clean_raster_data(data, index)

                valid_pixels = np.sum(~np.isnan(data))

                if valid_pixels >= min_pixels:
                    mean_val = np.nanmean(data)

                    if not np.isnan(mean_val):
                        date = pd.Timestamp(year=year, month=month, day=1)
                        data_list.append({'date': date, 'value': mean_val})

        except Exception as e:
            continue

    if not data_list:
        return None

    df = pd.DataFrame(data_list)
    df = df.drop_duplicates(subset='date', keep='first')
    df = df.set_index('date').sort_index()
    df.columns = [index]

    return df

def load_all_data(land_type):
    """Load all indices for a given land type"""

    print(f"\nLoading data for {land_type}...")

    dfs = []
    for index in INDICES:
        print(f"  Loading {index}...", end=' ')
        df = load_timeseries_data(land_type, index)
        if df is not None and len(df) > 0:
            dfs.append(df)
            print(f"✓ {len(df)} months")
        else:
            print("✗ No data")

    if not dfs:
        return None

    result = dfs[0]
    for df in dfs[1:]:
        result = result.join(df, how='outer')

    result = result.ffill(limit=2)

    print(f"  ✓ Combined: {len(result)} months x {len(result.columns)} indices")

    return result

# ============================================================================
# CROSS-CORRELATION ANALYSIS
# ============================================================================

def compute_ccf_proper(x, y, max_lag=12, alpha=0.05):
    """Proper cross-correlation function using statsmodels"""

    valid = ~(pd.isna(x) | pd.isna(y))
    x_clean = x[valid].values
    y_clean = y[valid].values

    if len(x_clean) < 24:
        return 0, 0, False, [], []

    x_std = (x_clean - np.mean(x_clean)) / np.std(x_clean)
    y_std = (y_clean - np.mean(y_clean)) / np.std(y_clean)

    ccf_pos = ccf(x_std, y_std, adjusted=False)[:max_lag+1]
    ccf_neg = ccf(y_std, x_std, adjusted=False)[1:max_lag+1]

    all_lags = list(range(-max_lag, max_lag+1))
    all_ccf = list(ccf_neg[::-1]) + list(ccf_pos)

    n = len(x_clean)
    conf_interval = stats.norm.ppf(1 - alpha/2) / np.sqrt(n)

    valid_indices = [i for i, c in enumerate(all_ccf) if c > conf_interval]

    if valid_indices:
        best_idx = max(valid_indices, key=lambda i: all_ccf[i])
        best_lag = all_lags[best_idx]
        best_corr = all_ccf[best_idx]
        significant = best_corr > conf_interval
    else:
        best_lag = 0
        best_corr = all_ccf[max_lag]
        significant = False

    return best_lag, best_corr, significant, all_lags, all_ccf

def compute_optimal_lags_fixed(data, max_lag=12):
    """Compute optimal lags using proper CCF methodology"""

    indices = data.columns.tolist()
    n = len(indices)

    lag_matrix = np.zeros((n, n))
    corr_matrix = np.zeros((n, n))
    sig_matrix = np.zeros((n, n), dtype=bool)

    for i, predictor in enumerate(indices):
        for j, target in enumerate(indices):
            if i == j:
                lag_matrix[i, j] = 0
                corr_matrix[i, j] = 1.0
                sig_matrix[i, j] = True
                continue

            x = data[predictor]
            y = data[target]

            best_lag, best_corr, significant, _, _ = compute_ccf_proper(x, y, max_lag)

            lag_matrix[i, j] = best_lag
            corr_matrix[i, j] = best_corr
            sig_matrix[i, j] = significant

    return lag_matrix, corr_matrix, sig_matrix, indices

# ============================================================================
# VISUALIZATION 1: LEAD-LAG MATRIX
# ============================================================================

def plot_lead_lag_matrix_enhanced(land_types_data):
    """Enhanced lead-lag matrix"""

    fig = plt.figure(figsize=(24, 22))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    for idx, (land_type, data) in enumerate(land_types_data.items()):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])

        lag_matrix, corr_matrix, sig_matrix, indices = compute_optimal_lags_fixed(data)

        cmap = plt.cm.RdYlGn_r
        im = ax.imshow(lag_matrix, cmap=cmap, aspect='auto',
                      vmin=-6, vmax=6, interpolation='nearest')

        for i in range(len(indices)):
            for j in range(len(indices)):
                if i != j:
                    lag = int(lag_matrix[i, j])
                    corr = corr_matrix[i, j]
                    sig = sig_matrix[i, j]

                    if abs(lag) > 4:
                        color = 'white'
                        fontsize = 9
                    else:
                        color = 'black'
                        fontsize = 9

                    weight = 'bold' if sig else 'normal'

                    if sig and corr > 0.7:
                        stars = '***'
                    elif sig and corr > 0.5:
                        stars = '**'
                    elif sig:
                        stars = '*'
                    else:
                        stars = ''

                    text = f"{lag:+d}{stars}\n(r={corr:.2f})"

                    ax.text(j, i, text, ha='center', va='center',
                           color=color, fontsize=fontsize, weight=weight)

        ax.set_xticks(range(len(indices)))
        ax.set_yticks(range(len(indices)))
        ax.set_xticklabels(indices, rotation=45, ha='right', fontsize=11)
        ax.set_yticklabels(indices, fontsize=11)

        ax.set_title(f'{land_type}\nOptimal Prediction Lag (months)\n' +
                    '*** r>0.7, ** r>0.5, * significant',
                    fontsize=13, weight='bold', pad=15)
        ax.set_xlabel('Target Index →', fontsize=12, weight='bold')
        ax.set_ylabel('← Predictor Index', fontsize=12, weight='bold')

        ax.set_xticks(np.arange(len(indices))-0.5, minor=True)
        ax.set_yticks(np.arange(len(indices))-0.5, minor=True)
        ax.grid(which='minor', color='white', linewidth=2.5)
        ax.grid(which='major', visible=False)

    cbar = fig.colorbar(im, ax=fig.axes, orientation='horizontal',
                       fraction=0.03, pad=0.08, aspect=50)
    cbar.set_label('Optimal Lag (months)\nNegative = Predictor LEADS (Early Warning Capability!)',
                   fontsize=13, weight='bold')
    cbar.ax.axvline(0, color='black', linewidth=3)

    fig.text(0.5, 0.02,
            'Negative lags indicate EARLY WARNING: predictor changes before target\n' +
            'Example: VPD leads NDVI by -2 months means VPD stress precedes vegetation decline',
            ha='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                     edgecolor='orange', linewidth=2))

    plt.savefig(OUTPUT_DIR / 'Fig1_LeadLag_Matrix_Fixed.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig1_LeadLag_Matrix_Fixed.pdf', bbox_inches='tight')
    print("✓ Saved: Lead-Lag Matrix")
    plt.close()

# ============================================================================
# VISUALIZATION 2: CROSS-CORRELATION HEATMAPS - FIXED
# ============================================================================

def plot_ccf_heatmaps(land_types_data):
    """CCF heatmap with FIXED colorbar positioning"""

    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25,
                         top=0.95, bottom=0.12)  # Reserve space for colorbar

    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    for idx, (land_type, data) in enumerate(land_types_data.items()):
        ax = axes[idx]

        indices = data.columns.tolist()
        max_lag = 6

        ccf_matrix = []
        pair_labels = []

        for i, predictor in enumerate(indices):
            for j, target in enumerate(indices):
                if i < j:
                    x = data[predictor]
                    y = data[target]

                    _, _, _, all_lags, all_ccf = compute_ccf_proper(x, y, max_lag)

                    if len(all_ccf) > 0:
                        ccf_matrix.append(all_ccf)
                        pair_labels.append(f"{predictor}→{target}")

        ccf_matrix = np.array(ccf_matrix)

        im = ax.imshow(ccf_matrix, cmap='RdBu_r', aspect='auto',
                      vmin=-1, vmax=1, interpolation='bilinear')

        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=8)

        lag_labels = list(range(-max_lag, max_lag+1))
        ax.set_xticks(range(len(lag_labels)))
        ax.set_xticklabels(lag_labels, fontsize=11)
        ax.set_xlabel('Lag (months)\nNegative = Predictor LEADS target',
                     fontsize=12, weight='bold')
        ax.set_ylabel('Predictor → Target (Unique Pairs)',
                     fontsize=12, weight='bold')
        ax.set_title(f'{land_type}\nCross-Correlation Function',
                    fontsize=13, weight='bold')

        ax.axvline(max_lag, color='black', linewidth=2.5, linestyle='--',
                  alpha=0.8, label='Lag = 0')
        ax.legend(loc='upper right', fontsize=10)

    # Colorbar positioned BELOW all subplots (fixed!)
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Cross-Correlation Coefficient\n(Red = Positive | Blue = Negative)',
                   fontsize=13, weight='bold')
    cbar.ax.tick_params(labelsize=11)

    plt.savefig(OUTPUT_DIR / 'Fig2_CCF_Heatmaps.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig2_CCF_Heatmaps.pdf', bbox_inches='tight')
    print("✓ Saved: Cross-Correlation Heatmaps (Fixed colorbar)")
    plt.close()

# ============================================================================
# VISUALIZATION 3: CRYSTAL-CLEAR PCA (COMPLETELY REDESIGNED)
# ============================================================================

def plot_pca_enhanced(land_types_data):
    """CRYSTAL CLEAR PCA - Easy to understand!"""

    fig, axes = plt.subplots(4, 3, figsize=(26, 32))

    for idx, (land_type, data) in enumerate(land_types_data.items()):

        data_clean = data.dropna()
        if len(data_clean) < 50:
            continue

        X = data_clean.values
        indices = data_clean.columns.tolist()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca = PCA(n_components=min(len(indices), X_scaled.shape[0]))
        pca.fit(X_scaled)

        # ========== PANEL 1: VARIANCE EXPLAINED (SIMPLE) ==========
        ax1 = axes[idx, 0]
        variance_exp = pca.explained_variance_ratio_[:6] * 100

        x_pos = np.arange(len(variance_exp))
        colors = ['#e74c3c', '#e67e22', '#f39c12', '#95a5a6', '#95a5a6', '#95a5a6']

        bars = ax1.bar(x_pos, variance_exp, color=colors,
                      edgecolor='black', linewidth=2, alpha=0.85)

        for i, (bar, val) in enumerate(zip(bars, variance_exp)):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
                    f'{val:.1f}%',
                    ha='center', va='bottom', fontsize=12, weight='bold')

        ax1.set_xlabel('Principal Component', fontsize=13, weight='bold')
        ax1.set_ylabel('% of Total Information', fontsize=13, weight='bold')
        ax1.set_title(f'{land_type}\nImportance of Each Pattern\n(Higher = More Important)',
                     fontsize=13, weight='bold', pad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f'PC{i+1}' for i in x_pos], fontsize=11)
        ax1.set_ylim(0, max(variance_exp) * 1.25)
        ax1.grid(alpha=0.3, axis='y')

        # Add text explanation
        ax1.text(0.5, 0.95,
                f'First 3 PCs capture {np.sum(variance_exp[:3]):.1f}% of drought patterns',
                transform=ax1.transAxes, ha='center', va='top',
                fontsize=11, weight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

        # ========== PANEL 2: BIPLOT (CLEAR ARROWS) ==========
        ax2 = axes[idx, 1]
        loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])

        # Category colors
        veg_indices = ['NDVI', 'NIRv', 'GNDVI', 'GOSIF', 'GPP']
        atm_indices = ['VPD', 'LST']
        water_indices = ['TWS', 'SPEI']

        for i, index in enumerate(indices):
            if index in veg_indices:
                color = '#2ecc71'
                label_bg = '#d5f4e6'
            elif index in atm_indices:
                color = '#e74c3c'
                label_bg = '#fadbd8'
            elif index in water_indices:
                color = '#3498db'
                label_bg = '#d6eaf8'
            else:
                color = 'gray'
                label_bg = 'lightgray'

            # Draw arrow
            ax2.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                     head_width=0.08, head_length=0.08, fc=color, ec='black',
                     linewidth=2.5, alpha=0.9, zorder=5)

            # Label
            offset_x = loadings[i, 0] * 1.25
            offset_y = loadings[i, 1] * 1.25
            ax2.text(offset_x, offset_y, index,
                    fontsize=11, weight='bold', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=label_bg,
                             edgecolor=color, linewidth=2.5, alpha=0.95),
                    zorder=10)

        # Set explicit axis limits to show full space
        max_val = np.max(np.abs(loadings)) * 1.5
        ax2.set_xlim(-max_val, max_val)
        ax2.set_ylim(-max_val, max_val)

        # Add axis lines at origin
        ax2.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
        ax2.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)

        # Clear axis labels with meaning
        pc1_var = pca.explained_variance_ratio_[0]*100
        pc2_var = pca.explained_variance_ratio_[1]*100

        ax2.set_xlabel(f'PC1 ({pc1_var:.1f}%): Overall Drought Intensity →\n' +
                      '(Right = More drought stress)',
                      fontsize=11, weight='bold')
        ax2.set_ylabel(f'PC2 ({pc2_var:.1f}%): Type of Drought ↑\n' +
                      '(Up = Atmospheric | Down = Water deficit)',
                      fontsize=11, weight='bold')

        ax2.set_title('How Indices Group Together\n' +
                     '(Arrows pointing together = similar behavior)',
                     fontsize=13, weight='bold', pad=10)

        # Legend at bottom
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', edgecolor='black', linewidth=2,
                 label='Vegetation Health'),
            Patch(facecolor='#e74c3c', edgecolor='black', linewidth=2,
                 label='Heat/Dryness Stress'),
            Patch(facecolor='#3498db', edgecolor='black', linewidth=2,
                 label='Water Supply')
        ]
        ax2.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.12), ncol=3,
                  fontsize=10, frameon=True)

        ax2.grid(alpha=0.2, linestyle=':')
        ax2.set_aspect('equal')

        # ========== PANEL 3: LOADINGS (CLEAR HEATMAP) ==========
        ax3 = axes[idx, 2]
        loadings_matrix = pca.components_[:3, :].T

        im = ax3.imshow(loadings_matrix, cmap='RdBu_r', aspect='auto',
                       vmin=-1, vmax=1, interpolation='nearest')

        ax3.set_xticks(range(3))
        ax3.set_xticklabels(['PC1\nDrought\nIntensity',
                            'PC2\nDrought\nType',
                            'PC3\nOther\nVariation'],
                           fontsize=11, weight='bold')
        ax3.set_yticks(range(len(indices)))
        ax3.set_yticklabels(indices, fontsize=11)

        ax3.set_title('How Each Index Loads on PCs\n' +
                     '(Red = Positive | Blue = Negative)',
                     fontsize=13, weight='bold', pad=10)

        # Add values
        for i in range(len(indices)):
            for j in range(3):
                val = loadings_matrix[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                weight = 'bold' if abs(val) > 0.5 else 'normal'

                ax3.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=10, weight=weight)

        cbar3 = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar3.set_label('Loading\nStrength', fontsize=10, weight='bold')

    fig.suptitle('Principal Component Analysis: Simplifying Complex Drought Patterns\n' +
                'LEFT: How important each pattern is | MIDDLE: Which indices behave similarly | RIGHT: Detailed relationships',
                fontsize=16, weight='bold', y=0.995)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig3_PCA_Enhanced.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig3_PCA_Enhanced.pdf', bbox_inches='tight')
    print("✓ Saved: PCA (Crystal clear and understandable!)")
    plt.close()

# ============================================================================
# VISUALIZATION 4: DROUGHT PROPAGATION NETWORK
# ============================================================================

def plot_drought_propagation_network(land_types_data):
    """Network visualization showing lead-lag relationships"""

    fig, axes = plt.subplots(2, 2, figsize=(22, 20))
    axes = axes.flatten()

    for idx, (land_type, data) in enumerate(land_types_data.items()):
        ax = axes[idx]

        lag_matrix, corr_matrix, sig_matrix, indices = compute_optimal_lags_fixed(data)

        n = len(indices)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        positions = {indices[i]: (np.cos(angles[i]), np.sin(angles[i]))
                    for i in range(n)}

        for i, predictor in enumerate(indices):
            for j, target in enumerate(indices):
                if i != j and sig_matrix[i, j]:
                    lag = lag_matrix[i, j]
                    corr = corr_matrix[i, j]

                    if lag < -1 and corr > 0.5:
                        x1, y1 = positions[predictor]
                        x2, y2 = positions[target]

                        width = corr * 3
                        alpha = min(1.0, corr)

                        if lag < -3:
                            color = '#e74c3c'
                        elif lag < -2:
                            color = '#e67e22'
                        else:
                            color = '#f39c12'

                        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                  arrowprops=dict(arrowstyle='->', lw=width,
                                                color=color, alpha=alpha,
                                                connectionstyle='arc3,rad=0.2'))

        for i, index in enumerate(indices):
            x, y = positions[index]

            if index in ['NDVI', 'NIRv', 'GNDVI', 'GOSIF', 'GPP']:
                node_color = '#2ecc71'
            elif index in ['VPD', 'LST']:
                node_color = '#e74c3c'
            else:
                node_color = '#3498db'

            circle = plt.Circle((x, y), 0.15, color=node_color, ec='black',
                              linewidth=3, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, index, ha='center', va='center',
                   fontsize=11, weight='bold', color='white', zorder=11)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{land_type}\nDrought Propagation Network\n(Arrows show lead relationships >1 month)',
                    fontsize=13, weight='bold', pad=15)

        ax.text(0, -1.35,
               '← Arrow direction shows propagation\n' +
               'Thickness = correlation strength\n' +
               'Red arrows = strong early warning (>3 months)',
               ha='center', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'Fig4_Propagation_Network.png', dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'Fig4_Propagation_Network.pdf', bbox_inches='tight')
    print("✓ Saved: Drought Propagation Network")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""

    print("="*70)
    print("ADVANCED DROUGHT ANALYSIS - FIXED CCF & PCA")
    print("="*70)
    print()

    print("Loading data...")
    land_types_data = {}

    for land_type in LAND_TYPES:
        data = load_all_data(land_type)
        if data is not None and len(data) > 50:
            land_types_data[land_type] = data
            print(f"  ✓ {land_type}: {len(data)} months ({data.index.min().year}-{data.index.max().year})")

    if not land_types_data:
        print("\nERROR: No data loaded!")
        return

    print(f"\n✓ Successfully loaded {len(land_types_data)} land types")
    print("\nCreating crystal-clear visualizations...\n")

    try:
        plot_lead_lag_matrix_enhanced(land_types_data)
    except Exception as e:
        print(f"Error in lead-lag matrix: {e}")

    try:
        plot_ccf_heatmaps(land_types_data)
    except Exception as e:
        print(f"Error in CCF heatmaps: {e}")

    try:
        plot_pca_enhanced(land_types_data)
    except Exception as e:
        print(f"Error in PCA: {e}")

    try:
        plot_drought_propagation_network(land_types_data)
    except Exception as e:
        print(f"Error in network: {e}")

    print("\n" + "="*70)
    print("✓ COMPLETE! All figures saved to:")
    print(f"  {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()