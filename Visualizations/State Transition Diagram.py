import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

"""
STATE TRANSITION HEATMAP (EWBI-based)
Clean, readable visualization of ecosystem state transitions
Uses deseasonalized EWBI (composite: SPEI + TWS − VPD − LST) and NIRv anomalies
"""

# -----------------------
# Configuration
# -----------------------
land_types = ['Grasslands', 'NaturalForests', 'Savannas', 'Shrublands']

# Directories
train_base = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data"
test_base = r"C:\Users\LOQ\OneDrive\Desktop\Test Data\Masked_LandTypes"

print("=" * 70)
print("STATE TRANSITION ANALYSIS - HEATMAP (EWBI vs NIRv)")
print("=" * 70)

# -----------------------
# Helpers
# -----------------------
def get_files(land_type, index):
    folder_map = {
        'Grasslands': 'CALIFORNIA GRASSLANDS',
        'NaturalForests': 'CALIFORNIA NATURAL FORESTS',
        'Savannas': 'CALIFORNIA SAVANNAS',
        'Shrublands': 'CALIFORNIA SHRUBLANDS'
    }
    folder = folder_map[land_type]

    # Allow VPDmax logical name while folder is "VPD"
    folder_index = 'VPD' if index == 'VPDmax' else index

    train_path = os.path.join(train_base, folder, folder_index)
    test_path  = os.path.join(test_base,  land_type, folder_index)

    files = []
    if os.path.exists(train_path):
        files.extend(glob.glob(os.path.join(train_path, "*.tif")))
    if os.path.exists(test_path):
        files.extend(glob.glob(os.path.join(test_path, "*.tif")))

    return sorted(list(set(files)))


def clean_data(data, index):
    data = data.astype(np.float64)
    nodata_values = [-9999, -3.4e+38, 32767, -32768, 65535, -32767]
    for nodata in nodata_values:
        data[np.abs(data - nodata) < 1.0] = np.nan
    data[np.abs(data) > 10000] = np.nan

    if index == 'SPEI':
        data[(data < -5) | (data > 5)] = np.nan
    elif index == 'NIRv':
        data[(data < -0.1) | (data > 0.6)] = np.nan
    elif index == 'TWS':
        data[(data < -100) | (data > 100)] = np.nan
    elif index in ['VPD', 'VPDmax']:
        data[(data < 0) | (data > 10)] = np.nan
    elif index == 'LST':
        data[(data < -50) | (data > 60)] = np.nan

    return data


def load_series(land_type, index, ref_shape, ref_transform, ref_crs):
    """Return land-type spatial-mean time series for an index (arrays aligned by file order)."""
    series = []
    files = get_files(land_type, index)
    for filepath in files:
        try:
            with rasterio.open(filepath) as src:
                data = src.read(1)
                if ref_shape is not None and src.shape != ref_shape:
                    output = np.empty(ref_shape, dtype=np.float32)
                    reproject(source=data, destination=output,
                              src_transform=src.transform, src_crs=src.crs,
                              dst_transform=ref_transform, dst_crs=ref_crs,
                              resampling=Resampling.bilinear,
                              src_nodata=np.nan, dst_nodata=np.nan)
                    data = output
                data = clean_data(data, index)
                series.append(np.nanmean(data))
        except Exception:
            continue
    return np.array(series, dtype=float)


def month_indices_for_length(n):
    """Infer month indices assuming contiguous monthly series (same as your original code)."""
    return np.array([i % 12 for i in range(n)])


def deseasonalize(series, month_idx):
    """Remove seasonality via calendar-month median, matching original approach."""
    clim = {}
    for m in range(12):
        vals = series[month_idx == m]
        clim[m] = np.nanmedian(vals) if vals.size > 0 else 0.0
    monthly_clim = np.array([clim[m] for m in month_idx])
    return series - monthly_clim


def zscore(series):
    mu = np.nanmean(series)
    sd = np.nanstd(series)
    if sd is None or np.isnan(sd) or sd == 0:
        return np.zeros_like(series)
    return (series - mu) / sd


def classify_state(ewbi_val, nirv_val, ewbi_med, nirv_med):
    """Classify ecosystem state based on EWBI and NIRv ANOMALIES (after deseasonalization)."""
    if ewbi_val > ewbi_med:
        if nirv_val > nirv_med:
            return "Wet/Healthy"
        else:
            return "Wet/Suppressed"
    else:
        if nirv_val > nirv_med:
            return "Dry/Resilient"
        else:
            return "Dry/Stressed"

# -----------------------
# Load reference grid (from any NIRv file)
# -----------------------
ref_transform = None
ref_crs = None
ref_shape = None

for land_type in land_types:
    files = get_files(land_type, 'NIRv')
    if files:
        with rasterio.open(files[0]) as src:
            ref_transform = src.transform
            ref_crs = src.crs
            ref_shape = (src.height, src.width)
            break

# -----------------------
# Extract time series and calculate transitions (EWBI vs NIRv)
# -----------------------
all_transitions = {}

for land_type in land_types:
    print(f"\nProcessing {land_type}...")

    # Load spatial-mean monthly series
    spei_series = load_series(land_type, 'SPEI',   ref_shape, ref_transform, ref_crs)
    tws_series  = load_series(land_type, 'TWS',    ref_shape, ref_transform, ref_crs)
    vpd_series  = load_series(land_type, 'VPDmax', ref_shape, ref_transform, ref_crs)  # folder name handled
    lst_series  = load_series(land_type, 'LST',    ref_shape, ref_transform, ref_crs)
    nirv_series = load_series(land_type, 'NIRv',   ref_shape, ref_transform, ref_crs)

    # Align series by minimum common length (assumes same ordering/frequency as in original script)
    lengths = [len(spei_series), len(tws_series), len(vpd_series), len(lst_series), len(nirv_series)]
    min_len = np.min(lengths) if all(l > 0 for l in lengths) else 0
    if min_len < 24:
        print(f"  Skipping {land_type}: insufficient overlapping data ({min_len} months).")
        continue

    spei_series = spei_series[:min_len]
    tws_series  = tws_series[:min_len]
    vpd_series  = vpd_series[:min_len]
    lst_series  = lst_series[:min_len]
    nirv_series = nirv_series[:min_len]

    # Month indices (same assumption as your original code)
    month_idx = month_indices_for_length(min_len)

    # Deseasonalize (monthly medians)
    spei_anom = deseasonalize(spei_series, month_idx)
    tws_anom  = deseasonalize(tws_series,  month_idx)
    vpd_anom  = deseasonalize(vpd_series,  month_idx)
    lst_anom  = deseasonalize(lst_series,  month_idx)
    nirv_anom = deseasonalize(nirv_series, month_idx)

    # Build EWBI anomaly time series using z-scored component anomalies (timewise), consistent with composite logic
    ewbi_comp = zscore(spei_anom) + zscore(tws_anom) - zscore(vpd_anom) - zscore(lst_anom)

    # Thresholds (median splits, same as original approach)
    ewbi_median = np.nanmedian(ewbi_comp)
    nirv_median = np.nanmedian(nirv_anom)

    # Classify monthly states
    states = [classify_state(e, n, ewbi_median, nirv_median) for e, n in zip(ewbi_comp, nirv_anom)]

    # Count transitions
    state_names = ["Wet/Healthy", "Wet/Suppressed", "Dry/Resilient", "Dry/Stressed"]
    transition_matrix = {(s1, s2): 0 for s1 in state_names for s2 in state_names}

    for i in range(len(states) - 1):
        transition_matrix[(states[i], states[i + 1])] += 1

    # Row-normalized probabilities
    transition_probs = {}
    for s1 in state_names:
        total = sum(transition_matrix[(s1, s2)] for s2 in state_names)
        for s2 in state_names:
            transition_probs[(s1, s2)] = (transition_matrix[(s1, s2)] / total) if total > 0 else 0.0

    # State occupancy (% of months)
    from collections import Counter
    counts = Counter(states)
    occupancy = {s: (counts[s] / len(states) * 100.0) for s in state_names}

    all_transitions[land_type] = {
        'probs': transition_probs,
        'states': states,
        'state_names': state_names,
        'occupancy': occupancy
    }

    print(f"  Total transitions: {len(states) - 1}")

# -----------------------
# Heatmap visualization
# -----------------------
fig, axes = plt.subplots(2, 2, figsize=(20, 18))
axes = axes.flatten()
state_names = ["Wet/Healthy", "Wet/Suppressed", "Dry/Resilient", "Dry/Stressed"]

for idx, land_type in enumerate(land_types):
    ax = axes[idx]
    if land_type not in all_transitions:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.axis('off')
        continue

    probs = all_transitions[land_type]['probs']
    occupancy = all_transitions[land_type]['occupancy']

    # Make probability matrix for heatmap
    matrix = np.zeros((4, 4))
    for i, s1 in enumerate(state_names):
        for j, s2 in enumerate(state_names):
            matrix[i, j] = probs[(s1, s2)] * 100.0

    sns.heatmap(matrix, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=state_names, yticklabels=state_names,
                cbar_kws={'label': 'Transition Probability (%)', 'shrink': 0.8},
                linewidths=2, linecolor='white', square=True,
                vmin=0, vmax=100, ax=ax, annot_kws={'size': 11, 'weight': 'bold'})

    ax.set_xlabel('To State →', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('From State →', fontsize=12, fontweight='bold', labelpad=10)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)

    # Title with occupancy
    title_text = f'{land_type}\nState Occupancy: '
    title_text += ', '.join([f'{s.split("/")[1]}: {occupancy[s]:.1f}%'
                             for s in state_names])
    ax.set_title(title_text, fontsize=13, fontweight='bold', pad=15)

    # Highlight persistence (diagonal)
    for i in range(4):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', linewidth=3, zorder=10))

plt.suptitle('Ecosystem State Transition Probabilities (Deseasonalized)\n'
             'Based on EWBI (forcing composite) vs NIRv (response) • Red boxes = state persistence • Each row sums to 100%',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0.02, 1, 0.96])
plt.subplots_adjust(hspace=0.4, wspace=0.3)

plt.savefig('state_transition_heatmap_EWBI.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('state_transition_heatmap_EWBI.pdf', dpi=300, bbox_inches='tight', facecolor='white')
print("\n✓ Saved: state_transition_heatmap_EWBI.png/pdf")
plt.show()

print("\n" + "=" * 70)
print("HEATMAP INTERPRETATION")
print("=" * 70)
print("• Each cell shows probability (%) of transitioning from row state to column state")
print("• Red boxes (diagonal) = probability of staying in the same state (persistence)")
print("• Darker colors = higher transition probability; each row sums to 100%")
print("• State occupancy shows % of time spent in each state")
print("• Forcing dimension = EWBI (z-scored component anomalies: SPEI + TWS − VPD − LST); Response = NIRv anomaly")
print("=" * 70)
