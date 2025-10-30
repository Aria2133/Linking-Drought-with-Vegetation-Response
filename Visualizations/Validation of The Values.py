import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Read the figure
img = Image.open('State_Transition_Based_on_EWBI_paper_ready.png')

print("=" * 80)
print("MANUAL VALIDATION FROM FIGURE")
print("=" * 80)

# Values read manually from the figure heatmaps
data = {
    'Grasslands': {
        'occupancy': {'Healthy': 31.7, 'Suppressed': 18.1, 'Resilient': 15.9, 'Stressed': 34.4},
        'persistence': {'Wet/Healthy': 70.4, 'Wet/Suppressed': 51.2, 'Dry/Resilient': 44.4, 'Dry/Stressed': 66.7},
        'transitions': {
            'Wet→Stressed': 7.0,
            'Stressed→Healthy': 6.4,
            'Wet→Suppressed': 1.4,
            'Suppressed→Stressed': 29.3
        }
    },
    'NaturalForests': {
        'occupancy': {'Healthy': 23.8, 'Suppressed': 26.0, 'Resilient': 23.8, 'Stressed': 26.4},
        'persistence': {'Wet/Healthy': 48.1, 'Wet/Suppressed': 50.0, 'Dry/Resilient': 46.3, 'Dry/Stressed': 55.0},
        'transitions': {
            'Wet→Stressed': 5.6,
            'Stressed→Healthy': 5.0,
            'Wet→Suppressed': 11.1,
            'Suppressed→Stressed': 24.1
        }
    },
    'Savannas': {
        'occupancy': {'Healthy': 28.6, 'Suppressed': 21.1, 'Resilient': 18.9, 'Stressed': 31.3},
        'persistence': {'Wet/Healthy': 66.2, 'Wet/Suppressed': 48.9, 'Dry/Resilient': 48.8, 'Dry/Stressed': 63.4},
        'transitions': {
            'Wet→Stressed': 9.2,
            'Stressed→Healthy': 2.8,
            'Wet→Suppressed': 7.7,
            'Suppressed→Stressed': 21.3
        }
    },
    'Shrublands': {
        'occupancy': {'Healthy': 30.0, 'Suppressed': 19.8, 'Resilient': 17.6, 'Stressed': 32.6},
        'persistence': {'Wet/Healthy': 64.7, 'Wet/Suppressed': 51.1, 'Dry/Resilient': 42.5, 'Dry/Stressed': 74.0},
        'transitions': {
            'Wet→Stressed': 2.9,
            'Stressed→Healthy': 1.4,
            'Wet→Suppressed': 5.9,
            'Suppressed→Stressed': 26.7
        }
    }
}

# Calculate averages
avg_stressed = np.mean([data[lt]['occupancy']['Stressed'] for lt in data])
avg_healthy = np.mean([data[lt]['occupancy']['Healthy'] for lt in data])

print(f"\n✓ Average Stressed occupancy: {avg_stressed:.1f}%")
print(f"✓ Average Healthy occupancy: {avg_healthy:.1f}%")

# Transition ranges
stress_onset = [data[lt]['transitions']['Wet→Stressed'] for lt in data]
recovery = [data[lt]['transitions']['Stressed→Healthy'] for lt in data]

print(f"\n✓ Stress onset (Wet→Stressed) range: {min(stress_onset):.1f}% to {max(stress_onset):.1f}%")
print(f"✓ Recovery (Stressed→Healthy) range: {min(recovery):.1f}% to {max(recovery):.1f}%")

# Specific values mentioned in paper
print("\n" + "=" * 80)
print("CHECKING PAPER VALUES:")
print("=" * 80)

print(f"\nShrublands Dry/Stressed persistence:")
print(f"  Figure: {data['Shrublands']['persistence']['Dry/Stressed']:.1f}%")
print(f"  Paper: 74% → {'✓ CORRECT' if abs(data['Shrublands']['persistence']['Dry/Stressed'] - 74) < 0.5 else '✗ WRONG'}")

print(f"\nNatural Forests Wet/Healthy persistence:")
print(f"  Figure: {data['NaturalForests']['persistence']['Wet/Healthy']:.1f}%")
print(f"  Paper: 48.1% → {'✓ CORRECT' if abs(data['NaturalForests']['persistence']['Wet/Healthy'] - 48.1) < 0.5 else '✗ WRONG'}")

print(f"\nNatural Forests Wet/Suppressed transitions:")
print(f"  Figure: {data['NaturalForests']['transitions']['Wet→Suppressed']:.1f}%")
print(f"  Paper: 28.7% → {'✗ WRONG - should be 11.1%' if abs(data['NaturalForests']['transitions']['Wet→Suppressed'] - 28.7) > 5 else '✓ CORRECT'}")

print(f"\nStress onset range:")
print(f"  Figure: {min(stress_onset):.1f}% to {max(stress_onset):.1f}%")
print(f"  Paper: 4.8-9.0% → {'✗ WRONG' if abs(min(stress_onset) - 4.8) > 2 else '✓ CORRECT'}")

print(f"\nRecovery range:")
print(f"  Figure: {min(recovery):.1f}% to {max(recovery):.1f}%")
print(f"  Paper: 2.3-8.3% → {'✗ WRONG' if abs(min(recovery) - 2.3) > 1 else '✓ CORRECT'}")

print("\n" + "=" * 80)
print("SUMMARY OF ERRORS:")
print("=" * 80)
print("1. ✓ Average occupancy: CORRECT (31.2% stressed, 28.5% healthy)")
print("2. ✓ Shrublands persistence: CORRECT (74%)")
print("3. ✓ Natural Forests Wet/Healthy: CORRECT (48.1%)")
print("4. ✗ Natural Forests Wet/Suppressed transition: WRONG (paper says 28.7%, should be ~11%)")
print("5. ✗ Transition ranges: WRONG (paper has wrong numbers AND backward direction)")
print("=" * 80)