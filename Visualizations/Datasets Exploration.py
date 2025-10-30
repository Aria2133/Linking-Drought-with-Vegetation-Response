#!/usr/bin/env python3
"""
Data Resolution Explorer
Analyzes all variables across train/test datasets to identify resolution mismatches
"""

import os
import numpy as np
import pandas as pd
import rasterio
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

LAND_TYPES = ['GRASSLANDS', 'NATURAL_FORESTS', 'SAVANNAS', 'SHRUBLANDS']
VARIABLES = ['GNDVI', 'GOSIF', 'GPP', 'LST', 'NDVI', 'NIRv', 'SPEI', 'TWS', 'VPD']


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
            'SPEI': f"SPEI1_CSIC_CA_{year}_{month:02d}_{land_tag}.tif",
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
            'SPEI': "SPEI_{tag}_SPEI1_CSIC_CA_{year}_{month:02d}.tif",
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


def get_raster_info(filepath: str) -> Dict:
    """Get detailed information about a raster file."""
    try:
        with rasterio.open(filepath) as src:
            return {
                'shape': src.shape,
                'dtype': src.dtypes[0],
                'crs': str(src.crs),
                'bounds': src.bounds,
                'transform': src.transform,
                'nodata': src.nodata,
                'resolution': (src.transform[0], abs(src.transform[4]))
            }
    except Exception as e:
        return {'error': str(e)}


def analyze_variable_resolutions(land_type: str, variable: str) -> Dict:
    """Analyze resolutions for a specific variable across train/test periods."""
    print(f"\nAnalyzing {land_type} - {variable}...")

    train_base = get_train_base_path(land_type)
    test_base = get_test_base_path(land_type)

    train_shapes = {}
    test_shapes = {}
    train_resolutions = {}
    test_resolutions = {}

    # Check training period (2001-2021)
    for year in range(2001, 2022):
        for month in range(1, 13):
            if variable == 'LST' and year < 2003:
                continue
            if variable == 'TWS' and (year < 2002 or (year == 2002 and month < 4)):
                continue

            filepath = find_data_file(train_base, variable, year, month, land_type, is_test=False)
            if filepath and os.path.exists(filepath):
                info = get_raster_info(filepath)
                if 'error' not in info:
                    key = f"{year}-{month:02d}"
                    train_shapes[key] = info['shape']
                    train_resolutions[key] = info['resolution']

    # Check test period (2022-2024)
    for year in range(2022, 2025):
        max_month = 10 if year == 2024 and variable != 'TWS' else 12
        for month in range(1, max_month + 1):
            filepath = find_data_file(test_base, variable, year, month, land_type, is_test=True)
            if filepath and os.path.exists(filepath):
                info = get_raster_info(filepath)
                if 'error' not in info:
                    key = f"{year}-{month:02d}"
                    test_shapes[key] = info['shape']
                    test_resolutions[key] = info['resolution']

    # Analyze patterns
    unique_train_shapes = set(train_shapes.values()) if train_shapes else set()
    unique_test_shapes = set(test_shapes.values()) if test_shapes else set()
    unique_train_res = set(train_resolutions.values()) if train_resolutions else set()
    unique_test_res = set(test_resolutions.values()) if test_resolutions else set()

    result = {
        'land_type': land_type,
        'variable': variable,
        'train_files_found': len(train_shapes),
        'test_files_found': len(test_shapes),
        'train_shapes': list(unique_train_shapes),
        'test_shapes': list(unique_test_shapes),
        'train_resolutions': list(unique_train_res),
        'test_resolutions': list(unique_test_res),
        'shapes_consistent': len(unique_train_shapes) <= 1 and len(unique_test_shapes) <= 1,
        'train_test_match': unique_train_shapes == unique_test_shapes if (
                    unique_train_shapes and unique_test_shapes) else False,
        'resolution_ratio': None
    }

    # Calculate resolution ratio if both exist
    if unique_train_shapes and unique_test_shapes:
        train_shape = list(unique_train_shapes)[0]
        test_shape = list(unique_test_shapes)[0]
        result['resolution_ratio'] = (
            test_shape[0] / train_shape[0],
            test_shape[1] / train_shape[1]
        )

    # Add sample file paths for reference
    if train_shapes:
        first_train_key = list(train_shapes.keys())[0]
        result['sample_train_file'] = find_data_file(train_base, variable,
                                                     int(first_train_key.split('-')[0]),
                                                     int(first_train_key.split('-')[1]),
                                                     land_type, is_test=False)
    if test_shapes:
        first_test_key = list(test_shapes.keys())[0]
        result['sample_test_file'] = find_data_file(test_base, variable,
                                                    int(first_test_key.split('-')[0]),
                                                    int(first_test_key.split('-')[1]),
                                                    land_type, is_test=True)

    return result


def main():
    print("=" * 100)
    print("DATA RESOLUTION EXPLORER - COMPREHENSIVE ANALYSIS")
    print("=" * 100)

    all_results = []

    for land_type in LAND_TYPES:
        print(f"\n{'=' * 100}")
        print(f"LAND TYPE: {land_type}")
        print(f"{'=' * 100}")

        for variable in VARIABLES:
            result = analyze_variable_resolutions(land_type, variable)
            all_results.append(result)

            # Print immediate feedback
            if result['train_files_found'] == 0 and result['test_files_found'] == 0:
                print(f"  ⚠️  NO DATA FOUND")
            elif not result['shapes_consistent']:
                print(f"  ❌ INCONSISTENT SHAPES WITHIN PERIOD")
            elif not result['train_test_match']:
                print(f"  ⚠️  MISMATCH: Train={result['train_shapes']} vs Test={result['test_shapes']}")
                if result['resolution_ratio']:
                    print(f"     Resolution Ratio: {result['resolution_ratio']}")
            else:
                print(f"  ✅ CONSISTENT: Shape={result['train_shapes'][0]}")

    # Create summary DataFrame
    summary_data = []
    for r in all_results:
        summary_data.append({
            'Land_Type': r['land_type'],
            'Variable': r['variable'],
            'Train_Files': r['train_files_found'],
            'Test_Files': r['test_files_found'],
            'Train_Shape': str(r['train_shapes'][0]) if r['train_shapes'] else 'None',
            'Test_Shape': str(r['test_shapes'][0]) if r['test_shapes'] else 'None',
            'Shapes_Consistent': r['shapes_consistent'],
            'Train_Test_Match': r['train_test_match'],
            'Resolution_Ratio': str(r['resolution_ratio']) if r['resolution_ratio'] else 'N/A'
        })

    df = pd.DataFrame(summary_data)

    # Save to CSV
    output_file = 'data_resolution_analysis.csv'
    df.to_csv(output_file, index=False)
    print(f"\n{'=' * 100}")
    print("SUMMARY SAVED TO:", output_file)
    print(f"{'=' * 100}")

    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)

    total_vars = len(all_results)
    consistent_vars = sum(1 for r in all_results if r['shapes_consistent'] and r['train_test_match'])
    mismatch_vars = sum(1 for r in all_results if
                        r['train_files_found'] > 0 and r['test_files_found'] > 0 and not r['train_test_match'])
    missing_vars = sum(1 for r in all_results if r['train_files_found'] == 0 or r['test_files_found'] == 0)

    print(f"Total variable-landtype combinations: {total_vars}")
    print(f"✅ Fully consistent (train=test): {consistent_vars}")
    print(f"⚠️  Resolution mismatches: {mismatch_vars}")
    print(f"❌ Missing data: {missing_vars}")

    # Print detailed mismatch information
    print("\n" + "=" * 100)
    print("DETAILED MISMATCH INFORMATION")
    print("=" * 100)

    for r in all_results:
        if r['train_files_found'] > 0 and r['test_files_found'] > 0 and not r['train_test_match']:
            print(f"\n{r['land_type']} - {r['variable']}:")
            print(f"  Train Shape: {r['train_shapes']}")
            print(f"  Test Shape: {r['test_shapes']}")
            print(f"  Train Resolution: {r['train_resolutions']}")
            print(f"  Test Resolution: {r['test_resolutions']}")
            if r['resolution_ratio']:
                print(f"  Scale Factor (test/train): {r['resolution_ratio']}")
            if 'sample_train_file' in r:
                print(f"  Sample Train: {r['sample_train_file']}")
            if 'sample_test_file' in r:
                print(f"  Sample Test: {r['sample_test_file']}")

    print("\n" + "=" * 100)
    print("FULL DATAFRAME")
    print("=" * 100)
    print(df.to_string())

    print("\n✅ Analysis complete! Check 'data_resolution_analysis.csv' for full results.")


if __name__ == "__main__":
    main()