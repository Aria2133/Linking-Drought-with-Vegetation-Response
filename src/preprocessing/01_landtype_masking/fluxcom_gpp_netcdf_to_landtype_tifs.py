import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd
from rasterio.transform import from_bounds
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Install cftime if not available
try:
    import cftime
except ImportError:
    print("Installing cftime package...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'cftime'])
    import cftime


def convert_time_to_datetime(time_values, time_units, calendar='standard'):
    """Convert NetCDF time values to datetime objects using cftime"""
    try:
        dates = cftime.num2date(time_values, time_units, calendar=calendar)
        datetime_list = []
        for date in dates:
            if hasattr(date, 'year') and hasattr(date, 'month') and hasattr(date, 'day'):
                if date.year >= 1900:
                    datetime_list.append(datetime(date.year, date.month, date.day))
                else:
                    datetime_list.append(date)
            else:
                datetime_list.append(date)
        return datetime_list
    except Exception as e:
        print(f"Error converting time: {e}")
        start_year = 1982
        dates = []
        for i, time_val in enumerate(time_values):
            year = start_year + (i // 12)
            month = (i % 12) + 1
            dates.append(datetime(year, month, 1))
        return dates


def read_and_reproject_mask(mask_path, target_transform, target_shape, target_crs='EPSG:4326'):
    """Read and reproject land type mask to match GPP data grid"""
    with rasterio.open(mask_path) as src:
        mask_data = src.read(1)
        reprojected_mask = np.zeros(target_shape, dtype=np.uint8)

        reproject(
            source=mask_data,
            destination=reprojected_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        return reprojected_mask > 0


def create_geotiff_transform(lons, lats):
    """Create affine transform for the GPP data"""
    lon_res = abs(lons[1] - lons[0])
    lat_res = abs(lats[1] - lats[0])

    transform = from_bounds(
        west=float(lons.min() - lon_res / 2),
        south=float(lats.min() - lat_res / 2),
        east=float(lons.max() + lon_res / 2),
        north=float(lats.max() + lat_res / 2),
        width=len(lons),
        height=len(lats)
    )

    return transform


def convert_gpp_units(gpp_data, input_units, output_units):
    """
    Convert GPP data from input units to desired output units

    Common conversions:
    - kg C m-2 s-1 to g C m-2 day-1: multiply by 86400 * 1000
    - kg C m-2 s-1 to kg C m-2 year-1: multiply by 31536000
    - kg C m-2 s-1 to g C m-2 month-1: multiply by ~2628000 * 1000 (30.4 days avg)
    """

    if input_units == "kg m-2 s-1" and output_units == "g C m-2 day-1":
        # Convert kg/m2/s to g/m2/day
        conversion_factor = 86400 * 1000  # seconds per day * g per kg
        converted_data = gpp_data * conversion_factor
        return converted_data, conversion_factor

    elif input_units == "kg m-2 s-1" and output_units == "kg C m-2 year-1":
        # Convert kg/m2/s to kg/m2/year
        conversion_factor = 31536000  # seconds per year (365.25 days)
        converted_data = gpp_data * conversion_factor
        return converted_data, conversion_factor

    elif input_units == "kg m-2 s-1" and output_units == "g C m-2 month-1":
        # Convert kg/m2/s to g/m2/month
        conversion_factor = 2628000 * 1000  # ~30.4 days * 86400 sec * 1000 g/kg
        converted_data = gpp_data * conversion_factor
        return converted_data, conversion_factor

    elif input_units == "kg m-2 s-1" and output_units == "Mg C ha-1 year-1":
        # Convert kg/m2/s to Mg/ha/year (common in forest literature)
        # 1 kg/m2 = 10 Mg/ha, so kg/m2/s * 31536000 * 10 = Mg/ha/year
        conversion_factor = 31536000 * 10
        converted_data = gpp_data * conversion_factor
        return converted_data, conversion_factor
    else:
        # No conversion needed
        return gpp_data, 1.0


def validate_gpp_values(gpp_data, landtype_name, units="g C m-2 day-1"):
    """Validate that GPP values are in expected range for given units"""
    valid_data = gpp_data[~np.isnan(gpp_data)]

    if len(valid_data) == 0:
        return False, "No valid data found"

    min_val, max_val = np.min(valid_data), np.max(valid_data)
    mean_val = np.mean(valid_data)

    # Expected ranges based on units
    if units == "g C m-2 day-1":
        expected_min = 0.01  # Very low GPP
        expected_max = 20.0  # Very high GPP (tropical forests)
        typical_range = (0.1, 10.0)

    elif units == "kg C m-2 year-1":
        expected_min = 0.01  # Very low GPP
        expected_max = 7.0  # Very high GPP
        typical_range = (0.1, 4.0)

    elif units == "g C m-2 month-1":
        expected_min = 1.0  # Very low GPP
        expected_max = 600.0  # Very high GPP
        typical_range = (5.0, 300.0)

    elif units == "Mg C ha-1 year-1":
        expected_min = 0.1  # Very low GPP
        expected_max = 70.0  # Very high GPP
        typical_range = (1.0, 40.0)
    else:
        # Default validation for unknown units
        expected_min = 1e-12
        expected_max = 1e6
        typical_range = (1e-10, 1e6)

    # Check if values are in reasonable range
    if min_val < 0:
        return False, f"Negative GPP values found (min: {min_val:.3f})"

    if max_val > expected_max:
        return False, f"GPP values too high (max: {max_val:.3f}, expected < {expected_max:.3f})"

    if mean_val < expected_min:
        return False, f"GPP values too low (mean: {mean_val:.3f}, expected > {expected_min:.3f})"

    # Values are valid
    status_msg = f"✓ Values valid: range {min_val:.3f} to {max_val:.3f}, mean {mean_val:.3f} {units}"

    # Add context about the values
    if typical_range[0] <= mean_val <= typical_range[1]:
        status_msg += " (typical GPP range)"
    elif mean_val > typical_range[1]:
        status_msg += " (high productivity area)"
    else:
        status_msg += " (low productivity area)"

    return True, status_msg


def clip_gpp_data_for_landtype(gpp_data, land_mask, dates, output_dir, landtype_name,
                               lons, lats, transform, input_units="kg m-2 s-1", output_units="g C m-2 day-1"):
    """Clip GPP data using land type mask and save as monthly TIF files with unit conversion"""

    landtype_dir = os.path.join(output_dir, landtype_name)
    os.makedirs(landtype_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"PROCESSING {landtype_name.upper()}")
    print(f"{'=' * 60}")
    print(f"Land mask has {np.sum(land_mask)} valid pixels")
    print(f"Converting from {input_units} to {output_units}")

    # Test first time step and validate values
    sample_gpp = gpp_data[0]
    sample_clipped = np.full_like(sample_gpp, np.nan, dtype=np.float64)
    sample_clipped[land_mask] = sample_gpp[land_mask]

    # Convert units for validation and output
    converted_sample, conversion_factor = convert_gpp_units(sample_clipped, input_units, output_units)
    print(f"Conversion factor: {conversion_factor}")

    # Validate converted values
    is_valid, validation_msg = validate_gpp_values(converted_sample, landtype_name, output_units)
    print(f"Validation: {validation_msg}")

    if not is_valid:
        print(f"❌ ERROR: {validation_msg}")
        return 0

    # Show before/after example
    original_mean = np.nanmean(sample_clipped)
    converted_mean = np.nanmean(converted_sample)
    print(f"Unit conversion example:")
    print(f"  Original: {original_mean:.2e} {input_units}")
    print(f"  Converted: {converted_mean:.3f} {output_units}")

    # Process each time step
    files_created = 0
    processing_stats = {
        'min_values': [],
        'max_values': [],
        'mean_values': [],
        'valid_pixels': []
    }

    for i, date in enumerate(dates):
        gpp_timestep = gpp_data[i]

        # Apply land type mask
        clipped_data = np.full_like(gpp_timestep, np.nan, dtype=np.float64)
        clipped_data[land_mask] = gpp_timestep[land_mask]

        # Convert units
        converted_data, _ = convert_gpp_units(clipped_data, input_units, output_units)

        # Collect statistics (on converted data)
        valid_data = converted_data[~np.isnan(converted_data)]
        if len(valid_data) > 0:
            processing_stats['min_values'].append(np.min(valid_data))
            processing_stats['max_values'].append(np.max(valid_data))
            processing_stats['mean_values'].append(np.mean(valid_data))
            processing_stats['valid_pixels'].append(len(valid_data))

        # Create output filename
        if hasattr(date, 'strftime'):
            date_str = date.strftime("%Y_%m")
        else:
            date_str = f"{date.year:04d}_{date.month:02d}"

        output_filename = f"{landtype_name}_GPP_{date_str}.tif"
        output_path = os.path.join(landtype_dir, output_filename)

        # Save converted data as GeoTIFF
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=converted_data.shape[0],
                width=converted_data.shape[1],
                count=1,
                dtype=np.float64,
                crs='EPSG:4326',
                transform=transform,
                compress='lzw',
                nodata=np.nan
        ) as dst:
            dst.write(converted_data, 1)

            # Enhanced metadata with converted units
            dst.set_band_description(1, f'GPP for {landtype_name} - {date_str} ({output_units})')

            # Calculate statistics for metadata
            if len(valid_data) > 0:
                min_val, max_val, mean_val = np.min(valid_data), np.max(valid_data), np.mean(valid_data)

                dst.update_tags(
                    1,
                    units=output_units,
                    original_units=input_units,
                    conversion_factor=f"{conversion_factor}",
                    long_name=f'Gross Primary Production - {landtype_name}',
                    date=date_str,
                    landtype=landtype_name,
                    valid_pixels=str(len(valid_data)),
                    data_min=f"{min_val:.6f}",
                    data_max=f"{max_val:.6f}",
                    data_mean=f"{mean_val:.6f}",
                    data_std=f"{np.std(valid_data):.6f}",
                    processing_note="Units converted from original NetCDF for better readability"
                )

        files_created += 1

        # Print progress every 5 years
        if i % 60 == 0:
            print(f"  Progress: {i + 1}/{len(dates)} time steps processed...")

    # Final statistics
    if processing_stats['mean_values']:
        overall_mean = np.mean(processing_stats['mean_values'])
        overall_min = np.min(processing_stats['min_values'])
        overall_max = np.max(processing_stats['max_values'])
        mean_valid_pixels = np.mean(processing_stats['valid_pixels'])

        print(f"\n📊 FINAL STATISTICS FOR {landtype_name}:")
        print(f"   Files created: {files_created}")
        print(f"   Value range: {overall_min:.3f} to {overall_max:.3f} {output_units}")
        print(f"   Overall mean: {overall_mean:.3f} {output_units}")
        print(f"   Average valid pixels per timestep: {mean_valid_pixels:.0f}")
        print(f"   Output directory: {landtype_dir}")

    return files_created


def main():
    # File paths
    gpp_file = r"<DOWNLOADS>\202582212913EnsembleGPP_GL.nc"

    mask_files = {
        "Grasslands": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Grasslands_2021.tif",
        "NaturalForests": r"<DATA_ROOT>\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
        "Savannas": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Savannas_2021.tif",
        "Shrublands": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Shrublands_2021.tif"
    }

    output_dir = r"<DATA_ROOT>\Data\GPP_Clipped_Output"
    os.makedirs(output_dir, exist_ok=True)

    # UNIT SELECTION - Choose your preferred output units
    # Common options used in scientific papers:
    output_units_options = {
        1: "g C m-2 day-1",  # Most common in daily GPP studies (0.1-20 range)
        2: "kg C m-2 year-1",  # Common in annual studies (0.1-7 range)
        3: "g C m-2 month-1",  # Monthly totals (5-600 range)
        4: "Mg C ha-1 year-1"  # Forest literature (1-70 range, 1 Mg = 1000 kg, 1 ha = 10000 m2)
    }

    print("=" * 80)
    print("GPP DATA PROCESSING FOR CALIFORNIA LAND TYPES")
    print("=" * 80)
    print("Available output units (commonly used in scientific papers):")
    for key, units in output_units_options.items():
        print(f"  {key}. {units}")

    # Select output units (you can change this default)
    selected_units = 1  # Change this number to select different units
    output_units = output_units_options[selected_units]

    print(f"\n🎯 Selected output units: {output_units}")
    print(f"   (You can change 'selected_units' variable in the code to choose different units)")

    print(f"\n📄 Loading GPP NetCDF file...")

    # Open and examine the dataset
    with xr.open_dataset(gpp_file, decode_times=False) as ds:
        print(f"✅ Dataset loaded successfully")
        print(f"   GPP data shape: {ds['gpp'].shape}")
        print(f"   Time range: {len(ds.time)} time steps")

        # Check units and attributes
        gpp_var = ds['gpp']
        input_units = gpp_var.attrs.get('units', 'kg m-2 s-1')  # Default assumption
        print(f"   Original units: {input_units}")
        print(f"   Output units: {output_units}")

        # Get coordinates and time
        lons = ds.lon.values
        lats = ds.lat.values
        time_values = ds.time.values

        # Convert time
        time_units = ds.time.attrs.get('units', 'days since 1582-10-14 00:00:00')
        time_calendar = ds.time.attrs.get('calendar', 'standard')
        dates = convert_time_to_datetime(time_values, time_units, time_calendar)
        print(f"   Date range: {dates[0]} to {dates[-1]}")

        # Create geospatial transform
        transform = create_geotiff_transform(lons, lats)

        # Get GPP data
        gpp_data = ds['gpp'].values

        print(f"   Coordinate ranges:")
        print(f"     Longitude: {lons.min():.3f}° to {lons.max():.3f}°")
        print(f"     Latitude: {lats.min():.3f}° to {lats.max():.3f}°")

        # Show unit conversion preview
        sample_data = gpp_data[0]
        valid_sample = sample_data[~np.isnan(sample_data)]
        converted_sample, conversion_factor = convert_gpp_units(valid_sample, input_units, output_units)

        print(f"\n🔄 UNIT CONVERSION PREVIEW:")
        print(f"   Conversion factor: {conversion_factor}")
        print(f"   Original range: {np.min(valid_sample):.2e} to {np.max(valid_sample):.2e} {input_units}")
        print(f"   Converted range: {np.min(converted_sample):.3f} to {np.max(converted_sample):.3f} {output_units}")
        print(f"   Original mean: {np.mean(valid_sample):.2e} {input_units}")
        print(f"   Converted mean: {np.mean(converted_sample):.3f} {output_units}")

        # Process each land type
        total_files = 0
        successful_landtypes = []

        for landtype_name, mask_path in mask_files.items():
            if not os.path.exists(mask_path):
                print(f"⚠️  Warning: Mask file not found for {landtype_name}: {mask_path}")
                continue

            # Read and reproject the land type mask
            land_mask = read_and_reproject_mask(
                mask_path,
                transform,
                (len(lats), len(lons))
            )

            if np.sum(land_mask) == 0:
                print(f"❌ ERROR: No valid pixels in {landtype_name} mask after reprojection!")
                continue

            # Process this land type with unit conversion
            files_created = clip_gpp_data_for_landtype(
                gpp_data,
                land_mask,
                dates,
                output_dir,
                landtype_name,
                lons,
                lats,
                transform,
                input_units,
                output_units
            )

            if files_created > 0:
                total_files += files_created
                successful_landtypes.append(landtype_name)

    # Final summary
    print(f"\n{'=' * 80}")
    print("🎉 PROCESSING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Total files created: {total_files}")
    print(f"🎯 Output units: {output_units}")
    print(f"✅ Successful land types: {len(successful_landtypes)}")

    # Expected value ranges for the selected units
    if output_units == "g C m-2 day-1":
        print(f"📊 Expected GPP ranges in {output_units}:")
        print(f"   • Low productivity (deserts, tundra): 0.1-1.0")
        print(f"   • Moderate productivity (grasslands, crops): 1.0-5.0")
        print(f"   • High productivity (forests): 5.0-15.0")
        print(f"   • Very high productivity (tropical forests): 10.0-20.0")
    elif output_units == "kg C m-2 year-1":
        print(f"📊 Expected GPP ranges in {output_units}:")
        print(f"   • Low productivity: 0.1-0.5")
        print(f"   • Moderate productivity: 0.5-2.0")
        print(f"   • High productivity: 2.0-4.0")
        print(f"   • Very high productivity: 4.0-7.0")
    elif output_units == "Mg C ha-1 year-1":
        print(f"📊 Expected GPP ranges in {output_units}:")
        print(f"   • Low productivity: 1-5")
        print(f"   • Moderate productivity: 5-20")
        print(f"   • High productivity: 20-40")
        print(f"   • Very high productivity: 40-70")

    for landtype_name in successful_landtypes:
        landtype_dir = os.path.join(output_dir, landtype_name)
        if os.path.exists(landtype_dir):
            tif_files = [f for f in os.listdir(landtype_dir) if f.endswith('.tif')]
            print(f"   • {landtype_name}: {len(tif_files)} files")

    print(f"\n💡 NOTE: Values are now in {output_units} - much more readable for scientific papers!")
    print(f"   Original small values were correct, just in inconvenient units for interpretation.")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("GPP DATA PROCESSING FOR CALIFORNIA LAND TYPES")
    print("=" * 80)
    print("📄 Loading GPP NetCDF file...")

    # Open and examine the dataset
    with xr.open_dataset(gpp_file, decode_times=False) as ds:
        print(f"✅ Dataset loaded successfully")
        print(f"   GPP data shape: {ds['gpp'].shape}")
        print(f"   Time range: {len(ds.time)} time steps")

        # Check units and attributes
        gpp_var = ds['gpp']
        units = gpp_var.attrs.get('units', 'unknown')
        print(f"   Units: {units}")

        # Get coordinates and time
        lons = ds.lon.values
        lats = ds.lat.values
        time_values = ds.time.values

        # Convert time
        time_units = ds.time.attrs.get('units', 'days since 1582-10-14 00:00:00')
        time_calendar = ds.time.attrs.get('calendar', 'standard')
        dates = convert_time_to_datetime(time_values, time_units, time_calendar)
        print(f"   Date range: {dates[0]} to {dates[-1]}")

        # Create geospatial transform
        transform = create_geotiff_transform(lons, lats)

        # Get GPP data
        gpp_data = ds['gpp'].values

        print(f"   Coordinate ranges:")
        print(f"     Longitude: {lons.min():.3f}° to {lons.max():.3f}°")
        print(f"     Latitude: {lats.min():.3f}° to {lats.max():.3f}°")

        # Global data validation
        sample_data = gpp_data[0]
        valid_global = sample_data[~np.isnan(sample_data)]
        print(f"   Global GPP range: {np.min(valid_global):.2e} to {np.max(valid_global):.2e} {units}")
        print(f"   Global GPP mean: {np.mean(valid_global):.2e} {units}")

        # Process each land type
        total_files = 0
        successful_landtypes = []

        for landtype_name, mask_path in mask_files.items():
            if not os.path.exists(mask_path):
                print(f"⚠️  Warning: Mask file not found for {landtype_name}: {mask_path}")
                continue

            # Read and reproject the land type mask
            land_mask = read_and_reproject_mask(
                mask_path,
                transform,
                (len(lats), len(lons))
            )

            if np.sum(land_mask) == 0:
                print(f"❌ ERROR: No valid pixels in {landtype_name} mask after reprojection!")
                continue

            # Process this land type
            files_created = clip_gpp_data_for_landtype(
                gpp_data,
                land_mask,
                dates,
                output_dir,
                landtype_name,
                lons,
                lats,
                transform,
                units
            )

            if files_created > 0:
                total_files += files_created
                successful_landtypes.append(landtype_name)

    # Final summary
    print(f"\n{'=' * 80}")
    print("🎉 PROCESSING COMPLETE!")
    print(f"{'=' * 80}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Total files created: {total_files}")
    print(f"✅ Successful land types: {len(successful_landtypes)}")

    for landtype_name in successful_landtypes:
        landtype_dir = os.path.join(output_dir, landtype_name)
        if os.path.exists(landtype_dir):
            tif_files = [f for f in os.listdir(landtype_dir) if f.endswith('.tif')]
            print(f"   • {landtype_name}: {len(tif_files)} files")

    print(f"\n💡 NOTE: GPP values in kg m⁻² s⁻¹ are naturally very small (1e-10 to 1e-7 range)")
    print(f"   This is normal and expected for carbon flux data at per-second resolution!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()