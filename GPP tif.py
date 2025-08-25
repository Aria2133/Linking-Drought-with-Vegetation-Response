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
        # Use cftime to handle the old reference date
        dates = cftime.num2date(time_values, time_units, calendar=calendar)
        # Convert cftime objects to standard datetime objects where possible
        datetime_list = []
        for date in dates:
            if hasattr(date, 'year') and hasattr(date, 'month') and hasattr(date, 'day'):
                # Convert to standard datetime for dates after 1900
                if date.year >= 1900:
                    datetime_list.append(datetime(date.year, date.month, date.day))
                else:
                    # Keep as cftime object for very old dates
                    datetime_list.append(date)
            else:
                datetime_list.append(date)
        return datetime_list
    except Exception as e:
        print(f"Error converting time: {e}")
        # Fallback: create approximate dates based on time values
        # Assuming the data spans roughly 1982-2011 (30 years, 360 months)
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
        # Read the mask data
        mask_data = src.read(1)

        # Create output array with target shape
        reprojected_mask = np.zeros(target_shape, dtype=np.uint8)

        # Reproject the mask to match GPP grid
        reproject(
            source=mask_data,
            destination=reprojected_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest
        )

        return reprojected_mask > 0  # Convert to boolean mask


def create_geotiff_transform(lons, lats):
    """Create affine transform for the GPP data"""
    lon_res = abs(lons[1] - lons[0])
    lat_res = abs(lats[1] - lats[0])

    # Create transform (west, pixel_width, 0, north, 0, -pixel_height)
    transform = from_bounds(
        west=float(lons.min() - lon_res / 2),
        south=float(lats.min() - lat_res / 2),
        east=float(lons.max() + lon_res / 2),
        north=float(lats.max() + lat_res / 2),
        width=len(lons),
        height=len(lats)
    )

    return transform


def clip_gpp_data_for_landtype(gpp_data, land_mask, dates, output_dir, landtype_name,
                               lons, lats, transform):
    """Clip GPP data using land type mask and save as monthly TIF files"""

    # Create output directory for this land type
    landtype_dir = os.path.join(output_dir, landtype_name)
    os.makedirs(landtype_dir, exist_ok=True)

    print(f"\nProcessing {landtype_name}...")
    print(f"Land mask has {np.sum(land_mask)} valid pixels")

    # Test first time step
    sample_gpp = gpp_data[0]
    sample_clipped = np.full_like(sample_gpp, np.nan, dtype=np.float64)
    sample_clipped[land_mask] = sample_gpp[land_mask]
    valid_count = np.sum(~np.isnan(sample_clipped))

    if valid_count > 0:
        print(f"Sample data range: {np.nanmin(sample_clipped):.2e} to {np.nanmax(sample_clipped):.2e}")
    else:
        print(f"WARNING: No valid data found for {landtype_name}")
        return

    # Process each time step
    files_created = 0
    for i, date in enumerate(dates):
        # Get GPP data for this time step
        gpp_timestep = gpp_data[i]

        # Apply land type mask - use float64 to preserve precision
        clipped_data = np.full_like(gpp_timestep, np.nan, dtype=np.float64)
        clipped_data[land_mask] = gpp_timestep[land_mask]

        # Create output filename
        if hasattr(date, 'strftime'):
            date_str = date.strftime("%Y_%m")
        else:
            # Handle cftime objects
            date_str = f"{date.year:04d}_{date.month:02d}"

        output_filename = f"{landtype_name}_GPP_{date_str}.tif"
        output_path = os.path.join(landtype_dir, output_filename)

        # Save as GeoTIFF with float64 precision
        with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=clipped_data.shape[0],
                width=clipped_data.shape[1],
                count=1,
                dtype=np.float64,  # Use float64 to preserve small values
                crs='EPSG:4326',
                transform=transform,
                compress='lzw',
                nodata=np.nan
        ) as dst:
            dst.write(clipped_data, 1)

            # Add comprehensive metadata
            dst.set_band_description(1, f'GPP for {landtype_name} - {date_str}')
            dst.update_tags(
                1,
                units='kg m-2 s-1',
                long_name=f'Gross Primary Production - {landtype_name}',
                date=date_str,
                landtype=landtype_name,
                valid_pixels=str(np.sum(~np.isnan(clipped_data))),
                data_min=f"{np.nanmin(clipped_data):.2e}" if np.any(~np.isnan(clipped_data)) else "NaN",
                data_max=f"{np.nanmax(clipped_data):.2e}" if np.any(~np.isnan(clipped_data)) else "NaN"
            )

        files_created += 1

        # Print progress
        if i % 60 == 0:  # Print every 5 years (60 months)
            print(f"  Processed {i + 1}/{len(dates)} time steps...")

    print(f"✓ Completed {landtype_name}: {files_created} files saved to {landtype_dir}")
    return files_created


def main():
    # File paths
    gpp_file = r"C:\Users\LOQ\Downloads\202582212913EnsembleGPP_GL.nc"

    # Land type mask files
    mask_files = {
        "Grasslands": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Grasslands_2021.tif",
        "NaturalForests": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
        "Savannas": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Savannas_2021.tif",
        "Shrublands": r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Shrublands_2021.tif"
    }

    # Output directory
    output_dir = r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\GPP_Clipped_Output"
    os.makedirs(output_dir, exist_ok=True)

    print("=== GPP DATA CLIPPING FOR CALIFORNIA LAND TYPES ===")
    print("Loading GPP NetCDF file...")

    # Open the NetCDF file with decode_times=False to avoid time conversion issues
    with xr.open_dataset(gpp_file, decode_times=False) as ds:
        print(f"Dataset loaded successfully")
        print(f"GPP data shape: {ds['gpp'].shape}")
        print(f"Time range: {len(ds.time)} time steps")

        # Get coordinates
        lons = ds.lon.values
        lats = ds.lat.values
        time_values = ds.time.values

        # Convert time to datetime manually
        time_units = ds.time.attrs.get('units', 'days since 1582-10-14 00:00:00')
        time_calendar = ds.time.attrs.get('calendar', 'standard')
        dates = convert_time_to_datetime(time_values, time_units, time_calendar)
        print(f"Date range: {dates[0]} to {dates[-1]}")

        # Create geospatial transform
        transform = create_geotiff_transform(lons, lats)

        # Get GPP data
        gpp_data = ds['gpp'].values  # Shape: (time, lat, lon)

        print(f"GPP data loaded: {gpp_data.shape}")
        print(f"Data range: {np.nanmin(gpp_data):.2e} to {np.nanmax(gpp_data):.2e} kg m-2 s-1")

        # Process each land type
        total_files = 0
        successful_landtypes = []

        for landtype_name, mask_path in mask_files.items():
            if not os.path.exists(mask_path):
                print(f"Warning: Mask file not found for {landtype_name}: {mask_path}")
                continue

            print(f"\nReading and reprojecting {landtype_name} mask...")

            # Read and reproject the land type mask
            land_mask = read_and_reproject_mask(
                mask_path,
                transform,
                (len(lats), len(lons))
            )

            print(f"Mask reprojected: {np.sum(land_mask)} valid pixels")

            if np.sum(land_mask) == 0:
                print(f"ERROR: No valid pixels in {landtype_name} mask after reprojection!")
                continue

            # Clip and save data for this land type
            files_created = clip_gpp_data_for_landtype(
                gpp_data,
                land_mask,
                dates,
                output_dir,
                landtype_name,
                lons,
                lats,
                transform
            )

            if files_created > 0:
                total_files += files_created
                successful_landtypes.append(landtype_name)

    print(f"\n{'=' * 60}")
    print("PROCESSING COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Output directory: {output_dir}")
    print(f"Total files created: {total_files}")
    print(f"Successful land types: {len(successful_landtypes)}")

    # Summary statistics
    for landtype_name in successful_landtypes:
        landtype_dir = os.path.join(output_dir, landtype_name)
        if os.path.exists(landtype_dir):
            tif_files = [f for f in os.listdir(landtype_dir) if f.endswith('.tif')]
            print(f"  ✓ {landtype_name}: {len(tif_files)} TIF files")

            # Show sample file info
            if tif_files:
                sample_file = os.path.join(landtype_dir, tif_files[0])
                with rasterio.open(sample_file) as src:
                    sample_data = src.read(1)
                    valid_pixels = np.sum(~np.isnan(sample_data))
                    if valid_pixels > 0:
                        data_range = f"{np.nanmin(sample_data):.2e} to {np.nanmax(sample_data):.2e}"
                        print(f"    Sample file: {tif_files[0]}")
                        print(f"    Data range: {data_range} kg m-2 s-1")
                        print(f"    Valid pixels: {valid_pixels}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback

        traceback.print_exc()