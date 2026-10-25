import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
import numpy as np
import os
import glob
from rasterio.transform import from_bounds
import warnings

warnings.filterwarnings("ignore")


def install_dependencies():
    """Install required packages"""
    import subprocess
    import sys

    packages = ['netcdf4', 'h5netcdf']
    for package in packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is available")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])


def open_netcdf_file(filepath):
    """Try to open NetCDF file with different engines"""
    engines = ['netcdf4', 'h5netcdf']

    for engine in engines:
        try:
            ds = xr.open_dataset(filepath, engine=engine)
            print(f"Opened {os.path.basename(filepath)} with {engine}")
            return ds
        except Exception as e:
            print(f"Failed with {engine}: {str(e)[:50]}...")
            continue

    raise Exception(f"Could not open {filepath} with any engine")


def get_gpp_variable(ds):
    """Find GPP variable in dataset"""
    for var_name in ds.data_vars.keys():
        if 'gpp' in var_name.lower():
            return var_name
    return None


def create_california_mask(lons, lats):
    """Create a simple mask for California region"""
    # California boundaries (approximate)
    ca_bounds = {
        'lon_min': -124.5, 'lon_max': -114.0,
        'lat_min': 32.0, 'lat_max': 42.5
    }

    # Create coordinate meshgrids
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Create California mask
    ca_mask = ((lon_grid >= ca_bounds['lon_min']) &
               (lon_grid <= ca_bounds['lon_max']) &
               (lat_grid >= ca_bounds['lat_min']) &
               (lat_grid <= ca_bounds['lat_max']))

    return ca_mask


def read_landtype_mask(mask_path, target_lons, target_lats):
    """Read and reproject landtype mask to match GPP grid"""
    if not os.path.exists(mask_path):
        return None

    try:
        # Create target transform
        lon_res = abs(target_lons[1] - target_lons[0])
        lat_res = abs(target_lats[1] - target_lats[0])

        target_transform = from_bounds(
            west=float(target_lons.min() - lon_res / 2),
            south=float(target_lats.min() - lat_res / 2),
            east=float(target_lons.max() + lon_res / 2),
            north=float(target_lats.max() + lat_res / 2),
            width=len(target_lons),
            height=len(target_lats)
        )

        # Read and reproject mask
        with rasterio.open(mask_path) as src:
            mask_data = src.read(1)
            reprojected_mask = np.zeros((len(target_lats), len(target_lons)), dtype=np.uint8)

            reproject(
                source=mask_data,
                destination=reprojected_mask,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=target_transform,
                dst_crs='EPSG:4326',
                resampling=Resampling.nearest
            )

        return reprojected_mask > 0

    except Exception as e:
        print(f"Error reading mask {mask_path}: {e}")
        return None


def save_monthly_tiff(data, lons, lats, output_path, metadata=None):
    """Save data as GeoTIFF"""
    # Create transform
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

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as GeoTIFF
    with rasterio.open(
            output_path,
            'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=np.float32,
            crs='EPSG:4326',
            transform=transform,
            compress='lzw',
            nodata=np.nan
    ) as dst:
        dst.write(data.astype(np.float32), 1)

        # Add metadata if provided
        if metadata:
            dst.update_tags(**metadata)


def process_netcdf_file(filepath, mask_files, output_dir):
    """Process a single NetCDF file"""
    print(f"\n{'=' * 50}")
    print(f"Processing: {os.path.basename(filepath)}")
    print(f"{'=' * 50}")

    try:
        # Extract year from filename
        filename = os.path.basename(filepath)
        year = int(filename.split('_')[1])

        # Open NetCDF file
        ds = open_netcdf_file(filepath)

        with ds:
            # Find GPP variable
            gpp_var = get_gpp_variable(ds)
            if not gpp_var:
                print("No GPP variable found")
                return 0

            print(f"GPP variable: {gpp_var}")
            print(f"Shape: {ds[gpp_var].shape}")
            print(f"Dimensions: {list(ds[gpp_var].dims)}")

            # Get coordinates
            lons = ds.lon.values
            lats = ds.lat.values

            print(f"Coordinate ranges:")
            print(f"  Longitude: {lons.min():.2f} to {lons.max():.2f}")
            print(f"  Latitude: {lats.min():.2f} to {lats.max():.2f}")

            # Create California mask
            ca_mask = create_california_mask(lons, lats)
            ca_pixels = np.sum(ca_mask)
            print(f"California pixels: {ca_pixels}")

            if ca_pixels == 0:
                print("No California pixels found")
                return 0

            # Get GPP data
            gpp_data = ds[gpp_var].values
            print(f"GPP data shape: {gpp_data.shape}")

            # Units
            units = ds[gpp_var].attrs.get('units', 'unknown')
            print(f"Units: {units}")

            # Process each land type
            total_files = 0

            for landtype_name, mask_path in mask_files.items():
                print(f"\n--- {landtype_name} ---")

                # Read landtype mask
                landtype_mask = read_landtype_mask(mask_path, lons, lats)
                if landtype_mask is None:
                    print(f"Could not load mask for {landtype_name}")
                    continue

                # Combine with California mask
                combined_mask = ca_mask & landtype_mask
                valid_pixels = np.sum(combined_mask)
                print(f"Valid pixels: {valid_pixels}")

                if valid_pixels == 0:
                    print("No valid pixels for this land type")
                    continue

                # Create output directory
                landtype_dir = os.path.join(output_dir, landtype_name)
                os.makedirs(landtype_dir, exist_ok=True)

                # Process each month
                n_months = gpp_data.shape[0] if len(gpp_data.shape) == 3 else 1

                for month in range(n_months):
                    # Get monthly data
                    if len(gpp_data.shape) == 3:
                        monthly_gpp = gpp_data[month, :, :]
                    else:
                        monthly_gpp = gpp_data
                        month = 0

                    # Apply mask
                    masked_gpp = np.full_like(monthly_gpp, np.nan, dtype=np.float32)
                    masked_gpp[combined_mask] = monthly_gpp[combined_mask]

                    # Check for valid data
                    valid_data = masked_gpp[~np.isnan(masked_gpp)]
                    if len(valid_data) == 0:
                        print(f"No valid data for month {month + 1}")
                        continue

                    # Create filename
                    output_filename = f"{landtype_name}_GPP_{year}_{month + 1:02d}.tif"
                    output_path = os.path.join(landtype_dir, output_filename)

                    # Metadata
                    metadata = {
                        'units': units,
                        'year': str(year),
                        'month': str(month + 1),
                        'landtype': landtype_name,
                        'source': 'FLUXCOM',
                        'valid_pixels': str(len(valid_data)),
                        'min_value': f"{np.min(valid_data):.6f}",
                        'max_value': f"{np.max(valid_data):.6f}",
                        'mean_value': f"{np.mean(valid_data):.6f}"
                    }

                    # Save as TIFF
                    save_monthly_tiff(masked_gpp, lons, lats, output_path, metadata)
                    total_files += 1

                # Print summary for this landtype
                landtype_files = len([f for f in os.listdir(landtype_dir)
                                      if f.startswith(f"{landtype_name}_GPP_{year}")])
                print(f"Created {landtype_files} files")

            return total_files

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return 0


def main():
    print("=" * 70)
    print("FLUXCOM GPP PROCESSING FOR CALIFORNIA LAND TYPES")
    print("=" * 70)

    # Install dependencies
    install_dependencies()

    # Paths
    input_dir = r"<DATA_ROOT>\NEWEST GPP"
    output_dir = r"<DATA_ROOT>\Data\FLUXCOM_GPP_California"

    # Land type masks
    mask_files = {
        "Grasslands": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Grasslands_2021.tif",
        "NaturalForests": r"<DATA_ROOT>\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
        "Savannas": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Savannas_2021.tif",
        "Shrublands": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Shrublands_2021.tif"
    }

    # Find NetCDF files
    pattern = os.path.join(input_dir, "GPP_*_050_monthly.nc")
    nc_files = glob.glob(pattern)

    if not nc_files:
        print(f"No files found matching: {pattern}")
        return

    nc_files.sort()
    print(f"Found {len(nc_files)} files")

    # Check masks
    print("\nChecking land type masks:")
    valid_masks = {}
    for name, path in mask_files.items():
        if os.path.exists(path):
            print(f"  ✅ {name}")
            valid_masks[name] = path
        else:
            print(f"  ❌ {name}: {path}")

    if not valid_masks:
        print("No valid masks found!")
        return

    # Process files
    total_files_created = 0
    successful_years = []

    for nc_file in nc_files:
        files_created = process_netcdf_file(nc_file, valid_masks, output_dir)
        if files_created > 0:
            total_files_created += files_created
            year = int(os.path.basename(nc_file).split('_')[1])
            successful_years.append(year)

    # Summary
    print(f"\n{'=' * 70}")
    print("PROCESSING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total files created: {total_files_created}")
    print(f"Successful years: {len(successful_years)}")

    if successful_years:
        print(f"Years processed: {min(successful_years)} to {max(successful_years)}")

        print(f"\nFiles by land type:")
        for landtype in valid_masks.keys():
            landtype_dir = os.path.join(output_dir, landtype)
            if os.path.exists(landtype_dir):
                tif_count = len([f for f in os.listdir(landtype_dir) if f.endswith('.tif')])
                print(f"  {landtype}: {tif_count} files")


if __name__ == "__main__":
    main()