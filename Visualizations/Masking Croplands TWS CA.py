import os
import glob
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.enums import Resampling
import numpy as np
from pathlib import Path
import re


def extract_year_month_from_filename(filename):
    """Extract year and month from filename like SPEI1_CSIC_CA_2001_01 or TWS_GRACE_CA_2001_01"""
    match = re.search(r'(\d{4})_(\d{2})', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def process_data_cropland_masking(data_type="SPEI"):
    """
    Process either SPEI or TWS GRACE data with cropland masking

    Parameters:
    data_type: str, either "SPEI" or "TWS" to specify which dataset to process
    """

    # Define paths based on data type
    if data_type.upper() == "SPEI":
        data_folder = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\SPEI1_CSIC_TIFs"
        output_folder = r"C:\Users\LOQ\OneDrive\Desktop\SPEI_Masked_Output"
        file_pattern = "SPEI*.tif"
        output_prefix = "SPEI1_CSIC_CA"
    elif data_type.upper() == "TWS":
        data_folder = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\TWS_GRACE_TIFs"
        output_folder = r"C:\Users\LOQ\OneDrive\Desktop\TWS_Masked_Output"
        file_pattern = "*.tif"
        output_prefix = "TWS_GRACE_CA"
    else:
        raise ValueError("data_type must be either 'SPEI' or 'TWS'")

    cropland_folder = r"C:\Users\LOQ\OneDrive\Desktop\California_Cropland_Masks"

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get all data files
    data_files = glob.glob(os.path.join(data_folder, file_pattern))
    print(f"Found {len(data_files)} {data_type} files")

    # Process each data file
    processed_count = 0

    for data_file in data_files:
        try:
            # Extract year and month from filename
            filename = os.path.basename(data_file)
            year, month = extract_year_month_from_filename(filename)

            if year is None or month is None:
                print(f"Could not extract year/month from {filename}, skipping...")
                continue

            # Find corresponding cropland mask
            cropland_file = os.path.join(cropland_folder, f"CA_CroplandMask_{year}.tif")

            if not os.path.exists(cropland_file):
                print(f"Cropland mask not found for year {year}, skipping {filename}")
                continue

            print(f"Processing {filename} (Year: {year}, Month: {month})")

            # Read data
            with rasterio.open(data_file) as data_src:
                data_array = data_src.read(1)
                data_profile = data_src.profile
                data_transform = data_src.transform
                data_crs = data_src.crs

            # Read cropland data
            with rasterio.open(cropland_file) as crop_src:
                crop_data = crop_src.read(1)
                crop_transform = crop_src.transform
                crop_crs = crop_src.crs

            # Reproject cropland data to match data resolution and extent
            # Create arrays to hold the reprojected cropland data
            reprojected_crop = np.zeros_like(data_array, dtype=np.uint8)

            # Reproject from cropland CRS/transform to data CRS/transform
            reproject(
                source=crop_data,
                destination=reprojected_crop,
                src_transform=crop_transform,
                src_crs=crop_crs,
                dst_transform=data_transform,
                dst_crs=data_crs,
                resampling=Resampling.nearest
            )

            # Create mask: cropland pixels typically have value 1, non-cropland 0
            # We want to keep non-cropland areas, so mask out where reprojected_crop == 1
            cropland_mask = reprojected_crop == 1

            # Apply mask to data
            masked_data = data_array.copy()
            masked_data[cropland_mask] = np.nan  # Set cropland pixels to NaN

            # Alternatively, if you want to use the original NoData value:
            # if data_profile.get('nodata') is not None:
            #     masked_data[cropland_mask] = data_profile['nodata']

            # Update profile for output
            output_profile = data_profile.copy()

            # Handle different data types appropriately
            if data_type.upper() == "SPEI":
                output_profile.update({
                    'nodata': np.nan,
                    'dtype': 'float32'
                })
            else:  # TWS
                output_profile.update({
                    'nodata': np.nan,
                    'dtype': 'float64'  # Keep double precision for TWS data
                })

            # Create output filename
            output_filename = f"{output_prefix}_{year:04d}_{month:02d}_masked.tif"
            output_path = os.path.join(output_folder, output_filename)

            # Write masked data
            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(masked_data, 1)

            processed_count += 1
            print(f"Successfully processed and saved: {output_filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    print(f"\nProcessing complete! {processed_count} {data_type} files processed successfully.")
    print(f"Output files saved to: {output_folder}")


# Keep the original function name for backward compatibility
def process_spei_cropland_masking():
    """Legacy function - calls the new generic function for SPEI"""
    process_data_cropland_masking("SPEI")


def validate_data_alignment(data_type="SPEI"):
    """Function to check and display information about the datasets"""

    if data_type.upper() == "SPEI":
        data_folder = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\SPEI1_CSIC_TIFs"
        dataset_name = "SPEI"
    elif data_type.upper() == "TWS":
        data_folder = r"C:\Users\LOQ\OneDrive\Desktop\CALIFORNIA INDICES\TWS_GRACE_TIFs"
        dataset_name = "TWS GRACE"
    else:
        raise ValueError("data_type must be either 'SPEI' or 'TWS'")

    cropland_folder = r"C:\Users\LOQ\OneDrive\Desktop\California_Cropland_Masks"

    # Check first data file
    data_files = glob.glob(os.path.join(data_folder, "*.tif"))
    if data_files:
        with rasterio.open(data_files[0]) as src:
            print(f"{dataset_name} Dataset Info:")
            print(f"  Shape: {src.shape}")
            print(f"  CRS: {src.crs}")
            print(f"  Transform: {src.transform}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Data type: {src.dtypes[0]}")

    # Check first cropland file
    crop_files = glob.glob(os.path.join(cropland_folder, "*.tif"))
    if crop_files:
        with rasterio.open(crop_files[0]) as src:
            print("\nCropland Dataset Info:")
            print(f"  Shape: {src.shape}")
            print(f"  CRS: {src.crs}")
            print(f"  Transform: {src.transform}")
            print(f"  Bounds: {src.bounds}")
            print(f"  Data type: {src.dtypes[0]}")

            # Check unique values in cropland data
            data = src.read(1)
            unique_vals = np.unique(data[~np.isnan(data)])
            print(f"  Unique values: {unique_vals}")


if __name__ == "__main__":
    # Choose which dataset to process: "SPEI" or "TWS" or "BOTH"
    dataset_to_process = "BOTH"  # Change this to "SPEI", "TWS", or "BOTH"

    if dataset_to_process.upper() in ["SPEI", "BOTH"]:
        print("=== SPEI Data Validation ===")
        validate_data_alignment("SPEI")

        print("\n" + "=" * 50)
        print("=== Starting SPEI Cropland Masking ===")
        process_data_cropland_masking("SPEI")

    if dataset_to_process.upper() in ["TWS", "BOTH"]:
        print("\n" + "=" * 50)
        print("=== TWS GRACE Data Validation ===")
        validate_data_alignment("TWS")

        print("\n" + "=" * 50)
        print("=== Starting TWS GRACE Cropland Masking ===")
        process_data_cropland_masking("TWS")

    print("\n" + "=" * 50)
    print("=== All Processing Complete ===")

    # Example of how to process just one dataset:
    # To process only SPEI:
    # process_data_cropland_masking("SPEI")

    # To process only TWS:
    # process_data_cropland_masking("TWS")