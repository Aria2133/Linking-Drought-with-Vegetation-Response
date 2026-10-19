import os
import rasterio
import numpy as np
from glob import glob
from rasterio.warp import reproject, Resampling

# -------------------------------------------------------------------
# 1) Paths and Configuration
# -------------------------------------------------------------------
# Base directory for all datasets
base_data_dir = r"<DATA_ROOT>\Test Data"

# Define all indices and their corresponding folders
# NOTE: For SPEI we now accept multiple patterns to catch 2024 GRIDMET names
indices_config = {
    "GNDVI": {
        "folder": os.path.join(base_data_dir, "GNDVI"),
        "pattern": "GNDVI_MOD09A1_CA_5km_*.tif",
    },
    "GOSIF": {
        "folder": os.path.join(base_data_dir, "GOSIF"),
        "pattern": "GOSIF_*.tif",
    },
    "GPP": {
        "folder": os.path.join(base_data_dir, "GPP"),
        "pattern": "GPP_MOD17A2H_CA_5km_*.tif",
    },
    "LST": {
        "folder": os.path.join(base_data_dir, "LST"),
        "pattern": "MYD21C1_CA_LST_*.tif",
    },
    "NDVI": {
        "folder": os.path.join(base_data_dir, "NDVI"),
        "pattern": "NDVI_MOD13Q1_CA_5km_*.tif",
    },
    "NIRv": {
        "folder": os.path.join(base_data_dir, "NIRv"),
        "pattern": "NIRv_MOD13Q1_CA_5km_*.tif",
    },
    "SPEI": {
        "folder": os.path.join(base_data_dir, "SPEI"),
        # Accept both historical CSIC and 2024 GRIDMET naming
        "patterns": [
            "SPEI1_CSIC_CA_*.tif",         # historical
            "SPEI1_GRIDMET_CA_2024_*.tif", # 2024 months
        ],
    },
    "TWS": {
        "folder": os.path.join(base_data_dir, "TWS"),
        "pattern": "TWS_GRACE_CA_*.tif",
    },
    "VPD": {
        "folder": os.path.join(base_data_dir, "VPD"),
        "pattern": "VPDmax_GRIDMET_CA_5km_*.tif",
    },
}

# Land masks (same as before)
land_masks = {
    "Grasslands": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Grasslands_2021.tif",
    "NaturalForests": r"<DATA_ROOT>\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
    "Savannas": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Savannas_2021.tif",
    "Shrublands": r"<DATA_ROOT>\Data\CA_LandMasks\CA_Shrublands_2021.tif",
}

# Output root directory
output_root = r"<DATA_ROOT>\Data\Masked_LandTypes"


# -------------------------------------------------------------------
# Helper: collect files for an index allowing single or multiple patterns
# -------------------------------------------------------------------
def collect_index_files(folder: str, cfg: dict) -> list[str]:
    """
    Return a sorted, de-duplicated list of files for an index.
    Supports either 'pattern' (str) or 'patterns' (list[str]) in cfg.
    """
    patterns = cfg.get("patterns")
    if patterns is None:
        # fall back to single 'pattern'
        pat = cfg.get("pattern", "*")
        patterns = [pat]

    found = set()
    print(f"Looking in: {folder}")
    for pat in patterns:
        full_pat = os.path.join(folder, pat)
        hits = glob(full_pat)
        print(f"  Pattern '{pat}': {len(hits)} files")
        found.update(hits)

    files = sorted(found)
    print(f"Total unique files found: {len(files)}")
    return files


# -------------------------------------------------------------------
# 2) Function to apply mask (same as before)
# -------------------------------------------------------------------
def apply_mask(input_path, mask_path, output_path):
    with rasterio.open(input_path) as src, rasterio.open(mask_path) as mask_src:
        data = src.read(1)
        meta = src.meta.copy()

        # Reproject mask to match data raster grid
        mask_resampled = np.empty(data.shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(mask_src, 1),
            destination=mask_resampled,
            src_transform=mask_src.transform,
            src_crs=mask_src.crs,
            dst_transform=src.transform,
            dst_crs=src.crs,
            resampling=Resampling.nearest,
        )

        # Mask out non-target pixels (keep only value==1 in mask)
        masked = np.where(mask_resampled == 1, data, np.nan)

        # Update metadata
        meta.update(dtype="float32", nodata=np.nan, compress="LZW")

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(masked.astype(np.float32), 1)


# -------------------------------------------------------------------
# 3) Process all indices for each land type
# -------------------------------------------------------------------
def main():
    total_processed = 0
    total_failed = 0

    # Process each index
    for index_name, config in indices_config.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {index_name} data...")
        print(f"{'=' * 60}")

        # Get all files for this index (supports multiple patterns)
        index_files = collect_index_files(config["folder"], config)

        if not index_files:
            # Show what we looked for
            pats = config.get("patterns") or [config.get("pattern", "*")]
            print(f"⚠ No {index_name} files found in {config['folder']}")
            print("  Looked for patterns:")
            for p in pats:
                print(f"   • {p}")
            continue

        # Process each land type
        for land_type, mask_path in land_masks.items():
            print(f"\n--- Processing {land_type} for {index_name} ---")

            # Create output directory: output_root/LandType/Index/
            out_dir = os.path.join(output_root, land_type, index_name)
            os.makedirs(out_dir, exist_ok=True)

            # Process each file for this index and land type
            for file_path in index_files:
                base_name = os.path.basename(file_path)
                name_without_ext, ext = os.path.splitext(base_name)
                # ensure we keep the original extension (usually .tif)
                out_name = f"{name_without_ext}_{land_type}{ext or '.tif'}"
                out_path = os.path.join(out_dir, out_name)

                try:
                    apply_mask(file_path, mask_path, out_path)
                    print(f"  ✔ {out_name}")
                    total_processed += 1
                except Exception as e:
                    print(f"  ❌ Failed {base_name} – {e}")
                    total_failed += 1

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total files processed successfully: {total_processed}")
    print(f"Total files failed: {total_failed}")
    print(f"Output directory structure:")
    print(f"  {output_root}/")
    for land_type in land_masks.keys():
        print(f"    ├── {land_type}/")
        for index_name in indices_config.keys():
            print(f"    │   ├── {index_name}/")
    print(f"✅ All indices masked for all land types.")


if __name__ == "__main__":
    main()
