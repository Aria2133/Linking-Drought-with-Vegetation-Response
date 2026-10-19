# Data access

**No data are stored in this repository.** All source datasets are public and can be obtained from the providers listed in
[`data_sources.csv`](data_sources.csv) (product, collection ID, resolution, period, and the script that processes it).

## How to obtain the inputs

1. **Earth Engine products** (MODIS NDVI/NIRv/GNDVI/LST/GPP, gridMET SPEI-1 and VPDmax, GRACE/GRACE-FO TWS, MCD12Q1 land cover):
   run the scripts in [`../gee/scripts`](../gee/scripts) with an Earth Engine account; they export monthly GeoTIFFs to Google Drive.
2. **GOSIF** monthly SIF: download from the GOSIF data portal (Li and Xiao, 2019) and clip to California.
3. **FLUXCOM GPP** (2001–2021): download the monthly GPP NetCDF cited in the manuscript; convert it with
   `src/preprocessing/01_landtype_masking/fluxcom_gpp_netcdf_to_landtype_tifs.py`.
4. Apply the land-type masks with the scripts in `src/preprocessing/01_landtype_masking/`.

## Expected local folder layout

The scripts were run with absolute Windows paths. In this repository those roots are written as placeholders:

| Placeholder | Meaning |
|---|---|
| `<DATA_ROOT>` | folder holding the raster archive: `Data\CA_LandMasks`, `Data\CALIFORNIA INDICES\<product folders>`, `Data\CALIFORNIA <LAND>\<VAR>` (2001–2021 land-type rasters), `Test Data\Masked_LandTypes\<Land>\<VAR>` (2022–2024) |
| `<PROJECT_ROOT>` | folder where the analysis scripts write their tables, caches and figures |
| `<DOWNLOADS>` | location of the downloaded FLUXCOM NetCDF |

Replace each placeholder with your own path before running a script.

## What is and is not available

See [`../docs/DATA_AVAILABILITY.md`](../docs/DATA_AVAILABILITY.md).
