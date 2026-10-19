# Google Earth Engine (JavaScript) component

These scripts run in the [Earth Engine Code Editor](https://code.earthengine.google.com/). They read public collections,
clip to California, aggregate to monthly images and export GeoTIFFs (EPSG:4326) to Google Drive.
Everything after the export is done in Python (`../src`).

| Script | Collection | Output |
|---|---|---|
| `mcd12q1_2021_landtype_masks_500m.js` | `MODIS/061/MCD12Q1` (LC_Type1, 2021) | 0/1 masks `CA_<Land>_2021.tif`, 500 m: forests IGBP 1–5, shrublands 6–7, savannas 8–9, grasslands 10 |
| `modis_mod13q1_ndvi_monthly_5km.js` | `MODIS/061/MOD13Q1` | monthly mean NDVI, 5 km |
| `modis_mod13q1_nirv_monthly_5km.js` | `MODIS/061/MOD13Q1` | monthly mean NIRv (NDVI × NIR reflectance), 5 km |
| `modis_mod09a1_gndvi_monthly_5km.js` | `MODIS/061/MOD09A1` | monthly mean GNDVI, 5 km |
| `modis_myd21c1_lst_monthly.js` | `MODIS/061/MYD21C1` | monthly mean daytime LST (QC-filtered) |
| `modis_myd21c1_lst_monthly_mcd12q1_variant.js` | `MODIS/061/MYD21C1` + `MCD12Q1` | variant of the LST export that also reads land cover; kept because the exported LST folder is shared and it could not be proven which of the two produced the final files |
| `gridmet_spei1_monthly_5km.js` | `GRIDMET/DROUGHT` (`spei30d`) | monthly SPEI-1, 5 km |
| `gridmet_vpdmax_monthly_5km.js` | `IDAHO_EPSCOR/GRIDMET` (`vpd`) | monthly maximum VPD, 5 km |
| `grace_mascon_tws_monthly.js` | `NASA/GRACE/MASS_GRIDS_V04/MASCON` (`lwe_thickness`) | monthly TWS anomaly (exported at 100 km scale) |
| `modis_mod17a2h_gpp_monthly_5km_2022_2024.js` | `MODIS/061/MOD17A2H` | monthly GPP total 2022–2024 (extends FLUXCOM) |

Not in Earth Engine: **GOSIF** and **FLUXCOM GPP** were downloaded from their providers and processed in Python
(see `../data_access/README.md`).

To rerun: open a script in the Code Editor, set the export folder/date range at the top if needed, run, and start the export
tasks in the *Tasks* tab. The exported file names are the ones the Python masking scripts expect.
