# Data availability

| Category | What it includes | Availability |
|---|---|---|
| **Public source data** | MODIS MOD13Q1, MOD09A1, MYD21C1, MOD17A2H, MCD12Q1; gridMET (SPEI-1, VPD); JPL GRACE/GRACE-FO mascons; GOSIF v2; FLUXCOM-X-BASE GPP | Available from the original providers (IDs and versions: [`../data_access/data_sources.csv`](../data_access/data_sources.csv)) |
| **Raw source data** | copies of the products above | Not redistributed through GitHub; obtain from the providers under their own terms |
| **Processed intermediate data** | land-type-masked monthly rasters, monthly land-type series, forecasting task caches, model predictions and metric tables | Not publicly deposited; available from the corresponding author upon reasonable request, subject to redistribution and file-size constraints |
| **Derived spatial products** | NIRv percentage-change maps, severe-response masks, land-type masks | Not publicly deposited; same conditions as processed intermediate data |
| **Code, configuration and documentation** | Earth Engine scripts, Python code, hyperparameters, folds, seeds, pixel-selection rules and counts, compact final source tables and statistical outputs (`results/`) | Public in this repository (MIT License) |

Suggested wording for the manuscript:

> All source datasets are publicly available from their original providers (listed in Table 1 and in
> `data_access/data_sources.csv` of the code repository); raw source data are not redistributed. Code, Google Earth Engine
> scripts, model configurations, fold definitions, random seeds, pixel-selection criteria and compact result tables are
> available at https://github.com/Aria2133/Linking-Drought-with-Vegetation-Response. Processed intermediate datasets and derived
> spatial products generated in this study are not publicly deposited and are available from the corresponding author upon
> reasonable request, subject to redistribution and file-size constraints.
