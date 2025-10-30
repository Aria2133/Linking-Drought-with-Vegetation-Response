# Linking Drought with Vegetation Response in California

> **Note:** This repository contains research code for an upcoming publication. The paper is currently under final revision before submission.

## 📋 Overview

This repository contains the complete analysis pipeline for investigating the relationships between drought conditions and vegetation responses across different land cover types in California. The study integrates multiple climate and vegetation indices to understand drought impacts, vegetation resilience, and recovery patterns using both traditional statistical methods and deep learning approaches.

## 🔬 Research Objectives

- Quantify the relationship between drought indices and vegetation productivity
- Develop deep learning models for drought forecasting
- Identify optimal lag times between drought onset and vegetation response
- Assess vegetation resilience and recovery across different ecosystems
- Compare drought sensitivity among California's major land cover types

## 📊 Data Sources & Indices

### Drought/Climate Indices
- **SPEI** - Standardized Precipitation-Evapotranspiration Index
- **TWS** - Total Water Storage (GRACE satellite)
- **LST** - Land Surface Temperature (MODIS)
- **VPD** - Vapor Pressure Deficit

### Vegetation Indices
- **GOSIF** - Global OCO-2 Solar-Induced Fluorescence
- **GPP** - Gross Primary Productivity
- **NDVI** - Normalized Difference Vegetation Index
- **NIRv** - Near-Infrared Reflectance of Vegetation
- **GNDVI** - Green Normalized Difference Vegetation Index
- **EVI** - Enhanced Vegetation Index

## 🛠️ Methodology

### Data Processing Pipeline
1. **Google Earth Engine Processing**
   - Satellite data acquisition (MODIS, Landsat, GRACE)
   - Cloud masking and quality control
   - Temporal aggregation (monthly means)
   - Spatial resampling (250m to 5km)
   - Export to GeoTIFF format

2. **Python Analysis Pipeline**
   - Spatial preprocessing and cropland masking
   - Deseasonalization using monthly climatology
   - Detrending using Singular Spectrum Analysis (SSA)
   - Cross-correlation analysis with multiple lag times
   - Statistical analysis and time series decomposition

3. **Deep Learning Models**
   - LSTM/GRU networks for drought forecasting
   - Multi-sensor feature integration
   - Temporal sequence prediction
   - Model validation and performance assessment

4. **Drought Impact Assessment**
   - Resilience metrics
   - Recovery time analysis
   - Land cover-specific response characterization

## 📁 Repository Contents

### Google Earth Engine Scripts (`gee_scripts/`)
JavaScript code for satellite data processing:
- MODIS vegetation indices (NDVI, NIRv, EVI, GNDVI)
- Land surface temperature extraction
- GRACE water storage processing
- Climate indices (SPEI, VPD)
- Multiple spatial resolutions (250m, 500m, 1km, 5km)
- Agricultural area masking

### Python Scripts
- Data preprocessing and spatial analysis
- Statistical analysis (correlation, SSA, time series)
- Resilience and recovery metrics
- Visualization and mapping
- Land cover-specific analysis (Grasslands, Forests, Savannas, Shrublands)

### Deep Learning Models
- Drought forecasting neural networks
- Feature engineering and data preparation
- Model training and validation
- Performance evaluation

### Results
- Correlation heatmaps and spatial maps
- Time series visualizations
- Statistical outputs and data tables
- Publication-quality figures

## 🔧 Requirements

### Python Environment
```python
# Core Scientific Computing
numpy, pandas, scipy, xarray, netCDF4

# Geospatial
rasterio, geopandas, shapely, pyproj, earthengine-api

# Analysis
scikit-learn, statsmodels

# Deep Learning
tensorflow, keras

# Visualization
matplotlib, seaborn, cartopy
```

### Installation
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Google Earth Engine Workflow
1. Open scripts from `gee_scripts/` in GEE Code Editor
2. Modify parameters (dates, region, resolution)
3. Run scripts and export to Google Drive
4. Download processed GeoTIFF files

### Python Analysis Workflow
1. Load exported GeoTIFF data
2. Run preprocessing scripts
3. Execute statistical analysis
4. Generate visualizations
5. Train deep learning models

## 📈 Key Features

- Multi-sensor integration (MODIS, GRACE, OCO-2, Landsat)
- 20+ years of time series analysis (2001-2024)
- State-wide spatial coverage (~424,000 km²)
- Ecosystem-specific analysis (4 land cover types)
- Traditional statistics + deep learning approaches
- Reproducible research workflow

## 📧 Contact

**Author:** Aria Ahmadi  
**Email:** Ariaahmadi2155@gmail.com  
**Institution:** University of Tehran  
**Department:** Water Resources Management

## ⚠️ Data Availability

Raw data sources:
- **SPEI**: [SPEIbase](http://spei.csic.es/database.html)
- **GRACE/TWS**: [NASA JPL](https://grace.jpl.nasa.gov/)
- **MODIS**: [NASA EARTHDATA](https://earthdata.nasa.gov/)
- **GOSIF**: [Global Ecology Group](http://globalecology.unh.edu/)

Processed datasets will be made available upon publication.

## 🔒 License

**Copyright © 2025 Aria Ahmadi. All Rights Reserved.**

This code is provided for review purposes. Redistribution or use for academic research requires permission from the author until publication. After publication, this repository will be released under the MIT License.

## 🔄 Status

- [x] Data collection and preprocessing
- [x] Statistical analysis pipeline
- [x] Deep learning model development
- [x] Visualization and results
- [ ] Paper under final revision
- [ ] Submitted for peer review
- [ ] Published

---

**Related Publication:** "A Multi-Sensor Remote Sensing Framework for Drought Forecasting and Vegetation Resilience Assessment in California Based on Deep Learning" (In preparation)

**Last Updated:** October 2025
```
