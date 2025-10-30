# Linking Drought with Vegetation Response in California

> **Note:** This repository contains research code for an upcoming publication. The paper is currently under final revision before submission.

## 📋 Overview

This repository contains the complete analysis pipeline for investigating the relationships between drought conditions and vegetation responses across different land cover types in California. The study integrates multiple climate and vegetation indices to understand drought impacts, vegetation resilience, and recovery patterns.

## 🔬 Research Objectives

- Quantify the relationship between drought indices and vegetation productivity
- Identify optimal lag times between drought onset and vegetation response
- Assess vegetation resilience and recovery across different ecosystems
- Compare drought sensitivity among California's major land cover types:
  - Grasslands
  - Natural Forests
  - Savannas
  - Shrublands

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
1. **Spatial Preprocessing**
   - Clipping to California state boundary
   - Masking agricultural/cropland areas
   - Resampling to common spatial resolution
   - Aligning temporal dimensions

2. **Temporal Analysis**
   - Deseasonalization using monthly climatology
   - Detrending using Singular Spectrum Analysis (SSA)
   - Monthly and yearly aggregation
   - Anomaly calculation (z-scores)

3. **Statistical Analysis**
   - Cross-correlation analysis with multiple lag times
   - Optimal lag identification
   - Spatial correlation mapping
   - Time series decomposition

4. **Drought Impact Assessment**
   - **Resilience**: Vegetation's ability to resist drought impacts
   - **Recovery**: Time and completeness of vegetation recovery post-drought
   - Land cover-specific response characterization

## 📁 Repository Structure

```
├── basic_drought_analysis/          # Initial exploratory analysis
├── drought_correlation_results/     # Correlation outputs
├── optimal_lag_results/             # Lag analysis results
├── results/                         # Final figures and data
│
├── Data Processing Scripts:
│   ├── California Drought Indices.py           # Main drought index processing
│   ├── California Drought Indices (Yearly).py  # Yearly aggregation
│   ├── CA MASKED 2.py                         # Cropland masking
│   ├── CA detrend yearly.py                    # SSA detrending
│   └── Without Crop Deseaonalized.py          # Deseasonalization
│
├── Vegetation Processing:
│   ├── CALIFORNIA GOSIF.py         # GOSIF processing
│   ├── FIXING GOSIF THE WORKING ONE.py
│   ├── GPP NetCDF.py              # GPP processing
│   ├── NIRv Deseasonalized.py     # NIRv processing
│   └── GNDVI LAND TYPES.py        # GNDVI by land type
│
├── Spatial Analysis:
│   ├── Clipping Code First Try.py  # Spatial clipping
│   ├── Clipping GRACE.py          # GRACE data processing
│   ├── Resize SPEI.py             # Resolution matching
│   └── GEE Upload.py              # Google Earth Engine integration
│
├── Time Series Analysis:
│   ├── SSA Indices.py             # Singular Spectrum Analysis
│   ├── CA DRought SSA.py          # SSA for drought indices
│   └── Tonzi Visualization for Drought Indices (SSA).py
│
├── Correlation Analysis:
│   ├── Corrolation Heatmaps.py    # Cross-correlation analysis
│   ├── Inter index correlation.py  # Inter-index relationships
│   └── CA DECROPPED COORELATION HEATMAP.py
│
├── Land Cover Specific:
│   ├── California Grasslands.py
│   ├── California Natural Forests.py
│   ├── California SAVANNA.py
│   └── California Shrublands.py
│
├── Resilience & Recovery:
│   ├── Res and Rec.py             # Main resilience/recovery analysis
│   ├── Res and Rec 2.py
│   └── CHAT CODE FOR RECOVERY AND RESILIENT.py
│
└── Visualization:
    ├── 9 Indices Heatmaps.py
    ├── 9 Indices Visualizing.py
    ├── Drought Visualization 1.py
    ├── Multi‑panel time‑series VISUALIZATION.py
    └── Various index-specific visualization scripts
```

## 🔧 Requirements

### Python Environment
```python
# Core Scientific Computing
numpy
pandas
scipy
xarray
netCDF4

# Geospatial
rasterio
geopandas
shapely
pyproj
earthengine-api

# Analysis
scikit-learn
statsmodels

# Visualization
matplotlib
seaborn
cartopy
```

### Installation
```bash
# Create conda environment
conda create -n drought_analysis python=3.9
conda activate drought_analysis

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Start
```python
# Example: Process California drought indices
python "California Drought Indices.py"

# Generate correlation heatmaps
python "Corrolation Heatmaps.py"

# Run resilience/recovery analysis
python "Res and Rec.py"
```

### Workflow
1. **Data Preparation**: Run spatial clipping and masking scripts
2. **Preprocessing**: Apply deseasonalization and detrending
3. **Analysis**: Execute correlation and lag analysis
4. **Visualization**: Generate figures using visualization scripts

## 📈 Key Findings

*[To be added after publication]*

The study reveals:
- Spatial and temporal patterns of drought-vegetation relationships across California
- Ecosystem-specific response times to drought conditions
- Differential resilience and recovery patterns among land cover types
- Critical thresholds and lag times for vegetation responses

## 📝 Citation

*[To be added after publication]*

```bibtex
@article{YourName2025,
  title={Linking Drought Conditions with Vegetation Response in California},
  author={Your Name},
  journal={Journal Name},
  year={2025},
  note={In preparation}
}
```

## 📧 Contact

For questions about this research, please contact:
- **Author**: [Your Name]
- **Email**: [Your Email]
- **Institution**: [Your Institution]

## ⚠️ Data Availability

Raw data used in this analysis is available from:
- **SPEI**: [SPEIbase](http://spei.csic.es/database.html)
- **GRACE/TWS**: [NASA JPL](https://grace.jpl.nasa.gov/)
- **MODIS**: [NASA EARTHDATA](https://earthdata.nasa.gov/)
- **GOSIF**: [Global Ecology Group](http://globalecology.unh.edu/)

Processed datasets and complete results will be made available upon publication.

## 🔒 License

**Copyright © 2025 [Your Name]. All Rights Reserved.**

This code is provided for review purposes. Redistribution or use for academic research requires permission from the author until publication. After publication, this repository will be released under the MIT License.

---

## 🔄 Status

- [x] Data collection and preprocessing
- [x] Analysis pipeline development
- [x] Statistical analysis
- [x] Visualization
- [ ] Paper under revision
- [ ] Submitted for peer review
- [ ] Published

---

**Last Updated**: October 2025
