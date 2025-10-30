# Google Earth Engine Scripts

This directory contains JavaScript code for satellite data processing using Google Earth Engine. These scripts support drought and vegetation analysis in California by acquiring, processing, and exporting multi-year time series data.

---

## 📋 Overview

These scripts process satellite imagery and climate data from multiple sources (MODIS, GRACE, ERA5) to create standardized datasets for drought-vegetation analysis. All data is:
- Clipped to California state boundary
- Resampled to consistent spatial resolution (500m-5km)
- Aggregated to monthly temporal resolution
- Exported as GeoTIFF files for Python analysis

**Key Features:**
- Automated monthly/yearly processing loops
- Agricultural area masking for natural vegetation analysis
- Multiple spatial resolutions for different analyses
- Site-specific data extraction for validation
- Quality control and cloud masking

---

## 📁 Script Inventory

### 🌍 **Study Area & Land Cover**

#### `4_types_land`
**Purpose:** Extracts and processes California's four major natural land cover types
- Grasslands
- Natural Forests  
- Savannas
- Shrublands

**Output:** Land cover masks for ecosystem-specific analysis

#### `CROPLAND`
**Purpose:** Identifies agricultural/cropland areas for exclusion from natural vegetation analysis
**Data Source:** USDA Cropland Data Layer
**Output:** Binary cropland mask

#### `HUC-8`
**Purpose:** Defines watershed boundaries using Hydrologic Unit Code (HUC-8) regions
**Use Case:** Regional drought analysis and water resource assessment

---

### 🌡️ **Climate & Drought Indices**

#### **Land Surface Temperature (LST)**

**`LST`**
- **Product:** MODIS MOD11A2 (8-day, 1km)
- **Processing:** Monthly mean temperature
- **Output:** Statewide LST time series
- **Resolution:** 1km native

**`California_LST_NO_AG`**
- Same as LST but with agricultural areas masked out
- Focuses on natural vegetation temperature response

**`LST_timelaps`**
- Creates temporal visualization of LST changes
- Useful for drought event identification

**`Tonzi_LST`**
- Site-specific LST extraction for Tonzi Ranch AmeriFlux site
- Used for ground-truthing and validation

---

#### **Drought Indices**

**`SPEI`**
- **Index:** Standardized Precipitation-Evapotranspiration Index
- **Source:** SPEI Global Database
- **Processing:** Monthly time series
- **Output:** Multi-year SPEI values

**`SPEI (2024)`**
- Updated version with latest 2024 data
- Includes recent drought events

**`VPD`**
- **Index:** Vapor Pressure Deficit
- **Source:** ERA5 climate reanalysis
- **Calculation:** From temperature and humidity
- **Output:** Monthly VPD time series

---

#### **Water Storage**

**`GRACE`**
- **Product:** GRACE/GRACE-FO Total Water Storage (TWS)
- **Resolution:** ~300km monthly
- **Processing:** Resampled to 5km for integration with other datasets
- **Output:** Monthly TWS anomalies

**`GRACE_1`**
- Alternative GRACE processing method
- Different temporal filtering approach

---

### 🌿 **Vegetation Indices**

#### **NDVI (Normalized Difference Vegetation Index)**

**`NDVI`**
- **Product:** MODIS MOD13Q1 (16-day, 250m)
- **Processing:** Monthly mean NDVI
- **Formula:** (NIR - Red) / (NIR + Red)
- **Scale Factor:** 0.0001 applied
- **Resolution:** 250m native

**`NDVI-5km`**
```javascript
