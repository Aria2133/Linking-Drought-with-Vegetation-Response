// ------------------------------------------------------
// Land cover masks for California (2021)
// MODIS/061/MCD12Q1 — LC_Type1 band
// ------------------------------------------------------

// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));

// 2) MODIS Land Cover for 2021
var lc2021 = ee.ImageCollection('MODIS/061/MCD12Q1')
                .filter(ee.Filter.calendarRange(2021, 2021, 'year'))
                .first()
                .select('LC_Type1');

// 3) Define masks
var naturalForests = lc2021.remap([1,2,3,4,5],[1,1,1,1,1],0).rename('naturalForests');
var savannas       = lc2021.remap([8,9],[1,1],0).rename('savannas');
var grasslands     = lc2021.remap([10],[1],0).rename('grasslands');
var shrublands     = lc2021.remap([6,7],[1,1],0).rename('shrublands');

// 4) Visualization
Map.centerObject(california, 6);
Map.addLayer(naturalForests.clip(california), {min:0, max:1, palette:['white','darkgreen']}, 'Natural Forests');
Map.addLayer(savannas.clip(california),       {min:0, max:1, palette:['white','orange']},    'Savannas');
Map.addLayer(grasslands.clip(california),     {min:0, max:1, palette:['white','yellow']},    'Grasslands');
Map.addLayer(shrublands.clip(california),     {min:0, max:1, palette:['white','brown']},     'Shrublands');

// 5) Export each mask as GeoTIFF
Export.image.toDrive({
  image: naturalForests.clip(california),
  description: 'CA_NaturalForests_2021',
  folder: 'CA_LandMasks',
  fileNamePrefix: 'CA_NaturalForests_2021',
  region: california,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: savannas.clip(california),
  description: 'CA_Savannas_2021',
  folder: 'CA_LandMasks',
  fileNamePrefix: 'CA_Savannas_2021',
  region: california,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: grasslands.clip(california),
  description: 'CA_Grasslands_2021',
  folder: 'CA_LandMasks',
  fileNamePrefix: 'CA_Grasslands_2021',
  region: california,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});

Export.image.toDrive({
  image: shrublands.clip(california),
  description: 'CA_Shrublands_2021',
  folder: 'CA_LandMasks',
  fileNamePrefix: 'CA_Shrublands_2021',
  region: california,
  scale: 500,
  crs: 'EPSG:4326',
  maxPixels: 1e13
});
