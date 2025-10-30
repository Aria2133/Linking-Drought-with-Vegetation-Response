// -------------------------------------------------------------------
//  Monthly GNDVI (MODIS MOD09A1 v061, resampled to 5km) – California
// -------------------------------------------------------------------
// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));

// 2) Year to process
var YEAR_TO_PROCESS = 2008;      // change as needed (2000-present)
print('Processing 12 months of GNDVI for:', YEAR_TO_PROCESS);

// 3) Loop through the 12 months
for (var month = 1; month <= 12; month++) {
  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate = startDate.advance(1, 'month');

  // MOD09A1 collection for the month
  var monthlyGNDVI = ee.ImageCollection('MODIS/061/MOD09A1')
        .filterDate(startDate, endDate)
        .filterBounds(california)
        .select(['sur_refl_b02', 'sur_refl_b04']) // Select NIR (b02) and Green (b04)
        .map(function(img) {
          // Apply scale factor (0.0001) to reflectance bands
          var nir = img.select('sur_refl_b02').multiply(0.0001);
          var green = img.select('sur_refl_b04').multiply(0.0001);

          // Calculate GNDVI
          var gndvi = nir.subtract(green)
                        .divide(nir.add(green))
                        .rename('GNDVI');

          // Set projection to MODIS sinusoidal
          return gndvi.setDefaultProjection('SR-ORG:6974', null, 463.312716527778) // MOD09A1 projection, 500m scale
                      .copyProperties(img, ['system:time_start']);
        })
        .mean()                       // Monthly average
        .clip(california)
        .rename('GNDVI');

  // Reproject to 5km resolution
  monthlyGNDVI = monthlyGNDVI
        .reproject({
          crs: 'EPSG:4326',
          scale: 5000                 // 5km = 5000 meters
        });

  // Metadata
  monthlyGNDVI = monthlyGNDVI.set({
    year: YEAR_TO_PROCESS,
    month: month,
    description: 'Monthly mean GNDVI (scaled, unitless) - 5km resolution',
    source: 'MODIS/061/MOD09A1',
    original_resolution: '500 m',
    resampled_resolution: '5 km'
  });

  // File name helpers
  var mm = (month < 10 ? '0' + month : month.toString());
  var filename = 'GNDVI_MOD09A1_CA_5km_' + YEAR_TO_PROCESS + '_' + mm;

  // Export to Drive
  Export.image.toDrive({
    image: monthlyGNDVI,
    description: filename,
    folder: 'GNDVI_MOD09A1_5km_TIFs',
    fileNamePrefix: filename,
    region: california,
    scale: 5000,             // 5km resolution
    crs: 'EPSG:4326',
    maxPixels: 1e9,         // Reduced from 1e13 since we have fewer pixels
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });

  print('Export queued for:', filename);
}

// -------------------------------------------------------------------
//  Quick visual check – June of the selected year
// -------------------------------------------------------------------
var vizMonth = 6;
var vizStart = ee.Date.fromYMD(YEAR_TO_PROCESS, vizMonth, 1);
var vizEnd = vizStart.advance(1, 'month');

var vizGNDVI = ee.ImageCollection('MODIS/061/MOD09A1')
      .filterDate(vizStart, vizEnd)
      .filterBounds(california)
      .select(['sur_refl_b02', 'sur_refl_b04'])
      .map(function(img) {
        var nir = img.select('sur_refl_b02').multiply(0.0001);
        var green = img.select('sur_refl_b04').multiply(0.0001);
        var gndvi = nir.subtract(green)
                      .divide(nir.add(green))
                      .rename('GNDVI');
        return gndvi.setDefaultProjection('SR-ORG:6974', null, 463.312716527778);
      })
      .mean()
      .clip(california);

// Reproject for visualization
vizGNDVI = vizGNDVI.reproject({
  crs: 'EPSG:4326',
  scale: 5000
});

var gndviVizParams = {
  min: 0.0,
  max: 0.9,
  palette: ['#654321', '#c2b280', '#ffff64', '#a1d99b', '#31a354', '#006837']
};

Map.centerObject(california, 6);
Map.addLayer(vizGNDVI, gndviVizParams, 'GNDVI 5km – June ' + YEAR_TO_PROCESS);

print('Data reduction: From ~500m to 5km resolution');
print('Pixel reduction factor: ~10x10 = 100x fewer pixels');
print('File size reduction: Approximately 100x smaller files');