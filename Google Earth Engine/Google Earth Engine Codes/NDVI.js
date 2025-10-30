// -------------------------------------------------------------------
//  Monthly NDVI (MODIS MOD13Q1 v061, resampled to 5km)  –  California
// -------------------------------------------------------------------
// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));

// 2) Year to process
var YEAR_TO_PROCESS = 2021;      // change as needed (2000-present)
print('Processing 12 months of NDVI for:', YEAR_TO_PROCESS);

// 3) Loop through the 12 months
for (var month = 1; month <= 12; month++) {
  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate   = startDate.advance(1, 'month');

  // NDVI collection for the month
  var monthlyNDVI = ee.ImageCollection('MODIS/061/MOD13Q1')
        .filterDate(startDate, endDate)
        .filterBounds(california)
        .select('NDVI')               // 16-day NDVI band
        // scale to real units and set projection immediately
        .map(function(img){
          return img.multiply(0.0001)
                    .setDefaultProjection('SR-ORG:6974', null, 231.656358263889) // MODIS projection and scale
                    .copyProperties(img, ['system:time_start']);
        })
        .mean()                       // monthly average
        .clip(california)
        .rename('NDVI');

  // Simple approach: reproject directly to 5km without reduceResolution
  monthlyNDVI = monthlyNDVI
        .reproject({
          crs: 'EPSG:4326',
          scale: 5000                 // 5km = 5000 meters
        });

  // metadata
  monthlyNDVI = monthlyNDVI.set({
    year : YEAR_TO_PROCESS,
    month: month,
    description: 'Monthly mean NDVI (scaled, unitless) - 5km resolution',
    source: 'MODIS/061/MOD13Q1',
    original_resolution: '250 m',
    resampled_resolution: '5 km'
  });

  // file name helpers
  var mm = (month < 10 ? '0' + month : month.toString());
  var filename = 'NDVI_MOD13Q1_CA_5km_' + YEAR_TO_PROCESS + '_' + mm;

  // Export to Drive
  Export.image.toDrive({
    image          : monthlyNDVI,
    description    : filename,
    folder         : 'NDVI_MOD13Q1_5km_TIFs',
    fileNamePrefix : filename,
    region         : california,
    scale          : 5000,             // 5km resolution
    crs            : 'EPSG:4326',
    maxPixels      : 1e9,              // Reduced from 1e13 since we have fewer pixels
    fileFormat     : 'GeoTIFF',
    formatOptions  : {cloudOptimized: true}
  });

  print('Export queued for:', filename);
}

// -------------------------------------------------------------------
//  Quick visual check – June of the selected year
// -------------------------------------------------------------------
var vizMonth = 6;
var vizStart = ee.Date.fromYMD(YEAR_TO_PROCESS, vizMonth, 1);
var vizEnd   = vizStart.advance(1, 'month');

var vizNDVI = ee.ImageCollection('MODIS/061/MOD13Q1')
      .filterDate(vizStart, vizEnd)
      .filterBounds(california)
      .select('NDVI')
      .map(function(img){
        return img.multiply(0.0001)
                  .setDefaultProjection('SR-ORG:6974', null, 231.656358263889); // MODIS projection
      })
      .mean()
      .clip(california);

// Simple reproject for visualization (GEE will handle the resampling automatically)
vizNDVI = vizNDVI.reproject({
  crs: 'EPSG:4326',
  scale: 5000
});

var ndviVizParams = {
  min: 0.0,
  max: 0.9,
  palette: ['#654321','#c2b280','#ffff64','#a1d99b','#31a354','#006837']
};

Map.centerObject(california, 6);
Map.addLayer(vizNDVI, ndviVizParams, 'NDVI 5km – June ' + YEAR_TO_PROCESS);

print('Data reduction: From ~231m to 5km resolution');
print('Pixel reduction factor: ~21x21 = 441x fewer pixels');
print('File size reduction: Approximately 441x smaller files');