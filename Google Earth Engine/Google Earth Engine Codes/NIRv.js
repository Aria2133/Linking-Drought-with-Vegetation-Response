// -------------------------------------------------------------------
//  Monthly NIRv (MODIS MOD13Q1 v061, resampled to 5km)  –  California
// -------------------------------------------------------------------
// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));

// 2) Year to process
var YEAR_TO_PROCESS = 2021;   // change as needed (2000-present)
print('Processing 12 months of NIRv for:', YEAR_TO_PROCESS);

// 3) Helper: scale + compute NIRv on each 16-day image
function addNIRv(img) {
  var ndvi = img.select('NDVI').multiply(0.0001);       // scale NDVI
  var nir  = img.select('sur_refl_b02').multiply(0.0001); // scale NIR reflectance
  var nirv = ndvi.multiply(nir).rename('NIRv');         // NDVI × ρNIR

  // Set default projection immediately after computation
  return nirv.setDefaultProjection('SR-ORG:6974', null, 231.656358263889)
             .copyProperties(img, ['system:time_start']);
}

// 4) Loop through the 12 months
for (var month = 1; month <= 12; month++) {
  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate   = startDate.advance(1, 'month');

  // Build monthly NIRv mean
  var monthlyNIRv = ee.ImageCollection('MODIS/061/MOD13Q1')
        .filterDate(startDate, endDate)
        .filterBounds(california)
        .map(addNIRv)          // compute NIRv per composite
        .mean()                // monthly average
        .clip(california)
        .rename('NIRv');

  // Reproject to 5km resolution
  monthlyNIRv = monthlyNIRv
        .reproject({
          crs: 'EPSG:4326',
          scale: 5000                 // 5km = 5000 meters
        });

  // Add metadata
  monthlyNIRv = monthlyNIRv.set({
    year : YEAR_TO_PROCESS,
    month: month,
    description: 'Monthly mean NIRv (unitless) - 5km resolution',
    source: 'MODIS/061/MOD13Q1',
    original_resolution: '250 m',
    resampled_resolution: '5 km'
  });

  // Export file name
  var mm = (month < 10 ? '0' + month : '' + month);
  var filename = 'NIRv_MOD13Q1_CA_5km_' + YEAR_TO_PROCESS + '_' + mm;

  // Export to Drive
  Export.image.toDrive({
    image          : monthlyNIRv,
    description    : filename,
    folder         : 'NIRv_MOD13Q1_5km_TIFs',
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

var vizNIRv = ee.ImageCollection('MODIS/061/MOD13Q1')
      .filterDate(vizStart, vizEnd)
      .filterBounds(california)
      .map(addNIRv)
      .mean()
      .clip(california);

// Reproject visualization to 5km as well
vizNIRv = vizNIRv.reproject({
  crs: 'EPSG:4326',
  scale: 5000
});

// Visualization parameters (tweak max if you like)
var nirvVizParams = {
  min: 0.0,
  max: 0.4,
  palette: ['#08306b','#41ab5d','#a1d76a','#ffffb2','#fe9929','#d95f0e','#993404']
};

Map.centerObject(california, 6);
Map.addLayer(vizNIRv, nirvVizParams, 'NIRv 5km – June ' + YEAR_TO_PROCESS);

print('Data reduction: From ~231m to 5km resolution');
print('Pixel reduction factor: ~21x21 = 441x fewer pixels');
print('File size reduction: Approximately 441x smaller files');