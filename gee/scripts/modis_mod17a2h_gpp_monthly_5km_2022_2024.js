// -------------------------------------------------------------------
//  Monthly GPP (MODIS MOD17A2H v061, resampled to 5km) – California
// -------------------------------------------------------------------
// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));

// 2) Year to process
var YEAR_TO_PROCESS = 2024;   // change as needed (2001–present)
print('Processing 12 months of GPP for:', YEAR_TO_PROCESS);

// 3) Helper: scale + select GPP on each 8-day image
function getGPP(img) {
  // MOD17A2H GPP band is 'Gpp' with scale factor 0.0001 (kg C m^-2 per 8-day)
  var gpp = img.select('Gpp').multiply(0.0001).rename('GPP');
  // Set default projection (to keep behavior consistent before reprojection)
  return gpp.setDefaultProjection('SR-ORG:6974', null, 231.656358263889)
            .copyProperties(img, ['system:time_start']);
}

// 4) Loop through the 12 months
for (var month = 1; month <= 12; month++) {
  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate   = startDate.advance(1, 'month');

  // Build monthly GPP total from 8-day composites (sum, not mean)
  var monthlyGPP = ee.ImageCollection('MODIS/061/MOD17A2H')
        .filterDate(startDate, endDate)
        .filterBounds(california)
        .map(getGPP)          // scale & select GPP per 8-day image
        .sum()                // monthly total GPP (kg C m^-2 month^-1)
        .clip(california)
        .rename('GPP');

  // Reproject to 5km resolution (match your NIRv grid spec)
  monthlyGPP = monthlyGPP.reproject({
    crs: 'EPSG:4326',
    scale: 5000               // 5 km
  });

  // Add metadata
  monthlyGPP = monthlyGPP.set({
    year : YEAR_TO_PROCESS,
    month: month,
    description: 'Monthly total GPP (kg C m^-2) - 5km resolution',
    source: 'MODIS/061/MOD17A2H',
    original_resolution: '500 m',
    resampled_resolution: '5 km',
    note: 'Monthly sum of 8-day GPP (Gpp × 0.0001, then summed)'
  });

  // Export file name
  var mm = (month < 10 ? '0' + month : '' + month);
  var filename = 'GPP_MOD17A2H_CA_5km_' + YEAR_TO_PROCESS + '_' + mm;

  // Export to Drive
  Export.image.toDrive({
    image          : monthlyGPP,
    description    : filename,
    folder         : 'GPP_MOD17A2H_5km_TIFs',
    fileNamePrefix : filename,
    region         : california,
    scale          : 5000,             // 5 km
    crs            : 'EPSG:4326',
    maxPixels      : 1e9,
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

var vizGPP = ee.ImageCollection('MODIS/061/MOD17A2H')
      .filterDate(vizStart, vizEnd)
      .filterBounds(california)
      .map(getGPP)
      .sum()
      .clip(california);

// Reproject visualization to 5km as well
vizGPP = vizGPP.reproject({
  crs: 'EPSG:4326',
  scale: 5000
});

// Visualization parameters (tweak max to match your biome)
var gppVizParams = {
  min: 0.0,
  max: 2.0, // kg C m^-2 month^-1 (adjust if needed)
  palette: ['#f7fcf5','#c7e9c0','#74c476','#238b45','#00441b']
};

Map.centerObject(california, 6);
Map.addLayer(vizGPP, gppVizParams, 'GPP 5km – June ' + YEAR_TO_PROCESS);

print('Monthly total GPP is the sum of 8-day composites (scaled by 0.0001).');
