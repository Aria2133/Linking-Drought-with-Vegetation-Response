// -------------------------------------------------------------------
// Annual GPP (MOD17A3HGF v061) – California
// -------------------------------------------------------------------

// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
  .filter(ee.Filter.eq('NAME', 'California'));

// 2) Year range to process
var startYear = 2001;
var endYear = 2021;

// Load MOD17A3HGF (annual, gap-filled)
var modisAnnual = ee.ImageCollection('MODIS/061/MOD17A3HGF')
  .filter(ee.Filter.calendarRange(startYear, endYear, 'year'))
  .filterBounds(california)
  .select('Gpp');

// Choose band: Gpp (Gross PP)

// Apply scale factor, mask fill values, and clip
var modisScaled = modisAnnual.map(function(img) {
  // Mask invalid/fill values (raw values > 65500 are fills)
  var masked = img.updateMask(img.gte(0).and(img.lte(65500)));
  return masked.multiply(0.0001) // original scale factor
    .clip(california)
    .copyProperties(img, ['system:time_start']);
});

// --- Loop through years and export ---
// The export loop remains the same, but it is commented out for
// direct visualization. Uncomment to run exports.

for (var year = startYear; year <= endYear; year++) {
  var yearlyImg = modisScaled.filter(ee.Filter.calendarRange(year, year, 'year'))
    .first(); // annual dataset has 1 image per year

  // Reproject to 500m (or coarser if needed)
  yearlyImg = yearlyImg.reproject({
    crs: 'EPSG:4326',
    scale: 5000
  });

  var filename = 'GPP_MOD17A3HGFCA' + year;
  Export.image.toDrive({
    image: yearlyImg,
    description: filename,
    folder: 'MOD17A3HGF_Annual_CA',
    fileNamePrefix: filename,
    region: california,
    scale: 5000,
    crs: 'EPSG:4326',
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });
  print('Export queued for:', filename);
}


// --- Improved visualization example for 2021 ---
var viz2021 = modisScaled.filter(ee.Filter.calendarRange(2021, 2021, 'year')).first();

// Define a new, more detailed visualization palette for GPP
var gppVizParams = {
  bands: ['Gpp'],
  min: 0,
  max: 6.55, // Adjusted max value based on raw max (65500) * scale (0.0001)
  palette: [
    'ffffff', // White for very low GPP
    'ce7e45', // Tan
    'df923d', // Light tan
    'f1b555', // Orange
    'fcd163', // Yellow
    '99b718', // Light green
    '74a901', // Medium green
    '66a000', // Darker green
    '529400', // Dark green
    '3e8601', // Very dark green
    '207401', // Forest green
    '056201', // Deep forest green
    '004c00', // Darker
    '023b01', // Even darker
    '012e01', // Deep green
    '011d01', // Almost black green
    '011301'  // Black green for highest
  ]
};

Map.centerObject(california, 6);
Map.addLayer(viz2021, gppVizParams, 'GPP 2021 (Improved Viz)');