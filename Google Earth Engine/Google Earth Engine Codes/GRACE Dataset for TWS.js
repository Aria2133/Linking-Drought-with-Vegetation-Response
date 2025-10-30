var california = ee.FeatureCollection('TIGER/2018/States')
                  .filter(ee.Filter.eq('NAME', 'California'));

// 📅 Set the year you want to process
var YEAR_TO_PROCESS = 2002;

print('Processing 12 months of TWS (GRACE Mascon) for:', YEAR_TO_PROCESS);
for (var month = 1; month <= 12; month++) {

  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate = startDate.advance(1, 'month');

  var twsImage = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON')
    .filterDate(startDate, endDate)
    .select('lwe_thickness')  // Liquid Water Equivalent thickness
    .mean()
    .clip(california)
    .set({
      'year': YEAR_TO_PROCESS,
      'month': month,
      'source': 'GRACE Mascon V04',
      'unit': 'cm (liquid water equivalent)'
    });

  var monthStr = month < 10 ? '0' + month : month.toString();
  var filename = 'TWS_GRACE_CA_' + YEAR_TO_PROCESS + '_' + monthStr;

  Export.image.toDrive({
    image: twsImage,
    description: filename,
    folder: 'TWS_GRACE_TIFs',
    fileNamePrefix: filename,
    region: california,
    scale: 100000, // GRACE native ~300km, 100km res is more appropriate
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    crs: 'EPSG:4326'
  });

  print('Export queued for:', filename);
}
// 🔍 Visualize June of the selected year
var vizMonth = 6;

var vizDateStart = ee.Date.fromYMD(YEAR_TO_PROCESS, vizMonth, 1);
var vizDateEnd = vizDateStart.advance(1, 'month');

var twsToVisualize = ee.ImageCollection('NASA/GRACE/MASS_GRIDS_V04/MASCON')
  .filterDate(vizDateStart, vizDateEnd)
  .select('lwe_thickness')
  .mean()
  .clip(california);

// Visualization parameters
var twsViz = {
  min: -25,
  max: 25,
  palette: ['#d7191c','#fdae61','#ffffbf','#a6d96a','#1a9641']
};

// Add to map
Map.centerObject(california, 6);
Map.addLayer(twsToVisualize, twsViz, 'TWS (June ' + YEAR_TO_PROCESS + ')');
