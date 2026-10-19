// ---------------------------
// 🔹 Setup
// ---------------------------
var california = ee.FeatureCollection('TIGER/2018/States')
                  .filter(ee.Filter.eq('NAME', 'California'));

// 📅 Set the year you want to process
var YEAR_TO_PROCESS = 2024;

print('Processing 12 months of SPEI-1 (GRIDMET/DROUGHT spei30d) for:', YEAR_TO_PROCESS);

// ---------------------------
// 🔁 Loop over all 12 months
// ---------------------------
for (var month = 1; month <= 12; month++) {

  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate = startDate.advance(1, 'month');

  var speiImage = ee.ImageCollection('GRIDMET/DROUGHT')
    .filterDate(startDate, endDate)
    .select('spei30d')  // ✅ 30-day SPEI (≈ 1-month) from GRIDMET/DROUGHT
    .mean()
    .clip(california)
    .set({
      'year': YEAR_TO_PROCESS,
      'month': month,
      'source': 'GRIDMET/DROUGHT',
      'spei_timescale': '1-month (30-day)'
    });

  var monthStr = month < 10 ? '0' + month : month.toString();
  var filename = 'SPEI1_GRIDMET_CA_' + YEAR_TO_PROCESS + '_' + monthStr;

  Export.image.toDrive({
    image: speiImage,
    description: filename,
    folder: 'SPEI1_GRIDMET_TIFs',
    fileNamePrefix: filename,
    region: california,
    scale: 5000,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    crs: 'EPSG:4326'
  });

  print('Export queued for:', filename);
}

// --------------------------------------
// 🔍 Visualization of one random month
// --------------------------------------

var vizMonth = 6;

var vizDateStart = ee.Date.fromYMD(YEAR_TO_PROCESS, vizMonth, 1);
var vizDateEnd = vizDateStart.advance(1, 'month');

var speiToVisualize = ee.ImageCollection('GRIDMET/DROUGHT')
  .filterDate(vizDateStart, vizDateEnd)
  .select('spei30d')  // ✅ 30-day SPEI
  .mean()
  .clip(california);

var speiViz = {
  min: -2.5,
  max: 2.5,
  palette: ['#762a83', '#af8dc3', '#e7d4e8', '#f7f7f7', '#d9f0d3', '#7fbf7b', '#1b7837']
};

Map.centerObject(california, 6);
Map.addLayer(speiToVisualize, speiViz, 'SPEI-1 (30-day) – June ' + YEAR_TO_PROCESS);
