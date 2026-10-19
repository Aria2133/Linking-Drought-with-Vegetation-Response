// -------------------------------------------------------------------
//  Monthly Maximum VPD (GRIDMET, resampled to 5km) – California
//  Period: 2001–2021
// -------------------------------------------------------------------

// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));

// 2) Years to process
var START_YEAR = 2001;
var END_YEAR   = 2021;

// 3) Loop through each year & month
for (var year = START_YEAR; year <= END_YEAR; year++) {
  for (var month = 1; month <= 12; month++) {
    var startDate = ee.Date.fromYMD(year, month, 1);
    var endDate   = startDate.advance(1, 'month');

    // GRIDMET daily VPD (kPa)
    var monthlyMaxVPD = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
          .filterDate(startDate, endDate)
          .filterBounds(california)
          .select('vpd')
          .max()                       // <-- monthly maximum
          .clip(california)
          .rename('VPD_max');

    // Reproject to 5 km for consistency with your other indices
    monthlyMaxVPD = monthlyMaxVPD.reproject({
      crs: 'EPSG:4326',
      scale: 5000
    });

    // metadata
    monthlyMaxVPD = monthlyMaxVPD.set({
      year : year,
      month: month,
      description: 'Monthly maximum VPD (kPa) – GRIDMET, 5 km resolution',
      source: 'IDAHO_EPSCOR/GRIDMET',
      original_resolution: '4 km approx.',
      resampled_resolution: '5 km'
    });

    // filename
    var mm = (month < 10 ? '0' + month : month.toString());
    var filename = 'VPDmax_GRIDMET_CA_5km_' + year + '_' + mm;

    // Export
    Export.image.toDrive({
      image          : monthlyMaxVPD,
      description    : filename,
      folder         : 'VPDmax_GRIDMET_CA_TIFs',
      fileNamePrefix : filename,
      region         : california,
      scale          : 5000,
      crs            : 'EPSG:4326',
      maxPixels      : 1e13,
      fileFormat     : 'GeoTIFF',
      formatOptions  : {cloudOptimized: true}
    });

    print('Export queued for:', filename);
  }
}

// -------------------------------------------------------------------
//  Quick visualization: July 2014
// -------------------------------------------------------------------
var vizYear = 2014;
var vizMonth = 7;
var vizStart = ee.Date.fromYMD(vizYear, vizMonth, 1);
var vizEnd   = vizStart.advance(1, 'month');

var vizVPD = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
      .filterDate(vizStart, vizEnd)
      .filterBounds(california)
      .select('vpd')
      .max()
      .clip(california)
      .reproject({crs: 'EPSG:4326', scale: 5000});

var vpdVizParams = {
  min: 0.0,
  max: 6.0,
  palette: ['#313695','#4575b4','#74add1','#abd9e9',
            '#ffffbf','#fdae61','#f46d43','#d73027','#a50026']
};

Map.centerObject(california, 6);
Map.addLayer(vizVPD, vpdVizParams, 'Max VPD – July ' + vizYear);

print('Visualization: GRIDMET monthly max VPD for California');
