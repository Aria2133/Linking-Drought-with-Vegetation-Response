/*******************************************************
*  Monthly LST (°C) — MODIS Aqua MYD21C1 v061          *
*  QA: keep QC_Day bits0-1 < 2  |  Drop daily images
*      with cloud > 30% over California                 *
*******************************************************/

// 1) Region
var california = ee.FeatureCollection('TIGER/2018/States')
                     .filter(ee.Filter.eq('NAME', 'California'));

// 2) Parametersa
var YEAR = 2001;          // سال مدنظر
var MAX_CLOUD = 0.30;     // 30 %

// 3) Prepare each daily image
function prep(img) {
  var qc = img.select('QC_Day');
  var goodPix = qc.bitwiseAnd(3).lt(2);   // bits 0-1: 0 or 1

  // cloud fraction over region = 1 - (good / total-valid)
  var ones = ee.Image.constant(1);
  var goodBand  = ones.updateMask(goodPix).rename('good');
  var totalBand = ones.updateMask(img.select('LST_Day').mask()).rename('total');

  var stats = ee.Image.cat(goodBand, totalBand).reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: california,
    scale: 5600,
    maxPixels: 1e11
  });

  var good  = ee.Number(stats.get('good', 0));
  var total = ee.Number(stats.get('total', 0));
  var cloudFrac = ee.Algorithms.If(total.gt(0),
                    ee.Number(1).subtract(good.divide(total)), 1);

  // K → °C  (بدون ×0.02 چون در EE از قبل K است)
  var lstC = img.select('LST_Day').subtract(273.15)
                .updateMask(goodPix)
                .rename('LST_Celsius');

  return lstC.copyProperties(img, ['system:time_start'])
             .set('cloudFrac', cloudFrac);
}

// 4) Monthly loop & export
print('Processing all 12 months of', YEAR);
for (var m = 1; m <= 12; m++) {
  var start = ee.Date.fromYMD(YEAR, m, 1);
  var end   = start.advance(1, 'month');

  var coll = ee.ImageCollection('MODIS/061/MYD21C1')
                .filterDate(start, end)
                .filterBounds(california)
                .map(prep)
                .filter(ee.Filter.lte('cloudFrac', MAX_CLOUD));

  var monthly = coll.mean()
                    .clip(california)
                    .rename('LST_Celsius')
                    .set({
                      year : YEAR,
                      month: m,
                      cloud_filter: '<= ' + (MAX_CLOUD*100) + '%',
                      src  : 'MODIS/061/MYD21C1'
                    });

  var mm = (m < 10 ? '0' + m : m);
  var name = 'MYD21C1_CA_LST_' + YEAR + '_' + mm;

  Export.image.toDrive({
    image: monthly,
    description: name,
    folder: 'MODIS_CA_Monthly_LST',
    fileNamePrefix: name,
    region: california,
    scale: 5600,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true}
  });

  print('Queued:', name);
}
print('✅ Exports queued.');

// 5) Quick visual check (June)
var vizMonth = 6;
var viz = ee.ImageCollection('MODIS/061/MYD21C1')
            .filterDate(ee.Date.fromYMD(YEAR, vizMonth, 1),
                        ee.Date.fromYMD(YEAR, vizMonth, 1).advance(1,'month'))
            .filterBounds(california)
            .map(prep)
            .filter(ee.Filter.lte('cloudFrac', MAX_CLOUD))
            .mean()
            .clip(california);

Map.centerObject(california, 6);
Map.addLayer(viz, {
  min: 5, max: 45,
  palette: ['040274','0502a3','307ef3','30c8e2','3be285','b5e22e',
            'ffd611','ff8b13','ff0000','a71001']
}, 'LST °C – Jun ' + YEAR);
