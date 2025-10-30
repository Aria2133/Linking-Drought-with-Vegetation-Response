/*******************************************************
*  Monthly LST (°C) — MODIS Aqua MYD21C1 v061
*  Mask croplands using:
*     2001–2007: MODIS LC_Type1
*     2008–2012: CDL cropland
*     2013–2021: CDL cultivated
*  QA: QC_Day bits0-1 < 2, drop cloudy days > 30%
*******************************************************/

// 1) California boundary
var california = ee.FeatureCollection('TIGER/2018/States')
                     .filter(ee.Filter.eq('NAME', 'California'));

// 2) Parameters
var YEAR = 2021;        // ← Change this year
var MAX_CLOUD = 0.30;   // 30% cloud limit

// 3) Ag mask function
function getAgMask(year) {
  if (year <= 2007) {
    var lc = ee.ImageCollection('MODIS/061/MCD12Q1')
                .filter(ee.Filter.calendarRange(year, year, 'year'))
                .first()
                .select('LC_Type1');
    return lc.neq(12).and(lc.neq(14)).rename('agMask');

  } else if (year <= 2012) {
    var cropland = ee.Image('USDA/NASS/CDL/' + year).select('cropland');
    return cropland.gte(61).rename('agMask');  // keep non-crop

  } else {
    var cultivated = ee.Image('USDA/NASS/CDL/' + year).select('cultivated');
    return cultivated.eq(1).rename('agMask');  // keep non-cultivated
  }
}
var agMask = getAgMask(YEAR);

// 4) QA + LST + cloud check per daily image
function prep(img) {
  var qc = img.select('QC_Day');
  var goodPix = qc.bitwiseAnd(3).lt(2); // QC bits 0-1 < 2

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

  var lstC = img.select('LST_Day')
                .subtract(273.15)
                .updateMask(goodPix)
                .rename('LST_Celsius');

  // Apply agMask
  lstC = lstC.updateMask(agMask);

  return lstC.copyProperties(img, ['system:time_start'])
             .set('cloudFrac', cloudFrac);
}

// 5) Monthly loop
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
                      mask_source: (YEAR <= 2007 ? 'MODIS LC_Type1' :
                                   YEAR <= 2012 ? 'USDA CDL cropland' :
                                                  'USDA CDL cultivated'),
                      cloud_filter: '<= ' + (MAX_CLOUD*100) + '%',
                      source: 'MODIS/061/MYD21C1'
                    });

  var mm = (m < 10 ? '0' + m : m);
  var name = 'MYD21C1_CA_LST_masked_' + YEAR + '_' + mm;

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

// 6) Visual preview for June
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

print('✅ LST masked export with cropland exclusion – Done:', YEAR);
