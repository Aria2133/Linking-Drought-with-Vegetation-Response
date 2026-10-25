// -------------------------------------------------------------------
//  Re-export of monthly MOD17A2H GPP for 2022 (California, 5 km)
// -------------------------------------------------------------------
// The 2022 files exported with modis_mod17a2h_gpp_monthly_5km_2022_2024.js are invalid: 9 of the 12 monthly GeoTIFFs
// are byte-identical and two more form a second identical pair (results/audit/gpp_table_s8/mod17_export_hashes_2022_2024.csv).
// Processing is identical to that script (Gpp x 0.0001, sum of the 8-day composites starting in the month, kg C m^-2 month^-1,
// 5 km, EPSG:4326). Exports go to a NEW Drive folder so the old files are not overwritten or mixed up.
// Before starting the tasks, check the printed table: the 12 California means must all differ and follow a seasonal cycle.
// After download, run: python src/ecological/gpp/mod17_monthly.py verify --folder <download folder> --year 2022

var california = ee.FeatureCollection('TIGER/2018/States')
                    .filter(ee.Filter.eq('NAME', 'California'));
var YEAR_TO_PROCESS = 2022;
var FOLDER = 'GPP_MOD17A2H_5km_TIFs_2022_reexport';

function getGPP(img) {
  var gpp = img.select('Gpp').multiply(0.0001).rename('GPP');
  return gpp.setDefaultProjection('SR-ORG:6974', null, 231.656358263889)
            .copyProperties(img, ['system:time_start']);
}

function monthlyGPP(month) {
  var startDate = ee.Date.fromYMD(YEAR_TO_PROCESS, month, 1);
  var endDate   = startDate.advance(1, 'month');
  var col = ee.ImageCollection('MODIS/061/MOD17A2H')
        .filterDate(startDate, endDate)
        .filterBounds(california);
  var img = col.map(getGPP).sum().clip(california).rename('GPP')
        .reproject({crs: 'EPSG:4326', scale: 5000})
        .set({
          year : YEAR_TO_PROCESS,
          month: month,
          n_8day_composites: col.size(),
          description: 'Monthly total GPP (kg C m^-2) - 5km resolution',
          source: 'MODIS/061/MOD17A2H',
          note: 'Monthly sum of 8-day GPP (Gpp x 0.0001, then summed); 2022 re-export'
        });
  return img;
}

// Pre-export check: one row per month with the number of 8-day composites and the California mean.
var check = ee.FeatureCollection(ee.List.sequence(1, 12).map(function (m) {
  var img = monthlyGPP(ee.Number(m).toInt());
  var mean = img.reduceRegion({reducer: ee.Reducer.mean(), geometry: california.geometry(), scale: 5000, maxPixels: 1e9});
  return ee.Feature(null, {month: m, n_8day_composites: img.get('n_8day_composites'), ca_mean_kgC_m2_month: mean.get('GPP')});
}));
print('2022 pre-export check (means must differ month to month):', check);

for (var month = 1; month <= 12; month++) {
  var mm = (month < 10 ? '0' + month : '' + month);
  var filename = 'GPP_MOD17A2H_CA_5km_' + YEAR_TO_PROCESS + '_' + mm;
  Export.image.toDrive({
    image          : monthlyGPP(month),
    description    : filename,
    folder         : FOLDER,
    fileNamePrefix : filename,
    region         : california,
    scale          : 5000,
    crs            : 'EPSG:4326',
    maxPixels      : 1e9,
    fileFormat     : 'GeoTIFF',
    formatOptions  : {cloudOptimized: true}
  });
  print('Export queued for:', filename);
}
