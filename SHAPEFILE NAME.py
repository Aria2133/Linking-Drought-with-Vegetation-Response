import geopandas as gpd

shapefile_path = r"C:\Users\LOQ\OneDrive\Desktop\SHAPEFILE OF CA\cb_2018_us_region_500k.shp"  # update if needed
states = gpd.read_file(shapefile_path)

print(states['NAME'].unique())
