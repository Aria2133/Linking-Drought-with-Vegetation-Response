import rasterio
import numpy as np

# File paths
files = {
    'Grasslands': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Grasslands_2021.tif",
    'Natural Forests': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_NaturalForests_2021.tif",
    'Savannas': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Savannas_2021.tif",
    'Shrublands': r"C:\Users\LOQ\OneDrive\Documents\Msc Water Resources Management\Thesis\Data\CA_LandMasks\CA_Shrublands_2021.tif"
}


# Function to calculate area
def calculate_area(file_path, land_type_name):
    with rasterio.open(file_path) as src:
        # Read the raster data
        data = src.read(1)

        # Get pixel resolution (in degrees)
        pixel_width = src.transform[0]  # degrees longitude
        pixel_height = abs(src.transform[4])  # degrees latitude

        # Get approximate center latitude for California
        bounds = src.bounds
        center_lat = (bounds.top + bounds.bottom) / 2

        # Convert degrees to km (approximate)
        # 1 degree longitude at this latitude ≈ 111 km * cos(latitude)
        # 1 degree latitude ≈ 111 km
        km_per_deg_lon = 111.32 * np.cos(np.radians(center_lat))
        km_per_deg_lat = 111.32

        # Calculate pixel area in km²
        pixel_area_km2 = (pixel_width * km_per_deg_lon) * (pixel_height * km_per_deg_lat)

        # Count pixels where value = 1
        count_ones = np.sum(data == 1)

        # Calculate total area
        total_area_km2 = count_ones * pixel_area_km2

        # Print results
        print(f"\n{land_type_name}:")
        print(f"  Pixel resolution: {pixel_width:.6f}° × {pixel_height:.6f}°")
        print(f"  Pixel area: {pixel_area_km2:.4f} km²")
        print(f"  Number of pixels (value=1): {count_ones:,}")
        print(f"  Total area: {total_area_km2:,.2f} km²")

        return total_area_km2, count_ones


# Calculate for all land types
print("=" * 60)
print("CALIFORNIA LAND TYPE AREAS")
print("=" * 60)

results = {}
for land_type, file_path in files.items():
    area, pixels = calculate_area(file_path, land_type)
    results[land_type] = {'area_km2': area, 'pixels': pixels}

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

total_area = sum(r['area_km2'] for r in results.values())
total_pixels = sum(r['pixels'] for r in results.values())

for land_type, data in results.items():
    percentage = (data['area_km2'] / total_area) * 100
    print(f"{land_type:20s}: {data['area_km2']:>12,.2f} km² ({percentage:>5.1f}%)")

print("-" * 60)
print(f"{'TOTAL':20s}: {total_area:>12,.2f} km²")
print("=" * 60)

# Format for your paper
print("\n" + "=" * 60)
print("FOR YOUR PAPER (copy-paste this):")
print("=" * 60)
print(f"""
This study focuses on California's natural ecosystems spanning 
{total_area:,.0f} km² across four land types: grasslands 
({results['Grasslands']['area_km2']:,.0f} km², {(results['Grasslands']['area_km2'] / total_area) * 100:.1f}%), 
natural forests ({results['Natural Forests']['area_km2']:,.0f} km², {(results['Natural Forests']['area_km2'] / total_area) * 100:.1f}%), 
savannas ({results['Savannas']['area_km2']:,.0f} km², {(results['Savannas']['area_km2'] / total_area) * 100:.1f}%), 
and shrublands ({results['Shrublands']['area_km2']:,.0f} km², {(results['Shrublands']['area_km2'] / total_area) * 100:.1f}%) 
(Figure 2).
""")