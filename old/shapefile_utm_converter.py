import geopandas as gpd
import os
from pathlib import Path

def convert_german_shapefile_to_utm():
    """
    Convert German shapefile to UTM Zone 32N for optimal spatial analysis.
    This eliminates coordinate transformation overhead and improves accuracy.
    """
    
    # Input and output paths
    input_shapefile = "german_shapefile/de.shp"  # Your current shapefile
    output_dir = "german_shapefile_utm32n"
    output_shapefile = f"{output_dir}/de_utm32n.shp"
    
    print("GERMAN SHAPEFILE UTM ZONE 32N CONVERSION")
    print("="*50)
    
    # Check if input exists
    if not os.path.exists(input_shapefile):
        print(f"❌ Error: Input shapefile not found: {input_shapefile}")
        print("Please update the input_shapefile path")
        return None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the shapefile
    print(f"Loading shapefile: {input_shapefile}")
    gdf = gpd.read_file(input_shapefile)
    
    print(f"✅ Loaded: {len(gdf)} regions")
    print(f"   Original CRS: {gdf.crs}")
    print(f"   Bounds: {gdf.total_bounds}")
    
    # Convert to UTM Zone 32N
    print(f"\nConverting to UTM Zone 32N (EPSG:32632)...")
    gdf_utm = gdf.to_crs('EPSG:32632')
    
    print(f"✅ Converted to UTM Zone 32N")
    print(f"   New CRS: {gdf_utm.crs}")
    print(f"   UTM Bounds (meters): {gdf_utm.total_bounds}")
    
    # Validate the conversion
    print(f"\nValidation:")
    print(f"   Geometry count: {len(gdf)} → {len(gdf_utm)}")
    print(f"   Columns preserved: {list(gdf.columns) == list(gdf_utm.columns)}")
    
    # Check for potential issues
    if gdf_utm.geometry.isna().any():
        print("⚠️  Warning: Some geometries became invalid during conversion")
    else:
        print("✅ All geometries valid after conversion")
    
    # Save the UTM version
    print(f"\nSaving UTM version to: {output_shapefile}")
    gdf_utm.to_file(output_shapefile)
    
    # Verify the saved file
    gdf_test = gpd.read_file(output_shapefile)
    print(f"✅ Verification: Saved file has {len(gdf_test)} regions in {gdf_test.crs}")
    
    print(f"\n" + "="*50)
    print("CONVERSION COMPLETE!")
    print("="*50)
    print(f"Files created in {output_dir}/:")
    for ext in ['.shp', '.shx', '.dbf', '.prj', '.cpg']:
        filename = f"de_utm32n{ext}"
        if os.path.exists(f"{output_dir}/{filename}"):
            size_kb = os.path.getsize(f"{output_dir}/{filename}") / 1024
            print(f"✅ {filename} ({size_kb:.1f} KB)")
    
    print(f"\n🎯 Usage in your code:")
    print(f'shapefile_path = "{output_shapefile}"')
    print(f'target_crs = "EPSG:32632"  # Already in UTM Zone 32N!')
    
    return output_shapefile

def compare_coordinate_systems():
    """
    Compare original vs UTM performance for spatial operations.
    """
    
    input_shapefile = "german_shapefile/de.shp"
    utm_shapefile = "german_shapefile_utm32n/de_utm32n.shp"
    
    if not os.path.exists(input_shapefile) or not os.path.exists(utm_shapefile):
        print("❌ Need both original and UTM shapefiles for comparison")
        return
    
    print("\nCOORDINATE SYSTEM COMPARISON")
    print("="*40)
    
    # Load both versions
    gdf_wgs84 = gpd.read_file(input_shapefile)
    gdf_utm = gpd.read_file(utm_shapefile)
    
    print(f"\nOriginal (WGS84):")
    print(f"   CRS: {gdf_wgs84.crs}")
    print(f"   Bounds: {gdf_wgs84.total_bounds}")
    print(f"   Units: degrees")
    
    print(f"\nUTM Zone 32N:")
    print(f"   CRS: {gdf_utm.crs}")
    print(f"   Bounds: {gdf_utm.total_bounds}")
    print(f"   Units: meters")
    
    # Test area calculations
    print(f"\nArea Calculations:")
    
    # WGS84 areas (in square degrees - not meaningful)
    wgs84_area_degrees = gdf_wgs84.geometry.area.sum()
    
    # Convert WGS84 to UTM for area calculation
    gdf_wgs84_to_utm = gdf_wgs84.to_crs('EPSG:32632')
    wgs84_converted_area = gdf_wgs84_to_utm.geometry.area.sum() / 1_000_000  # km²
    
    # UTM areas (native)
    utm_area = gdf_utm.geometry.area.sum() / 1_000_000  # km²
    
    print(f"   WGS84 (degrees²): {wgs84_area_degrees:.2f} - NOT MEANINGFUL")
    print(f"   WGS84→UTM (km²): {wgs84_converted_area:.2f}")
    print(f"   Native UTM (km²): {utm_area:.2f}")
    print(f"   Difference: {abs(wgs84_converted_area - utm_area):.6f} km²")
    
    print(f"\n🎯 Benefits of Native UTM:")
    print(f"✅ No coordinate transformation overhead")
    print(f"✅ Direct area/distance calculations in meters")
    print(f"✅ Superior numerical precision")
    print(f"✅ Faster spatial operations")

def update_spatial_mapper_for_utm_shapefile():
    """
    Show optimized SpatialPersonalityMapper for native UTM shapefile.
    """
    
    code_example = '''
# OPTIMIZED: When shapefile is already in UTM Zone 32N

class SpatialPersonalityMapper:
    def __init__(self, shapefile_path=None, target_crs='EPSG:32632'):
        # If shapefile is already in UTM Zone 32N
        self.native_utm_shapefile = True
        self.target_crs = target_crs
        
        if shapefile_path:
            self.load_utm_shapefile(shapefile_path)
    
    def load_utm_shapefile(self, shapefile_path):
        """Load shapefile that's already in UTM Zone 32N."""
        self.germany_gdf_utm = gpd.read_file(shapefile_path)
        
        if self.germany_gdf_utm.crs.to_string() == self.target_crs:
            print("✅ Shapefile already in UTM Zone 32N - optimal!")
            # No conversion needed!
        else:
            print(f"Converting from {self.germany_gdf_utm.crs} to {self.target_crs}")
            self.germany_gdf_utm = self.germany_gdf_utm.to_crs(self.target_crs)
        
        # Create WGS84 version only for visualization
        self.germany_gdf = self.germany_gdf_utm.to_crs('EPSG:4326')
    
    def create_germany_grid_utm_optimized(self, data_utm):
        """Optimized grid creation with native UTM shapefile."""
        
        # Use UTM shapefile bounds directly - no conversion needed!
        bounds = self.germany_gdf_utm.total_bounds
        x_min, y_min, x_max, y_max = bounds
        
        # Create grid (all in UTM meters)
        x_coords = np.arange(x_min, x_max, self.grid_resolution_meters)
        y_coords = np.arange(y_min, y_max, self.grid_resolution_meters)
        
        # ... rest of grid creation
        
        # Spatial filtering using native UTM coordinates - fastest possible!
        germany_union_utm = self.germany_gdf_utm.geometry.unary_union
        # No coordinate transformation needed here!
        
        return grid_points_utm, grid_shape

# Usage with UTM shapefile:
mapper = SpatialPersonalityMapper(
    shapefile_path="german_shapefile_utm32n/de_utm32n.shp",
    target_crs='EPSG:32632'
)
'''
    
    print("\nOPTIMIZED CODE FOR UTM SHAPEFILE:")
    print("="*40)
    print(code_example)

if __name__ == "__main__":
    # Convert the shapefile to UTM Zone 32N
    utm_shapefile = convert_german_shapefile_to_utm()
    
    # Compare coordinate systems
    if utm_shapefile:
        compare_coordinate_systems()
        
        # Show optimized code approach
        update_spatial_mapper_for_utm_shapefile()
        
        print(f"\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Use the converted UTM shapefile in your analysis:")
        print(f'   shapefile_path = "{utm_shapefile}"')
        print("2. Update your main() function:")
        print('   target_crs = "EPSG:32632"  # Already in target CRS!')
        print("3. Enjoy faster, more accurate spatial analysis! 🚀")
