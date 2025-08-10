import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import time
from tqdm import tqdm
import warnings
import os
from pyproj import CRS, Transformer

warnings.filterwarnings('ignore')

class SpatialPersonalityGridComputer:
    """
    spatial grid computer for personality traits using Ebert et al.'s methodology.
    
    Computes distance-weighted personality scores on a spatial grid using:
    1. UTM coordinate transformations for accurate distance calculations
    2. Grid creation in meter-based coordinates
    3. Distance-weighted personality score calculations
    4. Data export for visualization
    """
    
    def __init__(self, r_km=45, s_slope=7, grid_resolution_km=5, 
                 target_crs='EPSG:32632', shapefile_path=None):
        """
        Initialize the spatial grid computer.
        
        Parameters:
        -----------
        r_km : float
            Distance radius in kilometers (Ebert's parameter)
        s_slope : float  
            Distance decay slope (Ebert's parameter)
        grid_resolution_km : float
            Grid resolution in kilometers
        target_crs : str
            Target coordinate reference system (default: UTM Zone 32N for Germany)
        shapefile_path : str, optional
            Path to boundary shapefile for grid clipping
        """
        # Store Ebert's distance decay parameters
        self.r_km = r_km
        self.s_slope = s_slope
        self.r_miles = r_km / 1.60934
        self.r_meters = self.r_km * 1000
        
        # Store grid parameters
        self.grid_resolution_km = grid_resolution_km
        self.grid_resolution_meters = grid_resolution_km * 1000
        
        # Set up coordinate transformations
        self.source_crs = 'EPSG:4326'  # WGS84 (input data)
        self.target_crs = target_crs   # UTM Zone 32N for Germany
        self.transformer_to_utm = Transformer.from_crs(self.source_crs, self.target_crs, always_xy=True)
        self.transformer_to_wgs84 = Transformer.from_crs(self.target_crs, self.source_crs, always_xy=True)
        
        # Define personality traits to process
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        
        # Load boundary shapefile if provided
        self.boundary_gdf_utm = None
        if shapefile_path:
            self.load_boundary_shapefile(shapefile_path)
        
        print("="*60)
        print("SPATIAL PERSONALITY GRID COMPUTER - UTM ENHANCED")
        print("="*60)
        print(f"Coordinate Systems:")
        print(f"  Input CRS: {self.source_crs} (WGS84)")
        print(f"  Analysis CRS: {self.target_crs}")
        print(f"Distance Parameters:")
        print(f"  Radius: {r_km} km ({self.r_miles:.1f} miles, {self.r_meters:.0f} m)")
        print(f"  Decay slope: {s_slope}")
        print(f"Grid Parameters:")
        print(f"  Resolution: {grid_resolution_km} km ({self.grid_resolution_meters:.0f} m)")
        print(f"Boundary Clipping: {'Enabled' if self.boundary_gdf_utm is not None else 'Disabled'}")
        print("="*60)
    
    def load_boundary_shapefile(self, shapefile_path):
        """Load boundary shapefile and convert to UTM for clipping operations."""
        try:
            print(f"Loading boundary shapefile: {shapefile_path}")
            boundary_gdf = gpd.read_file(shapefile_path)
            
            # Convert to UTM for accurate spatial operations
            self.boundary_gdf_utm = boundary_gdf.to_crs(self.target_crs)
            
            print(f"Boundary shapefile loaded:")
            print(f"   {len(self.boundary_gdf_utm)} regions")
            print(f"   Converted to {self.target_crs}")
            print(f"   UTM bounds: {self.boundary_gdf_utm.total_bounds}")
            
        except Exception as e:
            print(f"Error loading boundary shapefile: {e}")
            print("   Proceeding without boundary clipping")
            self.boundary_gdf_utm = None
    
    def transform_coordinates_to_utm(self, data):
        """Transform lat/lon coordinates to UTM for accurate distance calculations."""
        print("\nSTEP 1: Coordinate Transformation")
        print("-" * 40)
        print("Transforming coordinates to UTM Zone 32N...")
        
        data_utm = data.copy()
        
        # Validate required columns exist
        required_cols = ['latitude', 'longitude'] + self.personality_traits
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Remove rows with missing data
        initial_count = len(data_utm)
        data_utm = data_utm.dropna(subset=required_cols)
        final_count = len(data_utm)
        
        if final_count < initial_count:
            print(f"   Removed {initial_count - final_count} rows with missing data")
        
        # Transform coordinates from WGS84 to UTM
        utm_x, utm_y = self.transformer_to_utm.transform(
            data_utm['longitude'].values, 
            data_utm['latitude'].values
        )
        
        data_utm['utm_x'] = utm_x
        data_utm['utm_y'] = utm_y
        
        print(f"Transformed {len(data_utm):,} points to UTM coordinates")
        print(f"   UTM X range: {utm_x.min():.0f} to {utm_x.max():.0f} meters")
        print(f"   UTM Y range: {utm_y.min():.0f} to {utm_y.max():.0f} meters")
        print(f"   Study area extent: ~{(utm_x.max()-utm_x.min())/1000:.0f} × {(utm_y.max()-utm_y.min())/1000:.0f} km")
        
        return data_utm
    
    def create_utm_grid(self, data_utm):
        """Create spatial grid in UTM coordinates with optional boundary clipping."""
        print("\nSTEP 2: UTM Grid Creation")
        print("-" * 40)
        
        # Determine grid bounds from boundary shapefile or data extent
        if self.boundary_gdf_utm is not None:
            bounds = self.boundary_gdf_utm.total_bounds  # [minx, miny, maxx, maxy]
            x_min, y_min, x_max, y_max = bounds
            print(f"Using boundary shapefile bounds:")
        else:
            buffer_meters = 10000  # 10km buffer around data
            x_min = data_utm['utm_x'].min() - buffer_meters
            x_max = data_utm['utm_x'].max() + buffer_meters
            y_min = data_utm['utm_y'].min() - buffer_meters
            y_max = data_utm['utm_y'].max() + buffer_meters
            print(f"Using data extent with {buffer_meters/1000:.0f}km buffer:")
        
        print(f"   X (Easting): {x_min:.0f} to {x_max:.0f} meters")
        print(f"   Y (Northing): {y_min:.0f} to {y_max:.0f} meters")
        print(f"   Grid extent: {(x_max-x_min)/1000:.1f} × {(y_max-y_min)/1000:.1f} km")
        
        # Create regular grid coordinates
        x_coords = np.arange(x_min, x_max + self.grid_resolution_meters, self.grid_resolution_meters)
        y_coords = np.arange(y_min, y_max + self.grid_resolution_meters, self.grid_resolution_meters)
        
        print(f"   Grid dimensions: {len(x_coords)} × {len(y_coords)} = {len(x_coords) * len(y_coords):,} cells")
        
        # Create meshgrid and convert to point dataframe
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        grid_points_utm = pd.DataFrame({
            'utm_x': x_grid.flatten(),
            'utm_y': y_grid.flatten(),
            'grid_idx': range(len(x_grid.flatten()))
        })
        
        # Transform grid points back to WGS84 for visualization
        print("   Converting grid coordinates to WGS84 for visualization...")
        grid_lon, grid_lat = self.transformer_to_wgs84.transform(
            grid_points_utm['utm_x'].values,
            grid_points_utm['utm_y'].values
        )
        
        grid_points_utm['grid_lon'] = grid_lon
        grid_points_utm['grid_lat'] = grid_lat
        
        # Apply boundary clipping if shapefile is available
        if self.boundary_gdf_utm is not None:
            print("   Applying boundary clipping...")
            grid_points_utm = self._clip_grid_to_boundaries(grid_points_utm)
        
        print(f"Final grid: {len(grid_points_utm):,} points")
        
        return grid_points_utm
    
    def _clip_grid_to_boundaries(self, grid_points_utm):
        """Clip grid points to boundary shapefile using UTM coordinates."""
        # Create point geometries in UTM
        geometry_utm = [Point(x, y) for x, y in zip(grid_points_utm['utm_x'], grid_points_utm['utm_y'])]
        grid_gdf_utm = gpd.GeoDataFrame(grid_points_utm, geometry=geometry_utm, crs=self.target_crs)
        
        # Filter to boundaries using UTM coordinates
        boundary_union_utm = self.boundary_gdf_utm.geometry.unary_union
        
        print("     Performing spatial filtering...")
        within_mask = grid_gdf_utm.geometry.within(boundary_union_utm)
        
        filtered_grid = grid_gdf_utm[within_mask].drop('geometry', axis=1).reset_index(drop=True)
        filtered_grid['grid_idx'] = range(len(filtered_grid))
        
        removed_count = len(grid_points_utm) - len(filtered_grid)
        print(f"     Removed {removed_count:,} points outside boundaries")
        
        return filtered_grid
    
    def euclidean_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between points in UTM coordinates (meters)."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def distance_decay_weight(self, distances_meters):
        """Calculate distance decay weights using Ebert's log-logistic function."""
        # Convert distances to kilometers for Ebert's formula
        distances_km = distances_meters / 1000
        
        # Apply Ebert's log-logistic distance decay: f(d) = 1 / (1 + (d/r)^s)
        weights = 1 / (1 + (distances_km / self.r_km) ** self.s_slope)
        return weights
    
    def calculate_weighted_scores(self, data_utm, grid_points_utm):
        """Calculate distance-weighted personality scores using UTM coordinates."""
        print("\nSTEP 3: Distance-Weighted Score Calculation")
        print("-" * 40)
        print(f"Processing {len(grid_points_utm):,} grid points with {len(data_utm):,} user data points")
        print(f"Using UTM-based Euclidean distance calculation")
        print(f"Distance parameters: r={self.r_km}km, s={self.s_slope}")
        print(f"  Participants at {self.r_km}km receive weight ≈ 0.5")
        print(f"  Participants at {self.r_km * 1.67:.0f}km receive weight ≈ 0")
        
        # Initialize results dataframe with grid coordinates
        results = grid_points_utm.copy()
        for trait in self.personality_traits:
            results[trait] = np.nan
            results[f'{trait}_weight_sum'] = 0.0
        
        # Process in batches for memory management
        batch_size = 100
        n_batches = (len(grid_points_utm) + batch_size - 1) // batch_size
        
        print(f"Processing in {n_batches:,} batches of {batch_size} grid points each...")
        
        start_time = time.time()
        
        for batch_idx in tqdm(range(n_batches), desc="Computing weighted scores", unit="batch"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(grid_points_utm))
            batch_grid = grid_points_utm.iloc[start_idx:end_idx]
            
            # Process each grid point in this batch
            for _, grid_point in batch_grid.iterrows():
                grid_idx = grid_point['grid_idx']
                
                # Calculate Euclidean distances to all users
                distances = self.euclidean_distance(
                    grid_point['utm_x'], grid_point['utm_y'],
                    data_utm['utm_x'].values, data_utm['utm_y'].values
                )
                
                # Calculate distance-decay weights
                weights = self.distance_decay_weight(distances)
                
                # Calculate weighted scores for each personality trait
                total_weight = np.sum(weights)
                
                if total_weight > 0:
                    for trait in self.personality_traits:
                        trait_values = data_utm[trait].values
                        weighted_score = np.sum(weights * trait_values) / total_weight
                        results.loc[results['grid_idx'] == grid_idx, trait] = weighted_score
                        results.loc[results['grid_idx'] == grid_idx, f'{trait}_weight_sum'] = total_weight
            
            # Report progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                remaining_batches = n_batches - (batch_idx + 1)
                eta_minutes = (elapsed / (batch_idx + 1)) * remaining_batches if batch_idx > 0 else 0
                
                print(f"   Progress {batch_idx + 1:,}/{n_batches:,} | "
                      f"Elapsed: {elapsed:.1f}min | ETA: {eta_minutes:.1f}min")
        
        total_time = (time.time() - start_time) / 60
        print(f"Weighted score calculation complete in {total_time:.1f} minutes")
        
        return results
    
    def standardize_scores(self, results):
        """Z-score standardize personality traits across all grid points."""
        print("\nSTEP 4: Score Standardization")
        print("-" * 40)
        
        standardized_results = results.copy()
        
        for trait in self.personality_traits:
            # Only standardize where we have valid scores
            valid_mask = results[f'{trait}_weight_sum'] > 0
            valid_scores = results.loc[valid_mask, trait]
            
            if len(valid_scores) > 1:
                mean_score = valid_scores.mean()
                std_score = valid_scores.std()
                
                if std_score > 0:
                    standardized_results.loc[valid_mask, f'{trait}_z'] = (
                        (results.loc[valid_mask, trait] - mean_score) / std_score
                    )
                    print(f"   {trait}: μ={mean_score:.3f}, σ={std_score:.3f} (n={len(valid_scores):,})")
                else:
                    print(f"   {trait}: No variation (σ=0), setting z-scores to 0")
                    standardized_results.loc[valid_mask, f'{trait}_z'] = 0.0
            else:
                print(f"   {trait}: Insufficient valid data for standardization")
                standardized_results[f'{trait}_z'] = np.nan
        
        print("Score standardization complete")
        
        return standardized_results
    
    def save_results(self, results, output_file):
        """Save computed grid results to CSV with metadata."""
        print(f"\nSTEP 5: Saving Results")
        print("-" * 40)
        
        # Add metadata columns
        results_with_meta = results.copy()
        results_with_meta['computation_date'] = pd.Timestamp.now().isoformat()
        results_with_meta['r_km'] = self.r_km
        results_with_meta['s_slope'] = self.s_slope
        results_with_meta['grid_resolution_km'] = self.grid_resolution_km
        results_with_meta['crs_source'] = self.source_crs
        results_with_meta['crs_target'] = self.target_crs
        
        # Save to CSV
        results_with_meta.to_csv(output_file, index=False)
        
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        print(f"Results saved to: {output_file}")
        print(f"   File size: {file_size_mb:.1f} MB")
        print(f"   Grid points: {len(results):,}")
        print(f"   Columns: {len(results_with_meta.columns)}")
        
        return output_file
    
    def generate_computation_summary(self, data_utm, results):
        """Generate comprehensive summary of the computation."""
        print("\n" + "="*70)
        print("COMPUTATION SUMMARY")
        print("="*70)
        
        print(f"\nInput Data:")
        print(f"   Total users: {len(data_utm):,}")
        print(f"   Geographic extent (WGS84):")
        print(f"     - Latitude: {data_utm['latitude'].min():.6f}° to {data_utm['latitude'].max():.6f}°")
        print(f"     - Longitude: {data_utm['longitude'].min():.6f}° to {data_utm['longitude'].max():.6f}°")
        print(f"   UTM extent (Zone 32N):")
        print(f"     - Easting: {data_utm['utm_x'].min():.0f} to {data_utm['utm_x'].max():.0f} m")
        print(f"     - Northing: {data_utm['utm_y'].min():.0f} to {data_utm['utm_y'].max():.0f} m")
        
        print(f"\nGrid Configuration:")
        print(f"   Resolution: {self.grid_resolution_km} km ({self.grid_resolution_meters:.0f} m)")
        print(f"   Total grid points: {len(results):,}")
        print(f"   Coordinate system: {self.target_crs}")
        print(f"   Boundary clipping: {'Yes' if self.boundary_gdf_utm is not None else 'No'}")
        
        print(f"\nDistance Parameters (Ebert et al. methodology):")
        print(f"   Radius: {self.r_km} km ({self.r_miles:.1f} miles, {self.r_meters:.0f} m)")
        print(f"   Decay slope: {self.s_slope}")
        print(f"   Weight at {self.r_km}km: 0.5 (exact match with Ebert et al.)")
        print(f"   Weight at {self.r_km*1.67:.0f}km: ~0.0")
        print(f"   Distance calculation: Euclidean (UTM) - Superior accuracy")

        

def compute_spatial_personality_grid(input_file, output_file, shapefile_path=None, 
                                   r_km=45, s_slope=7, grid_resolution_km=5):
    """
    Compute spatial personality grid with UTM enhancement.
    
    Parameters:
    -----------
    input_file : str
        Path to CSV file with user data (latitude, longitude, personality traits)
    output_file : str
        Path to save computed grid results
    shapefile_path : str, optional
        Path to boundary shapefile for grid clipping
    r_km, s_slope, grid_resolution_km : float
        Ebert's methodology parameters
        
    Returns:
    --------
    str : Path to saved results file
    """
    
    print("STARTING UTM-ENHANCED SPATIAL PERSONALITY GRID COMPUTATION")
    print("="*80)
    
    # Check for required dependencies
    try:
        import pyproj
        print("pyproj library available for coordinate transformations")
    except ImportError:
        print("pyproj library required for UTM transformations")
        print("   Install with: pip install pyproj")
        return None
    
    # Load input data
    print(f"\nLoading input data: {input_file}")
    try:
        data = pd.read_csv(input_file)
        print(f"   Loaded {len(data):,} records")
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
        return None
    
    # Initialize grid computer
    computer = SpatialPersonalityGridComputer(
        r_km=r_km,
        s_slope=s_slope, 
        grid_resolution_km=grid_resolution_km,
        target_crs='EPSG:32632',  # UTM Zone 32N for Germany
        shapefile_path=shapefile_path
    )
    
    # Execute computation pipeline
    start_time = time.time()
    
    data_utm = computer.transform_coordinates_to_utm(data)
    grid_points_utm = computer.create_utm_grid(data_utm)
    weighted_results = computer.calculate_weighted_scores(data_utm, grid_points_utm)
    standardized_results = computer.standardize_scores(weighted_results)
    saved_file = computer.save_results(standardized_results, output_file)
    
    # Generate summary
    computer.generate_computation_summary(data_utm, standardized_results)
    
    total_time = (time.time() - start_time) / 60
    print(f"\nCOMPUTATION COMPLETE!")
    print(f"   Total time: {total_time:.1f} minutes")
    print(f"   Results saved to: {saved_file}")
    print(f"   Ready for visualization with separate mapping component")
    print("="*80)
    
    return saved_file

if __name__ == "__main__":
    # Configuration
    input_file = "final_users_for_spatial_visualization.csv" 
    output_file = "spatial_personality_grid_results.csv" 
    shapefile_path = "german_shapefile/de.shp"
    
    # Ebert's parameters
    r_km = 45           # Distance radius in kilometers
    s_slope = 7         # Distance decay slope
    grid_resolution = 5 # Grid resolution in km
    
    # Run computation
    result_file = compute_spatial_personality_grid(
        input_file=input_file,
        output_file=output_file,
        shapefile_path=shapefile_path,
        r_km=r_km,
        s_slope=s_slope,
        grid_resolution_km=grid_resolution
    )
    
    if result_file:
        print(f"\nSUCCESS: Grid computation complete!")
        print(f"Results file: {result_file}")
        print(f"Next: Use this file with your visualization component")
    else:
        print(f"\nFAILED: Grid computation unsuccessful")