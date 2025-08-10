import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
import esda # For Moran's I calculation
import libpysal # For spatial weights
import splot.esda # For Moran scatter plot visualization

warnings.filterwarnings('ignore')

class SpatialAutocorrelationAnalyzer:
    """
    Component for performing Spatial Autocorrelation analysis (Moran's I)
    on interpolated personality trait data.
    """

    def __init__(self, shapefile_path=None):
        """
        Initializes the analyzer with an optional shapefile for boundaries.
        
        Parameters:
        -----------
        shapefile_path : str, optional
            Path to boundary shapefile (e.g., German states) in a suitable projected CRS.
        """
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        self.germany_gdf_proj = None  # For plotting boundaries and creating spatial weights
        self.target_crs = None        # CRS of the projected shapefile
        self.spatial_weights = None   # Spatial weights matrix (e.g., W)

        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("SPATIAL AUTOCORRELATION (MORAN'S I) ANALYZER")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf_proj is not None else 'Not provided'}")
        print("Ready to compute Moran's I and generate plots.")
        print("="*60)

    def load_shapefile(self, shapefile_path):
        """
        Loads the shapefile for boundaries and sets the target CRS.
        """
        try:
            print(f"Loading shapefile: {shapefile_path}")
            self.germany_gdf_proj = gpd.read_file(shapefile_path)
            self.target_crs = self.germany_gdf_proj.crs
            print(f"Loaded shapefile in native CRS: {self.target_crs}")
            print(f"Loaded {len(self.germany_gdf_proj)} regions")
                
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            self.germany_gdf_proj = None
            self.target_crs = None

    def load_grid_results(self, grid_results_file):
        """
        Loads the pre-computed grid results, which should contain grid coordinates
        and Z-scores for each trait.
        
        Parameters:
        -----------
        grid_results_file : str
            Path to CSV file with pre-computed grid results (e.g., from SpatialPersonalityGridComputer).
            
        Returns:
        --------
        pandas.DataFrame : Grid results with grid_x, grid_y, and trait_z columns.
        """
        print(f"\nLoading grid results for Moran's I analysis from: {grid_results_file}")
        
        if not os.path.exists(grid_results_file):
            raise FileNotFoundError(f"Grid results file not found: {grid_results_file}")
        
        try:
            grid_data = pd.read_csv(grid_results_file)
            
            # Ensure Z-score columns exist
            z_score_cols = [f'{trait}_z' for trait in self.personality_traits]
            required_cols = ['grid_lon', 'grid_lat'] + z_score_cols

            missing_cols = [col for col in required_cols if col not in grid_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in grid results: {missing_cols}")
            
            # Project grid points if not already projected, for consistency with shapefile
            if self.target_crs and grid_data['grid_lon'].dtype == float:
                print(f"Projecting grid points from EPSG:4326 to {self.target_crs} for analysis...")
                geometry = [Point(lon, lat) for lon, lat in zip(grid_data['grid_lon'], grid_data['grid_lat'])]
                grid_gdf_wgs84 = gpd.GeoDataFrame(grid_data, geometry=geometry, crs='EPSG:4326')
                grid_data_proj = grid_gdf_wgs84.to_crs(self.target_crs)
                grid_data['grid_x'] = grid_data_proj.geometry.x
                grid_data['grid_y'] = grid_data_proj.geometry.y
                print("Grid points projected successfully.")

            print(f"   Loaded {len(grid_data):,} grid points with Z-scores.")
            return grid_data
            
        except Exception as e:
            print(f"Error loading grid results: {e}")
            raise

    def create_spatial_weights(self, grid_data, method='knn', k=8):
        """
        Creates a spatial weights matrix (W) based on the grid points.
        
        Parameters:
        -----------
        grid_data : pandas.DataFrame
            DataFrame containing grid points with 'grid_x' and 'grid_y' columns.
        method : str
            Method for creating spatial weights ('knn' for K-nearest neighbors or 'contiguity' for polygon contiguity).
            'contiguity' requires a GeoDataFrame of polygons.
        k : int, optional
            Number of nearest neighbors for 'knn' method.
            
        Returns:
        --------
        libpysal.weights.weights.W : The spatial weights matrix.
        """
        print(f"\nCreating spatial weights matrix using '{method}' method...")
        
        coords = grid_data[['grid_x', 'grid_y']].values
        
        if method == 'knn':
            if k >= len(coords):
                raise ValueError(f"K ({k}) must be less than the number of grid points ({len(coords)}).")
            self.spatial_weights = libpysal.weights.KNN.from_array(coords, k=k)
            print(f"   Created K-Nearest Neighbors (k={k}) weights.")
        elif method == 'contiguity':
            if self.germany_gdf_proj is None:
                raise ValueError("Contiguity weights require a loaded shapefile (polygons).")
            # For grid data, contiguity is not directly applicable unless we convert grid points to polygons.
            # If the user intends contiguity between states, then the W should be built on states.
            # For grid points, KNN or distance-based weights are more common.
            print("   Warning: Contiguity weights are typically for polygons. Using KNN as fallback for grid points.")
            self.spatial_weights = libpysal.weights.KNN.from_array(coords, k=k) # Fallback to KNN
        else:
            raise ValueError(f"Unsupported spatial weights method: {method}. Choose 'knn'.")
            
        self.spatial_weights.transform = 'R' # Row-standardization
        print(f"   Weights matrix transformed to row-standardized.")
        
        return self.spatial_weights

    def calculate_morans_i(self, grid_data, trait):
        """
        Calculates Moran's I for a given personality trait's Z-scores.
        
        Parameters:
        -----------
        grid_data : pandas.DataFrame
            DataFrame containing grid points with the trait's Z-score column.
        trait : str
            Name of the personality trait (e.g., 'Openness').
            
        Returns:
        --------
        esda.moran.Moran : Moran's I object.
        """
        if self.spatial_weights is None:
            raise ValueError("Spatial weights matrix not created. Call create_spatial_weights first.")
        
        z_col = f'{trait}_z'
        if z_col not in grid_data.columns:
            raise ValueError(f"Z-score column '{z_col}' not found for trait '{trait}'.")
        
        # Filter out NaN values before calculating Moran's I
        valid_data = grid_data.dropna(subset=[z_col])
        if valid_data.empty:
            print(f"Warning: No valid data for {trait} after dropping NaNs. Cannot compute Moran's I.")
            return None

        coords_valid = valid_data[['grid_x', 'grid_y']].values
        if len(coords_valid) < 2: # Moran's I needs at least 2 points
            print(f"Warning: Insufficient valid data points ({len(coords_valid)}) for {trait} to compute Moran's I.")
            return None
        
        # Rebuild weights for only the valid data points
        try:
            temp_weights = libpysal.weights.KNN.from_array(coords_valid, k=min(8, len(coords_valid)-1))
            temp_weights.transform = 'R'
        except Exception as e:
            print(f"Error rebuilding spatial weights for valid data subset: {e}")
            print("This might happen if there are too few valid points or all points are identical.")
            return None

        print(f"\nCalculating Moran's I for {trait}...")
        moran = esda.moran.Moran(valid_data[z_col], temp_weights)
        
        print(f"   Moran's I for {trait}: {moran.I:.4f}")
        print(f"   P-value (randomization): {moran.p_sim:.4f}")
        
        return moran

    def plot_morans_scatter(self, moran_obj, trait, save_path=None):
        """
        Generates a Moran scatter plot for a given trait.
        
        Parameters:
        -----------
        moran_obj : esda.moran.Moran
            The Moran's I object returned by calculate_morans_i.
        trait : str
            Name of the personality trait.
        save_path : str, optional
            Path to save the plot image.
        """
        if moran_obj is None:
            print("Moran's I object is None. Cannot generate scatter plot.")
            return

        print(f"\nGenerating Moran Scatter Plot for {trait}...")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot the scatter plot using splot.esda.moran_scatterplot (corrected function name)
        splot.esda.moran_scatterplot(moran_obj, ax=ax, zstandard=True) # Corrected keyword argument
        
        ax.set_title(f'Moran Scatter Plot: {trait}', fontsize=16, fontweight='bold')
        ax.set_xlabel(f'{trait} (Z-score)', fontsize=12)
        ax.set_ylabel(f'Spatially Lagged {trait} (Z-score)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add Moran's I value to the plot
        ax.text(0.05, 0.95, f"Moran's I = {moran_obj.I:.4f}\nP-value = {moran_obj.p_sim:.4f}", 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', lw=0.5, alpha=0.8))

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Moran scatter plot saved: {save_path}")
        plt.show()
        plt.close(fig)


def run_morans_i_analysis(grid_results_file, shapefile_path, trait_to_analyze, output_prefix="morans_i"):
    """
    Main function to run the Moran's I spatial autocorrelation analysis.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to the CSV file containing pre-computed grid results (with Z-scores).
    shapefile_path : str
        Path to the boundary shapefile (e.g., German states), used for spatial context.
    trait_to_analyze : str
        Name of the personality trait for which to calculate Moran's I (e.g., 'Openness').
    output_prefix : str, optional
        Prefix for the output image file.
    """
    print("\n" + "="*70)
    print(f"STARTING MORAN'S I ANALYSIS FOR {trait_to_analyze.upper()}")
    print("="*70)

    # Initialize analyzer
    analyzer = SpatialAutocorrelationAnalyzer(shapefile_path=shapefile_path)
    
    if analyzer.germany_gdf_proj is None:
        print("Error: Shapefile could not be loaded. Cannot proceed with spatial analysis.")
        return {}

    # Load grid data
    grid_data = analyzer.load_grid_results(grid_results_file)
    
    if grid_data is None or grid_data.empty:
        print("Error: Could not load valid grid data. Cannot proceed with analysis.")
        return {}

    # Create spatial weights matrix (using KNN for grid points)
    # Adjust k based on density of your grid points. A common choice is 8 for a grid.
    analyzer.create_spatial_weights(grid_data, method='knn', k=8)
    
    # Calculate Moran's I
    moran_result = analyzer.calculate_morans_i(grid_data, trait_to_analyze)
    
    output_files = {}
    if moran_result:
        # Plot Moran Scatter Plot
        scatter_plot_path = f"{output_prefix}_{trait_to_analyze.lower()}_moran_scatter.png"
        analyzer.plot_morans_scatter(moran_result, trait_to_analyze, save_path=scatter_plot_path)
        output_files['moran_scatter_plot'] = scatter_plot_path
    else:
        print(f"Moran's I calculation failed for {trait_to_analyze}. No plot generated.")

    print(f"\nMoran's I analysis complete for {trait_to_analyze}!")
    print(f"Output files: {list(output_files.values())}")

    return output_files

if __name__ == "__main__":
    # Configuration for running the script directly
    # This should be the output file from your spatial_grid_computer.py
    grid_results_file_path = "spatial_personality_grid_results.csv" 
    shapefile_path = "german_shapefile/de.shp"
    
    # Define the personality trait for which you want to calculate Moran's I
    # You can change this to 'Conscientiousness', 'Extraversion', 'Agreeableness', or 'Neuroticism'
    trait_for_analysis = 'Openness' 

    output_prefix = "morans_i_analysis"
    
    output_files = run_morans_i_analysis(
        grid_results_file=grid_results_file_path,
        shapefile_path=shapefile_path,
        trait_to_analyze=trait_for_analysis,
        output_prefix=output_prefix
    )
    
    print(f"\nAll Moran's I visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
