import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os

warnings.filterwarnings('ignore')

class SpatialUncertaintyVisualizer:
    """
    Component for visualizing the spatial uncertainty of interpolated grid data.
    Uncertainty is represented by the inverse of the distance-weighted sum of
    contributions (weight_sum) from the grid computation. This measure reflects
    the data support/reliability at each grid point and is consistent across all traits.
    """

    def __init__(self, shapefile_path=None):
        """
        Initializes the visualizer with an optional shapefile for boundaries.
        
        Parameters:
        -----------
        shapefile_path : str, optional
            Path to boundary shapefile (e.g., German states) in a suitable projected CRS.
        """
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        self.germany_gdf_wgs84 = None # For spatial joins with lat/lon data (if needed for other plots)
        self.germany_gdf_proj = None  # For plotting boundaries to maintain correct aspect
        self.target_crs = None        # CRS of the projected shapefile

        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("SPATIAL UNCERTAINTY VISUALIZER (GRID-BASED)")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf_proj is not None else 'Not provided'}")
        print("Ready to visualize grid-based spatial uncertainty.")
        print("="*60)

    def load_shapefile(self, shapefile_path):
        """
        Loads the shapefile for boundaries. Loads it in its native CRS for plotting
        and creates a WGS84 version for spatial joins (though not used in this specific uncertainty plot).
        """
        try:
            print(f"Loading shapefile: {shapefile_path}")
            self.germany_gdf_proj = gpd.read_file(shapefile_path)
            self.target_crs = self.germany_gdf_proj.crs
            print(f"Loaded shapefile in native CRS: {self.target_crs}")
            
            # Create a WGS84 version, though not directly used for this grid-based uncertainty map
            if self.germany_gdf_proj.crs != 'EPSG:4326':
                self.germany_gdf_wgs84 = self.germany_gdf_proj.to_crs('EPSG:4326')
            else:
                self.germany_gdf_wgs84 = self.germany_gdf_proj.copy()
            
            print(f"Loaded {len(self.germany_gdf_proj)} regions")
                
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            self.germany_gdf_wgs84 = None
            self.germany_gdf_proj = None
            self.target_crs = None

    def load_grid_results_for_uncertainty(self, grid_results_file):
        """
        Loads the pre-computed grid results, which should contain grid coordinates
        and the 'weight_sum' for each trait.
        
        Parameters:
        -----------
        grid_results_file : str
            Path to CSV file with pre-computed grid results (e.g., from SpatialPersonalityGridComputer).
            
        Returns:
        --------
        pandas.DataFrame : Grid results with grid_lon, grid_lat, and trait_weight_sum columns.
        """
        print(f"\nLoading grid results for uncertainty visualization from: {grid_results_file}")
        
        if not os.path.exists(grid_results_file):
            raise FileNotFoundError(f"Grid results file not found: {grid_results_file}")
        
        try:
            grid_data = pd.read_csv(grid_results_file)
            
            # We only need one weight_sum column as they are all identical for a given grid point
            # We'll use Openness_weight_sum as the representative for general data support.
            required_cols = ['grid_lon', 'grid_lat', 'Openness_weight_sum'] 

            missing_cols = [col for col in required_cols if col not in grid_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in grid results: {missing_cols}")
            
            # Convert grid points to projected CRS for consistent plotting bounds
            if self.target_crs and grid_data['grid_lon'].dtype == float:
                print(f"Projecting grid points from EPSG:4326 to {self.target_crs} for plotting...")
                geometry = [Point(lon, lat) for lon, lat in zip(grid_data['grid_lon'], grid_data['grid_lat'])]
                grid_gdf_wgs84 = gpd.GeoDataFrame(grid_data, geometry=geometry, crs='EPSG:4326')
                grid_data_proj = grid_gdf_wgs84.to_crs(self.target_crs)
                grid_data['grid_x'] = grid_data_proj.geometry.x
                grid_data['grid_y'] = grid_data_proj.geometry.y
                print("Grid points projected successfully.")

            print(f"   Loaded {len(grid_data):,} grid points with weight sums.")
            return grid_data
            
        except Exception as e:
            print(f"Error loading grid results: {e}")
            raise

    def calculate_plot_bounds_from_grid(self, grid_data):
        """
        Calculates plot bounds and aspect ratio using projected grid data.
        Returns bounds in projected coordinates (x_min, x_max, y_min, y_max).
        """
        if 'grid_x' not in grid_data.columns or 'grid_y' not in grid_data.columns:
            raise ValueError("Grid data must contain 'grid_x' and 'grid_y' (projected coordinates).")

        x_min, x_max = grid_data['grid_x'].min(), grid_data['grid_x'].max()
        y_min, y_max = grid_data['grid_y'].min(), grid_data['grid_y'].max()
        
        width = x_max - x_min
        height = y_max - y_min
        
        if height == 0: height = 1 
        aspect_ratio = width / height
        
        print(f"   Projected grid extent: {width:.1f}m × {height:.1f}m (aspect ratio: {aspect_ratio:.2f})")
        
        return aspect_ratio, (x_min, x_max, y_min, y_max)

    def create_uncertainty_map(self, grid_data, save_path=None):
        """
        Creates a map visualizing the uncertainty (inverse of weight sum) of the interpolated grid data.
        
        Parameters:
        -----------
        grid_data : pandas.DataFrame
            DataFrame with grid data, including grid_x, grid_y, and a trait_weight_sum column.
            (e.g., 'Openness_weight_sum' used as the general weight sum).
        save_path : str, optional
            Path to save the generated map image.
        """
        if self.germany_gdf_proj is None or grid_data is None:
            print("Error: Missing projected shapefile or grid data. Cannot create uncertainty map.")
            return None
        
        # Use Openness_weight_sum as the representative weight sum, as all are identical
        weight_sum_col = 'Openness_weight_sum' 
        if weight_sum_col not in grid_data.columns:
            raise ValueError(f"Representative weight sum column '{weight_sum_col}' not found in grid data. "
                             f"Ensure grid results contain this column.")

        print(f"\nCreating Uncertainty Map: Data Support (Inverse Weight Sum)...")

        # Filter out grid points with no data (weight_sum is 0 or NaN) first, then calculate metric on this subset
        plot_data = grid_data[grid_data[weight_sum_col] > 0].copy()
        
        # Calculate uncertainty metric: inverse of weight_sum. Add a small epsilon to avoid division by zero.
        # Higher values of this metric mean higher uncertainty.
        plot_data['uncertainty_metric'] = 1 / (plot_data[weight_sum_col] + 1e-6) 
        
        if plot_data.empty:
            print(f"Warning: No valid grid points with weight sum > 0. Cannot create map.")
            return None

        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'Uncertainty Map: Data Support (Inverse Weight Sum)', 
                     fontsize=16, fontweight='bold')

        # Use a sequential colormap where higher values (higher uncertainty) are darker/more intense.
        cmap = 'YlOrRd' 
        
        # Define vmin and vmax based on the actual uncertainty metric range
        vmin = plot_data['uncertainty_metric'].min()
        vmax = plot_data['uncertainty_metric'].max()
        print(f"   Uncertainty metric range: Min={vmin:.4f}, Max={vmax:.4f}")


        # Scatter plot for grid points, colored by uncertainty metric
        scatter = ax.scatter(plot_data['grid_x'], plot_data['grid_y'], 
                             c=plot_data['uncertainty_metric'], 
                             cmap=cmap, 
                             s=5, # Smaller size for grid points
                             alpha=0.8,
                             vmin=vmin, vmax=vmax)

        # Add boundaries if available
        if self.germany_gdf_proj is not None:
            self.germany_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)

        # Set limits based on grid data bounds for consistency
        aspect_ratio, bounds = self.calculate_plot_bounds_from_grid(grid_data)
        x_min, x_max, y_min, y_max = bounds
        x_buffer = (x_max - x_min) * 0.02
        y_buffer = (y_max - y_min) * 0.02
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
        ax.set_aspect('equal', adjustable='box') 

        # Remove grid lines and frames
        ax.grid(False) 
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        ax.set_xlabel('') 
        ax.set_ylabel('') 
        for spine in ax.spines.values(): 
            spine.set_visible(False)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', 
                            label='Data Support (Inverse Weight Sum)', # Updated label
                            shrink=0.7, aspect=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"   Uncertainty map saved: {save_path}")
        
        plt.show()
        plt.close(fig)
        return fig


def run_uncertainty_analysis(grid_results_file, shapefile_path, output_prefix="uncertainty_map"):
    """
    Main function to run the spatial uncertainty analysis and generate the map.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to the CSV file containing pre-computed grid results (from SpatialPersonalityGridComputer).
    shapefile_path : str
        Path to the boundary shapefile (e.g., German states).
    output_prefix : str, optional
        Prefix for the output image file.
    """
    print("\n" + "="*70)
    print("STARTING SPATIAL UNCERTAINTY VISUALIZATION (GRID-BASED)")
    print("="*70)

    visualizer = SpatialUncertaintyVisualizer(shapefile_path=shapefile_path)
    
    if visualizer.target_crs is None:
        print("Error: Shapefile could not be loaded or has no CRS. Cannot proceed.")
        return {}

    # Load grid results which contain the weight_sum for uncertainty
    grid_data = visualizer.load_grid_results_for_uncertainty(grid_results_file)
    
    if grid_data is None:
        print("Error: Could not load grid data. Cannot create uncertainty map.")
        return {}

    # Output filename is now generic as the map is not trait-specific
    output_file_path = f"{output_prefix}_inverse_weight_sum_map.png"
    visualizer.create_uncertainty_map(grid_data, save_path=output_file_path)
    
    print(f"\nSpatial uncertainty visualization complete!")
    print(f"Output file: {output_file_path}")

    return {'uncertainty_map': output_file_path}

if __name__ == "__main__":
    # Configuration for running the script directly
    # This should be the output file from your spatial_grid_computer.py
    grid_results_file_path = "spatial_personality_grid_results.csv" 
    shapefile_path = "german_shapefile/de.shp" 
    
    # The uncertainty map is not trait-specific, as it reflects data support.
    # No need to define a specific trait here.

    output_prefix = "spatial_uncertainty_grid"
    
    output_files = run_uncertainty_analysis(
        grid_results_file=grid_results_file_path,
        shapefile_path=shapefile_path,
        output_prefix=output_prefix
    )
    
    print(f"\nAll spatial uncertainty visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
