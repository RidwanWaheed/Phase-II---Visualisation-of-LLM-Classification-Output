import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
import seaborn as sns # For density plots if desired

warnings.filterwarnings('ignore')

class DataCoverageVisualizer:
    """
    Component for visualizing the spatial coverage and density of original user data points.
    """

    def __init__(self, shapefile_path=None):
        """
        Initializes the visualizer with an optional shapefile for boundaries.
        
        Parameters:
        -----------
        shapefile_path : str, optional
            Path to boundary shapefile (e.g., German states) in a suitable projected CRS.
        """
        self.germany_gdf_proj = None  # For plotting boundaries
        self.target_crs = None        # CRS of the projected shapefile

        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("DATA COVERAGE/DENSITY VISUALIZER")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf_proj is not None else 'Not provided'}")
        print("Ready to visualize original user data distribution.")
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

    def load_original_data(self, data_file):
        """
        Loads the original user data, ensuring it has latitude and longitude.
        
        Parameters:
        -----------
        data_file : str
            Path to CSV file with original user data (expected lat/lon).
            
        Returns:
        --------
        pandas.DataFrame : Original data with 'latitude' and 'longitude' columns.
        """
        print(f"\nLoading original user data from: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        try:
            data = pd.read_csv(data_file)
            
            required_cols = ['latitude', 'longitude']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in data: {missing_cols}")
            
            # Drop rows with any missing latitude or longitude
            original_rows = len(data)
            data.dropna(subset=['latitude', 'longitude'], inplace=True)
            if len(data) < original_rows:
                print(f"   Removed {original_rows - len(data)} rows due to missing latitude/longitude.")

            if data.empty:
                raise ValueError("No valid user data found after dropping missing coordinates.")

            print(f"   Loaded {len(data):,} valid user records.")
            return data
            
        except Exception as e:
            print(f"Error loading original data: {e}")
            raise

    def plot_data_coverage(self, original_data, save_path=None, plot_type='scatter'):
        """
        Generates a map visualizing the spatial coverage and density of original user data.
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            DataFrame containing original user data with 'latitude' and 'longitude'.
        save_path : str, optional
            Path to save the plot image.
        plot_type : str, optional
            Type of plot to generate: 'scatter' for individual points, or 'kde' for kernel density estimate.
        """
        print(f"\nGenerating Data Coverage Map ({plot_type.upper()})...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle('Spatial Coverage and Density of Original User Data', 
                     fontsize=16, fontweight='bold')

        # Project original data points to the same CRS as the shapefile for consistent plotting
        geometry = [Point(lon, lat) for lon, lat in zip(original_data['longitude'], original_data['latitude'])]
        original_gdf = gpd.GeoDataFrame(original_data, geometry=geometry, crs='EPSG:4326')
        original_gdf_proj = original_gdf.to_crs(self.target_crs)

        # Add boundaries if available
        if self.germany_gdf_proj is not None:
            self.germany_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)

        # Plot based on type
        if plot_type == 'scatter':
            ax.scatter(original_gdf_proj.geometry.x, original_gdf_proj.geometry.y, 
                       color='blue', s=5, alpha=0.3, label='User Data Points')
        elif plot_type == 'kde':
            # Use seaborn's kdeplot for density estimation
            sns.kdeplot(x=original_gdf_proj.geometry.x, y=original_gdf_proj.geometry.y, 
                        cmap="Blues", fill=True, alpha=0.7, ax=ax,
                        cbar=True, cbar_kws={'label': 'Density of Users'})
        else:
            raise ValueError("Invalid plot_type. Choose 'scatter' or 'kde'.")

        # Set limits based on shapefile bounds for consistency
        if self.germany_gdf_proj is not None:
            x_min, y_min, x_max, y_max = self.germany_gdf_proj.total_bounds
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
        
        if plot_type == 'scatter':
            ax.legend(loc='upper right', frameon=True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"   Data coverage map saved: {save_path}")
        plt.show()
        plt.close(fig)


def run_data_coverage_analysis(data_file, shapefile_path, output_prefix="data_coverage", plot_type='scatter'):
    """
    Main function to run the data coverage analysis and generate the map.
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing original user data (with lat/lon).
    shapefile_path : str
        Path to the boundary shapefile (e.g., German states).
    output_prefix : str, optional
        Prefix for the output image file.
    plot_type : str, optional
        Type of plot to generate: 'scatter' for individual points, or 'kde' for kernel density estimate.
    """
    print("\n" + "="*70)
    print("STARTING DATA COVERAGE/DENSITY VISUALIZATION")
    print("="*70)

    visualizer = DataCoverageVisualizer(shapefile_path=shapefile_path)
    
    if visualizer.germany_gdf_proj is None:
        print("Error: Shapefile could not be loaded. Cannot proceed with visualization.")
        return {}

    # Load original user data
    original_data = visualizer.load_original_data(data_file)
    
    if original_data is None or original_data.empty:
        print("Error: Could not load valid original data. Cannot proceed with visualization.")
        return {}

    output_file_path = f"{output_prefix}_{plot_type}_map.png"
    visualizer.plot_data_coverage(original_data, save_path=output_file_path, plot_type=plot_type)
    
    print(f"\nData coverage visualization complete!")
    print(f"Output file: {output_file_path}")

    return {'data_coverage_map': output_file_path}

if __name__ == "__main__":
    # Configuration for running the script directly
    original_data_file_path = "final_users_for_spatial_visualization.csv" 
    shapefile_path = "german_shapefile/de.shp"
    
    output_prefix = "data_coverage"
    
    # Choose plot_type: 'scatter' for individual points, 'kde' for density heatmap
    plot_type_to_generate = 'kde' # or 'scatter'

    output_files = run_data_coverage_analysis(
        data_file=original_data_file_path,
        shapefile_path=shapefile_path,
        output_prefix=output_prefix,
        plot_type=plot_type_to_generate
    )
    
    print(f"\nAll data coverage visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
