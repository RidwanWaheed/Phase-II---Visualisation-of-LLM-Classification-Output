import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
import esda # For Getis-Ord Gi*
import libpysal # For spatial weights
import splot.esda # For Getis-Ord Gi* visualization

warnings.filterwarnings('ignore')

class HotSpotAnalyzer:
    """
    Component for performing Hot Spot Analysis (Getis-Ord Gi*)
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
        print("HOT SPOT ANALYSIS (GETIS-ORD GI*) ANALYZER")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf_proj is not None else 'Not provided'}")
        print("Ready to compute Getis-Ord Gi* and generate hot/cold spot maps.")
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
        print(f"\nLoading grid results for Hot Spot analysis from: {grid_results_file}")
        
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
            Method for creating spatial weights ('knn' for K-nearest neighbors).
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
        else:
            raise ValueError(f"Unsupported spatial weights method: {method}. Choose 'knn'.")
            
        self.spatial_weights.transform = 'R' # Row-standardization
        print(f"   Weights matrix transformed to row-standardized.")
        
        return self.spatial_weights

    def calculate_getis_ord_gi_star(self, grid_data, trait):
        """
        Calculates Getis-Ord Gi* for a given personality trait's Z-scores.
        
        Parameters:
        -----------
        grid_data : pandas.DataFrame
            DataFrame containing grid points with the trait's Z-score column.
        trait : str
            Name of the personality trait (e.g., 'Openness').
            
        Returns:
        --------
        pandas.DataFrame : Original grid_data with 'Gi_star', 'Gi_star_p_value', and 'hot_cold_spot' columns added.
        """
        if self.spatial_weights is None:
            raise ValueError("Spatial weights matrix not created. Call create_spatial_weights first.")
        
        z_col = f'{trait}_z'
        if z_col not in grid_data.columns:
            raise ValueError(f"Z-score column '{z_col}' not found for trait '{trait}'.")
        
        # Filter out NaN values before calculating Gi*
        valid_data = grid_data.dropna(subset=[z_col]).copy() # Use .copy() to avoid SettingWithCopyWarning
        if valid_data.empty:
            print(f"Warning: No valid data for {trait} after dropping NaNs. Cannot compute Gi*.")
            return None

        # Rebuild weights for only the valid data points to ensure alignment
        coords_valid = valid_data[['grid_x', 'grid_y']].values
        if len(coords_valid) < 2: 
            print(f"Warning: Insufficient valid data points ({len(coords_valid)}) for {trait} to compute Gi*.")
            return None
        
        try:
            # Use the same k as the main weights, but ensure it's less than valid data points
            temp_weights = libpysal.weights.KNN.from_array(coords_valid, k=min(self.spatial_weights.k, len(coords_valid)-1))
            temp_weights.transform = 'R'
        except Exception as e:
            print(f"Error rebuilding spatial weights for valid data subset for Gi*: {e}")
            print("This might happen if there are too few valid points or all points are identical.")
            return None

        print(f"\nCalculating Getis-Ord Gi* for {trait}...")
        gi_star = esda.G_Local(valid_data[z_col], temp_weights) 
        
        # Add Gi* statistics to the valid_data DataFrame
        valid_data['Gi_star'] = gi_star.Gs
        valid_data['Gi_star_p_value'] = gi_star.p_sim # p-value from permutation test
        
        # Classify hot/cold spots based on significance (e.g., p < 0.05)
        # 1: Hot Spot (Gi* positive and significant)
        # -1: Cold Spot (Gi* negative and significant)
        # 0: Not Significant
        valid_data['hot_cold_spot'] = 0 # Default to not significant
        # Hot spots (high values clustered)
        valid_data.loc[(valid_data['Gi_star'] > 0) & (valid_data['Gi_star_p_value'] < 0.05), 'hot_cold_spot'] = 1
        # Cold spots (low values clustered)
        valid_data.loc[(valid_data['Gi_star'] < 0) & (valid_data['Gi_star_p_value'] < 0.05), 'hot_cold_spot'] = -1
        
        print(f"   Calculated Gi* for {len(valid_data):,} points.")
        print(f"   Gi* range: Min={valid_data['Gi_star'].min():.4f}, Max={valid_data['Gi_star'].max():.4f}")
        print(f"   Hot spots (Gi* > 0, p < 0.05): {(valid_data['hot_cold_spot'] == 1).sum():,} points")
        print(f"   Cold spots (Gi* < 0, p < 0.05): {(valid_data['hot_cold_spot'] == -1).sum():,} points")
        
        return valid_data

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

    def plot_hot_cold_spots(self, gi_star_results_df, trait, save_path=None):
        """
        Generates a map visualizing statistically significant hot and cold spots for a given trait.
        
        Parameters:
        -----------
        gi_star_results_df : pandas.DataFrame
            DataFrame containing grid points with 'Gi_star' and 'hot_cold_spot' columns.
        trait : str
            Name of the personality trait.
        save_path : str, optional
            Path to save the plot image.
        """
        if gi_star_results_df is None or gi_star_results_df.empty:
            print("Gi* results DataFrame is empty or None. Cannot generate hot/cold spot map.")
            return

        print(f"\nGenerating Hot/Cold Spot Map for {trait}...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'Hot and Cold Spots: {trait}', fontsize=16, fontweight='bold')

        # Define colors for hot/cold spots
        colors = {1: 'red', -1: 'blue', 0: 'lightgray'}
        
        # Plot each category separately to ensure correct color mapping and legend
        plot_data = gi_star_results_df.dropna(subset=['hot_cold_spot']).copy() # Ensure 'hot_cold_spot' is not NaN

        # Plot the grid points by category
        for category_val, color in colors.items():
            subset = plot_data[plot_data['hot_cold_spot'] == category_val]
            if not subset.empty:
                ax.scatter(subset['grid_x'], subset['grid_y'], 
                           color=color, 
                           s=5, alpha=0.8, 
                           label={1: 'Hot Spot', -1: 'Cold Spot', 0: 'Not Significant'}.get(category_val))

        # Add boundaries if available
        if self.germany_gdf_proj is not None:
            self.germany_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)

        # Set limits based on grid data bounds for consistency
        aspect_ratio, bounds = self.calculate_plot_bounds_from_grid(gi_star_results_df) 
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
        
        # Create custom legend handles for colors
        legend_handles = [plt.Line2D([0], [0], marker='o', color='w', label='Hot Spot', markerfacecolor='red', markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', label='Cold Spot', markerfacecolor='blue', markersize=10),
                          plt.Line2D([0], [0], marker='o', color='w', label='Not Significant', markerfacecolor='lightgray', markersize=10)]
        ax.legend(handles=legend_handles, title="Significance", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Hot/Cold Spot map saved: {save_path}")
        plt.show()
        plt.close(fig)

    def plot_continuous_gi_star(self, gi_star_results_df, trait, save_path=None):
        """
        Generates a continuous map of Getis-Ord Gi* values for a given trait.
        This helps visualize the full range of local clustering, not just significant spots.
        
        Parameters:
        -----------
        gi_star_results_df : pandas.DataFrame
            DataFrame containing grid points with 'Gi_star' column.
        trait : str
            Name of the personality trait.
        save_path : str, optional
            Path to save the plot image.
        """
        if gi_star_results_df is None or gi_star_results_df.empty:
            print("Gi* results DataFrame is empty or None. Cannot generate continuous Gi* map.")
            return

        print(f"\nGenerating Continuous Gi* Map for {trait}...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'Continuous Getis-Ord Gi* Values: {trait}', fontsize=16, fontweight='bold')

        # Use a diverging colormap for Gi* values:
        # Red for positive (hot), Blue for negative (cold), White/light for near zero.
        cmap = 'RdBu_r' # Red-Blue reversed, so red is high Gi* (hot), blue is low Gi* (cold)
        
        # Define vmin and vmax symmetrically around 0 for diverging colormap
        max_abs_gi = gi_star_results_df['Gi_star'].abs().max()
        vmin = -max_abs_gi
        vmax = max_abs_gi

        # Plot the grid points, colored by Gi_star value
        scatter = ax.scatter(gi_star_results_df['grid_x'], gi_star_results_df['grid_y'], 
                             c=gi_star_results_df['Gi_star'], 
                             cmap=cmap, 
                             s=5, alpha=0.8,
                             vmin=vmin, vmax=vmax)

        # Add boundaries if available
        if self.germany_gdf_proj is not None:
            self.germany_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)

        # Set limits based on grid data bounds for consistency
        aspect_ratio, bounds = self.calculate_plot_bounds_from_grid(gi_star_results_df) 
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
                            label='Getis-Ord Gi* Value', 
                            shrink=0.7, aspect=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Continuous Gi* map saved: {save_path}")
        plt.show()
        plt.close(fig)


def run_hot_spot_analysis(grid_results_file, shapefile_path, trait_to_analyze, output_prefix="hot_spot"):
    """
    Main function to run the Hot Spot Analysis (Getis-Ord Gi*).
    
    Parameters:
    -----------
    grid_results_file : str
        Path to the CSV file containing pre-computed grid results (with Z-scores).
    shapefile_path : str
        Path to the boundary shapefile (e.g., German states), used for spatial context.
    trait_to_analyze : str
        Name of the personality trait for which to calculate Gi* (e.g., 'Openness').
    output_prefix : str, optional
        Prefix for the output image file.
    """
    print("\n" + "="*70)
    print(f"STARTING HOT SPOT ANALYSIS (GETIS-ORD GI*) FOR {trait_to_analyze.upper()}")
    print("="*70)

    # Initialize analyzer
    analyzer = HotSpotAnalyzer(shapefile_path=shapefile_path)
    
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
    
    # Calculate Getis-Ord Gi* and get the DataFrame with results
    gi_star_results_df = analyzer.calculate_getis_ord_gi_star(grid_data, trait_to_analyze)
    
    output_files = {}
    if gi_star_results_df is not None:
        # Plot Hot/Cold Spots (significant clusters)
        hot_cold_spot_map_path = f"{output_prefix}_{trait_to_analyze.lower()}_hot_cold_spot_map.png"
        analyzer.plot_hot_cold_spots(gi_star_results_df, trait_to_analyze, save_path=hot_cold_spot_map_path)
        output_files['hot_cold_spot_map'] = hot_cold_spot_map_path

        # Plot Continuous Gi* values (full range of local clustering)
        continuous_gi_map_path = f"{output_prefix}_{trait_to_analyze.lower()}_continuous_gi_map.png"
        analyzer.plot_continuous_gi_star(gi_star_results_df, trait_to_analyze, save_path=continuous_gi_map_path)
        output_files['continuous_gi_map'] = continuous_gi_map_path
    else:
        print(f"Getis-Ord Gi* calculation failed for {trait_to_analyze}. No maps generated.")

    print(f"\nHot Spot Analysis complete for {trait_to_analyze}!")
    print(f"Output files: {list(output_files.values())}")

    return output_files

if __name__ == "__main__":
    # Configuration for running the script directly
    # This should be the output file from your spatial_grid_computer.py
    grid_results_file_path = "spatial_personality_grid_results.csv" 
    shapefile_path = "german_shapefile/de.shp" 
    
    # Define the personality trait for which you want to perform Hot Spot Analysis
    # You can change this to 'Conscientiousness', 'Extraversion', 'Agreeableness', or 'Neuroticism'
    trait_for_hot_spot_analysis = 'Neuroticism' 

    output_prefix = "hot_spot_analysis"
    
    output_files = run_hot_spot_analysis(
        grid_results_file=grid_results_file_path,
        shapefile_path=shapefile_path,
        trait_to_analyze=trait_for_hot_spot_analysis,
        output_prefix=output_prefix
    )
    
    print(f"\nAll Hot Spot Analysis visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
