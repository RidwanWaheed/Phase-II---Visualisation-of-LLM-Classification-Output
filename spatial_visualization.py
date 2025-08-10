import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SpatialPersonalityVisualizer:
    """
    Visualization component for pre-computed spatial personality grid results.
    
    This class focuses on:
    1. Loading pre-computed grid results.
    2. Creating publication-quality maps.
    3. State-level aggregation for comparison.
    4. Grid-based and state-level visualizations.
    """
    
    def __init__(self, shapefile_path=None):
        """
        Initialize the visualizer with an optional shapefile for boundaries and state aggregation.
        
        Parameters:
        -----------
        shapefile_path : str, optional
            Path to boundary shapefile for visualization and state aggregation.
            This shapefile should ideally be in a projected CRS suitable for the region (e.g., UTM for Germany).
        """
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        
        # GeoDataFrames for Germany boundaries:
        # germany_gdf_wgs84: Used for spatial joins with original lat/lon user data.
        # germany_gdf_proj: Used for plotting to maintain correct aspect ratio in a projected CRS.
        self.germany_gdf_wgs84 = None 
        self.germany_gdf_proj = None  
        self.target_crs = None        # Stores the CRS of the projected shapefile.
        
        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("SPATIAL PERSONALITY VISUALIZER")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf_proj is not None else 'Not provided'}")
        print("Ready to load pre-computed grid results")
        print("="*60)
    
    def load_shapefile(self, shapefile_path):
        """
        Loads the shapefile for boundaries and state aggregation.
        Loads it in its native CRS for plotting and creates a WGS84 version for joins.
        """
        try:
            print(f"Loading shapefile: {shapefile_path}")
            self.germany_gdf_proj = gpd.read_file(shapefile_path)
            self.target_crs = self.germany_gdf_proj.crs
            print(f"Loaded shapefile in native CRS: {self.target_crs}")
            
            # Create a WGS84 (latitude/longitude) version for spatial joins with user data.
            if self.germany_gdf_proj.crs != 'EPSG:4326':
                print(f"Creating WGS84 version for spatial joins...")
                self.germany_gdf_wgs84 = self.germany_gdf_proj.to_crs('EPSG:4326')
            else:
                self.germany_gdf_wgs84 = self.germany_gdf_proj.copy()
            
            print(f"Loaded {len(self.germany_gdf_proj)} regions")
            
            # Identify the column containing state names.
            name_columns = ['name', 'NAME', 'NAME_1', 'ADMIN_NAME']
            state_col = None
            for col in name_columns:
                if col in self.germany_gdf_proj.columns:
                    state_col = col
                    break
            
            if state_col:
                states = self.germany_gdf_proj[state_col].unique()
                print(f"   State column: '{state_col}' with {len(states)} states")
                self.state_column = state_col
            else:
                print(f"   Warning: No recognized state name column found in shapefile.")
                print(f"   Available columns: {list(self.germany_gdf_proj.columns)}")
                self.state_column = None
                
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            self.germany_gdf_wgs84 = None
            self.germany_gdf_proj = None
            self.state_column = None
            self.target_crs = None
    
    def load_grid_results(self, grid_results_file):
        """
        Loads pre-computed grid results. Grid points (grid_lon, grid_lat) are assumed to be in WGS84
        and will be projected to the target_crs for consistent plotting.
        
        Parameters:
        -----------
        grid_results_file : str
            Path to CSV file with pre-computed grid results.
            
        Returns:
        --------
        pandas.DataFrame : Grid results with computed scores, including projected 'grid_x' and 'grid_y'.
        """
        print(f"\nLoading pre-computed grid results: {grid_results_file}")
        
        if not os.path.exists(grid_results_file):
            raise FileNotFoundError(f"Grid results file not found: {grid_results_file}")
        
        try:
            results = pd.read_csv(grid_results_file)
            
            required_cols = ['grid_lon', 'grid_lat']
            missing_cols = [col for col in required_cols if col not in results.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in grid results: {missing_cols}")
            
            trait_cols = []
            z_score_cols = []
            for trait in self.personality_traits:
                if trait in results.columns:
                    trait_cols.append(trait)
                if f'{trait}_z' in results.columns:
                    z_score_cols.append(f'{trait}_z')
            
            print(f"Loaded grid results:")
            print(f"   Grid points: {len(results):,}")
            print(f"   Personality traits: {len(trait_cols)}")
            print(f"   Z-score columns: {len(z_score_cols)}")
            
            if 'computation_date' in results.columns:
                comp_date = results['computation_date'].iloc[0]
                print(f"   Computed: {comp_date}")
            
            if 'r_km' in results.columns:
                r_km = results['r_km'].iloc[0]
                s_slope = results['s_slope'].iloc[0] if 's_slope' in results.columns else 'Unknown'
                grid_res = results['grid_resolution_km'].iloc[0] if 'grid_resolution_km' in results.columns else 'Unknown'
                print(f"   Parameters: r={r_km} km, s={s_slope}, grid={grid_res}km")
            
            valid_points = 0
            for trait in self.personality_traits:
                if f'{trait}_weight_sum' in results.columns:
                    valid_count = (results[f'{trait}_weight_sum'] > 0).sum()
                    valid_points = max(valid_points, valid_count)
            
            if valid_points > 0:
                coverage = (valid_points / len(results)) * 100
                print(f"   Data coverage: {valid_points:,}/{len(results):,} points ({coverage:.1f}%)")
            
            # Project grid points from WGS84 to the target projected CRS for plotting consistency.
            if self.target_crs and results['grid_lon'].dtype == float: 
                print(f"Projecting grid points from EPSG:4326 to {self.target_crs}...")
                geometry = [Point(lon, lat) for lon, lat in zip(results['grid_lon'], results['grid_lat'])]
                grid_gdf_wgs84 = gpd.GeoDataFrame(results, geometry=geometry, crs='EPSG:4326')
                results_proj = grid_gdf_wgs84.to_crs(self.target_crs)
                results['grid_x'] = results_proj.geometry.x
                results['grid_y'] = results_proj.geometry.y
                print("Grid points projected successfully.")
            
            return results
            
        except Exception as e:
            print(f"Error loading grid results: {e}")
            raise
    
    def aggregate_by_states(self, original_data_file):
        """
        Creates state-level aggregation from original user data.
        User data is assumed to be in WGS84 (latitude, longitude).
        
        Parameters:
        -----------
        original_data_file : str
            Path to original user data CSV file.
            
        Returns:
        --------
        pandas.DataFrame : State-level aggregated results with mean and Z-scores for traits.
        """
        if self.germany_gdf_wgs84 is None or self.state_column is None:
            print("Warning: No WGS84 shapefile or state column available for state aggregation.")
            return None
        
        print(f"\nCreating state-level aggregation from: {original_data_file}")
        
        try:
            data = pd.read_csv(original_data_file)
            required_cols = ['latitude', 'longitude'] + self.personality_traits
            data = data.dropna(subset=required_cols)
            
            print(f"   Loaded {len(data):,} user records")
            
            geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
            data_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
            
            data_with_states = gpd.sjoin(data_gdf, self.germany_gdf_wgs84, how='left', predicate='within')
            
            state_results = []
            for state_name in self.germany_gdf_wgs84[self.state_column].unique():
                state_data = data_with_states[data_with_states[self.state_column] == state_name]
                
                if len(state_data) > 0:
                    state_row = {'state': state_name, 'n_users': len(state_data)}
                    
                    for trait in self.personality_traits:
                        if trait in state_data.columns:
                            state_row[trait] = state_data[trait].mean()
                            state_row[f'{trait}_std'] = state_data[trait].std()
                    
                    state_results.append(state_row)
            
            if not state_results:
                print("Warning: No state results generated from aggregation.")
                return None
            
            state_results_df = pd.DataFrame(state_results)
            
            for trait in self.personality_traits:
                if trait in state_results_df.columns:
                    mean_score = state_results_df[trait].mean()
                    std_score = state_results_df[trait].std()
                    if std_score > 0:
                        state_results_df[f'{trait}_z'] = (state_results_df[trait] - mean_score) / std_score
                    else:
                        state_results_df[f'{trait}_z'] = 0.0 
            
            print(f"State aggregation complete: {len(state_results_df)} states.")
            return state_results_df
            
        except Exception as e:
            print(f"Error in state aggregation: {e}")
            return None
    
    def calculate_germany_plot_bounds(self, grid_results):
        """
        Calculates plot bounds and aspect ratio using projected grid data for consistent scaling.
        
        Parameters:
        -----------
        grid_results : pandas.DataFrame
            DataFrame containing projected grid coordinates ('grid_x', 'grid_y').
            
        Returns:
        --------
        tuple : (aspect_ratio, (x_min, x_max, y_min, y_max))
            The calculated aspect ratio and the bounding box coordinates in the projected CRS.
        """
        if 'grid_x' not in grid_results.columns or 'grid_y' not in grid_results.columns:
            raise ValueError("Grid results must contain 'grid_x' and 'grid_y' (projected coordinates).")

        x_min, x_max = grid_results['grid_x'].min(), grid_results['grid_x'].max()
        y_min, y_max = grid_results['grid_y'].min(), grid_results['grid_y'].max()
        
        width = x_max - x_min
        height = y_max - y_min
        
        if height == 0: height = 1 
        aspect_ratio = width / height
        
        print(f"   Projected extent: {width:.1f}m × {height:.1f}m (aspect ratio: {aspect_ratio:.2f})")
        
        return aspect_ratio, (x_min, x_max, y_min, y_max)

    def create_grid_visualization(self, grid_results, save_path=None):
        """
        Creates grid-based visualization using projected coordinates for accurate proportions.
        Removes all axis frames, ticks, and labels for a cleaner map appearance.
        
        Parameters:
        -----------
        grid_results : pandas.DataFrame
            DataFrame containing grid data, including projected 'grid_x' and 'grid_y'.
        save_path : str, optional
            Path to save the generated plot image.
            
        Returns:
        --------
        matplotlib.figure.Figure : The generated matplotlib Figure object.
        """
        print("\nCreating grid-based visualization with accurate proportions and no frames...")
        
        if 'grid_x' not in grid_results.columns or 'grid_y' not in grid_results.columns:
            print("Error: Projected grid coordinates (grid_x, grid_y) not found. Cannot create grid visualization.")
            return None

        aspect_ratio, bounds = self.calculate_germany_plot_bounds(grid_results)
        x_min, x_max, y_min, y_max = bounds
        
        subplot_width_inches = 6  
        subplot_height_inches = subplot_width_inches / aspect_ratio
        
        x_buffer = (x_max - x_min) * 0.02
        y_buffer = (y_max - y_min) * 0.02
        
        fig, axes = plt.subplots(2, 3, figsize=(subplot_width_inches * 3, subplot_height_inches * 2))
        fig.suptitle('Spatial Distribution of Big Five Personality Traits\n'
                    'Grid-Based Analysis', 
                    fontsize=16, fontweight='bold')
        
        cmap = plt.cm.RdYlBu_r 
        vmin, vmax = -1.96, 1.96 
        
        for idx, trait in enumerate(self.personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            if z_col in grid_results.columns:
                valid_mask = (~pd.isna(grid_results[z_col])) & (grid_results[z_col] != 0)
                valid_data = grid_results[valid_mask]
                
                if len(valid_data) > 0:
                    ax.scatter(valid_data['grid_x'], valid_data['grid_y'], 
                               c=valid_data[z_col], cmap=cmap, 
                               vmin=vmin, vmax=vmax, s=8, alpha=0.8)
                    
                    ax.set_aspect('equal', adjustable='box') 
                    
                    ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
                    ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
                    
                    if self.germany_gdf_proj is not None:
                        self.germany_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)
                    
                    ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
                    
                    ax.set_xlabel('') 
                    ax.set_ylabel('') 
                    ax.set_xticks([]) 
                    ax.set_yticks([]) 
                    for spine in ax.spines.values():
                        spine.set_visible(False) 
                    ax.grid(False) 
                    
                    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0)) 
                    
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'Missing {z_col}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
        
        if len(self.personality_traits) == 5:
            axes[1, 2].remove()
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95]) 
        
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z Score', rotation=270, labelpad=15, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"   Grid visualization saved: {save_path}")
        
        plt.show() 
        return fig
    
    def create_state_visualization(self, state_results, grid_results=None, save_path=None):
        """
        Creates state-level choropleth visualization using projected coordinates
        and consistent bounds from the grid data. Removes all axis frames, ticks, and labels.
        
        Parameters:
        -----------
        state_results : pandas.DataFrame
            DataFrame containing state-level aggregated personality data.
        grid_results : pandas.DataFrame, optional
            DataFrame containing grid data with projected coordinates. Used to ensure
            consistent plot bounds and aspect ratio with the grid visualization.
        save_path : str, optional
            Path to save the generated plot image.
            
        Returns:
        --------
        matplotlib.figure.Figure : The generated matplotlib Figure object.
        """
        if self.germany_gdf_proj is None or state_results is None:
            print("Cannot create state visualization: missing projected shapefile or state results.")
            return None
        
        print("\nCreating state-level visualization with consistent and accurate proportions (no frames)...")
        
        if grid_results is not None:
            if 'grid_x' not in grid_results.columns or 'grid_y' not in grid_results.columns:
                print("Warning: Grid results do not contain projected coordinates (grid_x, grid_y). Falling back to shapefile bounds.")
                x_min, y_min, x_max, y_max = self.germany_gdf_proj.total_bounds
                width = x_max - x_min
                height = y_max - y_min
                if height == 0: height = 1
                aspect_ratio = width / height
            else:
                aspect_ratio, bounds = self.calculate_germany_plot_bounds(grid_results)
                x_min, x_max, y_min, y_max = bounds
            print(f"   Using projected grid data bounds for consistency.")
        else:
            x_min, y_min, x_max, y_max = self.germany_gdf_proj.total_bounds
            width = x_max - x_min
            height = y_max - y_min
            if height == 0: height = 1
            aspect_ratio = width / height
            print(f"   Using projected shapefile bounds.")
        
        print(f"   Target aspect ratio: {aspect_ratio:.3f}")
        
        x_buffer = (x_max - x_min) * 0.02
        y_buffer = (y_max - y_min) * 0.02

        subplot_width_inches = 6  
        subplot_height_inches = subplot_width_inches / aspect_ratio
        
        fig, axes = plt.subplots(2, 3, figsize=(subplot_width_inches * 3, subplot_height_inches * 2))
        fig.suptitle('Spatial Distribution of Big Five Personality Traits\n'
                    'State-Level Administrative Aggregation', 
                    fontsize=16, fontweight='bold')
        
        merged_gdf = self.germany_gdf_proj.merge(state_results, left_on=self.state_column, 
                                          right_on='state', how='left')
        
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(self.personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            if z_col in merged_gdf.columns:
                merged_gdf.plot(column=z_col, cmap=cmap, vmin=vmin, vmax=vmax,
                               ax=ax, edgecolor='black', linewidth=0.7, 
                               missing_kwds={'color': 'lightgray', 'edgecolor': 'black'})
                
                ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
                ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
                ax.set_aspect('equal', adjustable='box') 
                
                ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
                
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False) 
                    
                final_xlim = ax.get_xlim()
                final_ylim = ax.get_ylim()
                final_ratio = (final_xlim[1] - final_xlim[0]) / (final_ylim[1] - final_ylim[0])
                if idx == 0:  
                    print(f"   Final subplot aspect ratio (projected): {final_ratio:.3f}")
                    
            else:
                ax.text(0.5, 0.5, f'Missing {z_col}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
        
        if len(self.personality_traits) == 5:
            axes[1, 2].remove()
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95]) 
        
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7]) 
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([]) 
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z Score', rotation=270, labelpad=15, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"   State visualization saved: {save_path}")
        
        plt.show() 
        return fig
    
def create_personality_visualizations(grid_results_file, original_data_file=None, 
                                    shapefile_path=None, output_prefix="personality_maps"):
    """
    Main function to create personality visualizations from pre-computed grid results.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to CSV file with pre-computed grid results (expected lat/lon).
    original_data_file : str, optional
        Path to original user data for state-level aggregation (expected lat/lon).
    shapefile_path : str, optional
        Path to boundary shapefile (preferably in a suitable projected CRS like UTM for Germany).
    output_prefix : str
        Prefix for output files.
        
    Returns:
    --------
    dict : Paths to created visualization files.
    """
    
    print("STARTING SPATIAL PERSONALITY VISUALIZATION (with projection for accuracy)")
    print("Ensuring geographically accurate maps by using a projected coordinate system.")
    print("="*70)
    
    visualizer = SpatialPersonalityVisualizer(shapefile_path=shapefile_path)
    
    if visualizer.target_crs is None:
        print("Error: Shapefile could not be loaded or has no CRS. Cannot proceed with visualizations.")
        return {}

    grid_results = visualizer.load_grid_results(grid_results_file)
    
    state_results = None
    if original_data_file and visualizer.germany_gdf_wgs84 is not None:
        state_results = visualizer.aggregate_by_states(original_data_file)
    
    output_files = {}
    
    grid_file = f"{output_prefix}_grid_based.png"
    visualizer.create_grid_visualization(grid_results, save_path=grid_file)
    output_files['grid'] = grid_file
    
    if state_results is not None:
        state_file = f"{output_prefix}_state_level.png"
        visualizer.create_state_visualization(state_results, grid_results, save_path=state_file)
        output_files['state'] = state_file
    
    print(f"\nVISUALIZATION COMPLETE")
    print(f"Created {len(output_files)} visualization files:")
    for viz_type, file_path in output_files.items():
        print(f"   {viz_type.title()}: {file_path}")
    
    
    return output_files

if __name__ == "__main__":
    shapefile_path = "german_shapefile/de.shp"
    grid_results_file = "spatial_personality_grid_results.csv"  
    original_data_file = "final_users_for_spatial_visualization.csv"  
    output_prefix = "personality_maps"
    
    output_files = create_personality_visualizations(
        grid_results_file=grid_results_file,
        original_data_file=original_data_file,
        shapefile_path=shapefile_path,
        output_prefix=output_prefix
    )
    
    print(f"\nAll visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
