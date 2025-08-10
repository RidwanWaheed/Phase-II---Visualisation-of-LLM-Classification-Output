import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os

warnings.filterwarnings('ignore')

class SpatialTraitInteractionVisualizer:
    """
    Component for visualizing the spatial interaction of two personality traits on a map.
    This class categorizes geographical areas (e.g., states) based on high/low values
    of two selected traits and plots these categories on a map.
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
        self.germany_gdf_wgs84 = None # For spatial joins with lat/lon data
        self.germany_gdf_proj = None  # For plotting to maintain correct aspect
        self.target_crs = None        # CRS of the projected shapefile
        self.state_column = None      # Column in GeoDataFrame holding state names

        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("SPATIAL TRAIT INTERACTION VISUALIZER")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf_proj is not None else 'Not provided'}")
        print("Ready to visualize trait interactions.")
        print("="*60)

    def load_shapefile(self, shapefile_path):
        """
        Loads the shapefile for boundaries. Loads it in its native CRS for plotting
        and creates a WGS84 version for spatial joins.
        """
        try:
            print(f"Loading shapefile: {shapefile_path}")
            self.germany_gdf_proj = gpd.read_file(shapefile_path)
            self.target_crs = self.germany_gdf_proj.crs
            print(f"Loaded shapefile in native CRS: {self.target_crs}")
            
            if self.germany_gdf_proj.crs != 'EPSG:4326':
                print(f"Creating WGS84 version for spatial joins...")
                self.germany_gdf_wgs84 = self.germany_gdf_proj.to_crs('EPSG:4326')
            else:
                self.germany_gdf_wgs84 = self.germany_gdf_proj.copy()
            
            print(f"Loaded {len(self.germany_gdf_proj)} regions")
            
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

    def aggregate_by_states(self, original_data_file):
        """
        Aggregates personality trait data by states.
        
        Parameters:
        -----------
        original_data_file : str
            Path to original user data CSV file (expected to contain lat/lon and trait scores).
            
        Returns:
        --------
        pandas.DataFrame : State-level aggregated results with mean trait scores.
        """
        if self.germany_gdf_wgs84 is None or self.state_column is None:
            print("Error: No WGS84 shapefile or state column available for state aggregation.")
            return None
        
        print(f"\nAggregating data by states from: {original_data_file}")
        
        try:
            data = pd.read_csv(original_data_file)
            required_cols = ['latitude', 'longitude'] + self.personality_traits
            data = data.dropna(subset=required_cols)
            
            print(f"   Loaded {len(data):,} user records.")
            
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
                    
                    state_results.append(state_row)
            
            if not state_results:
                print("Warning: No state results generated from aggregation.")
                return None
            
            state_results_df = pd.DataFrame(state_results)
            print(f"State aggregation complete: {len(state_results_df)} states.")
            return state_results_df
            
        except Exception as e:
            print(f"Error in state aggregation: {e}")
            return None

    def create_spatial_trait_interaction_map(self, aggregated_data, trait1, trait2, save_path=None):
        """
        Creates a map visualizing the spatial interaction of two personality traits.
        Categorizes states based on high/low values of the two traits (relative to their means).
        
        Parameters:
        -----------
        aggregated_data : pandas.DataFrame
            DataFrame with state-level aggregated personality trait means.
        trait1 : str
            Name of the first personality trait (e.g., 'Openness').
        trait2 : str
            Name of the second personality trait (e.g., 'Neuroticism').
        save_path : str, optional
            Path to save the generated map image.
        """
        if self.germany_gdf_proj is None or aggregated_data is None:
            print("Error: Missing projected shapefile or aggregated data. Cannot create interaction map.")
            return None
        
        if trait1 not in aggregated_data.columns or trait2 not in aggregated_data.columns:
            raise ValueError(f"Traits '{trait1}' or '{trait2}' not found in aggregated data.")

        print(f"\nCreating spatial interaction map for {trait1} vs {trait2}...")

        # Calculate means for categorization
        mean_trait1 = aggregated_data[trait1].mean()
        mean_trait2 = aggregated_data[trait2].mean()

        # Merge aggregated data with the projected GeoDataFrame
        merged_gdf = self.germany_gdf_proj.merge(aggregated_data, left_on=self.state_column, 
                                                  right_on='state', how='left')

        # Define categories based on high/low relative to the mean
        def categorize_traits(row):
            if pd.isna(row[trait1]) or pd.isna(row[trait2]):
                return 'No Data'
            
            high_trait1 = row[trait1] > mean_trait1
            high_trait2 = row[trait2] > mean_trait2

            if high_trait1 and high_trait2:
                return f'High {trait1}, High {trait2}'
            elif high_trait1 and not high_trait2:
                return f'High {trait1}, Low {trait2}'
            elif not high_trait1 and high_trait2:
                return f'Low {trait1}, High {trait2}'
            else: # not high_trait1 and not high_trait2
                return f'Low {trait1}, Low {trait2}'

        merged_gdf['trait_category'] = merged_gdf.apply(categorize_traits, axis=1)

        # Define colors for categories using lighter shades
        category_colors = {
            f'High {trait1}, High {trait2}': '#FFC0CB', # Light Pink (lighter red)
            f'High {trait1}, Low {trait2}': '#90EE90',    # Light Green (lighter green)
            f'Low {trait1}, High {trait2}': '#ADD8E6', # Light Blue (lighter blue)
            f'Low {trait1}, Low {trait2}': '#FFDAB9',     # Peach (lighter orange)
            'No Data': '#D3D3D3'                          # Light Gray
        }
        
        # Ensure all categories are in the merged_gdf for consistent legend
        all_categories = list(category_colors.keys())
        merged_gdf['trait_category'] = pd.Categorical(merged_gdf['trait_category'], categories=all_categories, ordered=False)


        # Plotting
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        fig.suptitle(f'Spatial Trait Interactions ({trait1} vs {trait2})', 
                     fontsize=16, fontweight='bold')

        # Plot each category separately to ensure correct color mapping and legend
        for category, color in category_colors.items():
            subset = merged_gdf[merged_gdf['trait_category'] == category]
            subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.7, label=category)

        # Set limits based on shapefile bounds
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
        
        # Create custom legend handles for colors
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=color, markersize=10) 
                   for label, color in category_colors.items()]
        
        # Position the legend outside the plot area
        ax.legend(handles=handles, title="Categories", loc='upper left', bbox_to_anchor=(1.02, 1), frameon=True)


        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"   Spatial trait interaction map saved: {save_path}")
        
        plt.show()
        plt.close(fig)
        return fig

def run_spatial_trait_interaction_analysis(data_file, shapefile_path, trait1, trait2, output_prefix="spatial_trait_interaction"):
    """
    Main function to run the spatial trait interaction analysis and generate the map.
    
    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing user data with personality traits and location.
    shapefile_path : str
        Path to the boundary shapefile (e.g., German states).
    trait1 : str
        Name of the first personality trait for interaction (e.g., 'Openness').
    trait2 : str
        Name of the second personality trait for interaction (e.g., 'Neuroticism').
    output_prefix : str, optional
        Prefix for the output image file.
    """
    print("\n" + "="*70)
    print("STARTING SPATIAL TRAIT INTERACTION VISUALIZATION")
    print("="*70)

    visualizer = SpatialTraitInteractionVisualizer(shapefile_path=shapefile_path)
    
    if visualizer.target_crs is None:
        print("Error: Shapefile could not be loaded or has no CRS. Cannot proceed.")
        return {}

    aggregated_data = visualizer.aggregate_by_states(data_file)
    
    if aggregated_data is None:
        print("Error: Could not aggregate data by states. Cannot create interaction map.")
        return {}

    output_file_path = f"{output_prefix}_{trait1.lower()}_vs_{trait2.lower()}.png"
    visualizer.create_spatial_trait_interaction_map(aggregated_data, trait1, trait2, save_path=output_file_path)
    
    print(f"\nSpatial trait interaction visualization complete!")
    print(f"Output file: {output_file_path}")

    return {'spatial_trait_interaction_map': output_file_path}

if __name__ == "__main__":
    # Configuration for running the script directly
    data_file_path = "final_users_for_spatial_visualization.csv" 
    shapefile_path = "german_shapefile/de.shp"
    
    # Define the two traits you want to visualize their interaction
    trait_1 = 'Openness'
    trait_2 = 'Neuroticism'

    output_prefix = "spatial_trait_interaction"
    
    output_files = run_spatial_trait_interaction_analysis(
        data_file=data_file_path,
        shapefile_path=shapefile_path,
        trait1=trait_1,
        trait2=trait_2,
        output_prefix=output_prefix
    )
    
    print(f"\nAll spatial trait interaction visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
