import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

class SpatialPersonalityVisualizer:
    """
    Visualization component for pre-computed spatial personality grid results.
    Creates individual trait maps using correct geographic proportions.
    CORRECTED VERSION: Fixed aspect ratio calculation.
    """
    
    def __init__(self, shapefile_path=None):
        """Initialize the visualizer with optional shapefile."""
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        
        # Shapefile for boundaries and state aggregation
        self.germany_gdf = None
        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("SPATIAL PERSONALITY VISUALIZER - CORRECTED")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.germany_gdf is not None else 'Not provided'}")
        print("Ready to load pre-computed grid results")
        print("="*60)
    
    def load_shapefile(self, shapefile_path):
        """Load shapefile for boundaries and state aggregation."""
        try:
            print(f"Loading shapefile: {shapefile_path}")
            self.germany_gdf = gpd.read_file(shapefile_path)
            
            # Ensure it's in WGS84 for visualization consistency
            if self.germany_gdf.crs != 'EPSG:4326':
                print(f"Converting shapefile from {self.germany_gdf.crs} to EPSG:4326")
                self.germany_gdf = self.germany_gdf.to_crs('EPSG:4326')
            
            print(f"Loaded {len(self.germany_gdf)} regions")
            
            # Check for state name column
            name_columns = ['name', 'NAME', 'NAME_1', 'ADMIN_NAME']
            state_col = None
            for col in name_columns:
                if col in self.germany_gdf.columns:
                    state_col = col
                    break
            
            if state_col:
                states = self.germany_gdf[state_col].unique()
                print(f"   State column: '{state_col}' with {len(states)} states")
                self.state_column = state_col
            else:
                print(f"   Warning: No recognized state name column found")
                print(f"   Available columns: {list(self.germany_gdf.columns)}")
                self.state_column = None
                
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            self.germany_gdf = None
            self.state_column = None
    
    def load_grid_results(self, grid_results_file):
        """Load pre-computed grid results from SpatialPersonalityGridComputer."""
        print(f"\nLoading pre-computed grid results: {grid_results_file}")
        
        if not os.path.exists(grid_results_file):
            raise FileNotFoundError(f"Grid results file not found: {grid_results_file}")
        
        try:
            results = pd.read_csv(grid_results_file)
            
            # Validate expected columns
            required_cols = ['grid_lon', 'grid_lat']
            missing_cols = [col for col in required_cols if col not in results.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in grid results: {missing_cols}")
            
            # Check for personality trait columns
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
            
            # Extract metadata if available
            if 'computation_date' in results.columns:
                comp_date = results['computation_date'].iloc[0]
                print(f"   Computed: {comp_date}")
            
            if 'r_miles' in results.columns:
                r_miles = results['r_miles'].iloc[0]
                s_slope = results['s_slope'].iloc[0] if 's_slope' in results.columns else 'Unknown'
                grid_res = results['grid_resolution_km'].iloc[0] if 'grid_resolution_km' in results.columns else 'Unknown'
                print(f"   Parameters: r={r_miles} miles, s={s_slope}, grid={grid_res}km")
            
            # Check data quality
            valid_points = 0
            for trait in self.personality_traits:
                if f'{trait}_weight_sum' in results.columns:
                    valid_count = (results[f'{trait}_weight_sum'] > 0).sum()
                    valid_points = max(valid_points, valid_count)
            
            if valid_points > 0:
                coverage = (valid_points / len(results)) * 100
                print(f"   Data coverage: {valid_points:,}/{len(results):,} points ({coverage:.1f}%)")
            
            return results
            
        except Exception as e:
            print(f"Error loading grid results: {e}")
            raise
    
    def calculate_optimal_aspect_ratio(self, grid_results, method='auto'):
        """
        Calculate optimal aspect ratio for geographic data visualization.
        
        Parameters:
        -----------
        method : str
            'corrected' - Use 1/cos(latitude) correction
            'equal' - Use matplotlib's equal aspect
            'auto' - Automatically choose best method
        """
        # Get actual data bounds
        lat_min, lat_max = grid_results['grid_lat'].min(), grid_results['grid_lat'].max()
        lon_min, lon_max = grid_results['grid_lon'].min(), grid_results['grid_lon'].max()
        
        # Calculate center latitude
        center_lat = (lat_min + lat_max) / 2
        
        # Calculate the theoretically correct aspect ratio
        cos_lat = np.cos(np.radians(center_lat))
        theoretical_correct = 1.0 / cos_lat
        
        print(f"   Geographic extent analysis:")
        print(f"   Latitude range: {lat_min:.3f} to {lat_max:.3f} degrees")
        print(f"   Longitude range: {lon_min:.3f} to {lon_max:.3f} degrees")
        print(f"   Center latitude: {center_lat:.2f} degrees")
        print(f"   Theoretical correct aspect: {theoretical_correct:.3f}")
        
        # Choose method
        if method == 'corrected':
            chosen_aspect = theoretical_correct
            print(f"   Using corrected aspect ratio: {chosen_aspect:.3f}")
        elif method == 'equal':
            chosen_aspect = 'equal'
            print(f"   Using equal aspect ratio")
        elif method == 'auto':
            # For Germany's size, equal aspect often works fine
            # Use corrected only if the distortion would be significant
            lat_span = lat_max - lat_min
            if lat_span > 10.0:  # Large area, use correction
                chosen_aspect = theoretical_correct
                print(f"   Large area detected, using corrected aspect: {chosen_aspect:.3f}")
            else:  # Smaller area, equal is often fine
                chosen_aspect = 'equal'
                print(f"   Moderate area, using equal aspect (distortion minimal)")
        
        return chosen_aspect, (lon_min, lon_max, lat_min, lat_max)
    
    def aggregate_by_states(self, original_data_file):
        """Create state-level aggregation from original user data."""
        if self.germany_gdf is None or self.state_column is None:
            print("Warning: No shapefile or state column available for state aggregation")
            return None
        
        print(f"\nCreating state-level aggregation from: {original_data_file}")
        
        try:
            # Load original data
            data = pd.read_csv(original_data_file)
            required_cols = ['latitude', 'longitude'] + self.personality_traits
            data = data.dropna(subset=required_cols)
            
            print(f"   Loaded {len(data):,} user records")
            
            # Create point geometries
            geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
            data_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
            
            # Spatial join to assign users to states
            data_with_states = gpd.sjoin(data_gdf, self.germany_gdf, how='left', predicate='within')
            
            # Aggregate by state
            state_results = []
            
            for state_name in self.germany_gdf[self.state_column].unique():
                state_data = data_with_states[data_with_states[self.state_column] == state_name]
                
                if len(state_data) > 0:
                    state_row = {'state': state_name, 'n_users': len(state_data)}
                    
                    # Calculate mean personality scores
                    for trait in self.personality_traits:
                        if trait in state_data.columns:
                            state_row[trait] = state_data[trait].mean()
                            state_row[f'{trait}_std'] = state_data[trait].std()
                    
                    state_results.append(state_row)
            
            if not state_results:
                print("Warning: No state results generated")
                return None
            
            state_results_df = pd.DataFrame(state_results)
            
            # Standardize scores for comparison
            for trait in self.personality_traits:
                if trait in state_results_df.columns:
                    mean_score = state_results_df[trait].mean()
                    std_score = state_results_df[trait].std()
                    if std_score > 0:
                        state_results_df[f'{trait}_z'] = (state_results_df[trait] - mean_score) / std_score
                    else:
                        state_results_df[f'{trait}_z'] = 0.0
            
            print(f"State aggregation complete: {len(state_results_df)} states")
            return state_results_df
            
        except Exception as e:
            print(f"Error in state aggregation: {e}")
            return None
    
    def create_individual_grid_trait_maps(self, grid_results, save_prefix="grid", aspect_method='auto'):
        """Create individual maps for each personality trait using grid data with optimal proportions."""
        print("\nCreating individual grid-based trait maps...")
        
        # Calculate optimal aspect ratio
        optimal_aspect, bounds = self.calculate_optimal_aspect_ratio(grid_results, method=aspect_method)
        lon_min, lon_max, lat_min, lat_max = bounds
        
        # Create individual map for each trait
        individual_files = []
        
        for trait in self.personality_traits:
            print(f"   Creating {trait} grid map...")
            
            # Calculate figure size maintaining reasonable proportions
            fig_width = 12
            fig_height = fig_width / 1.2  # Reasonable height
            
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            
            z_col = f'{trait}_z'
            if z_col in grid_results.columns:
                # Filter to valid data points
                valid_mask = (~pd.isna(grid_results[z_col])) & (grid_results[z_col] != 0)
                valid_data = grid_results[valid_mask]
                
                if len(valid_data) > 0:
                    # Create the scatter plot
                    scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                       c=valid_data[z_col], cmap=plt.cm.RdYlBu_r, 
                                       vmin=-1.96, vmax=1.96, s=15, alpha=0.8)
                    
                    # Add boundaries if available
                    if self.germany_gdf is not None:
                        self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1.0, alpha=0.8)
                    
                    # Apply optimal aspect ratio
                    ax.set_aspect(optimal_aspect)
                    
                    # Styling
                    aspect_note = f"Optimal Proportions ({optimal_aspect if isinstance(optimal_aspect, str) else f'aspect={optimal_aspect:.2f}'})"
                    ax.set_title(f'Spatial Distribution of {trait} in Germany\n'
                                f'Grid-Based Analysis - {aspect_note}', 
                                fontsize=16, fontweight='bold', pad=20)
                    ax.set_xlabel('Longitude', fontsize=14)
                    ax.set_ylabel('Latitude', fontsize=14)
                    ax.grid(True, alpha=0.3)
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
                    cbar.set_label('Z Score', rotation=270, labelpad=20, fontsize=14)
                    
                    # Add statistics
                    mean_z = valid_data[z_col].mean()
                    std_z = valid_data[z_col].std()
                    n_points = len(valid_data)
                    ax.text(0.02, 0.98, f'μ={mean_z:.3f}\nσ={std_z:.3f}\nn={n_points:,}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                           fontsize=12)
                    
                    # Save individual file
                    filename = f"{save_prefix}_{trait.lower()}.png"
                    plt.savefig(filename, dpi=300, bbox_inches='tight', 
                               facecolor='white', edgecolor='none')
                    individual_files.append(filename)
                    print(f"     Saved: {filename}")
                else:
                    print(f"     Warning: No valid data for {trait}")
            else:
                print(f"     Error: Missing {z_col} column")
            
            plt.close(fig)  # Close to save memory
        
        return individual_files
    
    def create_individual_state_trait_maps(self, state_results, save_prefix="state"):
        """Create individual maps for each personality trait using state data."""
        if self.germany_gdf is None or state_results is None:
            print("Cannot create individual state maps: missing shapefile or state results")
            return []
        
        print("\nCreating individual state-level trait maps...")
        
        # For state maps, equal aspect usually works fine since geopandas handles the projection
        print(f"   Using equal aspect ratio for state maps (geopandas projection)")
        
        # Merge state results with shapefile
        merged_gdf = self.germany_gdf.merge(state_results, left_on=self.state_column, 
                                          right_on='state', how='left')
        
        # Create individual map for each trait
        individual_files = []
        
        for trait in self.personality_traits:
            print(f"   Creating {trait} state map...")
            
            # Calculate figure size
            fig_width = 12
            fig_height = fig_width / 1.2
            
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
            
            z_col = f'{trait}_z'
            if z_col in merged_gdf.columns:
                # Plot using geopandas
                merged_gdf.plot(column=z_col, cmap=plt.cm.RdYlBu_r, 
                               vmin=-1.96, vmax=1.96,
                               ax=ax, edgecolor='black', linewidth=0.8, 
                               missing_kwds={'color': 'lightgray', 'edgecolor': 'black'})
                
                # Use equal aspect for state maps
                ax.set_aspect('equal')
                
                # Styling
                ax.set_title(f'Spatial Distribution of {trait} in Germany\n'
                            f'State-Level Administrative Aggregation', 
                            fontsize=16, fontweight='bold', pad=20)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, 
                                         norm=plt.Normalize(vmin=-1.96, vmax=1.96))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30)
                cbar.set_label('Z Score', rotation=270, labelpad=20, fontsize=14)
                
                # Add statistics
                valid_data = merged_gdf[merged_gdf[z_col].notna()]
                if len(valid_data) > 0:
                    mean_z = valid_data[z_col].mean()
                    std_z = valid_data[z_col].std()
                    n_states = len(valid_data)
                    ax.text(0.02, 0.98, f'μ={mean_z:.3f}\nσ={std_z:.3f}\nn={n_states} states', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                           fontsize=12)
                
                # Save individual file
                filename = f"{save_prefix}_{trait.lower()}.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                individual_files.append(filename)
                print(f"     Saved: {filename}")
            else:
                print(f"     Error: Missing {z_col} column")
            
            plt.close(fig)  # Close to save memory
        
        return individual_files

def create_individual_personality_maps(grid_results_file, original_data_file=None, 
                                     shapefile_path=None, output_prefix="trait",
                                     aspect_method='auto'):
    """
    Create individual maps for each personality trait using optimal geographic proportions.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to CSV file with pre-computed grid results
    original_data_file : str, optional
        Path to original user data for state-level maps
    shapefile_path : str, optional
        Path to boundary shapefile
    output_prefix : str
        Prefix for output files
    aspect_method : str
        'auto', 'equal', or 'corrected' for aspect ratio method
        
    Returns:
    --------
    dict : Lists of created visualization files by type
    """
    
    print("CREATING INDIVIDUAL TRAIT MAPS - CORRECTED VERSION")
    print("Each trait gets its own map using optimal geographic proportions")
    print("="*70)
    
    # Initialize visualizer
    visualizer = SpatialPersonalityVisualizer(shapefile_path=shapefile_path)
    
    # Load pre-computed grid results
    grid_results = visualizer.load_grid_results(grid_results_file)
    
    output_files = {
        'grid_maps': [],
        'state_maps': []
    }
    
    # 1. Create individual grid-based trait maps
    grid_files = visualizer.create_individual_grid_trait_maps(grid_results, 
                                                              save_prefix=f"{output_prefix}_grid",
                                                              aspect_method=aspect_method)
    output_files['grid_maps'] = grid_files
    
    # 2. Create individual state-level trait maps
    if original_data_file and visualizer.germany_gdf is not None:
        state_results = visualizer.aggregate_by_states(original_data_file)
        
        if state_results is not None:
            state_files = visualizer.create_individual_state_trait_maps(state_results,
                                                                        save_prefix=f"{output_prefix}_state")
            output_files['state_maps'] = state_files
        else:
            print("   Warning: Could not create state aggregation")
    else:
        if not original_data_file:
            print("   Info: No original data file provided - skipping state-level maps")
        if not visualizer.germany_gdf:
            print("   Info: No shapefile loaded - skipping state-level maps")
    
    # Summary
    total_maps = len(output_files['grid_maps']) + len(output_files['state_maps'])
    
    print(f"\nINDIVIDUAL TRAIT MAPS COMPLETE!")
    print(f"Created {total_maps} individual maps:")
    
    if output_files['grid_maps']:
        print(f"\nGrid-based trait maps ({len(output_files['grid_maps'])}):")
        for file in output_files['grid_maps']:
            print(f"   {file}")
    
    if output_files['state_maps']:
        print(f"\nState-level trait maps ({len(output_files['state_maps'])}):")
        for file in output_files['state_maps']:
            print(f"   {file}")
    
    print(f"\nMap Characteristics:")
    print(f"   Each trait: Individual focused map")
    print(f"   Grid maps: Use optimal aspect ratio (auto-selected)")
    print(f"   State maps: Use equal aspect ratio (works well with geopandas)")
    print(f"   All maps: Maintain proper geographic accuracy")
    
    return output_files

if __name__ == "__main__":
    # Configuration
    grid_results_file = "utm_spatial_personality_grid_results.csv"
    original_data_file = "final_users_for_spatial_visualization.csv"
    shapefile_path = "german_shapefile_utm32n/de_utm32n.shp"
    output_prefix = "trait"
    
    # Create individual trait maps with auto aspect ratio selection
    output_files = create_individual_personality_maps(
        grid_results_file=grid_results_file,
        original_data_file=original_data_file,
        shapefile_path=shapefile_path,
        output_prefix=output_prefix,
        aspect_method='auto'  # Let the code choose the best method
    )
    
    total_maps = len(output_files['grid_maps']) + len(output_files['state_maps'])
    
    print(f"\nIndividual trait maps created successfully!")
    print(f"Total maps created: {total_maps}")
    print(f"\nTo try different aspect ratios, use:")
    print(f"   aspect_method='equal' - For equal aspect (may work fine for Germany)")
    print(f"   aspect_method='corrected' - For latitude-corrected aspect")
    print(f"   aspect_method='auto' - Let the code choose automatically")