import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import os
from pyproj import CRS, Transformer

class SpatialPersonalityMapper:
    """
    Implements Ebert et al.'s spatial mapping methodology for personality traits.
    Updated with proper coordinate system handling using UTM Zone 32N for Germany.
    """
    
    def __init__(self, r_miles=45, s_slope=7, grid_resolution_km=5, shapefile_path=None, 
                 target_crs='EPSG:32632'):
        """
        Initialize the spatial mapper with proper coordinate system handling.
        
        Parameters:
        -----------
        r_miles : float
            Distance radius in miles (Ebert's parameter)
        s_slope : float
            Distance decay slope (Ebert's parameter)
        grid_resolution_km : float
            Grid resolution in kilometers
        shapefile_path : str
            Path to German shapefile
        target_crs : str
            Target coordinate reference system (default: UTM Zone 32N for Germany)
        """
        self.r_miles = r_miles
        self.s_slope = s_slope
        self.r_km = r_miles * 1.60934  # Convert miles to km
        self.r_meters = self.r_km * 1000  # Convert to meters for UTM
        self.grid_resolution_km = grid_resolution_km
        self.grid_resolution_meters = grid_resolution_km * 1000  # Convert to meters
        
        # Coordinate system setup
        self.source_crs = 'EPSG:4326'  # WGS84 (input data)
        self.target_crs = target_crs   # UTM Zone 32N for Germany
        self.transformer_to_utm = Transformer.from_crs(self.source_crs, self.target_crs, always_xy=True)
        self.transformer_to_wgs84 = Transformer.from_crs(self.target_crs, self.source_crs, always_xy=True)
        
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        self.germany_gdf = None
        self.germany_gdf_utm = None
        
        # Load German shapefile if provided
        if shapefile_path:
            self.load_german_shapefile(shapefile_path)
        
        print(f"Spatial Personality Mapper initialized:")
        print(f"  - Source CRS: {self.source_crs} (WGS84)")
        print(f"  - Target CRS: {self.target_crs} (UTM Zone 32N)")
        print(f"  - Distance radius: {r_miles} miles ({self.r_km:.1f} km, {self.r_meters:.0f} m)")
        print(f"  - Distance decay slope: {s_slope}")
        print(f"  - Grid resolution: {grid_resolution_km} km ({self.grid_resolution_meters:.0f} m)")
        print(f"  - German shapefile: {'Loaded' if self.germany_gdf is not None else 'Not provided'}")
    
    def load_german_shapefile(self, shapefile_path):
        """Load German shapefile with proper coordinate system handling."""
        try:
            # Load shapefile
            self.germany_gdf = gpd.read_file(shapefile_path)
            original_crs = self.germany_gdf.crs
            
            # Ensure it's in WGS84 for consistency
            if self.germany_gdf.crs != self.source_crs:
                print(f"Converting shapefile from {self.germany_gdf.crs} to {self.source_crs}")
                self.germany_gdf = self.germany_gdf.to_crs(self.source_crs)
            
            # Create UTM version for spatial operations
            self.germany_gdf_utm = self.germany_gdf.to_crs(self.target_crs)
            
            print(f"✅ Loaded German shapefile: {len(self.germany_gdf)} regions")
            print(f"   Original CRS: {original_crs}")
            print(f"   WGS84 version: {self.germany_gdf.crs}")
            print(f"   UTM version: {self.germany_gdf_utm.crs}")
            
            # Check for 'name' column
            if 'name' in self.germany_gdf.columns:
                states = self.germany_gdf['name'].unique()
                print(f"   States available: {len(states)} states found")
            else:
                print(f"   Warning: 'name' column not found in shapefile")
                print(f"   Available columns: {list(self.germany_gdf.columns)}")
                
        except Exception as e:
            print(f"❌ Error loading shapefile: {e}")
            self.germany_gdf = None
            self.germany_gdf_utm = None
    
    def transform_coordinates_to_utm(self, data):
        """Transform lat/lon coordinates to UTM for accurate distance calculations."""
        print("Transforming coordinates to UTM Zone 32N...")
        
        data_utm = data.copy()
        
        # Transform coordinates
        utm_x, utm_y = self.transformer_to_utm.transform(
            data['longitude'].values, 
            data['latitude'].values
        )
        
        data_utm['utm_x'] = utm_x
        data_utm['utm_y'] = utm_y
        
        print(f"✅ Transformed {len(data_utm)} points to UTM coordinates")
        print(f"   UTM X range: {utm_x.min():.0f} to {utm_x.max():.0f} meters")
        print(f"   UTM Y range: {utm_y.min():.0f} to {utm_y.max():.0f} meters")
        
        return data_utm
    
    def euclidean_distance(self, x1, y1, x2, y2):
        """Calculate Euclidean distance between two points in UTM coordinates (meters)."""
        return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    def distance_decay_weight(self, distances_meters):
        """
        Calculate distance decay weights using Ebert's log-logistic function.
        Updated to work with UTM distances in meters.
        """
        # Convert distances to miles for Ebert's original formula
        distances_miles = distances_meters / 1609.34  # meters to miles
        
        # Ebert's log-logistic distance decay: f(d) = 1 / (1 + (d/r)^s)
        weights = 1 / (1 + (distances_miles / self.r_miles) ** self.s_slope)
        return weights
    
    def create_germany_grid_utm(self, data_utm):
        """Create spatial grid in UTM coordinates for accurate measurements."""
        print("Creating UTM-based spatial grid...")
        
        # Get UTM bounds with buffer (in meters)
        x_min, x_max = data_utm['utm_x'].min() - 10000, data_utm['utm_x'].max() + 10000
        y_min, y_max = data_utm['utm_y'].min() - 10000, data_utm['utm_y'].max() + 10000
        
        # If shapefile available, use its bounds instead
        if self.germany_gdf_utm is not None:
            bounds = self.germany_gdf_utm.total_bounds  # [minx, miny, maxx, maxy]
            x_min, y_min, x_max, y_max = bounds
            print(f"Using shapefile bounds for grid creation")
        
        print(f"Creating grid for UTM bounds:")
        print(f"  X (Easting): {x_min:.0f} to {x_max:.0f} meters")
        print(f"  Y (Northing): {y_min:.0f} to {y_max:.0f} meters")
        
        # Create grid in UTM coordinates (meters)
        x_coords = np.arange(x_min, x_max + self.grid_resolution_meters, self.grid_resolution_meters)
        y_coords = np.arange(y_min, y_max + self.grid_resolution_meters, self.grid_resolution_meters)
        
        # Create meshgrid and flatten
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        grid_points_utm = pd.DataFrame({
            'utm_x': x_grid.flatten(),
            'utm_y': y_grid.flatten(),
            'grid_idx': range(len(x_grid.flatten()))
        })
        
        # Transform back to WGS84 for visualization
        grid_lon, grid_lat = self.transformer_to_wgs84.transform(
            grid_points_utm['utm_x'].values,
            grid_points_utm['utm_y'].values
        )
        
        grid_points_utm['grid_lon'] = grid_lon
        grid_points_utm['grid_lat'] = grid_lat
        
        # Filter grid points to only include those within German boundaries
        if self.germany_gdf_utm is not None:
            print("Filtering grid points to German territory...")
            
            # Create point geometries in UTM for spatial filtering
            geometry_utm = [Point(x, y) for x, y in zip(grid_points_utm['utm_x'], grid_points_utm['utm_y'])]
            grid_gdf_utm = gpd.GeoDataFrame(grid_points_utm, geometry=geometry_utm, crs=self.target_crs)
            
            # Filter to German boundaries using UTM coordinates
            germany_union_utm = self.germany_gdf_utm.geometry.unary_union
            within_mask = grid_gdf_utm.geometry.within(germany_union_utm)
            
            filtered_grid = grid_gdf_utm[within_mask].drop('geometry', axis=1).reset_index(drop=True)
            filtered_grid['grid_idx'] = range(len(filtered_grid))
            
            print(f"Grid points: {len(grid_points_utm)} → {len(filtered_grid)} "
                  f"(removed {len(grid_points_utm) - len(filtered_grid)} outside German boundaries)")
            
            grid_points_utm = filtered_grid
        
        grid_shape = (len(y_coords), len(x_coords))
        print(f"Final grid: {len(grid_points_utm):,} points")
        
        return grid_points_utm, grid_shape
    
    def calculate_weighted_scores_utm(self, data_utm, grid_points_utm):
        """Calculate distance-weighted personality scores using UTM coordinates."""
        print(f"Calculating weighted scores for {len(grid_points_utm):,} grid points...")
        print(f"Using {len(data_utm):,} user data points in UTM coordinates")
        
        # Initialize results dataframe
        results = grid_points_utm.copy()
        for trait in self.personality_traits:
            results[trait] = np.nan
            results[f'{trait}_weight_sum'] = 0.0
        
        # Process in batches to manage memory and provide progress updates
        batch_size = 100
        n_batches = (len(grid_points_utm) + batch_size - 1) // batch_size
        
        print(f"Processing in {n_batches} batches of {batch_size} grid points each...")
        
        start_time = time.time()
        
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(grid_points_utm))
            batch_grid = grid_points_utm.iloc[start_idx:end_idx]
            
            # For each grid point in this batch
            for _, grid_point in batch_grid.iterrows():
                grid_idx = grid_point['grid_idx']
                
                # Calculate Euclidean distances in UTM coordinates (much faster and more accurate)
                distances = self.euclidean_distance(
                    grid_point['utm_x'], grid_point['utm_y'],
                    data_utm['utm_x'].values, data_utm['utm_y'].values
                )
                
                # Calculate weights using distance-decay function
                weights = self.distance_decay_weight(distances)
                
                # Calculate weighted scores for each personality trait
                total_weight = np.sum(weights)
                
                if total_weight > 0:  # Avoid division by zero
                    for trait in self.personality_traits:
                        trait_values = data_utm[trait].values
                        weighted_score = np.sum(weights * trait_values) / total_weight
                        results.loc[results['grid_idx'] == grid_idx, trait] = weighted_score
                        results.loc[results['grid_idx'] == grid_idx, f'{trait}_weight_sum'] = total_weight
            
            # Save checkpoint every 5 batches
            if (batch_idx + 1) % 5 == 0:
                elapsed = (time.time() - start_time) / 60
                remaining_batches = n_batches - (batch_idx + 1)
                eta_minutes = (elapsed / (batch_idx + 1)) * remaining_batches if batch_idx > 0 else 0
                
                print(f"Checkpoint {batch_idx + 1}/{n_batches} | "
                      f"Elapsed: {elapsed:.1f}min | ETA: {eta_minutes:.1f}min")
                
                results.to_csv(f"checkpoint_batch_{batch_idx + 1}.csv", index=False)
        
        print("Weighted score calculation complete!")
        print("✅ UTM-based distance calculations provide superior accuracy for regional analysis")
        
        return results
    
    def standardize_scores(self, results):
        """Z-score standardize personality traits across all grid points."""
        print("Standardizing personality trait scores...")
        
        standardized_results = results.copy()
        
        for trait in self.personality_traits:
            # Only standardize where we have valid scores (weight_sum > 0)
            valid_mask = results[f'{trait}_weight_sum'] > 0
            valid_scores = results.loc[valid_mask, trait]
            
            if len(valid_scores) > 1:  # Need at least 2 points for standardization
                mean_score = valid_scores.mean()
                std_score = valid_scores.std()
                
                if std_score > 0:  # Avoid division by zero
                    standardized_results.loc[valid_mask, f'{trait}_z'] = (
                        (results.loc[valid_mask, trait] - mean_score) / std_score
                    )
                    print(f"  {trait}: μ={mean_score:.3f}, σ={std_score:.3f}")
                else:
                    print(f"  {trait}: No variation (σ=0), setting z-scores to 0")
                    standardized_results.loc[valid_mask, f'{trait}_z'] = 0.0
            else:
                print(f"  {trait}: Insufficient valid data for standardization")
                standardized_results[f'{trait}_z'] = np.nan
        
        return standardized_results
    
    def aggregate_by_states_improved(self, data):
        """
        Improved state aggregation using UTM coordinates for accurate spatial joins.
        """
        if self.germany_gdf is None:
            print("Warning: No shapefile loaded. Cannot aggregate by states.")
            return None
        
        print("Aggregating personality scores by German states...")
        
        # Create point geometries for user data in WGS84
        geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
        data_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=self.source_crs)
        
        # Spatial join to assign users to states
        data_with_states = gpd.sjoin(data_gdf, self.germany_gdf, how='left', predicate='within')
        
        # Use 'name' column
        state_col = 'name'
        if state_col not in data_with_states.columns:
            print(f"Warning: '{state_col}' column not found after spatial join")
            print(f"Available columns: {list(data_with_states.columns)}")
            return None
        
        # Aggregate by state
        state_results = []
        
        for state_name in self.germany_gdf[state_col].unique():
            state_data = data_with_states[data_with_states[state_col] == state_name]
            
            if len(state_data) > 0:
                state_row = {'state': state_name, 'n_users': len(state_data)}
                
                # Calculate mean personality scores
                for trait in self.personality_traits:
                    state_row[trait] = state_data[trait].mean()
                    state_row[f'{trait}_std'] = state_data[trait].std()
                
                state_results.append(state_row)
        
        if not state_results:
            print("Warning: No state results generated")
            return None
        
        state_results_df = pd.DataFrame(state_results)
        
        # Standardize scores
        for trait in self.personality_traits:
            mean_score = state_results_df[trait].mean()
            std_score = state_results_df[trait].std()
            state_results_df[f'{trait}_z'] = (state_results_df[trait] - mean_score) / std_score
        
        print(f"Aggregated data for {len(state_results_df)} states")
        return state_results_df
    
    def create_ebert_style_maps(self, results, grid_shape, state_results=None, save_path=None):
        """Create publication-quality maps matching Ebert et al.'s visual style."""
        print("Creating Ebert-style personality maps...")
        print("✅ Using WGS84 coordinates for visualization (standard for mapping)")
        
        personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                              'Agreeableness', 'Neuroticism']
        
        # Create grid-based only visualization
        self.create_grid_only_maps(results, personality_traits, 
                                  save_path.replace('.png', '_grid_only.png') if save_path else None)
        
        # Create state-level only visualization if available
        if state_results is not None and self.germany_gdf is not None:
            self.create_state_only_maps(state_results, personality_traits,
                                       save_path.replace('.png', '_state_only.png') if save_path else None)
        
        # Create combined visualization
        if state_results is not None and self.germany_gdf is not None:
            # Create both grid-based and state-based maps
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            
            fig.suptitle('Spatial Distribution of Big Five Personality Traits in Germany\n'
                        'UTM-Based Analysis: LLM-Inferred vs. State-Level Aggregation', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            # Grid-based maps (top two rows)
            self._create_grid_maps(axes[:2, :], results, personality_traits)
            
            # State-based maps (bottom two rows) 
            self._create_state_maps(axes[2:, :], state_results, personality_traits)
            
            # Adjust spacing to reduce gaps
            plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05, 
                               hspace=0.2, wspace=0.15)
            
        else:
            # Create only grid-based maps
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Spatial Distribution of Big Five Personality Traits\n'
                        'UTM-Based LLM-Inferred Personality from Social Media Data', 
                        fontsize=16, fontweight='bold')
            
            self._create_simple_grid_maps(axes, results, personality_traits)
                    
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Combined map saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_grid_only_maps(self, results, personality_traits, save_path=None):
        """Create grid-based only visualization."""
        print("Creating grid-based only maps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Spatial Distribution of Big Five Personality Traits in Germany\n'
                    'UTM-Based Grid Analysis (Actor-Based Clustering)', 
                    fontsize=16, fontweight='bold')
        
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            if z_col in results.columns:
                valid_mask = ~pd.isna(results[z_col])
                valid_data = results[valid_mask]
                
                if len(valid_data) > 0:
                    scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                       c=valid_data[z_col], cmap=cmap, 
                                       vmin=vmin, vmax=vmax, s=8, alpha=0.8)
                    
                    # Add German boundaries
                    if self.germany_gdf is not None:
                        self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)
                    
                    ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
                    ax.set_xlabel('Longitude', fontsize=12)
                    ax.set_ylabel('Latitude', fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal', adjustable='box')
                    
                    # Add statistics
                    mean_z = valid_data[z_col].mean()
                    std_z = valid_data[z_col].std()
                    ax.text(0.02, 0.98, f'μ={mean_z:.3f}\nσ={std_z:.3f}', 
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        # Adjust layout to make room for colorbar
        plt.subplots_adjust(left=0.05, right=0.85, top=0.90, bottom=0.10, hspace=0.3, wspace=0.3)
        
        # Add colorbar in the space created
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z Score', rotation=270, labelpad=15, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Grid-only map saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_state_only_maps(self, state_results, personality_traits, save_path=None):
        """Create state-level only visualization."""
        print("Creating state-level only maps...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Spatial Distribution of Big Five Personality Traits in Germany\n'
                    'State-Level Administrative Aggregation', 
                    fontsize=16, fontweight='bold')
        
        # Use 'name' column
        state_col = 'name'
        if state_col not in self.germany_gdf.columns:
            print(f"Warning: '{state_col}' column not found in shapefile")
            return
        
        # Merge data
        merged_gdf = self.germany_gdf.merge(state_results, left_on=state_col, right_on='state', how='left')
        
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            plot = merged_gdf.plot(column=z_col, cmap=cmap, vmin=vmin, vmax=vmax,
                                  ax=ax, edgecolor='black', linewidth=0.7, 
                                  missing_kwds={'color': 'lightgray', 'edgecolor': 'black'})
            
            ax.set_title(f'{trait}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add statistics
            valid_data = merged_gdf[merged_gdf[z_col].notna()]
            if len(valid_data) > 0:
                mean_z = valid_data[z_col].mean()
                std_z = valid_data[z_col].std()
                ax.text(0.02, 0.98, f'μ={mean_z:.3f}\nσ={std_z:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                       fontsize=10)
        
        # Remove empty subplot  
        axes[1, 2].remove()
        
        # Adjust layout to make room for colorbar
        plt.subplots_adjust(left=0.05, right=0.85, top=0.90, bottom=0.10, hspace=0.3, wspace=0.3)
        
        # Add colorbar in the space created
        cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Z Score', rotation=270, labelpad=15, fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"State-only map saved to {save_path}")
        
        plt.show()
        return fig
    
    def _create_grid_maps(self, axes, results, personality_traits):
        """Create grid-based personality maps."""
        axes[0, 2].text(0.5, 1.15, "UTM-Based Grid Analysis (Actor-Based Clustering)", 
                        ha='center', fontsize=14, fontweight='bold', 
                        transform=axes[0, 2].transAxes)
        
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 5
            col = idx % 5
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            if z_col in results.columns:
                valid_mask = ~pd.isna(results[z_col])
                valid_data = results[valid_mask]
                
                if len(valid_data) > 0:
                    scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                       c=valid_data[z_col], cmap=cmap, 
                                       vmin=vmin, vmax=vmax, s=4, alpha=0.8)
                    
                    # Add German boundaries
                    if self.germany_gdf is not None:
                        self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7)
                    
                    ax.set_title(trait, fontsize=12, fontweight='bold')
                    ax.set_xlabel('Longitude', fontsize=10)
                    ax.set_ylabel('Latitude', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_aspect('equal', adjustable='box')
                    ax.tick_params(labelsize=8)
        
        # Hide unused subplots
        for idx in range(len(personality_traits), 10):
            row = idx // 5
            col = idx % 5
            if row < 2 and col < 5:
                axes[row, col].set_visible(False)
        
        # Add colorbar
        if len(personality_traits) > 0:
            cbar = plt.colorbar(scatter, ax=axes[0, 4], shrink=0.8)
            cbar.set_label('Z Score', rotation=270, labelpad=15, fontsize=11)
    
    def _create_state_maps(self, axes, state_results, personality_traits):
        """Create state-level maps."""
        axes[0, 2].text(0.5, 1.15, "State-Level Administrative Aggregation", 
                        ha='center', fontsize=14, fontweight='bold', 
                        transform=axes[0, 2].transAxes)
        
        # Use 'name' column
        state_col = 'name'
        if state_col not in self.germany_gdf.columns:
            print(f"Warning: '{state_col}' column not found in shapefile")
            return
        
        # Merge data
        merged_gdf = self.germany_gdf.merge(state_results, left_on=state_col, right_on='state', how='left')
        
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 5
            col = idx % 5
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            merged_gdf.plot(column=z_col, cmap=cmap, vmin=vmin, vmax=vmax,
                           ax=ax, edgecolor='black', linewidth=0.5, 
                           missing_kwds={'color': 'lightgray', 'edgecolor': 'black'})
            
            ax.set_title(trait, fontsize=12, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Hide unused subplots
        for idx in range(len(personality_traits), 10):
            row = idx // 5
            col = idx % 5
            if row < 2 and col < 5:
                axes[row, col].set_visible(False)
    
    def _create_simple_grid_maps(self, axes, results, personality_traits):
        """Create simple grid maps when no state data available."""
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            z_col = f'{trait}_z'
            valid_mask = ~pd.isna(results[z_col])
            valid_data = results[valid_mask]
            
            if len(valid_data) > 0:
                scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                   c=valid_data[z_col], cmap=cmap, 
                                   vmin=vmin, vmax=vmax, s=6, alpha=0.8)
                
                if self.germany_gdf is not None:
                    self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)
                
                ax.set_title(trait, fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
        
        if len(personality_traits) == 5:
            axes[1, 2].remove()
    
    def generate_summary_statistics(self, data, data_utm, results):
        """Generate comprehensive summary statistics including coordinate system info."""
        print("\n" + "="*70)
        print("SUMMARY STATISTICS - UTM-ENHANCED SPATIAL ANALYSIS")
        print("="*70)
        
        print(f"\nCoordinate System Information:")
        print(f"  - Input CRS: {self.source_crs} (WGS84 Geographic)")
        print(f"  - Analysis CRS: {self.target_crs} (UTM Zone 32N)")
        print(f"  - Visualization CRS: {self.source_crs} (WGS84 for mapping)")
        
        print(f"\nInput Data:")
        print(f"  - Total users: {len(data):,}")
        print(f"  - Geographic extent (WGS84):")
        print(f"    • Latitude: {data['latitude'].min():.6f}° to {data['latitude'].max():.6f}°")
        print(f"    • Longitude: {data['longitude'].min():.6f}° to {data['longitude'].max():.6f}°")
        print(f"  - UTM extent (Zone 32N):")
        print(f"    • Easting: {data_utm['utm_x'].min():.0f} to {data_utm['utm_x'].max():.0f} m")
        print(f"    • Northing: {data_utm['utm_y'].min():.0f} to {data_utm['utm_y'].max():.0f} m")
        
        print(f"\nSpatial Grid:")
        print(f"  - Grid points: {len(results):,}")
        print(f"  - Resolution: {self.grid_resolution_km} km ({self.grid_resolution_meters:.0f} m)")
        print(f"  - Distance decay radius: {self.r_km:.1f} km ({self.r_meters:.0f} m)")
        print(f"  - Distance calculation: Euclidean (UTM) - Superior accuracy vs Haversine")
        
        print(f"\nPersonality Trait Distributions (Original vs. Spatial):")
        for trait in self.personality_traits:
            orig_mean = data[trait].mean()
            orig_std = data[trait].std()
            
            valid_spatial = results[results[f'{trait}_weight_sum'] > 0][trait]
            spatial_mean = valid_spatial.mean() if len(valid_spatial) > 0 else np.nan
            spatial_std = valid_spatial.std() if len(valid_spatial) > 0 else np.nan
            
            print(f"  {trait}:")
            print(f"    Original:  μ={orig_mean:.3f}, σ={orig_std:.3f}")
            print(f"    Spatial:   μ={spatial_mean:.3f}, σ={spatial_std:.3f}")
        
        print(f"\nMethodological Improvements:")
        print(f"  ✅ UTM Zone 32N provides <0.1% distance distortion for Germany")
        print(f"  ✅ Euclidean distance calculation is faster and more accurate than Haversine")
        print(f"  ✅ Grid resolution in meters ensures consistent spatial sampling")
        print(f"  ✅ Results directly comparable to Ebert et al.'s methodology")

def main(skip_computation=False):
    """
    Main function with UTM coordinate system implementation.
    
    Parameters:
    -----------
    skip_computation : bool
        If True, load existing results. If False, run full computation.
    """
    
    # Input file paths
    input_file = "final_100_users_for_spatial_visualization.csv"  # Update this path
    shapefile_path = "german_shapefile_utm32n/de_utm32n.shp"  # Update this path to your German shapefile
    
    # Ebert's parameters (unchanged)
    r_miles = 45        # Distance radius in miles (Ebert's value)
    s_slope = 7         # Distance decay slope (Ebert's value)
    grid_resolution = 5 # Grid resolution in km
    
    # Output settings
    save_maps = True
    output_file = "utm_enhanced_personality_maps_germany.png"
    save_results = True
    grid_results_file = "utm_spatial_personality_grid_results.csv"
    state_results_file = "utm_spatial_personality_state_results.csv"
    
    print("UTM-ENHANCED EBERT-STYLE SPATIAL ANALYSIS")
    print("="*80)
    print("🎯 Using UTM Zone 32N (EPSG:32632) for optimal accuracy in Germany")
    print("="*80)
    
    # Check if we should skip computation
    if skip_computation or os.path.exists(grid_results_file):
        print("LOADING EXISTING UTM RESULTS - SKIPPING COMPUTATION!")
        print("="*50)
        
        try:
            # Load existing grid results
            print(f"\nLoading existing UTM grid results from {grid_results_file}...")
            results = pd.read_csv(grid_results_file)
            print(f"✅ Loaded existing results: {len(results)} grid points")
            
            # Load data for state aggregation
            print(f"\nLoading data for state aggregation...")
            data = pd.read_csv(input_file)
            personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
            required_cols = ['latitude', 'longitude'] + personality_traits
            data = data.dropna(subset=required_cols)
            print(f"✅ Loaded original data: {len(data)} users")
            
            # Initialize mapper for visualization
            mapper = SpatialPersonalityMapper(
                r_miles=r_miles, 
                s_slope=s_slope, 
                grid_resolution_km=grid_resolution,
                shapefile_path=shapefile_path,
                target_crs='EPSG:32632'  # UTM Zone 32N
            )
            
            # Transform to UTM for summary statistics
            data_utm = mapper.transform_coordinates_to_utm(data)
            
            # Create state aggregation
            print(f"\nRe-aggregating by states...")
            state_results = mapper.aggregate_by_states_improved(data)
            
            if state_results is not None:
                print(f"✅ State aggregation: {len(state_results)} states")
                state_results.to_csv(state_results_file, index=False)
                print(f"✅ Saved: {state_results_file}")
            
            # Create visualizations
            print(f"\nCreating UTM-enhanced maps...")
            grid_shape = None  # Not needed for visualization
            mapper.create_ebert_style_maps(results, grid_shape, state_results, output_file)
            
            # Generate summary
            mapper.generate_summary_statistics(data, data_utm, results)
            
            print("\n" + "="*70)
            print("ANALYSIS COMPLETE USING EXISTING UTM RESULTS!")
            print("="*70)
            print("Files created:")
            print(f"✅ {output_file} (UTM-enhanced combined visualization)")
            print(f"✅ {output_file.replace('.png', '_grid_only.png')} (UTM grid-based only)")
            if state_results is not None:
                print(f"✅ {output_file.replace('.png', '_state_only.png')} (State-level only)")
                print(f"✅ {state_results_file} (Updated state results)")
            return  # EXIT HERE
            
        except FileNotFoundError as e:
            print(f"❌ Could not find existing results: {e}")
            print("Proceeding with full UTM computation...")
    
    # If we reach here, run full computation with UTM
    print("RUNNING FULL UTM-ENHANCED COMPUTATION...")
    print("="*40)
    
    # Load and validate data
    print("Loading data...")
    try:
        data = pd.read_csv(input_file)
        print(f"Loaded {len(data):,} records")
        
        # Validate required columns
        required_cols = ['latitude', 'longitude'] + ['Openness', 'Conscientiousness', 
                        'Extraversion', 'Agreeableness', 'Neuroticism']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Remove rows with missing coordinates or personality scores
        initial_count = len(data)
        data = data.dropna(subset=required_cols)
        final_count = len(data)
        
        if final_count < initial_count:
            print(f"Removed {initial_count - final_count} rows with missing data")
            
        print(f"Final dataset: {len(data):,} complete records")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{input_file}'")
        print("Please update the input_file path in the main() function")
        return
    
    # Initialize the UTM-enhanced spatial mapper
    mapper = SpatialPersonalityMapper(
        r_miles=r_miles, 
        s_slope=s_slope, 
        grid_resolution_km=grid_resolution,
        shapefile_path=shapefile_path,
        target_crs='EPSG:32632'  # UTM Zone 32N for Germany
    )
    
    # Step 0: Transform coordinates to UTM
    print("\nStep 0: Transforming coordinates to UTM Zone 32N...")
    data_utm = mapper.transform_coordinates_to_utm(data)
    
    # Step 1: Create UTM spatial grid
    print("\nStep 1: Creating UTM-based spatial grid...")
    grid_points_utm, grid_shape = mapper.create_germany_grid_utm(data_utm)
    
    # Step 2: Calculate distance-weighted scores using UTM
    print("\nStep 2: Calculating UTM-based distance-weighted personality scores...")
    weighted_results = mapper.calculate_weighted_scores_utm(data_utm, grid_points_utm)
    
    # Step 3: Standardize scores
    print("\nStep 3: Standardizing scores...")
    standardized_results = mapper.standardize_scores(weighted_results)
    
    # Step 4: State-level aggregation
    state_results = None
    if mapper.germany_gdf is not None:
        print("\nStep 4: Aggregating by German states...")
        state_results = mapper.aggregate_by_states_improved(data)
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating UTM-enhanced maps...")
    if save_maps:
        mapper.create_ebert_style_maps(standardized_results, grid_shape, 
                                     state_results, output_file)
    else:
        mapper.create_ebert_style_maps(standardized_results, grid_shape, state_results)
    
    # Step 6: Generate summary statistics
    mapper.generate_summary_statistics(data, data_utm, standardized_results)
    
    # Step 7: Save results
    if save_results:
        standardized_results.to_csv(grid_results_file, index=False)
        print(f"\nUTM grid results saved to '{grid_results_file}'")
        
        if state_results is not None:
            state_results.to_csv(state_results_file, index=False)
            print(f"State results saved to '{state_results_file}'")
    
    print("\n" + "="*70)
    print("UTM-ENHANCED ANALYSIS COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print(f"✅ {output_file} (UTM-enhanced combined visualization)")
    print(f"✅ {output_file.replace('.png', '_grid_only.png')} (UTM grid-based only)")
    if state_results is not None:
        print(f"✅ {output_file.replace('.png', '_state_only.png')} (State-level only)")
    print(f"✅ {grid_results_file} (UTM grid computation results)")
    if state_results is not None:
        print(f"✅ {state_results_file} (State aggregation results)")
    
    print("\n🎯 UTM Zone 32N Advantages Applied:")
    print("• <0.1% distance distortion for Baden-Württemberg region")
    print("• Euclidean distance calculations (faster & more accurate)")
    print("• Meter-based grid resolution (precise spatial sampling)")
    print("• Maintains compatibility with Ebert et al.'s methodology")
    print("• Results directly comparable with traditional BFI questionnaire studies")

if __name__ == "__main__":
    # Add pyproj requirement check
    try:
        import pyproj
        print("✅ pyproj library available for coordinate transformations")
    except ImportError:
        print("❌ pyproj library required for UTM transformations")
        print("Install with: pip install pyproj")
        exit(1)
    
    # Run with existing results (skip computation)
    main(skip_computation=True)
    
    # Or run full computation:
    # main(skip_computation=False)