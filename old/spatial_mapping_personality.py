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

class SpatialPersonalityMapper:
    """
    Implements Ebert et al.'s spatial mapping methodology for personality traits.
    Updated with skip computation logic and improved shapefile handling.
    """
    
    def __init__(self, r_miles=45, s_slope=7, grid_resolution_km=5, shapefile_path=None):
        """Initialize the spatial mapper with Ebert's parameters."""
        self.r_miles = r_miles
        self.s_slope = s_slope
        self.r_km = r_miles * 1.60934  # Convert miles to km
        self.grid_resolution_km = grid_resolution_km
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        self.germany_gdf = None
        
        # Load German shapefile if provided
        if shapefile_path:
            self.load_german_shapefile(shapefile_path)
        
        print(f"Spatial Personality Mapper initialized:")
        print(f"  - Distance radius: {r_miles} miles ({self.r_km:.1f} km)")
        print(f"  - Distance decay slope: {s_slope}")
        print(f"  - Grid resolution: {grid_resolution_km} km")
        print(f"  - German shapefile: {'Loaded' if self.germany_gdf is not None else 'Not provided'}")
    
    def load_german_shapefile(self, shapefile_path):
        """Load German shapefile with improved error handling."""
        try:
            self.germany_gdf = gpd.read_file(shapefile_path)
            if self.germany_gdf.crs != 'EPSG:4326':
                print(f"Converting shapefile from {self.germany_gdf.crs} to EPSG:4326")
                self.germany_gdf = self.germany_gdf.to_crs('EPSG:4326')
            
            print(f"✅ Loaded German shapefile: {len(self.germany_gdf)} regions")
            
            # Check for 'name' column (updated from update_shapefile.py)
            if 'name' in self.germany_gdf.columns:
                states = self.germany_gdf['name'].unique()
                print(f"States in shapefile ({len(states)}): {len(states)} states found")
            else:
                print(f"Warning: 'name' column not found in shapefile")
                print(f"Available columns: {list(self.germany_gdf.columns)}")
                
        except Exception as e:
            print(f"❌ Error loading shapefile: {e}")
            self.germany_gdf = None
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate Haversine distance between two points in kilometers."""
        R = 6371  # Earth's radius in kilometers
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def distance_decay_weight(self, distances_km):
        """Calculate distance decay weights using Ebert's log-logistic function."""
        # Convert distances to miles for Ebert's formula
        distances_miles = distances_km / 1.60934
        
        # Ebert's log-logistic distance decay: f(d) = 1 / (1 + (d/r)^s)
        weights = 1 / (1 + (distances_miles / self.r_miles) ** self.s_slope)
        return weights
    
    def create_germany_grid(self, data):
        """Create spatial grid clipped to German boundaries."""
        # Get geographic bounds with small buffer
        lat_min, lat_max = data['latitude'].min() - 0.1, data['latitude'].max() + 0.1
        lon_min, lon_max = data['longitude'].min() - 0.1, data['longitude'].max() + 0.1
        
        # If shapefile available, use its bounds instead
        if self.germany_gdf is not None:
            bounds = self.germany_gdf.total_bounds  # [minx, miny, maxx, maxy]
            lon_min, lat_min, lon_max, lat_max = bounds
            print(f"Using shapefile bounds for grid creation")
        
        print(f"Creating grid for bounds:")
        print(f"  Latitude: {lat_min:.3f} to {lat_max:.3f}")
        print(f"  Longitude: {lon_min:.3f} to {lon_max:.3f}")
        
        # Convert grid resolution from km to degrees (approximately)
        lat_step = self.grid_resolution_km / 111.0
        lon_step = self.grid_resolution_km / (111.0 * np.cos(np.radians(np.mean([lat_min, lat_max]))))
        
        # Create grid
        lats = np.arange(lat_min, lat_max + lat_step, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step, lon_step)
        
        # Create meshgrid and flatten
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        grid_points = pd.DataFrame({
            'grid_lat': lat_grid.flatten(),
            'grid_lon': lon_grid.flatten(),
            'grid_idx': range(len(lat_grid.flatten()))
        })
        
        # Filter grid points to only include those within German boundaries
        if self.germany_gdf is not None:
            print("Filtering grid points to German territory...")
            
            # Create point geometries
            geometry = []
            for _, row in tqdm(grid_points.iterrows(), 
                             total=len(grid_points), 
                             desc="Creating point geometries"):
                geometry.append(Point(row['grid_lon'], row['grid_lat']))
            
            grid_gdf = gpd.GeoDataFrame(grid_points, geometry=geometry, crs='EPSG:4326')
            
            # Filter to German boundaries
            germany_union = self.germany_gdf.geometry.unary_union
            within_mask = grid_gdf.geometry.within(germany_union)
            
            filtered_grid = grid_gdf[within_mask].drop('geometry', axis=1).reset_index(drop=True)
            filtered_grid['grid_idx'] = range(len(filtered_grid))
            
            print(f"Grid points: {len(grid_points)} → {len(filtered_grid)} "
                  f"(removed {len(grid_points) - len(filtered_grid)} outside German boundaries)")
            
            grid_points = filtered_grid
        
        grid_shape = (len(lats), len(lons))
        print(f"Final grid: {len(grid_points):,} points")
        
        return grid_points, grid_shape
    
    def calculate_weighted_scores(self, data, grid_points):
        """Calculate distance-weighted personality scores for each grid point."""
        print(f"Calculating weighted scores for {len(grid_points):,} grid points...")
        print(f"Using {len(data):,} user data points")
        
        # Initialize results dataframe
        results = grid_points.copy()
        for trait in self.personality_traits:
            results[trait] = np.nan
            results[f'{trait}_weight_sum'] = 0.0
        
        # Process in batches to manage memory and provide progress updates
        batch_size = 100
        n_batches = (len(grid_points) + batch_size - 1) // batch_size
        
        print(f"Processing in {n_batches} batches of {batch_size} grid points each...")
        
        start_time = time.time()
        
        for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(grid_points))
            batch_grid = grid_points.iloc[start_idx:end_idx]
            
            # For each grid point in this batch
            for _, grid_point in batch_grid.iterrows():
                grid_idx = grid_point['grid_idx']
                
                # Calculate distances from this grid point to all users
                distances = []
                for _, user in data.iterrows():
                    dist = self.haversine_distance(
                        grid_point['grid_lat'], grid_point['grid_lon'],
                        user['latitude'], user['longitude']
                    )
                    distances.append(dist)
                
                distances = np.array(distances)
                
                # Calculate weights using distance-decay function
                weights = self.distance_decay_weight(distances)
                
                # Calculate weighted scores for each personality trait
                total_weight = np.sum(weights)
                
                if total_weight > 0:  # Avoid division by zero
                    for trait in self.personality_traits:
                        trait_values = data[trait].values
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
        Improved state aggregation using 'name' column from update_shapefile.py logic.
        """
        if self.germany_gdf is None:
            print("Warning: No shapefile loaded. Cannot aggregate by states.")
            return None
        
        print("Aggregating personality scores by German states...")
        
        # Create point geometries for user data
        geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
        data_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
        
        # Spatial join to assign users to states
        data_with_states = gpd.sjoin(data_gdf, self.germany_gdf, how='left', predicate='within')
        
        # Use 'name' column (updated from update_shapefile.py)
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
                        'LLM-Inferred vs. State-Level Aggregation', 
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
                        'LLM-Inferred Personality from Social Media Data', 
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
                    'Grid-Based Actor-Based Clustering (LLM-Inferred)', 
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
                    'State-Level Administrative Aggregation (LLM-Inferred)', 
                    fontsize=16, fontweight='bold')
        
        # Use 'name' column (updated from update_shapefile.py)
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
        axes[0, 2].text(0.5, 1.15, "Grid-Based (Actor-Based Clustering)", 
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
        axes[0, 2].text(0.5, 1.15, "State-Level Aggregation", 
                        ha='center', fontsize=14, fontweight='bold', 
                        transform=axes[0, 2].transAxes)
        
        # Use 'name' column (updated from update_shapefile.py)
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
    
    def generate_summary_statistics(self, data, results):
        """Generate comprehensive summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nInput Data:")
        print(f"  - Total users: {len(data):,}")
        print(f"  - Geographic extent: {data['latitude'].min():.3f}° to {data['latitude'].max():.3f}° lat")
        print(f"                      {data['longitude'].min():.3f}° to {data['longitude'].max():.3f}° lon")
        
        print(f"\nSpatial Grid:")
        print(f"  - Grid points: {len(results):,}")
        print(f"  - Resolution: {self.grid_resolution_km} km")
        print(f"  - Distance decay radius: {self.r_km:.1f} km")
        
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

def main(skip_computation=False):
    """
    Main function with skip computation logic from update_shapefile.py.
    
    Parameters:
    -----------
    skip_computation : bool
        If True, load existing results. If False, run full computation.
    """
    
    # Input file paths
    input_file = "final_users_for_spatial_visualization.csv"  # Update this path
    shapefile_path = "german_shapefile/de.shp"  # Update this path to your German shapefile
    
    # Ebert's parameters
    r_miles = 45        # Distance radius in miles (Ebert's value)
    s_slope = 7         # Distance decay slope (Ebert's value)
    grid_resolution = 5 # Grid resolution in km
    
    # Output settings
    save_maps = True
    output_file = "ebert_style_personality_maps_germany.png"
    save_results = True
    grid_results_file = "spatial_personality_grid_results.csv"
    state_results_file = "spatial_personality_state_results.csv"
    
    print("EBERT-STYLE SPATIAL ANALYSIS WITH SKIP COMPUTATION LOGIC")
    print("="*70)
    
    # Check if we should skip computation (from update_shapefile.py logic)
    if skip_computation or os.path.exists(grid_results_file):
        print("LOADING EXISTING RESULTS - SKIPPING 5+ HOUR COMPUTATION!")
        print("="*50)
        print("This will use your existing computed results - no recomputation needed!")
        
        try:
            # Load existing grid results
            print(f"\nLoading existing grid results from {grid_results_file}...")
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
                shapefile_path=shapefile_path
            )
            
            # Create state aggregation with improved logic
            print(f"\nRe-aggregating by states with improved logic...")
            state_results = mapper.aggregate_by_states_improved(data)
            
            if state_results is not None:
                print(f"✅ State aggregation: {len(state_results)} states")
                state_results.to_csv(state_results_file, index=False)
                print(f"✅ Saved: {state_results_file}")
            
            # Create visualizations
            print(f"\nCreating maps with existing results...")
            grid_shape = None  # Not needed for visualization
            mapper.create_ebert_style_maps(results, grid_shape, state_results, output_file)
            
            # Generate summary
            mapper.generate_summary_statistics(data, results)
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE USING EXISTING RESULTS!")
            print("="*60)
            print("Files created:")
            print(f"✅ {output_file} (Combined visualization)")
            print(f"✅ {output_file.replace('.png', '_grid_only.png')} (Grid-based only)")
            if state_results is not None:
                print(f"✅ {output_file.replace('.png', '_state_only.png')} (State-level only)")
                print(f"✅ {state_results_file} (Updated state results)")
            return  # EXIT HERE - don't continue to full computation
            
        except FileNotFoundError as e:
            print(f"❌ Could not find existing results: {e}")
            print("Proceeding with full computation...")
            # Continue to full computation below
    
    # If we reach here, run full computation
    print("RUNNING FULL COMPUTATION...")
    print("="*30)
    
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
    
    # Initialize the spatial mapper with shapefile
    mapper = SpatialPersonalityMapper(
        r_miles=r_miles, 
        s_slope=s_slope, 
        grid_resolution_km=grid_resolution,
        shapefile_path=shapefile_path
    )
    
    # Step 1: Create spatial grid (clipped to German boundaries)
    print("\nStep 1: Creating spatial grid...")
    grid_points, grid_shape = mapper.create_germany_grid(data)
    
    # Step 2: Calculate distance-weighted scores
    print("\nStep 2: Calculating distance-weighted personality scores...")
    weighted_results = mapper.calculate_weighted_scores(data, grid_points)
    
    # Step 3: Standardize scores
    print("\nStep 3: Standardizing scores...")
    standardized_results = mapper.standardize_scores(weighted_results)
    
    # Step 4: State-level aggregation (improved)
    state_results = None
    if mapper.germany_gdf is not None:
        print("\nStep 4: Aggregating by German states...")
        state_results = mapper.aggregate_by_states_improved(data)
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating maps...")
    if save_maps:
        mapper.create_ebert_style_maps(standardized_results, grid_shape, 
                                     state_results, output_file)
    else:
        mapper.create_ebert_style_maps(standardized_results, grid_shape, state_results)
    
    # Step 6: Generate summary statistics
    mapper.generate_summary_statistics(data, standardized_results)
    
    # Step 7: Save results
    if save_results:
        standardized_results.to_csv(grid_results_file, index=False)
        print(f"\nGrid results saved to '{grid_results_file}'")
        
        if state_results is not None:
            state_results.to_csv(state_results_file, index=False)
            print(f"State results saved to '{state_results_file}'")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nFiles created:")
    print(f"✅ {output_file} (Combined grid + state visualization)")
    print(f"✅ {output_file.replace('.png', '_grid_only.png')} (Grid-based only visualization)")
    if state_results is not None:
        print(f"✅ {output_file.replace('.png', '_state_only.png')} (State-level only visualization)")
    print(f"✅ {grid_results_file} (Grid computation results)")
    if state_results is not None:
        print(f"✅ {state_results_file} (State aggregation results)")
    print("\nVisualization types:")
    print("1. Grid-based continuous maps (actor-based clustering)")
    print("2. State-level choropleth maps (administrative aggregation)")
    print("3. Combined comparison view")
    print("\nThese results can be directly compared with Ebert et al.'s")
    print("traditional BFI questionnaire approach using identical methodology.")

if __name__ == "__main__":
    # Run with existing results (skip 5+ hour computation)
    main(skip_computation=True)
    
    # Or run full computation:
    # main(skip_computation=False)