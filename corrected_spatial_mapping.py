import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SpatialPersonalityMapper:
    """
    Implements Ebert et al.'s actor-based clustering approach for mapping
    personality traits across geographic space using distance-decay weighting.
    """
    
    def __init__(self, r_miles=45, s_slope=7, grid_resolution_km=5, shapefile_path=None):
        """
        Initialize the spatial mapper with Ebert's parameters.
        
        Parameters:
        -----------
        r_miles : float
            Distance (in miles) where decay function reaches 0.5 weight
            Ebert used 45 miles (72.4 km)
        s_slope : float
            Slope parameter for distance decay function
            Ebert used s=7
        grid_resolution_km : float
            Resolution of output grid in kilometers
        shapefile_path : str
            Path to German states shapefile (optional but recommended)
        """
        self.r_km = r_miles * 1.60934  # Convert miles to kilometers
        self.s = s_slope
        self.grid_resolution_km = grid_resolution_km
        self.shapefile_path = shapefile_path
        self.germany_gdf = None
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        
        # Load German shapefile if provided
        if shapefile_path:
            self.load_german_shapefile()
        
        print(f"Initialized with Ebert parameters:")
        print(f"  - Distance decay radius: {r_miles} miles ({self.r_km:.1f} km)")
        print(f"  - Slope parameter: {s_slope}")
        print(f"  - Grid resolution: {grid_resolution_km} km")
        if self.germany_gdf is not None:
            print(f"  - German shapefile loaded: {len(self.germany_gdf)} states/regions")
    
    def load_german_shapefile(self):
        """
        Load German administrative boundaries shapefile.
        
        Expected shapefile should contain German states (Länder) boundaries.
        Common sources: 
        - Bundesamt für Kartographie und Geodäsie (BKG)
        - Natural Earth
        - GADM database
        """
        try:
            self.germany_gdf = gpd.read_file(self.shapefile_path)
            
            # Ensure CRS is WGS84 for consistency with lat/lon data
            if self.germany_gdf.crs != 'EPSG:4326':
                print(f"Converting shapefile from {self.germany_gdf.crs} to EPSG:4326")
                self.germany_gdf = self.germany_gdf.to_crs('EPSG:4326')
            
            print(f"Loaded German shapefile with {len(self.germany_gdf)} regions")
            
            # Print shapefile info for debugging
            print("Shapefile columns:", list(self.germany_gdf.columns))
            if 'NAME' in self.germany_gdf.columns:
                print("States/regions:", self.germany_gdf['NAME'].tolist())
            elif 'GEN' in self.germany_gdf.columns:  # Common German shapefile naming
                print("States/regions:", self.germany_gdf['GEN'].tolist())
                
        except Exception as e:
            print(f"Warning: Could not load shapefile from {self.shapefile_path}")
            print(f"Error: {e}")
            print("Continuing without shapefile boundaries...")
            self.germany_gdf = None
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between two points on Earth.
        
        This is the same distance calculation used by Ebert et al.
        Returns distance in kilometers.
        """
        # Convert coordinates to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth's radius in kilometers
        R = 6371.0
        return R * c
    
    def distance_decay_weight(self, distance_km):
        """
        Calculate distance-decay weights using Ebert's log-logistic function.
        
        Formula: f(d) = 1 / (1 + (d/r)^s)
        
        Where:
        - d = distance in kilometers
        - r = distance where weight = 0.5 (72.4 km for Ebert's 45 miles)
        - s = slope parameter (7 for Ebert)
        
        Weight interpretation:
        - Distance ≤ 48 km (30 miles): weight ≈ 1.0
        - Distance = 72 km (45 miles): weight = 0.5
        - Distance ≥ 121 km (75 miles): weight ≈ 0.0
        """
        return 1 / (1 + (distance_km / self.r_km) ** self.s)
    
    def create_germany_grid(self, data):
        """
        Create a regular grid covering Germany, optionally clipped to actual boundaries.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with 'latitude' and 'longitude' columns
            
        Returns:
        --------
        grid_points : pandas.DataFrame
            DataFrame with grid coordinates and indices
        """
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
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
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
            
            # Spatial join to keep only points within Germany
            germany_union = self.germany_gdf.geometry.unary_union
            
            within_mask = []
            for geom in tqdm(grid_gdf.geometry, desc="Checking German boundaries"):
                within_mask.append(geom.within(germany_union))
            
            grid_points_filtered = grid_gdf[within_mask]
            
            # Reset index and grid_idx
            grid_points = grid_points_filtered.drop('geometry', axis=1).reset_index(drop=True)
            grid_points['grid_idx'] = range(len(grid_points))
            
            print(f"Filtered from {len(grid_gdf)} to {len(grid_points)} points within Germany")
        
        print(f"Final grid with {len(grid_points)} points")
        
        return grid_points, (len(lats), len(lons))
    
    def calculate_weighted_scores(self, data, grid_points, batch_size=1000):
        """
        Calculate distance-weighted personality scores for each grid point.
        
        This is the core of Ebert's actor-based clustering approach.
        For each grid point, we calculate a weighted average of ALL users' 
        personality scores, where weights decrease with distance.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            User data with coordinates and personality scores
        grid_points : pandas.DataFrame
            Grid coordinates
        batch_size : int
            Process grid in batches to manage memory
            
        Returns:
        --------
        results : pandas.DataFrame
            Grid points with weighted personality scores
        """
        print(f"Calculating weighted scores for {len(grid_points)} grid points...")
        print(f"Using {len(data)} user data points")
        
        # Initialize results
        results = grid_points.copy()
        for trait in self.personality_traits:
            results[trait] = 0.0
            results[f'{trait}_weight_sum'] = 0.0
        
        # Process in batches to manage memory
        n_batches = (len(grid_points) + batch_size - 1) // batch_size
        
        print(f"Processing {n_batches} batches (batch size: {batch_size})")
        
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
        """
        Z-score standardize personality traits across all grid points.
        
        This follows Ebert's approach of standardizing scores before mapping
        to ensure comparable scales across traits.
        """
        print("Standardizing personality scores...")
        
        standardized_results = results.copy()
        
        for trait in self.personality_traits:
            # Extract valid scores (where weight_sum > 0)
            valid_mask = results[f'{trait}_weight_sum'] > 0
            valid_scores = results.loc[valid_mask, trait]
            
            if len(valid_scores) > 1:
                # Calculate z-scores
                mean_score = valid_scores.mean()
                std_score = valid_scores.std()
                
                # Standardize
                standardized_results.loc[valid_mask, f'{trait}_z'] = (
                    (valid_scores - mean_score) / std_score
                )
                
                print(f"{trait}: mean={mean_score:.3f}, std={std_score:.3f}, "
                      f"n_valid={len(valid_scores)}")
            else:
                standardized_results[f'{trait}_z'] = np.nan
        
        return standardized_results
    
    def aggregate_by_states(self, data):
        """
        Aggregate personality scores by German states using shapefile boundaries.
        
        This provides an alternative view similar to Ebert's state-level analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            User data with coordinates and personality scores
            
        Returns:
        --------
        state_results : pandas.DataFrame
            State-level aggregated personality scores
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
        
        # Identify state name column (varies by shapefile source)
        state_col = None
        for col in ['NAME', 'GEN', 'NAME_1', 'VARNAME_1']:
            if col in data_with_states.columns:
                state_col = col
                break
        
        if state_col is None:
            print("Warning: Could not identify state name column in shapefile")
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
        
        state_results_df = pd.DataFrame(state_results)
        
        # Standardize scores
        for trait in self.personality_traits:
            mean_score = state_results_df[trait].mean()
            std_score = state_results_df[trait].std()
            state_results_df[f'{trait}_z'] = (state_results_df[trait] - mean_score) / std_score
        
        print(f"Aggregated data for {len(state_results_df)} states")
        return state_results_df
    
    def create_ebert_style_maps(self, results, grid_shape, state_results=None, save_path=None):
        """
        Create publication-quality maps matching Ebert et al.'s visual style.
        
        Creates both grid-based continuous maps and state-level choropleth maps
        if shapefile is available.
        """
        print("Creating Ebert-style personality maps...")
        
        personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                              'Agreeableness', 'Neuroticism']
        
        if state_results is not None and self.germany_gdf is not None:
            # Create both grid-based and state-based maps
            fig, axes = plt.subplots(4, 5, figsize=(20, 16))
            
            fig.suptitle('Spatial Distribution of Big Five Personality Traits in Germany\n'
                        'LLM-Inferred vs. State-Level Aggregation', 
                        fontsize=16, fontweight='bold', y=0.98)
            
            # Grid-based maps (top two rows)
            self._create_grid_maps(axes[:2, :], results, personality_traits)
            
            # State-based maps (bottom two rows) 
            self._create_state_maps(axes[2:, :], state_results, personality_traits)
            
        else:
            # Create only grid-based maps
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Spatial Distribution of Big Five Personality Traits\n'
                        'LLM-Inferred Personality from Social Media Data', 
                        fontsize=16, fontweight='bold')
            
            self._create_simple_grid_maps(axes, results, personality_traits)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Map saved to {save_path}")
        
        plt.show()
        return fig
    
    def _create_grid_maps(self, axes, results, personality_traits):
        """Create grid-based personality maps."""
        # Add section title
        axes[0, 2].text(0.5, 1.15, "Grid-Based (Actor-Based Clustering)", 
                        ha='center', fontsize=14, fontweight='bold', 
                        transform=axes[0, 2].transAxes)
        
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 5
            col = idx % 5
            ax = axes[row, col]
            
            # Get valid data
            z_scores = results[f'{trait}_z'].values
            valid_mask = ~np.isnan(z_scores)
            
            if np.sum(valid_mask) > 0:
                # Create scatter plot for irregular grid
                valid_results = results[valid_mask]
                scatter = ax.scatter(valid_results['grid_lon'], valid_results['grid_lat'], 
                                   c=valid_results[f'{trait}_z'], cmap=cmap, 
                                   vmin=vmin, vmax=vmax, s=4, alpha=0.8)
                
                # Add German state boundaries if available
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
        """Create state-level choropleth maps."""
        if self.germany_gdf is None or state_results is None:
            for i in range(2):
                for j in range(5):
                    axes[i, j].set_visible(False)
            return
        
        # Add section title
        axes[0, 2].text(0.5, 1.15, "State-Level Aggregation", 
                        ha='center', fontsize=14, fontweight='bold', 
                        transform=axes[0, 2].transAxes)
        
        # Merge state results with shapefile
        state_col = self._get_state_column()
        if state_col is None:
            return
        
        merged_gdf = self.germany_gdf.merge(state_results, left_on=state_col, right_on='state', how='left')
        
        # Ebert's color scheme and breaks
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 5
            col = idx % 5
            ax = axes[row, col]
            
            # Create choropleth map
            merged_gdf.plot(column=f'{trait}_z', cmap=cmap, vmin=vmin, vmax=vmax,
                           ax=ax, edgecolor='black', linewidth=0.5, 
                           missing_kwds={'color': 'lightgray', 'edgecolor': 'black'})
            
            ax.set_title(trait, fontsize=12, fontweight='bold')
            ax.set_aspect('equal', adjustable='box')
            
            # Remove axis ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
        
        # Hide unused subplots
        for idx in range(len(personality_traits), 10):
            row = idx // 5
            col = idx % 5
            if row < 2 and col < 5:
                axes[row, col].set_visible(False)
    
    def _create_simple_grid_maps(self, axes, results, personality_traits):
        """Create simple grid-based maps when no state data available."""
        # Ebert's color scheme
        cmap = plt.cm.RdYlBu_r
        vmin, vmax = -1.96, 1.96
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Get valid data
            z_scores = results[f'{trait}_z'].values
            valid_mask = ~np.isnan(z_scores)
            
            if np.sum(valid_mask) > 0:
                # Create scatter plot
                valid_results = results[valid_mask]
                scatter = ax.scatter(valid_results['grid_lon'], valid_results['grid_lat'], 
                                   c=valid_results[f'{trait}_z'], cmap=cmap, 
                                   vmin=vmin, vmax=vmax, s=6, alpha=0.8)
                
                # Add German state boundaries if available
                if self.germany_gdf is not None:
                    self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)
                
                ax.set_title(trait, fontsize=14, fontweight='bold')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add statistics
                mean_z = np.nanmean(z_scores)
                std_z = np.nanstd(z_scores)
                ax.text(0.02, 0.98, f'μ={mean_z:.2f}\nσ={std_z:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Remove empty subplot if needed
        if len(personality_traits) == 5:
            axes[1, 2].remove()
        
        # Add colorbar
        if len(personality_traits) > 0:
            cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(scatter, cax=cbar_ax)
            cbar.set_label('Z Score', rotation=270, labelpad=20, fontsize=12)
    
    def _get_state_column(self):
        """Identify the state name column in the shapefile."""
        if self.germany_gdf is None:
            return None
        
        for col in ['NAME', 'GEN', 'NAME_1', 'VARNAME_1']:
            if col in self.germany_gdf.columns:
                return col
        return None
    
    def create_comparison_summary(self, state_results, save_path=None):
        """
        Create a summary comparison table similar to Ebert's supplementary materials.
        """
        if state_results is None:
            print("No state results available for comparison summary")
            return
        
        print("Creating comparison summary...")
        
        # Create summary statistics table
        summary_stats = []
        
        for trait in self.personality_traits:
            stats = {
                'Trait': trait,
                'Mean': state_results[trait].mean(),
                'SD': state_results[trait].std(),
                'Min': state_results[trait].min(),
                'Max': state_results[trait].max(),
                'Range': state_results[trait].max() - state_results[trait].min(),
                'States_N': state_results[trait].count()
            }
            summary_stats.append(stats)
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Display formatted table
        print("\n" + "="*80)
        print("PERSONALITY TRAIT SUMMARY BY GERMAN STATES")
        print("="*80)
        print(f"{'Trait':<15} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max':<8} {'Range':<8} {'N States':<8}")
        print("-" * 80)
        
        for _, row in summary_df.iterrows():
            print(f"{row['Trait']:<15} {row['Mean']:<8.3f} {row['SD']:<8.3f} {row['Min']:<8.3f} "
                  f"{row['Max']:<8.3f} {row['Range']:<8.3f} {int(row['States_N']):<8}")
        
        # Show top/bottom states for each trait
        print("\n" + "="*80)
        print("EXTREME STATES BY PERSONALITY TRAIT")
        print("="*80)
        
        for trait in self.personality_traits:
            sorted_states = state_results.sort_values(trait)
            highest = sorted_states.iloc[-1]
            lowest = sorted_states.iloc[0]
            
            print(f"\n{trait}:")
            print(f"  Highest: {highest['state']} ({highest[trait]:.3f})")
            print(f"  Lowest:  {lowest['state']} ({lowest[trait]:.3f})")
        
        if save_path:
            summary_df.to_csv(save_path.replace('.png', '_summary.csv'), index=False)
            print(f"\nSummary table saved to {save_path.replace('.png', '_summary.csv')}")
        
        return summary_df
    
    def generate_summary_statistics(self, data, results):
        """
        Generate summary statistics comparing input data to spatial results.
        """
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

def main():
    """
    Main function to run the complete Ebert-style spatial analysis with German shapefiles.
    """
    
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
    
    # Step 4: State-level aggregation (if shapefile available)
    state_results = None
    if mapper.germany_gdf is not None:
        print("\nStep 4: Aggregating by German states...")
        state_results = mapper.aggregate_by_states(data)
    
    # Step 5: Create visualizations
    print("\nStep 5: Creating maps...")
    if save_maps:
        mapper.create_ebert_style_maps(standardized_results, grid_shape, 
                                     state_results, output_file)
    else:
        mapper.create_ebert_style_maps(standardized_results, grid_shape, state_results)
    
    # Step 6: Generate summary statistics and comparison tables
    mapper.generate_summary_statistics(data, standardized_results)
    
    if state_results is not None:
        print("\nCreating state-level comparison summary...")
        mapper.create_comparison_summary(state_results, output_file)
    
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
    print("1. Grid-based continuous maps (actor-based clustering)")
    print("2. State-level choropleth maps (administrative aggregation)")
    print("3. Summary statistics and comparison tables")
    print("\nThese results can be directly compared with Ebert et al.'s")
    print("traditional BFI questionnaire approach using identical methodology.")

if __name__ == "__main__":
    main()
