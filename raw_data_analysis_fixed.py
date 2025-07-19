import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class RawDataAnalyzer:
    """
    Analyze raw user distribution and create explicit grid-based personality maps.
    This shows the actual data reality before any distance-decay smoothing.
    """
    
    def __init__(self, grid_resolution_km=5, shapefile_path=None):
        """
        Initialize raw data analyzer.
        
        Parameters:
        -----------
        grid_resolution_km : float
            Grid resolution in kilometers (same as used in Ebert analysis)
        shapefile_path : str
            Path to German shapefile for boundaries
        """
        self.grid_resolution_km = grid_resolution_km
        self.shapefile_path = shapefile_path
        self.germany_gdf = None
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        
        # Load German shapefile if provided
        if shapefile_path:
            self.load_german_shapefile()
        
        print(f"Raw Data Analyzer initialized:")
        print(f"  - Grid resolution: {grid_resolution_km} km")
        print(f"  - Shapefile: {'Loaded' if self.germany_gdf is not None else 'Not provided'}")
    
    def load_german_shapefile(self):
        """Load German shapefile for boundary visualization."""
        try:
            self.germany_gdf = gpd.read_file(self.shapefile_path)
            if self.germany_gdf.crs != 'EPSG:4326':
                self.germany_gdf = self.germany_gdf.to_crs('EPSG:4326')
            print(f"  - German shapefile loaded: {len(self.germany_gdf)} regions")
        except Exception as e:
            print(f"  - Warning: Could not load shapefile: {e}")
            self.germany_gdf = None
    
    def create_explicit_grid(self, data):
        """
        Create explicit 5km grid cells covering the data extent.
        
        Unlike the distance-decay approach, this creates actual discrete grid cells
        that users can be assigned to.
        """
        print("Creating explicit grid structure...")
        
        # Calculate grid bounds with buffer
        lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
        lon_min, lon_max = data['longitude'].min(), data['longitude'].max()
        
        # Add buffer around data
        buffer = 0.1  # degrees
        lat_min, lat_max = lat_min - buffer, lat_max + buffer
        lon_min, lon_max = lon_min - buffer, lon_max + buffer
        
        # Convert grid resolution from km to degrees (approximate)
        lat_step = self.grid_resolution_km / 111.0  # 1 degree latitude ≈ 111 km
        
        # Longitude step varies with latitude
        avg_lat = (lat_min + lat_max) / 2
        lon_step = self.grid_resolution_km / (111.0 * np.cos(np.radians(avg_lat)))
        
        # Create grid coordinates
        lats = np.arange(lat_min, lat_max, lat_step)
        lons = np.arange(lon_min, lon_max, lon_step)
        
        print(f"Grid dimensions: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} cells")
        
        # Create grid cells as polygons
        grid_cells = []
        cell_id = 0
        
        for i, lat in enumerate(lats[:-1]):
            for j, lon in enumerate(lons[:-1]):
                # Cell boundaries
                lat_bottom, lat_top = lat, lats[i + 1]
                lon_left, lon_right = lon, lons[j + 1]
                
                # Create polygon for this cell
                cell_polygon = Polygon([
                    (lon_left, lat_bottom), (lon_right, lat_bottom),
                    (lon_right, lat_top), (lon_left, lat_top)
                ])
                
                grid_cells.append({
                    'cell_id': cell_id,
                    'row': i,
                    'col': j,
                    'lat_center': (lat_bottom + lat_top) / 2,
                    'lon_center': (lon_left + lon_right) / 2,
                    'lat_bottom': lat_bottom,
                    'lat_top': lat_top,
                    'lon_left': lon_left,
                    'lon_right': lon_right,
                    'geometry': cell_polygon
                })
                cell_id += 1
        
        # Convert to GeoDataFrame
        grid_gdf = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')
        
        # Filter to German boundaries if shapefile available
        if self.germany_gdf is not None:
            print("Filtering grid cells to German territory...")
            
            # Keep cells that intersect with Germany
            germany_union = self.germany_gdf.geometry.unary_union
            intersects_mask = grid_gdf.geometry.intersects(germany_union)
            
            grid_gdf_filtered = grid_gdf[intersects_mask].reset_index(drop=True)
            grid_gdf_filtered['cell_id'] = range(len(grid_gdf_filtered))
            
            print(f"Filtered from {len(grid_gdf)} to {len(grid_gdf_filtered)} cells within Germany")
            return grid_gdf_filtered
        
        return grid_gdf
    
    def assign_users_to_grid(self, data, grid_cells):
        """
        Assign each user to their corresponding grid cell.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            User data with coordinates and personality scores
        grid_cells : geopandas.GeoDataFrame
            Grid cells as polygons
            
        Returns:
        --------
        data_with_cells : pandas.DataFrame
            User data with assigned cell IDs
        """
        print("Assigning users to grid cells...")
        
        # Create user points
        geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
        users_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
        
        # Spatial join to assign users to cells
        users_with_cells = gpd.sjoin(users_gdf, grid_cells[['cell_id', 'geometry']], 
                                   how='left', predicate='within')
        
        # Convert back to regular DataFrame
        data_with_cells = users_with_cells.drop('geometry', axis=1)
        
        # Check assignment success
        assigned_users = data_with_cells['cell_id'].notna().sum()
        total_users = len(data_with_cells)
        
        print(f"User assignment: {assigned_users}/{total_users} ({assigned_users/total_users*100:.1f}%)")
        
        if assigned_users < total_users * 0.95:
            print("Warning: Many users not assigned to grid cells - check coordinate systems")
        
        return data_with_cells
    
    def calculate_grid_statistics(self, data_with_cells, grid_cells):
        """
        Calculate user counts and personality statistics for each grid cell.
        
        Parameters:
        -----------
        data_with_cells : pandas.DataFrame
            User data with cell assignments
        grid_cells : geopandas.GeoDataFrame
            Grid cells
            
        Returns:
        --------
        grid_stats : geopandas.GeoDataFrame
            Grid cells with user counts and personality statistics
        """
        print("Calculating grid cell statistics...")
        
        # Initialize results
        grid_stats = grid_cells.copy()
        
        # Calculate statistics for each cell
        cell_statistics = []
        
        for cell_id in tqdm(grid_cells['cell_id'], desc="Processing cells"):
            cell_users = data_with_cells[data_with_cells['cell_id'] == cell_id]
            
            stats = {
                'cell_id': cell_id,
                'user_count': len(cell_users),
                'has_users': len(cell_users) > 0,
                'reliable': len(cell_users) >= 5,  # Minimum for reliable estimate
                'high_confidence': len(cell_users) >= 20  # High confidence threshold
            }
            
            # Calculate personality statistics if users exist
            if len(cell_users) > 0:
                for trait in self.personality_traits:
                    stats[f'{trait}_mean'] = cell_users[trait].mean()
                    stats[f'{trait}_std'] = cell_users[trait].std()
                    stats[f'{trait}_min'] = cell_users[trait].min()
                    stats[f'{trait}_max'] = cell_users[trait].max()
            else:
                # No users in cell
                for trait in self.personality_traits:
                    stats[f'{trait}_mean'] = np.nan
                    stats[f'{trait}_std'] = np.nan
                    stats[f'{trait}_min'] = np.nan
                    stats[f'{trait}_max'] = np.nan
            
            cell_statistics.append(stats)
        
        # Merge statistics with grid cells
        stats_df = pd.DataFrame(cell_statistics)
        grid_stats = grid_stats.merge(stats_df, on='cell_id', how='left')
        
        # Summary statistics
        total_cells = len(grid_stats)
        cells_with_users = grid_stats['has_users'].sum()
        reliable_cells = grid_stats['reliable'].sum()
        high_conf_cells = grid_stats['high_confidence'].sum()
        
        print(f"\nGrid Statistics Summary:")
        print(f"  Total cells: {total_cells}")
        print(f"  Cells with users: {cells_with_users} ({cells_with_users/total_cells*100:.1f}%)")
        print(f"  Reliable cells (≥5 users): {reliable_cells} ({reliable_cells/total_cells*100:.1f}%)")
        print(f"  High confidence cells (≥20 users): {high_conf_cells} ({high_conf_cells/total_cells*100:.1f}%)")
        
        # User count distribution
        user_counts = grid_stats[grid_stats['has_users']]['user_count']
        print(f"\nUser count distribution (cells with users):")
        print(f"  Mean: {user_counts.mean():.1f}")
        print(f"  Median: {user_counts.median():.1f}")
        print(f"  Range: {user_counts.min()}-{user_counts.max()}")
        
        return grid_stats
    
    def create_raw_data_visualizations(self, data_with_cells, grid_stats, save_prefix="raw_analysis"):
        """
        Create comprehensive raw data visualizations.
        
        This is Panel A: "What the data actually looks like"
        """
        print("Creating raw data visualizations...")
        
        # Figure 1: User Distribution and Grid Structure
        self._create_user_distribution_map(data_with_cells, grid_stats, 
                                         f"{save_prefix}_user_distribution.png")
        
        # Figure 2: User Count Analysis
        self._create_user_count_analysis(grid_stats, f"{save_prefix}_user_counts.png")
        
        # Figure 3: Raw Personality Maps with CONSISTENT SCALING
        self._create_raw_personality_maps_consistent(grid_stats, f"{save_prefix}_raw_personality.png")
        
        # Figure 4: Data Quality Assessment
        self._create_data_quality_maps(grid_stats, f"{save_prefix}_data_quality.png")
        
        print("Raw data visualizations complete!")
    
    def _create_user_distribution_map(self, data_with_cells, grid_stats, save_path):
        """Create user distribution map with proper colorbar and fixed range."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('User Distribution Analysis with 5km Grid System', fontsize=14, fontweight='bold')
        
        # Left panel: All users with grid
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)
        
        # Plot users
        assigned_users = data_with_cells.dropna(subset=['cell_id'])
        ax1.scatter(assigned_users['longitude'], assigned_users['latitude'], 
                   s=0.5, alpha=0.6, color='red')
        
        # Add grid overlay (sample of cells for visibility)
        sample_cells = grid_stats.sample(min(500, len(grid_stats)))
        sample_cells.boundary.plot(ax=ax1, color='gray', linewidth=0.3, alpha=0.5)
        
        ax1.set_title(f'User Distribution with 5km Grid\n{len(assigned_users):,} users shown')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_aspect('equal', adjustable='box')
        
        # Right panel: User density with colorbar (FIXED RANGE 0-50)
        plot = grid_stats.plot(column='user_count', ax=ax2, cmap='YlOrRd', 
                              legend=False, edgecolor='none', alpha=0.8,
                              vmin=0, vmax=50)  # Fixed range
        
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.8)
        
        ax2.set_title('Grid Cells by User Count')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_aspect('equal', adjustable='box')
        
        # Add colorbar with fixed range 0-50
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=50))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, shrink=0.8, extend='max')
        cbar.set_label('Users per cell (capped at 50)', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def detect_personality_columns(self, grid_stats):
        """
        Automatically detect personality trait columns in the grid stats dataframe.
        Handles shapefile column name truncation.
        """
        # Known truncation patterns from shapefile (10-character limit)
        truncated_mapping = {
            'Openness': 'Openness_m',
            'Conscientiousness': 'Conscienti', 
            'Extraversion': 'Extraversi',
            'Agreeableness': 'Agreeablen',
            'Neuroticism': 'Neuroticis'
        }
        
        personality_mapping = {}
        available_cols = list(grid_stats.columns)
        
        for trait in self.personality_traits:
            # First try the known truncated pattern
            if truncated_mapping[trait] in available_cols:
                personality_mapping[trait] = truncated_mapping[trait]
                print(f"✓ Found {trait} data in column: {truncated_mapping[trait]}")
            else:
                # Try original column name
                full_col = f'{trait}_mean'
                if full_col in available_cols:
                    personality_mapping[trait] = full_col
                    print(f"✓ Found {trait} data in column: {full_col}")
                else:
                    print(f"✗ Warning: No column found for {trait}")
        
        return personality_mapping
    
    def _create_raw_personality_maps_consistent(self, grid_stats, save_path):
        """Create raw personality maps with CONSISTENT 1-5 scaling across all traits."""
        
        # Detect personality columns
        personality_mapping = self.detect_personality_columns(grid_stats)
        
        if not personality_mapping:
            print("Error: No personality trait columns found!")
            return
        
        personality_traits = list(personality_mapping.keys())
        
        # Create figure with proper spacing
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle('Raw Personality Scores by Grid Cell - CONSISTENT SCALING\n'
                    '(Only cells with ≥5 users shown, 1-5 personality scale)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # CONSISTENT SCALING: Use 1-5 personality scale for ALL traits
        vmin, vmax = 1.0, 5.0
        cmap = plt.cm.RdYlBu_r
        norm = Normalize(vmin=vmin, vmax=vmax)
        
        # Only show reliable cells
        reliable_cells = grid_stats[grid_stats['reliable']].copy()
        print(f"Creating maps with consistent 1-5 scaling for {len(reliable_cells)} reliable cells...")
        
        for idx, trait in enumerate(personality_traits):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            if len(reliable_cells) > 0 and trait in personality_mapping:
                trait_col = personality_mapping[trait]
                
                if trait_col in reliable_cells.columns:
                    trait_data = reliable_cells[trait_col].dropna()
                    
                    if len(trait_data) > 0:
                        print(f"{trait}: range [{trait_data.min():.3f}, {trait_data.max():.3f}], "
                              f"mean {trait_data.mean():.3f}")
                        
                        # Plot with CONSISTENT scaling
                        reliable_cells.plot(column=trait_col, ax=ax, cmap=cmap,
                                          vmin=vmin, vmax=vmax, alpha=0.8,
                                          edgecolor='black', linewidth=0.1)
                        
                        # Add German boundaries
                        if self.germany_gdf is not None:
                            self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)
                        
                        ax.set_title(f'{trait}\n({len(reliable_cells)} reliable cells)')
                        ax.set_xlabel('Longitude')
                        ax.set_ylabel('Latitude')
                        ax.set_aspect('equal', adjustable='box')
                        
                        # Add statistics with ABSOLUTE VALUES
                        mean_val = trait_data.mean()
                        std_val = trait_data.std()
                        min_val = trait_data.min()
                        max_val = trait_data.max()
                        
                        stats_text = f'μ={mean_val:.3f}\nσ={std_val:.3f}\nRange: {min_val:.2f}-{max_val:.2f}'
                        ax.text(0.02, 0.98, stats_text, 
                               transform=ax.transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                               fontsize=9)
                    else:
                        ax.text(0.5, 0.5, f'No reliable data\nfor {trait}', 
                               transform=ax.transAxes, ha='center', va='center')
                        ax.set_aspect('equal', adjustable='box')
                else:
                    ax.text(0.5, 0.5, f'Column {trait_col}\nnot found', 
                           transform=ax.transAxes, ha='center', va='center')
                    ax.set_aspect('equal', adjustable='box')
            else:
                ax.text(0.5, 0.5, 'No reliable cells', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_aspect('equal', adjustable='box')
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        # Fix spacing to prevent overlapping titles
        fig.subplots_adjust(left=0.05, right=0.94, top=0.88, bottom=0.08, hspace=0.7, wspace=0.3)
        
        # Add CONSISTENT colorbar for all traits
        cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label('Personality Score (1-5 scale)', rotation=270, labelpad=20, fontsize=12)
        
        # Add scale reference
        cbar.set_ticks([1, 2, 3, 4, 5])
        cbar.set_ticklabels(['1\n(Very Low)', '2\n(Low)', '3\n(Moderate)', '4\n(High)', '5\n(Very High)'])
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Saved: {save_path}")
        print("All traits now use consistent 1-5 personality scale!")
    
    def _create_user_count_analysis(self, grid_stats, save_path):
        """Create user count distribution analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. User count histogram
        user_counts = grid_stats[grid_stats['has_users']]['user_count']
        ax1.hist(user_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(user_counts.mean(), color='red', linestyle='--', label=f'Mean: {user_counts.mean():.1f}')
        ax1.axvline(user_counts.median(), color='orange', linestyle='--', label=f'Median: {user_counts.median():.1f}')
        ax1.set_xlabel('Users per cell')
        ax1.set_ylabel('Number of cells')
        ax1.set_title('Distribution of Users per Grid Cell\n(Cells with users only)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative distribution
        sorted_counts = np.sort(user_counts)
        cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
        ax2.plot(sorted_counts, cumulative, color='blue', linewidth=2)
        ax2.axvline(5, color='red', linestyle='--', label='Reliable threshold (5)')
        ax2.axvline(20, color='orange', linestyle='--', label='High confidence (20)')
        ax2.set_xlabel('Users per cell')
        ax2.set_ylabel('Cumulative proportion')
        ax2.set_title('Cumulative Distribution of User Counts')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Empty vs filled cells
        empty_cells = (~grid_stats['has_users']).sum()
        filled_cells = grid_stats['has_users'].sum()
        reliable_cells = grid_stats['reliable'].sum()
        
        categories = ['Empty\ncells', 'Cells with\nusers', 'Reliable cells\n(≥5 users)']
        counts = [empty_cells, filled_cells, reliable_cells]
        colors = ['lightgray', 'lightblue', 'orange']
        
        bars = ax3.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
        ax3.set_ylabel('Number of cells')
        ax3.set_title('Grid Cell Categories')
        
        # Add percentage labels
        total_cells = len(grid_stats)
        for bar, count in zip(bars, counts):
            percentage = count / total_cells * 100
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total_cells*0.01,
                    f'{count:,}\n({percentage:.1f}%)', ha='center', va='bottom')
        
        # 4. Reliability zones
        reliability_data = {
            'Empty (0 users)': empty_cells,
            'Very Low (1-4)': ((grid_stats['user_count'] >= 1) & (grid_stats['user_count'] < 5)).sum(),
            'Reliable (5-19)': ((grid_stats['user_count'] >= 5) & (grid_stats['user_count'] < 20)).sum(),
            'High Conf (20+)': (grid_stats['user_count'] >= 20).sum()
        }
        
        wedges, texts, autotexts = ax4.pie(reliability_data.values(), 
                                          labels=reliability_data.keys(),
                                          autopct='%1.1f%%',
                                          colors=['lightgray', 'lightcoral', 'lightyellow', 'lightgreen'])
        ax4.set_title('Data Reliability Distribution')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def _create_data_quality_maps(self, grid_stats, save_path):
        """Create data quality assessment maps."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality and Reliability Assessment', fontsize=16, fontweight='bold')
        
        # 1. User count map
        grid_stats.plot(column='user_count', ax=ax1, cmap='viridis', 
                       legend=True, legend_kwds={'shrink': 0.8})
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax1, color='white', linewidth=1)
        ax1.set_title('User Count per Grid Cell')
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Reliability zones
        reliability_colors = {
            0: 'lightgray',    # Empty
            1: 'lightcoral',   # 1-4 users
            2: 'lightyellow',  # 5-19 users  
            3: 'lightgreen'    # 20+ users
        }
        
        grid_stats['reliability_zone'] = 0
        grid_stats.loc[grid_stats['user_count'].between(1, 4), 'reliability_zone'] = 1
        grid_stats.loc[grid_stats['user_count'].between(5, 19), 'reliability_zone'] = 2
        grid_stats.loc[grid_stats['user_count'] >= 20, 'reliability_zone'] = 3
        
        for zone, color in reliability_colors.items():
            subset = grid_stats[grid_stats['reliability_zone'] == zone]
            if len(subset) > 0:
                subset.plot(ax=ax2, color=color, alpha=0.8, edgecolor='none')
        
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax2, color='black', linewidth=1)
        ax2.set_title('Reliability Zones')
        ax2.set_aspect('equal', adjustable='box')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='lightgray', label='Empty (0 users)'),
            Patch(facecolor='lightcoral', label='Low reliability (1-4)'),
            Patch(facecolor='lightyellow', label='Reliable (5-19)'),
            Patch(facecolor='lightgreen', label='High confidence (20+)')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        # 3. Data coverage
        empty_cells = grid_stats[~grid_stats['has_users']]
        filled_cells = grid_stats[grid_stats['has_users']]
        
        if len(empty_cells) > 0:
            empty_cells.plot(ax=ax3, color='lightgray', alpha=0.8, label='No data')
        if len(filled_cells) > 0:
            filled_cells.plot(ax=ax3, color='blue', alpha=0.6, label='Has data')
        
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax3, color='black', linewidth=1)
        ax3.set_title('Data Coverage')
        ax3.set_aspect('equal', adjustable='box')
        ax3.legend()
        
        # 4. Summary statistics table
        ax4.axis('off')
        
        # Create summary table
        summary_data = {
            'Metric': [
                'Total grid cells',
                'Cells with users',
                'Empty cells',
                'Reliable cells (≥5 users)',
                'High confidence (≥20 users)',
                'Mean users per cell (with data)',
                'Median users per cell (with data)',
                'Max users in single cell'
            ],
            'Value': [
                f"{len(grid_stats):,}",
                f"{grid_stats['has_users'].sum():,} ({grid_stats['has_users'].mean()*100:.1f}%)",
                f"{(~grid_stats['has_users']).sum():,} ({(~grid_stats['has_users']).mean()*100:.1f}%)",
                f"{grid_stats['reliable'].sum():,} ({grid_stats['reliable'].mean()*100:.1f}%)",
                f"{grid_stats['high_confidence'].sum():,} ({grid_stats['high_confidence'].mean()*100:.1f}%)",
                f"{grid_stats[grid_stats['has_users']]['user_count'].mean():.1f}",
                f"{grid_stats[grid_stats['has_users']]['user_count'].median():.0f}",
                f"{grid_stats['user_count'].max():,}"
            ]
        }
        
        table = ax4.table(cellText=[[metric, value] for metric, value in zip(summary_data['Metric'], summary_data['Value'])],
                         colLabels=['Metric', 'Value'],
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.6, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax4.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")

def main():
    """
    Main function to run the complete raw data analysis with consistent scaling fixes.
    
    This creates Panel A: "What the data actually looks like"
    """
    
    # Configuration
    data_file = "final_users_for_spatial_visualization.csv"  # UPDATE THIS PATH
    shapefile_path = "german_shapefile/de.shp"   # UPDATE THIS PATH
    grid_resolution = 5  # Same as used in Ebert analysis
    
    print("RAW DATA ANALYSIS WITH CONSISTENT SCALING FIXES")
    print("="*60)
    print("This analysis shows the actual data distribution before any smoothing.")
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv(data_file)
        personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                             'Agreeableness', 'Neuroticism']
        required_cols = ['latitude', 'longitude'] + personality_traits
        data = data.dropna(subset=required_cols)
        print(f"Loaded {len(data):,} users")
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}")
        return
    
    # Initialize analyzer
    analyzer = RawDataAnalyzer(grid_resolution_km=grid_resolution, 
                              shapefile_path=shapefile_path)
    
    # Step 1: Create explicit grid
    print("\nStep 1: Creating explicit grid structure...")
    grid_cells = analyzer.create_explicit_grid(data)
    
    # Step 2: Assign users to grid cells
    print("\nStep 2: Assigning users to grid cells...")
    data_with_cells = analyzer.assign_users_to_grid(data, grid_cells)
    
    # Step 3: Calculate grid statistics
    print("\nStep 3: Calculating grid cell statistics...")
    grid_stats = analyzer.calculate_grid_statistics(data_with_cells, grid_cells)
    
    # Step 4: Create visualizations
    print("\nStep 4: Creating visualizations...")
    analyzer.create_raw_data_visualizations(data_with_cells, grid_stats)
    
    # Step 5: Save data for comparison analysis
    print("\nStep 5: Saving results for comparison...")
    grid_stats.to_file("raw_grid_analysis.shp")
    data_with_cells.to_csv("users_with_grid_assignments.csv", index=False)
    
    print("\n" + "="*60)
    print("RAW DATA ANALYSIS COMPLETE!")
    print("="*60)
    print("Files created:")
    print("✅ raw_analysis_user_distribution.png (WITH PROPER COLORBAR 0-50 RANGE)")
    print("✅ raw_analysis_user_counts.png") 
    print("✅ raw_analysis_raw_personality.png (CONSISTENT 1-5 SCALING)")
    print("✅ raw_analysis_data_quality.png")
    print("✅ raw_grid_analysis.shp (grid cells with statistics)")
    print("✅ users_with_grid_assignments.csv (users assigned to cells)")
    print("\nKey improvements:")
    print("✅ User distribution map now has proper colorbar with 0-50 range")
    print("✅ Personality maps use consistent 1-5 scaling across all traits")
    print("✅ Fixed overlapping labels with proper spacing")
    print("✅ Automatic detection of shapefile column truncation")
    print("\nNext step: Compare with distance-decay results!")

if __name__ == "__main__":
    main()