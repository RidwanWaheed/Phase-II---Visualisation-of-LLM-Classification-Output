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

class SpatialDataAnalyzer:
    """
    Analyze spatial user distribution and create grid-based maps.
    Focuses on geographic distribution without personality analysis.
    """
    
    def __init__(self, grid_resolution_km=5, shapefile_path=None):
        """
        Initialize spatial data analyzer.
        
        Parameters:
        -----------
        grid_resolution_km : float
            Grid resolution in kilometers
        shapefile_path : str
            Path to shapefile for boundary clipping
        """
        self.grid_resolution_km = grid_resolution_km
        self.shapefile_path = shapefile_path
        self.boundary_gdf = None
        
        # Load shapefile if provided
        if shapefile_path:
            self.load_boundary_shapefile()
        
        print(f"Spatial Data Analyzer initialized:")
        print(f"  - Grid resolution: {grid_resolution_km} km")
        print(f"  - Shapefile: {'Loaded' if self.boundary_gdf is not None else 'Not provided'}")
    
    def load_boundary_shapefile(self):
        """Load boundary shapefile for grid clipping."""
        try:
            self.boundary_gdf = gpd.read_file(self.shapefile_path)
            if self.boundary_gdf.crs != 'EPSG:4326':
                self.boundary_gdf = self.boundary_gdf.to_crs('EPSG:4326')
            print(f"  - Boundary shapefile loaded: {len(self.boundary_gdf)} regions")
        except Exception as e:
            print(f"  - Warning: Could not load shapefile: {e}")
            self.boundary_gdf = None
    
    def create_explicit_grid(self, data):
        """
        Create explicit grid cells covering the data extent.
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
        
        # Filter to shapefile boundaries if available
        if self.boundary_gdf is not None:
            print("Clipping grid cells to shapefile boundaries...")
            
            # Keep cells that intersect with the boundary
            boundary_union = self.boundary_gdf.geometry.unary_union
            intersects_mask = grid_gdf.geometry.intersects(boundary_union)
            
            grid_gdf_filtered = grid_gdf[intersects_mask].reset_index(drop=True)
            grid_gdf_filtered['cell_id'] = range(len(grid_gdf_filtered))
            
            print(f"Clipped from {len(grid_gdf)} to {len(grid_gdf_filtered)} cells within boundaries")
            return grid_gdf_filtered
        
        return grid_gdf
    
    def assign_users_to_grid(self, data, grid_cells):
        """
        Assign each user to their corresponding grid cell.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            User data with coordinates
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
        Calculate user counts and basic statistics for each grid cell.
        
        Parameters:
        -----------
        data_with_cells : pandas.DataFrame
            User data with cell assignments
        grid_cells : geopandas.GeoDataFrame
            Grid cells
            
        Returns:
        --------
        grid_stats : geopandas.GeoDataFrame
            Grid cells with user counts and basic statistics
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
            
            # Add state information if available
            if 'state' in data_with_cells.columns and len(cell_users) > 0:
                # Most common state in this cell
                state_counts = cell_users['state'].value_counts()
                stats['primary_state'] = state_counts.index[0] if len(state_counts) > 0 else None
                stats['state_diversity'] = len(state_counts)  # Number of different states
            
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
        if cells_with_users > 0:
            user_counts = grid_stats[grid_stats['has_users']]['user_count']
            print(f"\nUser count distribution (cells with users):")
            print(f"  Mean: {user_counts.mean():.1f}")
            print(f"  Median: {user_counts.median():.1f}")
            print(f"  Range: {user_counts.min()}-{user_counts.max()}")
        
        return grid_stats
    
    def create_spatial_visualizations(self, data_with_cells, grid_stats, save_prefix="spatial_analysis"):
        """
        Create comprehensive spatial visualizations.
        """
        print("Creating spatial visualizations...")
        
        # Figure 1: User Distribution and Grid Structure
        self._create_user_distribution_map(data_with_cells, grid_stats, 
                                         f"{save_prefix}_user_distribution.png")
        
        # Figure 2: User Count Analysis
        self._create_user_count_analysis(grid_stats, f"{save_prefix}_user_counts.png")
        
        # Figure 3: State Distribution (if state data available)
        if 'state' in data_with_cells.columns:
            self._create_state_analysis(data_with_cells, grid_stats, f"{save_prefix}_state_distribution.png")
        
        # Figure 4: Data Quality Assessment
        self._create_data_quality_maps(grid_stats, f"{save_prefix}_data_quality.png")
        
        print("Spatial visualizations complete!")
    
    def _create_user_distribution_map(self, data_with_cells, grid_stats, save_path):
        """Create user distribution map with proper colorbar."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('User Distribution Analysis with Grid System', fontsize=14, fontweight='bold')
        
        # Left panel: All users with grid
        # Add boundary overlay first if available
        if self.boundary_gdf is not None:
            self.boundary_gdf.boundary.plot(ax=ax1, color='black', linewidth=1, alpha=0.8)
        
        assigned_users = data_with_cells.dropna(subset=['cell_id'])
        ax1.scatter(assigned_users['longitude'], assigned_users['latitude'], 
                   s=20, alpha=0.7, color='red', edgecolors='black', linewidth=0.5)
        
        # Add grid overlay (sample of cells for visibility)
        sample_size = min(200, len(grid_stats))
        sample_cells = grid_stats.sample(sample_size)
        sample_cells.boundary.plot(ax=ax1, color='gray', linewidth=0.5, alpha=0.7)
        
        ax1.set_title(f'User Distribution with {self.grid_resolution_km}km Grid\n{len(assigned_users):,} users shown')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_aspect('equal', adjustable='box')
        ax1.grid(True, alpha=0.3)
        
        # Right panel: User density with colorbar
        max_users = grid_stats['user_count'].max()
        vmax = min(max_users, 10)  # Cap at 10 for better visualization
        
        plot = grid_stats.plot(column='user_count', ax=ax2, cmap='YlOrRd', 
                              legend=False, edgecolor='gray', alpha=0.8, linewidth=0.2,
                              vmin=0, vmax=vmax)
        
        # Add boundary overlay to right panel too
        if self.boundary_gdf is not None:
            self.boundary_gdf.boundary.plot(ax=ax2, color='black', linewidth=1, alpha=0.8)
        
        ax2.set_title('Grid Cells by User Count')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_aspect('equal', adjustable='box')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2, shrink=0.8)
        if max_users > vmax:
            cbar.set_label(f'Users per cell (capped at {vmax})', rotation=270, labelpad=15)
        else:
            cbar.set_label('Users per cell', rotation=270, labelpad=15)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def _create_user_count_analysis(self, grid_stats, save_path):
        """Create user count distribution analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Only analyze cells with users
        cells_with_users = grid_stats[grid_stats['has_users']]
        
        if len(cells_with_users) == 0:
            fig.suptitle('No cells with users found', fontsize=16)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            return
        
        user_counts = cells_with_users['user_count']
        
        # 1. User count histogram
        ax1.hist(user_counts, bins=min(20, len(user_counts)), alpha=0.7, color='skyblue', edgecolor='black')
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
        
        # Only plot if we have data
        if sum(reliability_data.values()) > 0:
            wedges, texts, autotexts = ax4.pie(reliability_data.values(), 
                                              labels=reliability_data.keys(),
                                              autopct='%1.1f%%',
                                              colors=['lightgray', 'lightcoral', 'lightyellow', 'lightgreen'])
            ax4.set_title('Data Reliability Distribution')
        else:
            ax4.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def _create_state_analysis(self, data_with_cells, grid_stats, save_path):
        """Create state distribution analysis if state data is available."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('State Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. User distribution by state
        state_counts = data_with_cells['state'].value_counts()
        ax1.bar(range(len(state_counts)), state_counts.values, alpha=0.7, color='steelblue')
        ax1.set_xticks(range(len(state_counts)))
        ax1.set_xticklabels(state_counts.index, rotation=45, ha='right')
        ax1.set_ylabel('Number of users')
        ax1.set_title('Users by State')
        ax1.grid(True, alpha=0.3)
        
        # 2. Geographic distribution by state
        for state in data_with_cells['state'].unique():
            state_users = data_with_cells[data_with_cells['state'] == state]
            ax2.scatter(state_users['longitude'], state_users['latitude'], 
                       label=state, alpha=0.7, s=30)
        
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title('Geographic Distribution by State')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. State diversity in grid cells
        cells_with_state_info = grid_stats[grid_stats['has_users'] & 
                                         grid_stats['state_diversity'].notna()]
        
        if len(cells_with_state_info) > 0:
            diversity_counts = cells_with_state_info['state_diversity'].value_counts().sort_index()
            ax3.bar(diversity_counts.index, diversity_counts.values, alpha=0.7, color='green')
            ax3.set_xlabel('Number of different states in cell')
            ax3.set_ylabel('Number of cells')
            ax3.set_title('State Diversity within Grid Cells')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No state diversity data', ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Summary statistics
        ax4.axis('off')
        
        # Create summary table
        total_users = len(data_with_cells)
        total_states = data_with_cells['state'].nunique()
        
        summary_data = {
            'Metric': [
                'Total users',
                'Total states',
                'Most common state',
                'Users in most common state',
                'State diversity (mean per cell)',
                'Cells with multiple states'
            ],
            'Value': [
                f"{total_users:,}",
                f"{total_states}",
                f"{state_counts.index[0]}",
                f"{state_counts.iloc[0]} ({state_counts.iloc[0]/total_users*100:.1f}%)",
                f"{cells_with_state_info['state_diversity'].mean():.1f}" if len(cells_with_state_info) > 0 else "N/A",
                f"{(cells_with_state_info['state_diversity'] > 1).sum()}" if len(cells_with_state_info) > 0 else "N/A"
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
    
    def _create_data_quality_maps(self, grid_stats, save_path):
        """Create data quality assessment maps."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality and Reliability Assessment', fontsize=16, fontweight='bold')
        
        # 1. User count map
        grid_stats.plot(column='user_count', ax=ax1, cmap='viridis', 
                       legend=True, legend_kwds={'shrink': 0.8}, alpha=0.8)
        if self.boundary_gdf is not None:
            self.boundary_gdf.boundary.plot(ax=ax1, color='white', linewidth=1)
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
                subset.plot(ax=ax2, color=color, alpha=0.8, edgecolor='black', linewidth=0.2)
        
        if self.boundary_gdf is not None:
            self.boundary_gdf.boundary.plot(ax=ax2, color='black', linewidth=1)
        
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
            empty_cells.plot(ax=ax3, color='lightgray', alpha=0.8, label='No data', edgecolor='black', linewidth=0.1)
        if len(filled_cells) > 0:
            filled_cells.plot(ax=ax3, color='blue', alpha=0.6, label='Has data', edgecolor='black', linewidth=0.1)
        
        if self.boundary_gdf is not None:
            self.boundary_gdf.boundary.plot(ax=ax3, color='black', linewidth=1)
        
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
                f"{grid_stats[grid_stats['has_users']]['user_count'].mean():.1f}" if grid_stats['has_users'].any() else "N/A",
                f"{grid_stats[grid_stats['has_users']]['user_count'].median():.0f}" if grid_stats['has_users'].any() else "N/A",
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
    Main function to run the spatial data analysis.
    """
    
    # Configuration - UPDATED FOR YOUR DATA
    data_file = "final_users_for_spatial_visualization.csv"
    shapefile_path = "german_shapefile/de.shp"  # UPDATE THIS PATH TO YOUR SHAPEFILE
    grid_resolution = 5  # Grid resolution in kilometers
    
    print("SPATIAL DATA ANALYSIS")
    print("="*60)
    print("This analysis shows spatial distribution patterns of users.")
    print("Grid cells will be clipped to shapefile boundaries if provided.")
    
    # Load data
    print("\nLoading data...")
    try:
        data = pd.read_csv(data_file)
        required_cols = ['latitude', 'longitude', 'user_guid']
        
        # Check for required columns
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
        
        # Remove rows with missing coordinates
        initial_count = len(data)
        data = data.dropna(subset=['latitude', 'longitude'])
        final_count = len(data)
        
        if final_count < initial_count:
            print(f"Removed {initial_count - final_count} rows with missing coordinates")
        
        print(f"Loaded {len(data):,} users with valid coordinates")
        
        # Show data summary
        print(f"\nData summary:")
        print(f"  Latitude range: {data['latitude'].min():.3f} to {data['latitude'].max():.3f}")
        print(f"  Longitude range: {data['longitude'].min():.3f} to {data['longitude'].max():.3f}")
        if 'state' in data.columns:
            print(f"  States: {data['state'].nunique()} unique states")
            print(f"  State distribution: {dict(data['state'].value_counts().head())}")
        
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}")
        print("Please make sure the file is in the current directory.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Initialize analyzer
    analyzer = SpatialDataAnalyzer(grid_resolution_km=grid_resolution, 
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
    analyzer.create_spatial_visualizations(data_with_cells, grid_stats)
    
    # Step 5: Save data for further analysis
    print("\nStep 5: Saving results...")
    try:
        grid_stats.to_file("spatial_grid_analysis.shp")
        print("✅ Grid analysis saved to: spatial_grid_analysis.shp")
    except Exception as e:
        print(f"Note: Could not save shapefile: {e}")
    
    data_with_cells.to_csv("users_with_grid_assignments.csv", index=False)
    print("✅ User assignments saved to: users_with_grid_assignments.csv")
    
    print("\n" + "="*60)
    print("SPATIAL ANALYSIS COMPLETE!")
    print("="*60)
    print("Files created:")
    print("✅ spatial_analysis_user_distribution.png")
    print("✅ spatial_analysis_user_counts.png")
    if 'state' in data.columns:
        print("✅ spatial_analysis_state_distribution.png")
    print("✅ spatial_analysis_data_quality.png")
    print("✅ users_with_grid_assignments.csv")
    print("\nAnalysis focuses on spatial distribution patterns without personality traits.")
    print("Grid cells are clipped to shapefile boundaries for geographic accuracy.")

if __name__ == "__main__":
    main()