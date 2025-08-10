"""
Spatial Data Analyzer with Interactive Map

Required packages:
pip install pandas numpy matplotlib geopandas shapely seaborn tqdm folium

The script creates both:
1. Static PNG map - for reports and presentations
2. Interactive HTML map - click on cells to see user counts
"""

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
import folium
from folium import plugins
import json
import warnings
warnings.filterwarnings('ignore')

class SpatialDataAnalyzer:
    """
    Analyze spatial user distribution and create grid-based maps.
    Focuses on user count ranges by grid cell.
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
        
        # Define user count ranges
        self.user_ranges = [
            (0, 0, 'Empty'),
            (1, 19, '1-19'),
            (20, 39, '20-39'),
            (40, 59, '40-59'),
            (60, 79, '60-79'),
            (80, 99, '80-99'),
            (100, float('inf'), '100+')
        ]
        
        # Define colors for each range
        self.range_colors = [
            '#fefefe',      # 0 (Empty) - almost pure white
            '#fee5d9',      # 1-19 (light red)
            '#fcae91',      # 20-39
            '#fb6a4a',      # 40-59
            '#de2d26',      # 60-79
            '#a50f15',      # 80-99
            '#67000d'       # 100+ (dark red)
        ]
        
        # Load shapefile if provided
        if shapefile_path:
            self.load_boundary_shapefile()
        
        print(f"Spatial Data Analyzer initialized:")
        print(f"  - Grid resolution: {grid_resolution_km} km")
        print(f"  - Shapefile: {'Loaded' if self.boundary_gdf is not None else 'Not provided'}")
        print(f"  - User count ranges: {', '.join([r[2] for r in self.user_ranges])}")
    
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
    
    def get_user_range_category(self, user_count):
        """Get the range category for a given user count."""
        for min_val, max_val, label in self.user_ranges:
            if min_val <= user_count <= max_val:
                return label
        return 'Unknown'
    
    def get_user_range_index(self, user_count):
        """Get the range index for a given user count."""
        for idx, (min_val, max_val, label) in enumerate(self.user_ranges):
            if min_val <= user_count <= max_val:
                return idx
        return 0
    
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
            user_count = len(cell_users)
            
            stats = {
                'cell_id': cell_id,
                'user_count': user_count,
                'has_users': user_count > 0,
                'user_range': self.get_user_range_category(user_count),
                'range_index': self.get_user_range_index(user_count)
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
        
        # Summary statistics by range
        print(f"\nGrid Statistics Summary:")
        print(f"  Total cells: {len(grid_stats)}")
        
        for min_val, max_val, label in self.user_ranges:
            if label == 'Empty':
                count = (grid_stats['user_count'] == 0).sum()
            elif max_val == float('inf'):
                count = (grid_stats['user_count'] >= min_val).sum()
            else:
                count = grid_stats['user_count'].between(min_val, max_val).sum()
            
            percentage = count / len(grid_stats) * 100
            print(f"  {label:10s}: {count:5d} cells ({percentage:5.1f}%)")
        
        # User count distribution
        cells_with_users = grid_stats[grid_stats['has_users']]
        if len(cells_with_users) > 0:
            user_counts = cells_with_users['user_count']
            print(f"\nUser count distribution (cells with users):")
            print(f"  Mean: {user_counts.mean():.1f}")
            print(f"  Median: {user_counts.median():.1f}")
            print(f"  Range: {user_counts.min()}-{user_counts.max()}")
        
        return grid_stats
    
    def create_user_range_visualization(self, grid_stats, save_path="user_count_ranges_map.png"):
        """
        Create map showing user count ranges by grid cell.
        """
        print("Creating user count ranges visualization...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        fig.suptitle('User Count Ranges by Grid Cell', fontsize=16, fontweight='bold')
        
        # Plot each range with its color
        for idx, (min_val, max_val, label) in enumerate(self.user_ranges):
            if label == 'Empty':
                subset = grid_stats[grid_stats['user_count'] == 0]
            elif max_val == float('inf'):
                subset = grid_stats[grid_stats['user_count'] >= min_val]
            else:
                subset = grid_stats[grid_stats['user_count'].between(min_val, max_val)]
            
            if len(subset) > 0:
                subset.plot(ax=ax, color=self.range_colors[idx], alpha=0.8, 
                          edgecolor='black', linewidth=0.2)
        
        # Add boundary overlay if available
        if self.boundary_gdf is not None:
            self.boundary_gdf.boundary.plot(ax=ax, color='black', linewidth=1)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Create legend
        from matplotlib.patches import Patch
        legend_elements = []
        for idx, (min_val, max_val, label) in enumerate(self.user_ranges):
            # Count cells in this range
            if label == 'Empty':
                count = (grid_stats['user_count'] == 0).sum()
            elif max_val == float('inf'):
                count = (grid_stats['user_count'] >= min_val).sum()
            else:
                count = grid_stats['user_count'].between(min_val, max_val).sum()
            
            # Add count to label
            legend_label = f'{label} ({count} cells)'
            legend_elements.append(Patch(facecolor=self.range_colors[idx], label=legend_label))
        
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), 
                 title='User Count Ranges', fontsize=10)
        
        # Add statistics text box
        stats_text = f"Grid Resolution: {self.grid_resolution_km} km\n"
        stats_text += f"Total Grid Cells: {len(grid_stats):,}\n"
        stats_text += f"Cells with Users: {grid_stats['has_users'].sum():,} ({grid_stats['has_users'].mean()*100:.1f}%)\n"
        
        if grid_stats['has_users'].any():
            cells_with_users = grid_stats[grid_stats['has_users']]
            stats_text += f"Avg Users/Cell: {cells_with_users['user_count'].mean():.1f}\n"
            stats_text += f"Max Users/Cell: {grid_stats['user_count'].max():,}"
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {save_path}")
    
    def create_interactive_map(self, grid_stats, save_path="interactive_user_map.html"):
        """
        Create an interactive HTML map where clicking on grid cells shows user count.
        """
        print("Creating interactive map...")
        
        # Calculate map center
        center_lat = grid_stats.lat_center.mean()
        center_lon = grid_stats.lon_center.mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:20px"><b>Interactive User Count Map - Click on Grid Cells</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create feature group for grid cells
        grid_group = folium.FeatureGroup(name='Grid Cells')
        
        # Add each grid cell as a clickable polygon
        for idx, row in grid_stats.iterrows():
            # Get color based on user count
            color = self.range_colors[row['range_index']]
            
            # Create polygon coordinates
            coords = [[row['lat_bottom'], row['lon_left']],
                     [row['lat_bottom'], row['lon_right']],
                     [row['lat_top'], row['lon_right']],
                     [row['lat_top'], row['lon_left']]]
            
            # Create popup text
            popup_text = f"""
            <div style='font-family: Arial; font-size: 12px;'>
                <b>Cell ID:</b> {row['cell_id']}<br>
                <b>Users:</b> {row['user_count']}<br>
                <b>Range:</b> {row['user_range']}<br>
                <b>Location:</b> ({row['lat_center']:.3f}, {row['lon_center']:.3f})
            """
            
            # Add state info if available
            if 'primary_state' in row and pd.notna(row.get('primary_state')):
                popup_text += f"<br><b>Primary State:</b> {row['primary_state']}"
            
            popup_text += "</div>"
            
            # Add polygon to map
            folium.Polygon(
                locations=coords,
                color='black',
                weight=0.5,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=200),
                tooltip=f"Users: {row['user_count']}"
            ).add_to(grid_group)
        
        # Add the grid group to the map
        grid_group.add_to(m)
        
        # Add boundaries if available
        if self.boundary_gdf is not None:
            boundary_group = folium.FeatureGroup(name='Boundaries')
            for idx, row in self.boundary_gdf.iterrows():
                # Convert geometry to geojson
                geojson = folium.GeoJson(
                    row.geometry.__geo_interface__,
                    style_function=lambda x: {
                        'fillColor': 'none',
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0
                    }
                )
                geojson.add_to(boundary_group)
            boundary_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 200px; height: auto; 
                    background-color: white; z-index:9999; font-size:12px;
                    border:2px solid grey; border-radius:5px; padding: 10px">
        <p style="margin: 0; font-weight: bold;">User Count Ranges</p>
        '''
        
        for idx, (min_val, max_val, label) in enumerate(self.user_ranges):
            color = self.range_colors[idx]
            # Count cells in this range
            if label == 'Empty':
                count = (grid_stats['user_count'] == 0).sum()
            elif max_val == float('inf'):
                count = (grid_stats['user_count'] >= min_val).sum()
            else:
                count = grid_stats['user_count'].between(min_val, max_val).sum()
            
            legend_html += f'''
            <p style="margin: 5px 0;">
                <span style="background-color:{color}; 
                           width:20px; height:10px; 
                           display:inline-block; 
                           border:1px solid black;"></span>
                {label} ({count})
            </p>
            '''
        
        legend_html += '''
        <hr style="margin: 10px 0;">
        <p style="margin: 5px 0; font-size: 10px;">
            <b>Total Cells:</b> ''' + f"{len(grid_stats):,}" + '''<br>
            <b>With Users:</b> ''' + f"{grid_stats['has_users'].sum():,}" + '''<br>
            <b>Grid Res:</b> ''' + f"{self.grid_resolution_km} km" + '''
        </p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen button
        plugins.Fullscreen(
            position='topleft',
            title='Fullscreen',
            title_cancel='Exit Fullscreen',
            force_separate_button=True
        ).add_to(m)
        
        # Save the map
        m.save(save_path)
        print(f"Saved interactive map: {save_path}")
        print("Open this HTML file in a web browser to interact with the map!")
        print("  - Click on any grid cell to see detailed information")
        print("  - Hover over cells to see user count")
        print("  - Use the layer control to toggle grid cells and boundaries")
        
        return m

def main():
    """
    Main function to run the spatial data analysis.
    """
    
    # Configuration - UPDATED FOR YOUR DATA
    data_file = "final_users_for_spatial_visualization.csv"
    shapefile_path = "german_shapefile/de.shp"  # UPDATE THIS PATH TO YOUR SHAPEFILE
    grid_resolution = 5  # Grid resolution in kilometers
    
    print("USER COUNT RANGES BY GRID CELL ANALYSIS")
    print("="*60)
    print("This analysis creates both static and interactive maps")
    print("showing user count ranges by grid cell")
    print("Ranges: 0, 1-19, 20-39, 40-59, 60-79, 80-99, 100+")
    print("="*60)
    
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
    
    # Step 4: Create visualizations (static and interactive)
    print("\nStep 4: Creating visualizations...")
    # Static map
    analyzer.create_user_range_visualization(grid_stats)
    # Interactive map
    analyzer.create_interactive_map(grid_stats)
    
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
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Files created:")
    print("✅ user_count_ranges_map.png (static map)")
    print("✅ interactive_user_map.html (interactive map - open in browser)")
    print("✅ users_with_grid_assignments.csv")
    print("\n📍 INTERACTIVE MAP FEATURES:")
    print("   • Click on any grid cell to see detailed user count")
    print("   • Hover over cells for quick user count tooltip")
    print("   • Toggle layers on/off using the layer control")
    print("   • Use fullscreen mode for better viewing")
    print("\nVisualization shows user density using color-coded ranges.")

if __name__ == "__main__":
    main()