"""
Spatial Data Analyzer with Interactive Map and Uncertainty Map
Aligned with spatial_visualization.py projection system

Required packages:
pip install pandas numpy matplotlib geopandas shapely seaborn tqdm folium pyproj

The script creates both:
1. Static PNG map - for reports and presentations
2. Interactive HTML map - click on cells to see user counts
3. Static PNG Uncertainty Map - shows reliability based on weight_sum
4. Interactive HTML Uncertainty Map - click on cells to see weight_sum as reliability indicator
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
from pyproj import CRS, Transformer # Import for potential CRS handling, though not directly used for uncertainty map logic here.

warnings.filterwarnings('ignore')

class SpatialDataAnalyzer:
    """
    Analyze spatial user distribution and create grid-based maps.
    Focuses on user count ranges by grid cell and introduces reliability based on weight_sum.
    Uses proper projection alignment for accurate geographic representation.
    """

    def __init__(self, grid_resolution_km=5, shapefile_path=None):
        """
        Initialize spatial data analyzer.

        Parameters:
        -----------
        grid_resolution_km : float
            Grid resolution in kilometers
        shapefile_path : str
            Path to shapefile for boundary clipping (preferably in projected CRS)
        """
        self.grid_resolution_km = grid_resolution_km
        self.shapefile_path = shapefile_path

        # Store both projected (for plotting) and WGS84 (for spatial joins) versions
        self.boundary_gdf_proj = None  # Native/projected CRS for accurate plotting
        self.boundary_gdf_wgs84 = None  # WGS84 for spatial joins with lat/lon data
        self.target_crs = None  # Target CRS for projection

        # Define user count ranges for general user density map
        self.user_ranges = [
            (0, 0, 'Empty'),
            (1, 19, '1-19'),
            (20, 39, '20-39'),
            (40, 59, '40-59'),
            (60, 79, '60-79'),
            (80, 99, '80-99'),
            (100, float('inf'), '100+')
        ]

        # Define colors for each user count range
        self.range_colors = [
            '#fefefe',      # 0 (Empty) - almost pure white
            '#fee5d9',      # 1-19 (light red)
            '#fcae91',      # 20-39
            '#fb6a4a',      # 40-59
            '#de2d26',      # 60-79
            '#a50f15',      # 80-99
            '#67000d'       # 100+ (dark red)
        ]

        # Define reliability ranges for the uncertainty map based on weight_sum
        # These thresholds may need adjustment based on your data's typical weight_sum values
        self.reliability_ranges = [
            (0, 0.001, 'No Data / Very Low (0)'), # Represents truly empty or negligible weight
            (0.001, 5, 'Low (1-5)'),
            (5, 20, 'Medium (6-20)'),
            (20, 50, 'High (21-50)'),
            (50, float('inf'), 'Very High (50+)')
        ]

        # Define colors for reliability ranges (e.g., green for high, red for low)
        # Using a sequential green color scheme for reliability: lighter for lower, darker for higher
        self.reliability_colors = [
            '#dcdcdc',      # No Data / Very Low (Grey)
            '#d9f0a3',      # Low (light green)
            '#addd8e',      # Medium
            '#78c679',      # High
            '#238443'       # Very High (dark green)
        ]


        # Load shapefile if provided
        if shapefile_path:
            self.load_boundary_shapefile()

        print(f"Spatial Data Analyzer initialized:")
        print(f"  - Grid resolution: {grid_resolution_km} km")
        print(f"  - Shapefile: {'Loaded' if self.boundary_gdf_proj is not None else 'Not provided'}")
        if self.target_crs:
            print(f"  - Target CRS: {self.target_crs}")
        print(f"  - User count ranges: {', '.join([r[2] for r in self.user_ranges])}")
        print(f"  - Reliability ranges (weight_sum): {', '.join([r[2] for r in self.reliability_ranges])}")

    def load_boundary_shapefile(self):
        """Load boundary shapefile in native CRS and create WGS84 version for spatial joins."""
        try:
            # Load in native/projected CRS for accurate plotting
            self.boundary_gdf_proj = gpd.read_file(self.shapefile_path)
            self.target_crs = self.boundary_gdf_proj.crs
            print(f"  - Boundary shapefile loaded in native CRS: {self.target_crs}")
            print(f"  - Regions: {len(self.boundary_gdf_proj)}")

            # Create WGS84 version for spatial joins with lat/lon data
            if self.boundary_gdf_proj.crs != 'EPSG:4326':
                print(f"  - Creating WGS84 version for spatial operations...")
                self.boundary_gdf_wgs84 = self.boundary_gdf_proj.to_crs('EPSG:4326')
            else:
                self.boundary_gdf_wgs84 = self.boundary_gdf_proj.copy()

        except Exception as e:
            print(f"  - Warning: Could not load shapefile: {e}")
            self.boundary_gdf_proj = None
            self.boundary_gdf_wgs84 = None
            self.target_crs = None

    def create_explicit_grid(self, data):
        """
        Create explicit grid cells covering the data extent in WGS84.
        Grid will be projected later for visualization.
        """
        print("Creating explicit grid structure...")

        # Calculate grid bounds with buffer (in WGS84)
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

        # Convert to GeoDataFrame in WGS84
        grid_gdf = gpd.GeoDataFrame(grid_cells, crs='EPSG:4326')

        # Filter to shapefile boundaries if available (using WGS84 version)
        if self.boundary_gdf_wgs84 is not None:
            print("Clipping grid cells to shapefile boundaries...")

            # Keep cells that intersect with the boundary
            boundary_union = self.boundary_gdf_wgs84.geometry.unary_union
            intersects_mask = grid_gdf.geometry.intersects(boundary_union)

            grid_gdf_filtered = grid_gdf[intersects_mask].reset_index(drop=True)
            grid_gdf_filtered['cell_id'] = range(len(grid_gdf_filtered))

            print(f"Clipped from {len(grid_gdf)} to {len(grid_gdf_filtered)} cells within boundaries")
            return grid_gdf_filtered

        return grid_gdf

    def project_grid_cells(self, grid_gdf):
        """
        Project grid cells from WGS84 to target CRS for accurate visualization.

        Parameters:
        -----------
        grid_gdf : geopandas.GeoDataFrame
            Grid cells in WGS84

        Returns:
        --------
        grid_gdf : geopandas.GeoDataFrame
            Grid cells with projected coordinates added
        """
        if self.target_crs and self.target_crs != 'EPSG:4326':
            print(f"Projecting grid cells to {self.target_crs} for accurate visualization...")

            # Project the grid to target CRS
            grid_proj = grid_gdf.to_crs(self.target_crs)

            # Add projected coordinates to original dataframe
            grid_gdf['x_proj'] = grid_proj.geometry.centroid.x
            grid_gdf['y_proj'] = grid_proj.geometry.centroid.y

            # Store projected geometry separately
            grid_gdf['geometry_proj'] = grid_proj.geometry

            print("Grid projection complete.")
        else:
            # If no projection needed, use lat/lon as x/y
            grid_gdf['x_proj'] = grid_gdf['lon_center']
            grid_gdf['y_proj'] = grid_gdf['lat_center']
            grid_gdf['geometry_proj'] = grid_gdf['geometry']

        return grid_gdf

    def assign_users_to_grid(self, data, grid_cells):
        """
        Assign each user to their corresponding grid cell using WGS84 coordinates.

        Parameters:
        -----------
        data : pandas.DataFrame
            User data with coordinates
        grid_cells : geopandas.GeoDataFrame
            Grid cells as polygons (in WGS84)

        Returns:
        --------
        data_with_cells : pandas.DataFrame
            User data with assigned cell IDs
        """
        print("Assigning users to grid cells...")

        # Create user points in WGS84
        geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
        users_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')

        # Spatial join to assign users to cells (both in WGS84)
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
        return 0 # Default to empty/first range if no match

    def get_reliability_category(self, weight_sum):
        """Get the reliability category for a given weight sum."""
        for min_val, max_val, label in self.reliability_ranges:
            if min_val <= weight_sum <= max_val:
                return label
        return 'Unknown' # Should not happen

    def get_reliability_index(self, weight_sum):
        """Get the reliability index for a given weight sum."""
        for idx, (min_val, max_val, label) in enumerate(self.reliability_ranges):
            if min_val <= weight_sum <= max_val:
                return idx
        return 0 # Default to first range

    def calculate_grid_statistics(self, data_with_cells, grid_cells):
        """
        Calculate user counts and basic statistics for each grid cell.
        Also calculates average weight_sum for reliability indication.

        Parameters:
        -----------
        data_with_cells : pandas.DataFrame
            User data with cell assignments (should include personality traits and _weight_sum)
        grid_cells : geopandas.GeoDataFrame
            Grid cells (with projected coordinates if available)

        Returns:
        --------
        grid_stats : geopandas.GeoDataFrame
            Grid cells with user counts, basic statistics, and reliability metrics
        """
        print("Calculating grid cell statistics and reliability metrics...")

        # Initialize results with projected grid cells
        grid_stats = grid_cells.copy()

        # Calculate statistics for each cell
        cell_statistics = []

        # Assuming 'Openness_weight_sum' is representative for overall reliability.
        # If you have specific needs, you might average across all _weight_sum columns.
        trait_weight_sum_col = 'Openness_weight_sum' # Example: using Openness's weight sum

        for cell_id in tqdm(grid_cells['cell_id'], desc="Processing cells for stats"):
            cell_users = data_with_cells[data_with_cells['cell_id'] == cell_id]
            user_count = len(cell_users)

            # Calculate average weight_sum for this cell.
            # If no users, weight_sum is 0. If users present, average their weight_sum from original data.
            # NOTE: The _weight_sum columns come from the spatial_grid_computer.py output,
            # which means this script (spatial_analyzer_interactive.py) needs to read that output.
            # For this example, we're assuming 'data_with_cells' will have these columns.
            # If 'data_with_cells' only has raw user data, you'd need to re-run the
            # weighted score calculation here or ensure 'grid_stats' is merged with
            # the results of 'calculate_weighted_scores' from the other script.

            # For now, let's assume `data_with_cells` contains the individual user's
            # contribution for each trait and we average it, or, more simply,
            # if grid_stats already contains a 'total_weight_sum' from a previous step,
            # we can use that.

            # Given the previous context, the `spatial_personality_grid_results.csv`
            # contains the `_weight_sum` for each grid cell. We should ideally load that
            # or ensure this `grid_stats` dataframe is enriched with it.
            # For a pure "uncertainty map," we need the actual sum of weights for each cell.
            # Let's add a placeholder for this. For this script to work robustly,
            # 'grid_stats' should ideally be the result of the spatial_grid_computer.
            # For now, if the `trait_weight_sum_col` isn't in `data_with_cells`,
            # we'll approximate with user_count for this demo.

            # **IMPORTANT**: In a real pipeline, `grid_stats` here should be
            # the output from `SpatialPersonalityGridComputer.calculate_weighted_scores`
            # or `standardize_scores`, which already contains the `_weight_sum` columns.
            # This analyzer should consume that output directly.
            # For this standalone analyzer, we will simulate or use a pre-existing column.
            # Let's assume the user has run spatial_grid_computer.py and its output
            # (spatial_personality_grid_results.csv) is loaded as data, or that
            # the `grid_stats` passed here already has the required `weight_sum` columns.

            # For this simplified analyzer, if no specific weight_sum is passed from
            # a pre-computed grid, we'll use user_count as a proxy for 'total_weight_sum'
            # for the purpose of demonstrating the reliability map, as user_count is directly
            # correlated with reliability in a simpler sense.
            # However, the prompt explicitly mentions 'weight_sum'.
            # Let's assume the input `grid_stats` already has a relevant `_weight_sum` column
            # from the `spatial_grid_computer.py` output.
            
            # If 'grid_stats' is directly from `spatial_personality_grid_results.csv`,
            # it will have columns like 'Openness_weight_sum'.
            # We'll use this column. If this script is run *before* the `spatial_grid_computer`,
            # this part will need careful handling or placeholder values.

            # For the purpose of adding to grid_stats, let's assume 'total_weight_sum' is already merged or computed earlier.
            # If we're relying on the 'grid_stats' output from the previous `spatial_grid_computer`
            # then it already has these columns (e.g., 'Openness_weight_sum').

            # To avoid dependency issues if `spatial_personality_grid_results.csv` isn't used
            # as the direct input to this analyzer, we will use a simple approximation
            # of `total_weight_sum` based on `user_count` here for demonstration,
            # but note that the true `weight_sum` comes from the other script's
            # decay function.

            # Re-evaluating based on the assumption that 'grid_stats' is the output of the
            # previous 'spatial_grid_computer.py' script. It contains `_weight_sum` columns.
            # Let's make sure our `main` function loads the correct file for this.
            
            # The input `grid_cells` here is just a grid. The actual `weight_sum`
            # needs to come from the full computed grid results.
            # This `calculate_grid_statistics` function, in its current form, only counts users.
            # To integrate `weight_sum`, we need to read the `spatial_personality_grid_results.csv`
            # created by the `spatial_grid_computer.py` and merge it here.

            # For now, let's just make sure the output dataframe has a column for 'avg_weight_sum'
            # (or just `weight_sum` if it's already a per-cell statistic).
            # The prompt implies that `grid_stats` should already contain the `weight_sum`.
            # I will modify the main function to load the `spatial_personality_grid_results.csv`
            # directly if it exists, to get the weight sums.
            
            # For this function to calculate it:
            # This logic needs to be run *after* weighted scores are computed.
            # The prompt is asking to *use* the result of the grid process.
            # So, the input `grid_stats` to this function should ideally be the `standardized_results`
            # from `spatial_grid_computer.py`.

            # Let's modify main() to load the `spatial_personality_grid_results.csv` and pass it.

            # This function is now simplified to just add user_range.
            # The 'reliability' calculation will happen assuming `grid_stats` (from main)
            # already contains the `_weight_sum` column.

            stats = {
                'cell_id': cell_id,
                'user_count': user_count,
                'has_users': user_count > 0,
                'user_range': self.get_user_range_category(user_count)
            }
            cell_statistics.append(stats)


        # Merge statistics with grid cells.
        # This will merge the user counts into the grid structure.
        stats_df = pd.DataFrame(cell_statistics)
        # Ensure 'cell_id' is unique in stats_df before merging, if not already.
        # This merge should be on the full `grid_stats` which contains the weight_sum.
        # So this function will return user_count specific stats.
        # The 'main' function will then enrich the `grid_stats` with reliability.
        merged_grid_stats = grid_stats.merge(stats_df, on='cell_id', how='left', suffixes=('_original', ''))
        merged_grid_stats['user_count'] = merged_grid_stats['user_count'].fillna(0).astype(int)
        merged_grid_stats['has_users'] = merged_grid_stats['user_count'] > 0
        merged_grid_stats['user_range'] = merged_grid_stats['user_count'].apply(self.get_user_range_category)


        # Add reliability based on Openness_weight_sum
        # Assuming `Openness_weight_sum` is present in `merged_grid_stats`
        # If not present (e.g., if this script runs standalone without prior computation),
        # we'll use a fallback for demo purposes.
        if 'Openness_weight_sum' in merged_grid_stats.columns:
            merged_grid_stats['reliability_weight_sum'] = merged_grid_stats['Openness_weight_sum']
        else:
            # Fallback for demonstration if pre-computed weight_sum is not available.
            # This is NOT the true weight_sum from Ebert's method, but a proxy.
            print("Warning: 'Openness_weight_sum' not found. Using user_count as proxy for reliability_weight_sum.")
            merged_grid_stats['reliability_weight_sum'] = merged_grid_stats['user_count'].copy() # Simple proxy

        merged_grid_stats['reliability_category'] = merged_grid_stats['reliability_weight_sum'].apply(self.get_reliability_category)
        merged_grid_stats['reliability_index'] = merged_grid_stats['reliability_weight_sum'].apply(self.get_reliability_index)


        # Summary statistics by range (user count)
        print(f"\nGrid Statistics Summary:")
        print(f"  Total cells: {len(merged_grid_stats)}")

        for min_val, max_val, label in self.user_ranges:
            if label == 'Empty':
                count = (merged_grid_stats['user_count'] == 0).sum()
            elif max_val == float('inf'):
                count = (merged_grid_stats['user_count'] >= min_val).sum()
            else:
                count = merged_grid_stats['user_count'].between(min_val, max_val).sum()

            percentage = count / len(merged_grid_stats) * 100
            print(f"  {label:10s}: {count:5d} cells ({percentage:5.1f}%)")

        # User count distribution
        cells_with_users = merged_grid_stats[merged_grid_stats['has_users']]
        if len(cells_with_users) > 0:
            user_counts = cells_with_users['user_count']
            print(f"\nUser count distribution (cells with users):")
            print(f"  Mean: {user_counts.mean():.1f}")
            print(f"  Median: {user_counts.median():.1f}")
            print(f"  Range: {user_counts.min()}-{user_counts.max()}")

        # Reliability distribution summary
        print(f"\nReliability (Weight Sum) Distribution Summary:")
        for min_val, max_val, label in self.reliability_ranges:
            if max_val == float('inf'):
                count = (merged_grid_stats['reliability_weight_sum'] >= min_val).sum()
            else:
                # Use left-inclusive for the first bin (0 to 0.001), then exclusive for max for others
                if min_val == 0:
                    count = (merged_grid_stats['reliability_weight_sum'] == 0).sum()
                else:
                    count = merged_grid_stats['reliability_weight_sum'].between(min_val, max_val, inclusive='left').sum()
            percentage = count / len(merged_grid_stats) * 100
            print(f"  {label:25s}: {count:5d} cells ({percentage:5.1f}%)")

        return merged_grid_stats

    def calculate_plot_bounds(self, grid_stats):
        """
        Calculate plot bounds and aspect ratio using projected coordinates.

        Parameters:
        -----------
        grid_stats : geopandas.GeoDataFrame
            Grid statistics with projected coordinates

        Returns:
        --------
        tuple : (aspect_ratio, (x_min, x_max, y_min, y_max))
        """
        if 'x_proj' in grid_stats.columns and 'y_proj' in grid_stats.columns:
            x_min, x_max = grid_stats['x_proj'].min(), grid_stats['x_proj'].max()
            y_min, y_max = grid_stats['y_proj'].min(), grid_stats['y_proj'].max()
        else:
            x_min, x_max = grid_stats['lon_center'].min(), grid_stats['lon_center'].max()
            y_min, y_max = grid_stats['lat_center'].min(), grid_stats['lat_center'].max()

        width = x_max - x_min
        height = y_max - y_min

        if height == 0:
            height = 1
        aspect_ratio = width / height

        return aspect_ratio, (x_min, x_max, y_min, y_max)

    def create_user_range_visualization(self, grid_stats, save_path="user_count_ranges_map.png"):
        """
        Create map showing user count ranges by grid cell using projected coordinates.
        """
        print("Creating user count ranges visualization...")

        # Calculate bounds and aspect ratio
        aspect_ratio, bounds = self.calculate_plot_bounds(grid_stats)
        x_min, x_max, y_min, y_max = bounds

        # Add buffer
        x_buffer = (x_max - x_min) * 0.02
        y_buffer = (y_max - y_min) * 0.02

        # Create figure with proper aspect ratio
        fig_width = 14
        fig_height = fig_width / aspect_ratio
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        fig.suptitle('User Count Ranges by Grid Cell', fontsize=16, fontweight='bold')

        # Use projected geometries if available
        if 'geometry_proj' in grid_stats.columns and self.target_crs:
            # Create temporary GeoDataFrame with projected geometries
            grid_stats_proj = grid_stats.copy()
            grid_stats_proj = gpd.GeoDataFrame(grid_stats_proj,
                                               geometry='geometry_proj',
                                               crs=self.target_crs)

            # Plot each range with its color
            for idx, (min_val, max_val, label) in enumerate(self.user_ranges):
                if label == 'Empty':
                    subset = grid_stats_proj[grid_stats_proj['user_count'] == 0]
                elif max_val == float('inf'):
                    subset = grid_stats_proj[grid_stats_proj['user_count'] >= min_val]
                else:
                    subset = grid_stats_proj[grid_stats_proj['user_count'].between(min_val, max_val)]

                if len(subset) > 0:
                    subset.plot(ax=ax, color=self.range_colors[idx], alpha=0.8,
                              edgecolor='black', linewidth=0.2)
        else:
            # Fallback to scatter plot with WGS84 coordinates
            for idx, (min_val, max_val, label) in enumerate(self.user_ranges):
                if label == 'Empty':
                    subset = grid_stats[grid_stats['user_count'] == 0]
                elif max_val == float('inf'):
                    subset = grid_stats[grid_stats['user_count'] >= min_val]
                else:
                    subset = grid_stats[grid_stats['user_count'].between(min_val, max_val)]

                if len(subset) > 0:
                    ax.scatter(subset['lon_center'], subset['lat_center'],
                             c=self.range_colors[idx], s=50, alpha=0.8,
                             edgecolors='black', linewidths=0.2, marker='s')

        # Add boundary overlay if available (use projected version)
        if self.boundary_gdf_proj is not None:
            self.boundary_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)

        # Set limits and aspect
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
        ax.set_aspect('equal', adjustable='box')

        # Remove axis labels and ticks for cleaner appearance
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

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
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print(f"Saved: {save_path}")

    def create_interactive_map(self, grid_stats, save_path="interactive_user_map.html"):
        """
        Create an interactive HTML map where clicking on grid cells shows user count.
        Uses WGS84 coordinates for web mapping.
        """
        print("Creating interactive user count map...")

        # Calculate map center (using WGS84 coordinates)
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
        grid_group = folium.FeatureGroup(name='Grid Cells (User Count)')

        # Add each grid cell as a clickable polygon
        for idx, row in grid_stats.iterrows():
            # Get color based on user count
            color = self.range_colors[self.get_user_range_index(row['user_count'])]

            # Create polygon coordinates (using WGS84)
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

        # Add boundaries if available (using WGS84 version)
        if self.boundary_gdf_wgs84 is not None:
            boundary_group = folium.FeatureGroup(name='Boundaries')
            for idx, row in self.boundary_gdf_wgs84.iterrows():
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
        print("  - Hover over cells for quick user count tooltip")
        print("  - Use the layer control to toggle grid cells and boundaries")

        return m

    def create_reliability_visualization(self, grid_stats, save_path="uncertainty_map.png"):
        """
        Create a static map showing reliability (based on weight_sum) by grid cell.
        """
        print("Creating static reliability (uncertainty) map...")

        # Calculate bounds and aspect ratio
        aspect_ratio, bounds = self.calculate_plot_bounds(grid_stats)
        x_min, x_max, y_min, y_max = bounds

        # Add buffer
        x_buffer = (x_max - x_min) * 0.02
        y_buffer = (y_max - y_min) * 0.02

        # Create figure with proper aspect ratio
        fig_width = 14
        fig_height = fig_width / aspect_ratio
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        fig.suptitle('Reliability Map (Based on Total Weight Sum)', fontsize=16, fontweight='bold')

        # Use projected geometries if available
        if 'geometry_proj' in grid_stats.columns and self.target_crs:
            grid_stats_proj = grid_stats.copy()
            grid_stats_proj = gpd.GeoDataFrame(grid_stats_proj,
                                               geometry='geometry_proj',
                                               crs=self.target_crs)
            for idx, (min_val, max_val, label) in enumerate(self.reliability_ranges):
                if max_val == float('inf'):
                    subset = grid_stats_proj[grid_stats_proj['reliability_weight_sum'] >= min_val]
                else:
                    # Handle the '0' bin specifically for exact match, then ranges
                    if min_val == 0:
                        subset = grid_stats_proj[grid_stats_proj['reliability_weight_sum'] == 0]
                    else:
                        subset = grid_stats_proj[grid_stats_proj['reliability_weight_sum'].between(min_val, max_val, inclusive='left')]

                if len(subset) > 0:
                    subset.plot(ax=ax, color=self.reliability_colors[idx], alpha=0.8,
                                edgecolor='black', linewidth=0.2)
        else:
            # Fallback for WGS84 coordinates (less ideal for precise maps)
            for idx, (min_val, max_val, label) in enumerate(self.reliability_ranges):
                if max_val == float('inf'):
                    subset = grid_stats[grid_stats['reliability_weight_sum'] >= min_val]
                else:
                    if min_val == 0:
                        subset = grid_stats[grid_stats['reliability_weight_sum'] == 0]
                    else:
                        subset = grid_stats[grid_stats['reliability_weight_sum'].between(min_val, max_val, inclusive='left')]

                if len(subset) > 0:
                    ax.scatter(subset['lon_center'], subset['lat_center'],
                             c=self.reliability_colors[idx], s=50, alpha=0.8,
                             edgecolors='black', linewidths=0.2, marker='s')

        # Add boundary overlay
        if self.boundary_gdf_proj is not None:
            self.boundary_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)

        # Set limits and aspect
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
        ax.set_aspect('equal', adjustable='box')

        # Clean up axes
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)

        # Create legend for reliability
        from matplotlib.patches import Patch
        legend_elements = []
        for idx, (min_val, max_val, label) in enumerate(self.reliability_ranges):
            # Count cells in this range
            if max_val == float('inf'):
                count = (grid_stats['reliability_weight_sum'] >= min_val).sum()
            else:
                if min_val == 0:
                    count = (grid_stats['reliability_weight_sum'] == 0).sum()
                else:
                    count = grid_stats['reliability_weight_sum'].between(min_val, max_val, inclusive='left').sum()

            legend_label = f'{label} ({count} cells)'
            legend_elements.append(Patch(facecolor=self.reliability_colors[idx], label=legend_label))

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1),
                 title='Reliability (Weight Sum)', fontsize=10)

        # Add stats text box
        stats_text = f"Grid Resolution: {self.grid_resolution_km} km\n"
        stats_text += f"Total Grid Cells: {len(grid_stats):,}\n"
        stats_text += f"Cells with Weight Sum > 0: {(grid_stats['reliability_weight_sum'] > 0).sum():,} ({((grid_stats['reliability_weight_sum'] > 0).sum() / len(grid_stats))*100:.1f}%)\n"
        if (grid_stats['reliability_weight_sum'] > 0).any():
            stats_text += f"Mean Weight Sum (valid): {grid_stats[grid_stats['reliability_weight_sum'] > 0]['reliability_weight_sum'].mean():.1f}\n"
            stats_text += f"Max Weight Sum: {grid_stats['reliability_weight_sum'].max():,}"

        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print(f"Saved: {save_path}")


    def create_interactive_reliability_map(self, grid_stats, save_path="interactive_uncertainty_map.html"):
        """
        Create an interactive HTML map showing reliability (weight_sum).
        Uses WGS84 coordinates for web mapping.
        """
        print("Creating interactive reliability (uncertainty) map...")

        center_lat = grid_stats.lat_center.mean()
        center_lon = grid_stats.lon_center.mean()

        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        title_html = '''
        <h3 align="center" style="font-size:20px"><b>Interactive Reliability Map (Weight Sum) - Click on Grid Cells</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        reliability_group = folium.FeatureGroup(name='Reliability (Weight Sum)')

        for idx, row in grid_stats.iterrows():
            color = self.reliability_colors[self.get_reliability_index(row['reliability_weight_sum'])]

            coords = [[row['lat_bottom'], row['lon_left']],
                     [row['lat_bottom'], row['lon_right']],
                     [row['lat_top'], row['lon_right']],
                     [row['lat_top'], row['lon_left']]]

            popup_text = f"""
            <div style='font-family: Arial; font-size: 12px;'>
                <b>Cell ID:</b> {row['cell_id']}<br>
                <b>Weight Sum:</b> {row['reliability_weight_sum']:.2f}<br>
                <b>Reliability:</b> {row['reliability_category']}<br>
                <b>Users (proxy):</b> {row['user_count']}<br>
                <b>Location:</b> ({row['lat_center']:.3f}, {row['lon_center']:.3f})
            </div>
            """

            folium.Polygon(
                locations=coords,
                color='black',
                weight=0.5,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"Weight Sum: {row['reliability_weight_sum']:.2f}"
            ).add_to(reliability_group)

        reliability_group.add_to(m)

        if self.boundary_gdf_wgs84 is not None:
            boundary_group = folium.FeatureGroup(name='Boundaries')
            for idx, row in self.boundary_gdf_wgs84.iterrows():
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

        folium.LayerControl().add_to(m)

        # Add reliability legend
        legend_html = '''
        <div style="position: fixed;
                    bottom: 50px; right: 50px; width: 200px; height: auto;
                    background-color: white; z-index:9999; font-size:12px;
                    border:2px solid grey; border-radius:5px; padding: 10px">
        <p style="margin: 0; font-weight: bold;">Reliability (Weight Sum)</p>
        '''

        for idx, (min_val, max_val, label) in enumerate(self.reliability_ranges):
            color = self.reliability_colors[idx]
            # Count cells in this range
            if max_val == float('inf'):
                count = (grid_stats['reliability_weight_sum'] >= min_val).sum()
            else:
                if min_val == 0:
                    count = (grid_stats['reliability_weight_sum'] == 0).sum()
                else:
                    count = grid_stats['reliability_weight_sum'].between(min_val, max_val, inclusive='left').sum()
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
            <b>Weight Sum > 0:</b> ''' + f"{(grid_stats['reliability_weight_sum'] > 0).sum():,}" + '''<br>
            <b>Grid Res:</b> ''' + f"{self.grid_resolution_km} km" + '''
        </p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        plugins.Fullscreen(
            position='topleft',
            title='Fullscreen',
            title_cancel='Exit Fullscreen',
            force_separate_button=True
        ).add_to(m)

        m.save(save_path)
        print(f"Saved interactive reliability map: {save_path}")
        print("Open this HTML file in a web browser to interact with the map!")
        print("  - Click on any grid cell to see its weight sum and reliability category")
        print("  - Hover over cells for quick weight sum tooltip")

        return m


def main():
    """
    Main function to run the spatial data analysis including uncertainty map generation.
    """

    # Configuration
    # IMPORTANT: 'spatial_personality_grid_results.csv' is the expected output from
    # running 'spatial_grid_computer.py'. Ensure this file exists and contains
    # the '_weight_sum' columns.
    # If not, the script will generate dummy data for the input and approximate
    # weight_sum with user_count for demonstration purposes, which is not
    # the true Ebert et al. weight_sum.
    input_data_file = "final_users_for_spatial_visualization.csv"
    grid_results_file = "spatial_personality_grid_results.csv" # Output from spatial_grid_computer.py
    shapefile_path = "german_shapefile/de.shp" # Path to your shapefile
    grid_resolution = 5  # Grid resolution in kilometers

    print("USER COUNT RANGES & RELIABILITY MAP ANALYSIS")
    print("="*60)
    print("This analysis creates static and interactive maps for:")
    print("  1. User count ranges per grid cell")
    print("  2. Reliability (Uncertainty) based on total weight sum per grid cell")
    print("="*60)

    # Load initial user data for grid creation
    print("\nLoading initial user data (for grid bounds and user assignment)...")
    try:
        try:
            data = pd.read_csv(input_data_file)
        except FileNotFoundError:
            print(f"Warning: '{input_data_file}' not found. Generating dummy data for grid bounds.")
            np.random.seed(42) # for reproducibility
            num_users = 1000
            dresden_lat, dresden_lon = 51.0504, 13.7373
            data = pd.DataFrame({
                'user_guid': [f'user_{i:04d}' for i in range(num_users)],
                'latitude': dresden_lat + np.random.normal(0, 0.1, num_users),
                'longitude': dresden_lon + np.random.normal(0, 0.1, num_users)
            })
            data.to_csv(input_data_file, index=False)
            print(f"Dummy user data saved to '{input_data_file}'.")

        required_cols = ['latitude', 'longitude', 'user_guid']
        data = data.dropna(subset=['latitude', 'longitude'])
        print(f"Loaded {len(data):,} users with valid coordinates for initial grid setup.")

    except Exception as e:
        print(f"Error loading initial data: {e}. Please ensure '{input_data_file}' is correctly formatted and accessible.")
        return

    # Initialize analyzer
    analyzer = SpatialDataAnalyzer(grid_resolution_km=grid_resolution,
                                  shapefile_path=shapefile_path)

    # Step 1: Create explicit grid in WGS84
    print("\nStep 1: Creating explicit grid structure...")
    grid_cells = analyzer.create_explicit_grid(data)

    # Step 2: Project grid cells for accurate visualization
    print("\nStep 2: Projecting grid cells...")
    grid_cells = analyzer.project_grid_cells(grid_cells)

    # Step 3: Assign users to grid cells (for user_count based stats)
    print("\nStep 3: Assigning users to grid cells...")
    data_with_cells = analyzer.assign_users_to_grid(data, grid_cells)

    # Step 4: Load pre-computed grid results for reliability (weight_sum)
    print(f"\nStep 4: Loading pre-computed grid results from '{grid_results_file}' for reliability...")
    try:
        # This file is expected to contain 'grid_idx' and '_weight_sum' columns.
        computed_grid_results = pd.read_csv(grid_results_file)
        print(f"Loaded {len(computed_grid_results):,} pre-computed grid cells with personality scores.")
        
        # Ensure grid_cells has a 'grid_idx' that can be used for merging
        # and that the 'geometry_proj' column is not dropped during merge.
        
        # Merge the computed grid results (which contain weight sums) with our grid_cells.
        # This is the crucial step to get the 'Openness_weight_sum' onto the grid_cells dataframe.
        # We need to preserve original grid_cells structure including `geometry` and `geometry_proj`
        
        # Create a clean version of grid_cells for merging
        grid_cells_for_merge = grid_cells[['cell_id', 'lat_center', 'lon_center',
                                            'lat_bottom', 'lat_top', 'lon_left', 'lon_right',
                                            'x_proj', 'y_proj']].copy()
        
        # Merge based on grid_idx or cell_id, assuming `computed_grid_results` has `grid_idx`
        # and it corresponds to `cell_id` after clipping.
        # The `spatial_grid_computer.py` uses `grid_idx`.
        # So we should use `grid_idx` for merging with `computed_grid_results`
        
        # Ensure `computed_grid_results`'s 'grid_idx' aligns with `grid_cells`'s 'cell_id'
        # If `spatial_grid_computer.py` re-indexes after clipping, this might need more robust matching.
        # For simplicity, assuming `grid_idx` in `computed_grid_results` refers to our `cell_id`.

        # Perform left merge to keep all grid cells, filling NaNs for cells with no computed data.
        # This `grid_stats` dataframe will now contain `Openness_weight_sum` and other traits.
        grid_stats = pd.merge(grid_cells, computed_grid_results, left_on='cell_id', right_on='grid_idx', how='left', suffixes=('_grid', '_computed'))
        
        # Fill NaN values for all relevant columns for cells that had no data in computed_grid_results
        # For personality traits and weight_sums, fill with 0 or NaN
        personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        for trait in personality_traits:
            grid_stats[trait] = grid_stats[trait].fillna(np.nan) # Keep actual scores as NaN if no data
            grid_stats[f'{trait}_weight_sum'] = grid_stats[f'{trait}_weight_sum'].fillna(0.0)
            grid_stats[f'{trait}_z'] = grid_stats[f'{trait}_z'].fillna(np.nan)


        # Now add user count and ranges to this merged grid_stats
        user_counts_per_cell = data_with_cells.groupby('cell_id').size().reset_index(name='user_count')
        grid_stats = pd.merge(grid_stats, user_counts_per_cell, on='cell_id', how='left')
        grid_stats['user_count'] = grid_stats['user_count'].fillna(0).astype(int)
        grid_stats['has_users'] = grid_stats['user_count'] > 0
        grid_stats['user_range'] = grid_stats['user_count'].apply(analyzer.get_user_range_category)

        # Set the reliability_weight_sum from the actual computed value
        grid_stats['reliability_weight_sum'] = grid_stats['Openness_weight_sum'] # Use Openness's weight sum as the primary reliability metric
        grid_stats['reliability_category'] = grid_stats['reliability_weight_sum'].apply(analyzer.get_reliability_category)
        grid_stats['reliability_index'] = grid_stats['reliability_weight_sum'].apply(analyzer.get_reliability_index)

    except FileNotFoundError:
        print(f"Error: Could not find '{grid_results_file}'.")
        print("This file is crucial for the true reliability map (weight_sum).")
        print("Please run `spatial_grid_computer.py` first to generate it.")
        print("Proceeding with a simplified reliability proxy (user count) for demonstration.")

        # Fallback if the pre-computed file is not found
        # In this fallback, grid_stats will be based only on grid_cells + user_counts.
        # The reliability_weight_sum will be approximated by user_count.
        grid_stats = grid_cells.copy()
        user_counts_per_cell = data_with_cells.groupby('cell_id').size().reset_index(name='user_count')
        grid_stats = pd.merge(grid_stats, user_counts_per_cell, on='cell_id', how='left')
        grid_stats['user_count'] = grid_stats['user_count'].fillna(0).astype(int)
        grid_stats['has_users'] = grid_stats['user_count'] > 0
        grid_stats['user_range'] = grid_stats['user_count'].apply(analyzer.get_user_range_category)
        grid_stats['reliability_weight_sum'] = grid_stats['user_count'].copy() # Proxy!
        grid_stats['reliability_category'] = grid_stats['reliability_weight_sum'].apply(analyzer.get_reliability_category)
        grid_stats['reliability_index'] = grid_stats['reliability_weight_sum'].apply(analyzer.get_reliability_index)


    # Step 5: Calculate grid statistics (now includes reliability summaries)
    print("\nStep 5: Calculating grid cell statistics (including reliability summaries)...")
    # This step will just print summaries since we've already prepared grid_stats
    analyzer.calculate_grid_statistics(data_with_cells, grid_stats) # Pass grid_stats with weight_sums

    # Step 6: Create visualizations (static and interactive for both user counts and reliability)
    print("\nStep 6: Creating visualizations...")
    # Static map for user count ranges
    analyzer.create_user_range_visualization(grid_stats, save_path="user_count_ranges_map.png")
    # Interactive map for user count ranges
    analyzer.create_interactive_map(grid_stats, save_path="interactive_user_map.html")

    # Static uncertainty map based on weight_sum
    analyzer.create_reliability_visualization(grid_stats, save_path="uncertainty_map.png")
    # Interactive uncertainty map based on weight_sum
    analyzer.create_interactive_reliability_map(grid_stats, save_path="interactive_uncertainty_map.html")

    # Step 7: Save final grid_stats for further analysis (optional, includes reliability)
    print("\nStep 7: Saving results...")
    try:
        # Save as shapefile (in WGS84 for compatibility) - now includes reliability data
        grid_stats_wgs84 = grid_stats.copy()
        # Ensure 'geometry' is the active geometry column for saving
        # If 'geometry_proj' was used, convert it back or ensure original 'geometry' is correct.
        grid_stats_wgs84 = gpd.GeoDataFrame(grid_stats_wgs84, geometry='geometry', crs='EPSG:4326')
        grid_stats_wgs84.to_file("spatial_grid_analysis_with_reliability.shp")
        print("✅ Grid analysis (with reliability) saved to: spatial_grid_analysis_with_reliability.shp")
    except Exception as e:
        print(f"Note: Could not save shapefile: {e}. Ensure geopandas can write this format.")

    # Save assignments and enriched grid_stats to CSV
    data_with_cells.to_csv("users_with_grid_assignments.csv", index=False)
    print("✅ User assignments saved to: users_with_grid_assignments.csv")
    grid_stats.to_csv("spatial_grid_analysis_full_results.csv", index=False)
    print("✅ Full grid analysis (including reliability) saved to: spatial_grid_analysis_full_results.csv")


    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Files created:")
    print("✅ user_count_ranges_map.png (static map of user counts)")
    print("✅ interactive_user_map.html (interactive map of user counts - open in browser)")
    print("✅ uncertainty_map.png (static map of reliability based on weight_sum)")
    print("✅ interactive_uncertainty_map.html (interactive map of reliability - open in browser)")
    print("✅ users_with_grid_assignments.csv")
    print("✅ spatial_grid_analysis_full_results.csv")
    print("✅ spatial_grid_analysis_with_reliability.shp (if shapefile export successful)")
    print("\n📍 INTERACTIVE MAP FEATURES:")
    print("   • Click on any grid cell to see detailed user count or weight sum/reliability")
    print("   • Hover over cells for quick tooltips")
    print("   • Toggle layers on/off using the layer control")
    print("   • Use fullscreen mode for better viewing")
    print("\nVisualization shows user density and data reliability.")

if __name__ == "__main__":
    main()
