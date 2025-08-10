import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

class SpatialDataStatsAnalyzer:
    """
    Analyzes spatial user distribution and provides detailed statistics.
    Focuses on summarizing grid cell properties based on user counts.
    """

    def __init__(self, grid_resolution_km=5, shapefile_path=None):
        """
        Initializes the statistical analyzer with a grid resolution.

        Parameters:
        ----------
        grid_resolution_km : float
            Resolution of the grid cells in kilometers.
        shapefile_path : str
            Path to shapefile for boundary clipping (preferably in projected CRS).
            If None, no clipping will be performed.
        """
        self.grid_resolution_km = grid_resolution_km
        self.shapefile_path = shapefile_path

        # Store both projected (for plotting) and WGS84 (for spatial joins) versions
        self.boundary_gdf_proj = None  # Native/projected CRS for accurate plotting
        self.boundary_gdf_wgs84 = None  # WGS84 for spatial joins with lat/lon data
        self.target_crs = None  # Target CRS for projection

        # Define user count ranges for categorization
        self.user_ranges = [
            (0, 0, 'Empty'),
            (1, 19, '1-19'),
            (20, 39, '20-39'),
            (40, 59, '40-59'),
            (60, 79, '60-79'),
            (80, 99, '80-99'),
            (100, float('inf'), '100+')
        ]

        # Define colors for visualization (consistent with previous script)
        self.range_colors = [
            '#fefefe',      # 0 (Empty)
            '#fee5d9',      # 1-19
            '#fcae91',      # 20-39
            '#fb6a4a',      # 40-59
            '#de2d26',      # 60-79
            '#a50f15',      # 80-99
            '#67000d'       # 100+
        ]

        # Load shapefile if provided
        if shapefile_path:
            self.load_boundary_shapefile()

        print(f"Spatial Data Statistics Analyzer initialized with {grid_resolution_km} km resolution.")
        if self.shapefile_path:
            print(f"  - Shapefile: {'Loaded' if self.boundary_gdf_proj is not None else 'Not provided/Loaded with error'}")
            if self.target_crs:
                print(f"  - Target CRS: {self.target_crs}")

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
        Creates explicit grid cells covering the data extent in WGS84 (lat/lon).
        Clips the grid cells to the loaded shapefile boundary if provided.
        """
        print("Creating explicit grid structure...")

        # Calculate grid bounds with a small buffer
        lat_min, lat_max = data['latitude'].min(), data['latitude'].max()
        lon_min, lon_max = data['longitude'].min(), data['longitude'].max()

        buffer_deg = 0.1 # Small buffer in degrees
        lat_min, lat_max = lat_min - buffer_deg, lat_max + buffer_deg
        lon_min, lon_max = lon_min - buffer_deg, lon_max + buffer_deg

        # Convert grid resolution from km to degrees (approximate)
        # 1 degree latitude is approx 111 km
        lat_step = self.grid_resolution_km / 111.0
        # Longitude step varies with latitude, using average latitude
        avg_lat = (lat_min + lat_max) / 2
        lon_step = self.grid_resolution_km / (111.0 * np.cos(np.radians(avg_lat)))

        # Create grid coordinates
        lats = np.arange(lat_min, lat_max + lat_step, lat_step)
        lons = np.arange(lon_min, lon_max + lon_step, lon_step)

        grid_cells = []
        cell_id = 0

        # Generate grid polygons
        for i in range(len(lats) - 1):
            for j in range(len(lons) - 1):
                lat_bottom, lat_top = lats[i], lats[i + 1]
                lon_left, lon_right = lons[j], lons[j + 1]

                cell_polygon = Polygon([
                    (lon_left, lat_bottom), (lon_right, lat_bottom),
                    (lon_right, lat_top), (lon_left, lat_top)
                ])

                grid_cells.append({
                    'cell_id': cell_id,
                    'lat_center': (lat_bottom + lat_top) / 2,
                    'lon_center': (lon_left + lon_right) / 2,
                    'geometry': cell_polygon
                })
                cell_id += 1

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

        print(f"Created {len(grid_gdf)} grid cells.")
        return grid_gdf

    def assign_users_to_grid(self, data, grid_cells):
        """
        Assigns each user to their corresponding grid cell using WGS84 coordinates.
        """
        print("Assigning users to grid cells...")

        # Create GeoDataFrame for users
        geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
        users_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')

        # Spatial join
        users_with_cells = gpd.sjoin(users_gdf, grid_cells[['cell_id', 'geometry']],
                                     how='left', predicate='within')

        data_with_cells = users_with_cells.drop('geometry', axis=1)

        assigned_users = data_with_cells['cell_id'].notna().sum()
        total_users = len(data_with_cells)
        print(f"Assigned {assigned_users} out of {total_users} users to grid cells.")

        return data_with_cells

    def get_user_range_category(self, user_count):
        """Helper to get the range label for a given user count."""
        for min_val, max_val, label in self.user_ranges:
            if min_val <= user_count <= max_val:
                return label
        return 'Unknown' # Should not happen with well-defined ranges

    def calculate_grid_statistics(self, data_with_cells, grid_cells):
        """
        Calculates user counts and basic statistics for each grid cell.
        """
        print("Calculating grid cell statistics...")

        # Count users per cell
        user_counts_per_cell = data_with_cells.groupby('cell_id').size().reset_index(name='user_count')

        # Merge with all grid cells to include empty cells
        grid_stats = grid_cells.merge(user_counts_per_cell, on='cell_id', how='left')
        grid_stats['user_count'] = grid_stats['user_count'].fillna(0).astype(int)

        # Add has_users and user_range columns
        grid_stats['has_users'] = grid_stats['user_count'] > 0
        grid_stats['user_range'] = grid_stats['user_count'].apply(self.get_user_range_category)

        print("Grid cell statistics calculated.")
        return grid_stats

    def display_statistics(self, grid_stats):
        """
        Displays a summary table of the calculated spatial statistics.
        """
        print("\n" + "="*50)
        print("SPATIAL DISTRIBUTION STATISTICS")
        print("="*50)

        total_cells = len(grid_stats)
        cells_with_users = grid_stats['has_users'].sum()
        empty_cells = total_cells - cells_with_users

        # Calculate percentages
        perc_cells_with_users = (cells_with_users / total_cells) * 100
        perc_empty_cells = (empty_cells / total_cells) * 100

        print(f"Total Grid Cells: {total_cells:,}")
        print(f"Cells with Users: {cells_with_users:,} ({perc_cells_with_users:.1f}%)")
        print(f"Empty Cells:      {empty_cells:,} ({perc_empty_cells:.1f}%)")

        cells_with_users_data = grid_stats[grid_stats['has_users']]['user_count']

        if not cells_with_users_data.empty:
            max_users = cells_with_users_data.max()
            mean_users = cells_with_users_data.mean()
            median_users = cells_with_users_data.median()
            min_users = cells_with_users_data.min()

            print(f"\nUser Count per Cell (Non-Empty Cells):")
            print(f"  Max Users in a Single Cell: {max_users:,}")
            print(f"  Min Users in a Single Cell: {min_users:,}")
            print(f"  Mean Users per Cell:      {mean_users:.1f}")
            print(f"  Median Users per Cell:    {median_users:.1f}")
        else:
            print("\nNo cells contain users.")

        print("\n" + "="*50)
        print("DISTRIBUTION BY USER COUNT RANGE")
        print("="*50)

        # Group by user range and count cells
        range_counts = grid_stats['user_range'].value_counts().reindex([r[2] for r in self.user_ranges], fill_value=0)
        range_percentages = (range_counts / total_cells) * 100

        range_summary = pd.DataFrame({
            'Cells': range_counts,
            'Percentage': range_percentages.map('{:.1f}%'.format)
        })
        print(range_summary.to_string())
        print("\n" + "="*50)

    def generate_charts(self, grid_stats):
        """
        Generates and displays charts for spatial statistics.
        """
        print("\nGenerating charts...")

        # --- Chart 1: Distribution of Cells by User Count Range ---
        plt.figure(figsize=(10, 6))
        range_counts = grid_stats['user_range'].value_counts().reindex([r[2] for r in self.user_ranges], fill_value=0)

        # Map range labels to their corresponding colors
        ordered_colors = [self.range_colors[self.user_ranges.index(next(filter(lambda r: r[2] == label, self.user_ranges)))] for label in range_counts.index]

        bars = plt.bar(range_counts.index, range_counts.values, color=ordered_colors, edgecolor='black')

        plt.title('Distribution of Grid Cells by User Count Range', fontsize=16)
        plt.xlabel('User Count Range', fontsize=12)
        plt.ylabel('Number of Grid Cells', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Add value labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.05 * yval, f'{int(yval)}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig('cells_by_user_range_chart.png', dpi=300)
        plt.show()
        print("Saved: cells_by_user_range_chart.png")

        # --- Chart 2: Histogram of User Counts in Non-Empty Cells ---
        cells_with_users_data = grid_stats[grid_stats['has_users']]['user_count']

        if not cells_with_users_data.empty:
            plt.figure(figsize=(10, 6))
            # Determine appropriate bin edges for the histogram based on user ranges
            # Exclude the 'Empty' range (0-0) for this histogram
            bins = [0] + [r[1] for r in self.user_ranges if r[1] != float('inf')]
            # Add a final bin for the maximum value
            bins.append(cells_with_users_data.max() + self.user_ranges[-1][1] / 10) # Small buffer for max bin

            plt.hist(cells_with_users_data, bins=bins, edgecolor='black', color='skyblue', alpha=0.8)
            plt.title('Histogram of User Counts in Non-Empty Grid Cells', fontsize=16)
            plt.xlabel('Number of Users per Cell', fontsize=12)
            plt.ylabel('Frequency (Number of Cells)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig('user_counts_histogram.png', dpi=300)
            plt.show()
            print("Saved: user_counts_histogram.png")
        else:
            print("No non-empty cells to generate user counts histogram.")

def main():
    """
    Main function to run the spatial data statistics analysis.
    """
    # --- Configuration ---
    # IMPORTANT: Replace 'final_users_for_spatial_visualization.csv' with your actual data file.
    # The file should contain 'latitude' and 'longitude' columns.
    data_file = "final_users_for_spatial_visualization.csv"
    grid_resolution = 5  # Grid resolution in kilometers (e.g., 5km)
    shapefile_path = "german_shapefile/de.shp"  # Path to your shapefile (e.g., boundaries of a country/region)

    print("Starting Spatial Data Statistics Analysis...")
    print(f"Expected data file: {data_file}")
    print(f"Grid Resolution: {grid_resolution} km")
    print(f"Shapefile Path: {shapefile_path}")

    # --- Load Data ---
    try:
        # Create dummy data if the file doesn't exist for demonstration purposes
        try:
            data = pd.read_csv(data_file)
        except FileNotFoundError:
            print(f"Warning: '{data_file}' not found. Generating dummy data for demonstration.")
            # Generate dummy data for 1000 users around a central point (e.g., Dresden, Germany)
            np.random.seed(42) # for reproducibility
            num_users = 1000

            # Approximate coordinates for Dresden, Germany
            dresden_lat, dresden_lon = 51.0504, 13.7373

            data = pd.DataFrame({
                'user_guid': [f'user_{i:04d}' for i in range(num_users)],
                'latitude': dresden_lat + np.random.normal(0, 0.1, num_users), # Spread within ~11km
                'longitude': dresden_lon + np.random.normal(0, 0.1, num_users) # Spread within ~7km
            })
            # Add some concentrated areas to create varying user counts
            data.loc[:50, 'latitude'] = dresden_lat + 0.01 + np.random.normal(0, 0.005, 51)
            data.loc[:50, 'longitude'] = dresden_lon + 0.01 + np.random.normal(0, 0.005, 51)
            data.loc[100:120, 'latitude'] = dresden_lat - 0.02 + np.random.normal(0, 0.002, 21)
            data.loc[100:120, 'longitude'] = dresden_lon - 0.02 + np.random.normal(0, 0.002, 21)
            data.to_csv(data_file, index=False) # Save dummy data for future runs
            print(f"Dummy data saved to '{data_file}'.")

        # Basic data cleaning
        initial_count = len(data)
        data = data.dropna(subset=['latitude', 'longitude'])
        if len(data) < initial_count:
            print(f"Removed {initial_count - len(data)} rows with missing coordinates.")
        print(f"Loaded {len(data):,} users with valid coordinates.")

    except Exception as e:
        print(f"Error loading data: {e}. Please ensure the CSV is correctly formatted and accessible.")
        return

    # --- Initialize Analyzer ---
    analyzer = SpatialDataStatsAnalyzer(grid_resolution_km=grid_resolution,
                                        shapefile_path=shapefile_path)

    # --- Perform Analysis Steps ---
    grid_cells = analyzer.create_explicit_grid(data)
    data_with_cells = analyzer.assign_users_to_grid(data, grid_cells)
    grid_stats = analyzer.calculate_grid_statistics(data_with_cells, grid_cells)

    # --- Display Results ---
    analyzer.display_statistics(grid_stats)
    analyzer.generate_charts(grid_stats)

    print("\nAnalysis Complete!")
    print("Check the generated PNG files for charts and console for table summary.")

if __name__ == "__main__":
    main()
