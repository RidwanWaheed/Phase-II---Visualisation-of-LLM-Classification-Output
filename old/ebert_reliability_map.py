"""
Ebert Methodology Reliability/Uncertainty Visualization
Based on weight_sum values from spatial interpolation

This script creates reliability maps showing the confidence/uncertainty
of personality trait estimates based on the weighted sum of user contributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import folium
from folium import plugins
import warnings
warnings.filterwarnings('ignore')

class EbertReliabilityVisualizer:
    """
    Visualize reliability/uncertainty of Ebert methodology results
    based on weight_sum values.
    """
    
    def __init__(self, shapefile_path=None):
        """
        Initialize the reliability visualizer.
        
        Parameters:
        -----------
        shapefile_path : str, optional
            Path to boundary shapefile for visualization
        """
        # Store both projected and WGS84 versions of shapefile
        self.boundary_gdf_proj = None  # Native/projected CRS for plotting
        self.boundary_gdf_wgs84 = None  # WGS84 for spatial operations
        self.target_crs = None
        
        # Define reliability categories and colors
        self.reliability_categories = [
            (0.0, 0.0, 'No Data', '#ffffff'),      # White
            (0.0, 0.2, 'Very Low', '#d73027'),     # Red
            (0.2, 0.4, 'Low', '#fc8d59'),          # Light red
            (0.4, 0.6, 'Medium', '#fee08b'),       # Yellow
            (0.6, 0.8, 'High', '#a6d96a'),         # Light green
            (0.8, 1.0, 'Very High', '#1a9850')     # Dark green
        ]
        
        if shapefile_path:
            self.load_shapefile(shapefile_path)
        
        print("="*60)
        print("EBERT METHODOLOGY RELIABILITY VISUALIZER")
        print("="*60)
        print(f"Boundary shapefile: {'Loaded' if self.boundary_gdf_proj is not None else 'Not provided'}")
        print("Ready to analyze weight_sum reliability")
        print("="*60)
    
    def load_shapefile(self, shapefile_path):
        """Load shapefile in native CRS and create WGS84 version."""
        try:
            print(f"Loading shapefile: {shapefile_path}")
            # Load in native/projected CRS for accurate plotting
            self.boundary_gdf_proj = gpd.read_file(shapefile_path)
            self.target_crs = self.boundary_gdf_proj.crs
            print(f"  - Loaded in native CRS: {self.target_crs}")
            
            # Create WGS84 version for web mapping
            if self.boundary_gdf_proj.crs != 'EPSG:4326':
                self.boundary_gdf_wgs84 = self.boundary_gdf_proj.to_crs('EPSG:4326')
            else:
                self.boundary_gdf_wgs84 = self.boundary_gdf_proj.copy()
                
            print(f"  - Loaded {len(self.boundary_gdf_proj)} regions")
            
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            self.boundary_gdf_proj = None
            self.boundary_gdf_wgs84 = None
            self.target_crs = None
    
    def load_and_process_grid_data(self, grid_results_file):
        """
        Load grid results and process weight_sum for reliability analysis.
        
        Parameters:
        -----------
        grid_results_file : str
            Path to CSV file with Ebert methodology results
            
        Returns:
        --------
        pandas.DataFrame : Processed grid data with reliability metrics
        """
        print(f"\nLoading grid results: {grid_results_file}")
        
        try:
            data = pd.read_csv(grid_results_file)
            
            # Check for required columns
            required_cols = ['grid_lon', 'grid_lat']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Find weight_sum column (use first trait's weight_sum as they're all the same)
            weight_cols = [col for col in data.columns if col.endswith('_weight_sum')]
            if not weight_cols:
                raise ValueError("No weight_sum columns found in data")
            
            # Use first weight_sum column (all should be identical)
            weight_col = weight_cols[0]
            print(f"  - Using weight column: {weight_col}")
            
            # Copy weight_sum to standard column
            data['weight_sum'] = data[weight_col]
            
            # Basic statistics
            print(f"  - Grid points: {len(data):,}")
            print(f"  - Weight_sum range: {data['weight_sum'].min():.2f} to {data['weight_sum'].max():.2f}")
            
            # Count data coverage
            has_data = data['weight_sum'] > 0
            coverage = has_data.sum() / len(data) * 100
            print(f"  - Data coverage: {has_data.sum():,}/{len(data):,} points ({coverage:.1f}%)")
            
            # Normalize weight_sum (0-1 scale)
            # Only normalize non-zero values
            non_zero_mask = data['weight_sum'] > 0
            if non_zero_mask.any():
                min_weight = data.loc[non_zero_mask, 'weight_sum'].min()
                max_weight = data.loc[non_zero_mask, 'weight_sum'].max()
                
                # Initialize normalized column
                data['weight_normalized'] = 0.0
                
                # Normalize only non-zero values
                if max_weight > min_weight:
                    data.loc[non_zero_mask, 'weight_normalized'] = (
                        (data.loc[non_zero_mask, 'weight_sum'] - min_weight) / 
                        (max_weight - min_weight)
                    )
                else:
                    # All non-zero values are the same
                    data.loc[non_zero_mask, 'weight_normalized'] = 1.0
            else:
                data['weight_normalized'] = 0.0
            
            # Assign reliability categories
            data['reliability_category'] = 'No Data'
            data['reliability_index'] = 0
            
            for idx, (min_val, max_val, category, color) in enumerate(self.reliability_categories):
                if category == 'No Data':
                    mask = data['weight_sum'] == 0
                else:
                    mask = (data['weight_normalized'] > min_val) & (data['weight_normalized'] <= max_val)
                
                data.loc[mask, 'reliability_category'] = category
                data.loc[mask, 'reliability_index'] = idx
            
            # Project coordinates if CRS available
            if self.target_crs and self.target_crs != 'EPSG:4326':
                print(f"  - Projecting coordinates to {self.target_crs}")
                geometry = [Point(lon, lat) for lon, lat in zip(data['grid_lon'], data['grid_lat'])]
                gdf_wgs84 = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
                gdf_proj = gdf_wgs84.to_crs(self.target_crs)
                data['x_proj'] = gdf_proj.geometry.x
                data['y_proj'] = gdf_proj.geometry.y
            else:
                data['x_proj'] = data['grid_lon']
                data['y_proj'] = data['grid_lat']
            
            # Print category distribution
            print("\n  Reliability Distribution:")
            for category in ['No Data', 'Very Low', 'Low', 'Medium', 'High', 'Very High']:
                count = (data['reliability_category'] == category).sum()
                pct = count / len(data) * 100
                print(f"    {category:10s}: {count:6,} points ({pct:5.1f}%)")
            
            return data
            
        except Exception as e:
            print(f"Error loading grid data: {e}")
            raise
    
    def create_reliability_map(self, data, save_path="reliability_map.png"):
        """
        Create static reliability/uncertainty map.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Processed grid data with reliability metrics
        save_path : str
            Path to save the figure
        """
        print("\nCreating reliability map...")
        
        # Calculate aspect ratio from data
        x_min, x_max = data['x_proj'].min(), data['x_proj'].max()
        y_min, y_max = data['y_proj'].min(), data['y_proj'].max()
        
        width = x_max - x_min
        height = y_max - y_min
        if height == 0:
            height = 1
        aspect_ratio = width / height
        
        # Create figure with proper aspect ratio
        fig_width = 12
        fig_height = fig_width / aspect_ratio
        
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
        fig.suptitle('Spatial Reliability Map - Ebert Methodology\n'
                    'Based on Weight Sum of User Contributions',
                    fontsize=16, fontweight='bold')
        
        # Plot each reliability category
        for idx, (min_val, max_val, category, color) in enumerate(self.reliability_categories):
            mask = data['reliability_category'] == category
            subset = data[mask]
            
            if len(subset) > 0:
                ax.scatter(subset['x_proj'], subset['y_proj'], 
                          c=color, s=8, alpha=0.8, label=f'{category} ({len(subset):,})')
        
        # Add boundaries if available
        if self.boundary_gdf_proj is not None:
            self.boundary_gdf_proj.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)
        
        # Set limits with buffer
        x_buffer = width * 0.02
        y_buffer = height * 0.02
        ax.set_xlim(x_min - x_buffer, x_max + x_buffer)
        ax.set_ylim(y_min - y_buffer, y_max + y_buffer)
        ax.set_aspect('equal', adjustable='box')
        
        # Remove axis elements for clean appearance
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(False)
        
        # Add legend
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                 title='Reliability Categories', fontsize=10)
        
        # Add statistics box
        stats_text = self._generate_statistics_text(data)
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.show()
        print(f"  - Saved: {save_path}")
        
        return fig
    
    def create_reliability_distribution(self, data, save_path="reliability_distribution.png"):
        """
        Create distribution plots for weight_sum analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Processed grid data
        save_path : str
            Path to save the figure
        """
        print("\nCreating reliability distribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Reliability Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Histogram of raw weight_sum (excluding zeros)
        ax = axes[0, 0]
        non_zero_weights = data[data['weight_sum'] > 0]['weight_sum']
        if len(non_zero_weights) > 0:
            ax.hist(non_zero_weights, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
            ax.axvline(non_zero_weights.median(), color='red', linestyle='--', 
                      label=f'Median: {non_zero_weights.median():.2f}')
            ax.axvline(non_zero_weights.mean(), color='green', linestyle='--', 
                      label=f'Mean: {non_zero_weights.mean():.2f}')
            ax.set_xlabel('Weight Sum')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Weight Sum (Non-Zero Values)')
            ax.legend()
        
        # 2. Histogram of normalized weights
        ax = axes[0, 1]
        normalized_non_zero = data[data['weight_sum'] > 0]['weight_normalized']
        if len(normalized_non_zero) > 0:
            ax.hist(normalized_non_zero, bins=50, color='darkgreen', alpha=0.7, edgecolor='black')
            
            # Add category boundaries
            for min_val, max_val, category, color in self.reliability_categories[1:]:  # Skip "No Data"
                if min_val > 0:
                    ax.axvline(min_val, color='gray', linestyle=':', alpha=0.5)
                    ax.text(min_val, ax.get_ylim()[1]*0.9, category, 
                           rotation=90, va='top', fontsize=8)
            
            ax.set_xlabel('Normalized Weight (0-1)')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution of Normalized Weights')
        
        # 3. Category bar chart
        ax = axes[1, 0]
        category_counts = data['reliability_category'].value_counts()
        categories_ordered = [cat for _, _, cat, _ in self.reliability_categories]
        counts = [category_counts.get(cat, 0) for cat in categories_ordered]
        colors = [color for _, _, _, color in self.reliability_categories]
        
        bars = ax.bar(range(len(categories_ordered)), counts, color=colors, 
                      alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(categories_ordered)))
        ax.set_xticklabels(categories_ordered, rotation=45, ha='right')
        ax.set_ylabel('Number of Grid Points')
        ax.set_title('Grid Points by Reliability Category')
        
        # Add percentage labels
        total = len(data)
        for bar, count in zip(bars, counts):
            if count > 0:
                pct = count / total * 100
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.005,
                       f'{count:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9)
        
        # 4. Spatial coverage pie chart
        ax = axes[1, 1]
        coverage_data = {
            'With Data': (data['weight_sum'] > 0).sum(),
            'No Data': (data['weight_sum'] == 0).sum()
        }
        
        colors_pie = ['#4CAF50', '#f0f0f0']
        wedges, texts, autotexts = ax.pie(coverage_data.values(), 
                                          labels=coverage_data.keys(),
                                          colors=colors_pie,
                                          autopct=lambda pct: f'{pct:.1f}%\n({int(pct/100*total):,})',
                                          startangle=90)
        ax.set_title('Spatial Data Coverage')
        
        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"  - Saved: {save_path}")
        
        return fig
    
    def create_interactive_reliability_map(self, data, save_path="interactive_reliability_map.html"):
        """
        Create interactive HTML map for reliability exploration.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Processed grid data
        save_path : str
            Path to save HTML file
        """
        print("\nCreating interactive reliability map...")
        
        # Calculate map center
        center_lat = data['grid_lat'].mean()
        center_lon = data['grid_lon'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:20px">
        <b>Interactive Reliability Map - Ebert Methodology</b><br>
        <span style="font-size:14px">Click on points for detailed information</span>
        </h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create feature groups for each reliability category
        feature_groups = {}
        for _, _, category, color in self.reliability_categories:
            fg = folium.FeatureGroup(name=f'{category}', show=True)
            feature_groups[category] = fg
        
        # Add points to respective feature groups
        for idx, row in data.iterrows():
            category = row['reliability_category']
            color = self.reliability_categories[row['reliability_index']][3]
            
            # Create popup text
            popup_text = f"""
            <div style='font-family: Arial; font-size: 12px; width: 200px;'>
                <b>Location:</b> ({row['grid_lat']:.3f}, {row['grid_lon']:.3f})<br>
                <b>Reliability:</b> {category}<br>
                <b>Weight Sum:</b> {row['weight_sum']:.3f}<br>
                <b>Normalized:</b> {row['weight_normalized']:.3f}<br>
            """
            
            # Add method parameters if available
            if 'r_km' in row:
                popup_text += f"<b>Radius:</b> {row['r_km']} km<br>"
            if 's_slope' in row:
                popup_text += f"<b>Slope:</b> {row['s_slope']}<br>"
            
            popup_text += "</div>"
            
            # Determine marker size based on reliability
            if category == 'No Data':
                radius = 2
                fillOpacity = 0.3
            else:
                radius = 3 + row['weight_normalized'] * 3
                fillOpacity = 0.6 + row['weight_normalized'] * 0.3
            
            folium.CircleMarker(
                location=[row['grid_lat'], row['grid_lon']],
                radius=radius,
                popup=folium.Popup(popup_text, max_width=250),
                tooltip=f"{category}: {row['weight_sum']:.2f}",
                color='black',
                fillColor=color,
                fillOpacity=fillOpacity,
                weight=0.5
            ).add_to(feature_groups[category])
        
        # Add all feature groups to map
        for fg in feature_groups.values():
            fg.add_to(m)
        
        # Add boundaries if available
        if self.boundary_gdf_wgs84 is not None:
            boundary_style = {
                'fillColor': 'none',
                'color': 'black',
                'weight': 2,
                'fillOpacity': 0
            }
            
            folium.GeoJson(
                self.boundary_gdf_wgs84.to_json(),
                style_function=lambda x: boundary_style,
                name='Boundaries'
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Create custom legend
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 220px; height: auto; 
                    background-color: white; z-index:9999; font-size:12px;
                    border:2px solid grey; border-radius:5px; padding: 10px">
        <p style="margin: 0; font-weight: bold; font-size: 13px;">Reliability Categories</p>
        <hr style="margin: 5px 0;">
        '''
        
        # Add legend items with counts
        for _, _, category, color in self.reliability_categories:
            count = (data['reliability_category'] == category).sum()
            pct = count / len(data) * 100
            legend_html += f'''
            <p style="margin: 3px 0;">
                <span style="background-color:{color}; 
                           width:15px; height:15px; 
                           display:inline-block; 
                           border:1px solid black;
                           vertical-align: middle;"></span>
                <span style="vertical-align: middle;">
                    {category}: {count:,} ({pct:.1f}%)
                </span>
            </p>
            '''
        
        legend_html += '''
        <hr style="margin: 5px 0;">
        <p style="margin: 3px 0; font-size: 11px;">
            <b>Total Points:</b> ''' + f"{len(data):,}" + '''<br>
            <b>Coverage:</b> ''' + f"{(data['weight_sum'] > 0).sum() / len(data) * 100:.1f}%" + '''
        </p>
        </div>
        '''
        
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Add fullscreen button
        plugins.Fullscreen(
            position='topleft',
            title='Fullscreen',
            title_cancel='Exit Fullscreen'
        ).add_to(m)
        
        # Save map
        m.save(save_path)
        print(f"  - Saved: {save_path}")
        print("  - Open in browser to explore interactively")
        
        return m
    
    def _generate_statistics_text(self, data):
        """Generate statistics text for the map."""
        total_points = len(data)
        with_data = (data['weight_sum'] > 0).sum()
        coverage = with_data / total_points * 100
        
        if with_data > 0:
            non_zero = data[data['weight_sum'] > 0]
            mean_weight = non_zero['weight_sum'].mean()
            median_weight = non_zero['weight_sum'].median()
            std_weight = non_zero['weight_sum'].std()
        else:
            mean_weight = median_weight = std_weight = 0
        
        stats_text = f"Grid Statistics:\n"
        stats_text += f"Total Points: {total_points:,}\n"
        stats_text += f"With Data: {with_data:,} ({coverage:.1f}%)\n"
        stats_text += f"Mean Weight: {mean_weight:.2f}\n"
        stats_text += f"Median Weight: {median_weight:.2f}\n"
        stats_text += f"Std Weight: {std_weight:.2f}"
        
        # Add method parameters if available
        if 'r_km' in data.columns and not data['r_km'].isna().all():
            stats_text += f"\n\nMethod Parameters:\n"
            stats_text += f"Radius: {data['r_km'].iloc[0]} km\n"
            if 's_slope' in data.columns:
                stats_text += f"Slope: {data['s_slope'].iloc[0]}"
        
        return stats_text

def create_reliability_visualizations(grid_results_file, shapefile_path=None, 
                                     output_prefix="ebert_reliability"):
    """
    Main function to create all reliability visualizations.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to CSV file with Ebert methodology results
    shapefile_path : str, optional
        Path to boundary shapefile
    output_prefix : str
        Prefix for output files
        
    Returns:
    --------
    dict : Paths to created files
    """
    print("STARTING EBERT METHODOLOGY RELIABILITY ANALYSIS")
    print("="*70)
    
    # Initialize visualizer
    visualizer = EbertReliabilityVisualizer(shapefile_path=shapefile_path)
    
    # Load and process data
    data = visualizer.load_and_process_grid_data(grid_results_file)
    
    # Create output files
    output_files = {}
    
    # 1. Main reliability map
    map_file = f"{output_prefix}_map.png"
    visualizer.create_reliability_map(data, save_path=map_file)
    output_files['map'] = map_file
    
    # 2. Distribution analysis
    dist_file = f"{output_prefix}_distribution.png"
    visualizer.create_reliability_distribution(data, save_path=dist_file)
    output_files['distribution'] = dist_file
    
    # 3. Interactive map
    interactive_file = f"{output_prefix}_interactive.html"
    visualizer.create_interactive_reliability_map(data, save_path=interactive_file)
    output_files['interactive'] = interactive_file
    
    # 4. Save processed data
    data_file = f"{output_prefix}_data.csv"
    data.to_csv(data_file, index=False)
    output_files['data'] = data_file
    print(f"\n  - Processed data saved: {data_file}")
    
    print("\n" + "="*70)
    print("RELIABILITY ANALYSIS COMPLETE!")
    print("="*70)
    print("Created files:")
    for file_type, file_path in output_files.items():
        print(f"  ✅ {file_type.capitalize()}: {file_path}")
    
    return output_files

if __name__ == "__main__":
    # Configuration
    grid_results_file = "spatial_personality_grid_results.csv"  # Your Ebert results file
    shapefile_path = "german_shapefile/de.shp"  # Your shapefile
    output_prefix = "ebert_reliability"
    
    # Create visualizations
    output_files = create_reliability_visualizations(
        grid_results_file=grid_results_file,
        shapefile_path=shapefile_path,
        output_prefix=output_prefix
    )
    
    print("\nAll reliability visualizations created successfully!")
    print("Check the output files for:")
    print("  - Static reliability map showing spatial confidence")
    print("  - Distribution plots for weight_sum analysis")
    print("  - Interactive HTML map for detailed exploration")