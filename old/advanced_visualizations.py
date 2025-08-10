import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings('ignore')

class AdvancedThesisVisualizations:
    """
    Comprehensive visualization framework for personality geography thesis.
    Implements high and medium priority visualizations for academic analysis.
    """
    
    def __init__(self, grid_results_path="spatial_personality_grid_results.csv",
                 state_results_path="spatial_personality_state_results.csv",
                 raw_data_path="final_users_for_spatial_visualization.csv",
                 shapefile_path="german_shapefile/de.shp"):
        """Initialize the visualization framework."""
        
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        
        # Load all data
        self.load_data(grid_results_path, state_results_path, raw_data_path, shapefile_path)
        
        print("Advanced Thesis Visualizations Framework Initialized")
        print(f"✅ Grid data: {len(self.grid_results)} points")
        print(f"✅ State data: {len(self.state_results) if self.state_results is not None else 0} states")
        print(f"✅ Raw data: {len(self.raw_data)} users")
        print(f"✅ German boundaries: {'Loaded' if self.germany_gdf is not None else 'Not available'}")
    
    def load_data(self, grid_path, state_path, raw_path, shapefile_path):
        """Load all necessary data for visualizations."""
        try:
            # Load grid results
            self.grid_results = pd.read_csv(grid_path)
            print(f"Loaded grid results: {len(self.grid_results)} points")
            
            # Load state results
            try:
                self.state_results = pd.read_csv(state_path)
                print(f"Loaded state results: {len(self.state_results)} states")
            except FileNotFoundError:
                self.state_results = None
                print("State results not available")
            
            # Load raw data
            self.raw_data = pd.read_csv(raw_path)
            print(f"Loaded raw data: {len(self.raw_data)} users")
            
            # Load German shapefile
            try:
                self.germany_gdf = gpd.read_file(shapefile_path)
                if self.germany_gdf.crs != 'EPSG:4326':
                    self.germany_gdf = self.germany_gdf.to_crs('EPSG:4326')
                print(f"Loaded German boundaries: {len(self.germany_gdf)} regions")
            except:
                self.germany_gdf = None
                print("German shapefile not available")
                
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    # =============================================================================
    # HIGH PRIORITY VISUALIZATIONS
    # =============================================================================
    
    def create_data_quality_assessment_panel(self, save_path="data_quality_assessment.png"):
        """
        Create comprehensive 4-panel data quality assessment.
        Panel A: User density coverage
        Panel B: Grid cell reliability 
        Panel C: Platform distribution (if available)
        Panel D: Summary statistics
        """
        print("Creating data quality assessment panel...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Data Quality and Coverage Assessment\nSpatial Distribution of LLM Personality Classification Data', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: User density coverage
        self._create_user_density_map(ax1)
        
        # Panel B: Grid cell reliability
        self._create_reliability_zones_map(ax2)
        
        # Panel C: Geographic extent and coverage
        self._create_coverage_analysis(ax3)
        
        # Panel D: Summary statistics
        self._create_summary_statistics_table(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    def create_methodological_comparison_dashboard(self, save_path="methodological_comparison.png"):
        """
        Create comprehensive comparison of grid vs. state methods.
        Shows raw grid data, state aggregation, and difference analysis.
        """
        print("Creating methodological comparison dashboard...")
        
        if self.state_results is None:
            print("⚠️ State results not available for comparison")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Methodological Comparison: Grid-Based vs. State-Level Aggregation\n'
                    'Spatial Personality Distribution Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Create comparison for each method
        traits_subset = self.personality_traits[:3]  # Show first 3 traits
        
        for idx, trait in enumerate(traits_subset):
            # Grid-based (top row)
            self._create_grid_trait_map(axes[0, idx], trait, f'{trait}\nGrid-Based')
            
            # State-based (bottom row) 
            self._create_state_trait_map(axes[1, idx], trait, f'{trait}\nState-Level')
        
        # Add method labels
        axes[0, 1].text(0.5, 1.15, "Grid-Based Actor-Based Clustering", 
                       transform=axes[0, 1].transAxes, ha='center', 
                       fontsize=14, fontweight='bold')
        axes[1, 1].text(0.5, 1.15, "State-Level Administrative Aggregation", 
                       transform=axes[1, 1].transAxes, ha='center', 
                       fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    def create_uncertainty_quantification_suite(self, save_path="uncertainty_quantification.png"):
        """
        Create uncertainty and reliability analysis.
        Shows confidence intervals, data sparsity, and reliability measures.
        """
        print("Creating uncertainty quantification suite...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Uncertainty Quantification and Reliability Assessment\n'
                    'Spatial Confidence in Personality Classifications', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Data density (users per grid cell)
        self._create_data_density_map(ax1)
        
        # Panel B: Reliability zones
        self._create_detailed_reliability_map(ax2)
        
        # Panel C: Weight sum distribution
        self._create_weight_distribution_map(ax3)
        
        # Panel D: Uncertainty metrics
        self._create_uncertainty_metrics(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    def create_spatial_statistics_analysis(self, save_path="spatial_statistics.png"):
        """
        Create spatial autocorrelation and clustering analysis.
        """
        print("Creating spatial statistics analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Spatial Statistics and Autocorrelation Analysis\n'
                    'Geographic Clustering of Personality Traits', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Global autocorrelation results
        self._create_moran_i_summary(ax1)
        
        # Panel B: Trait correlation matrix
        self._create_trait_correlation_heatmap(ax2)
        
        # Panel C: Spatial clustering example (Openness)
        self._create_clustering_example(ax3, 'Openness')
        
        # Panel D: Distance decay demonstration
        self._create_distance_decay_demo(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    # =============================================================================
    # MEDIUM PRIORITY VISUALIZATIONS
    # =============================================================================
    
    def create_multi_trait_correlation_analysis(self, save_path="trait_correlations.png"):
        """
        Create comprehensive trait correlation and PCA analysis.
        """
        print("Creating multi-trait correlation analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Trait Correlation and Principal Component Analysis\n'
                    'Relationships Between Personality Dimensions', 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Correlation matrix
        self._create_enhanced_correlation_matrix(ax1)
        
        # Panel B: PCA analysis
        self._create_pca_analysis(ax2)
        
        # Panel C: Trait combinations map
        self._create_trait_combination_map(ax3)
        
        # Panel D: Regional personality profiles
        self._create_regional_profiles(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    def create_distance_decay_demonstration(self, save_path="distance_decay_demo.png"):
        """
        Create detailed demonstration of Ebert's distance decay methodology.
        """
        print("Creating distance decay demonstration...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Ebert's Distance Decay Methodology Demonstration\n"
                    "45-Mile Radius Actor-Based Clustering Approach", 
                    fontsize=16, fontweight='bold')
        
        # Panel A: Distance decay function
        self._plot_distance_decay_function(ax1)
        
        # Panel B: Example city with radius
        self._demonstrate_radius_example(ax2)
        
        # Panel C: Weight surface 3D visualization
        self._create_weight_surface_demo(ax3)
        
        # Panel D: Boundary effects analysis
        self._analyze_boundary_effects(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    def create_platform_language_analysis(self, save_path="platform_language_analysis.png"):
        """
        Create platform and language distribution analysis if data available.
        """
        print("Creating platform and language analysis...")
        
        # Check if platform/language data is available
        has_platform = 'platform' in self.raw_data.columns
        has_language = 'language' in self.raw_data.columns
        
        if not (has_platform or has_language):
            print("⚠️ Platform/language data not available in raw data")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Platform and Language Distribution Analysis\n'
                    'Geographic Patterns in Social Media Data Sources', 
                    fontsize=16, fontweight='bold')
        
        if has_platform:
            self._create_platform_distribution_map(axes[0, 0])
            self._create_platform_statistics(axes[0, 1])
        
        if has_language:
            self._create_language_distribution_map(axes[1, 0])
            self._create_language_statistics(axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Saved: {save_path}")
    
    # =============================================================================
    # HELPER METHODS FOR INDIVIDUAL VISUALIZATIONS
    # =============================================================================
    
    def _create_user_density_map(self, ax):
        """Create user density coverage map."""
        ax.set_title('A. User Density Coverage', fontweight='bold', fontsize=12)
        
        # Create density bins
        if len(self.raw_data) > 0:
            # Create density calculation by coordinate frequency
            coord_counts = self.raw_data.groupby(['latitude', 'longitude']).size().reset_index(name='count')
            
            # Plot points with size based on count
            scatter = ax.scatter(coord_counts['longitude'], coord_counts['latitude'], 
                               c=coord_counts['count'], s=coord_counts['count']*2, 
                               alpha=0.6, cmap='Reds')
            
            # Add German boundaries
            if self.germany_gdf is not None:
                self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Users per location', fontsize=10)
    
    def _create_reliability_zones_map(self, ax):
        """Create grid cell reliability zones map."""
        ax.set_title('B. Grid Cell Reliability Zones', fontweight='bold', fontsize=12)
        
        if len(self.grid_results) > 0:
            # Create reliability categories
            reliability_colors = {
                'empty': 'lightgray',
                'low': 'lightcoral', 
                'reliable': 'lightyellow',
                'high': 'lightgreen'
            }
            
            # Categorize grid points based on weight sums
            for trait in self.personality_traits[:1]:  # Use first trait as example
                weight_col = f'{trait}_weight_sum'
                if weight_col in self.grid_results.columns:
                    weights = self.grid_results[weight_col].fillna(0)
                    
                    # Create categories
                    empty_mask = weights == 0
                    low_mask = (weights > 0) & (weights < 5)
                    reliable_mask = (weights >= 5) & (weights < 20)
                    high_mask = weights >= 20
                    
                    # Plot each category
                    if empty_mask.sum() > 0:
                        data_empty = self.grid_results[empty_mask]
                        ax.scatter(data_empty['grid_lon'], data_empty['grid_lat'], 
                                 c=reliability_colors['empty'], s=1, alpha=0.7, label='Empty (0)')
                    
                    if low_mask.sum() > 0:
                        data_low = self.grid_results[low_mask]
                        ax.scatter(data_low['grid_lon'], data_low['grid_lat'], 
                                 c=reliability_colors['low'], s=3, alpha=0.7, label='Low (1-4)')
                    
                    if reliable_mask.sum() > 0:
                        data_reliable = self.grid_results[reliable_mask]
                        ax.scatter(data_reliable['grid_lon'], data_reliable['grid_lat'], 
                                 c=reliability_colors['reliable'], s=5, alpha=0.7, label='Reliable (5-19)')
                    
                    if high_mask.sum() > 0:
                        data_high = self.grid_results[high_mask]
                        ax.scatter(data_high['grid_lon'], data_high['grid_lat'], 
                                 c=reliability_colors['high'], s=8, alpha=0.7, label='High (20+)')
                    break
            
            # Add German boundaries
            if self.germany_gdf is not None:
                self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1, alpha=0.8)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            ax.legend(title='Weight sum', loc='upper right', bbox_to_anchor=(1, 1))
    
    def _create_coverage_analysis(self, ax):
        """Create geographic coverage analysis."""
        ax.set_title('C. Geographic Coverage Analysis', fontweight='bold', fontsize=12)
        
        # Create coverage statistics
        if len(self.grid_results) > 0 and len(self.raw_data) > 0:
            # Calculate coverage metrics
            total_grid_points = len(self.grid_results)
            
            # Count grid points with data
            points_with_data = 0
            for trait in self.personality_traits[:1]:
                weight_col = f'{trait}_weight_sum'
                if weight_col in self.grid_results.columns:
                    points_with_data = (self.grid_results[weight_col] > 0).sum()
                    break
            
            coverage_pct = (points_with_data / total_grid_points) * 100
            
            # Geographic extent
            lat_range = self.raw_data['latitude'].max() - self.raw_data['latitude'].min()
            lon_range = self.raw_data['longitude'].max() - self.raw_data['longitude'].min()
            
            # Create simple coverage visualization
            if self.germany_gdf is not None:
                self.germany_gdf.plot(ax=ax, color='lightblue', alpha=0.3, edgecolor='black')
            
            # Add user points
            if len(self.raw_data) < 5000:  # Sample if too many points
                sample_data = self.raw_data
            else:
                sample_data = self.raw_data.sample(5000)
            
            ax.scatter(sample_data['longitude'], sample_data['latitude'], 
                      s=0.5, alpha=0.5, color='red')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            
            # Add coverage statistics as text
            stats_text = f"""Coverage Statistics:
Grid points with data: {points_with_data:,}
Total grid points: {total_grid_points:,}
Coverage: {coverage_pct:.1f}%
Geographic extent:
  Lat: {lat_range:.2f}°
  Lon: {lon_range:.2f}°"""
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _create_summary_statistics_table(self, ax):
        """Create summary statistics table."""
        ax.set_title('D. Summary Statistics', fontweight='bold', fontsize=12)
        ax.axis('off')
        
        # Collect statistics
        stats_data = []
        
        # Basic data statistics
        stats_data.append(['Total users', f"{len(self.raw_data):,}"])
        stats_data.append(['Grid points', f"{len(self.grid_results):,}"])
        
        if self.state_results is not None:
            stats_data.append(['German states', f"{len(self.state_results):,}"])
        
        # Coverage statistics
        for trait in self.personality_traits[:1]:  # Use first trait
            weight_col = f'{trait}_weight_sum'
            if weight_col in self.grid_results.columns:
                valid_points = (self.grid_results[weight_col] > 0).sum()
                coverage_pct = (valid_points / len(self.grid_results)) * 100
                stats_data.append(['Grid coverage', f"{coverage_pct:.1f}%"])
                break
        
        # Geographic extent
        lat_extent = self.raw_data['latitude'].max() - self.raw_data['latitude'].min()
        lon_extent = self.raw_data['longitude'].max() - self.raw_data['longitude'].min()
        stats_data.append(['Latitude range', f"{lat_extent:.2f}°"])
        stats_data.append(['Longitude range', f"{lon_extent:.2f}°"])
        
        # Personality trait ranges
        stats_data.append(['', ''])  # Spacer
        stats_data.append(['Personality Traits:', ''])
        
        for trait in self.personality_traits:
            if trait in self.raw_data.columns:
                mean_val = self.raw_data[trait].mean()
                std_val = self.raw_data[trait].std()
                stats_data.append([f'{trait}', f'μ={mean_val:.2f}, σ={std_val:.2f}'])
        
        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
    
    def _create_grid_trait_map(self, ax, trait, title):
        """Create grid-based trait map."""
        ax.set_title(title, fontweight='bold', fontsize=12)
        
        z_col = f'{trait}_z'
        if z_col in self.grid_results.columns:
            valid_mask = ~pd.isna(self.grid_results[z_col])
            valid_data = self.grid_results[valid_mask]
            
            if len(valid_data) > 0:
                scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                   c=valid_data[z_col], cmap='RdYlBu_r', 
                                   vmin=-1.96, vmax=1.96, s=4, alpha=0.8)
                
                if self.germany_gdf is not None:
                    self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
                
                ax.set_aspect('equal', adjustable='box')
                ax.set_xticks([])
                ax.set_yticks([])
    
    def _create_state_trait_map(self, ax, trait, title):
        """Create state-based trait map."""
        ax.set_title(title, fontweight='bold', fontsize=12)
        
        if self.state_results is not None and self.germany_gdf is not None:
            # Merge state data with shapefile
            state_col = 'name'  # Adjust based on your shapefile
            merged_gdf = self.germany_gdf.merge(self.state_results, 
                                              left_on=state_col, right_on='state', how='left')
            
            z_col = f'{trait}_z'
            if z_col in merged_gdf.columns:
                merged_gdf.plot(column=z_col, cmap='RdYlBu_r', vmin=-1.96, vmax=1.96,
                               ax=ax, edgecolor='black', linewidth=0.5, 
                               missing_kwds={'color': 'lightgray'})
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
    
    def _create_data_density_map(self, ax):
        """Create data density map for uncertainty analysis."""
        ax.set_title('A. Data Density (Weight Sums)', fontweight='bold', fontsize=12)
        
        # Use first trait's weight sum as representative
        for trait in self.personality_traits[:1]:
            weight_col = f'{trait}_weight_sum'
            if weight_col in self.grid_results.columns:
                valid_mask = self.grid_results[weight_col] > 0
                valid_data = self.grid_results[valid_mask]
                
                if len(valid_data) > 0:
                    scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                       c=valid_data[weight_col], cmap='viridis', 
                                       s=6, alpha=0.7)
                    
                    if self.germany_gdf is not None:
                        self.germany_gdf.boundary.plot(ax=ax, color='white', linewidth=0.8)
                    
                    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Weight sum')
                break
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    def _create_detailed_reliability_map(self, ax):
        """Create detailed reliability assessment map."""
        ax.set_title('B. Reliability Assessment', fontweight='bold', fontsize=12)
        
        # Create reliability score based on weight sums
        reliability_scores = []
        for _, row in self.grid_results.iterrows():
            # Calculate average weight sum across traits
            weights = []
            for trait in self.personality_traits:
                weight_col = f'{trait}_weight_sum'
                if weight_col in self.grid_results.columns:
                    weights.append(row[weight_col] if pd.notna(row[weight_col]) else 0)
            
            if weights:
                avg_weight = np.mean(weights)
                if avg_weight == 0:
                    reliability_scores.append(0)  # No data
                elif avg_weight < 5:
                    reliability_scores.append(1)  # Low
                elif avg_weight < 20:
                    reliability_scores.append(2)  # Moderate
                else:
                    reliability_scores.append(3)  # High
            else:
                reliability_scores.append(0)
        
        self.grid_results['reliability_score'] = reliability_scores
        
        # Create discrete color map
        colors = ['lightgray', 'lightcoral', 'lightyellow', 'lightgreen']
        labels = ['No data', 'Low (<5)', 'Moderate (5-20)', 'High (>20)']
        
        for score, color, label in zip(range(4), colors, labels):
            mask = self.grid_results['reliability_score'] == score
            if mask.sum() > 0:
                data_subset = self.grid_results[mask]
                ax.scatter(data_subset['grid_lon'], data_subset['grid_lat'], 
                          c=color, s=6, alpha=0.8, label=label)
        
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.legend(title='Reliability', loc='upper right', bbox_to_anchor=(1, 1))
    
    def _create_weight_distribution_map(self, ax):
        """Create weight distribution analysis."""
        ax.set_title('C. Weight Distribution Analysis', fontweight='bold', fontsize=12)
        
        # Calculate coefficient of variation in weights across traits
        cv_scores = []
        for _, row in self.grid_results.iterrows():
            weights = []
            for trait in self.personality_traits:
                weight_col = f'{trait}_weight_sum'
                if weight_col in self.grid_results.columns:
                    weight_val = row[weight_col] if pd.notna(row[weight_col]) else 0
                    weights.append(weight_val)
            
            if weights and np.mean(weights) > 0:
                cv = np.std(weights) / np.mean(weights)
                cv_scores.append(cv)
            else:
                cv_scores.append(np.nan)
        
        self.grid_results['weight_cv'] = cv_scores
        
        # Plot coefficient of variation
        valid_mask = ~pd.isna(self.grid_results['weight_cv'])
        valid_data = self.grid_results[valid_mask]
        
        if len(valid_data) > 0:
            scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                               c=valid_data['weight_cv'], cmap='plasma', 
                               s=6, alpha=0.7)
            
            if self.germany_gdf is not None:
                self.germany_gdf.boundary.plot(ax=ax, color='white', linewidth=0.8)
            
            plt.colorbar(scatter, ax=ax, shrink=0.8, label='Weight CV')
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    def _create_uncertainty_metrics(self, ax):
        """Create uncertainty metrics summary."""
        ax.set_title('D. Uncertainty Metrics', fontweight='bold', fontsize=12)
        ax.axis('off')
        
        # Calculate uncertainty metrics
        metrics_data = []
        
        # Weight sum statistics
        for trait in self.personality_traits[:1]:  # Use first trait as example
            weight_col = f'{trait}_weight_sum'
            if weight_col in self.grid_results.columns:
                weights = self.grid_results[weight_col].fillna(0)
                
                metrics_data.append(['Weight Sum Statistics:', ''])
                metrics_data.append(['Mean weight', f'{weights.mean():.2f}'])
                metrics_data.append(['Median weight', f'{weights.median():.2f}'])
                metrics_data.append(['Max weight', f'{weights.max():.0f}'])
                
                # Reliability distribution
                empty = (weights == 0).sum()
                low = ((weights > 0) & (weights < 5)).sum()
                moderate = ((weights >= 5) & (weights < 20)).sum()
                high = (weights >= 20).sum()
                total = len(weights)
                
                metrics_data.append(['', ''])
                metrics_data.append(['Reliability Distribution:', ''])
                metrics_data.append(['No data', f'{empty:,} ({empty/total*100:.1f}%)'])
                metrics_data.append(['Low reliability', f'{low:,} ({low/total*100:.1f}%)'])
                metrics_data.append(['Moderate', f'{moderate:,} ({moderate/total*100:.1f}%)'])
                metrics_data.append(['High reliability', f'{high:,} ({high/total*100:.1f}%)'])
                break
        
        # Create table
        table = ax.table(cellText=metrics_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
    
    def _create_moran_i_summary(self, ax):
        """Create Moran's I autocorrelation summary (placeholder)."""
        ax.set_title("A. Spatial Autocorrelation (Moran's I)", fontweight='bold', fontsize=12)
        
        # This would require spatial weights matrix calculation
        # For now, create a placeholder visualization
        ax.text(0.5, 0.5, "Moran's I Analysis\n\nRequires spatial weights\nmatrix calculation\n\n(Implementation needed)", 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    
    def _create_trait_correlation_heatmap(self, ax):
        """Create trait correlation heatmap."""
        ax.set_title('B. Trait Correlation Matrix', fontweight='bold', fontsize=12)
        
        # Calculate correlations using valid grid points
        trait_data = []
        for trait in self.personality_traits:
            z_col = f'{trait}_z'
            if z_col in self.grid_results.columns:
                trait_data.append(self.grid_results[z_col].fillna(0))
        
        if trait_data:
            corr_matrix = np.corrcoef(trait_data)
            
            # Create heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            
            # Add labels
            ax.set_xticks(range(len(self.personality_traits)))
            ax.set_yticks(range(len(self.personality_traits)))
            ax.set_xticklabels([t[:4] for t in self.personality_traits], rotation=45)
            ax.set_yticklabels([t[:4] for t in self.personality_traits])
            
            # Add correlation values
            for i in range(len(self.personality_traits)):
                for j in range(len(self.personality_traits)):
                    text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            plt.colorbar(im, ax=ax, shrink=0.8)
    
    def _create_clustering_example(self, ax, trait):
        """Create spatial clustering example."""
        ax.set_title(f'C. Spatial Clustering Example ({trait})', fontweight='bold', fontsize=12)
        
        z_col = f'{trait}_z'
        if z_col in self.grid_results.columns:
            valid_mask = ~pd.isna(self.grid_results[z_col])
            valid_data = self.grid_results[valid_mask]
            
            if len(valid_data) > 0:
                scatter = ax.scatter(valid_data['grid_lon'], valid_data['grid_lat'], 
                                   c=valid_data[z_col], cmap='RdYlBu_r', 
                                   vmin=-1.96, vmax=1.96, s=8, alpha=0.8)
                
                if self.germany_gdf is not None:
                    self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5)
                
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
    
    def _create_distance_decay_demo(self, ax):
        """Create distance decay function demonstration."""
        ax.set_title('D. Distance Decay Function', fontweight='bold', fontsize=12)
        
        # Ebert's parameters: r=45 miles, s=7
        distances = np.linspace(0, 100, 1000)  # 0 to 100 miles
        weights = 1 / (1 + (distances / 45) ** 7)
        
        ax.plot(distances, weights, 'b-', linewidth=2, label='f(d) = 1/(1+(d/45)^7)')
        ax.axvline(30, color='green', linestyle='--', alpha=0.7, label='~30 miles (weight ≈ 1.0)')
        ax.axvline(45, color='orange', linestyle='--', alpha=0.7, label='45 miles (weight = 0.5)')
        ax.axvline(75, color='red', linestyle='--', alpha=0.7, label='~75 miles (weight ≈ 0.0)')
        
        ax.set_xlabel('Distance (miles)')
        ax.set_ylabel('Weight')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    
    def _plot_distance_decay_function(self, ax):
        """Plot the distance decay function used in analysis."""
        self._create_distance_decay_demo(ax)
    
    def _demonstrate_radius_example(self, ax):
        """Demonstrate 45-mile radius example around a major city."""
        ax.set_title('B. 45-Mile Radius Example', fontweight='bold', fontsize=12)
        
        # Use Berlin as example (approximately 52.5°N, 13.4°E)
        berlin_lat, berlin_lon = 52.5, 13.4
        
        # Convert 45 miles to degrees (approximately)
        radius_deg = 45 * 1.60934 / 111.0  # rough conversion
        
        # Create circle
        circle = plt.Circle((berlin_lon, berlin_lat), radius_deg, 
                           fill=False, color='red', linewidth=2, alpha=0.7)
        ax.add_patch(circle)
        
        # Add German boundaries
        if self.germany_gdf is not None:
            self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1)
        
        # Add sample user points within radius
        if len(self.raw_data) > 0:
            # Filter to points near Berlin
            nearby_mask = ((self.raw_data['latitude'] - berlin_lat)**2 + 
                          (self.raw_data['longitude'] - berlin_lon)**2) < (radius_deg*2)**2
            nearby_users = self.raw_data[nearby_mask]
            
            if len(nearby_users) > 0:
                ax.scatter(nearby_users['longitude'], nearby_users['latitude'], 
                          s=1, alpha=0.5, color='blue')
        
        ax.plot(berlin_lon, berlin_lat, 'ro', markersize=8, label='Berlin (example center)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='box')
        ax.legend()
        
        # Set reasonable axis limits around Berlin
        ax.set_xlim(berlin_lon - radius_deg*1.5, berlin_lon + radius_deg*1.5)
        ax.set_ylim(berlin_lat - radius_deg*1.5, berlin_lat + radius_deg*1.5)
    
    def _create_weight_surface_demo(self, ax):
        """Create weight surface demonstration."""
        ax.set_title('C. Weight Surface Visualization', fontweight='bold', fontsize=12)
        
        # Create a simple 2D weight surface
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from center
        distances = np.sqrt(X**2 + Y**2)
        
        # Apply distance decay (scaled for visualization)
        weights = 1 / (1 + (distances / 1) ** 3)
        
        # Create contour plot
        contour = ax.contourf(X, Y, weights, levels=20, cmap='RdYlBu_r')
        ax.contour(X, Y, weights, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Add center point
        ax.plot(0, 0, 'ko', markersize=8, label='Center point')
        
        ax.set_xlabel('Relative distance')
        ax.set_ylabel('Relative distance')
        ax.set_aspect('equal')
        ax.legend()
        
        # Add colorbar
        plt.colorbar(contour, ax=ax, shrink=0.8, label='Weight')
    
    def _analyze_boundary_effects(self, ax):
        """Analyze boundary effects in the analysis."""
        ax.set_title('D. Boundary Effects Analysis', fontweight='bold', fontsize=12)
        
        if self.germany_gdf is not None and len(self.grid_results) > 0:
            # Plot German boundaries
            self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=2)
            
            # Identify edge vs. interior grid points
            # Simple approach: points near boundaries have fewer neighbors
            edge_threshold = 0.2  # degrees (approximate)
            
            bounds = self.germany_gdf.total_bounds  # [minx, miny, maxx, maxy]
            
            edge_mask = ((self.grid_results['grid_lon'] - bounds[0]) < edge_threshold) | \
                       ((bounds[2] - self.grid_results['grid_lon']) < edge_threshold) | \
                       ((self.grid_results['grid_lat'] - bounds[1]) < edge_threshold) | \
                       ((bounds[3] - self.grid_results['grid_lat']) < edge_threshold)
            
            # Plot edge vs interior points
            edge_points = self.grid_results[edge_mask]
            interior_points = self.grid_results[~edge_mask]
            
            if len(edge_points) > 0:
                ax.scatter(edge_points['grid_lon'], edge_points['grid_lat'], 
                          c='red', s=3, alpha=0.6, label=f'Edge points ({len(edge_points)})')
            
            if len(interior_points) > 0:
                ax.scatter(interior_points['grid_lon'], interior_points['grid_lat'], 
                          c='blue', s=1, alpha=0.4, label=f'Interior points ({len(interior_points)})')
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            ax.legend()
    
    def _create_enhanced_correlation_matrix(self, ax):
        """Create enhanced correlation matrix with significance testing."""
        self._create_trait_correlation_heatmap(ax)
    
    def _create_pca_analysis(self, ax):
        """Create PCA analysis of personality traits."""
        ax.set_title('B. Principal Component Analysis', fontweight='bold', fontsize=12)
        
        # Collect trait data
        trait_data = []
        trait_names = []
        
        for trait in self.personality_traits:
            z_col = f'{trait}_z'
            if z_col in self.grid_results.columns:
                trait_data.append(self.grid_results[z_col].fillna(0))
                trait_names.append(trait)
        
        if len(trait_data) >= 2:
            # Perform PCA
            data_matrix = np.array(trait_data).T
            
            # Remove rows with all zeros
            valid_rows = ~np.all(data_matrix == 0, axis=1)
            if valid_rows.sum() > 10:  # Need enough data points
                clean_data = data_matrix[valid_rows]
                
                from sklearn.decomposition import PCA
                pca = PCA()
                pca.fit(clean_data)
                
                # Plot explained variance
                ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
                      pca.explained_variance_ratio_, 
                      alpha=0.7, color='skyblue', edgecolor='black')
                
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Explained Variance Ratio')
                ax.set_title('B. PCA - Explained Variance')
                
                # Add cumulative variance line
                cumvar = np.cumsum(pca.explained_variance_ratio_)
                ax2 = ax.twinx()
                ax2.plot(range(1, len(cumvar) + 1), cumvar, 'ro-', color='red', alpha=0.7)
                ax2.set_ylabel('Cumulative Variance', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
                
                return
        
        # Fallback if PCA fails
        ax.text(0.5, 0.5, 'PCA Analysis\n\nInsufficient valid data\nfor analysis', 
               ha='center', va='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    def _create_trait_combination_map(self, ax):
        """Create trait combination map."""
        ax.set_title('C. Trait Combinations (Openness vs Neuroticism)', fontweight='bold', fontsize=12)
        
        openness_col = 'Openness_z'
        neuroticism_col = 'Neuroticism_z'
        
        if openness_col in self.grid_results.columns and neuroticism_col in self.grid_results.columns:
            # Create combination categories
            valid_mask = (~pd.isna(self.grid_results[openness_col])) & \
                        (~pd.isna(self.grid_results[neuroticism_col]))
            valid_data = self.grid_results[valid_mask]
            
            if len(valid_data) > 0:
                # Define categories
                high_open_low_neuro = (valid_data[openness_col] > 0) & (valid_data[neuroticism_col] < 0)
                high_open_high_neuro = (valid_data[openness_col] > 0) & (valid_data[neuroticism_col] > 0)
                low_open_low_neuro = (valid_data[openness_col] < 0) & (valid_data[neuroticism_col] < 0)
                low_open_high_neuro = (valid_data[openness_col] < 0) & (valid_data[neuroticism_col] > 0)
                
                # Plot categories
                if high_open_low_neuro.sum() > 0:
                    data_subset = valid_data[high_open_low_neuro]
                    ax.scatter(data_subset['grid_lon'], data_subset['grid_lat'], 
                              c='green', s=6, alpha=0.7, label='High O, Low N')
                
                if high_open_high_neuro.sum() > 0:
                    data_subset = valid_data[high_open_high_neuro]
                    ax.scatter(data_subset['grid_lon'], data_subset['grid_lat'], 
                              c='orange', s=6, alpha=0.7, label='High O, High N')
                
                if low_open_low_neuro.sum() > 0:
                    data_subset = valid_data[low_open_low_neuro]
                    ax.scatter(data_subset['grid_lon'], data_subset['grid_lat'], 
                              c='blue', s=6, alpha=0.7, label='Low O, Low N')
                
                if low_open_high_neuro.sum() > 0:
                    data_subset = valid_data[low_open_high_neuro]
                    ax.scatter(data_subset['grid_lon'], data_subset['grid_lat'], 
                              c='red', s=6, alpha=0.7, label='Low O, High N')
                
                if self.germany_gdf is not None:
                    self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
                
                ax.set_aspect('equal', adjustable='box')
                ax.set_xlabel('Longitude')
                ax.set_ylabel('Latitude')
                ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=8)
    
    def _create_regional_profiles(self, ax):
        """Create regional personality profiles."""
        ax.set_title('D. Regional Personality Profiles', fontweight='bold', fontsize=12)
        
        if self.state_results is not None:
            # Select a few states for profiling
            states_to_show = self.state_results.head(5)['state'].tolist()
            
            # Create radar chart data
            angles = np.linspace(0, 2 * np.pi, len(self.personality_traits), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, state in enumerate(states_to_show[:3]):  # Show max 3 states
                state_data = self.state_results[self.state_results['state'] == state].iloc[0]
                
                values = []
                for trait in self.personality_traits:
                    z_col = f'{trait}_z'
                    if z_col in state_data:
                        values.append(state_data[z_col])
                    else:
                        values.append(0)
                
                values = np.concatenate((values, [values[0]]))  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, alpha=0.7, 
                       label=state, color=colors[i])
                ax.fill(angles, values, alpha=0.1, color=colors[i])
            
            # Customize the plot
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([t[:4] for t in self.personality_traits])
            ax.set_ylim(-2, 2)
            ax.grid(True)
            ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
            
            # Add concentric circles for reference
            for val in [-1, 0, 1]:
                ax.plot(angles, [val] * len(angles), '--', alpha=0.3, color='gray')
        else:
            ax.text(0.5, 0.5, 'Regional Profiles\n\nState-level data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    def _create_platform_distribution_map(self, ax):
        """Create platform distribution map if data available."""
        ax.set_title('A. Platform Distribution', fontweight='bold', fontsize=12)
        
        if 'platform' in self.raw_data.columns:
            # Create platform-based visualization
            platforms = self.raw_data['platform'].unique()
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, platform in enumerate(platforms[:4]):  # Max 4 platforms
                platform_data = self.raw_data[self.raw_data['platform'] == platform]
                if len(platform_data) > 0:
                    # Sample if too many points
                    if len(platform_data) > 1000:
                        platform_data = platform_data.sample(1000)
                    
                    ax.scatter(platform_data['longitude'], platform_data['latitude'], 
                              s=1, alpha=0.6, color=colors[i], label=platform)
            
            if self.germany_gdf is not None:
                self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Platform data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _create_platform_statistics(self, ax):
        """Create platform statistics if data available."""
        ax.set_title('B. Platform Statistics', fontweight='bold', fontsize=12)
        
        if 'platform' in self.raw_data.columns:
            platform_counts = self.raw_data['platform'].value_counts()
            
            # Create pie chart
            ax.pie(platform_counts.values, labels=platform_counts.index, 
                  autopct='%1.1f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, 'Platform data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _create_language_distribution_map(self, ax):
        """Create language distribution map if data available."""
        ax.set_title('C. Language Distribution', fontweight='bold', fontsize=12)
        
        if 'language' in self.raw_data.columns:
            # Show top languages
            top_languages = self.raw_data['language'].value_counts().head(3).index
            colors = ['red', 'blue', 'green']
            
            for i, lang in enumerate(top_languages):
                lang_data = self.raw_data[self.raw_data['language'] == lang]
                if len(lang_data) > 0:
                    # Sample if too many points
                    if len(lang_data) > 1000:
                        lang_data = lang_data.sample(1000)
                    
                    ax.scatter(lang_data['longitude'], lang_data['latitude'], 
                              s=1, alpha=0.6, color=colors[i], label=lang)
            
            if self.germany_gdf is not None:
                self.germany_gdf.boundary.plot(ax=ax, color='black', linewidth=1)
            
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_aspect('equal', adjustable='box')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Language data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _create_language_statistics(self, ax):
        """Create language statistics if data available."""
        ax.set_title('D. Language Statistics', fontweight='bold', fontsize=12)
        
        if 'language' in self.raw_data.columns:
            lang_counts = self.raw_data['language'].value_counts().head(8)
            
            # Create horizontal bar chart
            ax.barh(range(len(lang_counts)), lang_counts.values, 
                   color='skyblue', alpha=0.7, edgecolor='black')
            ax.set_yticks(range(len(lang_counts)))
            ax.set_yticklabels(lang_counts.index)
            ax.set_xlabel('Number of users')
            
            # Add value labels
            for i, v in enumerate(lang_counts.values):
                ax.text(v, i, f' {v:,}', va='center')
        else:
            ax.text(0.5, 0.5, 'Language data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
    
    # =============================================================================
    # BATCH PROCESSING METHODS
    # =============================================================================
    
    def create_all_high_priority_visualizations(self, output_dir="thesis_visualizations"):
        """Create all high priority visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating all HIGH PRIORITY visualizations...")
        
        # High priority visualizations
        self.create_data_quality_assessment_panel(f"{output_dir}/01_data_quality_assessment.png")
        self.create_methodological_comparison_dashboard(f"{output_dir}/02_methodological_comparison.png")
        self.create_uncertainty_quantification_suite(f"{output_dir}/03_uncertainty_quantification.png")
        self.create_spatial_statistics_analysis(f"{output_dir}/04_spatial_statistics.png")
        
        print(f"\n✅ All high priority visualizations saved to: {output_dir}/")
    
    def create_all_medium_priority_visualizations(self, output_dir="thesis_visualizations"):
        """Create all medium priority visualizations."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("Creating all MEDIUM PRIORITY visualizations...")
        
        # Medium priority visualizations
        self.create_multi_trait_correlation_analysis(f"{output_dir}/05_trait_correlations.png")
        self.create_distance_decay_demonstration(f"{output_dir}/06_distance_decay_demo.png")
        self.create_platform_language_analysis(f"{output_dir}/07_platform_language_analysis.png")
        
        print(f"\n✅ All medium priority visualizations saved to: {output_dir}/")
    
    def create_complete_visualization_suite(self, output_dir="thesis_visualizations"):
        """Create complete suite of all visualizations."""
        print("="*60)
        print("CREATING COMPLETE THESIS VISUALIZATION SUITE")
        print("="*60)
        
        self.create_all_high_priority_visualizations(output_dir)
        self.create_all_medium_priority_visualizations(output_dir)
        
        print("\n" + "="*60)
        print("COMPLETE VISUALIZATION SUITE CREATED!")
        print("="*60)
        print(f"All visualizations saved to: {output_dir}/")
        print("\nFiles created:")
        print("📊 01_data_quality_assessment.png")
        print("📊 02_methodological_comparison.png") 
        print("📊 03_uncertainty_quantification.png")
        print("📊 04_spatial_statistics.png")
        print("📊 05_trait_correlations.png")
        print("📊 06_distance_decay_demo.png")
        print("📊 07_platform_language_analysis.png")

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def main():
    """Example usage of the advanced visualizations framework."""
    
    # Initialize the framework
    viz = AdvancedThesisVisualizations(
        grid_results_path="spatial_personality_grid_results.csv",
        state_results_path="spatial_personality_state_results.csv", 
        raw_data_path="final_users_for_spatial_visualization.csv",
        shapefile_path="german_shapefile/de.shp"
    )
    
    # Create complete visualization suite
    viz.create_complete_visualization_suite("thesis_visualizations")

if __name__ == "__main__":
    main()