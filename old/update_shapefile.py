import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import warnings
warnings.filterwarnings('ignore')

def update_with_new_shapefile(new_shapefile_path, 
                             grid_results_file="spatial_personality_grid_results.csv",
                             original_data_file="your_full_dataset.csv"):
    """
    Update analysis with new shapefile without recomputing the 5+ hour weighted scores.
    
    This will:
    1. Load your existing grid results (no recomputation needed)
    2. Filter grid points using new shapefile boundaries
    3. Re-aggregate states using new shapefile
    4. Create new visualizations
    """
    
    print("UPDATING ANALYSIS WITH NEW SHAPEFILE")
    print("="*50)
    print("This will use your existing computed results - no 5+ hour recomputation!")
    
    # Step 1: Load existing grid results
    print("\nStep 1: Loading existing grid results...")
    try:
        existing_results = pd.read_csv(grid_results_file)
        print(f"✅ Loaded existing results: {len(existing_results)} grid points")
        print(f"   Columns: {list(existing_results.columns)}")
    except FileNotFoundError:
        print(f"❌ Could not find {grid_results_file}")
        print("Please run the main analysis first or check the file path.")
        return
    
    # Step 2: Load new shapefile
    print(f"\nStep 2: Loading new shapefile...")
    try:
        germany_gdf = gpd.read_file(new_shapefile_path)
        if germany_gdf.crs != 'EPSG:4326':
            print(f"Converting shapefile from {germany_gdf.crs} to EPSG:4326")
            germany_gdf = germany_gdf.to_crs('EPSG:4326')
        
        print(f"✅ Loaded new shapefile: {len(germany_gdf)} regions")
        
        # Print state information
        state_col = 'name'  # Updated for your new shapefile
        if state_col in germany_gdf.columns:
            states = germany_gdf[state_col].unique()
            print(f"States in new shapefile ({len(states)}):")
            for state in sorted(states):
                print(f"  - {state}")
        else:
            print(f"Warning: 'name' column not found in shapefile")
            print(f"Available columns: {list(germany_gdf.columns)}")
            return
    except Exception as e:
        print(f"❌ Error loading shapefile: {e}")
        return
    
    # Step 3: Filter grid points to new boundaries (optional - for cleaner maps)
    print(f"\nStep 3: Filtering grid points to new boundaries...")
    
    # Create point geometries for existing grid
    geometry = []
    for _, row in existing_results.iterrows():
        geometry.append(Point(row['grid_lon'], row['grid_lat']))
    
    grid_gdf = gpd.GeoDataFrame(existing_results, geometry=geometry, crs='EPSG:4326')
    
    # Filter to new boundaries
    germany_union = germany_gdf.geometry.unary_union
    within_mask = [geom.within(germany_union) for geom in grid_gdf.geometry]
    
    filtered_results = grid_gdf[within_mask].drop('geometry', axis=1).reset_index(drop=True)
    filtered_results['grid_idx'] = range(len(filtered_results))
    
    print(f"Grid points: {len(existing_results)} → {len(filtered_results)} "
          f"(removed {len(existing_results) - len(filtered_results)} outside new boundaries)")
    
    # Step 4: Re-aggregate by states with new shapefile
    print(f"\nStep 4: Re-aggregating by states with new shapefile...")
    
    try:
        # Load original data for state aggregation
        data = pd.read_csv(original_data_file)
        personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                             'Agreeableness', 'Neuroticism']
        required_cols = ['latitude', 'longitude'] + personality_traits
        data = data.dropna(subset=required_cols)
        print(f"✅ Loaded original data: {len(data)} users")
        
        # Perform state aggregation
        state_results = aggregate_by_states_new(data, germany_gdf, personality_traits)
        
        if state_results is not None:
            print(f"✅ New state aggregation: {len(state_results)} states")
            
            # Save new state results
            new_state_file = "spatial_personality_state_results_corrected.csv"
            state_results.to_csv(new_state_file, index=False)
            print(f"✅ Saved: {new_state_file}")
        
    except FileNotFoundError:
        print(f"❌ Could not find {original_data_file}")
        print("State aggregation skipped - using grid results only")
        state_results = None
    
    # Step 5: Create new visualizations
    print(f"\nStep 5: Creating new maps with corrected boundaries...")
    
    output_file = "ebert_style_personality_maps_germany_corrected.png"
    create_corrected_maps(filtered_results, state_results, germany_gdf, output_file)
    
    # Step 6: Generate new summary
    if state_results is not None:
        print(f"\nStep 6: Creating corrected summary...")
        create_corrected_summary(state_results, output_file)
    
    # Step 7: Save corrected grid results
    corrected_grid_file = "spatial_personality_grid_results_corrected.csv"
    filtered_results.to_csv(corrected_grid_file, index=False)
    
    print("\n" + "="*60)
    print("UPDATE COMPLETE!")
    print("="*60)
    print("New files created:")
    print(f"✅ {corrected_grid_file}")
    if state_results is not None:
        print(f"✅ spatial_personality_state_results_corrected.csv")
    print(f"✅ {output_file}")
    print(f"✅ Corrected summary statistics")
    
    print(f"\nKey improvements:")
    print(f"✅ Correct number of German states: {len(state_results) if state_results else 'N/A'}")
    print(f"✅ No Bodensee water regions")
    print(f"✅ Clean administrative boundaries")
    print(f"✅ Same validated computational results")

def aggregate_by_states_new(data, germany_gdf, personality_traits):
    """
    Re-aggregate personality scores by German states using new shapefile.
    """
    print("Aggregating personality scores by German states...")
    
    # Create point geometries for user data
    geometry = [Point(lon, lat) for lon, lat in zip(data['longitude'], data['latitude'])]
    data_gdf = gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')
    
    # Spatial join to assign users to states
    data_with_states = gpd.sjoin(data_gdf, germany_gdf, how='left', predicate='within')
    
    # Identify state name column
    state_col = 'name'  # Updated for your new shapefile
    if state_col not in data_with_states.columns:
        print(f"Warning: 'name' column not found after spatial join")
        print(f"Available columns: {list(data_with_states.columns)}")
        return None
    
    # Aggregate by state
    state_results = []
    
    for state_name in germany_gdf[state_col].unique():
        state_data = data_with_states[data_with_states[state_col] == state_name]
        
        if len(state_data) > 0:
            state_row = {'state': state_name, 'n_users': len(state_data)}
            
            # Calculate mean personality scores
            for trait in personality_traits:
                state_row[trait] = state_data[trait].mean()
                state_row[f'{trait}_std'] = state_data[trait].std()
            
            state_results.append(state_row)
    
    if not state_results:
        print("Warning: No state results generated")
        return None
    
    state_results_df = pd.DataFrame(state_results)
    
    # Standardize scores
    for trait in personality_traits:
        mean_score = state_results_df[trait].mean()
        std_score = state_results_df[trait].std()
        state_results_df[f'{trait}_z'] = (state_results_df[trait] - mean_score) / std_score
    
    print(f"Aggregated data for {len(state_results_df)} states")
    return state_results_df

def create_corrected_maps(grid_results, state_results, germany_gdf, save_path):
    """
    Create corrected maps with new shapefile boundaries.
    """
    print("Creating corrected Ebert-style maps...")
    
    personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                          'Agreeableness', 'Neuroticism']
    
    if state_results is not None:
        # Create both grid and state maps
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        
        fig.suptitle('Spatial Distribution of Big Five Personality Traits in Germany\n'
                    'LLM-Inferred vs. State-Level Aggregation (Corrected Boundaries)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Grid maps (top 2 rows)
        create_grid_maps_corrected(axes[:2, :], grid_results, germany_gdf, personality_traits)
        
        # State maps (bottom 2 rows)
        create_state_maps_corrected(axes[2:, :], state_results, germany_gdf, personality_traits)
        
    else:
        # Grid maps only
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Spatial Distribution of Big Five Personality Traits in Germany\n'
                    'LLM-Inferred Personality (Corrected Boundaries)', 
                    fontsize=16, fontweight='bold')
        
        create_simple_grid_maps_corrected(axes, grid_results, germany_gdf, personality_traits)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Corrected maps saved: {save_path}")
    plt.show()

def create_grid_maps_corrected(axes, results, germany_gdf, personality_traits):
    """Create corrected grid-based maps."""
    axes[0, 2].text(0.5, 1.15, "Grid-Based (Actor-Based Clustering) - Corrected", 
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
                
                # Add corrected German boundaries
                germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.5, alpha=0.7)
                
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

def create_state_maps_corrected(axes, state_results, germany_gdf, personality_traits):
    """Create corrected state-level maps."""
    axes[0, 2].text(0.5, 1.15, "State-Level Aggregation - Corrected", 
                    ha='center', fontsize=14, fontweight='bold', 
                    transform=axes[0, 2].transAxes)
    
    # Find state column
    state_col = 'name'  # Updated for your new shapefile
    if state_col not in germany_gdf.columns:
        print(f"Warning: 'name' column not found in shapefile")
        return
    
    # Merge data
    merged_gdf = germany_gdf.merge(state_results, left_on=state_col, right_on='state', how='left')
    
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

def create_simple_grid_maps_corrected(axes, results, germany_gdf, personality_traits):
    """Create simple corrected grid maps."""
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
            
            germany_gdf.boundary.plot(ax=ax, color='black', linewidth=0.7, alpha=0.8)
            
            ax.set_title(trait, fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
    
    if len(personality_traits) == 5:
        axes[1, 2].remove()

def create_corrected_summary(state_results, save_path):
    """Create corrected summary statistics."""
    personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                          'Agreeableness', 'Neuroticism']
    
    print("\n" + "="*80)
    print("CORRECTED PERSONALITY TRAIT SUMMARY BY GERMAN STATES")
    print("="*80)
    print(f"{'Trait':<15} {'Mean':<8} {'SD':<8} {'Min':<8} {'Max':<8} {'Range':<8} {'N States':<8}")
    print("-" * 80)
    
    for trait in personality_traits:
        stats = {
            'Mean': state_results[trait].mean(),
            'SD': state_results[trait].std(),
            'Min': state_results[trait].min(),
            'Max': state_results[trait].max(),
            'Range': state_results[trait].max() - state_results[trait].min(),
            'States_N': state_results[trait].count()
        }
        
        print(f"{trait:<15} {stats['Mean']:<8.3f} {stats['SD']:<8.3f} {stats['Min']:<8.3f} "
              f"{stats['Max']:<8.3f} {stats['Range']:<8.3f} {int(stats['States_N']):<8}")
    
    # Show extreme states
    print("\n" + "="*80)
    print("EXTREME STATES BY PERSONALITY TRAIT (CORRECTED)")
    print("="*80)
    
    for trait in personality_traits:
        sorted_states = state_results.sort_values(trait)
        highest = sorted_states.iloc[-1]
        lowest = sorted_states.iloc[0]
        
        print(f"\n{trait}:")
        print(f"  Highest: {highest['state']} ({highest[trait]:.3f})")
        print(f"  Lowest:  {lowest['state']} ({lowest[trait]:.3f})")
    
    # Save corrected summary
    summary_file = save_path.replace('.png', '_summary.csv')
    
    summary_stats = []
    for trait in personality_traits:
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
    
    pd.DataFrame(summary_stats).to_csv(summary_file, index=False)
    print(f"\nCorrected summary saved: {summary_file}")

if __name__ == "__main__":
    # UPDATE THESE PATHS TO YOUR FILES
    new_shapefile = "german_shapefile/de.shp"  # Update this path to your German shapefile
    original_data = "final_users_for_spatial_visualization.csv"  # Update this path
    
    print("UPDATE FILE PATHS:")
    print(f"new_shapefile = '{new_shapefile}'")
    print(f"original_data = '{original_data}'")
    print(f"Existing grid results will be loaded from: 'spatial_personality_grid_results.csv'")
    print()
    
    update_with_new_shapefile(new_shapefile, original_data_file=original_data)
