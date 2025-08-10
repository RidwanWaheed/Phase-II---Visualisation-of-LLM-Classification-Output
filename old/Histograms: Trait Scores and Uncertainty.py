import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For enhanced visualizations
import warnings
import os

warnings.filterwarnings('ignore')

class HistogramVisualizer:
    """
    Component for generating histograms of interpolated personality trait Z-scores
    and the spatial uncertainty metric.
    """

    def __init__(self):
        """
        Initializes the visualizer.
        """
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        print("="*60)
        print("HISTOGRAM VISUALIZER: TRAIT SCORES AND UNCERTAINTY")
        print("="*60)
        print("Ready to visualize the distributions of personality traits and data support.")
        print("="*60)

    def load_grid_results(self, grid_results_file):
        """
        Loads the pre-computed grid results, which should contain grid coordinates,
        Z-scores for each trait, and the weight_sum for uncertainty.
        
        Parameters:
        -----------
        grid_results_file : str
            Path to CSV file with pre-computed grid results (e.g., from SpatialPersonalityGridComputer).
            
        Returns:
        --------
        pandas.DataFrame : Grid results with relevant columns for histogram plotting.
        """
        print(f"\nLoading grid results for histogram analysis from: {grid_results_file}")
        
        if not os.path.exists(grid_results_file):
            raise FileNotFoundError(f"Grid results file not found: {grid_results_file}")
        
        try:
            grid_data = pd.read_csv(grid_results_file)
            
            # Ensure Z-score columns and weight_sum columns exist
            z_score_cols = [f'{trait}_z' for trait in self.personality_traits]
            
            # For uncertainty, we use Openness_weight_sum as the representative for general data support.
            representative_weight_sum_col = 'Openness_weight_sum' 
            
            required_cols = z_score_cols + [representative_weight_sum_col]

            missing_cols = [col for col in required_cols if col not in grid_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in grid results: {missing_cols}")
            
            # Calculate the uncertainty metric (inverse weight sum)
            grid_data['uncertainty_metric'] = 1 / (grid_data[representative_weight_sum_col] + 1e-6)
            
            # Select only the relevant columns for analysis
            analysis_data = grid_data[z_score_cols + ['uncertainty_metric']].copy()
            
            # Drop rows with any missing values in the selected columns
            original_rows = len(analysis_data)
            analysis_data.dropna(inplace=True)
            if len(analysis_data) < original_rows:
                print(f"   Removed {original_rows - len(analysis_data)} rows due to missing data for analysis.")

            if analysis_data.empty:
                raise ValueError("No valid data found after dropping missing values for histogram analysis.")

            print(f"   Loaded and prepared {len(analysis_data):,} grid points for histogram analysis.")
            return analysis_data
            
        except Exception as e:
            print(f"Error loading or preparing grid results: {e}")
            raise

    def plot_histogram(self, data, column_name, title_suffix="", x_label="", save_path=None):
        """
        Generates a histogram for a specified column in the DataFrame.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing the data.
        column_name : str
            The name of the column to plot the histogram for.
        title_suffix : str, optional
            Suffix to add to the plot title (e.g., "Z-score" or "Inverse Weight Sum").
        x_label : str, optional
            Label for the x-axis.
        save_path : str, optional
            Path to save the plot image.
        """
        if column_name not in data.columns:
            print(f"Error: Column '{column_name}' not found in data. Cannot generate histogram.")
            return

        print(f"\nGenerating Histogram for {column_name}...")
        
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column_name], kde=True, bins=30, color='skyblue', edgecolor='black')
        
        plt.title(f'Distribution of {column_name.replace("_z", "")} {title_suffix}', fontsize=16, fontweight='bold')
        plt.xlabel(x_label if x_label else column_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Histogram for {column_name} saved: {save_path}")
        plt.show()
        plt.close()


def run_histogram_analysis(grid_results_file, output_prefix="histograms"):
    """
    Main function to run the histogram analysis for personality traits and uncertainty.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to the CSV file containing pre-computed grid results (with Z-scores and weight_sum).
    output_prefix : str, optional
        Prefix for the output image files.
    """
    print("\n" + "="*70)
    print("STARTING HISTOGRAM ANALYSIS")
    print("="*70)

    visualizer = HistogramVisualizer()
    
    # Load and prepare data
    try:
        analysis_data = visualizer.load_grid_results(grid_results_file)
    except Exception as e:
        print(f"Error: {e}")
        return {}

    output_files = {}

    # Generate histograms for each personality trait's Z-score
    for trait in visualizer.personality_traits:
        z_col = f'{trait}_z'
        if z_col in analysis_data.columns:
            hist_path = f"{output_prefix}_{trait.lower()}_z_score_histogram.png"
            visualizer.plot_histogram(analysis_data, z_col, 
                                      title_suffix="Z-score Distribution", 
                                      x_label=f'{trait} Z-score', 
                                      save_path=hist_path)
            output_files[f'{trait.lower()}_histogram'] = hist_path

    # Generate histogram for the uncertainty metric
    uncertainty_hist_path = f"{output_prefix}_uncertainty_histogram.png"
    visualizer.plot_histogram(analysis_data, 'uncertainty_metric', 
                              title_suffix="Distribution (Inverse Weight Sum)", 
                              x_label='Uncertainty (Inverse Weight Sum)', 
                              save_path=uncertainty_hist_path)
    output_files['uncertainty_histogram'] = uncertainty_hist_path

    print(f"\nHistogram Analysis complete!")
    print(f"Output files: {list(output_files.values())}")

    return output_files

if __name__ == "__main__":
    # Configuration for running the script directly
    grid_results_file_path = "spatial_personality_grid_results.csv" 
    
    output_prefix = "histograms"
    
    output_files = run_histogram_analysis(
        grid_results_file=grid_results_file_path,
        output_prefix=output_prefix
    )
    
    print(f"\nAll histogram visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
