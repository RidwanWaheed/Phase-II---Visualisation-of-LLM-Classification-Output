import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For enhanced visualizations
import warnings
import os

warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Component for analyzing the correlation between interpolated personality trait Z-scores
    and the spatial uncertainty metric (inverse weight sum).
    """

    def __init__(self):
        """
        Initializes the analyzer.
        """
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        print("="*60)
        print("CORRELATION ANALYSIS: TRAITS VS. UNCERTAINTY")
        print("="*60)
        print("Ready to analyze relationships between personality traits and data support.")
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
        pandas.DataFrame : Grid results with relevant columns for correlation.
        """
        print(f"\nLoading grid results for correlation analysis from: {grid_results_file}")
        
        if not os.path.exists(grid_results_file):
            raise FileNotFoundError(f"Grid results file not found: {grid_results_file}")
        
        try:
            grid_data = pd.read_csv(grid_results_file)
            
            # Ensure Z-score columns and weight_sum columns exist
            z_score_cols = [f'{trait}_z' for trait in self.personality_traits]
            weight_sum_cols = [f'{trait}_weight_sum' for trait in self.personality_traits]
            
            # For uncertainty, we only need one weight_sum column as they are identical for a given grid point
            # We'll use Openness_weight_sum as the representative for general data support.
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
                raise ValueError("No valid data found after dropping missing values for correlation analysis.")

            print(f"   Loaded and prepared {len(analysis_data):,} grid points for correlation analysis.")
            return analysis_data
            
        except Exception as e:
            print(f"Error loading or preparing grid results: {e}")
            raise

    def calculate_correlations(self, data):
        """
        Calculates Pearson correlation coefficients between personality trait Z-scores
        and the uncertainty metric.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing personality trait Z-scores and 'uncertainty_metric'.
            
        Returns:
        --------
        pandas.Series : Series of correlation coefficients.
        """
        print("\nCalculating correlations between traits and uncertainty...")
        
        correlations = {}
        for trait in self.personality_traits:
            z_col = f'{trait}_z'
            if z_col in data.columns:
                corr = data[z_col].corr(data['uncertainty_metric'])
                correlations[f'{trait}_vs_Uncertainty'] = corr
        
        correlations_series = pd.Series(correlations)
        print("   Correlations calculated:")
        print(correlations_series.to_string())
        
        return correlations_series

    def plot_correlation_heatmap(self, data, save_path=None):
        """
        Generates a heatmap of the correlation matrix including all traits and uncertainty.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing personality trait Z-scores and 'uncertainty_metric'.
        save_path : str, optional
            Path to save the plot image.
        """
        print("\nGenerating correlation heatmap...")
        
        # Calculate the full correlation matrix
        correlation_matrix = data.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", 
                    linewidths=.5, linecolor='black', cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Correlation Heatmap: Personality Traits and Uncertainty', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Correlation heatmap saved: {save_path}")
        plt.show()
        plt.close()

    def plot_scatter_trait_vs_uncertainty(self, data, trait, save_path=None):
        """
        Generates a scatter plot of a specific trait's Z-score against the uncertainty metric.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame containing personality trait Z-scores and 'uncertainty_metric'.
        trait : str
            Name of the personality trait to plot (e.g., 'Openness').
        save_path : str, optional
            Path to save the plot image.
        """
        z_col = f'{trait}_z'
        if z_col not in data.columns:
            print(f"Error: Z-score column '{z_col}' not found for trait '{trait}'. Cannot generate scatter plot.")
            return

        print(f"\nGenerating Scatter Plot: {trait} vs. Uncertainty...")
        
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=z_col, y='uncertainty_metric', data=data, alpha=0.6, s=20)
        
        # Add correlation value to the plot title
        correlation_value = data[z_col].corr(data['uncertainty_metric'])
        plt.title(f'Scatter Plot: {trait} (Z-score) vs. Uncertainty (Correlation: {correlation_value:.2f})', 
                  fontsize=14, fontweight='bold')
        plt.xlabel(f'{trait} (Z-score)', fontsize=12)
        plt.ylabel('Uncertainty (Inverse Weight Sum)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Scatter plot for {trait} saved: {save_path}")
        plt.show()
        plt.close()


def run_correlation_analysis(grid_results_file, traits_to_plot=None, output_prefix="correlation_analysis"):
    """
    Main function to run the correlation analysis between personality traits and uncertainty.
    
    Parameters:
    -----------
    grid_results_file : str
        Path to the CSV file containing pre-computed grid results (with Z-scores and weight_sum).
    traits_to_plot : list, optional
        A list of specific traits for which to generate individual scatter plots.
        If None, scatter plots will be generated for all default personality traits.
    output_prefix : str, optional
        Prefix for the output image files.
    """
    print("\n" + "="*70)
    print("STARTING CORRELATION ANALYSIS")
    print("="*70)

    analyzer = CorrelationAnalyzer()
    
    # Load and prepare data
    try:
        analysis_data = analyzer.load_grid_results(grid_results_file)
    except Exception as e:
        print(f"Error: {e}")
        return {}

    output_files = {}

    # Calculate and print correlations
    correlations = analyzer.calculate_correlations(analysis_data)
    # You might want to save these correlations to a text file or CSV for your thesis
    # e.g., correlations.to_csv(f"{output_prefix}_correlations.csv")

    # Generate correlation heatmap
    heatmap_path = f"{output_prefix}_heatmap.png"
    analyzer.plot_correlation_heatmap(analysis_data, save_path=heatmap_path)
    output_files['correlation_heatmap'] = heatmap_path

    # Generate individual scatter plots for each trait vs. uncertainty
    traits_for_scatter = traits_to_plot if traits_to_plot is not None else analyzer.personality_traits
    for trait in traits_for_scatter:
        scatter_path = f"{output_prefix}_{trait.lower()}_vs_uncertainty_scatter.png"
        analyzer.plot_scatter_trait_vs_uncertainty(analysis_data, trait, save_path=scatter_path)
        output_files[f'{trait.lower()}_scatter'] = scatter_path

    print(f"\nCorrelation Analysis complete!")
    print(f"Output files: {list(output_files.values())}")

    return output_files

if __name__ == "__main__":
    # Configuration for running the script directly
    grid_results_file_path = "spatial_personality_grid_results.csv" 
    
    # Optionally specify a subset of traits for individual scatter plots.
    # If None, scatter plots for all 5 traits will be generated.
    traits_to_plot_individually = None # ['Openness', 'Neuroticism'] 

    output_prefix = "correlation_analysis"
    
    output_files = run_correlation_analysis(
        grid_results_file=grid_results_file_path,
        traits_to_plot=traits_to_plot_individually,
        output_prefix=output_prefix
    )
    
    print(f"\nAll correlation analysis visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
