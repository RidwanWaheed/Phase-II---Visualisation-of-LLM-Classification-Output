import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

class PersonalityPCAVisualizer:
    """
    Component for performing Principal Component Analysis (PCA) on personality trait data
    and generating visualizations.
    """

    def __init__(self, personality_traits=None):
        """
        Initializes the PCA visualizer.

        Parameters:
        -----------
        personality_traits : list, optional
            A list of strings representing the names of the personality trait columns
            in the input DataFrame. If None, it will attempt to use default Big Five traits.
        """
        self.personality_traits = personality_traits if personality_traits is not None else \
                                 ['Openness', 'Conscientiousness', 'Extraversion', 
                                  'Agreeableness', 'Neuroticism']
        self.pca_model = None
        self.scaled_data = None
        self.principal_components = None
        self.explained_variance_ratio = None
        self.loadings = None

        print("="*60)
        print("PERSONALITY PCA VISUALIZER")
        print("="*60)
        print(f"Target personality traits: {self.personality_traits}")
        print("Ready to perform PCA and generate visualizations.")
        print("="*60)

    def load_and_prepare_data(self, data_file):
        """
        Loads the personality data from a CSV file and prepares it for PCA.
        This involves selecting personality trait columns and scaling the data.

        Parameters:
        -----------
        data_file : str
            Path to the CSV file containing user data with personality traits.

        Returns:
        --------
        pandas.DataFrame : The prepared DataFrame containing only the personality traits.
        """
        print(f"\nLoading and preparing data from: {data_file}")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")

        try:
            data = pd.read_csv(data_file)
            
            # Select only the personality trait columns
            personality_data = data[self.personality_traits].copy()
            
            # Drop rows with any missing values in personality traits
            original_rows = len(personality_data)
            personality_data.dropna(inplace=True)
            if len(personality_data) < original_rows:
                print(f"   Removed {original_rows - len(personality_data)} rows due to missing personality data.")

            if personality_data.empty:
                raise ValueError("No valid personality data found after dropping missing values.")

            print(f"   Successfully loaded {len(personality_data)} records for PCA.")
            print("   Scaling data...")
            
            # Standardize the data (mean=0, variance=1)
            scaler = StandardScaler()
            self.scaled_data = scaler.fit_transform(personality_data)
            self.scaled_data_df = pd.DataFrame(self.scaled_data, columns=self.personality_traits) # Keep as DataFrame for clarity

            print("   Data scaled successfully.")
            return personality_data

        except Exception as e:
            print(f"Error loading or preparing data: {e}")
            raise

    def perform_pca(self, n_components=None):
        """
        Performs PCA on the scaled personality data.

        Parameters:
        -----------
        n_components : int, optional
            The number of principal components to compute. If None, all components
            (up to min(n_samples, n_features)) will be kept.
        """
        if self.scaled_data is None:
            raise ValueError("Data not loaded or prepared. Call load_and_prepare_data first.")

        print(f"\nPerforming PCA with n_components={n_components if n_components else 'all'}...")
        self.pca_model = PCA(n_components=n_components)
        self.principal_components = self.pca_model.fit_transform(self.scaled_data)
        self.explained_variance_ratio = self.pca_model.explained_variance_ratio_
        
        # Calculate loadings (correlation between original variables and principal components)
        # Loadings are the eigenvectors scaled by the square root of the eigenvalues
        # Or simply, the components multiplied by the square root of explained variance
        self.loadings = pd.DataFrame(self.pca_model.components_.T, 
                                     columns=[f'PC{i+1}' for i in range(self.pca_model.n_components_)],
                                     index=self.personality_traits)
        
        print(f"   PCA completed. Explained variance ratio: {np.sum(self.explained_variance_ratio):.2f}")
        for i, ratio in enumerate(self.explained_variance_ratio):
            print(f"     PC{i+1}: {ratio*100:.2f}%")

    def plot_scree(self, save_path=None):
        """
        Generates a scree plot to visualize the explained variance by each principal component.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the scree plot image.
        """
        if self.explained_variance_ratio is None:
            raise ValueError("PCA not performed. Call perform_pca first.")

        print("\nGenerating Scree Plot...")
        plt.figure(figsize=(10, 6))
        
        # Plot explained variance ratio
        plt.plot(range(1, len(self.explained_variance_ratio) + 1), 
                 self.explained_variance_ratio, 
                 marker='o', linestyle='-', color='skyblue', label='Individual Explained Variance')
        
        # Plot cumulative explained variance
        cumulative_variance = np.cumsum(self.explained_variance_ratio)
        plt.plot(range(1, len(self.explained_variance_ratio) + 1), 
                 cumulative_variance, 
                 marker='x', linestyle='--', color='red', label='Cumulative Explained Variance')

        plt.title('Scree Plot: Explained Variance by Principal Components', fontsize=16, fontweight='bold')
        plt.xlabel('Principal Component', fontsize=12)
        plt.ylabel('Explained Variance Ratio', fontsize=12)
        plt.xticks(range(1, len(self.explained_variance_ratio) + 1))
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Scree plot saved: {save_path}")
        plt.show()
        plt.close()

    def plot_pca_scatter(self, save_path=None):
        """
        Generates a scatter plot of the first two principal components.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the scatter plot image.
        """
        if self.principal_components is None or self.principal_components.shape[1] < 2:
            print("Cannot generate 2D scatter plot: PCA not performed or less than 2 components.")
            return

        print("\nGenerating PCA Scatter Plot (PC1 vs PC2)...")
        plt.figure(figsize=(10, 8))
        
        plt.scatter(self.principal_components[:, 0], self.principal_components[:, 1], 
                    alpha=0.6, s=50, edgecolors='w', linewidth=0.5)
        
        plt.title('PCA: Principal Components 1 vs 2', fontsize=16, fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({self.explained_variance_ratio[0]*100:.2f}% Variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({self.explained_variance_ratio[1]*100:.2f}% Variance)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   PCA scatter plot saved: {save_path}")
        plt.show()
        plt.close()

    def plot_biplot(self, save_path=None):
        """
        Generates a biplot showing the first two principal components and the
        loadings of the original variables.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the biplot image.
        """
        if self.principal_components is None or self.principal_components.shape[1] < 2 or self.loadings is None:
            print("Cannot generate biplot: PCA not performed, less than 2 components, or loadings missing.")
            return

        print("\nGenerating Biplot (PC1 vs PC2 with Loadings)...")
        plt.figure(figsize=(12, 10))
        
        # Scatter plot of the data points
        plt.scatter(self.principal_components[:, 0], self.principal_components[:, 1], 
                    alpha=0.6, s=50, edgecolors='w', linewidth=0.5, zorder=2)
        
        # Plot the loadings as vectors
        # Scaling factor for the arrows for better visualization
        scale_x = max(abs(self.principal_components[:, 0])) * 0.8 / max(abs(self.loadings['PC1']))
        scale_y = max(abs(self.principal_components[:, 1])) * 0.8 / max(abs(self.loadings['PC2']))

        for i, feature in enumerate(self.personality_traits):
            plt.arrow(0, 0, self.loadings.loc[feature, 'PC1'] * scale_x, 
                      self.loadings.loc[feature, 'PC2'] * scale_y, 
                      color='r', alpha=0.8, lw=2, head_width=0.05 * max(abs(self.principal_components[:, 0]))/10, 
                      head_length=0.05 * max(abs(self.principal_components[:, 1]))/10, zorder=3)
            plt.text(self.loadings.loc[feature, 'PC1'] * scale_x * 1.1, 
                     self.loadings.loc[feature, 'PC2'] * scale_y * 1.1, 
                     feature, color='r', ha='center', va='center', fontsize=10, fontweight='bold', zorder=4)

        plt.title('Biplot of Principal Components 1 and 2', fontsize=16, fontweight='bold')
        plt.xlabel(f'Principal Component 1 ({self.explained_variance_ratio[0]*100:.2f}% Variance)', fontsize=12)
        plt.ylabel(f'Principal Component 2 ({self.explained_variance_ratio[1]*100:.2f}% Variance)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='gray', lw=0.5)
        plt.axvline(0, color='gray', lw=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   Biplot saved: {save_path}")
        plt.show()
        plt.close()

def run_pca_analysis(data_file, output_prefix="pca_results"):
    """
    Main function to run the PCA analysis and generate all visualizations.

    Parameters:
    -----------
    data_file : str
        Path to the CSV file containing user data with personality traits.
    output_prefix : str, optional
        Prefix for the output image files.
    """
    print("\n" + "="*70)
    print("STARTING PCA ANALYSIS AND VISUALIZATION")
    print("="*70)

    # Initialize the PCA visualizer
    # Assuming Big Five traits are 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'
    # If your CSV columns are different, update this list:
    personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    pca_visualizer = PersonalityPCAVisualizer(personality_traits=personality_traits)

    # Load and prepare data
    try:
        pca_visualizer.load_and_prepare_data(data_file)
    except Exception as e:
        print(f"Failed to prepare data: {e}")
        return

    # Perform PCA (compute all components by default)
    pca_visualizer.perform_pca()

    # Generate and save visualizations
    output_files = {}

    # Scree Plot
    scree_plot_path = f"{output_prefix}_scree_plot.png"
    pca_visualizer.plot_scree(save_path=scree_plot_path)
    output_files['scree_plot'] = scree_plot_path

    # PCA Scatter Plot (PC1 vs PC2)
    if pca_visualizer.principal_components.shape[1] >= 2:
        scatter_plot_path = f"{output_prefix}_scatter_plot.png"
        pca_visualizer.plot_pca_scatter(save_path=scatter_plot_path)
        output_files['scatter_plot'] = scatter_plot_path
    else:
        print("Skipping scatter plot: Less than 2 principal components.")

    # Biplot (PC1 vs PC2 with loadings)
    if pca_visualizer.principal_components.shape[1] >= 2:
        biplot_path = f"{output_prefix}_biplot.png"
        pca_visualizer.plot_biplot(save_path=biplot_path)
        output_files['biplot'] = biplot_path
    else:
        print("Skipping biplot: Less than 2 principal components.")

    print(f"\nPCA ANALYSIS COMPLETE!")
    print(f"Created {len(output_files)} visualization files:")
    for viz_type, file_path in output_files.items():
        print(f"   {viz_type.replace('_', ' ').title()}: {file_path}")

    return output_files

if __name__ == "__main__":
    # Configuration for running the script directly
    # IMPORTANT: Replace 'final_users_for_spatial_visualization.csv' with your actual data file path.
    # Ensure this file contains the personality trait columns specified in PersonalityPCAVisualizer.
    data_file_path = "final_users_for_spatial_visualization.csv" 
    output_prefix = "personality_pca_results"
    
    # Run the PCA analysis and visualizations
    output_files = run_pca_analysis(
        data_file=data_file_path,
        output_prefix=output_prefix
    )

    print(f"\nAll PCA visualizations created successfully!")
    print(f"Output files: {list(output_files.values())}")
