import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from math import pi

class GermanStatesSpiderChart:
    """
    Create professional spider/radar charts comparing personality traits across German states.
    Designed for academic thesis visualization comparing LLM-inferred personality profiles.
    """
    
    def __init__(self, state_results_path="spatial_personality_state_results.csv"):
        """Initialize with state-level personality data."""
        self.personality_traits = ['Openness', 'Conscientiousness', 'Extraversion', 
                                 'Agreeableness', 'Neuroticism']
        self.trait_labels = ['Open.', 'Consc.', 'Extra.', 'Agree.', 'Neuro.']
        
        # Load state results
        try:
            self.state_results = pd.read_csv(state_results_path)
            print(f"Loaded state results: {len(self.state_results)} states")
            print(f"Available states: {list(self.state_results['state'].unique())}")
        except FileNotFoundError:
            print(f"Error: Could not find {state_results_path}")
            self.state_results = None
    
    def create_single_spider_chart(self, states_to_compare, save_path=None, 
                                 use_z_scores=True, title_suffix=""):
        """
        Create a single spider chart comparing selected German states.
        
        Parameters:
        -----------
        states_to_compare : list
            List of state names to compare (max 6 for readability)
        save_path : str, optional
            Path to save the figure
        use_z_scores : bool
            Use standardized z-scores (True) or raw scores (False)
        title_suffix : str
            Additional text for the title
        """
        if self.state_results is None:
            print("No state results loaded")
            return
        
        # Limit to 6 states for readability
        if len(states_to_compare) > 6:
            print("Warning: Too many states selected. Using first 6.")
            states_to_compare = states_to_compare[:6]
        
        # Filter data for selected states
        selected_states = self.state_results[
            self.state_results['state'].isin(states_to_compare)
        ].copy()
        
        if len(selected_states) == 0:
            print("No matching states found")
            return
        
        # Setup the spider chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each personality trait
        angles = np.linspace(0, 2 * pi, len(self.personality_traits), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Color palette for states
        colors = plt.cm.Set3(np.linspace(0, 1, len(selected_states)))
        
        # Plot each state
        for idx, (_, state_row) in enumerate(selected_states.iterrows()):
            state_name = state_row['state']
            
            # Get personality values
            if use_z_scores:
                values = [state_row[f'{trait}_z'] for trait in self.personality_traits]
                value_range = (-2, 2)
                chart_title = f'Personality Profiles Comparison (Z-Scores)\n{title_suffix}'
            else:
                values = [state_row[trait] for trait in self.personality_traits]
                value_range = (1, 5)
                chart_title = f'Personality Profiles Comparison (Raw Scores)\n{title_suffix}'
            
            values += values[:1]  # Complete the circle
            
            # Plot the line and fill
            ax.plot(angles, values, 'o-', linewidth=2.5, alpha=0.8, 
                   label=state_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Customize the chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.trait_labels, fontsize=12, fontweight='bold')
        
        # Set y-axis limits and labels
        ax.set_ylim(value_range)
        ax.set_yticks(np.linspace(value_range[0], value_range[1], 5))
        ax.set_yticklabels([f'{val:.1f}' for val in np.linspace(value_range[0], value_range[1], 5)], 
                          fontsize=10, alpha=0.7)
        
        # Add gridlines
        ax.grid(True, alpha=0.3)
        
        # Add reference circles
        for val in np.linspace(value_range[0], value_range[1], 5):
            ax.plot(angles, [val] * len(angles), '--', alpha=0.2, color='gray', linewidth=0.8)
        
        # Customize title and legend
        plt.title(chart_title, pad=20, fontsize=14, fontweight='bold')
        
        legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), 
                          fontsize=11, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_alpha(0.9)
        
        # Add methodology note
        method_note = "LLM-Inferred from Social Media Data | Ebert et al. (2022) Methodology"
        plt.figtext(0.5, 0.02, method_note, ha='center', fontsize=9, 
                   style='italic', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Spider chart saved to {save_path}")
        
        plt.show()
        return fig, ax
    
    def create_multi_panel_comparison(self, save_path=None):
        """
        Create a 2x2 grid of spider charts showing different state groupings.
        """
        if self.state_results is None:
            print("No state results loaded")
            return
        
        # Get all available states
        all_states = self.state_results['state'].tolist()
        
        # Define interesting state groupings
        if len(all_states) >= 12:
            # Large states vs small states (by population)
            large_states = all_states[:4]  # Assuming sorted by population
            small_states = all_states[-4:]
            
            # North vs South
            north_states = [s for s in all_states if any(x in s.lower() for x in 
                          ['schleswig', 'hamburg', 'bremen', 'niedersachsen', 'mecklenburg'])][:3]
            south_states = [s for s in all_states if any(x in s.lower() for x in 
                          ['bayern', 'baden', 'württemberg'])][:3]
            
            # East vs West  
            east_states = [s for s in all_states if any(x in s.lower() for x in 
                         ['sachsen', 'thüringen', 'brandenburg', 'berlin'])][:3]
            west_states = [s for s in all_states if any(x in s.lower() for x in 
                         ['nordrhein', 'hessen', 'rheinland'])][:3]
        else:
            # Fallback: create groups from available states
            n_states = len(all_states)
            large_states = all_states[:n_states//4] if n_states >= 4 else all_states[:2]
            small_states = all_states[n_states//2:n_states//2+2] if n_states >= 4 else all_states[-2:]
            north_states = all_states[:3] if n_states >= 3 else all_states
            south_states = all_states[-3:] if n_states >= 3 else all_states
            east_states = all_states[:2] if n_states >= 2 else all_states
            west_states = all_states[-2:] if n_states >= 2 else all_states
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 16), 
                                                    subplot_kw=dict(projection='polar'))
        
        fig.suptitle('German States Personality Profile Comparisons\nLLM-Inferred Big Five Traits', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Define the four comparison groups
        comparisons = [
            (large_states, "Major Population Centers", ax1),
            (small_states, "Smaller States/Regions", ax2), 
            (north_states + south_states, "North vs South", ax3),
            (east_states + west_states, "East vs West", ax4)
        ]
        
        angles = np.linspace(0, 2 * pi, len(self.personality_traits), endpoint=False).tolist()
        angles += angles[:1]
        
        for states_list, title, ax in comparisons:
            # Filter valid states
            valid_states = [s for s in states_list if s in all_states]
            if not valid_states:
                continue
                
            selected_data = self.state_results[
                self.state_results['state'].isin(valid_states)
            ].copy()
            
            # Color coding
            if "North vs South" in title:
                colors = ['blue'] * len([s for s in valid_states if s in north_states]) + \
                        ['red'] * len([s for s in valid_states if s in south_states])
            elif "East vs West" in title:
                colors = ['green'] * len([s for s in valid_states if s in east_states]) + \
                        ['orange'] * len([s for s in valid_states if s in west_states])
            else:
                colors = plt.cm.Set2(np.linspace(0, 1, len(selected_data)))
            
            # Plot each state in this group
            for idx, (_, state_row) in enumerate(selected_data.iterrows()):
                state_name = state_row['state']
                values = [state_row[f'{trait}_z'] for trait in self.personality_traits]
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, alpha=0.8, 
                       label=state_name, color=colors[idx])
                ax.fill(angles, values, alpha=0.1, color=colors[idx])
            
            # Customize each subplot
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(self.trait_labels, fontsize=10)
            ax.set_ylim(-2, 2)
            ax.set_yticks([-2, -1, 0, 1, 2])
            ax.set_yticklabels(['-2', '-1', '0', '1', '2'], fontsize=8, alpha=0.7)
            ax.grid(True, alpha=0.3)
            ax.set_title(title, pad=15, fontsize=12, fontweight='bold')
            
            # Add legend
            legend = ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), 
                              fontsize=9, frameon=True)
            legend.get_frame().set_alpha(0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-panel comparison saved to {save_path}")
        
        plt.show()
        return fig
    
    def create_top_states_comparison(self, trait='Openness', n_states=5, save_path=None):
        """
        Create spider chart comparing states with highest/lowest scores for a specific trait.
        
        Parameters:
        -----------
        trait : str
            Personality trait to focus on ('Openness', 'Conscientiousness', etc.)
        n_states : int
            Number of top and bottom states to show
        save_path : str, optional
            Path to save the figure
        """
        if self.state_results is None:
            print("No state results loaded")
            return
        
        if trait not in self.personality_traits:
            print(f"Invalid trait. Choose from: {self.personality_traits}")
            return
        
        # Sort states by the selected trait
        sorted_states = self.state_results.sort_values(f'{trait}_z', ascending=False)
        
        # Get top and bottom states
        top_states = sorted_states.head(n_states)['state'].tolist()
        bottom_states = sorted_states.tail(n_states)['state'].tolist()
        
        # Combine for comparison
        states_to_compare = top_states + bottom_states
        
        # Create the chart
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * pi, len(self.personality_traits), endpoint=False).tolist()
        angles += angles[:1]
        
        # Create two color groups: top states (warm colors) and bottom states (cool colors)
        warm_colors = plt.cm.Reds(np.linspace(0.4, 0.9, n_states))
        cool_colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_states))
        colors = list(warm_colors) + list(cool_colors)
        
        # Plot states
        for idx, state_name in enumerate(states_to_compare):
            state_data = self.state_results[self.state_results['state'] == state_name].iloc[0]
            values = [state_data[f'{t}_z'] for t in self.personality_traits]
            values += values[:1]
            
            # Different line styles for top vs bottom
            linestyle = '-' if idx < n_states else '--'
            alpha = 0.9 if idx < n_states else 0.7
            linewidth = 2.5 if idx < n_states else 2
            
            label = f"{state_name} ({state_data[f'{trait}_z']:.2f})"
            
            ax.plot(angles, values, linestyle, linewidth=linewidth, alpha=alpha, 
                   label=label, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        # Customize chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.trait_labels, fontsize=12, fontweight='bold')
        ax.set_ylim(-2.5, 2.5)
        ax.set_yticks(np.linspace(-2, 2, 5))
        ax.set_yticklabels([f'{val:.1f}' for val in np.linspace(-2, 2, 5)], 
                          fontsize=10, alpha=0.7)
        ax.grid(True, alpha=0.3)
        
        # Title with emphasis on selected trait
        title = f'German States: Highest vs Lowest {trait} Scores\nComplete Personality Profiles (Z-Scores)'
        plt.title(title, pad=20, fontsize=14, fontweight='bold')
        
        # Create custom legend
        top_patch = mpatches.Patch(color='red', alpha=0.7, label=f'Highest {trait} (solid)')
        bottom_patch = mpatches.Patch(color='blue', alpha=0.7, label=f'Lowest {trait} (dashed)')
        
        # Regular legend with state names
        legend1 = ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), 
                           fontsize=9, frameon=True)
        legend1.get_frame().set_alpha(0.9)
        
        # Add custom legend for top/bottom groups
        legend2 = ax.legend(handles=[top_patch, bottom_patch], 
                           loc='center left', bbox_to_anchor=(1.1, 0.2),
                           fontsize=10, frameon=True, title='Groups')
        legend2.get_frame().set_alpha(0.9)
        
        # Add both legends
        ax.add_artist(legend1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Top states comparison saved to {save_path}")
        
        plt.show()
        return fig, ax
    
    def print_state_summary(self):
        """Print a summary of available states and their personality profiles."""
        if self.state_results is None:
            print("No state results loaded")
            return
        
        print("\n" + "="*80)
        print("GERMAN STATES PERSONALITY PROFILE SUMMARY")
        print("="*80)
        print(f"Total states analyzed: {len(self.state_results)}")
        print(f"Users per state: {self.state_results['n_users'].sum():,} total")
        print("\nState rankings by personality traits (Z-scores):")
        print("-"*80)
        
        for trait in self.personality_traits:
            sorted_states = self.state_results.sort_values(f'{trait}_z', ascending=False)
            print(f"\n{trait.upper()}:")
            print("  Highest: " + ", ".join([
                f"{row['state']} ({row[f'{trait}_z']:.2f})" 
                for _, row in sorted_states.head(3).iterrows()
            ]))
            print("  Lowest:  " + ", ".join([
                f"{row['state']} ({row[f'{trait}_z']:.2f})" 
                for _, row in sorted_states.tail(3).iterrows()
            ]))

# Example usage and demonstration
def main():
    """Demonstrate the German States Spider Chart functionality."""
    
    # Initialize the spider chart creator
    spider = GermanStatesSpiderChart("spatial_personality_state_results.csv")
    
    # Print summary of available data
    spider.print_state_summary()
    
    # Example 1: Compare specific states
    if spider.state_results is not None and len(spider.state_results) >= 4:
        states_list = spider.state_results['state'].tolist()
        
        # Compare first 4 states
        spider.create_single_spider_chart(
            states_to_compare=states_list[:4],
            save_path="spider_chart_example_states.png",
            use_z_scores=True,
            title_suffix="Example State Comparison"
        )
        
        # Create multi-panel comparison
        spider.create_multi_panel_comparison(
            save_path="spider_chart_multi_panel.png"
        )
        
        # Compare top/bottom states for Openness
        spider.create_top_states_comparison(
            trait='Openness',
            n_states=3,
            save_path="spider_chart_openness_comparison.png"
        )

if __name__ == "__main__":
    main()
