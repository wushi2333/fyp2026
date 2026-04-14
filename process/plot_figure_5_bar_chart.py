import numpy as np
import matplotlib.pyplot as plt

def plot_figure_5():
    # Use academic standard font uniformly
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # Extract your data (keep two decimals for display)
    subjects = ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
    
    # Standard Transformer Best Acc (%)
    std_acc = [96.55, 79.31, 100.00, 86.21, 93.10, 75.86, 86.21, 96.55, 89.66]
    
    # MoE (No Transfer / From Scratch) Best Acc (%)
    moe_acc = [96.55, 75.86, 100.00, 89.66, 86.21, 75.86, 89.66, 96.55, 93.10]
    
    # Set bar width and positions
    x = np.arange(len(subjects))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot grouped bar chart
    # Blue represents the standard Transformer baseline, orange represents the MoE architecture
    rects1 = ax.bar(x - width/2, std_acc, width, label='Standard Transformer', color='#1f77b4', alpha=0.85)
    rects2 = ax.bar(x + width/2, moe_acc, width, label='MoE Transformer (From Scratch)', color='#ff7f0e', alpha=0.85)
    
    # Function to add value labels
    def autolabel(rects, is_moe=False):
        for i, rect in enumerate(rects):
            height = rect.get_height()
            
            # Mark subjects to highlight (A04: index 3, A07: index 6, A09: index 8)
            highlight_idx = [3, 6, 8]
            
            # If this is MoE and these three subjects have higher accuracy, highlight with bold red text and an upward arrow
            if is_moe and i in highlight_idx and moe_acc[i] > std_acc[i]:
                ax.annotate(f'↑ {height:.1f}%',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=11, color='#d62728', fontweight='bold')
                
                # Add a bold red border to the bar
                rect.set_edgecolor('#d62728')
                rect.set_linewidth(2)
            else:
                ax.annotate(f'{height:.1f}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), 
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10, color='black')

    # Call the annotation function
    autolabel(rects1, is_moe=False)
    autolabel(rects2, is_moe=True)
    
    # Adjust chart details
    ax.set_ylabel('Best Classification Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Subjects (BCIC IV 2a)', fontsize=14, fontweight='bold')
    ax.set_title('Peak Accuracy Comparison Illuminating MoE Performance Ceilings', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=12)
    
    # Set y-axis limits to leave space for value labels
    ax.set_ylim(60, 105)
    
    # Add grid lines for readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Adjust the legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add an annotation box explaining the red arrow meaning
    props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#d62728', alpha=0.9)
    ax.text(0.02, 0.95, 'Highlighted (↑): Effectively \nextends in MoE architecture', 
            transform=ax.transAxes, fontsize=11, color='#d62728', fontweight='bold',
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save image
    save_path = 'Figure_5_MoE_Peak_Accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure generated and saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    plot_figure_5()