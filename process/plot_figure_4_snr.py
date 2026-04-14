import numpy as np
import matplotlib.pyplot as plt

def plot_real_snr_augmentation(npy_file_path, target_class=0, band_idx=0, csp_idx=0):
    """
    Plot an S&R data augmentation illustration using real FBCSP-processed EEG data.
    
    Args:
    npy_file_path: Full path to your train_data.npy file.
    target_class: Which class to display (0 or 1).
    band_idx: Frequency band index to visualize (0 is typically the first band, e.g., 8-10Hz).
    csp_idx: CSP component index to visualize (0 is usually the most discriminative component).
    """
    # 1. Load real data
    print(f"Loading real data from: {npy_file_path}")
    data = np.load(npy_file_path, allow_pickle=True).item()
    X_all = data['X']  # Expected shape: (Batch, Bands, Time, CSP)
    y_all = data['y']
    
    # Filter samples of the selected class
    class_indices = np.where(y_all == target_class)[0]
    if len(class_indices) < 3:
        raise ValueError("Not enough samples in this class to generate the figure (at least 3 required)!")
    
    # Randomly select 3 sample indices
    np.random.seed(42) # Fix random seed to ensure reproducible figures
    selected_indices = np.random.choice(class_indices, 3, replace=False)
    
    # Extract 1D time series: (Time,)
    # Time length should be 512 (4 sec * 128Hz)
    src_data_1 = X_all[selected_indices[0], band_idx, :, csp_idx]
    src_data_2 = X_all[selected_indices[1], band_idx, :, csp_idx]
    src_data_3 = X_all[selected_indices[2], band_idx, :, csp_idx]
    
    time_points = len(src_data_1)
    t = np.linspace(0, 4.0, time_points) # 4-second trial
    
    # Font and figure settings
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1, 1, 1.5]})
    fig.suptitle('Divide and Recombine (S&R) Data Augmentation Strategy', 
                 fontsize=16, fontweight='bold', y=0.98)

    # Parameter setup
    num_segments = 10
    segment_len = time_points // num_segments
    
    sources = [
        {'name': f'Source Trial 1 (Class {target_class})', 'color': '#1f77b4', 'data': src_data_1},
        {'name': f'Source Trial 2 (Class {target_class})', 'color': '#ff7f0e', 'data': src_data_2},
        {'name': f'Source Trial 3 (Class {target_class})', 'color': '#2ca02c', 'data': src_data_3}
    ]
    
    # Manually specify a recombination pattern for visualization
    synthesis_pattern = [0, 2, 1, 0, 2, 2, 1, 0, 1, 2]
    synthetic_data = np.zeros_like(t)

    # Get a unified Y-axis range for consistent subplot scaling
    y_min = min([src['data'].min() for src in sources]) * 1.1
    y_max = max([src['data'].min() for src in sources]) * 1.1

    # 2. Plot the three source trials in the top section
    for i, src in enumerate(sources):
        ax = axes[i]
        ax.plot(t, src['data'], color=src['color'], linewidth=1.2)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(src['name'], fontsize=12, loc='left', color=src['color'], fontweight='bold')
        ax.set_xlim(0, 4.0)
        # Hide some ticks to make chart cleaner
        ax.set_xticks([]) 
        ax.set_yticks([])

        # Draw vertical segment boundaries and background patches
        for seg in range(num_segments):
            x_start = seg * 4.0 / num_segments
            x_end = (seg + 1) * 4.0 / num_segments
            ax.axvline(x=x_start, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            
            # Highlight the segment if it is used in synthesis
            if synthesis_pattern[seg] == i:
                ax.axvspan(x_start, x_end, facecolor=src['color'], alpha=0.15)

    # 3. Plot the synthetic augmented trial in the bottom section
    ax_syn = axes[3]
    for seg in range(num_segments):
        start_idx = seg * segment_len
        end_idx = (seg + 1) * segment_len if seg < num_segments - 1 else time_points
        
        src_idx = synthesis_pattern[seg]
        src_color = sources[src_idx]['color']
        
        # Fill the synthetic signal segment with real data
        synthetic_data[start_idx:end_idx] = sources[src_idx]['data'][start_idx:end_idx]
        
        # Plot the recombined segment
        ax_syn.plot(t[start_idx:end_idx], synthetic_data[start_idx:end_idx], color=src_color, linewidth=1.5)
        
        # Draw separators and filled spans
        x_start = seg * 4.0 / num_segments
        x_end = (seg + 1) * 4.0 / num_segments
        ax_syn.axvline(x=x_start, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax_syn.axvspan(x_start, x_end, facecolor=src_color, alpha=0.2)
        
        # Label the source trial above each segment
        y_text_pos = np.max(synthetic_data) * 1.05
        ax_syn.text((x_start + x_end)/2, y_text_pos, f"T{src_idx+1}", ha='center', va='center', 
                    fontsize=10, color=src_color, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor=src_color, boxstyle='round,pad=0.2'))

    ax_syn.set_title('Synthetic Augmented Trial (Recombined from Real FBCSP Features)', fontsize=14, loc='left', fontweight='bold')
    ax_syn.set_xlabel('Time (Seconds) - Partitioned into 10 Temporal Segments', fontsize=12)
    ax_syn.set_ylabel('Amplitude', fontsize=10)
    ax_syn.set_xlim(0, 4.0)
    ax_syn.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)
    
    save_path = 'Figure_4_Divide_and_Recombine_Real.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure successfully generated and saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    # point to a preprocessed local numpy file
    REAL_DATA_FILE = r"D:\fyp\dataset_processed_fbcsp_all\BCIC_IV_2a\A01T\train_data.npy"
    
    try:
        plot_real_snr_augmentation(REAL_DATA_FILE)
    except FileNotFoundError:
        print(f"Error: File not found {REAL_DATA_FILE}. Please ensure the path is correct and the file was generated by data_process_fbcsp_all.py.")