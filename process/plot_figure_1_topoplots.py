import mne
import matplotlib.pyplot as plt
from mne.decoding import CSP
import numpy as np
import warnings

warnings.filterwarnings("ignore")
mne.set_log_level('ERROR')

def plot_fbcsp_topoplots(gdf_file_path):
    # Use the same 22 target channels as in your preprocessing script
    target_channels = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 
        'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 
        'P1', 'Pz', 'P2', 'POz'
    ]

    print(f"Loading data from {gdf_file_path}...")
    # 1. Load and align data [cite: 1853, 1854]
    raw = mne.io.read_raw_gdf(gdf_file_path, preload=True)
    
    current_names = raw.ch_names
    rename_map = {current_names[i]: target_channels[i] for i in range(22)}
    raw.rename_channels(rename_map)
    raw.pick_channels(target_channels)
    
    # [Key step] Set standard 10-20 coordinates, which is a prerequisite for plotting topoplots
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    
    # Get event labels (Left: 769, Right: 770)
    events, annot_map = mne.events_from_annotations(raw)
    event_id = {'Left': annot_map.get('769'), 'Right': annot_map.get('770')}
    
    # 2. Filter in Mu and Beta bands respectively
    print("Filtering into Mu (8-12 Hz) and Beta (13-30 Hz) bands...")
    raw_mu = raw.copy().filter(l_freq=8.0, h_freq=12.0, fir_design='firwin')
    raw_beta = raw.copy().filter(l_freq=13.0, h_freq=30.0, fir_design='firwin')
    
    # Split Epochs, duration set to 4.0 seconds
    epochs_mu = mne.Epochs(raw_mu, events, event_id, tmin=0, tmax=4.0, baseline=None, preload=True)
    epochs_beta = mne.Epochs(raw_beta, events, event_id, tmin=0, tmax=4.0, baseline=None, preload=True)
    
    y = epochs_mu.events[:, -1]
    
    # 3. Fit CSP
    print("Fitting CSP for both bands...")
    csp_mu = CSP(n_components=8, reg=None, log=True, norm_trace=False)
    csp_mu.fit(epochs_mu.get_data(), y)
    
    csp_beta = CSP(n_components=8, reg=None, log=True, norm_trace=False)
    csp_beta.fit(epochs_beta.get_data(), y)
    
    # 4. Plot Topoplots (2x2)
    # CSP patterns_ are sorted by eigenvalues. Usually, index 0 represents Class 1 (Left), and index -1 represents Class 2 (Right)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('FBCSP Spatial Energy Distributions (Patterns)', fontsize=16, fontfamily='Times New Roman', fontweight='bold')
    
    def format_ax(ax, title):
        ax.set_title(title, fontsize=14, fontfamily='Times New Roman')

    # First row: Mu band
    mne.viz.plot_topomap(csp_mu.patterns_[0], epochs_mu.info, axes=axes[0, 0], show=False, cmap='RdBu_r')
    format_ax(axes[0, 0], 'Mu Band (8-12 Hz) - Left Hand')
    
    mne.viz.plot_topomap(csp_mu.patterns_[-1], epochs_mu.info, axes=axes[0, 1], show=False, cmap='RdBu_r')
    format_ax(axes[0, 1], 'Mu Band (8-12 Hz) - Right Hand')
    
    # Second row: Beta band
    mne.viz.plot_topomap(csp_beta.patterns_[0], epochs_beta.info, axes=axes[1, 0], show=False, cmap='RdBu_r')
    format_ax(axes[1, 0], 'Beta Band (13-30 Hz) - Left Hand')
    
    mne.viz.plot_topomap(csp_beta.patterns_[-1], epochs_beta.info, axes=axes[1, 1], show=False, cmap='RdBu_r')
    format_ax(axes[1, 1], 'Beta Band (13-30 Hz) - Right Hand')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90) # Leave space for the main title
    
    # Save image with Gaussian clarity for reporting
    save_path = 'Figure_1_FBCSP_Topoplots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Figure successfully saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    TEST_FILE = r"D:\fyp\dataset\BCIC_IV_2a\BCICIV_2a_gdf\A01T.gdf"
    plot_fbcsp_topoplots(TEST_FILE)