import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

# Configure paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
process_path = os.path.join(project_root, 'process')

if structure_path not in sys.path: sys.path.append(structure_path)
if process_path not in sys.path: sys.path.append(process_path)

# Import your unified preprocessor
from test_csp_baseline import UnifiedEEGPreprocessor

# 1. Dynamically override the preprocessor to completely disable Z-score
class UnnormalizedPreprocessor(UnifiedEEGPreprocessor):
    def _robust_z_score(self, data: np.ndarray) -> np.ndarray:
        # Magic here: intercept the Z-score operation and return raw physical voltages unchanged!
        return data

def extract_raw_energy_features(raw_data_path, dataset_type, processor):
    """Extract unnormalized band energy features (simulate raw spatial manifold before FBCSP)."""
    print(f"Loading unnormalized data from {dataset_type}...")
    
    # Load data with modified processor
    if dataset_type == 'BCIC IV 2a':
        X, _ = processor.process_bcic_iv_2a(raw_data_path)
    elif dataset_type == 'OpenBMI':
        X, _ = processor.process_openbmi(raw_data_path)
    elif dataset_type == 'PhysioNet':
        X, _ = processor.process_physionet(raw_data_path, runs=[4, 8, 12])
        
    if X is None:
        raise ValueError(f"Failed to load data for {dataset_type}")
        
    # X shape is (Trials, Channels, Time)
    # Compute log-variance for each channel, the core CSP feature representation
    # Without Z-score, variance reflects the amplifier's absolute energy level
    variances = np.var(X, axis=2)
    log_variances = np.log(variances + 1e-8)
    
    # Randomly select up to 300 samples to keep chart clear
    max_samples = min(300, len(log_variances))
    indices = np.random.choice(len(log_variances), max_samples, replace=False)
    
    return log_variances[indices]

def plot_tsne_domain_drift_unnormalized():
    plt.rcParams['font.family'] = 'Times New Roman'
    

    paths = {
        'BCIC IV 2a': r"D:\fyp\dataset\BCIC_IV_2a\BCICIV_2a_gdf\A01T.gdf",
        'OpenBMI': r"D:\fyp\dataset\openBMI\MNE-lee2019-mi-data\gigadb-datasets\live\pub\10.5524\100001_101000\100542\session1\s4\sess01_subj04_EEG_MI.mat",
        'PhysioNet': r"D:\fyp\dataset\PhysioNet\MNE-eegbci-data\files\eegmmidb\1.0.0\S001\S001R04.edf"
    }
    
    processor = UnnormalizedPreprocessor(target_rate=128)
    
    X_bcic = extract_raw_energy_features(paths['BCIC IV 2a'], 'BCIC IV 2a', processor)
    X_physio = extract_raw_energy_features(paths['PhysioNet'], 'PhysioNet', processor)
    X_openbmi = extract_raw_energy_features(paths['OpenBMI'], 'OpenBMI', processor)
    
    X_all = np.vstack([X_bcic, X_physio, X_openbmi])
    labels = ['BCIC IV 2a (Target Domain)'] * len(X_bcic) + \
             ['PhysioNet (Source Domain)'] * len(X_physio) + \
             ['OpenBMI (Source Domain)'] * len(X_openbmi)

    print("\nRunning t-SNE dimensionality reduction on UNNORMALIZED features...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca', learning_rate='auto')
    X_tsne = tsne.fit_transform(X_all)
    
    df = pd.DataFrame({
        't-SNE Dimension 1': X_tsne[:, 0],
        't-SNE Dimension 2': X_tsne[:, 1],
        'Dataset / Domain': labels
    })
    
    plt.figure(figsize=(9, 7))
    palette = {
        'BCIC IV 2a (Target Domain)': '#d62728', 
        'PhysioNet (Source Domain)': '#1f77b4',  
        'OpenBMI (Source Domain)': '#2ca02c'     
    }
    
    sns.scatterplot(
        data=df, x='t-SNE Dimension 1', y='t-SNE Dimension 2', 
        hue='Dataset / Domain', palette=palette, alpha=0.8, edgecolor=None, s=50
    )
    
    plt.title('t-SNE Visualization of Feature Space Mismatch (Domain Drift)', 
              fontsize=16, fontweight='bold', pad=15)
              
    plt.legend(title='Domain Status', fontsize=11, title_fontsize=12, loc='best')
    plt.tight_layout()
    
    save_path = os.path.join(current_dir, 'Figure_6_Unnormalized_Drift.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure generated and saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    plot_tsne_domain_drift_unnormalized()