import torch
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 0. Path Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from model_fbcsp_no_cnn import Model_MoE_FBCSP

# Force saving to a specified directory
SAVE_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\XAI"
os.makedirs(SAVE_DIR, exist_ok=True)

# Your best weights path for 20 runs (e.g., No Transfer)
WEIGHTS_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\experiment_20runs\no_transfer\best_weights"

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'batch_size': 32,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    'subjects': ['A01', 'A03', 'A04'], # Select representative subjects for explainability analysis
    'n_bands': 55,
    'n_csp': 8,
    'time_steps': 512,
    'embed_dim': 128,
    'classes': ['Left Hand', 'Right Hand']
}

# Restore labels for 55 frequency bands, used for plotting Y-axis
freq_bands = []
for start in range(8, 29): freq_bands.append(f"{start}-{start + 2}Hz")
for start in range(8, 27): freq_bands.append(f"{start}-{start + 4}Hz")
for start in range(8, 23): freq_bands.append(f"{start}-{start + 8}Hz")

# ==========================================
# Experiment A: Expert Routing Visualization
# ==========================================
def experiment_a_routing_visualization():
    print(f"\n{'='*50}\nExperiment A: Expert Routing Visualization\n{'='*50}")
    
    routing_data = []
    
    for subj in CONFIG['subjects']:
        weight_path = os.path.join(WEIGHTS_DIR, f"{subj}_best_model.pth")
        if not os.path.exists(weight_path):
            print(f"Weights for {subj} not found. Skipping...")
            continue
            
        model = Model_MoE_FBCSP(n_classes=2, n_bands=CONFIG['n_bands'], n_csp=CONFIG['n_csp'], 
                                time_steps=CONFIG['time_steps'], embed_dim=CONFIG['embed_dim']).to(CONFIG['device'])
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']))
        model.eval()

        # Hook mechanism to intercept Gate activations
        gates_record = {'Left Hand': [], 'Right Hand': []}
        def get_gates_hook(module, input, output):
            gates, aux_loss = output
            # gates shape: (Batch, Time, Experts) -> (Batch, Experts)
            active_count = (gates > 0).float().mean(dim=1).detach().cpu().numpy()
            return active_count
        
        # Attach to the Router of the last Transformer layer
        hook = model.layers[-1].router.register_forward_hook(
            lambda m, i, o: gates_record[current_class].append((o[0] > 0).float().mean(dim=1).detach().cpu().numpy())
        )

        test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subj)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(CONFIG['device'])
                for i in range(len(y)):
                    label = y[i].item()
                    current_class = CONFIG['classes'][0 if label == y.min() else 1]
                    # Forward pass for each sample to accurately record classification
                    _ = model(x[i].unsqueeze(0))
                    
        hook.remove()
        
        # Calculate average routing load rate
        for cls_name in CONFIG['classes']:
            if len(gates_record[cls_name]) > 0:
                avg_load = np.vstack(gates_record[cls_name]).mean(axis=0)
                for exp_idx, load in enumerate(avg_load):
                    routing_data.append({
                        'Subject': subj,
                        'Class': cls_name,
                        'Expert': f"E{exp_idx+1}",
                        'Load_Rate': load
                    })

    df_routing = pd.DataFrame(routing_data)
    
    # Plot grouped heatmap
    g = sns.FacetGrid(df_routing, col="Subject", height=4, aspect=1.2)
    def draw_heatmap(*args, **kwargs):
        data = kwargs.pop('data')
        pivot = data.pivot(index='Class', columns='Expert', values='Load_Rate')
        sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=".2f", vmin=0, vmax=1, 
                    cbar_kws={'label': 'Activation Rate'}, **kwargs)
    g.map_dataframe(draw_heatmap)
    g.figure.suptitle("Sparse Expert Activation Distribution Across Subjects and Tasks", y=1.05, fontsize=14)
    
    plt.savefig(os.path.join(SAVE_DIR, "ExpA_Expert_Routing_Heatmap.png"), dpi=300, bbox_inches='tight')
    df_routing.to_csv(os.path.join(SAVE_DIR, "ExpA_Expert_Routing_Data.csv"), index=False)
    print(f"Exp A completed. Saved to {SAVE_DIR}\\ExpA_Expert_Routing_Heatmap.png")

# ==========================================
# Experiment B: Physiological Mechanism Explanation - Spatial-Spectral Gradient Saliency Analysis (Grad x Input Saliency)
# ==========================================
def experiment_b_physiological_saliency():
    print(f"\n{'='*50}\nExperiment B: Physiological Saliency Analysis\n{'='*50}")
    
    for subj in CONFIG['subjects']:
        weight_path = os.path.join(WEIGHTS_DIR, f"{subj}_best_model.pth")
        if not os.path.exists(weight_path): continue
            
        model = Model_MoE_FBCSP(n_classes=2, n_bands=CONFIG['n_bands'], n_csp=CONFIG['n_csp'], 
                                time_steps=CONFIG['time_steps'], embed_dim=CONFIG['embed_dim']).to(CONFIG['device'])
        model.load_state_dict(torch.load(weight_path, map_location=CONFIG['device']))
        model.eval()

        test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subj)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
        global_saliency = np.zeros((CONFIG['n_bands'], CONFIG['n_csp']))
        sample_count = 0
        
        for x, y in test_loader:
            x = x.to(CONFIG['device'])
            x.requires_grad_(True) # Enable gradient capture
            
            logits, _ = model(x)
            
            # Calculate gradients for the predicted class (Grad-CAM basic logic)
            score = logits.gather(1, logits.argmax(dim=1).unsqueeze(1)).sum()
            model.zero_grad()
            score.backward()
            
            # Saliency = |Gradient * Input|
            saliency = (x.grad * x).abs()
            
            # shape: (Batch, Bands, Time, CSP) -> Average over Batch and Time
            saliency_spatial_spectral = saliency.mean(dim=(0, 2)).detach().cpu().numpy()
            global_saliency += saliency_spatial_spectral
            sample_count += 1
            
        global_saliency /= sample_count
        
        # Normalize for visualization
        global_saliency = (global_saliency - global_saliency.min()) / (global_saliency.max() - global_saliency.min() + 1e-8)
        
        # Plot spatial-spectral saliency map
        plt.figure(figsize=(8, 12))
        ax = sns.heatmap(global_saliency, cmap="jet", yticklabels=freq_bands, 
                         xticklabels=[f"CSP {i+1}" for i in range(CONFIG['n_csp'])])
        
        # Highlight approximate row numbers for Mu band (8-12Hz) and Beta band (13-30Hz)
        # Our first few rows are 8-10, 9-11, 10-12, belonging to the Mu band
        ax.axhspan(0, 5, color='white', alpha=0.3, label='Mu Band (8-12 Hz)')
        ax.axhspan(20, 40, color='grey', alpha=0.3, label='Beta Band (13-30 Hz)')
        
        plt.title(f"Spatial-Spectral Attention Saliency Map - Subject {subj}\n(Where the model looks at)", fontsize=14)
        plt.ylabel("Frequency Bands", fontsize=12)
        plt.xlabel("Spatial Components (CSP Filters)", fontsize=12)
        plt.legend(loc="upper right")
        
        save_path = os.path.join(SAVE_DIR, f"ExpB_Saliency_Map_{subj}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Exp B completed. Heatmaps saved to {SAVE_DIR}")

if __name__ == "__main__":
    experiment_a_routing_visualization()
    experiment_b_physiological_saliency()
    print("All XAI analysis complete!")