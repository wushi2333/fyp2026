import torch
import torch.nn as nn
import time
import os
import sys
import pandas as pd
from thop import profile, clever_format

# 0. Basic path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

# Import models for comparison
from model_fbcsp_no_cnn import Model_MoE_FBCSP       # Proposed Linear + MoE model
from ablation_models import Ablation_CNNOnly         # CNN-only frontend baseline
from ablation_models import Ablation_StandardCNN_MoE # Standard CNN + MoE

# Force output to the advisor-specified directory
SAVE_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\engineering_metrics"
os.makedirs(SAVE_DIR, exist_ok=True)

# Define hyperparameters (must be consistent with actual training)
CONFIG = {
    'batch_size_train': 16,     # Batch size for training
    'batch_size_infer': 1,      # BCI online inference usually uses a single sample (Batch=1)
    'n_bands': 55,
    'n_csp': 8,
    'time_steps': 512,
    'embed_dim': 128,
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

def measure_latency(model, dummy_input, n_warmup=50, n_runs=500, tta_times=1):
    """
    Precisely measure the model inference latency (milliseconds).
    tta_times: Simulate the number of forward passes for Test Time Augmentation.
    """
    model.eval()
    
    # GPU warm-up to mitigate initial latency spikes
    with torch.no_grad():
        for _ in range(n_warmup):
            for _ in range(tta_times):
                _ = model(dummy_input)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    # Actual timing loop
    with torch.no_grad():
        for _ in range(n_runs):
            for _ in range(tta_times):
                _ = model(dummy_input)
                
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    # Compute average latency per trial (including TTA runs)
    avg_latency_ms = ((end_time - start_time) / n_runs) * 1000
    return avg_latency_ms

def evaluate_model_complexity(model, model_name, device):
    """
    Use thop to compute parameter count and computational cost (FLOPs).
    """
    model = model.to(device)
    # Simulate real input: (Batch, Bands, Time, CSP)
    dummy_input = torch.randn(CONFIG['batch_size_infer'], CONFIG['n_bands'], CONFIG['time_steps'], CONFIG['n_csp']).to(device)
    
    # 1. Compute Params and FLOPs (MACs)
    # custom_ops suppresses warnings for unsupported ops; the impact on magnitude is small
    macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.2f")
    
    # 2. Measure runtime latency: single forward pass (No TTA)
    latency_single = measure_latency(model, dummy_input, tta_times=1)
    
    # 3. Measure runtime latency: 7 forward passes (TTA=7)
    latency_tta7 = measure_latency(model, dummy_input, tta_times=7)
    
    return {
        'Model_Architecture': model_name,
        'Parameters': params_fmt,
        'FLOPs': macs_fmt,
        'Latency_NoTTA_ms': f"{latency_single:.2f}",
        'Latency_TTA7_ms': f"{latency_tta7:.2f}",
        'Real-Time Capability': 'Pass' if latency_tta7 < 100 else 'Warning' # BCI typically requires latency < 100ms
    }

def main():
    print(f"{'='*60}")
    print("STARTING ENGINEERING METRICS EVALUATION")
    print(f"Device: {CONFIG['device']}")
    print(f"Input Shape: (Batch=1, Bands=55, Time=512, CSP=8)")
    print(f"{'='*60}")
    
    results = []

    # 1. Instantiate our main model
    print("Evaluating: Proposed FBCSP-Linear-MoE...")
    model_proposed = Model_MoE_FBCSP(
        n_classes=2, n_bands=CONFIG['n_bands'], n_csp=CONFIG['n_csp'], 
        time_steps=CONFIG['time_steps'], embed_dim=CONFIG['embed_dim']
    )
    results.append(evaluate_model_complexity(model_proposed, "Proposed (Linear + MoE)", CONFIG['device']))

    # 2. Instantiate CNN-only baseline
    print("Evaluating: CNN Only Baseline...")
    model_cnn_only = Ablation_CNNOnly(
        n_bands=CONFIG['n_bands'], n_csp=CONFIG['n_csp'], 
        time_steps=CONFIG['time_steps'], embed_dim=CONFIG['embed_dim']
    )
    results.append(evaluate_model_complexity(model_cnn_only, "Baseline (CNN Only)", CONFIG['device']))

    # 3. Instantiate standard CNN + MoE model (show how replacing CNN with Linear can save parameters)
    print("Evaluating: Standard CNN + MoE...")
    model_cnn_moe = Ablation_StandardCNN_MoE(
        n_bands=CONFIG['n_bands'], n_csp=CONFIG['n_csp'], 
        time_steps=CONFIG['time_steps'], embed_dim=CONFIG['embed_dim']
    )
    results.append(evaluate_model_complexity(model_cnn_moe, "Standard CNN + MoE", CONFIG['device']))

    # Aggregate and save
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("ENGINEERING METRICS RESULTS:")
    print(df.to_string(index=False))
    
    # Extract the TTA cost analysis table
    tta_df = df[['Model_Architecture', 'Latency_NoTTA_ms', 'Latency_TTA7_ms']]
    
    csv_path = os.path.join(SAVE_DIR, "model_complexity_comparison.csv")
    df.to_csv(csv_path, index=False)
    tta_df.to_csv(os.path.join(SAVE_DIR, "tta_latency_cost.csv"), index=False)
    
    print(f"\nAll engineering reports saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()