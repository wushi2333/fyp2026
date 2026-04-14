import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
import gc
import random

# 0. Basic path and environment configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from model_fbcsp_no_cnn import Model_MoE_FBCSP

# Force output to the advisor-specified directory
SAVE_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\ablation_params"
os.makedirs(SAVE_DIR, exist_ok=True)

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'batch_size': 16,
    'lr': 0.0001,
    'epochs': 60,           # Reduce ablation epochs to 60 to speed up the search
    'rounds_per_cond': 10,  # Run each parameter setting 10 times to compute mean and std
    'n_bands': 55,
    'device': 'cuda:0',
    # Select representative subjects: A03(best), A01(medium), A04(worst/largest gap)
    'subjects': ['A01', 'A03', 'A04'], 
    # Ablation parameter grid
    'experts_list':[4, 8, 16],
    'topk_list': [1, 2, 4]
}

# 1. Helper functions
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 2. Single-condition training function (Train from scratch, no transfer learning to compare architecture capacity)
def train_single_condition(subject_id, n_experts, top_k, run_idx):
    set_seed(42 + run_idx * 100)
    
    train_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id, 
        snr_aug=True, snr_prob=0.8, num_segments=10
    )
    test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    # Dynamically pass ablation parameters
    model = Model_MoE_FBCSP(
        n_classes=2, n_bands=CONFIG['n_bands'], n_csp=8, time_steps=512, 
        embed_dim=128, depth=4, heads=8, 
        num_experts=n_experts, top_k=top_k, dropout=0.5
    ).to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.1)
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1)
    
    best_acc = 0.0

    for epoch in range(CONFIG['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            out, aux = model(x)
            loss = criterion(out, y) + 0.1 * aux
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Validation evaluation (with 7x TTA)
        model.eval()
        preds, targets = [],[]
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                
                tta_logits =[]
                for _ in range(7):
                    logits, _ = model(x)
                    tta_logits.append(logits)
                
                avg_logits = torch.stack(tta_logits).mean(0)
                _, p = torch.max(avg_logits, 1)
                preds.extend(p.cpu().numpy())
                targets.extend(y.cpu().numpy())
        
        acc = accuracy_score(targets, preds) * 100
        if acc > best_acc:
            best_acc = acc
            
    return best_acc

# 3. Result visualization and report generation
def generate_report_and_plots(raw_results):
    df = pd.DataFrame(raw_results)
    df.to_csv(os.path.join(SAVE_DIR, "raw_ablation_results.csv"), index=False)
    
    # Group by N_Experts and Top_K to compute mean and std
    summary = df.groupby(['N_Experts', 'Top_K'])['Best_Acc'].agg(['mean', 'std']).reset_index()
    summary['Mean±Std'] = summary.apply(lambda row: f"{row['mean']:.2f} ± {row['std']:.2f}", axis=1)
    
    # 1. Save a paper-ready formatted table
    pivot_table_str = summary.pivot(index='N_Experts', columns='Top_K', values='Mean±Std')
    pivot_table_str.to_csv(os.path.join(SAVE_DIR, "thesis_ready_table.csv"))
    print("\n[Thesis Ready Table] (Mean ± Std)")
    print(pivot_table_str)

    # 2. Plot a performance heatmap
    pivot_table_num = summary.pivot(index='N_Experts', columns='Top_K', values='mean')
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_table_num, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average Accuracy (%)'})
    plt.title("Parameter Ablation: Number of Experts vs Top-K Routing", fontsize=14)
    plt.xlabel("Top-K Selected", fontsize=12)
    plt.ylabel("Total Number of Sparse Experts", fontsize=12)
    plt.savefig(os.path.join(SAVE_DIR, "heatmap_experts_vs_topk.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot grouped bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='N_Experts', y='Best_Acc', hue='Top_K', capsize=.1, errorbar='sd', palette='viridis')
    plt.title("Ablation Study of MoE Routing Parameters", fontsize=14)
    plt.xlabel("Total Number of Experts", fontsize=12)
    plt.ylabel("Classification Accuracy (%)", fontsize=12)
    plt.legend(title='Top-K')
    plt.ylim(bottom=df['Best_Acc'].min() - 5, top=min(100, df['Best_Acc'].max() + 5))
    plt.savefig(os.path.join(SAVE_DIR, "barplot_experts_vs_topk.png"), dpi=300, bbox_inches='tight')
    plt.close()

# 4. Main scheduling logic
def main():
    print(f"{'='*60}")
    print("STARTING MoE PARAMETER ABLATION STUDY")
    print(f"Target Subjects: {CONFIG['subjects']}")
    print(f"Experts Grid: {CONFIG['experts_list']}")
    print(f"Top-K Grid: {CONFIG['topk_list']}")
    print(f"Results will be strictly saved to: {SAVE_DIR}")
    print(f"{'='*60}")

    raw_results = []
    total_combinations = len(CONFIG['experts_list']) * len(CONFIG['topk_list'])
    current_combo = 0

    for n_experts in CONFIG['experts_list']:
        for top_k in CONFIG['topk_list']:
            current_combo += 1
            # Skip unreasonable parameter combinations (e.g. Top-K larger than number of experts)
            if top_k > n_experts:
                continue
            
            print(f"\n>>> Combo {current_combo}/{total_combinations}: N_Experts={n_experts}, Top_K={top_k} <<<")
            
            for subj in CONFIG['subjects']:
                subj_accs = []
                for r in range(CONFIG['rounds_per_cond']):
                    acc = train_single_condition(subj, n_experts, top_k, r)
                    subj_accs.append(acc)
                    
                    # Collect in real-time for easy interruption
                    raw_results.append({
                        'Subject': subj,
                        'N_Experts': n_experts,
                        'Top_K': top_k,
                        'Run': r + 1,
                        'Best_Acc': acc
                    })
                    print(f"  [{subj}] Run {r+1}/{CONFIG['rounds_per_cond']} - Acc: {acc:.2f}%", end='\r')
                
                avg_acc = np.mean(subj_accs)
                std_acc = np.std(subj_accs)
                print(f"  [{subj}] Done! Mean: {avg_acc:.2f}% ± {std_acc:.2f}%{' '*20}")
                
            # After each parameter combination, immediately save a backup in case of interruption
            generate_report_and_plots(raw_results)
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY COMPLETED. All plots and CSVs are saved to:")
    print(SAVE_DIR)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()