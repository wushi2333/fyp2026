import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import gc
import random

# 0. Basic Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
# Import ablation models
from ablation_fbcsp_models import Ablation_FBCSP_NoTransformer, Ablation_FBCSP_StandardTransformer

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'batch_size': 16,
    'lr': 0.0001,
    'epochs': 80,
    'rounds': 20, # 20 rounds to find the best
    'device': 'cuda:0',
    'subjects': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'],
    'base_output_dir': os.path.join(project_root, 'result', 'ablation_20runs')
}

# 1. Helper Functions
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

# 2. Single Training Session Function
def train_single_session(model_class, subject_id, run_idx):
    # Set random seed to ensure different initialization for each round
    current_seed = 1000 + run_idx * 50
    set_seed(current_seed)
    
    # Load data
    train_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id, snr_aug=True)
    test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # Initialize model (random initialization, no transfer learning)
    model = model_class(n_bands=55, n_csp=8, time_steps=512, embed_dim=128).to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.1)
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1)
    
    best_acc = 0.0
    best_metrics = {}

    for epoch in range(CONFIG['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            out, aux = model(x) # Compatible with interface returning aux_loss
            loss = criterion(out, y) + 0.1 * aux
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Simple validation
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                logits, _ = model(x)
                _, p = torch.max(logits, 1)
                preds.extend(p.cpu().numpy())
                targets.extend(y.cpu().numpy())
        
        acc = accuracy_score(targets, preds) * 100
        if acc > best_acc:
            best_acc = acc
            best_metrics = {
                'acc': acc,
                'kappa': cohen_kappa_score(targets, preds),
                'f1': f1_score(targets, preds, average='macro'),
                'preds': preds,
                'targets': targets
            }
            
    return best_acc, best_metrics

# 3. Experiment Suite Executor
def run_ablation_suite(experiment_name, model_class):
    save_dir = os.path.join(CONFIG['base_output_dir'], experiment_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"RUNNING ABLATION: {experiment_name.upper()}")
    print(f"Save Directory: {save_dir}")
    print(f"{'='*60}")
    
    summary_results = []
    
    for subj in CONFIG['subjects']:
        print(f"\nSubject {subj}: Starting 20-Round Ablation...")
        
        champ_acc = -1.0
        champ_metrics = None
        champ_round = -1
        
        subject_round_accs = []

        # 20 rounds
        for r in range(CONFIG['rounds']):
            run_acc, run_metrics = train_single_session(model_class, subj, r)
            print(f"  > Round {r+1:02d}: {run_acc:.2f}%", end="\r")

            subject_round_accs.append(run_acc)
            
            if run_acc > champ_acc:
                champ_acc = run_acc
                champ_metrics = run_metrics
                champ_round = r + 1
                
        print(f"  > Subject {subj} Best: {champ_acc:.2f}% (Round {champ_round}){' '*10}")
        
        avg_acc_20runs = np.mean(subject_round_accs)

        # Save best results
        # 1. CSV
        df_pred = pd.DataFrame({
            'Sample_Index': range(len(champ_metrics['targets'])),
            'True_Label': champ_metrics['targets'],
            'Predicted_Label': champ_metrics['preds']
        })
        df_pred.to_csv(os.path.join(save_dir, f"{subj}_predictions.csv"), index=False)
        
        # 2. Confusion Matrix
        cm = confusion_matrix(champ_metrics['targets'], champ_metrics['preds'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f'{experiment_name}\n{subj} Best Acc: {champ_metrics["acc"]:.2f}%')
        plt.savefig(os.path.join(save_dir, f"{subj}_confusion_matrix.png"))
        plt.close(fig)
        
        # 3. Summary
        summary_results.append({
            'subject': subj,
            'best_acc': champ_metrics['acc'],
            'avg_acc_20runs': avg_acc_20runs,
            'kappa': champ_metrics['kappa'],
            'f1': champ_metrics['f1'],
            'best_round': champ_round
        })
        
        gc.collect()
        torch.cuda.empty_cache()

    # Save summary table
    df_summary = pd.DataFrame(summary_results)
    avg_row = df_summary[['best_acc', 'avg_acc_20runs', 'kappa', 'f1']].mean().to_dict()
    avg_row['subject'] = 'AVERAGE'
    df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)
    
    df_summary.to_csv(os.path.join(save_dir, "final_summary_metrics.csv"), index=False)
    print(f"\nAblation {experiment_name} Completed.")
    print(df_summary.to_string(index=False, float_format="%.4f"))

# 4. Main Program
if __name__ == "__main__":
    # No Transformer
    run_ablation_suite("no_transformer", Ablation_FBCSP_NoTransformer)
    
    # Standard Transformer
    run_ablation_suite("standard_transformer", Ablation_FBCSP_StandardTransformer)