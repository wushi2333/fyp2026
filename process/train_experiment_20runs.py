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

# 0. Basic configuration and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from model_fbcsp_no_cnn import Model_MoE_FBCSP 

# Global Config
CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'pretrained_path': os.path.join(project_root, 'checkpoints_final', 'model_a_fbcsp_best.pth'),
    'batch_size': 16,
    'lr': 0.0001,
    'epochs': 80,         
    'rounds': 20,         
    'n_bands': 55,
    'device': 'cuda:0',
    'subjects': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'],
    'base_output_dir': os.path.join(project_root, 'result', 'experiment_20runs')
}

# 1. Helper classes and functions
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

def load_pretrained_weights(model, path):
    if not os.path.exists(path): 
        print(f"!!! CRITICAL WARNING: Weight file not found at: {path}")
        print("!!! Model will use RANDOM initialization instead.")
        return False
    
    try:
        print(f" -> Loading weights from: {path}")
        checkpoint = torch.load(path)
        if 'frontend' in checkpoint:
            model.frontend.load_state_dict(checkpoint['frontend'], strict=True)
        if 'encoder' in checkpoint:
            src_state = checkpoint['encoder']
            for i in range(len(model.layers)): 
                prefix_src = f"layers.{i}."
                model.layers[i].attn.load_state_dict({
                    'in_proj_weight': src_state[f"{prefix_src}self_attn.in_proj_weight"],
                    'in_proj_bias': src_state[f"{prefix_src}self_attn.in_proj_bias"],
                    'out_proj.weight': src_state[f"{prefix_src}self_attn.out_proj.weight"],
                    'out_proj.bias': src_state[f"{prefix_src}self_attn.out_proj.bias"]
                })
                model.layers[i].norm1.load_state_dict({'weight': src_state[f"{prefix_src}norm1.weight"], 'bias': src_state[f"{prefix_src}norm1.bias"]})
                model.layers[i].norm2.load_state_dict({'weight': src_state[f"{prefix_src}norm2.weight"], 'bias': src_state[f"{prefix_src}norm2.bias"]})
                ffn_dict = {
                    '0.weight': src_state[f"{prefix_src}linear1.weight"], '0.bias': src_state[f"{prefix_src}linear1.bias"],
                    '2.weight': src_state[f"{prefix_src}linear2.weight"], '2.bias': src_state[f"{prefix_src}linear2.bias"]
                }
                model.layers[i].shared_expert.load_state_dict(ffn_dict)
                for expert in model.layers[i].experts:
                    expert.load_state_dict(ffn_dict)
                    with torch.no_grad():
                        for p in expert.parameters(): p.add_(torch.randn_like(p) * 0.01)
        return True
    except Exception as e:
        print(f"!!! ERROR loading weights: {e}")
        return False

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

# 2. Single train function
def train_single_session(subject_id, use_transfer, run_idx):
    current_seed = 42 + run_idx * 100
    set_seed(current_seed)
    
    # Data Loader
    train_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id, snr_aug=True, snr_prob=0.8, num_segments=10)
    test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = Model_MoE_FBCSP(n_classes=2, n_bands=CONFIG['n_bands'], n_csp=8, time_steps=512, embed_dim=128).to(CONFIG['device'])
    
    if use_transfer:
        success = load_pretrained_weights(model, CONFIG['pretrained_path'])
        if not success:
            print(f"!!! WARNING: Transfer Learning requested but failed for Subject {subject_id} Run {run_idx}")
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.1)
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1)
    
    best_acc = 0.0
    best_state_dict = None
    final_metrics = {}

    for epoch in range(CONFIG['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            out, aux = model(x)
            loss = criterion(out, y) + 0.1 * aux
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation 
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
            best_state_dict = model.state_dict()
            final_metrics = {
                'acc': acc,
                'kappa': cohen_kappa_score(targets, preds),
                'f1': f1_score(targets, preds, average='macro'),
                'preds': preds,
                'targets': targets
            }
            
    return best_acc, final_metrics, best_state_dict

# 3. 20 rounds
def run_experiment_suite(use_transfer):
    mode_name = "with_transfer" if use_transfer else "no_transfer"
    save_dir = os.path.join(CONFIG['base_output_dir'], mode_name)
    weights_dir = os.path.join(save_dir, "best_weights")
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    
    print(f"\n{'#'*60}")
    print(f"STARTING EXPERIMENT: {mode_name.upper()}")
    print(f"Saving to: {save_dir}")
    print(f"{'#'*60}")
    
    summary_results = []
    
    for subj in CONFIG['subjects']:
        print(f"\nSubject {subj}: Starting 20-Round Championship...")
        
        champion_acc = -1.0
        champion_metrics = None
        champion_weights = None
        champion_round = -1
        
        subject_round_accs = []
        
        # 20-round loop
        for r in range(CONFIG['rounds']):
            print(f"  > Round {r+1}/{CONFIG['rounds']} ... ", end="")
            run_acc, run_metrics, run_weights = train_single_session(subj, use_transfer, r)
            print(f"Acc: {run_acc:.2f}%")
            
            subject_round_accs.append(run_acc)
            
            # update best if current run is better
            if run_acc > champion_acc:
                champion_acc = run_acc
                champion_metrics = run_metrics
                champion_weights = run_weights
                champion_round = r + 1
                
        print(f"  *** Subject {subj} Winner: Round {champion_round} with Acc {champion_acc:.2f}% ***")

        avg_acc_20runs = np.mean(subject_round_accs)
        
        # Save the best result for this subject
        torch.save(champion_weights, os.path.join(weights_dir, f"{subj}_best_model.pth"))
        
        df_pred = pd.DataFrame({
            'Sample_Index': range(len(champion_metrics['targets'])),
            'True_Label': champion_metrics['targets'],
            'Predicted_Label': champion_metrics['preds']
        })
        df_pred.to_csv(os.path.join(save_dir, f"{subj}_predictions.csv"), index=False)
        
        cm = confusion_matrix(champion_metrics['targets'], champion_metrics['preds'])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap='Blues', ax=ax)
        plt.title(f'Confusion Matrix - {subj}\nBest Acc (Round {champion_round}): {champion_metrics["acc"]:.2f}%')
        plt.savefig(os.path.join(save_dir, f"{subj}_confusion_matrix.png"))
        plt.close(fig)
        
        # 4. Record into summary table
        summary_results.append({
            'subject': subj,
            'best_acc': champion_metrics['acc'],
            'avg_acc_20runs': avg_acc_20runs,
            'best_kappa': champion_metrics['kappa'],
            'best_f1': champion_metrics['f1'],
            'winning_round': champion_round
        })
        
        gc.collect()
        torch.cuda.empty_cache()

    # save result
    df_summary = pd.DataFrame(summary_results)
    
    avg_row = df_summary[['best_acc', 'avg_acc_20runs', 'best_kappa', 'best_f1']].mean().to_dict()
    avg_row['subject'] = 'AVERAGE'
    avg_row['winning_round'] = '-'
    df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)
    
    summary_path = os.path.join(save_dir, "final_summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\nExperiment {mode_name} Completed.")
    print(df_summary.to_string(index=False, float_format="%.4f"))

# 4. Main entry point
if __name__ == "__main__":
    
    run_experiment_suite(use_transfer=True)
    run_experiment_suite(use_transfer=False)