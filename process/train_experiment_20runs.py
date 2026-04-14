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

# Basic configuration and paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from model_fbcsp_no_cnn import Model_MoE_FBCSP 

BASE_OUTPUT_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result\new_experiment_20runs"

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'pretrained_path': os.path.join(project_root, 'checkpoints_final', 'model_a_fbcsp_best.pth'),
    'batch_size': 16,
    'lr': 0.0001,
    'epochs': 80,         
    'rounds': 20,         # Run independently 20 times
    'n_bands': 55,
    'device': 'cuda:0',
    'subjects': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
}

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
        return False
    try:
        checkpoint = torch.load(path)
        if 'frontend' in checkpoint: model.frontend.load_state_dict(checkpoint['frontend'], strict=True)
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
        return False

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_single_session(subject_id, use_transfer, run_idx):
    current_seed = 42 + run_idx * 100
    set_seed(current_seed)
    
    train_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id, snr_aug=True, snr_prob=0.8, num_segments=10)
    test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = Model_MoE_FBCSP(n_classes=2, n_bands=CONFIG['n_bands'], n_csp=8, time_steps=512, embed_dim=128).to(CONFIG['device'])
    
    if use_transfer:
        load_pretrained_weights(model, CONFIG['pretrained_path'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.1)
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1)
    
    best_acc, best_metrics, best_state_dict = 0.0, {}, None

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
            best_metrics = {
                'acc': acc,
                'kappa': cohen_kappa_score(targets, preds),
                'f1': f1_score(targets, preds, average='macro'),
                'preds': preds, 'targets': targets
            }
            
    return best_acc, best_metrics, best_state_dict

def run_experiment_suite(use_transfer):
    mode_name = "with_transfer" if use_transfer else "no_transfer"
    save_dir = os.path.join(BASE_OUTPUT_DIR, mode_name)
    os.makedirs(save_dir, exist_ok=True)
    
    # Record each run's metrics
    detailed_results = []
    
    print(f"\n{'#'*60}")
    print(f"STARTING EXPERIMENT: {mode_name.upper()}")
    print(f"Saving to: {save_dir}")
    print(f"{'#'*60}")
    
    for subj in CONFIG['subjects']:
        print(f"\nSubject {subj}: Starting 20-Round...")
        
        for r in range(CONFIG['rounds']):
            print(f"  > Round {r+1}/{CONFIG['rounds']} ... ", end="\r")
            run_acc, run_metrics, _ = train_single_session(subj, use_transfer, r)
            print(f"  > Round {r+1}/{CONFIG['rounds']} - Acc: {run_acc:.2f}%, Kappa: {run_metrics['kappa']:.4f}")
            
            # Save each run's scores
            detailed_results.append({
                'Subject': subj,
                'Run': r + 1,
                'Acc': run_acc,
                'Kappa': run_metrics['kappa'],
                'F1': run_metrics['f1']
            })
            
            gc.collect()
            torch.cuda.empty_cache()

    # Save the full 20-run detailed record as CSV
    df_detailed = pd.DataFrame(detailed_results)
    detailed_csv_path = os.path.join(save_dir, "raw_20runs_metrics.csv")
    df_detailed.to_csv(detailed_csv_path, index=False)
    
    print(f"\nExperiment {mode_name} Completed.")
    print(f"Detailed 20-run records saved to: {detailed_csv_path}")

if __name__ == "__main__":
    #run_experiment_suite(use_transfer=True)
    run_experiment_suite(use_transfer=False) 