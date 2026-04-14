import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')
sys.path.append(structure_path)
sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from baseline_dl_models import EEGNet, DeepConvNet

# ======= Global absolute path configuration =======
BASE_RESULT_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result"
BASELINE_OUT_DIR = os.path.join(BASE_RESULT_DIR, "baselines")

# Create the directory if it does not exist
if not os.path.exists(BASELINE_OUT_DIR):
    os.makedirs(BASELINE_OUT_DIR)
    print(f"Created baseline result directory: {BASELINE_OUT_DIR}")

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_no_csp",  # Use No-CSP (original 22-channel) dataset
    'batch_size': 16,
    'lr': 0.001,           
    'epochs': 80,         
    'rounds': 20,          # 20 independent runs
    'device': 'cuda:0',
    'subjects': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'],
    'base_output_dir': BASELINE_OUT_DIR # Use the adjusted absolute path
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def train_single_session(model_name, subject_id, run_idx):
    set_seed(2024 + run_idx * 10)
    
    # [Fairness guarantee]: Enable the S&R augmentation used in your paper for the baseline model
    train_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id, 
        snr_aug=True, snr_prob=0.8, num_segments=10
    )
    test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # Instantiate the corresponding classic model
    if model_name == "EEGNet":
        model = EEGNet().to(CONFIG['device'])
    elif model_name == "DeepConvNet":
        model = DeepConvNet().to(CONFIG['device'])
        
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    criterion = nn.CrossEntropyLoss() 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40)
    
    best_metrics = {'acc': 0.0, 'kappa': 0.0, 'f1': 0.0}

    for epoch in range(CONFIG['epochs']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            out, _ = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Run validation testing
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
        if acc > best_metrics['acc']:
            best_metrics['acc'] = acc
            best_metrics['kappa'] = cohen_kappa_score(targets, preds)
            best_metrics['f1'] = f1_score(targets, preds, average='macro')
            
    return best_metrics

def run_baseline_evaluation(model_name):
    save_dir = os.path.join(CONFIG['base_output_dir'], model_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Starting evaluation of classical deep learning baseline: {model_name}")
    print(f"Results will be saved to: {save_dir}")
    print(f"{'='*60}")
    
    final_stats = []
    raw_detailed_results = []
    
    for subj in CONFIG['subjects']:
        print(f"Subject {subj} [{model_name}] - running 20 independent trials...", end=" ")
        
        subj_accs, subj_kappas, subj_f1s = [], [], []
        for r in range(CONFIG['rounds']):
            metrics = train_single_session(model_name, subj, r)
            subj_accs.append(metrics['acc'])
            subj_kappas.append(metrics['kappa'])
            subj_f1s.append(metrics['f1'])
            
            raw_detailed_results.append({
                'Subject': subj,
                'Run': r + 1,
                'Acc': metrics['acc'],
                'Kappa': metrics['kappa'],
                'F1': metrics['f1']
            })
            
        # Extract statistical features (mean, standard deviation, maximum)
        mean_acc, std_acc, max_acc = np.mean(subj_accs), np.std(subj_accs), np.max(subj_accs)
        mean_kappa, std_kappa = np.mean(subj_kappas), np.std(subj_kappas)
        mean_f1, std_f1 = np.mean(subj_f1s), np.std(subj_f1s)
        
        print(f"Done! Highest Acc: {max_acc:.2f}%, Mean: {mean_acc:.2f}±{std_acc:.2f}%")
        
        final_stats.append({
            'Subject': subj,
            'Acc_Max(%)': max_acc,
            'Acc_Mean±Std(%)': f"{mean_acc:.2f}±{std_acc:.2f}",
            'Kappa_Mean±Std': f"{mean_kappa:.4f}±{std_kappa:.4f}",
            'F1_Mean±Std': f"{mean_f1:.4f}±{std_f1:.4f}",
            '_mean_acc_val': mean_acc,
            '_mean_kappa_val': mean_kappa,
            '_mean_f1_val': mean_f1
        })

    # Compute overall averages
    overall_mean_acc = np.mean([x['_mean_acc_val'] for x in final_stats])
    overall_mean_kappa = np.mean([x['_mean_kappa_val'] for x in final_stats])
    overall_mean_f1 = np.mean([x['_mean_f1_val'] for x in final_stats])
    
    final_stats.append({
        'Subject': 'AVERAGE',
        'Acc_Max(%)': '-',
        'Acc_Mean±Std(%)': f"{overall_mean_acc:.2f}",
        'Kappa_Mean±Std': f"{overall_mean_kappa:.4f}",
        'F1_Mean±Std': f"{overall_mean_f1:.4f}"
    })
    
    # Save and print
    df = pd.DataFrame(final_stats).drop(columns=['_mean_acc_val', '_mean_kappa_val', '_mean_f1_val'])
    csv_path = os.path.join(save_dir, f"{model_name}_20runs_stats.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n>>> Final statistical results table for {model_name} <<<")
    print(f"CSV saved to: {csv_path}")
    print(df.to_markdown(index=False))

    df_raw = pd.DataFrame(raw_detailed_results)
    raw_csv_path = os.path.join(save_dir, f"{model_name}_raw_20runs.csv")
    df_raw.to_csv(raw_csv_path, index=False)
    print(f"Raw data for significance testing saved to: {raw_csv_path}")

if __name__ == "__main__":
    # Run EEGNet and DeepConvNet sequentially
    run_baseline_evaluation("EEGNet")
    run_baseline_evaluation("DeepConvNet")