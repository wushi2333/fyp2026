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

# 1. Path and environment configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')
sys.path.append(structure_path)
sys.path.append(model_path)

# Import data loader
from dataset_loader import UniversalEEGDataset

# Import all model variants
from model_moe import Model_MoE_Final
from ablation_models import (
    Ablation_NoCNN_MoE, 
    Ablation_StandardCNN_MoE, 
    Ablation_CNNOnly, 
    Ablation_StandardTransformer
)

# Basic configuration
CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all", 
    'batch_size': 16,
    'lr': 0.0001,
    'epochs': 80,  
    'device': 'cuda:0',
    'n_bands': 55,
    'subjects': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09']
}

# Results root directory
ABLATION_ROOT = os.path.join(project_root, 'result', 'ablation')

# 2. Define experiment configuration dictionary
# Define the model class, save folder name, and whether to load pre-trained weights for each experiment
EXPERIMENTS = {
    # Experiment 1: No CNN frontend
    'no_cnn': {
        'model_class': Ablation_NoCNN_MoE,
        'load_weights': False, 
        'folder': 'no_cnn',
        'desc': 'MoE Transformer without CNN Frontend'
    },
    # Experiment 2: Standard CNN frontend (not depthwise separable)
    'standard_cnn': {
        'model_class': Ablation_StandardCNN_MoE,
        'load_weights': False,
        'folder': 'standard_cnn',
        'desc': 'MoE Transformer with Standard Conv1D'
    },
    # Experiment 3: No Transformer (CNN + classification head only)
    'no_transformer': {
        'model_class': Ablation_CNNOnly,
        'load_weights': False,
        'folder': 'no_transformer',
        'desc': 'CNN Frontend + Classifier (No Transformer)'
    },
    # Experiment 4: Standard Transformer (No MoE)
    'standard_transformer': {
        'model_class': Ablation_StandardTransformer,
        'load_weights': False,
        'folder': 'standard_transformer',
        'desc': 'Standard Transformer (No MoE)'
    },
    # Experiment 5: Full model but no transfer learning (No Transfer)
    'no_transfer': {
        'model_class': Model_MoE_Final, # Use the final model architecture
        'load_weights': False,          # Do not load pre-trained weights
        'folder': 'no_transfer',
        'desc': 'Final MoE Model trained from scratch (No Transfer Learning)'
    }
}

# 3. Helper functions, consistent with the main code
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

def save_results(subject_id, metrics, output_dir):
    """Save prediction CSV and confusion matrix"""
    # 1. Save CSV
    df_pred = pd.DataFrame({
        'Sample_Index': range(len(metrics['targets'])),
        'True_Label': metrics['targets'],
        'Predicted_Label': metrics['preds'],
        'Correct': [t == p for t, p in zip(metrics['targets'], metrics['preds'])]
    })
    df_pred.to_csv(os.path.join(output_dir, f"{subject_id}_predictions.csv"), index=False)
    
    # 2. Save confusion matrix image
    cm = confusion_matrix(metrics['targets'], metrics['preds'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix - {subject_id}\nAcc: {metrics["acc"]:.2f}%')
    plt.savefig(os.path.join(output_dir, f"{subject_id}_confusion_matrix.png"))
    plt.close(fig)

# 4. Core training logic
def run_single_experiment(exp_key):
    cfg = EXPERIMENTS[exp_key]
    save_dir = os.path.join(ABLATION_ROOT, cfg['folder'])
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Running Experiment: {cfg['desc']}")
    print(f"Output Directory: {save_dir}")
    print(f"{'='*60}")
    
    results_summary = []
    
    for subject_id in CONFIG['subjects']:
        print(f"\nTraining Subject: {subject_id} ...")
        
        # 1. Data loading
        train_dataset = UniversalEEGDataset(
            CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id,
            snr_aug=True, snr_prob=0.8, num_segments=10
        )
        test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)
        
        # 2. Initialize model
        # Unified parameter instantiation
        model = cfg['model_class'](
            n_bands=55, n_csp=8, time_steps=512, embed_dim=128, 
            depth=4, heads=8, dropout=0.5
        ).to(CONFIG['device'])
        
        
        # 3. Optimizer and loss
        optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
        criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1)
        
        best_metrics = {'acc': 0.0, 'kappa': 0.0, 'f1': 0.0, 'preds': [], 'targets': []}
        
        # 4. Training loop
        for epoch in range(CONFIG['epochs']):
            model.train()
            for x, y in train_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                
                optimizer.zero_grad()
                outputs, aux_loss = model(x) 
                
                # Calculate loss (main loss + 0.1 * auxiliary loss)
                loss = criterion(outputs, y) + 0.1 * aux_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # 5. Validation
            model.eval() 
            epoch_preds, epoch_targets = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                    if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                    
                    # --- TTA (Test Time Augmentation) Start ---
                    tta_logits = []
                    # Perform 7 forward passes (consistent with the main experiment)
                    for _ in range(7):
                        # Get logits (ignore aux_loss)
                        logits_i, _ = model(x)
                        tta_logits.append(logits_i)
                    
                    # Stack the 7 results and take the average -> (Batch, 2)
                    avg_logits = torch.stack(tta_logits).mean(0)
                    # --- TTA End ---

                    _, predicted = torch.max(avg_logits, 1)
                    epoch_preds.extend(predicted.cpu().numpy())
                    epoch_targets.extend(y.cpu().numpy())
            
            cur_acc = accuracy_score(epoch_targets, epoch_preds) * 100
            
            # Save best results
            if cur_acc > best_metrics['acc']:
                best_metrics['acc'] = cur_acc
                best_metrics['kappa'] = cohen_kappa_score(epoch_targets, epoch_preds)
                best_metrics['f1'] = f1_score(epoch_targets, epoch_preds, average='macro')
                best_metrics['preds'] = epoch_preds
                best_metrics['targets'] = epoch_targets
        
        print(f" -> Best Acc: {best_metrics['acc']:.2f}%")
        
        # 6. Save detailed results for this subject
        save_results(subject_id, best_metrics, save_dir)
        
        # Record summary information
        results_summary.append({
            'subject': subject_id,
            'acc': best_metrics['acc'],
            'kappa': best_metrics['kappa'],
            'f1': best_metrics['f1']
        })

    # 5. Experiment finished, save summary CSV
    df_summary = pd.DataFrame(results_summary)
    # Calculate average row
    avg_row = df_summary[['acc', 'kappa', 'f1']].mean().to_dict()
    avg_row['subject'] = 'AVERAGE'
    df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)
    
    summary_path = os.path.join(save_dir, "final_summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)
    
    print(f"\nExperiment '{exp_key}' Completed.")
    print(f"Summary saved to: {summary_path}")
    print(df_summary.to_string(index=False, float_format="%.4f"))

if __name__ == "__main__":
    # Options: 'no_cnn', 'standard_cnn', 'no_transformer', 'standard_transformer', 'no_transfer'
    # Run No Transfer experiment
    #run_single_experiment('no_transfer')
    
    # Run No CNN experiment
    #run_single_experiment('no_cnn')
    
    # Run No Transformer experiment
    run_single_experiment('no_transformer')

    # Run Standard Transformer experiment
    #run_single_experiment('standard_transformer')

    # Run Standard CNN experiment
    #run_single_experiment('standard_cnn')
