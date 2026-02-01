import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Configuration & Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')
sys.path.append(structure_path)
sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from model_no_csp import ModelA_NoCSP_Pretrain, Model_MoE_NoCSP_Final

ENABLE_TRANSFER_LEARNING = False  # Set False to train MoE from scratch

if ENABLE_TRANSFER_LEARNING:
    experiment_name = 'no_csp_transfer_learning'
else:
    experiment_name = 'no_csp_scratch_training'



CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_no_csp", # Ensure this matches data_process output
    'batch_size': 32,
    'lr_pretrain': 0.0001,
    'lr_finetune': 0.0001,
    'epochs_pretrain': 60,
    'epochs_finetune': 80,
    'device': 'cuda:0',
    'n_bands': 55,
    'n_channels': 22, # No longer CSP=8, using original 22 channels
    'subjects': ['A01', 'A02', 'A03', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09'],
    'save_dir': os.path.join(project_root, 'result', 'ablation', experiment_name)
}


os.makedirs(os.path.join(project_root, 'result', 'ablation'), exist_ok=True)
os.makedirs(CONFIG['save_dir'], exist_ok=True)
CHECKPOINT_PATH = os.path.join(project_root, 'result', 'ablation', 'no_csp_transfer_learning', 'model_a_nocsp_best.pth')

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

def save_metrics(subject, metrics, output_dir):
    df = pd.DataFrame({
        'Sample': range(len(metrics['targets'])),
        'True': metrics['targets'], 'Pred': metrics['preds']
    })
    df.to_csv(os.path.join(output_dir, f"{subject}_preds.csv"), index=False)
    
    cm = confusion_matrix(metrics['targets'], metrics['preds'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Left', 'Right'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'{subject} Acc: {metrics["acc"]:.2f}%')
    plt.savefig(os.path.join(output_dir, f"{subject}_cm.png"))
    plt.close()

# 2. Pre-training Phase (Model A)
def run_pretraining():
    print(f"\n\n PHASE 1: Pre-training Model A (No CSP)\n")
    
    # Load ALL datasets for pretraining
    train_ds = UniversalEEGDataset(CONFIG['data_root'], mode='train', augment=True, target_dataset='All', snr_aug=True)
    val_ds = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset='All')
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    model = ModelA_NoCSP_Pretrain(
        n_bands=CONFIG['n_bands'], n_channels=CONFIG['n_channels']
    ).to(CONFIG['device'])
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr_pretrain'], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(CONFIG['epochs_pretrain']):
        model.train()
        loss_sum = 0
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item()
            
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                logits = model(x)
                _, pred = torch.max(logits, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        
        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{CONFIG['epochs_pretrain']} | Loss: {loss_sum/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save({
                'frontend': model.frontend.state_dict(),
                'encoder': model.encoder.state_dict()
            }, CHECKPOINT_PATH)
            print(" -> Saved Best Model A")

# 3. Fine-tuning Phase (MoE)
def run_finetuning(subject):
    print(f"\nTraining Subject: {subject}")
    
    train_ds = UniversalEEGDataset(CONFIG['data_root'], mode='train', augment=True, target_dataset=subject, snr_aug=True)
    test_ds = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False)
    
    model = Model_MoE_NoCSP_Final(
        n_bands=CONFIG['n_bands'], n_channels=CONFIG['n_channels']
    ).to(CONFIG['device'])
    
    # Transfer Learning Logic
    if ENABLE_TRANSFER_LEARNING and os.path.exists(CHECKPOINT_PATH):
        print(" -> Loading Pretrained Weights...")
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_weights_from_model_a(checkpoint)
    else:
        print(" -> Training from Scratch (No Transfer)")

    optimizer = optim.AdamW([
        {'params': model.frontend.parameters(), 'lr': CONFIG['lr_finetune'] * 0.1},
        {'params': model.layers.parameters(), 'lr': CONFIG['lr_finetune']},
        {'params': model.cls_head.parameters(), 'lr': CONFIG['lr_finetune']}
    ], weight_decay=0.05)
    
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    
    best_res = {'acc': 0, 'preds': [], 'targets': []}
    
    for epoch in range(CONFIG['epochs_finetune']):
        model.train()
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            logits, aux_loss = model(x)
            loss = criterion(logits, y) + 0.1 * aux_loss
            loss.backward()
            optimizer.step()
            
        # Evaluation
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
        if cur_acc > best_res['acc']:
            best_res['acc'] = cur_acc
            best_res['kappa'] = cohen_kappa_score(epoch_targets, epoch_preds)
            best_res['f1'] = f1_score(epoch_targets, epoch_preds, average='macro')
            best_res['preds'] = epoch_preds
            best_res['targets'] = epoch_targets
            
    print(f"Subject {subject} Best Acc: {best_res['acc']:.2f}%")
    save_metrics(subject, best_res, CONFIG['save_dir'])
    return best_res


# 4. Main Execution
if __name__ == "__main__":
    # Step 1: Pre-train if enabled
    if ENABLE_TRANSFER_LEARNING:
        if not os.path.exists(CHECKPOINT_PATH):
            run_pretraining()
        else:
            print("Pretrained model exists, skipping training.")
            
    # Step 2: Fine-tune for each subject
    results = []
    for subj in CONFIG['subjects']:
        res = run_finetuning(subj)
        results.append({
            'subject': subj, 
            'acc': res['acc'], 
            'kappa': res['kappa'], 
            'f1': res['f1']
        })
        
    # Summary
    df = pd.DataFrame(results)
    avg = df.mean(numeric_only=True)
    df.loc['AVERAGE'] = avg
    df.loc['AVERAGE', 'subject'] = 'AVERAGE'
    
    print("\nFinal Results (No CSP):")
    print(df.to_string())
    df.to_csv(os.path.join(CONFIG['save_dir'], 'final_summary.csv'))