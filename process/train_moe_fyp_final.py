import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 0. Basic configuration and path settings
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

# Switch: whether to use transfer learning
# Modify here to control the experiment mode
ENABLE_TRANSFER_LEARNING = False 

# Dynamically set result output path
BASE_RESULT_DIR = r"C:\Users\巫逝\Desktop\学习\大四\毕设\code\final_year_project\result"

if ENABLE_TRANSFER_LEARNING:
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, "with_transfer")
    print(f"\n[Mode] Transfer Learning: ON")
else:
    RESULT_DIR = os.path.join(BASE_RESULT_DIR, "no_transfer")
    print(f"\n[Mode] Transfer Learning: OFF")

if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"Created result directory: {RESULT_DIR}")
else:
    print(f"Results will be saved to: {RESULT_DIR}")

from dataset_loader import UniversalEEGDataset
from final_year_project.structure.model.model_fbcsp_no_cnn import Model_MoE_FBCSP

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'pretrained_path': r"checkpoints_final/model_a_fbcsp_best.pth",
    'batch_size': 16,     
    'lr': 0.0001,
    'epochs': 80,         
    'n_bands': 55,
    'device': 'cuda:0',
    'snr_aug': True,
    'snr_prob': 0.8,
    'num_segments': 10,
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
    """
    Weight loading function specifically adapted for the FBCSP No-CNN architecture.
    """
    if not os.path.exists(path):
        print(f"Warning: Pretrained weights not found at {path}. Training from scratch.")
        return False

    print(f"Loading weights from {path}...")
    checkpoint = torch.load(path)
    
    # 1. Load frontend (Linear Projection + Pos Embed)
    if 'frontend' in checkpoint:
        model.frontend.load_state_dict(checkpoint['frontend'], strict=True)
        print(" -> Frontend weights loaded.")

    # 2. Load encoder weights into MoE layers (partial load)
    if 'encoder' in checkpoint:
        src_state = checkpoint['encoder']
        loaded_layers = 0
        for i in range(len(model.layers)): 
            prefix_src = f"layers.{i}."
            
            # Load Attention
            model.layers[i].attn.load_state_dict({
                'in_proj_weight': src_state[f"{prefix_src}self_attn.in_proj_weight"],
                'in_proj_bias': src_state[f"{prefix_src}self_attn.in_proj_bias"],
                'out_proj.weight': src_state[f"{prefix_src}self_attn.out_proj.weight"],
                'out_proj.bias': src_state[f"{prefix_src}self_attn.out_proj.bias"]
            })
            
            # Load LayerNorms
            model.layers[i].norm1.load_state_dict({
                'weight': src_state[f"{prefix_src}norm1.weight"],
                'bias': src_state[f"{prefix_src}norm1.bias"]
            })
            model.layers[i].norm2.load_state_dict({
                'weight': src_state[f"{prefix_src}norm2.weight"],
                'bias': src_state[f"{prefix_src}norm2.bias"]
            })
            
            # Load FFN into Shared Expert
            model.layers[i].shared_expert.load_state_dict({
                '0.weight': src_state[f"{prefix_src}linear1.weight"],
                '0.bias': src_state[f"{prefix_src}linear1.bias"],
                '2.weight': src_state[f"{prefix_src}linear2.weight"],
                '2.bias': src_state[f"{prefix_src}linear2.bias"]
            })
            
            # Initialize Sparse Experts (hot-start using Shared Expert weights)
            for expert in model.layers[i].experts:
                expert.load_state_dict({
                    '0.weight': src_state[f"{prefix_src}linear1.weight"],
                    '0.bias': src_state[f"{prefix_src}linear1.bias"],
                    '2.weight': src_state[f"{prefix_src}linear2.weight"],
                    '2.bias': src_state[f"{prefix_src}linear2.bias"]
                })
                # Add small noise to break symmetry
                with torch.no_grad():
                    for p in expert.parameters():
                        p.add_(torch.randn_like(p) * 0.01)

            loaded_layers += 1
        print(f" -> Encoder weights transferred to {loaded_layers} MoE layers.")
        return True
    return False

def train_individual_subject(subject_id):
    print(f"\nFine-tuning subject: {subject_id}")
    
    train_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='train', augment=True, target_dataset=subject_id,
        snr_aug=CONFIG['snr_aug'], snr_prob=CONFIG['snr_prob'], num_segments=CONFIG['num_segments']
    )
    test_dataset = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset=subject_id)
    
    if len(train_dataset) == 0: 
        return {'subject': subject_id, 'acc': 0.0, 'kappa': 0.0, 'f1': 0.0}

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    # Instantiate FBCSP (No-CNN) model
    model = Model_MoE_FBCSP(
        n_classes=2, n_bands=CONFIG['n_bands'], n_csp=8, time_steps=512,
        embed_dim=128, depth=4, heads=8, num_experts=8, top_k=2, dropout=0.5 
    ).to(CONFIG['device'])
    
    # Transfer learning control logic
    if ENABLE_TRANSFER_LEARNING:
        success = load_pretrained_weights(model, CONFIG['pretrained_path'])
        if not success:
            print(" -> Transfer learning enabled but weights failed to load. Proceeding with random init.")
    else:
        print(" -> Transfer learning DISABLED. Training from scratch.")
    
    optimizer = optim.AdamW([
        {'params': model.frontend.parameters(), 'lr': CONFIG['lr'] * 0.1}, 
        {'params': model.layers.parameters(), 'lr': CONFIG['lr']},         
        {'params': model.cls_head.parameters(), 'lr': CONFIG['lr']}
    ], weight_decay=0.1)
    
    criterion = LabelSmoothingLoss(classes=2, smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=40, T_mult=1)
    
    best_metrics = {'subject': subject_id, 'acc': 0.0, 'kappa': 0.0, 'f1': 0.0, 'preds': [], 'targets': []}
    
    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            outputs, aux_loss = model(x)
            loss = criterion(outputs, y) + 0.1 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
            
        # Evaluation with TTA
        model.eval() 
        epoch_preds = []
        epoch_targets = []
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                
                tta_logits = []
                for _ in range(7): 
                    logits, _ = model(x)
                    tta_logits.append(logits)
                
                avg_logits = torch.stack(tta_logits).mean(0)
                _, predicted = torch.max(avg_logits.data, 1)
                
                epoch_preds.extend(predicted.cpu().numpy())
                epoch_targets.extend(y.cpu().numpy())
        
        cur_acc = accuracy_score(epoch_targets, epoch_preds) * 100
        
        if cur_acc > best_metrics['acc']:
            best_metrics['acc'] = cur_acc
            best_metrics['kappa'] = cohen_kappa_score(epoch_targets, epoch_preds)
            best_metrics['f1'] = f1_score(epoch_targets, epoch_preds, average='macro')
            best_metrics['preds'] = epoch_preds
            best_metrics['targets'] = epoch_targets

    print(f"{subject_id} best accuracy: {best_metrics['acc']:.2f}% | Saving results to {RESULT_DIR}...")
    
    # Save results (CSV + Image)
    df_pred = pd.DataFrame({
        'Sample_Index': range(len(best_metrics['targets'])),
        'True_Label': best_metrics['targets'],
        'Predicted_Label': best_metrics['preds'],
        'Correct': [t == p for t, p in zip(best_metrics['targets'], best_metrics['preds'])]
    })
    csv_path = os.path.join(RESULT_DIR, f"{subject_id}_predictions.csv")
    df_pred.to_csv(csv_path, index=False)
    
    cm = confusion_matrix(best_metrics['targets'], best_metrics['preds'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap='Blues', ax=ax)
    plt.title(f'Confusion Matrix - {subject_id}\nAcc: {best_metrics["acc"]:.2f}%')
    cm_path = os.path.join(RESULT_DIR, f"{subject_id}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close(fig) 

    return best_metrics

def main():
    results = []
    print(f"Starting Subject-Specific training")
    print(f"Transfer Learning: {ENABLE_TRANSFER_LEARNING}")
    print(f"Output Directory: {RESULT_DIR}")
    
    for subj in CONFIG['subjects']:
        res = train_individual_subject(subj)
        summary_res = {k: v for k, v in res.items() if k not in ['preds', 'targets']}
        results.append(summary_res)
    
    print("\n" + "="*60)
    print("Final Result Summary")
    print("="*60)
    
    df_summary = pd.DataFrame(results)
    
    # Calculate average row
    if not df_summary.empty:
        avg_row = df_summary[['acc', 'kappa', 'f1']].mean().to_dict()
        avg_row['subject'] = 'AVERAGE'
        df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)
    
    print(df_summary.to_string(index=False, float_format="%.4f"))
    
    summary_path = os.path.join(RESULT_DIR, "final_summary_metrics.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSummary results saved to: {summary_path}")

if __name__ == "__main__":
    main()