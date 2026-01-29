import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import math
import matplotlib.pyplot as plt

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
# Import the new model
from model_a_fscfp_2 import ModelA_FSCFP_2

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all", 
    'batch_size': 64,
    'lr': 0.0001,         
    'epochs': 60,         
    'device': 'cuda:0',
    'n_bands': 55,        
    
    # Augmentation parameters
    'snr_aug': True,
    'snr_prob': 0.5,      # 50% probability of recombination
    'num_segments': 10,
    'mask_ratio': 0.0,    # It is recommended to set to 0 in the early stage of pre-training classification to ensure stability
    'warmup_epochs': 5,   # 5 rounds of warm-up
}

def train_supervised_pretrain():
    print("Starting Model A (FSCFP-2) supervised pre-training...")
    
    # 1. Load data
    train_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='train', augment=True, target_dataset='All',
        snr_aug=CONFIG['snr_aug'], snr_prob=CONFIG['snr_prob'], num_segments=CONFIG['num_segments']
    )
    val_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='test', augment=False, target_dataset='All'
    )
    
    # Windows optimized loading
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    print(f"Data loading complete: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 2. Initialize model
    model = ModelA_FSCFP_2(
        n_bands=CONFIG['n_bands'], 
        n_csp=8, 
        time_steps=512, 
        embed_dim=128, 
        depth=4, 
        heads=8, 
        dropout=0.5
    ).to(CONFIG['device'])
    
    # 3. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    # Custom Warmup + Cosine Scheduler
    def get_lr_factor(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return float(epoch + 1) / CONFIG['warmup_epochs']
        else:
            progress = (epoch - CONFIG['warmup_epochs']) / (CONFIG['epochs'] - CONFIG['warmup_epochs'])
            return 0.5 * (1 + math.cos(math.pi * progress))
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)
    
    best_acc = 0.0
    # Change save directory to checkpoints_final
    os.makedirs("checkpoints_final", exist_ok=True)
    
    train_acc_hist, val_acc_hist = [], []

    # 4. Training loop
    for epoch in range(CONFIG['epochs']):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            # Label correction
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            
            # Forward propagation (classification)
            logits = model(x, mask_ratio=CONFIG['mask_ratio']) 
            loss = criterion(logits, y)
            
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_sum += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        scheduler.step()
        train_acc = 100 * correct / total
        avg_loss = loss_sum / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
                if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
                
                logits = model(x, mask_ratio=0.0)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        val_acc = 100 * correct / total
        
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        
        print(f"Epoch {epoch+1} [LR={current_lr:.6f}]: Loss={avg_loss:.4f} | Train Acc={train_acc:.2f}% | Val Acc={val_acc:.2f}%")
        
        # Save best weights
        if val_acc > best_acc:
            best_acc = val_acc
            # We only need to save the frontend and encoder for MoE
            torch.save({
                'frontend': model.frontend.state_dict(),
                'encoder': model.encoder.state_dict()
            }, "checkpoints_final/model_a_fscfp_2_best.pth")

    print(f"Supervised pre-training complete! Best Val Acc: {best_acc:.2f}%")
    
    plt.figure()
    plt.plot(train_acc_hist, label='Train Acc')
    plt.plot(val_acc_hist, label='Val Acc')
    plt.title('Model A Supervised Pretraining')
    plt.legend()
    plt.savefig("checkpoints_final/fscfp_2_acc.png")

if __name__ == "__main__":
    train_supervised_pretrain()