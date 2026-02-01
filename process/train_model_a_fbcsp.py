import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import math

# Path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')
sys.path.append(structure_path)
sys.path.append(model_path)

from dataset_loader import UniversalEEGDataset
from model_fbcsp_no_cnn import ModelA_FBCSP  

CONFIG = {
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all",
    'batch_size': 64,
    'lr': 0.0001,
    'epochs': 20,
    'device': 'cuda:0',
    'save_path': os.path.join(r"D:\fyp\dataset_processed_fbcsp_all", "checkpoints_final", "model_a_fbcsp_best.pth")
}

def train_pretrain():
    print("Starting Model A (FBCSP No-CNN) Pre-training...")
    os.makedirs(os.path.dirname(CONFIG['save_path']), exist_ok=True)

    # Load all data
    train_ds = UniversalEEGDataset(CONFIG['data_root'], mode='train', augment=True, target_dataset='All', snr_aug=True)
    val_ds = UniversalEEGDataset(CONFIG['data_root'], mode='test', augment=False, target_dataset='All')
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    model = ModelA_FBCSP().to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        loss_sum = 0
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
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
        print(f"Epoch {epoch+1} | Loss: {loss_sum/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            # Save weights; these will be read by the final script
            torch.save({
                'frontend': model.frontend.state_dict(),
                'encoder': model.encoder.state_dict()
            }, CONFIG['save_path'])
            print(f" -> Saved Best Model to {CONFIG['save_path']}")

if __name__ == "__main__":
    train_pretrain()