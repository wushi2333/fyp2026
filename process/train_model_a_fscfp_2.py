import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import math
import matplotlib.pyplot as plt

# ================= 路径配置 =================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
structure_path = os.path.join(project_root, 'structure')
model_path = os.path.join(structure_path, 'model')

if structure_path not in sys.path: sys.path.append(structure_path)
if model_path not in sys.path: sys.path.append(model_path)
# ===========================================

from dataset_loader import UniversalEEGDataset
# 导入你刚才新建的模型
from model_a_fscfp_2 import ModelA_FSCFP_2

CONFIG = {
    # 确保路径指向你的 55 频带数据
    'data_root': r"D:\fyp\dataset_processed_fbcsp_all", 
    'batch_size': 64,
    'lr': 0.0001,         # 1e-4，分类任务的标准学习率
    'epochs': 60,         # 监督学习收敛较快，60轮通常足够
    'device': 'cuda:0',
    'n_bands': 55,        # 必须匹配数据
    
    # 增强参数
    'snr_aug': True,
    'snr_prob': 0.5,      # 50% 概率进行重组
    'num_segments': 10,
    'mask_ratio': 0.0,    # 预训练分类初期建议设为 0，先求稳
    'warmup_epochs': 5,   # 5轮热身
}

def train_supervised_pretrain():
    print("🚀 开始 Model A (FSCFP-2) 监督预训练...")
    
    # 1. 加载数据
    train_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='train', augment=True, target_dataset='All',
        snr_aug=CONFIG['snr_aug'], snr_prob=CONFIG['snr_prob'], num_segments=CONFIG['num_segments']
    )
    val_dataset = UniversalEEGDataset(
        CONFIG['data_root'], mode='test', augment=False, target_dataset='All'
    )
    
    # Windows 优化加载
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    print(f"数据加载完毕: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # 2. 初始化模型
    model = ModelA_FSCFP_2(
        n_bands=CONFIG['n_bands'], 
        n_csp=8, 
        time_steps=512, 
        embed_dim=128, 
        depth=4, 
        heads=8, 
        dropout=0.5
    ).to(CONFIG['device'])
    
    # 3. 优化器与 Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.05)
    criterion = nn.CrossEntropyLoss()
    
    # 自定义 Warmup + Cosine Scheduler
    def get_lr_factor(epoch):
        if epoch < CONFIG['warmup_epochs']:
            return float(epoch + 1) / CONFIG['warmup_epochs']
        else:
            progress = (epoch - CONFIG['warmup_epochs']) / (CONFIG['epochs'] - CONFIG['warmup_epochs'])
            return 0.5 * (1 + math.cos(math.pi * progress))
            
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_factor)
    
    best_acc = 0.0
    # 保存目录改为 checkpoints_final
    os.makedirs("checkpoints_final", exist_ok=True)
    
    train_acc_hist, val_acc_hist = [], []

    # 4. 训练循环
    for epoch in range(CONFIG['epochs']):
        model.train()
        correct = 0
        total = 0
        loss_sum = 0
        
        current_lr = optimizer.param_groups[0]['lr']
        
        for x, y in train_loader:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            
            # 标签修正
            if y.max() > 1: y = torch.where(y == y.min(), torch.tensor(0).to(y.device), torch.tensor(1).to(y.device))
            
            optimizer.zero_grad()
            
            # 前向传播 (分类)
            logits = model(x, mask_ratio=CONFIG['mask_ratio']) 
            loss = criterion(logits, y)
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_sum += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
        scheduler.step()
        train_acc = 100 * correct / total
        avg_loss = loss_sum / len(train_loader)
        
        # --- 验证 ---
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
        
        # 保存最佳权重
        if val_acc > best_acc:
            best_acc = val_acc
            # 我们只需要保存前端和编码器给 MoE 用
            torch.save({
                'frontend': model.frontend.state_dict(),
                'encoder': model.encoder.state_dict()
            }, "checkpoints_final/model_a_fscfp_2_best.pth")

    print(f"✅ 监督预训练完成！最佳 Val Acc: {best_acc:.2f}%")
    
    plt.figure()
    plt.plot(train_acc_hist, label='Train Acc')
    plt.plot(val_acc_hist, label='Val Acc')
    plt.title('Model A Supervised Pretraining')
    plt.legend()
    plt.savefig("checkpoints_final/fscfp_2_acc.png")

if __name__ == "__main__":
    train_supervised_pretrain()