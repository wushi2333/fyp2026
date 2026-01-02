import torch
import torch.nn as nn
# 假设你的 cnn.py 已经在 structure/model/ 目录下
from cnn import TemporalCNNFrontend

class ModelA_FSCFP_2(nn.Module):
    """
    [Model A - Supervised Version 2]
    用途：通过全量数据的“监督分类”任务，强制 CNN 和 Transformer 学习鲁棒特征。
    输入：(Batch, 55, 512, 8) -> FBCSP 特征
    输出：(Batch, 2) -> Left vs Right 分类结果
    """
    def __init__(self, 
                 n_bands=55, 
                 n_csp=8, 
                 time_steps=512, 
                 embed_dim=128, 
                 depth=4, 
                 heads=8, 
                 dropout=0.5): # 分类任务 Dropout 设高一点防止过拟合
        super().__init__()
        
        # 1. 前端特征提取 (CNN)
        self.frontend = TemporalCNNFrontend(
            n_bands=n_bands, 
            n_csp_channels=n_csp, 
            input_time_len=time_steps,
            embed_dim=embed_dim, 
            temporal_stride=8
        )
        
        # 2. Transformer 编码器 (学习时序依赖)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim*4,
            dropout=dropout, 
            batch_first=True, 
            activation='gelu', 
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. 分类头 (Classification Head)
        # 结构：归一化 -> 线性层 -> 2分类
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2)
        )
        
        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mask_ratio=0.0):
        # x: (Batch, Bands, Time, CSP)
        
        # 1. CNN 提取特征 -> (Batch, Seq_Len, Dim)
        x = self.frontend(x)
        
        # 2. (可选) 训练时的随机 Mask 增强，增加难度
        if self.training and mask_ratio > 0:
            B, L, D = x.shape
            # 生成随机掩码 (Batch, Seq_Len)
            mask = torch.rand(B, L, device=x.device) < mask_ratio
            # 将被遮盖的 Token 置为 0
            x[mask] = 0 
            
        # 3. Transformer 编码
        x = self.encoder(x)
        
        # 4. 全局平均池化 (GAP)
        # 将时间序列维度压缩：(Batch, Seq_Len, Dim) -> (Batch, Dim)
        x_cls = x.mean(dim=1)
        
        # 5. 输出分类 Logits
        logits = self.cls_head(x_cls)
        
        return logits