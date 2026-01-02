import torch
import torch.nn as nn
import numpy as np

class TemporalCNNFrontend(nn.Module):
    """
    [修复版] Temporal CNN
    修复点：恢复深度可分离卷积 (Depthwise Separable Conv)，大幅减少参数量，防止梯度爆炸。
    """
    def __init__(self, 
                 n_bands: int = 55,  
                 n_csp_channels: int = 8, 
                 input_time_len: int = 512, 
                 embed_dim: int = 128, 
                 temporal_stride: int = 8):
        super().__init__()
        
        self.in_channels = n_bands * n_csp_channels
        
        # 1. 深度卷积 (Depthwise Conv)
        # 关键修改：groups=self.in_channels
        # 意义：每个通道独立进行时间卷积，不进行通道融合
        self.temporal_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels, 
            kernel_size=31,      
            stride=temporal_stride, 
            padding=15,          
            groups=self.in_channels, # <--- 必须改回这个！
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.act1 = nn.GELU() 
        
        # 2. 逐点卷积 (Pointwise Conv)
        # 意义：在这里进行通道融合和降维
        self.projection_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=1, 
            bias=True
        )
        
        # 位置编码
        self.seq_len = input_time_len // temporal_stride
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.seq_len))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.ln_pre = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Bands, Time, CSP)
        b, n_bands, t, n_csp = x.shape
        
        # 维度变换 -> (Batch, Channels, Time)
        x = x.permute(0, 1, 3, 2).contiguous().reshape(b, n_bands * n_csp, t)
        
        # CNN 前向
        x = self.temporal_conv(x) 
        x = self.bn1(x)
        x = self.act1(x)
        x = self.projection_conv(x) 
        
        # 加位置编码
        x = x + self.pos_embed
        
        # 转 Transformer 格式 -> (B, Seq_Len, Embed)
        x = x.permute(0, 2, 1) 
        x = self.ln_pre(x)
        
        return x