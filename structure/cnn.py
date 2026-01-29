import torch
import torch.nn as nn
import numpy as np

class TemporalCNNFrontend(nn.Module):
    # Use Depthwise Separable Conv to significantly reduce the number of parameters and prevent gradient explosion.
    def __init__(self, 
                 n_bands: int = 55,  
                 n_csp_channels: int = 8, 
                 input_time_len: int = 512, 
                 embed_dim: int = 128, 
                 temporal_stride: int = 8):
        super().__init__()
        
        self.in_channels = n_bands * n_csp_channels
        
        # 1. Depthwise Conv
        # Each channel is convolved independently in time, without channel fusion
        self.temporal_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels, 
            kernel_size=31,      
            stride=temporal_stride, 
            padding=15,          
            groups=self.in_channels, 
            bias=False
        )
        
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.act1 = nn.GELU() 
        
        # 2. Pointwise Conv
        # Channel fusion and dimensionality reduction are performed here
        self.projection_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=embed_dim,
            kernel_size=1, 
            bias=True
        )
        
        # Positional encoding
        self.seq_len = input_time_len // temporal_stride
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.seq_len))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.ln_pre = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        # x: (Batch, Bands, Time, CSP)
        b, n_bands, t, n_csp = x.shape
        
        # Dimension transformation -> (Batch, Channels, Time)
        x = x.permute(0, 1, 3, 2).contiguous().reshape(b, n_bands * n_csp, t)
        
        # CNN forward
        x = self.temporal_conv(x) 
        x = self.bn1(x)
        x = self.act1(x)
        x = self.projection_conv(x) 
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Convert to Transformer format -> (B, Seq_Len, Embed)
        x = x.permute(0, 2, 1) 
        x = self.ln_pre(x)
        
        return x