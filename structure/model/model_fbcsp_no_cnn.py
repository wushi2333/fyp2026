import torch
import torch.nn as nn
from model_moe import MoETransformerBlock  # Reuse MoE module

# 1. Linear projection frontend (replacing the original CNN)
class FBCSP_LinearFrontend(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128):
        super().__init__()
        self.input_dim = n_bands * n_csp
        
        # Use a linear layer for direct projection, without temporal convolution
        self.projection = nn.Linear(self.input_dim, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.ln_pre = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (Batch, Bands, Time, CSP)
        b, n_bands, t, n_csp = x.shape
        
        # Rearrange dimensions -> (Batch, Time, Bands * CSP)
        x = x.permute(0, 2, 1, 3).contiguous().reshape(b, t, -1)
        
        # Linear mapping
        x = self.projection(x)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :t, :]
        
        # Normalization
        x = self.ln_pre(x)
        return x

# 2. Model A (general Transformer for pretraining)
class ModelA_FBCSP(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128, depth=4, heads=8):
        super().__init__()
        self.frontend = FBCSP_LinearFrontend(n_bands, n_csp, time_steps, embed_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, 
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))

    def forward(self, x):
        x = self.frontend(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.cls_head(x)

# 3. Model MoE (final model)
class Model_MoE_FBCSP(nn.Module):
    def __init__(self, n_classes=2, n_bands=55, n_csp=8, time_steps=512, 
                 embed_dim=128, depth=4, heads=8, num_experts=4, top_k=2, dropout=0.6):
        super().__init__()
        self.frontend = FBCSP_LinearFrontend(n_bands, n_csp, time_steps, embed_dim)
        
        self.layers = nn.ModuleList([
            MoETransformerBlock(embed_dim, heads, num_experts, top_k, dropout)
            for _ in range(depth)
        ])
        
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))

    def forward(self, x):
        x = self.frontend(x)
        total_aux_loss = 0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            
        x = x.mean(dim=1)
        return self.cls_head(x), total_aux_loss