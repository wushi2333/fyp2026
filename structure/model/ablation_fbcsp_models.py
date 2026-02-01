import torch
import torch.nn as nn
from model_fbcsp_no_cnn import FBCSP_LinearFrontend 


# 1. Ablation model: No Transformer (FBCSP + Linear + MLP)
# Purpose: Verify the necessity of Self-Attention mechanism
class Ablation_FBCSP_NoTransformer(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128):
        super().__init__()
        # Keep frontend consistent
        self.frontend = FBCSP_LinearFrontend(n_bands, n_csp, time_steps, embed_dim)
        
        # Simple classifier to replace Transformer
        # Frontend output: (Batch, Time, Embed)
        # Direct Global Average Pooling, followed by two-layer MLP
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 2)
        )

    def forward(self, x):
        # x: (B, 55, T, 8) -> Frontend -> (B, T, 128)
        x = self.frontend(x)
        
        # Global Average Pooling (remove time dimension)
        x = x.mean(dim=1) 
        
        return self.classifier(x), torch.tensor(0.0).to(x.device) # Return 0 aux_loss to maintain interface consistency


# 2. Ablation model: Standard Transformer (No MoE)
# Purpose: Verify the necessity of MoE (Mixture of Experts)
class Ablation_FBCSP_StandardTransformer(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128, 
                 depth=4, heads=8, dropout=0.5):
        super().__init__()
        self.frontend = FBCSP_LinearFrontend(n_bands, n_csp, time_steps, embed_dim)
        
        # Standard PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4,
            dropout=dropout, batch_first=True, norm_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))

    def forward(self, x):
        x = self.frontend(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.cls_head(x), torch.tensor(0.0).to(x.device)