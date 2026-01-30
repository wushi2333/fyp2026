import torch
import torch.nn as nn
from cnn import TemporalCNNFrontend
from model_moe import MoETransformerBlock, NoisyTopKRouter

# 1: No CNN Frontend
# Verify the necessity of the CNN frontend in extracting temporal features and reducing noise
class Ablation_NoCNN_MoE(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128, 
                 depth=4, heads=8, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        self.input_dim = n_bands * n_csp
        
        # Replace CNN: Use a simple linear layer to project the input to embed_dim
        # Removes the temporal smoothing ability of convolution
        self.projection = nn.Linear(self.input_dim, embed_dim)
        
        # Manually add positional encoding (originally included in the CNN frontend)
        self.pos_embed = nn.Parameter(torch.zeros(1, time_steps, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Backend remains unchanged: MoE Transformer
        self.layers = nn.ModuleList([
            MoETransformerBlock(embed_dim, heads, num_experts, top_k, dropout)
            for _ in range(depth)
        ])
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))

    def forward(self, x):
        # x: (Batch, Bands, Time, CSP) -> (Batch, Time, Bands*CSP)
        b, n_bands, t, n_csp = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().reshape(b, t, -1)
        
        # Linear projection + positional encoding
        x = self.projection(x)
        x = x + self.pos_embed[:, :t, :]
        
        total_aux_loss = 0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
            
        x = x.mean(dim=1)
        return self.cls_head(x), total_aux_loss


# 2: Use Standard CNN frontend
# Verify the role of Depthwise Separable convolution in reducing parameters and preventing overfitting
class StandardCNNFrontend(nn.Module):
    def __init__(self, n_bands, n_csp, input_time_len, embed_dim, stride=8):
        super().__init__()
        in_channels = n_bands * n_csp
        
        # Standard convolution: groups=1 (fully connected convolution), number of parameters increases dramatically
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim, # Directly convolve to the embedding dimension
            kernel_size=31,
            stride=stride,
            padding=15,
            bias=False
        )
        self.bn = nn.BatchNorm1d(embed_dim)
        self.act = nn.GELU()
        
        seq_len = input_time_len // stride
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, seq_len))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        b, n_bands, t, n_csp = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().reshape(b, -1, t)
        
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        
        x = x + self.pos_embed
        return x.permute(0, 2, 1)

class Ablation_StandardCNN_MoE(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128, 
                 depth=4, heads=8, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        # Replace frontend
        self.frontend = StandardCNNFrontend(n_bands, n_csp, time_steps, embed_dim)
        
        # Backend unchanged
        self.layers = nn.ModuleList([
            MoETransformerBlock(embed_dim, heads, num_experts, top_k, dropout)
            for _ in range(depth)
        ])
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))

    def forward(self, x):
        x = self.frontend(x)
        total_aux_loss = 0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
        x = x.mean(dim=1)
        return self.cls_head(x), total_aux_loss


# 3: No Transformer (CNN Only / FBCSP+CNN)
# Function: Verify the ability of Transformer to capture Long-term Dependency
class Ablation_CNNOnly(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128, **kwargs):
        super().__init__()
        # Keep your lightweight frontend unchanged
        self.frontend = TemporalCNNFrontend(n_bands, n_csp, time_steps, embed_dim, temporal_stride=8)
        
        # Remove all Transformer layers and connect directly to the classification head
        # Add a fully connected layer to compensate for the reduced depth
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(embed_dim, 2)
        )
        

    def forward(self, x):
        x = self.frontend(x) # (B, Seq, Dim)
        x = x.mean(dim=1)    # Global Average Pooling
        return self.classifier(x), torch.tensor(0.0).to(x.device) # To be compatible with the code, return 0 loss



# 4: Use Standard Transformer
# Function: Verify the advantage of the MoE (Mixture of Experts) architecture in handling individual differences
class Ablation_StandardTransformer(nn.Module):
    def __init__(self, n_bands=55, n_csp=8, time_steps=512, embed_dim=128, 
                 depth=4, heads=8, dropout=0.5):
        super().__init__()
        self.frontend = TemporalCNNFrontend(n_bands, n_csp, time_steps, embed_dim, temporal_stride=8)
        
        # Use PyTorch standard Transformer Encoder
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