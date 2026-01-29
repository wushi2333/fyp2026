import torch
import torch.nn as nn
from cnn import TemporalCNNFrontend

class ModelA_FSCFP_2(nn.Module):
    """
    Force CNN and Transformer to learn robust features through the "supervised classification" task of full data.
    Input: (Batch, 55, 512, 8) -> FBCSP features
    Output: (Batch, 2) -> Left vs Right classification result
    """
    def __init__(self, 
                 n_bands=55, 
                 n_csp=8, 
                 time_steps=512, 
                 embed_dim=128, 
                 depth=4, 
                 heads=8, 
                 dropout=0.5): # Set Dropout higher for classification tasks to prevent overfitting
        super().__init__()
        
        # 1. Front-end feature extraction (CNN)
        self.frontend = TemporalCNNFrontend(
            n_bands=n_bands, 
            n_csp_channels=n_csp, 
            input_time_len=time_steps,
            embed_dim=embed_dim, 
            temporal_stride=8
        )
        
        # 2. Transformer Encoder (learn temporal dependencies)
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
        
        # 3. Classification Head
        # Structure: Normalization -> Linear Layer -> 2-class classification
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 2)
        )
        
        # Initialize weights
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
        
        # 1. CNN feature extraction -> (Batch, Seq_Len, Dim)
        x = self.frontend(x)
        
        # 2. Random Mask enhancement during training to increase difficulty
        if self.training and mask_ratio > 0:
            B, L, D = x.shape
            # Generate random mask (Batch, Seq_Len)
            mask = torch.rand(B, L, device=x.device) < mask_ratio
            # Set masked tokens to 0
            x[mask] = 0 
            
        # 3. Transformer encoding
        x = self.encoder(x)
        
        # 4. Global Average Pooling (GAP)
        # Compress the time series dimension: (Batch, Seq_Len, Dim) -> (Batch, Dim)
        x_cls = x.mean(dim=1)
        
        # 5. Output classification Logits
        logits = self.cls_head(x_cls)
        
        return logits