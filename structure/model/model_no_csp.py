import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Temporal CNN Frontend (No CSP Version)
class TemporalCNNFrontend_NoCSP(nn.Module):
    """
    Input: (Batch, Bands=55, Time, Channels=22)
    Uses Depthwise Separable Conv to process all Band-Channel pairs.
    """
    def __init__(self, 
                 n_bands: int = 55,  
                 n_channels: int = 22, # Original EEG channels instead of CSP components
                 input_time_len: int = 512, 
                 embed_dim: int = 128, 
                 temporal_stride: int = 8):
        super().__init__()
        
        # Flatten input channels: 55 bands * 22 channels = 1210 input maps
        self.in_channels = n_bands * n_channels
        
        # 1. Depthwise Conv
        # Convolve each (Band, Channel) timeline independently
        self.temporal_conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.in_channels, 
            kernel_size=31,      
            stride=temporal_stride, 
            padding=15,          
            groups=self.in_channels, # Depthwise
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.act1 = nn.GELU() 
        
        # 2. Pointwise Conv
        # Project 1210 features down to embed_dim (e.g., 128)
        # This layer learns the spatial and spectral mixing
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
        # x: (Batch, Bands, Time, Channels)
        b, n_bands, t, n_ch = x.shape
        
        # Reshape -> (Batch, Bands * Channels, Time)
        x = x.permute(0, 1, 3, 2).contiguous().reshape(b, n_bands * n_ch, t)
        
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

# 2. Noisy Router & MoE Blocks
class NoisyTopKRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=2, noise_epsilon=0.2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.noise_epsilon = noise_epsilon
        self.w_gate = nn.Parameter(torch.zeros(embed_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(embed_dim, num_experts))
        nn.init.normal_(self.w_gate, mean=0, std=1.0) 
        nn.init.normal_(self.w_noise, mean=0, std=1.0)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        clean_logits = x @ self.w_gate 
        if self.training and self.noise_epsilon > 0:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise_stddev * eps)
        else:
            noisy_logits = clean_logits
            
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        top_k_gates = self.softmax(top_k_logits)
        zeros = torch.zeros_like(noisy_logits, requires_grad=True)
        gates = zeros.scatter(-1, indices, top_k_gates)
        
        probs = self.softmax(clean_logits).reshape(-1, self.num_experts)
        P = probs.mean(0) 
        has_token = (gates > 0).float().reshape(-1, self.num_experts)
        f = has_token.mean(0)
        aux_loss = self.num_experts * torch.sum(P * f)
        
        return gates, aux_loss

class MoETransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_experts=4, top_k=2, dropout=0.3):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.shared_expert = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
        )
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim), nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])
        self.router = NoisyTopKRouter(embed_dim, num_experts, top_k)
        
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        residual = x
        x = self.norm2(x)
        shared_out = self.shared_expert(x)
        gates, aux_loss = self.router(x)
        
        expert_outputs = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            gate_i = gates[:, :, i].unsqueeze(-1) 
            if gate_i.sum() > 0:
                out_i = expert(x)
                expert_outputs += gate_i * out_i
        
        return residual + shared_out + expert_outputs, aux_loss

# 3. Final Model Assemblies
class ModelA_NoCSP_Pretrain(nn.Module):
    """
    Pretraining Model (Model A equivalent) for No-CSP data
    """
    def __init__(self, n_bands=55, n_channels=22, time_steps=512, embed_dim=128, depth=4, heads=8):
        super().__init__()
        self.frontend = TemporalCNNFrontend_NoCSP(n_bands, n_channels, time_steps, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, dim_feedforward=embed_dim*4, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 2))

    def forward(self, x):
        x = self.frontend(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.cls_head(x)

class Model_MoE_NoCSP_Final(nn.Module):
    """
    Final MoE Model for No-CSP data
    """
    def __init__(self, n_classes=2, n_bands=55, n_channels=22, time_steps=512, embed_dim=128, 
                 depth=4, heads=8, num_experts=8, top_k=2, dropout=0.5):
        super().__init__()
        self.frontend = TemporalCNNFrontend_NoCSP(n_bands, n_channels, time_steps, embed_dim)
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
    
    def load_weights_from_model_a(self, state_dict):
        # Load Frontend
        self.frontend.load_state_dict(state_dict['frontend'], strict=True)
        # Load Encoder weights into MoE Layers (Partial load)
        enc_dict = state_dict['encoder']
        for i in range(len(self.layers)):
            prefix = f"layers.{i}."
            # Load Attention
            self.layers[i].attn.load_state_dict({
                'in_proj_weight': enc_dict[f'{prefix}self_attn.in_proj_weight'],
                'in_proj_bias': enc_dict[f'{prefix}self_attn.in_proj_bias'],
                'out_proj.weight': enc_dict[f'{prefix}self_attn.out_proj.weight'],
                'out_proj.bias': enc_dict[f'{prefix}self_attn.out_proj.bias']
            })
            # Load Norms
            self.layers[i].norm1.load_state_dict({'weight': enc_dict[f'{prefix}norm1.weight'], 'bias': enc_dict[f'{prefix}norm1.bias']})
            self.layers[i].norm2.load_state_dict({'weight': enc_dict[f'{prefix}norm2.weight'], 'bias': enc_dict[f'{prefix}norm2.bias']})
            # Init Experts with FFN weights
            ffn_dict = {
                '0.weight': enc_dict[f'{prefix}linear1.weight'], '0.bias': enc_dict[f'{prefix}linear1.bias'],
                '2.weight': enc_dict[f'{prefix}linear2.weight'], '2.bias': enc_dict[f'{prefix}linear2.bias']
            }
            self.layers[i].shared_expert.load_state_dict(ffn_dict)
            for expert in self.layers[i].experts:
                expert.load_state_dict(ffn_dict)