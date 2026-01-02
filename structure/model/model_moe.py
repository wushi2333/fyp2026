import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from cnn import TemporalCNNFrontend 

# ==========================================
# 1. 带噪门控网络 (Switch Loss 版本)
# ==========================================
class NoisyTopKRouter(nn.Module):
    def __init__(self, embed_dim, num_experts, top_k=2, noise_epsilon=0.2):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.noise_epsilon = noise_epsilon
        
        # 门控权重
        self.w_gate = nn.Parameter(torch.zeros(embed_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(embed_dim, num_experts))
        
        # 初始化: 使用较大的 std 确保初始不平衡
        nn.init.normal_(self.w_gate, mean=0, std=1.0) 
        nn.init.normal_(self.w_noise, mean=0, std=1.0)
        
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=-1)
        
        # 调试标记
        self.debug_printed = False

    def forward(self, x):
        # x: (Batch, S, D)
        B, S, D = x.shape
        
        # 1. Logits
        clean_logits = x @ self.w_gate 
        
        # 2. Noise
        if self.training and self.noise_epsilon > 0:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + self.noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + (noise_stddev * eps)
        else:
            noisy_logits = clean_logits
            
        # 3. Top-K
        # top_k_logits: (B, S, K)
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        top_k_gates = self.softmax(top_k_logits)
        
        # 4. 构建稀疏门控 (用于后续计算)
        zeros = torch.zeros_like(noisy_logits, requires_grad=True)
        gates = zeros.scatter(-1, indices, top_k_gates)
        
        # 5. === Switch Transformer Load Balancing Loss ===
        # loss = N * sum(P_i * f_i)
        # P_i: expert_prob (softmax of clean_logits, summed over batch)
        # f_i: expert_freq (fraction of tokens dispatched to expert i)
        
        # (B*S, E)
        probs = self.softmax(clean_logits).reshape(-1, self.num_experts)
        
        # 哪些专家被选中了? (B*S, E) - One-hot like
        # 简化计算 f_i: 直接用 gates > 0 来近似 f_i
        
        # P_i: 每个专家获得的概率总和 / 总Token数
        P = probs.mean(0) 
        
        # f_i: 每个专家被选中的频率 (基于 Gate 是否非零)
        # gates: (B, S, E)
        has_token = (gates > 0).float().reshape(-1, self.num_experts)
        f = has_token.mean(0)
        
        # Switch Loss
        aux_loss = self.num_experts * torch.sum(P * f)
        
        # === 调试打印 (只打印一次) ===
        if self.training and not self.debug_printed:
            print(f"\n[DEBUG Router] Logits Mean: {clean_logits.mean().item():.4f}, Std: {clean_logits.std().item():.4f}")
            print(f"[DEBUG Router] Prob dist (P): {P.detach().cpu().numpy()}")
            print(f"[DEBUG Router] Freq dist (f): {f.detach().cpu().numpy()}")
            print(f"[DEBUG Router] Aux Loss: {aux_loss.item():.4f}")
            self.debug_printed = True

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

class Model_MoE_Final(nn.Module):
    def __init__(self, n_classes=2, n_bands=5, n_csp=8, time_steps=512, embed_dim=64, 
                 depth=4, heads=4, num_experts=4, top_k=2, dropout=0.5):
        super().__init__()
        self.frontend = TemporalCNNFrontend(n_bands, n_csp, time_steps, embed_dim, 8)
        self.layers = nn.ModuleList([
            MoETransformerBlock(embed_dim, heads, num_experts, top_k, dropout)
            for _ in range(depth)
        ])
        self.cls_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, n_classes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.frontend(x)
        total_aux_loss = 0
        for layer in self.layers:
            x, aux_loss = layer(x)
            total_aux_loss += aux_loss
        x = x.mean(dim=1)
        return self.cls_head(x), total_aux_loss

    def load_from_model_a(self, checkpoint_path):
        print(f"🔄 正在从 {checkpoint_path} 初始化 MoE 模型...")
        checkpoint = torch.load(checkpoint_path)
        if 'frontend' in checkpoint: self.frontend.load_state_dict(checkpoint['frontend'])
        if 'encoder' in checkpoint:
            src = checkpoint['encoder']
            for i in range(len(self.layers)):
                pre = f"layers.{i}."
                self.layers[i].attn.load_state_dict({
                    'in_proj_weight': src[f'{pre}self_attn.in_proj_weight'],
                    'in_proj_bias': src[f'{pre}self_attn.in_proj_bias'],
                    'out_proj.weight': src[f'{pre}self_attn.out_proj.weight'],
                    'out_proj.bias': src[f'{pre}self_attn.out_proj.bias']
                })
                self.layers[i].norm1.load_state_dict({'weight': src[f'{pre}norm1.weight'], 'bias': src[f'{pre}norm1.bias']})
                self.layers[i].norm2.load_state_dict({'weight': src[f'{pre}norm2.weight'], 'bias': src[f'{pre}norm2.bias']})
                shared_dict = {
                    '0.weight': src[f'{pre}linear1.weight'], '0.bias': src[f'{pre}linear1.bias'],
                    '2.weight': src[f'{pre}linear2.weight'], '2.bias': src[f'{pre}linear2.bias']
                }
                self.layers[i].shared_expert.load_state_dict(shared_dict)
                for expert in self.layers[i].experts:
                    expert.load_state_dict(shared_dict)
                    with torch.no_grad():
                        # === 差异化初始化 ===
                        for p in expert.parameters(): p.add_(torch.randn_like(p) * 0.1)
        print(" -> ✅ MoE 初始化完成 (Debug Mode)")