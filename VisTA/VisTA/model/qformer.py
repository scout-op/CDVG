import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def pos2d_sincos(d_model: int, h: int, w: int, device=None, dtype=None):
    """2D sine-cosine positional encoding (H*W, C)."""
    if d_model % 4 != 0:
        raise ValueError(f"pos2d_sincos expects d_model % 4 == 0, got {d_model}")
    y, x = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    omega = torch.arange(d_model // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / (d_model // 4)))

    x = x.reshape(-1).unsqueeze(1).to(dtype)
    y = y.reshape(-1).unsqueeze(1).to(dtype)

    out_x = torch.cat([torch.sin(x * omega), torch.cos(x * omega)], dim=1)
    out_y = torch.cat([torch.sin(y * omega), torch.cos(y * omega)], dim=1)
    out = torch.cat([out_x, out_y], dim=1)  # (H*W, C)
    return out


class MultiheadAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, use_rope: bool = True, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.rope_base = rope_base

    def _build_rope_cache(self, seq_len: int, device, dtype):
        # Standard 1D RoPE over flattened tokens
        dim = self.head_dim
        half = dim // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, half, device=device, dtype=dtype) / half))
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum('i,j->ij', t, inv_freq)  # [seq, half]
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)  # [1,1,seq,half]
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)  # [1,1,seq,half]
        # expand pairwise to full dim
        cos = torch.repeat_interleave(cos, 2, dim=-1)  # [1,1,seq,dim]
        sin = torch.repeat_interleave(sin, 2, dim=-1)
        return cos, sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        # pairwise rotation on even/odd dims
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack([-x_odd, x_even], dim=-1).reshape_as(x)
        return x_rot

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: (B, nH, L, d), cos/sin: (1,1,L,d)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q, kv: (B, Lq, C) / (B, Lkv, C)
        B, Lq, C = q.shape
        Lkv = kv.shape[1]
        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # (B, nH, L, d)
        q = q.view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Lkv, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Lkv, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            cos_k, sin_k = self._build_rope_cache(Lkv, k.device, k.dtype)
            k = self._apply_rope(k, cos_k, sin_k)
            # Queries are learnable tokens without spatial pos; leave as-is (equiv. zero-angle RoPE)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v  # (B, nH, Lq, d)
        out = out.transpose(1, 2).contiguous().view(B, Lq, C)
        out = self.out_proj(out)
        return out


class CrossBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8, mlp_ratio: int = 4, dropout: float = 0.0, use_rope: bool = True, rope_base: float = 10000.0):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadAttentionRoPE(d_model, n_heads, dropout=dropout, use_rope=use_rope, rope_base=rope_base)

        self.self_ln = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True, dropout=dropout)

        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q, kv: (B, L, C)
        q = q + self.cross_attn(self.q_ln(q), self.kv_ln(kv))
        q = q + self.self_attn(self.self_ln(q), self.self_ln(q), self.self_ln(q), need_weights=False)[0]
        q = q + self.ffn(self.ffn_ln(q))
        return q


class QFormer(nn.Module):
    """
    Minimal Q-Former:
    - Projects multi-scale CNN features to a common dim.
    - Flattens to tokens with 2D positional encoding.
    - Applies several cross-attention blocks from learnable queries to visual tokens.
    - Initializes queries by injecting text "state" as conditioning.
    - Outputs a pooled query vector (B, d_model) as refined state.
    """

    def __init__(self,
                 d_model: int = 512,
                 n_heads: int = 8,
                 num_queries: int = 16,
                 num_layers: int = 2,
                 in_chans=(512, 1024, 512),
                 use_rope: bool = True,
                 rope_base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.use_rope = use_rope
        self.rope_base = rope_base
        # project each scale to d_model
        c1, c2, c3 = in_chans
        self.proj1 = nn.Conv2d(c1, d_model, 1)
        self.proj2 = nn.Conv2d(c2, d_model, 1)
        self.proj3 = nn.Conv2d(c3, d_model, 1)

        # learnable queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, d_model) / math.sqrt(d_model))
        self.state_proj = nn.Linear(d_model, d_model)

        self.blocks = nn.ModuleList([CrossBlock(d_model=d_model, n_heads=n_heads, use_rope=use_rope, rope_base=rope_base) for _ in range(num_layers)])
        self.out_ln = nn.LayerNorm(d_model)

    @staticmethod
    def _flatten_tokens(x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return x

    def forward(self, v1: torch.Tensor, v2: torch.Tensor, v3: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # Project to common dim
        dtype = v1.dtype
        device = v1.device
        p1 = self.proj1(v1)
        p2 = self.proj2(v2)
        p3 = self.proj3(v3)

        # Flatten + add 2D pos
        t1 = self._flatten_tokens(p1)
        t2 = self._flatten_tokens(p2)
        t3 = self._flatten_tokens(p3)

        if self.use_rope:
            kv = torch.cat([t1, t2, t3], dim=1)
        else:
            pe1 = pos2d_sincos(self.d_model, p1.shape[2], p1.shape[3], device=device, dtype=dtype)
            pe2 = pos2d_sincos(self.d_model, p2.shape[2], p2.shape[3], device=device, dtype=dtype)
            pe3 = pos2d_sincos(self.d_model, p3.shape[2], p3.shape[3], device=device, dtype=dtype)
            kv = torch.cat([t1 + pe1.unsqueeze(0), t2 + pe2.unsqueeze(0), t3 + pe3.unsqueeze(0)], dim=1)

        # Build batch queries conditioned on text state
        B = state.shape[0]
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
        cond = self.state_proj(state.to(dtype))  # (B, C)
        q = q + cond.unsqueeze(1)

        # Cross-attention blocks
        for blk in self.blocks:
            q = blk(q, kv)

        q = self.out_ln(q)
        # pooled refined state
        state_refined = q.mean(dim=1)  # (B, C)
        return state_refined
