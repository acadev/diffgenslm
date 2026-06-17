"""
DiffGenome: Bidirectional Gemma-style transformer for discrete diffusion.

Key differences from a standard decoder:
  - No causal mask — all positions attend to all positions (bidirectional)
  - RoPE from HiSAN's hisan_v2.py (ported here for self-containment)
  - RMSNorm + SwiGLU FFN + Grouped-Query Attention (GQA)
  - Accepts a mask_token_id input; predicts original tokens at masked positions

Architecture sizes (approximate param counts):
  small  : hidden=512,  layers=8,  heads=8,  kv_heads=4,  ffn=1366  →  ~28M params
  medium : hidden=1024, layers=16, heads=16, kv_heads=8,  ffn=2730  → ~220M params
  large  : hidden=2048, layers=24, heads=16, kv_heads=8,  ffn=5461  → ~1.7B params
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DiffGenomeConfig:
    vocab_size: int = 8280        # composite tokenizer vocab (adjust after training BPE)
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    num_kv_heads: int = 4         # GQA: key/value heads (must divide num_heads)
    ffn_intermediate_size: int = 1366   # SwiGLU: typically 8/3 * hidden, rounded
    rope_theta: float = 10_000.0
    max_seq_len: int = 4096
    dropout: float = 0.1
    pad_token_id: int = 0
    mask_token_id: int = 4        # <mask> token in composite vocab

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def kv_repeats(self) -> int:
        return self.num_heads // self.num_kv_heads


# ---------------------------------------------------------------------------
# RoPE (ported from HiSAN hisan_v2.py)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(
    head_dim: int, max_len: int, theta: float = 10_000.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len).float()
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, H, L, Dh]  →  rotated q, k"""
    q_r, q_i = q.float().reshape(*q.shape[:-1], -1, 2).unbind(-1)
    k_r, k_i = k.float().reshape(*k.shape[:-1], -1, 2).unbind(-1)
    cos = freqs_cos[None, None, :, :]   # [1, 1, L, Dh//2]
    sin = freqs_sin[None, None, :, :]
    q_out = torch.stack([q_r * cos - q_i * sin, q_r * sin + q_i * cos], -1).flatten(-2)
    k_out = torch.stack([k_r * cos - k_i * sin, k_r * sin + k_i * cos], -1).flatten(-2)
    return q_out.type_as(q), k_out.type_as(k)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, hidden: int, intermediate: int, dropout: float = 0.0):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj   = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class BidirectionalGQA(nn.Module):
    """
    Grouped-Query Attention without causal masking.

    All positions attend to all positions — required for diffusion, where
    the model must reason about the full (partially masked) context at once.
    """

    def __init__(self, cfg: DiffGenomeConfig):
        super().__init__()
        self.num_heads    = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.kv_repeats   = cfg.kv_repeats
        self.head_dim     = cfg.head_dim

        self.q_proj  = nn.Linear(cfg.hidden_size, cfg.num_heads * cfg.head_dim, bias=False)
        self.k_proj  = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.v_proj  = nn.Linear(cfg.hidden_size, cfg.num_kv_heads * cfg.head_dim, bias=False)
        self.o_proj  = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,                      # [B, L, D]
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        padding_mask: Optional[torch.Tensor],  # [B, L] bool, True = keep
    ) -> torch.Tensor:
        B, L, _ = x.shape

        q = self.q_proj(x).view(B, L, self.num_heads,    self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, freqs_cos, freqs_sin)

        # Expand KV heads to match Q heads (GQA)
        if self.kv_repeats > 1:
            k = k.repeat_interleave(self.kv_repeats, dim=1)
            v = v.repeat_interleave(self.kv_repeats, dim=1)

        # Attention scores — NO causal mask
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, L, L]

        if padding_mask is not None:
            # Mask out pad positions from attending and being attended to
            key_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            scores = scores.masked_fill(~key_mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_drop(weights)

        out = torch.matmul(weights, v)                           # [B, H, L, Dh]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)


class DiffGenomeBlock(nn.Module):
    """Pre-LN transformer block (bidirectional GQA + SwiGLU FFN)."""

    def __init__(self, cfg: DiffGenomeConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size)
        self.attn  = BidirectionalGQA(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size)
        self.ffn   = SwiGLU(cfg.hidden_size, cfg.ffn_intermediate_size, cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin, padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DiffGenomeModel(nn.Module):
    """
    Bidirectional transformer backbone for discrete genomic diffusion.

    Input:  token_ids [B, L]  (may contain mask_token_id at diffused positions)
    Output: logits    [B, L, vocab_size]

    The model is trained to predict the original (clean) token at every
    masked position; the diffusion loss ignores unmasked positions.
    """

    def __init__(self, cfg: DiffGenomeConfig):
        super().__init__()
        self.cfg = cfg

        self.embed   = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_token_id)
        self.blocks  = nn.ModuleList([DiffGenomeBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm    = RMSNorm(cfg.hidden_size)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        # Tie input/output embeddings (common in LMs, reduces parameters)
        self.lm_head.weight = self.embed.weight

        freqs_cos, freqs_sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,                # [B, L]
        padding_mask: Optional[torch.Tensor] = None,  # [B, L] bool True=real token
    ) -> torch.Tensor:
        """Returns logits [B, L, vocab_size]."""
        _, L = input_ids.shape

        if padding_mask is None:
            padding_mask = input_ids.ne(self.cfg.pad_token_id)

        x = self.embed(input_ids)              # [B, L, D]

        fc = self.freqs_cos[:L]
        fs = self.freqs_sin[:L]

        for block in self.blocks:
            x = block(x, fc, fs, padding_mask)

        x = self.norm(x)
        return self.lm_head(x)                 # [B, L, V]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, cfg: DiffGenomeConfig) -> "DiffGenomeModel":
        return cls(cfg)
