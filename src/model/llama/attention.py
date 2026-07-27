import torch
import torch.nn as nn
from einops import rearrange

from ..causal_attention import causal_attention
from .rope import apply_rotary_emb


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(num_heads * self.head_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        xq = rearrange(self.wq(x), 'b s (h d) -> b s h d', h=self.num_heads)
        xk = rearrange(self.wk(x), 'b s (h d) -> b s h d', h=self.num_kv_heads)
        xv = rearrange(self.wv(x), 'b s (h d) -> b s h d', h=self.num_kv_heads)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        output = causal_attention(xq, xk, xv)
        output = rearrange(output, 'b s h d -> b s (h d)')
        return self.wo(output)
