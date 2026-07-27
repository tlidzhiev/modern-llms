import torch
import torch.nn as nn
from einops import rearrange

from ..causal_attention import causal_attention


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout

        self.attn = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.attn(x).chunk(3, dim=-1)
        q, k, v = (rearrange(t, 'b s (h d) -> b s h d', h=self.num_heads) for t in (q, k, v))

        output = causal_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        output = rearrange(output, 'b s h d -> b s (h d)')
        return self.resid_dropout(self.proj(output))
