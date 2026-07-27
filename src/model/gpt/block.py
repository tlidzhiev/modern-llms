import torch
import torch.nn as nn

from .attention import Attention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        hidden_ff: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim, eps=norm_eps)
        self.attn = Attention(dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(dim, eps=norm_eps)
        self.mlp = MLP(dim, hidden_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
