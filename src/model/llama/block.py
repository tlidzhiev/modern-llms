import torch
import torch.nn as nn

from .attention import Attention
from .mlp import FeedForward


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        hidden_ff: int | None = None,
        multiple_of: int = 1024,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attention = Attention(dim, num_heads, num_kv_heads)
        self.feed_forward = FeedForward(dim, hidden_ff, multiple_of)
        self.attention_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), freqs_cis)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x
