import torch
import torch.nn.functional as F
from einops import rearrange


def causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    num_heads, num_kv_heads = q.shape[2], k.shape[2]
    q, k, v = (rearrange(t, 'b s h d -> b h s d') for t in (q, k, v))
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=True,
        enable_gqa=num_heads != num_kv_heads,
        dropout_p=dropout_p,
    )
    return rearrange(output, 'b h s d -> b s h d')
