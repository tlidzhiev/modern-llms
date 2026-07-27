import torch
from einops import rearrange


def precompute_freqs_cis(
    head_dim: int,
    seq_len: int,
    theta: float = 500000.0,
    scaling_factor: float = 1.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device)[: head_dim // 2].float() / head_dim)
    )
    positions = torch.arange(seq_len, device=device).float() / scaling_factor
    freqs = torch.outer(positions, freqs)
    return torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_re, x_im = rearrange(x.float(), 'b s h (d two) -> b s h d two', two=2).unbind(-1)
    out = torch.stack([x_re * cos - x_im * sin, x_re * sin + x_im * cos], dim=-1)
    return rearrange(out, 'b s h d two -> b s h (d two)')


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rearrange(freqs_cis, 's d two -> 1 s 1 d two').unbind(-1)
    return _rotate(xq, cos, sin).type_as(xq), _rotate(xk, cos, sin).type_as(xk)
