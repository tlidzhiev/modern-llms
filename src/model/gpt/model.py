from math import sqrt

import torch
import torch.nn as nn

from .block import TransformerBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 50257,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        hidden_ff: int | None = None,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        max_seq_len: int = 1024,
        tie_embeddings: bool = True,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads}).')

        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.tie_embeddings = tie_embeddings
        self.initializer_range = initializer_range

        self.wte = nn.Embedding(vocab_size, dim)
        self.wpe = nn.Embedding(max_seq_len, dim)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                hidden_ff=hidden_ff,
                dropout=dropout,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        )
        self.ln_f = nn.LayerNorm(dim, eps=norm_eps)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Identify the residual output projections by identity so _init_weights
        # can scale them down without subclassing nn.Linear or matching on
        # parameter names, which are unavailable in HuggingFace's
        # module-by-module _init_weights calls.
        self._residual_proj_ids = {
            id(block.attn.proj)  # ty:ignore[unresolved-attribute]
            for block in self.h
        } | {
            id(block.mlp.proj)  # ty:ignore[unresolved-attribute]
            for block in self.h
        }

        # Init before tying so the shared tensor is drawn only once.
        self.apply(self._init_weights)

        if tie_embeddings:
            self.lm_head.weight = self.wte.weight

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            std = self.initializer_range
            if id(module) in self._residual_proj_ids:
                std /= sqrt(2 * self.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        tokens = input_ids
        _, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f'Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.')

        positions = torch.arange(seq_len, device=tokens.device)
        h = self.drop(self.wte(tokens) + self.wpe(positions))

        for block in self.h:
            h = block(h)

        h = self.ln_f(h)
        return {'logits': self.lm_head(h)}
