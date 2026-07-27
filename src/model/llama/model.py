import torch
import torch.nn as nn

from .block import TransformerBlock
from .rope import precompute_freqs_cis


class LLaMA(nn.Module):
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_kv_heads: int = 4,
        hidden_ff: int | None = None,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_seq_len: int = 1024,
        original_max_seq_len: int | None = None,
        tie_embeddings: bool = True,
        initializer_range: float = 0.02,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f'dim ({dim}) must be divisible by num_heads ({num_heads}).')
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f'num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads}).'
            )

        rope_scaling_factor = 1.0
        if original_max_seq_len is not None:
            if original_max_seq_len <= 0:
                raise ValueError(
                    f'original_max_seq_len must be positive, got {original_max_seq_len}.'
                )
            rope_scaling_factor = max(1.0, max_seq_len / original_max_seq_len)

        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.initializer_range = initializer_range

        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                hidden_ff=hidden_ff,
                multiple_of=multiple_of,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        )
        self.norm = nn.RMSNorm(dim, eps=norm_eps)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Init before tying so the shared tensor is drawn only once.
        self.apply(self._init_weights)

        if tie_embeddings:
            self.output.weight = self.tok_embeddings.weight

        head_dim = dim // num_heads
        freqs_cis = precompute_freqs_cis(
            head_dim, max_seq_len, rope_theta, scaling_factor=rope_scaling_factor
        )
        self.register_buffer('freqs_cis', freqs_cis, persistent=False)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> dict[str, torch.Tensor]:
        tokens = input_ids
        _, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f'Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}.')

        h = self.tok_embeddings(tokens)
        freqs_cis = self.freqs_cis[:seq_len].to(h.device)  # ty:ignore[not-subscriptable]

        for layer in self.layers:
            h = layer(h, freqs_cis)

        h = self.norm(h)
        return {'logits': self.output(h)}
