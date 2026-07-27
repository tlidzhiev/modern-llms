from transformers import PretrainedConfig


class CustomGPTConfig(PretrainedConfig):
    model_type = 'custom_gpt'

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
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_ff = hidden_ff
        self.dropout = dropout
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.tie_embeddings = tie_embeddings
        self.initializer_range = initializer_range

        self.hidden_size = dim
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads
        self.max_position_embeddings = max_seq_len

        kwargs.pop('tie_word_embeddings', None)
        super().__init__(tie_word_embeddings=tie_embeddings, **kwargs)  # ty:ignore[unknown-argument]
