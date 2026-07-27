from transformers import PretrainedConfig


class CustomLLaMAConfig(PretrainedConfig):
    model_type = 'custom_llama'

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
        **kwargs,
    ) -> None:
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_ff = hidden_ff
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len
        self.original_max_seq_len = original_max_seq_len
        self.tie_embeddings = tie_embeddings
        self.initializer_range = initializer_range

        self.hidden_size = dim
        self.num_hidden_layers = num_layers
        self.num_attention_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.max_position_embeddings = max_seq_len

        kwargs.pop('tie_word_embeddings', None)
        super().__init__(tie_word_embeddings=tie_embeddings, **kwargs)  # ty:ignore[unknown-argument]
