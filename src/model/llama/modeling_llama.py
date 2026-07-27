import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_llama import CustomLLaMAConfig
from .model import LLaMA
from .rope import precompute_freqs_cis


class CustomLLaMAForCausalLM(PreTrainedModel):
    config_class = CustomLLaMAConfig
    base_model_prefix = 'model'
    _tied_weights_keys = {'model.output.weight': 'model.tok_embeddings.weight'}
    _no_split_modules = ['TransformerBlock']

    def __init__(self, config: CustomLLaMAConfig) -> None:
        super().__init__(config)
        self.model = LLaMA(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            hidden_ff=config.hidden_ff,
            multiple_of=config.multiple_of,
            norm_eps=config.norm_eps,
            rope_theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            original_max_seq_len=config.original_max_seq_len,
            tie_embeddings=config.tie_embeddings,
            initializer_range=config.initializer_range,
        )
        self._rope_device: torch.device | None = None
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        self.model._init_weights(module)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.tok_embeddings

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.tok_embeddings = value

    def get_output_embeddings(self) -> nn.Module:
        return self.model.output

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.model.output = value

    def _refresh_rope_cache(self, device: torch.device) -> None:
        head_dim = self.config.dim // self.config.num_heads
        scaling_factor = 1.0
        if self.config.original_max_seq_len is not None:
            scaling_factor = max(1.0, self.config.max_seq_len / self.config.original_max_seq_len)
        self.model.freqs_cis = precompute_freqs_cis(
            head_dim,
            self.config.max_seq_len,
            self.config.rope_theta,
            scaling_factor=scaling_factor,
            device=device,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if self._rope_device != input_ids.device:
            self._refresh_rope_cache(input_ids.device)
            self._rope_device = input_ids.device

        logits = self.model(input_ids=input_ids)['logits']

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)
