import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_gpt import CustomGPTConfig
from .model import GPT


class CustomGPTForCausalLM(PreTrainedModel):
    config_class = CustomGPTConfig
    base_model_prefix = 'model'
    _tied_weights_keys = {'model.lm_head.weight': 'model.wte.weight'}
    _no_split_modules = ['TransformerBlock']

    def __init__(self, config: CustomGPTConfig) -> None:
        super().__init__(config)
        self.model = GPT(
            vocab_size=config.vocab_size,
            dim=config.dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            hidden_ff=config.hidden_ff,
            dropout=config.dropout,
            norm_eps=config.norm_eps,
            max_seq_len=config.max_seq_len,
            tie_embeddings=config.tie_embeddings,
            initializer_range=config.initializer_range,
        )
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        self.model._init_weights(module)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.wte

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.wte = value

    def get_output_embeddings(self) -> nn.Module:
        return self.model.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        self.model.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
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
