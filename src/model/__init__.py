from transformers import AutoConfig, AutoModelForCausalLM

from .build import build_model
from .gpt import GPT, CustomGPTConfig, CustomGPTForCausalLM
from .llama import CustomLLaMAConfig, CustomLLaMAForCausalLM, LLaMA

for config_class, model_class in (
    (CustomGPTConfig, CustomGPTForCausalLM),
    (CustomLLaMAConfig, CustomLLaMAForCausalLM),
):
    AutoConfig.register(config_class.model_type, config_class, exist_ok=True)
    AutoModelForCausalLM.register(config_class, model_class, exist_ok=True)

__all__ = [
    'GPT',
    'LLaMA',
    'CustomGPTConfig',
    'CustomGPTForCausalLM',
    'CustomLLaMAConfig',
    'CustomLLaMAForCausalLM',
    'build_model',
]
