import logging

from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, PreTrainedModel

logger = logging.getLogger(__name__)


def build_model(cfg: DictConfig) -> PreTrainedModel:
    config = instantiate(cfg.model)
    if cfg.from_pretrained is None:
        return AutoModelForCausalLM.from_config(config)

    logger.info(f'Loading weights from {cfg.from_pretrained}')
    return AutoModelForCausalLM.from_pretrained(cfg.from_pretrained, config=config)
