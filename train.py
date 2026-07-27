import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, Trainer, TrainingArguments, default_data_collator

from src.dataset import resolve_tokenizer
from src.logger import setup_comet
from src.metrics import compute_metrics, token_loss_for_metrics
from src.model import build_model

logger = logging.getLogger(Path(__file__).name)


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    """
    Main script for training. Instantiates the model, tokenizer, dataset,
    logger, and metrics from the Hydra config. Runs Trainer to train
    and evaluate the model.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment config.
    """
    logger.info(f'Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}')
    setup_comet(cfg)

    training_args = TrainingArguments(**cfg.training_args)
    logger.info(f'Device: {training_args.device}')

    with training_args.main_process_first(desc='dataset preparation'):
        partitions = instantiate(cfg.dataset)
    train_dataset = partitions['train']
    eval_datasets = {name: ds for name, ds in partitions.items() if name != 'train'}
    logger.info(
        f'Train blocks: {len(train_dataset)}'
        + ''.join(f', {name} blocks: {len(ds)}' for name, ds in eval_datasets.items())
    )

    model = build_model(cfg)
    logger.info(
        f'Model:\n{model}\n'
        f'Parameters: {model.num_parameters() / 1e6:.2f}M total, '
        f'{model.num_parameters(exclude_embeddings=True) / 1e6:.2f}M non-embedding'
    )

    tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer(cfg.dataset.tokenizer))
    tokenizer_size = len(tokenizer)  # ty:ignore[invalid-argument-type]
    if tokenizer_size > model.config.vocab_size:
        logger.warning(
            f'Tokenizer {cfg.tokenizer} has {tokenizer_size} tokens but the model was built '
            f'with vocab_size={model.config.vocab_size}; ids above the embedding size will crash.'
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets or None,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if eval_datasets else None,
        preprocess_logits_for_metrics=token_loss_for_metrics if eval_datasets else None,
    )

    resume = cfg.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=resume if resume else None)

    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':
    main()
