import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

logger = logging.getLogger(Path(__file__).name)


@hydra.main(version_base='1.3', config_path='src/configs', config_name='train')
def main(cfg: DictConfig) -> None:
    """
    Main script for dataset preparation. Instantiates the dataset from the
    Hydra config. Tokenizes and packs it ahead of training, so training can
    reuse the result instead of redoing the work.

    Running this is optional, but worth doing when preprocessing should happen
    on a CPU-only machine, or before a multi-GPU run so that the ranks find the
    packed dataset already on disk.

    Parameters
    ----------
    cfg : DictConfig
        Hydra experiment config; only ``cfg.dataset`` is used.
    """
    partitions = instantiate(cfg.dataset)
    for name, partition in partitions.items():
        logger.info(f'{name}: {len(partition)} blocks')


if __name__ == '__main__':
    main()
