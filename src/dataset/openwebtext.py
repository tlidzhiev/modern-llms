import logging
import os
from itertools import chain
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from datasets import List as ListFeature
from transformers import AutoTokenizer

from src.utils.io import get_root

logger = logging.getLogger(__name__)

TOKENIZERS = {
    'mistralai': 'mistralai/Mistral-7B-v0.1',
    'gpt2': 'openai-community/gpt2',
    'llama': 'meta-llama/Llama-2-7b-hf',
}

DATASET_NAME = 'ashaba1in/small_openwebtext'


def resolve_tokenizer(tokenizer: str) -> str:
    return TOKENIZERS.get(tokenizer, tokenizer)


def add_labels(batch: dict[str, Any]) -> dict[str, torch.Tensor]:
    input_ids = torch.as_tensor(batch['input_ids'], dtype=torch.long)
    return {'input_ids': input_ids, 'labels': input_ids.clone()}


def blocks_dir(dataset_name: str, tokenizer: str, max_seq_len: int) -> Path:
    slug = f'{resolve_tokenizer(tokenizer)}-{max_seq_len}'.replace('/', '__')
    return get_root() / 'data' / dataset_name.replace('/', '__') / slug


def prepare_blocks(
    max_seq_len: int = 1024,
    tokenizer: str = 'mistralai',
    dataset_name: str = DATASET_NAME,
    save_dir: Path | str | None = None,
    force_rebuild: bool = False,
    num_proc: int | None = None,
    group_batch_size: int = 10_000,
) -> Dataset:
    path = (
        Path(save_dir) if save_dir is not None else blocks_dir(dataset_name, tokenizer, max_seq_len)
    )

    if path.exists() and not force_rebuild:
        blocks = Dataset.load_from_disk(str(path))
        logger.info(f'Loaded {len(blocks)} packed blocks of {max_seq_len} from {path}')
        return blocks

    if num_proc is None:
        num_proc = max(1, (os.cpu_count() or 1) - 2)

    hf_tokenizer = AutoTokenizer.from_pretrained(resolve_tokenizer(tokenizer))

    eos_token_id = hf_tokenizer.eos_token_id or hf_tokenizer.pad_token_id  # ty:ignore[unresolved-attribute]
    bos_token_id = hf_tokenizer.bos_token_id  # ty:ignore[unresolved-attribute]
    if bos_token_id == eos_token_id:
        bos_token_id = None

    vocab_size = len(hf_tokenizer)  # ty:ignore[invalid-argument-type]
    dtype = 'uint16' if vocab_size <= 65535 else 'uint32'
    logger.info(f'Vocab size: {vocab_size} -> dtype: {dtype}')

    def tokenize(examples: dict[str, list]) -> dict[str, list]:
        outputs = hf_tokenizer(examples['text'], add_special_tokens=False)  # ty:ignore[call-non-callable]

        formatted_ids = []
        for ids in outputs['input_ids']:
            sequence = [] if bos_token_id is None else [bos_token_id]
            sequence.extend(ids)
            sequence.append(eos_token_id)
            formatted_ids.append(sequence)

        return {'token_ids': formatted_ids}

    def group(examples: dict[str, list]) -> dict[str, list]:
        stream = list(chain.from_iterable(examples['token_ids']))
        end = (len(stream) // max_seq_len) * max_seq_len
        return {'input_ids': [stream[i : i + max_seq_len] for i in range(0, end, max_seq_len)]}

    logger.info(f'Loading {dataset_name} and tokenizing using {num_proc} CPU processes...')
    dataset = load_dataset(dataset_name, split='train')

    tokenized = dataset.map(
        tokenize,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc='Tokenizing texts',
    )

    blocks = tokenized.map(
        group,
        batched=True,
        batch_size=group_batch_size,
        num_proc=num_proc,
        remove_columns=tokenized.column_names,
        features=Features({'input_ids': ListFeature(Value(dtype), length=max_seq_len)}),
        desc=f'Packing into blocks of {max_seq_len}',
    )

    logger.info(
        f'Packed {len(blocks)} blocks of {max_seq_len} ({len(blocks) * max_seq_len} tokens)'
    )
    blocks.save_to_disk(str(path))
    logger.info(f'Saved packed dataset to {path}')

    return Dataset.load_from_disk(str(path))


def build_openwebtext(
    max_seq_len: int = 1024,
    tokenizer: str = 'mistralai',
    dataset_name: str = DATASET_NAME,
    save_dir: Path | str | None = None,
    force_rebuild: bool = False,
    limit: int | None = None,
    shuffle: bool = True,
    seed: int = 42,
    val_size: float | int = 0,
    num_proc: int | None = None,
    group_batch_size: int = 10_000,
) -> DatasetDict:
    blocks = prepare_blocks(
        max_seq_len=max_seq_len,
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        save_dir=save_dir,
        force_rebuild=force_rebuild,
        num_proc=num_proc,
        group_batch_size=group_batch_size,
    )

    if shuffle:
        blocks = blocks.shuffle(seed=seed)
    if limit is not None:
        blocks = blocks.select(range(min(limit, len(blocks))))

    if val_size:
        split = blocks.train_test_split(test_size=val_size, seed=seed, shuffle=not shuffle)
        partitions = DatasetDict({'train': split['train'], 'val': split['test']})
    else:
        partitions = DatasetDict({'train': blocks})

    for partition in partitions.values():
        partition.set_transform(add_labels)
    return partitions
