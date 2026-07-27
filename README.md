# Modern LLMs

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#examples">Examples</a> •
</p>

## About

This repository contains from-scratch implementations of modern LLM architectures — GPT-2 and LLaMA — including their attention, MLP, and block variants (e.g. RoPE for LLaMA)

## Installation

### Requirements
- Python 3.12.11
- CUDA (optional, for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/tlidzhiev/modern-llms.git
cd modern-llms

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv --python 3.12.11
source .venv/bin/activate

# Install dependencies via uv
uv sync --all-groups

# Install pre-commit
pre-commit install
```

### CometML Configuration

CometML is used by default for experiment tracking. Create a `.comet.config` file in the project root:

```ini
[comet]
api_key=YOUR_API_KEY
workspace=YOUR_WORKSPACE
project_name=YOUR_PROJECT_NAME
```

## How To Use

To run the training script, use the following command:

```bash
uv run train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

- `CONFIG_NAME`: A config file from `src/configs`
- `HYDRA_CONFIG_ARGUMENTS`: Optional Hydra overrides

Optionally, run `prepare_data.py` beforehand to tokenize and pack the dataset ahead of time, so training can reuse the result instead of redoing the work. This is useful for preprocessing on a CPU-only machine, or before a multi-GPU run so that the ranks find the packed dataset already on disk:

```bash
uv run prepare_data.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

## Examples

Train a LLaMA model with online logging and asset uploads enabled:

```bash
uv run train.py model=llama_small writer.run_name=llama_small \
  dataset.val_size=0.2 training_args.per_device_train_batch_size=64 training_args.gradient_accumulation_steps=2 \
  training_args.learning_rate=6e-4 training_args.num_train_epochs=3 training_args.seed=42 \
  writer.mode=online writer.log_assets=true
```

Train a GPT-2 model with online logging and asset uploads enabled:

```bash
uv run train.py model=gpt_small writer.run_name=gpt_small \
  dataset.val_size=0.2 training_args.per_device_train_batch_size=64 training_args.gradient_accumulation_steps=2 \
  training_args.learning_rate=6e-4 training_args.num_train_epochs=3 training_args.seed=42 \
  writer.mode=online writer.log_assets=true
```
