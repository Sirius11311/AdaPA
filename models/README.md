# Model Configuration Guide

This document describes how to download and configure the models required for AdaPA-Agent.

## Overview

AdaPA-Agent supports two generation modes:

1. **API-based Generation** (Default): Uses OpenAI-compatible API (e.g., GPT-4o-mini) for strength estimation and generation. No local GPU required.

2. **Preference Arithmetic Generation**: Uses local Llama models for preference arithmetic-based controllable generation. Requires GPU with sufficient memory.

---

## API-based Generation (Recommended for Quick Start)

### Requirements
- OpenAI API key or compatible endpoint
- No GPU required

### Configuration
Set the following environment variables or pass as command-line arguments:

```bash
export ADAPA_API_KEY="your-api-key"
export ADAPA_API_BASE="https://api.openai.com/v1/"
```

### Supported Models
- `gpt-4o-mini` (Recommended)
- `gpt-4o`
- `gpt-3.5-turbo`
- Other OpenAI-compatible models

---

## Preference Arithmetic Generation (Advanced)

### Requirements
- NVIDIA GPU with at least 16GB VRAM
- CUDA 11.7 or later
- Local model files

### Model Download

#### Option 1: Download from Hugging Face

```bash
# Install huggingface-cli if not installed
pip install huggingface-hub

# Login to Hugging Face (required for gated models)
huggingface-cli login

# Download Llama-2-7b-chat-hf
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir ./Llama-2-7b-chat-hf
```

#### Option 2: Download via Python

```python
from huggingface_hub import snapshot_download

# Download model
snapshot_download(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    local_dir="./Llama-2-7b-chat-hf",
    token="your-hf-token"
)
```

### Recommended Models

| Model | Size | VRAM | Description |
|-------|------|------|-------------|
| Llama-2-7b-chat-hf | 7B | ~16GB | Recommended for most use cases |
| Llama-2-13b-chat-hf | 13B | ~32GB | Better quality, higher resource usage |

### Configuration

Set the following environment variables:

```bash
# Path to local model
export ADAPA_GENERATION_MODEL="/path/to/Llama-2-7b-chat-hf"

# Hugging Face token (optional, for gated models)
export ADAPA_HF_TOKEN="your-hf-token"

# GPU selection
export CUDA_VISIBLE_DEVICES="0"
```

### Enable Preference Arithmetic

When running evaluation, add the `--use_preference_arithmetic` flag:

```bash
cd evaluation
./run_evaluation.sh --use_pa --gpu 0
```

Or in Python:

```python
from algorithms import AdaPAAgent

agent = AdaPAAgent(
    api_key="your-api-key",
    generation_model="/path/to/Llama-2-7b-chat-hf",
    hf_token="your-hf-token",
    use_preference_arithmetic=True
)
```

---

## Troubleshooting

### Out of Memory (OOM) Errors

If you encounter CUDA OOM errors:
1. Use a smaller model (7B instead of 13B)
2. Reduce batch size by setting `--max_workers 1`
3. Use quantization (requires additional setup)

### Slow Model Loading

Models are cached after first load. If loading is slow:
1. Ensure model is stored on SSD
2. Use offline mode for local models:
   ```bash
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   ```

### Network Issues

If you experience connectivity issues with Hugging Face:
1. Download models locally first
2. Set offline mode as shown above
3. Use a mirror if available in your region

---

## Model Arithmetic Dependency

The preference arithmetic feature requires the `model-arithmetic` package from ETH Zurich:

```bash
pip install model-arithmetic
```

For more information, see: https://github.com/eth-sri/language-model-arithmetic

