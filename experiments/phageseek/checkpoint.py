"""Load original checkpoints or inference-only safetensors releases."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def load_checkpoint(reference):
    """Resolve a local checkpoint, release directory, or Hugging Face model ID."""
    reference = os.fspath(reference)
    path = Path(reference).expanduser()
    if path.suffix in ('.pt', '.safetensors') and not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix == '.pt':
        return torch.load(path, map_location='cpu', weights_only=True)
    if path.is_dir():
        weights = path / 'model.safetensors'
        config = path / 'config.json'
    elif path.is_file() and path.suffix == '.safetensors':
        weights = path
        config = path.with_name('config.json')
    elif path.exists():
        raise ValueError(f'Unsupported checkpoint: {path}')
    else:
        weights = Path(hf_hub_download(repo_id=reference, filename='model.safetensors'))
        config = Path(hf_hub_download(repo_id=reference, filename='config.json'))
    metadata = json.loads(config.read_text(encoding='utf-8'))
    return dict(
        model_state=load_file(str(weights), device='cpu'),
        model_config=metadata['model_config'],
        step=metadata.get('step'),
    )
