"""Round-trip tests for inference-only model releases."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import call, patch

import torch
from safetensors.torch import save_model

from experiments.phageseek.config import ModelConfig
from experiments.phageseek.model import PhageSeek
from experiments.phadf.model import DiffusionPhageSeek


class ReleaseCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        self.config = ModelConfig(
            dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32,
            engram_slots=96, engram_dim=8, engram_heads_per_order=2,
            embed_layers=(0, 1), use_attn_residual=False,
            swa_window=None, max_seq_len=64)

    def write_release(self, directory, model):
        directory = Path(directory)
        save_model(model, str(directory / 'model.safetensors'), metadata={'format': 'pt'})
        (directory / 'config.json').write_text(json.dumps({
            'model_config': model.config.to_dict(), 'step': 17,
        }), encoding='utf-8')

    def assert_round_trip(self, model, reference):
        restored, state = type(model).from_checkpoint(reference)
        self.assertEqual(state['step'], 17)
        self.assertEqual(restored.config, model.config)
        self.assertEqual(restored.state_dict().keys(), model.state_dict().keys())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_safetensors_directory_round_trip(self):
        for model_type in (PhageSeek, DiffusionPhageSeek):
            with self.subTest(model=model_type.__name__), tempfile.TemporaryDirectory() as directory:
                model = model_type(self.config)
                self.write_release(directory, model)
                self.assert_round_trip(model, directory)

    def test_huggingface_repo_download(self):
        with tempfile.TemporaryDirectory() as directory:
            model = PhageSeek(self.config)
            self.write_release(directory, model)
            with patch('experiments.phageseek.checkpoint.hf_hub_download',
                       side_effect=lambda repo_id, filename: str(Path(directory) / filename)) as download:
                self.assert_round_trip(model, 'creatiphage/example')
            self.assertEqual(download.call_args_list, [
                call(repo_id='creatiphage/example', filename='model.safetensors'),
                call(repo_id='creatiphage/example', filename='config.json'),
            ])


if __name__ == '__main__':
    unittest.main()
