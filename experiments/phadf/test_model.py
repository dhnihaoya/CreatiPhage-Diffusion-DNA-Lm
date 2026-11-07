"""CPU tests for the released CPDD adapter and restoration path."""
import copy
from dataclasses import asdict
from pathlib import Path
import random
import sqlite3
import tempfile
import unittest

import torch
from torch.nn import functional as F

from experiments.phageseek.config import ModelConfig
from experiments.phageseek.model import PhageSeek
from experiments.phadf.model import DiffusionPhageSeek, encode, corrupt, masked_loss, confidence_restore, MASK
from experiments.phadf.data import SourceSampler


class DiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(1)
        self.cfg = ModelConfig(
            vocab_size=6, dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32,
            engram_slots=96, engram_dim=8, engram_ks=(3, 6, 12), engram_heads_per_order=2,
            engram_layer_ids=(1,), engram_direct_threshold=65536,
            embed_layers=(0, 1), use_attn_residual=False, swa_window=None, max_seq_len=128)
        self.model = DiffusionPhageSeek(copy.deepcopy(self.cfg))
        self.clean = encode('ATCGATCGATCG', torch.device('cpu'))

    def test_pretrained_checkpoint_reuses_every_old_row(self):
        original = PhageSeek(self.cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'base.pt'
            torch.save(dict(model_state=original.state_dict(), model_config=asdict(self.cfg)), path)
            restored, _ = DiffusionPhageSeek.from_checkpoint(path)
            torch.testing.assert_close(restored.token_emb.base, original.token_emb.weight, rtol=0, atol=0)
            torch.testing.assert_close(
                restored.token_emb.mask, original.token_emb.weight[1:5].mean(0, keepdim=True), rtol=0, atol=0)
            old, new = dict(original.named_parameters()), dict(restored.named_parameters())
            for name, value in old.items():
                if name == 'token_emb.weight':
                    continue
                if name.endswith('short_conv.conv.weight'):
                    torch.testing.assert_close(new[name][..., :4], value, rtol=0, atol=0)
                    self.assertEqual(float(new[name][..., 4:].abs().sum()), 0)
                elif name.endswith('memory_tables.0.weight') or name.endswith('memory_tables.1.weight'):
                    torch.testing.assert_close(new[name.replace('.weight', '.atcg')], value[:-1], rtol=0, atol=0)
                else:
                    torch.testing.assert_close(new[name], value, rtol=0, atol=0)
            rejected = Path(directory) / 'old_diffusion.pt'
            torch.save(dict(model_state={'token_emb.weight': torch.zeros(7, self.cfg.dim)},
                            model_config=asdict(self.cfg)), rejected)
            with self.assertRaises(ValueError):
                DiffusionPhageSeek.from_checkpoint(rejected)

    def test_corruption_bounds_shift_and_four_class_loss(self):
        ids, positions, rate = corrupt(self.clean, 2, 9, random.Random(3), fixed_rate=1.)
        self.assertEqual(positions.tolist(), list(range(3, 10)))
        self.assertTrue((ids[:, positions] == MASK).all())
        self.assertTrue(torch.equal(ids[:, :3], self.clean[:, :3]))
        self.assertTrue(torch.equal(ids[:, 10:], self.clean[:, 10:]))
        loss, correct, count = masked_loss(self.model, ids, self.clean, positions)
        reference = F.cross_entropy(
            self.model.logits(ids)[:, positions, 1:5].float().reshape(-1, 4),
            (self.clean[:, positions] - 1).reshape(-1))
        torch.testing.assert_close(loss, reference, rtol=0, atol=0)
        self.assertEqual(count, 7)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_one_and_many_rounds_preserve_context_and_fill_only_bases(self):
        for steps in (1, 3, 32):
            sample = confidence_restore(self.model, self.clean, 2, 9, steps=steps, seed=21)
            self.assertTrue(torch.equal(sample[:, :3], self.clean[:, :3]))
            self.assertTrue(torch.equal(sample[:, 10:], self.clean[:, 10:]))
            self.assertTrue(((sample[:, 3:10] >= 1) & (sample[:, 3:10] <= 4)).all())
            again = confidence_restore(self.model, self.clean, 2, 9, steps=steps, seed=21)
            torch.testing.assert_close(sample, again, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            confidence_restore(self.model, self.clean, -1, 3)

    def test_generation_does_not_read_hidden_reference(self):
        changed = self.clean.clone()
        changed[:, 3:10] = 4
        one = confidence_restore(self.model, self.clean, 2, 9, steps=4, seed=9)
        two = confidence_restore(self.model, changed, 2, 9, steps=4, seed=9)
        torch.testing.assert_close(one, two, rtol=0, atol=0)

    def test_optimizer_checkpoint_resume_and_mask_rng(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-5)
        rng = random.Random(18)
        ids, positions, _ = corrupt(self.clean, 2, 9, rng)
        masked_loss(self.model, ids, self.clean, positions)[0].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snapshot.pt'
            torch.save(dict(model_state=self.model.state_dict(), model_config=asdict(self.model.config),
                            optimizer_state=optimizer.state_dict(), rng=rng.getstate()), path)
            restored, state = DiffusionPhageSeek.from_checkpoint(path)
            other_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-5)
            other_optimizer.load_state_dict(state['optimizer_state'])
            other_rng = random.Random()
            other_rng.setstate(state['rng'])
            for model, opt, generator in ((self.model, optimizer, rng), (restored, other_optimizer, other_rng)):
                ids, positions, _ = corrupt(self.clean, 2, 9, generator)
                masked_loss(model, ids, self.clean, positions)[0].backward()
                opt.step()
            for first, second in zip(self.model.parameters(), restored.parameters()):
                torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_release_swa_and_attnres_forward(self):
        config = ModelConfig(**self.cfg.to_dict())
        config.swa_window = 4
        config.use_attn_residual = True
        model = DiffusionPhageSeek(config)
        with torch.inference_mode():
            logits = model.logits(self.clean)
        self.assertEqual(logits.shape, (1, 14, 7))
        self.assertTrue(torch.isfinite(logits).all())

    def test_source_sampler_resume_and_genome_uniformity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'targets').mkdir()
            connection = sqlite3.connect(root / 'targets/shard_0000.sqlite')
            connection.execute('CREATE TABLE targets(target_id TEXT, genome_idx INTEGER, genome_id TEXT, split TEXT)')
            connection.executemany(
                'INSERT INTO targets VALUES(?,?,?,?)',
                [(f'{genome}-{target}', genome, str(genome), 'train')
                 for genome in range(4) for target in range(genome + 1)])
            connection.commit()
            connection.close()
            sampler = SourceSampler(root, 77, excluded=['3'])
            first = [sampler.next()[1].split('-')[0] for _ in range(3)]
            self.assertEqual(sorted(first), ['0', '1', '2'])
            sampler.next()
            saved = sampler.state_dict()
            again = SourceSampler(root, 77, excluded=['3'])
            again.load_state_dict(saved)
            self.assertEqual([sampler.next() for _ in range(12)], [again.next() for _ in range(12)])


if __name__ == '__main__':
    unittest.main()
