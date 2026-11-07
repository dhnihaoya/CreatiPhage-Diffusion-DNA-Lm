"""CPU checks for the restart recipe, grouped learning rates, and AR-only entry."""
from pathlib import Path
import random
import tempfile
import unittest

import torch

from experiments.phadf.engram_adapt import checkpoint_kind
from experiments.phadf.grouped_lr import apply_lrs, build_optimizer, parameter_group, scaled_lrs
from experiments.phadf.model import DiffusionPhageSeek, masked_loss
from experiments.phadf.recipe import (
    MIN_VISIBLE_GAP, SUPERVISION_BUDGET, build_supervised_window, radius_for_stage,
    supervision_budget)
from experiments.phadf.restart import (
    RECIPE, RESUMABLE, BatchPlan, assert_training_source, backward_example,
    build_parser, checkpoint_paths, next_batch, parse_keep_steps, sample_rng)
from experiments.phadf.corruption import BRANCHES, BranchSampler
from experiments.phageseek.config import ModelConfig


class _FakeParallel:
    def __init__(self, rank, world):
        self.rank, self.world = rank, world


class _FakeCursor:
    """Stands in for the sibling-CDS query; the column order defines the dict keys."""

    def __init__(self, rows):
        self.description = [(name,) for name in ('target_id', 'target_start', 'target_end')]
        self.rows = [(row['target_id'], row['target_start'], row['target_end']) for row in rows]

    def __iter__(self):
        return iter(self.rows)


class _FakeSampler:
    def __init__(self, rows):
        self.rows, self.calls = rows, 0
        self.connections = {'targets/shard_0000.sqlite': self}

    def execute(self, statement, parameters):
        return _FakeCursor(self.rows)

    def next(self):
        self.calls += 1
        return 'targets/shard_0000.sqlite', self.rows[self.calls % len(self.rows)]['target_id']


class _FakeReader:
    def __init__(self, sequence, rows, log):
        self.sequence = sequence
        self.index = {row['target_id']: row for row in rows}
        self.log = log

    def read(self, database, target_id):
        self.log.append(target_id)
        return dict(sequence=self.sequence, target=dict(self.index[target_id], genome_idx=0))


def _tiny():
    cfg = ModelConfig(
        vocab_size=6, dim=16, layers=2, heads=2, kv_heads=1, ffn_dim=32,
        engram_slots=96, engram_dim=8, engram_ks=(3, 6, 12), engram_heads_per_order=2,
        engram_layer_ids=(1,), engram_direct_threshold=65536, embed_layers=(0, 1),
        use_attn_residual=False, swa_window=None, max_seq_len=128)
    return cfg, DiffusionPhageSeek(cfg)


class RestartTests(unittest.TestCase):
    def _spaced_window(self, gap=600, count=6, length=400):
        """Genes laid out with `gap` of visible sequence between them."""
        sequence = 'ACGT' * 2000
        rows = [dict(target_id=f'g{i}', target_start=500 + i * (length + gap),
                     target_end=500 + i * (length + gap) + length)
                for i in range(count)]
        return sequence, rows

    def test_window_masks_several_cds_and_splits_full_mask_boundaries(self):
        sequence, rows = self._spaced_window()
        primary = rows[2]
        example = build_supervised_window(sequence, primary, rows, radius_for_stage('A'),
                                          random.Random(1), 'partial', 0.6)
        self.assertEqual(example['window_start'], 0)
        chosen = [item['target_id'] for item in example['intervals']]
        self.assertIn('g2', chosen)
        self.assertGreater(len(chosen), 1)
        self.assertEqual(example['positions'].unique().numel(), example['positions'].numel())
        modes = []
        for seed in range(400):
            drawn = build_supervised_window(sequence, primary, rows, None, random.Random(seed), 'full', 1.0)
            modes.append(drawn['intervals'][0]['mode'])
        exact = modes.count('exact_boundary')
        self.assertGreater(exact, 140)
        self.assertLess(exact, 260)

    def test_overlapping_cds_is_never_co_selected(self):
        sequence, rows = self._spaced_window()
        overlapping = dict(target_id='overlap', target_start=rows[2]['target_start'] + 100,
                           target_end=rows[2]['target_end'] + 100)
        for seed in range(50):
            example = build_supervised_window(sequence, rows[2], rows + [overlapping],
                                              radius_for_stage('A'), random.Random(seed), 'partial', 0.5)
            self.assertNotIn('overlap', [item['target_id'] for item in example['intervals']])
            self.assertEqual(example['positions'].unique().numel(), example['positions'].numel())

    def test_masked_cds_keep_a_visible_gap_between_them(self):
        """Adjacent genes must not be masked together, otherwise the ones in the middle
        lose the flanking context they are supposed to be reconstructed from."""
        sequence = 'ACGT' * 2000
        # Genes packed back to back with only 20 bp between them.
        dense = [dict(target_id=f'd{i}', target_start=500 + i * 420,
                      target_end=500 + i * 420 + 400) for i in range(12)]
        for seed in range(30):
            example = build_supervised_window(sequence, dense[5], dense, radius_for_stage('A'),
                                              random.Random(seed), 'full', 1.0)
            picked = sorted((item['target_start'], item['target_end'])
                            for item in example['intervals'])
            for (_, end), (start, _) in zip(picked, picked[1:]):
                self.assertGreaterEqual(start - end, MIN_VISIBLE_GAP)

    def test_mask_fraction_controls_the_share_of_the_window(self):
        sequence, rows = self._spaced_window(gap=600, count=12)
        shares = {}
        for fraction in (0.12, 0.25):
            measured = []
            for seed in range(40):
                example = build_supervised_window(sequence, rows[5], rows, radius_for_stage('A'),
                                                  random.Random(seed), 'full', 1.0, fraction)
                measured.append(example['positions'].numel() / example['window_tokens'])
            shares[fraction] = sum(measured) / len(measured)
        self.assertLess(shares[0.12], shares[0.25])
        self.assertLess(shares[0.25], 0.45)

    def test_budget_is_absolute_but_a_short_window_is_still_capped(self):
        """Supervision per sample must not scale with the window, otherwise a 3x longer
        window costs 3x the FLOPs for the same gradient. The fraction only guards the
        case where the absolute budget would hide most of a short crop."""
        self.assertEqual(supervision_budget(32770, 7000, 0.25), 7000)
        self.assertEqual(supervision_budget(48000, 7000, 0.25), 7000)
        self.assertEqual(supervision_budget(10242, 7000, 0.25), 2560)
        with self.assertRaises(ValueError):
            supervision_budget(32770, 0, 0.25)

    def test_longer_window_does_not_inflate_supervision(self):
        """A 3x longer window must not cost 3x the FLOPs for the same gradient signal."""
        sequence = 'ACGT' * 15000
        rows = [dict(target_id=f'g{index}', target_start=600 + index * 1200,
                     target_end=600 + index * 1200 + 600) for index in range(45)]
        measured = {}
        for radius in (5120, 16384, None):
            counts = [build_supervised_window(sequence, rows[22], rows, radius,
                                              random.Random(seed), 'full', 1.0)['positions'].numel()
                      for seed in range(20)]
            measured[radius] = sum(counts) / len(counts)
        self.assertLess(measured[16384], 1.15 * measured[None])
        self.assertGreater(measured[16384], 0.85 * measured[None])
        self.assertLessEqual(measured[16384], SUPERVISION_BUDGET + 800)
        # The stage-A crop is small enough that the fraction cap binds instead.
        self.assertLess(measured[5120], measured[16384])

    def test_groups_schedule_and_one_update(self):
        cfg, model = _tiny()
        names = [parameter_group(name) for name, _ in model.named_parameters()]
        self.assertEqual(set(names), {'new', 'adapter', 'backbone'})
        self.assertIn('new', [parameter_group(name) for name, _ in model.named_parameters() if name.endswith('.mask_rows')])
        optimizer = build_optimizer(model, 1e-4)
        # WSD: warmup ramps, stable stays flat forever, decay only once asked for.
        self.assertAlmostEqual(scaled_lrs(0, 1e-4, warmup=2000)['new'], 1e-7)
        self.assertAlmostEqual(scaled_lrs(1999, 1e-4, warmup=2000)['adapter'], 1e-4)
        self.assertAlmostEqual(scaled_lrs(50000, 1e-4, warmup=2000)['adapter'], 1e-4)
        self.assertAlmostEqual(scaled_lrs(10000, 1e-4, warmup=2000)['backbone'], 5e-5)
        self.assertAlmostEqual(
            scaled_lrs(5999, 1e-4, warmup=2000, decay_start=4000, decay_steps=2000)['adapter'], 1e-5)
        self.assertAlmostEqual(
            scaled_lrs(4000, 1e-4, warmup=2000, decay_start=4000, decay_steps=2000)['adapter'], 1e-4)
        # A stage switch re-warms from the floor without touching the stable level.
        self.assertAlmostEqual(
            scaled_lrs(9000, 1e-4, warmup=2000, switch_step=9000, rewarmup_steps=300)['adapter'], 5e-5)
        self.assertAlmostEqual(
            scaled_lrs(9300, 1e-4, warmup=2000, switch_step=9000, rewarmup_steps=300)['adapter'], 1e-4)
        sequence = 'ACGT' * 10
        primary = dict(target_id='b', target_start=12, target_end=20)
        candidates = [dict(target_id='a', target_start=2, target_end=8), primary]
        example = build_supervised_window(sequence, primary, candidates, 4096, random.Random(3), 'partial', 0.5)
        apply_lrs(optimizer, 0, 1e-4, warmup=2000)
        before = float(masked_loss(model, example['ids'], example['clean'], example['positions'])[0])
        optimizer.zero_grad(set_to_none=True)
        stats = backward_example(model, example, aa_weight=0.0)
        optimizer.step()
        after = float(masked_loss(model, example['ids'], example['clean'], example['positions'])[0])
        self.assertTrue(stats['supervised'] > 0)
        self.assertNotEqual(before, after)

    def test_every_rank_draws_the_same_update_but_a_different_sample(self):
        """All ranks share the update policy but read different source targets."""
        sequence, rows = self._spaced_window()
        plan = BatchPlan(radius=radius_for_stage('B'), seed=7, long_probability=0.3)
        seen, reads = [], []

        for rank in range(6):
            DP = _FakeParallel(rank, 6)
            sampler = _FakeSampler(rows)
            reader = _FakeReader(sequence, rows, reads)
            branches, masks = BranchSampler(3), random.Random(11)
            draws = []
            for step in range(40):
                example, draw = next_batch(DP, sampler, reader, plan, branches, masks,
                                           torch.device('cpu'), step)
                draws.append((draw['branch'], draw['rate'], draw['substitution'],
                              draw['whole_source']))
            seen.append(draws)
            # One read per step, not one per rank: the window build is no longer replayed.
            self.assertEqual(sampler.calls, 40 * 6)
        self.assertEqual(len(set(map(tuple, seen))), 1)
        self.assertEqual(len(reads), 40 * 6)
        # Ranks sharing a step must look at different genomes.
        for step in range(40):
            picked = {reads[rank * 40 + step] for rank in range(6)}
            self.assertEqual(len(picked), 6)
        modes = [draw[3] for draw in seen[0]]
        self.assertTrue(any(modes) and not all(modes))

    def test_whole_source_share_tracks_the_requested_probability(self):
        plan = BatchPlan(radius=radius_for_stage('B'), seed=1, long_probability=0.3)
        masks = random.Random(5)
        drawn = [masks.random() < plan.long_probability for _ in range(20000)]
        self.assertAlmostEqual(sum(drawn) / len(drawn), 0.3, delta=0.02)
        with self.assertRaises(ValueError):
            BatchPlan(radius=None, seed=1, long_probability=1.4)

    def test_sample_noise_is_reproducible_from_step_and_rank_alone(self):
        self.assertEqual(sample_rng(7, 10, 2).random(), sample_rng(7, 10, 2).random())
        self.assertNotEqual(sample_rng(7, 10, 2).random(), sample_rng(7, 10, 3).random())
        self.assertNotEqual(sample_rng(7, 10, 2).random(), sample_rng(7, 11, 2).random())

    def test_resume_migrates_the_v3_policy_to_v4(self):
        sampler = BranchSampler(5)
        for _ in range(20):
            sampler.next()
        stored = sampler.state_dict()
        stored['policy'] = dict(stored['policy'], version=3, full_probability=0.25)
        with self.assertRaises(ValueError):
            BranchSampler(0).load_state_dict(stored)
        migrated = BranchSampler(0)
        migrated.load_state_dict(stored, accept_policy_change=True)
        self.assertEqual(migrated.draws, 0)
        self.assertEqual(migrated.counts, dict.fromkeys(BRANCHES, 0))
        self.assertEqual(migrated.rng.getstate(), sampler.rng.getstate())
        self.assertEqual(migrated.state_dict()['policy']['version'], 4)
        self.assertEqual(migrated.state_dict()['policy']['full_probability'], 0.20)

    def test_keep_steps_add_named_snapshots_without_replacing_latest(self):
        keep = parse_keep_steps('40000,46000')
        args = build_parser().parse_args([
            '--output', 'out', '--dataset-dir', 'data', '--keep-steps', '40000,46000'])
        self.assertEqual(args.keep_steps, keep)
        self.assertEqual(checkpoint_paths(Path('out'), 39999, keep), [Path('out/latest.pt')])
        self.assertEqual(checkpoint_paths(Path('out'), 40000, keep), [
            Path('out/latest.pt'), Path('out/step_40000.pt')])
        self.assertEqual(checkpoint_paths(Path('out'), 46000, keep), [
            Path('out/latest.pt'), Path('out/step_46000.pt')])

    def test_training_entry_accepts_only_the_ar_base(self):
        cfg, model = _tiny()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ar = root / 'ar.pt'
            from experiments.phageseek.model import PhageSeek
            ar_cfg = ModelConfig(**cfg.to_dict())
            ar_cfg.vocab_size = 6
            base = PhageSeek(ar_cfg)
            torch.save(dict(model_state=base.state_dict(), model_config=ar_cfg.to_dict()), ar)
            assert_training_source(torch.load(ar, map_location='cpu', weights_only=False), resume=False)
            adapted, _ = DiffusionPhageSeek.from_checkpoint(ar)
            restart = root / 'restart.pt'
            torch.save(dict(
                model_state=adapted.state_dict(), model_config=adapted.config.to_dict(),
                run_config=dict(recipe=RECIPE)), restart)
            assert_training_source(torch.load(restart, map_location='cpu', weights_only=False), resume=True)
            old = root / 'old.pt'
            torch.save(dict(model_state={'token_emb.weight': torch.zeros(7, cfg.dim)}, model_config=cfg.to_dict()), old)
            loaded = torch.load(old, map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint_kind(loaded['model_state']), 'old_diffusion')
            with self.assertRaises(ValueError):
                assert_training_source(loaded, resume=False)
            with self.assertRaises(ValueError):
                assert_training_source(dict(model_state=adapted.state_dict(), run_config=dict(recipe='other')), resume=True)
        self.assertIn('token_emb.base', dict(model.named_parameters()))
        self.assertEqual(RECIPE, 'cpdd_restart_v4')
        self.assertEqual(RESUMABLE, ('cpdd_restart_v3', RECIPE))


if __name__ == '__main__':
    unittest.main()
