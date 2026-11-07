"""Tests for gap placement and CDS-level validation aggregation."""
import unittest

from experiments.phadf.generation_eval import CONDITIONS, SEEDS, aggregate, gap_interval, per_cds, tasks

GAP = CONDITIONS[2][0]


def _run(target, stop, seed=SEEDS[0], condition=GAP):
    return dict(target_id=target, condition=condition, seed=seed, internal_stop=stop,
                aa_identity=0.9, hole_nt_accuracy=0.3, distinct_codons=50, top_codon_share=0.1)


class GapTests(unittest.TestCase):
    def test_gap_is_codon_aligned_and_centred_in_coding_direction(self):
        start, end = 100, 400
        forward = gap_interval(start, end, 1, 24)
        reverse = gap_interval(start, end, -1, 24)
        self.assertEqual(forward[1] - forward[0], 24)
        self.assertEqual((forward[0] - start) % 3, 0)
        self.assertEqual(end - reverse[1], forward[0] - start)
        self.assertTrue(start < forward[0] and forward[1] < end)

    def test_gap_must_leave_start_and_stop_visible(self):
        with self.assertRaises(ValueError):
            gap_interval(0, 30, 1, 25)
        with self.assertRaises(ValueError):
            gap_interval(0, 30, 1, 27)


class TaskTests(unittest.TestCase):
    def test_seeds_apply_to_sampled_conditions_only(self):
        listed = tasks([dict(target_id='a')])
        expected = sum(len(seeds) for *_, seeds in CONDITIONS)
        self.assertEqual(len(listed), expected)
        argmax = [task for task in listed if task['decoding'] == 'argmax']
        self.assertEqual([task['seed'] for task in argmax], [SEEDS[0]])
        self.assertFalse(any(name.startswith('whole_categorical') for name, *_ in CONDITIONS))


class AggregateTests(unittest.TestCase):
    def test_rates_average_seeds_inside_a_cds_first(self):
        runs = [_run('a', True, seed) for seed in SEEDS] + [_run('b', False)]
        self.assertEqual(per_cds(runs, GAP, 'internal_stop'), {'a': 1.0, 'b': 0.0})
        summary = aggregate(runs)[GAP]
        self.assertEqual(summary['n_cds'], 2)
        self.assertEqual(summary['n_runs'], 5)
        self.assertAlmostEqual(summary['internal_stop_rate'], 0.5)


if __name__ == '__main__':
    unittest.main()
