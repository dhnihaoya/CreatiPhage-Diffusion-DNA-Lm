"""The one axis on which training and evaluation corruption differ, pinned down."""
import collections
import random
import unittest

from experiments.phadf.corruption import (
    BASES, SUBSTITUTION_RANGE, draw_subset, noise_count, substitute_code, substitute_letter)


class NoiseCountTests(unittest.TestCase):
    def test_count_is_fixed_rather_than_a_coin_per_position(self):
        self.assertEqual(noise_count(100, 0.3), 30)
        self.assertEqual(noise_count(19, 0.3), 6)
        self.assertEqual(noise_count(0, 0.3), 0)
        self.assertEqual(noise_count(100, 0.0), 0)
        # A nonzero share never rounds away to nothing, and never exceeds the population.
        self.assertEqual(noise_count(3, 0.01), 1)
        self.assertEqual(noise_count(3, 1.0), 3)
        with self.assertRaises(ValueError):
            noise_count(10, 1.5)

    def test_subset_is_sorted_without_replacement(self):
        drawn = draw_subset(50, 12, random.Random(4))
        self.assertEqual(len(drawn), 12)
        self.assertEqual(drawn, sorted(set(drawn)))
        self.assertTrue(all(0 <= index < 50 for index in drawn))
        self.assertEqual(draw_subset(50, 0, random.Random(4)), [])
        with self.assertRaises(ValueError):
            draw_subset(10, 11, random.Random(4))


class AlphabetTests(unittest.TestCase):
    def test_training_keeps_the_truth_in_the_alphabet_and_evaluation_does_not(self):
        """Training needs the visible-and-correct class; without it a fully masked gene
        has no correct visible base and the model learns to revise everything. Evaluation
        needs every planted edit to be a real error so its labels stay exact."""
        rng = random.Random(0)
        counts = collections.Counter(substitute_letter('A', rng, True) for _ in range(4000))
        self.assertEqual(set(counts), set(BASES))
        self.assertAlmostEqual(counts['A'] / 4000, 0.25, delta=0.03)

        rng = random.Random(0)
        strict = collections.Counter(substitute_letter('A', rng, False) for _ in range(4000))
        self.assertEqual(set(strict), set('TCG'))
        self.assertNotIn('A', strict)

    def test_code_form_round_trips_the_encoding_used_by_the_trainer(self):
        rng = random.Random(1)
        for code, letter in enumerate(BASES, start=1):
            drawn = {substitute_code(code, rng, False) for _ in range(200)}
            self.assertNotIn(code, drawn)
            self.assertTrue(drawn <= {1, 2, 3, 4})
            self.assertEqual(BASES[code - 1], letter)
        with self.assertRaises(ValueError):
            substitute_code(6, rng, True)
        with self.assertRaises(ValueError):
            substitute_letter('N', rng, True)

    def test_substitution_range_is_declared_once(self):
        from experiments.phadf import corruption
        self.assertIs(corruption.SUBSTITUTION_RANGE, SUBSTITUTION_RANGE)
        self.assertEqual(corruption.POLICY['substitution_range'], SUBSTITUTION_RANGE)


if __name__ == '__main__':
    unittest.main()
