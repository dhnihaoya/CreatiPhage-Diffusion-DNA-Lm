"""CPU checks for amino-acid supervision: strand handling, codon tables, boundaries."""
import itertools
import math
import unittest

import torch

from Bio.Data import CodonTable
from Bio.Seq import Seq

from experiments.phadf.amino_acid import (
    BASE_ORDER, SUPPORTED_CODES, _oriented_ids, amino_acid_loss, amino_acid_nll, codon_index,
    codon_indices_from_ids, codon_log_probs, codon_to_class, coding_log_probs,
    internal_codon_span)
from experiments.phadf.model import encode


def random_log_probs(length, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(length, 4, generator=generator, dtype=torch.float64), dim=-1)


class CodonTableTests(unittest.TestCase):
    def test_every_codon_maps_and_tables_differ_only_on_tag_tga(self):
        labels = {}
        for code in SUPPORTED_CODES:
            mapping, names = codon_to_class(code)
            self.assertEqual(mapping.shape[0], 64)
            self.assertEqual(mapping.min().item(), 0)
            self.assertLess(mapping.max().item(), len(names))
            table = CodonTable.unambiguous_dna_by_id[code]
            labels[code] = {}
            for triplet in (''.join(x) for x in itertools.product(BASE_ORDER, repeat=3)):
                expected = 'STOP' if triplet in table.stop_codons else table.forward_table[triplet]
                self.assertEqual(names[mapping[codon_index(triplet)]], expected)
                labels[code][triplet] = expected
        disagreeing = {t for t in labels[4] if len({labels[c][t] for c in SUPPORTED_CODES}) > 1}
        self.assertEqual(disagreeing, {'TAG', 'TGA'})

    def test_codon_index_round_trips_through_ids(self):
        sequence = 'ATGCGTTACGGG'
        ids = encode(sequence, torch.device('cpu'))[0]
        indices = codon_indices_from_ids(ids, 1, len(sequence))
        rebuilt = []
        for value in indices.tolist():
            rebuilt.append(''.join(BASE_ORDER[(value // 4 ** power) % 4] for power in (2, 1, 0)))
        self.assertEqual(''.join(rebuilt), sequence)

    def test_unsupported_code_is_rejected(self):
        with self.assertRaises(ValueError):
            codon_to_class(1)


class CodonProbabilityTests(unittest.TestCase):
    def test_codon_log_probs_match_the_product_of_marginals(self):
        coding = random_log_probs(6, 11)
        table = codon_log_probs(coding)
        self.assertEqual(table.shape, (2, 64))
        for position in range(2):
            for first, second, third in itertools.product(range(4), repeat=3):
                expected = (coding[position * 3 + 0, first] + coding[position * 3 + 1, second]
                            + coding[position * 3 + 2, third])
                got = table[position, first * 16 + second * 4 + third]
                self.assertAlmostEqual(float(got), float(expected), places=10)
        # Each codon row is a proper distribution over the 64 codons.
        self.assertTrue(torch.allclose(table.logsumexp(-1), torch.zeros(2, dtype=torch.float64), atol=1e-9))

    def test_amino_acid_nll_equals_manual_sum_over_synonymous_codons(self):
        coding = random_log_probs(3, 7)
        truth = torch.tensor([codon_index('CTG')])  # leucine, 6 synonymous codons
        loss = amino_acid_nll(coding, truth, 11)
        table = CodonTable.unambiguous_dna_by_id[11]
        synonyms = [c for c, a in table.forward_table.items() if a == 'L']
        self.assertEqual(len(synonyms), 6)
        per_codon = codon_log_probs(coding)[0]
        manual = torch.logsumexp(torch.stack([per_codon[codon_index(c)] for c in synonyms]), 0)
        self.assertAlmostEqual(float(loss), float(-manual), places=9)

    def test_wobble_errors_are_free_but_first_position_errors_are_not(self):
        # A distribution certain about CT? scores leucine perfectly regardless of the
        # third base, which is exactly the property nt cross entropy lacks.
        confident = torch.full((3, 4), -1e4, dtype=torch.float64)
        confident[0, BASE_ORDER.index('C')] = 0.0
        confident[1, BASE_ORDER.index('T')] = 0.0
        confident[2] = math.log(0.25)
        confident = torch.log_softmax(confident, dim=-1)
        leucine = amino_acid_nll(confident, torch.tensor([codon_index('CTG')]), 11)
        self.assertLess(float(leucine), 1e-3)
        wrong_first = amino_acid_nll(confident, torch.tensor([codon_index('ATG')]), 11)
        self.assertGreater(float(wrong_first), 5.0)


class StrandTests(unittest.TestCase):
    def test_negative_strand_logits_match_reverse_complement_of_the_sequence(self):
        forward = 'ATGCGTTACGGGTAA'
        reverse = str(Seq(forward).reverse_complement())
        log_probs = random_log_probs(len(forward), 3)
        coding = coding_log_probs(log_probs, -1)
        # The reoriented row must assign each reverse-complement base the same probability
        # the forward row gave to its complement at the mirrored position.
        for position, base in enumerate(reverse):
            mirrored = len(forward) - 1 - position
            complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}[base]
            self.assertAlmostEqual(
                float(coding[position, BASE_ORDER.index(base)]),
                float(log_probs[mirrored, BASE_ORDER.index(complement)]), places=12)

    def test_permutation_is_an_involution(self):
        log_probs = random_log_probs(12, 5)
        twice = coding_log_probs(coding_log_probs(log_probs, -1), -1)
        self.assertTrue(torch.allclose(twice, log_probs, atol=1e-12))

    def test_negative_strand_end_to_end_matches_scoring_the_revcomp_as_positive(self):
        """The real check: a minus-strand CDS scored through the forward row must equal
        scoring its reverse complement as if it were a plus-strand gene."""
        forward = 'TTACCCGTAACGCAT'            # revcomp = ATGCGTTACGGGTAA
        reverse = str(Seq(forward).reverse_complement())
        self.assertEqual(reverse, 'ATGCGTTACGGGTAA')
        log_probs = random_log_probs(len(forward), 23)

        reoriented = coding_log_probs(log_probs, -1)
        truth_reverse = codon_indices_from_ids(encode(reverse, torch.device('cpu'))[0], 1, len(reverse))
        through_minus = amino_acid_nll(reoriented, truth_reverse, 11)

        # Independent path: pretend the revcomp sequence is a plus-strand gene whose
        # per-base distributions are the reoriented ones, built by hand.
        by_hand = torch.empty_like(log_probs)
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        for position in range(len(reverse)):
            mirrored = len(forward) - 1 - position
            for base in BASE_ORDER:
                by_hand[position, BASE_ORDER.index(base)] = \
                    log_probs[mirrored, BASE_ORDER.index(complement[base])]
        as_plus = amino_acid_nll(coding_log_probs(by_hand, 1), truth_reverse, 11)
        self.assertAlmostEqual(float(through_minus), float(as_plus), places=12)

    def test_oriented_ids_reverse_complements_tokens(self):
        forward = 'TTACCCGTAACGCAT'
        row = encode(forward, torch.device('cpu'))[0]
        oriented = _oriented_ids(row, 1, 1 + len(forward), -1)
        expected = encode(str(Seq(forward).reverse_complement()), torch.device('cpu'))[0][1:-1]
        self.assertTrue(torch.equal(oriented, expected))
        self.assertTrue(torch.equal(_oriented_ids(row, 1, 1 + len(forward), 1), row[1:-1]))

    def test_positive_strand_is_untouched(self):
        log_probs = random_log_probs(9, 9)
        self.assertTrue(torch.equal(coding_log_probs(log_probs, 1), log_probs))


class VisibleBaseTests(unittest.TestCase):
    def test_visible_bases_are_pinned_to_truth(self):
        coding = random_log_probs(3, 13)
        truth = torch.tensor([codon_index('CTG')])
        # Reveal positions 1 and 2; only the first base stays masked.
        visible = torch.tensor([False, True, True])
        loss = amino_acid_nll(coding, truth, 11, visible=visible)
        # With T and G pinned, leucine is reachable only via CTG and TTG.
        per_base = coding[0]
        expected = -torch.logsumexp(torch.stack([
            per_base[BASE_ORDER.index('C')], per_base[BASE_ORDER.index('T')]]), 0)
        normaliser = torch.logsumexp(torch.stack([
            per_base[BASE_ORDER.index(b)] for b in BASE_ORDER]), 0)
        self.assertAlmostEqual(float(loss), float(expected + normaliser), places=6)

    def test_fully_visible_codon_is_rejected(self):
        coding = random_log_probs(3, 17)
        with self.assertRaises(ValueError):
            amino_acid_nll(coding, torch.tensor([codon_index('CTG')]), 11,
                           visible=torch.tensor([True, True, True]))


class EndToEndLossTests(unittest.TestCase):
    """amino_acid_loss over a whole window, including strand and visibility bookkeeping."""

    def _window(self, gene, strand, flank='GGCATTACGATTCCGA'):
        sequence = flank + gene + flank
        row = encode(sequence, torch.device('cpu'))[0]
        interval = dict(target_id='g', target_start=len(flank), target_end=len(flank) + len(gene),
                        strand=strand, genetic_code=11, target_kind='whole_cds')
        # Mask the whole CDS (token indices carry the +1 BOS offset).
        positions = torch.arange(len(flank) + 1, len(flank) + len(gene) + 1)
        example = dict(intervals=[interval], positions=positions, window_start=0,
                       window_tokens=int(row.shape[0]))
        return row, example

    def test_minus_strand_window_equals_plus_strand_window_on_the_revcomp(self):
        gene = 'ATGCGTTACGGGCCATTTAAATAA'
        self.assertEqual(len(gene) % 3, 0)
        generator = torch.Generator().manual_seed(31)
        plus_row, plus_example = self._window(gene, 1)
        logits = torch.randn(plus_row.shape[0], 4, generator=generator, dtype=torch.float64)
        plus_loss, plus_codons = amino_acid_loss(logits, plus_row, plus_example)

        # Same gene on the minus strand: the stored forward sequence is its revcomp, and
        # the logits are mirrored and complemented so the model "sees" the same evidence.
        minus_gene = str(Seq(gene).reverse_complement())
        minus_row, minus_example = self._window(
            minus_gene, -1, flank=str(Seq('GGCATTACGATTCCGA').reverse_complement()))
        mirrored = logits.flip(0)[:, list((1, 0, 3, 2))]
        minus_loss, minus_codons = amino_acid_loss(mirrored, minus_row, minus_example)

        self.assertEqual(plus_codons, minus_codons)
        self.assertGreater(plus_codons, 0)
        self.assertAlmostEqual(float(plus_loss), float(minus_loss), places=10)

    def test_loss_is_normalised_per_base_and_counts_internal_codons_only(self):
        gene = 'ATGCGTTACGGGCCATTTAAATAA'   # 8 codons; start and stop are dropped
        row, example = self._window(gene, 1)
        generator = torch.Generator().manual_seed(41)
        logits = torch.randn(row.shape[0], 4, generator=generator, dtype=torch.float64)
        loss, codons = amino_acid_loss(logits, row, example)
        self.assertEqual(codons, 6)
        self.assertTrue(math.isfinite(float(loss)))
        self.assertGreater(float(loss), 0.0)

    def test_window_with_no_masked_coding_bases_returns_zero(self):
        gene = 'ATGCGTTACGGGCCATTTAAATAA'
        row, example = self._window(gene, 1)
        example['positions'] = torch.tensor([1, 2, 3])     # only flank is masked
        generator = torch.Generator().manual_seed(43)
        logits = torch.randn(row.shape[0], 4, generator=generator, dtype=torch.float64)
        loss, codons = amino_acid_loss(logits, row, example)
        self.assertEqual(codons, 0)
        self.assertEqual(float(loss), 0.0)

    def test_partial_mask_still_scores_codons(self):
        gene = 'ATGCGTTACGGGCCATTTAAATAA'
        row, example = self._window(gene, 1)
        example['positions'] = example['positions'][::2]   # every other base masked
        generator = torch.Generator().manual_seed(47)
        logits = torch.randn(row.shape[0], 4, generator=generator, dtype=torch.float64)
        loss, codons = amino_acid_loss(logits, row, example)
        self.assertEqual(codons, 6)
        self.assertTrue(math.isfinite(float(loss)))


class SpanTests(unittest.TestCase):
    def _interval(self, **overrides):
        base = dict(target_start=10, target_end=40, strand=1, genetic_code=11,
                    target_kind='whole_cds')
        base.update(overrides)
        return base

    def test_span_drops_start_and_stop_codons_and_offsets_for_bos(self):
        span = internal_codon_span(self._interval(), window_start=0, window_tokens=100)
        self.assertEqual(span, (14, 38, 1))  # 10+3+1 .. 40-3+1
        self.assertEqual((span[1] - span[0]) % 3, 0)

    def test_window_offset_is_applied(self):
        span = internal_codon_span(self._interval(target_start=110, target_end=140),
                                   window_start=100, window_tokens=100)
        self.assertEqual(span, (14, 38, 1))

    def test_non_codon_multiple_and_partial_windows_are_skipped(self):
        self.assertIsNone(internal_codon_span(self._interval(target_end=41), 0, 100))
        self.assertIsNone(internal_codon_span(
            self._interval(target_kind='partial_cds_window'), 0, 100))

    def test_too_short_and_out_of_window_are_skipped(self):
        self.assertIsNone(internal_codon_span(self._interval(target_end=16), 0, 100))
        self.assertIsNone(internal_codon_span(self._interval(), 0, 20))
        self.assertIsNone(internal_codon_span(self._interval(strand=0), 0, 100))


if __name__ == '__main__':
    unittest.main()
