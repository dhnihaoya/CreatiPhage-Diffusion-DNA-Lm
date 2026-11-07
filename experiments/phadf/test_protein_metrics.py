"""CPU checks for protein-level scoring."""
import unittest

from Bio.Seq import Seq

from experiments.phadf.protein_metrics import (
    aggregate, global_identity, orf_report, score_generated, translate)

# start M, two internal residues, terminal stop
GENE = 'ATGCGTTACGGGCCATTTAAATAA'


class TranslationTests(unittest.TestCase):
    def test_translate_matches_biopython_and_rejects_ragged_length(self):
        self.assertEqual(translate(GENE, 11), str(Seq(GENE).translate(table=11)))
        self.assertIsNone(translate(GENE[:-1], 11))


class OrfTests(unittest.TestCase):
    def test_intact_gene_passes_every_check(self):
        report = orf_report(GENE, 11)
        self.assertTrue(report['codon_aligned'])
        self.assertTrue(report['start_codon'])
        self.assertTrue(report['terminal_stop'])
        self.assertFalse(report['internal_stop'])
        self.assertTrue(report['intact'])

    def test_missing_start_and_internal_stop_are_caught(self):
        self.assertFalse(orf_report('CCC' + GENE[3:], 11)['start_codon'])
        broken = GENE[:6] + 'TAA' + GENE[9:]
        self.assertTrue(orf_report(broken, 11)['internal_stop'])
        self.assertFalse(orf_report(broken, 11)['intact'])

    def test_ragged_length_is_not_intact(self):
        report = orf_report(GENE[:-1], 11)
        self.assertFalse(report['codon_aligned'])
        self.assertFalse(report['intact'])

    def test_tga_is_a_stop_under_code_11_but_tryptophan_under_code_4(self):
        gene = 'ATGCGTTGAGGGCCATTTAAATAA'
        self.assertTrue(orf_report(gene, 11)['internal_stop'])
        self.assertFalse(orf_report(gene, 4)['internal_stop'])


class IdentityTests(unittest.TestCase):
    def test_identical_and_disjoint_sequences(self):
        self.assertAlmostEqual(global_identity('MRYG', 'MRYG'), 1.0)
        self.assertLess(global_identity('MRYG', 'WWWW'), 0.3)

    def test_identity_is_reported_over_alignment_length(self):
        self.assertAlmostEqual(global_identity('MRY', 'MRY'), 1.0)
        self.assertLessEqual(global_identity('MRY', 'MRYGGG'), 0.5 + 1e-9)


class ScoringTests(unittest.TestCase):
    def test_synonymous_substitution_keeps_protein_but_lowers_nt(self):
        # CGT -> CGC is still arginine, so AA identity must stay perfect.
        synonymous = 'ATGCGCTACGGGCCATTTAAATAA'
        row = score_generated(synonymous, GENE, 11)
        self.assertLess(row['nt_accuracy'], 1.0)
        self.assertAlmostEqual(row['aa_identity'], 1.0)
        self.assertTrue(row['aa_exact'])

    def test_perfect_reconstruction(self):
        row = score_generated(GENE, GENE, 11)
        self.assertAlmostEqual(row['nt_accuracy'], 1.0)
        self.assertAlmostEqual(row['aa_identity'], 1.0)
        self.assertTrue(row['orf']['intact'])

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            score_generated(GENE[:-3], GENE, 11)

    def test_aggregate_weights_by_residues_and_bases(self):
        rows = [score_generated(GENE, GENE, 11),
                score_generated('ATGCGCTACGGGCCATTTAAATAA', GENE, 11)]
        summary = aggregate(rows)
        self.assertEqual(summary['examples'], 2)
        self.assertAlmostEqual(summary['aa_identity'], 1.0)
        self.assertAlmostEqual(summary['orf_intact_rate'], 1.0)
        self.assertLess(summary['nt_accuracy'], 1.0)


if __name__ == '__main__':
    unittest.main()
