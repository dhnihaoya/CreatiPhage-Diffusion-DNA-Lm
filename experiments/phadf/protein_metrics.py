"""Protein identity, ORF integrity, and codon-diversity metrics for restored CDS."""
from __future__ import annotations

from collections import Counter

from Bio.Data import CodonTable
from Bio.Seq import Seq

GAP_OPEN = -11
GAP_EXTEND = -1


def translate(dna, code):
    """Translate a coding-direction CDS; returns None when it is not codon aligned."""
    if len(dna) % 3:
        return None
    return str(Seq(dna).translate(table=code, to_stop=False))


def orf_report(dna, code):
    """Structural legality of a generated CDS, independent of whether it matches the truth."""
    table = CodonTable.unambiguous_dna_by_id[code]
    report = dict(length=len(dna), codon_aligned=len(dna) % 3 == 0)
    if not report['codon_aligned'] or len(dna) < 6:
        return {**report, 'start_codon': False, 'terminal_stop': False,
                'internal_stop': None, 'intact': False}
    protein = translate(dna, code)
    report['start_codon'] = dna[:3] in table.start_codons
    report['terminal_stop'] = dna[-3:] in table.stop_codons
    report['internal_stop'] = '*' in protein[:-1]
    report['intact'] = bool(report['start_codon'] and report['terminal_stop']
                            and not report['internal_stop'])
    return report


def global_identity(query, reference):
    """Needleman-Wunsch identity with affine gaps, reported over the alignment length."""
    if not query or not reference:
        return 0.0
    try:
        from Bio import Align
    except ImportError:  # pragma: no cover - Biopython always ships Align
        raise
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.open_gap_score = GAP_OPEN
    aligner.extend_gap_score = GAP_EXTEND
    try:
        aligner.substitution_matrix = Align.substitution_matrices.load('BLOSUM62')
    except Exception:
        aligner.match_score, aligner.mismatch_score = 1, -1
    alignment = aligner.align(query, reference)[0]
    top, bottom = alignment[0], alignment[1]
    matches = sum(1 for a, b in zip(top, bottom) if a == b and a != '-')
    return matches / max(len(top), 1)


def codon_diversity(dna):
    """Return codon count, distinct-codon count, and dominant-codon share."""
    codons = [dna[index:index + 3] for index in range(0, len(dna) - len(dna) % 3, 3)]
    if not codons:
        return dict(codons=0, distinct_codons=0, top_codon_share=None)
    counts = Counter(codons)
    return dict(codons=len(codons), distinct_codons=len(counts),
                top_codon_share=counts.most_common(1)[0][1] / len(codons))


def score_generated(generated_dna, reference_dna, code):
    """Compare one generated CDS against its reference at nt, AA, and ORF level."""
    if len(generated_dna) != len(reference_dna):
        raise ValueError('Generated and reference CDS must share a length')
    nt_matches = sum(1 for a, b in zip(generated_dna, reference_dna) if a == b)
    generated_protein = translate(generated_dna, code)
    reference_protein = translate(reference_dna, code)
    row = dict(bases=len(reference_dna), nt_matches=nt_matches,
               nt_accuracy=nt_matches / len(reference_dna),
               orf=orf_report(generated_dna, code),
               diversity=codon_diversity(generated_dna),
               reference_diversity=codon_diversity(reference_dna))
    if generated_protein is None or reference_protein is None:
        row.update(aa_identity=None, aa_exact=None, residues=0)
        return row
    # Drop the terminal stop before comparing residues.
    query = generated_protein[:-1] if generated_protein.endswith('*') else generated_protein
    target = reference_protein[:-1] if reference_protein.endswith('*') else reference_protein
    row.update(aa_identity=global_identity(query, target),
               aa_exact=query == target, residues=len(target))
    return row


def aggregate(rows):
    scored = [row for row in rows if row.get('aa_identity') is not None]
    bases = sum(row['bases'] for row in rows) or 1
    residues = sum(row['residues'] for row in scored) or 1
    diverse = [row for row in rows if row.get('diversity', {}).get('codons')]
    rolled = dict(
        examples=len(rows),
        nt_accuracy=sum(row['nt_matches'] for row in rows) / bases,
        aa_identity=sum(row['aa_identity'] * row['residues'] for row in scored) / residues,
        aa_exact_rate=sum(1 for row in scored if row['aa_exact']) / max(len(scored), 1),
        orf_intact_rate=sum(1 for row in rows if row['orf']['intact']) / max(len(rows), 1),
        start_codon_rate=sum(1 for row in rows if row['orf']['start_codon']) / max(len(rows), 1),
        terminal_stop_rate=sum(1 for row in rows if row['orf']['terminal_stop']) / max(len(rows), 1),
        internal_stop_rate=sum(1 for row in rows if row['orf'].get('internal_stop')) / max(len(rows), 1))
    if diverse:
        # Reported next to orf_intact_rate on purpose: a collapsed reconstruction scores
        # well on ORF integrity and must not be readable as a structural capability.
        rolled.update(
            distinct_codons=sum(row['diversity']['distinct_codons'] for row in diverse) / len(diverse),
            top_codon_share=sum(row['diversity']['top_codon_share'] for row in diverse) / len(diverse),
            reference_distinct_codons=sum(row['reference_diversity']['distinct_codons']
                                          for row in diverse) / len(diverse))
    return rolled
