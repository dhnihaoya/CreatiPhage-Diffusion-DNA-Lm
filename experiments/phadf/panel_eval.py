"""Single-forward nucleotide, amino-acid, ORF, and codon-diversity validation."""
from __future__ import annotations

from contextlib import nullcontext

import torch
from torch.nn import functional as F

from Bio.Seq import Seq

from experiments.phadf.amino_acid import SUPPORTED_CODES, amino_acid_loss
from experiments.phadf.model import BASES, corrupt, encode
from experiments.phadf.protein_metrics import aggregate, score_generated
from experiments.phadf.data import evaluation_rng, window_bounds

DEFAULT_RATES = (1.0, 0.5)


def _autocast(device):
    if device.type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def coding_direction(dna, strand):
    """Orient a CDS so codon 0 is the start codon."""
    if strand not in (1, -1):
        raise ValueError('Strand must be +1 or -1')
    return dna if strand == 1 else str(Seq(dna).reverse_complement())


def codon_position_counts(positions, rel_start, rel_end, strand, hit):
    """Split supervised bases by codon position, counted in the translation direction.

    A model can lift nt accuracy purely on the third position, which carries almost no
    amino-acid information, so the aggregate number is reported alongside this split.
    """
    length = rel_end - rel_start
    if length % 3 or strand not in (1, -1):
        return None
    inside = positions - (rel_start + 1)
    coding = inside if strand == 1 else (length - 1 - inside)
    slot = (coding % 3).to(torch.long)
    total = torch.bincount(slot, minlength=3)
    correct = torch.bincount(slot, weights=hit.to(torch.float64), minlength=3)
    return correct.tolist(), total.tolist()


def _as_interval(target, rel_start, rel_end):
    return dict(target_id=target['target_id'], target_start=rel_start, target_end=rel_end,
                strand=target.get('strand'), genetic_code=target.get('genetic_code'),
                target_kind=target.get('target_kind'))


def score_one(model, clean, sequence, target, rel_start, rel_end, rate, rng):
    """Mask [rel_start, rel_end) at ``rate`` and report every metric off one forward."""
    ids, positions, _ = corrupt(clean, rel_start, rel_end, rng, fixed_rate=rate)
    with _autocast(clean.device):
        row = model.logits(ids)[0, :, 1:5]
    row = row.float()
    selected = row[positions]
    truth = clean[0, positions] - 1
    hit = selected.argmax(-1) == truth
    out = dict(ce_sum=float(F.cross_entropy(selected, truth, reduction='sum')),
               correct=int(hit.sum()), bases=int(truth.numel()))

    strand, code = target.get('strand'), target.get('genetic_code')
    split = codon_position_counts(positions, rel_start, rel_end, strand, hit)
    if split is not None and target.get('target_kind') == 'whole_cds':
        out['position_correct'], out['position_bases'] = split

    example = dict(positions=positions, window_start=0, window_tokens=int(clean.shape[1]),
                   intervals=[_as_interval(target, rel_start, rel_end)])
    aa, codons = amino_acid_loss(row, clean[0], example)
    if codons:
        # amino_acid_loss normalises per base; undo it so panels of different length
        # aggregate by weight rather than by example count.
        out['aa_nll_sum'], out['aa_codons'] = float(aa) * codons * 3, codons

    if target.get('target_kind') == 'whole_cds' and code in SUPPORTED_CODES:
        filled = clean.clone()
        filled[0, positions] = selected.argmax(-1) + 1
        written = ''.join(BASES[value - 1] for value in filled[0, rel_start + 1:rel_end + 1].tolist())
        reference = sequence[rel_start:rel_end]
        if len(written) == len(reference) and len(reference) % 3 == 0:
            out['protein'] = score_generated(coding_direction(written, strand),
                                             coding_direction(reference, strand), code)
    return out


def score_panel(model, reader, rows, device, radius, rates=DEFAULT_RATES, seed=1701):
    """Score the primary CDS of every panel row, either cropped to the stage or whole."""
    was = model.training
    model.eval()
    totals = {rate: dict(ce_sum=0.0, correct=0, bases=0, aa_nll_sum=0.0, aa_codons=0,
                         position_correct=[0.0] * 3, position_bases=[0] * 3, proteins=[])
              for rate in rates}
    try:
        with torch.inference_mode():
            for row in rows:
                record = reader.read(row['target_db'], row['target_id'])
                target = record['target']
                start, end = target['target_start'], target['target_end']
                if radius is None:
                    sequence, rel_start, rel_end = record['sequence'], start, end
                else:
                    left, right = window_bounds(len(record['sequence']), start, end, radius)
                    sequence = record['sequence'][left:right]
                    rel_start, rel_end = start - left, end - left
                clean = encode(sequence, device)
                for rate in rates:
                    scored = score_one(model, clean, sequence, target, rel_start, rel_end,
                                       rate, evaluation_rng(seed, row['target_id'], rate))
                    bucket = totals[rate]
                    bucket['ce_sum'] += scored['ce_sum']
                    bucket['correct'] += scored['correct']
                    bucket['bases'] += scored['bases']
                    bucket['aa_nll_sum'] += scored.get('aa_nll_sum', 0.0)
                    bucket['aa_codons'] += scored.get('aa_codons', 0)
                    for index in range(3):
                        bucket['position_correct'][index] += scored.get('position_correct', [0] * 3)[index]
                        bucket['position_bases'][index] += scored.get('position_bases', [0] * 3)[index]
                    if 'protein' in scored:
                        bucket['proteins'].append(scored['protein'])
    finally:
        model.train(was)
    return {str(rate): summarize(bucket) for rate, bucket in totals.items()}


def summarize(bucket):
    """Panel-level means, weighted by base or residue rather than by example."""
    bases = bucket['bases'] or 1
    panel = dict(ce=bucket['ce_sum'] / bases, accuracy=bucket['correct'] / bases,
                 bases=bucket['bases'])
    if bucket['aa_codons']:
        panel['aa_nll'] = bucket['aa_nll_sum'] / (bucket['aa_codons'] * 3)
        panel['aa_codons'] = bucket['aa_codons']
    if any(bucket['position_bases']):
        panel['position_accuracy'] = [
            correct / count if count else None
            for correct, count in zip(bucket['position_correct'], bucket['position_bases'])]
    if bucket['proteins']:
        rolled = aggregate(bucket['proteins'])
        panel.update(aa_identity=rolled['aa_identity'], aa_exact_rate=rolled['aa_exact_rate'],
                     orf_intact_rate=rolled['orf_intact_rate'],
                     start_codon_rate=rolled['start_codon_rate'],
                     internal_stop_rate=rolled['internal_stop_rate'],
                     distinct_codons=rolled.get('distinct_codons'),
                     top_codon_share=rolled.get('top_codon_share'),
                     reference_distinct_codons=rolled.get('reference_distinct_codons'),
                     scored_cds=rolled['examples'])
    return panel
