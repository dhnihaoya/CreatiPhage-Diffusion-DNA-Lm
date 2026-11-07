"""Amino-acid supervision from factorized nucleotide logits.

Codon probabilities are products of three nucleotide marginals, and amino-acid
probabilities are sums over synonymous codons. Negative-strand CDSs are handled
by reversing the sequence axis and complementing the logit classes.
"""
from __future__ import annotations

from functools import lru_cache
import itertools

import torch
from torch.nn import functional as F

from Bio.Data import CodonTable

# model.encode uses A=1, T=2, C=3, G=4; masked logits are sliced to [1:5],
# so the class axis is ordered A, T, C, G.
BASE_ORDER = 'ATCG'
COMPLEMENT_PERMUTATION = (1, 0, 3, 2)  # A<->T, C<->G
CODON_COUNT = 64
SUPPORTED_CODES = (4, 11, 15)
# Internal residues only. Start and stop codons follow different distributions
# (the three tables disagree exactly on TAG/TGA), so they get their own classes.
STOP_CLASS = 'STOP'


def codon_index(triplet):
    """Base-4 index of a codon written in the A,T,C,G class order."""
    index = 0
    for base in triplet:
        index = index * 4 + BASE_ORDER.index(base)
    return index


@lru_cache(maxsize=8)
def codon_to_class(code):
    """Map each of the 64 codons to a class id, plus the ordered class names.

    Classes are the 20 amino acids present in the table, then STOP. START is tracked
    separately by the caller because it depends on position, not on the codon alone.
    """
    if code not in SUPPORTED_CODES:
        raise ValueError(f'Unsupported genetic code {code}; dataset uses {SUPPORTED_CODES}')
    table = CodonTable.unambiguous_dna_by_id[code]
    names = sorted({amino for amino in table.forward_table.values()}) + [STOP_CLASS]
    lookup = {name: position for position, name in enumerate(names)}
    mapping = torch.empty(CODON_COUNT, dtype=torch.long)
    for triplet in (''.join(x) for x in itertools.product(BASE_ORDER, repeat=3)):
        name = STOP_CLASS if triplet in table.stop_codons else table.forward_table[triplet]
        mapping[codon_index(triplet)] = lookup[name]
    return mapping, tuple(names)


@lru_cache(maxsize=8)
def class_membership(code, device_str):
    """[num_classes, 64] boolean mask of which codons realise each class."""
    mapping, names = codon_to_class(code)
    membership = torch.zeros(len(names), CODON_COUNT, dtype=torch.bool)
    membership[mapping, torch.arange(CODON_COUNT)] = True
    return membership.to(torch.device(device_str)), names


def coding_log_probs(log_probs, strand):
    """Reorient per-base log probabilities into the translation direction."""
    if strand not in (1, -1):
        raise ValueError('Strand must be +1 or -1')
    if strand == 1:
        return log_probs
    return log_probs.flip(0)[:, list(COMPLEMENT_PERMUTATION)]


def codon_log_probs(coding):
    """[n_codons, 64] log probability of every codon, from three independent marginals."""
    if coding.shape[0] % 3:
        raise ValueError('Coding region must be a whole number of codons')
    triplets = coding.view(-1, 3, 4)
    combined = (triplets[:, 0].unsqueeze(-1).unsqueeze(-1)
                + triplets[:, 1].unsqueeze(1).unsqueeze(-1)
                + triplets[:, 2].unsqueeze(1).unsqueeze(1))
    return combined.reshape(-1, CODON_COUNT)


def amino_acid_nll(coding, truth_codon_index, code, visible=None):
    """Mean -log P(true amino acid) over the supplied codons.

    ``visible`` is a [n_codons*3] boolean marking bases the model can already see; those
    positions are replaced by the one-hot truth so a codon with only one masked base is
    still scored (that is the common case in the partial branch).
    """
    if truth_codon_index.numel() * 3 != coding.shape[0]:
        raise ValueError('Codon count does not match the coding slice')
    if visible is not None:
        if visible.shape[0] != coding.shape[0]:
            raise ValueError('Visible mask must cover every coding base')
        if bool(visible.all()):
            raise ValueError('At least one base must be masked to supervise')
        truth_bases = _codon_index_to_bases(truth_codon_index, coding.device)
        one_hot = F.one_hot(truth_bases, num_classes=4).to(coding.dtype)
        floor = torch.finfo(coding.dtype).min
        known = torch.where(one_hot.bool(), torch.zeros_like(one_hot), torch.full_like(one_hot, floor))
        coding = torch.where(visible.unsqueeze(-1), known, coding)
    per_codon = codon_log_probs(coding)
    mapping, names = codon_to_class(code)
    mapping = mapping.to(coding.device)
    # Segment-logsumexp over synonymous codons via a max-shifted index_add_, which keeps
    # the intermediate at [n_codons, n_classes] instead of materialising [n, classes, 64].
    shift = per_codon.max(dim=-1, keepdim=True).values
    weights = (per_codon - shift).exp()
    pooled = torch.zeros(per_codon.shape[0], len(names), dtype=weights.dtype, device=weights.device)
    pooled = pooled.index_add(1, mapping, weights)
    class_log_probs = pooled.clamp_min(torch.finfo(coding.dtype).tiny).log() + shift
    class_log_probs = class_log_probs - torch.logsumexp(class_log_probs, dim=-1, keepdim=True)
    return F.nll_loss(class_log_probs, mapping[truth_codon_index])


def _codon_index_to_bases(indices, device):
    first = torch.div(indices, 16, rounding_mode='floor')
    second = torch.div(indices % 16, 4, rounding_mode='floor')
    third = indices % 4
    return torch.stack((first, second, third), dim=-1).reshape(-1).to(device)


def codon_indices_from_ids(ids, offset, length):
    """Truth codon indices from encoded ids (A=1..G=4) over [offset, offset+length)."""
    if length % 3:
        raise ValueError('Length must be a whole number of codons')
    bases = ids[offset:offset + length] - 1
    if bool(((bases < 0) | (bases > 3)).any()):
        raise ValueError('Coding region contains a non-base token')
    triplets = bases.view(-1, 3)
    return triplets[:, 0] * 16 + triplets[:, 1] * 4 + triplets[:, 2]


def internal_codon_span(interval, window_start, window_tokens):
    """Codon-aligned slice of one CDS inside the window, excluding start and stop codons.

    Coordinates returned are indices into the token row (BOS occupies index 0). The span
    is aligned to the CDS reading frame, which the flank expansion does not respect, and
    drops the first and last codon of a whole CDS.
    """
    start = interval['target_start'] - window_start
    end = interval['target_end'] - window_start
    if start < 0 or end > window_tokens - 2 or end - start < 9:
        return None
    if interval.get('target_kind') != 'whole_cds':
        return None
    length = end - start
    if length % 3:
        return None
    strand = interval.get('strand')
    if strand not in (1, -1):
        return None
    # Drop the terminal codon at each end: the first is a start codon and the last a stop,
    # and both have distributions unlike internal residues.
    inner_start, inner_end = start + 3, end - 3
    if inner_end - inner_start < 3:
        return None
    return inner_start + 1, inner_end + 1, strand  # +1 for BOS


def amino_acid_loss(logits, clean, example, codes=None):
    """Sum of per-CDS amino-acid NLL over the supervised whole CDS in one window.

    ``logits`` are the full-row [T, 4] log-its in A,T,C,G order; ``clean`` is the token row.
    Returns (loss, number of scored codons); the loss is normalised per base so it sits on
    the same scale as the nt cross entropy.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    masked_positions = torch.zeros(clean.shape[0], dtype=torch.bool, device=clean.device)
    masked_positions[example['positions']] = True
    total, codons = logits.new_zeros(()), 0
    for interval in example['intervals']:
        span = internal_codon_span(interval, example['window_start'], example['window_tokens'])
        if span is None:
            continue
        lo, hi, strand = span
        if not bool(masked_positions[lo:hi].any()):
            continue
        code = (codes or {}).get(interval['target_id'], interval.get('genetic_code'))
        if code not in SUPPORTED_CODES:
            continue
        coding = coding_log_probs(log_probs[lo:hi], strand)
        truth = codon_indices_from_ids(
            _oriented_ids(clean, lo, hi, strand), 0, hi - lo)
        visible = ~masked_positions[lo:hi]
        if strand == -1:
            visible = visible.flip(0)
        if bool(visible.all()):
            continue
        total = total + amino_acid_nll(coding, truth, code, visible) * truth.numel()
        codons += int(truth.numel())
    if not codons:
        return logits.new_zeros(()), 0
    return total / (codons * 3), codons


def _oriented_ids(clean, lo, hi, strand):
    """Token ids of the span in translation direction (A=1..G=4 preserved)."""
    segment = clean[lo:hi]
    if strand == 1:
        return segment
    complement = torch.tensor([0, 2, 1, 4, 3, 5, 6], device=segment.device, dtype=segment.dtype)
    return complement[segment].flip(0)
