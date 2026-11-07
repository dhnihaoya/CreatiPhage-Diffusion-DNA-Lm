"""Short-then-long windows that supervise every non-overlapping CDS inside the crop.

Two properties matter for gradient quality and are enforced here:
  1. one mask rate per optimizer update (passed in, never drawn per sample);
  2. a roughly constant number of supervised bases per sample, so the gradient scale
     does not swing with however many CDS happen to fall inside the window.
"""
from __future__ import annotations

import random

import torch

from experiments.phadf.model import corrupt, encode
from experiments.phadf.data import window_bounds
from experiments.phadf.corruption import BRANCHES, apply_substitution, corrupt_branch

STAGE_RADIUS = {'A': 5120, 'B': 16384, 'C': None}
# Maximum corrupted share of each window; the absolute base budget may be lower.
MASK_FRACTION = 0.25
# Supervised bases and target count per example.
SUPERVISION_BUDGET = 7000
MAX_TARGETS = 8
# Visible sequence retained between independently masked CDS targets.
MIN_VISIBLE_GAP = 300


def radius_for_stage(stage):
    try:
        return STAGE_RADIUS[stage]
    except KeyError:
        raise ValueError(f'Stage must be one of {sorted(STAGE_RADIUS)}') from None


def select_nonoverlapping(primary, candidates, rng, rate=1.0, budget=None,
                          max_targets=MAX_TARGETS, min_gap=MIN_VISIBLE_GAP):
    """Keep the primary CDS, then add non-overlapping CDS until the masked-base budget is met.

    Candidates are visited in RANDOM order. Taking them in coordinate order masked a run
    of adjacent genes, so the ones in the middle lost the neighbours they were supposed to
    be reconstructed from; a scattered selection leaves every masked gene with visible
    flanks, which is what the evaluation setting looks like.

    The budget counts EXPECTED masked bases (length * rate), so a low-rate update pulls in
    more genes and a full-mask update may use only the primary one. That keeps the
    supervised-token count per sample roughly constant across rates and branches.
    """
    chosen = [primary]
    expected = (primary['target_end'] - primary['target_start']) * rate
    pool = [row for row in candidates if row['target_id'] != primary['target_id']]
    pool.sort(key=lambda row: (row['target_start'], row['target_end'], row['target_id']))
    rng.shuffle(pool)
    for row in pool:
        if budget is not None and expected >= budget:
            break
        if len(chosen) >= max_targets:
            break
        start, end = row['target_start'], row['target_end']
        # Flank expansion can add up to 100 bp per side later, so the gap is checked
        # against the raw CDS bounds with that allowance already inside min_gap.
        if all(end + min_gap <= item['target_start'] or start >= item['target_end'] + min_gap
               for item in chosen):
            chosen.append(row)
            expected += (end - start) * rate
    chosen.sort(key=lambda row: (row['target_start'], row['target_end'], row['target_id']))
    return chosen


def _expand(start, end, length, rng):
    left, right = rng.randint(0, 100), rng.randint(0, 100)
    return max(0, start - left), min(length, end + right)


def supervised_intervals(selected, window_start, window_end, source_length, rng, branch):
    """Full-mask draws sit on exact CDS boundaries half the time. Partial draws keep flank expansion."""
    intervals = []
    for index, row in enumerate(selected):
        if index == 0:
            left_limit = window_start
        else:
            left_limit = (selected[index - 1]['target_end'] + row['target_start']) // 2
        if index + 1 == len(selected):
            right_limit = window_end
        else:
            right_limit = (row['target_end'] + selected[index + 1]['target_start']) // 2
        if branch == 'full' and rng.random() < 0.5:
            start, end, mode = row['target_start'], row['target_end'], 'exact_boundary'
        else:
            start, end = _expand(row['target_start'], row['target_end'], source_length, rng)
            mode = 'flank'
        start, end = max(start, left_limit, window_start), min(end, right_limit, window_end)
        if end <= start:
            raise ValueError('Expansion left no supervised bases')
        intervals.append(dict(
            target_id=row['target_id'], target_start=row['target_start'], target_end=row['target_end'],
            strand=row.get('strand'), genetic_code=row.get('genetic_code'),
            target_kind=row.get('target_kind'),
            expanded_start=start, expanded_end=end, mode=mode))
    return intervals


def supervision_budget(window_tokens, budget=SUPERVISION_BUDGET, mask_fraction=MASK_FRACTION):
    """Absolute budget, capped so a short window can never be mostly hidden.

    The absolute term is what holds supervision-per-step steady as the window grows; the
    fraction is only a guard, because 7,000 bases out of a 10,242-token stage-A crop
    would leave almost no visible context to condition on.
    """
    if budget < 1:
        raise ValueError('Supervision budget must be positive')
    if not 0 < mask_fraction < 1:
        raise ValueError('Mask fraction must be in (0, 1)')
    return max(1, min(int(budget), int(window_tokens * mask_fraction)))


def build_supervised_window(sequence, primary, candidates, radius, rng, branch, rate,
                            mask_fraction=MASK_FRACTION, min_gap=MIN_VISIBLE_GAP,
                            substitution=0.0, budget=SUPERVISION_BUDGET):
    """Corrupt every selected CDS in one cropped forward. Loss positions are the corrupted bases.

    "Corrupted" is not the same as "masked": a share of the positions carries a
    substituted base instead, and those are supervised too. That is what gives the model
    a wrong visible token to revise, which plain masked diffusion never provides.
    """
    if branch not in BRANCHES:
        raise ValueError('Unknown training branch')
    if not 0 < rate <= 1:
        raise ValueError('Mask rate must be in (0, 1]')
    window_start, window_end = window_bounds(len(sequence), primary['target_start'], primary['target_end'], radius)
    budget = supervision_budget(window_end - window_start, budget, mask_fraction)
    inside = [row for row in candidates
              if window_start <= row['target_start'] and row['target_end'] <= window_end]
    if not any(row['target_id'] == primary['target_id'] for row in inside):
        inside.append(primary)
    selected = select_nonoverlapping(primary, inside, rng, rate=rate, budget=budget,
                                     min_gap=min_gap)
    intervals = supervised_intervals(selected, window_start, window_end, len(sequence), rng, branch)
    clean = encode(sequence[window_start:window_end], torch.device('cpu'))
    ids = clean.clone()
    pieces, substituted, agreeing = [], 0, 0
    for item in intervals:
        rel_start = item['expanded_start'] - window_start
        rel_end = item['expanded_end'] - window_start
        if branch == 'partial' and rel_end - rel_start < 2:
            corrupted, positions, used = corrupt(clean, rel_start, rel_end, rng, fixed_rate=1.0)
            noise = dict(substituted=0, substituted_correct=0)
            noise['substituted'], noise['substituted_correct'] = apply_substitution(
                corrupted, clean, positions, rng, substitution)
        else:
            corrupted, positions, used, noise = corrupt_branch(
                clean, rel_start, rel_end, rng, branch, rate, substitution)
        ids[0, positions] = corrupted[0, positions]
        item['mask_rate'] = used
        item['window_start'] = rel_start
        item['window_end'] = rel_end
        substituted += noise['substituted']
        agreeing += noise['substituted_correct']
        pieces.append(positions)
    positions = torch.cat(pieces)
    if positions.unique().numel() != positions.numel():
        raise RuntimeError('Supervised CDS intervals overlapped')
    return dict(
        clean=clean, ids=ids, positions=positions, branch=branch, rate=rate,
        window_start=window_start, window_end=window_end, intervals=intervals,
        supervised_targets=len(selected), window_tokens=int(clean.shape[1]),
        mask_budget=budget, substitution=substitution, substituted=substituted,
        substituted_correct=agreeing)


def move_example(example, device):
    moved = dict(example)
    moved['clean'] = example['clean'].to(device)
    moved['ids'] = example['ids'].to(device)
    moved['positions'] = example['positions'].to(device)
    return moved
