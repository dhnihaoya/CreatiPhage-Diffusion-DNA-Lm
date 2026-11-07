"""Synchronized masking and uniform-base substitution for CPDD training and evaluation."""
from __future__ import annotations

import random

import torch

from experiments.phadf.model import corrupt

BASES = 'ATCG'
SUBSTITUTION_RANGE = (0.0, 0.3)
POLICY = dict(version=4, full_probability=0.20,
              synchronized_branches=True, synchronized_mask_rate=True,
              substitution_range=SUBSTITUTION_RANGE, synchronized_substitution=True,
              substitution_alphabet='uniform over all four bases, truth included',
              supervision='every corrupted position, MASK or substituted',
              optimizer_updates_per_batch=1)
BRANCHES = ('partial', 'full')
PARTIAL_RATE_RANGE = (0.05, 1.0)


def noise_count(total, share, minimum=1):
    """Return a fixed corruption count rather than independent Bernoulli draws."""
    if total < 0:
        raise ValueError('Cannot corrupt a negative number of positions')
    if not 0 <= share <= 1:
        raise ValueError('Noise share must lie in [0, 1]')
    if not total or not share:
        return 0
    return min(total, max(round(total * share), minimum))


def draw_subset(total, count, rng):
    """Draw sorted indices without replacement."""
    if not 0 <= count <= total:
        raise ValueError('Subset size must fit inside the population')
    if not count:
        return []
    pool = list(range(total))
    rng.shuffle(pool)
    return sorted(pool[:count])


def substitute_letter(letter, rng, include_truth):
    """Draw a replacement base with or without the original letter."""
    if letter not in BASES:
        raise ValueError(f'Cannot corrupt non-ACGT base {letter!r}')
    if include_truth:
        return rng.choice(BASES)
    return rng.choice(tuple(other for other in BASES if other != letter))


def substitute_code(code, rng, include_truth):
    """Draw a replacement for an encoded base id, where A=1, T=2, C=3, G=4."""
    if code not in (1, 2, 3, 4):
        raise ValueError(f'Cannot corrupt non-base token {code}')
    return BASES.index(substitute_letter(BASES[code - 1], rng, include_truth)) + 1


class BranchSampler:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.draws = 0
        self.counts = dict.fromkeys(BRANCHES, 0)

    def next(self):
        branch = 'full' if self.rng.random() < POLICY['full_probability'] else 'partial'
        self.draws += 1
        self.counts[branch] += 1
        return branch

    def state_dict(self):
        return dict(policy=dict(POLICY), rng=self.rng.getstate(), draws=self.draws, counts=dict(self.counts))

    def load_state_dict(self, state, accept_policy_change=False):
        """Restore the branch policy and RNG, with an explicit migration opt-out."""
        if state['draws'] != sum(state['counts'].values()):
            raise ValueError('Invalid branch history')
        if accept_policy_change:
            self.rng.setstate(state['rng'])
            self.draws, self.counts = 0, dict.fromkeys(BRANCHES, 0)
            return
        if state['policy'] != POLICY:
            raise ValueError('Training branch policy changed across resume')
        if set(state['counts']) != set(BRANCHES):
            raise ValueError('Training branch set changed across resume')
        self.rng.setstate(state['rng'])
        self.draws, self.counts = state['draws'], dict(state['counts'])


def draw_rate(branch, rng):
    """Draw one shared mask rate per optimizer update."""
    if branch not in BRANCHES:
        raise ValueError('Unknown training branch')
    return 1.0 if branch == 'full' else rng.uniform(*PARTIAL_RATE_RANGE)


def draw_substitution(rng, span=SUBSTITUTION_RANGE):
    """Draw one shared visible-substitution share per optimizer update."""
    low, high = span
    if not 0 <= low <= high <= 1:
        raise ValueError('Substitution share range must be ordered inside [0, 1]')
    return rng.uniform(low, high)


def apply_substitution(ids, clean, positions, rng, share, include_truth=True):
    """Replace a fixed share of corrupted positions with visible bases."""
    count = noise_count(positions.numel(), share)
    if not count:
        return 0, 0
    chosen = draw_subset(positions.numel(), count, rng)
    where = positions[torch.tensor(chosen, dtype=torch.long)]
    drawn = torch.tensor([substitute_code(int(clean[0, index]), rng, include_truth)
                          for index in where], dtype=ids.dtype)
    ids[0, where] = drawn
    return count, int((drawn == clean[0, where]).sum())


def corrupt_branch(clean, start, end, rng, branch, rate, substitution=0.0):
    """Apply the selected full or partial corruption branch."""
    if branch not in BRANCHES:
        raise ValueError('Unknown training branch')
    if branch == 'partial' and end-start < 2:
        raise ValueError('Partial masking requires at least two target bases')
    if branch == 'full' and rate != 1.0:
        raise ValueError('The full-mask branch must use rate 1')
    if not 0 < rate <= 1:
        raise ValueError('Mask rate must be in (0, 1]')
    ids, positions, _ = corrupt(clean, start, end, rng, fixed_rate=rate)
    if branch == 'partial' and positions.numel() == end-start:
        visible = start+1+rng.randrange(end-start)
        ids[0, visible] = clean[0, visible]
        positions = positions[positions != visible]
    substituted, agreeing = apply_substitution(ids, clean, positions, rng, substitution)
    return ids, positions, rate, dict(substituted=substituted, substituted_correct=agreeing)
