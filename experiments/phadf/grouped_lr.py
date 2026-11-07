"""Three AdamW groups with warmup, stable, decay, and stage-transition re-warmup."""
from __future__ import annotations

import math

import torch

DEFAULT_WARMUP = 500
DEFAULT_MIN_RATIO = 0.1
DEFAULT_REWARMUP_STEPS = 300
DEFAULT_REWARMUP_FLOOR = 0.5
GROUP_MULTIPLIERS = {'new': 2.0, 'adapter': 1.0, 'backbone': 0.5}


def parameter_group(name):
    """MASK-specific rows learn fastest. Attention, Engram projections, and the tied base embedding are the adapter."""
    if name.endswith('token_emb.mask') or name.endswith('.mask_rows'):
        return 'new'
    if (name.endswith('token_emb.base') or 'lm_head' in name or name.endswith('short_conv.conv.weight')
            or '.attn.' in name or '.attn_residual' in name
            or name.endswith('key_proj.weight') or name.endswith('key_proj.bias')
            or name.endswith('value_proj.weight') or name.endswith('value_proj.bias')):
        return 'adapter'
    return 'backbone'


def wsd_scale(step, warmup, decay_start=None, decay_steps=None, min_ratio=DEFAULT_MIN_RATIO):
    """Linear warmup, flat stable, then cosine decay to ``min_ratio`` once decay starts."""
    if warmup < 1 or not 0 < min_ratio < 1:
        raise ValueError('Invalid WSD schedule')
    if decay_start is not None:
        if decay_start < warmup or not decay_steps or decay_steps < 1:
            raise ValueError('Decay must start after warmup and span at least one step')
    if step < warmup:
        return (step + 1) / warmup
    if decay_start is not None and step >= decay_start:
        progress = min(1.0, (step - decay_start) / decay_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0


def rewarmup_scale(step, switch_step=None, steps=DEFAULT_REWARMUP_STEPS, floor=DEFAULT_REWARMUP_FLOOR):
    """Short ramp after a stage switch; the input distribution changes, the schedule does not."""
    if switch_step is None or steps <= 0 or step < switch_step or step >= switch_step + steps:
        return 1.0
    if not 0 < floor <= 1:
        raise ValueError('Re-warmup floor must be in (0, 1]')
    return floor + (1.0 - floor) * (step - switch_step) / steps


def scaled_lrs(step, peak, warmup=DEFAULT_WARMUP, decay_start=None, decay_steps=None,
               min_ratio=DEFAULT_MIN_RATIO, switch_step=None, rewarmup_steps=DEFAULT_REWARMUP_STEPS):
    if peak <= 0:
        raise ValueError('Peak learning rate must be positive')
    base = peak * wsd_scale(step, warmup, decay_start, decay_steps, min_ratio)
    base *= rewarmup_scale(step, switch_step, rewarmup_steps)
    return {name: multiplier * base for name, multiplier in GROUP_MULTIPLIERS.items()}


def build_optimizer(model, peak, weight_decay=0.01, warmup=DEFAULT_WARMUP):
    buckets = {name: [] for name in GROUP_MULTIPLIERS}
    for name, param in model.named_parameters():
        buckets[parameter_group(name)].append(param)
    if any(not params for params in buckets.values()):
        missing = [name for name, params in buckets.items() if not params]
        raise ValueError(f'Parameter group is empty: {missing}')
    initial = scaled_lrs(0, peak, warmup=warmup)
    groups = [{'params': buckets[name], 'lr': initial[name], 'name': name}
              for name in GROUP_MULTIPLIERS]
    return torch.optim.AdamW(groups, lr=peak, betas=(0.9, 0.95), eps=1e-8, weight_decay=weight_decay)


def apply_lrs(optimizer, step, peak, warmup=DEFAULT_WARMUP, decay_start=None, decay_steps=None,
              min_ratio=DEFAULT_MIN_RATIO, switch_step=None, rewarmup_steps=DEFAULT_REWARMUP_STEPS):
    values = scaled_lrs(step, peak, warmup, decay_start, decay_steps, min_ratio,
                        switch_step, rewarmup_steps)
    for group in optimizer.param_groups:
        group['lr'] = values[group['name']]
    return values
