"""Isolated masked-diffusion adaptation of pretrained PhageSeek weights."""
from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
import math
import types

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import flex_attention, _create_sparse_block_from_block_mask

from experiments.phageseek.checkpoint import load_checkpoint
from experiments.phageseek.config import ModelConfig
from experiments.phageseek.model import PhageSeek
from experiments.phageseek.modules import CausalSelfAttention, EngramMemory, apply_rope
from experiments.phadf.engram_adapt import TiedSplitEmbedding, install_diffusion_engram, load_pretrained

MASK = 6
BASES = 'ATCG'
_BLOCK_MASKS = OrderedDict()
_BLOCK_MASK_CACHE = 64
_FLEX = None


def _get_compiled_flex_attention():
    global _FLEX
    if _FLEX is None:
        _FLEX = torch.compile(flex_attention, dynamic=True)
    return _FLEX
@lru_cache(maxsize=8)
def symmetric_rule(window):
    def rule(b, h, q, k):
        return (q - k).abs() < window
    return rule


def symmetric_mask(window, n, device):
    """Use symmetric radius without a dense token-square mask."""
    rule = symmetric_rule(window)
    bs = 128
    nb = (n + bs - 1) // bs
    qb = torch.arange(nb, device=device).view(-1, 1)
    kb = torch.arange(nb, device=device).view(1, -1)
    q0, q1 = qb * bs, qb * bs + bs - 1
    k0, k1 = kb * bs, kb * bs + bs - 1
    visible = (k0 < q1 + window) & (k1 > q0 - window)
    full = (k1 < q0 + window) & (k0 > q1 - window)
    return _create_sparse_block_from_block_mask(
        ((visible & ~full)[None, None].contiguous(), full[None, None].contiguous()),
        rule, (n, n), Q_BLOCK_SIZE=bs, KV_BLOCK_SIZE=bs)


def bidirectional_forward(self, x, *, cos=None, sin=None):
    b, n, _ = x.shape
    q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
    if self.use_rope:
        if cos is None:
            cos, sin = self.rope(n, x.device)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
    if self.use_qk_norm:
        q = F.normalize(q, dim=-1) * self.q_scale.to(q.dtype)
        k = F.normalize(k, dim=-1) * self.k_scale.to(k.dtype)
    if self.swa_window is None:
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=self.kv_repeats > 1)
    else:
        key = (self.swa_window, n, str(x.device))
        if key not in _BLOCK_MASKS:
            # Window lengths cluster at the stage radius but have a long tail (~22% of
            # steps), and a block mask is only a block-level bool matrix (a few KB), so
            # keep plenty of them rather than rebuilding on every tail length.
            while len(_BLOCK_MASKS) >= _BLOCK_MASK_CACHE:
                _BLOCK_MASKS.popitem(last=False)
            # Build outside inference mode: a mask first created under inference_mode
            # holds inference tensors, and the cache would then poison later backward
            # passes. Fixed-radius windows make training and validation share one key.
            with torch.inference_mode(False):
                _BLOCK_MASKS[key] = symmetric_mask(self.swa_window, n, x.device)
        y = _get_compiled_flex_attention()(q, k, v, block_mask=_BLOCK_MASKS[key], enable_gqa=self.kv_repeats > 1)
    return self.out_proj(y.transpose(1, 2).reshape(b, n, self.heads * self.head_dim))


class DiffusionPhageSeek(PhageSeek):
    """Bidirectional adapter. Engram uses base-5 MASK digits and a centered convolution."""
    def __init__(self, config):
        config.vocab_size = 7
        super().__init__(config)
        self.token_emb = TiedSplitEmbedding(self.token_emb.weight.detach())
        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                module.forward = types.MethodType(bidirectional_forward, module)
            elif isinstance(module, EngramMemory):
                install_diffusion_engram(module)

    def logits(self, ids):
        if ids.shape[1] > self.max_seq_len:
            raise ValueError('Full source exceeds verified model context; never truncate')
        return super().forward(ids, return_value='logits')

    @classmethod
    def from_checkpoint(cls, path):
        state = load_checkpoint(path)
        config = ModelConfig.from_dict(state['model_config'])
        model = cls(config)
        load_pretrained(model, state['model_state'])
        return model, state


def encode(sequence, device):
    lookup = {'A': 1, 'T': 2, 'C': 3, 'G': 4}
    return torch.tensor([[0] + [lookup[base] for base in sequence] + [5]], dtype=torch.long, device=device)


def corrupt(clean, start, end, rng, full_probability=.25, fixed_rate=None):
    """Coordinates are zero-based DNA intervals; BOS adds one token offset."""
    if not 0 <= start < end <= clean.shape[1] - 2:
        raise ValueError('Target outside full source context')
    count = end-start
    rate = fixed_rate if fixed_rate is not None else (1.0 if rng.random() < full_probability else rng.uniform(.05, 1.0))
    chosen = [i for i in range(count) if rng.random() < rate]
    if not chosen:
        chosen = [rng.randrange(count)]
    positions = torch.tensor([start+1+i for i in chosen], dtype=torch.long, device=clean.device)
    ids = clean.clone()
    ids[:, positions] = MASK
    return ids, positions, rate


def masked_loss(model, ids, clean, positions):
    """Score each masked base from its own hidden state."""
    logits = model.logits(ids)[:, positions, 1:5].float()
    truth = clean[:, positions]-1
    loss = F.cross_entropy(logits.reshape(-1, 4), truth.reshape(-1))
    correct = (logits.argmax(-1) == truth).sum()
    return loss, correct, truth.numel()


@torch.inference_mode()
def confidence_restore(model, clean, start, end, *, steps=32, seed=0, temperature=1.0, stop_check=None, decoding='categorical'):
    """Fill entropy-ranked positions using categorical sampling or argmax."""
    if steps < 1 or temperature <= 0:
        raise ValueError('Positive steps and temperature required')
    if decoding not in ('categorical', 'argmax'):
        raise ValueError('Unknown completion decoding mode')
    if clean.ndim != 2 or clean.shape[0] != 1 or not 0 <= start < end <= clean.shape[1]-2:
        raise ValueError('Expected one source and a valid DNA target interval')
    ids = clean.clone()
    positions = torch.arange(start+1, end+1, device=ids.device)
    ids[:, positions] = MASK
    initial = positions.numel()
    generator = torch.Generator(device=ids.device).manual_seed(seed) if decoding == 'categorical' else None
    was_training = model.training
    model.eval()
    try:
        for step in range(1, min(steps, initial)+1):
            if stop_check is not None:
                stop_check()
            logits = model.logits(ids)[0, positions, 1:5].float() / temperature
            probs = logits.softmax(-1)
            if not torch.isfinite(probs).all():
                raise FloatingPointError('Nonfinite generation probabilities')
            entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1)
            remaining = math.floor(initial * (1-step/min(steps, initial)))
            number = positions.numel()-remaining
            selected = entropy.argsort(stable=True)[:number]
            sampled = (logits[selected].argmax(-1)+1 if decoding == 'argmax' else
                       torch.multinomial(probs[selected], 1, generator=generator).squeeze(-1)+1)
            ids[0, positions[selected]] = sampled
            positions = positions[ids[0, positions] == MASK]
        if positions.numel():
            raise RuntimeError('Sampler left unfilled MASK positions')
        return ids
    finally:
        model.train(was_training)
