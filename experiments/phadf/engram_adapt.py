"""Diffusion-only Engram: MASK is its own base-5 digit, and the short conv is bidirectional.

The causal Engram in ``experiments.phageseek.modules`` stays unchanged. AR checkpoints
are migrated when a diffusion model is constructed from them.
"""
from __future__ import annotations

import types

import torch
from torch import nn

from experiments.phageseek.modules import RMSNorm

MASK = 6
MASK_DIGIT = 4
KERNEL = 7
ROW_SEED = 20261104


def init_rows(rows, dim, seed, dtype, device):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(int(seed))
    return (torch.randn(rows, dim, generator=generator) * 0.02).to(device=device, dtype=dtype)


def transplant_conv_weight(causal):
    """Place the 4 causal taps on the current base and its left context; zero the future."""
    if causal.shape[-1] != 4:
        raise ValueError('Expected the AR causal kernel of 4 taps')
    out = causal.new_zeros(*causal.shape[:-1], KERNEL)
    out[..., :4].copy_(causal)
    return out


def build_luts(order):
    """Map a base-5 k-mer code to (group, local row). Group 0 is ATCG, 1 is MASK, 2 is OOV."""
    order = int(order)
    size = 5 ** order
    group = torch.empty(size + 1, dtype=torch.long)
    local = torch.empty(size + 1, dtype=torch.long)
    mask_count = 0
    for code in range(size):
        digits = []
        value = code
        for _ in range(order):
            digits.append(value % 5)
            value //= 5
        if any(digit == MASK_DIGIT for digit in digits):
            group[code] = 1
            local[code] = mask_count
            mask_count += 1
        else:
            group[code] = 0
            acc = 0
            place = 1
            for digit in digits:
                acc += digit * place
                place *= 4
            local[code] = acc
    if mask_count != size - 4 ** order:
        raise RuntimeError('MASK-containing k-mer count does not match 5**k - 4**k')
    group[size] = 2
    local[size] = 0
    return group, local


class GatherSplit(torch.autograd.Function):
    """Gather ATCG, MASK-containing, or OOV rows without leaking gradients across groups."""

    @staticmethod
    def forward(ctx, atcg, mask, oov, group, local):
        ctx.save_for_backward(group, local)
        ctx.n_atcg = atcg.shape[0]
        ctx.n_mask = mask.shape[0]
        out = atcg.new_empty(*group.shape, atcg.shape[-1])
        flat = out.view(-1, atcg.shape[-1])
        groups = group.reshape(-1)
        rows = local.reshape(-1)
        for code, table in ((0, atcg), (1, mask)):
            chosen = groups == code
            if bool(chosen.any()):
                flat[chosen] = table.index_select(0, rows[chosen])
        chosen = groups == 2
        if bool(chosen.any()):
            flat[chosen] = oov.expand(int(chosen.sum()), -1)
        return out

    @staticmethod
    def backward(ctx, grad):
        group, local = ctx.saved_tensors
        flat = grad.reshape(-1, grad.shape[-1])
        groups = group.reshape(-1)
        rows = local.reshape(-1)
        atcg = flat.new_zeros(ctx.n_atcg, flat.shape[-1])
        mask = flat.new_zeros(ctx.n_mask, flat.shape[-1])
        oov = flat.new_zeros(1, flat.shape[-1])
        chosen = groups == 0
        if bool(chosen.any()):
            atcg.index_add_(0, rows[chosen], flat[chosen])
        chosen = groups == 1
        if bool(chosen.any()):
            mask.index_add_(0, rows[chosen], flat[chosen])
        chosen = groups == 2
        if bool(chosen.any()):
            oov.add_(flat[chosen].sum(0, keepdim=True))
        return atcg, mask, oov, None, None


class DirectBase5Table(nn.Module):
    """Exact k-mer table. ATCG codes keep the AR rows; codes containing MASK are new."""

    def __init__(self, order, dim, old_weight):
        super().__init__()
        self.order = int(order)
        pure = 4 ** self.order
        if tuple(old_weight.shape) != (pure + 1, dim):
            raise ValueError(f'Base-4 table for k={self.order} has shape {tuple(old_weight.shape)}')
        self.atcg = nn.Parameter(old_weight[:pure].detach().clone())
        self.mask_rows = nn.Parameter(init_rows(
            5 ** self.order - pure, dim, ROW_SEED + self.order * 17 + 1, old_weight.dtype, old_weight.device))
        self.oov = nn.Parameter(init_rows(
            1, dim, ROW_SEED + self.order * 17 + 2, old_weight.dtype, old_weight.device))
        group, local = build_luts(self.order)
        self.register_buffer('group', group, persistent=False)
        self.register_buffer('local', local, persistent=False)

    def forward(self, idx):
        return GatherSplit.apply(self.atcg, self.mask_rows, self.oov, self.group[idx], self.local[idx])


class DepthwiseBidirectionalConv(nn.Module):
    """Depthwise conv with a centered kernel. Kernel 7 and dilation 12 see ±36 bp."""

    def __init__(self, dim, *, kernel_size=KERNEL, dilation=12):
        super().__init__()
        if int(kernel_size) % 2 != 1:
            raise ValueError('Bidirectional kernel needs a center tap')
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.radius = (self.kernel_size // 2) * self.dilation
        self.history_size = self.radius
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(
            dim, dim, self.kernel_size, groups=dim, dilation=self.dilation,
            padding=self.radius, bias=False)
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        time = x.shape[1]
        y = self.conv(self.norm(x).transpose(1, 2))
        if y.shape[-1] != time:
            raise RuntimeError('Bidirectional conv changed the sequence length')
        return self.act(y).transpose(1, 2).contiguous()


class TiedSplitEmbedding(nn.Module):
    """Tied input/output embedding with a separate MASK row for its own learning rate."""

    def __init__(self, weight):
        super().__init__()
        if weight.shape[0] != 7:
            raise ValueError('Diffusion embedding has 7 rows')
        self.base = nn.Parameter(weight[:-1].detach().clone())
        self.mask = nn.Parameter(weight[-1:].detach().clone())

    @property
    def weight(self):
        return torch.cat((self.base, self.mask), 0)

    def forward(self, ids):
        return nn.functional.embedding(ids, self.weight)


def radix5_ngram(module, ids, order):
    """Base-5 code of the causal k-mer ending at each position. OOV is 5**order."""
    acc = torch.zeros(ids.shape, device=ids.device, dtype=torch.long)
    oov = torch.zeros(ids.shape, device=ids.device, dtype=torch.bool)
    place = 1
    lut = module.base_lut
    for lag in range(order):
        digit = lut[module._shift_ids(ids, lag)]
        bad = digit < 0
        oov = oov | bad
        acc = acc + torch.where(bad, torch.zeros_like(digit), digit) * place
        place *= 5
    return torch.where(oov, torch.full_like(acc, 5 ** order), acc)


def masked_engram_forward(self, hidden_states, ids):
    """Score MASK k-mers instead of zeroing every k-mer that touches MASK."""
    pieces = []
    for oi, order in enumerate(self.ks):
        if self.mechanisms[oi] == 'direct':
            emb = self.memory_tables[oi](radix5_ngram(self, ids, order))
        else:
            hashed = self._hash_ngram(ids, oi, order)
            offsets = self.table_offsets[oi].to(ids.device).view(1, 1, -1)
            emb = self.memory_tables[oi](hashed + offsets).flatten(start_dim=-2)
        pieces.append(emb)
    memory = torch.cat(pieces, dim=-1).to(self.value_proj.weight.dtype)
    key = self.key_norm(self.key_proj(memory))
    value = self.value_proj(memory)
    query = self.query_norm(hidden_states)
    gate = (query.float() * key.float()).sum(-1) * self.inv_sqrt_dim
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated = torch.sigmoid(gate).to(dtype=value.dtype).unsqueeze(-1) * value
    return gated + self.short_conv(gated)


def install_diffusion_engram(module):
    """Replace direct tables and the causal conv. Hash tables stay, but MASK is no longer dropped."""
    if getattr(module, 'diffusion_engram', False):
        return module
    lut = torch.full((max(int(module.vocab_size), MASK + 1),), -1, dtype=torch.long)
    for digit, base in enumerate(module.dna_bases):
        lut[int(base)] = digit
    lut[MASK] = MASK_DIGIT
    module.register_buffer('base_lut', lut, persistent=False)
    module.vocab_size = int(lut.numel())
    for oi, order in enumerate(module.ks):
        if module.mechanisms[oi] != 'direct':
            continue
        old = module.memory_tables[oi]
        module.memory_tables[oi] = DirectBase5Table(order, old.embedding_dim, old.weight)
    old_conv = module.short_conv
    new_conv = DepthwiseBidirectionalConv(
        old_conv.conv.weight.shape[0], kernel_size=KERNEL, dilation=old_conv.dilation)
    with torch.no_grad():
        new_conv.norm.weight.copy_(old_conv.norm.weight)
        new_conv.conv.weight.copy_(transplant_conv_weight(old_conv.conv.weight))
    module.short_conv = new_conv
    module.forward = types.MethodType(masked_engram_forward, module)
    module.diffusion_engram = True
    return module


def checkpoint_kind(weights):
    if 'token_emb.base' in weights:
        return 'restart'
    emb = weights.get('token_emb.weight')
    if emb is None:
        raise ValueError('Checkpoint has no token embedding')
    if emb.shape[0] == 6:
        return 'ar'
    if emb.shape[0] == 7:
        return 'old_diffusion'
    raise ValueError(f'Unrecognized token embedding shape {tuple(emb.shape)}')


def _table_order(model, param_name):
    return model.get_submodule(param_name.rsplit('.', 1)[0]).order


def load_pretrained(model, weights):
    """Load an AR base by migrating it, or a restart checkpoint as-is."""
    kind = checkpoint_kind(weights)
    if kind == 'old_diffusion':
        raise ValueError(
            'Old diffusion checkpoints predate the base-5 Engram and aligned readout; '
            'restart training only accepts the AR base')
    if kind == 'restart':
        model.load_state_dict(weights, strict=True)
        return kind
    mapped = {key: value.clone() for key, value in model.state_dict().items()}
    emb = weights['token_emb.weight']
    mapped['token_emb.base'] = emb.detach().clone()
    mapped['token_emb.mask'] = emb[1:5].mean(0, keepdim=True).detach().clone()
    for name, param in model.named_parameters():
        if name in ('token_emb.base', 'token_emb.mask'):
            continue
        if name.endswith('.atcg'):
            order = _table_order(model, name)
            old = weights[name[:-len('atcg')] + 'weight']
            mapped[name] = old[: 4 ** order].detach().clone()
            continue
        if name.endswith('.mask_rows'):
            order = _table_order(model, name)
            mapped[name] = init_rows(
                param.shape[0], param.shape[1], ROW_SEED + order * 17 + 1, param.dtype, param.device)
            continue
        if name.endswith('.oov'):
            order = _table_order(model, name)
            mapped[name] = init_rows(
                1, param.shape[1], ROW_SEED + order * 17 + 2, param.dtype, param.device)
            continue
        if name.endswith('short_conv.conv.weight'):
            mapped[name] = transplant_conv_weight(weights[name])
            continue
        if name not in weights:
            raise KeyError(name)
        if tuple(weights[name].shape) != tuple(param.shape):
            raise ValueError(f'Shape mismatch for {name}')
        mapped[name] = weights[name].detach().clone()
    model.load_state_dict(mapped, strict=True)
    return kind
