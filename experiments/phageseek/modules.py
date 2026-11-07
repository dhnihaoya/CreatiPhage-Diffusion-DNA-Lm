"""Core neural network blocks for PhageSeek."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn.attention.flex_attention import flex_attention


# Pre-compiled flex_attention singleton. The surrounding frame can fall back to
# eager at runtime (dynamo recompile limit: _run_attn_res_layer recompiles per
# blocks-tuple length/dtype signature and hits the limit of 8 in any AttnRes
# model), and EAGER flex is an unfused fallback that materialises the full
# [L, L] scores — fatal at 98k. Calling an explicitly compiled flex keeps the
# kernel fused whether the caller frame is compiled or not; under an outer
# torch.compile dynamo simply inlines it.
_compiled_flex_attention = None


def _get_compiled_flex_attention():
    global _compiled_flex_attention
    if _compiled_flex_attention is None:
        _compiled_flex_attention = torch.compile(flex_attention)
    return _compiled_flex_attention


def causal_swa_mask_mod(window: int):
    """Return the causal sliding-window predicate used by the AR backbone."""

    def mask_mod(b, h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & (q_idx - kv_idx < window)

    return mask_mod


def build_causal_swa_block_mask(window: int, n: int, device: torch.device):
    """BlockMask for a causal sliding window, built directly at BLOCK granularity.

    torch's create_block_mask evaluates mask_mod on the DENSE [L, L] index grid
    before blockifying — that is 72GiB of int64 intermediates at 98k and OOMs.
    The causal-window pattern is a simple band, so the block layout is derived
    analytically: blocks fully inside the band become "full" (no per-element
    mask), only the diagonal / window-edge blocks stay "partial" and get their
    element masks from mask_mod at kernel time. The block-bool -> BlockMask
    conversion reuses flex's own _create_sparse_block_from_block_mask (private
    API, verified against torch 2.12.1 — kv_indices must keep FULL nb columns;
    its _ordered_to_dense assumes index values < kv_indices.shape[-1], so
    trimming to the per-row max count is NOT allowed).
    """
    from torch.nn.attention.flex_attention import _create_sparse_block_from_block_mask

    BS = 128  # flex_attention._DEFAULT_SPARSE_BLOCK_SIZE
    nb = (n + BS - 1) // BS
    qb = torch.arange(nb, device=device).view(-1, 1)
    kb = torch.arange(nb, device=device).view(1, -1)
    q_min, q_max = qb * BS, qb * BS + BS - 1
    kv_min, kv_max = kb * BS, kb * BS + BS - 1
    # Element rule (must match causal_swa_mask_mod): kv <= q and kv > q - window.
    visible = (kv_min <= q_max) & (kv_max > q_min - window)  # any element visible
    full = (kv_max <= q_min) & (kv_min > q_max - window)  # every element visible
    partial = visible & ~full
    # Standard BlockMask rank is [B, H, Q_BLOCKS, KV_BLOCKS] — inductor's flex
    # template indexes sizes by absolute dim, 2-D inputs crash its codegen.
    partial = partial.view(1, 1, nb, nb).contiguous()
    full = full.view(1, 1, nb, nb).contiguous()

    return _create_sparse_block_from_block_mask(
        (partial, full),
        causal_swa_mask_mod(window),
        (n, n),
        Q_BLOCK_SIZE=BS,
        KV_BLOCK_SIZE=BS,
    )


@dataclass(slots=True)
class AttentionDecodeCache:
    """Autoregressive K/V cache for one attention layer."""

    k: torch.Tensor
    v: torch.Tensor
    length: int | None = None


@dataclass(slots=True)
class LayerDecodeCache:
    """Autoregressive decode cache for one transformer layer.

    `attn` stores past projected K/V tensors. `engram_state` stores the recent
    gated Engram values needed by the depthwise causal convolution. AttnRes
    depth-mixer state is intentionally not cached across time: it mixes only
    layer-depth states for the current token, so the step path rebuilds it for
    each generated token.
    """

    attn: AttentionDecodeCache | None = None
    engram_state: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to fp32 for the normalization statistics.  Under bf16 autocast
        # the mean/rsqrt of a 98k-long vector in bf16 loses precision; computing
        # the whole reduction in fp32 and casting back is the standard pattern.
        in_dtype = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (self.weight * out).to(in_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10_000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward_position(self, position: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        t = torch.tensor([position], device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :].to(dtype=x.dtype)
    sin = sin[None, None, :, :].to(dtype=x.dtype)
    return (x * cos) + (rotate_half(x) * sin)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _next_distinct_primes(start: int, count: int, seen: set[int]) -> list[int]:
    out: list[int] = []
    candidate = max(2, int(start))
    while len(out) < count:
        if _is_prime(candidate) and candidate not in seen:
            seen.add(candidate)
            out.append(candidate)
        candidate += 1
    return out


class DepthwiseCausalConv(nn.Module):
    """Short causal depthwise convolution used by Engram fusion."""

    def __init__(self, dim: int, *, kernel_size: int, dilation: int):
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.history_size = (self.kernel_size - 1) * self.dilation
        self.norm = RMSNorm(dim)
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=self.kernel_size,
            groups=dim,
            padding=self.history_size,
            dilation=self.dilation,
            bias=False,
        )
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time = x.shape[1]
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y)[..., :time]
        return self.act(y).transpose(1, 2).contiguous()


class EngramMemory(nn.Module):
    """DNA-specialized Engram: causal multi-head N-gram lookup plus QK/V gate.

    The NLP reference compresses tokenizer IDs before hashing. DNA tokens are
    already canonical and tiny, so this module keeps raw IDs but preserves the
    important Engram structure: per-layer deterministic multi-head hashes,
    concatenated retrieved memories, context-dependent QK gating, and a short
    causal convolution over gated values.

    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        slots: int,
        memory_dim: int,
        model_dim: int,
        ks: tuple[int, ...],
        heads_per_order: int,
        layer_idx: int,
        zero_init_residual: bool,
        direct_threshold: int = 65_536,
        dna_bases: tuple[int, ...] = (1, 2, 3, 4),
    ):
        super().__init__()
        if not ks:
            raise ValueError("Engram ks must not be empty")
        if heads_per_order <= 0:
            raise ValueError(f"heads_per_order must be positive, got {heads_per_order}")
        if memory_dim % heads_per_order != 0:
            raise ValueError(
                f"engram_dim ({memory_dim}) must be divisible by heads_per_order ({heads_per_order})"
            )

        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.ks = tuple(int(k) for k in ks)
        self.heads_per_order = int(heads_per_order)
        self.head_dim = int(memory_dim) // self.heads_per_order
        self.memory_dim = int(memory_dim)
        self.model_dim = int(model_dim)
        self.total_memory_dim = len(self.ks) * self.memory_dim
        self.inv_sqrt_dim = 1.0 / math.sqrt(self.model_dim)
        self.zero_init_residual = bool(zero_init_residual)

        # Per-k mechanism. DNA has only 4 bases, so there are 4**k distinct
        # k-mers. When that cardinality is affordable (<= direct_threshold) use an
        # EXACT direct radix table (collision-free, no dead rows, biologically a
        # codon/motif table); only fall back to the hashed multi-head count-sketch
        # table when the key space is too large to store. With ks=(3,6,12) and the
        # default threshold 65536: k=3 (64) and k=6 (4096) are direct, k=12 (16.7M)
        # is hashed. A direct table stores the full memory_dim per k-mer row; a
        # hashed table keeps the per-head distinct-prime layout (head_dim per slot).
        self.mechanisms = tuple(
            "direct" if (4 ** k) <= int(direct_threshold) else "hash" for k in self.ks
        )
        self.dna_bases = tuple(int(b) for b in dna_bases)

        slots_per_order = max(1, int(slots) // len(self.ks))
        seen_primes: set[int] = set()
        table_sizes: list[list[int]] = []
        table_offsets: list[list[int]] = []
        self.memory_tables = nn.ModuleList()
        for order_idx, k in enumerate(self.ks):
            if self.mechanisms[order_idx] == "direct":
                # +1 row for OOV (k-mers spanning pad/BOS/EOS or any non-ACGT token).
                self.memory_tables.append(nn.Embedding(4 ** k + 1, self.memory_dim))
                table_sizes.append([1] * self.heads_per_order)    # dummy, unused by direct
                table_offsets.append([0] * self.heads_per_order)  # dummy, unused by direct
            else:
                sizes = _next_distinct_primes(slots_per_order, self.heads_per_order, seen_primes)
                offsets = [0]
                for size in sizes[:-1]:
                    offsets.append(offsets[-1] + size)
                table_sizes.append(sizes)
                table_offsets.append(offsets)
                self.memory_tables.append(nn.Embedding(sum(sizes), self.head_dim))

        self.register_buffer("table_sizes", torch.tensor(table_sizes, dtype=torch.long), persistent=False)
        self.register_buffer("table_offsets", torch.tensor(table_offsets, dtype=torch.long), persistent=False)

        max_k = max(self.ks)
        generator = torch.Generator()
        generator.manual_seed(17_071 + 10_007 * int(layer_idx))
        multipliers = torch.randint(
            low=1,
            high=2**30,
            size=(len(self.ks), max_k, self.heads_per_order),
            generator=generator,
            dtype=torch.long,
        )
        multipliers = multipliers * 2 + 1
        self.register_buffer("hash_multipliers", multipliers, persistent=False)

        # Direct-radix lookup: token id -> base digit (0..len(dna_bases)-1) or -1
        # for non-bases (pad/BOS/EOS). Assumes the megaDNA tokenization (DNA bases
        # at small contiguous ids; default A=1,C=2,G=3,T=4 in the DNA encoding
        # -- note T=2,C=3, NOT alphabetical, but the digit assignment is arbitrary).
        base_lut = torch.full((self.vocab_size,), -1, dtype=torch.long)
        for digit, base_id in enumerate(self.dna_bases):
            if 0 <= base_id < self.vocab_size:
                base_lut[base_id] = digit
        self.register_buffer("base_lut", base_lut, persistent=False)

        self.key_proj = nn.Linear(self.total_memory_dim, self.model_dim, bias=True)
        self.value_proj = nn.Linear(self.total_memory_dim, self.model_dim, bias=True)
        self.query_norm = RMSNorm(self.model_dim)
        self.key_norm = RMSNorm(self.model_dim)
        self.short_conv = DepthwiseCausalConv(
            self.model_dim,
            kernel_size=4,
            dilation=max(self.ks),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for table in self.memory_tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.key_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.key_proj.bias)
        if self.zero_init_residual:
            nn.init.zeros_(self.value_proj.weight)
        else:
            nn.init.normal_(self.value_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.value_proj.bias)

    def _shift_ids(self, ids: torch.Tensor, lag: int) -> torch.Tensor:
        if lag == 0:
            return ids
        shifted = torch.full_like(ids, self.pad_id)
        if lag < ids.shape[1]:
            shifted[:, lag:] = ids[:, : ids.shape[1] - lag]
        return shifted

    def _hash_ngram(self, ids: torch.Tensor, order_idx: int, order: int) -> torch.Tensor:
        b, t = ids.shape
        mix = torch.zeros((b, t, self.heads_per_order), device=ids.device, dtype=torch.long)
        multipliers = self.hash_multipliers[order_idx]
        for lag in range(order):
            token = self._shift_ids(ids, lag).long() + 1
            mix = torch.bitwise_xor(mix, token.unsqueeze(-1) * multipliers[lag].view(1, 1, -1))
        sizes = self.table_sizes[order_idx].to(device=ids.device).view(1, 1, -1)
        return torch.remainder(mix, sizes)

    def _radix_ngram(self, ids: torch.Tensor, order: int) -> torch.Tensor:
        """Direct base-4 radix index of the k-mer ending at each position (causal).

        k-mer = (ids[p], ids[p-1], ..., ids[p-order+1]); any non-base token (pad,
        BOS, EOS) or a position before the sequence start marks the whole k-mer as
        OOV -> index 4**order (the shared last row). Returns [b, t] in [0, 4**k].
        """
        b, t = ids.shape
        acc = torch.zeros((b, t), device=ids.device, dtype=torch.long)
        oov = torch.zeros((b, t), device=ids.device, dtype=torch.bool)
        place = 1  # 4 ** lag
        base_lut = self.base_lut.to(ids.device)
        for lag in range(order):
            tok = self._shift_ids(ids, lag)
            digit = base_lut[tok]                      # -1 for non-base, else 0..3
            is_base = digit >= 0
            oov = oov | ~is_base
            acc = acc + torch.where(is_base, digit, torch.zeros_like(digit)) * place
            place *= 4
        return torch.where(oov, torch.full_like(acc, 4 ** order), acc)

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        pieces = []
        for order_idx, order in enumerate(self.ks):
            if self.mechanisms[order_idx] == "direct":
                idx = self._radix_ngram(ids, order)
                emb = self.memory_tables[order_idx](idx)            # [b, t, memory_dim]
            else:
                hashed = self._hash_ngram(ids, order_idx, order)
                offsets = self.table_offsets[order_idx].to(device=ids.device).view(1, 1, -1)
                emb = self.memory_tables[order_idx](hashed + offsets).flatten(start_dim=-2)
            pieces.append(emb)
        return torch.cat(pieces, dim=-1).to(dtype=self.value_proj.weight.dtype)

    def forward(self, hidden_states: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        memory = self.lookup(ids)
        key = self.key_proj(memory)
        value = self.value_proj(memory)
        query = self.query_norm(hidden_states)
        key = self.key_norm(key)
        gate_logits = (query.float() * key.float()).sum(dim=-1) * self.inv_sqrt_dim
        gate_logits = gate_logits.abs().clamp_min(1e-6).sqrt() * gate_logits.sign()
        gated = torch.sigmoid(gate_logits).to(dtype=value.dtype).unsqueeze(-1) * value
        return gated + self.short_conv(gated)

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        ids: torch.Tensor,
        state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode one token exactly, including the Engram causal conv state.

        `ids` is the full generated prefix including the current token. Lookup
        only needs the last max(k) token ids; the short conv only needs the last
        `(kernel_size - 1) * dilation` gated values.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
            raise ValueError(f"Engram forward_step expects [B, 1, D], got {tuple(hidden_states.shape)}")
        if ids.ndim != 2:
            raise ValueError(f"Engram forward_step expects ids [B, T], got {tuple(ids.shape)}")

        tail_ids = ids[:, -max(self.ks) :]
        memory = self.lookup(tail_ids)[:, -1:]
        key = self.key_proj(memory)
        value = self.value_proj(memory)
        query = self.query_norm(hidden_states)
        key = self.key_norm(key)
        gate_logits = (query.float() * key.float()).sum(dim=-1) * self.inv_sqrt_dim
        gate_logits = gate_logits.abs().clamp_min(1e-6).sqrt() * gate_logits.sign()
        gated = torch.sigmoid(gate_logits).to(dtype=value.dtype).unsqueeze(-1) * value

        if state is None:
            conv_input = gated
        else:
            conv_input = torch.cat((state.to(device=gated.device, dtype=gated.dtype), gated), dim=1)
        conv_out = self.short_conv(conv_input)[:, -1:]

        history = self.short_conv.history_size
        if history > 0:
            next_state = conv_input[:, -history:].detach()
        else:
            next_state = None
        return gated + conv_out, next_state


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        kv_heads: int,
        dropout: float,
        rope_theta: float,
        use_qk_norm: bool,
        zero_init_residual: bool,
        use_rope: bool = True,
        swa_window: int | None = None,
    ):
        super().__init__()
        if heads % kv_heads != 0:
            raise ValueError(f"heads ({heads}) must be divisible by kv_heads ({kv_heads})")
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads})")

        self.dim = dim
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads
        self.dropout = dropout
        self.use_qk_norm = use_qk_norm
        self.kv_repeats = heads // kv_heads
        # Sliding window: None = full-sequence (the only behaviour in old
        # checkpoints); W = causal banded attention over the most recent W
        # tokens via FlexAttention. Parameterless.
        self.swa_window = swa_window
        # BlockMask cache keyed by (seq_len, device); plain dict, not a module
        # buffer — it never enters state_dict and old checkpoints are untouched.
        self._swa_block_masks: dict = {}

        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        # NoPE runs (use_rope=False) skip RoPE entirely; self.rope stays None.
        self.use_rope = use_rope
        self.rope = RotaryEmbedding(self.head_dim, rope_theta) if use_rope else None

        if use_qk_norm:
            scale = math.sqrt(self.head_dim)
            self.q_scale = nn.Parameter(torch.full((1, heads, 1, 1), scale))
            self.k_scale = nn.Parameter(torch.full((1, kv_heads, 1, 1), scale))
        else:
            self.q_scale = None
            self.k_scale = None

        if zero_init_residual:
            nn.init.zeros_(self.out_proj.weight)

    @torch.compiler.disable
    def _flex_swa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
        """Windowed attention via compiled FlexAttention.

        The whole method is kept OUT of the outer model compile: dynamo's
        recompile limit on _run_attn_res_layer (per-blocks-tuple signature
        recompiles) silently demotes that frame to eager at runtime, and any
        flex call inlined into it would then run as the unfused eager fallback
        that materialises full [L, L] scores (432GiB at 98k — fatal). With the
        call isolated here, flex is always invoked through its own compiled
        wrapper, fused, regardless of what the surrounding frames do.
        """
        return _get_compiled_flex_attention()(
            q,
            k,
            v,
            block_mask=self._swa_block_mask(n, device),
            enable_gqa=self.kv_repeats > 1,
        )

    @torch.compiler.disable
    def _swa_block_mask(self, n: int, device: torch.device):
        """BlockMask for this length, built on demand and cached per (seq_len, device)
        with a small bound (2 entries) — see the eviction note below.

        The @torch.compiler.disable is load-bearing: the mask build is plain
        eager Python and must stay out of the compiled graph (the model is
        compiled as a whole). Uses build_causal_swa_block_mask — block-level
        construction, NOT create_block_mask, whose dense [L, L] intermediate
        is 72GiB of int64 at 98k. The cached BlockMask is a pytree-registered
        object that flows into the compiled flex_attention kernel cleanly."""
        key = (n, str(device))
        bm = self._swa_block_masks.get(key)
        if bm is None:
            bm = build_causal_swa_block_mask(self.swa_window, n, device)
            # Bound the cache to the two most recent shapes so variable-length
            # inputs cannot grow resident mask storage without limit.
            while len(self._swa_block_masks) >= 2:
                self._swa_block_masks.pop(next(iter(self._swa_block_masks)))
            self._swa_block_masks[key] = bm
        return bm

    def forward(
        self,
        x: torch.Tensor,
        *,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)

        # RoPE cos/sin are shared across all layers (computed once in
        # TransformerStack.forward); fall back to the local table only when a
        # caller (generate / standalone use) didn't pass them. Skipped entirely
        # in NoPE runs (self.use_rope=False).
        if self.use_rope:
            if cos is None or sin is None:
                cos, sin = self.rope(n, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale.to(dtype=q.dtype)
            k = F.normalize(k, dim=-1) * self.k_scale.to(dtype=k.dtype)
        # GQA: pass kv_heads K/V directly with enable_gqa instead of materialising
        # the heads/kv_heads copy via repeat_kv — bit-identical, far less memory
        # and bandwidth at long context.
        if self.swa_window is None:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                enable_gqa=self.kv_repeats > 1,
            )
        else:
            # Sliding-window layer: FlexAttention with a causal banded BlockMask
            # (SDPA has no window support), via the compile-isolated _flex_swa
            # helper — see its docstring for why the isolation is load-bearing.
            # (flex has no dropout arg; every preset runs attn_dropout=0.0.)
            out = self._flex_swa(q, k, v, n, x.device)
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.out_proj(out)

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        position: int,
        cache: AttentionDecodeCache | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, AttentionDecodeCache]:
        """Decode a single token with a K/V cache.

        The cache already contains only positions <= the current token, so the
        SDPA call must be non-causal. Passing `is_causal=True` with a query
        length of 1 would incorrectly mask the query down to key position 0.
        """
        b, n, _ = x.shape
        if n != 1:
            raise ValueError(f"forward_step expects [B, 1, D], got {tuple(x.shape)}")

        q = self.q_proj(x).view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if cos is None or sin is None:
                cos, sin = self.rope.forward_position(position, x.device)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)

        if self.use_qk_norm:
            q = F.normalize(q, dim=-1) * self.q_scale.to(dtype=q.dtype)
            k = F.normalize(k, dim=-1) * self.k_scale.to(dtype=k.dtype)

        if cache is not None and cache.length is not None:
            past_len = int(cache.length)
            new_len = past_len + n
            if new_len > cache.k.shape[2]:
                raise ValueError(
                    f"attention decode cache capacity {cache.k.shape[2]} exceeded by length {new_len}"
                )
            cache.k[:, :, past_len:new_len, :].copy_(k)
            cache.v[:, :, past_len:new_len, :].copy_(v)
            cache.length = new_len
            # Use the contiguous prefix view directly instead of cloning the
            # growing decode cache on every token.
            k = cache.k[:, :, :new_len, :]
            v = cache.v[:, :, :new_len, :]
            next_cache = cache
        elif cache is not None:
            k = torch.cat((cache.k.to(device=k.device, dtype=k.dtype), k), dim=2)
            v = torch.cat((cache.v.to(device=v.device, dtype=v.dtype), v), dim=2)
            next_cache = AttentionDecodeCache(k=k.detach(), v=v.detach())
        else:
            next_cache = AttentionDecodeCache(k=k.detach(), v=v.detach())

        attn_mask = None
        if self.swa_window is not None:
            # The cache holds only keys <= position, so causality is already
            # guaranteed; the window just drops keys older than the window,
            # matching the training-time banded mask (kv > position - window).
            keep = torch.arange(k.shape[2], device=k.device) > (position - self.swa_window)
            attn_mask = keep.view(1, 1, 1, -1)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
            enable_gqa=self.kv_repeats > 1,
            attn_mask=attn_mask,
        )
        out = out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)
        return self.out_proj(out), next_cache


class SwiGLU(nn.Module):
    def __init__(self, *, dim: int, hidden_dim: int, dropout: float, zero_init_residual: bool,
                 ffn_bias: bool = False):
        super().__init__()
        # Optional gate/up bias; down_proj remains bias-free.
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=ffn_bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=ffn_bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        if zero_init_residual:
            nn.init.zeros_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(self.dropout(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        kv_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ffn_dropout: float,
        resid_dropout: float,
        rope_theta: float,
        use_qk_norm: bool,
        zero_init_residual: bool,
        engram: EngramMemory | None,
        use_attn_residual: bool,
        attn_impl: str = "sdpa",
        use_rope: bool = True,
        swa_window: int | None = None,
        ffn_bias: bool = False,
    ):
        super().__init__()
        self.engram = engram
        self.attn_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        if attn_impl == "sdpa":
            self.attn = CausalSelfAttention(
                dim=dim,
                heads=heads,
                kv_heads=kv_heads,
                dropout=attn_dropout,
                rope_theta=rope_theta,
                use_qk_norm=use_qk_norm,
                zero_init_residual=zero_init_residual,
                use_rope=use_rope,
                swa_window=swa_window,
            )
        else:
            raise ValueError(f"unknown attn_impl {attn_impl!r}")
        self.ffn = SwiGLU(
            dim=dim,
            hidden_dim=ffn_dim,
            dropout=ffn_dropout,
            zero_init_residual=zero_init_residual,
            ffn_bias=ffn_bias,
        )
        self.resid_dropout = nn.Dropout(resid_dropout)
        self.attn_residual = (
            AttentionResidualMixer(dim)
            if use_attn_residual else None
        )
        self.ffn_residual = (
            AttentionResidualMixer(dim)
            if use_attn_residual else None
        )

    @staticmethod
    def _add_partial(partial_block: torch.Tensor | None, update: torch.Tensor) -> torch.Tensor:
        return update if partial_block is None else partial_block + update

    def forward(
        self,
        x: torch.Tensor,
        ids: torch.Tensor | None = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.engram is not None:
            if ids is None:
                raise ValueError("Engram-enabled blocks require token ids")
            x = x + self.engram(x, ids)
        x = x + self.resid_dropout(self.attn(self.attn_norm(x), cos=cos, sin=sin))
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x

    def forward_step(
        self,
        x: torch.Tensor,
        ids: torch.Tensor | None,
        position: int,
        cache: LayerDecodeCache | None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerDecodeCache]:
        if cache is None:
            cache = LayerDecodeCache()
        if self.engram is not None:
            if ids is None:
                raise ValueError("Engram-enabled blocks require token ids")
            engram_update, cache.engram_state = self.engram.forward_step(x, ids, cache.engram_state)
            x = x + engram_update
        attn_out, cache.attn = self.attn.forward_step(
            self.attn_norm(x),
            position=position,
            cache=cache.attn,
            cos=cos,
            sin=sin,
        )
        x = x + self.resid_dropout(attn_out)
        x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x, cache

    def forward_attn_res(
        self,
        blocks: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor,
        ids: torch.Tensor | None,
        start_new_block: bool,
        window: int | None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Apply attention, Engram, and FFN updates to AttnRes block state."""
        if self.attn_residual is None or self.ffn_residual is None:
            raise RuntimeError("forward_attn_res called on a block without AttnRes")

        h = self.attn_residual(blocks, partial_block)
        if start_new_block:
            blocks = blocks + (partial_block,)
            if window is not None and len(blocks) > window:
                blocks = blocks[-window:]
            partial_block = None

        if self.engram is not None:
            if ids is None:
                raise ValueError("Engram-enabled blocks require token ids")
            engram_update = self.engram(h, ids)
            h = h + engram_update
            partial_block = self._add_partial(partial_block, engram_update)

        attn_out = self.resid_dropout(self.attn(self.attn_norm(h), cos=cos, sin=sin))
        partial_block = self._add_partial(partial_block, attn_out)

        h = self.ffn_residual(blocks, partial_block)
        ffn_out = self.resid_dropout(self.ffn(self.ffn_norm(h)))
        return blocks, partial_block + ffn_out

    def forward_step_attn_res(
        self,
        blocks: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor,
        ids: torch.Tensor | None,
        position: int,
        cache: LayerDecodeCache | None,
        start_new_block: bool,
        window: int | None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, LayerDecodeCache]:
        if self.attn_residual is None or self.ffn_residual is None:
            raise RuntimeError("forward_step_attn_res called on a block without AttnRes")
        if cache is None:
            cache = LayerDecodeCache()

        h = self.attn_residual(blocks, partial_block)
        if start_new_block:
            blocks = blocks + (partial_block,)
            if window is not None and len(blocks) > window:
                blocks = blocks[-window:]
            partial_block = None

        if self.engram is not None:
            if ids is None:
                raise ValueError("Engram-enabled blocks require token ids")
            engram_update, cache.engram_state = self.engram.forward_step(h, ids, cache.engram_state)
            h = h + engram_update
            partial_block = self._add_partial(partial_block, engram_update)

        attn_out, cache.attn = self.attn.forward_step(
            self.attn_norm(h),
            position=position,
            cache=cache.attn,
            cos=cos,
            sin=sin,
        )
        attn_out = self.resid_dropout(attn_out)
        partial_block = self._add_partial(partial_block, attn_out)

        h = self.ffn_residual(blocks, partial_block)
        ffn_out = self.resid_dropout(self.ffn(self.ffn_norm(h)))
        return blocks, partial_block + ffn_out, cache


class AttentionResidualMixer(nn.Module):
    """Block AttnRes with two-pass, low-memory depth mixing."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.pseudo_query = nn.Parameter(torch.zeros(dim))

    def forward(self, blocks: Sequence[torch.Tensor], partial_block: torch.Tensor) -> torch.Tensor:
        states = (*blocks, partial_block)
        dt = partial_block.dtype
        qg = (self.pseudo_query.float() * self.norm.weight.float()).to(dt)
        logits = []
        for value in states:
            mean_square = value.pow(2).mean(dim=-1, dtype=torch.float32)
            rstd = torch.rsqrt(mean_square + self.norm.eps)
            logits.append(torch.einsum("h,bth->bt", qg, value).float() * rstd)
        weights = torch.stack(logits, dim=0).softmax(dim=0)
        acc: torch.Tensor | None = None
        for weight, value in zip(weights, states):
            term = (weight.unsqueeze(-1).to(dt) * value).float()
            acc = term if acc is None else acc + term
        return acc.to(dt)


class TransformerStack(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        layers: int,
        heads: int,
        kv_heads: int,
        ffn_dim: int,
        attn_dropout: float,
        ffn_dropout: float,
        resid_dropout: float,
        rope_theta: float,
        use_qk_norm: bool,
        use_attn_residual: bool,
        attn_residual_block_size: int,
        attn_residual_window: int | None = None,
        zero_init_residual: bool,
        vocab_size: int,
        pad_id: int,
        use_engram: bool,
        engram_slots: int,
        engram_dim: int,
        engram_ks: tuple[int, ...],
        engram_layer_ids: tuple[int, ...],
        engram_heads_per_order: int,
        engram_direct_threshold: int = 65_536,
        attn_impl: str = "sdpa",
        use_rope: bool = True,
        swa_window: int | None = None,
        swa_full_every: int = 4,
        ffn_bias: bool = False,
    ):
        super().__init__()
        if use_attn_residual and zero_init_residual:
            # AttnRes requires a nonzero residual stream at block boundaries.
            zero_init_residual = False
        engram_layers = set(engram_layer_ids) if use_engram else set()
        bad_layers = sorted(idx for idx in engram_layers if idx < 0 or idx >= layers)
        if bad_layers:
            raise ValueError(f"engram_layer_ids out of range for {layers} layers: {bad_layers}")

        self.layers = nn.ModuleList()
        for idx in range(layers):
            engram = None
            if idx in engram_layers:
                engram = EngramMemory(
                    vocab_size=vocab_size,
                    pad_id=pad_id,
                    slots=engram_slots,
                    memory_dim=engram_dim,
                    model_dim=dim,
                    ks=engram_ks,
                    heads_per_order=engram_heads_per_order,
                    layer_idx=idx,
                    zero_init_residual=zero_init_residual,
                    direct_threshold=engram_direct_threshold,
                )
            block_attn_impl = "sdpa"
            # Optional sliding window on SDPA layers: idx % swa_full_every == 0
            # keeps full-sequence attention; the remaining layers use the window.
            block_swa_window = (
                swa_window
                if (
                    swa_window is not None
                    and block_attn_impl == "sdpa"
                    and idx % swa_full_every != 0
                )
                else None
            )
            self.layers.append(
                TransformerBlock(
                    dim=dim,
                    heads=heads,
                    kv_heads=kv_heads,
                    ffn_dim=ffn_dim,
                    attn_dropout=attn_dropout,
                    ffn_dropout=ffn_dropout,
                    resid_dropout=resid_dropout,
                    rope_theta=rope_theta,
                    use_qk_norm=use_qk_norm,
                    zero_init_residual=zero_init_residual,
                    engram=engram,
                    use_attn_residual=use_attn_residual,
                    attn_impl=block_attn_impl,
                    use_rope=use_rope,
                    swa_window=block_swa_window,
                    ffn_bias=ffn_bias,
                )
            )
        self.norm = RMSNorm(dim)
        self.use_attn_residual = use_attn_residual
        self.use_rope = use_rope
        # The reference counts block_size in sublayers (attention + MLP).
        self.attn_residual_block_layers = max(1, attn_residual_block_size // 2)
        # Sliding window over AttnRes block states: None = keep all (bit-equivalent
        # to the original AttnRes); K = each layer's depth attention sees only the
        # most recent K blocks. Caps the otherwise O(layers) resident block memory,
        # which is the dominant long-context cost and cannot be saved by any
        # checkpoint (these blocks are downstream inputs, not droppable activations).
        self.attn_residual_window = attn_residual_window
        self.gradient_checkpointing = False
        # Partial (selective) checkpointing: leave this many layers OUTSIDE the
        # checkpoint (evenly spread over the stack), 0 = checkpoint every layer
        # (the historical behaviour).  Each free layer keeps its activations
        # resident (the AttnRes depth-attention KV and SwiGLU saves are heavy
        # at long context), trading memory for less
        # recompute.  Numerics are unaffected (checkpointing only controls
        # recompute), so a run can switch the count at resume without breaking
        # comparability.
        self.gradient_checkpointing_free: int = 0
        self._ckpt_free_set: set[int] = set()

    def set_gradient_checkpointing(self, enabled: bool, free: int = 0) -> None:
        # Both forward paths now checkpoint at the whole-layer level through this
        # single flag: the standard path in _run_standard_layer and the AttnRes
        # path in _run_attn_res_layer. forward_attn_res is a pure function, so the
        # whole layer (mixer + engram + attn + ffn) replays correctly under
        # checkpoint — no per-block flag is needed anymore.
        self.gradient_checkpointing = enabled
        n_layers = len(self.layers)
        k = max(0, min(int(free), n_layers))
        self.gradient_checkpointing_free = k
        if not k:
            self._ckpt_free_set = set()
            return
        # Which layers stay OUT of checkpointing (free = activations resident,
        # no forward recompute in backward). Full-attention layers are the most
        # expensive to recompute in an SWA stack and have small resident state,
        # so they are freed first. Layer 0 is always a full-attention layer.
        full = [
            i for i, ly in enumerate(self.layers)
            if isinstance(ly.attn, CausalSelfAttention) and ly.attn.swa_window is None
        ]
        if full and len(full) < n_layers:
            others = [i for i in range(n_layers) if i not in set(full)]
            chosen = (full + others)[:k]
        else:
            # Uniform full-attention stack: spread the free layers by index.
            chosen = [i * n_layers // k for i in range(k)]
        self._ckpt_free_set = set(chosen)

    def _ckpt_layer(self, idx: int) -> bool:
        """Whether layer `idx` runs under activation checkpoint this step."""
        return (
            self.gradient_checkpointing
            and self.training
            and idx not in self._ckpt_free_set
        )

    def empty_decode_cache(self) -> list[LayerDecodeCache]:
        return [LayerDecodeCache() for _ in self.layers]

    def empty_decode_cache_prealloc(
        self,
        *,
        batch_size: int,
        max_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[LayerDecodeCache]:
        caches: list[LayerDecodeCache] = []
        for layer in self.layers:
            attn = layer.attn
            k = torch.empty(
                batch_size,
                attn.kv_heads,
                max_length,
                attn.head_dim,
                device=device,
                dtype=dtype,
            )
            v = torch.empty_like(k)
            caches.append(LayerDecodeCache(attn=AttentionDecodeCache(k=k, v=v, length=0)))
        return caches

    def _run_standard_layer(
        self,
        layer: TransformerBlock,
        x: torch.Tensor,
        ids: torch.Tensor | None,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        layer_idx: int,
    ) -> torch.Tensor:
        if self._ckpt_layer(layer_idx):
            return checkpoint(layer, x, ids, cos, sin, use_reentrant=False)
        return layer(x, ids, cos=cos, sin=sin)

    def _run_attn_res_layer(
        self,
        layer: TransformerBlock,
        blocks: tuple[torch.Tensor, ...],
        partial_block: torch.Tensor,
        ids: torch.Tensor | None,
        cos: torch.Tensor | None,
        sin: torch.Tensor | None,
        start_new_block: bool,
        layer_idx: int,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        if self._ckpt_layer(layer_idx):
            return checkpoint(
                layer.forward_attn_res, blocks, partial_block, ids, start_new_block,
                self.attn_residual_window, cos, sin, use_reentrant=False)
        return layer.forward_attn_res(
            blocks, partial_block, ids, start_new_block,
            self.attn_residual_window, cos=cos, sin=sin)

    def forward(
        self,
        x: torch.Tensor,
        *,
        ids: torch.Tensor | None = None,
        hidden_indices: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        """Run the stack and optionally collect per-layer hidden states."""
        collect = set(hidden_indices) if hidden_indices is not None else None
        collected: dict[int, torch.Tensor] = {}
        n = x.shape[1]
        cos, sin = self.layers[0].attn.rope(n, x.device) if self.use_rope else (None, None)

        if not self.use_attn_residual:
            for idx, layer in enumerate(self.layers):
                x = self._run_standard_layer(layer, x, ids, cos, sin, idx)
                if collect is not None and idx in collect:
                    collected[idx] = x
            return self.norm(x), collected

        blocks: tuple[torch.Tensor, ...] = (x,)
        partial_block = x
        for idx, layer in enumerate(self.layers):
            start_new_block = (idx + 1) % self.attn_residual_block_layers == 0
            blocks, partial_block = self._run_attn_res_layer(
                layer, blocks, partial_block, ids, cos, sin, start_new_block, idx
            )
            x = partial_block

            if collect is not None and idx in collect:
                collected[idx] = x
        return self.norm(x), collected

    def forward_step(
        self,
        x: torch.Tensor,
        *,
        ids: torch.Tensor | None,
        position: int,
        cache: list[LayerDecodeCache] | None = None,
        hidden_indices: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, list[LayerDecodeCache], dict[int, torch.Tensor]]:
        """Run one autoregressive decode step with per-layer caches."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError(f"forward_step expects [B, 1, D], got {tuple(x.shape)}")
        if cache is None:
            cache = self.empty_decode_cache()
        if len(cache) != len(self.layers):
            raise ValueError(f"decode cache has {len(cache)} layers, expected {len(self.layers)}")

        collect = set(hidden_indices) if hidden_indices is not None else None
        collected: dict[int, torch.Tensor] = {}
        cos, sin = self.layers[0].attn.rope.forward_position(position, x.device) if self.use_rope else (None, None)

        if not self.use_attn_residual:
            for idx, layer in enumerate(self.layers):
                x, cache[idx] = layer.forward_step(x, ids, position, cache[idx], cos=cos, sin=sin)
                if collect is not None and idx in collect:
                    collected[idx] = x
            return self.norm(x), cache, collected

        blocks: tuple[torch.Tensor, ...] = (x,)
        partial_block = x
        for idx, layer in enumerate(self.layers):
            start_new_block = (idx + 1) % self.attn_residual_block_layers == 0
            blocks, partial_block, cache[idx] = layer.forward_step_attn_res(
                blocks,
                partial_block,
                ids,
                position,
                cache[idx],
                start_new_block,
                self.attn_residual_window,
                cos=cos,
                sin=sin,
            )
            x = partial_block
            if collect is not None and idx in collect:
                collected[idx] = x
        return self.norm(x), cache, collected
