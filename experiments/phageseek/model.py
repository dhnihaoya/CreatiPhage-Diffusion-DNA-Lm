"""CreatiPhage 145M SWA autoregressive DNA language model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import load_checkpoint
from .config import ModelConfig, get_model_config
from .modules import LayerDecodeCache, RMSNorm, TransformerStack

ReturnValue = Literal["loss", "logits", "embedding"]


@dataclass(slots=True)
class PhageSeekDecodeCache:
    """Model-level autoregressive inference cache.

    `layers` owns the per-transformer-layer K/V and Engram convolution state.
    `position` is the absolute RoPE position for the next token to append.
    `token_suffix` stores only the recent token ids needed to reconstruct
    Engram k-mers; callers do not need to retain or pass the full prefix.
    """

    layers: list[LayerDecodeCache]
    position: int
    token_suffix: torch.Tensor | None = None


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


class PhageSeek(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pad_id = config.pad_id
        self.max_seq_len = config.max_seq_len
        self.embed_layers = tuple(config.embed_layers)
        if min(self.embed_layers) < 0 or max(self.embed_layers) >= config.layers:
            raise ValueError(
                f"embed_layers {self.embed_layers} must all be in [0, {config.layers})"
            )

        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        self.stack = TransformerStack(
            dim=config.dim,
            layers=config.layers,
            heads=config.heads,
            kv_heads=config.kv_heads,
            ffn_dim=config.ffn_dim,
            attn_dropout=config.attn_dropout,
            ffn_dropout=config.ffn_dropout,
            resid_dropout=config.resid_dropout,
            rope_theta=config.rope_theta,
            use_qk_norm=config.use_qk_norm,
            use_attn_residual=config.use_attn_residual,
            attn_residual_block_size=config.attn_residual_block_size,
            attn_residual_window=config.attn_residual_window,
            zero_init_residual=config.zero_init_residual,
            vocab_size=config.vocab_size,
            pad_id=config.pad_id,
            use_engram=config.use_engram,
            engram_slots=config.engram_slots,
            engram_dim=config.engram_dim,
            engram_ks=config.engram_ks,
            engram_layer_ids=config.engram_layer_ids,
            engram_heads_per_order=config.engram_heads_per_order,
            engram_direct_threshold=config.engram_direct_threshold,
            attn_impl=config.attn_impl,
            use_rope=config.use_rope,
            swa_window=config.swa_window,
            swa_full_every=config.swa_full_every,
            ffn_bias=config.ffn_bias,
        )

        self.final_norm = RMSNorm(config.dim)
        self.logit_scale = config.dim ** -0.5
        self.lm_head = None if config.tie_embeddings else nn.Linear(config.dim, config.vocab_size, bias=False)
        self.decode_token_history = max(config.engram_ks) if config.use_engram and config.engram_layer_ids else 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)

    @classmethod
    def from_checkpoint(cls, path):
        state = load_checkpoint(path)
        model = cls(ModelConfig.from_dict(state['model_config']))
        model.load_state_dict(state['model_state'], strict=True)
        return model, state

    def set_gradient_checkpointing(self, enabled: bool, free: int = 0) -> None:
        self.stack.set_gradient_checkpointing(enabled, free)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        # The released tied-embedding readout includes the dimension scale.
        x = self.final_norm(x)
        if self.lm_head is not None:
            return self.lm_head(x)
        return F.linear(x, self.token_emb.weight) * self.logit_scale

    def forward(
        self,
        ids: torch.Tensor,
        return_value: ReturnValue = "loss",
        *,
        labels: torch.Tensor | None = None,
    ):
        if return_value not in {"loss", "logits", "embedding"}:
            raise ValueError('return_value must be one of "embedding", "logits", or "loss"')
        if ids.ndim != 2:
            raise ValueError(f"expected ids to be [batch, seq], got shape {tuple(ids.shape)}")
        seq_len = ids.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f"sequence length {seq_len} exceeds configured max_seq_len {self.max_seq_len}")
        x = self.token_emb(ids)

        hidden_indices = list(self.embed_layers) if return_value == "embedding" else None
        out, collected = self.stack(x, ids=ids, hidden_indices=hidden_indices)
        logits = self._logits(out)[:, :seq_len]

        if return_value == "logits":
            return logits
        if return_value == "embedding":
            return [collected[idx][:, :seq_len] for idx in self.embed_layers]
        if seq_len < 2:
            return logits.sum() * 0.0
        targets = ids[:, 1:seq_len] if labels is None else labels[:, :seq_len - 1]
        pred = logits[:, :seq_len - 1].transpose(1, 2)
        return F.cross_entropy(pred, targets, ignore_index=self.pad_id)

    @torch.no_grad()
    def forward_cached(
        self,
        ids: torch.Tensor,
        cache: PhageSeekDecodeCache | None = None,
        max_cache_len: int | None = None,
    ) -> tuple[torch.Tensor, PhageSeekDecodeCache]:
        """Append one or more tokens to the decode cache and return their logits.

        This is an inference-only API: `ids` are the *new* tokens to append, not
        the whole prefix. Internally it advances one token at a time so the
        cached attention path only ever attends to past+current keys with no
        non-square causal-mask ambiguity.
        """
        if ids.ndim != 2:
            raise ValueError(f"expected ids to be [batch, seq], got shape {tuple(ids.shape)}")
        if ids.shape[1] < 1:
            raise ValueError("forward_cached needs at least one token")

        batch, steps = ids.shape
        device = next(self.parameters()).device
        ids = ids.to(device)
        if cache is None:
            if max_cache_len is not None:
                layer_cache: list[LayerDecodeCache] | None = self.stack.empty_decode_cache_prealloc(
                    batch_size=batch,
                    max_length=max_cache_len,
                    device=device,
                    dtype=self.token_emb.weight.dtype,
                )
            else:
                layer_cache = None
            position = 0
            token_suffix = None
        else:
            layer_cache = cache.layers
            position = cache.position
            token_suffix = cache.token_suffix
            if token_suffix is not None and token_suffix.shape[0] != batch:
                raise ValueError(
                    f"cache batch size {token_suffix.shape[0]} does not match ids batch size {batch}"
                )
        if position + steps > self.max_seq_len:
            raise ValueError(
                f"cached sequence length {position + steps} exceeds configured max_seq_len {self.max_seq_len}"
            )

        logits: list[torch.Tensor] = []
        for offset in range(steps):
            current_ids = ids[:, offset : offset + 1]
            if token_suffix is None or token_suffix.numel() == 0:
                ids_for_engram = current_ids
            else:
                ids_for_engram = torch.cat(
                    (token_suffix.to(device=device, dtype=current_ids.dtype), current_ids),
                    dim=1,
                )

            x = self.token_emb(current_ids)
            out, layer_cache, _ = self.stack.forward_step(
                x,
                ids=ids_for_engram,
                position=position + offset,
                cache=layer_cache,
            )
            logits.append(self._logits(out))

            if self.decode_token_history > 0:
                token_suffix = ids_for_engram[:, -self.decode_token_history :].detach()

        if layer_cache is None:
            raise RuntimeError("internal error: cached forward produced no layer cache")
        next_cache = PhageSeekDecodeCache(
            layers=layer_cache,
            position=position + steps,
            token_suffix=token_suffix,
        )
        return torch.cat(logits, dim=1), next_cache

    def _sample_next(
        self,
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        suppress_ids: Sequence[int] | None,
    ) -> torch.Tensor:
        step_logits = logits / max(temperature, 1e-6)
        if suppress_ids:
            blocked = torch.tensor(
                sorted({int(tok) for tok in suppress_ids}),
                device=step_logits.device,
                dtype=torch.long,
            )
            blocked = blocked[(blocked >= 0) & (blocked < step_logits.shape[-1])]
            if blocked.numel():
                step_logits = step_logits.clone()
                step_logits.index_fill_(-1, blocked, -torch.finfo(step_logits.dtype).max)
        if top_k is not None and top_k > 0:
            values, indices = torch.topk(step_logits, min(top_k, step_logits.shape[-1]), dim=-1)
            masked = torch.full_like(step_logits, -torch.finfo(step_logits.dtype).max)
            step_logits = masked.scatter(-1, indices, values)
        return torch.multinomial(step_logits.softmax(dim=-1), num_samples=1)

    @torch.no_grad()
    def generate_cached(
        self,
        prime: torch.Tensor | None = None,
        seq_len: int = 1024,
        temperature: float = 1.0,
        top_k: int | None = None,
        default_batch_size: int = 1,
        suppress_ids: Sequence[int] | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample with a reusable per-layer decode cache.

        When ``eos_id`` is set, each batch row stops being sampled as soon as it
        emits that token: the EOS is recorded at its sampled position and every
        later position is filled with ``pad_id``. The whole batch returns once
        every row has emitted EOS or ``seq_len`` is reached (so short rows cost
        no extra sampling). This is the stop-at-first-EOS protocol for de-novo
        genome generation: training sequences are ``[BOS, bases, EOS]`` with
        ``BOS=0`` and ``EOS=5``. Anything emitted after EOS is outside the
        training distribution.
        """
        device = next(self.parameters()).device
        was_training = self.training
        self.eval()
        try:
            if seq_len < 1:
                raise ValueError(f"seq_len must be positive, got {seq_len}")
            if seq_len > self.max_seq_len:
                raise ValueError(f"seq_len {seq_len} exceeds configured max_seq_len {self.max_seq_len}")
            if prime is None:
                prefix = torch.full((default_batch_size, 1), self.pad_id, dtype=torch.long, device=device)
            else:
                prefix = prime.to(device)
                if prefix.shape[1] < 1:
                    prefix = torch.full((prefix.shape[0], 1), self.pad_id, dtype=torch.long, device=device)
            if prefix.shape[1] >= seq_len:
                return prefix[:, :seq_len]

            batch = prefix.shape[0]
            # Initialise with pad_id rather than torch.empty: under eos-early-stop
            # a row can return before seq_len, and the unwritten tail must read as
            # padding (ignored by the token->contig decoder), not uninitialised.
            seq = torch.full((batch, seq_len), self.pad_id, dtype=torch.long, device=device)
            seq[:, : prefix.shape[1]] = prefix

            logits, cache = self.forward_cached(prefix, max_cache_len=seq_len)
            next_pos = prefix.shape[1]
            done = torch.zeros(batch, dtype=torch.bool, device=device) if eos_id is not None else None
            # done.all() forces a GPU->CPU sync; evaluating it every step roughly
            # halves decode throughput. Finished rows only emit pad afterwards, so
            # checking every 64 steps is numerically identical (tail is pad either
            # way) while letting the CPU run ahead of the GPU between checks.
            while next_pos < seq_len:
                sampled = self._sample_next(
                    logits[:, -1],
                    temperature=temperature,
                    top_k=top_k,
                    suppress_ids=suppress_ids,
                )
                if done is not None:
                    # Rows already done keep emitting pad; active rows keep their
                    # sampled token (which may itself be the EOS that stops them).
                    next_id = torch.where(
                        done.unsqueeze(1),
                        torch.full_like(sampled, self.pad_id),
                        sampled,
                    )
                    done = done | sampled.squeeze(-1).eq(eos_id)
                else:
                    next_id = sampled
                seq[:, next_pos : next_pos + 1] = next_id
                logits, cache = self.forward_cached(next_id, cache)
                next_pos += 1
                if done is not None and (next_pos & 63) == 0 and bool(done.all()):
                    break
            return seq
        finally:
            if was_training:
                self.train()

    @torch.no_grad()
    def generate(
        self,
        prime: torch.Tensor | None = None,
        seq_len: int = 1024,
        temperature: float = 1.0,
        top_k: int | None = None,
        default_batch_size: int = 1,
        use_cache: bool = False,
        suppress_ids: Sequence[int] | None = None,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively sample up to `seq_len` tokens.

        When ``eos_id`` is set, sampling for each batch row stops at the first
        emitted EOS (see ``generate_cached`` for the protocol rationale).

        Note: for `seq_len > max_seq_len` the sliding window renumbers RoPE
        positions from 0 at every window boundary, so relative-position
        information across boundaries is imperfect.  Keep seq_len <= max_seq_len
        for correct long-form generation.
        """
        if use_cache:
            return self.generate_cached(
                prime=prime,
                seq_len=seq_len,
                temperature=temperature,
                top_k=top_k,
                default_batch_size=default_batch_size,
                suppress_ids=suppress_ids,
                eos_id=eos_id,
            )

        device = next(self.parameters()).device
        was_training = self.training
        self.eval()  # disable dropout even if the caller left us in train mode
        try:
            if prime is None:
                seq = torch.full((default_batch_size, 1), self.pad_id, dtype=torch.long, device=device)
            else:
                seq = prime.to(device)
            batch = seq.shape[0]
            done = torch.zeros(batch, dtype=torch.bool, device=device) if eos_id is not None else None
            while seq.shape[1] < seq_len and not (done is not None and bool(done.all())):
                logits = self(seq[:, -self.max_seq_len :], return_value="logits")[:, -1] / max(temperature, 1e-6)
                sampled = self._sample_next(
                    logits,
                    temperature=1.0,
                    top_k=top_k,
                    suppress_ids=suppress_ids,
                )
                if done is not None:
                    next_id = torch.where(
                        done.unsqueeze(1),
                        torch.full_like(sampled, self.pad_id),
                        sampled,
                    )
                    done = done | sampled.squeeze(-1).eq(eos_id)
                else:
                    next_id = sampled
                seq = torch.cat((seq, next_id), dim=1)
            return seq
        finally:
            if was_training:
                self.train()


def build_model(config: str | ModelConfig | None = None) -> PhageSeek:
    cfg = get_model_config(config)
    return PhageSeek(cfg)
