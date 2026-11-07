"""Configuration for the released CreatiPhage 145M SWA backbone and CPDD."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(slots=True)
class ModelConfig:
    name: str = "phageseek_145m_engram2_swa"
    vocab_size: int = 6
    pad_id: int = 0
    max_seq_len: int = 98_304

    dim: int = 768
    layers: int = 13
    heads: int = 12
    kv_heads: int = 2
    ffn_dim: int = 3776

    engram_slots: int = 131_072
    engram_dim: int = 256
    engram_ks: tuple[int, ...] = (3, 6, 12)
    engram_layer_ids: tuple[int, ...] = (1,)
    engram_heads_per_order: int = 4
    engram_direct_threshold: int = 65_536
    use_engram: bool = True

    attn_residual_block_size: int = 4
    use_attn_residual: bool = True
    attn_residual_window: int | None = None

    use_qk_norm: bool = True
    attn_impl: str = "sdpa"
    swa_window: int | None = 4096
    swa_full_every: int = 4
    rope_theta: float = 10_000.0
    use_rope: bool = True

    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    resid_dropout: float = 0.0
    tie_embeddings: bool = True
    ffn_bias: bool = False
    zero_init_residual: bool = True

    embed_layers: tuple[int, int, int] = (3, 7, 12)

    def __post_init__(self) -> None:
        if self.attn_impl != "sdpa":
            raise ValueError(f"Unsupported attention implementation: {self.attn_impl!r}")
        if self.swa_window is not None and self.swa_window <= 0:
            raise ValueError(f"swa_window must be a positive int or None, got {self.swa_window}")
        if self.swa_full_every < 1:
            raise ValueError(f"swa_full_every must be >= 1, got {self.swa_full_every}")
        if self.use_attn_residual and self.zero_init_residual:
            self.zero_init_residual = False

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["engram_ks"] = list(self.engram_ks)
        out["engram_layer_ids"] = list(self.engram_layer_ids)
        out["embed_layers"] = list(self.embed_layers)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {field.name for field in fields(cls)}
        clean = {key: value for key, value in data.items() if key in known}
        for name in ("engram_ks", "engram_layer_ids", "embed_layers"):
            if name in clean:
                clean[name] = tuple(clean[name])
        clean.setdefault("attn_residual_window", None)
        clean.setdefault("swa_window", None)
        return cls(**clean)


def phageseek_145m_engram2_swa() -> ModelConfig:
    return ModelConfig()


def get_model_config(config: str | ModelConfig | None = None) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return config
    if config in (None, "phageseek_145m_engram2_swa"):
        return phageseek_145m_engram2_swa()
    raise ValueError("This release supports only phageseek_145m_engram2_swa")
