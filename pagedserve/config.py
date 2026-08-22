"""Configuration dataclasses. The single place device, dtype, and shape live.

Nothing else in this package may call ``.cuda()``, hardcode ``torch.bfloat16``,
or bury a shape constant in a function body. This repo runs on macOS (CPU/MPS),
on a T4 or P100 notebook, and on an A100, and anything the roadmap says to sweep
has to be a field here rather than a literal somewhere (AGENTS.md §4, §5).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

__all__ = [
    "CacheConfig",
    "EngineConfig",
    "ModelConfig",
    "SchedulerConfig",
    "resolve_device",
    "resolve_dtype",
]


def resolve_device(spec: str | None = None) -> torch.device:
    """Pick a device once, here, so no other module has to guess.

    Explicit spec wins. Otherwise CUDA, then Apple MPS, then CPU — the order the
    three environments in AGENTS.md §4 appear in.
    """
    if spec:
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(spec: str | None, device: torch.device) -> torch.dtype:
    """Pick a dtype the device can actually run.

    Native bf16 starts at Ampere (compute capability 8.0). Turing (T4) and
    Volta (V100) lack it, and PyTorch will happily *emulate* bf16 there — which
    is correct and slow, and is why this checks the capability directly rather
    than asking ``torch.cuda.is_bf16_supported()``. CPU and MPS get float32 for
    the same reason: emulated bf16 makes a measurement meaningless.
    """
    if spec:
        dtype = getattr(torch, spec, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"not a torch dtype: {spec!r}")
        return dtype
    if device.type == "cuda":
        # Compute capability, NOT torch.cuda.is_bf16_supported(). That returns
        # True on a Turing T4, because recent PyTorch counts *emulated* bf16 as
        # supported. Emulated bf16 is correct and slow, so trusting it would
        # silently pick the wrong dtype on exactly the card most free GPU
        # sessions hand out, and any throughput measured that way would be
        # meaningless. Native bf16 starts at Ampere (8.0).
        index = device.index if device.index is not None else torch.cuda.current_device()
        major, _ = torch.cuda.get_device_capability(index)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


@dataclass(frozen=True)
class ModelConfig:
    """Architecture, read from the checkpoint rather than assumed.

    Covers the Llama-style family: RoPE, GQA, SwiGLU, RMSNorm. Qwen2 is in that
    family but differs in two ways that the layers must honour — its Q/K/V
    projections carry bias terms where Llama's do not, and it ties the output
    projection to the input embedding. Both are read from the checkpoint here
    rather than branched on by name.
    """

    name: str
    num_layers: int
    hidden_size: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    tie_word_embeddings: bool
    attention_bias: bool
    eos_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"num_q_heads ({self.num_q_heads}) must be divisible by "
                f"num_kv_heads ({self.num_kv_heads}); GQA groups would be ragged"
            )

    @property
    def num_queries_per_kv(self) -> int:
        """How many query heads share one KV head. The GQA compression ratio."""
        return self.num_q_heads // self.num_kv_heads

    def kv_bytes_per_token(self, dtype: torch.dtype) -> int:
        """Bytes of KV cache one token costs, across every layer.

        ``2`` for K and V. This is the number the whole project is about: it
        turns a sequence length into a memory footprint, and it is what makes
        "how many sequences fit" a calculation rather than a guess.
        """
        return (
            2 * self.num_layers * self.num_kv_heads * self.head_dim * torch.finfo(dtype).bits // 8
        )

    @classmethod
    def from_pretrained(cls, path: str | Path, name: str | None = None) -> ModelConfig:
        """Parse a HuggingFace ``config.json`` from a local directory."""
        directory = Path(path)
        raw = json.loads((directory / "config.json").read_text())

        # generation_config.json is authoritative for stop tokens and often
        # lists more of them than config.json does -- Qwen2.5 names one EOS in
        # config.json and two here. Missing a stop token means generating past
        # the end of a turn, which looks like a model quality problem and is
        # really a config-parsing bug.
        gen_path = directory / "generation_config.json"
        if gen_path.exists():
            gen_raw = json.loads(gen_path.read_text())
            if gen_raw.get("eos_token_id") is not None:
                raw = {**raw, "eos_token_id": gen_raw["eos_token_id"]}
        return cls.from_hf_dict(raw, name=name or str(path))

    @classmethod
    def from_hf_dict(cls, raw: dict[str, Any], name: str = "") -> ModelConfig:
        """Build from a parsed HuggingFace config dict.

        Kept separate from file reading so it can be tested on a literal without
        a checkpoint on disk.
        """
        hidden = raw["hidden_size"]
        num_q_heads = raw["num_attention_heads"]
        # head_dim is explicit in newer configs and implied in older ones.
        head_dim = raw.get("head_dim") or hidden // num_q_heads

        eos = raw.get("eos_token_id")
        if eos is None:
            eos_ids: tuple[int, ...] = ()
        elif isinstance(eos, int):
            eos_ids = (eos,)
        else:
            eos_ids = tuple(eos)

        model_type = raw.get("model_type", "")
        return cls(
            name=name,
            num_layers=raw["num_hidden_layers"],
            hidden_size=hidden,
            num_q_heads=num_q_heads,
            num_kv_heads=raw.get("num_key_value_heads", num_q_heads),
            head_dim=head_dim,
            intermediate_size=raw["intermediate_size"],
            vocab_size=raw["vocab_size"],
            rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
            rope_theta=raw.get("rope_theta", 10000.0),
            max_position_embeddings=raw.get("max_position_embeddings", 2048),
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            # Qwen2 always biases Q/K/V and never biases the output projection.
            # Llama exposes the choice and defaults to off.
            attention_bias=raw.get("attention_bias", model_type == "qwen2"),
            eos_token_ids=eos_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tie_word_embeddings,
            "attention_bias": self.attention_bias,
        }


@dataclass(frozen=True)
class CacheConfig:
    """KV cache sizing.

    ``block_size`` is unused in Phase 1 — the contiguous cache has no blocks —
    but it lives here from the start because Phase 2 needs it and the roadmap
    calls for sweeping it in Phase 8. A knob that is swept must be a config
    field, never a literal.
    """

    max_seq_len: int = 2048
    max_num_seqs: int = 32
    block_size: int = 16
    gpu_memory_utilization: float = 0.90
    # Skips capacity profiling. Required off CUDA, where there is no way to
    # measure free device memory and guessing one would put a fabricated number
    # under every capacity decision.
    num_blocks_override: int | None = None
    # Host blocks for SWAP preemption. Only allocated when the policy needs it.
    swap_space_blocks: int = 512
    # Prefix caching must be switchable, because "identical output with it
    # on and off" is the property that makes it safe, and that is only
    # testable if both states exist.
    enable_prefix_caching: bool = False

    def __post_init__(self) -> None:
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_seq_len": self.max_seq_len,
            "max_num_seqs": self.max_num_seqs,
            "block_size": self.block_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "num_blocks_override": self.num_blocks_override,
            "swap_space_blocks": self.swap_space_blocks,
            "enable_prefix_caching": self.enable_prefix_caching,
        }


@dataclass(frozen=True)
class SchedulerConfig:
    """Batch composition limits.

    Inert in Phase 1, which batches statically. Phase 3's scheduler budgets on
    both of these at once, because a step holding one 2000-token prefill is
    ~2000x the work of a step holding one decode token, and a sequence-count
    budget alone produces wildly uneven step times.
    """

    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 32
    preemption_policy: str = "recompute"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "preemption_policy": self.preemption_policy,
        }


@dataclass
class EngineConfig:
    """Everything the engine needs, and everything a result file must record.

    ``to_dict`` output is the ``config`` block of a benchmark result. If a knob
    is not in here, a run that changed it is not reproducible.
    """

    model: ModelConfig
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    attn_backend: str = "contiguous"
    debug_invariants: bool = False
    _device: torch.device = field(default_factory=lambda: resolve_device(None))
    _dtype: torch.dtype | None = None

    @classmethod
    def build(
        cls,
        model: ModelConfig,
        *,
        device: str | None = None,
        dtype: str | None = None,
        cache: CacheConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        attn_backend: str = "contiguous",
        debug_invariants: bool = False,
    ) -> EngineConfig:
        """Resolve device and dtype once, here, and hand them down."""
        resolved_device = resolve_device(device)
        return cls(
            model=model,
            cache=cache or CacheConfig(),
            scheduler=scheduler or SchedulerConfig(),
            attn_backend=attn_backend,
            debug_invariants=debug_invariants,
            _device=resolved_device,
            _dtype=resolve_dtype(dtype, resolved_device),
        )

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        if self._dtype is None:
            self._dtype = resolve_dtype(None, self._device)
        return self._dtype

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "cache": self.cache.to_dict(),
            "scheduler": self.scheduler.to_dict(),
            "attn_backend": self.attn_backend,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "debug_invariants": self.debug_invariants,
        }
