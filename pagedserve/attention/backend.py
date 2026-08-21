"""The attention backend interface. Required indirection, not optional polish.

Three implementations will exist by the end of this project and all three must
be selectable at runtime:

- ``contiguous`` (Phase 1) — one dense ``[seq_slot, max_seq_len, ...]`` buffer
  per layer. Wastes most of what it reserves. Survives forever as the
  ``--no-paging`` ablation arm, which is what turns the headline comparison into
  a controlled experiment where paging is the only variable.
- ``gather`` (Phase 2) — paged storage, read by copying blocks into a scratch
  buffer. Correct and slow. Survives forever as the oracle the CUDA kernel is
  diffed against.
- ``cuda`` (Phase 4) — paged storage read in-place through block tables.

Because a slower backend is never deleted, the seam between "how KV is stored"
and "how attention is computed" has to be real from the first phase. Retrofitting
it after the model code has reached into a concrete cache layout is a rewrite.

The split of responsibilities:

- The **engine** owns the batch, decides what runs this step, and builds a
  ``StepInput`` describing it in layout-independent terms.
- The **backend** turns that into its own ``AttentionMetadata`` once per step,
  and owns the KV storage entirely.
- The **model** calls ``forward`` once per layer and knows nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

__all__ = [
    "AttentionBackend",
    "AttentionMetadata",
    "KVMemoryStats",
    "StepInput",
]


@dataclass(frozen=True)
class KVMemoryStats:
    """Reserved KV bytes versus KV bytes holding real tokens.

    The whole thesis of this project in two integers. A contiguous cache
    reserves ``max_seq_len`` per sequence whether or not those tokens exist, so
    ``utilization`` is expected to be low — the vLLM paper measured 20.4%-38.2%
    in existing systems, and seeing our own number land in that range is the
    Phase 1 exit criterion.
    """

    allocated_bytes: int
    live_bytes: int

    @property
    def utilization(self) -> float | None:
        """Fraction of reserved KV actually holding tokens, or ``None``.

        ``None`` rather than ``0.0`` when nothing is allocated: no cache means
        the question is unanswerable, not that the answer is zero.
        """
        if self.allocated_bytes <= 0:
            return None
        return self.live_bytes / self.allocated_bytes

    @property
    def wasted_bytes(self) -> int:
        return max(0, self.allocated_bytes - self.live_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allocated_bytes": self.allocated_bytes,
            "live_bytes": self.live_bytes,
            "wasted_bytes": self.wasted_bytes,
            "utilization": self.utilization,
        }


@dataclass(frozen=True)
class StepInput:
    """What the engine knows about one forward pass, independent of KV layout.

    Deliberately a dataclass rather than a widening argument list: Phase 2 adds
    ``block_tables`` and ``slot_mapping``, and a new field with a default does
    not break a backend that ignores it.

    Attributes:
        query_len: Tokens per sequence in this step. The prompt length during
            prefill, 1 during decode. Prefill and decode are different
            computations with opposite bottlenecks, and this is what tells them
            apart.
        context_len: Tokens already cached per sequence *before* this step.
        seq_lens: Per-sequence count of real, non-padding tokens after this
            step. Drives the live-bytes half of the utilization number.
        padding_mask: ``[num_seqs, context_len + query_len]``, True where a
            position holds a real token. Left padding means a batch's sequences
            do not start at index 0, and attending to a pad would silently
            corrupt the output rather than crash.
        is_prefill: Whether this step processes prompts.
        query_positions: ``[num_seqs, query_len]`` logical position of each
            query token within its own sequence. A dense cache can infer this
            from the batch's padded indices; a paged cache cannot, because it
            stores a sequence at its own logical offsets and never stores
            padding at all.
        block_tables: Per-sequence logical-to-physical block map.
        slot_mapping: One flat KV slot index per token in the batch.
    """

    query_len: int
    context_len: int
    seq_lens: torch.Tensor
    padding_mask: torch.Tensor
    is_prefill: bool
    query_positions: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    slot_mapping: torch.Tensor | None = None

    @property
    def num_seqs(self) -> int:
        return int(self.padding_mask.shape[0])

    @property
    def key_len(self) -> int:
        """Cache positions readable this step: everything up to and including it."""
        return self.context_len + self.query_len


class AttentionMetadata:
    """Backend-private per-step state, built once and reused by every layer.

    A marker base rather than an interface: backends share no per-step fields,
    because what a step needs is precisely what differs between a dense buffer
    and a block table.

    Exists so that per-step work — mask construction, block-table staging — is
    paid once rather than ``num_layers`` times. With 24 layers, rebuilding a
    mask inside the layer loop would be 24x the cost for an identical result.
    """


class AttentionBackend(ABC):
    """How KV is stored, written, and read.

    Implementations own their KV storage completely. Nothing outside a backend
    may index into a KV cache, because the layout is exactly what changes
    between phases.
    """

    name: str = "abstract"

    @abstractmethod
    def allocate(self) -> None:
        """Reserve KV storage up front.

        Takes no arguments on purpose. Sizing is a property of the backend and
        its ``CacheConfig``, not something a caller supplies — a dense cache is
        sized by sequence slots times max length, a paged one by a block count,
        and a signature that named either would leak that layout into every
        caller. This is the seam that lets Phase 4 drop in without touching the
        engine.

        Called once at startup, never in the decode loop. Allocator calls in the
        hot loop cost microseconds, can synchronise the stream, and fragment
        device memory in exactly the way this project exists to avoid.
        """

    @abstractmethod
    def free(self) -> None:
        """Release KV storage. Safe to call when nothing is allocated."""

    @abstractmethod
    def begin_step(self, step: StepInput) -> AttentionMetadata:
        """Prepare per-step state before any layer runs."""

    @abstractmethod
    def forward(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        """Write this step's K/V into the cache and attend over everything cached.

        Writing and reading are one call rather than two because the fused form
        is what the Phase 4 kernel wants, and because it leaves the model code
        with a single point of contact.

        Args:
            layer_idx: Which layer's cache to use.
            query: ``[num_seqs, query_len, num_q_heads, head_dim]``, RoPE applied.
            key: ``[num_seqs, query_len, num_kv_heads, head_dim]``, RoPE applied.
            value: ``[num_seqs, query_len, num_kv_heads, head_dim]``.
            metadata: The object returned by ``begin_step``.

        Returns:
            ``[num_seqs, query_len, num_q_heads, head_dim]``.
        """

    @abstractmethod
    def memory_stats(self) -> KVMemoryStats:
        """Reserved versus live KV bytes, as of the last completed step."""
