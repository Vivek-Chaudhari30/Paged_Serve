"""Phase 1: one dense KV buffer per sequence slot. The ablation arm.

Storage is ``[num_layers, 2, num_seq_slots, max_seq_len, num_kv_heads, head_dim]``
— every sequence slot reserves room for ``max_seq_len`` tokens at startup,
whether or not it ever generates them. This is what HuggingFace ``generate()``
effectively does, and it wastes memory three ways:

1. **Internal fragmentation.** A slot reserves 2048 token-slots and the request
   generates 120. The other 94% is dead for the request's whole lifetime.
2. **Reservation fragmentation.** Room held for a running request's *future*
   tokens cannot be lent out, even though nothing has been written there.
3. **Left-padding waste.** Batched generation pads short prompts up to the
   longest in the batch, and those pad positions occupy real cache.

This backend is never deleted. Kept behind ``--no-paging``, it makes the Phase 2
comparison a controlled experiment with paging as the only variable, instead of
a cross-system comparison confounded by a hundred implementation differences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from pagedserve.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    KVMemoryStats,
    StepInput,
)
from pagedserve.config import ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["ContiguousAttentionBackend", "ContiguousMetadata"]


@dataclass(frozen=True)
class ContiguousMetadata(AttentionMetadata):
    """Per-step state for a dense cache.

    Because static batching advances every sequence in lockstep, a single
    ``write_start`` covers the whole batch — every sequence writes this step's
    tokens at the same absolute cache index. That is only true with left
    padding, and it stops being true the moment Phase 3 lets sequences join and
    leave mid-flight.
    """

    write_start: int
    query_len: int
    key_len: int
    attn_bias: torch.Tensor | None
    live_tokens: int


class ContiguousAttentionBackend(AttentionBackend):
    """Dense per-slot KV cache with SDPA over the whole valid prefix."""

    name = "contiguous"

    def __init__(
        self,
        model: ModelConfig,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.kv_cache: torch.Tensor | None = None
        self.num_seq_slots = 0
        self.max_seq_len = 0
        self._live_tokens = 0

    def allocate(self, num_seq_slots: int, max_seq_len: int) -> None:
        """Reserve the entire cache once, at startup.

        Deliberately the worst-case shape: this is the arm whose waste is being
        measured, so it must reserve exactly what a naive implementation would.
        """
        self.free()
        self.num_seq_slots = num_seq_slots
        self.max_seq_len = max_seq_len
        self.kv_cache = torch.zeros(
            (
                self.model.num_layers,
                2,  # K and V
                num_seq_slots,
                max_seq_len,
                self.model.num_kv_heads,
                self.model.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        logger.info(
            "allocated contiguous KV cache: %d slots x %d tokens = %.1f MiB",
            num_seq_slots,
            max_seq_len,
            self.kv_cache.numel() * self.kv_cache.element_size() / 2**20,
        )

    def free(self) -> None:
        self.kv_cache = None
        self.num_seq_slots = 0
        self.max_seq_len = 0
        self._live_tokens = 0

    def begin_step(self, step: StepInput) -> ContiguousMetadata:
        """Build the additive attention mask once per step, not once per layer."""
        if self.kv_cache is None:
            raise RuntimeError("allocate() must be called before begin_step()")
        if step.key_len > self.max_seq_len:
            raise ValueError(
                f"sequence length {step.key_len} exceeds the cache's max_seq_len "
                f"{self.max_seq_len}; raise CacheConfig.max_seq_len"
            )

        # Live bytes count real tokens only. Pad positions occupy cache but hold
        # nothing, and counting them would flatter the utilization number that
        # this entire phase exists to expose.
        self._live_tokens = int(step.seq_lens.sum().item())

        return ContiguousMetadata(
            write_start=step.context_len,
            query_len=step.query_len,
            key_len=step.key_len,
            attn_bias=self._build_attn_bias(step),
            live_tokens=self._live_tokens,
        )

    def _build_attn_bias(self, step: StepInput) -> torch.Tensor:
        """Additive mask of shape ``[num_seqs, 1, query_len, key_len]``.

        Two things are masked, and missing either one corrupts output silently
        rather than raising:

        - **Padding.** Left padding puts non-tokens at the start of every short
          sequence. Attending to them mixes garbage into real activations.
        - **Causality.** A query at absolute position *p* may only see keys at
          positions <= *p*.

        Uses ``finfo.min`` rather than ``-inf``: a row that is entirely masked
        would softmax ``-inf`` values into NaN, and a large negative number
        degrades to a uniform row instead of poisoning the whole tensor.
        """
        num_seqs = step.num_seqs
        query_len = step.query_len
        key_len = step.key_len
        neg = torch.finfo(self.dtype).min

        # [num_seqs, 1, 1, key_len] -- True where the key is a real token.
        keep = step.padding_mask[:, :key_len].view(num_seqs, 1, 1, key_len)

        # Absolute cache index of each query in this step.
        query_pos = torch.arange(
            step.context_len, step.context_len + query_len, device=self.device
        ).view(1, 1, query_len, 1)
        key_pos = torch.arange(key_len, device=self.device).view(1, 1, 1, key_len)
        causal = key_pos <= query_pos

        allowed = keep & causal
        return torch.zeros(
            (num_seqs, 1, query_len, key_len), device=self.device, dtype=self.dtype
        ).masked_fill_(~allowed, neg)

    def forward(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        assert isinstance(metadata, ContiguousMetadata)
        if self.kv_cache is None:
            raise RuntimeError("allocate() must be called before forward()")

        start = metadata.write_start
        end = start + metadata.query_len
        num_seqs = query.shape[0]

        # Write this step's K/V into the slots it owns.
        self.kv_cache[layer_idx, 0, :num_seqs, start:end] = key
        self.kv_cache[layer_idx, 1, :num_seqs, start:end] = value

        # Read back the whole valid prefix. [num_seqs, key_len, kv_heads, dim]
        keys = self.kv_cache[layer_idx, 0, :num_seqs, : metadata.key_len]
        values = self.kv_cache[layer_idx, 1, :num_seqs, : metadata.key_len]

        # SDPA wants [batch, heads, seq, dim].
        q = query.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        # GQA: every KV head is shared by num_queries_per_kv query heads. Expand
        # rather than materialise where possible -- expand is a view, and the
        # copy only happens if SDPA needs contiguous input.
        repeats = self.model.num_queries_per_kv
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=metadata.attn_bias)
        return out.transpose(1, 2)

    def memory_stats(self) -> KVMemoryStats:
        """Reserved versus live KV bytes.

        Allocated is the entire reservation, which is the honest denominator:
        those bytes are unavailable to anyone else for the batch's whole
        lifetime, touched or not.
        """
        if self.kv_cache is None:
            return KVMemoryStats(allocated_bytes=0, live_bytes=0)
        element_size = self.kv_cache.element_size()
        allocated = self.kv_cache.numel() * element_size
        per_token = (
            2 * self.model.num_layers * self.model.num_kv_heads * self.model.head_dim
        ) * element_size
        return KVMemoryStats(
            allocated_bytes=allocated,
            live_bytes=self._live_tokens * per_token,
        )
