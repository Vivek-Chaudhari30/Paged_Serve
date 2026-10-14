"""Phase 4: the paged cache read in place by a CUDA kernel.

Storage, allocation, and block tables are exactly the gather backend's — this
class *is* a ``PagedAttentionBackend`` — and only two things change:

- **Write:** ``reshape_and_cache`` scatters the step's K/V straight into their
  blocks (``csrc/cache_kernels.cu``).
- **Decode read:** ``paged_attention_v1`` (``csrc/paged_attention.cu``) walks each
  sequence's block table and never materialises the gathered scratch buffer
  that makes the oracle slow.

Prefill rows still go through the oracle's per-row path. A prefill happens once
per request, decodes happen on every step, and the roadmap is explicit that
prefill should come from an existing kernel rather than a hand-written one.

The ops are injectable. In production they are the compiled ``pagedserve._C``
module; tests pass ``paged_reference.ReferenceOps`` so the backend's plumbing is
golden-tested on a CPU and only the kernel itself is GPU-gated.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from pagedserve import extension
from pagedserve.attention.backend import AttentionMetadata
from pagedserve.attention.gather import PagedAttentionBackend, RaggedMetadata
from pagedserve.config import CacheConfig, ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["CudaPagedAttentionBackend", "DecodeMetadata"]


@dataclass(frozen=True)
class DecodeMetadata(AttentionMetadata):
    """A pure-decode step in fixed-shape tensors.

    What the CUDA-graph runner feeds the model: every field is a persistent
    buffer whose address never changes, so a captured graph can be replayed
    after the buffers are overwritten in place. Padding rows point at the trash
    block with ``seq_len == 1``.

    Attributes:
        slot_mapping: ``[batch]`` int64 destination slot of each new token.
        block_tables: ``[batch, max_blocks]`` int64.
        seq_lens: ``[batch]`` int32 tokens to attend over, new token included.
    """

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    live_tokens: int = 0
    used_blocks: int = 0


class CudaPagedAttentionBackend(PagedAttentionBackend):
    """Block-table KV storage read in place by a custom kernel."""

    name = "cuda"

    def __init__(
        self,
        model: ModelConfig,
        cache: CacheConfig,
        *,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
        ops: Any | None = None,
    ) -> None:
        super().__init__(model, cache, num_blocks=num_blocks, device=device, dtype=dtype)
        if ops is None:
            ops = extension.load()
            if ops is None:
                raise RuntimeError(
                    f"the cuda attention backend needs the compiled extension: "
                    f"{extension.unavailable_reason()}"
                )
        self.ops = ops
        self.scale = model.head_dim**-0.5

    def forward(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if self.kv_cache is None:
            raise RuntimeError("allocate() must be called before forward()")
        if isinstance(metadata, DecodeMetadata):
            return self._forward_decode(layer_idx, query, key, value, metadata)
        if isinstance(metadata, RaggedMetadata):
            return self._forward_ragged(layer_idx, query, key, value, metadata)
        # Static batching (PagedMetadata) is an ablation arm, measured on the
        # oracle's read path; continuous batching is where the kernel matters.
        return super().forward(layer_idx, query, key, value, metadata)

    def _write(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, slots) -> tuple:
        kv_heads, head_dim = self.model.num_kv_heads, self.model.head_dim
        k_cache = self.kv_cache[layer_idx, 0]
        v_cache = self.kv_cache[layer_idx, 1]
        self.ops.reshape_and_cache(
            key.reshape(-1, kv_heads, head_dim),
            value.reshape(-1, kv_heads, head_dim),
            k_cache,
            v_cache,
            slots,
        )
        return k_cache, v_cache

    def _forward_ragged(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        md: RaggedMetadata,
    ) -> torch.Tensor:
        num_q_heads, head_dim = self.model.num_q_heads, self.model.head_dim
        flat_q = query.reshape(-1, num_q_heads, head_dim)
        k_cache, v_cache = self._write(layer_idx, key, value, md.slot_mapping)

        out = torch.empty_like(flat_q)
        if md.decode_rows:
            plan = self._plan(md)
            out[plan.decode_starts] = self.ops.paged_attention_v1(
                flat_q[plan.decode_starts],
                k_cache,
                v_cache,
                plan.decode_block_tables,
                plan.decode_seq_lens,
                self.scale,
            )
        self._attend_prefill_rows(layer_idx, flat_q, out, md)
        return out.view(1, -1, num_q_heads, head_dim)

    def _forward_decode(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        md: DecodeMetadata,
    ) -> torch.Tensor:
        """Every row is one token: write, then one kernel launch. No host work."""
        num_q_heads, head_dim = self.model.num_q_heads, self.model.head_dim
        k_cache, v_cache = self._write(layer_idx, key, value, md.slot_mapping)
        out = self.ops.paged_attention_v1(
            query.reshape(-1, num_q_heads, head_dim).contiguous(),
            k_cache,
            v_cache,
            md.block_tables,
            md.seq_lens,
            self.scale,
        )
        return out.view(1, -1, num_q_heads, head_dim)
