"""Phase 2: paged KV storage, read by gathering blocks into a scratch buffer.

Storage is ``[num_layers, 2, num_blocks + 1, block_size, num_kv_heads, head_dim]``.
A sequence's tokens live wherever its block table points, so admitting a
sequence costs *k* blocks off a free list rather than a contiguous hole of the
right size. Internal fragmentation drops from "up to ``max_seq_len - len``" to
"at most ``block_size - 1``, in the last block only".

**This backend is intentionally slow and stays in the repo forever.**

The read path copies every active sequence's entire KV into a contiguous scratch
buffer on every single decode step, then calls SDPA on it. That is a full
re-read of the working set per step, and it will be slower than the Phase 1
dense cache even though it uses far less memory. Reporting that regression
honestly is the point: it is the motivation for the Phase 4 CUDA kernel, which
reads KV in place through the block tables and never materialises the copy.

Its second job is to be the correctness oracle. Separating "is my allocator
right?" from "is my CUDA kernel right?" is only possible if one of them is known
good, and a kernel diffed against nothing is a kernel debugged by guessing.

The ``+ 1`` block is a trash block. Static batching left-pads, and pad positions
still flow through the model and produce K and V that must go somewhere that is
not a real sequence's cache. Pointing them at a block no sequence owns keeps the
write path a single fused scatter with no per-sequence branching.
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
from pagedserve.config import CacheConfig, ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["PagedAttentionBackend", "PagedMetadata", "RaggedMetadata"]


@dataclass(frozen=True)
class PagedMetadata(AttentionMetadata):
    """Per-step state for a paged cache.

    Attributes:
        slot_mapping: ``[num_seqs * query_len]`` flat destination slot for each
            token in the batch, padding included (which lands in the trash
            block). One fused ``index_copy_`` per layer, never a Python loop
            over sequences — a per-sequence loop in the hot path is the single
            most common reason a hand-rolled engine sits at 30% GPU utilization.
        block_tables: ``[num_seqs, max_blocks]`` physical block ids.
        attn_bias: ``[num_seqs, 1, query_len, gathered_len]`` additive mask.
        gathered_len: Key positions the scratch buffer exposes.
        live_tokens: Real tokens currently cached, for the utilization stat.
        used_blocks: Blocks currently held by some sequence.
    """

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    attn_bias: torch.Tensor
    gathered_len: int
    live_tokens: int
    used_blocks: int


@dataclass(frozen=True)
class RaggedMetadata(AttentionMetadata):
    """Per-step state for a batch that mixes prefills and decodes.

    Continuous batching means one step can hold a 500-token prefill, a
    7-token prefill, and thirty single-token decodes. Padding them to a common
    length would waste compute proportional to the spread, which on a
    heavy-tailed workload is most of the batch.

    Instead every token in the step is flattened into one sequence of length
    ``sum(query_lens)``, exactly the varlen layout FlashAttention's API wants —
    so the Phase 4 kernel drops into the same shape. ``query_offsets`` is the
    cumulative-sequence-length array that makes the flattening addressable.

    Attributes:
        query_lens: Tokens contributed by each sequence this step.
        query_offsets: Exclusive prefix sum of ``query_lens``; where each
            sequence's tokens begin in the flattened batch.
        seq_lens: Total real tokens per sequence after this step.
        decode_rows: Indices of sequences contributing exactly one token. These
            are batched into a single attention call with zero padding, because
            decode is the steady state and the hot path.
        prefill_rows: Indices contributing more than one. Handled one at a time;
            see ``PagedAttentionBackend.forward`` for why that is acceptable
            here and what replaces it.
    """

    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    query_lens: tuple[int, ...]
    query_offsets: tuple[int, ...]
    seq_lens: torch.Tensor
    query_positions: torch.Tensor
    decode_rows: tuple[int, ...]
    prefill_rows: tuple[int, ...]
    gathered_len: int
    live_tokens: int
    used_blocks: int

    @property
    def total_tokens(self) -> int:
        return sum(self.query_lens)


class PagedAttentionBackend(AttentionBackend):
    """Block-table KV storage with a Python gather read path."""

    name = "gather"

    def __init__(
        self,
        model: ModelConfig,
        cache: CacheConfig,
        *,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.model = model
        self.cache = cache
        self.num_blocks = num_blocks
        self.block_size = cache.block_size
        self.device = device
        self.dtype = dtype
        self.kv_cache: torch.Tensor | None = None
        self._live_tokens = 0
        self._used_blocks = 0

    @property
    def trash_block_id(self) -> int:
        """Must match ``BlockManager.trash_block_id``; both are ``num_blocks``."""
        return self.num_blocks

    def allocate(self) -> None:
        """Grab the whole cache at t=0 and manage it ourselves.

        Allocating blocks on demand from the caching allocator would cost
        microseconds per call in the decode loop, can synchronise the stream,
        and would fragment device memory in precisely the way this project
        exists to avoid. Same reason kernels use arena allocators.
        """
        self.free()
        self.kv_cache = torch.zeros(
            (
                self.model.num_layers,
                2,  # K and V
                self.num_blocks + 1,  # the trailing trash block
                self.block_size,
                self.model.num_kv_heads,
                self.model.head_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )
        logger.info(
            "allocated paged KV cache: %d blocks x %d tokens = %d token slots, %.1f MiB",
            self.num_blocks,
            self.block_size,
            self.num_blocks * self.block_size,
            self.kv_cache.numel() * self.kv_cache.element_size() / 2**20,
        )

    def free(self) -> None:
        self.kv_cache = None
        self._live_tokens = 0
        self._used_blocks = 0

    def begin_step(self, step: StepInput) -> PagedMetadata:
        if self.kv_cache is None:
            raise RuntimeError("allocate() must be called before begin_step()")
        if step.slot_mapping is None or step.block_tables is None:
            raise ValueError(
                "the paged backend needs slot_mapping and block_tables in StepInput; "
                "the engine builds them from the BlockManager"
            )
        if step.query_positions is None:
            raise ValueError("the paged backend needs query_positions in StepInput")

        self._live_tokens = int(step.seq_lens.sum().item())
        gathered_len = int(step.block_tables.shape[1]) * self.block_size

        return PagedMetadata(
            slot_mapping=step.slot_mapping,
            block_tables=step.block_tables,
            attn_bias=self._build_attn_bias(step, gathered_len),
            gathered_len=gathered_len,
            live_tokens=self._live_tokens,
            used_blocks=self._used_blocks,
        )

    def set_used_blocks(self, used: int) -> None:
        """Told by the engine, which owns the allocator, for the memory stat."""
        self._used_blocks = used

    def _build_attn_bias(self, step: StepInput, gathered_len: int) -> torch.Tensor:
        """Additive mask over gathered *logical* positions.

        Two differences from the dense backend, both consequences of paging:

        - Keys are indexed by logical position within the sequence, not by a
          shared padded index. Left padding does not exist in a paged cache at
          all, which is itself memory saved.
        - The gathered buffer is a whole number of blocks, so it exposes up to
          ``block_size - 1`` slots past the end of the sequence. Those hold
          stale KV from a previous tenant of the block and must be masked, or a
          sequence attends to another request's data.
        """
        num_seqs = step.num_seqs
        query_len = step.query_len
        neg = torch.finfo(self.dtype).min

        key_pos = torch.arange(gathered_len, device=self.device).view(1, 1, 1, gathered_len)

        # Key is real iff its logical position is inside the sequence.
        key_valid = key_pos < step.seq_lens.view(num_seqs, 1, 1, 1)

        # Causal in logical space. Pad *queries* carry position 0, so they
        # attend only to key 0 -- a finite row whose output is discarded, rather
        # than an all-masked row that would softmax into NaN.
        query_pos = step.query_positions.view(num_seqs, 1, query_len, 1)
        causal = key_pos <= query_pos

        allowed = key_valid & causal
        return torch.zeros(
            (num_seqs, 1, query_len, gathered_len), device=self.device, dtype=self.dtype
        ).masked_fill_(~allowed, neg)

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
        if isinstance(metadata, RaggedMetadata):
            return self._forward_ragged(layer_idx, query, key, value, metadata)
        assert isinstance(metadata, PagedMetadata)

        num_seqs, query_len = query.shape[0], query.shape[1]
        kv_heads, head_dim = self.model.num_kv_heads, self.model.head_dim

        # ---- write: one fused scatter, no loop over sequences ----
        # A flat view over every token slot in the cache, so slot_mapping can be
        # a single index_copy_ rather than B separate small copies.
        k_flat = self.kv_cache[layer_idx, 0].view(-1, kv_heads, head_dim)
        v_flat = self.kv_cache[layer_idx, 1].view(-1, kv_heads, head_dim)
        k_flat.index_copy_(0, metadata.slot_mapping, key.reshape(-1, kv_heads, head_dim))
        v_flat.index_copy_(0, metadata.slot_mapping, value.reshape(-1, kv_heads, head_dim))

        # ---- read: gather this sequence's blocks into scratch ----
        # THE slow step, and the whole reason Phase 4 exists. Every active
        # sequence's entire KV is copied, every layer, every decode step.
        blocks = self.kv_cache[layer_idx, 0].index_select(0, metadata.block_tables.reshape(-1))
        keys = blocks.view(num_seqs, metadata.gathered_len, kv_heads, head_dim)
        blocks = self.kv_cache[layer_idx, 1].index_select(0, metadata.block_tables.reshape(-1))
        values = blocks.view(num_seqs, metadata.gathered_len, kv_heads, head_dim)

        q = query.transpose(1, 2)
        k = keys.transpose(1, 2)
        v = values.transpose(1, 2)

        repeats = self.model.num_queries_per_kv
        if repeats > 1:
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=metadata.attn_bias)
        assert out.shape[2] == query_len
        return out.transpose(1, 2)

    def memory_stats(self) -> KVMemoryStats:
        """Reserved versus live KV bytes.

        ``allocated`` counts only blocks currently held by a sequence, not the
        whole cache. That is the honest denominator for a paged allocator: an
        unallocated block is genuinely available to the next arrival, which is
        exactly what a contiguous reservation cannot say about its unused tail.
        The two backends therefore answer slightly different questions, and the
        comparison between them is the Phase 2 result.
        """
        if self.kv_cache is None:
            return KVMemoryStats(allocated_bytes=0, live_bytes=0)
        per_token = self.model.kv_bytes_per_token(self.dtype)
        return KVMemoryStats(
            allocated_bytes=self._used_blocks * self.block_size * per_token,
            live_bytes=self._live_tokens * per_token,
        )

    def cache_bytes(self) -> int:
        """Total bytes the cache tensor occupies, held or not."""
        if self.kv_cache is None:
            return 0
        return self.kv_cache.numel() * self.kv_cache.element_size()

    # ---- ragged path (Phase 3) -----------------------------------------

    def _forward_ragged(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        md: RaggedMetadata,
    ) -> torch.Tensor:
        """Attention over a batch mixing prefills and single-token decodes.

        Inputs arrive as ``[1, total_tokens, heads, head_dim]`` — the model ran
        on a flattened batch, so no position was padded and every projection,
        norm, and MLP saw exactly the real tokens.

        Attention is where the shapes stop being uniform, and it is split in two:

        - **All decodes in one call.** They share ``query_len == 1``, so they
          batch with zero padding. Decode is the steady state — it runs on every
          step for every live sequence — so this is the path that must not have
          a Python loop in it.
        - **Prefills one at a time.** Their lengths differ, and a single dense
          call would need padding to the longest. A prefill happens once per
          request while decodes happen thousands of times, so the loop sits on
          the cold path by construction. Phase 4 replaces it with
          ``flash_attn_varlen_func``, which is exactly why this metadata already
          carries cumulative sequence lengths.
        """
        kv_heads, head_dim = self.model.num_kv_heads, self.model.head_dim
        num_q_heads = self.model.num_q_heads

        flat_q = query.reshape(-1, num_q_heads, head_dim)
        flat_k = key.reshape(-1, kv_heads, head_dim)
        flat_v = value.reshape(-1, kv_heads, head_dim)

        # Write is layout-agnostic: one fused scatter, ragged or not.
        k_cache = self.kv_cache[layer_idx, 0].view(-1, kv_heads, head_dim)
        v_cache = self.kv_cache[layer_idx, 1].view(-1, kv_heads, head_dim)
        k_cache.index_copy_(0, md.slot_mapping, flat_k)
        v_cache.index_copy_(0, md.slot_mapping, flat_v)

        out = torch.empty_like(flat_q)
        repeats = self.model.num_queries_per_kv

        if md.decode_rows:
            rows = torch.tensor(md.decode_rows, dtype=torch.long, device=self.device)
            starts = torch.tensor(
                [md.query_offsets[r] for r in md.decode_rows],
                dtype=torch.long,
                device=self.device,
            )
            keys, values = self._gather(layer_idx, md.block_tables[rows], len(md.decode_rows))
            # [n, heads, dim] -> [n, heads, 1, dim]: one query row per sequence.
            q = flat_q[starts].unsqueeze(2)
            bias = self._ragged_bias(md, md.decode_rows, query_len=1, gathered_len=md.gathered_len)
            attended = self._sdpa(q, keys, values, bias, repeats)
            out[starts] = attended[:, :, 0]

        for row in md.prefill_rows:
            start = md.query_offsets[row]
            length = md.query_lens[row]
            keys, values = self._gather(layer_idx, md.block_tables[row : row + 1], 1)
            q = flat_q[start : start + length].unsqueeze(0).transpose(1, 2)
            bias = self._ragged_bias(md, (row,), query_len=length, gathered_len=md.gathered_len)
            attended = self._sdpa(q, keys, values, bias, repeats)
            out[start : start + length] = attended[0].transpose(0, 1)

        return out.view(1, -1, num_q_heads, head_dim)

    def _gather(
        self, layer_idx: int, block_tables: torch.Tensor, num_seqs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Copy the named blocks into a contiguous scratch buffer.

        The slow step, and the whole reason Phase 4 exists.
        """
        kv_heads, head_dim = self.model.num_kv_heads, self.model.head_dim
        flat = block_tables.reshape(-1)
        gathered = block_tables.shape[1] * self.block_size
        keys = (
            self.kv_cache[layer_idx, 0]
            .index_select(0, flat)
            .view(num_seqs, gathered, kv_heads, head_dim)
            .transpose(1, 2)
        )
        values = (
            self.kv_cache[layer_idx, 1]
            .index_select(0, flat)
            .view(num_seqs, gathered, kv_heads, head_dim)
            .transpose(1, 2)
        )
        return keys, values

    def _sdpa(self, q, keys, values, bias, repeats: int) -> torch.Tensor:
        if repeats > 1:
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        return F.scaled_dot_product_attention(q, keys, values, attn_mask=bias)

    def _ragged_bias(
        self, md: RaggedMetadata, rows: tuple[int, ...], *, query_len: int, gathered_len: int
    ) -> torch.Tensor:
        """Additive mask for a subset of the step's sequences.

        Same two rules as the static paged path: a key is real only inside its
        own sequence, and causality is enforced in logical space. The gathered
        buffer is a whole number of blocks, so it exposes up to
        ``block_size - 1`` slots holding a previous tenant's KV — masking those
        is what stops one request reading another's data.
        """
        neg = torch.finfo(self.dtype).min
        num = len(rows)
        row_index = torch.tensor(rows, dtype=torch.long, device=self.device)

        key_pos = torch.arange(gathered_len, device=self.device).view(1, 1, 1, gathered_len)
        key_valid = key_pos < md.seq_lens[row_index].view(num, 1, 1, 1)

        positions = []
        for row in rows:
            start = md.query_offsets[row]
            positions.append(md.query_positions[start : start + query_len])
        query_pos = torch.stack(positions).view(num, 1, query_len, 1)

        allowed = key_valid & (key_pos <= query_pos)
        return torch.zeros(
            (num, 1, query_len, gathered_len), device=self.device, dtype=self.dtype
        ).masked_fill_(~allowed, neg)
