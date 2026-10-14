"""Phase 7: replay pure-decode steps from captured CUDA graphs.

A decode step for a 0.5B model is dozens of tiny kernels per layer, and at
small batch sizes the CPU time spent launching them can exceed the GPU time
spent running them. Capturing the whole forward pass once per batch-size bucket
and replaying it turns those launches into one.

Graphs need fixed shapes and fixed addresses, so everything the forward pass
reads lives in persistent buffers sized for the largest bucket. A step copies
its inputs into those buffers and replays the bucket it fits; rows past the
real batch are padding that writes into the trash block and attends over one
token, so their output is finite and discarded.

Only the ``cuda`` backend's ``DecodeMetadata`` path is graph-safe: it does no
host work and no data-dependent allocation. Prefills and mixed batches always
run eagerly.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import torch

from pagedserve.attention.cuda_paged import DecodeMetadata

if TYPE_CHECKING:
    from pagedserve.memory.block_manager import BlockManager
    from pagedserve.sequence import Sequence

logger = logging.getLogger(__name__)

__all__ = ["DecodeGraphRunner", "build_decode_inputs", "default_batch_sizes"]


def default_batch_sizes(max_batch_size: int) -> list[int]:
    """Buckets 1, 2, 4, 8, then every 8 up to the maximum (vLLM's scheme)."""
    sizes = {b for b in (1, 2, 4, 8) if b <= max_batch_size}
    sizes.update(range(16, max_batch_size + 1, 8))
    sizes.add(max_batch_size)
    return sorted(sizes)


def build_decode_inputs(
    scheduled: list[Sequence],
    manager: BlockManager,
    *,
    bucket: int,
    max_blocks: int,
) -> dict[str, torch.Tensor]:
    """Host tensors for one padded decode step.

    Padding rows feed token 0 at position 0, write to the trash block's first
    slot, read a table of trash blocks, and attend over a single token.
    """
    n = len(scheduled)
    if n > bucket:
        raise ValueError(f"{n} sequences do not fit a bucket of {bucket}")
    trash = manager.trash_block_id
    input_ids = [0] * bucket
    positions = [0] * bucket
    slots = [trash * manager.block_size] * bucket
    seq_lens = [1] * bucket
    tables = [[trash] * max_blocks for _ in range(bucket)]
    for row, sequence in enumerate(scheduled):
        position = sequence.num_computed_tokens
        input_ids[row] = sequence.all_token_ids[position]
        positions[row] = position
        slots[row] = manager.slots(sequence.seq_id, position, 1)[0]
        seq_lens[row] = position + 1
        blocks = list(manager.block_table(sequence.seq_id))
        if len(blocks) > max_blocks:
            raise ValueError(f"sequence {sequence.seq_id} has more than {max_blocks} blocks")
        tables[row][: len(blocks)] = blocks
    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "positions": torch.tensor([positions], dtype=torch.long),
        "slot_mapping": torch.tensor(slots, dtype=torch.long),
        "block_tables": torch.tensor(tables, dtype=torch.long),
        "seq_lens": torch.tensor(seq_lens, dtype=torch.int32),
    }


class DecodeGraphRunner:
    """Captures one graph per batch-size bucket and replays it for decode steps."""

    def __init__(
        self,
        model: Any,
        backend: Any,
        *,
        max_batch_size: int,
        max_seq_len: int,
        block_size: int,
        device: torch.device,
        batch_sizes: list[int] | None = None,
    ) -> None:
        self.model = model
        self.backend = backend
        self.device = device
        self.batch_sizes = batch_sizes or default_batch_sizes(max_batch_size)
        self.max_batch_size = max(self.batch_sizes)
        self.max_blocks = math.ceil(max_seq_len / block_size)
        b = self.max_batch_size
        self.input_ids = torch.zeros((1, b), dtype=torch.long, device=device)
        self.positions = torch.zeros((1, b), dtype=torch.long, device=device)
        self.slot_mapping = torch.zeros(b, dtype=torch.long, device=device)
        self.block_tables = torch.zeros((b, self.max_blocks), dtype=torch.long, device=device)
        self.seq_lens = torch.ones(b, dtype=torch.int32, device=device)
        self.logits_indices = torch.arange(b, dtype=torch.long, device=device)
        self._graphs: dict[int, tuple[Any, torch.Tensor]] = {}
        self._pool = None

    def bucket_for(self, n: int) -> int | None:
        return next((b for b in self.batch_sizes if b >= n), None)

    def can_run(self, scheduled: list[Sequence]) -> bool:
        if not self._graphs or not scheduled or len(scheduled) > self.max_batch_size:
            return False
        return all(s.num_uncomputed_tokens == 1 for s in scheduled)

    def _metadata(self, bucket: int) -> DecodeMetadata:
        return DecodeMetadata(
            slot_mapping=self.slot_mapping[:bucket],
            block_tables=self.block_tables[:bucket],
            seq_lens=self.seq_lens[:bucket],
        )

    def _forward(self, bucket: int) -> torch.Tensor:
        return self.model(
            self.input_ids[:, :bucket],
            self.positions[:, :bucket],
            self.backend,
            self._metadata(bucket),
            logits_indices=self.logits_indices[:bucket],
        )

    def _reset_to_padding(self, trash_block: int, block_size: int) -> None:
        self.input_ids.zero_()
        self.positions.zero_()
        self.slot_mapping.fill_(trash_block * block_size)
        self.block_tables.fill_(trash_block)
        self.seq_lens.fill_(1)

    @torch.inference_mode()
    def capture(self, trash_block: int, block_size: int) -> None:
        """Warm up, then capture every bucket. Must follow ``backend.allocate()``.

        The graphs bake in the KV cache's address, so a re-allocated cache
        needs a fresh capture.
        """
        self._graphs.clear()
        self._reset_to_padding(trash_block, block_size)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for bucket in reversed(self.batch_sizes):
                for _ in range(2):
                    self._forward(bucket)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        # Largest first, so smaller graphs reuse the largest one's memory pool.
        for bucket in reversed(self.batch_sizes):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool):
                logits = self._forward(bucket)
            if self._pool is None:
                self._pool = graph.pool()
            self._graphs[bucket] = (graph, logits)
        torch.cuda.synchronize()
        logger.info("captured decode CUDA graphs for batch sizes %s", self.batch_sizes)

    def run(self, scheduled: list[Sequence], manager: BlockManager) -> torch.Tensor:
        """Replay the smallest bucket that fits; returns ``[len(scheduled), vocab]``."""
        n = len(scheduled)
        bucket = self.bucket_for(n)
        assert bucket is not None
        host = build_decode_inputs(scheduled, manager, bucket=bucket, max_blocks=self.max_blocks)
        self.input_ids[:, :bucket].copy_(host["input_ids"], non_blocking=True)
        self.positions[:, :bucket].copy_(host["positions"], non_blocking=True)
        self.slot_mapping[:bucket].copy_(host["slot_mapping"], non_blocking=True)
        self.block_tables[:bucket].copy_(host["block_tables"], non_blocking=True)
        self.seq_lens[:bucket].copy_(host["seq_lens"], non_blocking=True)
        graph, logits = self._graphs[bucket]
        graph.replay()
        return logits[:n]
