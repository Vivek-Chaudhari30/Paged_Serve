"""The KV allocator: a free list, block tables, and reference counts.

This is the piece the whole project rests on, so the invariants it maintains are
written down rather than assumed:

1. ``len(free_blocks) + len(allocated_blocks) == num_blocks``. A block is either
   on the free list or referenced by at least one sequence, never both and never
   neither. A block that is in neither set has leaked and the cache silently
   shrinks over a long run.
2. The sum of all reference counts equals the total number of block references
   held across every sequence's block table. A refcount that drifts high leaks;
   one that drifts low hands a live block to a second writer and corrupts it.
3. A sequence's block table holds exactly ``ceil(len(sequence) / block_size)``
   blocks — never fewer (the next append would write out of bounds) and never
   more (waste).
4. No sequence holds a block whose reference count is zero.

These run under ``EngineConfig.debug_invariants``: on in tests, off in
benchmarks, because checking them is O(num_blocks) and belongs nowhere near a
measured hot loop.

**Why allocation is cheap enough to sit in the decode loop.** ``append_slot``
touches the free list only when a sequence crosses a block boundary — once every
``block_size`` tokens. The other 15 steps out of 16 are pure writes into a block
the sequence already owns. That amortisation is what makes per-step allocation
affordable at all.
"""

from __future__ import annotations

import logging
from collections import deque
from enum import Enum

from pagedserve.memory.block import BlockTable, PhysicalBlock

logger = logging.getLogger(__name__)

__all__ = ["AllocStatus", "BlockManager"]


class AllocStatus(Enum):
    """Whether a sequence can be admitted, and if not, whether waiting helps.

    The distinction matters to the Phase 3 scheduler: ``LATER`` means requeue
    and retry when something frees, while ``NEVER`` means the request cannot run
    at this cache size no matter what and must be rejected outright rather than
    sitting in the queue forever.
    """

    OK = "ok"
    LATER = "later"
    NEVER = "never"


class BlockManager:
    """Hands out fixed-size KV blocks and tracks who holds them."""

    def __init__(self, num_blocks: int, block_size: int, *, watermark: float = 0.01) -> None:
        """
        Args:
            num_blocks: Physical blocks available to sequences. The cache itself
                holds one more — see ``trash_block_id``.
            block_size: Token slots per block.
            watermark: Fraction of the cache kept in reserve when deciding
                whether to admit a *new* sequence. Admitting a request that
                leaves zero headroom means the very next decode step preempts
                something, and preemption thrash costs far more than the
                admission gained.
        """
        if num_blocks < 1:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if block_size < 1:
            raise ValueError(f"block_size must be positive, got {block_size}")

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.watermark_blocks = int(num_blocks * watermark)

        self.blocks = [PhysicalBlock(block_id=i) for i in range(num_blocks)]
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.block_tables: dict[int, BlockTable] = {}

    # ---- the extra block ------------------------------------------------

    @property
    def trash_block_id(self) -> int:
        """A block no sequence ever owns, for padding tokens to write into.

        Static batching left-pads, and those pad positions still flow through
        the model and produce K and V. They need somewhere to go that is not a
        real sequence's cache. Pointing them at a dedicated block keeps the
        write path a single fused scatter with no per-sequence branch, at a cost
        of one block.
        """
        return self.num_blocks

    @property
    def num_cache_blocks(self) -> int:
        """Blocks the KV tensor must actually hold, trash block included."""
        return self.num_blocks + 1

    # ---- accounting -----------------------------------------------------

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    @property
    def utilization(self) -> float:
        """Fraction of blocks currently held by some sequence."""
        return self.num_used_blocks / self.num_blocks

    # ---- allocation -----------------------------------------------------

    def blocks_needed(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        return -(-num_tokens // self.block_size)

    def can_allocate(self, num_tokens: int) -> AllocStatus:
        """Whether a new sequence of ``num_tokens`` can be admitted now."""
        needed = self.blocks_needed(num_tokens)
        if needed > self.num_blocks:
            return AllocStatus.NEVER
        if needed + self.watermark_blocks > self.num_free_blocks:
            return AllocStatus.LATER
        return AllocStatus.OK

    def allocate(self, seq_id: int, num_tokens: int) -> BlockTable:
        """Give a sequence enough blocks to hold ``num_tokens``."""
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} is already allocated")
        needed = self.blocks_needed(num_tokens)
        if needed > self.num_free_blocks:
            raise MemoryError(
                f"need {needed} blocks for sequence {seq_id}, {self.num_free_blocks} free"
            )

        table = BlockTable(self.block_size)
        for _ in range(needed):
            table.append(self._take_free_block())
        self.block_tables[seq_id] = table
        return table

    def append_slot(self, seq_id: int, new_length: int) -> int | None:
        """Extend a sequence to ``new_length`` tokens, growing it if needed.

        Returns the newly allocated block id, or ``None`` when the sequence
        still fits in what it already owns — which is the common case, since a
        boundary is crossed only once every ``block_size`` tokens.
        """
        table = self._table(seq_id)
        needed = self.blocks_needed(new_length)
        if needed <= len(table):
            return None
        if needed - len(table) > self.num_free_blocks:
            raise MemoryError(
                f"sequence {seq_id} needs another block at length {new_length}, none free"
            )
        block_id = self._take_free_block()
        table.append(block_id)
        return block_id

    def free(self, seq_id: int) -> None:
        """Release a sequence's blocks.

        Decrements refcounts and returns to the free list only what reaches
        zero, so a block still shared with another sequence survives. Freeing an
        unknown sequence is a no-op: retirement and preemption paths can race to
        free the same sequence, and making that an error would turn a benign
        double-free into a crashed benchmark.
        """
        table = self.block_tables.pop(seq_id, None)
        if table is None:
            return
        for block_id in table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError(
                    f"block {block_id} refcount went negative freeing sequence {seq_id}"
                )
            if block.ref_count == 0:
                self.free_blocks.append(block_id)

    def fork(self, parent_id: int, child_id: int) -> BlockTable:
        """Share a parent's blocks with a child, copy-on-write.

        Copies the block *table* — a list of integers — and bumps refcounts. The
        KV itself is shared until the two sequences diverge, which is why
        ``n>1`` sampling and beam search cost almost nothing in memory.
        """
        if child_id in self.block_tables:
            raise ValueError(f"sequence {child_id} is already allocated")
        parent = self._table(parent_id)
        child = parent.copy()
        for block_id in child:
            self.blocks[block_id].ref_count += 1
        self.block_tables[child_id] = child
        return child

    # ---- lookups --------------------------------------------------------

    def block_table(self, seq_id: int) -> BlockTable:
        return self._table(seq_id)

    def slots(self, seq_id: int, start: int, count: int) -> list[int]:
        """Flat physical slot indices for a run of logical positions."""
        return self._table(seq_id).slots(start, count)

    def _table(self, seq_id: int) -> BlockTable:
        table = self.block_tables.get(seq_id)
        if table is None:
            raise KeyError(f"sequence {seq_id} has no block table")
        return table

    def _take_free_block(self) -> int:
        block_id = self.free_blocks.popleft()
        block = self.blocks[block_id]
        if block.ref_count != 0:
            raise RuntimeError(
                f"block {block_id} was on the free list with refcount {block.ref_count}"
            )
        block.ref_count = 1
        return block_id

    # ---- invariants -----------------------------------------------------

    def check_invariants(self, sequence_lengths: dict[int, int] | None = None) -> None:
        """Assert the four invariants from AGENTS.md §5.

        O(num_blocks); gated behind ``EngineConfig.debug_invariants`` so it runs
        in tests and never in a measured loop.

        Args:
            sequence_lengths: Optional real token count per sequence. When
                given, also checks that each block table is exactly the right
                size — the invariant that catches an ``append_slot`` that
                silently did not grow.
        """
        free = set(self.free_blocks)
        if len(free) != len(self.free_blocks):
            raise AssertionError("a block appears on the free list more than once")

        held: dict[int, int] = {}
        for seq_id, table in self.block_tables.items():
            for block_id in table:
                held[block_id] = held.get(block_id, 0) + 1
            if free & set(table.blocks):
                raise AssertionError(
                    f"sequence {seq_id} holds a block that is also on the free list"
                )

        allocated = set(held)
        if len(free) + len(allocated) != self.num_blocks:
            raise AssertionError(
                f"free ({len(free)}) + allocated ({len(allocated)}) != "
                f"num_blocks ({self.num_blocks})"
            )
        if free & allocated:
            raise AssertionError(f"blocks are both free and allocated: {sorted(free & allocated)}")

        for block_id, references in held.items():
            actual = self.blocks[block_id].ref_count
            if actual != references:
                raise AssertionError(
                    f"block {block_id} refcount is {actual} but {references} sequences reference it"
                )
            if actual == 0:
                raise AssertionError(f"block {block_id} is held but has refcount zero")

        total_refs = sum(b.ref_count for b in self.blocks)
        total_held = sum(held.values())
        if total_refs != total_held:
            raise AssertionError(
                f"refcount sum ({total_refs}) != total block references ({total_held})"
            )

        if sequence_lengths is not None:
            for seq_id, length in sequence_lengths.items():
                table = self.block_tables.get(seq_id)
                if table is None:
                    continue
                expected = self.blocks_needed(length)
                if len(table) != expected:
                    raise AssertionError(
                        f"sequence {seq_id} has {len(table)} blocks for {length} "
                        f"tokens, expected {expected}"
                    )

    def reset(self) -> None:
        """Return every block to the free list."""
        self.block_tables.clear()
        for block in self.blocks:
            block.ref_count = 0
        self.free_blocks = deque(range(self.num_blocks))
