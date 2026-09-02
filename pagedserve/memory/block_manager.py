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
from collections.abc import Sequence
from enum import Enum

from pagedserve.memory.block import BlockTable, PhysicalBlock
from pagedserve.memory.prefix_cache import PrefixCache, block_hashes, chain_hash

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

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        *,
        watermark: float = 0.01,
        enable_prefix_caching: bool = False,
    ) -> None:
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

        # Prefix caching introduces a THIRD block state. Without it a block is
        # either on the free list or referenced by a sequence; with it, a block
        # can be referenced by nobody and still be worth keeping, because its
        # contents may be wanted again shortly. Those blocks live in the cache's
        # LRU pool: not free, not referenced, reclaimable on demand.
        self.prefix_cache = PrefixCache(block_size, enabled=enable_prefix_caching)
        # Chained hash per full block, per sequence. Kept so that sealing a
        # newly filled block costs one hash rather than re-hashing the sequence.
        self._chains: dict[int, list[int]] = {}

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
        """Blocks obtainable right now, reclaimable cached blocks included.

        A cached block with no references is available: taking it costs an
        eviction, not a wait. Reporting only the free list would make the
        scheduler defer admissions it could serve, and the more effective the
        cache became the more it would look like memory pressure.
        """
        return len(self.free_blocks) + self.prefix_cache.num_reclaimable

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

    def allocate(
        self, seq_id: int, num_tokens: int, token_ids: Sequence[int] | None = None
    ) -> tuple[BlockTable, int]:
        """Give a sequence enough blocks to hold ``num_tokens``.

        When ``token_ids`` is supplied and prefix caching is on, the leading
        blocks are matched against the cache and reused rather than allocated.
        A reused block already holds the right K and V, so the tokens it covers
        need no forward pass at all -- which is where the time-to-first-token
        saving comes from, since prefill is what dominates TTFT.

        Returns:
            The block table, and how many leading tokens are already computed.
        """
        if seq_id in self.block_tables:
            raise ValueError(f"sequence {seq_id} is already allocated")
        needed = self.blocks_needed(num_tokens)

        table = BlockTable(self.block_size)
        chain: list[int] = []
        cached_tokens = 0

        if token_ids is not None and self.prefix_cache.enabled:
            for block_hash in block_hashes(token_ids, self.block_size)[:needed]:
                block_id = self.prefix_cache.lookup(block_hash)
                if block_id is None:
                    break
                # The refcount is what stops a reused block being reclaimed out
                # from under us; the cache index does not own it.
                self.blocks[block_id].ref_count += 1
                self._discard_free(block_id)
                table.append(block_id)
                chain.append(block_hash)
                cached_tokens += self.block_size
            if cached_tokens:
                self.prefix_cache.stats.blocks_reused += len(table)
                self.prefix_cache.stats.tokens_saved += cached_tokens

        # A fully cached prompt would leave nothing to run, and a forward pass
        # over zero tokens produces no logits to sample from. Hold one token
        # back so there is always work to do.
        cached_tokens = min(cached_tokens, max(0, num_tokens - 1))

        remaining = needed - len(table)
        if remaining > self.num_free_blocks:
            for block_id in table:
                self.blocks[block_id].ref_count -= 1
            raise MemoryError(
                f"need {remaining} more blocks for sequence {seq_id}, "
                f"{self.num_free_blocks} available"
            )
        for _ in range(remaining):
            table.append(self._take_free_block())

        self.block_tables[seq_id] = table
        self._chains[seq_id] = chain
        return table, cached_tokens

    def _discard_free(self, block_id: int) -> None:
        """Take a block off the free list if it happens to be there.

        A cache hit can land on a block that was never released into the pool
        but is sitting on the free list; handing it out twice would corrupt
        both holders.
        """
        if block_id in self.free_blocks:
            self.free_blocks.remove(block_id)

    def seal(self, seq_id: int, token_ids: Sequence[int], num_computed: int) -> None:
        """Index blocks whose KV is now computed and whose block is full.

        Sealing happens *after* the forward pass, never at allocation. A block
        indexed before its K and V exist would be served to another request as
        a cache hit, and that request would attend over uninitialised memory and
        generate fluent nonsense. The tokens are known early; the KV is not.
        """
        if not self.prefix_cache.enabled:
            return
        table = self.block_tables.get(seq_id)
        if table is None:
            return
        chain = self._chains.setdefault(seq_id, [])
        complete = min(num_computed, len(token_ids)) // self.block_size
        while len(chain) < min(complete, len(table)):
            index = len(chain)
            parent = chain[-1] if chain else None
            start = index * self.block_size
            block_hash = chain_hash(parent, token_ids[start : start + self.block_size])
            chain.append(block_hash)
            self.prefix_cache.insert(block_hash, table.blocks[index])

    def copy_on_write(self, seq_id: int, logical_block: int) -> tuple[int, int] | None:
        """Give a sequence a private copy of a block it shares with someone else.

        Returns ``(source, destination)`` for the caller to copy in the KV
        cache, or ``None`` when the block is already exclusive.

        Only full blocks are ever cached and a full block is never written to
        again, so prefix caching alone never triggers this. Forking does: n>1
        sampling shares a partially filled block between samples that then
        diverge, and without the copy the second writer would overwrite the
        first one's tokens in place.
        """
        table = self._table(seq_id)
        source = table.blocks[logical_block]
        if self.blocks[source].ref_count <= 1:
            return None
        destination = self._take_free_block()
        table.blocks[logical_block] = destination
        self.blocks[source].ref_count -= 1
        # The copy is about to diverge from the prefix its hash identifies, so
        # it must not be reachable as a cache hit.
        self.prefix_cache.forget(destination)
        return source, destination

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
        self._chains.pop(seq_id, None)
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
                if self.prefix_cache.is_cached(block_id):
                    # Keep it warm rather than reclaiming it. A finished
                    # request's system-prompt blocks are the most likely blocks
                    # to be wanted a second later.
                    self.prefix_cache.release(block_id)
                else:
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
        if not self.free_blocks:
            # Nothing free, but a cached block nobody references can be
            # reclaimed. This is the "under memory pressure" half of the LRU
            # pool: the cache holds blocks until the allocator actually needs
            # them, and not one step longer.
            evicted = self.prefix_cache.evict()
            if evicted is None:
                raise MemoryError("no free blocks and nothing reclaimable")
            self.free_blocks.append(evicted)
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
        # Prefix caching adds a third state: cached, unreferenced, reclaimable.
        # Such a block is deliberately on neither the free list nor any block
        # table, so the two-way partition no longer holds and asserting it would
        # report a leak every time the cache did its job.
        reclaimable = set(self.prefix_cache.reclaimable_blocks()) - allocated
        total = len(free) + len(allocated) + len(reclaimable)
        if total != self.num_blocks:
            raise AssertionError(
                f"free ({len(free)}) + allocated ({len(allocated)}) + "
                f"reclaimable ({len(reclaimable)}) != num_blocks ({self.num_blocks})"
            )
        if free & reclaimable:
            raise AssertionError(
                f"blocks are both free and reclaimable: {sorted(free & reclaimable)}"
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
        """Return every block to the free list and empty the cache."""
        self.block_tables.clear()
        self._chains.clear()
        self.prefix_cache.reset()
        for block in self.blocks:
            block.ref_count = 0
        self.free_blocks = deque(range(self.num_blocks))
