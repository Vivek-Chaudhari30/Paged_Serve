"""Unit tests for the KV allocator, including the AGENTS.md §5 invariants.

Pure Python, no torch, no model. The allocator is the piece everything later
rests on, so it gets tested on its own before any tensor touches it.
"""

from __future__ import annotations

import pytest

from pagedserve.memory.block import BlockTable, PhysicalBlock
from pagedserve.memory.block_manager import AllocStatus, BlockManager

BLOCK_SIZE = 4


@pytest.fixture
def manager():
    # watermark=0 keeps admission arithmetic exact in tests; the reserve is
    # exercised separately.
    return BlockManager(num_blocks=8, block_size=BLOCK_SIZE, watermark=0.0)


class TestBlockTable:
    def test_slot_arithmetic(self):
        table = BlockTable(block_size=4, blocks=[5, 2])
        # Logical 0..3 live in block 5, logical 4..7 in block 2.
        assert table.slot(0) == 5 * 4 + 0
        assert table.slot(3) == 5 * 4 + 3
        assert table.slot(4) == 2 * 4 + 0
        assert table.slot(7) == 2 * 4 + 3

    def test_slots_returns_a_consecutive_run(self):
        table = BlockTable(block_size=4, blocks=[1, 0])
        assert table.slots(2, 4) == [6, 7, 0, 1]

    def test_slot_beyond_capacity_is_an_error(self):
        table = BlockTable(block_size=4, blocks=[0])
        with pytest.raises(IndexError, match="needs block 1"):
            table.slot(4)

    def test_negative_position_is_an_error(self):
        with pytest.raises(IndexError, match="negative"):
            BlockTable(block_size=4, blocks=[0]).slot(-1)

    def test_blocks_needed_rounds_up(self):
        table = BlockTable(block_size=16)
        assert table.blocks_needed(0) == 0
        assert table.blocks_needed(1) == 1
        assert table.blocks_needed(16) == 1
        assert table.blocks_needed(17) == 2

    def test_copy_shares_physical_blocks_but_not_the_list(self):
        original = BlockTable(block_size=4, blocks=[3, 7])
        clone = original.copy()
        assert clone.blocks == [3, 7]
        clone.append(1)
        assert original.blocks == [3, 7]

    def test_physical_block_is_free_when_unreferenced(self):
        assert PhysicalBlock(block_id=0).is_free
        assert not PhysicalBlock(block_id=0, ref_count=1).is_free


class TestAllocation:
    def test_allocates_ceil_blocks(self, manager):
        table, _ = manager.allocate(seq_id=1, num_tokens=5)  # 5 tokens, block 4
        assert len(table) == 2
        assert manager.num_free_blocks == 6

    def test_zero_token_sequence_takes_nothing(self, manager):
        table, _ = manager.allocate(1, 0)
        assert len(table) == 0
        assert manager.num_free_blocks == 8

    def test_double_allocation_is_an_error(self, manager):
        manager.allocate(1, 4)
        with pytest.raises(ValueError, match="already allocated"):
            manager.allocate(1, 4)

    def test_allocating_more_than_exists_raises(self, manager):
        with pytest.raises(MemoryError, match="need 9 more blocks"):
            manager.allocate(1, 33)

    def test_blocks_are_not_handed_out_twice(self, manager):
        first = set(manager.allocate(1, 16)[0].blocks)
        second = set(manager.allocate(2, 16)[0].blocks)
        assert first.isdisjoint(second)
        assert manager.num_free_blocks == 0


class TestCanAllocate:
    def test_ok_when_there_is_room(self, manager):
        assert manager.can_allocate(8) is AllocStatus.OK

    def test_later_when_the_cache_is_full_right_now(self, manager):
        manager.allocate(1, 32)  # takes all 8 blocks
        assert manager.can_allocate(4) is AllocStatus.LATER

    def test_never_when_it_could_not_fit_in_an_empty_cache(self, manager):
        # The scheduler must reject this outright rather than queue it forever.
        assert manager.can_allocate(33) is AllocStatus.NEVER

    def test_watermark_reserves_headroom_for_running_sequences(self):
        # Admitting a request that leaves zero headroom means the next decode
        # step immediately preempts something, and thrash costs more than the
        # admission gained.
        reserved = BlockManager(num_blocks=100, block_size=4, watermark=0.10)
        assert reserved.watermark_blocks == 10
        assert reserved.can_allocate(4 * 90) is AllocStatus.OK
        assert reserved.can_allocate(4 * 91) is AllocStatus.LATER


class TestAppendSlot:
    def test_returns_none_while_the_current_block_has_room(self, manager):
        manager.allocate(1, 1)
        # block_size is 4, so lengths 2, 3, 4 all fit in the block already held.
        assert manager.append_slot(1, 2) is None
        assert manager.append_slot(1, 3) is None
        assert manager.append_slot(1, 4) is None
        assert manager.num_free_blocks == 7

    def test_allocates_exactly_on_a_boundary_crossing(self, manager):
        manager.allocate(1, 4)
        assert manager.num_free_blocks == 7
        block_id = manager.append_slot(1, 5)
        assert block_id is not None
        assert manager.num_free_blocks == 6

    def test_touches_the_allocator_once_per_block_size_tokens(self, manager):
        """The amortisation that makes per-step allocation affordable."""
        manager.allocate(1, 1)
        allocations = sum(
            1 for length in range(2, 17) if manager.append_slot(1, length) is not None
        )
        # Lengths 2..16 cross a boundary at 5, 9, 13 -- three times in fifteen
        # steps, not fifteen.
        assert allocations == 3

    def test_raises_when_the_cache_is_exhausted(self, manager):
        manager.allocate(1, 32)
        with pytest.raises(MemoryError, match="none free"):
            manager.append_slot(1, 33)

    def test_unknown_sequence_is_an_error(self, manager):
        with pytest.raises(KeyError, match="no block table"):
            manager.append_slot(99, 4)


class TestFree:
    def test_returns_blocks_to_the_free_list(self, manager):
        manager.allocate(1, 16)
        assert manager.num_free_blocks == 4
        manager.free(1)
        assert manager.num_free_blocks == 8

    def test_freeing_an_unknown_sequence_is_a_no_op(self, manager):
        # Retirement and preemption can race to free the same sequence; making
        # that an error turns a benign double-free into a crashed benchmark.
        manager.free(12345)
        assert manager.num_free_blocks == 8

    def test_double_free_is_harmless(self, manager):
        manager.allocate(1, 8)
        manager.free(1)
        manager.free(1)
        assert manager.num_free_blocks == 8

    def test_freed_blocks_can_be_reallocated(self, manager):
        manager.allocate(1, 32)
        manager.free(1)
        manager.allocate(2, 32)
        assert manager.num_free_blocks == 0


class TestFork:
    def test_child_shares_the_parent_blocks(self, manager):
        parent, _ = manager.allocate(1, 8)
        before = manager.num_free_blocks
        child = manager.fork(1, 2)
        # Sharing costs no new blocks -- the point of refcounting.
        assert manager.num_free_blocks == before
        assert child.blocks == parent.blocks

    def test_refcounts_rise_on_fork(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        for block_id in manager.block_table(1):
            assert manager.blocks[block_id].ref_count == 2

    def test_freeing_one_side_keeps_the_other_alive(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        manager.free(1)
        assert manager.num_free_blocks == 6  # still held by the child
        for block_id in manager.block_table(2):
            assert manager.blocks[block_id].ref_count == 1

    def test_freeing_both_sides_releases_the_blocks(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        manager.free(1)
        manager.free(2)
        assert manager.num_free_blocks == 8

    def test_forking_onto_a_live_id_is_an_error(self, manager):
        manager.allocate(1, 4)
        manager.allocate(2, 4)
        with pytest.raises(ValueError, match="already allocated"):
            manager.fork(1, 2)


class TestInvariants:
    """The four invariants from AGENTS.md §5."""

    def test_hold_on_a_fresh_manager(self, manager):
        manager.check_invariants()

    def test_hold_through_allocate_append_and_free(self, manager):
        manager.allocate(1, 6)
        manager.allocate(2, 3)
        manager.append_slot(1, 9)
        manager.check_invariants()
        manager.free(1)
        manager.check_invariants()
        manager.free(2)
        manager.check_invariants()
        assert manager.num_free_blocks == 8

    def test_hold_across_a_fork(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        manager.check_invariants()
        manager.free(1)
        manager.check_invariants()

    def test_block_table_length_is_checked_against_sequence_length(self, manager):
        manager.allocate(1, 5)
        manager.check_invariants({1: 5})
        # Claiming the sequence is longer than its table can hold is exactly the
        # bug an append_slot that silently did not grow would cause.
        with pytest.raises(AssertionError, match="expected 3"):
            manager.check_invariants({1: 9})

    def test_detects_a_leaked_block(self, manager):
        manager.allocate(1, 4)
        # Simulate a free path that dropped the table without returning blocks.
        manager.block_tables.pop(1)
        with pytest.raises(AssertionError, match="!= num_blocks"):
            manager.check_invariants()

    def test_detects_a_refcount_that_drifted(self, manager):
        manager.allocate(1, 4)
        manager.blocks[manager.block_table(1).blocks[0]].ref_count = 5
        with pytest.raises(AssertionError, match="refcount is 5"):
            manager.check_invariants()

    def test_detects_a_block_that_is_free_and_held_at_once(self, manager):
        manager.allocate(1, 4)
        manager.free_blocks.append(manager.block_table(1).blocks[0])
        with pytest.raises(AssertionError, match="also on the free list"):
            manager.check_invariants()

    def test_detects_a_held_block_with_refcount_zero(self, manager):
        manager.allocate(1, 4)
        held = manager.block_table(1).blocks[0]
        manager.blocks[held].ref_count = 0
        with pytest.raises(AssertionError, match="refcount"):
            manager.check_invariants()


class TestFragmentation:
    def test_waste_is_bounded_by_block_size_minus_one(self):
        """The headline property of paging, checked directly.

        A contiguous cache wastes up to ``max_seq_len - actual_len`` per
        sequence. A paged one wastes at most ``block_size - 1``, and only in the
        last block.
        """
        manager = BlockManager(num_blocks=1000, block_size=16, watermark=0.0)
        for seq_id, length in enumerate([1, 15, 16, 17, 100, 129]):
            manager.allocate(seq_id, length)
            capacity = manager.block_table(seq_id).capacity
            assert 0 <= capacity - length <= 15

    def test_scattered_blocks_still_serve_a_contiguous_sequence(self):
        """Fragmentation in the cache does not block admission.

        This is why continuous batching needs paging: a new sequence needs *k*
        blocks from a free list, any *k*, not a contiguous hole.
        """
        manager = BlockManager(num_blocks=8, block_size=4, watermark=0.0)
        manager.allocate(1, 4)
        manager.allocate(2, 4)
        manager.allocate(3, 4)
        manager.allocate(4, 4)
        # Free alternating blocks, leaving the free list badly fragmented.
        manager.free(1)
        manager.free(3)
        assert manager.num_free_blocks == 6
        # A 24-token sequence still gets admitted from the scattered remainder.
        table, _ = manager.allocate(5, 24)
        assert len(table) == 6
        manager.check_invariants()


class TestReset:
    def test_returns_everything(self, manager):
        manager.allocate(1, 16)
        manager.allocate(2, 8)
        manager.reset()
        assert manager.num_free_blocks == 8
        assert manager.block_tables == {}
        manager.check_invariants()


class TestConstruction:
    def test_rejects_nonsense_sizes(self):
        with pytest.raises(ValueError, match="num_blocks"):
            BlockManager(num_blocks=0, block_size=16)
        with pytest.raises(ValueError, match="block_size"):
            BlockManager(num_blocks=16, block_size=0)

    def test_cache_holds_one_more_block_than_is_allocatable(self, manager):
        # The trash block that left-padding tokens write into.
        assert manager.num_cache_blocks == manager.num_blocks + 1
        assert manager.trash_block_id == manager.num_blocks


class TestCopyOnWrite:
    """Giving a sequence a private copy of a block it shares.

    Not reachable from prefix caching alone: only *full* blocks are cached and a
    full block is never written to again. It exists for fork(), where n>1
    sampling shares a partially filled block between samples that then diverge —
    without the copy, the second writer would overwrite the first one's tokens
    in place.
    """

    def test_exclusive_block_needs_no_copy(self, manager):
        manager.allocate(1, 8)
        assert manager.copy_on_write(1, 0) is None

    def test_shared_block_is_copied(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        original = manager.block_table(2).blocks[0]

        result = manager.copy_on_write(2, 0)
        assert result is not None
        source, destination = result
        assert source == original
        assert destination != original
        # The child now points at its own block; the parent is untouched.
        assert manager.block_table(2).blocks[0] == destination
        assert manager.block_table(1).blocks[0] == source

    def test_the_source_loses_a_reference(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        source = manager.block_table(1).blocks[0]
        assert manager.blocks[source].ref_count == 2
        manager.copy_on_write(2, 0)
        assert manager.blocks[source].ref_count == 1

    def test_the_copy_is_exclusively_owned(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        _, destination = manager.copy_on_write(2, 0)
        assert manager.blocks[destination].ref_count == 1

    def test_copying_twice_is_a_no_op_the_second_time(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        manager.copy_on_write(2, 0)
        # Now exclusive, so nothing more to do.
        assert manager.copy_on_write(2, 0) is None

    def test_the_copy_is_not_reachable_as_a_cache_hit(self):
        """Its contents are about to diverge from the prefix its hash names."""
        from pagedserve.memory.prefix_cache import chain_hash

        manager = BlockManager(8, BLOCK_SIZE, watermark=0.0, enable_prefix_caching=True)
        manager.allocate(1, BLOCK_SIZE, token_ids=list(range(BLOCK_SIZE)))
        manager.seal(1, list(range(BLOCK_SIZE)), BLOCK_SIZE)
        block_hash = chain_hash(None, list(range(BLOCK_SIZE)))
        assert manager.prefix_cache.lookup(block_hash) is not None

        manager.fork(1, 2)
        _, destination = manager.copy_on_write(2, 0)
        assert not manager.prefix_cache.is_cached(destination)

    def test_invariants_hold_after_a_copy(self, manager):
        manager.allocate(1, 8)
        manager.fork(1, 2)
        manager.copy_on_write(2, 0)
        manager.check_invariants()
        manager.free(1)
        manager.free(2)
        manager.check_invariants()
        assert manager.num_free_blocks == 8

    def test_raises_when_there_is_nothing_to_copy_into(self):
        manager = BlockManager(2, BLOCK_SIZE, watermark=0.0)
        manager.allocate(1, BLOCK_SIZE * 2)  # takes both blocks
        manager.fork(1, 2)
        with pytest.raises(MemoryError):
            manager.copy_on_write(2, 0)


class TestPrefixReuse:
    """Allocator-level prefix caching: reuse, sealing, and eviction."""

    def cache_manager(self, num_blocks: int = 16) -> BlockManager:
        return BlockManager(num_blocks, BLOCK_SIZE, watermark=0.0, enable_prefix_caching=True)

    def test_nothing_is_reused_before_anything_is_sealed(self):
        """Sealing follows the forward pass, so a first arrival cannot hit.

        Two requests admitted in the same step cannot share either: no KV
        exists yet for the blocks they would share.
        """
        manager = self.cache_manager()
        tokens = list(range(BLOCK_SIZE * 2))
        _, cached = manager.allocate(1, len(tokens), token_ids=tokens)
        assert cached == 0

    def test_a_later_request_reuses_a_sealed_prefix(self):
        manager = self.cache_manager()
        tokens = list(range(BLOCK_SIZE * 2))
        manager.allocate(1, len(tokens), token_ids=tokens)
        manager.seal(1, tokens, len(tokens))

        _, cached = manager.allocate(2, len(tokens), token_ids=tokens)
        # One token is held back so the forward pass has work to do.
        assert cached == len(tokens) - 1
        assert manager.block_table(2).blocks == manager.block_table(1).blocks

    def test_reuse_shares_blocks_rather_than_allocating(self):
        manager = self.cache_manager()
        tokens = list(range(BLOCK_SIZE * 2))
        manager.allocate(1, len(tokens), token_ids=tokens)
        manager.seal(1, tokens, len(tokens))
        before = manager.num_free_blocks
        manager.allocate(2, len(tokens), token_ids=tokens)
        assert manager.num_free_blocks == before
        manager.check_invariants()

    def test_a_divergent_prompt_reuses_only_the_shared_prefix(self):
        manager = self.cache_manager()
        shared = list(range(BLOCK_SIZE))
        first = shared + [100] * BLOCK_SIZE
        manager.allocate(1, len(first), token_ids=first)
        manager.seal(1, first, len(first))

        second = shared + [200] * BLOCK_SIZE
        _, cached = manager.allocate(2, len(second), token_ids=second)
        assert cached == BLOCK_SIZE
        assert manager.block_table(2).blocks[0] == manager.block_table(1).blocks[0]
        assert manager.block_table(2).blocks[1] != manager.block_table(1).blocks[1]

    def test_freed_blocks_stay_cached_rather_than_going_free(self):
        manager = self.cache_manager()
        tokens = list(range(BLOCK_SIZE * 2))
        manager.allocate(1, len(tokens), token_ids=tokens)
        manager.seal(1, tokens, len(tokens))
        manager.free(1)
        # Still obtainable, but reclaimable rather than on the free list.
        assert manager.prefix_cache.num_reclaimable == 2
        assert len(manager.free_blocks) == 14
        manager.check_invariants()

    def test_a_reclaimable_block_can_still_be_reused(self):
        manager = self.cache_manager()
        tokens = list(range(BLOCK_SIZE * 2))
        manager.allocate(1, len(tokens), token_ids=tokens)
        manager.seal(1, tokens, len(tokens))
        original = list(manager.block_table(1))
        manager.free(1)
        manager.allocate(2, len(tokens), token_ids=tokens)
        assert list(manager.block_table(2)) == original

    def test_pressure_reclaims_the_least_recently_used(self):
        manager = self.cache_manager(num_blocks=4)
        for seq_id in range(2):
            tokens = [seq_id * 1000 + i for i in range(BLOCK_SIZE * 2)]
            manager.allocate(seq_id, len(tokens), token_ids=tokens)
            manager.seal(seq_id, tokens, len(tokens))
            manager.free(seq_id)
        assert manager.prefix_cache.num_reclaimable == 4

        # A wholly new prompt must evict rather than fail.
        fresh = [9999 + i for i in range(BLOCK_SIZE * 2)]
        manager.allocate(9, len(fresh), token_ids=fresh)
        assert manager.prefix_cache.stats.evictions > 0
        manager.check_invariants()

    def test_caching_off_never_reuses(self):
        manager = BlockManager(16, BLOCK_SIZE, watermark=0.0, enable_prefix_caching=False)
        tokens = list(range(BLOCK_SIZE * 2))
        manager.allocate(1, len(tokens), token_ids=tokens)
        manager.seal(1, tokens, len(tokens))
        _, cached = manager.allocate(2, len(tokens), token_ids=tokens)
        assert cached == 0
        assert manager.block_table(2).blocks != manager.block_table(1).blocks
