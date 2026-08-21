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
        table = manager.allocate(seq_id=1, num_tokens=5)  # 5 tokens, block 4
        assert len(table) == 2
        assert manager.num_free_blocks == 6

    def test_zero_token_sequence_takes_nothing(self, manager):
        assert len(manager.allocate(1, 0)) == 0
        assert manager.num_free_blocks == 8

    def test_double_allocation_is_an_error(self, manager):
        manager.allocate(1, 4)
        with pytest.raises(ValueError, match="already allocated"):
            manager.allocate(1, 4)

    def test_allocating_more_than_exists_raises(self, manager):
        with pytest.raises(MemoryError, match="need 9 blocks"):
            manager.allocate(1, 33)

    def test_blocks_are_not_handed_out_twice(self, manager):
        first = set(manager.allocate(1, 16).blocks)
        second = set(manager.allocate(2, 16).blocks)
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
        parent = manager.allocate(1, 8)
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
        table = manager.allocate(5, 24)
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
