"""Unit tests for the Phase 2 paged attention backend.

Random tensors, no model. The decisive property is that a paged read reproduces
a dense read exactly — same tokens, different memory layout — so several of
these diff the two backends directly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pagedserve.attention.backend import StepInput  # noqa: E402
from pagedserve.attention.contiguous import ContiguousAttentionBackend  # noqa: E402
from pagedserve.attention.gather import PagedAttentionBackend  # noqa: E402
from pagedserve.config import CacheConfig, ModelConfig  # noqa: E402
from pagedserve.memory.block_manager import BlockManager  # noqa: E402

BLOCK_SIZE = 4
NUM_BLOCKS = 16

TINY = ModelConfig(
    name="tiny",
    num_layers=2,
    hidden_size=32,
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=8,
    intermediate_size=64,
    vocab_size=100,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=128,
    tie_word_embeddings=False,
    attention_bias=False,
)
CACHE = CacheConfig(max_seq_len=64, max_num_seqs=4, block_size=BLOCK_SIZE)


@pytest.fixture
def backend():
    b = PagedAttentionBackend(
        TINY,
        CACHE,
        num_blocks=NUM_BLOCKS,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    b.allocate()
    return b


@pytest.fixture
def manager():
    return BlockManager(NUM_BLOCKS, BLOCK_SIZE, watermark=0.0)


def paged_step(manager, seq_lens, query_len, logical_starts, query_positions):
    """Build a StepInput the way the engine does, for one batch."""
    num_seqs = len(seq_lens)
    slots = []
    for seq_id in range(num_seqs):
        logical = logical_starts[seq_id]
        table = manager.block_table(seq_id)
        for _ in range(query_len):
            slots.append(table.slot(logical))
            logical += 1
    max_blocks = max(len(manager.block_table(i)) for i in range(num_seqs))
    tables = [
        list(manager.block_table(i))
        + [manager.trash_block_id] * (max_blocks - len(manager.block_table(i)))
        for i in range(num_seqs)
    ]
    return StepInput(
        query_len=query_len,
        context_len=0,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        padding_mask=torch.ones((num_seqs, max(seq_lens)), dtype=torch.bool),
        query_positions=torch.tensor(query_positions, dtype=torch.long),
        is_prefill=query_len > 1,
        block_tables=torch.tensor(tables, dtype=torch.long),
        slot_mapping=torch.tensor(slots, dtype=torch.long),
    )


def qkv(num_seqs, query_len):
    return (
        torch.randn(num_seqs, query_len, TINY.num_q_heads, TINY.head_dim),
        torch.randn(num_seqs, query_len, TINY.num_kv_heads, TINY.head_dim),
        torch.randn(num_seqs, query_len, TINY.num_kv_heads, TINY.head_dim),
    )


class TestAllocation:
    def test_cache_has_one_extra_block_for_padding(self, backend):
        # [layers, 2, num_blocks + 1, block_size, kv_heads, head_dim]
        assert backend.kv_cache.shape == (2, 2, NUM_BLOCKS + 1, BLOCK_SIZE, 2, 8)

    def test_trash_block_matches_the_allocator(self, backend, manager):
        # A mismatch would scribble padding K/V into a real sequence's last
        # block, and the symptom would be wrong tokens, not a crash.
        assert backend.trash_block_id == manager.trash_block_id

    def test_begin_step_before_allocate_is_an_error(self):
        b = PagedAttentionBackend(
            TINY, CACHE, num_blocks=4, device=torch.device("cpu"), dtype=torch.float32
        )
        with pytest.raises(RuntimeError, match="allocate"):
            b.begin_step(
                StepInput(
                    query_len=1,
                    context_len=0,
                    seq_lens=torch.tensor([1]),
                    padding_mask=torch.ones((1, 1), dtype=torch.bool),
                    is_prefill=False,
                )
            )

    def test_requires_paged_fields_in_the_step(self, backend):
        bare = StepInput(
            query_len=1,
            context_len=0,
            seq_lens=torch.tensor([1]),
            padding_mask=torch.ones((1, 1), dtype=torch.bool),
            is_prefill=False,
        )
        with pytest.raises(ValueError, match="slot_mapping and block_tables"):
            backend.begin_step(bare)

    def test_free_releases_the_cache(self, backend):
        backend.free()
        assert backend.kv_cache is None
        assert backend.cache_bytes() == 0


class TestWritePath:
    def test_scatters_tokens_to_their_blocks(self, backend, manager):
        manager.allocate(0, 6)  # 6 tokens -> 2 blocks
        step = paged_step(manager, [6], 6, [0], [[0, 1, 2, 3, 4, 5]])
        md = backend.begin_step(step)
        q, k, v = qkv(1, 6)
        backend.forward(0, q, k, v, md)

        table = manager.block_table(0)
        for logical in range(6):
            block = table.blocks[logical // BLOCK_SIZE]
            offset = logical % BLOCK_SIZE
            assert torch.equal(backend.kv_cache[0, 0, block, offset], k[0, logical])
            assert torch.equal(backend.kv_cache[0, 1, block, offset], v[0, logical])

    def test_non_contiguous_blocks_still_hold_a_contiguous_sequence(self, backend, manager):
        """The indirection working: scattered frames, one logical sequence."""
        # Genuinely fragment the free list: take every block, then release
        # alternating ones. Freed blocks return to the back of the queue, so
        # simply freeing a couple early would still hand out adjacent blocks.
        for filler in range(100, 100 + NUM_BLOCKS):
            manager.allocate(filler, BLOCK_SIZE)
        for filler in range(100, 100 + NUM_BLOCKS, 2):
            manager.free(filler)
        manager.allocate(0, 8)
        blocks = manager.block_table(0).blocks
        assert blocks[1] != blocks[0] + 1  # genuinely scattered

        step = paged_step(manager, [8], 8, [0], [list(range(8))])
        md = backend.begin_step(step)
        q, k, v = qkv(1, 8)
        backend.forward(0, q, k, v, md)
        for logical in range(8):
            block = blocks[logical // BLOCK_SIZE]
            assert torch.equal(backend.kv_cache[0, 0, block, logical % BLOCK_SIZE], k[0, logical])

    def test_padding_lands_in_the_trash_block(self, backend, manager):
        manager.allocate(0, 4)
        trash_slot = manager.trash_block_id * BLOCK_SIZE
        table = manager.block_table(0)
        step = StepInput(
            query_len=3,
            context_len=0,
            seq_lens=torch.tensor([2], dtype=torch.int32),
            padding_mask=torch.tensor([[False, True, True]]),
            query_positions=torch.tensor([[0, 0, 1]]),
            is_prefill=True,
            block_tables=torch.tensor([list(table)], dtype=torch.long),
            slot_mapping=torch.tensor([trash_slot, table.slot(0), table.slot(1)]),
        )
        md = backend.begin_step(step)
        q, k, v = qkv(1, 3)
        backend.forward(0, q, k, v, md)
        # The pad's K went to the trash block, not into the sequence.
        assert torch.equal(backend.kv_cache[0, 0, manager.trash_block_id, 0], k[0, 0])
        assert torch.equal(backend.kv_cache[0, 0, table.blocks[0], 0], k[0, 1])

    def test_layers_have_separate_caches(self, backend, manager):
        manager.allocate(0, 4)
        step = paged_step(manager, [4], 4, [0], [[0, 1, 2, 3]])
        md = backend.begin_step(step)
        q, k0, v0 = qkv(1, 4)
        _, k1, v1 = qkv(1, 4)
        backend.forward(0, q, k0, v0, md)
        backend.forward(1, q, k1, v1, md)
        block = manager.block_table(0).blocks[0]
        assert torch.equal(backend.kv_cache[0, 0, block, 0], k0[0, 0])
        assert torch.equal(backend.kv_cache[1, 0, block, 0], k1[0, 0])


class TestReadPath:
    def test_masks_slots_past_the_end_of_the_sequence(self, backend, manager):
        """The gathered buffer is a whole number of blocks.

        It therefore exposes up to ``block_size - 1`` slots past the sequence,
        holding stale KV from a previous tenant. Failing to mask them means one
        request attends to another's data.
        """
        manager.allocate(0, 5)  # 5 tokens in 2 blocks -> 8 gathered slots
        step = paged_step(manager, [5], 5, [0], [[0, 1, 2, 3, 4]])
        md = backend.begin_step(step)
        assert md.gathered_len == 8
        neg = torch.finfo(torch.float32).min
        # The last query may see keys 0..4 and must not see 5..7.
        assert (md.attn_bias[0, 0, 4, :5] == 0).all()
        assert (md.attn_bias[0, 0, 4, 5:] == neg).all()

    def test_is_causal_in_logical_space(self, backend, manager):
        manager.allocate(0, 4)
        step = paged_step(manager, [4], 4, [0], [[0, 1, 2, 3]])
        md = backend.begin_step(step)
        neg = torch.finfo(torch.float32).min
        assert md.attn_bias[0, 0, 0, 0] == 0
        assert md.attn_bias[0, 0, 0, 1] == neg
        assert (md.attn_bias[0, 0, 3, :4] == 0).all()

    def test_matches_the_dense_backend_exactly(self, manager):
        """The Phase 2 assertion: a memory layout change must not alter output."""
        torch.manual_seed(0)
        num_seqs, seq_len = 2, 6
        q, k, v = qkv(num_seqs, seq_len)

        dense = ContiguousAttentionBackend(
            TINY,
            CacheConfig(max_seq_len=16, max_num_seqs=num_seqs, block_size=BLOCK_SIZE),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        dense.allocate()
        dense_md = dense.begin_step(
            StepInput(
                query_len=seq_len,
                context_len=0,
                seq_lens=torch.tensor([seq_len] * num_seqs, dtype=torch.int32),
                padding_mask=torch.ones((num_seqs, seq_len), dtype=torch.bool),
                is_prefill=True,
            )
        )
        dense_out = dense.forward(0, q, k, v, dense_md)

        paged = PagedAttentionBackend(
            TINY, CACHE, num_blocks=NUM_BLOCKS, device=torch.device("cpu"), dtype=torch.float32
        )
        paged.allocate()
        for seq_id in range(num_seqs):
            manager.allocate(seq_id, seq_len)
        paged_md = paged.begin_step(
            paged_step(
                manager,
                [seq_len] * num_seqs,
                seq_len,
                [0] * num_seqs,
                [list(range(seq_len))] * num_seqs,
            )
        )
        paged_out = paged.forward(0, q, k, v, paged_md)
        assert torch.allclose(dense_out, paged_out, atol=1e-6)

    def test_decode_reads_back_what_prefill_wrote(self, backend, manager):
        manager.allocate(0, 3)
        md = backend.begin_step(paged_step(manager, [3], 3, [0], [[0, 1, 2]]))
        q, k, v = qkv(1, 3)
        backend.forward(0, q, k, v, md)

        manager.append_slot(0, 4)
        md2 = backend.begin_step(paged_step(manager, [4], 1, [3], [[3]]))
        q2, k2, v2 = qkv(1, 1)
        out = backend.forward(0, q2, k2, v2, md2)
        assert out.shape == (1, 1, TINY.num_q_heads, TINY.head_dim)
        block = manager.block_table(0).blocks[0]
        assert torch.equal(backend.kv_cache[0, 0, block, 0], k[0, 0])
        assert torch.equal(backend.kv_cache[0, 0, block, 3], k2[0, 0])


class TestMemoryStats:
    def test_counts_only_held_blocks_as_allocated(self, backend, manager):
        """An unheld block is genuinely available to the next arrival.

        That is the claim a contiguous reservation cannot make about its unused
        tail, and it is the Phase 2 result.
        """
        manager.allocate(0, 5)  # 2 blocks held out of 16
        backend.set_used_blocks(manager.num_used_blocks)
        backend.begin_step(paged_step(manager, [5], 5, [0], [[0, 1, 2, 3, 4]]))
        stats = backend.memory_stats()
        per_token = TINY.kv_bytes_per_token(torch.float32)
        assert stats.allocated_bytes == 2 * BLOCK_SIZE * per_token
        assert stats.live_bytes == 5 * per_token
        assert stats.utilization == pytest.approx(5 / 8)

    def test_waste_is_bounded_by_the_block_size(self, backend, manager):
        for length in (1, 4, 5, 13):
            manager.reset()
            manager.allocate(0, length)
            backend.set_used_blocks(manager.num_used_blocks)
            backend.begin_step(paged_step(manager, [length], length, [0], [list(range(length))]))
            stats = backend.memory_stats()
            per_token = TINY.kv_bytes_per_token(torch.float32)
            assert 0 <= stats.wasted_bytes / per_token <= BLOCK_SIZE - 1

    def test_reports_nothing_when_unallocated(self):
        b = PagedAttentionBackend(
            TINY, CACHE, num_blocks=4, device=torch.device("cpu"), dtype=torch.float32
        )
        assert b.memory_stats().allocated_bytes == 0
        assert b.memory_stats().utilization is None
