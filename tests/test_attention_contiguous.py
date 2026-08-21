"""Unit tests for the Phase 1 contiguous attention backend.

Runs on random tensors with no model involved, so these are fast and isolate
the cache mechanics from the transformer's numerics. The golden test covers the
two together.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pagedserve.attention.backend import KVMemoryStats, StepInput  # noqa: E402
from pagedserve.attention.contiguous import ContiguousAttentionBackend  # noqa: E402
from pagedserve.config import ModelConfig  # noqa: E402

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


@pytest.fixture
def backend():
    return ContiguousAttentionBackend(TINY, device=torch.device("cpu"), dtype=torch.float32)


def step(num_seqs=2, query_len=4, context_len=0, seq_lens=None, mask=None, prefill=True):
    key_len = context_len + query_len
    if mask is None:
        mask = torch.ones((num_seqs, key_len), dtype=torch.bool)
    if seq_lens is None:
        seq_lens = torch.full((num_seqs,), key_len, dtype=torch.int32)
    return StepInput(
        query_len=query_len,
        context_len=context_len,
        seq_lens=seq_lens,
        padding_mask=mask,
        is_prefill=prefill,
    )


def qkv(num_seqs=2, query_len=4):
    q = torch.randn(num_seqs, query_len, TINY.num_q_heads, TINY.head_dim)
    k = torch.randn(num_seqs, query_len, TINY.num_kv_heads, TINY.head_dim)
    v = torch.randn(num_seqs, query_len, TINY.num_kv_heads, TINY.head_dim)
    return q, k, v


class TestAllocation:
    def test_allocates_the_full_worst_case_shape(self, backend):
        backend.allocate(num_seq_slots=4, max_seq_len=64)
        assert backend.kv_cache.shape == (2, 2, 4, 64, 2, 8)

    def test_forward_before_allocate_is_an_error(self, backend):
        with pytest.raises(RuntimeError, match="allocate"):
            backend.begin_step(step())

    def test_free_is_idempotent(self, backend):
        backend.allocate(2, 16)
        backend.free()
        backend.free()
        assert backend.kv_cache is None

    def test_reallocating_replaces_the_old_cache(self, backend):
        backend.allocate(2, 16)
        backend.allocate(4, 32)
        assert backend.kv_cache.shape[2] == 4
        assert backend.kv_cache.shape[3] == 32

    def test_rejects_a_sequence_longer_than_the_cache(self, backend):
        backend.allocate(2, 8)
        with pytest.raises(ValueError, match="exceeds"):
            backend.begin_step(step(query_len=16))


class TestMask:
    def test_is_causal_within_a_prefill(self, backend):
        backend.allocate(2, 16)
        md = backend.begin_step(step(num_seqs=2, query_len=4))
        bias = md.attn_bias[0, 0]
        neg = torch.finfo(torch.float32).min
        # Query 0 sees only key 0; query 3 sees keys 0..3.
        assert bias[0, 0] == 0 and bias[0, 1] == neg
        assert (bias[3, :4] == 0).all()

    def test_masks_left_padding(self, backend):
        backend.allocate(2, 16)
        # Sequence 0 has two pad slots at the front.
        mask = torch.ones((2, 4), dtype=torch.bool)
        mask[0, :2] = False
        md = backend.begin_step(step(num_seqs=2, query_len=4, mask=mask))
        neg = torch.finfo(torch.float32).min
        # Even the last query of the padded sequence must not see the pads.
        assert (md.attn_bias[0, 0, 3, :2] == neg).all()
        assert (md.attn_bias[0, 0, 3, 2:] == 0).all()
        # The unpadded sequence sees everything up to itself.
        assert (md.attn_bias[1, 0, 3, :] == 0).all()

    def test_decode_sees_the_whole_prefix(self, backend):
        backend.allocate(2, 16)
        md = backend.begin_step(step(query_len=1, context_len=5, prefill=False))
        assert md.attn_bias.shape == (2, 1, 1, 6)
        assert (md.attn_bias == 0).all()

    def test_uses_finfo_min_not_negative_infinity(self, backend):
        # A fully masked row would softmax -inf into NaN and poison the whole
        # tensor; a large negative degrades to a uniform row instead.
        backend.allocate(1, 16)
        mask = torch.zeros((1, 2), dtype=torch.bool)
        md = backend.begin_step(step(num_seqs=1, query_len=2, mask=mask))
        assert torch.isfinite(md.attn_bias).all()


class TestForward:
    def test_output_shape(self, backend):
        backend.allocate(2, 16)
        md = backend.begin_step(step())
        q, k, v = qkv()
        out = backend.forward(0, q, k, v, md)
        assert out.shape == (2, 4, TINY.num_q_heads, TINY.head_dim)

    def test_writes_kv_into_the_cache(self, backend):
        backend.allocate(2, 16)
        md = backend.begin_step(step())
        q, k, v = qkv()
        backend.forward(0, q, k, v, md)
        assert torch.equal(backend.kv_cache[0, 0, :2, :4], k)
        assert torch.equal(backend.kv_cache[0, 1, :2, :4], v)

    def test_layers_do_not_share_cache(self, backend):
        backend.allocate(2, 16)
        md = backend.begin_step(step())
        q, k0, v0 = qkv()
        _, k1, v1 = qkv()
        backend.forward(0, q, k0, v0, md)
        backend.forward(1, q, k1, v1, md)
        assert torch.equal(backend.kv_cache[0, 0, :2, :4], k0)
        assert torch.equal(backend.kv_cache[1, 0, :2, :4], k1)

    def test_decode_reads_back_what_prefill_wrote(self, backend):
        backend.allocate(1, 16)
        md = backend.begin_step(step(num_seqs=1, query_len=3))
        q, k, v = qkv(num_seqs=1, query_len=3)
        backend.forward(0, q, k, v, md)

        md2 = backend.begin_step(step(num_seqs=1, query_len=1, context_len=3, prefill=False))
        q2, k2, v2 = qkv(num_seqs=1, query_len=1)
        backend.forward(0, q2, k2, v2, md2)
        # The prefill's keys survived and the new key landed after them.
        assert torch.equal(backend.kv_cache[0, 0, :1, :3], k)
        assert torch.equal(backend.kv_cache[0, 0, :1, 3:4], k2)

    def test_gqa_matches_a_manual_head_expansion(self, backend):
        """Each KV head must serve exactly num_queries_per_kv query heads."""
        import torch.nn.functional as F

        backend.allocate(1, 16)
        md = backend.begin_step(step(num_seqs=1, query_len=4))
        q, k, v = qkv(num_seqs=1, query_len=4)
        out = backend.forward(0, q, k, v, md)

        repeats = TINY.num_queries_per_kv
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2).repeat_interleave(repeats, dim=1),
            v.transpose(1, 2).repeat_interleave(repeats, dim=1),
            attn_mask=md.attn_bias,
        ).transpose(1, 2)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_padding_does_not_leak_into_the_output(self, backend):
        """A padded sequence must produce the same result as an unpadded one.

        This is the failure mode that produces fluent, plausible, wrong text
        rather than a crash, so it is worth an explicit test.
        """
        torch.manual_seed(0)
        q, k, v = qkv(num_seqs=1, query_len=3)

        backend.allocate(1, 16)
        md = backend.begin_step(step(num_seqs=1, query_len=3))
        alone = backend.forward(0, q, k, v, md)

        # Same sequence, now with two pad slots in front.
        pad_q = torch.randn(1, 2, TINY.num_q_heads, TINY.head_dim)
        pad_k = torch.randn(1, 2, TINY.num_kv_heads, TINY.head_dim)
        pad_v = torch.randn(1, 2, TINY.num_kv_heads, TINY.head_dim)
        mask = torch.ones((1, 5), dtype=torch.bool)
        mask[0, :2] = False

        backend.allocate(1, 16)
        md = backend.begin_step(
            step(num_seqs=1, query_len=5, mask=mask, seq_lens=torch.tensor([3]))
        )
        padded = backend.forward(
            0,
            torch.cat([pad_q, q], dim=1),
            torch.cat([pad_k, k], dim=1),
            torch.cat([pad_v, v], dim=1),
            md,
        )
        assert torch.allclose(alone, padded[:, 2:], atol=1e-6)


class TestMemoryStats:
    def test_reports_nothing_when_unallocated(self, backend):
        stats = backend.memory_stats()
        assert stats.allocated_bytes == 0
        assert stats.utilization is None

    def test_utilization_is_null_not_zero_without_a_cache(self):
        # A zero would read as a measurement; there is nothing to measure.
        assert KVMemoryStats(allocated_bytes=0, live_bytes=0).utilization is None

    def test_counts_only_real_tokens_as_live(self, backend):
        backend.allocate(num_seq_slots=2, max_seq_len=100)
        # Two sequences, 10 real tokens each, in a cache sized for 100 apiece.
        backend.begin_step(
            step(num_seqs=2, query_len=10, seq_lens=torch.tensor([10, 10], dtype=torch.int32))
        )
        stats = backend.memory_stats()
        per_token = TINY.kv_bytes_per_token(torch.float32)
        assert stats.live_bytes == 20 * per_token
        assert stats.allocated_bytes == 2 * 100 * per_token
        assert stats.utilization == pytest.approx(0.10)

    def test_padding_is_not_counted_as_live(self, backend):
        """Pad slots occupy cache but hold nothing.

        Counting them would flatter the utilization number this phase exists to
        expose.
        """
        backend.allocate(2, 100)
        mask = torch.ones((2, 10), dtype=torch.bool)
        mask[0, :4] = False  # sequence 0 is 4 tokens of padding, 6 real
        backend.begin_step(
            step(
                num_seqs=2,
                query_len=10,
                mask=mask,
                seq_lens=torch.tensor([6, 10], dtype=torch.int32),
            )
        )
        per_token = TINY.kv_bytes_per_token(torch.float32)
        assert backend.memory_stats().live_bytes == 16 * per_token

    def test_wasted_bytes_is_the_complement(self):
        stats = KVMemoryStats(allocated_bytes=1000, live_bytes=250)
        assert stats.wasted_bytes == 750
        assert stats.utilization == pytest.approx(0.25)

    def test_stats_serialize_for_a_result_file(self):
        d = KVMemoryStats(allocated_bytes=100, live_bytes=25).to_dict()
        assert d["utilization"] == pytest.approx(0.25)
        assert set(d) == {"allocated_bytes", "live_bytes", "wasted_bytes", "utilization"}
