"""Phase 4: the CUDA paged-attention backend and its kernels.

Three layers, each diffed against the one below it:

1. ``paged_reference`` — the kernel's algorithm in PyTorch — against a plain
   masked SDPA over a gathered buffer. Runs everywhere.
2. ``CudaPagedAttentionBackend`` driven by the reference ops against the gather
   oracle on identical ragged batches. Runs everywhere, and covers every line of
   the backend except the kernel launch.
3. The compiled kernels against the reference. ``gpu`` + ``cuda_ext``: skipped
   (not passed) on any machine without a built extension.
"""

from __future__ import annotations

import math
import random

import pytest

torch = pytest.importorskip("torch")

from pagedserve import extension  # noqa: E402
from pagedserve.attention import paged_reference as ref  # noqa: E402
from pagedserve.attention.cuda_paged import (  # noqa: E402
    CudaPagedAttentionBackend,
    DecodeMetadata,
)
from pagedserve.attention.gather import PagedAttentionBackend, RaggedMetadata  # noqa: E402
from pagedserve.config import CacheConfig, EngineConfig, ModelConfig  # noqa: E402

BLOCK_SIZE = 4
NUM_BLOCKS = 32

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
CACHE = CacheConfig(max_seq_len=64, max_num_seqs=8, block_size=BLOCK_SIZE, num_blocks_override=32)

requires_extension = pytest.mark.skipif(
    not (extension.is_available() and torch.cuda.is_available()),
    reason="CUDA extension not built or no GPU. This is a SKIP, not a pass.",
)


def naive_paged_attention(query, key_cache, value_cache, block_tables, seq_lens, scale):
    """Gather every block, mask the tail, softmax. Obviously correct, obviously slow."""
    num_seqs, num_heads, head_dim = query.shape
    block_size, kv_heads = key_cache.shape[1], key_cache.shape[2]
    out = torch.empty_like(query, dtype=torch.float32)
    for i in range(num_seqs):
        n = int(seq_lens[i])
        k = key_cache[block_tables[i]].reshape(-1, kv_heads, head_dim)[:n].float()
        v = value_cache[block_tables[i]].reshape(-1, kv_heads, head_dim)[:n].float()
        k = k.repeat_interleave(num_heads // kv_heads, dim=1)
        v = v.repeat_interleave(num_heads // kv_heads, dim=1)
        scores = torch.einsum("hd,thd->ht", query[i].float(), k) * scale
        out[i] = torch.einsum("ht,thd->hd", scores.softmax(dim=-1), v)
    del block_size
    return out


def random_tables(seq_lens, block_size, num_blocks, *, seed=0):
    """Non-contiguous, shuffled block assignments: the layout paging produces."""
    rng = random.Random(seed)
    pool = list(range(num_blocks))
    rng.shuffle(pool)
    max_blocks = max(math.ceil(n / block_size) for n in seq_lens)
    tables = []
    for n in seq_lens:
        need = math.ceil(n / block_size)
        blocks = [pool.pop() for _ in range(need)]
        tables.append(blocks + [num_blocks - 1] * (max_blocks - need))
    return torch.tensor(tables, dtype=torch.long)


class TestReferenceAlgorithm:
    @pytest.mark.parametrize("block_size", [1, 4, 16])
    def test_matches_naive_attention(self, block_size):
        torch.manual_seed(0)
        seq_lens = [1, block_size, block_size + 1, 3 * block_size - 1, 13]
        num_blocks = 64
        kc = torch.randn(num_blocks, block_size, 2, 8)
        vc = torch.randn(num_blocks, block_size, 2, 8)
        tables = random_tables(seq_lens, block_size, num_blocks - 1)
        lens = torch.tensor(seq_lens, dtype=torch.int32)
        q = torch.randn(len(seq_lens), 4, 8)
        got = ref.paged_attention_v1(q, kc, vc, tables, lens, 8**-0.5)
        want = naive_paged_attention(q, kc, vc, tables, lens, 8**-0.5)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_stale_data_past_the_end_is_never_read(self):
        """Slots past seq_len hold a previous tenant's KV; poisoning them must not matter."""
        torch.manual_seed(1)
        kc = torch.randn(8, 4, 2, 8)
        vc = torch.randn(8, 4, 2, 8)
        tables = torch.tensor([[3, 5]], dtype=torch.long)
        lens = torch.tensor([6], dtype=torch.int32)
        q = torch.randn(1, 4, 8)
        before = ref.paged_attention_v1(q, kc, vc, tables, lens, 0.3)
        kc[5, 2:] = 1e4
        vc[5, 2:] = float("nan")
        after = ref.paged_attention_v1(q, kc, vc, tables, lens, 0.3)
        torch.testing.assert_close(before, after)

    def test_zero_length_sequence_yields_zeros(self):
        kc = torch.randn(4, 4, 2, 8)
        out = ref.paged_attention_v1(
            torch.randn(1, 4, 8),
            kc,
            kc.clone(),
            torch.tensor([[0]]),
            torch.tensor([0], dtype=torch.int32),
            1.0,
        )
        assert torch.equal(out, torch.zeros_like(out))

    def test_reshape_and_cache_skips_negative_slots(self):
        kc = torch.zeros(4, 2, 2, 8)
        vc = torch.zeros(4, 2, 2, 8)
        key = torch.randn(3, 2, 8)
        value = torch.randn(3, 2, 8)
        ref.reshape_and_cache(key, value, kc, vc, torch.tensor([5, -1, 0]))
        assert torch.equal(kc[2, 1], key[0]) and torch.equal(vc[0, 0], value[2])
        assert torch.count_nonzero(kc) == 2 * 2 * 8

    def test_copy_blocks_copies_every_layer_and_both_k_and_v(self):
        cache = torch.randn(3, 2, 6, 4, 2, 8)
        expected = cache.clone()
        expected[:, :, 4] = expected[:, :, 1]
        expected[:, :, 0] = expected[:, :, 2]
        ref.copy_blocks(cache, torch.tensor([[1, 4], [2, 0]]))
        assert torch.equal(cache, expected)

    def test_swap_blocks_round_trip(self):
        device_side = torch.randn(6, 4, 2, 8)
        host = torch.zeros(6, 4, 2, 8)
        ref.swap_blocks(device_side, host, torch.tensor([[1, 3], [5, 0]]))
        assert torch.equal(host[3], device_side[1]) and torch.equal(host[0], device_side[5])
        restored = torch.zeros_like(device_side)
        ref.swap_blocks(host, restored, torch.tensor([[3, 1]]))
        assert torch.equal(restored[1], device_side[1])


def ragged_batch(rows, *, seed=0):
    """A RaggedMetadata for ``rows`` of (context_len, query_len), plus its inputs.

    Context tokens are assumed already cached; the batch writes the query tokens.
    """
    torch.manual_seed(seed)
    tables = random_tables([c + q for c, q in rows], BLOCK_SIZE, NUM_BLOCKS, seed=seed)
    slot_mapping, positions, query_lens, offsets, seq_lens = [], [], [], [], []
    decode_rows, prefill_rows = [], []
    offset = 0
    for i, (context, query_len) in enumerate(rows):
        for pos in range(context, context + query_len):
            slot_mapping.append(int(tables[i, pos // BLOCK_SIZE]) * BLOCK_SIZE + pos % BLOCK_SIZE)
            positions.append(pos)
        query_lens.append(query_len)
        offsets.append(offset)
        seq_lens.append(context + query_len)
        (decode_rows if query_len == 1 else prefill_rows).append(i)
        offset += query_len
    md = RaggedMetadata(
        slot_mapping=torch.tensor(slot_mapping, dtype=torch.long),
        block_tables=tables,
        query_lens=tuple(query_lens),
        query_offsets=tuple(offsets),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        query_positions=torch.tensor(positions, dtype=torch.long),
        decode_rows=tuple(decode_rows),
        prefill_rows=tuple(prefill_rows),
        gathered_len=tables.shape[1] * BLOCK_SIZE,
        live_tokens=sum(seq_lens),
        used_blocks=0,
    )
    total = sum(query_lens)
    q = torch.randn(1, total, TINY.num_q_heads, TINY.head_dim)
    k = torch.randn(1, total, TINY.num_kv_heads, TINY.head_dim)
    v = torch.randn(1, total, TINY.num_kv_heads, TINY.head_dim)
    return md, q, k, v


def make_pair():
    kwargs = dict(num_blocks=NUM_BLOCKS, device=torch.device("cpu"), dtype=torch.float32)
    oracle = PagedAttentionBackend(TINY, CACHE, **kwargs)
    cuda = CudaPagedAttentionBackend(TINY, CACHE, ops=ref.ReferenceOps, **kwargs)
    oracle.allocate()
    cuda.allocate()
    torch.manual_seed(42)
    oracle.kv_cache.normal_()
    cuda.kv_cache.copy_(oracle.kv_cache)
    return oracle, cuda


class TestBackendAgainstOracle:
    @pytest.mark.parametrize(
        "rows",
        [
            [(0, 1)],
            [(5, 1), (12, 1), (3, 1)],
            [(0, 7), (9, 1), (0, 3), (15, 1)],
            [(8, 5), (4, 1)],  # a prefill over a cached prefix (prefix-cache hit / chunk)
        ],
    )
    def test_ragged_output_and_cache_match(self, rows):
        oracle, cuda = make_pair()
        md, q, k, v = ragged_batch(rows)
        for layer in range(TINY.num_layers):
            want = oracle.forward(layer, q, k, v, md)
            got = cuda.forward(layer, q, k, v, md)
            torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
        assert torch.equal(cuda.kv_cache, oracle.kv_cache)

    def test_decode_metadata_matches_ragged_decode(self):
        oracle, cuda = make_pair()
        md, q, k, v = ragged_batch([(5, 1), (11, 1), (2, 1)])
        decode = DecodeMetadata(
            slot_mapping=md.slot_mapping, block_tables=md.block_tables, seq_lens=md.seq_lens
        )
        want = oracle.forward(0, q, k, v, md)
        got = cuda.forward(0, q, k, v, decode)
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    def test_plan_is_rebuilt_for_every_step(self):
        """Per-step tensors are memoised per metadata object, never across steps."""
        oracle, _ = make_pair()
        md1, q1, k1, v1 = ragged_batch([(5, 1), (3, 1)], seed=1)
        md2, q2, k2, v2 = ragged_batch([(9, 1)], seed=2)
        oracle.forward(0, q1, k1, v1, md1)
        first = oracle._step_plan
        oracle.forward(1, q1, k1, v1, md1)
        assert oracle._step_plan is first
        oracle.forward(0, q2, k2, v2, md2)
        assert oracle._step_plan is not first and oracle._step_plan.metadata is md2


class TestSelection:
    def test_missing_extension_is_a_clear_error_when_constructed_directly(self):
        if extension.is_available():
            pytest.skip("extension is built here")
        with pytest.raises(RuntimeError, match="extension"):
            CudaPagedAttentionBackend(
                TINY, CACHE, num_blocks=4, device=torch.device("cpu"), dtype=torch.float32
            )

    def test_engine_falls_back_to_gather_and_records_it(self):
        from pagedserve.engine import _build_backend

        config = EngineConfig.build(
            TINY, device="cpu", dtype="float32", cache=CACHE, attn_backend="cuda"
        )
        backend, manager = _build_backend(config)
        assert isinstance(backend, PagedAttentionBackend)
        assert not isinstance(backend, CudaPagedAttentionBackend)
        assert manager is not None
        assert config.attn_backend == "gather"
        assert config.to_dict()["attn_backend"] == "gather"


@pytest.mark.gpu
@pytest.mark.cuda_ext
@requires_extension
class TestKernels:
    @pytest.fixture
    def ops(self):
        return extension.load()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("block_size", [8, 16, 32])
    @pytest.mark.parametrize("heads", [(14, 2), (8, 8)])
    def test_paged_attention_matches_reference(self, ops, dtype, head_dim, block_size, heads):
        torch.manual_seed(0)
        num_heads, kv_heads = heads
        seq_lens = [1, block_size, block_size + 1, 5 * block_size - 3, 333]
        num_blocks = 128
        kc = torch.randn(num_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=dtype)
        vc = torch.randn_like(kc)
        tables = random_tables(seq_lens, block_size, num_blocks - 1).cuda()
        lens = torch.tensor(seq_lens, dtype=torch.int32, device="cuda")
        q = torch.randn(len(seq_lens), num_heads, head_dim, device="cuda", dtype=dtype)
        scale = head_dim**-0.5
        got = ops.paged_attention_v1(q, kc, vc, tables, lens, scale)
        want = ref.paged_attention_v1(q, kc, vc, tables, lens, scale)
        tol = {torch.float32: 1e-4, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]
        torch.testing.assert_close(got.float(), want.float(), rtol=tol, atol=tol)

    def test_paged_attention_accepts_strided_query(self, ops):
        kc = torch.randn(8, 16, 2, 64, device="cuda")
        q_big = torch.randn(3, 8, 128, device="cuda")
        q = q_big[..., :64]
        tables = torch.tensor([[0, 1], [2, 3], [4, 5]], device="cuda")
        lens = torch.tensor([20, 1, 32], dtype=torch.int32, device="cuda")
        got = ops.paged_attention_v1(q, kc, kc, tables, lens, 0.125)
        want = ref.paged_attention_v1(q.contiguous(), kc, kc, tables, lens, 0.125)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_reshape_and_cache_matches_reference(self, ops, dtype):
        kc = torch.zeros(16, 16, 2, 64, device="cuda", dtype=dtype)
        vc = torch.zeros_like(kc)
        key = torch.randn(10, 2, 64, device="cuda", dtype=dtype)
        value = torch.randn_like(key)
        slots = torch.randperm(16 * 16, device="cuda")[:10]
        slots[3] = -1
        kc_ref, vc_ref = kc.clone(), vc.clone()
        ops.reshape_and_cache(key, value, kc, vc, slots)
        ref.reshape_and_cache(key, value, kc_ref, vc_ref, slots)
        assert torch.equal(kc, kc_ref) and torch.equal(vc, vc_ref)

    def test_copy_blocks_matches_reference(self, ops):
        cache = torch.randn(3, 2, 10, 16, 2, 64, device="cuda", dtype=torch.float16)
        expected = cache.clone()
        mapping = torch.tensor([[1, 7], [2, 0], [9, 3]], device="cuda")
        ops.copy_blocks(cache, mapping)
        ref.copy_blocks(expected, mapping)
        assert torch.equal(cache, expected)

    def test_swap_blocks_round_trip(self, ops):
        gpu = torch.randn(10, 16, 2, 64, device="cuda", dtype=torch.float16)
        host = torch.zeros(10, 16, 2, 64, dtype=torch.float16).pin_memory()
        ops.swap_blocks(gpu, host, torch.tensor([[1, 4], [6, 0]]))
        torch.cuda.synchronize()
        assert torch.equal(host[4], gpu[1].cpu()) and torch.equal(host[0], gpu[6].cpu())
        back = torch.zeros_like(gpu)
        ops.swap_blocks(host, back, torch.tensor([[4, 1]]))
        torch.cuda.synchronize()
        assert torch.equal(back[1], gpu[1])
