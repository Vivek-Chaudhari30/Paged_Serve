"""bench/kernel_bench.py on CPU: the harness, not the kernel.

The compiled kernel is only timed on a GPU. What runs here is everything that
must already be right before that session is paid for: the inputs, the oracle
check, the bandwidth arithmetic and the result file the charts read.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from bench import kernel_bench as kb  # noqa: E402
from pagedserve.attention import paged_reference  # noqa: E402


def test_gather_sdpa_matches_the_reference():
    inputs = kb.make_inputs(
        batch=3,
        seq_len=37,
        block_size=8,
        num_heads=4,
        num_kv_heads=2,
        head_dim=16,
        dtype=torch.float32,
        device=torch.device("cpu"),
        seed=1,
    )
    scale = 16**-0.5
    got = kb.gather_sdpa(*inputs, scale)
    want = paged_reference.paged_attention_v1(*inputs, scale)
    assert torch.allclose(got, want, atol=1e-5)


def test_block_tables_are_shuffled_and_cover_every_block():
    _, _, _, tables, lens = kb.make_inputs(
        batch=4,
        seq_len=64,
        block_size=8,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        seed=0,
    )
    flat = tables.flatten().tolist()
    assert sorted(flat) == list(range(32))
    assert flat != list(range(32))
    assert lens.tolist() == [64] * 4


def test_kv_bytes_counts_k_and_v():
    assert kb.kv_bytes(batch=2, seq_len=10, num_kv_heads=3, head_dim=4, elem_size=2) == 960


def test_cli_writes_a_result_the_charts_recognise(tmp_path):
    out = tmp_path / "mb.json"
    rc = kb.main(
        [
            "--num-heads",
            "4",
            "--num-kv-heads",
            "2",
            "--head-dim",
            "16",
            "--dtype",
            "float32",
            "--device",
            "cpu",
            "--seq-lens",
            "16",
            "40",
            "--batch-sizes",
            "2",
            "--block-sizes",
            "8",
            "--iters",
            "2",
            "--warmup",
            "0",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    result = json.loads(out.read_text())
    assert result["kind"] == "kernel_microbench"
    assert {r["impl"] for r in result["rows"]} >= {"reference", "gather-sdpa"}
    assert all(r["matches_reference"] for r in result["rows"])
    assert all(r["effective_gbps"] > 0 for r in result["rows"])


@pytest.mark.gpu
@pytest.mark.cuda_ext
def test_cuda_kernel_is_benchmarked_and_correct(tmp_path):
    from pagedserve import extension

    if not (extension.is_available() and torch.cuda.is_available()):
        pytest.skip("CUDA extension not built or no GPU. This is a SKIP, not a pass.")
    out = tmp_path / "mb.json"
    rc = kb.main(
        [
            "--num-heads",
            "14",
            "--num-kv-heads",
            "2",
            "--head-dim",
            "64",
            "--dtype",
            "float16",
            "--seq-lens",
            "128",
            "1024",
            "--batch-sizes",
            "1",
            "8",
            "--iters",
            "5",
            "--warmup",
            "1",
            "--output",
            str(out),
        ]
    )
    rows = json.loads(out.read_text())["rows"]
    assert rc == 0
    assert any(r["impl"] == "cuda" for r in rows)
