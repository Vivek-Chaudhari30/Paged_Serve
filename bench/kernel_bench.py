"""Decode-attention microbenchmark: the paged kernel in isolation.

The end-to-end sweeps say whether the CUDA kernel made serving faster. They
cannot say *why*, or how far from the hardware it is, because a decode step is
also the MLP, the sampler, the scheduler and Python. This script times the one
call that Phase 4 replaced, with everything else removed:

- ``cuda``        — ``pagedserve._C.paged_attention_v1``, the kernel itself
- ``reference``   — ``attention/paged_reference.py``, the same algorithm in PyTorch ops
- ``gather-sdpa`` — gather the blocks into a contiguous buffer, then SDPA: the
  Phase 2 read path the kernel exists to remove

For each (batch size, block size, context length) it reports the per-call time
distribution and the **effective bandwidth**: the bytes of K and V the call has
to read, divided by its time. Decode attention is bandwidth-bound -- every key
is used for one dot product -- so that number against the card's peak is the
honest measure of how good the kernel is, and the gap is what K6 (vectorised
loads, split-K, warp shuffles) is for.

Every implementation is checked against ``reference`` before it is timed, and the
result is recorded per row. A fast kernel that computes the wrong answer is not
a data point.

Usage::

    python -m bench.kernel_bench --model Qwen/Qwen2.5-0.5B-Instruct \\
        --dtype float16 --output results/kernel/microbench.json

On a machine without a GPU it runs on CPU for a smoke test. CPU timings are not
evidence (AGENTS.md section 4), and bench/plot.py skips any file whose
environment names no GPU.
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import random
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SEQ_LENS = (128, 256, 512, 1024, 2048, 4096)
BATCH_SIZES = (1, 8, 32)
BLOCK_SIZES = (16,)
# Bandwidth-bound kernels are compared at the tolerance the parity tests use.
ATOL = 1e-2

Impl = Callable[..., Any]


def gather_sdpa(query, key_cache, value_cache, block_tables, seq_lens, scale):
    """The Phase 2 read path, reduced to the decode case.

    Copies every sequence's blocks into one padded buffer, masks the tail, and
    runs SDPA. The copy is the cost the paged kernel removes, so it is timed as
    part of the call, exactly as the gather backend pays it every step.
    """
    import torch
    import torch.nn.functional as F

    num_seqs, num_heads, head_dim = query.shape
    _, block_size, kv_heads, _ = key_cache.shape
    max_len = block_tables.shape[1] * block_size
    k = key_cache[block_tables].reshape(num_seqs, max_len, kv_heads, head_dim)
    v = value_cache[block_tables].reshape(num_seqs, max_len, kv_heads, head_dim)
    group = num_heads // kv_heads
    k = k.repeat_interleave(group, dim=2).transpose(1, 2)  # [n, h, L, d]
    v = v.repeat_interleave(group, dim=2).transpose(1, 2)
    positions = torch.arange(max_len, device=query.device).view(1, 1, 1, max_len)
    mask = positions < seq_lens.to(torch.long).view(num_seqs, 1, 1, 1)
    out = F.scaled_dot_product_attention(
        query.unsqueeze(2), k, v, attn_mask=mask, scale=scale
    )  # [n, h, 1, d]
    return out.squeeze(2)


def make_inputs(
    *,
    batch: int,
    seq_len: int,
    block_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any,
    device: Any,
    seed: int,
) -> tuple[Any, ...]:
    """Random Q/K/V and *shuffled* block tables.

    Shuffled because that is the layout paging produces: a sequence's blocks
    are wherever the free list had them. Contiguous tables would let the
    hardware prefetcher flatter the kernel in a way serving never does.
    """
    import torch

    blocks_per_seq = math.ceil(seq_len / block_size)
    num_blocks = batch * blocks_per_seq
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    key_cache = torch.randn(shape, generator=gen).to(device=device, dtype=dtype)
    value_cache = torch.randn(shape, generator=gen).to(device=device, dtype=dtype)
    query = torch.randn((batch, num_heads, head_dim), generator=gen).to(device=device, dtype=dtype)
    pool = list(range(num_blocks))
    random.Random(seed).shuffle(pool)
    tables = torch.tensor(pool, dtype=torch.int64).view(batch, blocks_per_seq).to(device)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device=device)
    return query, key_cache, value_cache, tables, seq_lens


def kv_bytes(*, batch: int, seq_len: int, num_kv_heads: int, head_dim: int, elem_size: int) -> int:
    """Bytes of K and V one decode call must read: the bandwidth numerator.

    Query, output and block tables are left out. They are a few KB against
    megabytes of KV, and leaving them out makes the figure a lower bound on the
    traffic rather than an estimate of it.
    """
    return 2 * batch * seq_len * num_kv_heads * head_dim * elem_size


def time_call(fn: Callable[[], Any], *, device: Any, warmup: int, iters: int) -> list[float]:
    """Per-call times in microseconds.

    On CUDA each call is bracketed by events, so the time is GPU time and the
    host's launch overhead between calls is excluded. Off CUDA it is wall time.
    """
    import torch

    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends, strict=True):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize(device)
        return [start.elapsed_time(end) * 1e3 for start, end in zip(starts, ends, strict=True)]
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e6)
    return out


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def available_impls(device: Any) -> dict[str, Impl]:
    """Implementations runnable here. The kernel only with an extension and a GPU."""
    from pagedserve import extension
    from pagedserve.attention import paged_reference

    impls: dict[str, Impl] = {
        "reference": paged_reference.paged_attention_v1,
        "gather-sdpa": gather_sdpa,
    }
    ext = extension.load()
    if ext is not None and device.type == "cuda":
        impls["cuda"] = ext.paged_attention_v1
    else:
        logger.warning(
            "CUDA kernel not benchmarked: %s",
            extension.unavailable_reason() or "no CUDA device",
        )
    return impls


def run(
    *,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: Any,
    device: Any,
    seq_lens: Sequence[int],
    batch_sizes: Sequence[int],
    block_sizes: Sequence[int],
    impls: dict[str, Impl],
    warmup: int,
    iters: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Time every implementation at every grid point."""
    import torch

    elem = torch.empty((), dtype=dtype).element_size()
    scale = head_dim**-0.5
    rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        for batch in batch_sizes:
            for seq_len in seq_lens:
                inputs = make_inputs(
                    batch=batch,
                    seq_len=seq_len,
                    block_size=block_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    device=device,
                    seed=seed,
                )
                expected = impls["reference"](*inputs, scale).float()
                nbytes = kv_bytes(
                    batch=batch,
                    seq_len=seq_len,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    elem_size=elem,
                )
                for name, fn in impls.items():
                    diff = float((fn(*inputs, scale).float() - expected).abs().max())
                    times = time_call(
                        functools.partial(fn, *inputs, scale),
                        device=device,
                        warmup=warmup,
                        iters=iters,
                    )
                    p50 = statistics.median(times)
                    rows.append(
                        {
                            "impl": name,
                            "batch_size": batch,
                            "block_size": block_size,
                            "seq_len": seq_len,
                            "us_p10": _percentile(times, 0.10),
                            "us_p50": p50,
                            "us_p90": _percentile(times, 0.90),
                            "iters": iters,
                            "kv_bytes": nbytes,
                            "effective_gbps": nbytes / (p50 * 1e-6) / 1e9,
                            "max_abs_diff_vs_reference": diff,
                            "matches_reference": diff <= ATOL,
                        }
                    )
                    if diff > ATOL:
                        logger.error(
                            "%s disagrees with the reference by %.3g at batch=%d seq_len=%d "
                            "block=%d; its time is recorded but is not a valid data point",
                            name,
                            diff,
                            batch,
                            seq_len,
                            block_size,
                        )
                    logger.info(
                        "%-11s b=%-3d bs=%-3d L=%-5d %9.1f us  %7.1f GB/s",
                        name,
                        batch,
                        block_size,
                        seq_len,
                        p50,
                        nbytes / (p50 * 1e-6) / 1e9,
                    )
                del inputs, expected
    return rows


def _head_shape(args: argparse.Namespace) -> tuple[int, int, int, str | None]:
    if args.model:
        from pagedserve.config import ModelConfig
        from pagedserve.model.loader import resolve_model_path

        cfg = ModelConfig.from_pretrained(resolve_model_path(args.model), name=args.model)
        return cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim, args.model
    if None in (args.num_heads, args.num_kv_heads, args.head_dim):
        raise SystemExit("pass --model, or all of --num-heads --num-kv-heads --head-dim")
    return args.num_heads, args.num_kv_heads, args.head_dim, None


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=None, help="Read head shapes from this model.")
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--num-kv-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16", "float32"))
    parser.add_argument("--device", default=None, help="Default: cuda if available, else cpu.")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=list(SEQ_LENS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES))
    parser.add_argument("--block-sizes", type=int, nargs="+", default=list(BLOCK_SIZES))
    parser.add_argument(
        "--impls",
        nargs="+",
        default=None,
        help="Subset of implementations to time (reference is always run as the oracle).",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--peak-gbps",
        type=float,
        default=None,
        help="The card's spec-sheet memory bandwidth, drawn as a reference line. "
        "Recorded as given; this script does not know it.",
    )
    parser.add_argument("--output", default=None, help="Result JSON path.")
    args = parser.parse_args(argv)

    import torch

    from bench.loadgen import collect_environment

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = getattr(torch, args.dtype)
    num_heads, num_kv_heads, head_dim, model = _head_shape(args)
    impls = available_impls(device)
    if args.impls:
        unknown = set(args.impls) - set(impls)
        if unknown:
            raise SystemExit(f"not available here: {sorted(unknown)} (have {sorted(impls)})")
        impls = {k: v for k, v in impls.items() if k in args.impls or k == "reference"}

    rows = run(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        device=device,
        seq_lens=args.seq_lens,
        batch_sizes=args.batch_sizes,
        block_sizes=args.block_sizes,
        impls=impls,
        warmup=args.warmup,
        iters=args.iters,
        seed=args.seed,
    )
    result = {
        "kind": "kernel_microbench",
        "config": {
            "model": model,
            "num_heads": num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_dim,
            "dtype": args.dtype,
            "device": str(device),
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "atol": ATOL,
            "peak_gbps": args.peak_gbps,
            "impls": sorted(impls),
        },
        "environment": collect_environment(),
        "rows": rows,
    }
    bad = [r for r in rows if not r["matches_reference"]]
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out}")
    else:
        print(json.dumps(rows, indent=2))
    if bad:
        print(f"{len(bad)} row(s) disagree with the reference; see max_abs_diff_vs_reference")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
