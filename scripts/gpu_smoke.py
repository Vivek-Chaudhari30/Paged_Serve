"""First-contact checks for the code paths that have only ever run on a laptop.

Three things in this repo are CUDA-only and have therefore never executed once:

1. ``profile_num_blocks``'s measurement branch — every run so far passed
   ``num_blocks_override`` because host memory cannot be profiled.
2. ``SwapSpace`` with genuinely pinned host memory and a real copy stream.
3. The golden test in float16 — a T4 is Turing and has no bfloat16 at all, so
   the fp16 fallback in ``resolve_dtype`` becomes load-bearing rather than
   theoretical.

Each check reports PASS or FAIL and keeps going, so one session produces a
complete picture instead of one error at a time.

**Nothing here is a benchmark.** A notebook GPU is shared and unpinnable, so
its timings are not evidence and must not reach ``results/`` (AGENTS.md §4).
This only answers "does it work at all on real hardware".

Usage:
    python scripts/gpu_smoke.py
    python scripts/gpu_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import traceback

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str):
    """Run a check, record the outcome, never abort the run."""

    def decorator(fn):
        try:
            detail = fn() or ""
            results.append((PASS, name, str(detail)))
        except Exception as exc:  # noqa: BLE001 - a failure here is the output
            results.append((FAIL, name, f"{type(exc).__name__}: {exc}"))
            print(f"--- traceback for {name} ---")
            traceback.print_exc()
        return fn

    return decorator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    import torch

    from pagedserve.config import CacheConfig, ModelConfig, resolve_device, resolve_dtype

    device = resolve_device(None)
    dtype = resolve_dtype(None, device)

    print("=" * 72)
    print("  ENVIRONMENT")
    print("=" * 72)
    print(f"  torch            {torch.__version__}")
    print(f"  cuda runtime     {torch.version.cuda}")
    print(f"  cuda available   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"  gpu              {props.name}")
        print(f"  compute cap      {props.major}.{props.minor}")
        print(f"  total memory     {props.total_memory / 2**30:.1f} GiB")
        print(f"  bf16 supported   {torch.cuda.is_bf16_supported()}")
    print(f"  resolved device  {device}")
    print(f"  resolved dtype   {dtype}")
    import transformers

    print(f"  transformers     {transformers.__version__}")

    if device.type != "cuda":
        print("\nNo CUDA device. This script has nothing to check here.")
        return 1

    print()
    print("=" * 72)
    print("  CHECKS")
    print("=" * 72)

    from pagedserve.engine import LLMEngine
    from pagedserve.model.loader import resolve_model_path

    path = str(resolve_model_path(args.model))
    model_config = ModelConfig.from_pretrained(path, name=args.model)

    # ---- 1. capacity profiling, the real measurement branch ----
    @check("profile_num_blocks measures free memory")
    def _profile():
        # No num_blocks_override: this is the whole point, the branch that
        # reads real device memory and has never run.
        engine = LLMEngine.from_pretrained(
            args.model, cache=CacheConfig(max_seq_len=512, max_num_seqs=8)
        )
        num_blocks = engine.block_manager.num_blocks
        return (
            f"{num_blocks} blocks x {engine.config.cache.block_size} tokens "
            f"= {num_blocks * engine.config.cache.block_size:,} token slots"
        )

    # ---- 2. swap space with real pinned memory and a copy stream ----
    @check("SwapSpace round-trips blocks through pinned host memory")
    def _swap():
        from pagedserve.worker.cache_engine import SwapSpace

        swap = SwapSpace(
            model_config,
            num_cpu_blocks=8,
            block_size=16,
            dtype=dtype,
            device=device,
        )
        gpu_cache = torch.randn(
            (model_config.num_layers, 2, 8, 16, model_config.num_kv_heads, model_config.head_dim),
            device=device,
            dtype=dtype,
        )
        original = gpu_cache[0, 0, 2].clone()
        cpu_blocks = swap.swap_out(gpu_cache, [2, 3])
        gpu_cache[0, 0, 2].zero_()
        swap.swap_in(gpu_cache, cpu_blocks, [2, 3])
        if not torch.equal(gpu_cache[0, 0, 2], original):
            raise AssertionError("swapped block did not come back identical")
        return f"pinned={swap.cpu_cache.is_pinned()}, 2 blocks out and back"

    # ---- 3. generation in the device's native dtype ----
    @check(f"end-to-end generation in {dtype}")
    def _generate():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        engine = LLMEngine.from_pretrained(
            args.model, cache=CacheConfig(max_seq_len=512, max_num_seqs=8)
        )
        ids = [tokenizer(p).input_ids for p in ("The capital of France is", "def add(a, b):")]
        sequences = engine.generate_continuous(ids, max_tokens=16)
        texts = [tokenizer.decode(s.output_token_ids) for s in sequences]
        return " | ".join(repr(t[:40]) for t in texts)

    # ---- 4. the contiguous ablation arm on GPU ----
    @check("contiguous (--no-paging) arm runs on GPU")
    def _contiguous():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        engine = LLMEngine.from_pretrained(
            args.model,
            cache=CacheConfig(max_seq_len=512, max_num_seqs=4),
            attn_backend="contiguous",
        )
        out = engine.generate([tokenizer("The capital of France is").input_ids], max_tokens=8)
        return out.utilization_report()

    # ---- 5. paged and contiguous must agree, on this hardware too ----
    @check("paged and contiguous agree bit-for-bit on GPU")
    def _agree():
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(path)
        ids = [tokenizer(p).input_ids for p in ("The capital of France is", "def add(a, b):")]
        cache = CacheConfig(max_seq_len=512, max_num_seqs=4, num_blocks_override=256)
        dense = LLMEngine.from_pretrained(
            args.model, cache=cache, attn_backend="contiguous"
        ).generate(ids, max_tokens=16)
        paged = LLMEngine.from_pretrained(args.model, cache=cache, attn_backend="gather").generate(
            ids, max_tokens=16
        )
        if dense.token_ids != paged.token_ids:
            raise AssertionError(
                f"backends diverged\n  dense: {dense.token_ids}\n  paged: {paged.token_ids}"
            )
        return "identical"

    # ---- 6. forced preemption, both policies, on GPU ----
    @check("forced preemption is invisible in the output (both policies)")
    def _preempt():
        from transformers import AutoTokenizer

        from pagedserve.config import SchedulerConfig

        tokenizer = AutoTokenizer.from_pretrained(path)
        ids = [
            tokenizer(p).input_ids
            for p in ("The capital of France is", "def add(a, b):", "Q: 2+2?\nA:")
        ]
        roomy = CacheConfig(max_seq_len=512, max_num_seqs=8, num_blocks_override=256)
        reference = [
            s.output_token_ids
            for s in LLMEngine.from_pretrained(args.model, cache=roomy).generate_continuous(
                ids, max_tokens=24
            )
        ]
        notes = []
        for policy in ("recompute", "swap"):
            engine = LLMEngine.from_pretrained(
                args.model,
                cache=CacheConfig(
                    max_seq_len=512,
                    max_num_seqs=8,
                    num_blocks_override=6,
                    swap_space_blocks=64,
                ),
                scheduler=SchedulerConfig(preemption_policy=policy, max_num_seqs=8),
            )
            got = [s.output_token_ids for s in engine.generate_continuous(ids, max_tokens=24)]
            if got != reference:
                raise AssertionError(f"{policy} changed the output")
            notes.append(f"{policy}={engine.scheduler.num_preemptions} preemptions")
        return ", ".join(notes)

    print()
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    for status, name, detail in results:
        print(f"  [{status}] {name}")
        if detail:
            print(f"         {detail}")
    failures = sum(1 for status, _, _ in results if status == FAIL)
    print()
    print(f"  {len(results) - failures} passed, {failures} failed")
    print()
    print("  Reminder: these are correctness checks, not benchmarks. A notebook")
    print("  GPU is shared and unpinnable, so nothing timed here belongs in results/.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
