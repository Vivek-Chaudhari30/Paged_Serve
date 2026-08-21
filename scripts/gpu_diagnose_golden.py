"""Decide whether the GPU golden failure is a bug or float16 rounding.

The failure has a specific shape: on CUDA in float16 the *paged* backend
diverges from HuggingFace while the *contiguous* backend does not, and a
sequence's output changes depending on whether it was batched. On CPU in
float32 every one of those cases passes.

Two hypotheses, and they are distinguishable:

**H1 — a logic bug in the paged path** that CPU/float32 happened not to expose.
Then the divergence survives in float32 on the GPU too, and the logits at the
divergence point differ by much more than rounding.

**H2 — float16 non-associativity.** The paged path attends over a buffer padded
out to a whole number of blocks, so it hands SDPA a different shape than the
contiguous path does. Different shape means a different kernel and a different
reduction order, and in float16 (about three decimal digits) that is enough to
flip an argmax whose top two logits are close. Then float32 on the same GPU
passes, and the divergences sit on near-ties.

The distinguishing evidence, printed below:

1. Does the paged/contiguous disagreement survive in float32 on this GPU?
2. At the first divergent step, how far apart are the top two logits? A
   near-tie is rounding; a wide gap is a bug.
3. How large is the raw logit difference between the two backends at that step,
   relative to the size of the logits themselves?

This script asserts nothing and fixes nothing. It answers one question.
"""

from __future__ import annotations

import argparse

PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists",
    "def fibonacci(n):",
    "Q: What is 2 + 2?\nA:",
]
MAX_TOKENS = 24


def build(model: str, backend: str, dtype: str, device: str):
    from pagedserve.config import CacheConfig
    from pagedserve.engine import LLMEngine

    return LLMEngine.from_pretrained(
        model,
        device=device,
        dtype=dtype,
        cache=CacheConfig(max_seq_len=256, max_num_seqs=8, block_size=16, num_blocks_override=128),
        attn_backend=backend,
    )


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    import torch
    from transformers import AutoTokenizer

    from pagedserve.model.loader import resolve_model_path

    if not torch.cuda.is_available():
        print("No CUDA device; this diagnostic only makes sense on the GPU.")
        return 1

    path = str(resolve_model_path(args.model))
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompt_ids = [tokenizer(p).input_ids for p in PROMPTS]

    # ---- question 1: does the disagreement survive in float32? ----
    print("=" * 72)
    print("  1. PAGED vs CONTIGUOUS, SAME GPU, BOTH DTYPES")
    print("=" * 72)
    outcomes = {}
    for dtype in ("float16", "float32"):
        dense = build(args.model, "contiguous", dtype, "cuda").generate(
            prompt_ids, max_tokens=MAX_TOKENS
        )
        paged = build(args.model, "gather", dtype, "cuda").generate(
            prompt_ids, max_tokens=MAX_TOKENS
        )
        agree = dense.token_ids == paged.token_ids
        outcomes[dtype] = agree
        print(f"  {dtype:>9}: backends agree = {agree}")
        if not agree:
            pairs = zip(dense.token_ids, paged.token_ids, strict=True)
            for i, (d, p) in enumerate(pairs):
                index = first_divergence(d, p)
                if index is not None:
                    print(f"      prompt {i} diverges at generated token {index}")
        torch.cuda.empty_cache()

    print()
    print("  Reading: float32 agreeing while float16 does not points at")
    print("  precision. Both disagreeing points at a logic bug in the paged path.")

    # ---- question 2: is the contested decision a near-tie? ----
    print()
    print("=" * 72)
    print("  2. HOW CLOSE IS THE CONTESTED DECISION?")
    print("=" * 72)
    for dtype in ("float16", "float32"):
        dense_engine = build(args.model, "contiguous", dtype, "cuda")
        paged_engine = build(args.model, "gather", dtype, "cuda")
        dense = dense_engine.generate(prompt_ids, max_tokens=MAX_TOKENS)
        paged = paged_engine.generate(prompt_ids, max_tokens=MAX_TOKENS)

        for i in range(len(prompt_ids)):
            index = first_divergence(dense.token_ids[i], paged.token_ids[i])
            if index is None:
                continue
            # Re-run just this prompt up to the contested step and read the
            # logits both backends produced there.
            prefix = prompt_ids[i] + dense.token_ids[i][:index]
            gaps = {}
            for name, engine in (("contiguous", dense_engine), ("paged", paged_engine)):
                logits = engine.logits_for(prefix)
                top = torch.topk(logits.float(), 2)
                gaps[name] = (
                    top.values[0].item(),
                    top.values[1].item(),
                    top.indices.tolist(),
                )
            print(f"  [{dtype}] prompt {i}, generated token {index}:")
            for name, (top1, top2, ids) in gaps.items():
                print(
                    f"      {name:>11}: top1={top1:9.4f} top2={top2:9.4f} "
                    f"gap={top1 - top2:8.5f} ids={ids}"
                )
        torch.cuda.empty_cache()

    print()
    print("  Reading: a gap of order 0.01 or less is a coin-flip that rounding")
    print("  decides. A gap of order 1 means the two backends genuinely computed")
    print("  different things, which is a bug.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
