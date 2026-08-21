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


def hf_reference(path: str, prompts: list[str], max_tokens: int, dtype: str) -> list[list[int]]:
    """HuggingFace greedy output, with the checkpoint's sampling knobs removed.

    Qwen2.5 ships repetition_penalty=1.1 and generate() applies logits
    processors even when do_sample is False, so an un-neutralised reference is
    a different algorithm entirely.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from bench.baseline_hf import dtype_kwarg

    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(path, **dtype_kwarg(getattr(torch, dtype)))
        .to("cuda")
        .eval()
    )
    encoded = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    config = GenerationConfig(
        max_new_tokens=max_tokens,
        do_sample=False,
        repetition_penalty=1.0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    with torch.inference_mode():
        out = model.generate(**encoded, generation_config=config)
    rows = out[:, encoded["input_ids"].shape[1] :].tolist()
    eos = tokenizer.eos_token_id
    eos_ids = set(eos) if isinstance(eos, list) else {eos}
    trimmed = []
    for row in rows:
        stop = len(row)
        for i, token in enumerate(row):
            if token in eos_ids:
                stop = i + 1
                break
        trimmed.append(row[:stop])
    del model
    torch.cuda.empty_cache()
    return trimmed


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

    # ---- question 0: is the paged path even deterministic? ----
    print("=" * 72)
    print("  0. IS THE PAGED PATH DETERMINISTIC?")
    print("=" * 72)
    print("  Every padding token in a left-padded batch is written to the SAME")
    print("  trash slot, so index_copy_ receives duplicate indices -- which")
    print("  PyTorch documents as undefined behaviour on CUDA. If repeated runs")
    print("  of identical input disagree, that race is the bug, and no amount of")
    print("  dtype analysis will explain it.")
    print()
    for dtype in ("float16", "float32"):
        for backend in ("contiguous", "gather"):
            engine = build(args.model, backend, dtype, "cuda")
            runs = [
                tuple(tuple(t) for t in engine.generate(prompt_ids, max_tokens=MAX_TOKENS).token_ids)
                for _ in range(5)
            ]
            unique = len(set(runs))
            verdict = "deterministic" if unique == 1 else f"NONDETERMINISTIC ({unique} distinct)"
            print(f"  {dtype:>9} {backend:>11}: {verdict}")
            del engine
            torch.cuda.empty_cache()
    print()

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
        reference = hf_reference(path, PROMPTS, MAX_TOKENS, dtype)
        print(
            f"  {dtype:>9}: paged==contiguous {agree} | "
            f"contiguous==HF {dense.token_ids == reference} | "
            f"paged==HF {paged.token_ids == reference}"
        )
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
