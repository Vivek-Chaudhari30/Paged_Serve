# PagedServe

A from-scratch high-throughput LLM inference server. It reimplements the core mechanisms that
make vLLM, TensorRT-LLM, and SGLang fast — a paged KV cache, an iteration-level continuous
batching scheduler, block-aligned prefix caching, and a custom CUDA paged attention kernel —
without depending on any of them at runtime.

**Status: the engine runs; nothing has been benchmarked yet.**

| Phase | | State |
|---|---|---|
| 0 | Benchmark harness | complete |
| 1 | Naive engine, contiguous KV cache | complete |
| 2 | Paged KV cache (Python gather backend) | complete |
| 3 | Continuous batching scheduler | complete |
| 4 | **Custom CUDA paged attention kernel** | **build scaffolding only — the kernel is not written** |
| 5 | Block-aligned prefix caching | complete |
| 6 | OpenAI-compatible HTTP server | complete |
| 7 | Chunked prefill, CUDA graphs | not started |
| 8 | Ablations | not started |
| 9 | Writeup | not started |

Every completed phase is CPU-verified, and correctness is additionally verified on a Tesla T4
(see below). Every completed phase also has exactly one open box, and it is the same box in
each: **run the benchmark sweep on a GPU.** `results/` does not exist yet.

## Scope

Single GPU, one Llama-style architecture family (RoPE, GQA, SwiGLU, RMSNorm), unquantized
weights. Tensor/pipeline parallelism, weight quantization, broad model-zoo support, and
speculative decoding are explicitly out of scope: the thesis of this project is KV memory
management, and those are different bottlenecks.

## Results

TODO(bench) — **no performance has been measured.** There is no throughput number, no
latency number, and no speedup ratio anywhere in this repository, because there is no
`results/` directory for one to come from.

Every number that eventually appears here will trace to a committed raw result JSON under
`results/`, produced by a script in `bench/`, on hardware named in the methodology section.
That rule is written down in `AGENTS.md` §2.1 and it is not negotiable.

## What has been verified

None of the following is a performance measurement. They are correctness results and
allocation bookkeeping, and they are listed separately from Results for exactly that reason.

**Correctness, on a Tesla T4 (sm_75, float16 and float32):**

- The golden gate — token-for-token equality with a HuggingFace reference — passes 44/44,
  on both attention backends, in both dtypes, with prefix caching on and off.
- The paged and contiguous backends produce bit-identical logits.
- Forced preemption is invisible in the output under both the recompute and swap policies.
- The CUDA extension's build canary passes 13/13, including a fresh-subprocess loader test.

A shared, unpinnable T4 cannot produce a reportable timing, so nothing above is timed.

**KV allocation bookkeeping** (4 prompts, `block_size=16`, CPU/fp32 — this ratio is
allocated-versus-live accounting and so is device-independent, but it depends entirely on
`max_seq_len` and on the workload, so it means nothing quoted without them):

| generated tokens | contiguous | paged |
|---|---|---|
| 16 | 2.1% | 68.0% |
| 48 | 5.1% | 80.9% |
| 96 | 8.6% | 87.8% |
| 256 | — | 94.4% |

Paged utilization rises with length because waste is bounded by `block_size - 1` in the last
block only, rather than by `max_seq_len`.

## Install

```bash
pip install -e ".[all]"
```

The base install is deliberately light so the benchmark harness runs on a laptop with no GPU.
`[engine]` pulls torch, `[baseline]` pulls transformers for the HuggingFace reference,
`[server]` pulls FastAPI.

The optional CUDA extension is built explicitly, never through `pip install` — pip builds in
an isolated environment with no torch, so it would silently fail to build:

```bash
python setup.py build_ext --inplace
```

`import pagedserve` works fine without it; the engine falls back to the `gather` backend.

## Run the server

```bash
python -m pagedserve.server --model Qwen/Qwen2.5-0.5B-Instruct --num-blocks 512
```

Pass `--num-blocks` explicitly for any run that must not OOM: KV capacity profiling is
optimistic without a profiling forward pass.

## Development

```bash
ruff check . && ruff format --check .
pytest -m "not gpu"
```

On a GPU machine, `pytest -m gpu` selects only the 10 tests that cannot run without CUDA —
it is **not** the full GPU suite, and the golden gate is not in it. The gate is
device-parametric, so run the whole suite against the GPU with:

```bash
PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest
```

## Further reading

- `docs/how-it-works.md` — plain-language explanation of the whole system.
- `AGENTS.md` — the rules every contributor and coding agent works under.
- `WORKTREE_SUMMARY.md` — an inventory of what is actually in the tree.
