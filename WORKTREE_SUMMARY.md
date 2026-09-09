# PagedServe — Worktree Build Summary

A from-scratch, high-throughput LLM inference server that reimplements the
core mechanisms behind vLLM / TensorRT-LLM / SGLang.

**What this file is.** An inventory of what exists in the tree right now. Every
path below was checked against the working tree; nothing is listed because a
roadmap says it should be there. Where a phase is incomplete, this file says so
rather than describing the finished shape.

**No performance numbers appear here.** `results/` does not exist yet, so under
`AGENTS.md` §2.1 there is nothing this file is allowed to claim about speed.

---

## What We Built (Phase by Phase)

### Phase 1 — Contiguous KV Cache · complete
- `pagedserve/attention/contiguous.py` — baseline SDPA over a contiguous
  max-length KV cache (kept forever as the ablation baseline via `--no-paging`).
- `pagedserve/model/llama.py`, `layers.py`, `loader.py` — our own forward pass
  (RMSNorm, RoPE, GQA, SwiGLU) over safetensors weights.

### Phase 2 — Paged KV Cache (Python gather backend) · complete
- `pagedserve/memory/block.py`, `block_manager.py` — physical blocks, per-sequence
  block tables, free list, refcounts, `allocate` / `append_slot` / `free` / `fork`.
- `pagedserve/attention/gather.py` — `index_select` blocks into a scratch buffer
  then run SDPA. This is the correctness oracle for the future CUDA kernel and is
  never deleted.
- `pagedserve/worker/cache_engine.py` — capacity profiling and one preallocation.

### Phase 3 — Continuous Batching · complete
- `pagedserve/core/scheduler.py` — iteration-level scheduler with
  waiting / running / swapped queues, admission budgeted on both
  `max_num_batched_tokens` and `max_num_seqs`, and preemption.
- `pagedserve/core/policy.py` — RECOMPUTE vs SWAP preemption policy.
- `pagedserve/sequence.py` — `Sequence`, `SequenceGroup`, `SequenceStatus`.
- `pagedserve/engine.py` — `LLMEngine.step()`, the top-level decode loop. Batch
  assembly and `slot_mapping` construction live here; there is **no**
  `pagedserve/worker/model_runner.py`, and the layout sketch in `AGENTS.md` §3
  showing one is a plan, not a description of the tree.

### Phase 4 — Custom CUDA Paged Attention Kernel · SCAFFOLDING ONLY (~10%)

Step 1 of this phase — proving the build toolchain works — is done. **The decode
kernel itself is not started.** What actually exists:

- `csrc/trivial.cu` — a build canary. It adds one to every element of a tensor.
  It contains no attention math of any kind.
- `csrc/bindings.cpp` — the pybind11 module definition that exposes the canary.
- `setup.py` — builds the extension. Build it **explicitly**, never through
  `pip install`: pip builds in an isolated environment with no torch, so the
  extension would silently fail to build.
  ```bash
  python setup.py build_ext --inplace
  ```
- `pagedserve/extension.py` — optional loader. A missing extension is a supported
  state; `import pagedserve` must work on a laptop with no `nvcc`.
- Verified on a Tesla T4 (sm_75): the toolchain compiles and links against
  libtorch, the canary round-trips tensors, and the module imports from a fresh
  process with torch not previously imported. **That is a build verification, not
  a kernel verification.**

Not written yet: `csrc/paged_attention.cu`, `csrc/cache_kernels.cu`, and
`pagedserve/attention/cuda_paged.py`. There is no `cuda` attention backend, so
the golden test currently runs against `contiguous` and `gather` only.

### Phase 5 — Prefix Cache · complete
- `pagedserve/memory/prefix_cache.py` — hash-chained block index, refcounted
  shared blocks, LRU eviction pool.
- Copy-on-write is implemented and tested, but is **not reachable from prefix
  caching alone**: only full blocks are cached and a full block is never written
  to again. It exists for `fork()`.
- Prefix caching is semantically invisible — the golden test passes with it on
  or off, producing identical tokens.

### Phase 6 — Full Sampling & OpenAI-Compatible Server · complete
- `pagedserve/model/sampler.py` — greedy (bit-for-bit identical to Phase 2, via
  an all-greedy short circuit to `argmax`), temperature, top-p, top-k,
  repetition penalty.
- `pagedserve/server/api.py` — FastAPI app with `/v1/completions` and
  `/v1/chat/completions`, SSE streaming, `n>1` via block-table forking, and
  block release on client disconnect.
- `pagedserve/server/protocol.py` — Pydantic schemas matching the OpenAI wire
  format.
- Stop *strings* live in the server, not the engine: detecting them needs a
  detokenizer, and `AGENTS.md` §2.5 keeps the tokenizer out of the hot loop. The
  engine handles stop *tokens*, which need no decoding.

### Phases 7, 8, 9 — not started
Chunked prefill and CUDA graphs (7), ablation sweeps (8), and the public writeup
(9) have no code in the tree.

---

## Supporting Infrastructure

| Area | Files |
|---|---|
| Model loading | `pagedserve/model/loader.py` (safetensors → state dict) |
| Llama forward pass | `pagedserve/model/llama.py`, `layers.py` |
| Engine / batch assembly | `pagedserve/engine.py` |
| KV cache allocation & swap | `pagedserve/worker/cache_engine.py` |
| Config | `pagedserve/config.py` (EngineConfig, CacheConfig, SchedulerConfig, ModelConfig) |
| Attention ABC | `pagedserve/attention/backend.py` — required indirection; all backends plug in here |
| Optional CUDA extension | `pagedserve/extension.py` |

---

## Benchmarking & Plotting

- `bench/baseline_hf.py` — HuggingFace `generate()` reference baseline.
- `bench/pagedserve_backend.py` — PagedServe driver, continuous and static arms.
- `bench/loadgen.py` — Poisson and closed-loop request generator.
- `bench/metrics.py` — TTFT / ITL / E2E percentiles, throughput.
- `bench/plot.py` — regenerates charts from committed result JSON.
- `scripts/explorer_job.sbatch` — SLURM sweep job for the A100 cluster.
- `scripts/fetch_dataset.sh`, `scripts/kaggle_bootstrap.sh`, `scripts/gpu_smoke.py`.

There is no `bench/sweep.py`; the sweep is driven by `scripts/explorer_job.sbatch`
calling `python -m bench.loadgen`.

---

## Test Suite

**17 test files, 482 collected tests.** On a laptop, 472 run and 10 are deselected
as GPU-only:

```bash
pytest -m "not gpu"       # 472 passed, 10 deselected
```

- `test_golden.py` — token-for-token match vs HuggingFace. The commit gate. It is
  device-parametric (`PAGEDSERVE_TEST_DEVICE`, `PAGEDSERVE_TEST_DTYPE`) rather
  than `gpu`-marked, so running it on CUDA is a matter of setting those, not of
  selecting a marker.
- `test_attention_contiguous.py`, `test_attention_gather.py` — both backends.
  There is no `test_attention_parity.py` yet; it arrives with the CUDA kernel.
- `test_block_manager.py`, `test_prefix_cache.py` — memory subsystem.
- `test_scheduler.py` — admission, preemption, queue transitions.
- `test_cache_engine.py` — capacity arithmetic (CPU) and real CUDA profiling (`gpu`).
- `test_extension.py` — optional-import behaviour (CPU) and the build canary (`cuda_ext`).
- `test_sampler.py`, `test_layers.py`, `test_config.py`, `test_server.py`.
- `test_loadgen.py`, `test_metrics.py`, `test_plot.py`, `test_baseline_hf.py`,
  `test_pagedserve_backend.py` — the bench harness.

`pytest -m gpu` selects only the 10 tests that cannot run without CUDA. It is
**not** the full GPU suite — the golden gate is not in it. To exercise the gate on
a GPU:

```bash
PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest
```

### GPU correctness status (Tesla T4, sm_75, float16 and float32)

Correctness has been verified on real CUDA hardware. None of it is a benchmark —
a shared, unpinnable T4 produces no reportable timing (`AGENTS.md` §4).

- Golden gate 44/44 on GPU, on both backends, in float16 and float32, with prefix
  caching on and off.
- Paged and contiguous produce bit-identical logits.
- Forced preemption is invisible in the output under both policies.
- CUDA extension 13/13, including the fresh-subprocess loader test.

---

## Docs & Design

- `README.md` — public-facing; the Results section stays `TODO(bench)` until
  measured results are committed.
- `docs/how-it-works.md` — plain-language system explainer.
- `AGENTS.md` — single source of truth for all AI agent instructions.
- `FINAL-COMPLETION.md` — the remaining plan, from here to done.
- `_private/` (gitignored) — DESIGN.md, ROADMAP.md, AGENT-PROMPTS.md.

---

## Key Engineering Rules Upheld

1. **No fabricated numbers** — every perf metric requires a committed benchmark result.
2. **Golden test gates correctness** — must pass across all backends and cache states.
3. **Every optimization is ablatable** — `--no-paging`, `--attn-backend=gather`, prefix cache flag.
4. **Hot loop discipline** — no Python loops, `.item()`, or allocations in the decode step.
5. **This file describes the tree, not the plan** — if a path is listed here, it exists.
