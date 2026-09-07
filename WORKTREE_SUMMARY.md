# PagedServe — Worktree Build Summary

A from-scratch, high-throughput LLM inference server that reimplements the
core mechanisms behind vLLM / TensorRT-LLM / SGLang.

---

## What We Built (Phase by Phase)

### Phase 1 — Contiguous KV Cache
- `pagedserve/attention/contiguous.py` — baseline SDPA over a contiguous
  max-length KV cache (kept as the ablation baseline via `--no-paging`).
- `pagedserve/memory/block.py`, `block_manager.py` — data structures for
  physical blocks and block tables (scaffolded here, fully used later).

### Phase 2 — Paged KV Cache (Python gather backend)
- `pagedserve/attention/gather.py` — `index_select` blocks into a scratch
  buffer then run SDPA; correctness oracle for the CUDA kernel.
- Block manager free list, refcounts, per-sequence block tables.

### Phase 3 — Continuous Batching
- `pagedserve/core/scheduler.py` — iteration-level scheduler with
  waiting / running / swapped queues, admission, and preemption.
- `pagedserve/core/policy.py` — RECOMPUTE vs SWAP preemption policy.
- `pagedserve/sequence.py` — `Sequence`, `SequenceGroup`, `SequenceStatus`.
- `pagedserve/engine.py` — `LLMEngine.step()` top-level decode loop.

### Phase 4 — Custom CUDA Paged Attention Kernel
- `csrc/paged_attention.cu` — decode kernel that reads KV through block tables.
- `csrc/cache_kernels.cu` — KV scatter / copy / swap kernels.
- `csrc/bindings.cpp` — pybind11 module wiring.
- `pagedserve/attention/cuda_paged.py` — Python wrapper for the extension.
- `setup.py` — builds the CUDA extension at install time.
- Verified green on a real T4 GPU (float16).

### Phase 5 — Prefix Cache
- `pagedserve/memory/prefix_cache.py` — hash-chained block index, refcounted
  shared blocks, LRU eviction pool, copy-on-write on first write.
- Prefix caching is semantically invisible — golden test passes with it on or off.

### Phase 6 — Full Sampling & OpenAI-Compatible Server
- `pagedserve/model/sampler.py` — greedy (bit-for-bit identical to Phase 2),
  temperature scaling, top-p, top-k.
- `pagedserve/server/api.py` — FastAPI app with `/v1/completions` and
  `/v1/chat/completions` (SSE streaming).
- `pagedserve/server/protocol.py` — Pydantic request/response schemas matching
  the OpenAI wire format.

---

## Supporting Infrastructure

| Area | Files |
|---|---|
| Model loading | `pagedserve/model/loader.py` (safetensors → state dict) |
| Llama forward pass | `pagedserve/model/llama.py`, `layers.py` (RMSNorm, RoPE, GQA, SwiGLU) |
| Worker / batch assembly | `pagedserve/worker/model_runner.py`, `cache_engine.py` |
| Config | `pagedserve/config.py` (EngineConfig, CacheConfig, SchedulerConfig, ModelConfig) |
| Attention ABC | `pagedserve/attention/backend.py` — required indirection; all backends plug in here |

---

## Benchmarking & Plotting

- `bench/baseline_hf.py` — HuggingFace throughput baseline.
- `bench/pagedserve_backend.py` — PagedServe throughput driver.
- `bench/loadgen.py` — synthetic request generator.
- `bench/metrics.py` — latency / throughput metric collection.
- `bench/plot.py` — matplotlib plots for benchmark results.
- `scripts/` — SLURM job scripts for cluster runs (`explorer_job.sbatch`).

---

## Test Suite

39 test files covering every layer:

- `test_golden.py` — token-for-token match vs HuggingFace (gating test).
- `test_attention_*.py` — contiguous and gather backends.
- `test_block_manager.py`, `test_prefix_cache.py` — memory subsystem.
- `test_scheduler.py` — admission, preemption, queue transitions.
- `test_sampler.py` — all sampling modes.
- `test_server.py` — FastAPI endpoints.
- `test_plot.py` — headless plotting (matplotlib Agg backend via `conftest.py`).
- `test_loadgen.py`, `test_metrics.py`, `test_pagedserve_backend.py` — bench harness.

---

## Docs & Design

- `README.md` — public-facing; results populated only from measured benchmarks.
- `docs/` — plain-language system explainer written during the project.
- `AGENTS.md` — single source of truth for all AI agent instructions.
- `_private/` (gitignored) — DESIGN.md, ROADMAP.md, AGENT-PROMPTS.md.

---

## Key Engineering Rules Upheld

1. **No fabricated numbers** — every perf metric requires a committed benchmark result.
2. **Golden test gates correctness** — must pass across all backends and cache states.
3. **Every optimization is ablatable** — `--no-paging`, `--attn-backend=gather`, prefix cache flag.
4. **Hot loop discipline** — no Python loops, `.item()`, or allocations in the decode step.
