# AGENTS.md — PagedServe

Instructions for AI coding agents (Claude Code, Codex, Cursor, or any other) working in this
repository. Read this file completely before writing any code. It is the source of truth for
scope, conventions, and constraints.

`CLAUDE.md` points here. Do not duplicate content between them.

---

## 1. What this project is

PagedServe is a **from-scratch high-throughput LLM inference server**. It reimplements the core
mechanisms that make vLLM, TensorRT-LLM, and SGLang fast:

- **Paged KV cache** — fixed-size KV blocks plus per-sequence block tables, replacing contiguous
  max-sequence-length preallocation.
- **Continuous batching** — an iteration-level scheduler that admits, preempts, and retires
  requests between decode steps.
- **Prefix caching** — block-aligned, refcounted, shared KV blocks with LRU eviction.
- **Custom CUDA paged attention kernel** — reads KV through block tables, bound to Python via
  pybind11.
- **OpenAI-compatible HTTP server** — FastAPI, SSE streaming.

The full design rationale lives in `_private/DESIGN.md` and the phased build plan in
`_private/ROADMAP.md`. **Read both before starting a phase.** They are gitignored, so if you
cannot see them, stop and tell the user rather than guessing at the design.

### Non-goals — do not implement these without an explicit request

- Tensor or pipeline parallelism (single GPU only)
- Weight quantization (GPTQ/AWQ/FP8 weights)
- Support for model architectures beyond one Llama-style family
- Speculative decoding
- Any dependency on vLLM, TGI, or another serving framework as a runtime component

If a task seems to require one of these, stop and ask.

---

## 2. Hard rules

These are not style preferences. Violating any of them silently breaks the project's purpose.

### 2.1 Never fabricate a number

**Do not write any performance number into code comments, docstrings, the README, or any
document unless it came from a benchmark script in this repo that was actually executed and
whose raw output is committed under `results/`.**

This includes:
- No "~3x faster" in a docstring based on what you know about the literature.
- No filled-in example numbers in README tables, not even as illustrative placeholders.
- No estimated speedups in commit messages or PR descriptions.

If a document needs a number that does not exist yet, write `TODO(bench)` and leave it. The
user has been burned by unverified metrics before; treat this as the single most important
rule in this file.

### 2.2 Correctness is gated by the golden test

`tests/test_golden.py` asserts that greedy decoding produces **token-for-token identical**
output to a HuggingFace reference on a fixed prompt set.

- It must pass before any commit.
- It must pass with every attention backend (`gather`, `cuda`, and any future backend).
- It must pass with prefix caching both on and off — a prefix cache is required to be
  semantically invisible.
- Never loosen the assertion, widen a tolerance, or skip a case to make it pass. A failing
  golden test is a bug report, not an obstacle.

### 2.3 Every optimization must be ablatable

Optimizations land behind a flag with the old path kept working:

- `--no-paging` keeps the Phase 1 contiguous KV path alive forever.
- `--attn-backend=gather` keeps the Phase 2 Python gather path alive forever as the
  correctness oracle for the CUDA kernel.
- Prefix caching, chunked prefill, and CUDA graphs are each independently toggleable.

Never delete a slower reference implementation. It is the experimental control.

### 2.4 One phase at a time

The roadmap has nine phases. The current phase is recorded in §7 of this file. Do not implement
work from a later phase because it seems convenient. If the user asks for something from a
later phase, confirm they want to jump ahead before doing it.

### 2.5 Nothing in the hot loop that does not belong there

The per-step decode loop is the performance-critical path. In it:

- No Python `for` loops over sequences. Batch into tensors.
- No `.item()`, `.cpu()`, `.tolist()`, or `print()` on GPU tensors — each forces a host-device
  sync and will silently halve throughput.
- No tensor allocation. Preallocate and reuse buffers.
- No tokenizer or detokenizer calls.

If you must add something to the loop, say so explicitly in your response so the user can
evaluate the cost.

---

## 3. Repository layout

```
pagedserve/
├── AGENTS.md               # this file
├── CLAUDE.md               # pointer to this file
├── README.md               # public-facing; results go here (measured only)
├── pyproject.toml
├── setup.py                # builds the CUDA extension
├── .gitignore
│
├── _private/               # GITIGNORED — design docs, not public
│   ├── DESIGN.md
│   ├── ROADMAP.md
│   └── AGENT-PROMPTS.md
│
├── pagedserve/
│   ├── __init__.py
│   ├── config.py           # EngineConfig, CacheConfig, SchedulerConfig, ModelConfig
│   ├── engine.py           # LLMEngine: owns step(), the top-level loop
│   ├── sequence.py         # Sequence, SequenceGroup, SequenceStatus
│   │
│   ├── memory/
│   │   ├── block_manager.py    # free list, block tables, refcounts, alloc/free/fork
│   │   ├── block.py            # PhysicalBlock, BlockTable
│   │   └── prefix_cache.py     # hash chain, LRU evictor  (Phase 5)
│   │
│   ├── core/
│   │   ├── scheduler.py        # waiting/running/swapped queues, admission, preemption
│   │   └── policy.py           # preemption policy: RECOMPUTE | SWAP
│   │
│   ├── model/
│   │   ├── loader.py           # safetensors -> state dict
│   │   ├── llama.py            # manual forward pass
│   │   ├── layers.py           # RMSNorm, RoPE, GQA attention, SwiGLU
│   │   └── sampler.py          # greedy, temperature, top-p, top-k
│   │
│   ├── attention/
│   │   ├── backend.py          # AttentionBackend ABC — REQUIRED indirection layer
│   │   ├── contiguous.py       # Phase 1: SDPA over a contiguous cache
│   │   ├── gather.py           # Phase 2: index_select blocks -> scratch -> SDPA
│   │   └── cuda_paged.py       # Phase 4: wraps the compiled extension
│   │
│   ├── worker/
│   │   ├── model_runner.py     # batch assembly, slot_mapping, forward, sample
│   │   └── cache_engine.py     # KV cache allocation, swap in/out, profiling
│   │
│   └── server/
│       ├── api.py              # FastAPI app, OpenAI-compatible routes
│       └── protocol.py         # pydantic request/response schemas
│
├── csrc/                   # C++/CUDA  (Phase 4)
│   ├── bindings.cpp        # pybind11 module definition
│   ├── paged_attention.cu  # decode kernel
│   └── cache_kernels.cu    # KV scatter/copy/swap kernels
│
├── bench/
│   ├── loadgen.py          # Poisson + closed-loop load generator
│   ├── baseline_hf.py      # HuggingFace generate() baselines
│   ├── metrics.py          # TTFT / ITL / E2E percentiles, throughput, goodput
│   ├── sweep.py            # concurrency and ablation sweeps
│   └── plot.py             # regenerates every README chart from results/
│
├── tests/
│   ├── test_golden.py      # token-for-token vs HuggingFace — the gate
│   ├── test_block_manager.py
│   ├── test_scheduler.py
│   ├── test_prefix_cache.py
│   └── test_attention_parity.py   # cuda backend vs gather backend
│
├── results/                # committed raw benchmark JSON — the evidence
└── scripts/
    ├── kaggle_bootstrap.sh # clone + install + run, for notebook environments
    └── explorer_job.sbatch # SLURM job template
```

**Create files as phases need them, not up front.** Do not scaffold empty modules for future
phases.

---

## 4. Environment matrix — this repo runs in three places

The code must work in all three. This is a design constraint, not an afterthought.

| Environment | Hardware | What runs here | What does not |
|---|---|---|---|
| **macOS (Apple Silicon)** | CPU / MPS | All editing. Unit tests. Logic tests on a tiny random-init model. Server layer. Load generator against a mock backend. | Anything CUDA. Any benchmark whose number will be reported. |
| **Kaggle / Colab notebook** | T4 or P100 | GPU smoke tests. Does the kernel compile. Does a real model generate correct tokens. Rough profiling. | Reported benchmarks (shared, unpinnable hardware). bf16 on T4 — Turing has no bf16, use fp16 there. |
| **Northeastern Explorer (SLURM)** | A100 80GB (pinned) | Every benchmark that appears in the README. Nsight profiling. | Interactive development (queue waits). |

### Consequences for how you write code

1. **Device-agnostic by default.** Resolve device once in `config.py`; never hardcode `.cuda()`.
   Use `torch.device` objects passed down from config.
2. **Dtype is configurable, not assumed.** Default bf16, but fp16 must work — T4 and V100 have
   no bf16. Never hardcode `torch.bfloat16`.
3. **The CUDA extension is optional at import time.** `import pagedserve` must succeed on a Mac
   with no `nvcc`. Wrap the extension import in a try/except and fall back to the gather
   backend, logging a warning once.
4. **Every test must declare its requirements.** Mark GPU-only tests with
   `@pytest.mark.gpu` and CUDA-extension-only tests with `@pytest.mark.cuda_ext`, so
   `pytest -m "not gpu"` passes cleanly on a Mac.
5. **Never assume a persistent filesystem.** Kaggle sessions are ephemeral. All scripts must be
   runnable as `git clone && pip install -e . && python bench/...` from nothing.

---

## 5. Conventions

- **Python 3.11+.** Type hints on every public function. `from __future__ import annotations`.
- **Formatting:** `ruff format`, line length 100. Lint with `ruff check`.
- **No new runtime dependencies without asking.** Current allowed set: torch, transformers
  (loading and the reference baseline only), safetensors, fastapi, uvicorn, pydantic, numpy.
  `matplotlib` is dev-only.
- **Config over constants.** Block size, dtype, GPU utilization target, max batched tokens,
  scheduling policy — all live in `config.py` dataclasses with defaults, never as literals
  buried in code. Anything the roadmap says to sweep must be a config field.
- **Comments explain why, not what.** `# block_size=16 balances internal fragmentation against
  kernel locality` is useful. `# increment counter` is noise.
- **Logging, not printing.** Use the stdlib `logging` module. Never `print()` in library code.
- **Docstrings on the tricky parts.** The block manager, the scheduler's preemption path, and
  the CUDA kernel each need a docstring explaining the invariants they maintain. Write these
  for a reader who has not read the design doc.

### Invariants worth asserting in debug mode

- Sum of refcounts over all physical blocks equals the total block references held by all
  sequences.
- `len(free_blocks) + len(allocated_blocks) == num_blocks` always.
- A sequence's block table length equals `ceil(len(sequence) / block_size)`.
- No sequence holds a block whose refcount is zero.

Put these behind a `config.debug_invariants` flag so they can run in tests and be off in
benchmarks.

---

## 6. Testing and benchmarking protocol

### Before every commit

```bash
ruff check . && ruff format --check .
pytest -m "not gpu"                    # on Mac
pytest                                 # on a GPU machine
```

### Benchmark protocol — follow exactly or the numbers are worthless

1. One machine per result set. Record GPU model, driver, CUDA version, torch version.
2. Fixed model, dtype, and sampling params. Log all of them into the result JSON.
3. Warm up and discard the first 30 seconds or first N requests.
4. Minimum three runs; report the **median** and the run-to-run spread.
5. Report **P50, P95, P99** for every latency metric. Never a bare mean.
6. Poisson arrivals for headline results; closed-loop bursts only for isolating batch scaling.
7. Every run writes a JSON to `results/` containing the full config, the environment, and the
   raw per-request timings. Commit it.
8. `bench/plot.py` regenerates every chart from `results/`. No chart is ever hand-made.

### The result JSON must contain

```json
{
  "config": { "...": "full EngineConfig dump" },
  "environment": {
    "gpu": "...", "driver": "...", "cuda": "...", "torch": "...",
    "host": "...", "timestamp": "..."
  },
  "workload": { "dataset": "...", "arrival": "poisson", "rate": 0.0, "num_requests": 0 },
  "requests": [ { "arrival": 0.0, "first_token": 0.0, "tokens": [], "finish": 0.0 } ],
  "summary": { "ttft_p50": 0.0, "...": 0.0 }
}
```

---

## 7. Current status

**Current phase: 0 — environment, baseline, and the measurement harness.**

Phase 0 deliverables:
- [x] `pyproject.toml`, package skeleton, ruff config
- [x] `bench/metrics.py` — percentile computation from raw per-request timings
- [x] `bench/loadgen.py` — Poisson and closed-loop modes, pluggable backend
- [x] `bench/baseline_hf.py` — HuggingFace sequential and static-batched baselines
- [x] `scripts/kaggle_bootstrap.sh` and `scripts/explorer_job.sbatch`
- [ ] First baseline sweep committed to `results/` — **needs a GPU; cannot be
      closed on macOS.** Everything above is verified on CPU against a tiny
      random-init Llama, which exercises the logic but measures nothing.

Do not begin Phase 1 until every box is checked and the user confirms.

Update this section when a phase completes.

---

## 8. When to stop and ask

Stop and ask the user rather than proceeding if:

- A task requires a non-goal from §1.
- The golden test fails and the fix is not obvious within a few minutes of investigation.
- You would need to add a runtime dependency.
- You would need to change a documented design decision (block size semantics, the block-aligned
  prefix cache constraint, the attention backend interface).
- A benchmark result looks implausible — a suspiciously large speedup is more often a broken
  measurement than a real win, and reporting it would be worse than reporting nothing.
- The user asks for a number that has not been measured.

When you finish a unit of work, report what you changed, what you tested, and what you did
**not** verify. Do not describe untested code as working.
