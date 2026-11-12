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

This describes the tree as it exists. If you add or remove a file, update it.

```
.
├── AGENTS.md               # this file
├── CLAUDE.md               # pointer to this file
├── README.md               # public-facing; results go here (measured only)
├── LICENSE
├── pyproject.toml          # package metadata, extras, ruff and pytest config
├── setup.py                # builds the CUDA extension (explicitly, never via pip)
├── .github/workflows/ci.yml  # ruff + `pytest -m "not gpu"` on every push / PR
│
├── _private/               # GITIGNORED — design docs and plans, not public
│   ├── DESIGN.md
│   ├── ROADMAP.md
│   ├── AGENT-PROMPTS.md
│   └── FINAL-COMPLETION.md # the completion plan (moved out of the public tree)
│
├── pagedserve/
│   ├── config.py           # EngineConfig, CacheConfig, SchedulerConfig, ModelConfig; device/dtype
│   ├── engine.py           # LLMEngine: step(), batch assembly, slot_mapping (no model_runner.py)
│   ├── sequence.py         # Sequence, SequenceGroup, SequenceStatus
│   ├── extension.py        # optional loader for pagedserve._C (imports torch first)
│   ├── memory/
│   │   ├── block.py            # PhysicalBlock, BlockTable
│   │   ├── block_manager.py    # free list, block tables, refcounts, alloc/free/fork, CoW
│   │   └── prefix_cache.py     # hash chain, LRU evictor
│   ├── core/
│   │   ├── scheduler.py        # waiting/running/swapped queues, admission, preemption, chunking
│   │   └── policy.py           # preemption policy: RECOMPUTE | SWAP
│   ├── model/
│   │   ├── loader.py           # safetensors -> state dict
│   │   ├── llama.py            # manual forward pass
│   │   ├── layers.py           # RMSNorm, RoPE, GQA attention, SwiGLU
│   │   └── sampler.py          # greedy, temperature, top-p, top-k, repetition penalty
│   ├── attention/
│   │   ├── backend.py          # AttentionBackend ABC — REQUIRED indirection layer
│   │   ├── contiguous.py       # Phase 1: SDPA over a contiguous cache
│   │   ├── gather.py           # Phase 2: index_select blocks -> scratch -> SDPA (the oracle)
│   │   ├── cuda_paged.py       # Phase 4: wraps the compiled extension, falls back to gather
│   │   └── paged_reference.py  # pure-PyTorch twins of the CUDA ops, same signatures
│   ├── worker/
│   │   ├── cache_engine.py     # KV cache allocation, swap in/out, capacity profiling
│   │   └── cuda_graph.py       # Phase 7: bucketed decode CUDA graphs
│   └── server/
│       ├── api.py              # FastAPI app, EngineRunner, OpenAI-compatible routes
│       ├── protocol.py         # pydantic request/response schemas
│       └── __main__.py         # `python -m pagedserve.server` CLI and ablation flags
│
├── csrc/
│   ├── bindings.cpp        # pybind11 module definition
│   ├── paged_attention.cu  # decode kernel
│   ├── cache_kernels.cu    # reshape_and_cache, copy_blocks, swap_blocks
│   └── trivial.cu          # build canary
│
├── bench/
│   ├── loadgen.py          # Poisson + closed-loop load generator
│   ├── baseline_hf.py      # HuggingFace generate() baselines
│   ├── pagedserve_backend.py  # PagedServe driver for loadgen
│   ├── metrics.py          # TTFT / ITL / E2E percentiles, throughput, goodput
│   ├── sweep.py            # ablation sweeps: arms, block-size, preemption, prefix
│   └── plot.py             # regenerates every README chart from results/
│
├── tests/                  # one file per module; test_golden.py is the gate,
│                           # test_attention_cuda.py diffs the kernels against the reference
├── docs/                   # design.md, how-it-works.md, vllm-gap.md, design-qa.md, aws-runbook.md
├── results/                # committed raw benchmark JSON — the evidence (does not exist yet)
└── scripts/
    ├── aws_bench.sh        # EC2 launch -> sweep -> teardown
    ├── explorer_job.sbatch # SLURM job template (A100)
    ├── kaggle_bootstrap.sh # clone + install + run, for notebook environments
    ├── make_charts.sh      # results/ -> docs/charts/
    ├── fetch_dataset.sh    # ShareGPT download
    ├── conductor_setup.sh  # links _private/ into a fresh worktree
    └── gpu_smoke.py, gpu_diagnose_*.py  # GPU smoke test and diagnostics
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

# On a GPU machine. The env vars are not optional: the golden gate is
# device-parametric, so a bare `pytest` runs it on CPU and proves nothing about
# the GPU.
PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest
```

**`pytest -m gpu` is not the GPU suite.** It selects only the ~10 tests that
cannot run at all without CUDA, and the golden gate is not among them. A clean
`-m gpu` run on a metered GPU box reads exactly like a passing GPU suite while
the commit gate never ran. Use the command above.

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

**Current phase: the first GPU benchmark sweep.** All nine phases have code. What is left is
GPU work: verify the CUDA kernel and CUDA graphs, then run the sweep and commit `results/`.

| Phase | State |
|---|---|
| 0 Benchmark harness | complete; CPU-verified |
| 1 Naive engine, contiguous KV | complete; golden-verified on CPU and T4 |
| 2 Paged KV, gather backend | complete; golden-verified on CPU and T4 |
| 3 Continuous batching | complete; golden-verified on CPU and T4, both preemption policies |
| 4 CUDA paged attention | **code complete, never compiled.** CPU-tested through `paged_reference.py` only |
| 5 Prefix caching | complete; golden-verified on CPU and T4, cache on and off |
| 6 OpenAI-compatible server | complete; CPU-verified |
| 7 Chunked prefill, CUDA graphs, async detok | code complete; chunked prefill golden-verified on CPU; CUDA graphs never run |
| 8 Ablations | tooling complete (`bench/sweep.py`, CLI flags, `scripts/make_charts.sh`); no sweep run |
| 9 Writeup | `docs/design.md`, `docs/vllm-gap.md`, `docs/design-qa.md` written; results await a sweep |

`results/` does not exist. Phases 0, 2, 3, 5 and 6 each hang on exactly one unchecked box, and
it is the same box: run the sweep on a GPU. One successful GPU session closes all five.

### Next GPU session, in order

1. `python setup.py build_ext --inplace`, then `pytest -m gpu tests/test_attention_cuda.py`
   (compiled kernels diffed against the reference ops).
2. The golden gate on the `cuda` backend:
   `PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest`.
3. Only then, `python -m bench.sweep ...` and commit the raw JSON under `results/`.

### Verified on a Tesla T4 (sm_75, torch 2.10+cu128, float16 and float32)

Correctness only. A Kaggle T4 is shared and unpinnable, so its timings are not evidence (§4).

- [x] Golden gate 44/44, `contiguous` and `gather`, both dtypes, prefix caching on and off
- [x] Paged and contiguous produce bit-identical logits
- [x] Forced preemption invisible in the output, both policies
- [x] Every configuration deterministic across repeated runs
- [x] CUDA extension build canary 13/13, including the fresh-subprocess loader test
- [x] `SwapSpace` round-trip through genuinely pinned host memory

### Known gaps

- **Capacity profiling is optimistic unless a forward pass is injected.** Without
  `run_max_shape_forward`, `profile_num_blocks` counts activation memory as zero — it handed
  13.1 GB of a 14.6 GiB T4 to KV. Pass `--num-blocks` explicitly for any run that must not OOM.
- **The client-disconnect test failed on a T4** while passing on CPU.
  `scripts/gpu_diagnose_disconnect.py` is the committed diagnostic; Phase 7 reworked abort
  handling (deferred aborts, prompt block release), and the test must be re-run on a GPU.
- **Copy-on-write is not reachable from prefix caching alone** — only full blocks are cached
  and a full block is never written again. It exists for `fork()` (`n > 1`).
- **Prefix reuse needs staggered arrivals.** Requests admitted in the same step cannot share:
  nothing has been through a forward pass yet. A 0% hit rate on a simultaneous burst is correct.

### Things that have bitten this project — do not relearn them

- **Never trust `torch.cuda.is_bf16_supported()` on Turing.** It returns True because it
  counts emulation. `resolve_dtype` picks by compute capability; ask it, do not reimplement it.
- **Build the extension explicitly, never via `pip install`.** pip builds in an isolated
  environment with no torch, and the build silently does not happen.
- **`extension.py` must import torch before `pagedserve._C`.** libc10/libtorch live in torch's
  package directory, not on the loader path. pytest imports torch first, so only a
  fresh-subprocess test catches a regression.
- **`setup.py` passes no `-arch`.** A hardcoded architecture produces a binary that will not
  load on any other card.
- **Greedy must stay bit-for-bit stable.** An all-greedy batch short-circuits to `argmax` and
  never touches softmax or multinomial; every golden test depends on it.
- **Stop strings live in the server, not the engine.** They need a detokenizer, and §2.5 keeps
  the tokenizer out of the hot loop. The engine handles stop *tokens* only.
- **`apply_chat_template(tokenize=True)` returns a `BatchEncoding` on transformers v5** and a
  list of ints on v4. `list()` on a `BatchEncoding` iterates its keys.
- **Bookkeeping ratios are not benchmarks.** The KV utilization figures in the README are
  allocated-versus-live accounting from CPU correctness runs. Quote `max_seq_len` and the
  workload with them, or they mean nothing.

Target model: `Qwen/Qwen2.5-0.5B-Instruct` (ungated). Llama-style, but with Q/K/V biases and
tied embeddings; both are read from the checkpoint, never branched on by name.

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
