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

**Current phase: 3 — continuous batching scheduler.**

### Phase 4 step 1 — CUDA extension scaffolding: VERIFIED on a T4

- [x] `nvcc` compiles `csrc/trivial.cu`; links `pagedserve._C` against libtorch
- [x] Architecture inferred as `sm_75` from the visible GPU — `setup.py`
      deliberately passes no `-arch`, since a hardcoded one yields a binary that
      will not load on any other card
- [x] Build canary passes: round-trip, non-contiguous input, empty tensor,
      1M elements, CPU rejection, dtype rejection, no input mutation
- [x] `importable: True` from a fresh process with torch not previously imported

Build it explicitly, never via `pip install` — pip builds in an isolated
environment with no torch, so the extension silently would not build:

```bash
python setup.py build_ext --inplace
```

**`extension.py` must import torch before `pagedserve._C`.** The extension links
against libc10/libtorch, which live in torch's package directory rather than on
the loader's search path. pytest imports torch first, so a test suite cannot
catch this — a fresh-subprocess regression test does.

### GPU verification status (Tesla T4, sm_75, torch 2.10+cu128, float16)

Correctness is verified on real CUDA hardware. Nothing below is a benchmark —
a Kaggle T4 is shared and unpinnable, so its timings are not evidence (§4).

- [x] Golden gate: **39/39 pass on GPU**, both backends, in float16 *and* float32
- [x] `profile_num_blocks` measurement branch (see the known gap below)
- [x] `SwapSpace` round-trip through genuinely pinned host memory
- [x] Paged and contiguous produce **bit-identical logits** (max diff 0.000000)
- [x] Forced preemption invisible in the output, both policies
- [x] Every configuration deterministic across repeated runs

**Do not run on emulated bfloat16.** Turing has no native bf16, and
`torch.cuda.is_bf16_supported()` returns True there anyway because it counts
emulation. `resolve_dtype` picks by compute capability instead; anything that
selects a dtype must ask it rather than reimplement the check.

**Known gap:** `profile_num_blocks` runs no profiling forward pass, so
activation memory counts as zero and the block count is optimistic — it handed
13.1 GB of a 14.6 GiB card to KV. Pass `--num-blocks` explicitly for any run
that must not OOM. A two-pass profile is the real fix.

Phase 3 deliverables:
- [x] `core/scheduler.py` — waiting/running/swapped queues, `schedule()` every
      iteration, admission budgeted on BOTH `max_num_batched_tokens` and
      `max_num_seqs`
- [x] Retire on EOS / max_tokens, freeing blocks in the same step
- [x] `core/policy.py` — RECOMPUTE and SWAP, selectable by config, tail
      victim selection with the fairness tradeoff documented
- [x] Ragged batch assembly — one step mixes prefills of different lengths with
      single-token decodes, flattened with cumulative sequence lengths, no padding
- [x] Forced-preemption tests for BOTH policies: resumed sequences produce
      output identical to an unpreempted run
- [ ] Full concurrency sweep committed to `results/` — **needs a GPU.**

The engine is now benchmarkable through the Phase 0 harness. Three ablation
arms, one command each — the moment a GPU is available these produce committable
result JSON:

| Arm | Flag | Isolates |
|---|---|---|
| contiguous + static | `--no-paging` | the naive floor |
| paged + static | `--static-batching` | paging alone |
| paged + continuous | (default) | iteration-level scheduling |

```bash
python bench/loadgen.py --backend pagedserve --model Qwen/Qwen2.5-0.5B-Instruct \
    --mode closed --concurrency 32 --num-requests 256 --max-tokens 128 \
    --output results/<run>.json
```

Continuous batching output is token-identical to static batching, and stays
identical under forced preemption with either policy.

**Previous phase: 2 — paged KV cache (Python gather path).**

Phase 2 deliverables:
- [x] `memory/block.py`, `memory/block_manager.py` — free list, block tables,
      refcounts, `can_allocate`/`allocate`/`append_slot`/`free`/`fork`
- [x] `worker/cache_engine.py` — capacity profiling, one preallocation
- [x] Write path: flat `slot_mapping`, one fused `index_copy_`, no per-sequence loop
- [x] `attention/gather.py` — the correctness oracle, kept forever
- [x] Paging off via `attn_backend="contiguous"` (the `--no-paging` path).
      **Not yet a CLI flag** — there is no engine CLI until Phase 6; Phase 8
      makes it first-class.
- [x] `tests/test_block_manager.py` with the §5 invariants
- [ ] Throughput comparison vs Phase 1 — needs a GPU; a CPU timing is not evidence.

Measured KV utilization, 4 prompts, `block_size=16` (CPU/fp32, so pure
allocated-versus-live bookkeeping — device-independent):

| generated tokens | contiguous | paged |
|---|---|---|
| 16 | 2.1% | 68.0% |
| 48 | 5.1% | 80.9% |
| 96 | 8.6% | 87.8% |
| 256 | — | 94.4% |

Paged utilization rises with length because waste is bounded by
`block_size - 1` in the last block only, instead of by `max_seq_len`. Output is
bit-identical between the two backends.

**Previous phase: 1 — naive engine with a contiguous KV cache.**

Phase 1 deliverables:
- [x] `model/loader.py` — safetensors into a state dict
- [x] `model/layers.py` — RMSNorm, RoPE, GQA attention, SwiGLU
- [x] `model/llama.py` — manual forward pass with KV cache write hooks
- [x] `attention/backend.py` ABC + `attention/contiguous.py`
- [x] Static batching loop, left padding, greedy decode
- [x] Memory instrumentation: allocated vs live KV bytes
- [x] `tests/test_golden.py` — token-for-token vs HuggingFace. **This is now
      the commit gate.**

Target model: `Qwen/Qwen2.5-0.5B-Instruct` (ungated, so no HF token is needed
in any of the three environments). It is Llama-style but biases Q/K/V and ties
its embeddings; both are read from the checkpoint, not branched on by name.

Measured utilization: **2.1%** on an 8-prompt heavy-tailed batch with
`max_seq_len=2048`, 32 new tokens each (CPU/fp32). The ratio is pure
allocated-versus-live bookkeeping and so is device-independent, but it depends
entirely on `max_seq_len` and the workload — quote both or the number means
nothing.

**Previous phase: 0 — environment, baseline, and the measurement harness.**

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
