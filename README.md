<div align="center">

<img src="docs/img/hero.svg" alt="PagedServe: an LLM inference server rebuilt from scratch" width="100%"/>

[![Correctness](https://img.shields.io/badge/GPU%20correctness-verified%20on%20Tesla%20T4-3fb950?style=for-the-badge)](results/verification/kaggle-t4-2026-10-14/README.md)
[![Golden](https://img.shields.io/badge/golden%20vs%20HuggingFace-58%2F58-3fb950?style=for-the-badge)](results/verification/kaggle-t4-2026-10-14/README.md)
[![Tests](https://img.shields.io/badge/tests-515%20passing%20on%20cuda-58a6ff?style=for-the-badge)](results/verification/kaggle-t4-2026-10-14/README.md)
[![Benchmarks](https://img.shields.io/badge/benchmarks-need%20a%20pinned%20GPU-d29922?style=for-the-badge)](#results)

**[How it works](#how-paging-works)** ·
**[Run evidence](#run-evidence)** ·
**[Architecture](#architecture)** ·
**[Results](#results)** ·
**[Run it](#run-it)** ·
**[Docs](#further-reading)**

</div>

---

## What this is

When you send a prompt to a chatbot, a server has to hold many conversations in GPU memory at
once and generate tokens for all of them efficiently. That server is an **inference engine**.
The well-known ones are vLLM, TensorRT-LLM and SGLang.

PagedServe rebuilds their four core ideas from nothing, with no dependency on any of them at
runtime, so every mechanism can be explained from memory:

| Idea | What it fixes |
|---|---|
| **Paged KV cache** | Reserving worst-case memory per request wastes most of the GPU. Fixed-size blocks and a block table waste almost none. |
| **Continuous batching** | Static batches wait for the slowest request. An iteration-level scheduler admits and retires requests every step. |
| **Prefix caching** | Requests that share a prompt prefix recompute it. Shared, refcounted, LRU-evicted blocks skip the repeat work. |
| **Custom CUDA attention kernel** | Reading K and V through a block table needs a kernel that understands the table. |

<br/>

## How paging works

<div align="center">
<img src="docs/img/paged-kv-demo.svg" alt="Animation: three requests grow block by block, one finishes, and a new request reuses its memory" width="100%"/>
</div>

Each request keeps a **block table** that maps its logical blocks to physical ones. Blocks need
not be contiguous, so memory is never reserved up front and a finished request hands its blocks
straight back to the pool. The measured effect on this repo's own harness:

<div align="center">
<img src="docs/img/utilization.svg" alt="Bar chart: allocated KV memory that is actually used, contiguous versus paged" width="100%"/>
</div>

<br/>

## Run evidence

<div align="center">
<img src="docs/img/verification.svg" alt="Checklist of passing GPU verification runs on a Tesla T4" width="100%"/>
</div>

Two independent notebook runs of this repository on a **Tesla T4 (sm_75, float16)**, with raw
logs committed in [`results/verification/kaggle-t4-2026-10-14/`](results/verification/kaggle-t4-2026-10-14/).

What that establishes: the hand-written CUDA kernels compile, and the engine produces the **same
tokens as HuggingFace** on a real GPU, across both attention backends, with prefix caching on
and off, and with forced preemption (recompute and swap) leaving the output unchanged.

What it does **not** establish: speed. A shared, unpinnable GPU cannot produce a reportable
timing, so nothing above is timed. See [Results](#results).

<details>
<summary><b>The exact numbers, per run</b></summary>

| Check | Result |
|---|---|
| CPU suite (`-m "not gpu"`) | 510 passed |
| CUDA extension build (nvcc, sm_75) | built and importable, 7/7 extension tests |
| GPU smoke checks | 6/6: KV profiling, swap round-trip, e2e generation, paged = contiguous bit-for-bit, preemption invisible |
| `pytest -m gpu` | 52 passed, 12 skipped (cuda-backend cases that need `device=cuda`) |
| CUDA kernel + CUDA graph + extension tests, `device=cuda` | 73 passed, 1 skipped |
| Golden suite vs HuggingFace on `device=cuda`, float16 | 58 passed, 0 skipped |
| Whole suite on `device=cuda` (minus golden) | 515 passed, 1 skipped |

The one skip applies only when the extension is not built, so it is expected here.
Environment: torch 2.11.0+cu128, CUDA 12.8, Tesla T4, float16.

</details>

<br/>

## Build status

<div align="center">
<img src="docs/img/phases.svg" alt="Phase-by-phase build status" width="100%"/>
</div>

<details>
<summary><b>What each phase is, and what is verified</b></summary>

| Phase | What | State |
|---|---|---|
| 0 | Benchmark harness: load generator, percentile metrics, HuggingFace baselines | complete |
| 1 | Naive engine, contiguous KV cache | complete |
| 2 | Paged KV cache (Python gather backend) | complete |
| 3 | Continuous batching scheduler, recompute and swap preemption | complete |
| 4 | Custom CUDA paged-attention and cache-copy kernels | **verified on a T4** (builds, passes the CUDA tests and golden) |
| 5 | Block-aligned prefix caching | complete |
| 6 | OpenAI-compatible HTTP server with SSE streaming and full sampling | complete |
| 7 | Chunked prefill, static-batching mode, CUDA graph decode, async detokenization | **verified on a T4** (`test_cuda_graph.py` and golden pass on cuda) |
| 8 | Ablations | tooling complete (CLI flags, `bench/sweep.py`, charts); **no sweep run** |
| 9 | Writeup | design, vLLM gap and Q&A docs written; results await a sweep |

</details>

<br/>

## Architecture

```mermaid
flowchart LR
    C([Client]) -->|OpenAI-style request| S[HTTP server<br/>FastAPI + SSE]
    S --> Q[Scheduler<br/>continuous batching]
    Q -->|admit / preempt| B[Block manager<br/>free pool + block tables]
    B --- P[(Prefix cache<br/>refcounted, LRU)]
    Q --> E[Model runner<br/>Llama-style forward]
    E --> A{{Attention backend}}
    A -->|contiguous| A1[Reference]
    A -->|gather| A2[Python gather]
    A -->|cuda| A3[CUDA paged kernel<br/>+ CUDA graphs]
    A3 --> K[(GPU KV cache blocks)]
    A2 --> K
    E --> D[Sampler] --> T[Async detokenizer] --> S
```

```mermaid
sequenceDiagram
    participant R as Request
    participant Q as Scheduler
    participant B as Block manager
    participant G as GPU
    R->>Q: prompt arrives
    Q->>B: allocate blocks (reuse any cached prefix)
    loop every step
        Q->>G: one batched forward over all running requests
        G-->>Q: next token per request
        Q->>B: append a block only when one fills
    end
    Q->>B: finished, return blocks to the pool
    Note over Q,B: out of memory, so preempt a request<br/>(recompute or swap to host) and resume later
```

<br/>

## Results

> **No performance has been measured yet, and none is claimed.**

This is deliberate. The rule in [`AGENTS.md`](AGENTS.md) §2.1 is absolute: every number in this
repository must trace to a committed raw result JSON under `results/`, produced by a script in
`bench/`, on named hardware. Timings from the shared Kaggle GPU above do not qualify, because it
cannot be pinned.

The sweep tooling is finished and rehearsed end to end off the GPU; what remains is one run on
a pinned GPU.

<details>
<summary><b>What the sweeps will produce</b></summary>

| Sweep | Question it answers |
|---|---|
| `arms` | Which feature caused which gain: paging, continuous batching, prefix cache, the CUDA kernel. |
| `block-size` | Why block size 16: the fragmentation versus kernel-efficiency trade-off. |
| `preemption` | Recompute versus swap under memory pressure. |
| `prefix` | Time to first token with and without a shared prefix. |

```bash
python -m bench.sweep {arms,block-size,preemption,prefix} --help
scripts/make_charts.sh results docs/charts
```

</details>

<br/>

## Run it

```bash
pip install -e ".[all]"                 # base install is light; [engine] torch, [baseline] transformers, [server] FastAPI
python setup.py build_ext --inplace     # optional CUDA extension; never through pip (isolated build has no torch)
```

```bash
python -m pagedserve.server --model Qwen/Qwen2.5-0.5B-Instruct --num-blocks 512 \
    [--attn-backend {contiguous,gather,cuda}] [--no-paging] [--static-batching] \
    [--enable-prefix-caching] [--enable-chunked-prefill] [--enable-cuda-graphs] \
    [--preemption-policy {recompute,swap}]
```

`--attn-backend` defaults to `cuda` when the extension is built and a GPU is present, else
`gather`. Pass `--num-blocks` explicitly for any run that must not run out of memory.

<details>
<summary><b>Reproduce the GPU verification yourself, free, on Kaggle</b></summary>

Open a Kaggle notebook with a **GPU** and **Internet on**, then run:

```python
!bash <(curl -sSL https://raw.githubusercontent.com/Vivek-Chaudhari30/Paged_Serve/main/scripts/kaggle_bootstrap.sh)
```

It builds the CUDA extension and runs the smoke checks and the golden test on the GPU.
Use **Save Version → Save & Run All** and it keeps running with your laptop closed.

</details>

<details>
<summary><b>Development and tests</b></summary>

```bash
ruff check . && ruff format --check .
pytest -m "not gpu"
```

On a GPU machine, `pytest -m gpu` selects only the tests that cannot run without CUDA. It is
**not** the full GPU suite and the golden gate is not in it. Run the whole suite against the GPU:

```bash
PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest
```

</details>

<br/>

## Scope

Single GPU, one Llama-style architecture family (RoPE, GQA, SwiGLU, RMSNorm), unquantized
weights. Tensor and pipeline parallelism, quantization, broad model support and speculative
decoding are out of scope on purpose: this project is about KV memory management, and those are
different bottlenecks.

## Further reading

- [`docs/how-it-works.md`](docs/how-it-works.md): a plain-language explanation of the whole system.
- [`docs/design.md`](docs/design.md): design decisions and why.
- [`docs/vllm-gap.md`](docs/vllm-gap.md): where this is slower than vLLM, and why.
- [`docs/interview-notes.md`](docs/interview-notes.md): the questions this project should survive.
- [`docs/aws-runbook.md`](docs/aws-runbook.md): running the benchmark on AWS.
- [`AGENTS.md`](AGENTS.md): the rules every contributor and coding agent works under.
