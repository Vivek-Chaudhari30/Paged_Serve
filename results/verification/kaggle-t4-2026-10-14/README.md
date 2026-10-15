# GPU correctness verification — Kaggle Tesla T4, 2026-10-14

Two notebook runs of the code at `main` (a7244ef, PR #9 merged), on a **Tesla T4 (sm_75,
float16)**, torch 2.11.0+cu128, CUDA 12.8. Raw logs are in this directory.

| Check | Result | Log |
|---|---|---|
| CPU suite (`-m "not gpu"`) | 510 passed | run1 |
| CUDA extension build (nvcc, sm_75) | built, importable; 7/7 extension tests | run1 |
| GPU smoke checks (profiling, swap, e2e, paged==contiguous, preemption) | 6/6 passed | run1 |
| Golden vs HuggingFace on GPU, float16 | 58/58 passed | run1, run2 |
| `pytest -m gpu` | 52 passed, 12 skipped (cuda-backend cases needing `device=cuda`) | run2 §A |
| CUDA kernel + CUDA-graph + extension tests, `device=cuda` | 73 passed, 1 skipped | run2 §B |
| Golden suite on `device=cuda` | 58 passed, 0 skipped | run2 §C |
| Whole suite on `device=cuda` (minus golden) | 515 passed, 1 skipped | run2 §D |

## What this is and is not

This is **correctness evidence**: the custom paged-attention kernel, cache-copy kernels,
CUDA-graph decode and chunked prefill produce the same tokens as HuggingFace on a real GPU.

It is **not benchmark evidence**. Kaggle GPUs are shared and cannot be pinned, so no timing
from these runs is reported anywhere (AGENTS.md §4). Throughput/latency numbers still need a
pinned GPU.

The single skip in §B/§D (`test_attention_cuda.py:253`, "extension is built here") applies
only when the extension is *not* built, so it is expected to skip here.
