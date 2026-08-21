# PagedServe

A from-scratch high-throughput LLM inference server. It reimplements the core mechanisms that
make vLLM, TensorRT-LLM, and SGLang fast — a paged KV cache, an iteration-level continuous
batching scheduler, block-aligned prefix caching, and a custom CUDA paged attention kernel —
without depending on any of them at runtime.

**Status: Phase 0 of 9 — building the measurement harness. No engine exists yet.**

## Scope

Single GPU, one Llama-style architecture family (RoPE, GQA, SwiGLU, RMSNorm), unquantized
weights. Tensor/pipeline parallelism, weight quantization, broad model-zoo support, and
speculative decoding are explicitly out of scope: the thesis of this project is KV memory
management, and those are different bottlenecks.

## Results

TODO(bench) — nothing has been measured yet. Every number that eventually appears here will
trace to a committed raw result JSON under `results/`, produced by a script in `bench/` on
hardware named in the methodology section.

## Install

```bash
pip install -e ".[all]"
```

The base install is deliberately light so the benchmark harness runs on a laptop with no GPU.
`[engine]` pulls torch, `[baseline]` pulls transformers for the HuggingFace reference,
`[server]` pulls FastAPI.

## Development

```bash
ruff check . && ruff format --check .
pytest -m "not gpu"
```
