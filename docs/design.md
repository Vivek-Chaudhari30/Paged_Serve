# PagedServe design

A public, trimmed design document (FINAL-COMPLETION.md W2). It describes what the code does
and why. It contains no performance numbers: per `AGENTS.md` §2.1, every figure must trace
to a committed raw result under `results/`, and no GPU sweep has been committed yet.

## 1. Request lifecycle

```
HTTP (FastAPI, SSE)  ->  EngineRunner  ->  LLMEngine.step()  ->  Scheduler
  pagedserve/server/      one asyncio      one forward pass      who runs, how many
  api.py                  task             per call              tokens each
                                |
                                v
                    Llama-family model (pagedserve/model/)
                                |
                                v
            attention backend: contiguous | gather | cuda
                                |
                                v
                 paged KV cache + BlockManager (pagedserve/memory/)
```

- `EngineRunner` owns the only engine loop. `engine.step()` runs in a worker thread so the
  event loop keeps streaming. Aborts that arrive mid-step are deferred until the step
  returns, because the block manager is not thread-safe.
- Detokenization runs on a single dedicated thread (`EngineRunner.detokenize`), so a slow
  decode never stalls the event loop or other streams.
- Finished sequences are freed *before* their clients are told they are done.

## 2. KV cache and block manager

- The cache is one tensor `[layers, 2, num_blocks, block_size, kv_heads, head_dim]`.
- `BlockManager` hands out fixed-size blocks and keeps a per-sequence block table. Waste is
  bounded by `block_size - 1` slots per sequence instead of `max_seq_len - len`.
- One extra *trash block* absorbs writes from padded rows (used by CUDA graph padding).
- Prefix caching: full blocks are hashed by their token content chain, refcounted and
  LRU-evicted. A block is sealed (made shareable) only after its K/V have been written.

## 3. Scheduler

Order inside `Scheduler.schedule()`:

1. retire finished sequences;
2. guarantee growth room for running sequences, preempting the most recently admitted
   (recompute, or swap to host blocks);
3. resume swapped sequences;
4. admit waiting sequences under the token and sequence budgets.

Options (`SchedulerConfig`):

| Field | Effect |
|---|---|
| `max_num_batched_tokens` | per-step token budget |
| `max_num_seqs` | per-step sequence budget |
| `preemption_policy` | `recompute` or `swap` |
| `enable_chunked_prefill` | split long prompts into budget-sized chunks; decodes are scheduled first |
| `static_batching` | admit only into an empty engine (ablation arm) |

Chunked prefill is expressed through `Sequence.num_scheduled_tokens`. The engine runs each
chunk with keys up to the end of that chunk (`seq_lens = start + chunk`) and samples only for
rows whose prefill completes in this step. Golden tests verify both scheduling variants
produce token-identical output to the static reference.

## 4. Attention backends

| Backend | Role | Path |
|---|---|---|
| `contiguous` | Phase 1 ablation arm, no paging | `attention/contiguous.py` |
| `gather` | correctness oracle: gathers the context through the block table, then SDPA | `attention/gather.py` |
| `cuda` | custom paged-attention decode kernel; prefill reuses the gather path | `attention/cuda_paged.py`, `csrc/` |

`attn_backend="cuda"` falls back to `gather` with a warning when there is no CUDA device or
the extension is not built; `config.attn_backend` is rewritten so result files record what
actually ran. `engine.default_attn_backend()` picks `cuda` only when it can run.

### 4.1 The CUDA kernel (`csrc/paged_attention.cu`)

- One thread block per `(sequence, query head)`; block dim is the next power of two >= 32
  and >= `head_dim`.
- Each thread owns one output dimension. Per token: shared-memory reduction for `q . k`,
  then an online-softmax update (`running_max`, `running_sum`, per-thread accumulator), all
  in fp32.
- K/V are read straight from physical blocks via the block table; the context is never
  gathered into scratch memory.
- GQA: `kv_head = head / (num_heads / num_kv_heads)`.
- dtypes: fp32, fp16, bf16. Positions at or beyond `seq_len` are never read.

`csrc/cache_kernels.cu` holds `reshape_and_cache` (scatter new K/V into slots; negative
slots skipped), `copy_blocks` (copy-on-write style block copies across layers) and
`swap_blocks` (host<->device block copies).

`pagedserve/attention/paged_reference.py` implements the same four ops in pure PyTorch with
the same signatures and the same online-softmax structure. CPU tests run the cuda backend
with these reference ops; GPU tests (`-m gpu`) diff the compiled ops against them.

**Status:** the CUDA sources have not been compiled or run in this repository's CI
environment (no CUDA toolkit or GPU). They must pass `pytest -m gpu tests/test_attention_cuda.py`
and the golden gate on the `cuda` backend before any claim is made about them.

## 5. CUDA graphs (`pagedserve/worker/cuda_graph.py`)

- Enabled with `enable_cuda_graphs=True`, `cuda` backend on a CUDA device only.
- One graph per batch-size bucket (1, 2, 4, 8, then every 8 up to the maximum), captured
  largest-first sharing one memory pool, after the KV cache is allocated (graphs bake in
  its address; `start()` recaptures).
- Inputs live in persistent buffers; a step copies host inputs in and replays. Padding rows
  feed token 0 at position 0, write into the trash block, and attend over one token.
- Only pure-decode batches replay; prefills and mixed batches always run eagerly.

## 6. Host synchronization

The decode step has one deliberate host sync: `sample(...).tolist()`, because the scheduler
must see the tokens to decide the next step. Metadata is built on the host from Python lists
once per step and copied once; per-layer work reuses it (`gather._StepPlan`).

## 7. What is intentionally not here

Tensor/pipeline parallelism, quantization, speculative decoding, other architectures, and any
runtime dependency on vLLM/TGI. See `AGENTS.md`.
