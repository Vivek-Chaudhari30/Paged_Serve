# Gap analysis: PagedServe vs vLLM

FINAL-COMPLETION.md W4. This is a qualitative, code-level comparison. **It contains no
measured ratios**: no vLLM comparison run (B5) has been committed under `results/`. When one
is, its numbers belong in the README with a link to the raw files, not here.

| Area | PagedServe | vLLM | Expected impact |
|---|---|---|---|
| Decode attention | one v1 kernel: one block per (seq, head), scalar loads, no split-K | v1/v2 kernels with vectorized loads, warp-level reductions, split-K partitioning for long contexts | largest expected gap at long contexts and large batches |
| Prefill attention | gather + SDPA (cuda backend reuses it) | FlashAttention varlen | prefill-heavy workloads |
| CUDA graphs | decode-only, bucketed, cuda backend only | decode graphs with broad coverage, piecewise compilation | CPU launch overhead at small batch |
| Scheduler | chunked prefill and static/continuous arms; Python | chunked prefill, priority, multi-step scheduling, async output processing | CPU overhead per step |
| Host syncs | one `.tolist()` per step; host-built metadata | overlapped CPU/GPU, persistent buffers | CPU-bound regime |
| Sampling | PyTorch ops per step | fused/flashinfer samplers | small |
| Prefix caching | block-hash, refcount, LRU | same design plus more eviction policies | small |
| Kernels for cache ops | `reshape_and_cache`, `copy_blocks`, `swap_blocks` | same set, tuned | small |
| Quantized KV / weights | out of scope | supported | out of scope by design |

## How to turn this into numbers

1. Build the extension on a GPU host and pass `pytest -m gpu` plus the golden gate with
   `PAGEDSERVE_TEST_DEVICE=cuda`.
2. Run `python -m bench.sweep arms ...` for PagedServe and the equivalent vLLM benchmark on
   the same machine, model, dataset and seeds.
3. Commit the raw JSONs, regenerate charts with `scripts/make_charts.sh`, and only then
   state a ratio.
