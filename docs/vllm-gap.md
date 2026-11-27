# Gap analysis: PagedServe vs vLLM

This is a qualitative, code-level comparison. **It contains no
measured ratios**: no vLLM comparison run has been committed under `results/`. When one
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

The comparison runs through one client against both servers: `bench/http_backend.py`
speaks the OpenAI streaming protocol, so the same prompts, seeds, timestamping and result
format measure PagedServe's server and `vllm serve` alike. A comparison where each side is
measured by its own harness compares the harnesses.

1. Build the extension on a GPU host and pass the GPU suite
   (`PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest`).
2. Run the `http` section of the sweep script. It starts PagedServe's server, runs
   `python -m bench.sweep http --http-arm ps-http`, stops it, then does the same for vLLM
   with `--http-arm vllm-http`:

   ```bash
   SECTIONS=http INSTALL_VLLM=1 NUM_BLOCKS=8192 bash scripts/explorer_job.sbatch
   ```

   Both servers get the same dtype, `max_seq_len`, `max_num_seqs`, block size and, with
   `NUM_BLOCKS`, the same KV capacity (`--num-gpu-blocks-override` on the vLLM side).
   vLLM's prefix cache is turned off to match PagedServe's default. vLLM lives in its own
   virtualenv (`.venv-vllm`) because it pins its own torch; installing it alongside the
   extension would replace the torch the extension was built against. Record the vLLM
   version it prints into `run_notes.txt`.
3. Each result file records `workload.http.sse_text_events` (tokens as the client counted
   them) and, when the server reports it, `usage_completion_tokens`. vLLM sends the usage
   block; PagedServe's server does not yet, so check that the two servers' event counts
   per request agree with `max_tokens` before trusting a throughput ratio.
4. Commit the raw JSONs, regenerate charts with `scripts/make_charts.sh`
   (`http_throughput_vs_concurrency.png`, `http_ttft_vs_concurrency.png`), and only then
   state a ratio.
