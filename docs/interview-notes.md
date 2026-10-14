# Design questions and answers

FINAL-COMPLETION.md W6. Answers are about mechanisms in this codebase. Any performance
figure must come from `results/`; there are none committed yet, so none appear here.

**Why paging?** A contiguous cache reserves `max_seq_len` slots per sequence up front.
Most sequences never use them, so memory that could hold more concurrent sequences sits
empty. Fixed-size blocks plus a block table bound the waste to under one block per sequence.
The repo's CPU bookkeeping run (README, "What has been verified") shows the allocated-vs-live
ratio; it is accounting, not a speed claim.

**Why block size 16?** A trade-off: smaller blocks waste less memory at the tail but mean
longer block tables and more scattered reads per token; larger blocks read better but waste
more. 16 is the common default; `bench.sweep block-size` exists to measure it on real
hardware instead of assuming it.

**Why is decode memory-bound?** Each decode step reads every weight and the whole KV context
to produce one token per sequence, so arithmetic per byte is low. Batching more sequences
amortizes the weight reads, which is why capacity (paging) and scheduling (continuous
batching) matter.

**Why does continuous batching need paging?** Continuous batching admits sequences whenever
there is room. With a contiguous cache, "room" means a whole `max_seq_len` slab; with paging
it means a few blocks, so admission is frequent and fine-grained.

**What does chunked prefill buy?** A long prompt otherwise occupies a whole step, stalling
every decoding sequence (ITL spikes). Chunking caps the prefill work per step and schedules
decodes first.

**Where is PagedServe slower than vLLM, and why?** See `docs/vllm-gap.md`: kernel maturity
(split-K, vectorized loads), prefill kernels, CUDA graph coverage, and Python scheduler
overhead.

**What next?** Validate and profile the CUDA kernel on a GPU, then optimize one change at a
time (vectorized loads, shared-memory tiling, split-K, warp shuffles) with Nsight between
each; use FlashAttention varlen for prefill.
