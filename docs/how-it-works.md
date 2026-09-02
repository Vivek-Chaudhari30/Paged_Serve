# How PagedServe works — a plain-language explanation

This document explains what this project does and why, assuming no background in GPUs,
CUDA kernels, or model serving. Every technical term is defined the first time it appears.

It is a companion to `AGENTS.md` (the rules) and the private design docs (the rationale).
Those are written for someone building the system. This one is written for someone trying
to understand it, or to explain it out loud.

> **On numbers.** This project has a hard rule (`AGENTS.md` §2.1): never write down a
> performance figure that did not come out of a script in this repo that was actually run.
> As of this writing **no performance measurement exists yet** — there is no `results/`
> directory. Every figure below is labelled with what kind of number it is: arithmetic you
> can re-derive, a citation from a published paper, or a bookkeeping ratio from a
> correctness run on a laptop. None of them are speed claims.

---

## 1. What this program actually does

When you type a message to an AI assistant, something on the other end runs the model and
sends words back. That something is an **inference server**.

- **Inference** — using a trained model to produce answers. As opposed to **training**,
  which is teaching the model in the first place. This project only does inference.
- **Serving** — doing inference for many users at once, over a network, continuously.

PagedServe is a homemade inference server. It holds a model in memory, accepts requests
over HTTP, and streams words back as they are generated.

It runs on a **GPU** (a graphics card). The one thing to understand about a GPU is that it
is *not* fast at doing a single thing quickly. It is fast at doing thousands of identical
things simultaneously. Almost every design decision in this project follows from that.

---

## 2. The model's notebook (the KV cache)

The model writes its answer one word at a time. To choose word 500 it must consider
everything before it — your question, plus the 499 words it already wrote.

Re-reading all of that from scratch for every new word would make long answers get slower
and slower. So instead, for every word it has seen, the model writes down two small
summaries and keeps them. Those summaries are called **keys** and **values**, and the pile
of them is the **KV cache**.

Think of it as the model's scratch notebook: it consults its notes instead of re-reading
the whole conversation.

**The notebook is much bigger than you would guess.** The size for one sequence is:

```
kv_bytes = 2 (one K and one V)
         x number of layers
         x number of KV heads
         x head dimension
         x sequence length
         x bytes per number
```

For a small model (16 layers, 8 KV heads, head dimension 64, 2 bytes per number):

```
per token = 2 x 16 x 8 x 64 x 2 = 32,768 bytes = 32 KB per word
```

*(Arithmetic — re-derivable, not a measurement.)*

So a 2,000-word conversation costs about **64 MB of notes for one user**. On a 24 GB card,
after the model itself and its working space take their share, there is room for roughly
300 conversations of that length.

That number is your capacity. Not how fast the chip is — **how many notebooks fit.** This
is the single most important idea in the project.

---

## 3. The waste that started all this

When a request arrives you do not know how long the answer will be. It might be 20 words,
it might be 2,000. The obvious implementation, which is roughly what HuggingFace's
`generate()` does, hands every request a full-size notebook up front — room for the
maximum, just in case.

Most requests write a hundred words and leave. The rest of the notebook sits blank,
reserved, and unusable by anyone else for the entire life of that request. And you cannot
lend out the blank part, because the notebook has to be one continuous chunk in memory —
the request might still grow into it.

Three kinds of waste, with their proper names:

- **Internal fragmentation** — space reserved inside a request's own allocation that it
  never uses.
- **Reservation fragmentation** — space held for a request's *future* tokens, which cannot
  be lent out even though nothing is in it yet.
- **External fragmentation** — unusable gaps left between differently-sized allocations.

**How bad is it in practice?** The vLLM paper (Kwon et al., SOSP 2023) measured effective
KV utilization in existing systems at **20.4%–38.2%** *(citation from published
literature)*. This project's own naive implementation measured **2.1%** on an 8-prompt
heavy-tailed batch with `max_seq_len=2048` and 32 new tokens each *(a bookkeeping ratio —
allocated bytes versus bytes holding real tokens — measured on CPU in float32; it is
device-independent but depends entirely on `max_seq_len` and the workload, so it is
meaningless quoted without both).*

---

## 4. The fix: loose-leaf pages instead of notebooks

Stop handing out notebooks. Chop the memory into small fixed-size **blocks** — 16 words
each — and keep them all in one shared pool. A request takes one block. When it fills it,
it takes another. When the request finishes, every block it held goes straight back to the
pool for the next person.

Each request keeps a small index: "my block 0 is pool-block #43, my block 1 is pool-block
#7." That index is called a **block table**. The blocks do not have to sit next to each
other in memory, and that turns out to matter enormously (see §6).

```
CONTIGUOUS - one big reservation per request

  request A   [####........................................]   reserved 2048, using ~120
  request B   [########....................................]   reserved 2048, using ~340
                                                               nothing left for anyone else

PAGED - 16-word blocks from one shared pool

  pool        [A][B][ ][ ][A][ ][B][ ][ ][B][ ][ ][ ][B][ ][ ]
  request A   block table: 0, 4
  request B   block table: 1, 6, 9, 13
                                                               10 blocks free right now
```

The worst possible waste is now **15 blank slots in the last block**, instead of nearly
2,000. Measured on this codebase at `block_size=16` *(CPU/float32 bookkeeping ratios, not
timings)*:

| generated words | contiguous | paged |
|---|---|---|
| 16  | 2.1% | 68.0% |
| 48  | 5.1% | 80.9% |
| 96  | 8.6% | 87.8% |
| 256 | —    | 94.4% |

Utilization *rises* with length under paging because the waste is capped at one partial
block regardless of how long the answer gets.

### This is not a new idea — it is virtual memory

The operating system on your laptop has solved exactly this problem since the 1960s. A
program wants a big continuous address space; physical RAM is fragmented and shared. The
answer was: chop RAM into fixed-size **pages**, give each program a **page table** mapping
its addresses to real ones, hand out pages on demand, and share read-only pages between
programs using **reference counting**.

PagedServe applies it one-for-one. That is where the name comes from.

| Operating system | PagedServe |
|---|---|
| Process | Sequence (one request) |
| Page / frame | KV block (16 word-slots) |
| Page table | Block table |
| Page fault, allocate a frame | Sequence crosses a block boundary, take a block |
| Swap to disk | Copy KV blocks out to CPU memory |
| Shared library page, reference counted | Shared prefix block, reference counted |
| Copy-on-write | Fork a shared block when two writers diverge |

**Reference counting** just means each block records how many requests are currently using
it. A block is only reclaimed when that count reaches zero. **Copy-on-write** means if two
requests share a block and one of them needs to write into it, it gets its own private copy
first, so it cannot corrupt the other.

---

## 5. Why batching is the whole game

Here is the piece that turns a memory trick into a speed trick.

To produce **one** next word for **one** user, the GPU must read the entire model — billions
of numbers — out of its memory. That read is the expensive part. It is like driving a truck
to a warehouse: the trip costs about the same whether you bring back one box or thirty.

So if 32 users all need their next word at the same moment, one trip serves all 32. Serving
32 users costs barely more than serving 1. That is **batching**, and it is why every serving
system fights to keep the batch full.

This is what people mean by the two phases of generation having opposite bottlenecks:

| | **Prefill** (reading the prompt) | **Decode** (writing the answer) |
|---|---|---|
| Words handled per step | the whole prompt at once | exactly one per request |
| Limited by | raw arithmetic — **compute-bound** | fetching data from memory — **memory-bandwidth-bound** |
| Benefit from batching | modest, the chip is already busy | **enormous**, it amortizes the memory read |

- **Compute-bound** — the chip's arithmetic units are the bottleneck.
- **Memory-bandwidth-bound** — moving data in and out of memory is the bottleneck, and the
  arithmetic units spend most of their time waiting.

Decode is memory-bound. That is precisely why batching helps so much, and it is the
mechanical reason a serving system's throughput can scale nearly linearly with batch size.

---

## 6. Static vs continuous batching

The old approach, **static batching**, collects a group of requests, runs them together, and
does not start anyone new until the *last* one in the group finishes. It is a tour bus that
will not stop until every passenger has reached their destination. If one person is going 20
stops and another 800, the short-trip seat sits empty — still paid for, still holding memory
— for 780 stops. That is called **head-of-line blocking**.

The new approach, **continuous batching** (also called *iteration-level scheduling*, from the
Orca paper, OSDI 2022), re-decides the batch between *every single word*: anyone finished?
Free their memory now. Anyone waiting? Let them in now. A city bus instead of a tour bus.

```
time ------------------------------------------------------------->

STATIC - the batch waits for its slowest member
  slot 1  [=====][...................][========][.............]
  slot 2  [=========================][==================.......]
  slot 3  [===][.....................][======][...............]
  slot 4  [============][............][===============][.......]
                                    ^ batch boundary

CONTINUOUS - a freed slot refills on the next step
  slot 1  [=====][========][============][====================]
  slot 2  [=========================][=======================]
  slot 3  [===][=====][==============][=======================]
  slot 4  [============][==============][====================]

  [===] generating words          [...] slot held, producing nothing
```

Every dotted stretch in the top panel is memory held hostage and compute burned on nothing.

### The link between the two halves of the project

**Continuous batching is only possible because of paging.** To let a new request in
mid-flight you need memory *right now, in whatever shape happens to be free*. With whole
notebooks you would need a continuous empty stretch of the right size, which almost never
exists once the system has been running. With blocks you just need *k* blocks off a free
list — any *k*.

**Paging is the enabler; continuous batching is the payoff.** If you remember one sentence
from this document, make it that one.

### When memory runs out anyway

Under pressure the scheduler must evict a running request. This project implements both
standard policies and keeps them selectable, because which one wins is hardware-dependent
and measuring the crossover is a genuinely interesting result:

- **Recompute** — throw the blocks away, put the request back in the queue, re-read its
  prompt later. Costs compute, no memory traffic. Wins on short prompts.
- **Swap** — copy the blocks out to ordinary CPU memory, free the GPU blocks, copy them back
  on resume. Costs two transfers across the bus, no recompute. Wins on long prompts.

Either way the evicted request must resume producing **identical** output, otherwise
eviction would be a behaviour change rather than an implementation detail. There are tests
that force eviction and check exactly this.

---

## 7. Prefix caching

Most requests to a real assistant begin with the same long block of text — the system
prompt, tool descriptions, few-shot examples. Reading that same preamble from scratch for
every user is pure waste, and reading the prompt is what you experience as the pause before
the first word appears.

So: remember the notes for that preamble once, and let every request that starts the same
way reuse them, skipping the reading entirely.

**The subtle part.** Two chunks of text that *look* identical are not necessarily
interchangeable, because a word's notes depend on everything that came before it. The words
"the cat" at position 32 of one prompt have different keys and values than the same words at
position 32 of a different prompt.

So the code fingerprints each block **chained** to the fingerprint of the block before it:

```
hash(block_i) = H( hash(block_{i-1}), tokens_of_block_i )
```

A match then means "identical all the way back to the very first word", which is the only
condition under which sharing notes is actually correct. Getting this wrong would not
crash — it would produce fluent, confident text conditioned on a conversation that never
happened, which is far worse than a crash.

Two more rules that make it safe:

- **Only completely full blocks are cached.** A partly-filled block is still being written
  into. Sharing something mutable would mean copying constantly. Sealing only full blocks
  makes every shared block immutable. The cost is up to 15 words of missed reuse at the tail.
- **Blocks linger after their last user leaves.** A finished request's system-prompt blocks
  are the *most* likely blocks to be wanted again a second later, so they go into a
  least-recently-used pool and are only reclaimed under real memory pressure. Freeing them
  immediately would throw away the best entries in the cache.

*Measured on a shared system prompt with staggered arrivals: 78.9% block hit rate, 240 words
of prompt-reading skipped (a CPU correctness run, not a benchmark).* Note that reuse
requires **staggered** arrivals — requests admitted in the same step cannot share, because
nothing has been through the model yet and there are no notes to reuse. A batch fired all at
once shows a 0% hit rate, and that is correct behaviour, not a bug.

---

## 8. What a "kernel" is, and the one that is not written yet

A **kernel** is simply a small program that runs directly on the graphics card. **CUDA** is
the language you write them in for NVIDIA cards.

The **attention** step is where the model looks back over all its notes and weighs which
ones matter for the word it is about to produce. Once the notes are scattered across
non-adjacent blocks, that step gets awkward — standard attention code expects one continuous
run of notes.

This project handles it in three planned stages:

1. **Gather** (built) — copy all of a request's scattered blocks into one neat scratch
   buffer, then call PyTorch's built-in attention. Obviously correct, and **deliberately
   slow**: it re-copies the entire working set on every single word. It stays in the repo
   forever as the reference that the fast version is checked against.
2. **FlashAttention with block tables** — a production-grade library fallback, if needed.
3. **A custom CUDA kernel** (**not started**) — teach the GPU to read the scattered blocks
   in place, no copying. This is the hardest part of the project.

What exists today for stage 3 is the build scaffolding: the toolchain compiles, a trivial
test kernel round-trips a tensor on a real GPU, and the Python side falls back gracefully to
the gather path on any machine without a CUDA toolkit. The attention kernel itself is
unwritten.

---

## 9. Where the project stands today

Nine planned phases. Seven have landed.

| Phase | What it added | Status |
|---|---|---|
| 0 | Measurement tools, built *before* any optimization | done |
| 1 | Hand-written model, contiguous cache, correctness gate | done |
| 2 | Paged memory allocator, gather read path | done |
| 3 | Continuous batching scheduler, both eviction policies | done |
| 4 | CUDA build scaffolding | scaffolding only — kernel not started |
| 5 | Prefix caching with chained hashes and LRU eviction | done |
| 6 | OpenAI-compatible HTTP server, streaming, full sampling | done |
| 7 | Chunked prefill, CUDA graphs, latency polish | not started |
| 8 | Ablations and honest comparison against vLLM | not started |
| 9 | Writeup, README, real numbers | not started |

**Correctness is genuinely established.** A "golden test" checks that this engine produces
*the exact same words* as HuggingFace's reference implementation — not similar, identical —
and it passes 44 of 44 cases on a real GPU (Tesla T4, float16), with paging on and off, and
with prefix caching on and off. The paged and contiguous paths produce bit-identical
intermediate values. Forced eviction is invisible in the output under both policies. There
are 423 test functions in total.

**Performance is entirely unmeasured.** There is no `results/` directory. Every phase
carries an unchecked box reading "needs a GPU". Two known open items are recorded honestly
in `AGENTS.md`: the memory-sizing routine skips its profiling pass and therefore
over-allocates, and one test about freeing memory after a client disconnects passes on CPU
and fails on a T4, with a diagnostic committed rather than a guess.

---

## 10. How this differs from what OpenAI or Anthropic run

Two honest answers, pointing in opposite directions.

**Conceptually, it does not differ much.** Paged attention and continuous batching are the
standard playbook for serving language models. vLLM, an open-source project out of Berkeley,
popularised them in 2023, and every serious serving stack now uses some version of these
ideas. This project is a careful reimplementation of a known approach, not a new invention.
That is deliberate.

**Practically, the gap is enormous.** Production systems run models orders of magnitude
larger, split across many GPUs at once (explicitly out of scope here — see below),
with weight compression, techniques that guess several words ahead, custom hardware, and
routing across datacenters. Each of those is a team's full-time work for years.

So the point is not to compete. The point is that the memory allocator, the scheduler, and
the kernel are written here rather than imported, which means every design decision can be
explained and defended. Being meaningfully slower than vLLM and saying so plainly is more
credible than any number one could claim otherwise.

### What is deliberately out of scope, and why

- **Multiple GPUs (tensor / pipeline parallelism)** — a different project with a different
  bottleneck: coordination between cards, not memory management.
- **Weight compression (quantization)** — orthogonal to the memory thesis.
- **Many model architectures** — supporting five architectures teaches configuration
  plumbing, not systems.
- **Speculative decoding** — a big win and a big complexity, and it interacts with the
  scheduler in ways that would consume the schedule.

"I scoped this to a single GPU because the interesting problem was memory fragmentation, not
coordination between cards" is a defensible sentence. "I did not get to it" is not.

---

## 11. Does this actually reduce latency?

This needs three separate answers, because the question hides a distinction.

**First: latency and throughput are not the same thing, and this project mostly improves the
second.**

- **Throughput** — total words per second across everyone. The capacity number.
- **Latency** — how long *your* request takes. Usually broken into **TTFT** (time to first
  token: how long before the first word appears) and **ITL** (inter-token latency: the gap
  between words once it is flowing).

Paging and continuous batching do not make the chip faster. They let far more requests fit
in memory and stop slots from sitting empty. That is throughput.

**Second: latency under load does improve, and that is the real claim.** When a request feels
slow on a busy server, it is usually not because the GPU is slow — it is because you are in a
queue. Fitting many more requests in memory and refilling slots instantly makes that queue
much shorter. The latency win comes from a shorter line, not a faster machine, and it shows
up in the tail (the P95 and P99 numbers) far more than in the average.

On a completely idle machine serving one user, expect roughly no improvement, and possibly
slightly worse, since paging adds bookkeeping that a single lonely request does not benefit
from. Prefix caching is the exception: it deletes work outright, so it cuts TTFT even on an
idle machine. The custom CUDA kernel, once written, would be the piece that reduces raw
per-word time and therefore ITL.

**Third, and most important: none of this has been measured yet.** There is no evidence in
this repository that any latency has been reduced, because no benchmark has been run. The
utilization percentages and the cache hit rate quoted above are correctness runs on a laptop
— bookkeeping ratios, not timings.

In fact the system today is quite possibly *slower* end-to-end than the naive version it
started from, because the gather read path re-copies the working set on every word and the
kernel that would replace it is not written. The roadmap predicted exactly that and
instructed that it be recorded honestly, because that regression is the entire argument for
building the kernel.

**Summary: the mechanisms that reduce latency under load are built and proven correct.
Whether they reduce it on real hardware, and by how much, is genuinely unknown. Finding out
is the remaining work.**

---

## 12. Glossary

| Term | Plain meaning |
|---|---|
| **Inference** | Using a trained model to produce answers |
| **GPU** | Graphics card; fast at thousands of identical operations at once, not at one operation |
| **CUDA** | NVIDIA's language for writing programs that run on the GPU |
| **Kernel** | A small program that runs directly on the GPU |
| **KV cache** | The model's notes: two summaries per word seen, so it need not re-read the conversation |
| **Key / Value** | The two summaries stored per word |
| **Token** | Roughly a word or word-piece; the unit the model actually reads and writes |
| **Prefill** | Reading the prompt. Done all at once. Compute-bound |
| **Decode** | Writing the answer, one word at a time. Memory-bandwidth-bound |
| **Compute-bound** | The arithmetic units are the bottleneck |
| **Memory-bandwidth-bound** | Moving data is the bottleneck; the arithmetic units wait |
| **Block** | A fixed-size chunk of the KV cache holding 16 words |
| **Block table** | A request's index from its own word positions to real block numbers |
| **Fragmentation** | Memory that is reserved or stranded and therefore unusable |
| **Batching** | Serving many requests in one pass so the expensive memory read is shared |
| **Static batching** | The batch runs to completion before anyone new is admitted |
| **Continuous batching** | The batch is re-decided between every word |
| **Head-of-line blocking** | One slow request holding up everyone behind it |
| **Preemption** | Evicting a running request when memory runs out |
| **Reference counting** | Tracking how many requests use a block, so it is freed only at zero |
| **Copy-on-write** | Giving a writer a private copy of a shared block before it modifies it |
| **Prefix caching** | Reusing the notes for a shared opening passage across requests |
| **TTFT** | Time to first token — the pause before the first word appears |
| **ITL** | Inter-token latency — the gap between words once generation is flowing |
| **Throughput** | Total words per second across all users |
| **Goodput** | Requests per second that actually met a stated latency target |
| **SSE** | Server-sent events; the HTTP mechanism used to stream words to the browser |

---

## 13. Where to look in the code

| Idea | File |
|---|---|
| Every configurable knob, device and dtype selection | `pagedserve/config.py` |
| One request's state and its lifecycle | `pagedserve/sequence.py` |
| Blocks and block tables (the OS analogy, in code) | `pagedserve/memory/block.py` |
| The allocator: free list, reference counts, invariants | `pagedserve/memory/block_manager.py` |
| Chained hashing and the LRU pool | `pagedserve/memory/prefix_cache.py` |
| Continuous batching, admission, eviction | `pagedserve/core/scheduler.py` |
| Recompute vs swap, and tail victim selection | `pagedserve/core/policy.py` |
| The naive contiguous cache (kept as the control) | `pagedserve/attention/contiguous.py` |
| The paged gather path (the correctness oracle) | `pagedserve/attention/gather.py` |
| The main loop: schedule, run, sample, retire | `pagedserve/engine.py` |
| HTTP server, streaming, disconnect handling | `pagedserve/server/api.py` |
| Load generator, latency percentiles | `bench/loadgen.py`, `bench/metrics.py` |
| The correctness gate | `tests/test_golden.py` |
| CUDA build canary | `csrc/trivial.cu` |

The module docstrings are unusually detailed on purpose — most of them explain *why* the
design is what it is, not just what the code does. `pagedserve/memory/prefix_cache.py` and
`pagedserve/core/policy.py` are the two best places to start reading.

---

## 14. Further reading, in the order it becomes useful

| Topic | Source |
|---|---|
| The memory problem and paged attention | vLLM paper, Kwon et al., SOSP 2023 |
| Iteration-level scheduling | Orca, OSDI 2022 |
| Virtual memory, pages, fragmentation | OSTEP, the paging chapters |
| Online softmax and memory-aware attention | FlashAttention 1 and 2 |
| The alternative prefix cache design not chosen here | SGLang's RadixAttention |
| Capping per-step work so long prompts do not spike latency | Sarathi-Serve |
