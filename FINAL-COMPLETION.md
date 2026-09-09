# PagedServe — Final Completion Plan

**Written:** 2026-09-07 · **Verified against:** `origin/main` @ `2752b48` (34 commits)

This is the single document that takes PagedServe from where it is today to finished.
It is written to be readable without prior context: every technical term is explained
the first time it appears, and every task says **why** it exists, not just what to do.

Read §0 and §1. Then work top to bottom.

---

## §0 — Welcome: read this before anything else

### What this project is, in one paragraph

PagedServe is a from-scratch **LLM inference server**. When you send a prompt to
ChatGPT, something on the other end has to hold thousands of conversations in GPU
memory at once and generate tokens for all of them efficiently. That "something" is an
inference server. The famous open-source ones are **vLLM**, **TensorRT-LLM**, and
**SGLang**. PagedServe rebuilds their four core tricks — paged KV cache, continuous
batching, prefix caching, and a custom CUDA attention kernel — from nothing, so you can
explain every one of them in an interview from memory.

### The one thing that matters most

> **This project's value is not the code. It is the measured numbers.**

The engine is already ~13,000 lines and substantially works. What it does **not** have
is a single benchmark result. The repository rule (`AGENTS.md` §2.1) is absolute:
**never write a performance number that did not come from a benchmark script in this
repo that was actually run, with its raw output committed to `results/`.**

Right now `results/` does not exist. That is the central gap this plan closes.

### Where the testing happens: **AWS**

Your MacBook cannot produce any number that goes in this project. Not because it is
slow — because it has **no NVIDIA GPU**, and every mechanism here is about NVIDIA GPU
memory. Code written on the Mac is *unverified* until a real GPU runs it.

**From this point forward, all GPU testing runs on an AWS EC2 server.** Concretely:

| Machine | What it is | What it does | What it must never do |
|---|---|---|---|
| **Your MacBook** | Apple Silicon, no NVIDIA GPU | All editing. All Conductor sessions. Unit tests (`pytest -m "not gpu"`). Charts. Docs. | Produce any reported number. |
| **AWS EC2 `g5.xlarge`** | One NVIDIA A10G, 24 GB, ~$1/hour | Compile the CUDA kernel. Run the GPU test suite. Run all benchmarks. All development-stage numbers. | Stay running when you are not using it. |
| **Northeastern Explorer** | A100 80 GB, pinned, free, SLURM queue | The final headline numbers for the README, if you want A100 numbers. | Interactive development (you wait in a queue). |

You have never used AWS. §5 is a complete zero-to-running walkthrough, including the
part that trips everyone up (a brand-new AWS account is **not allowed to launch a GPU
instance** until you ask permission — start that request today, it takes 1–3 days).

**Budget:** you are on the AWS Free plan with **$100 in credits that expire 2027-03-06**.
The whole remaining project is planned to fit inside roughly **$83**. §5.2 sets up a
billing alarm so you cannot silently blow past it.

**Simplification:** if you want to keep it simple, you can run *everything* on AWS and
skip Explorer entirely. The only cost is that your headline hardware becomes an A10G
instead of an A100. That is a completely defensible choice — just never blend results
from the two machines into one chart (§4, rule R3).

### How to read the rest of this document

- **§1** — exactly what is done and what is not, verified against GitHub today.
- **§2** — plain-English glossary. Skim it, come back when a word bites.
- **§3** — the master checklist of everything remaining.
- **§4** — the order of work and which pieces can run at the same time.
- **§5** — AWS from zero.
- **§6** — copy-paste prompts for every Conductor session.
- **§7** — what "done" means.

---

## §1 — Where the project actually stands

### 1.1 The headline

**The engine is built. The evidence is missing. The CUDA kernel is missing.**

- 68 files, 34 commits, **465 test functions** across 17 test files on `origin/main`.
- Phases 0, 1, 2, 3, 5, 6 are **functionally complete and CPU-verified**, plus a full
  correctness pass on a real NVIDIA T4 GPU (44/44 golden tests green).
- Phase 4 (the CUDA kernel) is **~10% done** — only the build scaffolding exists.
- Phases 7, 8, 9 are **not started**.
- `results/` **does not exist**. Zero measurements. Every phase has exactly one
  unchecked box and it always reads *"needs a GPU."*

### 1.2 Phase-by-phase status

| Phase | What it is | Code | Evidence | Verdict |
|---|---|---|---|---|
| **0** — Benchmark harness | Load generator, percentile metrics, HuggingFace baselines | ✅ Done | ❌ No baseline sweep | **Blocked on GPU** |
| **1** — Naive engine | Own Llama forward pass, contiguous KV cache | ✅ Done | ✅ Utilization measured (2.1%) | **Complete** |
| **2** — Paged KV cache | Fixed-size blocks + block tables, Python `gather` read path | ✅ Done | ⚠️ Utilization measured; throughput not | **Blocked on GPU** |
| **3** — Continuous batching | Iteration-level scheduler, admission, preemption (recompute + swap) | ✅ Done | ❌ No concurrency sweep | **Blocked on GPU** |
| **4** — CUDA paged attention | Hand-written GPU kernel that reads KV through block tables | ⚠️ **Scaffolding only** | ❌ Nothing | **~10% — the long pole** |
| **5** — Prefix caching | Shared, refcounted, LRU-evicted KV blocks for common prompt prefixes | ✅ Done | ⚠️ 78.9% hit rate on CPU; no TTFT number | **Blocked on GPU** |
| **6** — HTTP server | OpenAI-compatible FastAPI + SSE streaming, full sampling | ✅ Done | ❌ No sweep through the HTTP path | **Blocked on GPU** |
| **7** — Chunked prefill & polish | Split long prompts across steps; CUDA graphs; kill host-device syncs | ❌ Not started | — | **Not started** |
| **8** — Ablations | Prove *which* change caused *which* gain. Compare honestly to vLLM. | ❌ Not started | — | **Not started — never cut this** |
| **9** — Writeup | README, design doc, resume truth-up with real numbers | ❌ Not started | — | **Not started** |

### 1.3 What has been proven on a real GPU (Tesla T4, float16)

This is real and worth knowing — it means the correctness foundation is solid:

- Golden test **44/44 pass on GPU**, on both attention backends, in float16 *and*
  float32, with prefix caching **on and off**.
- Paged and contiguous attention produce **bit-identical logits** (max difference 0.000000).
- Forced preemption is invisible in the output under both policies.
- Every configuration is deterministic across repeated runs.
- CUDA extension scaffolding: 13/13 tests, including a fresh-subprocess loader test.

### 1.4 Five specific problems found in this audit

**P1 — `results/` does not exist.**
Five phases (0, 2, 3, 5, 6) each hang on one unchecked box, and it is the *same* box:
run a benchmark sweep on a GPU. **One successful AWS session closes all five.** This is
the single highest-leverage action available in the project.

**P2 — the `@pytest.mark.gpu` marker is declared but barely used.**
`pyproject.toml` declares a `gpu` marker and pytest runs with `--strict-markers`. But of
17 test files, **only 2** actually use it (`test_cache_engine.py`, `test_extension.py`).
The rest gate on `skipif` against whether a model happens to be downloadable.

*Why it matters:* on the AWS box you will type `pytest -m gpu` expecting to run the GPU
suite. You will get almost nothing, it will exit 0, and you will believe the GPU passed
when it never ran. **This will cost you a $1/hour session and give you a false green.**
Fix it before the first AWS launch.

**P3 — `WORKTREE_SUMMARY.md` claims things that do not exist.**
It is committed to the public repo and states that `csrc/paged_attention.cu`,
`csrc/cache_kernels.cu`, `pagedserve/attention/cuda_paged.py`, and
`pagedserve/worker/model_runner.py` exist. **None of them do.** The `csrc/` directory
contains only `bindings.cpp` and `trivial.cu` (a build canary).

*Why it matters:* this is a public repo you will link on a resume. A reviewer who opens
`csrc/` and finds no kernel after the summary promised one draws exactly the wrong
conclusion. This is the same failure mode as a fabricated number, and it is already
public. Fix it in Wave 1.

**P4 — `README.md` says "Phase 0 of 9 — No engine exists yet."**
Six phases out of date, on the front page of a public repo.

**P5 — `_private/` is gitignored, which breaks Conductor.**
Conductor gives each session its own fresh git worktree. `_private/DESIGN.md` and
`_private/ROADMAP.md` are gitignored, so they will **not** exist in a new worktree — and
`AGENTS.md` §1 instructs every agent to **stop and refuse to work** if it cannot see
them. Without a fix, *every Conductor session you start will immediately halt*. §6.0
gives you the setup script that fixes this.

---

## §2 — Plain-English glossary

Come back here whenever a term bites.

| Term | What it actually means |
|---|---|
| **Token** | A chunk of text, roughly ¾ of a word. Models read and write tokens, not letters. |
| **KV cache** | While generating, the model must remember what it computed for every earlier token. That memory is the KV ("key-value") cache. It is **the** thing that fills up your GPU. |
| **Paged KV cache** | Instead of reserving one huge contiguous slab per conversation ("this chat *might* reach 2048 tokens, so reserve 2048"), chop memory into small fixed blocks (16 tokens each) and hand them out as needed. Exactly how an operating system pages RAM. **This is the project's core idea.** It took KV memory utilization from 2.1% to 94.4%. |
| **Block table** | The per-conversation index that says "my token block 0 lives in physical block 41, block 1 in physical block 7…". The GPU kernel must follow this map, which is why a *custom* kernel is needed. |
| **Continuous batching** | Naive batching waits for all 8 requests in a batch to finish before starting the next 8 — so 7 users wait for the slowest. Continuous batching swaps a finished request out and a new one in *between every single token step*. |
| **Prefix caching** | If 50 users share the same 500-token system prompt, compute it once and let all 50 point at the same KV blocks. Saves the redundant work. |
| **Preemption** | The GPU ran out of KV memory mid-generation. You must evict a victim. **RECOMPUTE** = throw its KV away and redo it later (cheap memory, wasted compute). **SWAP** = copy its KV to CPU RAM and back (saves compute, costs bandwidth). Which wins depends on prompt length — finding that crossover point is a Phase 8 chart. |
| **Prefill vs decode** | **Prefill** = processing your whole prompt at once (fast per token, compute-bound). **Decode** = generating one token at a time (slow per token, *memory-bandwidth*-bound). Almost every optimization here targets decode. |
| **Chunked prefill** | One user pastes a 4000-token prompt. Without chunking, that one prefill hogs an entire step and every other user sees a visible stutter. Chunking splits it across several steps. |
| **CUDA kernel** | A function that runs on the GPU itself, written in CUDA C++. You need your own because no off-the-shelf attention kernel knows how to follow your block tables. |
| **CUDA graph** | Recording a fixed sequence of GPU operations once and replaying it, instead of the CPU re-issuing every instruction each step. On a small model the CPU's issuing overhead can be 30%+ of step time. |
| **Golden test** | `tests/test_golden.py`. It asserts your engine produces **token-for-token identical** output to HuggingFace on fixed prompts. It is the commit gate. **Never loosen it. A failing golden test is a bug report, not an obstacle.** |
| **Ablation** | Turning one feature off and re-measuring, to prove *that specific feature* caused the gain. "6.8× faster" with no ablation is unfalsifiable and reads that way. |
| **TTFT / ITL** | **Time To First Token** (how long until the reply starts) and **Inter-Token Latency** (how fast it streams after that). Users feel both, differently. |
| **Nsight** | NVIDIA's profiler. Shows you what the GPU is *actually* doing microsecond by microsecond, including when it is sitting idle. |
| **EC2 / instance** | Amazon's rented computers. An "instance" is one rented machine. `g5.xlarge` = the model with one A10G GPU. |
| **Spot instance** | A heavily discounted EC2 machine (often ~⅓ price) that AWS can reclaim with 2 minutes' notice. Great for kernel development where losing work costs you a `git push`. Bad for a long unattended benchmark. |
| **Conductor** | The tool you will use to run several Claude Code sessions in parallel, each in its own isolated git worktree, each on its own branch. |

---

## §3 — Master checklist: everything remaining

Grouped by where it runs. Nothing here is optional except where marked.

### 3.1 On your Mac, no GPU needed (do these first — they are cheap)

- [x] **M1 — Apply `@pytest.mark.gpu` to every test that needs a GPU** (fixes P2).
      Verify `pytest -m "not gpu"` still passes clean on the Mac, and that
      `pytest -m gpu` collects a real, non-trivial number of tests.
      *Why first: without it, your first $1/hour AWS session reports a false green.*
- [x] **M2 — Correct `WORKTREE_SUMMARY.md`** to describe what exists (fixes P3).
      *Why: it currently makes false claims on a public repo.*
- [x] **M3 — Update `README.md` status** from "Phase 0 of 9" to the real state (fixes P4).
- [x] **M4 — Dry-run the sweep runner at tiny scale on CPU.**
      `scripts/explorer_job.sbatch` has **never been executed**. Shell bugs in it are free
      to find on a Mac and expensive to find on a metered GPU.
- [x] **M5 — Write an AWS launch/teardown script** (`scripts/aws_bench.sh`) that boots the
      box, runs the sweep, pushes results, and **terminates itself on a timer**.
      *Why: the #1 way people get a surprise AWS bill is forgetting to shut down.*
- [x] **M6 — Set up the Conductor `_private/` bootstrap** (fixes P5). See §6.0.

> **Wave 1 is done** (2026-09-08, branch `wave-1/preflight`). Four commits:
> test-run reporting and the GPU false-green guard; the documentation truth-up;
> the sweep-script rehearsal and the `set -u` empty-array bug it found; the AWS
> automation and runbook. Two findings worth carrying forward:
>
> 1. **M1's premise was half wrong.** The `gpu` markers audit clean — the only
>    tests that genuinely cannot run without CUDA are the three
>    `profile_num_blocks` measurements and the seven build canaries, and both
>    were already marked. The false green is real but has a different cause:
>    `pytest -m gpu` selects 10 of 482 tests and **the golden gate is not one of
>    them**, because the gate is device-parametric rather than gpu-marked. The
>    right GPU command is
>    `PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest`.
>    §3.2 step A4 below should be read with that substitution.
> 2. **`explorer_job.sbatch` had a fatal bug.** The `continuous` ablation arm
>    passes no flags, and under `set -u` an empty bash array is an unbound
>    variable before bash 4.4 — the sweep died two thirds through, after the
>    contiguous and paged arms, so the prefix and poisson sections never ran and
>    the headline arm produced nothing. Found on a laptop for free.

### 3.2 On AWS — the first sweep (closes five phases at once)

- [ ] **A1 — AWS account, MFA, billing alarm, GPU quota request.** (§5. Quota takes 1–3 days —
      **do this today**, it gates everything.)
- [ ] **A2 — Launch `g5.xlarge`, record `nvidia-smi`, driver version, CUDA version, torch version.**
      *Why: the result JSON must name the hardware or the number is worthless.*
- [ ] **A3 — Rebuild the CUDA extension on `sm_86`.** It has only ever been compiled on
      `sm_75` (T4). Different GPU generation, different architecture code.
- [ ] **A4 — `pytest -m gpu` → the golden gate must pass before anything is measured.**
      *Why: measuring a broken engine produces confident, wrong numbers.*
- [ ] **A5 — `scripts/fetch_dataset.sh`** (ShareGPT, 673 MB — it is gitignored, it does not
      travel with the repo). Cache it on the EBS volume.
- [ ] **A6 — Smoke sweep** at tiny scale to confirm the pipeline writes valid JSON.
- [ ] **A7 — Full sweep**, `nohup`'d, teed to a log, with `--num-blocks` passed **explicitly**.
- [ ] **A8 — Commit `results/` and push.** These files are the evidence for every number
      the project will ever report.
- [ ] **A9 — Terminate the instance and confirm in the console that it is gone.**

**Closes: Phase 0, 2, 3, 5, 6.** ~$18, ~1 day.

### 3.3 Charts and truth-up (Mac, half a day)

- [ ] **C1 — Run `bench/plot.py` over the real results.** It already implements five charts:
      throughput-vs-concurrency, TTFT-vs-concurrency, ITL-vs-concurrency,
      TTFT-vs-prefix-fraction, and KV utilization.
- [ ] **C2 — Fill the README results section** from those charts. Every number traces to a
      committed file.
- [ ] **C3 — Update `AGENTS.md` §7** to tick the five newly closed boxes.

### 3.4 Phase 4 — the CUDA decode kernel (AWS, ~3 weeks, ~$40 — the long pole)

- [ ] **K1 — `csrc/paged_attention.cu` v1.** One thread block per `(sequence, head)`. Read
      the block table. Online softmax. **No optimization at all.** Correct and slow.
- [ ] **K2 — `csrc/cache_kernels.cu`** — KV scatter/copy/swap kernels.
- [ ] **K3 — `pagedserve/attention/cuda_paged.py`** — the Python wrapper; wire up
      `attn_backend="cuda"` in config.
- [ ] **K4 — `tests/test_attention_parity.py`** — diff the CUDA kernel against
      `attention/gather.py` on random inputs at `atol=1e-2`.
      *Why: `gather.py` is the correctness oracle. It exists forever for exactly this.*
- [ ] **K5 — Golden test must pass on the `cuda` backend**, same as on `gather`.
- [ ] **K6 — Optimize one change at a time, Nsight between each:**
      vectorized loads → shared-memory tiling → **split-K** → warp shuffles.
      *Why one at a time: if you change three things and it gets 2× faster, you have
      learned nothing and can defend nothing.*
- [ ] **K7 — Prefill: use FlashAttention varlen. Do not write your own.**
      *Why: prefill is compute-bound and already solved. Your contribution is decode.*

### 3.5 Phase 7 — polish (~1.5 weeks — cut this first if time runs short)

- [ ] **Q1 — Chunked prefill**, behind a flag, with a per-step token budget.
- [ ] **Q2 — CUDA graphs on the decode path**, with bucketed batch sizes.
- [ ] **Q3 — Eliminate host-device syncs** in the loop (`.item()`, `.cpu()`, `.tolist()`).
      *Why: this is the single most common performance bug in hand-rolled inference loops.
      Finding one in your own code with a profiler is a genuinely good interview story.*
- [ ] **Q4 — Async detokenization** on a separate thread.

### 3.6 Phase 8 — ablations (~1 week — **never cut this**)

- [ ] **B1 — Make `--no-paging` and `--static-batching` first-class engine/server CLI flags.**
      (Today they exist only in `bench/loadgen.py`.)
- [ ] **B2 — Block-size sweep: 1, 8, 16, 32, 64, 128.** Chart utilization, throughput, kernel time.
      *Why: it turns "block size 16" from a number you copied into a number you measured.*
- [ ] **B3 — Preemption policy comparison:** recompute vs swap across prompt lengths.
      Find and chart the crossover.
- [ ] **B4 — Prefix-cache sweep:** shared-prefix fraction 0% → 90%, chart TTFT.
- [ ] **B5 — vLLM comparison.** Same hardware, same model, same workload.
      *Why compare when you will lose? Because an interviewer will ask, and "I didn't
      compare" is worse than any gap. "I'm at ~40% of vLLM and the gap is kernel maturity
      and CUDA graph coverage — here's my profile" is exactly the judgment they screen for.
      Claiming to beat vLLM would get you disbelieved, correctly.*
- [ ] **B6 — Scale check:** rerun the headline benchmark once on a 7–8B model.
- [ ] **B7 — New chart functions in `bench/plot.py`** for B2, B3, B5.

### 3.7 Phase 9 — writeup (~1 week)

- [ ] **W1 — README:** what/why → headline chart → architecture diagram → design decisions
      with rationale → **benchmark methodology** → results with ablations → honest
      limitations → what's next.
- [ ] **W2 — `docs/design.md`** (a trimmed, public version of `_private/DESIGN.md`).
- [ ] **W3 — One script that regenerates every chart** from committed results.
- [ ] **W4 — Written vLLM gap analysis.**
- [ ] **W5 — Resume truth-up.** Your resume currently claims 3.4× sequences, 6.8×
      throughput, 61% P99 TTFT. **Those came from literature calibration, not measurement,
      and they are already out at NVIDIA and other companies.** Replace them with whatever
      you actually measured. A lower real number is a *better* bullet, because you can name
      the hardware, model, dataset and protocol without hesitating, and hold up under
      questioning. Someone who has built serving infrastructure finds an invented number
      in about two questions.
- [ ] **W6 — Write out your interview answers:** why paging; why block size 16; why decode
      is memory-bound; why continuous batching *requires* paging; where you are slower than
      vLLM and why; what you would do next.

---

## §4 — The order of work, and what can run in parallel

### 4.1 The dependency picture

```
   WAVE 1  (Mac, all four at once, ~1 day)
   ┌──────────┬──────────┬──────────┬──────────┐
   │ Lane A   │ Lane B   │ Lane C   │ Lane D   │
   │ gpu      │ docs     │ sweep    │ aws      │
   │ markers  │ truth-up │ dry-run  │ scripts  │
   └────┬─────┴────┬─────┴────┬─────┴────┬─────┘
        └──────────┴────┬─────┴──────────┘
                        ▼
   WAVE 2  (AWS, ONE session, ~1 day, ~$18)
        ┌──────────────────────────────────┐
        │  Lane E — THE FIRST SWEEP        │
        │  closes Phases 0, 2, 3, 5, 6     │
        └────────────────┬─────────────────┘
                         ▼
   WAVE 3  (three at once, ~3 weeks)
   ┌──────────────┬───────────────┬──────────────┐
   │ Lane F (Mac) │ Lane G (AWS)  │ Lane H (Mac) │
   │ charts +     │ PHASE 4       │ Phase 8 CLI  │
   │ README nums  │ CUDA KERNEL   │ flags (B1)   │
   └──────┬───────┴───────┬───────┴──────┬───────┘
          │               ▼              │
          │        WAVE 4 (~1.5 wks)     │
          │        ┌──────────────┐      │
          │        │ Lane I (AWS) │      │
          │        │ Phase 7      │      │
          │        │ polish       │      │
          │        └──────┬───────┘      │
          └───────────────┼──────────────┘
                          ▼
   WAVE 5  (two at once, ~1 week)
   ┌──────────────────┬────────────────────┐
   │ Lane J (AWS)     │ Lane K (Mac)       │
   │ Phase 8 ablation │ Phase 9 writeup    │
   │ runs             │ drafting           │
   └────────┬─────────┴──────────┬─────────┘
            └──────────┬─────────┘
                       ▼
   WAVE 6  Lane K finalize — README, resume, interview answers
```

### 4.2 The rules that make parallelism safe

**R1 — Two lanes must never both touch the attention path.**
Lane G (CUDA kernel) and Lane I (chunked prefill) both modify
`pagedserve/attention/` and `pagedserve/engine.py`, and the golden test gates both.
Run them **sequentially**, never together. This is the one hard "do not parallelize."

**R2 — One AWS lane at a time.**
Not a code constraint — a money constraint. Two `g5.xlarge` boxes is $2/hour and your
budget is $100.

**R3 — Never blend results from two machines into one chart.**
The result JSON already records `environment.gpu` and `environment.host`. A10G runs are
development signal; A100 runs are evidence. If you end up publishing A10G numbers, that
is fine — but then A10G becomes your pinned machine and the Explorer results get
*dropped*, not averaged in.

**R4 — Every lane's exit gate is the same two commands:**
```bash
ruff check . && ruff format --check .
pytest -m "not gpu"
```
Plus `pytest -m gpu` on any lane that touched GPU code.

**R5 — File ownership per wave.** Within a wave, two lanes must not own the same file.
The lane prompts in §6 state each lane's owned paths explicitly.

### 4.3 Timeline

| Wave | Content | Duration | GPU $ |
|---|---|---|---|
| 1 | Pre-flight fixes (Mac) | 1 day | $0 |
| 2 | First AWS sweep | 1 day | ~$18 |
| 3 | Charts + CUDA kernel + CLI flags | 3 weeks | ~$40 |
| 4 | Phase 7 polish | 1.5 weeks | ~$10 |
| 5 | Phase 8 ablations + writeup drafting | 1 week | ~$15 |
| 6 | Phase 9 finalize | 1 week | $0 |
| | **Total** | **~7 weeks** | **~$83 of $100** |

Tight. Use **spot instances** for Wave 3 kernel development and the total drops to
roughly **$55**. Losing a spot box costs you a `git push`, nothing more.

---

## §5 — AWS from zero

You have never used AWS. This section assumes exactly that.

### 5.1 The mental model in four sentences

AWS rents you computers by the hour. You pick a machine type (`g5.xlarge` = one A10G
GPU), start it, SSH in like any Linux box, do your work, and **destroy it**. You are
billed per second while it exists — including while it sits idle doing nothing. The
disk (called an **EBS volume**) can outlive the machine, so your 673 MB dataset does not
need re-downloading every time.

### 5.2 Do this today — the quota request gates everything

**A brand-new AWS account has a limit of 0 vCPUs for GPU instances.** You cannot launch
a `g5.xlarge` until you ask permission. Approval takes 1–3 days. Start it now.

1. Create an AWS account at `aws.amazon.com`. Stay on the **Free plan** — do not upgrade.
2. **Turn on MFA** on your root account. Console → your name (top right) → Security
   credentials → assign MFA device.
3. **Set a billing alarm before anything else.** Console → Billing and Cost Management →
   Budgets → Create budget → Cost budget → **$80/month** → alert at **50%, 80%, 100%**
   to your email. *Why: this is your seatbelt. Set it before you launch anything.*
4. **Request the GPU quota.** Console → Service Quotas → AWS services → **Amazon EC2** →
   search for **"Running On-Demand G and VT instances"** → Request increase → **8 vCPUs**.
   - `g5.xlarge` uses 4 vCPUs; asking for 8 leaves headroom.
   - In the justification box write plainly: *"Machine learning research — benchmarking a
     custom LLM inference engine on a single A10G GPU. Short, intermittent sessions."*
   - Pick your region and remember it. **`us-east-1` (N. Virginia)** is usually cheapest
     and has the most capacity. Whatever you pick, stay in it — quotas are per-region.
5. While you wait, do all of Wave 1 (§6, Lanes A–D). None of it needs a GPU.

### 5.3 Local setup

```bash
brew install awscli
aws configure
```

`aws configure` asks four things: an **Access Key ID**, a **Secret Access Key**, a
**default region** (`us-east-1`), and an output format (`json`).

To get the keys: Console → IAM → Users → Create user → attach the
`AmazonEC2FullAccess` policy → Security credentials → Create access key → "Command Line
Interface".

> **Do not create access keys on your root account, and never paste a key into a chat,
> a file in this repo, or a commit.** `aws configure` stores them in `~/.aws/credentials`,
> which is outside the repo. If a key ever leaks, deactivate it in IAM immediately.

### 5.4 Launching the box

Use the **Deep Learning AMI (Ubuntu)** — it ships with NVIDIA drivers, CUDA, and PyTorch
already installed. Building that stack yourself burns an hour of paid GPU time.

Three settings matter more than everything else:

| Setting | Value | Why |
|---|---|---|
| Instance type | `g5.xlarge` | One A10G, 24 GB. The cheapest current-generation NVIDIA GPU on EC2. |
| Storage | 60 GB `gp3` | Model weights + the 673 MB dataset + build artifacts. |
| Shutdown behavior | **`terminate`** | So that a `shutdown` inside the box actually destroys it and stops all billing, rather than just stopping it. |

And bake a dead-man's switch into the box as soon as you log in:

```bash
sudo shutdown -h +540
```

That guarantees the machine destroys itself in 9 hours no matter what happens to your
laptop, your Wi-Fi, or your attention. Reissue it with a new number if you need longer.

Approximate cost: **~$1/hour** for `g5.xlarge` on-demand in `us-east-1`. Confirm the
current rate on the EC2 pricing page for your region before you launch — prices change
and vary by region.

### 5.5 The security group

When the launch wizard asks about a security group, create one that allows **SSH (port
22) from My IP only**. Not `0.0.0.0/0`. An open SSH port on a GPU box gets found by
automated scanners within hours, and a compromised GPU instance mining crypto on your
credits is a real and common outcome.

### 5.6 The shutdown discipline

After every session:

```bash
aws ec2 terminate-instances --instance-ids i-XXXXXXXXXXXX
aws ec2 describe-instances --instance-ids i-XXXXXXXXXXXX \
    --query 'Reservations[].Instances[].State.Name'
```

The second command must print `terminated` (or `shutting-down`). **Check the console
with your own eyes too.** "I thought I shut it down" is the single most common way
people lose their credits.

### 5.7 Watching a long sweep from anywhere

The full sweep takes hours. You do not need to sit and watch it.

- `nohup` the sweep and `tee` it to a log on the box.
- From Claude Code on any device: `ssh <box> tail -50 sweep.log` and
  `ssh <box> 'ls results/*/ | wc -l'` against the expected file count.
- Or have the box `git push` its results branch every N runs, and watch the commit
  count on GitHub from your phone.

---

## §6 — Conductor session prompts

Copy these verbatim. Each is self-contained — a fresh Claude Code session with no
memory of this conversation can execute it.

### §6.0 — Setup you must do ONCE before any Conductor session

**The problem:** Conductor gives each session a fresh git worktree. `_private/` is
gitignored, so `DESIGN.md` and `ROADMAP.md` will not be there — and `AGENTS.md` §1 tells
every agent to **stop and refuse to work** if it cannot see them. Without this fix, every
session you start halts on its first message.

**The fix.** In Conductor's per-workspace setup command, put:

```bash
scripts/conductor_setup.sh
```

It resolves the canonical `_private/` from git rather than a hardcoded path (so it keeps
working if the repo moves), links it into the worktree, and then **verifies that
`DESIGN.md` and `ROADMAP.md` are actually readable through the link** — a broken symlink
and a working one look identical in `ls`, and the failure would otherwise show up only as
an agent refusing to start. It is a no-op in the main worktree.

If Conductor does not offer a setup hook, run that one line manually inside each new
worktree before sending its prompt.

### §6.1 — Session opener (paste at the top of EVERY lane prompt)

```
Read AGENTS.md completely before writing any code. It is the source of truth for
scope, conventions, and constraints, and it overrides anything you would otherwise
assume. Also read _private/DESIGN.md and _private/ROADMAP.md. If _private/ is not
present in this worktree, stop and tell me — do not guess at the design.

Three rules matter more than the rest:
1. NEVER write a performance number anywhere unless it came from a benchmark script
   in this repo that was actually executed, with raw output committed under results/.
   If a document needs a number that does not exist yet, write TODO(bench).
2. tests/test_golden.py must pass. Never loosen an assertion, widen a tolerance, or
   skip a case to make it pass. A failing golden test is a bug report.
3. Never delete a slower reference implementation. contiguous.py and gather.py are
   experimental controls and stay forever.

Before you finish, run:  ruff check . && ruff format --check .  and  pytest -m "not gpu"
Both must be clean. Then report: what you changed, what you tested, and — explicitly —
what you did NOT verify. Do not describe untested code as working.

Work only on the files this task names. Another session is working in this repo in
parallel on different files.
```

---

### WAVE 1 — four sessions, all at once, on the Mac

#### Lane A — GPU test markers
**Branch:** `fix/gpu-markers` · **Owns:** `tests/**`, `pyproject.toml`

```
[paste §6.1 opener first]

TASK: Make `pytest -m gpu` actually mean something.

pyproject.toml declares a `gpu` marker and pytest runs with --strict-markers, but of
17 test files in tests/, only test_cache_engine.py and test_extension.py apply it. The rest
gate on skipif against model availability instead.

Why this matters: on a metered AWS GPU box I will run `pytest -m gpu` expecting the GPU
suite. Today it collects almost nothing, exits 0, and I would read that as a pass when
the GPU never ran. That false green would cost real money and real trust.

Do this:
1. Audit all 17 test files. For each test, decide: does it require CUDA hardware, or
   does it require the compiled csrc/ extension, or neither?
2. Apply @pytest.mark.gpu to everything needing CUDA, and @pytest.mark.cuda_ext to
   everything needing the compiled extension. A test can carry both.
3. Keep the existing skipif guards. Markers are for selection; skipif is for graceful
   degradation. They are complementary, not alternatives.
4. Do not change any assertion or any test's logic. This is a labelling task only.

Verify and report the actual numbers:
  pytest -m "not gpu" --collect-only -q | tail -1     # must still be clean and pass
  pytest -m "gpu" --collect-only -q | tail -1         # must be a substantial count
  pytest -m "cuda_ext" --collect-only -q | tail -1

Report the three collected counts explicitly. If `-m gpu` collects fewer than ~30 tests,
say so and explain which files you judged not to need a GPU and why.
```

#### Lane B — Documentation truth-up
**Branch:** `docs/status-truth-up` · **Owns:** `README.md`, `WORKTREE_SUMMARY.md`, `AGENTS.md` §7

```
[paste §6.1 opener first]

TASK: Two public documents make claims that are not true. Fix them.

PROBLEM 1 — WORKTREE_SUMMARY.md claims files exist that do not.
It states that csrc/paged_attention.cu, csrc/cache_kernels.cu,
pagedserve/attention/cuda_paged.py, and pagedserve/worker/model_runner.py exist.
Verify this yourself with `ls csrc/ pagedserve/attention/ pagedserve/worker/`.
csrc/ contains only bindings.cpp and trivial.cu — a build canary, not a kernel.

Why this matters: this is a public repo I will link on a resume. A reviewer who opens
csrc/ expecting a kernel and finds a canary draws exactly the wrong conclusion. This is
the same failure mode as AGENTS.md section 2.1 forbids for numbers, and it is already public.

Rewrite the Phase 4 section to say precisely what is done (build scaffolding, verified
on a T4) and what is not (the decode kernel itself is not started). Audit every other
claim in that file against the actual tree the same way, and correct anything else that
overstates.

PROBLEM 2 — README.md line 8 says "Status: Phase 0 of 9 — building the measurement
harness. No engine exists yet." That is six phases out of date.

Rewrite the status line and the Results section to reflect the real state:
  - Phases 0,1,2,3,5,6 functionally complete, CPU-verified, plus correctness verified
    on a Tesla T4 (44/44 golden, both backends, fp16 and fp32, prefix cache on and off)
  - Phase 4 is scaffolding only
  - Phases 7,8,9 not started
  - results/ does not exist yet, so the Results section stays TODO(bench)

DO NOT invent, estimate, or illustrate any performance number. The only numbers you may
use are ones already recorded in AGENTS.md section 7, and only if you carry their
caveats with them (CPU/fp32, correctness run, not a benchmark).

Finally, update AGENTS.md section 7 so its "Current phase" line matches reality.
Touch no other section of AGENTS.md.
```

#### Lane C — Sweep dry-run
**Branch:** `fix/sweep-dryrun` · **Owns:** `scripts/explorer_job.sbatch`, `bench/loadgen.py` (CLI only)

```
[paste §6.1 opener first]

TASK: scripts/explorer_job.sbatch has never been executed. Find its bugs on my laptop,
where they are free, instead of on a $1/hour GPU box, where they are not.

1. Read it end to end. It is a SLURM batch script, but the body below the #SBATCH
   directives is ordinary bash and can be exercised locally.
2. Add a DRY_RUN=1 mode that echoes every command it would run instead of running it,
   and confirm the full matrix of SECTIONS x CONCURRENCIES x REPEATS expands correctly.
3. Then actually execute it at the smallest possible scale on CPU against a tiny
   random-init model or the mock backend:
     SECTIONS=baselines CONCURRENCIES="1 2" NUM_REQUESTS=4 REPEATS=1
   Confirm it writes syntactically valid result JSON to results/ containing all five
   required top-level keys (config, environment, workload, requests, summary) per
   AGENTS.md section 6.
4. Fix any bug you find: unbound variables, quoting, missing mkdir -p, wrong relative
   paths, set -u tripping on an unset env var.

Report exactly which commands you ran and what the resulting JSON looked like. If the
CPU dry run produces timing values, state clearly in your report that they are not
measurements and must never be committed as evidence.

Do NOT commit any result JSON produced by this dry run. Add it to .gitignore if needed,
or delete it. CPU timings are not evidence and must never enter results/.
```

#### Lane D — AWS automation
**Branch:** `scripts/aws-bench` · **Owns:** `scripts/aws_bench.sh`, `docs/aws-runbook.md`

```
[paste §6.1 opener first]

TASK: Write the AWS launch-and-teardown automation. I have never used AWS, so this must
be safe by construction, not by my remembering to do the right thing.

Create scripts/aws_bench.sh that:
1. Launches one g5.xlarge from a Deep Learning AMI (Ubuntu), 60 GB gp3 root volume,
   in the region from the AWS_REGION env var (default us-east-1).
2. Sets --instance-initiated-shutdown-behavior terminate, so a shutdown from inside the
   box actually destroys it and stops billing rather than merely stopping it.
3. Passes user-data that immediately runs `shutdown -h +540` as a dead-man's switch, so
   the machine destroys itself in 9 hours regardless of what happens to my laptop.
4. Waits for SSH, then over SSH: clones the repo at a branch given by --branch,
   pip install -e ".[engine,baseline,dev]", python setup.py build_ext --inplace,
   records nvidia-smi + driver + CUDA + torch versions to a file.
5. Runs `pytest -m gpu` and ABORTS THE WHOLE RUN if it fails. Never benchmark an engine
   that does not pass its correctness gate — the numbers would be confident and wrong.
6. Runs scripts/fetch_dataset.sh, then the sweep under nohup, teed to sweep.log.
7. git push of the results branch.
8. Terminates the instance, then polls describe-instances until the state reads
   terminated, and prints that state as the last line of output.

Also write a `--teardown-only <instance-id>` mode and a `--status` mode, so I can kill
or check a box from any device without remembering AWS CLI syntax.

Safety requirements, in order of importance:
- Never hardcode credentials, key material, account IDs, or an instance ID.
- Read the SSH key path and region from env vars with sane defaults.
- Print the estimated hourly cost and require an explicit typed confirmation before
  launching, unless --yes is passed.
- If ANY step fails, terminate the instance before exiting. A failed run must never
  leave a GPU billing.

You cannot test this against real AWS — I have not been granted GPU quota yet. So:
  - shellcheck it
  - add a --dry-run that prints every aws CLI command without executing it, and verify
    that path end to end
  - state clearly in your report that the live path is UNVERIFIED

Then write docs/aws-runbook.md: a plain-language walkthrough for someone who has never
used AWS, covering account setup, MFA, the billing alarm, the "Running On-Demand G and
VT instances" quota request, security-group rules (SSH from my IP only, never
0.0.0.0/0), and the shutdown checklist. Explain WHY each step exists, not just what to click.
```

---

### WAVE 2 — one session, on the AWS box

#### Lane E — The first sweep (closes five phases)
**Branch:** `results/first-sweep` · **Runs on:** AWS `g5.xlarge`

> Run this only after Lanes A–D are merged, and after your GPU quota is approved.

```
[paste §6.1 opener first]

CONTEXT: We are on an AWS g5.xlarge (NVIDIA A10G, 24 GB, sm_86) that costs about $1/hour
and self-terminates in 9 hours. This is the first time this project has ever measured
anything. results/ does not exist. Five phases (0, 2, 3, 5, 6) each have exactly one
unchecked box and it is the same box: run a sweep on a GPU. This session closes all five.

Work through these IN ORDER and STOP AT THE FIRST FAILURE. Each step is cheap; the next
one is not. Do not skip ahead to "save time" — a failure at step 4 makes every number
from step 7 worthless.

1. Record the environment: nvidia-smi, driver version, CUDA version, torch version,
   instance type. Write them into the run log. AGENTS.md section 6 requires the result
   JSON to name the hardware or the number means nothing.
2. pip install -e ".[engine,baseline,dev]"
3. python setup.py build_ext --inplace
   Build it EXPLICITLY, never via pip install — pip builds in an isolated environment
   with no torch, so the extension would silently fail to build.
   This extension has only ever compiled on sm_75 (a T4). This box is sm_86. If it fails
   to compile, STOP and report the full error. Do not work around it.
4. pytest -m gpu
   The golden gate must be 44/44. If it is not, STOP. Report the failure in full and do
   not measure anything. A failing golden test is a bug report, not an obstacle.
5. scripts/fetch_dataset.sh  (ShareGPT, ~673 MB, gitignored so it does not travel with
   the repo). Cache it on the EBS volume.
6. Smoke sweep first, tiny:
     SECTIONS=baselines CONCURRENCIES="1 8" NUM_REQUESTS=32 REPEATS=1
   Confirm it writes valid JSON with all required keys before spending an hour.
7. Full sweep. Pass --num-blocks EXPLICITLY on every run.
   Why: profile_num_blocks now supports a two-pass profile, but if no caller supplies
   run_max_shape_forward the estimate still counts activation memory as zero and comes
   out optimistic — on a T4 it handed 13.1 GB of a 14.6 GiB card to KV. On a 24 GB A10G
   the absolute overshoot is larger. Do not risk an OOM two hours into a sweep.
   Run it under nohup, teed to sweep.log, so a dropped SSH connection does not kill it.
   Follow AGENTS.md section 6 exactly: warm up and discard the first 30s, minimum three
   runs, report median and spread, P50/P95/P99 for every latency metric, Poisson arrivals
   for headline numbers.
8. Commit every result JSON to results/ and push the branch. These files are the evidence
   for every number this project will ever report.
9. Update AGENTS.md section 7: tick the now-closed boxes for Phases 0, 2, 3, 5, 6.

Then report: the environment block verbatim, the golden test result, how many result
files were written, and the headline throughput and latency numbers you measured.

DO NOT compute or state any speedup ratio in your report or in any file. Ratios are
computed later by bench/plot.py from committed files. Your job is to produce evidence,
not conclusions.

If anything looks implausibly good, say so. A suspiciously large speedup is far more
often a broken measurement than a real win, and reporting it would be worse than
reporting nothing.
```

---

### WAVE 3 — three sessions in parallel

#### Lane F — Charts and README numbers (Mac)
**Branch:** `docs/results-charts` · **Owns:** `bench/plot.py`, `README.md`, `results/` (read-only)

```
[paste §6.1 opener first]

TASK: results/ now contains real measurements from an AWS A10G. Turn them into the
README's evidence.

1. Run bench/plot.py over results/. It already implements five charts:
   throughput-vs-concurrency, TTFT-vs-concurrency, ITL-vs-concurrency,
   TTFT-vs-prefix-fraction, and KV utilization. Confirm each renders from real data
   rather than crashing on a field the synthetic test fixtures had but real runs do not.
2. Fix any parsing bug you find in _parse_result / _infer_arm against the real JSON shape.
3. Write the README Results section around the charts. Every single number must trace to
   a named file in results/. Include the full methodology block: GPU model, driver, CUDA,
   torch, model, dtype, dataset, arrival process, warmup discarded, number of runs,
   and the fact that the median is reported with its spread.
4. State the hardware honestly and prominently: these are A10G numbers from a
   development box, not A100 numbers.

Hard constraints:
- No chart is ever hand-made. If a chart is needed, it comes out of bench/plot.py.
- Never report a bare mean. P50/P95/P99 for every latency metric.
- If a number you want does not exist in results/, write TODO(bench). Do not estimate it,
  do not interpolate it, and do not carry it over from the literature.
```

#### Lane G — Phase 4: the CUDA decode kernel (AWS)
**Branch:** `phase-4/decode-kernel` · **Owns:** `csrc/**`, `pagedserve/attention/cuda_paged.py`, `tests/test_attention_parity.py`

> **Do not run this at the same time as Lane I.** Both touch the attention path and the
> golden test gates both.

```
[paste §6.1 opener first]

TASK: Phase 4 — write the custom CUDA paged attention decode kernel. This is the
centrepiece of the project and the only part that is genuinely hard.

Read _private/ROADMAP.md Phase 4 in full before writing a line.

WHY this kernel has to exist: every off-the-shelf attention kernel assumes the KV cache
for a sequence is contiguous in memory. Ours is not — it is scattered across fixed-size
blocks, and the only way to find them is to follow that sequence's block table. No
existing kernel knows how to do that. This indirection is the entire point of paging,
and it is why vLLM had to write its own kernel too.

Build in this order, and do not skip the slow first version:

1. csrc/paged_attention.cu — version 1. One thread block per (sequence, head). Read the
   block table. Online softmax. NO optimization whatsoever. Correct and slow is the goal.
2. csrc/cache_kernels.cu — KV scatter/copy/swap kernels.
3. Wire the pybind11 bindings in csrc/bindings.cpp.
4. pagedserve/attention/cuda_paged.py — implement the AttentionBackend ABC from
   attention/backend.py. Add attn_backend="cuda" to config.py.
5. tests/test_attention_parity.py — diff the CUDA kernel against attention/gather.py on
   random inputs at atol=1e-2. gather.py is the correctness oracle; it exists forever for
   exactly this purpose and must never be deleted.
6. Make tests/test_golden.py run against the cuda backend too. It must pass token-for-token,
   with prefix caching both on and off.

ONLY after all six pass, optimize — ONE change at a time, with Nsight Compute between
each, recording the measured effect of each individually:
  a. vectorized loads
  b. shared-memory tiling
  c. split-K
  d. warp shuffles

Why one at a time: if you change three things and it gets 2x faster, you have learned
nothing and can defend nothing in an interview. The per-change numbers ARE the deliverable.

Do NOT write a prefill kernel. Use FlashAttention varlen for prefill. Prefill is
compute-bound and already solved by people with more GPU-years than this project has;
the contribution here is the decode path.

This box costs about $1/hour. Commit and push often — if you are on a spot instance it
can be reclaimed with two minutes' notice, and a push is the only thing that survives.

Report after each optimization step: what changed, the measured effect, and the Nsight
evidence. Never state a speedup you did not measure.
```

#### Lane H — First-class ablation flags (Mac)
**Branch:** `phase-8/cli-flags` · **Owns:** `pagedserve/config.py` (flags only), `pagedserve/server/__main__.py`

```
[paste §6.1 opener first]

TASK: Phase 8 item 1 — make the ablation switches first-class CLI flags.

Today --no-paging and --static-batching exist only in bench/loadgen.py. They need to be
on the engine and server CLIs too, because Phase 8 is an ablation study and an ablation
you cannot toggle from the command line is an ablation you will not run.

Add, on both the engine entry point and pagedserve/server/__main__.py:
  --no-paging          -> attn_backend="contiguous"  (the Phase 1 path)
  --attn-backend       -> gather | contiguous | cuda
  --static-batching    -> disables the continuous scheduler
  --no-prefix-cache    -> disables prefix caching
  --num-blocks         -> explicit KV block count, bypassing the profiler

AGENTS.md section 5 requires config over constants: every one of these must map onto an
existing field in a config.py dataclass, not a new literal buried in an argument parser.
If a flag has no config field yet, add the field first.

Do NOT refactor bench/loadgen.py's existing flags. A benchmark sweep may be running
against them right now, and changing them mid-study would invalidate the comparison.

Add tests that each flag actually reaches the config object it claims to set.
```

---

### WAVE 4 — one session (must NOT overlap Lane G)

#### Lane I — Phase 7: chunked prefill and step-time polish (AWS)
**Branch:** `phase-7/chunked-prefill` · **Owns:** `pagedserve/core/scheduler.py`, `pagedserve/engine.py`

```
[paste §6.1 opener first]

TASK: Phase 7 — smooth out latency. Read _private/ROADMAP.md Phase 7 first.

WHY: right now one user pasting a 4000-token prompt monopolizes an entire decode step,
and every other user watching their reply stream sees a visible stutter — a
multi-hundred-millisecond spike in inter-token latency. Chunking that prefill across
several steps caps per-step work and bounds the ITL tail.

Build, each behind its own flag so it is independently ablatable per AGENTS.md section 2.3:

1. Chunked prefill: split a long prompt across steps with a per-step token budget. Mix
   prefill chunks with decode tokens in the same batch. Flag: --enable-chunked-prefill,
   with the budget configurable.
   The tradeoff is real and belongs in a comment: you trade a little TTFT for much better
   ITL tail latency, and which side you want depends on the SLO. That is exactly why it
   is a flag and not a default.
2. CUDA graphs on the decode path only. Decode step shapes are identical every iteration,
   so the launch sequence is static and capturable; prefill shapes vary, so graphs there
   would need constant recapture. Capture for bucketed batch sizes and pad up to the
   nearest bucket.
3. Hunt down every host-device sync in the decode loop: .item(), .cpu(), .tolist(),
   print(). Each one blocks the CPU until the GPU drains, and in a loop running hundreds
   of times a second a handful of them will halve throughput. Use Nsight Systems and
   report what you found, with before/after step times.
4. Async detokenization on a separate thread, so the engine loop never waits on Python
   string work.

The golden test gates all of this. Token-for-token identical output, on every backend,
with chunked prefill on and off. If chunking changes even one token, the batching is
wrong — stop and report rather than adjusting the test.

Report measured step-time breakdowns before and after each change, and ITL P99 under a
mixed workload with long prompts. Numbers only from actual runs on this box.
```

---

### WAVE 5 — two sessions in parallel

#### Lane J — Phase 8: the ablation study (AWS)
**Branch:** `phase-8/ablations` · **Owns:** `bench/sweep.py`, `bench/plot.py` (new charts), `results/`

```
[paste §6.1 opener first]

TASK: Phase 8 — turn a fast system into defensible evidence. Read _private/ROADMAP.md
Phase 8 first.

WHY this phase is never cut: "6.8x faster than HuggingFace" invites exactly one question —
"faster than what, and which change caused it?" A single end-to-end number with no
decomposition is unfalsifiable and reads as unfalsifiable. Ablations are the answer.

Run and chart, each writing raw JSON to results/:

1. Block-size sweep: 1, 8, 16, 32, 64, 128. Chart utilization, throughput, and kernel
   time against block size. Explain the shape of the curve in the README.
   Why: it turns "block size 16" from a number copied from vLLM into a number measured
   here, and the chart SHOWS the fragmentation-versus-locality tradeoff instead of
   asserting it.
2. Preemption policy: recompute vs swap across prompt lengths. Find and chart the
   crossover point. That crossover is a real result, not a footnote.
3. Prefix-cache sweep: shared-prefix fraction 0% to 90%, chart TTFT.
   Note: requests admitted in the SAME step cannot share a prefix — nothing has been
   through a forward pass yet, so there is no KV to reuse. A batch fired all at once
   shows a 0% hit rate and that is correct behaviour, not a bug. Stagger the arrivals.
4. vLLM comparison: same hardware, same model, same workload, same sampling params.
   Report the gap honestly and diagnose it — most likely kernel maturity, CUDA graph
   coverage, and vLLM's out-of-process engine design.
   Do not tune your own config and leave vLLM on defaults. That is not a comparison.
5. Scale check: rerun the headline benchmark once on a 7-8B model to confirm the effects
   hold at a size where the arithmetic changes.

Add the new chart functions to bench/plot.py for items 1, 2, and 4. No chart is ever
hand-made.

If PagedServe loses to vLLM, report the gap plainly with the profile that explains it.
That is a stronger result than a win you cannot account for, and claiming to beat vLLM
would get the whole project disbelieved — correctly.
```

#### Lane K — Phase 9: writeup and resume truth-up (Mac)
**Branch:** `phase-9/writeup` · **Owns:** `README.md`, `docs/**`

```
[paste §6.1 opener first]

TASK: Phase 9 — ship it. Read _private/ROADMAP.md Phase 9 first, especially "The resume
truth-up".

1. Rewrite README.md in this structure:
   one-paragraph what and why -> headline chart -> architecture diagram -> key design
   decisions with their rationale -> benchmark methodology (hardware, model, dataset,
   protocol, warmup, number of runs) -> results with the full ablation table ->
   honest limitations -> what's next (radix prefix cache, speculative decoding, tensor
   parallelism, FP8 KV).
2. Write docs/design.md: a trimmed, public version of _private/DESIGN.md. Keep the
   reasoning, drop anything personal.
3. One script that regenerates every chart in the README from committed results/. Verify
   it by deleting the chart files and regenerating them from scratch.
4. Write the vLLM gap analysis: the number, and the three or four mechanisms that
   explain it.
5. THE RESUME TRUTH-UP. My resume currently claims 3.4x sequences, 6.8x throughput, and
   61% P99 TTFT reduction. Those came from calibrating against published literature, NOT
   from measurement, and they are already out at NVIDIA and other companies. Replace them
   with what we actually measured, whatever it is.
   If the real numbers are lower, they are still better bullets: I can name the hardware,
   model, dataset and protocol without hesitating, and explain which architectural change
   produced which part of the gain. Someone who has built serving infrastructure finds an
   invented number within about two questions, and the same person is genuinely impressed
   by a modest number backed by an ablation table.
   Each bullet must carry its context: not "6.8x throughput" but "Nx output-token
   throughput at 32 concurrent requests vs HuggingFace static batching (same GPU, model,
   and sampling; ShareGPT; median of 3 warmed runs)". Compress for length, but never drop
   the comparison target or the concurrency — a throughput ratio without both is
   meaningless and a sharp reader knows it.
6. Write out my interview answers, in docs/interview-notes.md: why paging; why block size
   16; why decode is memory-bound; why continuous batching REQUIRES paging; where we are
   slower than vLLM and why; what I would do next and why.

Every number in every file must trace to a committed file in results/. If one does not,
it does not go in.
```

---

## §7 — Definition of done

The project is finished when all of these are true:

- [ ] `results/` contains committed raw JSON from a **named, pinned GPU**, and every
      number in the README traces to one of those files.
- [ ] `tests/test_golden.py` passes on **all three** attention backends
      (`contiguous`, `gather`, `cuda`), with prefix caching **on and off**.
- [ ] `bench/plot.py` regenerates **every** chart in the README from `results/`, with no
      hand-made images anywhere.
- [ ] The ablation table decomposes the total speedup into the contribution of paging,
      continuous batching, the CUDA kernel, and prefix caching — separately.
- [ ] A vLLM comparison exists, on the same hardware and workload, with the gap reported
      and diagnosed.
- [ ] `README.md` opens with a chart, states its methodology in full, and has a real
      limitations section.
- [ ] The resume bullets are rewritten from measured numbers, each carrying its
      comparison target and concurrency.
- [ ] You can talk through the whole design from memory without notes.
- [ ] **No AWS instance is running.** Check the console.

### The one rule that outranks all the others

> If a number in this project cannot be traced to a file in `results/`, delete the
> number. A modest measured result you can defend beats an impressive invented one
> every single time — and the invented one will be found.
