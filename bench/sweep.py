"""Phase 8 sweep driver: expand an ablation grid into ``bench.loadgen`` runs.

Every run writes one raw result JSON under ``--out``; ``bench/plot.py`` turns
that directory into charts. This script produces no numbers itself, and it
never fabricates a result for a run that failed — a failed run is logged and
leaves no file behind.

Sweeps:

- ``arms``            — contiguous / paged-static / paged-continuous across concurrency
- ``block-size``      — block sizes 1..128 on the continuous arm
- ``preemption``      — recompute vs swap across prompt lengths, under a starved cache
- ``prefix``          — shared-prefix fraction 0..0.9, cache off vs on
- ``kernel``          — gather vs the CUDA kernel vs the kernel plus CUDA graphs
- ``chunked-prefill`` — chunking off vs on at several token budgets, long prompts mixed in
- ``http``            — any OpenAI-compatible server over HTTP (PagedServe or vLLM)

Usage::

    python -m bench.sweep block-size --model Qwen/Qwen2.5-0.5B-Instruct \\
        --dataset data/sharegpt.json --out results/block_size --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

logger = logging.getLogger(__name__)

SWEEPS = ("arms", "block-size", "preemption", "prefix", "kernel", "chunked-prefill", "http")
CONCURRENCY = (1, 2, 4, 8, 16, 32)
BLOCK_SIZES = (1, 8, 16, 32, 64, 128)
PREFIX_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.9)
# Long enough to span many 16-token blocks, short enough that the prefix is a
# system prompt rather than the whole request. Matches explorer_job.sbatch.
PREFIX_WORDS = 400
# Recompute costs a prefill and swap costs two PCIe copies; both grow with the
# sequence, at different rates. The crossover is a function of prompt length,
# so prompt length is the axis. 1536 + max_tokens stays inside max_seq_len 2048.
PREEMPTION_PROMPT_WORDS = (128, 256, 512, 1024, 1536)
PREEMPTION_CONCURRENCY = 16
DEFAULT_BLOCK_SIZE = 16
# Per-step token budgets for chunked prefill. The off arm keeps the engine's
# default budget: it is the configuration being improved on, not a fourth knob.
CHUNK_BUDGETS = (256, 512, 1024)
LONG_PROMPT_WORDS = 1536
LONG_PROMPT_FRACTION = 0.125
CHUNKED_CONCURRENCY = 16


@dataclass(frozen=True)
class Run:
    """One loadgen invocation: a name for its result file and its extra flags."""

    name: str
    arm: str
    flags: tuple[str, ...] = field(default_factory=tuple)
    backend: str = "pagedserve"
    # Written to workload.sweep, so bench/plot.py can keep runs that hold some
    # other variable fixed out of the concurrency charts.
    sweep: str = ""


def starved_cache_flags(prompt_words: int, *, max_tokens: int, concurrency: int) -> list[str]:
    """KV and swap sizes that force preemption at this prompt length.

    The cache holds about half of the in-flight sequences at full length, so
    the scheduler has to evict to make progress -- a preemption comparison on a
    cache that never fills compares two policies that never ran. Swap space
    holds every sequence, so a swap run is never refused for lack of host room.

    Words stand in for tokens here (synthetic prompts are one repeated word,
    which tokenizes to one token per word), so the sizing is approximate. The
    preemption count each run records is the evidence of how starved it was.
    """
    per_seq = -(-(prompt_words + max_tokens) // DEFAULT_BLOCK_SIZE) + 1
    num_blocks = max(per_seq * concurrency // 2, per_seq * 2)
    return [
        "--num-blocks",
        str(num_blocks),
        "--swap-space-blocks",
        str(per_seq * concurrency),
    ]


def expand(
    sweep: str,
    *,
    concurrency: Sequence[int] = CONCURRENCY,
    max_tokens: int = 128,
    http_arm: str = "ps-http",
) -> Iterator[Run]:
    """The runs a sweep consists of, in execution order."""
    for run in _expand(sweep, concurrency=concurrency, max_tokens=max_tokens, http_arm=http_arm):
        yield replace(run, sweep=sweep)


def _expand(
    sweep: str, *, concurrency: Sequence[int], max_tokens: int, http_arm: str
) -> Iterator[Run]:
    if sweep == "arms":
        for c in concurrency:
            yield Run(f"contiguous_c{c}", "ps-contiguous", ("--no-paging", "--concurrency", str(c)))
            yield Run(
                f"paged_static_c{c}", "ps-paged", ("--static-batching", "--concurrency", str(c))
            )
            yield Run(f"continuous_c{c}", "ps-continuous", ("--concurrency", str(c)))
    elif sweep == "block-size":
        for b in BLOCK_SIZES:
            yield Run(f"block{b}", "ps-continuous", ("--block-size", str(b), "--concurrency", "16"))
    elif sweep == "preemption":
        for words in PREEMPTION_PROMPT_WORDS:
            cache = starved_cache_flags(
                words, max_tokens=max_tokens, concurrency=PREEMPTION_CONCURRENCY
            )
            for policy in ("recompute", "swap"):
                yield Run(
                    f"{policy}_p{words}",
                    "ps-continuous",
                    (
                        "--preemption-policy",
                        policy,
                        "--prompt-words",
                        str(words),
                        "--concurrency",
                        str(PREEMPTION_CONCURRENCY),
                        *cache,
                    ),
                )
    elif sweep == "prefix":
        for fraction in PREFIX_FRACTIONS:
            for cache in (False, True):
                flags = [
                    "--shared-prefix-words",
                    str(PREFIX_WORDS),
                    "--shared-prefix-fraction",
                    str(fraction),
                    "--concurrency",
                    "8",
                ]
                if cache:
                    flags.append("--enable-prefix-caching")
                tag = "on" if cache else "off"
                yield Run(f"prefix{int(fraction * 100)}_{tag}", "ps-continuous", tuple(flags))
    elif sweep == "kernel":
        # Each arm adds one thing to the one before it: the kernel replaces the
        # gather copy, then graphs remove the launch overhead around it.
        for c in concurrency:
            yield Run(
                f"gather_c{c}", "ps-gather", ("--attn-backend", "gather", "--concurrency", str(c))
            )
            yield Run(f"cuda_c{c}", "ps-cuda", ("--attn-backend", "cuda", "--concurrency", str(c)))
            yield Run(
                f"cuda_graphs_c{c}",
                "ps-cuda-graphs",
                ("--attn-backend", "cuda", "--enable-cuda-graphs", "--concurrency", str(c)),
            )
    elif sweep == "chunked-prefill":
        mix = (
            "--long-prompt-words",
            str(LONG_PROMPT_WORDS),
            "--long-prompt-fraction",
            str(LONG_PROMPT_FRACTION),
            "--concurrency",
            str(CHUNKED_CONCURRENCY),
        )
        yield Run("chunked_off", "ps-chunked-off", mix)
        for budget in CHUNK_BUDGETS:
            yield Run(
                f"chunked_b{budget}",
                "ps-chunked-on",
                (*mix, "--enable-chunked-prefill", "--max-num-batched-tokens", str(budget)),
            )
    elif sweep == "http":
        # The server is started outside this script; see docs/vllm-gap.md.
        for c in concurrency:
            yield Run(f"http_c{c}", http_arm, ("--concurrency", str(c)), backend="openai")
    else:
        raise ValueError(f"unknown sweep {sweep!r}; expected one of {SWEEPS}")


def command(run: Run, *, base: Sequence[str], out_dir: Path, repeat: int) -> list[str]:
    """The full loadgen command line for one run and repeat."""
    output = out_dir / f"{run.name}_r{repeat}.json"
    return [
        sys.executable,
        "-m",
        "bench.loadgen",
        "--backend",
        run.backend,
        *base,
        *run.flags,
        "--arm",
        run.arm,
        *(("--sweep", run.sweep) if run.sweep else ()),
        "--seed",
        str(repeat),
        "--output",
        str(output),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sweep", choices=SWEEPS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--out", required=True, type=Path, help="Directory for result JSONs.")
    parser.add_argument("--repeats", type=int, default=3, help="Seeds per point (median plotted).")
    parser.add_argument("--num-requests", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=32,
        help="Requests dropped from each run's summary (AGENTS.md section 6, rule 3).",
    )
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=list(CONCURRENCY), help="Concurrency grid."
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra loadgen flags, shell-quoted (e.g. '--attn-backend cuda').",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000/v1",
        help="http sweep: the server under test.",
    )
    parser.add_argument(
        "--http-arm",
        default="ps-http",
        help="http sweep: arm label for the server under test (ps-http or vllm-http).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = parser.parse_args(argv)

    base = ["--model", args.model, "--num-requests", str(args.num_requests)]
    base += ["--max-tokens", str(args.max_tokens), "--mode", "closed"]
    base += ["--warmup-requests", str(args.warmup_requests)]
    if args.dataset:
        base += ["--dataset", args.dataset]
    if args.num_blocks is not None:
        base += ["--num-blocks", str(args.num_blocks)]
    if args.sweep == "http":
        base += ["--base-url", args.base_url]
    base += shlex.split(args.extra)

    failures = 0
    runs = expand(
        args.sweep,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        http_arm=args.http_arm,
    )
    for run in runs:
        for repeat in range(args.repeats):
            cmd = command(run, base=base, out_dir=args.out, repeat=repeat)
            if args.dry_run:
                print(shlex.join(cmd))
                continue
            args.out.mkdir(parents=True, exist_ok=True)
            logger.info("running %s (repeat %d)", run.name, repeat)
            if subprocess.run(cmd, check=False).returncode != 0:
                failures += 1
                logger.error("run %s repeat %d failed; no result recorded", run.name, repeat)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
