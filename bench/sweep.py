"""Phase 8 sweep driver: expand an ablation grid into ``bench.loadgen`` runs.

Every run writes one raw result JSON under ``--out``; ``bench/plot.py`` turns
that directory into charts. This script produces no numbers itself, and it
never fabricates a result for a run that failed — a failed run is logged and
leaves no file behind.

Sweeps (FINAL-COMPLETION.md §3.6):

- ``arms``       — contiguous / paged-static / paged-continuous across concurrency
- ``block-size`` — block sizes 1..128 on the continuous arm (B2)
- ``preemption`` — recompute vs swap under a starved cache (B3)
- ``prefix``     — shared-prefix fraction 0..0.9, cache off vs on (B4)

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
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SWEEPS = ("arms", "block-size", "preemption", "prefix")
CONCURRENCY = (1, 2, 4, 8, 16, 32)
BLOCK_SIZES = (1, 8, 16, 32, 64, 128)
PREFIX_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 0.9)


@dataclass(frozen=True)
class Run:
    """One loadgen invocation: a name for its result file and its extra flags."""

    name: str
    arm: str
    flags: tuple[str, ...] = field(default_factory=tuple)


def expand(sweep: str, *, concurrency: Sequence[int] = CONCURRENCY) -> Iterator[Run]:
    """The runs a sweep consists of, in execution order."""
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
        for policy in ("recompute", "swap"):
            for c in concurrency:
                yield Run(
                    f"{policy}_c{c}",
                    "ps-continuous",
                    ("--preemption-policy", policy, "--concurrency", str(c)),
                )
    elif sweep == "prefix":
        for fraction in PREFIX_FRACTIONS:
            for cache in (False, True):
                flags = ["--shared-prefix-fraction", str(fraction), "--concurrency", "8"]
                if cache:
                    flags.append("--enable-prefix-caching")
                tag = "on" if cache else "off"
                yield Run(f"prefix{int(fraction * 100)}_{tag}", "ps-continuous", tuple(flags))
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
        "pagedserve",
        *base,
        *run.flags,
        "--arm",
        run.arm,
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
    parser.add_argument("--num-blocks", type=int, default=None)
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=list(CONCURRENCY), help="Concurrency grid."
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra loadgen flags, shell-quoted (e.g. '--attn-backend cuda').",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    args = parser.parse_args(argv)

    base = ["--model", args.model, "--num-requests", str(args.num_requests)]
    base += ["--max-tokens", str(args.max_tokens), "--mode", "closed"]
    if args.dataset:
        base += ["--dataset", args.dataset]
    if args.num_blocks is not None:
        base += ["--num-blocks", str(args.num_blocks)]
    base += shlex.split(args.extra)

    failures = 0
    for run in expand(args.sweep, concurrency=args.concurrency):
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
