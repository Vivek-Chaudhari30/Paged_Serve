"""bench/sweep.py expands grids into loadgen command lines; it runs nothing in --dry-run."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import loadgen
from bench.sweep import BLOCK_SIZES, command, expand, main


def test_block_size_sweep_covers_the_grid():
    runs = list(expand("block-size"))
    assert [r.flags[1] for r in runs] == [str(b) for b in BLOCK_SIZES]
    assert {r.arm for r in runs} == {"ps-continuous"}


def test_arms_sweep_has_three_arms_per_concurrency():
    runs = list(expand("arms", concurrency=(1, 4)))
    assert len(runs) == 6
    assert {r.arm for r in runs} == {"ps-contiguous", "ps-paged", "ps-continuous"}


def test_prefix_sweep_pairs_cache_off_and_on():
    runs = list(expand("prefix"))
    on = [r for r in runs if "--enable-prefix-caching" in r.flags]
    assert len(on) * 2 == len(runs)


def test_unknown_sweep_is_rejected():
    with pytest.raises(ValueError):
        list(expand("nope"))


def test_every_generated_command_parses_as_loadgen_flags(tmp_path: Path):
    """A sweep that emits a flag loadgen does not know would fail every run."""
    parser = loadgen.build_parser()
    for sweep in ("arms", "block-size", "preemption", "prefix"):
        for run in expand(sweep):
            cmd = command(run, base=["--model", "m"], out_dir=tmp_path, repeat=0)
            args = parser.parse_args(cmd[3:])
            assert args.arm == run.arm
            assert args.output.endswith("_r0.json")


def test_dry_run_prints_and_writes_nothing(tmp_path: Path, capsys):
    out = tmp_path / "res"
    assert (
        main(["block-size", "--model", "m", "--out", str(out), "--repeats", "1", "--dry-run"]) == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(BLOCK_SIZES)
    assert not out.exists()
