"""bench/sweep.py expands grids into loadgen command lines; it runs nothing in --dry-run."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import loadgen
from bench.sweep import (
    BLOCK_SIZES,
    CHUNK_BUDGETS,
    PREEMPTION_PROMPT_WORDS,
    SWEEPS,
    command,
    expand,
    main,
    starved_cache_flags,
)


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


def _parse(run, tmp_path, base=("--model", "m")):
    cmd = command(run, base=list(base), out_dir=tmp_path, repeat=0)
    return loadgen.build_parser().parse_args(cmd[3:])


def test_prefix_sweep_actually_injects_a_prefix(tmp_path: Path):
    """--shared-prefix-fraction alone does nothing: with no prefix length, every
    arm of the sweep would measure the same prefix-free workload."""
    for run in expand("prefix"):
        assert _parse(run, tmp_path).shared_prefix_words > 0


def test_preemption_sweep_varies_prompt_length_under_a_starved_cache(tmp_path: Path):
    runs = list(expand("preemption"))
    assert len(runs) == 2 * len(PREEMPTION_PROMPT_WORDS)
    for run in runs:
        # The sweep's own cache sizing must win over a --num-blocks in the base.
        args = _parse(run, tmp_path, base=("--model", "m", "--num-blocks", "8192"))
        per_seq = -(-(args.prompt_words + args.max_tokens) // 16) + 1
        assert args.num_blocks < per_seq * args.concurrency
        assert args.swap_space_blocks >= per_seq * args.concurrency
    words = {_parse(r, tmp_path).prompt_words for r in runs}
    assert words == set(PREEMPTION_PROMPT_WORDS)


def test_starved_cache_always_fits_one_sequence():
    for words in (1, 16, 2000):
        flags = starved_cache_flags(words, max_tokens=128, concurrency=1)
        per_seq = -(-(words + 128) // 16) + 1
        assert int(flags[1]) >= per_seq


def test_kernel_sweep_adds_one_variable_per_arm(tmp_path: Path):
    runs = list(expand("kernel", concurrency=(8,)))
    parsed = {r.arm: _parse(r, tmp_path) for r in runs}
    assert parsed["ps-gather"].attn_backend == "gather"
    assert parsed["ps-cuda"].attn_backend == "cuda" and not parsed["ps-cuda"].enable_cuda_graphs
    assert parsed["ps-cuda-graphs"].enable_cuda_graphs


def test_chunked_sweep_holds_the_workload_fixed(tmp_path: Path):
    runs = list(expand("chunked-prefill"))
    parsed = [_parse(r, tmp_path) for r in runs]
    assert sum(not a.enable_chunked_prefill for a in parsed) == 1
    budgets = sorted(a.max_num_batched_tokens for a in parsed if a.enable_chunked_prefill)
    assert budgets == sorted(CHUNK_BUDGETS)
    assert len({(a.long_prompt_words, a.long_prompt_fraction, a.concurrency) for a in parsed}) == 1
    assert all(a.long_prompt_words > 0 for a in parsed)


def test_http_sweep_targets_the_given_server(tmp_path: Path, capsys):
    url = "http://127.0.0.1:9000/v1"
    assert (
        main(
            [
                "http",
                "--model",
                "m",
                "--out",
                str(tmp_path),
                "--repeats",
                "1",
                "--dry-run",
                "--concurrency",
                "1",
                "4",
                "--base-url",
                url,
                "--http-arm",
                "vllm-http",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        args = loadgen.build_parser().parse_args(line.split()[3:])
        assert (args.backend, args.base_url, args.arm) == ("openai", url, "vllm-http")


def test_unknown_sweep_is_rejected():
    with pytest.raises(ValueError):
        list(expand("nope"))


def test_every_generated_command_parses_as_loadgen_flags(tmp_path: Path):
    """A sweep that emits a flag loadgen does not know would fail every run."""
    parser = loadgen.build_parser()
    for sweep in SWEEPS:
        for run in expand(sweep):
            cmd = command(run, base=["--model", "m"], out_dir=tmp_path, repeat=0)
            args = parser.parse_args(cmd[3:])
            assert args.arm == run.arm
            assert args.sweep == sweep
            assert args.backend == run.backend
            assert args.output.endswith("_r0.json")


def test_dry_run_prints_and_writes_nothing(tmp_path: Path, capsys):
    out = tmp_path / "res"
    assert (
        main(["block-size", "--model", "m", "--out", str(out), "--repeats", "1", "--dry-run"]) == 0
    )
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(BLOCK_SIZES)
    assert not out.exists()
