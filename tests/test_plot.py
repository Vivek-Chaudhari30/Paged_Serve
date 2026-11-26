"""Tests for bench.plot — chart generation against synthetic result files.

The synthetic result dicts use the same JSON schema as real committed result
files (AGENTS.md §6), so these tests exercise the exact parsing and plotting
logic that will run against real GPU data.

No GPU is required.  matplotlib is forced to the Agg (non-interactive) backend
via conftest.py before any import of bench.plot, so the tests run cleanly in a
headless CI environment and on macOS with no display server.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.plot import (
    _hw_safe,
    _infer_arm,
    _median_band,
    _parse_result,
    run,
)

# ---------------------------------------------------------------------------
# Constants used across fixtures and helpers
# ---------------------------------------------------------------------------

_GPU = "NVIDIA A100-SXM4-80GB"
_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
_DATASET = "sharegpt"


# ---------------------------------------------------------------------------
# Synthetic data builders
# ---------------------------------------------------------------------------


def _make_summary(
    *,
    output_throughput: float = 1000.0,
    ttft_p50: float = 0.05,
    ttft_p95: float = 0.12,
    ttft_p99: float = 0.20,
    itl_p50: float = 0.010,
    itl_p95: float = 0.020,
    itl_p99: float = 0.035,
    kv_utilization: float | None = None,
) -> dict[str, object]:
    d: dict[str, object] = {
        "units": "seconds",
        "num_requests": 64,
        "num_succeeded": 64,
        "num_failed": 0,
        "total_prompt_tokens": 4096,
        "total_output_tokens": 8192,
        "duration": 8.192,
        "output_throughput": output_throughput,
        "prompt_throughput": 500.0,
        "request_throughput": 7.8,
        "ttft_p50": ttft_p50,
        "ttft_p95": ttft_p95,
        "ttft_p99": ttft_p99,
        "ttft_mean": ttft_p50,
        "itl_p50": itl_p50,
        "itl_p95": itl_p95,
        "itl_p99": itl_p99,
        "itl_mean": itl_p50,
        "slo": None,
        "num_good": None,
        "goodput": None,
    }
    if kv_utilization is not None:
        d["kv_utilization"] = kv_utilization
    return d


def _make_result(
    *,
    arm: str,
    concurrency: int = 8,
    gpu: str = _GPU,
    model: str = _MODEL,
    dataset: str = _DATASET,
    max_tokens: int = 128,
    prefix_fraction: float | None = None,
    prefix_cache: dict[str, object] | None = None,
    kv_utilization: float | None = None,
    output_throughput: float = 1000.0,
    ttft_p50: float = 0.05,
    ttft_p95: float = 0.12,
    ttft_p99: float = 0.20,
    itl_p50: float = 0.010,
    itl_p95: float = 0.020,
    itl_p99: float = 0.035,
) -> dict[str, object]:
    """Build a minimal result dict that matches AGENTS.md §6 schema.

    Uses ``workload.arm`` to set the arm label explicitly so that tests are
    not sensitive to the arm-inference heuristics.
    """
    workload: dict[str, object] = {
        "arm": arm,
        "dataset": dataset,
        "arrival": "closed_loop",
        "rate": None,
        "concurrency": concurrency,
        "num_requests": 64,
        "max_tokens": max_tokens,
        "seed": 0,
        "warmup_requests": 0,
        "warmup_seconds": 0.0,
        "num_requests_measured": 64,
    }
    if prefix_fraction is not None:
        workload["prefix_fraction"] = prefix_fraction
    if prefix_cache is not None:
        workload["prefix_cache"] = prefix_cache
    if kv_utilization is not None:
        workload["kv_utilization"] = kv_utilization

    return {
        "config": {
            "backend": "pagedserve",
            "mode": "closed",
            "backend_config": None,
            "engine": {"model": model, "attn_backend": "gather"},
        },
        "environment": {
            "gpu": gpu,
            "driver": None,
            "cuda": "12.1",
            "torch": "2.2.0",
            "host": "explorer",
            "timestamp": "2024-01-01T00:00:00+00:00",
        },
        "workload": workload,
        "requests": [],
        "summary": _make_summary(
            output_throughput=output_throughput,
            ttft_p50=ttft_p50,
            ttft_p95=ttft_p95,
            ttft_p99=ttft_p99,
            itl_p50=itl_p50,
            itl_p95=itl_p95,
            itl_p99=itl_p99,
            kv_utilization=kv_utilization,
        ),
    }


def _write(rdir: Path, name: str, data: dict[str, object]) -> Path:
    p = rdir / name
    p.write_text(json.dumps(data))
    return p


def _chart_dir(out: Path) -> Path:
    """Return the per-hardware subdirectory that run() creates for the test GPU."""
    return out / _hw_safe(_GPU, _MODEL, _DATASET)


# ---------------------------------------------------------------------------
# Unit tests: _infer_arm
# ---------------------------------------------------------------------------


def test_infer_arm_explicit_field_wins() -> None:
    result = {"workload": {"arm": "hf-sequential"}, "config": {"backend": "pagedserve"}}
    assert _infer_arm(result) == "hf-sequential"


def test_infer_arm_hf_sequential() -> None:
    result = {
        "config": {"backend": "hf", "backend_config": {"max_batch_size": 1}},
        "workload": {},
    }
    assert _infer_arm(result) == "hf-sequential"


def test_infer_arm_hf_static() -> None:
    result = {
        "config": {"backend": "hf", "backend_config": {"max_batch_size": 16}},
        "workload": {},
    }
    assert _infer_arm(result) == "hf-static"


def test_infer_arm_ps_contiguous() -> None:
    result = {
        "config": {
            "backend": "pagedserve",
            "engine": {"attn_backend": "contiguous", "model": "m"},
        },
        "workload": {},
    }
    assert _infer_arm(result) == "ps-contiguous"


def test_infer_arm_ps_continuous_no_batching_signal() -> None:
    # No explicit batching info → default to continuous.
    result = {
        "config": {
            "backend": "pagedserve",
            "engine": {"attn_backend": "gather", "model": "m"},
        },
        "workload": {},
    }
    assert _infer_arm(result) == "ps-continuous"


def test_infer_arm_ps_paged_via_steps_equals_batches() -> None:
    # steps ≈ num_batches signals static batching (one step per batch).
    result = {
        "config": {
            "backend": "pagedserve",
            "engine": {"attn_backend": "gather", "model": "m"},
        },
        "workload": {
            "scheduler": {"steps": 4},
            "batching": {"num_batches": 4},
        },
    }
    assert _infer_arm(result) == "ps-paged"


def test_infer_arm_unknown_backend_returns_none() -> None:
    result = {"config": {"backend": "frobnosticator"}, "workload": {}}
    assert _infer_arm(result) is None


# ---------------------------------------------------------------------------
# Unit tests: _median_band
# ---------------------------------------------------------------------------


def test_median_band_single_repeat() -> None:
    xs, meds, lo, hi = _median_band({1.0: [5.0], 2.0: [10.0]})
    assert xs == [1.0, 2.0]
    assert meds == pytest.approx([5.0, 10.0])
    assert lo == pytest.approx(meds)
    assert hi == pytest.approx(meds)


def test_median_band_multiple_repeats() -> None:
    xs, meds, lo, hi = _median_band({4.0: [2.0, 4.0, 6.0]})
    assert xs == [4.0]
    assert meds == pytest.approx([4.0])
    assert lo == pytest.approx([2.0])
    assert hi == pytest.approx([6.0])


def test_median_band_skips_empty_x() -> None:
    xs, meds, lo, hi = _median_band({1.0: [3.0], 2.0: [], 3.0: [9.0]})
    assert xs == [1.0, 3.0]
    assert len(meds) == 2


# ---------------------------------------------------------------------------
# Unit tests: _parse_result
# ---------------------------------------------------------------------------


def test_parse_result_valid(tmp_path: Path) -> None:
    p = _write(tmp_path, "run.json", _make_result(arm="ps-continuous", concurrency=16))
    rec = _parse_result(p)
    assert rec is not None
    assert rec.arm == "ps-continuous"
    assert rec.concurrency == 16
    assert rec.gpu == _GPU
    assert rec.model == _MODEL
    assert rec.ttft_p50 == pytest.approx(0.05)


def test_parse_result_missing_gpu_returns_none(tmp_path: Path) -> None:
    data = _make_result(arm="ps-continuous")
    data["environment"]["gpu"] = None  # type: ignore[index]
    p = _write(tmp_path, "bad.json", data)
    assert _parse_result(p) is None


def test_parse_result_missing_model_returns_none(tmp_path: Path) -> None:
    data = _make_result(arm="ps-continuous")
    data["config"]["engine"]["model"] = None  # type: ignore[index]
    p = _write(tmp_path, "bad.json", data)
    assert _parse_result(p) is None


def test_parse_result_bad_json_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "corrupt.json"
    p.write_text("{not valid json")
    assert _parse_result(p) is None


def test_parse_result_reads_kv_utilization_from_summary(tmp_path: Path) -> None:
    p = _write(tmp_path, "run.json", _make_result(arm="ps-contiguous", kv_utilization=0.68))
    rec = _parse_result(p)
    assert rec is not None
    assert rec.kv_utilization == pytest.approx(0.68)


def test_parse_result_reads_kv_utilization_from_workload(tmp_path: Path) -> None:
    data = _make_result(arm="ps-contiguous")
    data["workload"]["kv_utilization"] = 0.42  # type: ignore[index]
    p = _write(tmp_path, "run.json", data)
    rec = _parse_result(p)
    assert rec is not None
    assert rec.kv_utilization == pytest.approx(0.42)


def test_parse_result_prefix_cache_fields(tmp_path: Path) -> None:
    pc = {"hit_rate": 0.75, "tokens_saved": 300, "evictions": 0}
    p = _write(
        tmp_path,
        "run.json",
        _make_result(arm="ps-continuous", prefix_fraction=0.8, prefix_cache=pc),
    )
    rec = _parse_result(p)
    assert rec is not None
    assert rec.prefix_cache_on is True
    assert rec.prefix_hit_rate == pytest.approx(0.75)
    assert rec.prefix_fraction == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Integration fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_sweep(tmp_path: Path) -> Path:
    """Results directory with three repeats per arm × concurrency level."""
    rdir = tmp_path / "results"
    rdir.mkdir()
    arms = ["hf-sequential", "hf-static", "ps-contiguous", "ps-paged", "ps-continuous"]
    concurrencies = [4, 8, 16]
    for arm in arms:
        for conc in concurrencies:
            for repeat in range(3):
                _write(
                    rdir,
                    f"{arm}_c{conc}_r{repeat}.json",
                    _make_result(
                        arm=arm,
                        concurrency=conc,
                        output_throughput=500.0 * conc + repeat * 10,
                        ttft_p50=0.10 / conc + repeat * 0.001,
                        ttft_p95=0.20 / conc,
                        ttft_p99=0.30 / conc,
                        itl_p50=0.010 + repeat * 0.001,
                        itl_p95=0.020,
                        itl_p99=0.035,
                    ),
                )
    return rdir


# ---------------------------------------------------------------------------
# Integration tests: chart production
# ---------------------------------------------------------------------------


def test_throughput_chart_produced(full_sweep: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    run(full_sweep, out)
    assert (_chart_dir(out) / "throughput_vs_concurrency.png").exists()


def test_ttft_chart_produced(full_sweep: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    run(full_sweep, out)
    assert (_chart_dir(out) / "ttft_vs_concurrency.png").exists()


def test_itl_chart_produced(full_sweep: Path, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    run(full_sweep, out)
    assert (_chart_dir(out) / "itl_vs_concurrency.png").exists()


def test_prefix_fraction_chart_produced(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    fractions = [0.0, 0.25, 0.50, 0.75]
    for frac in fractions:
        pc = {"hit_rate": frac, "tokens_saved": int(frac * 100)} if frac > 0 else None
        _write(
            rdir,
            f"prefix_f{frac}_on{pc is not None}.json",
            _make_result(
                arm="ps-continuous",
                concurrency=8,
                prefix_fraction=frac,
                prefix_cache=pc,
                ttft_p50=0.20 - frac * 0.08,
            ),
        )
    out = tmp_path / "figures"
    run(rdir, out)
    assert (_chart_dir(out) / "ttft_vs_prefix_fraction.png").exists()


def test_kv_utilization_chart_produced(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    for arm, util, max_tok in [
        ("ps-contiguous", 0.10, 64),
        ("ps-contiguous", 0.21, 128),
        ("ps-continuous", 0.80, 64),
        ("ps-continuous", 0.88, 128),
    ]:
        _write(
            rdir,
            f"kv_{arm}_{max_tok}.json",
            _make_result(arm=arm, max_tokens=max_tok, kv_utilization=util),
        )
    out = tmp_path / "figures"
    run(rdir, out)
    assert (_chart_dir(out) / "kv_utilization.png").exists()


# ---------------------------------------------------------------------------
# Integration tests: graceful skips
# ---------------------------------------------------------------------------


def test_empty_results_dir_does_not_raise(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    run(rdir, tmp_path / "figures")  # must not raise


def test_run_returns_one_on_missing_dir(tmp_path: Path) -> None:
    from bench.plot import main

    code = main(["--results", str(tmp_path / "nonexistent"), "--out", str(tmp_path / "out")])
    assert code == 1


def test_all_gpu_missing_produces_no_charts(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    data = _make_result(arm="ps-continuous")
    data["environment"]["gpu"] = None  # type: ignore[index]
    _write(rdir, "bad.json", data)
    out = tmp_path / "figures"
    run(rdir, out)
    captured = capsys.readouterr()
    assert "nothing to plot" in captured.out.lower()


def test_partial_sweep_still_plots(tmp_path: Path) -> None:
    """A single arm at a single concurrency still produces a throughput chart."""
    rdir = tmp_path / "results"
    rdir.mkdir()
    _write(rdir, "one.json", _make_result(arm="ps-continuous", concurrency=8))
    out = tmp_path / "figures"
    run(rdir, out)
    assert (_chart_dir(out) / "throughput_vs_concurrency.png").exists()


def test_missing_throughput_skips_only_that_chart(tmp_path: Path) -> None:
    """When output_throughput is absent, chart 1 is skipped; latency charts still render."""
    rdir = tmp_path / "results"
    rdir.mkdir()
    data = _make_result(arm="ps-continuous", concurrency=8)
    data["summary"]["output_throughput"] = None  # type: ignore[index]
    _write(rdir, "no_throughput.json", data)
    out = tmp_path / "figures"
    run(rdir, out)
    chart_dir = _chart_dir(out)
    assert not (chart_dir / "throughput_vs_concurrency.png").exists()
    assert (chart_dir / "ttft_vs_concurrency.png").exists()
    assert (chart_dir / "itl_vs_concurrency.png").exists()


def test_multiple_hardware_contexts_produce_separate_dirs(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    for gpu_name, model_name in [
        ("NVIDIA A100-SXM4-80GB", "Qwen/Qwen2.5-0.5B-Instruct"),
        ("NVIDIA H100-SXM5-80GB", "meta-llama/Llama-3-8B"),
    ]:
        _write(
            rdir,
            f"{gpu_name.replace(' ', '_')}.json",
            _make_result(arm="ps-continuous", gpu=gpu_name, model=model_name),
        )
    out = tmp_path / "figures"
    run(rdir, out)
    subdirs = list(out.iterdir())
    assert len(subdirs) == 2


# ---------------------------------------------------------------------------
# Phase 8 charts: block-size sweep and preemption policies
# ---------------------------------------------------------------------------


def _with_engine(data: dict, *, block_size: int, policy: str = "recompute", preemptions=0):
    data["config"]["engine"].update(
        {"cache": {"block_size": block_size}, "scheduler": {"preemption_policy": policy}}
    )
    data["workload"]["scheduler"] = {"steps": 100, "preemptions": preemptions}
    return data


def test_parse_result_reads_block_size_and_policy(tmp_path: Path) -> None:
    data = _with_engine(_make_result(arm="ps-continuous"), block_size=32, policy="swap")
    data["workload"]["scheduler"]["preemptions"] = 7
    rec = _parse_result(_write(tmp_path, "r.json", data))
    assert rec is not None
    assert (rec.block_size, rec.preemption_policy, rec.preemptions) == (32, "swap", 7)


def test_block_size_chart_produced(tmp_path: Path) -> None:
    rdir = tmp_path / "results" / "block_size"
    rdir.mkdir(parents=True)
    for b in (8, 16, 32):
        data = _make_result(arm="ps-continuous", kv_utilization=0.9, output_throughput=100.0 * b)
        _write(rdir, f"b{b}.json", _with_engine(data, block_size=b))
    out = tmp_path / "out"
    run(tmp_path / "results", out)
    assert (_chart_dir(out) / "block_size_sweep.png").exists()


def test_block_size_chart_needs_two_sizes(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    _write(rdir, "b16.json", _with_engine(_make_result(arm="ps-continuous"), block_size=16))
    out = tmp_path / "out"
    run(rdir, out)
    assert not (_chart_dir(out) / "block_size_sweep.png").exists()


def test_preemption_chart_produced(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    for policy in ("recompute", "swap"):
        for c in (4, 16):
            data = _make_result(arm="ps-continuous", concurrency=c)
            _write(
                rdir,
                f"{policy}{c}.json",
                _with_engine(data, block_size=16, policy=policy, preemptions=c),
            )
    out = tmp_path / "out"
    run(rdir, out)
    assert (_chart_dir(out) / "preemption_policies.png").exists()


def test_preemption_chart_skipped_without_preemptions(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    for policy in ("recompute", "swap"):
        data = _make_result(arm="ps-continuous")
        _write(rdir, f"{policy}.json", _with_engine(data, block_size=16, policy=policy))
    out = tmp_path / "out"
    run(rdir, out)
    assert not (_chart_dir(out) / "preemption_policies.png").exists()


# ---------------------------------------------------------------------------
# Fields as loadgen actually writes them, and the charts for the new sweeps
# ---------------------------------------------------------------------------


def test_parse_result_reads_realised_prefix_fraction_from_loadgen_shape(tmp_path: Path) -> None:
    """loadgen writes workload.shared_prefix.realised_fraction, never a flat
    prefix_fraction; a cache-off run has no other way to get an x value."""
    data = _make_result(arm="ps-continuous")
    data["workload"]["shared_prefix"] = {"words": 400, "realised_fraction": 0.0}
    rec = _parse_result(_write(tmp_path, "r.json", data))
    assert rec is not None and rec.prefix_fraction == 0.0


def test_prefix_chart_plots_cache_off_from_loadgen_shape(tmp_path: Path) -> None:
    from bench.plot import plot_ttft_vs_prefix_fraction

    rdir = tmp_path / "results"
    rdir.mkdir()
    for frac in (0.0, 0.5):
        for cache in (False, True):
            data = _make_result(
                arm="ps-continuous", prefix_cache={"hit_rate": frac} if cache else None
            )
            data["workload"]["shared_prefix"] = {"words": 400, "realised_fraction": frac}
            _write(rdir, f"f{frac}_{cache}.json", data)
    recs = [_parse_result(p) for p in sorted(rdir.glob("*.json"))]
    off = [r for r in recs if r is not None and not r.prefix_cache_on]
    assert len(off) == 2 and all(r.prefix_fraction is not None for r in off)
    assert plot_ttft_vs_prefix_fraction([r for r in recs if r], tmp_path, "hw")


def test_model_name_is_read_from_a_nested_engine_config(tmp_path: Path) -> None:
    data = _make_result(arm="ps-continuous")
    data["config"]["engine"]["model"] = {"name": _MODEL, "num_layers": 24}
    rec = _parse_result(_write(tmp_path, "r.json", data))
    assert rec is not None and rec.model == _MODEL


def test_sweeps_that_fix_another_variable_stay_out_of_concurrency_charts(tmp_path: Path) -> None:
    from bench.plot import _concurrency_records

    rdir = tmp_path / "r"
    rdir.mkdir()
    a = _make_result(arm="ps-continuous", concurrency=16)
    b = _make_result(arm="ps-continuous", concurrency=16)
    b["workload"]["sweep"] = "block-size"
    recs = [_parse_result(_write(rdir, "a.json", a)), _parse_result(_write(rdir, "b.json", b))]
    kept = _concurrency_records([r for r in recs if r], ["ps-continuous"])
    assert [r.path.name for r in kept] == ["a.json"]


def test_preemption_chart_uses_prompt_length_when_present(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    for policy in ("recompute", "swap"):
        for words in (128, 1024):
            data = _with_engine(
                _make_result(arm="ps-continuous", concurrency=16),
                block_size=16,
                policy=policy,
                preemptions=3,
            )
            data["workload"]["prompt_words"] = words
            data["workload"]["sweep"] = "preemption"
            _write(rdir, f"{policy}{words}.json", data)
    out = tmp_path / "out"
    run(rdir, out)
    assert (_chart_dir(out) / "preemption_policies.png").exists()
    # And none of them leaked into the arms chart as a concurrency=16 point.
    assert not (_chart_dir(out) / "throughput_vs_concurrency.png").exists()


def test_kernel_and_http_charts_produced(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    for arm in ("ps-gather", "ps-cuda", "ps-cuda-graphs", "ps-http", "vllm-http"):
        for c in (1, 8):
            _write(rdir, f"{arm}{c}.json", _make_result(arm=arm, concurrency=c))
    out = tmp_path / "out"
    run(rdir, out)
    d = _chart_dir(out)
    for name in (
        "kernel_throughput_vs_concurrency",
        "kernel_itl_vs_concurrency",
        "http_throughput_vs_concurrency",
        "http_ttft_vs_concurrency",
    ):
        assert (d / f"{name}.png").exists(), name


def test_chunked_prefill_chart_needs_both_sides(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()

    def chunked(on: bool, budget: int, name: str) -> None:
        data = _make_result(arm="ps-chunked-on" if on else "ps-chunked-off", concurrency=16)
        data["config"]["engine"]["scheduler"] = {
            "enable_chunked_prefill": on,
            "max_num_batched_tokens": budget,
        }
        data["workload"]["sweep"] = "chunked-prefill"
        _write(rdir, name, data)

    chunked(True, 256, "on256.json")
    out = tmp_path / "out"
    run(rdir, out)
    assert not (_chart_dir(out) / "chunked_prefill.png").exists()
    chunked(False, 8192, "off.json")
    chunked(True, 512, "on512.json")
    run(rdir, out)
    assert (_chart_dir(out) / "chunked_prefill.png").exists()


def test_kernel_microbench_chart_and_skip_from_run_records(tmp_path: Path) -> None:
    rdir = tmp_path / "results"
    rdir.mkdir()
    rows = [
        {
            "impl": impl,
            "batch_size": 8,
            "block_size": 16,
            "seq_len": L,
            "us_p50": 10.0 * L,
            "effective_gbps": 100.0,
        }
        for impl in ("cuda", "reference")
        for L in (128, 1024)
    ]
    mb = {
        "kind": "kernel_microbench",
        "config": {"peak_gbps": 600.0},
        "environment": {"gpu": _GPU},
        "rows": rows,
    }
    path = _write(rdir, "mb.json", mb)
    assert _parse_result(path) is None
    out = tmp_path / "out"
    run(rdir, out)
    assert (out / _hw_safe(_GPU, "kernel", "microbench") / "kernel_microbench_b8_bs16.png").exists()
