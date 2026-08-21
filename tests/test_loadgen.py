"""Unit tests for bench/loadgen.py.

Timing assertions use loose bounds on purpose — these run on a laptop under an
unknown load, and a flaky harness test would train us to ignore the harness.
Anything that can be tested exactly (the arrival schedule, the result schema,
error handling, concurrency limits) is tested exactly instead.
"""

from __future__ import annotations

import asyncio
import json
import random

import pytest

from bench.loadgen import (
    MockBackend,
    PromptRequest,
    build_result,
    collect_environment,
    load_sharegpt,
    poisson_offsets,
    run_closed_loop,
    run_open_loop,
    synthetic_prompts,
    write_result,
)
from bench.metrics import SLO, RequestRecord


def prompts(n: int, max_tokens: int = 3) -> list[PromptRequest]:
    return synthetic_prompts(n, max_tokens=max_tokens)


class TestPoissonOffsets:
    def test_starts_at_zero_and_is_non_decreasing(self):
        offsets = poisson_offsets(10.0, 50, random.Random(0))
        assert offsets[0] == 0.0
        assert all(b >= a for a, b in zip(offsets, offsets[1:], strict=False))

    def test_is_reproducible_under_a_seed(self):
        assert poisson_offsets(5.0, 20, random.Random(42)) == poisson_offsets(
            5.0, 20, random.Random(42)
        )

    def test_different_seeds_give_different_schedules(self):
        assert poisson_offsets(5.0, 20, random.Random(1)) != poisson_offsets(
            5.0, 20, random.Random(2)
        )

    def test_mean_gap_approaches_one_over_rate(self):
        # Exponential(rate) has mean 1/rate. With 20k samples the sample mean
        # sits well inside 10% of it, so this is a real check without being
        # flaky.
        rate = 20.0
        offsets = poisson_offsets(rate, 20_000, random.Random(7))
        mean_gap = offsets[-1] / (len(offsets) - 1)
        assert mean_gap == pytest.approx(1.0 / rate, rel=0.1)

    def test_requires_a_positive_rate(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="rate must be positive"):
                poisson_offsets(bad, 5, random.Random(0))

    def test_zero_requests(self):
        assert poisson_offsets(1.0, 0, random.Random(0)) == []


class TestOpenLoop:
    def test_records_every_request(self):
        backend = MockBackend(ttft=0.001, itl=0.0)
        records, _ = asyncio.run(
            run_open_loop(backend, prompts(8), rate=1000.0, rng=random.Random(0))
        )
        assert len(records) == 8
        assert all(r.succeeded for r in records)
        assert all(r.output_tokens == 3 for r in records)

    def test_arrival_is_the_scheduled_time_not_the_dispatch_time(self):
        # The whole coordinated-omission defence rests on this. Arrivals must
        # reproduce the seeded schedule's *gaps*, regardless of dispatch jitter.
        rng_for_run = random.Random(11)
        backend = MockBackend(ttft=0.001, itl=0.0)
        records, _ = asyncio.run(run_open_loop(backend, prompts(12), rate=200.0, rng=rng_for_run))
        expected = poisson_offsets(200.0, 12, random.Random(11))
        actual = [r.arrival - records[0].arrival for r in records]
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_does_not_wait_for_completion_before_issuing(self):
        # A slow backend must not stretch the arrival schedule: 6 requests at
        # 500/s should all be issued long before a 0.2s-per-request backend
        # finishes even the first one.
        backend = MockBackend(ttft=0.2, itl=0.0)
        records, _ = asyncio.run(
            run_open_loop(backend, prompts(6, max_tokens=1), rate=500.0, rng=random.Random(3))
        )
        span = records[-1].arrival - records[0].arrival
        assert span < 0.15
        assert backend.max_in_flight > 1

    def test_unknown_prompt_tokens_stay_unknown_in_the_record(self):
        # Synthetic prompts have no tokenizer behind them; the None must survive
        # into the record rather than being coerced to 0.
        backend = MockBackend(ttft=0.0, itl=0.0)
        records, _ = asyncio.run(
            run_open_loop(backend, prompts(3), rate=1000.0, rng=random.Random(0))
        )
        assert all(r.prompt_tokens is None for r in records)

    def test_known_prompt_tokens_are_carried_through(self):
        backend = MockBackend(ttft=0.0, itl=0.0)
        counted = [PromptRequest(prompt="hi", max_tokens=2, prompt_tokens=7)]
        records, _ = asyncio.run(run_open_loop(backend, counted, rate=1000.0, rng=random.Random(0)))
        assert records[0].prompt_tokens == 7

    def test_reports_dispatch_lag(self):
        backend = MockBackend(ttft=0.0, itl=0.0)
        _, lag = asyncio.run(run_open_loop(backend, prompts(5), rate=1000.0, rng=random.Random(0)))
        assert lag.max >= 0.0
        assert lag.mean >= 0.0
        assert set(lag.to_dict()) == {"max", "mean", "num_late"}

    def test_backend_failures_become_records_not_exceptions(self):
        # Every second call fails. The run must complete and the failures must
        # appear in the output rather than disappearing from the percentiles.
        backend = MockBackend(ttft=0.0, itl=0.0, fail_every=2)
        records, _ = asyncio.run(
            run_open_loop(backend, prompts(6), rate=1000.0, rng=random.Random(0))
        )
        assert len(records) == 6
        failed = [r for r in records if not r.succeeded]
        assert len(failed) == 3
        assert all(r.error is not None and "mock backend failure" in r.error for r in failed)
        assert all(r.finish is None for r in failed)

    def test_duration_stops_issuing_early(self):
        backend = MockBackend(ttft=0.0, itl=0.0)
        records, _ = asyncio.run(
            run_open_loop(backend, prompts(10_000), rate=200.0, rng=random.Random(0), duration=0.05)
        )
        assert 0 < len(records) < 10_000


class TestClosedLoop:
    def test_never_exceeds_the_configured_concurrency(self):
        backend = MockBackend(ttft=0.002, itl=0.0)
        asyncio.run(run_closed_loop(backend, prompts(30), concurrency=4))
        assert backend.max_in_flight <= 4

    def test_actually_reaches_the_configured_concurrency(self):
        backend = MockBackend(ttft=0.02, itl=0.0)
        asyncio.run(run_closed_loop(backend, prompts(24), concurrency=4))
        assert backend.max_in_flight == 4

    def test_runs_every_prompt_exactly_once(self):
        backend = MockBackend(ttft=0.0, itl=0.0)
        records = asyncio.run(run_closed_loop(backend, prompts(20), concurrency=3))
        assert len(records) == 20
        assert backend.num_calls == 20

    def test_single_worker_is_serial(self):
        backend = MockBackend(ttft=0.001, itl=0.0)
        asyncio.run(run_closed_loop(backend, prompts(5), concurrency=1))
        assert backend.max_in_flight == 1

    def test_rejects_nonsense_concurrency(self):
        with pytest.raises(ValueError, match="concurrency must be at least 1"):
            asyncio.run(run_closed_loop(MockBackend(), prompts(1), concurrency=0))

    def test_duration_stops_workers_early(self):
        backend = MockBackend(ttft=0.005, itl=0.0)
        records = asyncio.run(
            run_closed_loop(backend, prompts(10_000), concurrency=2, duration=0.05)
        )
        assert 0 < len(records) < 10_000


class TestDatasets:
    def test_synthetic_prompts_leave_token_counts_unknown(self):
        # Guessing a token count without a tokenizer would put a fabricated
        # number into prompt_throughput.
        for p in synthetic_prompts(3, max_tokens=8):
            assert p.prompt_tokens is None
            assert p.max_tokens == 8
            assert p.prompt

    def test_load_sharegpt_takes_the_first_human_turn(self, tmp_path):
        path = tmp_path / "sharegpt.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "conversations": [
                            {"from": "human", "value": "first human turn"},
                            {"from": "gpt", "value": "a reply"},
                            {"from": "human", "value": "second human turn"},
                        ]
                    }
                ]
            )
        )
        loaded = load_sharegpt(path, 1, rng=random.Random(0), max_tokens=16)
        assert loaded[0].prompt == "first human turn"
        assert loaded[0].prompt_tokens is None

    def test_load_sharegpt_cycles_when_the_dataset_is_short(self, tmp_path):
        path = tmp_path / "sharegpt.json"
        path.write_text(json.dumps([{"conversations": [{"from": "human", "value": "only one"}]}]))
        loaded = load_sharegpt(path, 5, rng=random.Random(0), max_tokens=16)
        assert len(loaded) == 5

    def test_load_sharegpt_skips_conversations_with_no_human_turn(self, tmp_path):
        path = tmp_path / "sharegpt.json"
        path.write_text(
            json.dumps(
                [
                    {"conversations": [{"from": "gpt", "value": "no human here"}]},
                    {"conversations": [{"from": "human", "value": "usable"}]},
                ]
            )
        )
        loaded = load_sharegpt(path, 2, rng=random.Random(0), max_tokens=16)
        assert {p.prompt for p in loaded} == {"usable"}

    def test_load_sharegpt_rejects_an_unusable_file(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text(json.dumps([]))
        with pytest.raises(ValueError, match="no usable human turns"):
            load_sharegpt(path, 1, rng=random.Random(0), max_tokens=16)


class TestEnvironment:
    def test_always_reports_observable_fields(self):
        env = collect_environment()
        assert env["host"]
        assert env["timestamp"]
        assert env["python"]

    def test_gpu_fields_are_none_rather_than_invented_when_absent(self):
        env = collect_environment()
        for key in ("torch", "cuda", "driver", "gpu"):
            assert key in env
            assert env[key] is None or isinstance(env[key], str)


class TestBuildResult:
    def records(self) -> list[RequestRecord]:
        # Four requests arriving one second apart, each 2 tokens.
        return [
            RequestRecord(
                arrival=float(i),
                first_token=float(i) + 0.5,
                tokens=[float(i) + 0.5, float(i) + 0.7],
                finish=float(i) + 0.7,
                prompt_tokens=10,
                output_tokens=2,
            )
            for i in range(4)
        ]

    def test_has_every_top_level_key_from_the_schema(self):
        result = build_result(
            self.records(), config={"backend": "mock"}, workload={"dataset": "synthetic"}
        )
        assert set(result) >= {"config", "environment", "workload", "requests", "summary"}

    def test_writes_all_records_including_warmup(self):
        result = build_result(
            self.records(),
            config={},
            workload={},
            warmup_requests=2,
        )
        # Raw evidence keeps everything; only the summary drops the warmup.
        assert len(result["requests"]) == 4
        assert result["workload"]["num_requests_measured"] == 2
        assert result["summary"]["num_requests"] == 2

    def test_records_the_warmup_parameters_used(self):
        result = build_result(
            self.records(), config={}, workload={}, warmup_requests=1, warmup_seconds=0.5
        )
        assert result["workload"]["warmup_requests"] == 1
        assert result["workload"]["warmup_seconds"] == 0.5

    def test_summary_reflects_the_slo(self):
        result = build_result(self.records(), config={}, workload={}, slo=SLO(ttft=1.0))
        assert result["summary"]["slo"] == {"ttft": 1.0, "tpot": None, "e2e": None}
        assert result["summary"]["num_good"] == 4

    def test_workload_fields_are_preserved(self):
        result = build_result(
            self.records(), config={}, workload={"dataset": "sharegpt", "rate": 8.0}
        )
        assert result["workload"]["dataset"] == "sharegpt"
        assert result["workload"]["rate"] == 8.0

    def test_result_is_json_serializable(self):
        result = build_result(self.records(), config={}, workload={})
        assert json.loads(json.dumps(result)) == result


class TestWriteResult:
    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "results" / "nested" / "run.json"
        write_result(out, {"summary": {"ttft_p50": None}})
        assert out.exists()
        assert json.loads(out.read_text())["summary"]["ttft_p50"] is None


class TestEndToEnd:
    def test_a_full_open_loop_run_produces_a_valid_result_file(self, tmp_path):
        backend = MockBackend(ttft=0.002, itl=0.001)
        records, lag = asyncio.run(
            run_open_loop(backend, prompts(20, max_tokens=4), rate=500.0, rng=random.Random(5))
        )
        result = build_result(
            records,
            config={"backend": "mock", "mode": "poisson"},
            workload={"dataset": "synthetic", "arrival": "poisson", "rate": 500.0},
            slo=SLO(ttft=1.0),
            warmup_requests=5,
        )
        result["workload"]["dispatch_lag"] = lag.to_dict()
        out = write_result(tmp_path / "run.json", result)

        reloaded = json.loads(out.read_text())
        assert len(reloaded["requests"]) == 20
        assert reloaded["summary"]["num_requests"] == 15
        assert reloaded["summary"]["num_succeeded"] == 15
        # Real measured timings, so only structure is asserted -- no expected
        # magnitudes, because those would be numbers nobody measured.
        assert reloaded["summary"]["ttft_p50"] > 0
        assert reloaded["summary"]["output_throughput"] > 0
        assert reloaded["summary"]["units"] == "seconds"

        # And the records round-trip back into the metrics layer.
        restored = [RequestRecord.from_dict(d) for d in reloaded["requests"]]
        assert all(r.succeeded for r in restored)
