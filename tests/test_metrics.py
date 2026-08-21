"""Unit tests for bench/metrics.py.

Every expected value here is worked out by hand in a comment. A metrics module
tested against its own output would validate nothing — these are the numbers
that every claim this project ever makes will be built on.
"""

from __future__ import annotations

import pytest

from bench.metrics import (
    SLO,
    RequestRecord,
    drop_warmup,
    meets_slo,
    percentile,
    percentile_fields,
    summarize,
    validate_record,
)


def make_record(**kwargs) -> RequestRecord:
    """A well-formed record, overridable field by field."""
    defaults = dict(
        arrival=0.0,
        first_token=1.0,
        tokens=[1.0, 1.5, 2.0],
        finish=2.0,
        prompt_tokens=10,
        output_tokens=3,
    )
    defaults.update(kwargs)
    return RequestRecord(**defaults)


# Two requests used across the summary tests.
#
#   A: arrival 0.0, tokens at 1.0 / 1.5 / 2.0
#      ttft = 1.0 - 0.0 = 1.0     itls = [0.5, 0.5]
#      tpot = (2.0 - 1.0) / 2 = 0.5
#      e2e  = 2.0 - 0.0 = 2.0
#   B: arrival 1.0, tokens at 1.2 / 1.4
#      ttft = 1.2 - 1.0 = 0.2     itls = [0.2]
#      tpot = (1.4 - 1.2) / 1 = 0.2
#      e2e  = 1.4 - 1.0 = 0.4
REQUEST_A = RequestRecord(
    arrival=0.0,
    first_token=1.0,
    tokens=[1.0, 1.5, 2.0],
    finish=2.0,
    prompt_tokens=10,
    output_tokens=3,
)
REQUEST_B = RequestRecord(
    arrival=1.0,
    first_token=1.2,
    tokens=[1.2, 1.4],
    finish=1.4,
    prompt_tokens=20,
    output_tokens=2,
)


class TestRequestRecord:
    def test_derived_latencies(self):
        assert REQUEST_A.ttft == pytest.approx(1.0)
        assert REQUEST_A.e2e == pytest.approx(2.0)
        assert REQUEST_A.itls == pytest.approx([0.5, 0.5])
        assert REQUEST_A.tpot == pytest.approx(0.5)
        assert REQUEST_A.succeeded

    def test_single_token_output_has_no_itl_or_tpot(self):
        # One token means zero gaps. The quantity is undefined, not zero --
        # reporting 0.0 here would make a single-token workload look infinitely
        # smooth.
        rec = make_record(tokens=[1.0], finish=1.0, output_tokens=1)
        assert rec.itls == []
        assert rec.tpot is None
        assert rec.ttft == pytest.approx(1.0)
        assert rec.e2e == pytest.approx(1.0)

    def test_no_tokens_produced(self):
        rec = make_record(first_token=None, tokens=[], finish=None, output_tokens=0)
        assert rec.ttft is None
        assert rec.e2e is None
        assert rec.tpot is None
        assert not rec.succeeded

    def test_errored_record_never_succeeds(self):
        rec = make_record(error="connection reset")
        assert not rec.succeeded
        # The timings are still readable; they are just not counted.
        assert rec.ttft == pytest.approx(1.0)

    def test_unfinished_record_keeps_ttft(self):
        # In flight when the run ended: first token arrived, completion did not.
        rec = make_record(finish=None)
        assert not rec.succeeded
        assert rec.ttft == pytest.approx(1.0)
        assert rec.e2e is None

    def test_dict_round_trip(self):
        restored = RequestRecord.from_dict(REQUEST_A.to_dict())
        assert restored == REQUEST_A

    def test_from_dict_tolerates_a_minimal_entry(self):
        rec = RequestRecord.from_dict({"arrival": 3.0})
        assert rec.arrival == 3.0
        assert rec.tokens == []
        assert rec.output_tokens == 0
        assert not rec.succeeded


class TestPercentile:
    def test_known_values(self):
        # np.percentile linear interpolation on [1, 2, 3, 4, 5]:
        #   p50 -> index 0.50 * 4 = 2.0        -> exactly 3
        #   p95 -> index 0.95 * 4 = 3.8        -> 4 + 0.8 * (5 - 4) = 4.8
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(values, 50) == pytest.approx(3.0)
        assert percentile(values, 95) == pytest.approx(4.8)
        assert percentile(values, 0) == pytest.approx(1.0)
        assert percentile(values, 100) == pytest.approx(5.0)

    def test_unsorted_input(self):
        assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 50) == pytest.approx(3.0)

    def test_empty_is_none_not_zero(self):
        # A zero in a committed result file reads as a measurement.
        assert percentile([], 50) is None

    def test_single_value(self):
        assert percentile([7.0], 99) == pytest.approx(7.0)


class TestPercentileFields:
    def test_key_names_and_mean(self):
        fields = percentile_fields("ttft", [1.0, 2.0, 3.0, 4.0, 5.0])
        assert set(fields) == {"ttft_p50", "ttft_p95", "ttft_p99", "ttft_mean"}
        assert fields["ttft_p50"] == pytest.approx(3.0)
        assert fields["ttft_mean"] == pytest.approx(3.0)

    def test_fractional_percentile_key(self):
        fields = percentile_fields("itl", [1.0, 2.0], percentiles=(99.9,))
        assert "itl_p99_9" in fields

    def test_empty_gives_none_everywhere(self):
        fields = percentile_fields("e2e", [])
        assert all(v is None for v in fields.values())


class TestSLO:
    def test_all_thresholds_must_hold(self):
        # A: ttft 1.0, tpot 0.5, e2e 2.0
        assert meets_slo(REQUEST_A, SLO(ttft=1.5, tpot=1.0, e2e=3.0))
        assert not meets_slo(REQUEST_A, SLO(ttft=0.5))
        assert not meets_slo(REQUEST_A, SLO(tpot=0.1))
        assert not meets_slo(REQUEST_A, SLO(e2e=1.0))

    def test_threshold_is_inclusive(self):
        assert meets_slo(REQUEST_A, SLO(ttft=1.0))

    def test_unconstrained_slo_passes_any_successful_request(self):
        assert meets_slo(REQUEST_A, SLO())

    def test_failed_request_is_never_good(self):
        assert not meets_slo(make_record(error="timeout"), SLO(ttft=10.0))
        assert not meets_slo(make_record(finish=None), SLO(ttft=10.0))

    def test_unmeasurable_constrained_stat_fails_rather_than_passing(self):
        # Single-token output has no TPOT. Counting it as good would inflate
        # goodput on exactly the workload that stresses TTFT hardest.
        rec = make_record(tokens=[1.0], finish=1.0, output_tokens=1)
        assert rec.tpot is None
        assert not meets_slo(rec, SLO(tpot=1.0))
        assert meets_slo(rec, SLO(ttft=2.0))


class TestValidateRecord:
    def test_well_formed_record_has_no_problems(self):
        assert validate_record(REQUEST_A) == []
        assert validate_record(REQUEST_B) == []

    def test_detects_token_count_disagreement(self):
        problems = validate_record(make_record(output_tokens=5))
        assert any("disagrees with" in p for p in problems)

    def test_detects_non_monotonic_timestamps(self):
        problems = validate_record(make_record(tokens=[1.0, 0.9, 2.0]))
        assert any("monotonically" in p for p in problems)

    def test_detects_first_token_before_arrival(self):
        problems = validate_record(make_record(arrival=5.0))
        assert any("precedes arrival" in p for p in problems)

    def test_detects_finish_before_first_token(self):
        problems = validate_record(
            RequestRecord(arrival=0.0, first_token=2.0, tokens=[], finish=1.0, output_tokens=1)
        )
        assert any("precedes first_token" in p for p in problems)

    def test_detects_tokens_head_mismatch(self):
        problems = validate_record(make_record(first_token=0.5))
        assert any("tokens[0]" in p for p in problems)

    def test_detects_negative_counts(self):
        problems = validate_record(make_record(prompt_tokens=-1))
        assert any("prompt_tokens is negative" in p for p in problems)


class TestDropWarmup:
    def test_drops_by_count_in_arrival_order(self):
        records = [make_record(arrival=float(i)) for i in range(5)]
        kept = drop_warmup(reversed(records), num_requests=2)
        assert [r.arrival for r in kept] == [2.0, 3.0, 4.0]

    def test_drops_by_elapsed_seconds(self):
        records = [make_record(arrival=float(i)) for i in range(5)]
        # Window starts at the first arrival (0.0); anything before 2.0 goes.
        kept = drop_warmup(records, seconds=2.0)
        assert [r.arrival for r in kept] == [2.0, 3.0, 4.0]

    def test_filters_compose_from_the_original_start(self):
        records = [make_record(arrival=float(i)) for i in range(6)]
        # Count filter removes arrivals 0 and 1; the 2.0s window is still
        # measured from arrival 0.0, so nothing further is dropped.
        kept = drop_warmup(records, num_requests=2, seconds=2.0)
        assert [r.arrival for r in kept] == [2.0, 3.0, 4.0, 5.0]

    def test_does_not_mutate_input(self):
        records = [make_record(arrival=2.0), make_record(arrival=1.0)]
        drop_warmup(records, num_requests=1)
        assert [r.arrival for r in records] == [2.0, 1.0]

    def test_dropping_everything_is_not_an_error(self):
        records = [make_record(arrival=float(i)) for i in range(3)]
        assert drop_warmup(records, num_requests=10) == []
        assert drop_warmup([], num_requests=1) == []

    def test_no_filters_is_a_sorted_copy(self):
        records = [make_record(arrival=2.0), make_record(arrival=1.0)]
        assert [r.arrival for r in drop_warmup(records)] == [1.0, 2.0]


class TestSummarize:
    def test_counts_and_totals(self):
        s = summarize([REQUEST_A, REQUEST_B])
        assert s["num_requests"] == 2
        assert s["num_succeeded"] == 2
        assert s["num_failed"] == 0
        assert s["total_output_tokens"] == 5  # 3 + 2
        assert s["total_prompt_tokens"] == 30  # 10 + 20
        assert s["units"] == "seconds"

    def test_latency_percentiles(self):
        s = summarize([REQUEST_A, REQUEST_B])
        # ttfts [1.0, 0.2] -> sorted [0.2, 1.0] -> p50 = midpoint = 0.6
        assert s["ttft_p50"] == pytest.approx(0.6)
        # e2e [2.0, 0.4] -> p50 = 1.2
        assert s["e2e_p50"] == pytest.approx(1.2)
        # tpot [0.5, 0.2] -> p50 = 0.35
        assert s["tpot_p50"] == pytest.approx(0.35)

    def test_itls_are_pooled_across_requests(self):
        s = summarize([REQUEST_A, REQUEST_B])
        # A contributes [0.5, 0.5], B contributes [0.2] -> pooled [0.2,0.5,0.5]
        # p50 of three values is the middle one: 0.5
        assert s["itl_p50"] == pytest.approx(0.5)
        # mean over the pool, not a mean of per-request means:
        # (0.5 + 0.5 + 0.2) / 3 = 0.4
        assert s["itl_mean"] == pytest.approx(0.4)

    def test_duration_and_throughput(self):
        s = summarize([REQUEST_A, REQUEST_B])
        # last finish 2.0 - first arrival 0.0 = 2.0s
        assert s["duration"] == pytest.approx(2.0)
        assert s["output_throughput"] == pytest.approx(2.5)  # 5 tokens / 2.0s
        assert s["prompt_throughput"] == pytest.approx(15.0)  # 30 tokens / 2.0s
        assert s["request_throughput"] == pytest.approx(1.0)  # 2 requests / 2.0s

    def test_explicit_duration_overrides_the_observed_window(self):
        s = summarize([REQUEST_A, REQUEST_B], duration=5.0)
        assert s["duration"] == pytest.approx(5.0)
        assert s["output_throughput"] == pytest.approx(1.0)  # 5 tokens / 5.0s

    def test_failures_are_excluded_from_latency_but_counted(self):
        failed = RequestRecord(arrival=0.0, first_token=None, finish=None, error="oom")
        s = summarize([REQUEST_A, REQUEST_B, failed])
        assert s["num_requests"] == 3
        assert s["num_succeeded"] == 2
        assert s["num_failed"] == 1
        # Unchanged from the two-request case: the failure did not dilute it.
        assert s["ttft_p50"] == pytest.approx(0.6)
        assert s["total_output_tokens"] == 5

    def test_goodput_requires_an_slo(self):
        s = summarize([REQUEST_A, REQUEST_B])
        assert s["goodput"] is None
        assert s["num_good"] is None
        assert s["slo"] is None

    def test_goodput_with_an_slo(self):
        # ttft <= 0.5s: A (1.0) misses, B (0.2) meets -> 1 good in 2.0s
        s = summarize([REQUEST_A, REQUEST_B], slo=SLO(ttft=0.5))
        assert s["num_good"] == 1
        assert s["goodput"] == pytest.approx(0.5)
        assert s["slo"] == {"ttft": 0.5, "tpot": None, "e2e": None}

    def test_empty_slo_is_treated_as_no_slo(self):
        s = summarize([REQUEST_A], slo=SLO())
        assert s["goodput"] is None

    def test_empty_input_produces_nulls_not_zeros(self):
        s = summarize([])
        assert s["num_requests"] == 0
        assert s["duration"] is None
        assert s["ttft_p50"] is None
        assert s["output_throughput"] is None
        assert s["total_output_tokens"] == 0

    def test_all_requests_failed(self):
        failed = RequestRecord(arrival=0.0, error="oom")
        s = summarize([failed, failed])
        assert s["num_failed"] == 2
        assert s["duration"] is None
        assert s["output_throughput"] is None
        assert s["itl_p99"] is None

    def test_zero_duration_does_not_divide_by_zero(self):
        instant = RequestRecord(
            arrival=1.0, first_token=1.0, tokens=[1.0], finish=1.0, output_tokens=1
        )
        s = summarize([instant])
        assert s["duration"] == pytest.approx(0.0)
        assert s["output_throughput"] is None

    def test_custom_percentiles(self):
        s = summarize([REQUEST_A, REQUEST_B], percentiles=(90.0,))
        assert "ttft_p90" in s
        assert "ttft_p99" not in s

    def test_summary_is_json_serializable(self):
        import json

        s = summarize([REQUEST_A, REQUEST_B], slo=SLO(ttft=0.5))
        assert json.loads(json.dumps(s)) == s
