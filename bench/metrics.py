"""Latency, throughput, and goodput metrics for the PagedServe benchmark harness.

Pure functions over per-request timing records. No torch, no GPU, no I/O, no
clock reads — everything here is a deterministic transform of numbers that were
recorded elsewhere. That is what makes it unit-testable on a laptop, and it is
why this module is the first thing built in Phase 0: a speedup is a ratio, and
the ratio is only meaningful if both sides were measured by the same code.

Conventions
-----------
**All durations are in seconds.** Timestamps are assumed to come from a single
monotonic clock (``time.perf_counter``) within one process, so only differences
are ever meaningful.

**Percentiles use linear interpolation between order statistics** (numpy's
default, and what vLLM's own benchmark scripts report), so numbers here are
comparable to numbers published by other serving systems.

**An unmeasurable statistic is ``None``, never ``0.0``.** A zero in a result
file reads as a measurement; a null reads as an absence. See AGENTS.md §2.1.

Metric definitions
------------------
TTFT
    ``first_token - arrival``. Includes queueing delay, which is the point:
    under open-loop load, queueing is a first-class part of what a user feels.
ITL
    Inter-token latency, the gap between consecutive output tokens. Computed
    per request from the token timestamps, then **pooled across all requests**
    before taking percentiles, so one request that stalls shows up in the tail
    rather than being averaged away inside its own request.
TPOT
    Time per output token, ``(finish - first_token) / (output_tokens - 1)``.
    A per-request mean, available even when per-token timestamps were not
    recorded. Undefined for single-token outputs.
E2E
    ``finish - arrival``.
Goodput
    Completed requests that met *every* configured per-request SLO, divided by
    wall-clock duration. Raw throughput can always be inflated by letting
    latency explode; goodput cannot, which is why it is the honest headline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_PERCENTILES",
    "RequestRecord",
    "SLO",
    "drop_warmup",
    "meets_slo",
    "percentile",
    "percentile_fields",
    "summarize",
    "validate_record",
]

DEFAULT_PERCENTILES: tuple[float, ...] = (50.0, 95.0, 99.0)


@dataclass
class RequestRecord:
    """Raw timings for one request. The unit of evidence.

    Field names match the ``requests`` entries of the result JSON schema in
    AGENTS.md §6, so ``to_dict`` output is committed verbatim.

    Attributes:
        arrival: When the request was *issued* by the load generator. Under
            open-loop load this is the scheduled arrival time, not the time the
            backend accepted it — the difference is queueing delay, which must
            be inside TTFT, not hidden before it.
        first_token: Timestamp of the first output token, or ``None`` if the
            request produced nothing.
        tokens: Timestamp of every output token, first included. May be empty if
            the backend cannot report per-token times; ITL is then unavailable
            but TTFT, E2E, and TPOT are not.
        finish: Timestamp of the last output token, or ``None`` if the request
            never completed (in flight at shutdown, cancelled, or errored).
        prompt_tokens: Prompt length in tokens, or ``None`` when no tokenizer
            was available to count it. Needed to interpret TTFT, since prefill
            cost is roughly linear in it. ``None`` rather than ``0`` so that an
            uncounted prompt cannot masquerade as an empty one.
        output_tokens: Number of tokens generated.
        error: Failure reason, or ``None``. A record with an error is excluded
            from latency statistics and counted separately, never silently
            dropped.
    """

    arrival: float
    first_token: float | None = None
    tokens: list[float] = field(default_factory=list)
    finish: float | None = None
    prompt_tokens: int | None = None
    output_tokens: int = 0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """True if this request completed and produced at least one token."""
        return self.error is None and self.finish is not None and self.output_tokens > 0

    @property
    def ttft(self) -> float | None:
        """Time to first token, or ``None`` if no token was ever produced."""
        if self.first_token is None:
            return None
        return self.first_token - self.arrival

    @property
    def e2e(self) -> float | None:
        """Arrival to last token, or ``None`` if the request never completed."""
        if self.finish is None:
            return None
        return self.finish - self.arrival

    @property
    def itls(self) -> list[float]:
        """Gaps between consecutive output tokens.

        Empty when fewer than two token timestamps were recorded. The gap from
        arrival to the first token is deliberately excluded — that is TTFT, and
        it is one to two orders of magnitude larger, so folding it in would
        swamp the ITL distribution it was mixed into.
        """
        return [b - a for a, b in zip(self.tokens, self.tokens[1:], strict=False)]

    @property
    def tpot(self) -> float | None:
        """Mean time per output token after the first.

        ``None`` for single-token outputs, where the quantity is undefined
        rather than zero.
        """
        if self.first_token is None or self.finish is None or self.output_tokens < 2:
            return None
        return (self.finish - self.first_token) / (self.output_tokens - 1)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the result JSON schema in AGENTS.md §6."""
        return {
            "arrival": self.arrival,
            "first_token": self.first_token,
            "tokens": list(self.tokens),
            "finish": self.finish,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RequestRecord:
        """Rebuild a record from committed JSON, so old results stay analyzable."""
        return cls(
            arrival=d["arrival"],
            first_token=d.get("first_token"),
            tokens=list(d.get("tokens") or []),
            finish=d.get("finish"),
            prompt_tokens=d.get("prompt_tokens"),
            output_tokens=d.get("output_tokens", 0),
            error=d.get("error"),
        )


@dataclass(frozen=True)
class SLO:
    """Per-request service-level objective. ``None`` means unconstrained.

    A request is "good" only if it satisfies every threshold that is set. All
    thresholds are in seconds.
    """

    ttft: float | None = None
    tpot: float | None = None
    e2e: float | None = None

    def is_empty(self) -> bool:
        """True if no threshold is set, in which case goodput is meaningless."""
        return self.ttft is None and self.tpot is None and self.e2e is None

    def to_dict(self) -> dict[str, float | None]:
        return {"ttft": self.ttft, "tpot": self.tpot, "e2e": self.e2e}


def meets_slo(record: RequestRecord, slo: SLO) -> bool:
    """Whether a request met every configured threshold.

    A failed or incomplete request never counts as good. A constrained
    statistic that is unmeasurable for this request (TPOT on a one-token
    output) also fails, rather than passing by default — counting an
    unmeasured request as good is how goodput gets quietly inflated.
    """
    if not record.succeeded:
        return False
    for threshold, value in (
        (slo.ttft, record.ttft),
        (slo.tpot, record.tpot),
        (slo.e2e, record.e2e),
    ):
        if threshold is None:
            continue
        if value is None or value > threshold:
            return False
    return True


def validate_record(record: RequestRecord) -> list[str]:
    """Return a list of internal inconsistencies, empty if the record is sound.

    Kept separate from construction on purpose: a benchmark that has been under
    load for ten minutes should not die on one malformed record. The load
    generator calls this and logs, tests call this and assert.
    """
    problems: list[str] = []
    if record.prompt_tokens is not None and record.prompt_tokens < 0:
        problems.append(f"prompt_tokens is negative: {record.prompt_tokens}")
    if record.output_tokens < 0:
        problems.append(f"output_tokens is negative: {record.output_tokens}")
    if not math.isfinite(record.arrival):
        problems.append(f"arrival is not finite: {record.arrival}")
    if record.first_token is not None and record.first_token < record.arrival:
        problems.append("first_token precedes arrival")
    if (
        record.finish is not None
        and record.first_token is not None
        and record.finish < record.first_token
    ):
        problems.append("finish precedes first_token")
    if record.tokens:
        if any(b < a for a, b in zip(record.tokens, record.tokens[1:], strict=False)):
            problems.append("token timestamps are not monotonically non-decreasing")
        if record.first_token is not None and record.tokens[0] != record.first_token:
            problems.append("tokens[0] does not equal first_token")
        if record.output_tokens and len(record.tokens) != record.output_tokens:
            problems.append(
                f"len(tokens)={len(record.tokens)} disagrees with "
                f"output_tokens={record.output_tokens}"
            )
    return problems


def percentile(values: Sequence[float], p: float) -> float | None:
    """The ``p``-th percentile by linear interpolation, or ``None`` if empty."""
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def percentile_fields(
    prefix: str,
    values: Sequence[float],
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, float | None]:
    """Build ``{"<prefix>_p50": ..., "<prefix>_p95": ...}`` for a result summary.

    Also emits ``<prefix>_mean``, which exists only so a reader can compare it
    against the tail and see the skew. Never report the mean alone.
    """
    out: dict[str, float | None] = {}
    for p in percentiles:
        out[f"{prefix}_{_percentile_key(p)}"] = percentile(values, p)
    out[f"{prefix}_mean"] = float(np.mean(values)) if len(values) else None
    return out


def _percentile_key(p: float) -> str:
    """``50.0 -> 'p50'``, ``99.9 -> 'p99_9'``."""
    if float(p).is_integer():
        return f"p{int(p)}"
    return f"p{p}".replace(".", "_")


def drop_warmup(
    records: Iterable[RequestRecord],
    *,
    num_requests: int = 0,
    seconds: float = 0.0,
) -> list[RequestRecord]:
    """Discard the warmup window, per the benchmark protocol in AGENTS.md §6.

    Drops the first ``num_requests`` records by arrival order and any record
    arriving within ``seconds`` of the first arrival. CUDA context creation,
    kernel autotuning, allocator growth, and cuBLAS handle setup all land in
    the first few seconds; leaving them in makes the system look worse than it
    is and inflates the run-to-run spread.

    Returns a new list sorted by arrival. Never mutates its input.
    """
    ordered = sorted(records, key=lambda r: r.arrival)
    if not ordered:
        return []
    # Pinned before any dropping: the window is measured from the start of the
    # *original* run, so the two filters compose to "whichever is stricter"
    # instead of stacking into an unintended longer warmup.
    run_start = ordered[0].arrival
    if num_requests > 0:
        ordered = ordered[num_requests:]
    if seconds > 0.0:
        cutoff = run_start + seconds
        ordered = [r for r in ordered if r.arrival >= cutoff]
    return ordered


def summarize(
    records: Sequence[RequestRecord],
    *,
    slo: SLO | None = None,
    duration: float | None = None,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
) -> dict[str, Any]:
    """Compute the ``summary`` block of a result JSON.

    Latency statistics are computed over successful requests only; failures are
    counted in ``num_failed`` and reported rather than dropped, because a
    configuration that goes fast by erroring out is not fast.

    Args:
        records: Per-request timings, warmup already removed by ``drop_warmup``.
        slo: Thresholds for goodput. Goodput is ``None`` when no SLO is given —
            there is no default SLO, because a default would be a number nobody
            chose.
        duration: Wall-clock seconds the throughput denominators divide by. When
            ``None``, inferred as last finish minus first arrival over these
            records. Pass it explicitly for a fixed-duration run so that an idle
            tail at the end is not silently excluded.
        percentiles: Which percentiles to report.

    Returns:
        A JSON-serializable dict. Every duration is in seconds; unmeasurable
        entries are ``None``.
    """
    succeeded = [r for r in records if r.succeeded]
    failed = [r for r in records if not r.succeeded]

    ttfts = [t for r in succeeded if (t := r.ttft) is not None]
    e2es = [t for r in succeeded if (t := r.e2e) is not None]
    tpots = [t for r in succeeded if (t := r.tpot) is not None]
    # Pooled across requests: one stalled request belongs in the global tail.
    itls = [gap for r in succeeded for gap in r.itls]

    total_output_tokens = sum(r.output_tokens for r in succeeded)
    # All-or-nothing: if any request could not report its prompt length, a sum
    # over the rest silently understates the total, and prompt throughput
    # derived from it would be a number nobody measured. Report null instead.
    total_prompt_tokens: int | None = (
        sum(r.prompt_tokens or 0 for r in succeeded)
        if succeeded and all(r.prompt_tokens is not None for r in succeeded)
        else None
    )

    if duration is None:
        duration = _observed_duration(records)

    summary: dict[str, Any] = {
        "units": "seconds",
        "num_requests": len(records),
        "num_succeeded": len(succeeded),
        "num_failed": len(failed),
        "total_prompt_tokens": total_prompt_tokens,
        "total_output_tokens": total_output_tokens,
        "duration": duration,
    }
    summary.update(percentile_fields("ttft", ttfts, percentiles))
    summary.update(percentile_fields("itl", itls, percentiles))
    summary.update(percentile_fields("tpot", tpots, percentiles))
    summary.update(percentile_fields("e2e", e2es, percentiles))

    usable_duration = duration is not None and duration > 0.0
    summary["output_throughput"] = total_output_tokens / duration if usable_duration else None
    summary["prompt_throughput"] = (
        total_prompt_tokens / duration
        if usable_duration and total_prompt_tokens is not None
        else None
    )
    summary["request_throughput"] = len(succeeded) / duration if usable_duration else None

    if slo is not None and not slo.is_empty():
        num_good = sum(1 for r in records if meets_slo(r, slo))
        summary["slo"] = slo.to_dict()
        summary["num_good"] = num_good
        summary["goodput"] = num_good / duration if usable_duration else None
    else:
        summary["slo"] = None
        summary["num_good"] = None
        summary["goodput"] = None

    return summary


def _observed_duration(records: Sequence[RequestRecord]) -> float | None:
    """Wall time from the first arrival to the last observed timestamp.

    Uses arrivals from *all* records but completions only from those that
    finished; an in-flight request at shutdown should not stretch the window.
    """
    if not records:
        return None
    start = min(r.arrival for r in records)
    ends = [r.finish for r in records if r.finish is not None]
    if not ends:
        return None
    return max(ends) - start
