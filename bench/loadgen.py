"""Load generator for the PagedServe benchmark harness.

Two arrival processes, one backend interface, one result file format.

**Open loop (Poisson).** Requests are issued on a schedule computed up front
from an exponential inter-arrival distribution, and the generator does *not*
wait for a response before issuing the next one. This is the headline mode:
real traffic arrives whether or not the server is keeping up, so queueing delay
is a first-class part of what a user experiences.

**Closed loop.** ``concurrency`` workers each issue a request, wait for it to
complete, and issue the next. Useful for isolating batch-scaling behaviour, and
flattering: a saturated server slows the arrival rate down, so the offered load
silently drops to whatever the server can handle. Never report it as a headline.

Coordinated omission
--------------------
In open-loop mode a request's ``arrival`` is its **scheduled** time, not the
moment the generator actually dispatched it. If the server stalls and the
generator falls behind, charging only from dispatch would hide exactly the
latency the stall caused — the classic coordinated-omission error, which makes
an overloaded system look fine. The generator separately tracks its own
dispatch lag and warns when it is large, because at that point the *generator*
has become a bottleneck and the run should be discarded rather than reported.

The backend is an injected async callable, so the same generator drives a mock,
an in-process engine, or a remote HTTP endpoint, and the numbers stay comparable
across all three.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import platform
import random
import socket
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bench.metrics import DEFAULT_PERCENTILES, SLO, RequestRecord, drop_warmup, summarize

logger = logging.getLogger(__name__)

__all__ = [
    "BackendFn",
    "MockBackend",
    "PromptRequest",
    "build_result",
    "collect_environment",
    "poisson_offsets",
    "run_closed_loop",
    "run_open_loop",
    "write_result",
]


@dataclass(frozen=True)
class PromptRequest:
    """One unit of work handed to a backend.

    Attributes:
        prompt: The prompt text.
        max_tokens: Generation cap.
        prompt_tokens: Prompt length in tokens, or ``None`` when no tokenizer
            was available to count it. Left as ``None`` rather than estimated —
            a guessed token count would flow straight into a reported
            prompt-throughput number.
        request_id: Stable identifier, useful when correlating with server logs.
    """

    prompt: str
    max_tokens: int
    prompt_tokens: int | None = None
    request_id: str = ""


# A backend is anything callable with a PromptRequest that yields output tokens
# as they become available. The generator timestamps each yield, so a backend
# that buffers its output will report a TTFT equal to its E2E -- which is the
# truth about that backend, not a measurement bug.
BackendFn = Callable[[PromptRequest], AsyncIterator[str]]


class MockBackend:
    """A backend with no model behind it, for testing the generator itself.

    Emits ``max_tokens`` tokens with a fixed time to first token and a fixed
    inter-token interval. It also records the peak number of concurrent calls,
    which is what makes closed-loop concurrency assertions possible.
    """

    def __init__(self, ttft: float = 0.01, itl: float = 0.001, fail_every: int = 0) -> None:
        self.ttft = ttft
        self.itl = itl
        self.fail_every = fail_every
        self.in_flight = 0
        self.max_in_flight = 0
        self.num_calls = 0

    async def __call__(self, request: PromptRequest) -> AsyncIterator[str]:
        self.num_calls += 1
        call_index = self.num_calls
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if self.fail_every and call_index % self.fail_every == 0:
                raise RuntimeError("mock backend failure")
            await asyncio.sleep(self.ttft)
            for i in range(request.max_tokens):
                if i:
                    await asyncio.sleep(self.itl)
                yield f"t{i}"
        finally:
            self.in_flight -= 1


def poisson_offsets(rate: float, num_requests: int, rng: random.Random) -> list[float]:
    """Arrival offsets in seconds for a Poisson process of the given rate.

    Poisson arrivals have exponentially distributed gaps. The first request goes
    out at offset 0 and each subsequent gap is drawn from ``Exponential(rate)``,
    so the expected offset of request *n* is ``n / rate``.

    Pure and seeded, so a run's arrival schedule is reproducible and testable
    without waiting on a clock.

    Args:
        rate: Requests per second. Must be positive.
        num_requests: How many arrivals to schedule.
        rng: Seeded random source.

    Returns:
        Non-decreasing offsets, starting at 0.0.
    """
    if rate <= 0.0:
        raise ValueError(f"rate must be positive, got {rate}")
    offsets: list[float] = []
    t = 0.0
    for i in range(num_requests):
        if i:
            t += rng.expovariate(rate)
        offsets.append(t)
    return offsets


async def _run_one(
    backend: BackendFn,
    request: PromptRequest,
    arrival: float,
) -> RequestRecord:
    """Drive one request to completion, timestamping every token.

    Never raises on backend failure: a failed request becomes a record with an
    ``error`` set, so it is counted in the result file rather than vanishing and
    quietly improving the percentiles.
    """
    token_times: list[float] = []
    error: str | None = None
    try:
        async for _token in backend(request):
            token_times.append(time.perf_counter())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - the failure is data, not an event
        error = f"{type(exc).__name__}: {exc}"

    completed = error is None and bool(token_times)
    return RequestRecord(
        arrival=arrival,
        first_token=token_times[0] if token_times else None,
        tokens=token_times,
        # finish is the last token's timestamp, so E2E and the token stream
        # agree. A request that died mid-stream has no finish at all.
        finish=token_times[-1] if completed else None,
        prompt_tokens=request.prompt_tokens,
        output_tokens=len(token_times),
        error=error,
    )


@dataclass
class DispatchLag:
    """How far behind its own schedule the generator ran.

    If this is not small relative to the latencies being measured, the
    generator was the bottleneck and the run is not a measurement of the
    server.
    """

    max: float = 0.0
    mean: float = 0.0
    num_late: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"max": self.max, "mean": self.mean, "num_late": self.num_late}


async def run_open_loop(
    backend: BackendFn,
    prompts: Sequence[PromptRequest],
    *,
    rate: float,
    rng: random.Random,
    duration: float | None = None,
    lag_warn_threshold: float = 0.05,
) -> tuple[list[RequestRecord], DispatchLag]:
    """Issue requests on a Poisson schedule without waiting for responses.

    Args:
        backend: The system under test.
        prompts: Work to issue, in order.
        rate: Target requests per second.
        rng: Seeded random source for the arrival process.
        duration: Stop issuing after this many seconds. ``None`` means issue
            every prompt. In-flight requests are always awaited.
        lag_warn_threshold: Warn if mean dispatch lag exceeds this many seconds.

    Returns:
        The per-request records and the generator's own dispatch-lag summary.
    """
    offsets = poisson_offsets(rate, len(prompts), rng)
    start = time.perf_counter()
    tasks: list[asyncio.Task[RequestRecord]] = []
    lags: list[float] = []

    for prompt, offset in zip(prompts, offsets, strict=True):
        scheduled = start + offset
        now = time.perf_counter()
        if scheduled > now:
            await asyncio.sleep(scheduled - now)
        now = time.perf_counter()
        if duration is not None and now - start >= duration:
            break
        lags.append(max(0.0, now - scheduled))
        # arrival is the SCHEDULED time: a request the generator was late to
        # dispatch was still, from the workload's point of view, offered then.
        tasks.append(asyncio.create_task(_run_one(backend, prompt, arrival=scheduled)))

    records = list(await asyncio.gather(*tasks))

    lag = DispatchLag(
        max=max(lags) if lags else 0.0,
        mean=sum(lags) / len(lags) if lags else 0.0,
        num_late=sum(1 for value in lags if value > lag_warn_threshold),
    )
    if lag.mean > lag_warn_threshold:
        logger.warning(
            "Load generator fell behind its own schedule (mean lag %.3fs, max %.3fs over %d "
            "requests). The generator, not the server, may be the bottleneck; treat this run "
            "as suspect.",
            lag.mean,
            lag.max,
            len(lags),
        )
    return records, lag


async def run_closed_loop(
    backend: BackendFn,
    prompts: Sequence[PromptRequest],
    *,
    concurrency: int,
    duration: float | None = None,
) -> list[RequestRecord]:
    """Run ``concurrency`` workers, each issuing one request at a time.

    There is no queueing by construction, so TTFT here excludes the arrival
    queueing that open-loop mode measures. Offered load is whatever the server
    can absorb, which is why this mode isolates batch scaling but must not be
    used for a headline throughput claim.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be at least 1, got {concurrency}")

    queue: asyncio.Queue[PromptRequest] = asyncio.Queue()
    for prompt in prompts:
        queue.put_nowait(prompt)

    start = time.perf_counter()
    records: list[RequestRecord] = []

    async def worker() -> None:
        while True:
            if duration is not None and time.perf_counter() - start >= duration:
                return
            try:
                prompt = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            arrival = time.perf_counter()
            records.append(await _run_one(backend, prompt, arrival=arrival))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return records


def collect_environment() -> dict[str, Any]:
    """Record what the run happened on.

    Every field is either observed or ``None``. torch and CUDA details are
    probed through a lazy import so this module still works on a machine with
    no torch installed.
    """
    env: dict[str, Any] = {
        "host": socket.gethostname(),
        "timestamp": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": None,
        "cuda": None,
        "driver": None,
        "gpu": None,
    }
    try:
        import torch
    except ImportError:
        return env

    env["torch"] = torch.__version__
    env["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(0)
        try:
            # Present on CUDA builds only; absent on ROCm and older torch.
            env["driver"] = torch.cuda.get_device_properties(0).driver_version  # type: ignore[attr-defined]
        except AttributeError:
            env["driver"] = None
    return env


def build_result(
    records: Sequence[RequestRecord],
    *,
    config: dict[str, Any],
    workload: dict[str, Any],
    slo: SLO | None = None,
    warmup_requests: int = 0,
    warmup_seconds: float = 0.0,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the result JSON described in AGENTS.md §6.

    Every record is written out, warmup included, because the file is the raw
    evidence. The ``summary`` block is computed *after* the warmup window is
    dropped, and the warmup parameters are recorded alongside it so a reader can
    recompute the summary differently from the same file.
    """
    measured = drop_warmup(records, num_requests=warmup_requests, seconds=warmup_seconds)
    return {
        "config": config,
        "environment": environment if environment is not None else collect_environment(),
        "workload": {
            **workload,
            "warmup_requests": warmup_requests,
            "warmup_seconds": warmup_seconds,
            "num_requests_measured": len(measured),
        },
        "requests": [r.to_dict() for r in records],
        "summary": summarize(measured, slo=slo, percentiles=percentiles),
    }


def write_result(path: str | Path, result: dict[str, Any]) -> Path:
    """Write a result file, creating parent directories as needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return out


def synthetic_prompts(
    num_requests: int,
    *,
    max_tokens: int,
    words: int = 128,
) -> list[PromptRequest]:
    """Fixed-length placeholder prompts, for driving a mock backend.

    ``prompt_tokens`` is left unset: without a tokenizer this generator does not
    know the token count, and inventing one would put a fabricated number into
    a result file. Real workloads come from the dataset loader.
    """
    text = " ".join(["token"] * words)
    return [
        PromptRequest(prompt=text, max_tokens=max_tokens, request_id=f"synthetic-{i}")
        for i in range(num_requests)
    ]


def load_sharegpt(
    path: str | Path,
    num_requests: int,
    *,
    rng: random.Random,
    max_tokens: int,
) -> list[PromptRequest]:
    """Load prompts from a ShareGPT-format conversation dump.

    Takes the first human turn of each conversation as the prompt. ShareGPT is
    the standard choice because its length distribution is heavy-tailed and
    highly variable, and variance is precisely what a paged allocator exploits;
    a uniform-length synthetic workload would understate the effect.

    ``prompt_tokens`` is left as ``None`` — counting tokens requires the
    server's own tokenizer, and the backend that has one should report it.
    """
    entries = json.loads(Path(path).read_text())
    prompts: list[str] = []
    for entry in entries:
        turns = entry.get("conversations") or []
        first_human = next((t for t in turns if t.get("from") == "human"), None)
        if first_human and first_human.get("value"):
            prompts.append(first_human["value"])
    if not prompts:
        raise ValueError(f"no usable human turns found in {path}")
    rng.shuffle(prompts)
    # Cycle if the dataset is smaller than the requested request count, so a
    # long sweep does not silently run short.
    chosen = [prompts[i % len(prompts)] for i in range(num_requests)]
    return [
        PromptRequest(prompt=text, max_tokens=max_tokens, request_id=f"sharegpt-{i}")
        for i, text in enumerate(chosen)
    ]


def _build_backend(name: str, args: argparse.Namespace) -> BackendFn:
    """Resolve a backend by name.

    Only the mock exists in Phase 0. The HuggingFace baseline registers here
    next, and the engine and HTTP backends after that; the generator itself
    does not change when they do.
    """
    if name == "mock":
        return MockBackend(ttft=args.mock_ttft, itl=args.mock_itl)
    if name == "hf":
        # Imported lazily: this module must stay importable on a machine with
        # no torch, and the harness is tested there (AGENTS.md section 4).
        from bench.baseline_hf import BaselineConfig, HFBaselineBackend, load_model_and_tokenizer

        if not args.model:
            raise ValueError("--model is required for the hf backend")
        config = BaselineConfig(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            # Sequential is static batching with a batch of one, so the two
            # modes share a single timing path and stay comparable.
            max_batch_size=1 if args.baseline_mode == "sequential" else args.concurrency,
            batch_timeout=args.batch_timeout,
        )
        model, tokenizer = load_model_and_tokenizer(config)
        return HFBaselineBackend(model, tokenizer, config)
    raise ValueError(f"unknown backend {name!r} (available: mock, hf)")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PagedServe load generator: Poisson open-loop and closed-loop modes."
    )
    parser.add_argument("--backend", default="mock", help="Backend to drive (currently: mock).")
    parser.add_argument(
        "--mode",
        default="poisson",
        choices=("poisson", "closed"),
        help="poisson = open-loop arrivals at --rate; closed = --concurrency workers.",
    )
    parser.add_argument("--rate", type=float, default=1.0, help="Requests/sec in poisson mode.")
    parser.add_argument("--concurrency", type=int, default=1, help="Workers in closed mode.")
    parser.add_argument("--num-requests", type=int, default=64)
    parser.add_argument("--duration", type=float, default=None, help="Stop issuing after N sec.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0, help="Seeds the arrival process.")
    parser.add_argument("--dataset", default=None, help="Path to a ShareGPT-format JSON file.")
    parser.add_argument("--warmup-requests", type=int, default=0)
    parser.add_argument("--warmup-seconds", type=float, default=0.0)
    parser.add_argument("--slo-ttft", type=float, default=None, help="Goodput TTFT bound (sec).")
    parser.add_argument("--slo-tpot", type=float, default=None, help="Goodput TPOT bound (sec).")
    parser.add_argument("--slo-e2e", type=float, default=None, help="Goodput E2E bound (sec).")
    parser.add_argument("--output", default=None, help="Result JSON path.")
    parser.add_argument("--model", default=None, help="Model id or path for the hf backend.")
    parser.add_argument("--device", default=None, help="Override device (cuda/mps/cpu).")
    parser.add_argument("--dtype", default=None, help="Override dtype (bfloat16/float16/float32).")
    parser.add_argument(
        "--baseline-mode",
        default="static",
        choices=("sequential", "static"),
        help="hf backend: sequential runs one request at a time; static batches up to "
        "--concurrency requests together.",
    )
    parser.add_argument(
        "--batch-timeout",
        type=float,
        default=0.05,
        help="hf static batching: seconds to wait for a batch to fill.",
    )
    parser.add_argument("--mock-ttft", type=float, default=0.01)
    parser.add_argument("--mock-itl", type=float, default=0.001)
    return parser.parse_args(argv)


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    backend = _build_backend(args.backend, args)

    if args.dataset:
        prompts = load_sharegpt(
            args.dataset, args.num_requests, rng=rng, max_tokens=args.max_tokens
        )
        dataset_name = str(args.dataset)
    else:
        prompts = synthetic_prompts(args.num_requests, max_tokens=args.max_tokens)
        dataset_name = "synthetic"

    # A backend that owns a tokenizer can fill in the prompt lengths the dataset
    # loader refused to guess, which is what makes prompt throughput reportable.
    tokenizer = getattr(backend, "tokenizer", None)
    if tokenizer is not None:
        from bench.baseline_hf import tokenize_prompts

        prompts = tokenize_prompts(prompts, tokenizer)

    lag: DispatchLag | None = None
    try:
        if args.mode == "poisson":
            records, lag = await run_open_loop(
                backend, prompts, rate=args.rate, rng=rng, duration=args.duration
            )
        else:
            records = await run_closed_loop(
                backend, prompts, concurrency=args.concurrency, duration=args.duration
            )
    finally:
        aclose = getattr(backend, "aclose", None)
        if aclose is not None:
            await aclose()

    slo = SLO(ttft=args.slo_ttft, tpot=args.slo_tpot, e2e=args.slo_e2e)
    workload: dict[str, Any] = {
        "dataset": dataset_name,
        "arrival": "poisson" if args.mode == "poisson" else "closed_loop",
        "rate": args.rate if args.mode == "poisson" else None,
        "concurrency": args.concurrency if args.mode == "closed" else None,
        "num_requests": len(records),
        "max_tokens": args.max_tokens,
        "seed": args.seed,
    }
    if lag is not None:
        workload["dispatch_lag"] = lag.to_dict()

    # Evidence that a batching backend actually batched. Without this, a static
    # batching run that silently degraded to batches of one would be
    # indistinguishable from a sequential run in the result file.
    batch_sizes = getattr(backend, "batch_sizes", None)
    if batch_sizes:
        workload["batching"] = {
            "num_batches": len(batch_sizes),
            "mean_batch_size": sum(batch_sizes) / len(batch_sizes),
            "max_batch_size": max(batch_sizes),
        }

    return build_result(
        records,
        config={
            "backend": args.backend,
            "mode": args.mode,
            "backend_config": (backend.config.to_dict() if hasattr(backend, "config") else None),
            # The engine config dump lands here once an engine exists to dump.
            "engine": None,
        },
        workload=workload,
        slo=slo if not slo.is_empty() else None,
        warmup_requests=args.warmup_requests,
        warmup_seconds=args.warmup_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    result = asyncio.run(_main_async(args))

    if args.output:
        path = write_result(args.output, result)
        print(f"wrote {path}")
    else:
        print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
