"""HuggingFace ``generate()`` baselines — the floor every claim is measured against.

Two modes, both driven through the same async backend interface the rest of the
harness uses, so the baseline and the engine are measured by identical code:

**Sequential.** One request at a time. Honest, and unflattering to HuggingFace:
during decode the entire model's weights are read from HBM to produce a single
token, so the GPU is almost entirely idle. This is the floor.

**Static batching.** Requests are collected into a batch of up to
``max_batch_size``, left-padded to a common length, and generated together. The
batch runs until the *longest* request in it finishes, and no new request can
start until then. That is head-of-line blocking, and it is the specific problem
continuous batching exists to solve — which makes this, not the sequential mode,
the fair headline comparison.

Sequential is the degenerate case of static batching with ``max_batch_size=1``,
and is implemented that way so both modes share one timing path. A difference in
*how* the two were measured would confound the comparison between them.

Two deliberate choices, both about fairness
-------------------------------------------
**Tokens are streamed as they are produced,** pushed from the generation thread
onto the event loop, so the load generator timestamps them at the moment they
appear. Buffering the batch and delivering it at the end would report a TTFT
equal to E2E for every request and erase the measurement.

**Detokenization is excluded from the measured path.** This module yields token
ids, not text. AGENTS.md §2.5 bars tokenizer calls from the engine's hot loop,
so excluding them here too keeps the comparison symmetric and isolates
scheduling and memory management rather than tokenizer performance.

EOS counts as an output token. The engine must use the same rule or the
throughput comparison is skewed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Device and dtype resolution lives in pagedserve.config and nowhere else
# (AGENTS.md section 4.1): the baseline and the engine must agree on what
# hardware they are running on, or the comparison between them is not
# controlled. Re-exported here so callers of this module need not care.
from bench.loadgen import PromptRequest
from pagedserve.config import resolve_device, resolve_dtype

logger = logging.getLogger(__name__)

__all__ = [
    "BaselineConfig",
    "HFBaselineBackend",
    "count_output_tokens",
    "load_model_and_tokenizer",
    "resolve_device",
    "resolve_dtype",
    "tokenize_prompts",
]


@dataclass
class BaselineConfig:
    """Everything that must be recorded for a baseline run to be reproducible.

    Device and dtype are resolved, never hardcoded: this repo runs on macOS
    (CPU/MPS), on a T4 or P100 notebook, and on an A100. See AGENTS.md §4.
    """

    model: str
    device: str | None = None
    dtype: str | None = None
    max_batch_size: int = 1
    # How long to wait for a batch to fill before running it short-handed. Only
    # meaningful when max_batch_size > 1.
    batch_timeout: float = 0.05
    max_new_tokens_cap: int = 2048
    trust_remote_code: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "device": self.device,
            "dtype": self.dtype,
            "max_batch_size": self.max_batch_size,
            "batch_timeout": self.batch_timeout,
            "max_new_tokens_cap": self.max_new_tokens_cap,
        }


def load_model_and_tokenizer(config: BaselineConfig) -> tuple[Any, Any]:
    """Load the reference model and a left-padding tokenizer.

    Left padding is required for batched generation: the newest token must sit
    at the same index for every sequence in the batch, which is what lets the
    streaming callback read one column and get every sequence's latest token.
    """
    device = resolve_device(config.device)
    dtype = resolve_dtype(config.dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model, trust_remote_code=config.trust_remote_code
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model, dtype=dtype, trust_remote_code=config.trust_remote_code
    )
    model.to(device)
    model.eval()
    logger.info("loaded %s on %s with %s", config.model, device, dtype)
    return model, tokenizer


def tokenize_prompts(prompts: Sequence[PromptRequest], tokenizer: Any) -> list[PromptRequest]:
    """Fill in ``prompt_tokens`` now that a real tokenizer is available.

    The dataset loader leaves the count unknown rather than estimating it; this
    is where it becomes known, which is what makes prompt throughput reportable
    for a baseline run.
    """
    counts = [len(tokenizer(p.prompt).input_ids) for p in prompts]
    return [
        PromptRequest(
            prompt=p.prompt,
            max_tokens=p.max_tokens,
            prompt_tokens=n,
            request_id=p.request_id,
        )
        for p, n in zip(prompts, counts, strict=True)
    ]


def count_output_tokens(
    generated_ids: Sequence[int], eos_token_id: int | None, max_tokens: int
) -> int:
    """How many tokens a sequence really produced, EOS included.

    ``generate()`` pads finished sequences out to the batch's longest run, so
    the raw tail length overcounts. Truncating at the first EOS (and counting
    that EOS) is what makes output-token throughput mean the same thing for the
    baseline and for the engine.

    Pure, so it can be tested without a model.
    """
    limit = min(len(generated_ids), max_tokens)
    for i in range(limit):
        if eos_token_id is not None and generated_ids[i] == eos_token_id:
            return i + 1
    return limit


@dataclass
class _Pending:
    """One in-flight request, plus the channel its tokens are delivered on."""

    request: PromptRequest
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)


_DONE = object()


class _StreamingCriteria(StoppingCriteria):
    """Per-step hook that streams each sequence's newest token to its consumer.

    ``generate()`` calls this once per decode step, after the step's token has
    been appended, with the whole batch. That makes it the one place where a
    token's production time is actually known. Timestamping happens on the load
    generator side, when the token is yielded, so every backend in this repo is
    timed by exactly the same code.

    This never stops generation; it returns all-False and lets ``generate()``
    handle real stopping. It does track per-sequence completion so a request
    that finished early stops receiving tokens while the batch grinds on — the
    visible signature of head-of-line blocking.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        pendings: Sequence[_Pending],
        eos_token_id: int | None,
    ) -> None:
        self._loop = loop
        self._pendings = list(pendings)
        self._eos = eos_token_id
        self.counts = [0] * len(pendings)
        self.finished = [False] * len(pendings)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor, **kwargs) -> torch.Tensor:
        # One host transfer per step for the whole batch, not one per sequence.
        # A streaming server pays this too; it is not an artefact of measuring.
        newest = input_ids[:, -1].tolist()
        for i, token_id in enumerate(newest):
            if self.finished[i]:
                continue
            self.counts[i] += 1
            self._emit(i, str(token_id))
            if token_id == self._eos or self.counts[i] >= self._pendings[i].request.max_tokens:
                self._finish(i)
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    def _emit(self, index: int, token: str) -> None:
        queue = self._pendings[index].queue
        self._loop.call_soon_threadsafe(queue.put_nowait, token)

    def _finish(self, index: int) -> None:
        self.finished[index] = True
        queue = self._pendings[index].queue
        self._loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    def finish_all(self) -> None:
        """Close out anything still open once ``generate()`` has returned."""
        for i in range(len(self._pendings)):
            if not self.finished[i]:
                self._finish(i)

    def fail_all(self, exc: BaseException) -> None:
        for i, pending in enumerate(self._pendings):
            if not self.finished[i]:
                self.finished[i] = True
                self._loop.call_soon_threadsafe(pending.queue.put_nowait, exc)


class HFBaselineBackend:
    """A ``BackendFn`` over HuggingFace ``generate()``.

    Requests queue up; a background task forms batches of up to
    ``max_batch_size`` and runs each one to completion in a worker thread. While
    a batch runs, arriving requests wait — that queueing delay is real, it is
    what static batching costs, and it belongs in TTFT.
    """

    def __init__(self, model: Any, tokenizer: Any, config: BaselineConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = resolve_device(config.device)
        self._incoming: asyncio.Queue[_Pending] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.batches_run = 0
        self.batch_sizes: list[int] = []

    async def __call__(self, request: PromptRequest) -> AsyncIterator[str]:
        self._ensure_started()
        pending = _Pending(request=request)
        await self._incoming.put(pending)
        while True:
            item = await pending.queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item  # type: ignore[misc]

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._batch_loop())

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _batch_loop(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._incoming.get()
            batch = [first]
            # Wait briefly for more arrivals, but only up to max_batch_size.
            # Running short-handed after the timeout is what a real static
            # batcher does; blocking forever for a full batch would deadlock
            # the tail of a run.
            deadline = loop.time() + self.config.batch_timeout
            while len(batch) < self.config.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._incoming.get(), remaining))
                except TimeoutError:
                    break
            await self._run_batch(batch, loop)

    async def _run_batch(self, batch: list[_Pending], loop: asyncio.AbstractEventLoop) -> None:
        self.batches_run += 1
        self.batch_sizes.append(len(batch))
        criteria = _StreamingCriteria(loop, batch, self.tokenizer.eos_token_id)
        try:
            # to_thread keeps the event loop free, so open-loop arrivals stay on
            # schedule while the GPU is busy. Without this the generator's own
            # dispatch lag would contaminate every measurement.
            await asyncio.to_thread(self._generate_sync, batch, criteria)
        except Exception as exc:  # noqa: BLE001 - delivered to the caller as data
            logger.exception("batch of %d failed", len(batch))
            criteria.fail_all(exc)
        finally:
            criteria.finish_all()

    def _generate_sync(self, batch: list[_Pending], criteria: _StreamingCriteria) -> None:
        """Run one batch. Executes on a worker thread, not the event loop."""
        prompts = [p.request.prompt for p in batch]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        # A static batch runs until its longest member is done. Every other
        # request in it holds memory and burns compute on padding until then.
        max_new = min(max(p.request.max_tokens for p in batch), self.config.max_new_tokens_cap)
        # A checkpoint's generation_config.json can carry sampling defaults --
        # Qwen2.5 ships repetition_penalty=1.1 -- and generate() applies its
        # logits processors even when do_sample is False. Left alone, the
        # baseline would run a penalty our engine does not, producing different
        # tokens and quietly making the comparison unfair in both directions.
        # Building the config from scratch replaces those defaults rather than
        # layering overrides on top of them.
        generation_config = GenerationConfig(
            max_new_tokens=max_new,
            do_sample=False,  # greedy: the golden test needs determinism
            repetition_penalty=1.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            output = self.model.generate(
                **encoded,
                generation_config=generation_config,
                stopping_criteria=StoppingCriteriaList([criteria]),
            )
        self._verify_streamed_counts(batch, criteria, output, encoded["input_ids"].shape[1])

    def _verify_streamed_counts(
        self,
        batch: list[_Pending],
        criteria: _StreamingCriteria,
        output: torch.Tensor,
        prompt_len: int,
    ) -> None:
        """Cross-check the streamed token counts against what generate() produced.

        The streaming hook is the only thing standing between this baseline and
        a wrong output-token count, and a wrong count silently corrupts every
        throughput number computed from it. Recounting the returned sequences
        independently and comparing costs one host transfer per batch and turns
        a silent measurement error into a log line.
        """
        generated = output[:, prompt_len:].tolist()
        for i, pending in enumerate(batch):
            expected = count_output_tokens(
                generated[i], self.tokenizer.eos_token_id, pending.request.max_tokens
            )
            if expected != criteria.counts[i]:
                logger.warning(
                    "request %s: streamed %d tokens but generate() produced %d. "
                    "Output-token throughput from this run is not trustworthy.",
                    pending.request.request_id or i,
                    criteria.counts[i],
                    expected,
                )
