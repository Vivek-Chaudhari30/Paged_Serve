"""Drive the PagedServe engine through the load generator's backend interface.

Phase 0 built the harness around an injected async callable specifically so the
engine could be measured by exactly the same code as the HuggingFace baseline.
This is the piece that connects them, and without it none of the three stalled
sweeps can run.

The engine is synchronous and step-based: ``step()`` schedules, runs one forward
pass, and appends one token to every sequence in the batch. The load generator
is asynchronous and per-request. Bridging them means a background loop that
calls ``step()`` and fans the new tokens out to per-request queues.

Two things this must get right, both about not corrupting the measurement:

**Tokens are delivered as they are produced.** Each ``step()`` pushes that
step's token to each live request immediately. Buffering a request's output
until it finishes would report a TTFT equal to its E2E and erase the metric.

**The engine runs on a worker thread.** ``step()`` is a blocking call that holds
the GIL through a full forward pass. Running it on the event loop would stall
the load generator's arrival schedule, and the generator would then be measuring
itself rather than the server — the dispatch-lag warning exists to catch exactly
that, and this is the way to avoid triggering it.

This is also a rehearsal for Phase 6: the server's engine loop has the same
shape, minus the HTTP.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Any

from bench.loadgen import PromptRequest

logger = logging.getLogger(__name__)

__all__ = ["PagedServeBackend", "StaticEngineBackend"]

_DONE = object()


class PagedServeBackend:
    """An async ``BackendFn`` over ``LLMEngine``."""

    def __init__(self, engine: Any, *, idle_sleep: float = 0.0005) -> None:
        """
        Args:
            engine: A started or unstarted ``LLMEngine``.
            idle_sleep: How long the loop naps when the engine has no work.
                Without it an idle loop spins a core and starves the event loop
                of the very arrivals it is waiting for.
        """
        self.engine = engine
        self.idle_sleep = idle_sleep
        self._queues: dict[int, asyncio.Queue] = {}
        self._delivered: dict[int, int] = {}
        self._loop_task: asyncio.Task | None = None
        self._started = False
        self.steps = 0
        self.batch_sizes: list[int] = []

    async def __call__(self, request: PromptRequest) -> AsyncIterator[str]:
        if request.prompt_token_ids is None:
            raise ValueError(
                "the pagedserve backend needs pre-tokenized prompts; call "
                "bench.loadgen.tokenize_prompts before the run so tokenization "
                "stays off the measured path"
            )
        self._ensure_started()
        queue: asyncio.Queue = asyncio.Queue()
        sequence = self.engine.add_request(list(request.prompt_token_ids), request.max_tokens)
        self._queues[sequence.seq_id] = queue
        self._delivered[sequence.seq_id] = 0

        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._queues.pop(sequence.seq_id, None)
            self._delivered.pop(sequence.seq_id, None)

    def _ensure_started(self) -> None:
        if not self._started:
            self.engine.start()
            self._started = True
        if self._loop_task is None or self._loop_task.done():
            self._loop_task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._loop_task is not None and not self._loop_task.done():
            self._loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._loop_task
        self._loop_task = None

    async def _run(self) -> None:
        """Step the engine forever, fanning tokens out as they appear."""
        while True:
            if not self.engine.scheduler.has_work:
                await asyncio.sleep(self.idle_sleep)
                continue
            try:
                output = await asyncio.to_thread(self.engine.step)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - delivered to callers as data
                logger.exception("engine step failed")
                self._fail_all(exc)
                return

            self.steps += 1
            self.batch_sizes.append(output.batch_size)
            self._deliver(output.scheduled)
            self._close_finished(output.scheduled)

    def _deliver(self, scheduled) -> None:
        """Push each sequence's newly generated tokens to its consumer."""
        for sequence in scheduled:
            queue = self._queues.get(sequence.seq_id)
            if queue is None:
                continue
            already = self._delivered.get(sequence.seq_id, 0)
            for token in sequence.output_token_ids[already:]:
                queue.put_nowait(str(token))
            self._delivered[sequence.seq_id] = len(sequence.output_token_ids)

    def _close_finished(self, scheduled) -> None:
        from pagedserve.sequence import SequenceStatus

        for sequence in scheduled:
            if sequence.status is SequenceStatus.FINISHED:
                queue = self._queues.get(sequence.seq_id)
                if queue is not None:
                    queue.put_nowait(_DONE)

    def _fail_all(self, exc: BaseException) -> None:
        for queue in self._queues.values():
            queue.put_nowait(exc)


class StaticEngineBackend:
    """The static-batching arm, over the same engine.

    Three arms have to be measurable through one harness for the ablations in
    Phase 8 to isolate anything:

    ==========================  ====================  =========================
    Arm                         Backend               What it isolates
    ==========================  ====================  =========================
    contiguous + static         ``--no-paging``       the naive floor
    paged + static              ``--static-batching`` paging alone
    paged + continuous          (default)             iteration-level scheduling
    ==========================  ====================  =========================

    Comparing the first two isolates paging with batching held fixed; comparing
    the last two isolates scheduling with paging held fixed. A single flag that
    changed both at once would produce a number nobody could attribute.

    Batching here means a request waits for the batch to fill, then for the
    batch's *longest* member to finish. That queueing is head-of-line blocking,
    it is exactly what continuous batching removes, and it belongs in the TTFT
    this backend reports.
    """

    def __init__(self, engine: Any, *, max_batch_size: int, batch_timeout: float = 0.05):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self._incoming: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self.steps = 0
        self.batch_sizes: list[int] = []

    async def __call__(self, request: PromptRequest) -> AsyncIterator[str]:
        if request.prompt_token_ids is None:
            raise ValueError("the static engine backend needs pre-tokenized prompts")
        self._ensure_started()
        queue: asyncio.Queue = asyncio.Queue()
        await self._incoming.put((request, queue))
        while True:
            item = await queue.get()
            if item is _DONE:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    def _ensure_started(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            first = await self._incoming.get()
            batch = [first]
            deadline = loop.time() + self.batch_timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self._incoming.get(), remaining))
                except TimeoutError:
                    break
            await self._run_batch(batch, loop)

    async def _run_batch(self, batch, loop: asyncio.AbstractEventLoop) -> None:
        self.batch_sizes.append(len(batch))
        requests = [r for r, _ in batch]
        queues = [q for _, q in batch]

        def on_step(tokens: list[int | None]) -> None:
            # Runs on the worker thread; hand each token to the event loop as it
            # is produced rather than after the batch completes.
            for queue, token in zip(queues, tokens, strict=True):
                if token is not None:
                    loop.call_soon_threadsafe(queue.put_nowait, str(token))

        try:
            await asyncio.to_thread(
                self.engine.generate,
                [list(r.prompt_token_ids) for r in requests],
                max(r.max_tokens for r in requests),
                on_step=on_step,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - delivered as data
            logger.exception("static batch failed")
            for queue in queues:
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            return
        finally:
            self.steps += 1
        for queue in queues:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)
