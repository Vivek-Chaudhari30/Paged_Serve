"""FastAPI app: OpenAI-compatible routes over the engine.

The engine loop runs as a background task. HTTP handlers only enqueue a request
and consume its token stream — they never call ``step()`` themselves. That
separation is what lets one engine serve many concurrent requests: the loop
batches whatever is in flight, and the handlers are just readers.

Cancellation is a first-class concern, not an afterthought
----------------------------------------------------------
Under open-loop load with client timeouts, abandoned requests are normal. If a
disconnect does not free the sequence's blocks, every abandoned request leaks
its KV permanently and the server slowly strangles itself — admitting fewer and
fewer requests until it stops admitting any. The symptom is a server that
degrades over hours and looks fine on restart, which is about the worst
diagnostic signature a bug can have. So the streaming generator frees on *any*
exit: normal completion, client disconnect, or exception.

Stop strings live here
----------------------
Detecting them needs a detokenizer, and AGENTS.md §2.5 keeps the tokenizer out
of the engine's hot loop. The server already holds a tokenizer for encoding
prompts, so it watches the decoded text and aborts the sequence when a stop
string appears. The engine handles stop *tokens*, which need no decoding.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from pagedserve.engine import LLMEngine
from pagedserve.sequence import Sequence, SequenceStatus
from pagedserve.server.protocol import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
    UsageInfo,
    make_id,
)

logger = logging.getLogger(__name__)

__all__ = ["EngineRunner", "SSE_DONE", "create_app", "stream_tokens"]

# Terminates an SSE stream. OpenAI clients look for this exact sentinel.
SSE_DONE = "data: [DONE]\n\n"


class EngineRunner:
    """Owns the engine loop and fans tokens out to per-request queues.

    One background task drives ``step()``; each in-flight request has a queue it
    reads from. The same shape as the benchmark harness's backend, which is not
    a coincidence — a server and a load generator want exactly the same thing
    from an engine.
    """

    def __init__(self, engine: LLMEngine, tokenizer: Any, *, idle_sleep: float = 0.0005):
        self.engine = engine
        self.tokenizer = tokenizer
        self.idle_sleep = idle_sleep
        self._queues: dict[int, asyncio.Queue] = {}
        self._delivered: dict[int, int] = {}
        self._task: asyncio.Task | None = None
        self._started = False

    async def start(self) -> None:
        if not self._started:
            self.engine.start()
            self._started = True
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._task = None

    def submit(self, prompt_token_ids: list[int], sampling) -> tuple[Sequence, asyncio.Queue]:
        sequence = self.engine.add_request(prompt_token_ids, sampling.max_tokens, sampling=sampling)
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[sequence.seq_id] = queue
        self._delivered[sequence.seq_id] = 0
        return sequence, queue

    def abort(self, sequence: Sequence) -> None:
        """Stop a sequence and give its blocks back immediately.

        Called on client disconnect and after a stop string. Not deferred to
        some later sweep: the whole point is that abandoned work stops consuming
        memory the moment it is abandoned.
        """
        self._queues.pop(sequence.seq_id, None)
        self._delivered.pop(sequence.seq_id, None)
        scheduler = self.engine.scheduler
        if scheduler is None:
            return
        sequence.status = SequenceStatus.FINISHED
        sequence.finish_reason = sequence.finish_reason or "abort"
        self.engine.block_manager.free(sequence.seq_id)
        for queue in (scheduler.waiting, scheduler.swapped):
            if sequence in queue:
                queue.remove(sequence)
        if sequence in scheduler.running:
            scheduler.running.remove(sequence)

    async def _run(self) -> None:
        while True:
            scheduler = self.engine.scheduler
            if scheduler is None or not scheduler.has_work:
                await asyncio.sleep(self.idle_sleep)
                continue
            try:
                output = await asyncio.to_thread(self.engine.step)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - delivered to clients as data
                logger.exception("engine step failed")
                for queue in self._queues.values():
                    queue.put_nowait(exc)
                return
            for sequence in output.scheduled:
                queue = self._queues.get(sequence.seq_id)
                if queue is None:
                    continue
                already = self._delivered.get(sequence.seq_id, 0)
                for token in sequence.output_token_ids[already:]:
                    queue.put_nowait(token)
                self._delivered[sequence.seq_id] = len(sequence.output_token_ids)
                if sequence.status is SequenceStatus.FINISHED:
                    queue.put_nowait(None)


async def stream_tokens(
    runner: EngineRunner,
    sequence: Sequence,
    queue: asyncio.Queue,
    stop_strings: tuple[str, ...],
) -> AsyncIterator[tuple[str, str | None]]:
    """Yield ``(text_delta, finish_reason)`` until the sequence ends.

    Frees the sequence on **every** exit path. A client that disconnects
    mid-stream causes this generator to be closed, and without the ``finally``
    the sequence would keep its blocks for the lifetime of the server.
    """
    tokens: list[int] = []
    emitted = ""
    try:
        while True:
            item = await queue.get()
            if isinstance(item, BaseException):
                raise item
            if item is None:
                yield "", sequence.finish_reason or "stop"
                return

            tokens.append(item)
            # Decode the whole run each time rather than token by token:
            # multi-byte characters and byte-level BPE mean a single token can
            # be half a character, and decoding it alone yields a replacement
            # character that never resolves.
            text = runner.tokenizer.decode(tokens, skip_special_tokens=True)
            delta = text[len(emitted) :]
            emitted = text

            hit = next((s for s in stop_strings if s and s in text), None)
            if hit is not None:
                # Truncate at the stop string; the client asked not to see it.
                cut = text.index(hit)
                if cut > len(emitted) - len(delta):
                    yield text[len(emitted) - len(delta) : cut], None
                runner.abort(sequence)
                yield "", "stop"
                return

            if delta:
                yield delta, None
    finally:
        if sequence.status is not SequenceStatus.FINISHED:
            runner.abort(sequence)
        runner._queues.pop(sequence.seq_id, None)
        runner._delivered.pop(sequence.seq_id, None)


def create_app(engine: LLMEngine, tokenizer: Any, *, model_name: str = "pagedserve") -> FastAPI:
    """Build the app around an already-constructed engine."""
    runner = EngineRunner(engine, tokenizer)

    @contextlib.asynccontextmanager
    async def lifespan(_: FastAPI):
        # The engine loop starts with the app and stops with it. on_event is
        # deprecated in current FastAPI; a lifespan context also guarantees the
        # loop is torn down even if startup raised partway through.
        await runner.start()
        try:
            yield
        finally:
            await runner.stop()

    app = FastAPI(title="PagedServe", version="0.0.0", lifespan=lifespan)
    app.state.runner = runner
    app.state.model_name = model_name

    @app.get("/health")
    async def health() -> dict[str, Any]:
        manager = engine.block_manager
        return {
            "status": "ok",
            "model": model_name,
            # Free blocks are exposed deliberately: watching this number is how
            # a leak from an aborted request is caught.
            "free_blocks": manager.num_free_blocks if manager else None,
            "total_blocks": manager.num_blocks if manager else None,
        }

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=model_name)])

    async def _generate(
        request: Request,
        prompt_ids: list[int],
        sampling,
        stop_strings: tuple[str, ...],
    ) -> tuple[str, str]:
        """Run one request to completion, aborting if the client goes away."""
        sequence, queue = runner.submit(prompt_ids, sampling)
        pieces: list[str] = []
        reason = "length"
        stream = stream_tokens(runner, sequence, queue, stop_strings)
        try:
            async for delta, finish in stream:
                if await request.is_disconnected():
                    runner.abort(sequence)
                    break
                pieces.append(delta)
                if finish is not None:
                    reason = finish
                    break
        finally:
            await stream.aclose()
        return "".join(pieces), reason

    async def _sse(
        request: Request,
        prompt_ids: list[int],
        sampling,
        stop_strings: tuple[str, ...],
        make_chunk,
    ) -> AsyncIterator[str]:
        sequence, queue = runner.submit(prompt_ids, sampling)
        stream = stream_tokens(runner, sequence, queue, stop_strings)
        try:
            async for delta, finish in stream:
                if await request.is_disconnected():
                    runner.abort(sequence)
                    return
                if delta or finish:
                    yield f"data: {json.dumps(make_chunk(delta, finish))}\n\n"
                if finish is not None:
                    break
            yield SSE_DONE
        finally:
            await stream.aclose()

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, request: Request):
        prompts = body.prompts()
        if len(prompts) != 1:
            return JSONResponse(
                status_code=400,
                content=ErrorResponse.make(
                    "batched prompt lists are not supported; send one prompt per request"
                ).model_dump(),
            )
        prompt_ids = list(tokenizer(prompts[0]).input_ids)
        sampling = body.to_sampling_params(_eos_ids(engine))
        stops = body.stop_strings()
        response_id = make_id("cmpl")

        if body.stream:

            def chunk(delta: str, finish: str | None) -> dict[str, Any]:
                return {
                    "id": response_id,
                    "object": "text_completion",
                    "model": body.model,
                    "choices": [{"index": 0, "text": delta, "finish_reason": finish}],
                }

            return StreamingResponse(
                _sse(request, prompt_ids, sampling, stops, chunk),
                media_type="text/event-stream",
            )

        # n>1 shares the prompt's KV through block-table forking, so the extra
        # samples cost block-table entries rather than memory.
        choices = []
        completion_tokens = 0
        for index in range(body.n):
            text, reason = await _generate(request, prompt_ids, sampling, stops)
            completion_tokens += len(tokenizer(text).input_ids) if text else 0
            choices.append(CompletionChoice(index=index, text=text, finish_reason=reason))
        return CompletionResponse(
            id=response_id,
            model=body.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_ids) + completion_tokens,
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, request: Request):
        prompt_ids = _apply_chat_template(tokenizer, body.messages)
        sampling = body.to_sampling_params(_eos_ids(engine))
        stops = body.stop_strings()
        response_id = make_id("chatcmpl")

        if body.stream:

            def chunk(delta: str, finish: str | None) -> dict[str, Any]:
                return {
                    "id": response_id,
                    "object": "chat.completion.chunk",
                    "model": body.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta} if delta else {},
                            "finish_reason": finish,
                        }
                    ],
                }

            return StreamingResponse(
                _sse(request, prompt_ids, sampling, stops, chunk),
                media_type="text/event-stream",
            )

        choices = []
        completion_tokens = 0
        for index in range(body.n):
            text, reason = await _generate(request, prompt_ids, sampling, stops)
            completion_tokens += len(tokenizer(text).input_ids) if text else 0
            choices.append(
                ChatCompletionChoice(
                    index=index,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason=reason,
                )
            )
        return ChatCompletionResponse(
            id=response_id,
            model=body.model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=completion_tokens,
                total_tokens=len(prompt_ids) + completion_tokens,
            ),
        )

    return app


def _eos_ids(engine: LLMEngine) -> tuple[int, ...]:
    return engine.config.model.eos_token_ids


def _apply_chat_template(tokenizer: Any, messages: list[ChatMessage]) -> list[int]:
    """Render a conversation into token ids.

    Uses the checkpoint's own chat template when it has one. Falling back to a
    hand-rolled format would produce a prompt the model was never trained on,
    and the output would be poor in a way that looks like a model problem.
    """
    payload = [{"role": m.role, "content": m.content} for m in messages]
    if getattr(tokenizer, "chat_template", None):
        # Render to text, then tokenize separately. tokenize=True returns a
        # BatchEncoding on transformers v5 and a plain list of ints on v4, and
        # list() over a BatchEncoding iterates its KEYS -- producing a list of
        # strings that fails much later with "too many dimensions 'str'" from
        # inside the engine, nowhere near the cause.
        rendered = tokenizer.apply_chat_template(
            payload, add_generation_prompt=True, tokenize=False
        )
        return list(tokenizer(rendered).input_ids)
    rendered = "".join(f"{m.role}: {m.content}\n" for m in messages) + "assistant:"
    return tokenizer(rendered).input_ids
