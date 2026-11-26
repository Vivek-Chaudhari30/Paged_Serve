"""bench/http_backend.py against a real socket.

The server here is a few lines of asyncio that speaks the wire format uvicorn
and vLLM speak -- chunked transfer encoding, SSE events, a usage event and
``[DONE]`` -- so the client's framing, parsing and token counting are tested
without a model. The end-to-end run against the real PagedServe server is in
the GPU session checklist, where a model is present.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from bench.http_backend import (
    HTTPBackendConfig,
    OpenAIHTTPBackend,
    fetch_json,
    parse_sse_events,
)
from bench.loadgen import PromptRequest, run_closed_loop


def test_sse_parser_keeps_partial_events_for_the_next_read():
    buf = bytearray(b'data: {"a": 1}\n\ndata: {"b"')
    assert parse_sse_events(buf) == ['{"a": 1}']
    assert buf == bytearray(b'data: {"b"')
    buf += b": 2}\r\n\r\n: comment\n\n"
    assert parse_sse_events(buf) == ['{"b": 2}']
    assert buf == bytearray()


def test_sse_parser_joins_multiline_data():
    buf = bytearray(b"data: one\ndata: two\nevent: x\n\n")
    assert parse_sse_events(buf) == ["one\ntwo"]


def _completion_events(n: int, *, usage: bool, chat: bool) -> list[str]:
    events = []
    for i in range(n):
        choice = {"index": 0, "finish_reason": None}
        if chat:
            choice["delta"] = {"content": f"t{i}"}
        else:
            choice["text"] = f"t{i}"
        events.append(json.dumps({"choices": [choice]}))
    if usage:
        events.append(json.dumps({"choices": [], "usage": {"completion_tokens": n}}))
    events.append("[DONE]")
    return events


async def _serve(handler):
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, f"http://127.0.0.1:{port}/v1"


def _sse_handler(*, usage: bool = True, status: int = 200, split: bool = True):
    seen: list[dict] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        head = await reader.readuntil(b"\r\n\r\n")
        length = int(
            next(
                line.split(b":")[1]
                for line in head.split(b"\r\n")
                if line.lower().startswith(b"content-length")
            )
        )
        request = json.loads(await reader.readexactly(length))
        request["_path"] = head.split(b" ")[1].decode()
        seen.append(request)
        if status != 200:
            body = b'{"error": "nope"}'
            writer.write(
                f"HTTP/1.1 {status} Bad\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
            )
            await writer.drain()
            writer.close()
            return
        writer.write(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n")
        chat = request["_path"].endswith("/chat/completions")
        for event in _completion_events(request["max_tokens"], usage=usage, chat=chat):
            payload = f"data: {event}\n\n".encode()
            # Split events across chunk boundaries, as TCP and proxies do.
            pieces = [payload[:7], payload[7:]] if split else [payload]
            for piece in pieces:
                writer.write(f"{len(piece):x}\r\n".encode() + piece + b"\r\n")
                await writer.drain()
                await asyncio.sleep(0)
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()

    return handle, seen


def _drive(handler, *, endpoint="completions", n=3, concurrency=2, max_tokens=4):
    async def go():
        server, url = await _serve(handler)
        try:
            backend = OpenAIHTTPBackend(
                HTTPBackendConfig(base_url=url, model="m", endpoint=endpoint)
            )
            reqs = [PromptRequest(prompt=f"p{i}", max_tokens=max_tokens) for i in range(n)]
            records = await run_closed_loop(backend, reqs, concurrency=concurrency)
            return backend, records
        finally:
            server.close()
            await server.wait_closed()

    return asyncio.run(go())


def test_streams_and_counts_one_token_per_text_event():
    handler, seen = _sse_handler()
    backend, records = _drive(handler, n=3, max_tokens=4)
    assert all(r.succeeded and r.output_tokens == 4 for r in records)
    assert all(r.ttft < r.e2e for r in records)
    stats = backend.stats()
    assert stats["sse_text_events"] == 12
    assert stats["usage_completion_tokens"] == 12
    assert seen[0]["stream"] is True and seen[0]["temperature"] == 0.0
    assert seen[0]["prompt"] == "p0" and seen[0]["_path"] == "/v1/completions"


def test_chat_endpoint_reads_deltas():
    handler, seen = _sse_handler(usage=False)
    backend, records = _drive(handler, endpoint="chat", n=1, max_tokens=3)
    assert records[0].output_tokens == 3
    assert seen[0]["messages"] == [{"role": "user", "content": "p0"}]
    assert backend.stats()["usage_completion_tokens"] is None


def test_http_error_becomes_a_failed_record_not_a_crash():
    handler, _ = _sse_handler(status=400)
    _, records = _drive(handler, n=2)
    assert all(not r.succeeded for r in records)
    assert all("HTTP 400" in (r.error or "") for r in records)


def test_rejects_https_and_unknown_endpoints():
    with pytest.raises(ValueError):
        OpenAIHTTPBackend(HTTPBackendConfig(base_url="https://x/v1", model="m"))
    with pytest.raises(ValueError):
        OpenAIHTTPBackend(HTTPBackendConfig(base_url="http://x/v1", model="m", endpoint="embed"))


def test_fetch_json_reads_a_content_length_body():
    async def handle(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        body = b'{"data": [{"id": "m"}]}'
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(body) + body)
        await writer.drain()
        writer.close()

    async def go():
        server, url = await _serve(handle)
        try:
            return await fetch_json(url, "/models")
        finally:
            server.close()
            await server.wait_closed()

    assert asyncio.run(go()) == {"data": [{"id": "m"}]}
