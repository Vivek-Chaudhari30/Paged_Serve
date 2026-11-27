"""Drive any OpenAI-compatible server through the load generator.

Two measurements need this, and neither is possible in-process:

**The HTTP path.** Phase 6's number is the one a client would see, so it has to
include tokenization on the server, JSON serialization, SSE framing and the
event loop sitting between the engine and the socket. The in-process
``pagedserve`` backend measures the engine with all of that removed.

**The vLLM comparison.** ``vllm serve`` speaks the same protocol, so the same
generator, the same prompts, the same seeds and the same timestamping code can
drive both servers. A comparison where each side was measured by its own
harness is a comparison of harnesses.

Stdlib only, on purpose. The harness must stay importable on a laptop with
nothing installed (AGENTS.md section 4), and a client library would add a
runtime dependency to the code that produces the evidence. HTTP/1.1 with one
connection per request is a small, fully specified protocol; what is needed of
it fits in this file and is tested against a real socket.

What counts as a token
----------------------
Each SSE event whose choice carries non-empty text is timestamped as one token.
That is also what vLLM's own ``benchmark_serving`` does. It is exact when a
server sends one event per generated token, and it undercounts when a server
coalesces tokens into one event -- PagedServe's server does that when a token
ends mid-way through a UTF-8 character and its text has to wait for the next
one. The usage block, when the server sends one, carries the true count, and it
is recorded per run next to the event count so the two can be compared rather
than assumed equal.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit

from bench.loadgen import PromptRequest

logger = logging.getLogger(__name__)

__all__ = ["HTTPBackendConfig", "OpenAIHTTPBackend", "parse_sse_events"]


@dataclass(frozen=True)
class HTTPBackendConfig:
    """Everything about the client side of an HTTP run.

    Written into the result file's ``backend_config``: a number measured against
    a server is not reproducible unless the request it sent is.
    """

    base_url: str
    model: str
    endpoint: str = "completions"
    temperature: float = 0.0
    timeout: float = 600.0
    # vLLM's stream_options.include_usage: a final event with the true token
    # count. Servers that do not know the field ignore it.
    include_usage: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HTTPError(RuntimeError):
    """The server answered, but not with a stream."""


def parse_sse_events(buffer: bytearray) -> list[str]:
    """Remove every complete SSE event from ``buffer`` and return its data.

    An event ends at a blank line. Only ``data:`` fields are kept, joined with
    newlines as the SSE spec requires; comments and other fields are dropped.
    Whatever follows the last blank line is left in the buffer for the next
    read, because TCP delivers bytes, not events.
    """
    events: list[str] = []
    while True:
        # Servers may frame with \n\n or \r\n\r\n; take whichever comes first.
        candidates = [i for i in (buffer.find(b"\n\n"), buffer.find(b"\r\n\r\n")) if i >= 0]
        if not candidates:
            return events
        end = min(candidates)
        sep = 4 if buffer[end : end + 4] == b"\r\n\r\n" else 2
        raw = bytes(buffer[:end]).decode("utf-8")
        del buffer[: end + sep]
        data = [
            line[5:].removeprefix(" ")
            for line in raw.replace("\r\n", "\n").split("\n")
            if line.startswith("data:")
        ]
        if data:
            events.append("\n".join(data))


class _Body:
    """The response body as a byte stream, whatever its framing.

    A streaming response arrives chunked (uvicorn's choice when it has no
    content length), but a proxy or a server error may send a fixed length or
    simply close the connection. All three are handled so a framing difference
    between two servers cannot turn into a timing difference.
    """

    def __init__(self, reader: asyncio.StreamReader, headers: dict[str, str]) -> None:
        self._reader = reader
        self._chunked = "chunked" in headers.get("transfer-encoding", "").lower()
        length = headers.get("content-length")
        self._remaining = int(length) if length is not None and not self._chunked else None
        self._done = False

    async def read(self) -> bytes:
        """The next piece of body, or ``b""`` at the end."""
        if self._done:
            return b""
        if self._chunked:
            size_line = await self._reader.readline()
            if not size_line:
                self._done = True
                return b""
            size = int(size_line.split(b";", 1)[0].strip() or b"0", 16)
            if size == 0:
                # Trailers, then the blank line that ends the message.
                while (await self._reader.readline()) not in (b"\r\n", b"\n", b""):
                    pass
                self._done = True
                return b""
            data = await self._reader.readexactly(size)
            await self._reader.readline()  # the CRLF after every chunk
            return data
        if self._remaining is not None:
            if self._remaining <= 0:
                self._done = True
                return b""
            data = await self._reader.read(min(self._remaining, 65536))
            if not data:
                self._done = True
            self._remaining -= len(data)
            return data
        data = await self._reader.read(65536)
        if not data:
            self._done = True
        return data


class OpenAIHTTPBackend:
    """An async ``BackendFn`` that streams completions from an HTTP server.

    One TCP connection per request, closed when the response ends. Connection
    reuse would be faster, but it would also make the client a variable: two
    servers with different keep-alive behaviour would be measured with
    different amounts of handshake in their TTFT. A fresh connection is the same
    cost against every server.
    """

    def __init__(self, config: HTTPBackendConfig) -> None:
        if config.endpoint not in ("completions", "chat"):
            raise ValueError(f"endpoint must be 'completions' or 'chat', got {config.endpoint!r}")
        parts = urlsplit(config.base_url)
        if parts.scheme != "http":
            # TLS would put a handshake in every TTFT and is never what a
            # benchmark against a local server wants.
            raise ValueError(f"only http:// base URLs are supported, got {config.base_url!r}")
        self.config = config
        self.host = parts.hostname or "127.0.0.1"
        self.port = parts.port or 80
        prefix = parts.path.rstrip("/")
        self.path = (
            f"{prefix}/completions"
            if config.endpoint == "completions"
            else (f"{prefix}/chat/completions")
        )
        # Per-run evidence that the event count is (or is not) a token count.
        self.num_events = 0
        self.num_usage_tokens = 0
        self.num_requests_with_usage = 0
        self.tokenizer: Any = None

    def _payload(self, request: PromptRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": request.max_tokens,
            "temperature": self.config.temperature,
            "stream": True,
        }
        if self.config.include_usage:
            body["stream_options"] = {"include_usage": True}
        if self.config.endpoint == "completions":
            body["prompt"] = request.prompt
        else:
            body["messages"] = [{"role": "user", "content": request.prompt}]
        return body

    async def __call__(self, request: PromptRequest) -> AsyncIterator[str]:
        body = json.dumps(self._payload(request)).encode("utf-8")
        head = (
            f"POST {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            "Accept: text/event-stream\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")

        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(self.host, self.port), timeout=self.config.timeout
        )
        try:
            writer.write(head + body)
            await writer.drain()
            status, headers = await self._read_head(reader)
            stream = _Body(reader, headers)
            if status != 200:
                error = bytearray()
                while chunk := await stream.read():
                    error += chunk
                raise HTTPError(f"HTTP {status}: {bytes(error[:500]).decode('utf-8', 'replace')}")

            buffer = bytearray()
            while True:
                chunk = await asyncio.wait_for(stream.read(), timeout=self.config.timeout)
                if not chunk:
                    return
                buffer += chunk
                for data in parse_sse_events(buffer):
                    if data.strip() == "[DONE]":
                        return
                    text = self._text_of(json.loads(data))
                    if text:
                        self.num_events += 1
                        yield text
        finally:
            writer.close()
            # A server that has already gone away must not turn a finished
            # measurement into an error.
            with contextlib.suppress(ConnectionError, OSError):
                await writer.wait_closed()

    def _text_of(self, event: dict[str, Any]) -> str:
        usage = event.get("usage")
        if usage and usage.get("completion_tokens") is not None:
            self.num_usage_tokens += int(usage["completion_tokens"])
            self.num_requests_with_usage += 1
        if "error" in event:
            raise HTTPError(f"server error in stream: {event['error']}")
        choices = event.get("choices") or []
        if not choices:
            return ""
        choice = choices[0]
        if self.config.endpoint == "completions":
            return choice.get("text") or ""
        return (choice.get("delta") or {}).get("content") or ""

    async def _read_head(self, reader: asyncio.StreamReader) -> tuple[int, dict[str, str]]:
        status_line = await asyncio.wait_for(reader.readline(), timeout=self.config.timeout)
        if not status_line:
            raise HTTPError("connection closed before a response")
        parts = status_line.decode("latin-1").split(" ", 2)
        if len(parts) < 2 or not parts[0].startswith("HTTP/"):
            raise HTTPError(f"malformed status line {status_line!r}")
        headers: dict[str, str] = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, _, value = line.decode("latin-1").partition(":")
            headers[key.strip().lower()] = value.strip()
        return int(parts[1]), headers

    def stats(self) -> dict[str, Any]:
        """What the result file records about how tokens were counted."""
        return {
            "sse_text_events": self.num_events,
            "usage_completion_tokens": (
                self.num_usage_tokens if self.num_requests_with_usage else None
            ),
            "requests_with_usage": self.num_requests_with_usage,
        }


async def fetch_json(base_url: str, path: str, *, timeout: float = 10.0) -> Any:
    """GET a small JSON document, for recording what server a run was against."""
    parts = urlsplit(base_url)
    host, port = parts.hostname or "127.0.0.1", parts.port or 80
    full = f"{parts.path.rstrip('/')}{path}"
    reader, writer = await asyncio.wait_for(asyncio.open_connection(host, port), timeout=timeout)
    try:
        writer.write(
            f"GET {full} HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode()
        )
        await writer.drain()
        backend = OpenAIHTTPBackend(HTTPBackendConfig(base_url=base_url, model=""))
        status, headers = await backend._read_head(reader)
        stream = _Body(reader, headers)
        body = bytearray()
        while chunk := await asyncio.wait_for(stream.read(), timeout=timeout):
            body += chunk
        if status != 200:
            raise HTTPError(f"GET {full}: HTTP {status}")
        return json.loads(bytes(body))
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError, OSError):
            await writer.wait_closed()
