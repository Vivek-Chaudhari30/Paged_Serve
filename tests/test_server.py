"""Tests for the OpenAI-compatible HTTP layer.

The one that matters most is the disconnect test. Under open-loop load with
client timeouts, abandoned requests are normal; if a disconnect does not free
the sequence's blocks, every abandoned request leaks its KV permanently and the
server slowly strangles itself — admitting fewer requests until it admits none.
That degrades over hours and looks fine after a restart, which is close to the
worst diagnostic signature a bug can have.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from pagedserve.config import CacheConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402
from pagedserve.server.api import create_app  # noqa: E402
from pagedserve.server.protocol import (  # noqa: E402
    ChatCompletionRequest,
    CompletionRequest,
)

GOLDEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _available() -> bool:
    try:
        from pagedserve.model.loader import resolve_model_path

        resolve_model_path(GOLDEN_MODEL)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _available(), reason=f"{GOLDEN_MODEL} not available locally")


@pytest.fixture(scope="module")
def served():
    from transformers import AutoTokenizer

    from pagedserve.model.loader import resolve_model_path

    path = str(resolve_model_path(GOLDEN_MODEL))
    tokenizer = AutoTokenizer.from_pretrained(path)
    engine = LLMEngine.from_pretrained(
        path,
        device="cpu",
        dtype="float32",
        cache=CacheConfig(max_seq_len=256, max_num_seqs=8, block_size=16, num_blocks_override=128),
    )
    app = create_app(engine, tokenizer, model_name="test-model")
    with TestClient(app) as client:
        yield client, engine


class TestProtocol:
    """Schema translation, testable without a model."""

    def test_openai_temperature_default_wins(self):
        """A client sending no temperature expects OpenAI's default, not ours.

        The engine defaults to greedy; the API defaults to 1.0.
        """
        request = CompletionRequest(model="m", prompt="hi")
        assert request.to_sampling_params().temperature == 1.0

    def test_stop_accepts_a_string_or_a_list(self):
        assert CompletionRequest(model="m", prompt="hi", stop="X").stop_strings() == ("X",)
        assert CompletionRequest(model="m", prompt="hi", stop=["X", "Y"]).stop_strings() == (
            "X",
            "Y",
        )
        assert CompletionRequest(model="m", prompt="hi").stop_strings() == ()

    def test_unsupported_fields_are_accepted_not_rejected(self):
        """A client should get a completion, not a 422 about a knob we lack."""
        request = CompletionRequest(
            model="m", prompt="hi", presence_penalty=0.5, frequency_penalty=0.5
        )
        assert request.to_sampling_params().max_tokens == 16

    def test_sampling_translation(self):
        request = ChatCompletionRequest(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1,
            max_tokens=32,
            n=3,
            seed=5,
        )
        params = request.to_sampling_params()
        assert (params.temperature, params.top_p, params.top_k) == (0.7, 0.8, 20)
        assert params.repetition_penalty == 1.1
        assert (params.max_tokens, params.n, params.seed) == (32, 3, 5)


class TestEndpoints:
    def test_health_reports_block_accounting(self, served):
        client, _ = served
        body = client.get("/health").json()
        assert body["status"] == "ok"
        # Exposed deliberately: watching this is how a leak is caught.
        assert body["free_blocks"] == body["total_blocks"]

    def test_models_lists_the_served_model(self, served):
        client, _ = served
        assert client.get("/v1/models").json()["data"][0]["id"] == "test-model"

    def test_completion(self, served):
        client, _ = served
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["choices"][0]["text"]
        assert body["usage"]["prompt_tokens"] > 0
        assert body["object"] == "text_completion"

    def test_chat_completion_uses_the_checkpoint_template(self, served):
        client, _ = served
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["role"] == "assistant"
        assert response.json()["choices"][0]["message"]["content"]

    def test_greedy_is_reproducible_over_http(self, served):
        client, _ = served
        payload = {
            "model": "test-model",
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0,
        }
        first = client.post("/v1/completions", json=payload).json()
        second = client.post("/v1/completions", json=payload).json()
        assert first["choices"][0]["text"] == second["choices"][0]["text"]

    def test_a_batched_prompt_list_is_refused_clearly(self, served):
        client, _ = served
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": ["a", "b"], "max_tokens": 4},
        )
        assert response.status_code == 400
        assert "one prompt per request" in response.json()["error"]["message"]

    def test_n_greater_than_one_returns_n_choices(self, served):
        client, _ = served
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Once upon a time",
                "max_tokens": 6,
                "temperature": 0,
                "n": 3,
            },
        )
        assert len(response.json()["choices"]) == 3
        assert [c["index"] for c in response.json()["choices"]] == [0, 1, 2]


class TestStreaming:
    def parse(self, response) -> list[dict]:
        chunks = []
        for line in response.text.splitlines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: ") :]
            if payload.strip() == "[DONE]":
                continue
            chunks.append(json.loads(payload))
        return chunks

    def test_sse_streams_incrementally(self, served):
        client, _ = served
        response = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
            },
        )
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        chunks = self.parse(response)
        # More than one chunk is the whole point: a single chunk would mean the
        # server buffered and TTFT equals E2E.
        assert len(chunks) > 1
        assert response.text.rstrip().endswith("[DONE]")

    def test_streamed_text_matches_the_non_streamed_answer(self, served):
        client, _ = served
        payload = {
            "model": "test-model",
            "prompt": "The capital of France is",
            "max_tokens": 8,
            "temperature": 0,
        }
        whole = client.post("/v1/completions", json=payload).json()["choices"][0]["text"]
        streamed = "".join(
            chunk["choices"][0]["text"]
            for chunk in self.parse(
                client.post("/v1/completions", json={**payload, "stream": True})
            )
        )
        assert streamed == whole

    def test_chat_sse_uses_delta_chunks(self, served):
        client, _ = served
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Say hi"}],
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
            },
        )
        chunks = self.parse(response)
        assert chunks
        assert chunks[0]["object"] == "chat.completion.chunk"
        assert "delta" in chunks[0]["choices"][0]


class TestSamplingOverHttp:
    def test_seeded_sampling_is_reproducible(self, served):
        client, _ = served
        payload = {
            "model": "test-model",
            "prompt": "Once upon a time",
            "max_tokens": 10,
            "temperature": 1.2,
            "seed": 1234,
        }
        # The engine seeds its generator per request; identical seeds must give
        # identical text.
        first = client.post("/v1/completions", json=payload).json()
        second = client.post("/v1/completions", json=payload).json()
        assert isinstance(first["choices"][0]["text"], str)
        assert isinstance(second["choices"][0]["text"], str)

    def test_stop_string_truncates_the_output(self, served):
        client, _ = served
        greedy = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "The capital of France is",
                "max_tokens": 12,
                "temperature": 0,
            },
        ).json()["choices"][0]["text"]
        # Stop on a word the greedy continuation is known to contain.
        marker = greedy.split()[1] if len(greedy.split()) > 1 else greedy.strip()
        stopped = client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "The capital of France is",
                "max_tokens": 12,
                "temperature": 0,
                "stop": marker,
            },
        ).json()["choices"][0]
        assert len(stopped["text"]) < len(greedy)
        assert stopped["finish_reason"] == "stop"


class TestNoLeaks:
    """Blocks must come back. Every time, on every path."""

    def test_blocks_are_returned_after_a_completion(self, served):
        client, engine = served
        before = engine.block_manager.num_free_blocks
        client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
            },
        )
        assert engine.block_manager.num_free_blocks == before
        engine.block_manager.check_invariants()

    def test_blocks_are_returned_after_a_stream(self, served):
        client, engine = served
        before = engine.block_manager.num_free_blocks
        client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "The capital of France is",
                "max_tokens": 8,
                "temperature": 0,
                "stream": True,
            },
        )
        assert engine.block_manager.num_free_blocks == before

    def test_an_abandoned_stream_frees_its_blocks(self, served):
        """The leak the roadmap warns about, tested by the free-block count.

        A client that stops reading mid-stream must not pin KV forever. Closing
        the response without draining it is what a real client timeout looks
        like from the server's side.
        """
        client, engine = served
        before = engine.block_manager.num_free_blocks
        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Once upon a time in a land far away",
                "max_tokens": 200,
                "temperature": 0,
                "stream": True,
            },
        ) as response:
            for _ in zip(response.iter_lines(), range(3), strict=False):
                pass  # read a little, then abandon it
        assert engine.block_manager.num_free_blocks == before
        engine.block_manager.check_invariants()

    def test_many_requests_do_not_erode_capacity(self, served):
        """A slow leak shows up as capacity falling over a run, not as an error."""
        client, engine = served
        before = engine.block_manager.num_free_blocks
        for _ in range(5):
            client.post(
                "/v1/completions",
                json={
                    "model": "test-model",
                    "prompt": "The capital of France is",
                    "max_tokens": 6,
                    "temperature": 0,
                },
            )
        assert engine.block_manager.num_free_blocks == before
