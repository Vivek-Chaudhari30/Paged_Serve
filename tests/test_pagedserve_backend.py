"""Tests for driving the engine through the load generator's backend interface.

Uses a tiny random-init Llama on CPU. The weights are nonsense, so nothing here
says anything about output quality — what is being tested is that the engine can
be measured at all, and measured honestly.
"""

from __future__ import annotations

import asyncio
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bench.loadgen import (  # noqa: E402
    PromptRequest,
    run_closed_loop,
    run_open_loop,
    tokenize_prompts,
)
from bench.pagedserve_backend import PagedServeBackend, StaticEngineBackend  # noqa: E402
from pagedserve.config import CacheConfig, SchedulerConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402


class FakeTokenizer:
    """Whitespace tokenizer, so the suite needs no network and no vocab file."""

    def __call__(self, text: str):
        ids = [4 + (abs(hash(word)) % 8) for word in text.split()]
        return type("Encoded", (), {"input_ids": ids})()


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    from transformers import LlamaConfig, LlamaForCausalLM

    path = tmp_path_factory.mktemp("tiny-llama")
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        eos_token_id=2,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    LlamaForCausalLM(config).eval().save_pretrained(path)
    return str(path)


def build_engine(model_path: str, backend: str = "gather", **kwargs) -> LLMEngine:
    return LLMEngine.from_pretrained(
        model_path,
        device="cpu",
        dtype="float32",
        cache=CacheConfig(max_seq_len=128, max_num_seqs=8, block_size=8, num_blocks_override=64),
        scheduler=SchedulerConfig(max_num_batched_tokens=512, max_num_seqs=8, **kwargs),
        attn_backend=backend,
        debug_invariants=True,
    )


def prompts(n: int, max_tokens: int = 4) -> list[PromptRequest]:
    raw = [
        PromptRequest(prompt=f"alpha beta gamma {i}", max_tokens=max_tokens, request_id=f"r{i}")
        for i in range(n)
    ]
    return tokenize_prompts(raw, FakeTokenizer())


class TestTokenizePrompts:
    def test_fills_ids_and_count_together(self):
        [request] = tokenize_prompts(
            [PromptRequest(prompt="alpha beta gamma", max_tokens=4)], FakeTokenizer()
        )
        assert request.prompt_token_ids is not None
        assert request.prompt_tokens == len(request.prompt_token_ids) == 3

    def test_leaves_the_original_untouched(self):
        original = PromptRequest(prompt="alpha", max_tokens=4)
        tokenize_prompts([original], FakeTokenizer())
        assert original.prompt_token_ids is None


class TestPagedServeBackend:
    def test_drives_requests_to_completion(self, tiny_model):
        backend = PagedServeBackend(build_engine(tiny_model))
        records = asyncio.run(run_closed_loop(backend, prompts(4), concurrency=4))
        asyncio.run(backend.aclose())
        assert len(records) == 4
        assert all(r.succeeded for r in records)
        assert all(0 < r.output_tokens <= 4 for r in records)

    def test_refuses_untokenized_prompts(self, tiny_model):
        """Tokenizing inside the backend would charge the engine for work the
        HuggingFace baseline is not charged for."""
        backend = PagedServeBackend(build_engine(tiny_model))
        raw = [PromptRequest(prompt="alpha beta", max_tokens=4)]
        records = asyncio.run(run_closed_loop(backend, raw, concurrency=1))
        asyncio.run(backend.aclose())
        assert records[0].error is not None
        assert "pre-tokenized" in records[0].error

    def test_streams_tokens_rather_than_buffering(self, tiny_model):
        """Buffering until completion would report TTFT equal to E2E."""
        backend = PagedServeBackend(build_engine(tiny_model))
        records = asyncio.run(run_closed_loop(backend, prompts(1, max_tokens=6), concurrency=1))
        asyncio.run(backend.aclose())
        record = records[0]
        if record.output_tokens < 2:
            pytest.skip("random weights stopped immediately")
        assert len(set(record.tokens)) == len(record.tokens)
        assert record.ttft < record.e2e

    def test_matches_the_engine_driven_directly(self, tiny_model):
        """The harness must not change what the engine produces."""
        requests = prompts(3, max_tokens=5)
        expected = [
            s.output_token_ids
            for s in build_engine(tiny_model).generate_continuous(
                [list(r.prompt_token_ids) for r in requests], max_tokens=5
            )
        ]
        backend = PagedServeBackend(build_engine(tiny_model))

        async def drive():
            out = []
            for request in requests:
                out.append([int(t) async for t in backend(request)])
            return out

        got = asyncio.run(drive())
        asyncio.run(backend.aclose())
        assert got == expected

    def test_open_loop_arrivals_work_through_the_engine(self, tiny_model):
        backend = PagedServeBackend(build_engine(tiny_model))
        records, lag = asyncio.run(
            run_open_loop(backend, prompts(6), rate=50.0, rng=random.Random(0))
        )
        asyncio.run(backend.aclose())
        assert len(records) == 6
        assert all(r.succeeded for r in records)
        assert lag.mean >= 0.0

    def test_reports_scheduler_activity(self, tiny_model):
        backend = PagedServeBackend(build_engine(tiny_model))
        asyncio.run(run_closed_loop(backend, prompts(4), concurrency=4))
        asyncio.run(backend.aclose())
        assert backend.steps > 0
        assert backend.batch_sizes


class TestStaticEngineBackend:
    def test_batches_and_completes(self, tiny_model):
        backend = StaticEngineBackend(build_engine(tiny_model), max_batch_size=4)
        records = asyncio.run(run_closed_loop(backend, prompts(4), concurrency=4))
        asyncio.run(backend.aclose())
        assert len(records) == 4
        assert all(r.succeeded for r in records)
        assert max(backend.batch_sizes) > 1

    def test_works_with_the_contiguous_ablation_arm(self, tiny_model):
        """--no-paging must be measurable, not just configurable.

        The contiguous backend cannot do continuous batching at all -- its
        layout assumes lockstep -- so the static arm is what makes the Phase 1
        cache benchmarkable through the same harness.
        """
        engine = build_engine(tiny_model, backend="contiguous")
        assert not engine.is_paged
        backend = StaticEngineBackend(engine, max_batch_size=4)
        records = asyncio.run(run_closed_loop(backend, prompts(4), concurrency=4))
        asyncio.run(backend.aclose())
        assert all(r.succeeded for r in records)

    def test_streams_per_step(self, tiny_model):
        backend = StaticEngineBackend(build_engine(tiny_model), max_batch_size=2)
        records = asyncio.run(run_closed_loop(backend, prompts(2, max_tokens=6), concurrency=2))
        asyncio.run(backend.aclose())
        for record in records:
            if record.output_tokens >= 2:
                assert record.ttft < record.e2e

    def test_paged_and_contiguous_agree_under_static_batching(self, tiny_model):
        """Paging is a memory change, not a semantic one."""
        requests = prompts(3, max_tokens=5)
        ids = [list(r.prompt_token_ids) for r in requests]
        paged = build_engine(tiny_model, backend="gather").generate(ids, max_tokens=5)
        dense = build_engine(tiny_model, backend="contiguous").generate(ids, max_tokens=5)
        assert paged.token_ids == dense.token_ids


class TestOnStepCallback:
    def test_reports_one_token_per_sequence_per_step(self, tiny_model):
        engine = build_engine(tiny_model)
        seen: list[list[int | None]] = []
        engine.generate([[4, 5, 6], [4, 5]], max_tokens=4, on_step=seen.append)
        assert seen
        assert all(len(step) == 2 for step in seen)

    def test_finished_sequences_report_none(self, tiny_model):
        """A sequence that stopped must not appear to keep producing tokens."""
        engine = build_engine(tiny_model)
        seen: list[list[int | None]] = []
        out = engine.generate(
            [[4, 5, 6], [4, 5]], max_tokens=6, eos_token_ids=(), on_step=seen.append
        )
        for index, tokens in enumerate(out.token_ids):
            emitted = sum(1 for step in seen if step[index] is not None)
            # Every emitted token is accounted for; the callback never invents
            # tokens for a sequence that has already retired.
            assert emitted <= len(tokens)
