"""Tests for bench/baseline_hf.py.

Runs against a tiny random-init Llama on the CPU. That is exactly what AGENTS.md
§4 says a Mac is for: the weights are nonsense, so no output here means anything
about quality, but the batching, streaming, and token-accounting logic is the
same code that will run on an A100.

Everything is built offline — a test suite that reaches for the HuggingFace Hub
fails on a plane and fails in CI.
"""

from __future__ import annotations

import asyncio
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from bench.baseline_hf import (  # noqa: E402
    BaselineConfig,
    HFBaselineBackend,
    count_output_tokens,
    resolve_device,
    resolve_dtype,
    tokenize_prompts,
)
from bench.loadgen import PromptRequest, run_closed_loop  # noqa: E402

VOCAB = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
    "alpha": 4,
    "beta": 5,
    "gamma": 6,
    "delta": 7,
}
EOS_ID = 2


@pytest.fixture(scope="module")
def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    backing = Tokenizer(models.WordLevel(VOCAB, unk_token="<unk>"))
    backing.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backing,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
    )
    tokenizer.padding_side = "left"
    return tokenizer


@pytest.fixture(scope="module")
def tiny_backend(tiny_tokenizer):
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=len(VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        eos_token_id=EOS_ID,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()

    def build(max_batch_size: int) -> HFBaselineBackend:
        baseline = BaselineConfig(
            model="<in-memory>", device="cpu", max_batch_size=max_batch_size, batch_timeout=0.05
        )
        return HFBaselineBackend(model, tiny_tokenizer, baseline)

    return build


def prompt(text: str = "alpha beta", max_tokens: int = 4, i: int = 0) -> PromptRequest:
    return PromptRequest(prompt=text, max_tokens=max_tokens, request_id=f"r{i}")


class TestCountOutputTokens:
    def test_truncates_at_the_first_eos_and_counts_it(self):
        # generate() pads finished sequences out to the batch's longest run, so
        # the raw tail overcounts. EOS itself counts as an output token.
        assert count_output_tokens([7, 8, EOS_ID, 0, 0], EOS_ID, 10) == 3

    def test_respects_the_request_cap(self):
        assert count_output_tokens([7, 8, 9, 10, 11], EOS_ID, 3) == 3

    def test_cap_wins_when_it_is_tighter_than_eos(self):
        assert count_output_tokens([7, 8, EOS_ID], EOS_ID, 2) == 2

    def test_no_eos_means_the_whole_run(self):
        assert count_output_tokens([7, 8, 9], EOS_ID, 10) == 3

    def test_eos_at_the_very_first_position(self):
        assert count_output_tokens([EOS_ID, 0, 0], EOS_ID, 10) == 1

    def test_tolerates_a_model_with_no_eos(self):
        assert count_output_tokens([7, 8, 9], None, 10) == 3

    def test_empty_generation(self):
        assert count_output_tokens([], EOS_ID, 10) == 0


class TestResolveDevice:
    def test_explicit_spec_wins(self):
        assert resolve_device("cpu") == torch.device("cpu")

    def test_falls_back_to_something_real(self):
        assert resolve_device().type in {"cuda", "mps", "cpu"}


class TestResolveDtype:
    def test_explicit_spec_wins(self):
        assert resolve_dtype("float16", torch.device("cpu")) == torch.float16

    def test_cpu_and_mps_get_float32(self):
        # bf16 on CPU is emulated and slow enough to make a baseline meaningless.
        assert resolve_dtype(None, torch.device("cpu")) == torch.float32
        assert resolve_dtype(None, torch.device("mps")) == torch.float32

    def test_cuda_prefers_bfloat16_when_supported(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        assert resolve_dtype(None, torch.device("cuda")) == torch.bfloat16

    def test_cuda_falls_back_to_float16_without_bf16(self, monkeypatch):
        # Turing (T4) and Volta (V100) have no bf16 at all. This fallback is a
        # correctness requirement, not a preference -- see AGENTS.md §4.2.
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        assert resolve_dtype(None, torch.device("cuda")) == torch.float16


class TestTokenizePrompts:
    def test_fills_in_previously_unknown_prompt_lengths(self, tiny_tokenizer):
        raw = [prompt("alpha beta gamma"), prompt("alpha")]
        assert all(p.prompt_tokens is None for p in raw)
        counted = tokenize_prompts(raw, tiny_tokenizer)
        assert [p.prompt_tokens for p in counted] == [3, 1]

    def test_preserves_the_other_fields(self, tiny_tokenizer):
        counted = tokenize_prompts([prompt("alpha", max_tokens=9, i=3)], tiny_tokenizer)
        assert counted[0].max_tokens == 9
        assert counted[0].request_id == "r3"
        assert counted[0].prompt == "alpha"


class TestSequentialMode:
    def test_generates_the_requested_number_of_tokens(self, tiny_backend):
        backend = tiny_backend(1)
        records = asyncio.run(run_closed_loop(backend, [prompt(max_tokens=4)], concurrency=1))
        asyncio.run(backend.aclose())
        assert records[0].succeeded
        # Random weights, so EOS is possible; never more than asked for.
        assert 0 < records[0].output_tokens <= 4

    def test_runs_one_batch_per_request(self, tiny_backend):
        backend = tiny_backend(1)
        prompts = [prompt(max_tokens=2, i=i) for i in range(3)]
        asyncio.run(run_closed_loop(backend, prompts, concurrency=1))
        asyncio.run(backend.aclose())
        assert backend.batches_run == 3
        assert backend.batch_sizes == [1, 1, 1]

    def test_tokens_are_streamed_not_delivered_in_one_lump(self, tiny_backend):
        # If the backend buffered the batch and flushed at the end, every token
        # would carry the same timestamp and TTFT would equal E2E, erasing the
        # measurement this whole harness exists to take.
        backend = tiny_backend(1)
        records = asyncio.run(run_closed_loop(backend, [prompt(max_tokens=6)], concurrency=1))
        asyncio.run(backend.aclose())
        record = records[0]
        if record.output_tokens < 2:
            pytest.skip("random weights emitted EOS immediately; nothing to compare")
        assert len(set(record.tokens)) == len(record.tokens)
        assert record.ttft < record.e2e


class TestStaticBatching:
    def test_forms_batches_up_to_the_configured_size(self, tiny_backend):
        backend = tiny_backend(4)
        prompts = [prompt(max_tokens=2, i=i) for i in range(8)]
        asyncio.run(run_closed_loop(backend, prompts, concurrency=8))
        asyncio.run(backend.aclose())
        assert max(backend.batch_sizes) > 1
        assert max(backend.batch_sizes) <= 4
        assert sum(backend.batch_sizes) == 8

    def test_every_request_in_a_batch_gets_its_own_tokens(self, tiny_backend):
        backend = tiny_backend(4)
        prompts = [prompt("alpha beta", max_tokens=3, i=i) for i in range(4)]
        records = asyncio.run(run_closed_loop(backend, prompts, concurrency=4))
        asyncio.run(backend.aclose())
        assert len(records) == 4
        assert all(r.succeeded for r in records)
        assert all(0 < r.output_tokens <= 3 for r in records)

    def test_a_short_request_stops_early_inside_a_long_batch(self, tiny_backend):
        # The signature of head-of-line blocking: the 2-token request stops
        # receiving tokens while the batch keeps running for the 8-token one.
        backend = tiny_backend(2)
        short = PromptRequest(prompt="alpha beta", max_tokens=2, request_id="short")
        long = PromptRequest(prompt="alpha beta", max_tokens=8, request_id="long")
        records = asyncio.run(run_closed_loop(backend, [short, long], concurrency=2))
        asyncio.run(backend.aclose())
        assert backend.batch_sizes == [2]
        by_tokens = sorted(r.output_tokens for r in records)
        assert by_tokens[0] <= 2

    def test_uneven_prompt_lengths_are_left_padded_without_error(self, tiny_backend):
        backend = tiny_backend(3)
        prompts = [
            PromptRequest(prompt="alpha", max_tokens=2, request_id="a"),
            PromptRequest(prompt="alpha beta gamma delta", max_tokens=2, request_id="b"),
            PromptRequest(prompt="alpha beta", max_tokens=2, request_id="c"),
        ]
        records = asyncio.run(run_closed_loop(backend, prompts, concurrency=3))
        asyncio.run(backend.aclose())
        assert all(r.succeeded for r in records)


class TestStreamedCountVerification:
    def test_warns_when_the_streamed_count_disagrees_with_generate(self, tiny_backend, caplog):
        # A wrong output-token count silently corrupts every throughput number
        # derived from it, so the guard must actually fire.
        import logging

        from bench.baseline_hf import _Pending, _StreamingCriteria

        backend = tiny_backend(1)

        async def check():
            loop = asyncio.get_running_loop()
            pending = _Pending(request=prompt(max_tokens=4))
            criteria = _StreamingCriteria(loop, [pending], EOS_ID)
            criteria.counts = [99]  # deliberately wrong
            # Prompt of length 2, then three generated tokens.
            output = torch.tensor([[4, 5, 6, 7, 4]])
            with caplog.at_level(logging.WARNING):
                backend._verify_streamed_counts([pending], criteria, output, 2)

        asyncio.run(check())
        assert "not trustworthy" in caplog.text

    def test_silent_when_the_counts_agree(self, tiny_backend, caplog):
        import logging

        from bench.baseline_hf import _Pending, _StreamingCriteria

        backend = tiny_backend(1)

        async def check():
            loop = asyncio.get_running_loop()
            pending = _Pending(request=prompt(max_tokens=4))
            criteria = _StreamingCriteria(loop, [pending], EOS_ID)
            criteria.counts = [3]
            output = torch.tensor([[4, 5, 6, 7, 4]])
            with caplog.at_level(logging.WARNING):
                backend._verify_streamed_counts([pending], criteria, output, 2)

        asyncio.run(check())
        assert caplog.text == ""


class TestLifecycle:
    def test_aclose_is_idempotent(self, tiny_backend):
        backend = tiny_backend(1)
        asyncio.run(run_closed_loop(backend, [prompt()], concurrency=1))

        async def close_twice():
            await backend.aclose()
            await backend.aclose()

        asyncio.run(close_twice())

    def test_backend_config_is_serializable_for_the_result_file(self, tiny_backend):
        backend = tiny_backend(2)
        as_dict = backend.config.to_dict()
        assert as_dict["max_batch_size"] == 2
        assert as_dict["device"] == "cpu"


class TestOpenLoopIntegration:
    def test_poisson_arrivals_through_the_real_backend(self, tiny_backend):
        from bench.loadgen import build_result, run_open_loop

        backend = tiny_backend(4)
        prompts = [prompt(max_tokens=2, i=i) for i in range(8)]
        records, lag = asyncio.run(
            run_open_loop(backend, prompts, rate=100.0, rng=random.Random(0))
        )
        asyncio.run(backend.aclose())
        result = build_result(
            records,
            config={"backend": "hf", "backend_config": backend.config.to_dict()},
            workload={"dataset": "synthetic", "arrival": "poisson", "rate": 100.0},
        )
        assert result["summary"]["num_succeeded"] == 8
        assert lag.mean >= 0.0
