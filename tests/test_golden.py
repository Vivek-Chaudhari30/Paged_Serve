"""The commit gate: our greedy output must equal HuggingFace's, token for token.

This is the test AGENTS.md §2.2 says must pass before any commit, with every
attention backend, and with prefix caching on or off once that exists. It is
never loosened, never widened to a tolerance, and never skipped to make a
change land. A failing golden test is a bug report.

**The reference must be neutralised.** A checkpoint's ``generation_config.json``
can carry sampling defaults, and ``generate()`` applies its logits processors
even when ``do_sample=False``. Qwen2.5-0.5B ships ``repetition_penalty=1.1``,
which is enough to move an argmax by a wide margin — it flips a decision whose
top-two logits differ by 1.6 — and would make this test fail against a perfectly
correct engine. Comparing against an un-neutralised reference is comparing
against a different algorithm.

Marked ``gpu`` where relevant but runs on CPU: the point is numerical agreement,
which is device-independent, and CPU float32 is the strictest setting to
demonstrate it in.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from pagedserve.config import CacheConfig, SchedulerConfig  # noqa: E402
from pagedserve.engine import LLMEngine  # noqa: E402

# Small, ungated, and already cached on the dev machine, so this test needs no
# token and no network. Override to run the gate against another checkpoint.
GOLDEN_MODEL = os.environ.get("PAGEDSERVE_GOLDEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
# The gate must pass on every device and dtype we run on, not just the laptop's.
# On a T4 that means float16, since Turing has no bfloat16 at all.
TEST_DEVICE = os.environ.get("PAGEDSERVE_TEST_DEVICE", "cpu")
TEST_DTYPE = os.environ.get("PAGEDSERVE_TEST_DTYPE", "float32")

# Fixed by design. A golden test whose prompts change is not a golden test.
GOLDEN_PROMPTS = [
    "The capital of France is",
    "In a shocking finding, scientists",
    "def fibonacci(n):",
    "Q: What is 2 + 2?\nA:",
]
MAX_NEW_TOKENS = 24


def _model_is_available(model: str) -> bool:
    """Whether the checkpoint can be loaded without reaching the network."""
    try:
        from pagedserve.model.loader import resolve_model_path

        resolve_model_path(model)
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _model_is_available(GOLDEN_MODEL),
    reason=(
        f"golden model {GOLDEN_MODEL} is not available locally. This is a SKIP, "
        f"not a pass: the commit gate has not run. Fetch it with "
        f"`hf download {GOLDEN_MODEL}` or set PAGEDSERVE_GOLDEN_MODEL."
    ),
)


@pytest.fixture(scope="module")
def model_path() -> str:
    from pagedserve.model.loader import resolve_model_path

    return str(resolve_model_path(GOLDEN_MODEL))


@pytest.fixture(scope="module")
def tokenizer(model_path):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def reference(model_path, tokenizer):
    """HuggingFace greedy output, with the checkpoint's sampling knobs removed."""
    from transformers import AutoModelForCausalLM, GenerationConfig

    from bench.baseline_hf import dtype_kwarg

    model = (
        AutoModelForCausalLM.from_pretrained(model_path, **dtype_kwarg(getattr(torch, TEST_DTYPE)))
        .to(TEST_DEVICE)
        .eval()
    )

    def generate(prompts: list[str], max_new_tokens: int) -> list[list[int]]:
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(TEST_DEVICE)
        config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.inference_mode():
            out = model.generate(**encoded, generation_config=config)
        prompt_len = encoded["input_ids"].shape[1]
        rows = out[:, prompt_len:].tolist()
        # generate() pads finished rows out to the batch's longest run; trim so
        # the comparison is against real tokens only.
        trimmed = []
        for row in rows:
            stop = len(row)
            for i, token in enumerate(row):
                if token in _eos_ids(tokenizer):
                    stop = i + 1
                    break
            trimmed.append(row[:stop])
        return trimmed

    return generate


def _eos_ids(tokenizer) -> set[int]:
    eos = tokenizer.eos_token_id
    return set(eos) if isinstance(eos, list) else {eos}


def build_engine(model_path: str, backend: str) -> LLMEngine:
    """An engine on the named backend, with allocator invariants checked.

    ``num_blocks_override`` is required off CUDA: there is no way to measure
    free host memory, and inventing a figure would put a fabricated number under
    every capacity decision. 128 blocks x 16 tokens = 2048 slots, which is
    exactly ``max_num_seqs x max_seq_len``, so the paged arm is given the same
    capacity the dense arm reserves.
    """
    return LLMEngine.from_pretrained(
        model_path,
        device=TEST_DEVICE,
        dtype=TEST_DTYPE,
        cache=CacheConfig(max_seq_len=256, max_num_seqs=8, block_size=16, num_blocks_override=128),
        attn_backend=backend,
        debug_invariants=True,
    )


@pytest.fixture(scope="module", params=["contiguous", "gather"])
def engine(request, model_path):
    """Every golden assertion runs against every backend.

    AGENTS.md §2.2 requires it, and it is what makes a new backend's first
    failure a test failure rather than a subtly wrong benchmark six weeks later.
    """
    return build_engine(model_path, request.param)


@pytest.fixture(scope="module")
def dense_engine(model_path):
    return build_engine(model_path, "contiguous")


@pytest.fixture(scope="module")
def paged_engine(model_path):
    return build_engine(model_path, "gather")


class TestGoldenOutput:
    def test_batched_greedy_matches_huggingface(self, engine, tokenizer, reference):
        """The gate. Every prompt, batched and left-padded, token for token."""
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        ours = engine.generate(prompt_ids, max_tokens=MAX_NEW_TOKENS).token_ids
        theirs = reference(GOLDEN_PROMPTS, MAX_NEW_TOKENS)

        for i, (mine, ref) in enumerate(zip(ours, theirs, strict=True)):
            assert mine == ref, (
                f"prompt {i} ({GOLDEN_PROMPTS[i]!r}) diverged.\n"
                f"  ours: {mine}\n  ref : {ref}\n"
                f"  ours text: {tokenizer.decode(mine)!r}\n"
                f"  ref  text: {tokenizer.decode(ref)!r}"
            )

    @pytest.mark.parametrize("index", range(len(GOLDEN_PROMPTS)))
    def test_each_prompt_alone_matches_huggingface(self, engine, tokenizer, reference, index):
        """Unbatched, so a failure here is the model rather than the padding."""
        prompt = GOLDEN_PROMPTS[index]
        ours = engine.generate([tokenizer(prompt).input_ids], max_tokens=MAX_NEW_TOKENS)
        theirs = reference([prompt], MAX_NEW_TOKENS)
        assert ours.token_ids[0] == theirs[0]

    def test_batching_does_not_change_a_sequence_output(self, engine, tokenizer):
        """Left padding must be invisible to the tokens a sequence produces.

        A sequence batched with a longer neighbour gets pad positions in its
        cache. If the mask is wrong those pads leak into attention, and the
        symptom is fluent, plausible, wrong text rather than a crash.
        """
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        batched = engine.generate(prompt_ids, max_tokens=12).token_ids
        for i, ids in enumerate(prompt_ids):
            alone = engine.generate([ids], max_tokens=12).token_ids[0]
            assert alone == batched[i], f"prompt {i} changed when batched"


class TestGenerationMechanics:
    def test_respects_max_tokens(self, engine, tokenizer):
        ids = [tokenizer("Count: 1 2 3").input_ids]
        out = engine.generate(ids, max_tokens=5)
        assert len(out.token_ids[0]) <= 5

    def test_stops_on_eos(self, engine, tokenizer):
        """A sequence that emits EOS stops there and reports why."""
        ids = [tokenizer("The capital of France is").input_ids]
        eos = next(iter(_eos_ids(tokenizer)))
        out = engine.generate(ids, max_tokens=8, eos_token_ids=(eos,))
        if out.finish_reasons[0] == "stop":
            assert out.token_ids[0][-1] == eos
            assert eos not in out.token_ids[0][:-1]

    def test_forced_stop_token_terminates_immediately(self, engine, tokenizer):
        """Treat the model's actual first token as a stop token."""
        ids = [tokenizer("The capital of France is").input_ids]
        first = engine.generate(ids, max_tokens=4).token_ids[0][0]
        out = engine.generate(ids, max_tokens=8, eos_token_ids=(first,))
        assert out.token_ids[0] == [first]
        assert out.finish_reasons[0] == "stop"

    def test_empty_batch(self, engine):
        out = engine.generate([], max_tokens=4)
        assert out.token_ids == []

    def test_rejects_a_prompt_longer_than_the_cache(self, engine, tokenizer):
        too_long = [list(range(300))]
        with pytest.raises(ValueError, match="exceeds"):
            engine.generate(too_long, max_tokens=8)

    def test_rejects_a_batch_wider_than_the_cache(self, engine, tokenizer):
        ids = tokenizer("hi").input_ids
        with pytest.raises(ValueError, match="max_num_seqs"):
            engine.generate([ids] * 9, max_tokens=4)


class TestPagedEquivalence:
    """Paging must be invisible to the tokens produced.

    A memory layout change that alters output is a bug, not an optimisation.
    This is the Phase 2 assertion: same tokens, far less memory.
    """

    def test_paged_and_dense_agree_bit_for_bit(self, dense_engine, paged_engine, tokenizer):
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        dense = dense_engine.generate(prompt_ids, max_tokens=MAX_NEW_TOKENS)
        paged = paged_engine.generate(prompt_ids, max_tokens=MAX_NEW_TOKENS)
        assert dense.token_ids == paged.token_ids
        assert dense.finish_reasons == paged.finish_reasons

    def test_paged_uses_far_less_memory_for_the_same_work(
        self, dense_engine, paged_engine, tokenizer
    ):
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        dense = dense_engine.generate(prompt_ids, max_tokens=MAX_NEW_TOKENS)
        paged = paged_engine.generate(prompt_ids, max_tokens=MAX_NEW_TOKENS)
        # Identical live bytes -- the same real tokens are cached either way.
        assert dense.final_stats.live_bytes == paged.final_stats.live_bytes
        # But the paged arm holds only the blocks it needs.
        assert paged.final_stats.allocated_bytes < dense.final_stats.allocated_bytes
        assert paged.final_stats.utilization > dense.final_stats.utilization

    def test_block_manager_returns_everything_between_runs(self, paged_engine, tokenizer):
        """No leaks across generate() calls.

        A block that escapes the free list shrinks the cache silently, and the
        symptom is a server that admits fewer requests the longer it runs.
        """
        manager = paged_engine.block_manager
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        paged_engine.generate(prompt_ids, max_tokens=8)
        manager.check_invariants()
        used_after_first = manager.num_used_blocks
        paged_engine.generate(prompt_ids, max_tokens=8)
        manager.check_invariants()
        assert manager.num_used_blocks == used_after_first


class TestPagedMemory:
    def test_utilization_clears_the_phase_2_bar(self, paged_engine, tokenizer):
        """Waste is bounded by the last partial block, not by max_seq_len.

        The bound is ``block_size - 1`` tokens per sequence, so utilization
        rises toward 100% as sequences lengthen. Short generations are dominated
        by that partial block, which is why the threshold here is modest and the
        trend matters more than the point.
        """
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        short = paged_engine.generate(prompt_ids, max_tokens=16).final_stats
        longer = paged_engine.generate(prompt_ids, max_tokens=64).final_stats
        assert short.utilization > 0.50
        assert longer.utilization > short.utilization

    def test_never_exceeds_the_block_size_waste_bound(self, paged_engine, tokenizer):
        """At most ``block_size - 1`` wasted token slots per sequence."""
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        stats = paged_engine.generate(prompt_ids, max_tokens=32).final_stats
        per_token = paged_engine.config.model.kv_bytes_per_token(paged_engine.config.dtype)
        wasted_slots = stats.wasted_bytes / per_token
        block_size = paged_engine.config.cache.block_size
        assert wasted_slots <= len(GOLDEN_PROMPTS) * (block_size - 1)


class TestMemoryInstrumentation:
    """The utilization ratio is the whole point of Phase 1."""

    def test_utilization_is_low_with_a_contiguous_cache(self, dense_engine, tokenizer):
        # Short sequences in a cache sized for long ones. The vLLM paper
        # measured 20.4%-38.2% effective utilization in existing systems; a
        # contiguous reservation should land in that neighbourhood or below.
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        out = dense_engine.generate(prompt_ids, max_tokens=16)
        stats = out.final_stats
        assert stats is not None
        assert stats.utilization is not None
        assert 0.0 < stats.utilization < 0.40, (
            f"expected heavy waste from a contiguous cache, got "
            f"{stats.utilization:.1%}. If this is high, the cache is being "
            f"sized to the batch's need rather than to max_seq_len, which "
            f"would understate the very waste this arm exists to measure."
        )

    def test_live_bytes_grow_as_tokens_are_generated(self, dense_engine, tokenizer):
        out = dense_engine.generate([tokenizer("The capital of France is").input_ids], max_tokens=8)
        live = [s.live_bytes for s in out.step_stats]
        assert live == sorted(live), "live bytes must be non-decreasing"
        assert live[-1] > live[0]

    def test_allocated_bytes_never_change_during_a_run(self, dense_engine, tokenizer):
        """The reservation is made once at startup, never grown in the loop."""
        out = dense_engine.generate([tokenizer("hello world").input_ids], max_tokens=8)
        allocated = {s.allocated_bytes for s in out.step_stats}
        assert len(allocated) == 1

    def test_utilization_matches_the_hand_computation(self, dense_engine, tokenizer):
        """Cross-check the instrumentation against the arithmetic by hand."""
        ids = tokenizer("The capital of France is").input_ids
        out = dense_engine.generate([ids], max_tokens=8)
        stats = out.final_stats
        model = dense_engine.config.model
        per_token = model.kv_bytes_per_token(dense_engine.config.dtype)

        # Minus one: generating N tokens takes N forward passes, and the last
        # token sampled never goes back through the model, so its K and V are
        # never computed. Its KV would only be needed to produce an (N+1)th
        # token. Counting it would overstate live bytes by one token.
        cached_tokens = len(ids) + len(out.token_ids[0]) - 1
        assert stats.live_bytes == cached_tokens * per_token

        # Sized to the server's configured capacity, not to this batch.
        cache = dense_engine.config.cache
        expected_alloc = cache.max_num_seqs * cache.max_seq_len * per_token
        assert stats.allocated_bytes == expected_alloc


class TestContinuousBatching:
    """Phase 3: iteration-level scheduling must not change what is produced.

    Continuous batching is a throughput change, not a semantic one. If the
    tokens differ from a static-batched run, the scheduler has a bug — the
    comparison is against the same engine on the same backend, so nothing else
    can account for a difference.
    """

    def reference(self, paged_engine, tokenizer):
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        return prompt_ids, paged_engine.generate(prompt_ids, max_tokens=MAX_NEW_TOKENS).token_ids

    def test_matches_static_batching(self, model_path, paged_engine, tokenizer):
        prompt_ids, expected = self.reference(paged_engine, tokenizer)
        engine = build_engine(model_path, "gather")
        sequences = engine.generate_continuous(prompt_ids, max_tokens=MAX_NEW_TOKENS)
        assert [s.output_token_ids for s in sequences] == expected

    def test_mixed_batch_composition_is_reached(self, model_path, tokenizer):
        """One step really does hold a prefill and decodes at once."""
        engine = build_engine(model_path, "gather")
        engine.start()
        engine.add_request(tokenizer(GOLDEN_PROMPTS[0]).input_ids, max_tokens=12)
        engine.step()  # sequence 0 is now decoding
        engine.add_request(tokenizer(GOLDEN_PROMPTS[1]).input_ids, max_tokens=12)
        output = engine.step()
        assert output.prefills and output.decodes
        # Ragged: prefill length plus one decode token, not padded to the max.
        assert output.num_batched_tokens == output.prefills[0].prompt_len + len(output.decodes)

    def test_a_request_added_mid_flight_still_matches(self, model_path, tokenizer):
        """Joining an in-progress batch must not perturb the joiner or the batch."""
        engine = build_engine(model_path, "gather")
        alone = engine.generate_continuous([tokenizer(GOLDEN_PROMPTS[1]).input_ids], max_tokens=16)[
            0
        ].output_token_ids

        engine = build_engine(model_path, "gather")
        engine.start()
        engine.add_request(tokenizer(GOLDEN_PROMPTS[0]).input_ids, max_tokens=40)
        for _ in range(5):
            engine.step()
        late = engine.add_request(tokenizer(GOLDEN_PROMPTS[1]).input_ids, max_tokens=16)
        while engine.scheduler.num_unfinished:
            engine.step()
        assert late.output_token_ids == alone

    @pytest.mark.parametrize("policy", ["recompute", "swap"])
    def test_forced_preemption_is_invisible_in_the_output(
        self, model_path, paged_engine, tokenizer, policy
    ):
        """The Phase 3 exit criterion, for both policies.

        The cache is starved so eviction is unavoidable, and the resumed
        sequences must produce exactly what an unpreempted run produced. A
        preemption that changes a token is a correctness bug wearing an
        optimisation's clothes.
        """
        prompt_ids, expected = self.reference(paged_engine, tokenizer)
        starved = LLMEngine.from_pretrained(
            model_path,
            device=TEST_DEVICE,
            dtype=TEST_DTYPE,
            cache=CacheConfig(
                max_seq_len=256,
                max_num_seqs=8,
                block_size=16,
                num_blocks_override=6,
                swap_space_blocks=64,
            ),
            scheduler=SchedulerConfig(
                max_num_batched_tokens=2048, max_num_seqs=8, preemption_policy=policy
            ),
            attn_backend="gather",
            debug_invariants=True,
        )
        sequences = starved.generate_continuous(prompt_ids, max_tokens=MAX_NEW_TOKENS)
        assert starved.scheduler.num_preemptions > 0, "the cache was not starved enough"
        assert [s.output_token_ids for s in sequences] == expected
        starved.block_manager.check_invariants()

    def test_blocks_are_all_returned_when_the_run_ends(self, model_path, tokenizer):
        engine = build_engine(model_path, "gather")
        prompt_ids = [tokenizer(p).input_ids for p in GOLDEN_PROMPTS]
        engine.generate_continuous(prompt_ids, max_tokens=12)
        engine.scheduler.free_finished()
        engine.block_manager.check_invariants()
