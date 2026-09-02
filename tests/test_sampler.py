"""Unit tests for sampling.

The load-bearing assertion is that **greedy stays exactly what it was**. Five
phases of correctness were gated on token-for-token equality under greedy
decoding; if adding sampling perturbs that path, the gate stops meaning
anything retroactively.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pagedserve.model.sampler import (  # noqa: E402
    SamplingParams,
    SamplingTensors,
    apply_repetition_penalty,
    apply_temperature,
    apply_top_k,
    apply_top_p,
    sample,
)

CPU = torch.device("cpu")


def tensors_for(params, histories=None):
    params = list(params)
    histories = histories or [[] for _ in params]
    return SamplingTensors.build(params, histories, device=CPU, dtype=torch.float32)


class TestSamplingParams:
    def test_defaults_are_greedy(self):
        assert SamplingParams().is_greedy
        assert not SamplingParams().needs_penalty

    def test_rejects_nonsense(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-1)
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=0.0)
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)
        with pytest.raises(ValueError, match="top_k"):
            SamplingParams(top_k=-1)
        with pytest.raises(ValueError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=0.0)
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)
        with pytest.raises(ValueError, match="n must be"):
            SamplingParams(n=0)


class TestGreedyIsUnchanged:
    def test_all_greedy_batch_is_pure_argmax(self):
        """The fast path must not route through softmax or multinomial.

        Five phases of verification rest on greedy being bit-for-bit stable.
        """
        logits = torch.randn(4, 100)
        result = sample(logits, tensors_for([SamplingParams()] * 4))
        assert torch.equal(result, torch.argmax(logits, dim=-1))

    def test_greedy_rows_survive_a_mixed_batch(self):
        logits = torch.randn(3, 50)
        params = [
            SamplingParams(),
            SamplingParams(temperature=1.0),
            SamplingParams(),
        ]
        result = sample(logits, tensors_for(params))
        expected = torch.argmax(logits, dim=-1)
        assert result[0] == expected[0]
        assert result[2] == expected[2]

    def test_greedy_is_deterministic_across_calls(self):
        logits = torch.randn(2, 40)
        first = sample(logits, tensors_for([SamplingParams()] * 2))
        second = sample(logits, tensors_for([SamplingParams()] * 2))
        assert torch.equal(first, second)

    def test_temperature_zero_never_divides_by_zero(self):
        """A zero divisor would produce infinities that poison the whole batch."""
        logits = torch.randn(2, 20)
        params = [SamplingParams(), SamplingParams(temperature=0.5)]
        result = sample(logits, tensors_for(params))
        assert torch.isfinite(result.float()).all()


class TestRepetitionPenalty:
    def test_penalises_a_seen_token(self):
        logits = torch.zeros(1, 10)
        logits[0, 3] = 4.0
        out = apply_repetition_penalty(
            logits, tensors_for([SamplingParams(repetition_penalty=2.0)], [[3]])
        )
        assert out[0, 3] == pytest.approx(2.0)

    def test_multiplies_a_negative_logit(self):
        """Dividing a negative logit would move it toward zero and *reward* it."""
        logits = torch.zeros(1, 10)
        logits[0, 3] = -4.0
        out = apply_repetition_penalty(
            logits, tensors_for([SamplingParams(repetition_penalty=2.0)], [[3]])
        )
        assert out[0, 3] == pytest.approx(-8.0)

    def test_leaves_unseen_tokens_alone(self):
        logits = torch.ones(1, 10) * 3.0
        out = apply_repetition_penalty(
            logits, tensors_for([SamplingParams(repetition_penalty=2.0)], [[3]])
        )
        assert out[0, 5] == pytest.approx(3.0)

    def test_a_repeated_token_is_penalised_once(self):
        # The penalty is about whether a token appeared, not how often.
        logits = torch.zeros(1, 10)
        logits[0, 3] = 8.0
        out = apply_repetition_penalty(
            logits, tensors_for([SamplingParams(repetition_penalty=2.0)], [[3, 3, 3]])
        )
        assert out[0, 3] == pytest.approx(4.0)

    def test_penalty_of_one_changes_nothing(self):
        logits = torch.randn(1, 10)
        out = apply_repetition_penalty(
            logits, tensors_for([SamplingParams(repetition_penalty=1.0)], [[3]])
        )
        assert torch.equal(out, logits)

    def test_only_the_requesting_row_is_penalised(self):
        logits = torch.zeros(2, 10)
        logits[:, 3] = 4.0
        params = [
            SamplingParams(repetition_penalty=2.0),
            SamplingParams(repetition_penalty=1.0),
        ]
        out = apply_repetition_penalty(logits, tensors_for(params, [[3], [3]]))
        assert out[0, 3] == pytest.approx(2.0)
        assert out[1, 3] == pytest.approx(4.0)

    def test_it_can_change_the_argmax(self):
        """The behaviour that bit us in Phase 1's golden test."""
        logits = torch.tensor([[5.0, 4.0, 0.0]])
        out = apply_repetition_penalty(
            logits, tensors_for([SamplingParams(repetition_penalty=2.0)], [[0]])
        )
        assert int(torch.argmax(out, dim=-1)) == 1


class TestTemperature:
    def test_higher_temperature_flattens(self):
        logits = torch.tensor([[4.0, 2.0, 0.0]])
        cold = torch.softmax(
            apply_temperature(logits, tensors_for([SamplingParams(temperature=0.5)])), -1
        )
        hot = torch.softmax(
            apply_temperature(logits, tensors_for([SamplingParams(temperature=2.0)])), -1
        )
        assert cold[0, 0] > hot[0, 0]

    def test_temperature_one_is_the_identity(self):
        logits = torch.randn(1, 10)
        out = apply_temperature(logits, tensors_for([SamplingParams(temperature=1.0)]))
        assert torch.allclose(out, logits)


class TestTopK:
    def test_keeps_exactly_k(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]])
        out = apply_top_k(logits, tensors_for([SamplingParams(top_k=2)]))
        assert torch.isfinite(out).sum() == 2
        assert torch.isfinite(out[0, 1]) and torch.isfinite(out[0, 4])

    def test_zero_disables_it(self):
        logits = torch.randn(1, 10)
        out = apply_top_k(logits, tensors_for([SamplingParams(top_k=0)]))
        assert torch.equal(out, logits)

    def test_k_larger_than_the_vocabulary_is_harmless(self):
        logits = torch.randn(1, 5)
        out = apply_top_k(logits, tensors_for([SamplingParams(top_k=100)]))
        assert torch.isfinite(out).all()

    def test_per_row_k(self):
        logits = torch.tensor([[1.0, 5.0, 3.0, 2.0, 4.0]] * 2)
        params = [SamplingParams(top_k=1), SamplingParams(top_k=3)]
        out = apply_top_k(logits, tensors_for(params))
        assert torch.isfinite(out[0]).sum() == 1
        assert torch.isfinite(out[1]).sum() == 3


class TestTopP:
    def test_keeps_the_nucleus(self):
        # Probabilities roughly [0.85, 0.11, 0.04] -- 0.9 keeps the top two.
        logits = torch.tensor([[4.0, 2.0, 1.0]])
        out = apply_top_p(logits, tensors_for([SamplingParams(top_p=0.9)]))
        assert torch.isfinite(out[0, 0])
        assert torch.isfinite(out[0, 1])
        assert out[0, 2] == float("-inf")

    def test_one_disables_it(self):
        logits = torch.randn(1, 10)
        out = apply_top_p(logits, tensors_for([SamplingParams(top_p=1.0)]))
        assert torch.equal(out, logits)

    def test_never_empties_a_row(self):
        """Even a tiny top_p must keep the top token, or softmax gives NaN."""
        logits = torch.tensor([[10.0, 1.0, 0.0]])
        out = apply_top_p(logits, tensors_for([SamplingParams(top_p=0.01)]))
        assert torch.isfinite(out).any()
        assert torch.isfinite(torch.softmax(out, dim=-1)).all()


class TestSampling:
    def test_seeded_sampling_is_reproducible(self):
        logits = torch.randn(4, 50)
        params = [SamplingParams(temperature=1.0)] * 4

        def run():
            generator = torch.Generator(device="cpu").manual_seed(1234)
            return sample(logits, tensors_for(params), generator=generator)

        assert torch.equal(run(), run())

    def test_different_seeds_differ(self):
        torch.manual_seed(0)
        logits = torch.randn(8, 200) * 3

        def run(seed):
            generator = torch.Generator(device="cpu").manual_seed(seed)
            return sample(
                logits, tensors_for([SamplingParams(temperature=2.0)] * 8), generator=generator
            )

        assert not torch.equal(run(1), run(2))

    def test_top_k_one_is_effectively_greedy(self):
        logits = torch.randn(4, 60)
        params = [SamplingParams(temperature=1.0, top_k=1)] * 4
        result = sample(logits, tensors_for(params))
        assert torch.equal(result, torch.argmax(logits, dim=-1))

    def test_sampled_tokens_are_in_range(self):
        logits = torch.randn(6, 30)
        params = [SamplingParams(temperature=1.0, top_p=0.9, top_k=10)] * 6
        result = sample(logits, tensors_for(params))
        assert result.shape == (6,)
        assert int(result.min()) >= 0 and int(result.max()) < 30

    def test_a_mixed_batch_applies_each_row_its_own_parameters(self):
        logits = torch.randn(3, 40)
        params = [
            SamplingParams(),
            SamplingParams(temperature=1.0, top_k=1),
            SamplingParams(),
        ]
        result = sample(logits, tensors_for(params))
        # Rows 0 and 2 are greedy; row 1 with top_k=1 is also the argmax.
        assert torch.equal(result, torch.argmax(logits, dim=-1))
