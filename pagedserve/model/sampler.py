"""Token selection: greedy, temperature, top-k, top-p, repetition penalty.

Phases 1 through 5 were greedy-only on purpose. The golden test asserts
token-for-token equality against HuggingFace, and sampling would have made that
assertion statistical instead of exact. Now that the hard parts are locked down
and verified, the rest of the sampling surface can land — and **greedy must
remain bit-for-bit what it was**, or the gate that protected all of it stops
meaning anything. ``temperature == 0`` therefore takes a pure ``argmax`` path
that never touches the probabilistic machinery.

Everything here is batched. A step can hold sequences with different sampling
parameters, and AGENTS.md §2.5 bars per-sequence Python loops from the decode
path, so the parameters are staged as per-row tensors and every operation is a
whole-batch tensor op. That is more code than a loop over sequences and it is
the difference between a GPU at 90% and one at 30%.

Order matters, and it is the same order HuggingFace uses:

1. **Repetition penalty** — on raw logits, before any scaling. Applying it after
   temperature would make the penalty's strength depend on the temperature.
2. **Temperature** — scales the distribution's sharpness.
3. **Top-k** then **top-p** — truncate the candidate set. Top-k first, because
   applying nucleus sampling to a set already limited to k is the cheaper order
   and is what every reference implementation does, so results match.
4. **Sample** from what remains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

__all__ = ["SamplingParams", "SamplingTensors", "greedy", "sample"]


@dataclass(frozen=True)
class SamplingParams:
    """How one request's tokens are chosen.

    Defaults are greedy and deterministic, so a caller that asks for nothing
    gets the behaviour the golden test pins.

    Attributes:
        temperature: 0 means greedy. Higher flattens the distribution.
        top_p: Nucleus threshold. 1.0 disables it.
        top_k: Keep only the k highest-scoring tokens. 0 disables it.
        repetition_penalty: Divides the logits of tokens already seen (or
            multiplies, when the logit is negative — dividing a negative number
            would *increase* it and reward repetition, which is the opposite of
            the intent). 1.0 disables it.
        max_tokens: Generation cap.
        stop_token_ids: Token ids that end generation.
        stop_strings: Text that ends generation. Handled in the server layer
            rather than here: detecting them needs a detokenizer, and AGENTS.md
            §2.5 keeps the tokenizer out of the engine's hot loop.
        n: How many independent samples to generate from this prompt.
        seed: Per-request seed, for reproducible sampling.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    max_tokens: int = 16
    stop_token_ids: tuple[int, ...] = ()
    stop_strings: tuple[str, ...] = ()
    n: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be non-negative, got {self.temperature}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.top_k < 0:
            raise ValueError(f"top_k must be non-negative, got {self.top_k}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be at least 1, got {self.max_tokens}")
        if self.n < 1:
            raise ValueError(f"n must be at least 1, got {self.n}")

    @property
    def is_greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def needs_penalty(self) -> bool:
        return self.repetition_penalty != 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "max_tokens": self.max_tokens,
            "n": self.n,
            "seed": self.seed,
        }


@dataclass
class SamplingTensors:
    """Per-row sampling parameters, staged as tensors.

    Built once per step from the batch's sequences so the sampler can run whole-
    batch operations instead of branching per sequence.
    """

    temperature: torch.Tensor
    top_p: torch.Tensor
    top_k: torch.Tensor
    repetition_penalty: torch.Tensor
    all_greedy: bool
    any_penalty: bool
    # Flat (row, token) pairs for every token already in each sequence, so the
    # repetition penalty is one scatter rather than a loop.
    penalty_rows: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    penalty_tokens: torch.Tensor = field(default_factory=lambda: torch.empty(0))

    @classmethod
    def build(
        cls,
        params: list[SamplingParams],
        histories: list[list[int]],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> SamplingTensors:
        any_penalty = any(p.needs_penalty for p in params)
        rows: list[int] = []
        tokens: list[int] = []
        if any_penalty:
            for row, (param, history) in enumerate(zip(params, histories, strict=True)):
                if not param.needs_penalty:
                    continue
                # A token repeated twice is penalised once: the penalty is about
                # whether a token has appeared, not how often.
                for token in set(history):
                    rows.append(row)
                    tokens.append(token)

        def column(values, tensor_dtype=dtype):
            return torch.tensor(values, dtype=tensor_dtype, device=device)

        return cls(
            temperature=column([p.temperature for p in params]),
            top_p=column([p.top_p for p in params]),
            top_k=column([p.top_k for p in params], torch.long),
            repetition_penalty=column([p.repetition_penalty for p in params]),
            all_greedy=all(p.is_greedy for p in params),
            any_penalty=any_penalty,
            penalty_rows=column(rows, torch.long),
            penalty_tokens=column(tokens, torch.long),
        )


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Pick the highest-scoring token per row."""
    return torch.argmax(logits, dim=-1)


def apply_repetition_penalty(logits: torch.Tensor, tensors: SamplingTensors) -> torch.Tensor:
    """Penalise tokens the sequence has already produced.

    A positive logit is divided by the penalty and a negative one multiplied.
    Dividing a negative logit would move it *towards* zero and so make the token
    more likely — rewarding the repetition the penalty exists to discourage.
    """
    if not tensors.any_penalty or tensors.penalty_rows.numel() == 0:
        return logits
    rows = tensors.penalty_rows
    tokens = tensors.penalty_tokens
    penalties = tensors.repetition_penalty[rows]
    selected = logits[rows, tokens]
    adjusted = torch.where(selected > 0, selected / penalties, selected * penalties)
    logits = logits.clone()
    logits[rows, tokens] = adjusted
    return logits


def apply_temperature(logits: torch.Tensor, tensors: SamplingTensors) -> torch.Tensor:
    """Scale by temperature, leaving greedy rows untouched.

    Greedy rows are given a divisor of 1 rather than 0: argmax is
    scale-invariant so the value is irrelevant, but dividing by zero would
    produce infinities that poison the softmax for the whole batch.
    """
    divisor = torch.where(
        tensors.temperature > 0, tensors.temperature, torch.ones_like(tensors.temperature)
    )
    return logits / divisor.unsqueeze(-1)


def apply_top_k(logits: torch.Tensor, tensors: SamplingTensors) -> torch.Tensor:
    """Mask everything outside each row's k highest-scoring tokens."""
    top_k = tensors.top_k
    if int(top_k.max()) <= 0:
        return logits
    vocab = logits.shape[-1]
    # 0 means disabled, which is the whole vocabulary.
    effective = torch.where(top_k > 0, top_k, torch.full_like(top_k, vocab))
    effective = effective.clamp(max=vocab)
    # One topk at the batch's largest k, then a per-row threshold. Cheaper than
    # a ragged topk and it keeps the operation whole-batch.
    largest = int(effective.max())
    values, _ = torch.topk(logits, largest, dim=-1)
    thresholds = values.gather(1, (effective - 1).unsqueeze(-1))
    return logits.masked_fill(logits < thresholds, float("-inf"))


def apply_top_p(logits: torch.Tensor, tensors: SamplingTensors) -> torch.Tensor:
    """Keep the smallest set of tokens whose probability mass reaches ``top_p``."""
    if float(tensors.top_p.min()) >= 1.0:
        return logits
    ordered, indices = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.softmax(ordered, dim=-1).cumsum(dim=-1)
    # Strictly greater, and shifted by one, so the token that carries the
    # threshold across is kept. Dropping it could empty a row whose top token
    # already exceeds top_p.
    remove = cumulative - torch.softmax(ordered, dim=-1) > tensors.top_p.unsqueeze(-1)
    remove[:, 0] = False
    ordered = ordered.masked_fill(remove, float("-inf"))
    return ordered.scatter(1, indices, ordered)


def sample(
    logits: torch.Tensor,
    tensors: SamplingTensors,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Choose one token per row.

    Args:
        logits: ``[batch, vocab_size]``.
        tensors: Per-row parameters from ``SamplingTensors.build``.
        generator: Seeded source, for reproducible sampling.

    Returns:
        ``[batch]`` token ids.
    """
    # The fast, exact path. An all-greedy batch is the golden test's world and
    # must not be routed through softmax and multinomial, whose rounding would
    # make previously-verified output subtly different.
    if tensors.all_greedy and not tensors.any_penalty:
        return greedy(logits)

    working = apply_repetition_penalty(logits.float(), tensors)
    greedy_choice = greedy(working)

    working = apply_temperature(working, tensors)
    working = apply_top_k(working, tensors)
    working = apply_top_p(working, tensors)

    probabilities = torch.softmax(working, dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)

    # Greedy rows take the argmax of the penalised logits, never the sample.
    is_greedy = (tensors.temperature == 0).to(sampled.device)
    return torch.where(is_greedy, greedy_choice, sampled)
