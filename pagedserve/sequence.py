"""Request state: what a sequence is and where it is in its life.

Phase 1 and 2 kept this implicit in parallel lists inside the generation loop,
which worked because static batching advances every sequence in lockstep. Phase
3 breaks lockstep — sequences join, finish, and get evicted between steps — so
the state has to become a thing with a name.

The state machine::

    WAITING  --admit-->  RUNNING  --EOS / stop / max_tokens-->  FINISHED
       ^                    |
       |                    v
       +---- preempt ---- (RECOMPUTE: blocks dropped, prompt re-prefilled)
                             or
                          SWAPPED  --resume-->  RUNNING
                          (SWAP: blocks copied to host, GPU blocks freed)

A preempted sequence must resume to *identical* output. That is what makes
preemption an implementation detail rather than a behaviour change, and it is
what the forced-preemption tests assert.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = ["Sequence", "SequenceStatus"]


class SequenceStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"

    @property
    def is_terminal(self) -> bool:
        return self is SequenceStatus.FINISHED


@dataclass
class Sequence:
    """One request, from arrival to retirement.

    Attributes:
        seq_id: Stable identity. Survives preemption — a recomputed sequence is
            the same request, not a new one.
        prompt_token_ids: The prompt. Kept for the whole lifetime because
            RECOMPUTE preemption re-prefills from it; dropping it after the
            first prefill would make that policy impossible.
        output_token_ids: Generated so far.
        max_tokens: Generation cap.
        stop_token_ids: Stop tokens, usually the model's EOS set.
        status: Where it is in the state machine.
        num_computed_tokens: How many of this sequence's tokens have KV in the
            cache. Normally ``len(prompt) + len(output) - 1``; RECOMPUTE
            preemption resets it to zero, which is exactly what "drop the
            blocks and start over" means.
        num_preemptions: How many times this sequence has been evicted. Recorded
            because preemption thrash is a real failure mode and a benchmark
            that hides it is not measuring what it claims.
    """

    seq_id: int
    prompt_token_ids: list[int]
    max_tokens: int
    stop_token_ids: tuple[int, ...] = ()
    output_token_ids: list[int] = field(default_factory=list)
    status: SequenceStatus = SequenceStatus.WAITING
    num_computed_tokens: int = 0
    num_preemptions: int = 0
    finish_reason: str | None = None
    arrival_index: int = 0

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        """Tokens this sequence occupies: prompt plus everything generated."""
        return self.prompt_len + self.output_len

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_uncomputed_tokens(self) -> int:
        """Tokens still needing a forward pass before the next one can be sampled.

        After a RECOMPUTE preemption this is the whole prompt again, which is
        the cost of that policy: O(prompt) compute, zero memory traffic.
        """
        return max(0, self.total_len - self.num_computed_tokens)

    @property
    def is_prefill(self) -> bool:
        """Whether the next step for this sequence is a prefill."""
        return self.num_uncomputed_tokens > 1

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_stop(self) -> bool:
        """Retire on a stop token or the length cap. Returns True if finished."""
        if self.output_token_ids and self.output_token_ids[-1] in self.stop_token_ids:
            self.status = SequenceStatus.FINISHED
            self.finish_reason = "stop"
            return True
        if self.output_len >= self.max_tokens:
            self.status = SequenceStatus.FINISHED
            self.finish_reason = "length"
            return True
        return False

    def reset_for_recompute(self) -> None:
        """Drop cached state, keeping generated tokens.

        The output so far is *not* thrown away — only the KV backing it. On
        resume the sequence re-prefills prompt plus output and continues from
        where it was, which is why a recomputed sequence produces identical
        tokens to one that was never preempted.
        """
        self.num_computed_tokens = 0
        self.num_preemptions += 1
        self.status = SequenceStatus.WAITING

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.value}, "
            f"prompt={self.prompt_len}, output={self.output_len}, "
            f"computed={self.num_computed_tokens})"
        )
