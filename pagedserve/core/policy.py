"""Preemption: what to do when memory runs out mid-flight.

Two policies, both implemented, because the crossover between them is
measurable and the measurement is a genuinely differentiated result. Nearly
every reimplementation picks one and never compares.

===========  ==========================================  ========================
Policy       Mechanism                                   Cost
===========  ==========================================  ========================
RECOMPUTE    Drop the blocks, requeue, re-prefill later  O(prompt) compute,
                                                         zero memory traffic
SWAP         Copy blocks to pinned host memory, free      2x PCIe transfer of
             the GPU blocks, copy back on resume         the KV, zero recompute
===========  ==========================================  ========================

Which wins depends on prompt length and PCIe bandwidth. Short prompts favour
RECOMPUTE — prefill is compute-bound, highly parallel, and cheap. Long prompts
favour SWAP, because re-prefilling 4000 tokens costs far more than moving their
KV across the bus twice. The crossover is hardware-specific, which is why this
is a flag and a sweep rather than a hardcoded choice.

Victim selection: preempt from the TAIL
---------------------------------------
The running set is ordered by admission, and preemption evicts from the end —
the most recently admitted sequence goes first.

The alternative, evicting the oldest, is worse on two counts. It throws away the
most accumulated work, since the oldest sequence has the most KV built up. And
it risks starvation: a long-running request would be the first victim every time
pressure recurred, so it could be preempted repeatedly and never finish.

Tail preemption approximates last-in-first-out under pressure, which bounds how
long any individual request can be delayed.

**The fairness tradeoff is real and worth stating plainly.** LIFO is not fair in
the queueing-theory sense — a request that arrives during a sustained burst can
be admitted and evicted several times while older requests proceed untouched,
and its latency is worse than FIFO would give it. What LIFO buys is that *some*
requests finish promptly instead of every request progressing slowly, and that
the system has a bounded worst case rather than a starvation hole. Under
overload, completing some work beats making uniform slow progress on all of it.
Aging — promoting a sequence's priority with each preemption — is the standard
fix if the tail latency turns out to matter, and ``num_preemptions`` is recorded
on every sequence so the question can be answered with data.
"""

from __future__ import annotations

import logging
from enum import Enum

from pagedserve.sequence import Sequence, SequenceStatus

logger = logging.getLogger(__name__)

__all__ = ["PreemptionMode", "select_victim"]


class PreemptionMode(Enum):
    """How an evicted sequence's KV is handled."""

    RECOMPUTE = "recompute"
    SWAP = "swap"

    @classmethod
    def parse(cls, value: str | PreemptionMode) -> PreemptionMode:
        if isinstance(value, PreemptionMode):
            return value
        try:
            return cls(value.lower())
        except ValueError as exc:
            valid = [m.value for m in cls]
            raise ValueError(f"unknown preemption policy {value!r}; expected {valid}") from exc


def select_victim(running: list[Sequence]) -> Sequence | None:
    """Pick the sequence to evict: the most recently admitted still-running one.

    Returns ``None`` when nothing can be evicted, which the scheduler must treat
    as a hard stop rather than looping — a preemption round that frees nothing
    and runs again is an infinite loop under memory pressure.
    """
    for sequence in reversed(running):
        if sequence.status is SequenceStatus.RUNNING:
            return sequence
    return None
