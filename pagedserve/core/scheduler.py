"""Iteration-level scheduling: the batch is decided fresh on every step.

Static batching collects N requests, runs them to completion together, and only
then starts the next batch. If one request generates 20 tokens and another 800,
the first one's slot is dead weight for 780 steps — compute burned on padding
and memory held hostage. That is head-of-line blocking, and it is the dominant
throughput killer.

Continuous batching (Orca's iteration-level scheduling) runs this scheduler
between *every* decode step. Finished sequences retire immediately and free
their blocks; waiting requests are admitted into the freed capacity on the very
next step. Batch composition changes constantly.

**This is only possible because of paging.** Admitting a sequence mid-flight
needs memory right now, in whatever shape happens to be free. A contiguous
allocator needs a contiguous hole of the right size, which usually does not
exist once the cache has been churned. Block tables need *k* blocks off a free
list, any *k*. Paging is the enabler; continuous batching is the payoff.

Budgets
-------
Admission is capped on **both** token count and sequence count, simultaneously.

``max_num_seqs`` alone is not enough, because a step holding one 2000-token
prefill is roughly 2000x the work of a step holding one decode token. Budgeting
only on sequences produces wildly uneven step times and terrible inter-token
latency variance — the tail that a mean would hide.

``max_num_batched_tokens`` alone is not enough either, because per-sequence
bookkeeping and block-table staging cost something regardless of token count.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from pagedserve.config import SchedulerConfig
from pagedserve.core.policy import PreemptionMode, select_victim
from pagedserve.memory.block_manager import AllocStatus, BlockManager
from pagedserve.sequence import Sequence, SequenceStatus

logger = logging.getLogger(__name__)

__all__ = ["Scheduler", "SchedulerOutput"]


@dataclass
class SchedulerOutput:
    """What runs this iteration, and what happened to make room for it."""

    prefills: list[Sequence] = field(default_factory=list)
    decodes: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    finished: list[Sequence] = field(default_factory=list)
    num_batched_tokens: int = 0

    @property
    def scheduled(self) -> list[Sequence]:
        """Prefills first, then decodes — the order the batch is laid out in."""
        return self.prefills + self.decodes

    @property
    def is_empty(self) -> bool:
        return not self.prefills and not self.decodes

    @property
    def batch_size(self) -> int:
        return len(self.prefills) + len(self.decodes)


class Scheduler:
    """Owns the waiting / running / swapped queues and decides each iteration."""

    def __init__(
        self,
        config: SchedulerConfig,
        block_manager: BlockManager,
        *,
        swap_space=None,
    ) -> None:
        self.config = config
        self.block_manager = block_manager
        self.swap_space = swap_space
        self.policy = PreemptionMode.parse(config.preemption_policy)
        if self.policy is PreemptionMode.SWAP and swap_space is None:
            raise ValueError(
                "preemption_policy='swap' needs a SwapSpace; the engine builds one "
                "when swap_space_blocks is set"
            )

        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.swapped: deque[Sequence] = deque()

        # Host block ids parked for each swapped-out sequence.
        self._swapped_blocks: dict[int, list[int]] = {}
        self.num_preemptions = 0
        self._arrivals = 0

    # ---- queue management ----------------------------------------------

    def add_request(self, sequence: Sequence) -> None:
        sequence.status = SequenceStatus.WAITING
        sequence.arrival_index = self._arrivals
        self._arrivals += 1
        self.waiting.append(sequence)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    # ---- the iteration --------------------------------------------------

    def schedule(self) -> SchedulerOutput:
        """Decide this iteration's batch.

        Order matters:

        1. **Retire** finished sequences and free their blocks *first*, so the
           capacity they release is available to admissions in this same step
           rather than the next one. Retiring late is a whole step of wasted
           headroom per completion.
        2. **Make room** for the sequences already running, preempting from the
           tail until every survivor can take its next token.
        3. **Resume** swapped sequences before admitting new ones — they have
           accumulated work and their memory is already accounted for.
        4. **Admit** from the waiting queue while both budgets allow.
        """
        output = SchedulerOutput()
        self._retire_finished(output)
        self._make_room_for_running(output)

        budget_tokens = self.config.max_num_batched_tokens
        budget_seqs = self.config.max_num_seqs

        # Running sequences have first claim on the budget: they are already
        # holding memory, and starving them to admit new work is how a system
        # ends up with everything half-finished.
        for sequence in self.running:
            if sequence.status is not SequenceStatus.RUNNING:
                continue
            tokens = sequence.num_uncomputed_tokens
            if tokens > budget_tokens or budget_seqs <= 0:
                break
            budget_tokens -= tokens
            budget_seqs -= 1
            if sequence.is_prefill:
                output.prefills.append(sequence)
            else:
                output.decodes.append(sequence)
            output.num_batched_tokens += tokens

        budget_tokens, budget_seqs = self._resume_swapped(output, budget_tokens, budget_seqs)
        self._admit_waiting(output, budget_tokens, budget_seqs)
        return output

    def _retire_finished(self, output: SchedulerOutput) -> None:
        still_running = []
        for sequence in self.running:
            if sequence.status is SequenceStatus.FINISHED:
                self.block_manager.free(sequence.seq_id)
                output.finished.append(sequence)
            else:
                still_running.append(sequence)
        self.running = still_running

    def _make_room_for_running(self, output: SchedulerOutput) -> None:
        """Preempt from the tail until every survivor can take its next token.

        The demand being checked is one block per sequence that is about to
        cross a block boundary — not a whole sequence's worth. That is the
        amortisation paging buys: most steps need nothing at all.
        """
        while True:
            needed = sum(
                1
                for s in self.running
                if s.status is SequenceStatus.RUNNING and self._needs_new_block(s)
            )
            if needed <= self.block_manager.num_free_blocks:
                return
            victim = select_victim(self.running)
            if victim is None:
                # Nothing left to evict. Returning rather than looping: a
                # preemption round that frees nothing and runs again is an
                # infinite loop under pressure.
                logger.warning(
                    "cannot free enough blocks: %d needed, %d free, nothing to preempt",
                    needed,
                    self.block_manager.num_free_blocks,
                )
                return
            self._preempt(victim, output)

    def _needs_new_block(self, sequence: Sequence) -> bool:
        """Whether this sequence crosses a block boundary on its next token."""
        table = self.block_manager.block_tables.get(sequence.seq_id)
        if table is None:
            return False
        return self.block_manager.blocks_needed(sequence.total_len + 1) > len(table)

    def _preempt(self, victim: Sequence, output: SchedulerOutput) -> None:
        self.running.remove(victim)
        self.num_preemptions += 1
        output.preempted.append(victim)

        if self.policy is PreemptionMode.RECOMPUTE:
            # Drop the KV and requeue. The generated tokens are kept, so the
            # sequence resumes exactly where it left off -- preemption must be
            # invisible in the output.
            self.block_manager.free(victim.seq_id)
            victim.reset_for_recompute()
            self.waiting.appendleft(victim)
        else:
            self._swap_out(victim)

        logger.debug(
            "preempted sequence %d by %s (preemption %d)",
            victim.seq_id,
            self.policy.value,
            victim.num_preemptions,
        )

    def _swap_out(self, victim: Sequence) -> None:
        blocks = list(self.block_manager.block_table(victim.seq_id))
        if self.swap_space is None or not self.swap_space.can_swap_out(len(blocks)):
            # Host space is finite too. Degrading to RECOMPUTE beats failing the
            # request, and the fallback is logged rather than silent because it
            # changes the cost model of the run being measured.
            logger.warning(
                "swap space full, falling back to recompute for sequence %d", victim.seq_id
            )
            self.block_manager.free(victim.seq_id)
            victim.reset_for_recompute()
            self.waiting.appendleft(victim)
            return
        self._swapped_blocks[victim.seq_id] = self._do_swap_out(blocks)
        self.block_manager.free(victim.seq_id)
        victim.num_preemptions += 1
        victim.status = SequenceStatus.SWAPPED
        self.swapped.appendleft(victim)

    def _do_swap_out(self, blocks: list[int]) -> list[int]:
        """Overridden in tests to exercise the scheduler without a KV tensor."""
        return self.swap_space.swap_out(self._gpu_cache(), blocks)

    def _do_swap_in(self, cpu_blocks: list[int], gpu_blocks: list[int]) -> None:
        self.swap_space.swap_in(self._gpu_cache(), cpu_blocks, gpu_blocks)

    def _gpu_cache(self):
        raise RuntimeError(
            "the scheduler needs a KV cache to swap; the engine installs one via attach_kv_cache()"
        )

    def attach_kv_cache(self, get_cache) -> None:
        """Give the scheduler access to the KV tensor, for SWAP only."""
        self._gpu_cache = get_cache

    def _running_growth_reserve(self) -> int:
        """Blocks the running set will claim when it appends this step's token.

        Admission must not spend these. Handing them to a new arrival
        guarantees either a MemoryError when block tables grow at the end of the
        step, or a preemption on the very next one -- and a system that admits a
        request only to evict something immediately is doing strictly worse than
        not admitting it.
        """
        return sum(
            1
            for s in self.running
            if s.status is SequenceStatus.RUNNING and self._needs_new_block(s)
        )

    def _resume_swapped(
        self, output: SchedulerOutput, budget_tokens: int, budget_seqs: int
    ) -> tuple[int, int]:
        """Bring swapped sequences back before admitting anything new.

        They already hold accumulated work, and their host blocks are occupying
        swap space that stays occupied until they resume.
        """
        reserve = self._running_growth_reserve()
        while self.swapped and budget_seqs > 0:
            sequence = self.swapped[0]
            cpu_blocks = self._swapped_blocks.get(sequence.seq_id, [])
            needed = self.block_manager.blocks_needed(sequence.total_len)
            # A resumed sequence decodes immediately, so it needs the same
            # boundary-crossing headroom a newly admitted one does. Without
            # this it resumes and then cannot grow at the end of the step.
            growth = self.block_manager.blocks_needed(sequence.total_len + 1) - needed
            if needed + growth > self.block_manager.num_free_blocks - reserve:
                break
            if len(cpu_blocks) > needed:
                break
            tokens = sequence.num_uncomputed_tokens
            if tokens > budget_tokens:
                break

            self.swapped.popleft()
            table, _ = self.block_manager.allocate(sequence.seq_id, sequence.total_len)
            self._do_swap_in(cpu_blocks, list(table)[: len(cpu_blocks)])
            self._swapped_blocks.pop(sequence.seq_id, None)

            sequence.status = SequenceStatus.RUNNING
            self.running.append(sequence)
            budget_tokens -= tokens
            budget_seqs -= 1
            reserve += growth
            output.decodes.append(sequence)
            output.num_batched_tokens += tokens
        return budget_tokens, budget_seqs

    def _admit_waiting(self, output: SchedulerOutput, budget_tokens: int, budget_seqs: int) -> None:
        reserve = self._running_growth_reserve()
        while self.waiting and budget_seqs > 0:
            sequence = self.waiting[0]
            tokens = sequence.num_uncomputed_tokens

            # Impossibility is checked BEFORE headroom. A request that can
            # never fit must be rejected, not deferred -- deferring it leaves it
            # at the head of the queue blocking everything behind it forever,
            # which is the starvation AllocStatus.NEVER exists to prevent.
            status = self.block_manager.can_allocate(sequence.total_len)
            if status is AllocStatus.NEVER:
                # Cannot ever fit. Rejecting outright beats letting it sit in
                # the queue forever blocking everything behind it.
                self.waiting.popleft()
                sequence.status = SequenceStatus.FINISHED
                sequence.finish_reason = "too_long"
                output.finished.append(sequence)
                logger.warning(
                    "rejecting sequence %d: %d tokens exceeds total cache capacity",
                    sequence.seq_id,
                    sequence.total_len,
                )
                continue
            if status is AllocStatus.LATER:
                break

            needed = self.block_manager.blocks_needed(sequence.total_len)
            # A newly admitted sequence also appends a token this step, so it
            # needs its own headroom if that token crosses a block boundary.
            growth = self.block_manager.blocks_needed(sequence.total_len + 1) - needed
            if needed + growth > self.block_manager.num_free_blocks - reserve:
                break
            if tokens > budget_tokens:
                break

            self.waiting.popleft()
            # Prompt tokens are handed to the allocator so it can match the
            # prefix cache. A hit returns blocks whose KV already exists, and
            # num_computed_tokens is how the engine learns to skip that span.
            _, cached = self.block_manager.allocate(
                sequence.seq_id, sequence.total_len, sequence.all_token_ids
            )
            if cached > sequence.num_computed_tokens:
                sequence.num_computed_tokens = cached
                tokens = sequence.num_uncomputed_tokens
            sequence.status = SequenceStatus.RUNNING
            self.running.append(sequence)
            budget_tokens -= tokens
            budget_seqs -= 1
            reserve += growth
            output.prefills.append(sequence)
            output.num_batched_tokens += tokens

    # ---- post-step ------------------------------------------------------

    def append_slots(self, sequences: list[Sequence]) -> None:
        """Grow block tables after a step's tokens have been appended."""
        for sequence in sequences:
            if sequence.status is SequenceStatus.RUNNING:
                self.block_manager.append_slot(sequence.seq_id, sequence.total_len)

    def free_finished(self) -> list[Sequence]:
        finished = [s for s in self.running if s.status is SequenceStatus.FINISHED]
        for sequence in finished:
            self.block_manager.free(sequence.seq_id)
        self.running = [s for s in self.running if s.status is not SequenceStatus.FINISHED]
        return finished

    def reset(self) -> None:
        self.waiting.clear()
        self.running.clear()
        self.swapped.clear()
        self._swapped_blocks.clear()
        self.num_preemptions = 0
        self._arrivals = 0
        self.block_manager.reset()
