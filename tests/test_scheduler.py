"""Unit tests for the iteration-level scheduler and the sequence state machine.

Pure Python — no model, no torch tensors. The scheduler's job is deciding
*which* sequences run, and that decision is testable on its own. Whether the
resulting tokens are right is the golden test's job.
"""

from __future__ import annotations

import pytest

from pagedserve.config import SchedulerConfig
from pagedserve.core.policy import PreemptionMode, select_victim
from pagedserve.core.scheduler import Scheduler
from pagedserve.memory.block_manager import BlockManager
from pagedserve.sequence import Sequence, SequenceStatus

BLOCK_SIZE = 4


class StubSwapSpace:
    """Records swaps without owning a KV tensor.

    Lets the scheduler's SWAP path be tested without a model, which keeps these
    tests fast and keeps a scheduler bug distinguishable from a copy bug.
    """

    def __init__(self, capacity: int = 64) -> None:
        self.capacity = capacity
        self.parked: dict[int, list[int]] = {}
        self.swapped_out = 0
        self.swapped_in = 0
        self._next = 0

    def can_swap_out(self, num_blocks: int) -> bool:
        return self._next + num_blocks <= self.capacity

    def swap_out(self, _cache, blocks: list[int]) -> list[int]:
        self.swapped_out += 1
        ids = list(range(self._next, self._next + len(blocks)))
        self._next += len(blocks)
        return ids

    def swap_in(self, _cache, cpu_blocks: list[int], gpu_blocks: list[int]) -> None:
        assert len(cpu_blocks) == len(gpu_blocks)
        self.swapped_in += 1


def make_scheduler(
    *,
    num_blocks: int = 16,
    max_num_seqs: int = 8,
    max_num_batched_tokens: int = 1024,
    policy: str = "recompute",
    swap: StubSwapSpace | None = None,
) -> Scheduler:
    manager = BlockManager(num_blocks, BLOCK_SIZE, watermark=0.0)
    config = SchedulerConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        preemption_policy=policy,
    )
    scheduler = Scheduler(config, manager, swap_space=swap)
    scheduler.attach_kv_cache(lambda: None)
    return scheduler


def make_sequence(seq_id: int, prompt_len: int = 4, max_tokens: int = 100) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        max_tokens=max_tokens,
        stop_token_ids=(999,),
    )


def advance(scheduler: Scheduler, output, token: int = 1) -> None:
    """Simulate the model: append a token to everything that ran."""
    for sequence in output.scheduled:
        sequence.num_computed_tokens = sequence.total_len
        sequence.append_token(token)
        sequence.check_stop()
    scheduler.append_slots(output.scheduled)


class TestSequenceState:
    def test_uncomputed_is_the_whole_prompt_before_any_step(self):
        seq = make_sequence(0, prompt_len=10)
        assert seq.num_uncomputed_tokens == 10
        assert seq.is_prefill

    def test_becomes_a_decode_after_prefill(self):
        seq = make_sequence(0, prompt_len=10)
        seq.num_computed_tokens = 10
        seq.append_token(5)
        assert seq.num_uncomputed_tokens == 1
        assert not seq.is_prefill

    def test_stops_on_a_stop_token(self):
        seq = make_sequence(0, max_tokens=100)
        seq.append_token(999)
        assert seq.check_stop()
        assert seq.finish_reason == "stop"

    def test_stops_on_the_length_cap(self):
        seq = make_sequence(0, max_tokens=2)
        seq.append_token(1)
        assert not seq.check_stop()
        seq.append_token(2)
        assert seq.check_stop()
        assert seq.finish_reason == "length"

    def test_recompute_reset_keeps_generated_tokens(self):
        """Only the KV is dropped, never the output.

        On resume the sequence re-prefills prompt plus output and continues from
        where it was, which is why a preempted sequence produces identical
        tokens to one that was never touched.
        """
        seq = make_sequence(0, prompt_len=4)
        seq.num_computed_tokens = 4
        seq.append_token(7)
        seq.append_token(8)
        seq.reset_for_recompute()
        assert seq.output_token_ids == [7, 8]
        assert seq.num_computed_tokens == 0
        assert seq.num_uncomputed_tokens == 6  # prompt 4 + output 2
        assert seq.num_preemptions == 1
        assert seq.status is SequenceStatus.WAITING


class TestAdmission:
    def test_admits_what_fits(self):
        scheduler = make_scheduler(num_blocks=16)
        for i in range(3):
            scheduler.add_request(make_sequence(i))
        output = scheduler.schedule()
        assert len(output.prefills) == 3
        assert len(scheduler.running) == 3

    def test_sequence_budget_caps_the_batch(self):
        scheduler = make_scheduler(num_blocks=64, max_num_seqs=2)
        for i in range(5):
            scheduler.add_request(make_sequence(i))
        output = scheduler.schedule()
        assert output.batch_size == 2
        assert len(scheduler.waiting) == 3

    def test_token_budget_caps_the_batch(self):
        """A step with one 2000-token prefill is not the same work as one decode.

        Budgeting on sequences alone produces wildly uneven step times.
        """
        scheduler = make_scheduler(num_blocks=64, max_num_batched_tokens=20)
        for i in range(4):
            scheduler.add_request(make_sequence(i, prompt_len=12))
        output = scheduler.schedule()
        assert output.num_batched_tokens <= 20
        assert output.batch_size == 1

    def test_both_budgets_apply_at_once(self):
        scheduler = make_scheduler(num_blocks=64, max_num_seqs=3, max_num_batched_tokens=8)
        for i in range(5):
            scheduler.add_request(make_sequence(i, prompt_len=4))
        output = scheduler.schedule()
        assert output.batch_size <= 3
        assert output.num_batched_tokens <= 8

    def test_defers_when_blocks_run_out(self):
        scheduler = make_scheduler(num_blocks=4)
        for i in range(4):
            scheduler.add_request(make_sequence(i, prompt_len=8))  # 2 blocks each
        output = scheduler.schedule()
        assert output.batch_size < 4
        assert scheduler.waiting

    def test_rejects_a_request_that_can_never_fit(self):
        """NEVER means reject, not queue.

        Leaving it in the queue would block everything behind it forever.
        """
        scheduler = make_scheduler(num_blocks=4)  # 16 token slots total
        scheduler.add_request(make_sequence(0, prompt_len=100))
        output = scheduler.schedule()
        assert output.finished and output.finished[0].finish_reason == "too_long"
        assert not scheduler.waiting
        assert output.is_empty

    def test_admission_leaves_headroom_for_running_sequences(self):
        """A new arrival must not eat the block a running sequence is about to need.

        Admitting into that headroom guarantees either an allocation failure at
        the end of the step or a preemption on the very next one.
        """
        scheduler = make_scheduler(num_blocks=6)
        scheduler.add_request(make_sequence(0, prompt_len=4))
        advance(scheduler, scheduler.schedule())
        for i in range(1, 6):
            scheduler.add_request(make_sequence(i, prompt_len=4))
        for _ in range(10):
            output = scheduler.schedule()
            advance(scheduler, output)
            scheduler.block_manager.check_invariants()


class TestRetirement:
    def test_frees_blocks_in_the_same_step(self):
        """Retiring late wastes a whole step of headroom per completion."""
        scheduler = make_scheduler(num_blocks=8)
        scheduler.add_request(make_sequence(0, prompt_len=4, max_tokens=1))
        advance(scheduler, scheduler.schedule())
        assert scheduler.running[0].status is SequenceStatus.FINISHED

        used_before = scheduler.block_manager.num_used_blocks
        output = scheduler.schedule()
        assert output.finished
        assert scheduler.block_manager.num_used_blocks < used_before
        assert not scheduler.running

    def test_a_finished_sequence_frees_capacity_for_a_waiting_one(self):
        scheduler = make_scheduler(num_blocks=4)
        scheduler.add_request(make_sequence(0, prompt_len=8, max_tokens=1))
        scheduler.add_request(make_sequence(1, prompt_len=8))
        first = scheduler.schedule()
        assert len(first.prefills) == 1  # only one fits
        advance(scheduler, first)

        second = scheduler.schedule()
        assert second.finished  # the first retired
        assert second.prefills  # and the second got in on the same step


class TestPreemption:
    def test_victim_is_the_most_recently_admitted(self):
        sequences = [make_sequence(i) for i in range(3)]
        for s in sequences:
            s.status = SequenceStatus.RUNNING
        assert select_victim(sequences) is sequences[-1]

    def test_no_victim_when_nothing_is_running(self):
        assert select_victim([]) is None
        finished = make_sequence(0)
        finished.status = SequenceStatus.FINISHED
        assert select_victim([finished]) is None

    def test_recompute_requeues_and_the_sequence_still_finishes(self):
        scheduler = make_scheduler(num_blocks=6, policy="recompute")
        sequences = [make_sequence(i, prompt_len=4, max_tokens=12) for i in range(4)]
        for sequence in sequences:
            scheduler.add_request(sequence)
        for _ in range(400):
            if not scheduler.num_unfinished:
                break
            advance(scheduler, scheduler.schedule())
            scheduler.block_manager.check_invariants()
        assert scheduler.num_preemptions > 0
        assert all(len(s.output_token_ids) == 12 for s in sequences)

    def test_swap_parks_blocks_and_brings_them_back(self):
        # Terminating sequences on purpose. Under LIFO preemption a swapped
        # sequence waits for the running set to drain, so with unbounded
        # generation it would legitimately never come back -- that is the
        # fairness tradeoff documented in core/policy.py, not a bug.
        # Pressure has to develop *after* admission. Admission reserves a
        # block of growth headroom per sequence, so it defers rather than
        # admitting into a shortfall -- preemption is triggered by sequences
        # outgrowing what was reserved, not by the initial admission.
        swap = StubSwapSpace()
        scheduler = make_scheduler(num_blocks=6, policy="swap", swap=swap)
        for i in range(4):
            scheduler.add_request(make_sequence(i, prompt_len=4, max_tokens=12))
        for _ in range(200):
            if not scheduler.num_unfinished:
                break
            advance(scheduler, scheduler.schedule())
            scheduler.block_manager.check_invariants()
        assert swap.swapped_out > 0
        assert swap.swapped_in > 0
        assert scheduler.num_unfinished == 0

    def test_swap_falls_back_to_recompute_when_host_space_runs_out(self):
        """Host memory is finite too, and the fallback must not be silent."""
        swap = StubSwapSpace(capacity=0)
        scheduler = make_scheduler(num_blocks=6, policy="swap", swap=swap)
        for i in range(4):
            scheduler.add_request(make_sequence(i, prompt_len=4, max_tokens=12))
        for _ in range(400):
            if not scheduler.num_unfinished:
                break
            advance(scheduler, scheduler.schedule())
        assert swap.swapped_out == 0
        assert scheduler.num_preemptions > 0

    def test_swap_policy_requires_swap_space(self):
        manager = BlockManager(8, BLOCK_SIZE)
        with pytest.raises(ValueError, match="needs a SwapSpace"):
            Scheduler(SchedulerConfig(preemption_policy="swap"), manager)

    def test_preemption_never_loses_or_leaks_blocks(self):
        for policy, swap in (("recompute", None), ("swap", StubSwapSpace())):
            scheduler = make_scheduler(num_blocks=6, policy=policy, swap=swap)
            for i in range(4):
                scheduler.add_request(make_sequence(i, prompt_len=4, max_tokens=12))
            for _ in range(200):
                if not scheduler.num_unfinished:
                    break
                advance(scheduler, scheduler.schedule())
                scheduler.block_manager.check_invariants()
            assert scheduler.num_unfinished == 0, policy
            scheduler.block_manager.check_invariants()

    def test_all_sequences_still_finish_under_pressure(self):
        """Tail preemption must not starve anything outright."""
        scheduler = make_scheduler(num_blocks=6, policy="recompute")
        sequences = [make_sequence(i, prompt_len=4, max_tokens=8) for i in range(4)]
        for s in sequences:
            scheduler.add_request(s)
        for _ in range(400):
            if not scheduler.num_unfinished:
                break
            advance(scheduler, scheduler.schedule())
        assert all(s.status is SequenceStatus.FINISHED for s in sequences)
        assert all(len(s.output_token_ids) == 8 for s in sequences)


class TestPreemptionMode:
    def test_parses_names(self):
        assert PreemptionMode.parse("recompute") is PreemptionMode.RECOMPUTE
        assert PreemptionMode.parse("SWAP") is PreemptionMode.SWAP
        assert PreemptionMode.parse(PreemptionMode.SWAP) is PreemptionMode.SWAP

    def test_rejects_an_unknown_policy(self):
        with pytest.raises(ValueError, match="unknown preemption policy"):
            PreemptionMode.parse("teleport")


class TestRaggedComposition:
    def test_one_step_mixes_prefills_and_decodes(self):
        """The shape continuous batching produces and static batching cannot."""
        scheduler = make_scheduler(num_blocks=32)
        scheduler.add_request(make_sequence(0, prompt_len=8))
        advance(scheduler, scheduler.schedule())  # sequence 0 is now decoding

        scheduler.add_request(make_sequence(1, prompt_len=12))
        output = scheduler.schedule()
        assert len(output.prefills) == 1
        assert len(output.decodes) == 1
        # 12 prefill tokens + 1 decode token, not 2 x 12 padded.
        assert output.num_batched_tokens == 13

    def test_a_new_arrival_joins_without_waiting_for_the_batch_to_drain(self):
        scheduler = make_scheduler(num_blocks=32)
        scheduler.add_request(make_sequence(0, prompt_len=4, max_tokens=50))
        advance(scheduler, scheduler.schedule())
        advance(scheduler, scheduler.schedule())

        scheduler.add_request(make_sequence(1, prompt_len=4))
        output = scheduler.schedule()
        # Admitted on the very next iteration, mid-flight.
        assert any(s.seq_id == 1 for s in output.prefills)
        assert any(s.seq_id == 0 for s in output.decodes)
