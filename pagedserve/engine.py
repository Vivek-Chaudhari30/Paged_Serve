"""The Phase 1 engine: static batching, contiguous cache, greedy decode.

This is deliberately the naive design, built to last rather than to be thrown
away. It becomes the ``--no-paging`` ablation arm, and an ablation arm is only
worth having if it is a fair implementation of the thing it represents.

Two properties of static batching are being demonstrated, not worked around:

**Reservation waste.** Every sequence slot reserves ``max_seq_len`` tokens of KV
at startup. A request that generates 40 tokens holds room for 2048 for its
entire life. That reservation is the utilization number this phase exists to
produce.

**Head-of-line blocking.** The batch runs until its longest member finishes. A
sequence that hits EOS at step 10 of a 200-step batch keeps its slot, keeps its
memory, and contributes padding compute for 190 more steps. Phase 3 fixes this;
Phase 1 measures it.

Phase 3 replaces ``generate()`` with an iteration-level ``step()``. The name
``LLMEngine`` is kept from the start so the seam is where AGENTS.md §3 says it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from pagedserve.attention.backend import AttentionBackend, KVMemoryStats, StepInput
from pagedserve.attention.contiguous import ContiguousAttentionBackend
from pagedserve.attention.gather import PagedAttentionBackend
from pagedserve.config import CacheConfig, EngineConfig, ModelConfig
from pagedserve.memory.block_manager import AllocStatus, BlockManager
from pagedserve.model.llama import CausalLM
from pagedserve.model.loader import load_state_dict, resolve_model_path
from pagedserve.model.sampler import greedy
from pagedserve.worker.cache_engine import profile_num_blocks

logger = logging.getLogger(__name__)

__all__ = ["GenerationOutput", "LLMEngine"]


@dataclass
class GenerationOutput:
    """Generated tokens plus the memory evidence from the run."""

    token_ids: list[list[int]]
    finish_reasons: list[str]
    step_stats: list[KVMemoryStats] = field(default_factory=list)

    @property
    def final_stats(self) -> KVMemoryStats | None:
        """Utilization at the end of the run, when the cache is fullest."""
        return self.step_stats[-1] if self.step_stats else None

    def utilization_report(self) -> str:
        """A human-readable summary of the reserved-versus-live picture.

        This ratio is the entire point of Phase 1. Seeing it with your own eyes
        is what turns "contiguous allocation is wasteful" from a claim in a
        paper into a number about your own machine.
        """
        stats = self.final_stats
        if stats is None:
            return "no steps recorded"
        util = stats.utilization
        return (
            f"KV cache: {stats.allocated_bytes / 2**20:.1f} MiB allocated, "
            f"{stats.live_bytes / 2**20:.1f} MiB live, "
            f"{stats.wasted_bytes / 2**20:.1f} MiB wasted "
            f"({util:.1%} utilization)"
            if util is not None
            else "no cache allocated"
        )


BACKENDS = ("contiguous", "gather")


def _build_backend(
    config: EngineConfig, *, weights_bytes: int = 0
) -> tuple[AttentionBackend, BlockManager | None]:
    """Construct the configured backend and, if it pages, its allocator.

    The engine owns the BlockManager rather than the backend, because Phase 3's
    scheduler needs it to make admission and preemption decisions and those are
    not the storage layer's business.
    """
    if config.attn_backend == "contiguous":
        # --no-paging. The Phase 1 arm, kept working forever so that the paged
        # comparison is a controlled experiment with exactly one variable.
        return (
            ContiguousAttentionBackend(
                config.model, config.cache, device=config.device, dtype=config.dtype
            ),
            None,
        )
    if config.attn_backend == "gather":
        num_blocks = profile_num_blocks(
            config.model,
            config.cache,
            device=config.device,
            dtype=config.dtype,
            weights_bytes=weights_bytes,
        )
        manager = BlockManager(num_blocks, config.cache.block_size)
        backend = PagedAttentionBackend(
            config.model,
            config.cache,
            num_blocks=num_blocks,
            device=config.device,
            dtype=config.dtype,
        )
        # Both derive the trash block id from num_blocks. Assert rather than
        # trust: a mismatch would scribble padding K/V into a real sequence's
        # last block, and the symptom would be wrong tokens, not a crash.
        assert backend.trash_block_id == manager.trash_block_id
        return backend, manager
    raise ValueError(f"unknown attention backend {config.attn_backend!r} (available: {BACKENDS})")


class LLMEngine:
    """Owns the model, the KV cache, and the generation loop."""

    def __init__(
        self,
        config: EngineConfig,
        model: CausalLM,
        backend: AttentionBackend,
        block_manager: BlockManager | None = None,
    ):
        self.config = config
        self.model = model
        self.backend = backend
        self.block_manager = block_manager

    @property
    def is_paged(self) -> bool:
        return self.block_manager is not None

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        device: str | None = None,
        dtype: str | None = None,
        cache: CacheConfig | None = None,
        attn_backend: str = "gather",
        debug_invariants: bool = False,
    ) -> LLMEngine:
        """Load a checkpoint and build an engine around it.

        Defaults to the paged backend. ``attn_backend="contiguous"`` selects the
        ``--no-paging`` ablation arm.
        """
        path = resolve_model_path(model)
        model_config = ModelConfig.from_pretrained(path, name=str(model))
        config = EngineConfig.build(
            model_config,
            device=device,
            dtype=dtype,
            cache=cache,
            attn_backend=attn_backend,
            debug_invariants=debug_invariants,
        )

        module = CausalLM(model_config, config.dtype, config.device)
        module.load_weights(load_state_dict(path, dtype=config.dtype, device=config.device))
        module.eval()

        weights_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
        backend, manager = _build_backend(config, weights_bytes=weights_bytes)
        return cls(config, module, backend, manager)

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[list[int]],
        max_tokens: int,
        *,
        eos_token_ids: tuple[int, ...] | None = None,
    ) -> GenerationOutput:
        """Generate greedily for a whole batch at once, left-padded.

        Args:
            prompt_token_ids: One list of token ids per request.
            max_tokens: Cap on generated tokens per request.
            eos_token_ids: Stop tokens. Defaults to the checkpoint's.

        Returns:
            Generated ids per request, why each stopped, and the per-step KV
            memory statistics.
        """
        if not prompt_token_ids:
            return GenerationOutput(token_ids=[], finish_reasons=[])

        eos = eos_token_ids if eos_token_ids is not None else self.config.model.eos_token_ids
        device = self.config.device
        batch = len(prompt_token_ids)
        prompt_lens = [len(ids) for ids in prompt_token_ids]
        max_prompt = max(prompt_lens)
        total_needed = max_prompt + max_tokens

        cache_len = self.config.cache.max_seq_len
        if total_needed > cache_len:
            raise ValueError(
                f"prompt ({max_prompt}) + max_tokens ({max_tokens}) exceeds "
                f"CacheConfig.max_seq_len ({cache_len})"
            )
        if batch > self.config.cache.max_num_seqs:
            raise ValueError(
                f"batch of {batch} exceeds CacheConfig.max_num_seqs "
                f"({self.config.cache.max_num_seqs})"
            )

        self.backend.allocate()
        if self.block_manager is not None:
            self.block_manager.reset()
            for seq_id, length in enumerate(prompt_lens):
                status = self.block_manager.can_allocate(length + max_tokens)
                if status is AllocStatus.NEVER:
                    raise ValueError(
                        f"sequence {seq_id} needs more blocks than the cache holds "
                        f"({length + max_tokens} tokens); raise num_blocks or lower "
                        f"max_tokens"
                    )
                # Phase 2 still batches statically, so every sequence is
                # admitted up front. Phase 3 is where LATER stops meaning
                # "fail" and starts meaning "wait in the queue".
                self.block_manager.allocate(seq_id, length)

        input_ids, padding_mask, position_ids = self._left_pad(prompt_token_ids, cache_len)

        # Real tokens per sequence: the live half of the utilization ratio.
        seq_lens = torch.tensor(prompt_lens, dtype=torch.int32, device=device)

        outputs: list[list[int]] = [[] for _ in range(batch)]
        finish_reasons = ["length"] * batch
        finished = [False] * batch
        step_stats: list[KVMemoryStats] = []

        # ---- prefill: the whole padded prompt in one pass ----
        step = self._build_step(
            query_len=max_prompt,
            context_len=0,
            seq_lens=seq_lens,
            padding_mask=padding_mask[:, :max_prompt],
            query_positions=position_ids[:, :max_prompt],
            is_prefill=True,
            token_is_real=padding_mask[:, :max_prompt],
            logical_starts=[0] * batch,
        )
        metadata = self.backend.begin_step(step)
        logits = self.model(
            input_ids[:, :max_prompt], position_ids[:, :max_prompt], self.backend, metadata
        )
        step_stats.append(self.backend.memory_stats())
        next_tokens = greedy(logits)

        # ---- decode: one token per sequence per step ----
        for offset in range(max_tokens):
            position = max_prompt + offset
            token_list = next_tokens.tolist()
            for i, token in enumerate(token_list):
                if finished[i]:
                    continue
                outputs[i].append(token)
                if token in eos:
                    finished[i] = True
                    finish_reasons[i] = "stop"
                elif len(outputs[i]) >= max_tokens:
                    finished[i] = True

            if all(finished):
                break
            if offset == max_tokens - 1:
                break

            # A finished sequence still occupies its slot and still runs through
            # the model. That wasted compute is head-of-line blocking, and it is
            # the thing Phase 3's scheduler removes.
            active = torch.tensor([not f for f in finished], dtype=torch.int32, device=device)
            seq_lens = seq_lens + active

            input_ids[:, position] = next_tokens
            padding_mask[:, position] = True
            # Under left padding each sequence's real position differs, so the
            # next position is its own length, not the shared step index.
            position_ids[:, position] = seq_lens - 1

            if self.block_manager is not None:
                # One allocator call per sequence per step, and it only touches
                # the free list when a block boundary is crossed -- once every
                # block_size tokens.
                for seq_id in range(batch):
                    self.block_manager.append_slot(seq_id, int(seq_lens[seq_id]))

            step = self._build_step(
                query_len=1,
                context_len=position,
                seq_lens=seq_lens,
                padding_mask=padding_mask[:, : position + 1],
                query_positions=position_ids[:, position : position + 1],
                is_prefill=False,
                token_is_real=torch.ones((batch, 1), dtype=torch.bool, device=device),
                logical_starts=[int(seq_lens[i]) - 1 for i in range(batch)],
            )
            metadata = self.backend.begin_step(step)
            logits = self.model(
                input_ids[:, position : position + 1],
                position_ids[:, position : position + 1],
                self.backend,
                metadata,
            )
            step_stats.append(self.backend.memory_stats())
            next_tokens = greedy(logits)

        result = GenerationOutput(
            token_ids=outputs, finish_reasons=finish_reasons, step_stats=step_stats
        )
        logger.info("%s", result.utilization_report())
        return result

    def _build_step(
        self,
        *,
        query_len: int,
        context_len: int,
        seq_lens: torch.Tensor,
        padding_mask: torch.Tensor,
        query_positions: torch.Tensor,
        is_prefill: bool,
        token_is_real: torch.Tensor,
        logical_starts: list[int],
    ) -> StepInput:
        """Describe this step for whichever backend is installed.

        The dense backend needs nothing beyond lengths and a mask. The paged one
        additionally needs to know where every token in the batch should be
        written (``slot_mapping``) and which physical blocks each sequence owns
        (``block_tables``). Building those here rather than inside the backend
        keeps the allocator's knowledge in one place.

        Args:
            token_is_real: ``[num_seqs, query_len]``, False for left padding.
                Padding still flows through the model and produces K and V, so
                it needs a destination that is not a real sequence's cache.
            logical_starts: Logical position within each sequence at which this
                step's first *real* token lands.
        """
        if self.block_manager is None:
            return StepInput(
                query_len=query_len,
                context_len=context_len,
                seq_lens=seq_lens,
                padding_mask=padding_mask,
                query_positions=query_positions,
                is_prefill=is_prefill,
            )

        manager = self.block_manager
        device = self.config.device
        num_seqs = int(token_is_real.shape[0])
        real = token_is_real.tolist()

        trash = manager.trash_block_id * manager.block_size
        slots: list[int] = []
        for seq_id in range(num_seqs):
            logical = logical_starts[seq_id]
            table = manager.block_table(seq_id)
            for j in range(query_len):
                if not real[seq_id][j]:
                    # Padding goes to the trash block, keeping the write path a
                    # single fused scatter instead of a per-sequence branch.
                    slots.append(trash)
                    continue
                slots.append(table.slot(logical))
                logical += 1

        max_blocks = max(len(manager.block_table(i)) for i in range(num_seqs))
        tables = []
        for seq_id in range(num_seqs):
            blocks = list(manager.block_table(seq_id))
            # Short tables are padded with the trash block. Those positions are
            # masked out by seq_lens, so what they point at is never read -- but
            # it must not be another sequence's live block.
            blocks += [manager.trash_block_id] * (max_blocks - len(blocks))
            tables.append(blocks)

        manager_used = manager.num_used_blocks
        if hasattr(self.backend, "set_used_blocks"):
            self.backend.set_used_blocks(manager_used)
        if self.config.debug_invariants:
            manager.check_invariants({i: int(seq_lens[i]) for i in range(num_seqs)})

        return StepInput(
            query_len=query_len,
            context_len=context_len,
            seq_lens=seq_lens,
            padding_mask=padding_mask,
            query_positions=query_positions,
            is_prefill=is_prefill,
            block_tables=torch.tensor(tables, dtype=torch.long, device=device),
            slot_mapping=torch.tensor(slots, dtype=torch.long, device=device),
        )

    def _left_pad(
        self, prompt_token_ids: list[list[int]], cache_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-pad prompts and build the matching mask and positions.

        Left rather than right padding, because it puts every sequence's newest
        token at the same index. That is what lets one ``write_start`` cover the
        whole batch, and it is why the decode step is a single slice assignment
        instead of a Python loop over sequences.

        Position ids are derived from the mask, not from ``arange``: a pad slot
        must not consume a position, or every real token in a short sequence
        shifts and RoPE quietly disagrees with the reference implementation.
        """
        device = self.config.device
        batch = len(prompt_token_ids)
        max_prompt = max(len(ids) for ids in prompt_token_ids)

        input_ids = torch.zeros((batch, cache_len), dtype=torch.long, device=device)
        padding_mask = torch.zeros((batch, cache_len), dtype=torch.bool, device=device)
        position_ids = torch.zeros((batch, cache_len), dtype=torch.long, device=device)

        for i, ids in enumerate(prompt_token_ids):
            start = max_prompt - len(ids)
            input_ids[i, start:max_prompt] = torch.tensor(ids, dtype=torch.long, device=device)
            padding_mask[i, start:max_prompt] = True
            position_ids[i, start:max_prompt] = torch.arange(
                len(ids), dtype=torch.long, device=device
            )
        return input_ids, padding_mask, position_ids

    def config_dict(self) -> dict[str, Any]:
        """The engine config, for a benchmark result file's ``config`` block."""
        return self.config.to_dict()
