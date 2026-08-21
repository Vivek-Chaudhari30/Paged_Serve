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
from pagedserve.config import CacheConfig, EngineConfig, ModelConfig
from pagedserve.model.llama import CausalLM
from pagedserve.model.loader import load_state_dict, resolve_model_path
from pagedserve.model.sampler import greedy

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


def _build_backend(config: EngineConfig) -> AttentionBackend:
    if config.attn_backend == "contiguous":
        return ContiguousAttentionBackend(config.model, device=config.device, dtype=config.dtype)
    raise ValueError(f"unknown attention backend {config.attn_backend!r} (available: contiguous)")


class LLMEngine:
    """Owns the model, the KV cache, and the generation loop."""

    def __init__(self, config: EngineConfig, model: CausalLM, backend: AttentionBackend):
        self.config = config
        self.model = model
        self.backend = backend

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path,
        *,
        device: str | None = None,
        dtype: str | None = None,
        cache: CacheConfig | None = None,
        attn_backend: str = "contiguous",
    ) -> LLMEngine:
        """Load a checkpoint and build an engine around it."""
        path = resolve_model_path(model)
        model_config = ModelConfig.from_pretrained(path, name=str(model))
        config = EngineConfig.build(
            model_config,
            device=device,
            dtype=dtype,
            cache=cache,
            attn_backend=attn_backend,
        )

        module = CausalLM(model_config, config.dtype, config.device)
        module.load_weights(load_state_dict(path, dtype=config.dtype, device=config.device))
        module.eval()

        backend = _build_backend(config)
        return cls(config, module, backend)

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

        # The naive reservation: every slot gets room for max_seq_len tokens
        # regardless of what it will use. Sizing this to the batch's actual need
        # would understate exactly the waste this arm exists to measure.
        self.backend.allocate(num_seq_slots=batch, max_seq_len=cache_len)

        input_ids, padding_mask, position_ids = self._left_pad(prompt_token_ids, cache_len)

        # Real tokens per sequence: the live half of the utilization ratio.
        seq_lens = torch.tensor(prompt_lens, dtype=torch.int32, device=device)

        outputs: list[list[int]] = [[] for _ in range(batch)]
        finish_reasons = ["length"] * batch
        finished = [False] * batch
        step_stats: list[KVMemoryStats] = []

        # ---- prefill: the whole padded prompt in one pass ----
        step = StepInput(
            query_len=max_prompt,
            context_len=0,
            seq_lens=seq_lens,
            padding_mask=padding_mask[:, :max_prompt],
            is_prefill=True,
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

            step = StepInput(
                query_len=1,
                context_len=position,
                seq_lens=seq_lens,
                padding_mask=padding_mask[:, : position + 1],
                is_prefill=False,
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
