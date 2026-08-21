"""Deciding how many KV blocks fit, then reserving them once.

The sizing question is "how much memory is left after the weights and the peak
activation footprint, and how many blocks is that?" Answering it by profiling
rather than guessing is what lets the cache saturate the card without OOMing at
3am under a burst — a hardcoded fraction is either wasteful on a big card or
fatal on a small one.

The arithmetic is a pure function and the measurement is separate, so the part
that can be wrong in an interesting way is testable on a laptop with no GPU.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch

from pagedserve.config import CacheConfig, ModelConfig

logger = logging.getLogger(__name__)

__all__ = ["blocks_from_budget", "bytes_per_block", "profile_num_blocks"]


def bytes_per_block(model: ModelConfig, block_size: int, dtype: torch.dtype) -> int:
    """Bytes one physical block costs across every layer.

    A block holds ``block_size`` tokens, and a token costs K and V in every
    layer, so this is just ``block_size x kv_bytes_per_token``.
    """
    return block_size * model.kv_bytes_per_token(dtype)


def blocks_from_budget(
    *,
    total_bytes: int,
    weights_bytes: int,
    peak_activation_bytes: int,
    utilization: float,
    block_bytes: int,
) -> int:
    """How many blocks fit in what is left over. Pure arithmetic.

    ``utilization`` is applied to the *total* device memory rather than to the
    remainder, so the fraction means "how much of this card am I willing to
    occupy" — which is the question an operator actually asks. The headroom it
    leaves absorbs allocator fragmentation and the CUDA context, neither of
    which appears in ``weights + activations``.

    Raises:
        ValueError: If nothing is left. Failing loudly at startup beats
            allocating a two-block cache and thrashing forever.
    """
    if block_bytes <= 0:
        raise ValueError(f"block_bytes must be positive, got {block_bytes}")
    budget = int(total_bytes * utilization) - weights_bytes - peak_activation_bytes
    if budget < block_bytes:
        raise ValueError(
            f"no room for a KV cache: {int(total_bytes * utilization)} bytes of budget "
            f"minus {weights_bytes} of weights and {peak_activation_bytes} of peak "
            f"activation leaves {budget}, less than one {block_bytes}-byte block. "
            f"Use a smaller model, a smaller max batch, or raise "
            f"gpu_memory_utilization."
        )
    return budget // block_bytes


def profile_num_blocks(
    model: ModelConfig,
    cache: CacheConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    weights_bytes: int,
    run_max_shape_forward: Callable[[], None] | None = None,
) -> int:
    """Measure what is free, then derive the block count.

    On CUDA this runs a forward pass at the largest shape the engine will ever
    see and records peak allocation, because activation memory is a real claim
    on the card that no static calculation predicts reliably — it depends on the
    attention implementation, on autograd being off, and on whatever workspace
    cuBLAS decides it wants.

    Off CUDA there is no equivalent measurement: ``torch`` cannot report a
    meaningful free-memory figure for host RAM, and inventing one would put a
    fabricated number at the root of every capacity decision. So a non-CUDA
    device requires ``CacheConfig.num_blocks_override``, and says so.
    """
    if cache.num_blocks_override is not None:
        logger.info("using num_blocks override: %d", cache.num_blocks_override)
        return cache.num_blocks_override

    if device.type != "cuda":
        raise ValueError(
            f"cannot profile KV cache capacity on device {device.type!r}: only CUDA "
            f"reports free device memory. Set CacheConfig.num_blocks_override to "
            f"size the cache explicitly on this device."
        )

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if run_max_shape_forward is not None:
        run_max_shape_forward()
        torch.cuda.synchronize()

    peak_activation = max(0, torch.cuda.max_memory_allocated() - weights_bytes)
    total = torch.cuda.get_device_properties(device).total_memory
    block_bytes = bytes_per_block(model, cache.block_size, dtype)

    num_blocks = blocks_from_budget(
        total_bytes=total,
        weights_bytes=weights_bytes,
        peak_activation_bytes=peak_activation,
        utilization=cache.gpu_memory_utilization,
        block_bytes=block_bytes,
    )
    logger.info(
        "profiled KV capacity: %.1f GiB total, %.1f GiB weights, %.1f GiB peak "
        "activation, utilization %.2f -> %d blocks (%d token slots)",
        total / 2**30,
        weights_bytes / 2**30,
        peak_activation / 2**30,
        cache.gpu_memory_utilization,
        num_blocks,
        num_blocks * cache.block_size,
    )
    return num_blocks
