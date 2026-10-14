"""Pure-PyTorch twins of the CUDA extension's ops.

Each function here has the same name, signature, and in-place semantics as the
matching entry point in ``csrc/``, and follows the same algorithm rather than
just producing the same answer: ``paged_attention_v1`` walks the block table one
block at a time with an online softmax, exactly as the kernel walks it one token
at a time. That makes this module useful twice over:

- It is what the kernel is diffed against on a GPU (``tests/test_attention_cuda.py``).
- It lets ``CudaPagedAttentionBackend`` run end to end on a CPU, so the backend's
  plumbing — the write path, the decode/prefill split, the metadata — is covered
  by the golden test on a laptop, leaving only the kernel itself GPU-gated.

It is not a fast path and is never selected by the engine on its own.
"""

from __future__ import annotations

import torch

__all__ = [
    "ReferenceOps",
    "copy_blocks",
    "paged_attention_v1",
    "reshape_and_cache",
    "swap_blocks",
]


def paged_attention_v1(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Decode attention read through block tables.

    Args:
        query: ``[num_seqs, num_heads, head_dim]``, one query token per sequence.
        key_cache: ``[num_blocks, block_size, num_kv_heads, head_dim]``.
        value_cache: Same layout as ``key_cache``.
        block_tables: ``[num_seqs, max_blocks]`` int64 physical block ids.
        seq_lens: ``[num_seqs]`` int32 tokens to attend over per sequence.
        scale: Softmax scale, normally ``head_dim ** -0.5``.

    Returns:
        ``[num_seqs, num_heads, head_dim]`` in ``query``'s dtype.
    """
    num_seqs, num_heads, head_dim = query.shape
    block_size, num_kv_heads = key_cache.shape[1], key_cache.shape[2]
    group = num_heads // num_kv_heads
    device = query.device

    # [n, kv_heads, group, dim]: query heads grouped under the KV head they read.
    q = query.float().reshape(num_seqs, num_kv_heads, group, head_dim) * scale
    running_max = torch.full((num_seqs, num_kv_heads, group), float("-inf"), device=device)
    running_sum = torch.zeros((num_seqs, num_kv_heads, group), device=device)
    acc = torch.zeros((num_seqs, num_kv_heads, group, head_dim), device=device)
    lens = seq_lens.to(device=device, dtype=torch.long).view(num_seqs, 1)
    offsets = torch.arange(block_size, device=device)

    for b in range(block_tables.shape[1]):
        blocks = block_tables[:, b]
        # Slots past seq_len hold a previous tenant's KV. The kernel never
        # loads them; zeroing them here keeps that true even for NaN garbage.
        valid = (b * block_size + offsets).view(1, block_size) < lens  # [n, block_size]
        token_mask = valid.view(num_seqs, block_size, 1, 1)
        k = key_cache[blocks].float().masked_fill(~token_mask, 0.0)  # [n, bs, kv_heads, dim]
        v = value_cache[blocks].float().masked_fill(~token_mask, 0.0)
        scores = torch.einsum("nhgd,nthd->nhgt", q, k)
        scores = scores.masked_fill(~valid.view(num_seqs, 1, 1, block_size), float("-inf"))

        new_max = torch.maximum(running_max, scores.amax(dim=-1))
        # A row with nothing valid yet keeps max -inf; shift by 0 so the
        # exponentials below are exp(-inf) == 0 rather than NaN.
        shift = torch.where(torch.isinf(new_max), torch.zeros_like(new_max), new_max)
        correction = torch.exp(running_max - shift)
        p = torch.exp(scores - shift.unsqueeze(-1))
        running_sum = running_sum * correction + p.sum(dim=-1)
        acc = acc * correction.unsqueeze(-1) + torch.einsum("nhgt,nthd->nhgd", p, v)
        running_max = new_max

    out = torch.where(
        running_sum.unsqueeze(-1) > 0,
        acc / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1),
        torch.zeros_like(acc),
    )
    return out.reshape(num_seqs, num_heads, head_dim).to(query.dtype)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Scatter ``[num_tokens, kv_heads, dim]`` K/V into the caches, in place.

    Tokens whose slot is negative are skipped, as in the kernel.
    """
    num_kv_heads, head_dim = key_cache.shape[2], key_cache.shape[3]
    keep = slot_mapping >= 0
    slots = slot_mapping[keep]
    key_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slots, key[keep])
    value_cache.view(-1, num_kv_heads, head_dim).index_copy_(0, slots, value[keep])


def copy_blocks(kv_cache: torch.Tensor, block_mapping: torch.Tensor) -> None:
    """Copy whole blocks within ``[layers, 2, blocks, ...]``, in place."""
    if block_mapping.numel() == 0:
        return
    mapping = block_mapping.to(kv_cache.device)
    src, dst = mapping[:, 0], mapping[:, 1]
    kv_cache[:, :, dst] = kv_cache[:, :, src]


def swap_blocks(src: torch.Tensor, dst: torch.Tensor, block_mapping: torch.Tensor) -> None:
    """Copy ``(src, dst)`` block pairs between two ``[blocks, ...]`` tensors."""
    if block_mapping.numel() == 0:
        return
    mapping = block_mapping.cpu()
    source = mapping[:, 0].to(src.device)
    destination = mapping[:, 1].to(dst.device)
    dst.index_copy_(0, destination, src.index_select(0, source).to(dst.device))


class ReferenceOps:
    """Drop-in for the ``pagedserve._C`` module, for tests and CPU runs."""

    paged_attention_v1 = staticmethod(paged_attention_v1)
    reshape_and_cache = staticmethod(reshape_and_cache)
    copy_blocks = staticmethod(copy_blocks)
    swap_blocks = staticmethod(swap_blocks)
