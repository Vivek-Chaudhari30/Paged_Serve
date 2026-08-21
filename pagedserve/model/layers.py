"""Transformer building blocks: RMSNorm, RoPE, GQA attention, SwiGLU.

Written by hand rather than imported, because the whole point of Phase 1 is
owning the place where K and V get written — the exact layer ``transformers``
abstracts away.

Numerics here match HuggingFace's reference implementation deliberately and
exactly. The golden test asserts token-for-token equality under greedy decoding,
so "close enough" is not a thing: a normalisation done in bf16 instead of fp32,
or a rotary convention off by an interleave, produces output that is plausible,
subtly wrong, and very hard to find later.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from pagedserve.attention.backend import AttentionBackend, AttentionMetadata
from pagedserve.config import ModelConfig

__all__ = ["Attention", "MLP", "RMSNorm", "RotaryEmbedding", "apply_rotary", "rotate_half"]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm, no mean subtraction and no bias.

    The float32 upcast is not optional. In bf16 the sum of 896 squared
    activations loses enough precision to move the argmax on near-ties, which
    shows up as a golden-test failure several tokens later with no obvious
    cause. HuggingFace upcasts here for the same reason; matching it is what
    makes the outputs comparable.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the halves of the last dimension: ``[a, b] -> [-b, a]``.

    This is the *split-half* convention, which is what HuggingFace's Llama and
    Qwen2 checkpoints are trained with. The alternative *interleaved* convention
    pairs adjacent elements instead. Both are valid rotary embeddings and they
    are not interchangeable — using the wrong one against trained weights gives
    fluent-looking garbage rather than an error.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key.

    Args:
        q: ``[batch, seq, num_q_heads, head_dim]``.
        k: ``[batch, seq, num_kv_heads, head_dim]``.
        cos, sin: ``[batch, seq, head_dim]``.

    Only Q and K are rotated; V carries no position information, which is why
    the KV cache can store V once and reuse it at any later position.
    """
    cos = cos.unsqueeze(2)  # [batch, seq, 1, head_dim] -- broadcast over heads
    sin = sin.unsqueeze(2)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEmbedding(nn.Module):
    """Precomputed inverse frequencies for RoPE.

    RoPE encodes position as a rotation in each 2D subspace of the head
    dimension, which means relative position falls out of the dot product
    automatically. That is what lets a cached key computed at step 5 stay valid
    forever: its rotation is absolute, and attention only ever sees differences.
    Learned positional embeddings have no such property, which is the reason
    every modern serving stack uses RoPE.
    """

    def __init__(self, head_dim: int, theta: float, device: torch.device):
        super().__init__()
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, position_ids: torch.Tensor, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build cos/sin for the given positions.

        Args:
            position_ids: ``[batch, seq]`` absolute positions. Under left
                padding these are *not* ``arange`` — pad slots must not consume
                positions, or every real token shifts and RoPE silently
                disagrees with the reference.

        Returns:
            ``cos`` and ``sin``, each ``[batch, seq, head_dim]``.
        """
        # Computed in float32 regardless of model dtype: these are angles, and
        # bf16 has ~3 decimal digits, which is not enough at position 30000.
        freqs = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


class MLP(nn.Module):
    """SwiGLU feed-forward: ``down(silu(gate(x)) * up(x))``.

    The gate is what makes it "gated linear": one branch decides how much of the
    other branch survives. Costs a third projection versus a plain FFN and buys
    enough quality that every model in this family uses it.
    """

    def __init__(self, config: ModelConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        kwargs = {"bias": False, "dtype": dtype, "device": device}
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, **kwargs)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, **kwargs)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    """Grouped-query attention. Delegates all KV storage to the backend.

    GQA gives several query heads one shared KV head. That ratio divides the KV
    cache size directly — Qwen2.5-0.5B's 14:2 means the cache is 7x smaller than
    multi-head attention would need — which is the cheapest possible win on the
    exact resource this project is about.

    This class never touches a KV cache. It computes Q/K/V, applies RoPE, and
    hands them to the backend. That is what lets the paged and CUDA backends
    replace the storage layer in later phases without editing this file.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.num_q_heads = config.num_q_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        q_size = config.num_q_heads * config.head_dim
        kv_size = config.num_kv_heads * config.head_dim
        common = {"dtype": dtype, "device": device}
        # Qwen2 biases Q/K/V and never the output projection; Llama biases none
        # by default. Read from the checkpoint's config rather than branched on
        # a model name.
        self.q_proj = nn.Linear(config.hidden_size, q_size, bias=config.attention_bias, **common)
        self.k_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias, **common)
        self.v_proj = nn.Linear(config.hidden_size, kv_size, bias=config.attention_bias, **common)
        self.o_proj = nn.Linear(q_size, config.hidden_size, bias=False, **common)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        backend: AttentionBackend,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        batch, seq, _ = hidden.shape

        q = self.q_proj(hidden).view(batch, seq, self.num_q_heads, self.head_dim)
        k = self.k_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(batch, seq, self.num_kv_heads, self.head_dim)

        q, k = apply_rotary(q, k, cos, sin)

        attn = backend.forward(self.layer_idx, q, k, v, metadata)
        return self.o_proj(attn.reshape(batch, seq, self.num_q_heads * self.head_dim))
