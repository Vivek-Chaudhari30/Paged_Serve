"""The manual forward pass, with KV cache write hooks.

Handles the Llama-style family — RoPE, GQA, SwiGLU, RMSNorm — which includes
Qwen2. Roughly 150 lines of transformer, written out so there is a place to
stand when attention numerics need debugging at 2am, and so the K/V write point
is ours rather than buried in a library's cache abstraction.

Weight loading is strict in both directions: an unexpected key and a missing key
are both errors. A checkpoint whose bias terms silently went unloaded still
runs, still produces fluent text, and is wrong — the golden test would catch it,
but a clear error at load time is a much shorter debugging path than a token
mismatch at position 37.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from pagedserve.attention.backend import AttentionBackend, AttentionMetadata
from pagedserve.config import ModelConfig
from pagedserve.model.layers import MLP, Attention, RMSNorm, RotaryEmbedding

logger = logging.getLogger(__name__)

__all__ = ["CausalLM", "DecoderLayer"]


class DecoderLayer(nn.Module):
    """Pre-norm transformer block: norm -> attention -> add, norm -> MLP -> add.

    Pre-norm (normalising the input to each sublayer rather than the output)
    keeps a clean residual path from embedding to logits, which is what makes
    deep stacks trainable without warmup tricks. It also means the residual
    stream is never normalised, so activations grow with depth — relevant here
    only because it is why the final ``model.norm`` exists.
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.self_attn = Attention(config, layer_idx, dtype, device)
        self.mlp = MLP(config, dtype, device)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, device)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, dtype, device
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        backend: AttentionBackend,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(self.input_layernorm(hidden), cos, sin, backend, metadata)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class CausalLM(nn.Module):
    """A Llama-style decoder stack with a language-model head."""

    def __init__(self, config: ModelConfig, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.device = device

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config, i, dtype, device) for i in range(config.num_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype, device)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta, device)

        if config.tie_word_embeddings:
            # Sharing the matrix rather than copying it: on a 0.5B model with a
            # 152k vocab the embedding is 136M parameters, a quarter of the
            # whole model. Duplicating it to make the code uniform would be a
            # measurable waste of the memory this project is about.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.hidden_size, config.vocab_size, bias=False, dtype=dtype, device=device
            )

    def load_weights(self, state: dict[str, torch.Tensor]) -> None:
        """Map a HuggingFace state dict onto this module, strictly.

        The checkpoint's names are kept as the source of truth and translated
        here, rather than naming our modules to match, so the mapping is one
        explicit table instead of an implicit dependency on a naming scheme we
        do not control.
        """
        mapping: dict[str, torch.Tensor] = {}
        remaining = dict(state)

        def take(key: str) -> torch.Tensor:
            if key not in remaining:
                raise KeyError(f"checkpoint is missing {key!r}")
            return remaining.pop(key)

        mapping["embed_tokens.weight"] = take("model.embed_tokens.weight")
        mapping["norm.weight"] = take("model.norm.weight")
        if not self.config.tie_word_embeddings:
            mapping["lm_head.weight"] = take("lm_head.weight")

        for i in range(self.config.num_layers):
            src = f"model.layers.{i}."
            dst = f"layers.{i}."
            mapping[dst + "input_layernorm.weight"] = take(src + "input_layernorm.weight")
            mapping[dst + "post_attention_layernorm.weight"] = take(
                src + "post_attention_layernorm.weight"
            )
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mapping[dst + f"self_attn.{proj}.weight"] = take(src + f"self_attn.{proj}.weight")
                if self.config.attention_bias and proj != "o_proj":
                    mapping[dst + f"self_attn.{proj}.bias"] = take(src + f"self_attn.{proj}.bias")
            for proj in ("gate_proj", "up_proj", "down_proj"):
                mapping[dst + f"mlp.{proj}.weight"] = take(src + f"mlp.{proj}.weight")

        if remaining:
            raise KeyError(
                f"checkpoint has {len(remaining)} unmapped tensors, e.g. "
                f"{sorted(remaining)[:5]}. Refusing to load a model we do not "
                f"fully understand."
            )

        missing, unexpected = self.load_state_dict(mapping, strict=False)
        # tie_word_embeddings means lm_head has no parameters of its own.
        missing = [k for k in missing if not k.startswith("rotary.")]
        if missing or unexpected:
            raise RuntimeError(f"weight load mismatch: missing={missing} unexpected={unexpected}")
        logger.info("loaded %d tensors into the model", len(mapping))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        backend: AttentionBackend,
        metadata: AttentionMetadata,
        logits_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the stack and return logits for the final position only.

        Args:
            input_ids: ``[batch, seq]``.
            position_ids: ``[batch, seq]`` absolute positions for RoPE.
            backend: Where KV lives.
            metadata: Per-step backend state from ``begin_step``.
            logits_indices: Flat positions to project, for a ragged batch where
                every sequence ends somewhere different. ``None`` means the last
                position of each row, which is what lockstep static batching
                wants.

        Returns:
            ``[num_sampled, vocab_size]``.

        Only the sampled positions are projected. During prefill the other
        positions' logits are never read, and an LM head over a 152k vocabulary
        is expensive enough that computing them all would dominate prefill time.
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary(position_ids, self.dtype)

        for layer in self.layers:
            hidden = layer(hidden, cos, sin, backend, metadata)

        if logits_indices is None:
            selected = hidden[:, -1:, :]
        else:
            # Ragged: one row per sequence, taken from the flattened batch.
            selected = hidden.view(-1, hidden.shape[-1])[logits_indices].unsqueeze(0)

        selected = self.norm(selected)
        weight = self.embed_tokens.weight if self.lm_head is None else self.lm_head.weight
        logits = torch.matmul(selected, weight.t())
        return logits.squeeze(0) if logits_indices is not None else logits.squeeze(1)
