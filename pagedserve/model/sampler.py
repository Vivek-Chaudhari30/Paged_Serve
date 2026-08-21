"""Token selection. Greedy only, for now.

Phase 1 is greedy-only on purpose: the golden test asserts token-for-token
equality against HuggingFace, and sampling would make that assertion statistical
instead of exact. Temperature, top-p, and top-k arrive in Phase 6, once the hard
parts are locked down and verified.
"""

from __future__ import annotations

import torch

__all__ = ["greedy"]


def greedy(logits: torch.Tensor) -> torch.Tensor:
    """Pick the highest-scoring token per sequence.

    Args:
        logits: ``[batch, vocab_size]``.

    Returns:
        ``[batch]`` token ids.
    """
    return torch.argmax(logits, dim=-1)
