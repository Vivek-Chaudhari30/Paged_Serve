"""Attention backends.

Every backend implements ``AttentionBackend``. Model code talks to that ABC and
never to a concrete backend, which is what makes the paged and CUDA paths drop
in later without touching a line of the forward pass — and what keeps the slow
reference implementations alive as correctness oracles (AGENTS.md §2.3).
"""

from __future__ import annotations
