"""PagedServe — a from-scratch high-throughput LLM inference server.

This package must import cleanly on a machine with no GPU and no CUDA toolchain
(see AGENTS.md section 4). Nothing here may import torch at module scope until a
phase actually needs it, and the compiled CUDA extension is always imported
lazily behind a try/except.
"""

from __future__ import annotations

__version__ = "0.0.0"

__all__ = ["__version__"]
