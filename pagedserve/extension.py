"""Loads the compiled CUDA extension, or explains clearly why it cannot.

``import pagedserve`` must succeed on a machine with no CUDA toolkit — that is
where most of this code gets written (AGENTS.md §4.3). So nothing imports the
extension at module scope. Callers ask this module, which caches the answer and
warns exactly once.

Warning once matters: a per-step warning in a decode loop would produce
thousands of identical lines and train everyone to ignore the one that says
something important.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["is_available", "load", "unavailable_reason"]

_extension: Any | None = None
_reason: str | None = None
_attempted = False
_warned = False


def load() -> Any | None:
    """The extension module, or ``None`` if it is not usable here.

    Never raises. A missing extension is an expected state — on a laptop, in
    CI, or before anyone has run ``build_ext`` — and the fallback to the gather
    backend is a supported configuration, not an error.
    """
    global _extension, _reason, _attempted, _warned
    if _attempted:
        return _extension

    _attempted = True
    try:
        # torch MUST be imported first. The extension links against libc10,
        # libtorch and libtorch_cuda, and those live in torch's package
        # directory rather than anywhere the dynamic loader searches by default.
        # Importing torch loads them into the process, after which the
        # extension resolves. Without it the import fails with
        # "libc10.so: cannot open shared object file" -- which reads like a
        # broken build even though the build was perfectly fine, and which only
        # appears when nothing else in the process happened to import torch
        # first. That makes it exactly the kind of bug that hides in a test
        # suite and surfaces in a script.
        import torch  # noqa: F401
    except ImportError as exc:
        _reason = (
            f"torch is not installed ({exc}), so the CUDA extension cannot be "
            f"loaded. Install the [engine] extra."
        )
        _extension = None
        _warned = True
        logger.warning("CUDA extension unavailable: %s", _reason)
        return None

    try:
        import pagedserve._C as extension  # type: ignore[import-not-found]
    except ImportError as exc:
        _reason = (
            f"{exc}. Build it with `python setup.py build_ext --inplace` on a "
            f"machine with a CUDA toolkit. Until then the gather backend is used, "
            f"which is correct and slower."
        )
        _extension = None
    else:
        _extension = extension
        _reason = None

    if _extension is None and not _warned:
        _warned = True
        logger.warning("CUDA extension unavailable: %s", _reason)
    return _extension


def is_available() -> bool:
    return load() is not None


def unavailable_reason() -> str | None:
    """Why the extension could not be loaded, or ``None`` if it loaded fine."""
    load()
    return _reason
