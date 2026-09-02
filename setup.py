"""Builds the optional CUDA extension.

Project metadata lives in ``pyproject.toml``; this file exists only for the
extension, because ``torch.utils.cpp_extension`` has no declarative equivalent.

**The extension is optional in both directions.** ``pip install -e .`` must work
on a laptop with no CUDA toolkit — that is where the code is written — so a
missing ``nvcc`` skips the extension rather than failing the install. And
``import pagedserve`` must work without the extension having been built, which
``pagedserve/extension.py`` handles at the other end.

Building it is therefore an explicit step rather than a side effect of install:

    python setup.py build_ext --inplace

Deliberate, because ``pip install`` runs in an isolated build environment that
has no torch, so the extension would silently not build there and the failure
would surface much later as a confusing fallback to the slow path.
"""

from __future__ import annotations

import os
import sys

from setuptools import setup


def cuda_toolkit_path() -> str | None:
    """Where nvcc lives, or None if this machine cannot compile CUDA."""
    try:
        from torch.utils.cpp_extension import CUDA_HOME
    except ImportError:
        # No torch in this environment. Normal during an isolated pip build.
        return None
    if CUDA_HOME and os.path.isdir(CUDA_HOME):
        return CUDA_HOME
    return None


def build_config() -> tuple[list, dict]:
    if os.environ.get("PAGEDSERVE_SKIP_CUDA_BUILD"):
        print("PAGEDSERVE_SKIP_CUDA_BUILD set; skipping the CUDA extension.")
        return [], {}

    cuda_home = cuda_toolkit_path()
    if cuda_home is None:
        print(
            "No CUDA toolkit found; skipping the CUDA extension. "
            "pagedserve will run on the gather backend.",
            file=sys.stderr,
        )
        return [], {}

    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    print(f"Building the CUDA extension against {cuda_home}")
    extension = CUDAExtension(
        name="pagedserve._C",
        sources=["csrc/bindings.cpp", "csrc/trivial.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            # No -arch flag: torch infers the architectures of the visible GPUs,
            # and hardcoding one produces a binary that fails at load time on
            # every other card. TORCH_CUDA_ARCH_LIST overrides when
            # cross-compiling for a machine that is not this one.
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        },
    )
    return [extension], {"build_ext": BuildExtension}


ext_modules, cmdclass = build_config()

setup(ext_modules=ext_modules, cmdclass=cmdclass)
