"""pytest configuration shared by all tests in this directory.

Two jobs.

**Headless plotting.** Sets the matplotlib backend to Agg before any test module
is imported. This must happen before the first ``import matplotlib`` or
``import bench.plot`` call, and conftest.py is the right place because pytest
loads it before importing test modules.

**Saying out loud what a run actually exercised.** Most of this suite is
device-agnostic on purpose (AGENTS.md §4), so a green run tells you nothing
about *where* it ran unless the run says so. On a metered GPU box that gap is
expensive: ``pytest -m gpu`` selects only the handful of tests that cannot run
without CUDA, which does not include the golden gate, and "10 passed" reads
exactly like "the GPU suite passed". The report header below states the device,
the dtype, whether CUDA and the compiled extension are actually present, and
whether the golden model can be loaded at all — so a run that proved less than
you thought says so on its first line.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

# Mirrors tests/test_golden.py. The golden gate is device-parametric rather than
# gpu-marked because numerical agreement with HuggingFace is device-independent,
# and CPU float32 is the strictest place to demonstrate it. That is why running
# it on CUDA is a matter of setting these, not of selecting a marker.
TEST_DEVICE = os.environ.get("PAGEDSERVE_TEST_DEVICE", "cpu")
TEST_DTYPE = os.environ.get("PAGEDSERVE_TEST_DTYPE", "float32")
GOLDEN_MODEL = os.environ.get("PAGEDSERVE_GOLDEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _torch_state() -> dict[str, object]:
    """What torch can actually do here. Never raises; torch may be absent."""
    state: dict[str, object] = {
        "torch": None,
        "cuda_available": False,
        "cuda_device": None,
        "extension": False,
    }
    try:
        import torch
    except Exception:
        return state

    state["torch"] = torch.__version__
    try:
        state["cuda_available"] = bool(torch.cuda.is_available())
        if state["cuda_available"]:
            state["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception:
        state["cuda_available"] = False

    try:
        from pagedserve import extension

        state["extension"] = bool(extension.is_available())
    except Exception:
        state["extension"] = False
    return state


def _golden_model_available() -> bool:
    """Whether the golden checkpoint is already on this machine.

    Forced offline for the duration of the probe. ``resolve_model_path`` will
    happily *download* a missing checkpoint, and a header line is the wrong place
    to start a multi-gigabyte fetch — ``pytest tests/test_metrics.py`` must stay
    a two-second command on a fresh machine.
    """
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from pagedserve.model.loader import resolve_model_path

        resolve_model_path(GOLDEN_MODEL)
    except Exception:
        return False
    else:
        return True
    finally:
        if previous is None:
            del os.environ["HF_HUB_OFFLINE"]
        else:
            os.environ["HF_HUB_OFFLINE"] = previous


def pytest_report_header() -> list[str]:
    state = _torch_state()
    gpu = state["cuda_device"] if state["cuda_available"] else "not available"
    golden = "loadable" if _golden_model_available() else "MISSING — the gate will SKIP"
    return [
        f"pagedserve: torch={state['torch']} cuda={gpu} extension={state['extension']}",
        f"pagedserve: golden gate device={TEST_DEVICE} dtype={TEST_DTYPE} "
        f"model={GOLDEN_MODEL} ({golden})",
    ]


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items) -> None:
    """Refuse to run CUDA-only tests on a machine with no CUDA.

    ``trylast`` matters: pytest's own mark plugin does its ``-m`` deselection in
    this same hook, so running first would see the full 482 items and reject
    ``-m "not gpu"`` — the one command that is correct on a laptop.

    Without this the three ``profile_num_blocks`` GPU tests fail on a laptop with
    "Torch not compiled with CUDA enabled" — three stack traces that look like a
    regression in the code under test rather than a wrong command. Deselecting
    them silently would be worse: it would make ``pytest -m gpu`` exit clean on a
    machine that has no GPU, which is the false green this guard exists to stop.
    """
    state = _torch_state()
    if state["cuda_available"]:
        return

    selected_gpu = [item for item in items if item.get_closest_marker("gpu")]
    if not selected_gpu:
        return

    # A bare `pytest` on a Mac selects them too, and AGENTS.md §6 says to use
    # -m "not gpu" there. Point at that rather than at the traceback.
    raise pytest.UsageError(
        f"{len(selected_gpu)} test(s) marked `gpu` were selected, but torch reports "
        f"no CUDA device (torch={state['torch']}).\n"
        "  On a machine without a GPU, run:  pytest -m 'not gpu'\n"
        "  On a GPU machine, a passing `pytest -m gpu` is NOT the full GPU suite: it "
        "selects only the tests that cannot run without CUDA. The golden gate is "
        "device-parametric, so run the whole suite against the GPU with:\n"
        "      PAGEDSERVE_TEST_DEVICE=cuda PAGEDSERVE_TEST_DTYPE=float16 pytest"
    )
