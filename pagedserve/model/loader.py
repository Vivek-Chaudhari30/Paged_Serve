"""Weight loading: a repo id or a directory in, a state dict out.

Resolving *where* the weights are and *reading* them are kept apart on purpose.
Resolution may touch the network; loading never does. That separation is what
lets a benchmark run be guaranteed offline (set ``HF_HUB_OFFLINE=1`` and a
missing file fails immediately instead of hanging on a socket halfway through a
two-hour measurement), and it lets the loader be tested against a fixture
directory with no hub involved at all.

We read ``.safetensors`` directly rather than going through ``transformers``.
Phase 1's entire point is that we own the forward pass, and ``transformers``'
cache API abstracts exactly the layer being replaced.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

__all__ = ["load_state_dict", "resolve_model_path", "shard_files"]

# Only these are needed to run a model. Pulling the whole repo drags in
# duplicate .bin weights, ONNX exports, and GGUF conversions -- gigabytes of
# files that will never be read.
WEIGHT_PATTERNS = [
    "config.json",
    "generation_config.json",
    "*.safetensors",
    "*.safetensors.index.json",
]
TOKENIZER_PATTERNS = ["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json"]


def resolve_model_path(
    model: str | Path,
    *,
    revision: str | None = None,
    include_tokenizer: bool = True,
) -> Path:
    """Turn a repo id or path into a local directory containing the weights.

    A path that already exists is returned untouched, so a pre-staged checkout
    on a cluster's shared filesystem needs no hub access at all. Anything else
    is treated as a hub repo id and downloaded.

    Gated repos (``meta-llama/*`` among them) require a token, supplied through
    ``hf auth login`` or the ``HF_TOKEN`` environment variable. No token is ever
    read, logged, or written by this project.
    """
    candidate = Path(model)
    if candidate.is_dir():
        return candidate

    from huggingface_hub import snapshot_download

    patterns = list(WEIGHT_PATTERNS)
    if include_tokenizer:
        patterns += TOKENIZER_PATTERNS
    logger.info("resolving %s from the HuggingFace hub", model)
    return Path(snapshot_download(str(model), revision=revision, allow_patterns=patterns))


def shard_files(path: str | Path) -> list[Path]:
    """The ``.safetensors`` files to read, in a deterministic order.

    Prefers the shard index when one exists so that only files the index
    actually references get read; falls back to a glob for single-file
    checkpoints, which is what most sub-1B models are.
    """
    directory = Path(path)
    index_files = sorted(directory.glob("*.safetensors.index.json"))
    if index_files:
        weight_map = json.loads(index_files[0].read_text())["weight_map"]
        return [directory / name for name in sorted(set(weight_map.values()))]

    shards = sorted(p for p in directory.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no .safetensors found in {directory}")
    return shards


def load_state_dict(
    path: str | Path,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Read every shard into one state dict.

    Loads to CPU first and casts there. Materialising on the target device
    shard-by-shard would hold both the source and the cast copy in device
    memory at once, and on a card sized for a specific model that peak is
    exactly what will not fit.

    Args:
        path: Directory holding the ``.safetensors``.
        dtype: Cast every floating-point tensor to this. ``None`` keeps the
            checkpoint's own dtype.
        device: Where the tensors end up.

    Returns:
        Parameter name to tensor, with the checkpoint's names unchanged.
    """
    state: dict[str, torch.Tensor] = {}
    for shard in shard_files(path):
        loaded = load_file(str(shard), device="cpu")
        for key, tensor in loaded.items():
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype)
            state[key] = tensor.to(device)
    logger.info("loaded %d tensors from %s", len(state), path)
    return state
