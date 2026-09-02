"""Run the server: ``python -m pagedserve.server --model <id>``.

Kept separate from ``api.py`` so the app can be constructed in a test without
starting a process, and so the CLI's argument surface is visible in one place.
"""

from __future__ import annotations

import argparse
import logging

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve a model over an OpenAI-compatible API.")
    parser.add_argument("--model", required=True, help="Model id or local path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=None,
        help="Size the KV cache explicitly. Required off CUDA, and worth setting "
        "on CUDA too until the profiler accounts for activation memory.",
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--no-paging",
        action="store_true",
        help="Use the Phase 1 contiguous cache. The ablation arm; it cannot do "
        "continuous batching, so it serves one request at a time.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    import uvicorn
    from transformers import AutoTokenizer

    from pagedserve.config import CacheConfig, SchedulerConfig
    from pagedserve.engine import LLMEngine
    from pagedserve.model.loader import resolve_model_path
    from pagedserve.server.api import create_app

    path = resolve_model_path(args.model)
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    engine = LLMEngine.from_pretrained(
        args.model,
        device=args.device,
        dtype=args.dtype,
        cache=CacheConfig(
            max_seq_len=args.max_seq_len,
            max_num_seqs=args.max_num_seqs,
            block_size=args.block_size,
            num_blocks_override=args.num_blocks,
            enable_prefix_caching=args.enable_prefix_caching,
        ),
        scheduler=SchedulerConfig(
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
        ),
        attn_backend="contiguous" if args.no_paging else "gather",
    )
    app = create_app(engine, tokenizer, model_name=args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
