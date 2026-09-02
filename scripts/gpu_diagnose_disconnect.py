"""Is the abandoned-stream failure a leak, or cleanup racing the assertion?

``test_an_abandoned_stream_frees_its_blocks`` passes on CPU and fails on a T4.
Two possibilities, and they need opposite fixes:

**A real leak.** The disconnect path does not free the sequence, so the blocks
never come back. Under open-loop load with client timeouts this is fatal — the
server admits fewer and fewer requests until it admits none, degrading over
hours and looking healthy after a restart.

**Cleanup lagging the assertion.** Closing the response schedules the streaming
generator's ``finally`` on the event loop, and the test reads the free-block
count before that has run. The property being defended is "abandoned work stops
consuming memory", not "cleanup is synchronous with the client's socket close" —
so if the blocks return promptly, the test is asserting the wrong thing and the
server is fine.

The difference is visible in one measurement: sample the free-block count
immediately after abandoning, then again after a moment. This settles it rather
than arguing about it.
"""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    from fastapi.testclient import TestClient
    from transformers import AutoTokenizer

    from pagedserve.config import CacheConfig
    from pagedserve.engine import LLMEngine
    from pagedserve.model.loader import resolve_model_path
    from pagedserve.server.api import create_app

    path = str(resolve_model_path(args.model))
    tokenizer = AutoTokenizer.from_pretrained(path)
    engine = LLMEngine.from_pretrained(
        path,
        cache=CacheConfig(max_seq_len=512, max_num_seqs=8, block_size=16, num_blocks_override=128),
    )
    app = create_app(engine, tokenizer, model_name="diagnose")
    manager = engine.block_manager

    print("=" * 68)
    print("  ABANDONED STREAM: DOES THE MEMORY COME BACK?")
    print("=" * 68)

    with TestClient(app) as client:
        baseline = manager.num_free_blocks
        print(f"  free blocks, idle                 : {baseline}")

        with client.stream(
            "POST",
            "/v1/completions",
            json={
                "model": "diagnose",
                "prompt": "Once upon a time in a land far away",
                "max_tokens": 200,
                "temperature": 0,
                "stream": True,
            },
        ) as response:
            for read, _ in enumerate(response.iter_lines(), start=1):
                if read >= 3:
                    break
            during = manager.num_free_blocks
        print(f"  free blocks, mid-stream           : {during}")

        samples = []
        for delay in (0.0, 0.05, 0.25, 1.0, 3.0):
            time.sleep(delay if not samples else delay - samples[-1][0])
            samples.append((delay, manager.num_free_blocks))
            print(f"  free blocks, {delay:>4.2f}s after abandon: {samples[-1][1]}")

        recovered = samples[-1][1] == baseline
        print()
        if recovered and samples[0][1] == baseline:
            print("  VERDICT: freed synchronously. The GPU failure is something else.")
        elif recovered:
            when = next(d for d, value in samples if value == baseline)
            print(f"  VERDICT: cleanup is ASYNCHRONOUS -- blocks returned by {when:.2f}s.")
            print("  Not a leak. The test asserts synchronous cleanup, which the")
            print("  server never promised; it should wait for the count instead.")
        else:
            print("  VERDICT: REAL LEAK. The blocks never came back.")
            print("  Under open-loop load this strangles the server over hours.")

        try:
            manager.check_invariants()
            print("  allocator invariants: OK")
        except AssertionError as exc:
            print(f"  allocator invariants: VIOLATED -- {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
