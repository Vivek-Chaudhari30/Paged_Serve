"""Unit tests for KV capacity sizing.

The arithmetic is pure, so it is tested here without a GPU. The measurement it
consumes is not, which is exactly why the two are separate functions.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from pagedserve.config import CacheConfig, ModelConfig  # noqa: E402
from pagedserve.worker.cache_engine import (  # noqa: E402
    blocks_from_budget,
    bytes_per_block,
    profile_num_blocks,
)

QWEN = ModelConfig.from_hf_dict(
    {
        "model_type": "qwen2",
        "hidden_size": 896,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "num_hidden_layers": 24,
        "intermediate_size": 4864,
        "vocab_size": 151936,
    }
)

GIB = 2**30


class TestBytesPerBlock:
    def test_by_hand(self):
        # 12288 bytes/token in bf16 x 16 tokens = 196608 bytes per block.
        assert QWEN.kv_bytes_per_token(torch.bfloat16) == 12288
        assert bytes_per_block(QWEN, 16, torch.bfloat16) == 196608

    def test_scales_linearly_with_block_size(self):
        small = bytes_per_block(QWEN, 8, torch.bfloat16)
        assert bytes_per_block(QWEN, 16, torch.bfloat16) == 2 * small

    def test_scales_with_dtype_width(self):
        half = bytes_per_block(QWEN, 16, torch.float16)
        assert bytes_per_block(QWEN, 16, torch.float32) == 2 * half


class TestBlocksFromBudget:
    def test_worked_example(self):
        # 24 GiB card at 0.90 -> 21.6 GiB budget; 2 GiB weights and 1 GiB peak
        # activation leave 18.6 GiB; at 196608 bytes per block that is 101,662.
        blocks = blocks_from_budget(
            total_bytes=24 * GIB,
            weights_bytes=2 * GIB,
            peak_activation_bytes=1 * GIB,
            utilization=0.90,
            block_bytes=196608,
        )
        expected = (int(24 * GIB * 0.90) - 3 * GIB) // 196608
        assert blocks == expected
        assert blocks == 101_580

    def test_utilization_applies_to_the_whole_card(self):
        """The fraction answers "how much of this card may I occupy".

        Applying it to the remainder instead would leave no headroom for the
        CUDA context and allocator fragmentation, neither of which shows up in
        weights plus activations.
        """
        low = blocks_from_budget(
            total_bytes=16 * GIB,
            weights_bytes=1 * GIB,
            peak_activation_bytes=0,
            utilization=0.50,
            block_bytes=GIB,
        )
        assert low == 7  # 8 GiB budget - 1 GiB weights

    def test_more_headroom_means_more_blocks(self):
        args = dict(
            total_bytes=16 * GIB, peak_activation_bytes=GIB, utilization=0.9, block_bytes=GIB
        )
        assert blocks_from_budget(weights_bytes=2 * GIB, **args) > blocks_from_budget(
            weights_bytes=6 * GIB, **args
        )

    def test_refuses_when_nothing_is_left(self):
        # Failing loudly at startup beats a two-block cache that thrashes.
        with pytest.raises(ValueError, match="no room for a KV cache"):
            blocks_from_budget(
                total_bytes=4 * GIB,
                weights_bytes=4 * GIB,
                peak_activation_bytes=GIB,
                utilization=0.9,
                block_bytes=196608,
            )

    def test_refuses_a_nonsense_block_size(self):
        with pytest.raises(ValueError, match="block_bytes must be positive"):
            blocks_from_budget(
                total_bytes=GIB,
                weights_bytes=0,
                peak_activation_bytes=0,
                utilization=0.9,
                block_bytes=0,
            )

    def test_rounds_down_never_up(self):
        # Rounding up would allocate a cache that does not fit.
        blocks = blocks_from_budget(
            total_bytes=1000,
            weights_bytes=0,
            peak_activation_bytes=0,
            utilization=1.0,
            block_bytes=300,
        )
        assert blocks == 3


class TestProfileNumBlocks:
    def test_override_wins_and_skips_profiling(self):
        cache = CacheConfig(num_blocks_override=512)
        assert (
            profile_num_blocks(
                QWEN,
                cache,
                device=torch.device("cpu"),
                dtype=torch.float32,
                weights_bytes=0,
            )
            == 512
        )

    def test_refuses_to_guess_off_cuda(self):
        """No fabricated capacity number.

        torch cannot report free host memory, and inventing a figure would put a
        made-up number under every capacity decision the engine makes.
        """
        with pytest.raises(ValueError, match="num_blocks_override"):
            profile_num_blocks(
                QWEN,
                CacheConfig(),
                device=torch.device("cpu"),
                dtype=torch.float32,
                weights_bytes=0,
            )

    def test_error_names_the_device_and_the_fix(self):
        with pytest.raises(ValueError) as exc:
            profile_num_blocks(
                QWEN,
                CacheConfig(),
                device=torch.device("mps"),
                dtype=torch.float32,
                weights_bytes=0,
            )
        assert "mps" in str(exc.value)
        assert "num_blocks_override" in str(exc.value)

    def test_snapshot_injection_bypasses_cuda_measurement(self):
        """_memory_snapshot lets the two-pass path be verified without a GPU.

        The injection skips the device check, the CUDA synchronize, and
        torch.cuda.get_device_properties — only the pure arithmetic runs.
        """
        total = 24 * GIB
        peak = 1 * GIB
        result = profile_num_blocks(
            QWEN,
            CacheConfig(),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            weights_bytes=2 * GIB,
            _memory_snapshot=(total, peak),
        )
        expected = blocks_from_budget(
            total_bytes=total,
            weights_bytes=2 * GIB,
            peak_activation_bytes=peak,
            utilization=CacheConfig().gpu_memory_utilization,
            block_bytes=bytes_per_block(QWEN, CacheConfig().block_size, torch.bfloat16),
        )
        assert result == expected

    def test_snapshot_injection_zero_activation(self):
        """When activation is zero the optimistic path produces the same result."""
        total = 16 * GIB
        result_injected = profile_num_blocks(
            QWEN,
            CacheConfig(),
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
            weights_bytes=GIB,
            _memory_snapshot=(total, 0),
        )
        expected = blocks_from_budget(
            total_bytes=total,
            weights_bytes=GIB,
            peak_activation_bytes=0,
            utilization=CacheConfig().gpu_memory_utilization,
            block_bytes=bytes_per_block(QWEN, CacheConfig().block_size, torch.bfloat16),
        )
        assert result_injected == expected


@pytest.mark.gpu
class TestProfileNumBlocksOnGpu:
    """Real CUDA measurement.  Skipped everywhere CUDA is not available."""

    def test_profiles_without_forward_pass(self):
        """Without run_max_shape_forward the function still returns a sane count."""
        result = profile_num_blocks(
            QWEN,
            CacheConfig(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            weights_bytes=0,
        )
        assert result > 0

    def test_run_max_shape_forward_is_called(self):
        """The two-pass: the profiling callable must be invoked."""
        called = []

        def noop_forward() -> None:
            called.append(True)

        result = profile_num_blocks(
            QWEN,
            CacheConfig(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            weights_bytes=0,
            run_max_shape_forward=noop_forward,
        )
        assert called, "run_max_shape_forward was not called"
        assert result > 0

    def test_activation_reduces_the_block_count(self):
        """A forward pass that allocates extra memory must shrink the cache.

        This is the whole point of the two-pass: activation memory is a real
        claim on the card, and counting it as zero hands the KV cache memory
        that something else will ask for.
        """
        # Baseline: no forward pass, activation counted as zero.
        baseline = profile_num_blocks(
            QWEN,
            CacheConfig(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            weights_bytes=0,
        )

        def allocating_forward() -> None:
            # Allocate 256 MiB, then free it.  max_memory_allocated() still
            # captures this high-water mark after the tensor is gone.
            t = torch.zeros(256 * 2**20 // 4, dtype=torch.float32, device="cuda")
            torch.cuda.synchronize()
            del t

        with_activation = profile_num_blocks(
            QWEN,
            CacheConfig(),
            device=torch.device("cuda"),
            dtype=torch.float16,
            weights_bytes=0,
            run_max_shape_forward=allocating_forward,
        )

        assert with_activation < baseline
