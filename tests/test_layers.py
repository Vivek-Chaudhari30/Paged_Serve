"""Unit tests for pagedserve/model/layers.py and the weight loader.

These check the conventions that produce *plausible wrong output* rather than a
crash when they are wrong: the rotary half-split, the RMSNorm float32 upcast,
and strict weight-name mapping.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pagedserve.config import ModelConfig  # noqa: E402
from pagedserve.model.layers import (  # noqa: E402
    MLP,
    RMSNorm,
    RotaryEmbedding,
    apply_rotary,
    rotate_half,
)
from pagedserve.model.loader import load_state_dict, shard_files  # noqa: E402

CPU = torch.device("cpu")

TINY = ModelConfig(
    name="tiny",
    num_layers=2,
    hidden_size=16,
    num_q_heads=4,
    num_kv_heads=2,
    head_dim=4,
    intermediate_size=32,
    vocab_size=50,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=64,
    tie_word_embeddings=False,
    attention_bias=True,
)


class TestRotateHalf:
    def test_uses_the_split_half_convention(self):
        # [a, b, c, d] -> [-c, -d, a, b]. The alternative interleaved convention
        # would give [-b, a, -d, c]. Both are valid rotary embeddings and they
        # are not interchangeable against trained weights.
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        assert torch.equal(rotate_half(x), torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))

    def test_applying_it_four_times_is_the_identity(self):
        x = torch.randn(2, 3, 8)
        assert torch.allclose(rotate_half(rotate_half(rotate_half(rotate_half(x)))), x)


class TestRotaryEmbedding:
    def test_cos_sin_shapes(self):
        rot = RotaryEmbedding(head_dim=8, theta=10000.0, device=CPU)
        cos, sin = rot(torch.arange(5).unsqueeze(0), torch.float32)
        assert cos.shape == (1, 5, 8)
        assert sin.shape == (1, 5, 8)

    def test_position_zero_is_no_rotation(self):
        rot = RotaryEmbedding(head_dim=8, theta=10000.0, device=CPU)
        cos, sin = rot(torch.zeros(1, 1, dtype=torch.long), torch.float32)
        q = torch.randn(1, 1, 2, 8)
        k = torch.randn(1, 1, 1, 8)
        rq, rk = apply_rotary(q, k, cos, sin)
        assert torch.allclose(rq, q, atol=1e-6)
        assert torch.allclose(rk, k, atol=1e-6)

    def test_rotation_preserves_norm(self):
        # A rotation changes direction, never length.
        rot = RotaryEmbedding(head_dim=8, theta=10000.0, device=CPU)
        cos, sin = rot(torch.arange(4).unsqueeze(0), torch.float32)
        q = torch.randn(1, 4, 2, 8)
        rq, _ = apply_rotary(q, q[:, :, :1], cos, sin)
        assert torch.allclose(rq.norm(dim=-1), q.norm(dim=-1), atol=1e-5)

    def test_dot_product_depends_only_on_relative_position(self):
        """The property that makes a KV cache valid at any later position.

        A key computed at step 5 stays correct forever because attention only
        ever sees the difference between two absolute rotations.
        """
        rot = RotaryEmbedding(head_dim=8, theta=10000.0, device=CPU)
        q = torch.randn(1, 1, 1, 8)
        k = torch.randn(1, 1, 1, 8)

        def score(pos_q: int, pos_k: int) -> float:
            cq, sq = rot(torch.tensor([[pos_q]]), torch.float32)
            ck, sk = rot(torch.tensor([[pos_k]]), torch.float32)
            rq, _ = apply_rotary(q, q, cq, sq)
            rk, _ = apply_rotary(k, k, ck, sk)
            return float((rq * rk).sum())

        assert score(3, 1) == pytest.approx(score(9, 7), abs=1e-4)
        assert score(3, 1) != pytest.approx(score(9, 4), abs=1e-4)


class TestRMSNorm:
    def test_matches_the_reference_formula(self):
        norm = RMSNorm(8, eps=1e-6, dtype=torch.float32, device=CPU)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 8))
        x = torch.randn(2, 3, 8)
        expected = norm.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))
        assert torch.allclose(norm(x), expected, atol=1e-6)

    def test_upcasts_to_float32_internally(self):
        """The upcast is not optional.

        In bf16 the sum of squared activations loses enough precision to move an
        argmax on near-ties, which surfaces as a golden-test failure several
        tokens later with no obvious cause.
        """
        norm = RMSNorm(64, eps=1e-6, dtype=torch.bfloat16, device=CPU)
        x = torch.randn(1, 1, 64, dtype=torch.bfloat16)
        out = norm(x)
        assert out.dtype == torch.bfloat16

        # Ours must reproduce the float32 normalisation exactly, then cast --
        # not normalise in bf16.
        precise = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)
        assert torch.equal(out, norm.weight * precise.to(torch.bfloat16))

        # And the bf16-throughout version really is different, so the upcast is
        # doing work rather than being decorative.
        naive = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        assert not torch.equal(naive, precise.to(torch.bfloat16))

    def test_output_is_scale_invariant(self):
        # RMSNorm divides out the magnitude, so scaling the input does nothing.
        norm = RMSNorm(16, eps=1e-12, dtype=torch.float32, device=CPU)
        x = torch.randn(1, 1, 16)
        assert torch.allclose(norm(x), norm(x * 10.0), atol=1e-4)


class TestMLP:
    def test_shape_round_trip(self):
        mlp = MLP(TINY, torch.float32, CPU)
        x = torch.randn(2, 3, TINY.hidden_size)
        assert mlp(x).shape == x.shape

    def test_has_no_bias_terms(self):
        mlp = MLP(TINY, torch.float32, CPU)
        assert mlp.gate_proj.bias is None
        assert mlp.up_proj.bias is None
        assert mlp.down_proj.bias is None

    def test_gate_of_zero_kills_the_branch(self):
        # silu(0) = 0, so a zeroed gate produces a zero output regardless of up.
        mlp = MLP(TINY, torch.float32, CPU)
        with torch.no_grad():
            mlp.gate_proj.weight.zero_()
        assert torch.allclose(
            mlp(torch.randn(1, 1, TINY.hidden_size)),
            torch.zeros(1, 1, TINY.hidden_size),
            atol=1e-7,
        )


class TestWeightLoading:
    def _write_checkpoint(self, tmp_path, tensors: dict[str, torch.Tensor]):
        from safetensors.torch import save_file

        save_file(tensors, str(tmp_path / "model.safetensors"))
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
        return tmp_path

    def test_reads_a_single_shard(self, tmp_path):
        self._write_checkpoint(tmp_path, {"a": torch.ones(2), "b": torch.zeros(3)})
        state = load_state_dict(tmp_path)
        assert set(state) == {"a", "b"}

    def test_casts_floating_point_tensors(self, tmp_path):
        self._write_checkpoint(tmp_path, {"w": torch.ones(4, dtype=torch.float32)})
        state = load_state_dict(tmp_path, dtype=torch.bfloat16)
        assert state["w"].dtype == torch.bfloat16

    def test_leaves_integer_tensors_alone(self, tmp_path):
        # Casting an integer buffer to bf16 would corrupt it silently.
        self._write_checkpoint(tmp_path, {"ids": torch.arange(4, dtype=torch.int64)})
        state = load_state_dict(tmp_path, dtype=torch.bfloat16)
        assert state["ids"].dtype == torch.int64

    def test_prefers_the_shard_index_when_present(self, tmp_path):
        from safetensors.torch import save_file

        save_file({"a": torch.ones(1)}, str(tmp_path / "part-1.safetensors"))
        save_file({"b": torch.ones(1)}, str(tmp_path / "part-2.safetensors"))
        save_file({"c": torch.ones(1)}, str(tmp_path / "orphan.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "part-1.safetensors", "b": "part-2.safetensors"}})
        )
        names = [p.name for p in shard_files(tmp_path)]
        assert names == ["part-1.safetensors", "part-2.safetensors"]
        assert "orphan.safetensors" not in names

    def test_errors_when_there_is_nothing_to_load(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no .safetensors"):
            shard_files(tmp_path)
