"""Unit tests for pagedserve/config.py."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pagedserve.config import (  # noqa: E402
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
    resolve_device,
    resolve_dtype,
)

QWEN_LIKE = {
    "model_type": "qwen2",
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_key_value_heads": 2,
    "num_hidden_layers": 24,
    "intermediate_size": 4864,
    "vocab_size": 151936,
    "rms_norm_eps": 1e-6,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 32768,
    "tie_word_embeddings": True,
    "eos_token_id": 151645,
}

LLAMA_LIKE = {
    "model_type": "llama",
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "num_hidden_layers": 16,
    "intermediate_size": 8192,
    "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "max_position_embeddings": 131072,
    "tie_word_embeddings": True,
    "eos_token_id": [128001, 128009],
}


class TestModelConfig:
    def test_parses_a_qwen_style_config(self):
        c = ModelConfig.from_hf_dict(QWEN_LIKE)
        assert c.num_layers == 24
        assert c.num_q_heads == 14
        assert c.num_kv_heads == 2
        assert c.head_dim == 64  # 896 / 14, implied rather than stated
        assert c.eos_token_ids == (151645,)

    def test_infers_qwen_attention_bias(self):
        # Qwen2 always biases Q/K/V and the config never says so. Getting this
        # wrong means silently dropping bias terms: the model still runs and
        # still produces fluent text, and it is wrong.
        assert ModelConfig.from_hf_dict(QWEN_LIKE).attention_bias is True

    def test_llama_defaults_to_no_attention_bias(self):
        assert ModelConfig.from_hf_dict(LLAMA_LIKE).attention_bias is False

    def test_explicit_attention_bias_overrides_the_family_default(self):
        raw = {**QWEN_LIKE, "attention_bias": False}
        assert ModelConfig.from_hf_dict(raw).attention_bias is False

    def test_accepts_a_list_of_eos_ids(self):
        assert ModelConfig.from_hf_dict(LLAMA_LIKE).eos_token_ids == (128001, 128009)

    def test_explicit_head_dim_wins_over_the_implied_one(self):
        raw = {**LLAMA_LIKE, "head_dim": 128}
        assert ModelConfig.from_hf_dict(raw).head_dim == 128

    def test_queries_per_kv_is_the_gqa_ratio(self):
        assert ModelConfig.from_hf_dict(QWEN_LIKE).num_queries_per_kv == 7
        assert ModelConfig.from_hf_dict(LLAMA_LIKE).num_queries_per_kv == 4

    def test_rejects_a_ragged_gqa_grouping(self):
        raw = {**QWEN_LIKE, "num_key_value_heads": 5}
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig.from_hf_dict(raw)

    def test_kv_bytes_per_token_by_hand(self):
        c = ModelConfig.from_hf_dict(QWEN_LIKE)
        # 2 (K and V) x 24 layers x 2 kv heads x 64 head_dim x 2 bytes = 12288
        assert c.kv_bytes_per_token(torch.bfloat16) == 12288
        assert c.kv_bytes_per_token(torch.float32) == 24576

    def test_kv_bytes_scales_with_the_gqa_ratio(self):
        # Halving KV heads halves the cache. This is the cheapest possible win
        # on the exact resource the project is about.
        few = ModelConfig.from_hf_dict({**QWEN_LIKE, "num_key_value_heads": 1})
        many = ModelConfig.from_hf_dict({**QWEN_LIKE, "num_key_value_heads": 2})
        assert many.kv_bytes_per_token(torch.float32) == 2 * few.kv_bytes_per_token(torch.float32)

    def test_from_pretrained_merges_generation_config_eos(self, tmp_path):
        # generation_config.json often lists more stop tokens than config.json.
        # Missing one means generating past the end of a turn, which reads as a
        # model quality problem and is really a config bug.
        (tmp_path / "config.json").write_text(json.dumps(QWEN_LIKE))
        (tmp_path / "generation_config.json").write_text(
            json.dumps({"eos_token_id": [151645, 151643]})
        )
        c = ModelConfig.from_pretrained(tmp_path)
        assert c.eos_token_ids == (151645, 151643)

    def test_from_pretrained_without_a_generation_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps(QWEN_LIKE))
        assert ModelConfig.from_pretrained(tmp_path).eos_token_ids == (151645,)

    def test_to_dict_is_json_serializable(self):
        c = ModelConfig.from_hf_dict(QWEN_LIKE)
        assert json.loads(json.dumps(c.to_dict()))["num_layers"] == 24


class TestResolveDtype:
    def test_explicit_spec_wins(self):
        assert resolve_dtype("float16", torch.device("cpu")) == torch.float16

    def test_rejects_a_name_that_is_not_a_dtype(self):
        with pytest.raises(ValueError, match="not a torch dtype"):
            resolve_dtype("banana", torch.device("cpu"))

    def test_cpu_and_mps_get_float32(self):
        assert resolve_dtype(None, torch.device("cpu")) == torch.float32
        assert resolve_dtype(None, torch.device("mps")) == torch.float32

    def test_ampere_and_later_get_bfloat16(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (8, 0))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        assert resolve_dtype(None, torch.device("cuda")) == torch.bfloat16

    def test_turing_falls_back_to_float16(self, monkeypatch):
        # A T4 is sm_75. Native bf16 starts at Ampere.
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        assert resolve_dtype(None, torch.device("cuda")) == torch.float16

    def test_ignores_is_bf16_supported_which_counts_emulation(self, monkeypatch):
        """The bug this replaced: is_bf16_supported() is True on a T4.

        Recent PyTorch counts emulated bf16 as supported, so trusting it picks
        a correct-but-slow dtype on the card most free GPU sessions hand out.
        """
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda *a, **k: True)
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5))
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        assert resolve_dtype(None, torch.device("cuda")) == torch.float16


class TestCacheConfig:
    def test_rejects_nonsense_sizes(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            CacheConfig(max_seq_len=0)
        with pytest.raises(ValueError, match="max_num_seqs"):
            CacheConfig(max_num_seqs=0)

    def test_block_size_exists_before_paging_does(self):
        # Phase 2 needs it and Phase 8 sweeps it. A knob that gets swept must be
        # a config field from the start, never a literal.
        assert CacheConfig().block_size == 16


class TestEngineConfig:
    def test_build_resolves_device_and_dtype_once(self):
        c = EngineConfig.build(ModelConfig.from_hf_dict(QWEN_LIKE), device="cpu")
        assert c.device == torch.device("cpu")
        assert c.dtype == torch.float32

    def test_to_dict_captures_everything_a_result_file_needs(self):
        c = EngineConfig.build(
            ModelConfig.from_hf_dict(QWEN_LIKE),
            device="cpu",
            dtype="float32",
            cache=CacheConfig(max_seq_len=512),
            scheduler=SchedulerConfig(max_num_batched_tokens=4096),
        )
        d = c.to_dict()
        assert d["device"] == "cpu"
        assert d["dtype"] == "float32"
        assert d["cache"]["max_seq_len"] == 512
        assert d["scheduler"]["max_num_batched_tokens"] == 4096
        assert d["attn_backend"] == "contiguous"
        assert json.loads(json.dumps(d)) == d

    def test_resolve_device_returns_something_real(self):
        assert resolve_device().type in {"cuda", "mps", "cpu"}
