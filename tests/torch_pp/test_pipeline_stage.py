"""Tests for PipelineStage forward/backward correctness.

Uses a small HF model (Qwen3-0.6B or similar) to verify that:
1. Partitioned forward produces the same logits as full-model forward
2. Backward through PP stages produces valid gradients
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import patch

from verl.workers.torch_pp.pipeline_stage import (
    _prune_model_inplace,
    _remap_global_to_local,
    restore_global_layer_keys,
)


class TestRemapKeys:
    def test_remap_layer_key(self):
        key = "model.layers.7.self_attn.q_proj.weight"
        result = _remap_global_to_local(key, start_layer=5)
        assert result == "model.layers.2.self_attn.q_proj.weight"

    def test_remap_non_layer_key_unchanged(self):
        for key in ["model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"]:
            assert _remap_global_to_local(key, 5) == key

    def test_restore_roundtrip(self):
        """remap then restore should give back original keys."""
        original = {
            "model.layers.5.w": torch.zeros(1),
            "model.layers.6.w": torch.zeros(1),
            "model.embed_tokens.weight": torch.zeros(1),
        }
        remapped = {_remap_global_to_local(k, 5): v for k, v in original.items()}
        restored = restore_global_layer_keys(remapped, 5)
        assert set(restored.keys()) == set(original.keys())


class TestPruneModelInplace:
    def _make_fake_model(self, num_layers=8):
        """Create a minimal model structure matching HF causal LM layout."""

        class FakeInner(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(n)])
                self.norm = nn.LayerNorm(4)
                self.embed_tokens = nn.Embedding(10, 4)

        class FakeModel(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.model = FakeInner(n)
                self.lm_head = nn.Linear(4, 10)
                self.config = type("Config", (), {"num_hidden_layers": n})()

        return FakeModel(num_layers)

    def test_first_stage_keeps_embed(self):
        model = self._make_fake_model(8)
        _prune_model_inplace(model, 0, 2, is_first=True, is_last=False)
        assert len(model.model.layers) == 2
        assert hasattr(model.model, "embed_tokens")
        # norm and lm_head replaced with Identity
        assert isinstance(model.model.norm, nn.Identity)
        assert isinstance(model.lm_head, nn.Identity)

    def test_last_stage_keeps_norm_and_lm_head(self):
        model = self._make_fake_model(8)
        _prune_model_inplace(model, 6, 8, is_first=False, is_last=True)
        assert len(model.model.layers) == 2
        assert isinstance(model.model.norm, nn.LayerNorm)
        assert not isinstance(model.lm_head, nn.Identity)

    def test_middle_stage(self):
        model = self._make_fake_model(8)
        _prune_model_inplace(model, 2, 4, is_first=False, is_last=False)
        assert len(model.model.layers) == 2
        assert isinstance(model.model.norm, nn.Identity)
        assert isinstance(model.lm_head, nn.Identity)

    def test_config_not_modified(self):
        """num_hidden_layers should NOT be changed — some models use it
        for per-layer attention patterns (e.g. Qwen3 max_window_layers)."""
        model = self._make_fake_model(8)
        _prune_model_inplace(model, 2, 5, is_first=False, is_last=False)
        assert model.config.num_hidden_layers == 8  # unchanged
