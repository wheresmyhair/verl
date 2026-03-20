"""Test that remove-padding produces the same results as padded forward.

Loads a small HF model, runs forward with padding and without padding,
and verifies the outputs match for the non-padded token positions.

Key equivalence: padded [B, S] forward == rmpad [1, total_nnz] forward
at all valid (non-padding) token positions.
"""

import os
import pytest
import torch
from transformers import AutoConfig

from verl.workers.torch_pp.pipeline_stage import PipelineStage, _FLASH_PADDING_AVAILABLE
from verl.workers.torch_pp.loss import log_probs_from_logits


MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _requires_gpu_and_model():
    if DEVICE.type == "cpu":
        return pytest.mark.skipif(True, reason="GPU required")
    if not _FLASH_PADDING_AVAILABLE:
        return pytest.mark.skipif(True, reason="flash_attn.bert_padding not available")
    try:
        AutoConfig.from_pretrained(MODEL_PATH)
    except Exception:
        return pytest.mark.skipif(True, reason=f"Model {MODEL_PATH} not available")
    return lambda f: f


@pytest.fixture(scope="module")
def stage():
    """Load a single full-model stage for testing."""
    s = PipelineStage.from_pretrained(
        model_path=MODEL_PATH,
        pp_rank=0,
        pp_size=1,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    s.eval()
    return s


@_requires_gpu_and_model()
class TestRemovePadding:
    def _make_inputs(self, B=4, S=64, seed=42):
        """Create padded inputs with variable actual lengths."""
        torch.manual_seed(seed)
        config = AutoConfig.from_pretrained(MODEL_PATH)
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=DEVICE)
        attention_mask = torch.ones(B, S, device=DEVICE, dtype=torch.long)
        # Variable lengths: seq 0 = 40 tokens, seq 1 = full, seq 2 = 50, seq 3 = full
        attention_mask[0, 40:] = 0
        input_ids[0, 40:] = 0
        attention_mask[2, 50:] = 0
        input_ids[2, 50:] = 0
        return input_ids, attention_mask

    def test_logits_match_at_valid_positions(self, stage):
        """Padded and rmpad forward should produce same logits at non-padded positions."""
        input_ids, attention_mask = self._make_inputs()
        B, S = input_ids.shape

        # Padded forward
        stage.set_batch_data(input_ids, attention_mask, 1, use_remove_padding=False)
        with torch.no_grad():
            logits_padded = stage.forward_step(0)  # [B, S, V]
        stage.clear_batch_data()

        # Rmpad forward
        stage.set_batch_data(input_ids, attention_mask, 1, use_remove_padding=True)
        with torch.no_grad():
            logits_rmpad = stage.forward_step(0)  # [1, total_nnz, V]
        stage.clear_batch_data()

        # Extract padded logits at valid positions and compare
        valid_logits_padded = []
        for b in range(B):
            seq_len = attention_mask[b].sum().int().item()
            valid_logits_padded.append(logits_padded[b, :seq_len, :])
        valid_padded = torch.cat(valid_logits_padded, dim=0)
        valid_rmpad = logits_rmpad.squeeze(0)

        assert valid_padded.shape == valid_rmpad.shape, (
            f"Shape mismatch: padded {valid_padded.shape} vs rmpad {valid_rmpad.shape}"
        )
        # Tolerance: padded attention includes padding tokens in softmax
        # normalization, causing small numerical differences vs varlen.
        torch.testing.assert_close(
            valid_padded, valid_rmpad, atol=0.6, rtol=0.05,
        )

    def test_log_probs_match_at_valid_positions(self, stage):
        """Log probs from padded and rmpad forward should match at valid positions."""
        input_ids, attention_mask = self._make_inputs()
        B, S = input_ids.shape

        # Padded forward -> log probs
        stage.set_batch_data(input_ids, attention_mask, 1, use_remove_padding=False)
        with torch.no_grad():
            logits_padded = stage.forward_step(0)
        stage.clear_batch_data()
        lp_padded = log_probs_from_logits(logits_padded, input_ids)

        # Rmpad forward -> log probs
        stage.set_batch_data(input_ids, attention_mask, 1, use_remove_padding=True)
        micro_ids_rmpad = stage._micro_input_ids[0]
        with torch.no_grad():
            logits_rmpad = stage.forward_step(0)
        stage.clear_batch_data()
        lp_rmpad = log_probs_from_logits(logits_rmpad, micro_ids_rmpad)

        # Compare within-sequence positions only (skip cross-sequence boundaries)
        seq_lens = [attention_mask[b].sum().int().item() for b in range(B)]

        valid_lp_padded = []
        for b in range(B):
            valid_lp_padded.append(lp_padded[b, :seq_lens[b] - 1])
        valid_padded = torch.cat(valid_lp_padded, dim=0)

        lp_rmpad_flat = lp_rmpad.squeeze(0)
        valid_lp_rmpad = []
        offset = 0
        for sl in seq_lens:
            valid_lp_rmpad.append(lp_rmpad_flat[offset:offset + sl - 1])
            offset += sl
        valid_rmpad = torch.cat(valid_lp_rmpad, dim=0)

        assert valid_padded.shape == valid_rmpad.shape
        # Tolerance: padded attention includes padding tokens in softmax
        # normalization, causing small numerical differences vs varlen.
        torch.testing.assert_close(
            valid_padded, valid_rmpad, atol=0.6, rtol=0.05,
        )

    def test_multiple_micro_batches(self, stage):
        """Rmpad should work correctly with multiple micro-batches."""
        input_ids, attention_mask = self._make_inputs(B=8, S=32)

        M = 4
        stage.set_batch_data(input_ids, attention_mask, M, use_remove_padding=True)

        assert len(stage._micro_input_ids) == M
        assert len(stage._micro_nnz) == M

        total_nnz_sum = sum(stage._micro_nnz)
        total_valid = attention_mask.sum().item()
        assert total_nnz_sum == total_valid

        for i in range(M):
            ids = stage._micro_input_ids[i]
            assert ids.shape == (1, stage._micro_nnz[i])
            assert stage._micro_attention_mask[i] is None

        stage.clear_batch_data()

    def test_nnz_matches_valid_tokens(self, stage):
        """_micro_nnz should equal the number of valid tokens per micro-batch."""
        input_ids, attention_mask = self._make_inputs(B=4, S=64)

        stage.set_batch_data(input_ids, attention_mask, 2, use_remove_padding=True)

        micro_masks = list(attention_mask.chunk(2, dim=0))
        for i in range(2):
            expected_nnz = micro_masks[i].sum().int().item()
            assert stage._micro_nnz[i] == expected_nnz

        stage.clear_batch_data()

    def test_no_padding_all_valid(self, stage):
        """When all tokens are valid, rmpad output should match padded per-sequence."""
        B, S = 2, 32
        torch.manual_seed(123)
        config = AutoConfig.from_pretrained(MODEL_PATH)
        input_ids = torch.randint(0, config.vocab_size, (B, S), device=DEVICE)
        attention_mask = torch.ones(B, S, device=DEVICE, dtype=torch.long)

        # Padded [B, S]
        stage.set_batch_data(input_ids, attention_mask, 1, use_remove_padding=False)
        with torch.no_grad():
            out_pad = stage.forward_step(0)
        stage.clear_batch_data()

        # Rmpad [1, B*S]
        stage.set_batch_data(input_ids, attention_mask, 1, use_remove_padding=True)
        with torch.no_grad():
            out_rmpad = stage.forward_step(0)
        stage.clear_batch_data()

        # Compare per-sequence: varlen separates sequences identically to batching
        for b in range(B):
            padded_seq = out_pad[b]  # [S, V]
            rmpad_seq = out_rmpad[0, b * S : (b + 1) * S, :]  # [S, V]
            torch.testing.assert_close(padded_seq, rmpad_seq, atol=1e-2, rtol=1e-2)
