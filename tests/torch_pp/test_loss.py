"""Tests for loss computation functions."""

import pytest
import torch
from verl.workers.torch_pp.loss import (
    log_probs_from_logits,
    entropy_from_logits,
    gather_response_log_probs,
    compute_grpo_loss,
)


@pytest.fixture
def device():
    return torch.device("cpu")


class TestLogProbsFromLogits:
    def test_shape(self, device):
        B, S, V = 4, 32, 100
        logits = torch.randn(B, S, V, device=device)
        labels = torch.randint(0, V, (B, S), device=device)
        result = log_probs_from_logits(logits, labels)
        assert result.shape == (B, S - 1)

    def test_values_are_negative(self, device):
        """Log probabilities should be <= 0."""
        logits = torch.randn(2, 16, 50, device=device)
        labels = torch.randint(0, 50, (2, 16), device=device)
        result = log_probs_from_logits(logits, labels)
        assert (result <= 0).all()

    def test_high_logit_gives_high_log_prob(self, device):
        """If the correct token has the highest logit, log_prob should be close to 0."""
        B, S, V = 1, 4, 10
        logits = torch.zeros(B, S, V, device=device)
        labels = torch.zeros(B, S, dtype=torch.long, device=device)
        # Make token 0 very likely at every position
        logits[:, :, 0] = 100.0
        result = log_probs_from_logits(logits, labels)
        # log_prob should be close to 0 (near certainty)
        assert (result > -0.01).all()

    def test_matches_manual_computation(self, device):
        B, S, V = 1, 3, 5
        logits = torch.randn(B, S, V, device=device)
        labels = torch.tensor([[2, 0, 4]], device=device)
        result = log_probs_from_logits(logits, labels)
        # Manual: shifted logits[:, :-1] predict labels[:, 1:]
        shift_logits = logits[:, :-1]
        shift_labels = labels[:, 1:]  # [[0, 4]]
        manual_lp = torch.log_softmax(shift_logits, dim=-1)
        manual_gathered = manual_lp[0, 0, 0], manual_lp[0, 1, 4]
        torch.testing.assert_close(result[0, 0], manual_gathered[0])
        torch.testing.assert_close(result[0, 1], manual_gathered[1])


class TestEntropyFromLogits:
    def test_shape(self, device):
        logits = torch.randn(4, 32, 100, device=device)
        result = entropy_from_logits(logits)
        assert result.shape == (4, 31)

    def test_entropy_non_negative(self, device):
        logits = torch.randn(2, 16, 50, device=device)
        result = entropy_from_logits(logits)
        assert (result >= -1e-5).all()

    def test_peaked_distribution_low_entropy(self, device):
        logits = torch.zeros(1, 4, 10, device=device)
        logits[:, :, 0] = 100.0  # Very peaked
        result = entropy_from_logits(logits)
        assert (result < 0.01).all()

    def test_uniform_distribution_max_entropy(self, device):
        V = 10
        logits = torch.zeros(1, 4, V, device=device)  # Uniform
        result = entropy_from_logits(logits)
        expected = torch.log(torch.tensor(float(V)))
        torch.testing.assert_close(result[0, 0], expected, atol=1e-5, rtol=1e-5)


class TestGatherResponseLogProbs:
    def test_basic_gather(self, device):
        B, S_minus_1, R = 2, 15, 5
        full_lp = torch.arange(S_minus_1, dtype=torch.float, device=device).unsqueeze(0).expand(B, -1)
        # Response starts at position 5 → shifted index = 4
        resp_starts = torch.tensor([5, 5], device=device)
        result = gather_response_log_probs(full_lp, resp_starts, R)
        assert result.shape == (B, R)
        # Adjusted = 5-1=4, so gather indices [4,5,6,7,8]
        expected = torch.tensor([4.0, 5.0, 6.0, 7.0, 8.0])
        torch.testing.assert_close(result[0], expected)

    def test_clamps_at_boundaries(self, device):
        B, S_minus_1, R = 1, 10, 5
        full_lp = torch.ones(B, S_minus_1, device=device)
        # Start near the end — should clamp indices
        resp_starts = torch.tensor([9], device=device)
        result = gather_response_log_probs(full_lp, resp_starts, R)
        assert result.shape == (1, R)
        # Should not raise — clamped indices are valid


class TestComputeGrpoLoss:
    @pytest.fixture
    def grpo_inputs(self, device):
        B, S, V, R = 2, 16, 50, 8
        logits = torch.randn(B, S, V, device=device, requires_grad=True)
        input_ids = torch.randint(0, V, (B, S), device=device)
        resp_starts = torch.tensor([S - R] * B, device=device)
        old_lp = torch.randn(B, R, device=device) * 0.1
        advantages = torch.randn(B, device=device)
        resp_mask = torch.ones(B, R, device=device)
        ref_lp = torch.randn(B, R, device=device) * 0.1
        return {
            "logits": logits,
            "input_ids": input_ids,
            "response_start_positions": resp_starts,
            "old_log_probs": old_lp,
            "advantages": advantages,
            "response_mask": resp_mask,
            "ref_log_probs": ref_lp,
        }

    def test_loss_is_scalar(self, grpo_inputs):
        loss, stats = compute_grpo_loss(**grpo_inputs)
        assert loss.dim() == 0

    def test_loss_is_differentiable(self, grpo_inputs):
        loss, _ = compute_grpo_loss(**grpo_inputs)
        loss.backward()
        assert grpo_inputs["logits"].grad is not None
        assert grpo_inputs["logits"].grad.shape == grpo_inputs["logits"].shape

    def test_stats_keys(self, grpo_inputs):
        _, stats = compute_grpo_loss(**grpo_inputs)
        assert "loss" in stats
        assert "pg_loss" in stats
        assert "ratio_mean" in stats
        assert "clip_frac" in stats
        assert "kl_per_token" in stats

    def test_zero_advantage_gives_near_zero_pg_loss(self, grpo_inputs):
        grpo_inputs["advantages"] = torch.zeros_like(grpo_inputs["advantages"])
        grpo_inputs["ref_log_probs"] = None
        loss, stats = compute_grpo_loss(**grpo_inputs, kl_coef=0.0)
        assert abs(stats["pg_loss"]) < 1e-5

    def test_masked_tokens_dont_contribute(self, grpo_inputs):
        # Mask out all tokens
        grpo_inputs["response_mask"] = torch.zeros_like(grpo_inputs["response_mask"])
        grpo_inputs["ref_log_probs"] = None
        loss, _ = compute_grpo_loss(**grpo_inputs, kl_coef=0.0)
        assert abs(loss.item()) < 1e-5
