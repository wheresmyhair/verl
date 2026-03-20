"""Numerical correctness test: partitioned PP forward == full model forward.

Loads a small HF model, runs full forward, then partitions into PP stages
and runs the same input through each stage sequentially. Verifies the
logits match.

Requires: Qwen/Qwen3-0.6B or set TEST_MODEL_PATH env var.
"""

import os
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from verl.workers.torch_pp.partitioner import compute_layer_assignment
from verl.workers.torch_pp.pipeline_stage import PipelineStage
from verl.workers.torch_pp.loss import log_probs_from_logits


MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen3-0.6B")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PP_SIZE = 4


def _requires_model():
    """Skip if model not available."""
    if DEVICE.type == "cpu":
        return pytest.mark.skipif(True, reason="GPU required for model test")
    try:
        AutoConfig.from_pretrained(MODEL_PATH)
    except Exception:
        return pytest.mark.skipif(True, reason=f"Model {MODEL_PATH} not available")
    return lambda f: f


@pytest.fixture(scope="module")
def full_model_output():
    """Run full model forward once and cache the result."""
    config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).to(DEVICE).eval()

    B, S = 2, 64
    torch.manual_seed(42)
    input_ids = torch.randint(0, config.vocab_size, (B, S), device=DEVICE)
    attention_mask = torch.ones(B, S, device=DEVICE, dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    logits = out.logits.clone()
    lp = log_probs_from_logits(logits, input_ids)

    # Free full model
    del model
    torch.cuda.empty_cache()

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "logits": logits,
        "log_probs": lp,
    }


@_requires_model()
class TestPPNumericalCorrectness:
    @pytest.fixture(scope="class")
    def pp_stages(self):
        """Build all PP stages once for the test class."""
        stages = []
        for rank in range(PP_SIZE):
            stage = PipelineStage.from_pretrained(
                model_path=MODEL_PATH,
                pp_rank=rank,
                pp_size=PP_SIZE,
                device=DEVICE,
                dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            stage.eval()
            stages.append(stage)
        return stages

    def test_pp_forward_matches_full_model(self, pp_stages, full_model_output):
        """Partitioned sequential forward should produce same logits as full model."""
        input_ids = full_model_output["input_ids"]
        attention_mask = full_model_output["attention_mask"]
        expected_logits = full_model_output["logits"]
        B, S = input_ids.shape

        hidden = None
        M = 1  # single micro-batch for simplicity
        with torch.no_grad():
            for rank, stage in enumerate(pp_stages):
                stage.set_batch_data(input_ids, attention_mask, M)
                if stage.is_first:
                    hidden = stage.forward_step(0)
                else:
                    hidden = stage.forward_step(0, hidden)
                stage.clear_batch_data()

        # hidden is now the logits from the last stage
        pp_logits = hidden
        assert pp_logits.shape == expected_logits.shape, (
            f"Shape mismatch: {pp_logits.shape} vs {expected_logits.shape}"
        )

        # Check logits are close (bf16 allows some tolerance)
        torch.testing.assert_close(
            pp_logits, expected_logits, atol=1e-2, rtol=1e-2,
        )

    def test_pp_log_probs_match(self, pp_stages, full_model_output):
        """Log probs from PP forward should match full model log probs."""
        input_ids = full_model_output["input_ids"]
        attention_mask = full_model_output["attention_mask"]
        expected_lp = full_model_output["log_probs"]

        hidden = None
        M = 1
        with torch.no_grad():
            for rank, stage in enumerate(pp_stages):
                stage.set_batch_data(input_ids, attention_mask, M)
                if stage.is_first:
                    hidden = stage.forward_step(0)
                else:
                    hidden = stage.forward_step(0, hidden)
                stage.clear_batch_data()

        pp_lp = log_probs_from_logits(hidden, input_ids)
        torch.testing.assert_close(pp_lp, expected_lp, atol=1e-2, rtol=1e-2)

    def test_all_stages_have_params(self, pp_stages):
        """Each PP stage should have a meaningful number of parameters."""
        for rank, stage in enumerate(pp_stages):
            num_params = sum(p.numel() for p in stage.parameters())
            assert num_params > 0, f"Stage {rank} has no parameters"
            # Each stage should have at least one transformer layer's worth
            assert num_params > 1_000_000, (
                f"Stage {rank} has suspiciously few params: {num_params:,}"
            )
