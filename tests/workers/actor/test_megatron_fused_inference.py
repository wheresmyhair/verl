"""Smoke test for MegatronFusedInferenceStage.

Phase 1 verification: load the HF model, run forward on a toy batch, check
output shape / dtype / no-NaN. Numerical equivalence against Megatron's
`compute_log_prob` is checked at integration time (run the fused script and
verify `pg_loss` is near zero on the first step before the policy updates).
"""

import torch

from verl.workers.actor.megatron_fused_inference import MegatronFusedInferenceStage


def _build_toy_batch(
    batch_size: int = 2,
    prompt_len: int = 8,
    response_len: int = 16,
    vocab_size: int = 1000,
    device: torch.device = torch.device("cuda"),
):
    """Build a toy padded batch (prompt + response, left-padded prompt)."""
    total_len = prompt_len + response_len
    # Fixed RNG so test is reproducible
    g = torch.Generator(device="cpu").manual_seed(42)
    input_ids = torch.randint(
        low=1, high=vocab_size, size=(batch_size, total_len), generator=g
    ).to(device)
    # All positions valid (no actual padding) for simplicity
    attention_mask = torch.ones(batch_size, total_len, dtype=torch.int64, device=device)
    position_ids = (
        torch.arange(total_len, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .clone()
    )
    return input_ids, attention_mask, position_ids


def test_inference_stage_smoke():
    """Load Qwen3-1.7B, run forward, check output shape + no NaN."""
    if not torch.cuda.is_available():
        print("[test_megatron_fused_inference] CUDA unavailable — skipping")
        return

    device = torch.device("cuda:0")
    stage = MegatronFusedInferenceStage(
        model_path="Qwen/Qwen3-1.7B",
        device=device,
        dtype=torch.bfloat16,
    )
    print(f"[test_megatron_fused_inference] loaded: {stage}")

    batch_size = 2
    prompt_len = 8
    response_len = 16
    input_ids, attn_mask, pos_ids = _build_toy_batch(
        batch_size=batch_size,
        prompt_len=prompt_len,
        response_len=response_len,
        vocab_size=min(1000, stage.model.config.vocab_size),
        device=device,
    )

    log_probs = stage.forward_log_probs(
        input_ids=input_ids,
        attention_mask=attn_mask,
        position_ids=pos_ids,
        response_length=response_len,
        temperature=1.0,
    )

    assert log_probs.shape == (batch_size, response_len), (
        f"expected ({batch_size}, {response_len}), got {tuple(log_probs.shape)}"
    )
    assert log_probs.dtype == torch.bfloat16, f"unexpected dtype {log_probs.dtype}"
    assert torch.isfinite(log_probs).all(), "log_probs contains NaN/Inf"
    # log prob should be <= 0 always
    assert (log_probs <= 1e-3).all(), (
        f"log_probs contain values > 0 (max={log_probs.max().item()})"
    )
    # And not pathologically small
    assert log_probs.min().item() > -100.0, (
        f"log_probs contain values <= -100 (min={log_probs.min().item()})"
    )

    print(
        f"[test_megatron_fused_inference] OK — log_probs shape={tuple(log_probs.shape)}"
        f" dtype={log_probs.dtype}"
        f" range=[{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]"
    )


def test_inference_stage_temperature():
    """Verify temperature scaling affects the distribution."""
    if not torch.cuda.is_available():
        print("[test_megatron_fused_inference] CUDA unavailable — skipping")
        return

    device = torch.device("cuda:0")
    stage = MegatronFusedInferenceStage(
        model_path="Qwen/Qwen3-1.7B",
        device=device,
        dtype=torch.bfloat16,
    )

    input_ids, attn_mask, pos_ids = _build_toy_batch(
        batch_size=1, prompt_len=4, response_len=8, vocab_size=1000, device=device
    )

    lp_t1 = stage.forward_log_probs(
        input_ids, attn_mask, pos_ids, response_length=8, temperature=1.0
    )
    lp_t2 = stage.forward_log_probs(
        input_ids, attn_mask, pos_ids, response_length=8, temperature=2.0
    )
    # Higher temperature flattens the softmax → log probs of chosen tokens move toward
    # log(1/V) (more negative, closer to mean). Typically |lp_t2| > |lp_t1| for the
    # high-probability continuations this toy input lands on.
    # We just assert they're not identical.
    diff = (lp_t1 - lp_t2).abs().max().item()
    assert diff > 1e-4, (
        f"temperature had no effect — max diff {diff}"
    )
    print(
        f"[test_megatron_fused_inference] temperature OK — t=1 vs t=2 max diff {diff:.4f}"
    )


if __name__ == "__main__":
    test_inference_stage_smoke()
    test_inference_stage_temperature()
