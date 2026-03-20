"""
GRPO / PPO-clip policy gradient loss.

Runs on the last PP stage (which has the LM head and produces logits).
All functions are differentiable — used inside autograd training loop.

Per response-token:
    ratio       = exp(new_log_prob - old_log_prob)
    clipped     = clamp(ratio, 1-e, 1+e)
    pg_loss     = -min(ratio * advantage, clipped * advantage)
    kl_loss     = beta * (new_log_prob - ref_log_prob)
    token_loss  = pg_loss + kl_loss - alpha * entropy
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from verl.utils.torch_functional import (
    logprobs_from_logits_v2 as _logprobs_from_logits_v2,
    entropy_from_logits_with_chunking as _entropy_from_logits_chunked,
)

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss as _flash_cross_entropy
    _FLASH_CE_AVAILABLE = True
except ImportError:
    _FLASH_CE_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def log_probs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Per-token log probs (shifted for causal LM).

    Uses memory-efficient implementation: flash-attn cross_entropy if
    available, otherwise logsumexp trick (avoids materializing full
    softmax). Falls back to row-by-row processing for bf16 stability.

    Args:
        logits: [B, S, V]
        labels: [B, S]

    Returns:
        [B, S-1]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    # Flash-attn Triton cross_entropy is fastest but requires CUDA tensors
    if _FLASH_CE_AVAILABLE and shift_logits.is_cuda:
        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        output = _flash_cross_entropy(flat_logits, flat_labels, inplace_backward=True)
        assert isinstance(output, tuple), (
            "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
        )
        return -output[0].view(shift_logits.shape[:-1])
    # Fallback: logsumexp trick (float32) or row-by-row log_softmax (bf16)
    return _logprobs_from_logits_v2(shift_logits, shift_labels)


def entropy_from_logits(
    logits: torch.Tensor,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """Per-token entropy (shifted), computed in chunks.

    Chunks along the flattened (B*S) dimension to cap peak memory from
    softmax/logsumexp intermediates over the vocab dimension.

    Args:
        logits: [B, S, V]
        chunk_size: number of token rows per chunk (default 1024)

    Returns:
        [B, S-1]
    """
    shift_logits = logits[:, :-1, :].contiguous()
    B, S_minus_1, V = shift_logits.shape
    flat = shift_logits.view(-1, V)  # [B*(S-1), V]
    ent_flat = _entropy_from_logits_chunked(flat, chunk_size=chunk_size)
    return ent_flat.view(B, S_minus_1)


def gather_response_log_probs(
    full_log_probs: torch.Tensor,
    response_start_positions: torch.Tensor,
    max_resp_len: int,
) -> torch.Tensor:
    """
    Extract response-portion log probs from full-sequence shifted log probs.

    Args:
        full_log_probs: [B, S-1]
        response_start_positions: [B] (original, un-shifted positions)
        max_resp_len: R (columns of old_log_probs)

    Returns:
        [B, R]
    """
    device = full_log_probs.device
    adjusted = (response_start_positions - 1).clamp(min=0)
    offsets = torch.arange(max_resp_len, device=device).unsqueeze(0)  # [1, R]
    indices = (adjusted.unsqueeze(1) + offsets).clamp(max=full_log_probs.size(1) - 1)
    return full_log_probs.gather(1, indices)


# ──────────────────────────────────────────────────────────────────────
# Main loss
# ──────────────────────────────────────────────────────────────────────


def compute_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_start_positions: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    kl_coef: float = 0.001,
    entropy_coef: float = 0.0,
    loss_agg: str = "token-mean",
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    GRPO / PPO-clip policy gradient loss.

    Args:
        logits:                   [B, S, V] from forward pass
        input_ids:                [B, S] token IDs
        response_start_positions: [B]
        old_log_probs:            [B, R] from behavior policy
        advantages:               [B] per-sequence
        response_mask:            [B, R]
        ref_log_probs:            [B, R] optional reference policy
        clip_ratio:               PPO epsilon
        kl_coef:                  KL penalty weight
        entropy_coef:             entropy bonus weight
        loss_agg:                 "token-mean" or "seq-mean"

    Returns:
        (loss, stats_dict)
    """
    R = old_log_probs.size(1)

    # New log probs (differentiable)
    new_full_lp = log_probs_from_logits(logits, input_ids)  # [B, S-1]
    new_resp_lp = gather_response_log_probs(new_full_lp, response_start_positions, R)
    new_resp_lp = new_resp_lp * response_mask  # [B, R]

    # Ratio
    log_ratio = new_resp_lp - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    # Policy gradient loss
    adv = advantages.unsqueeze(1)  # [B, 1]
    pg_loss = torch.max(-ratio * adv, -clipped * adv)  # [B, R]

    # KL penalty
    kl_loss = torch.zeros_like(pg_loss)
    if ref_log_probs is not None and kl_coef > 0.0:
        kl_loss = kl_coef * (new_resp_lp - ref_log_probs)

    # Entropy bonus
    ent_loss = torch.zeros_like(pg_loss)
    per_token_entropy = None
    if entropy_coef > 0.0:
        full_ent = entropy_from_logits(logits)
        per_token_entropy = gather_response_log_probs(
            full_ent, response_start_positions, R
        ) * response_mask
        ent_loss = -entropy_coef * per_token_entropy

    # Combine and aggregate
    token_loss = (pg_loss + kl_loss + ent_loss) * response_mask
    num_tokens = response_mask.sum().clamp(min=1)

    if loss_agg == "token-mean":
        loss = token_loss.sum() / num_tokens
    elif loss_agg == "seq-mean":
        per_seq = response_mask.sum(dim=1).clamp(min=1)
        loss = (token_loss.sum(dim=1) / per_seq).mean()
    else:
        raise ValueError(f"Unknown loss_agg: {loss_agg}")

    # Logging stats (detached)
    with torch.no_grad():
        valid = response_mask.bool()
        stats = {
            "loss": loss.item(),
            "pg_loss": (pg_loss * response_mask).sum().item() / num_tokens.item(),
            "ratio_mean": ratio[valid].mean().item() if valid.any() else 0.0,
            "ratio_max": ratio[valid].max().item() if valid.any() else 0.0,
            "clip_frac": ((ratio[valid] - 1.0).abs() > clip_ratio).float().mean().item()
            if valid.any()
            else 0.0,
        }
        if ref_log_probs is not None and kl_coef > 0.0:
            stats["kl_per_token"] = (
                (new_resp_lp - ref_log_probs)[valid].mean().item() if valid.any() else 0.0
            )
        if per_token_entropy is not None:
            stats["entropy"] = per_token_entropy[valid].mean().item() if valid.any() else 0.0

    return loss, stats
