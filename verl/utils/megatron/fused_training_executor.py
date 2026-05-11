"""Step 5b-C scaffolding: wires `FusedPPExecutor` (5b-B) into actual
actor/infer model forward+backward.

Architecture
------------
- Schedule comes from `build_fused_schedule(P, M)` (5b-A).
- `FusedPPExecutor` (5b-B) walks the schedule and at each op boundary
  co-batches per-peer P2P. Compute is delegated to `compute_op_fn`.
- This module supplies `compute_op_fn` for real models:
    tF.k → actor_module forward; save (input, output) for tB.k
    tB.k → torch.autograd.backward on saved output with received grad
    iF.k → infer_module forward (no grad); log_probs at last_inf

Status (Step 5b-C V1, 2026-05-11)
---------------------------------
- Real iF path: uses infer_module + vocab_parallel_log_probs_from_logits
  (same compute as Step 5a `compute_log_prob_reverse_pp`).
- tF/tB path: uses verl's `gptmodel_forward` for actor; saves activation
  tensors with `requires_grad_(True)` on input; backward via
  `torch.autograd.backward` with received gradient (from upstream rank).
- Loss: PLACEHOLDER `(output ** 2).sum()` on last_train rank. Real PPO
  loss (with old_log_probs from iF + advantages + KL) is Step 5b-D.
- Optimizer.step() and grad allreduce: NOT done here; caller must wrap
  this with proper grad handling. For V1 testing, we just verify the
  schedule runs without deadlock and produces matching tensor shapes.

The placeholder loss means gradients are wrong for actual training, so
this V1 path is for ARCHITECTURE validation only — env-gated by
`RLPIPE_FUSED_USE_DUALPIPE_V2=1`.
"""
from __future__ import annotations
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


def _parse_op(op: str):
    kind, mb = op.split(".")
    return kind, int(mb)


def run_fused_forward_backward(
    *,
    actor_module,
    infer_module,
    micro_batches: List[Dict[str, torch.Tensor]],
    pp_group,
    pp_world_ranks: List[int],
    temperature: float = 1.0,
    hidden_size: Optional[int] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Drive one round of fused fwd+bwd+iF via FusedPPExecutor.

    Returns:
        log_probs: (total_B, response_length) on every PP rank
                   (broadcast from last_inf rank 0).
        metrics: dict of scalar metrics (placeholder values for V1).
    """
    from verl.utils.megatron.fused_schedule import build_fused_schedule, verify_dependencies
    from verl.utils.megatron.fused_pp_executor import FusedPPExecutor, FusedOpCtx
    from verl.utils.megatron_utils import unwrap_model
    from verl.models.mcore.model_forward import gptmodel_forward
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    rank = dist.get_rank(group=pp_group)
    pp_size = dist.get_world_size(group=pp_group)
    M = len(micro_batches)
    if M == 0:
        return torch.zeros(0), {}

    schedule = build_fused_schedule(pp_size, M)
    if rank == 0:
        verify_dependencies(schedule)
        logger.info("[fused] schedule (P=%d M=%d):", pp_size, M)
        for r in range(pp_size):
            logger.info("  r%d: %s", r, " ".join(schedule[r]))

    my_sched = schedule[rank]

    # Probe hidden shape from first mb (assumes same shape across mbs)
    sample = micro_batches[0]
    bsz, seq_len = sample["input_ids"].shape
    if hidden_size is None:
        hidden_size = infer_module.config.hidden_size
    # Megatron PP transfer uses (S, B, H) layout
    hidden_shape = (seq_len, bsz, hidden_size)
    device = next(infer_module.parameters()).device

    executor = FusedPPExecutor(
        schedule=my_sched, rank=rank, pp_group=pp_group, pp_world_ranks=pp_world_ranks,
        hidden_shape=hidden_shape, dtype=dtype, device=device,
    )

    # Per-mb state
    tF_saved: Dict[int, Tuple[Optional[torch.Tensor], torch.Tensor]] = {}
    tF_loss: Dict[int, torch.Tensor] = {}
    iF_log_probs: Dict[int, torch.Tensor] = {}

    actor_unwrapped = unwrap_model(actor_module[0])
    is_first_train = (rank == 0)
    is_last_train = (rank == pp_size - 1)
    is_first_inf = (rank == pp_size - 1)
    is_last_inf = (rank == 0)

    def compute_op(ctx: FusedOpCtx):
        mb = micro_batches[ctx.mb]
        if ctx.kind == "tF":
            input_tensor = ctx.recv_hidden
            if input_tensor is not None:
                input_tensor = input_tensor.detach().requires_grad_(True)
                actor_unwrapped.set_input_tensor(input_tensor)
            # Forward through actor directly (bypass verl's pack/recover-padding
            # wrappers; we need fixed (S, B, H) shape for our pre-allocated
            # P2P buffers, which the wrappers compress via remove_left_padding).
            output = actor_module[0](
                input_ids=mb["input_ids"],
                attention_mask=mb["attention_mask"],
                position_ids=mb["position_ids"],
            )
            tF_saved[ctx.mb] = (input_tensor, output)
            if ctx.is_last_train:
                # Placeholder loss; real PPO loss is Step 5b-D
                loss = (output ** 2).sum() / M
                tF_loss[ctx.mb] = loss
                return None  # last_train tF has no send (loss stays local)
            return output
        elif ctx.kind == "tB":
            input_tensor, output_tensor = tF_saved.pop(ctx.mb)
            output_grad = ctx.recv_grad_output
            if ctx.is_last_train:
                loss = tF_loss.pop(ctx.mb)
                loss.backward()
            else:
                if output_grad is None:
                    raise RuntimeError(f"r{ctx.rank} tB.{ctx.mb}: missing recv_grad_output on non-last_train")
                torch.autograd.backward([output_tensor], [output_grad])
            if ctx.is_first_train:
                return None  # first_train tB doesn't send
            return input_tensor.grad if input_tensor is not None else None
        elif ctx.kind == "iF":
            with torch.no_grad():
                hidden_in = ctx.recv_hidden_inf
                if hidden_in is not None:
                    infer_module.set_input_tensor(hidden_in)
                output = infer_module(
                    input_ids=mb["input_ids"],
                    position_ids=mb["position_ids"],
                    attention_mask=mb["attention_mask"],
                )
            if ctx.is_last_inf:
                # logits → log_probs over responses (same as Step 5a)
                logits = output / temperature
                responses = mb["responses"]
                position_ids = mb["position_ids"]
                response_length = responses.size(1)
                label = position_ids.clone()
                label[:, -response_length - 1 : -1] = responses
                log_probs = vocab_parallel_log_probs_from_logits(logits, label)
                log_probs = log_probs[:, -response_length - 1 : -1].contiguous()
                iF_log_probs[ctx.mb] = log_probs
                return None
            return output.contiguous()

    executor.run(compute_op)

    # Aggregate log_probs across mb at last_inf, then broadcast to all PP ranks
    last_inf_world = pp_world_ranks[pp_size - 1 - (pp_size - 1)]  # rank with rev_pp=P-1 = global rank 0
    response_length = micro_batches[0]["responses"].shape[1]
    total_B = sum(mb["responses"].shape[0] for mb in micro_batches)
    if is_last_inf:
        all_log_probs = torch.cat([iF_log_probs[k] for k in range(M)], dim=0).to(torch.float32)
    else:
        all_log_probs = torch.empty((total_B, response_length), dtype=torch.float32, device=device)
    dist.broadcast(all_log_probs, src=last_inf_world, group=pp_group)

    # V1 metrics: just shape + placeholder loss; real metrics in 5b-D
    metrics = {
        "fused/n_micro_batches": float(M),
        "fused/n_ops_per_rank": float(len(my_sched)),
    }
    return all_log_probs, metrics
