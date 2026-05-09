"""Gloo pair-group P2P for megatron fused-forward.

Why gloo for fused phase
------------------------
Megatron's stock NCCL P2P enforces symmetric rendezvous (sender +
receiver must submit matching ops in `groupStart/End` blocks); pre-post
on receiver side returns only after the matching peer call. This makes
the standard fused schedule (rank P-1 does iF burst before recv'ing
upstream tF) deadlock — exhaustively verified 2026-05-08~09 across all
pytorch entries (batched/raw/coalescing-manager).

Gloo `isend` is CPU-staged and **truly async**: the call returns
immediately with a Work handle; data lands on the receiver when the
matching `dist.recv` is issued. Cost is one D2H + H2D PCIe round-trip
per tensor (~6ms for ~144MB on our hardware), small relative to the
fused-forward saving (~30s/step on 16K-context Qwen3-8B).

Pair-group structure (mirrors torch_pp `_create_pp_pair_groups`)
----------------------------------------------------------------
- Train PP forward: per-edge gloo groups `[i, i+1]` for i in 0..P-2.
- Infer PP reverse: separate gloo groups for the same edges; the same
  edge carries train-backward and infer-forward in opposite logical
  directions, but gloo matches sends/recvs FIFO per direction, so we
  use distinct groups to avoid size-mismatch from interleaving.
- OLP (old-log-probs): rank 0 ↔ rank P-1 group (only when P >= 3).

API
---
    pair_groups = create_megatron_fused_pair_groups(pp_size, pp_rank)
    # pair_groups is a dict:
    #   pair_groups[i]            → adjacent train-pp edge i↔i+1 (i = min)
    #   pair_groups[f"infer_{i}"] → infer-pp edge i↔i+1
    #   pair_groups["olp"]        → 0↔P-1 for log-prob transfer

    # Send / recv via these groups using gloo isend/recv. See
    # verl/workers/torch_pp_workers.py:1690+ for reference impl.
"""
from __future__ import annotations
import logging
from typing import Dict, Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


# Module-level cache so repeated calls (e.g. actor + ref-model init)
# reuse the same groups.
_PAIR_GROUPS_CACHE: Optional[Dict] = None


def get_megatron_fused_pair_groups() -> Optional[Dict]:
    return _PAIR_GROUPS_CACHE


def set_megatron_fused_pair_groups(groups: Dict) -> None:
    global _PAIR_GROUPS_CACHE
    _PAIR_GROUPS_CACHE = groups


def create_megatron_fused_pair_groups(
    pp_size: int,
    pp_rank: int,
    pp_global_ranks: Optional[list[int]] = None,
) -> Dict:
    """Create gloo pair groups for the fused-forward phase.

    Must be called collectively by ALL ranks in the world (not just PP
    ranks) since `dist.new_group` is a world-wide collective. Caller is
    responsible for ensuring all ranks reach this point.

    Args:
        pp_size: pipeline parallel size
        pp_rank: this rank's PP index (0..pp_size-1)
        pp_global_ranks: optional global rank ids of the PP ranks. If
            None, assumes [0, 1, ..., pp_size-1] (single-PP world).
            For Megatron with TP+DP+PP, pass the actual global ranks for
            this PP group from `mpu.get_pipeline_model_parallel_group()`.

    Returns:
        Dict mapping group key → ProcessGroup.

    Cached at module level; subsequent calls return the cached dict.
    """
    cached = get_megatron_fused_pair_groups()
    if cached is not None:
        return cached

    if pp_global_ranks is None:
        pp_global_ranks = list(range(pp_size))
    assert len(pp_global_ranks) == pp_size

    pair_groups: Dict = {}

    # Adjacent train-pp edges
    for i in range(pp_size - 1):
        ranks = [pp_global_ranks[i], pp_global_ranks[i + 1]]
        g = dist.new_group(ranks=ranks, backend="gloo")
        if pp_rank == i or pp_rank == i + 1:
            pair_groups[i] = g

    # Adjacent infer-pp edges (same physical edges, separate gloo group
    # so train-backward and infer-forward don't FIFO-interfere on the
    # same group)
    for i in range(pp_size - 1):
        ranks = [pp_global_ranks[i], pp_global_ranks[i + 1]]
        g = dist.new_group(ranks=ranks, backend="gloo")
        if pp_rank == i or pp_rank == i + 1:
            pair_groups[f"infer_{i}"] = g

    # OLP: 0 ↔ P-1 (non-adjacent for P >= 3)
    if pp_size >= 3:
        ranks = [pp_global_ranks[0], pp_global_ranks[-1]]
        g = dist.new_group(ranks=ranks, backend="gloo")
        if pp_rank == 0 or pp_rank == pp_size - 1:
            pair_groups["olp"] = g

    set_megatron_fused_pair_groups(pair_groups)
    if pp_rank == 0:
        logger.info(
            "[megatron-fused] created %d train + %d infer pair gloo groups (P=%d)",
            pp_size - 1, pp_size - 1, pp_size,
        )
    return pair_groups


def get_pair_group(rank_a: int, rank_b: int, is_infer: bool = False,
                   pair_groups: Optional[Dict] = None):
    """Look up the gloo pair group for the (rank_a, rank_b) edge.

    For adjacent ranks (|a-b|==1), use train_i or infer_i where i = min(a,b).
    For non-adjacent (PP rank 0 ↔ P-1), use the "olp" group.
    """
    if pair_groups is None:
        pair_groups = _PAIR_GROUPS_CACHE
    if pair_groups is None:
        raise RuntimeError("megatron fused pair groups not created; "
                           "call create_megatron_fused_pair_groups first")
    if abs(rank_a - rank_b) == 1:
        i = min(rank_a, rank_b)
        key = f"infer_{i}" if is_infer else i
        return pair_groups[key]
    return pair_groups["olp"]


def gloo_p2p_send(tensor: torch.Tensor, dst_pp_rank: int,
                  pair_groups: Dict, is_infer: bool = False,
                  src_pp_rank: int = -1) -> dist.Work:
    """Truly-async CPU-staged send via gloo pair group.

    Caller must keep a reference to `tensor` (or the cpu-staged copy)
    until the returned Work completes. Easiest: stash in a list and
    drain after iF/tF/tB compute.

    Returns the Work handle; caller should `.wait()` before reusing
    or freeing the staged buffer.
    """
    t = tensor.detach().contiguous().cpu()
    group = get_pair_group(src_pp_rank, dst_pp_rank, is_infer=is_infer,
                           pair_groups=pair_groups)
    return dist.isend(t, dst=dst_pp_rank, group=group)


def gloo_p2p_recv(shape, src_pp_rank: int, dtype: torch.dtype,
                  pair_groups: Dict, device: torch.device,
                  is_infer: bool = False, dst_pp_rank: int = -1) -> torch.Tensor:
    """Synchronous CPU-staged recv via gloo pair group.

    Returns tensor on `device` (after H2D copy from CPU staging buffer).
    """
    buf = torch.empty(shape, dtype=dtype, device="cpu")
    group = get_pair_group(dst_pp_rank, src_pp_rank, is_infer=is_infer,
                           pair_groups=pair_groups)
    dist.recv(buf, src=src_pp_rank, group=group)
    return buf.to(device)
