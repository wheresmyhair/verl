"""
Point-to-point communication for pipeline parallelism — NCCL-native.

Uses ``torch.distributed.send`` / ``recv`` directly on GPU tensors via NCCL.
Receiver knows the expected shape from model config, so no shape headers needed.

Design:
- Training activations flow forward:   stage i  ->  stage i+1
- Training gradients flow backward:    stage i  <-  stage i+1
- Inference activations flow reverse:  stage i  ->  stage i-1
- old_log_probs flow:                  rank 0   ->  rank P-1

Tag ranges (avoid collisions between concurrent micro-batches):
    training activations:   0      + micro_batch_id
    training gradients:     10000  + micro_batch_id
    inference activations:  20000  + micro_batch_id
    old_log_probs:          30000  + micro_batch_id
"""

from typing import Optional

import torch
import torch.distributed as dist


# Tags to avoid collisions between concurrent micro-batches
_ACT_TAG_BASE = 0
_GRAD_TAG_BASE = 10000
_INFER_ACT_TAG_BASE = 20000
_OLD_LOG_PROBS_TAG_BASE = 30000


def _act_tag(micro_batch_id: int) -> int:
    return _ACT_TAG_BASE + micro_batch_id


def _grad_tag(micro_batch_id: int) -> int:
    return _GRAD_TAG_BASE + micro_batch_id


def _infer_act_tag(micro_batch_id: int) -> int:
    return _INFER_ACT_TAG_BASE + micro_batch_id


def _old_log_probs_tag(micro_batch_id: int) -> int:
    return _OLD_LOG_PROBS_TAG_BASE + micro_batch_id


# -----------------------------------------------------------------------
# Global default PP group — set via set_pp_group() at init time
# -----------------------------------------------------------------------

_PP_GROUP: Optional[dist.ProcessGroup] = None
_PP_PAIR_GROUPS: Optional[dict] = None


def set_pp_group(group: dist.ProcessGroup):
    """Set the default process group for all PP communication."""
    global _PP_GROUP
    _PP_GROUP = group


def get_pp_group() -> Optional[dist.ProcessGroup]:
    """Get the default PP process group."""
    return _PP_GROUP


def set_pp_pair_groups(groups: dict):
    """Set the per-pair PP process groups."""
    global _PP_PAIR_GROUPS
    _PP_PAIR_GROUPS = groups


def get_pp_pair_groups() -> Optional[dict]:
    """Get the per-pair PP process groups."""
    return _PP_PAIR_GROUPS


# -----------------------------------------------------------------------
# Internal helpers — NCCL native, no shape headers
# -----------------------------------------------------------------------

def _send_tensor(
    tensor: torch.Tensor,
    dst_rank: int,
    tag: int,
    group: Optional[dist.ProcessGroup] = None,
) -> dist.Work:
    """Send a contiguous GPU tensor via non-blocking isend.

    Returns a Work handle. The caller must call handle.wait() before
    the step ends.
    """
    t = tensor.contiguous()
    return dist.isend(t, dst=dst_rank, tag=tag, group=group)


def _recv_tensor(
    shape: tuple,
    src_rank: int,
    tag: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    requires_grad: bool = False,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Receive a tensor of known shape via blocking recv.

    The receiver pre-allocates a buffer of the expected shape.
    """
    buf = torch.empty(shape, dtype=dtype, device=device)
    dist.recv(buf, src=src_rank, tag=tag, group=group)
    if requires_grad:
        buf.requires_grad_(True)
    return buf


# -----------------------------------------------------------------------
# Training activations (forward direction: rank r -> rank r+1)
# -----------------------------------------------------------------------


def send_activation(
    tensor: torch.Tensor,
    dst_rank: int,
    micro_batch_id: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> dist.Work:
    """Send activation (non-blocking). Returns Work handle."""
    return _send_tensor(tensor, dst_rank, _act_tag(micro_batch_id), group)


def recv_activation(
    shape: tuple,
    src_rank: int,
    micro_batch_id: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Receive activation tensor from the previous PP stage.

    Returns a tensor with ``requires_grad=True`` so gradients can flow back.
    """
    dev = device or torch.device("cuda")
    return _recv_tensor(
        shape, src_rank, _act_tag(micro_batch_id), dev, dtype,
        requires_grad=True, group=group,
    )


# -----------------------------------------------------------------------
# Training gradients (backward direction: rank r <- rank r+1)
# -----------------------------------------------------------------------


def send_grad(
    grad: torch.Tensor,
    dst_rank: int,
    micro_batch_id: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> dist.Work:
    """Send gradient (non-blocking). Returns Work handle."""
    return _send_tensor(grad, dst_rank, _grad_tag(micro_batch_id), group)


def recv_grad(
    shape: tuple,
    src_rank: int,
    micro_batch_id: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Receive gradient tensor from the next PP stage."""
    dev = device or torch.device("cuda")
    return _recv_tensor(
        shape, src_rank, _grad_tag(micro_batch_id), dev, dtype,
        requires_grad=False, group=group,
    )


# -----------------------------------------------------------------------
# Inference activations (reverse direction: rank r -> rank r-1)
# -----------------------------------------------------------------------


def send_infer_activation(
    tensor: torch.Tensor,
    dst_rank: int,
    micro_batch_id: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> dist.Work:
    """Send inference activation (non-blocking). rank r -> rank r-1."""
    return _send_tensor(tensor, dst_rank, _infer_act_tag(micro_batch_id), group)


def recv_infer_activation(
    shape: tuple,
    src_rank: int,
    micro_batch_id: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Receive inference activation (blocking). requires_grad=False."""
    dev = device or torch.device("cuda")
    return _recv_tensor(
        shape, src_rank, _infer_act_tag(micro_batch_id), dev, dtype,
        requires_grad=False, group=group,
    )


# -----------------------------------------------------------------------
# old_log_probs transfer: rank 0 -> rank P-1
# -----------------------------------------------------------------------


def send_old_log_probs(
    log_probs: torch.Tensor,
    dst_rank: int,
    micro_batch_id: int = 0,
    group: Optional[dist.ProcessGroup] = None,
) -> dist.Work:
    """Send old_log_probs from rank 0 (last infer stage) to rank P-1 (last train stage)."""
    return _send_tensor(log_probs, dst_rank, _old_log_probs_tag(micro_batch_id), group)


def recv_old_log_probs(
    shape: tuple,
    src_rank: int,
    micro_batch_id: int = 0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Receive old_log_probs on rank P-1 (blocking). requires_grad=False."""
    dev = device or torch.device("cuda")
    return _recv_tensor(
        shape, src_rank, _old_log_probs_tag(micro_batch_id), dev, dtype,
        requires_grad=False, group=group,
    )
