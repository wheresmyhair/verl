"""Generic fused-PP executor for mixed tF/tB/iF schedules over NCCL P2P.

Builds on the DualPipe insight (see `dualpipe_executor.py`):
- Each rank executes its own op list independently
- At each op boundary, co-batch all of this rank's P2P traffic in a
  single `dist.batch_isend_irecv` group
- NCCL FIFO matches sends/recvs across ranks; no deadlock as long as
  the schedule is dep-correct

Design
------
This module is a thin scheduling shell. It doesn't know how to compute
tF/tB/iF — the caller injects `compute_op_fn(kind, mb, ctx)` that does
the actual work. `ctx` carries per-op state across the boundary:
- For tF: `ctx.recv_hidden` (or None if first stage); output goes into
  `ctx.send_hidden` for transmission to the next rank.
- For tB: `ctx.recv_grad_output` (or None if last stage); output goes
  into `ctx.send_grad_input`.
- For iF: `ctx.recv_hidden_inf` (reverse direction); output goes into
  `ctx.send_hidden_inf`.

The executor handles:
- P2P direction (tF: r→r+1, tB: r→r-1, iF: r→r-1 reverse)
- Co-batched commit at each boundary
- Final drain for last op's send

Tags
----
Each (kind, mb) gets a distinct tag so NCCL FIFO doesn't mix:
  tF tag = 100 + mb
  tB tag = 200 + mb
  iF tag = 300 + mb
(Up to mb=99 supported; trivial to extend.)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import logging
import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


TAG_BASE = {"tF": 100, "tB": 200, "iF": 300}


def _parse_op(op: str) -> Tuple[str, int]:
    kind, mb = op.split(".")
    return kind, int(mb)


def _need_send(kind: str, rank: int, pp_size: int) -> bool:
    """Does this rank's op produce output to send to a peer?"""
    if kind == "tF":
        return rank < pp_size - 1  # tF flows 0→P-1; last stage doesn't send
    if kind == "tB":
        return rank > 0  # tB flows P-1→0; first stage doesn't send
    if kind == "iF":
        return rank > 0  # iF flows P-1→0 (reverse); rank-0 (last_inf) doesn't send
    raise ValueError(kind)


def _need_recv(kind: str, rank: int, pp_size: int) -> bool:
    """Does this rank's op consume input from a peer?"""
    if kind == "tF":
        return rank > 0
    if kind == "tB":
        return rank < pp_size - 1
    if kind == "iF":
        return rank < pp_size - 1  # first_inf (rank P-1) doesn't recv
    raise ValueError(kind)


def _peer_send(kind: str, rank: int) -> int:
    """Peer global rank this op's output goes to."""
    if kind == "tF":
        return rank + 1
    if kind == "tB":
        return rank - 1
    if kind == "iF":
        return rank - 1
    raise ValueError(kind)


def _peer_recv(kind: str, rank: int) -> int:
    """Peer global rank this op's input comes from."""
    if kind == "tF":
        return rank - 1
    if kind == "tB":
        return rank + 1
    if kind == "iF":
        return rank + 1
    raise ValueError(kind)


@dataclass
class FusedOpCtx:
    """Per-op context passed to user-provided compute_op_fn.

    The executor fills the `recv_*` slots before calling compute_op_fn;
    compute_op_fn fills the `send_*` slots (or returns the output, which
    the executor copies into the pre-allocated send buffer).

    Shape: hidden tensors are always `(hidden_shape)` tuples; the
    executor pre-allocates with that shape and dtype.
    """
    op: str  # "kind.mb"
    kind: str
    mb: int
    rank: int
    pp_size: int
    recv_hidden: Optional[torch.Tensor] = None         # tF input
    recv_grad_output: Optional[torch.Tensor] = None    # tB grad-of-output
    recv_hidden_inf: Optional[torch.Tensor] = None     # iF input
    is_first_train: bool = False
    is_last_train: bool = False
    is_first_inf: bool = False
    is_last_inf: bool = False


class FusedPPExecutor:
    """Run a per-rank fused schedule (tF/tB/iF mix) via DualPipe co-batched
    NCCL P2P.

    Args:
        schedule: list of "kind.mb" strings (rank-local op order)
        rank: this rank's index in pp_group (== global rank for PP=world)
        pp_group: distributed PP process group
        pp_world_ranks: list of global ranks in pp_group (in pp-order)
        hidden_shape: shape for activation tensors (used for send/recv buffers)
        dtype: tensor dtype (typically bf16)
        device: device for buffer alloc
    """

    def __init__(
        self,
        *,
        schedule: List[str],
        rank: int,
        pp_group,
        pp_world_ranks: List[int],
        hidden_shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.schedule = schedule
        self.rank = rank
        self.pp_group = pp_group
        self.pp_world_ranks = pp_world_ranks
        self.pp_size = len(pp_world_ranks)
        self.hidden_shape = hidden_shape
        self.dtype = dtype
        self.device = device

        self.is_first_train = (rank == 0)
        self.is_last_train = (rank == self.pp_size - 1)
        self.is_first_inf = (rank == self.pp_size - 1)
        self.is_last_inf = (rank == 0)

        # Pre-allocate send/recv buffers by (kind, mb)
        self.send_bufs: Dict[Tuple[str, int], torch.Tensor] = {}
        self.recv_bufs: Dict[Tuple[str, int], torch.Tensor] = {}
        for op in schedule:
            kind, mb = _parse_op(op)
            if _need_send(kind, rank, self.pp_size):
                self.send_bufs[(kind, mb)] = torch.empty(hidden_shape, dtype=dtype, device=device)
            if _need_recv(kind, rank, self.pp_size):
                self.recv_bufs[(kind, mb)] = torch.empty(hidden_shape, dtype=dtype, device=device)

    def _make_tag(self, kind: str, mb: int) -> int:
        return TAG_BASE[kind] + mb

    def _build_recv_op(self, kind: str, mb: int) -> dist.P2POp:
        peer = self._global_rank_of(_peer_recv(kind, self.rank))
        buf = self.recv_bufs[(kind, mb)]
        return dist.P2POp(op=dist.irecv, tensor=buf, peer=peer, group=self.pp_group, tag=self._make_tag(kind, mb))

    def _build_send_op(self, kind: str, mb: int) -> dist.P2POp:
        peer = self._global_rank_of(_peer_send(kind, self.rank))
        buf = self.send_bufs[(kind, mb)]
        return dist.P2POp(op=dist.isend, tensor=buf, peer=peer, group=self.pp_group, tag=self._make_tag(kind, mb))

    def _global_rank_of(self, in_group_rank: int) -> int:
        return self.pp_world_ranks[in_group_rank]

    def run(self, compute_op_fn: Callable[[FusedOpCtx], torch.Tensor]) -> Dict[Tuple[str, int], torch.Tensor]:
        """Execute the schedule.

        compute_op_fn(ctx) returns the output tensor for this op (or None
        if no send is needed, e.g., on last-rank for tF where output is loss).
        The executor copies the returned output into the pre-allocated send
        buffer (if applicable) for transmission at the next boundary.

        Returns: dict of (kind, mb) -> output for ops whose output isn't
                 sent over P2P (e.g., last-stage outputs). Caller uses this
                 to collect final results (loss for last train rank,
                 log_probs for last_inf rank).
        """
        local_outputs: Dict[Tuple[str, int], torch.Tensor] = {}
        # Track compute outputs that need to be sent at NEXT boundary
        pending_compute_out: Dict[Tuple[str, int], torch.Tensor] = {}

        n = len(self.schedule)
        for i, op in enumerate(self.schedule):
            kind, mb = _parse_op(op)

            comm_ops = []

            # Sends for previous op's output (co-batched at this boundary)
            if i > 0:
                prev_kind, prev_mb = _parse_op(self.schedule[i - 1])
                if _need_send(prev_kind, self.rank, self.pp_size):
                    prev_key = (prev_kind, prev_mb)
                    if prev_key in pending_compute_out:
                        self.send_bufs[prev_key].copy_(pending_compute_out[prev_key])
                        del pending_compute_out[prev_key]
                    comm_ops.append(self._build_send_op(prev_kind, prev_mb))

            # Recv for this op's input
            if _need_recv(kind, self.rank, self.pp_size):
                comm_ops.append(self._build_recv_op(kind, mb))

            if comm_ops:
                import os
                debug = os.environ.get("RLPIPE_FUSED_DEBUG", "0") == "1"
                mode = os.environ.get("RLPIPE_FUSED_COMM_MODE", "batched")  # "batched" | "unbatched"
                if debug:
                    summary = [(o.op.__name__, o.peer, o.tag) for o in comm_ops]
                    print(f"[r{self.rank}] b{i} {op}: mode={mode} comm={summary}", flush=True)
                if mode == "unbatched":
                    # Submit each P2P op individually via dist.isend/irecv.
                    # This avoids NCCL coalescing across heterogeneous peers,
                    # which we suspect of hanging in some schedule patterns.
                    all_reqs = []
                    for o in comm_ops:
                        if o.op is dist.isend:
                            req = dist.isend(o.tensor, o.peer, group=o.group, tag=o.tag)
                        else:
                            req = dist.irecv(o.tensor, o.peer, group=o.group, tag=o.tag)
                        all_reqs.append(req)
                    for r in all_reqs:
                        r.wait()
                else:
                    reqs = dist.batch_isend_irecv(comm_ops)
                    for r in reqs:
                        r.wait()
                if debug:
                    print(f"[r{self.rank}] b{i} {op}: comm done", flush=True)

            # Local compute
            ctx = FusedOpCtx(
                op=op, kind=kind, mb=mb, rank=self.rank, pp_size=self.pp_size,
                is_first_train=self.is_first_train, is_last_train=self.is_last_train,
                is_first_inf=self.is_first_inf, is_last_inf=self.is_last_inf,
            )
            if _need_recv(kind, self.rank, self.pp_size):
                if kind == "tF":
                    ctx.recv_hidden = self.recv_bufs[(kind, mb)]
                elif kind == "tB":
                    ctx.recv_grad_output = self.recv_bufs[(kind, mb)]
                elif kind == "iF":
                    ctx.recv_hidden_inf = self.recv_bufs[(kind, mb)]

            output = compute_op_fn(ctx)

            if _need_send(kind, self.rank, self.pp_size):
                # output will be sent at NEXT boundary
                if output is None:
                    raise RuntimeError(f"compute_op_fn returned None but op {op} needs to send")
                pending_compute_out[(kind, mb)] = output
            else:
                # Last-stage output (loss for tF on last_train, log_probs for iF on last_inf, or no output)
                if output is not None:
                    local_outputs[(kind, mb)] = output

        # Final drain: send the last op's output if needed
        if n > 0:
            last_kind, last_mb = _parse_op(self.schedule[-1])
            if _need_send(last_kind, self.rank, self.pp_size):
                last_key = (last_kind, last_mb)
                if last_key not in pending_compute_out:
                    raise RuntimeError(
                        f"final op {self.schedule[-1]} needs to send but compute output is missing"
                    )
                self.send_bufs[last_key].copy_(pending_compute_out[last_key])
                reqs = dist.batch_isend_irecv([self._build_send_op(last_kind, last_mb)])
                for r in reqs:
                    r.wait()

        return local_outputs
