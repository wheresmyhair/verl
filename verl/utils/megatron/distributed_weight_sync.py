"""NCCL-collective weight sync between megatron actor and sglang engine.

Replaces verl's CUDA-IPC-based `update_weights_from_tensor` path when
the container lacks `CAP_SYS_PTRACE` (so `pidfd_getfd` fails). NCCL P2P
goes over NVLink/PCIe and doesn't need ptrace permission.

See `docs/rlpipe/option_b_distributed_weight_sync_plan.md` for the
full design + rationale.

Usage (per-rank, both actor and rollout sides)
---------------------------------------------

    coord = DistributedWeightSyncCoordinator(
        sglang_tp_size=1, master_addr="127.0.0.1", master_port=29600,
        group_name="rlpipe_weight_update",
    )

    # On actor PP rank 0 only — joins the update group as group rank 0
    if actor_pp_rank == 0:
        actor_pg = coord.init_actor_side()

    # On the verl rollout worker — forwards init to sglang engine
    coord.init_sglang_side(rollout_engine)

    # Per-step (after per_tensor_generator yields per-rank tensors)
    if actor_pp_rank == 0:
        # Step 1: send sglang the (name, dtype, shape) meta + kick off recv
        names, dtypes, shapes = [], [], []
        tensors = []
        for name, t in per_tensor_param:
            names.append(name); dtypes.append(str(t.dtype).removeprefix("torch."))
            shapes.append(list(t.shape)); tensors.append(t)
        # Sglang side allocates buffers and posts dist.broadcast(empty, src=0)
        await rollout.update_weights_distributed(
            coord, names=names, dtypes=dtypes, shapes=shapes,
        )
        # Step 2: actor rank 0 broadcasts each tensor (matches sglang's recvs)
        for t in tensors:
            coord.broadcast(t)

Critical constraints (from sglang code 2026-05-09)
--------------------------------------------------

- `update_weights_from_distributed` always uses `src=0` for the
  collective broadcast. ⇒ actor rank 0 must be group rank 0.
- Group size = 1 + sglang_tp_size (one actor sender + N sglang
  receivers). PP ranks > 0 do NOT join.
- The group is created via TCPStore rendezvous (master_addr:port),
  separate from megatron's existing pp_group / world group. Both
  actor and sglang sides must reach `init_custom_process_group`
  collectively or it'll hang on rendezvous.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.distributed as dist


logger = logging.getLogger(__name__)


@dataclass
class DistributedWeightSyncCoordinator:
    """Manages the NCCL update group between actor PP rank 0 and sglang TP ranks.

    Stateful — call `init_actor_side` once per actor process (or once on
    rank 0 only), and `init_sglang_side` once per rollout worker.
    """
    sglang_tp_size: int
    master_addr: str = "127.0.0.1"
    master_port: int = 29600
    group_name: str = "rlpipe_weight_update"
    # Backend: "nccl" requires distinct GPUs per rank in the group.
    # In hybrid mode actor + sglang scheduler share the SAME GPU per
    # DP replica → NCCL fails with "Duplicate GPU detected". Use "gloo"
    # which CPU-stages tensors (one D2H + one H2D per broadcast).
    backend: str = "gloo"

    # Filled at init_actor_side
    _update_pg: Optional[dist.ProcessGroup] = field(default=None, init=False)
    _initialized: bool = field(default=False, init=False)

    @property
    def world_size(self) -> int:
        """Group spans actor rank 0 + sglang TP ranks."""
        return 1 + self.sglang_tp_size

    def init_actor_side(self) -> dist.ProcessGroup:
        """Called by actor PP rank 0. Returns the new ProcessGroup spanning
        actor (rank 0 in this group) + sglang TP ranks (ranks 1..N).

        WARNING: this MUST be called collectively with `init_sglang_side`
        on the rollout side, both reaching their TCPStore rendezvous at
        `master_addr:master_port`. Otherwise the call hangs.
        """
        from sglang.srt.utils.common import init_custom_process_group
        if self._initialized:
            return self._update_pg

        logger.info(
            "[DistributedWeightSync] actor side init: master=%s:%d, world=%d, rank=0, group=%s",
            self.master_addr, self.master_port, self.world_size, self.group_name,
        )
        self._update_pg = init_custom_process_group(
            backend=self.backend,
            init_method=f"tcp://{self.master_addr}:{self.master_port}",
            world_size=self.world_size,
            rank=0,
            group_name=self.group_name,
        )
        self._initialized = True
        return self._update_pg

    def init_sglang_side(self, engine) -> None:
        """Called on the rollout worker; forwards to engine.init_weights_update_group.

        engine: an `sglang.srt.entrypoints.engine.Engine` (or
        `AsyncEngine` / `AsyncHttpServerAdapter` — the API is mirrored).
        """
        logger.info(
            "[DistributedWeightSync] sglang side init: master=%s:%d, world=%d, rank_offset=1, group=%s",
            self.master_addr, self.master_port, self.world_size, self.group_name,
        )
        # `rank_offset=1` because actor is at rank 0; sglang TP=0 is rank 1, etc.
        result = engine.init_weights_update_group(
            master_address=self.master_addr,
            master_port=self.master_port,
            rank_offset=1,
            world_size=self.world_size,
            group_name=self.group_name,
            backend=self.backend,
        )
        # `init_weights_update_group` returns a tuple from the dispatcher:
        # (success, message). For AsyncEngine it's wrapped in awaitable.
        if hasattr(result, "__await__"):
            # Caller's responsibility to await — this is a sync wrapper that
            # returns the awaitable. Async-side is in
            # SGLangRollout.init_weights_update_group.
            return result
        return result

    def broadcast(self, tensor: torch.Tensor) -> None:
        """Actor rank 0 sends `tensor` to all sglang TP ranks via the update group.

        For gloo backend (default; required when actor+sglang share GPU
        in hybrid mode), tensor is CPU-staged before broadcast. Sglang side
        receives via empty CPU buffer + dist.broadcast then moves to GPU.

        For nccl backend (requires distinct GPUs per rank), tensor stays on GPU.
        """
        if not self._initialized:
            raise RuntimeError("init_actor_side() must be called first")
        if self.backend == "gloo":
            # Stage to CPU; gloo's send is truly async w.r.t. receiver.
            send_buf = tensor.detach().contiguous().cpu()
            dist.broadcast(send_buf, src=0, group=self._update_pg, async_op=False)
        else:
            dist.broadcast(tensor, src=0, group=self._update_pg, async_op=False)

    def destroy(self) -> None:
        if self._update_pg is not None:
            try:
                dist.destroy_process_group(self._update_pg)
            except Exception:
                logger.warning("[DistributedWeightSync] destroy failed", exc_info=True)
            self._update_pg = None
        self._initialized = False


def make_meta_lists(per_tensor_iter):
    """Helper: split a (name, tensor) iterator into parallel name/dtype/shape/tensor lists.

    Used to feed `update_weights_from_distributed` (needs meta) + the
    matching `broadcast` calls (need the tensors).
    """
    names: List[str] = []
    dtypes: List[str] = []
    shapes: List[List[int]] = []
    tensors: List[torch.Tensor] = []
    for name, t in per_tensor_iter:
        names.append(name)
        # sglang accepts torch.dtype string without "torch." prefix
        dtype_name = str(t.dtype).removeprefix("torch.")
        dtypes.append(dtype_name)
        shapes.append(list(t.shape))
        tensors.append(t)
    return names, dtypes, shapes, tensors
