"""Pre-posted P2P recv buffers for Megatron pipeline-parallel fused forward.

Problem
-------
Megatron's stock `P2PCommunicator._communicate` calls
`batch_isend_irecv` + `wait()` — blocking. When a rank is busy doing
local compute (e.g. inference forward iF on the last PP rank during
fused schedule warmup), upstream `send_forward` calls **stall** until
the busy rank finally posts a matching recv. The stall lasts ≈ iF
duration; if long enough, NCCL watchdog (~30 min) kills the job.

Fix
---
Pre-post all expected `irecv` requests before the busy phase starts.
NCCL has the recv buffer ready, so upstream `isend`s complete
immediately. The busy rank later "consumes" pre-posted recvs in
schedule order (each `recv_forward` call just `wait()`s the queued
request and returns its buffer).

API
---
    comm = PrePostP2PCommunicator(stock_comm)            # wraps existing P2PCommunicator
    comm.pre_post_recv_forward(num_micro_batches, shape) # before iF burst
    comm.pre_post_recv_backward(num_micro_batches, shape)
    # ... fused schedule executes; each recv_forward / recv_backward
    # call consumes from the queue.

Memory cost
-----------
Pre-posting all M recvs simultaneously costs M × tensor_size bytes
of GPU HBM per direction. For Qwen3-8B PP=4 TP=1 (mb=1, seq≈18K,
hidden=4096, bf16): ~144 MB per buffer × 128 = 18 GB per rank
(forward only). With TP=4 sharding: ~36 MB × 128 = 4.6 GB.

K-window optimization (TODO if memory-bound):
    pre_post_recv_forward(K, shape, refill_to=M)
where K << M, and a callback re-posts when each completes.
"""
from __future__ import annotations
import logging
from collections import deque
from typing import Any, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class PrePostP2PCommunicator:
    """Wraps `megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator`
    with pre-post recv queues for fused forward.

    Compatible drop-in: passes through `send_forward`, `send_backward`,
    `_communicate`, etc. Only `recv_forward` and `recv_backward` are
    overridden to consume from queue when one exists.
    """

    def __init__(self, stock: Any):
        """stock: an instance of megatron.core.pipeline_parallel.p2p_communication.P2PCommunicator"""
        self._stock = stock
        self._pending_fwd: deque[Tuple[Any, torch.Tensor]] = deque()
        self._pending_bwd: deque[Tuple[Any, torch.Tensor]] = deque()

    # ---- Pass-through stock attrs ----
    @property
    def pp_group(self):
        return self._stock.pp_group

    @property
    def config(self):
        return self._stock.config

    @property
    def next_rank(self):
        return self._stock.next_rank

    @property
    def prev_rank(self):
        return self._stock.prev_rank

    @property
    def virtual_pipeline_model_parallel_size(self):
        return self._stock.virtual_pipeline_model_parallel_size

    # ---- Pre-post API ----
    def pre_post_recv_forward(self, num_buffers: int, tensor_shape) -> None:
        """Issue `num_buffers` non-blocking irecv ops from prev_rank.
        Buffers + req handles are stashed; subsequent `recv_forward()`
        calls consume them in FIFO order.
        """
        if num_buffers <= 0:
            return
        for _ in range(num_buffers):
            tensor, _, reqs = self._stock._communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=True,
                recv_next=False,
                tensor_shape=tensor_shape,
                wait_on_reqs=False,
            )
            self._pending_fwd.append((reqs, tensor))
        logger.debug(
            "pre_post_recv_forward: queued %d buffers (shape=%s, total=%d)",
            num_buffers, tensor_shape, len(self._pending_fwd),
        )

    def pre_post_recv_backward(self, num_buffers: int, tensor_shape) -> None:
        """Issue `num_buffers` non-blocking irecv ops from next_rank."""
        if num_buffers <= 0:
            return
        for _ in range(num_buffers):
            _, tensor, reqs = self._stock._communicate(
                tensor_send_next=None,
                tensor_send_prev=None,
                recv_prev=False,
                recv_next=True,
                tensor_shape=tensor_shape,
                wait_on_reqs=False,
            )
            self._pending_bwd.append((reqs, tensor))
        logger.debug(
            "pre_post_recv_backward: queued %d buffers (shape=%s, total=%d)",
            num_buffers, tensor_shape, len(self._pending_bwd),
        )

    def has_pending_fwd(self) -> bool:
        return len(self._pending_fwd) > 0

    def has_pending_bwd(self) -> bool:
        return len(self._pending_bwd) > 0

    def drain_pending(self) -> None:
        """Wait on all unconsumed pre-posted recvs (cleanup)."""
        while self._pending_fwd:
            reqs, _ = self._pending_fwd.popleft()
            self._wait_reqs(reqs)
        while self._pending_bwd:
            reqs, _ = self._pending_bwd.popleft()
            self._wait_reqs(reqs)

    @staticmethod
    def _wait_reqs(reqs):
        """`reqs` from `_communicate` may be list or dict-of-handles."""
        if reqs is None:
            return
        if isinstance(reqs, dict):
            for r in reqs.values():
                r.wait()
        else:
            for r in reqs:
                r.wait()

    # ---- Override recv_* ----
    def recv_forward(self, tensor_shapes, is_first_stage: bool):
        """Consume pre-posted forward recvs if available; else fall back to stock."""
        if is_first_stage:
            return self._stock.recv_forward(tensor_shapes, is_first_stage)
        # tensor_shapes might be a single shape or list of shapes (multimodule).
        # We support the single-shape case only for now (the standard config).
        if not _is_single_shape(tensor_shapes):
            # Fall back; multimodule pre-post not implemented.
            return self._stock.recv_forward(tensor_shapes, is_first_stage)
        if not self._pending_fwd:
            return self._stock.recv_forward(tensor_shapes, is_first_stage)
        reqs, tensor = self._pending_fwd.popleft()
        self._wait_reqs(reqs)
        return tensor

    def recv_backward(self, tensor_shapes, is_last_stage: bool):
        if is_last_stage:
            return self._stock.recv_backward(tensor_shapes, is_last_stage)
        if not _is_single_shape(tensor_shapes):
            return self._stock.recv_backward(tensor_shapes, is_last_stage)
        if not self._pending_bwd:
            return self._stock.recv_backward(tensor_shapes, is_last_stage)
        reqs, tensor = self._pending_bwd.popleft()
        self._wait_reqs(reqs)
        return tensor

    # ---- Pass-through send / combined ----
    def send_forward(self, output_tensors, is_last_stage: bool) -> None:
        return self._stock.send_forward(output_tensors, is_last_stage)

    def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
        return self._stock.send_backward(input_tensor_grads, is_first_stage)

    def send_forward_recv_backward(self, *args, **kwargs):
        # If we have pre-posted bwd, peel off the recv portion to consume queue.
        # For paranoia, only override when no kwargs ambiguity. Current Megatron
        # signature: (output_tensors, tensor_shapes, is_last_stage)
        return self._stock.send_forward_recv_backward(*args, **kwargs)

    def send_backward_recv_forward(self, *args, **kwargs):
        return self._stock.send_backward_recv_forward(*args, **kwargs)

    def send_forward_recv_forward(self, *args, **kwargs):
        return self._stock.send_forward_recv_forward(*args, **kwargs)

    def send_backward_recv_backward(self, *args, **kwargs):
        return self._stock.send_backward_recv_backward(*args, **kwargs)

    def send_forward_backward_recv_forward_backward(self, *args, **kwargs):
        return self._stock.send_forward_backward_recv_forward_backward(*args, **kwargs)

    def _communicate(self, *args, **kwargs):
        return self._stock._communicate(*args, **kwargs)

    def _communicate_shapes(self, *args, **kwargs):
        return self._stock._communicate_shapes(*args, **kwargs)


def _is_single_shape(tensor_shapes) -> bool:
    """Return True if `tensor_shapes` is a single shape (e.g. tuple of ints,
    torch.Size) rather than a list of shapes."""
    if hasattr(tensor_shapes, "__len__") and len(tensor_shapes) == 0:
        return True
    if isinstance(tensor_shapes, torch.Size):
        return True
    if isinstance(tensor_shapes, (tuple, list)):
        # If any element is an int, this is a single shape; if all elements are
        # iterable (tuples/lists/sizes), this is a list-of-shapes.
        if len(tensor_shapes) > 0 and isinstance(tensor_shapes[0], int):
            return True
        return False
    return True
