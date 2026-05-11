"""M4 — Option B: schedule reorder to avoid deadlock.

Rank 3 schedule: tF.0 → iF burst → tB.0, tF.1, tB.1, ...
(vs deadlock-prone: iF burst → tF.0 → tB.0, ...)

Single-PP setup: each rank does standard 1F1B; rank 3 alone has iF burst
inserted between tF.0 and tB.0.

The cascade during r3's iF burst is bounded by:
  - 1F1B warmup depth (r2 has 2 pending tFs, r1 has 3, r0 has 4)
  - NCCL internal P2P buffer

For PP=4 M=8 schedule:
  r0: tF.0 tF.1 tF.2 tF.3 tB.0 tF.4 tB.1 tF.5 tB.2 tF.6 tB.3 tF.7 tB.4 tB.5 tB.6 tB.7
  r1: tF.0 tF.1 tF.2 tB.0 tF.3 tB.1 tF.4 tB.2 tF.5 tB.3 tF.6 tB.4 tF.7 tB.5 tB.6 tB.7
  r2: tF.0 tF.1 tB.0 tF.2 tB.1 tF.3 tB.2 tF.4 tB.3 tF.5 tB.4 tF.6 tB.5 tF.7 tB.6 tB.7
  r3: tF.0 [iF.0..iF.M-1 burst] tB.0 tF.1 tB.1 tF.2 tB.2 ... tF.M-1 tB.M-1

Each rank does send_forward / send_backward / recv_forward / recv_backward
via Megatron-style _communicate (blocking batch_isend_irecv + wait).

If iF burst is short enough that the bounded cascade resolves before
NCCL's 30-min watchdog, schedule completes successfully.
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.distributed as dist


PP = 4
M = 8                  # micro-batches
SEQ = 4096
HIDDEN = 4096
DTYPE = torch.bfloat16
IF_DURATION_S = 4.0    # rank-3 iF burst (single LOCAL compute, no P2P)
COMPUTE_PER_OP_S = 0.1
TEST_TIMEOUT_S = 120.0


def communicate_send_recv(send_to_next, recv_from_prev, send_to_prev, recv_from_next, my_rank):
    """Megatron-style _communicate: combined batch_isend_irecv + wait."""
    ops = []
    if send_to_next is not None and my_rank < PP - 1:
        ops.append(dist.P2POp(dist.isend, send_to_next, my_rank + 1))
    if recv_from_prev is not None and my_rank > 0:
        ops.append(dist.P2POp(dist.irecv, recv_from_prev, my_rank - 1))
    if send_to_prev is not None and my_rank > 0:
        ops.append(dist.P2POp(dist.isend, send_to_prev, my_rank - 1))
    if recv_from_next is not None and my_rank < PP - 1:
        ops.append(dist.P2POp(dist.irecv, recv_from_next, my_rank + 1))
    if not ops:
        return
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    torch.cuda.synchronize()


def fake_compute(secs: float):
    end = time.perf_counter() + secs
    while time.perf_counter() < end:
        t = torch.randn((2048, 2048), device="cuda", dtype=DTYPE)
        _ = (t @ t).sum()
        torch.cuda.synchronize()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == PP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    print(f"[r{rank}] init", flush=True)
    dist.barrier()
    t_start = time.perf_counter()

    shape = (SEQ, 1, HIDDEN)
    is_first = (rank == 0)
    is_last = (rank == PP - 1)

    # Per-rank schedule: list of ("op", mb_id) where op ∈ {tF, tB, iF}
    # Standard 1F1B for r0..r2; r3 has iF burst between tF.0 and tB.0.
    # Build schedule:
    schedule = []
    n_warmup = PP - rank  # for non-rank-3 in 1F1B
    if rank == PP - 1:
        # rank 3 special: tF.0, iF burst, tB.0, tF.1, tB.1, ..., tF.M-1, tB.M-1
        schedule.append(("tF", 0))
        for k in range(M):
            schedule.append(("iF", k))
        for k in range(M):
            schedule.append(("tB", k))
            if k + 1 < M:
                schedule.append(("tF", k + 1))
    else:
        # Standard 1F1B
        # Warmup: n_warmup forwards
        for k in range(n_warmup):
            schedule.append(("tF", k))
        # Steady-state: alternating tB/tF
        for k in range(M - n_warmup):
            schedule.append(("tB", k))
            schedule.append(("tF", k + n_warmup))
        # Cooldown: remaining backwards
        for k in range(M - n_warmup, M):
            schedule.append(("tB", k))

    # Verify schedule has M tF + M tB (+ M iF for rank 3)
    counts = {"tF": 0, "tB": 0, "iF": 0}
    for op, _ in schedule:
        counts[op] += 1
    print(f"[r{rank}] schedule len={len(schedule)} {counts}", flush=True)

    # Storage for activations / grads to use across ops
    activations = {}  # mb_id → tensor (output of tF, used by tB)
    incoming = {}     # mb_id → tensor (input from prev rank, used by tF)
    grads_in = {}     # mb_id → grad tensor for tB

    def do_tF(mb):
        # recv from prev, compute, send to next
        if not is_first:
            buf = torch.empty(shape, device="cuda", dtype=DTYPE)
            ops = [dist.P2POp(dist.irecv, buf, rank - 1)]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs: r.wait()
            incoming[mb] = buf
        else:
            incoming[mb] = torch.randn(shape, device="cuda", dtype=DTYPE)
        # fake compute
        fake_compute(COMPUTE_PER_OP_S)
        out = incoming[mb] + 0.001  # fake forward
        activations[mb] = out
        if not is_last:
            ops = [dist.P2POp(dist.isend, out, rank + 1)]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs: r.wait()

    def do_tB(mb):
        if not is_last:
            buf = torch.empty(shape, device="cuda", dtype=DTYPE)
            ops = [dist.P2POp(dist.irecv, buf, rank + 1)]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs: r.wait()
            grads_in[mb] = buf
        else:
            grads_in[mb] = torch.randn(shape, device="cuda", dtype=DTYPE)
        fake_compute(COMPUTE_PER_OP_S)
        grad_out = grads_in[mb] + 0.001
        if not is_first:
            ops = [dist.P2POp(dist.isend, grad_out, rank - 1)]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs: r.wait()

    def do_iF_burst():
        """Local compute on rank 3 simulating M iFs."""
        fake_compute(IF_DURATION_S)

    # Execute schedule
    for op_idx, (op, mb) in enumerate(schedule):
        elapsed = time.perf_counter() - t_start
        if elapsed > TEST_TIMEOUT_S:
            print(f"[r{rank}] !!! TIMEOUT @ {elapsed:.1f}s op={op}.{mb}", flush=True)
            sys.exit(99)
        if op == "iF":
            # Only first iF op triggers the burst (we lump all M iFs into one wall block)
            if mb == 0:
                t0 = time.perf_counter()
                do_iF_burst()
                print(f"[r{rank}] iF burst done in {time.perf_counter()-t0:.1f}s @ {time.perf_counter()-t_start:.1f}s", flush=True)
            continue  # don't do P2P for individual iFs in this smoke
        elif op == "tF":
            t0 = time.perf_counter()
            do_tF(mb)
            t_op = time.perf_counter() - t0
        elif op == "tB":
            t0 = time.perf_counter()
            do_tB(mb)
            t_op = time.perf_counter() - t0
        elapsed = time.perf_counter() - t_start
        if op_idx % 3 == 0 or t_op > 0.3:
            print(f"[r{rank}] op{op_idx} {op}.{mb} took {t_op:.2f}s @ {elapsed:.1f}s", flush=True)

    dist.barrier()
    elapsed = time.perf_counter() - t_start
    print(f"[r{rank}] all done @ {elapsed:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
