"""M1 — smoke that reproduces the megatron NCCL P2P deadlock when fused
schedule places iF on rank 3 before any recv_forward.

Setup: PP=4 mock. Stock-megatron-style _communicate uses
batch_isend_irecv + wait(), which is blocking — same path as megatron's
P2PCommunicator._communicate.

Expected: rank 3 starts iF burst (just sleeps). rank 0..2 chain
send_forward(tF.k → next_rank) for k = 0..M-1. Rank 2's first
send_forward(tF.0 → r3) blocks because r3 isn't recv-ing. Cascading
block on r1, r0. After watchdog timeout (~30 min) NCCL kills.

Run:
  torchrun --nproc-per-node=4 \
    /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m1_deadlock.py

Pass criterion (paradoxically): we WANT to observe the hang.
We use a 60s test-side timeout (not NCCL watchdog) to assert the hang
exists. If all sends complete within 60s, deadlock did NOT occur.
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.distributed as dist


PP = 4
M = 16                 # micro-batches
SEQ = 8192             # tensor shape — large enough to fill NCCL internal buffers
HIDDEN = 4096
DTYPE = torch.bfloat16
IF_DURATION_S = 8.0    # rank-3 iF burst real wall time
TEST_TIMEOUT_S = 90.0


def _communicate_blocking(send_to_next, recv_from_prev, my_rank):
    """Stock megatron-style blocking _communicate: batch_isend_irecv + wait."""
    ops = []
    if send_to_next is not None and my_rank < PP - 1:
        ops.append(dist.P2POp(dist.isend, send_to_next, my_rank + 1))
    if recv_from_prev is not None and my_rank > 0:
        ops.append(dist.P2POp(dist.irecv, recv_from_prev, my_rank - 1))
    if not ops:
        return
    reqs = dist.batch_isend_irecv(ops)
    for r in reqs:
        r.wait()
    torch.cuda.synchronize()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == PP, f"need {PP} ranks, got {world_size}"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    print(f"[r{rank}] init done", flush=True)
    dist.barrier()
    t_start = time.perf_counter()

    if rank == PP - 1:
        # Rank 3: iF burst — DOES NOT recv_forward during this time.
        # Use real GEMM at scale + sleep to ensure r3 stays busy long enough
        # for upstream cascade to develop.
        print(f"[r{rank}] starting iF burst ({IF_DURATION_S}s, no recvs posted)", flush=True)
        t_iF_start = time.perf_counter()
        while time.perf_counter() - t_iF_start < IF_DURATION_S:
            t = torch.randn((4096, 4096), device="cuda", dtype=DTYPE)
            _ = (t @ t).sum()
            torch.cuda.synchronize()
        print(f"[r{rank}] iF burst done @ {time.perf_counter()-t_start:.1f}s", flush=True)
        # Now begin 1F1B-like backward+recv cycle:
        # for each k: recv tF.k from r2, do tB.k, send_backward to r2
        for k in range(M):
            buf = torch.empty((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
            _communicate_blocking(send_to_next=None, recv_from_prev=buf, my_rank=rank)
            grad = torch.randn((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
            # send_backward to r2 (in this minimal model: send "to prev" goes the other way)
            ops = [dist.P2POp(dist.isend, grad, rank - 1)]
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            print(f"[r{rank}] recv tF.{k} + send tB.{k} ok @ {time.perf_counter()-t_start:.1f}s", flush=True)
    else:
        # Ranks 0,1,2: send tF.k forward, recv tB.k backward (1F1B-like)
        for k in range(M):
            send = torch.randn((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
            recv_buf = None
            if rank > 0:
                recv_buf = torch.empty((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
            t0 = time.perf_counter()
            _communicate_blocking(
                send_to_next=send, recv_from_prev=recv_buf, my_rank=rank,
            )
            t_op = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_start
            print(f"[r{rank}] tF.{k} send/recv done in {t_op:.2f}s @ {elapsed:.1f}s", flush=True)
            if elapsed > TEST_TIMEOUT_S:
                print(f"[r{rank}] !!! TIMEOUT — deadlock observed @ {elapsed:.1f}s", flush=True)
                sys.exit(99)
            # Backward: recv grad from next, send grad to prev
            if rank < PP - 1:
                grad_in = torch.empty((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
                grad_send = torch.randn((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE) if rank > 0 else None
                ops = [dist.P2POp(dist.irecv, grad_in, rank + 1)]
                if grad_send is not None:
                    ops.append(dist.P2POp(dist.isend, grad_send, rank - 1))
                reqs = dist.batch_isend_irecv(ops)
                for r in reqs:
                    r.wait()
                elapsed = time.perf_counter() - t_start
                print(f"[r{rank}] tB.{k} recv/send done @ {elapsed:.1f}s", flush=True)
                if elapsed > TEST_TIMEOUT_S:
                    print(f"[r{rank}] !!! TIMEOUT (backward) @ {elapsed:.1f}s", flush=True)
                    sys.exit(99)

    dist.barrier()
    print(f"[r{rank}] all done @ {time.perf_counter()-t_start:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
