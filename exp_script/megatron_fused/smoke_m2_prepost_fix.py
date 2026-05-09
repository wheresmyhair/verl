"""M2 — verify pre-post irecv eliminates the rank-3 iF stall.

Setup mirrors M1, but rank 3 pre-posts M irecv ops BEFORE its iF burst.
Stock megatron's `_communicate` allocates recv buffer + irecv inline at
each `recv_forward` call — that's why upstream stalls.

If pre-post works: upstream send_forwards complete immediately even while
r3 does iF; r0/r1/r2 also enter their own iF burst in parallel; total
wall ≈ max(iF) instead of (upstream stall) + (iF) + ...

Run:
  torchrun --nproc-per-node=4 \
    /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m2_prepost_fix.py

Pass: r0/r1/r2's tF.0 sends should complete in << IF_DURATION_S, not = IF_DURATION_S.
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.distributed as dist


PP = 4
M = 16
SEQ = 8192
HIDDEN = 4096
DTYPE = torch.bfloat16
IF_DURATION_S = 8.0
TEST_TIMEOUT_S = 90.0


def pre_post_irecvs(num_buffers, src_rank, shape, dtype):
    """Pre-allocate `num_buffers` recv buffers and post irecv on each.
    Returns list of (req_handle, buffer)."""
    pending = []
    for _ in range(num_buffers):
        buf = torch.empty(shape, device="cuda", dtype=dtype)
        op = dist.P2POp(dist.irecv, buf, src_rank)
        reqs = dist.batch_isend_irecv([op])
        # batch_isend_irecv returns list of work objects
        pending.append((reqs[0], buf))
    return pending


def consume_pre_posted(pending, idx):
    """Wait for the idx-th pre-posted recv to complete, return buffer."""
    req, buf = pending[idx]
    req.wait()
    return buf


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == PP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    print(f"[r{rank}] init done", flush=True)
    dist.barrier()
    t_start = time.perf_counter()

    if rank == PP - 1:
        # Step 1: PRE-POST M irecvs from r2 (before iF burst!)
        t_pp = time.perf_counter()
        pending_fwd = pre_post_irecvs(M, rank - 1, (SEQ, 1, HIDDEN), DTYPE)
        t_pp_done = time.perf_counter() - t_pp
        print(f"[r{rank}] pre-posted {M} irecvs in {t_pp_done*1000:.1f}ms", flush=True)

        # Step 2: iF burst (real GEMM at scale)
        print(f"[r{rank}] starting iF burst ({IF_DURATION_S}s)", flush=True)
        t_iF_start = time.perf_counter()
        while time.perf_counter() - t_iF_start < IF_DURATION_S:
            t = torch.randn((4096, 4096), device="cuda", dtype=DTYPE)
            _ = (t @ t).sum()
            torch.cuda.synchronize()
        print(f"[r{rank}] iF burst done @ {time.perf_counter()-t_start:.1f}s", flush=True)

        # Step 3: consume pre-posted recvs (verify all M arrived)
        for k in range(M):
            buf = consume_pre_posted(pending_fwd, k)
            print(f"[r{rank}] consume tF.{k} ok @ {time.perf_counter()-t_start:.1f}s", flush=True)
    else:
        # Ranks 0/1/2: send tF.k → next (forward direction only — that's
        # what gets stalled by r3-no-recv). Skip backward direction in
        # smoke (would need real 1F1B schedule).
        for k in range(M):
            send = torch.randn((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
            ops = []
            recv_buf = None
            if rank > 0:
                recv_buf = torch.empty((SEQ, 1, HIDDEN), device="cuda", dtype=DTYPE)
                ops.append(dist.P2POp(dist.irecv, recv_buf, rank - 1))
            ops.append(dist.P2POp(dist.isend, send, rank + 1))
            t0 = time.perf_counter()
            reqs = dist.batch_isend_irecv(ops)
            for r in reqs:
                r.wait()
            torch.cuda.synchronize()
            t_op = time.perf_counter() - t0
            elapsed = time.perf_counter() - t_start
            print(f"[r{rank}] tF.{k} send/recv done in {t_op:.2f}s @ {elapsed:.1f}s", flush=True)
            if elapsed > TEST_TIMEOUT_S:
                sys.exit(99)

        # In a real fused schedule rank 0/1/2 also do iF here. We simulate
        # that by checking whether the upstream cascade was UNBLOCKED, i.e.
        # whether tF.0 send took ~0s instead of IF_DURATION_S.

    dist.barrier()
    print(f"[r{rank}] all done @ {time.perf_counter()-t_start:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
