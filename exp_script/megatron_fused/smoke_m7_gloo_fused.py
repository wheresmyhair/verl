"""M7 — gloo fused-schedule smoke.

Same setup as M1/M3 (4-rank PP, rank 3 in iF burst) but P2P uses gloo
pair groups instead of NCCL. Expected: rank 3 sends iF activations
asynchronously, no stall on r0/r1/r2 tF sends.

Pass criterion:
  - Total wall ≈ max(iF burst, tF/tB chain) — overlap achieved
  - r0/r1/r2 first-tF latency < 1s (vs 8.11s with NCCL stall)

This validates that gloo's truly-async CPU-staged isend lets the fused
schedule complete without the NCCL rendezvous deadlock.
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.distributed as dist


PP = 4
M = 8
SEQ = 4096
HIDDEN = 4096
DTYPE = torch.bfloat16
IF_DURATION_S = 4.0
COMPUTE_PER_OP_S = 0.05
TEST_TIMEOUT_S = 60.0


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == PP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    print(f"[r{rank}] init", flush=True)

    # Create gloo pair groups (must call collectively)
    sys.path.insert(0, "/home/user/rlpipe/verl")
    from verl.utils.megatron.fused_p2p import (
        create_megatron_fused_pair_groups, gloo_p2p_send, gloo_p2p_recv,
    )
    pair_groups = create_megatron_fused_pair_groups(pp_size=PP, pp_rank=rank)
    print(f"[r{rank}] pair_groups keys: {sorted([str(k) for k in pair_groups.keys()])}", flush=True)

    dist.barrier()
    t_start = time.perf_counter()

    shape = (SEQ, 1, HIDDEN)
    is_first = (rank == 0)
    is_last = (rank == PP - 1)

    # Simple fused-pattern smoke:
    # - r3 does iF burst (local "compute"; we simulate with sleep)
    # - r0/r1/r2 do M tF sends in chain via gloo
    # - r3 doesn't recv during iF; sends should NOT stall (gloo async)
    pending_handles = []
    _send_bufs = []  # keep CPU staging refs alive

    if rank == PP - 1:
        # r3: simulate iF burst, then recv M tFs from r2
        print(f"[r{rank}] starting iF burst {IF_DURATION_S}s (no recv'ing)", flush=True)
        t_iF = time.perf_counter()
        while time.perf_counter() - t_iF < IF_DURATION_S:
            # fake compute
            t = torch.randn((4096, 4096), device="cuda", dtype=DTYPE)
            _ = (t @ t).sum()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
        print(f"[r{rank}] iF burst done @ {elapsed:.1f}s", flush=True)
        # Now drain pending tFs from r2
        for k in range(M):
            buf = gloo_p2p_recv(shape, src_pp_rank=rank - 1, dtype=DTYPE,
                                pair_groups=pair_groups, device=torch.device("cuda"),
                                dst_pp_rank=rank)
            print(f"[r{rank}] recv tF.{k} @ {time.perf_counter()-t_start:.1f}s", flush=True)
    else:
        for k in range(M):
            # If not first, recv from prev first
            if not is_first:
                buf = gloo_p2p_recv(shape, src_pp_rank=rank - 1, dtype=DTYPE,
                                    pair_groups=pair_groups, device=torch.device("cuda"),
                                    dst_pp_rank=rank)
            # fake compute (small)
            t_op = torch.randn(shape, device="cuda", dtype=DTYPE) + 0.001
            time.sleep(COMPUTE_PER_OP_S)
            # Send to next via gloo (async)
            t0 = time.perf_counter()
            handle = gloo_p2p_send(t_op, dst_pp_rank=rank + 1, pair_groups=pair_groups,
                                    src_pp_rank=rank)
            send_wall = time.perf_counter() - t0
            pending_handles.append(handle)
            elapsed = time.perf_counter() - t_start
            print(f"[r{rank}] tF.{k} sent (issue {send_wall*1000:.1f}ms) @ {elapsed:.1f}s", flush=True)
            if elapsed > TEST_TIMEOUT_S:
                print(f"[r{rank}] !!! TIMEOUT @ {elapsed:.1f}s", flush=True)
                sys.exit(99)

    # Drain pending sends
    for h in pending_handles:
        h.wait()

    dist.barrier()
    elapsed = time.perf_counter() - t_start
    print(f"[r{rank}] all done @ {elapsed:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
