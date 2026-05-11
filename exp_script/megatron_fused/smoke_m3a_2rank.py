"""Minimal 2-rank pre-post test: r1 pre-posts irecvs from r0; r0 sends later."""
from __future__ import annotations
import os, sys, time
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[r{rank}] init", flush=True)

    # Use DEFAULT world group (no new_group), matches M2 pattern
    pg = None
    # warmup default group
    dummy = torch.zeros(1, device="cuda")
    dist.all_reduce(dummy)
    print(f"[r{rank}] default pg warmed", flush=True)

    M = 4
    shape = (1024, 1024)

    if rank == 1:
        print(f"[r{rank}] pre-post {M} irecvs from r0", flush=True)
        bufs = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(M)]
        reqs = []
        for i in range(M):
            op = dist.P2POp(dist.irecv, bufs[i], 0)  # no group → default
            r = dist.batch_isend_irecv([op])
            reqs.append(r[0])
            print(f"[r{rank}]   irecv {i} posted @ {time.perf_counter():.3f}", flush=True)
        print(f"[r{rank}] all {M} pre-posted, sleep 2s", flush=True)
        time.sleep(2.0)
        for i in range(M):
            reqs[i].wait()
            print(f"[r{rank}]   recv {i} arrived", flush=True)
    else:
        print(f"[r{rank}] sleep 1s before sending", flush=True)
        time.sleep(1.0)
        for i in range(M):
            t = torch.full(shape, float(i), device="cuda", dtype=torch.bfloat16)
            op = dist.P2POp(dist.isend, t, 1)
            reqs = dist.batch_isend_irecv([op])
            for r in reqs: r.wait()
            print(f"[r{rank}]   send {i} done @ {time.perf_counter():.3f}", flush=True)

    dist.barrier()
    print(f"[r{rank}] all done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
