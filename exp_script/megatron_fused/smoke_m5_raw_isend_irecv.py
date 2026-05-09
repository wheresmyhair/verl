"""M5 — test raw dist.isend / dist.irecv (NOT batch_isend_irecv) for true async.

Hypothesis: torch.distributed's raw `isend/irecv` (no batch wrapper) might
be truly non-blocking on NCCL, allowing pre-post of recv buffers without
matching peer participation.

Test: rank 1 pre-posts irecvs from rank 0 via raw dist.irecv, then
rank 0 sends after a delay. If pre-post returns immediately, raw irecv
is truly async.
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[r{rank}] init", flush=True)
    dummy = torch.zeros(1, device="cuda")
    dist.all_reduce(dummy)
    print(f"[r{rank}] warmed @ {time.perf_counter():.3f}", flush=True)

    M = 4
    shape = (1024, 1024)

    if rank == 1:
        t0 = time.perf_counter()
        bufs = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(M)]
        reqs = []
        for i in range(M):
            t = time.perf_counter()
            req = dist.irecv(bufs[i], src=0)  # raw irecv, no batch wrapper
            print(f"[r{rank}]   raw irecv {i} returned in {(time.perf_counter()-t)*1000:.1f}ms", flush=True)
            reqs.append(req)
        elapsed = time.perf_counter() - t0
        print(f"[r{rank}] all {M} pre-posted in {elapsed*1000:.1f}ms (sleep 2s before wait)", flush=True)
        time.sleep(2.0)
        for i in range(M):
            reqs[i].wait()
            print(f"[r{rank}]   recv {i} arrived", flush=True)
    elif rank == 0:
        print(f"[r{rank}] sleep 1s then send", flush=True)
        time.sleep(1.0)
        for i in range(M):
            t = torch.full(shape, float(i), device="cuda", dtype=torch.bfloat16)
            req = dist.isend(t, dst=1)  # raw isend
            req.wait()
            print(f"[r{rank}]   send {i} done @ {time.perf_counter():.3f}", flush=True)

    dist.barrier()
    print(f"[r{rank}] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
