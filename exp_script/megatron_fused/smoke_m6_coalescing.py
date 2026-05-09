"""M6 — Option C: NCCL groupStart/End via pytorch `_coalescing_manager`.

Hypothesis: ncclGroupStart + multiple ncclRecv + ncclGroupEnd schedules
recvs without per-op rendezvous. ncclGroupEnd returns when ops are
ENQUEUED (not completed). Waits happen via Work.wait() later.

If true, we can pre-post M irecvs on a rank without any matching sender,
get back work handles immediately, and consume them later as the
schedule progresses (when senders eventually issue matching ncclSends).

Test: 2-rank then 4-rank.
  - rank N pre-posts M irecvs from rank-1 inside coalescing block
  - measure pre-post wall (should be ~ms)
  - sender starts AFTER (delayed)
  - receiver waits on work handles → data arrives

Pass: pre-post returns in << than (sender delay).
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _coalescing_manager


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    print(f"[r{rank}] init", flush=True)
    dummy = torch.zeros(1, device="cuda")
    dist.all_reduce(dummy)
    dist.barrier()

    M = 8
    shape = (1024, 1024)
    SENDER_DELAY_S = 2.0

    if rank == world_size - 1:
        # Receiver: pre-post M irecvs from rank-1
        bufs = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(M)]
        t0 = time.perf_counter()
        with _coalescing_manager(group=None, device=torch.device("cuda"), async_ops=True) as cm:
            for i in range(M):
                dist.irecv(bufs[i], src=rank - 1)
        elapsed_pp = time.perf_counter() - t0
        print(f"[r{rank}] pre-post {M} irecvs in coalescing block: {elapsed_pp*1000:.1f}ms", flush=True)
        # cm.works holds the Work handles
        works = cm.works
        print(f"[r{rank}] got {len(works)} work handles", flush=True)
        time.sleep(0.5)  # let sender start
        for i, w in enumerate(works):
            t = time.perf_counter()
            w.wait()
            print(f"[r{rank}]   recv {i} arrived (wait {(time.perf_counter()-t)*1000:.1f}ms)", flush=True)
    elif rank == 0:
        # Sender: wait then send
        print(f"[r{rank}] sleep {SENDER_DELAY_S}s", flush=True)
        time.sleep(SENDER_DELAY_S)
        with _coalescing_manager(group=None, device=torch.device("cuda"), async_ops=True) as cm:
            for i in range(M):
                t = torch.full(shape, float(i), device="cuda", dtype=torch.bfloat16)
                dist.isend(t, dst=world_size - 1)
        print(f"[r{rank}] {M} isends issued", flush=True)
        for w in cm.works:
            w.wait()
        print(f"[r{rank}] all sends done", flush=True)
    else:
        # Other ranks (in 4-rank world): no-op
        pass

    dist.barrier()
    print(f"[r{rank}] done", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
