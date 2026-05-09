"""M3 — two-PP fused-forward smoke with pre-post on both directions.

Setup
-----
PP=4. Two channels:
  - train PP: forward direction r0→r1→r2→r3 (and backward r3→r0)
  - infer PP: REVERSE direction r3→r2→r1→r0 (PP-sharded inference,
    rank P-1 is first inference stage)

Without pre-post the two NCCL channels form a circular wait (memory file
`project_megatron_two_pp_deadlock_tbd.md`):
  r3.send_inf(→r2)    waits for r2.recv_inf
  r2.recv_inf         needs r2.tF.0 done
  r2.tF.0 done        needs r2.send_train(→r3)
  r2.send_train(→r3)  waits for r3.recv_train
  r3.recv_train       needs r3 finishes iFs
  r3 iF chain         needs r3.send_inf  ← back to start

Pre-post fix
------------
Each rank pre-posts ALL expected recvs on BOTH PPs before the busy
phase. Both NCCL channels have receiver buffers ready, so no send
blocks on the rendezvous; the circular wait dissolves.

Per-rank pre-post (PP=4):
  rank 0: train_recv_bwd (from r1) × M, infer_recv_fwd (from r1) × M
  rank 1: train_recv_fwd (from r0) × M, train_recv_bwd (from r2) × M
          infer_recv_fwd (from r2) × M, infer_recv_bwd (to r0) × 0
  rank 2: train_recv_fwd × M, train_recv_bwd × M, infer_recv_fwd × M, infer_recv_bwd × M-skipped
  rank 3: train_recv_fwd (from r2) × M, infer_recv_bwd (from r2) × M

(Backward direction for inf only present if we propagate iF gradients.
RL inference forward usually has no_grad, so inf has no backward. We
omit infer_recv_bwd for simplicity.)

Run:
  torchrun --nproc-per-node=4 \
    /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m3_dual_pp.py
"""
from __future__ import annotations
import os
import sys
import time
from collections import deque

import torch
import torch.distributed as dist


PP = 4
M = 8
SEQ = 4096
HIDDEN = 4096
DTYPE = torch.bfloat16
COMPUTE_PER_OP_S = 0.05    # tF/tB/iF compute time
TEST_TIMEOUT_S = 60.0


def pre_post_irecvs(num: int, src_rank: int, group, shape, dtype):
    """Issue `num` non-blocking irecvs from `src_rank` on `group`. Returns
    deque of (req, buffer).

    NOTE: must use default world group (group=None). NCCL on sub-group
    (dist.new_group) blocks batch_isend_irecv until matching peer call.
    """
    pending = deque()
    for _ in range(num):
        buf = torch.empty(shape, device="cuda", dtype=dtype)
        op = dist.P2POp(dist.irecv, buf, src_rank)  # no group → default world
        reqs = dist.batch_isend_irecv([op])
        pending.append((reqs[0], buf))
    return pending


def consume_recv(pending: deque):
    if not pending:
        raise RuntimeError("pending queue empty — schedule mismatch")
    req, buf = pending.popleft()
    req.wait()
    return buf


def fake_compute(t: torch.Tensor, secs: float) -> torch.Tensor:
    """Fake compute returning a derived tensor of the same shape, taking ≈secs."""
    end = time.perf_counter() + secs
    out = t
    while time.perf_counter() < end:
        out = out + 0.001
    return out.contiguous()


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == PP
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)

    # Use DEFAULT world group for both train and infer P2P. NCCL on
    # new_group blocks batch_isend_irecv until the matching peer also
    # calls (sub-group rendezvous semantics), making asymmetric pre-post
    # impossible. Default world group permits asymmetric calls. Train vs
    # infer ops are distinguished by their (src,dst) pairs and ordering —
    # not by group.
    train_pg = None
    infer_pg = None

    print(f"[r{rank}] using default world group", flush=True)
    dummy = torch.zeros(1, device="cuda")
    dist.all_reduce(dummy)
    print(f"[r{rank}] default warmed", flush=True)
    dist.barrier()
    t_start = time.perf_counter()

    shape = (SEQ, 1, HIDDEN)

    # ===== Step 1: pre-post all expected recvs =====
    pending_train_fwd = deque()  # for tF: recv from rank-1
    pending_train_bwd = deque()  # for tB: recv from rank+1
    pending_infer_fwd = deque()  # for iF: recv from rank+1 (reverse)

    print(f"[r{rank}] pre-post tF starting", flush=True)
    if rank > 0:
        pending_train_fwd = pre_post_irecvs(M, rank - 1, train_pg, shape, DTYPE)
    print(f"[r{rank}] pre-post tF done ({len(pending_train_fwd)})", flush=True)
    if rank < PP - 1:
        pending_train_bwd = pre_post_irecvs(M, rank + 1, train_pg, shape, DTYPE)
    print(f"[r{rank}] pre-post tB done ({len(pending_train_bwd)})", flush=True)
    if rank < PP - 1:
        pending_infer_fwd = pre_post_irecvs(M, rank + 1, infer_pg, shape, DTYPE)
    print(f"[r{rank}] pre-posted: tF={len(pending_train_fwd)} tB={len(pending_train_bwd)} iF={len(pending_infer_fwd)}", flush=True)
    dist.barrier()

    # ===== Step 2: build per-rank schedule =====
    # Use a simplified fused schedule: rank r does (PP-1-r)*K iFs first,
    # then alternates tF/tB. This mimics filling the warmup bubble.
    # For rank 3: all M iFs first, then tF/tB (the deepest bubble).
    # For rank 0: minimal iFs (no warmup bubble).

    # Schedule each rank — each entry is (op_type, mb_id):
    schedule = []
    # iF burst proportional to warmup depth
    n_iF_warmup = M  # rank 3 does all; smaller ranks do less
    if rank == PP - 1:
        for k in range(M):
            schedule.append(("iF", k))
    elif rank == PP - 2:
        for k in range(M // 2):
            schedule.append(("iF", k))
    # All ranks do all tF then tB
    for k in range(M):
        schedule.append(("tF", k))
    for k in range(M):
        schedule.append(("tB", k))
    # Remaining iFs (for non-rank-3)
    if rank == PP - 2:
        for k in range(M // 2, M):
            schedule.append(("iF", k))

    # ===== Step 3: execute schedule =====
    for op_idx, (op, mb) in enumerate(schedule):
        elapsed = time.perf_counter() - t_start
        if elapsed > TEST_TIMEOUT_S:
            print(f"[r{rank}] !!! TIMEOUT @ {elapsed:.1f}s on op={op}.{mb}", flush=True)
            sys.exit(99)
        if op == "tF":
            # recv from prev (if not first), compute, send to next (if not last)
            if rank > 0:
                inp = consume_recv(pending_train_fwd)
            else:
                inp = torch.randn(shape, device="cuda", dtype=DTYPE)
            out = fake_compute(inp, COMPUTE_PER_OP_S)
            if rank < PP - 1:
                send_op = dist.P2POp(dist.isend, out, rank + 1)
                reqs = dist.batch_isend_irecv([send_op])
                for r in reqs: r.wait()
        elif op == "tB":
            # recv from next (if not last), compute, send to prev (if not first)
            if rank < PP - 1:
                grad = consume_recv(pending_train_bwd)
            else:
                grad = torch.randn(shape, device="cuda", dtype=DTYPE)
            grad_in = fake_compute(grad, COMPUTE_PER_OP_S)
            if rank > 0:
                send_op = dist.P2POp(dist.isend, grad_in, rank - 1)
                reqs = dist.batch_isend_irecv([send_op])
                for r in reqs: r.wait()
        elif op == "iF":
            # Inference (reverse PP): recv from next (rank+1), compute, send to prev (rank-1)
            if rank < PP - 1:
                inp = consume_recv(pending_infer_fwd)
            else:
                inp = torch.randn(shape, device="cuda", dtype=DTYPE)
            out = fake_compute(inp, COMPUTE_PER_OP_S)
            if rank > 0:
                send_op = dist.P2POp(dist.isend, out, rank - 1)
                reqs = dist.batch_isend_irecv([send_op])
                for r in reqs: r.wait()
        if op_idx % 5 == 0:
            print(f"[r{rank}] op {op_idx}/{len(schedule)}: {op}.{mb} done @ {time.perf_counter()-t_start:.2f}s", flush=True)

    dist.barrier()
    elapsed = time.perf_counter() - t_start
    print(f"[r{rank}] all done @ {elapsed:.1f}s", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
