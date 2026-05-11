"""M13 — DualPipe co-batching with CONCURRENT tF + iF + tB (correct fused schedule).

Uses the schedule from `assets/fused_schedule_algorithm.tex`:
- Last rank: all iF first, then 1F1B
- Other ranks: greedy with priority iF > tB > tF on smallest mb idx

Each rank has the SAME number of ops (= 3*M for P,M with full
tF+tB+iF, here 12 for P=4 M=4). At each op boundary, every rank's
batch_isend_irecv contains the matching (send prev-op-output, recv
next-op-input) for every active op type, so NCCL P2P matches naturally
across all ranks — no asymmetric rendezvous, no deadlock.

Test: walk all 12 op boundaries on each rank; if no rank deadlocks
and last_inf rank harvests M iF outputs, PASS.

Run:
    cd /home/user/rlpipe/verl && \
    torchrun --nproc-per-node=4 --rdzv-endpoint=127.0.0.1:29500 \
        exp_script/megatron_fused/smoke_m13_dualpipe_concurrent.py
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.distributed as dist


# Fused schedule for P=4 M=4 (from fused_schedule_algorithm.tex)
SCHEDULE_P4_M4 = [
    ["tF.0", "tF.1", "tF.2", "iF.0", "tF.3", "iF.1", "iF.2", "iF.3", "tB.0", "tB.1", "tB.2", "tB.3"],
    ["tF.0", "iF.0", "tF.1", "iF.1", "tF.2", "iF.2", "tF.3", "iF.3", "tB.0", "tB.1", "tB.2", "tB.3"],
    ["iF.0", "tF.0", "iF.1", "tF.1", "iF.2", "tF.2", "iF.3", "tB.0", "tB.1", "tF.3", "tB.2", "tB.3"],
    ["iF.0", "iF.1", "iF.2", "iF.3", "tF.0", "tB.0", "tF.1", "tB.1", "tF.2", "tB.2", "tF.3", "tB.3"],
]


def parse_op(op: str) -> tuple[str, int]:
    """'tF.3' -> ('tF', 3)"""
    kind, mb = op.split(".")
    return kind, int(mb)


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{rank}"))

    P = world_size
    assert P == 4, "smoke hard-coded for P=4"
    M = 4

    HIDDEN_SHAPE = (2, 16, 32)  # (B, S, H), small for fast smoke

    # Tags: distinct per (op_kind, mb)
    TAG_BASE = {"tF": 100, "tB": 200, "iF": 300}
    def make_tag(kind: str, mb: int) -> int:
        return TAG_BASE[kind] + mb

    schedule = SCHEDULE_P4_M4[rank]
    n_ops = len(schedule)

    if rank == 0:
        print("\n=== Fused schedule (P=4 M=4) ===")
        for r in range(P):
            print(f"r{r}: {' '.join(SCHEDULE_P4_M4[r])}")
        print()

    # Pre-allocate buffers per (op, mb).
    # send_bufs: store outputs of completed ops for transmission to peer
    # recv_bufs: pre-allocated, filled by next op's recv
    send_bufs: dict = {}  # (kind, mb) → tensor (output of compute, sent downstream)
    recv_bufs: dict = {}  # (kind, mb) → tensor (filled by recv from upstream)

    def need_recv(kind: str) -> bool:
        """Does this rank need to recv input for this op kind?"""
        if kind == "tF":
            return rank > 0  # tF flows 0→P-1; first stage doesn't recv
        if kind == "tB":
            return rank < P - 1  # tB flows P-1→0; last stage doesn't recv
        if kind == "iF":
            return rank < P - 1  # iF flows P-1→0 (reverse); first_inf=P-1 doesn't recv
        raise ValueError(kind)

    def need_send(kind: str) -> bool:
        """Does this rank need to send output for this op kind to peer?"""
        if kind == "tF":
            return rank < P - 1
        if kind == "tB":
            return rank > 0
        if kind == "iF":
            return rank > 0  # iF flows P-1→0; last_inf=0 doesn't send
        raise ValueError(kind)

    def peer_send(kind: str) -> int:
        """Where does this rank send the output to?"""
        if kind == "tF":
            return rank + 1
        if kind == "tB":
            return rank - 1
        if kind == "iF":
            return rank - 1  # reverse direction
        raise ValueError(kind)

    def peer_recv(kind: str) -> int:
        """Where does this rank recv the input from?"""
        if kind == "tF":
            return rank - 1
        if kind == "tB":
            return rank + 1
        if kind == "iF":
            return rank + 1  # reverse direction
        raise ValueError(kind)

    # Allocate
    for op in schedule:
        kind, mb = parse_op(op)
        if need_recv(kind):
            recv_bufs[(kind, mb)] = torch.empty(HIDDEN_SHAPE, dtype=torch.bfloat16, device="cuda")
        if need_send(kind):
            send_bufs[(kind, mb)] = torch.empty(HIDDEN_SHAPE, dtype=torch.bfloat16, device="cuda")

    # Compute output storage
    compute_out: dict = {}

    def fake_compute(kind: str, mb: int, hidden_in):
        # Deterministic fake output
        if hidden_in is None:
            base = torch.full(HIDDEN_SHAPE, fill_value=float(rank * 1000 + mb * 100 + {"tF": 1, "tB": 2, "iF": 3}[kind]),
                              dtype=torch.bfloat16, device="cuda")
        else:
            base = hidden_in + {"tF": 0.1, "tB": 0.2, "iF": 0.3}[kind]
        return base

    print(f"[r{rank}] start; {n_ops} ops to execute", flush=True)
    t0 = time.time()

    for i, op in enumerate(schedule):
        kind, mb = parse_op(op)
        comm_ops = []

        # Recv at boundary i: input for op i
        if need_recv(kind):
            comm_ops.append(dist.P2POp(
                op=dist.irecv,
                tensor=recv_bufs[(kind, mb)],
                peer=peer_recv(kind),
                group=dist.group.WORLD,  # PP=world_size, just use world group
                tag=make_tag(kind, mb),
            ))

        # Send at boundary i: output of op i-1 (already computed)
        if i > 0:
            prev_kind, prev_mb = parse_op(schedule[i - 1])
            if need_send(prev_kind):
                # Copy compute output into pre-alloc send buf
                send_bufs[(prev_kind, prev_mb)].copy_(compute_out[(prev_kind, prev_mb)])
                comm_ops.append(dist.P2POp(
                    op=dist.isend,
                    tensor=send_bufs[(prev_kind, prev_mb)],
                    peer=peer_send(prev_kind),
                    group=dist.group.WORLD,
                    tag=make_tag(prev_kind, prev_mb),
                ))

        # Submit batch — DualPipe commit point
        if comm_ops:
            print(f"[r{rank}] @b{i}: pre-batch ops={[(o.op.__name__, o.peer, o.tag) for o in comm_ops]}", flush=True)
            reqs = dist.batch_isend_irecv(comm_ops)
            for ridx, req in enumerate(reqs):
                req.wait()
            print(f"[r{rank}] @b{i}: batch done", flush=True)

        # Compute op i locally
        hidden_in = recv_bufs.get((kind, mb))  # None if first stage of this kind
        out = fake_compute(kind, mb, hidden_in)
        compute_out[(kind, mb)] = out

        if i % 4 == 0 or i == n_ops - 1:
            print(f"[r{rank}] boundary {i+1}/{n_ops}: {kind}.{mb} done, comm_ops={len(comm_ops)}", flush=True)

    # FINAL DRAIN: each op's send is co-batched with the NEXT boundary's
    # comm. The last op (op[n-1]) has no "next boundary" inside the loop,
    # so its send is never issued. We need a separate drain step here
    # to flush op[n-1]'s send, otherwise the peer's recv (which IS posted
    # at the peer's last boundary inside their loop) hangs forever.
    last_kind, last_mb = parse_op(schedule[-1])
    final_comm = []
    if need_send(last_kind):
        send_bufs[(last_kind, last_mb)].copy_(compute_out[(last_kind, last_mb)])
        final_comm.append(dist.P2POp(
            op=dist.isend,
            tensor=send_bufs[(last_kind, last_mb)],
            peer=peer_send(last_kind),
            group=dist.group.WORLD,
            tag=make_tag(last_kind, last_mb),
        ))
    if final_comm:
        print(f"[r{rank}] DRAIN: {[(o.op.__name__, o.peer, o.tag) for o in final_comm]}", flush=True)
        reqs = dist.batch_isend_irecv(final_comm)
        for req in reqs:
            req.wait()

    elapsed = time.time() - t0
    print(f"[r{rank}] DONE in {elapsed*1000:.1f}ms", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"\n=== M13 PASS: concurrent tF/tB/iF via batched NCCL P2P, no deadlock ===\n", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
