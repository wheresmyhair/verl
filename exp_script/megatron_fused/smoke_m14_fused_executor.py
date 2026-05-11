"""M14 — validate FusedPPExecutor on a generated schedule (5b-A) with
fake compute. Like smoke_m13 but driven by `build_fused_schedule(P, M)`
instead of a hardcoded schedule.

Validates that the SCHEDULE GENERATOR and the EXECUTOR work together
end-to-end without deadlock. Real model compute is wired in 5b-C.

Run:
    cd /home/user/rlpipe/verl && \
    torchrun --nproc-per-node=4 --rdzv-endpoint=127.0.0.1:29500 \
        exp_script/megatron_fused/smoke_m14_fused_executor.py
"""
from __future__ import annotations
import os
import sys
import time

import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{rank}"))

    sys.path.insert(0, "/home/user/rlpipe/verl")
    from verl.utils.megatron.fused_schedule import build_fused_schedule, verify_dependencies
    from verl.utils.megatron.fused_pp_executor import FusedPPExecutor, FusedOpCtx, _parse_op

    P, M = world_size, 4
    if rank == 0:
        sched_all = build_fused_schedule(P, M)
        verify_dependencies(sched_all)
        print("\n=== Generated schedule ===")
        for r in range(P):
            print(f"r{r}: {' '.join(sched_all[r])}")
        print()
    else:
        sched_all = None

    # Broadcast the schedule string list from rank 0 to all
    sched_obj = [sched_all]
    dist.broadcast_object_list(sched_obj, src=0)
    sched_all = sched_obj[0]
    my_sched = sched_all[rank]

    HIDDEN_SHAPE = (2, 16, 32)  # small for speed
    executor = FusedPPExecutor(
        schedule=my_sched,
        rank=rank,
        pp_group=dist.group.WORLD,
        pp_world_ranks=list(range(P)),
        hidden_shape=HIDDEN_SHAPE,
        dtype=torch.bfloat16,
        device=torch.device("cuda"),
    )

    # Fake compute: deterministic output based on rank+kind+mb
    def fake_compute(ctx: FusedOpCtx):
        # Combine recv inputs if present + a per-op signature
        sig = float(ctx.rank * 1000 + ctx.mb * 100 + {"tF": 1, "tB": 2, "iF": 3}[ctx.kind])
        base = torch.full(HIDDEN_SHAPE, sig, dtype=torch.bfloat16, device="cuda")
        if ctx.kind == "tF":
            if ctx.recv_hidden is not None:
                base = base + ctx.recv_hidden
            # For last_train: this is the "loss" path; return None (no send)
            if ctx.is_last_train:
                # In real model, would compute loss; in smoke just return base for harvest
                return base  # last_train doesn't need_send for tF → harvested
            return base
        elif ctx.kind == "tB":
            if ctx.recv_grad_output is not None:
                base = base + ctx.recv_grad_output
            if ctx.is_first_train:
                return base  # first_train doesn't need_send for tB → harvested (grad_input would terminate here)
            return base
        elif ctx.kind == "iF":
            if ctx.recv_hidden_inf is not None:
                base = base + ctx.recv_hidden_inf
            if ctx.is_last_inf:
                return base  # last_inf's iF output (= log_probs in real use) — harvested
            return base
        raise ValueError(ctx.kind)

    print(f"[r{rank}] start; {len(my_sched)} ops in schedule", flush=True)
    t0 = time.time()

    outputs = executor.run(fake_compute)

    elapsed = time.time() - t0
    print(f"[r{rank}] DONE in {elapsed*1000:.1f}ms; harvested {len(outputs)} outputs: {sorted(outputs.keys())}", flush=True)

    # Validate: each rank should harvest the ops that DON'T need_send
    # - last_train (rank P-1): tF.0..tF.M-1 (last stage tF doesn't send, it's the loss)
    # - first_train (rank 0): tB.0..tB.M-1 (first stage tB doesn't send, grad terminates)
    # - last_inf (rank 0): iF.0..iF.M-1
    expected_count = 0
    if rank == 0:  # first_train + last_inf
        expected_count = M + M  # tB + iF
    if rank == P - 1:  # last_train + first_inf
        expected_count = M  # tF only (iF is first_inf which DOES need_send)
    # Other ranks: 0 harvested

    assert len(outputs) == expected_count, (
        f"[r{rank}] expected {expected_count} harvested outputs, got {len(outputs)}: {sorted(outputs.keys())}"
    )

    dist.barrier()
    if rank == 0:
        print(f"\n=== M14 PASS: FusedPPExecutor runs generated schedule end-to-end ===\n", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
