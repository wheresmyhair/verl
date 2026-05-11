"""M14b — run REF (tex) schedule via FusedPPExecutor.
Validates the executor, not the schedule generator.
"""
from __future__ import annotations
import os, sys, time
import torch
import torch.distributed as dist


REF = [
    ["tF.0", "tF.1", "tF.2", "iF.0", "tF.3", "iF.1", "iF.2", "iF.3", "tB.0", "tB.1", "tB.2", "tB.3"],
    ["tF.0", "iF.0", "tF.1", "iF.1", "tF.2", "iF.2", "tF.3", "iF.3", "tB.0", "tB.1", "tB.2", "tB.3"],
    ["iF.0", "tF.0", "iF.1", "tF.1", "iF.2", "tF.2", "iF.3", "tB.0", "tB.1", "tF.3", "tB.2", "tB.3"],
    ["iF.0", "iF.1", "iF.2", "iF.3", "tF.0", "tB.0", "tF.1", "tB.1", "tF.2", "tB.2", "tF.3", "tB.3"],
]


def main():
    rank = int(os.environ["RANK"])
    P = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", device_id=torch.device(f"cuda:{rank}"))

    sys.path.insert(0, "/home/user/rlpipe/verl")
    from verl.utils.megatron.fused_pp_executor import FusedPPExecutor, FusedOpCtx

    HIDDEN_SHAPE = (2, 16, 32)
    executor = FusedPPExecutor(
        schedule=REF[rank], rank=rank, pp_group=dist.group.WORLD,
        pp_world_ranks=list(range(P)),
        hidden_shape=HIDDEN_SHAPE, dtype=torch.bfloat16, device=torch.device("cuda"),
    )

    def fake_compute(ctx: FusedOpCtx):
        sig = float(ctx.rank * 1000 + ctx.mb * 100 + {"tF": 1, "tB": 2, "iF": 3}[ctx.kind])
        out = torch.full(HIDDEN_SHAPE, sig, dtype=torch.bfloat16, device="cuda")
        return out

    print(f"[r{rank}] start", flush=True)
    t0 = time.time()
    outs = executor.run(fake_compute)
    print(f"[r{rank}] DONE in {(time.time()-t0)*1000:.1f}ms; harvested={len(outs)}", flush=True)

    dist.barrier()
    if rank == 0:
        print(f"\n=== M14b PASS: ref schedule via FusedPPExecutor ===", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
