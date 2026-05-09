"""M10 — verify reverse pp_group in-group rank is correctly inverted.

The fused-forward inference model needs a process group whose in-group
rank for each world rank is (P-1 - actor_pp_rank). This smoke checks
that DistributedWeightSyncCoordinator-style new_group(reversed) gives
this property.

Run:
  torchrun --nproc-per-node=4 \
    /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m10_reverse_pp_group.py
"""
from __future__ import annotations
import os, sys
import torch
import torch.distributed as dist


def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    # Train pp_group: ranks 0,1,2,3 in order (just like megatron's pp group at PP=4 DP=1)
    train_pp = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    # warmup (to actually init nccl comm)
    t = torch.zeros(1, device="cuda")
    dist.all_reduce(t, group=train_pp)

    train_pp_rank = dist.get_rank(group=train_pp)
    assert train_pp_rank == rank, f"expected train pp rank == world rank, got {train_pp_rank} != {rank}"

    # Build reverse pp_group via our helper
    sys.path.insert(0, "/home/user/rlpipe/verl")
    from verl.utils.megatron.reverse_pp_model import build_reverse_pp_group

    infer_pp = build_reverse_pp_group(train_pp)
    dist.all_reduce(t, group=infer_pp)

    infer_pp_rank = dist.get_rank(group=infer_pp)
    expected_infer_rank = (world_size - 1) - rank
    print(f"[r{rank}] train_pp_rank={train_pp_rank}, infer_pp_rank={infer_pp_rank}, expected={expected_infer_rank}", flush=True)
    assert infer_pp_rank == expected_infer_rank, (
        f"r{rank}: expected reverse pp rank {expected_infer_rank}, got {infer_pp_rank}"
    )

    # Also check that pre/post determination works
    pp_size = dist.get_world_size(group=infer_pp)
    pre_process = (infer_pp_rank == 0)            # = (world rank == P-1)
    post_process = (infer_pp_rank == pp_size - 1)  # = (world rank == 0)
    expected_pre = (rank == world_size - 1)
    expected_post = (rank == 0)
    assert pre_process == expected_pre, f"r{rank}: pre={pre_process}, expected={expected_pre}"
    assert post_process == expected_post, f"r{rank}: post={post_process}, expected={expected_post}"
    print(f"[r{rank}] ✅ pre_process={pre_process} (=embedding here for rev_pp rank 0), post_process={post_process} (=lm_head here for rev_pp rank P-1)", flush=True)

    dist.barrier()
    if rank == 0:
        print("\n=== M10 PASS: reverse pp_group has correctly inverted in-group ranks ===", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
