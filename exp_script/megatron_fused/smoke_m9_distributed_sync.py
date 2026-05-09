"""M9 — minimal smoke for DistributedWeightSyncCoordinator.

Goal: validate that NCCL collective broadcast across two SEPARATE
processes (different PIDs, different default-group dist init) works
without CUDA IPC / pidfd_getfd. This is the core mechanism behind
Option B's replacement for verl's IPC weight transfer.

Setup
-----
We spawn 2 child processes via Python multiprocessing:
  - "actor" process: picks rank 0 in the update group; allocates a
    GPU tensor with known content; broadcasts via the update group.
  - "sglang" process: picks rank 1; allocates an empty buffer;
    receives via dist.broadcast(buf, src=0, group=...).

Both processes call init_custom_process_group with the SAME
master_addr:master_port (TCPStore rendezvous), separate ranks,
world_size=2.

If the recv'd tensor on sglang side matches the actor's tensor by
checksum, we've validated the NCCL P2P path works in our docker
without CUDA IPC.

Run
---
  python3 /home/user/rlpipe/verl/exp_script/megatron_fused/smoke_m9_distributed_sync.py

This is NOT a torchrun script — we deliberately spawn two
independent processes (mimicking actor + sglang scheduler subprocess
running in the same node but different PIDs).
"""
from __future__ import annotations
import multiprocessing as mp
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, "/home/user/rlpipe/verl")
sys.path.insert(0, "/home/user/rlpipe/sglang-fork/python")


MASTER_ADDR = "127.0.0.1"
MASTER_PORT = 29701  # picked to not collide with default 29600
GROUP_NAME = "rlpipe_smoke_m9"
WORLD_SIZE = 2
SHAPE = (256, 256)
DTYPE = torch.bfloat16


def actor_process(role: str, gpu_id: int, result_queue):
    """Plays the actor role: rank 0 in the update group."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)  # local index after CUDA_VISIBLE_DEVICES

        from verl.utils.megatron.distributed_weight_sync import (
            DistributedWeightSyncCoordinator,
        )

        coord = DistributedWeightSyncCoordinator(
            sglang_tp_size=1,
            master_addr=MASTER_ADDR, master_port=MASTER_PORT,
            group_name=GROUP_NAME,
        )
        print(f"[{role}] init starting…", flush=True)
        t0 = time.perf_counter()
        coord.init_actor_side()
        print(f"[{role}] init done in {time.perf_counter()-t0:.1f}s", flush=True)

        # Build a known tensor (so receiver can verify content)
        torch.manual_seed(0xa07)
        tensor = torch.randn(SHAPE, dtype=DTYPE, device="cuda")
        checksum = tensor.float().sum().item()
        print(f"[{role}] tensor shape={tuple(tensor.shape)} checksum={checksum:.6f}", flush=True)

        # Broadcast to receiver (rank 1)
        t0 = time.perf_counter()
        coord.broadcast(tensor)
        torch.cuda.synchronize()
        bcast_wall = time.perf_counter() - t0
        print(f"[{role}] broadcast done in {bcast_wall*1000:.1f}ms", flush=True)

        result_queue.put((role, "OK", checksum))
        coord.destroy()
    except Exception as e:
        import traceback
        result_queue.put((role, "FAIL", f"{e}\n{traceback.format_exc()}"))


def sglang_process(role: str, gpu_id: int, result_queue):
    """Plays the sglang side: rank 1 in the update group."""
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)

        from sglang.srt.utils.common import init_custom_process_group

        # Mimics what sglang.model_runner.init_weights_update_group does
        rank = 1  # rank_offset=1 + sglang_tp_local_rank=0
        print(f"[{role}] init starting…", flush=True)
        t0 = time.perf_counter()
        pg = init_custom_process_group(
            backend="nccl",
            init_method=f"tcp://{MASTER_ADDR}:{MASTER_PORT}",
            world_size=WORLD_SIZE,
            rank=rank,
            group_name=GROUP_NAME,
        )
        print(f"[{role}] init done in {time.perf_counter()-t0:.1f}s", flush=True)

        # Mimics what sglang.update_weights_from_distributed does
        recv = torch.empty(SHAPE, dtype=DTYPE, device="cuda")
        t0 = time.perf_counter()
        dist.broadcast(recv, src=0, group=pg, async_op=False)
        torch.cuda.synchronize()
        bcast_wall = time.perf_counter() - t0
        recv_checksum = recv.float().sum().item()
        print(f"[{role}] recv done in {bcast_wall*1000:.1f}ms checksum={recv_checksum:.6f}", flush=True)

        result_queue.put((role, "OK", recv_checksum))
        try:
            dist.destroy_process_group(pg)
        except Exception:
            pass
    except Exception as e:
        import traceback
        result_queue.put((role, "FAIL", f"{e}\n{traceback.format_exc()}"))


def main():
    mp.set_start_method("spawn", force=True)
    queue = mp.Queue()

    actor_p = mp.Process(target=actor_process, args=("actor", 0, queue))
    sgl_p = mp.Process(target=sglang_process, args=("sglang", 1, queue))

    actor_p.start()
    sgl_p.start()
    actor_p.join(timeout=60)
    sgl_p.join(timeout=60)

    results = []
    while not queue.empty():
        results.append(queue.get_nowait())

    print("\n=== M9 RESULTS ===")
    role_results = {r[0]: (r[1], r[2]) for r in results}
    actor = role_results.get("actor", ("MISSING", None))
    sglang = role_results.get("sglang", ("MISSING", None))
    print(f"  actor : {actor[0]}  checksum={actor[1]}")
    print(f"  sglang: {sglang[0]}  checksum={sglang[1]}")
    if actor[0] == "OK" and sglang[0] == "OK":
        if abs(actor[1] - sglang[1]) < 1e-2:
            print("  ✅ checksums match — NCCL path works without CUDA IPC")
            return 0
        else:
            print(f"  ❌ checksums DIFFER (delta={actor[1]-sglang[1]})")
            return 1
    else:
        if actor[0] == "FAIL":
            print(f"  actor error: {actor[1]}")
        if sglang[0] == "FAIL":
            print(f"  sglang error: {sglang[1]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
