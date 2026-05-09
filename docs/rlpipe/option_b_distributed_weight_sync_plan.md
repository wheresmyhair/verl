# Option B: NCCL-distributed weight sync for verl-megatron-sglang

**目的**: 在容器无 `CAP_SYS_PTRACE` 的环境下，绕开 verl megatron rollout 的 CUDA IPC 路径，改用 NCCL collective broadcast。

**Status**: 已设计；实施中

**Why not other paths**:
- Docker fix (Option A) 是更干净的解，但不在我们控制内（云服务商重启容器加 `--cap-add=SYS_PTRACE --security-opt seccomp=unconfined`）
- Server mode (Option C) 实测仍用 CUDA IPC（HTTP 只发 meta，真 weights 还走 IPC handle），不解决问题
- Drop megatron (Option D) 放弃 framework-independence claim，paper 缩水

---

## 现有失败路径

```
verl megatron worker (rollout_mode)
  ├─ per_tensor_generator → (name, full_tensor) on every PP rank
  └─ self.rollout.update_weights(per_tensor_param)
        └─ sgl_update_weights(engine, params_batch, device_mesh_key="infer_tp")
              └─ MultiprocessingSerializer.serialize(tensor)        ← CUDA IPC handle
              └─ dist.gather_object → tp rank 0 collects
              └─ engine.update_weights_from_tensor(serialized_tensors)
                    └─ tp_worker.update_weights_from_tensor
                          └─ MultiprocessingSerializer.deserialize  ← ❌ pidfd_getfd 拒绝
```

## 目标路径 (B)

```
verl megatron worker (rollout_mode)
  ├─ init: setup NCCL update group spanning {actor rank 0, sglang TP ranks}
  ├─ per_tensor_generator → (name, full_tensor) on every PP rank
  └─ for each (name, tensor):
        if actor PP rank == 0:
            dist.broadcast(tensor, src=0, group=update_group)        ← NCCL 不走 IPC
        ↑↓ NCCL 匹配 ↑↓
        sglang.update_weights_from_distributed(name, dtype, shape)
            └─ tp_worker.update_weights_from_distributed
                  └─ torch.empty(shape) + dist.broadcast(empty, src=0, group=...)
                  └─ self.model.load_weights([(name, tensor)])
```

NCCL P2P 走 NVLink/PCIe，不需要 `pidfd_getfd`，docker capability 不影响。

## 关键约束

1. **Sglang `update_weights_from_distributed` 写死 `src=0`**
   (`sglang/srt/model_executor/model_runner.py: dist.broadcast(weight, src=0, group=...)`)
   ⇒ 必须由 group 内 rank 0 发送
   ⇒ 我们用 actor PP rank 0 当 group rank 0

2. **Group composition**:
   - `update_group` = { actor PP rank 0 } ∪ { sglang TP ranks }
   - For PP=2 TP=1 hybrid: 2 members (1 actor + 1 sglang TP=0)
   - For PP=4 TP=4: 5 members (1 actor + 4 sglang)

3. **Actor PP rank > 0** 不直接参与 update_group。`per_tensor_generator` 已经做了 PP gather + broadcast，所以 rank 0 拿到的是 full tensor。

4. **Hybrid mode 下** actor 进程和 sglang scheduler 子进程在不同 PID。`init_weights_update_group` 用 TCPStore master_address/port 做 rendezvous。

## 实施计划

### 文件清单

1. **新建** `verl/utils/megatron/distributed_weight_sync.py` (~150 lines)
   - `DistributedWeightSyncCoordinator`：封装 init + per-tensor broadcast 状态
   - `init_distributed_update_group(actor_rank0_global, sglang_global_ranks, master_port)`
   - `broadcast_tensor_to_sglang(tensor, group)` (actor side)
   - `prepare_update_meta(per_tensor_generator)` → list of (name, dtype, shape) for sglang side

2. **改** `verl/workers/megatron_workers.py`
   - `init_model`：collective 创建 update_group（在 actor + rollout 都 init 完毕之后）
   - `rollout_mode`：if `weight_sync_mode == "distributed"`，走新路径
   - 加 `weight_sync_mode` 字段到 worker config

3. **改** `verl/workers/rollout/sglang_rollout/sglang_rollout.py`
   - 加 method `update_weights_distributed(meta_list)`
   - 调用 `engine.update_weights_from_distributed(names, dtypes, shapes, group_name)`

4. **加 config flag** `verl/workers/config/rollout.py`
   - `weight_sync_mode: Literal["tensor", "distributed"] = "tensor"`

### 代码骨架

```python
# verl/utils/megatron/distributed_weight_sync.py
class DistributedWeightSyncCoordinator:
    def __init__(self, actor_pp_size, sglang_tp_size,
                 master_addr="127.0.0.1", master_port=29600):
        self.world_size = 1 + sglang_tp_size  # actor rank 0 + sglang ranks
        self.master_addr = master_addr
        self.master_port = master_port
        self.group_name = "rlpipe_weight_update"
        self._update_group = None

    def init_actor_side(self, actor_pp_rank):
        """Called on each actor PP rank.
        Only PP rank 0 actually participates in the update group as rank 0.
        Other PP ranks return None (no group ref needed)."""
        if actor_pp_rank != 0:
            return None
        # Use init_process_group on a NEW process group (TCPStore-based)
        # spanning this rank + sglang TP ranks.
        # Actually we use dist.new_group(...) on existing world if possible,
        # or torch.distributed.init_device_mesh. Need to research best API.
        ...
        self._update_group = ...
        return self._update_group

    def broadcast(self, tensor: torch.Tensor):
        """Actor PP rank 0 broadcasts a tensor to sglang."""
        assert self._update_group is not None, "not initialized"
        dist.broadcast(tensor, src=0, group=self._update_group, async_op=False)

    def init_sglang_side(self, engine, sglang_local_rank):
        """Called by SGLangRollout. Forwards to engine.init_weights_update_group."""
        engine.init_weights_update_group(
            master_address=self.master_addr,
            master_port=self.master_port,
            rank_offset=1,                            # actor rank 0 takes index 0
            world_size=self.world_size,
            group_name=self.group_name,
            backend="nccl",
        )
```

### 调用 flow

```python
# === verl/workers/megatron_workers.py (modified) ===
class MegatronActorRolloutRefWorker:
    def init_model(self):
        # ... existing actor init ...
        # ... existing rollout init ...
        if self.config.rollout.weight_sync_mode == "distributed":
            from verl.utils.megatron.distributed_weight_sync import DistributedWeightSyncCoordinator
            self._weight_sync = DistributedWeightSyncCoordinator(
                actor_pp_size=self.actor_pp_size,
                sglang_tp_size=self.config.rollout.tensor_model_parallel_size,
            )
            # Init group on both sides (collective)
            actor_group = self._weight_sync.init_actor_side(self.actor_pp_rank)
            self._weight_sync.init_sglang_side(self.rollout._engine, self.sglang_local_rank)

    async def rollout_mode(self):
        # ... existing loads + per_tensor_param generation ...
        if self.config.rollout.weight_sync_mode == "distributed":
            # New path: NCCL broadcast
            tensor_list = list(per_tensor_param)
            meta = [(name, t.dtype, list(t.shape)) for name, t in tensor_list]
            # Sglang side: kick off async receive (it'll do dist.broadcast(empty, src=0))
            await self.rollout.update_weights_distributed(meta)
            # Actor rank 0: broadcast each tensor (matches sglang's broadcasts)
            if self.actor_pp_rank == 0:
                for (name, tensor), (_, dtype, shape) in zip(tensor_list, meta):
                    self._weight_sync.broadcast(tensor)
        else:
            # Old IPC path
            await self.rollout.update_weights(per_tensor_param)
```

## 测试 / 验证策略

1. **Unit-level smoke** (`exp_script/megatron_fused/smoke_m9_distributed_sync.py`)
   - 不启动完整 verl，只测：actor proc + sglang Engine 在不同 PID
   - 跑 init_weights_update_group + 一次 update_weights_from_distributed
   - 验证 weights 实际传到 sglang（compare hash on sglang side）

2. **Integration smoke** (mod of `idea2_megatron_baseline.sh`)
   - 加 `actor_rollout_ref.rollout.weight_sync_mode=distributed`
   - 期望：step:1 emit + 无 pidfd_getfd 错误
   - 比对：log_probs 与 docker-fixed 环境（如果能拿到）一致

3. **Performance comparison**
   - Distributed vs IPC weight transfer wall time（在 docker-fixed 环境对比）
   - 期望：distributed ≤ 2x IPC 时长（NCCL collective vs CUDA IPC zero-copy）

## 已知风险

1. **Process group setup race**: actor + sglang 异步 init 时 master TCPStore 必须先就位
2. **NCCL group versioning**: sglang fork 可能假定特定 torch 版本的 dist API
3. **Hybrid mode lifecycle**: actor 卸载/加载 GPU 时 NCCL group 状态保持？需确认
4. **Multi-node**: 当前设计假设单节点；多节点 master_addr 需要升级到协调 IP

## 估时

- 文件修改: 1 day
- Smoke 调试: 0.5 day
- 集成测试 + 与 IPC 对比: 0.5 day

总计 **2 day** 完成 minimum viable B。

## Done criteria

- [ ] `idea2_megatron_baseline.sh` with `weight_sync_mode=distributed` 跑通 1 step，无 pidfd 错误
- [ ] log_probs 与 stock IPC path 数值等价（< 1e-5 max diff）
- [ ] B 路径 wall time 与 IPC path 同数量级（report ratio）
