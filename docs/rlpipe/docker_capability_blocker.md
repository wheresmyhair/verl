# Megatron + sglang 训练在容器内被 docker 权限阻塞 — 原因 + 解决方案

**作者**: rlpipe team
**日期**: 2026-05-09
**关键词**: docker, CAP_SYS_PTRACE, seccomp, pidfd_getfd, CUDA IPC, megatron, sglang

---

## 概要

我们的 RL 训练流水线由两条主路径组成：

1. **torch_pp + sglang**: 已经跑通，1 step at Qwen3-1.7B PP=4 在我们容器内 38s 完成
2. **megatron + sglang**: 论文最终路径，目前**完全跑不通**——容器内被 docker 权限阻塞

阻塞的具体原因和需要云服务商做的修改如下。

---

## 问题表象

```
RuntimeError: pidfd_getfd: Operation not permitted

  File ".../sglang/srt/utils/patch_torch.py", line 81, in _rebuild_cuda_tensor_modified
  File ".../torch/multiprocessing/reductions.py", line 181, in rebuild_cuda_tensor
  File ".../torch/storage.py", line 1457, in _new_shared_cuda
```

错误发生在：megatron actor 把训练好的 actor 权重交给 sglang 推理引擎做 rollout 时。具体调用链：

```
megatron actor (PP=2, ray worker)         ← 持有训练后的模型权重
    └── 序列化 CUDA tensor 句柄 (pickle)
         └── 跨进程传给 sglang TP rank
              └── sglang 调 torch._new_shared_cuda 反序列化
                   └── 内核调 pidfd_getfd 跨进程取 file descriptor
                        └── ❌ Operation not permitted（被 seccomp 拦掉）
```

## 容器现状（来自 `/proc/self/status`）

```
CapInh: 0000000000000000     # 无可继承 capability
CapPrm: 0000000000000000     # 无 permitted capability
CapEff: 0000000000000000     # 无 effective capability
CapBnd: 00000000a80425fb     # bounding set 缺 cap_sys_ptrace
Seccomp: 2                   # filter 模式启用 (SECCOMP_MODE_FILTER)
Seccomp_filters: 1
```

`capsh --print` 输出确认 `cap_sys_ptrace` 在 negated 列表里。

`/proc/sys/kernel/yama/ptrace_scope = 1`（restricted）。

## 为什么 torch_pp 路径不受影响

`torch_pp` 路径用 **gloo CPU broadcast** 在 ranks 间同步权重，不走 CUDA IPC：

```python
# verl/workers/torch_pp_workers.py:980
# Broadcast on CPU (gloo-compatible, no CUDA IPC issues)
dist.broadcast(tensor_cpu, src=owning_rank)
```

不需要 `pidfd_getfd`，所以 docker 权限不影响。

## 为什么 megatron 路径必须 CUDA IPC

`verl + megatron` 路径里 sglang 跑在独立 ray worker 进程，必须跨进程传 GPU tensor 句柄：

```python
# sglang-fork/python/sglang/srt/weight_sync/utils.py
named_tensors_batch = [
    (name, MultiprocessingSerializer.serialize(tensor))   # ← CUDA IPC 序列化
    for name, tensor in params_batch
]
```

`MultiprocessingSerializer.serialize` 内部走 PyTorch 的 CUDA IPC 路径，PyTorch 2.x 该路径**强制**调 `pidfd_getfd`。环境变量、torch flag 都没法回退到旧机制。

## 修复方案

容器启动时需要加两个 docker flag：

```bash
docker run \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    [...其他参数]
```

或者更直接：

```bash
docker run --privileged [...其他参数]
```

**这不是镜像 (image) 的问题，是容器 runtime 启动配置的问题**——不需要重新 build image，只要重新 start 容器。

### 不同部署方式的对应改法

| 部署方式 | 改法 |
|---|---|
| 直接 `docker run` | 加上面两个 flag 重启容器 |
| Kubernetes pod | `securityContext.capabilities.add: ["SYS_PTRACE"]` + `seccompProfile.type: Unconfined` |
| Docker Compose | 在 compose.yml 的 service 下加 `cap_add: [SYS_PTRACE]` + `security_opt: ["seccomp=unconfined"]` |
| 托管云 GPU 实例 | 一般在实例配置里有 "privileged" / "高级 capabilities" 选项 |

### 验证 fix 是否生效

容器重启后在容器内执行：

```bash
# 1. 检查 capability
grep ^CapEff /proc/self/status
# 期望: 包含 cap_sys_ptrace 对应位 (bit 19, 即 0x80000)

# 2. 实测 pidfd_getfd
python3 -c "
import os, ctypes
SYS_pidfd_open = 434
SYS_pidfd_getfd = 438
libc = ctypes.CDLL('libc.so.6', use_errno=True)
pidfd = libc.syscall(SYS_pidfd_open, os.getpid(), 0)
fd = libc.syscall(SYS_pidfd_getfd, pidfd, 0, 0)
print('pidfd_getfd:', 'OK' if fd >= 0 else f'FAIL ({ctypes.get_errno()})')
"
# 期望: OK（修复前: FAIL (1) Operation not permitted）

# 3. 跑实际训练 smoke
cd /home/user/rlpipe/verl
bash exp_script/megatron_fused/idea2_megatron_baseline.sh
# 期望: 出现 "step:1" 行 + 无 pidfd_getfd 错误
```

## 安全考量（给云服务商参考）

| 选项 | 增加的风险 | 推荐度 |
|---|---|---|
| `--cap-add=SYS_PTRACE` 单独加 | 容器内进程可以 ptrace **同容器内**的其他进程；**无法**逃逸到 host | ✓ 最小化授权 |
| `--security-opt seccomp=unconfined` | 解开 syscall 黑名单（约 70 条），但**容器隔离仍生效**（namespace + cgroup）| ✓ 必须搭配 SYS_PTRACE |
| `--privileged` | 容器**等同 host root**，可访问 host 设备 | ✗ 过度授权 |

CUDA IPC 在多 GPU 训练里是**通行需求**——主流 ML 平台（NVIDIA NGC, PyTorch official, vLLM/sglang docker images）的官方文档都标注需要 `--cap-add=SYS_PTRACE` 或 privileged。例：

- NVIDIA: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch> 启动示例用 `--privileged`
- vLLM: 文档里"distributed inference"段建议 SYS_PTRACE
- sglang: 多卡部署文档同样

所以这是**业界标准做法**，不是不寻常的需求。

## 影响 / 紧迫性

- **直接影响**：megatron 训练路径完全不可用 → 论文 fused forward 在 megatron backbone 上不能验证 → 老师之前提出的 framework-independence claim 需要更慎重的 framing
- **现状的临时方案**：在 verl 代码里把 CUDA IPC 路径替换成 NCCL collective broadcast (`update_weights_from_distributed`)，绕开 pidfd。但这是 1-2 天的 verl 核心 patch，且会引入 maintenance cost (上游 verl 更新得手动 merge)
- **首选**：拿到 docker 权限即可立即 unblock；零代码改动

## 相关文件 / 证据

- 错误 trace: `/home/user/profiling_rlpipe/idea2_megatron_baseline/seed_42/train.log`
- 我们的 verl prctl workaround（已尝试，无效）: `verl/verl/__init__.py:23-44`
- sglang 强制 IPC 序列化: `sglang-fork/python/sglang/srt/weight_sync/utils.py`
- torch 的 pidfd-only IPC: `torch/storage.py:_new_shared_cuda` (PyTorch >= 2.1)
- 测试通过的对比 (torch_pp): `/home/user/rlpipe/verl/exp_script/idea2_baseline.sh` — 38s/step OK
- 测试失败的 megatron: `/home/user/rlpipe/verl/exp_script/megatron_fused/idea2_megatron_baseline.sh` — pidfd_getfd 错

## 一句话给云服务商

> 我们容器内跑 PyTorch + sglang 多卡推理，需要 CUDA IPC（业界主流多卡推理框架的标准依赖）。麻烦容器启动时加 `--cap-add=SYS_PTRACE` + `--security-opt seccomp=unconfined`（或 `--privileged`），不需要新建镜像。验证方法：容器内 `grep ^CapEff /proc/self/status` 应包含 `cap_sys_ptrace`。

## 一句话给老师

> Megatron 路径被 docker 权限阻塞（容器没 SYS_PTRACE，sglang 跨进程传 CUDA tensor 失败）。两条平行路径：(1) 联系云服务商加 capability，5 分钟 unblock；(2) 改 verl 用 NCCL collective 替代 CUDA IPC，1-2 天工程量。我已经在做 (2) 不让进度卡住，同时希望 (1) 走得通——后者干净，前者上游 verl 更新维护成本高。
