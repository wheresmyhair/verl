# Torch Naive PP Implementation for verl

## Overview

New `torch_naive_pp` strategy alongside `fsdp` and `megatron`. Each GPU = 1 PP stage (training) + 1 vLLM instance (DP rollout). Optional fused forward computes old_log_probs inline during PP bubbles.

## Files Created

### `verl/workers/torch_pp/` package (8 files)
| File | Purpose |
|---|---|
| `__init__.py` | Package init, re-exports all public symbols |
| `partitioner.py` | `compute_layer_assignment()` — distributes layers across PP stages |
| `pipeline_stage.py` | HF model wrapper with **partial loading via safetensors** (meta device → prune → materialize → selective shard load). `forward_step()`, `backward_step()`, `get_global_state_dict()` with global/local layer key remapping |
| `comm.py` | NCCL-native P2P: `send/recv_activation`, `send/recv_grad`, `send/recv_infer_activation`, `send/recv_old_log_probs`. No shape headers — receiver pre-allocates from model config. Tag ranges: [0,10k), [10k,20k), [20k,30k), [30k,40k) |
| `schedule.py` | `build_1f1b_schedule()` and `ScheduleOp` for standard 1F1B pipeline |
| `fused_schedule.py` | `FusedScheduleOp`, `parse_schedule()`, `validate_schedule()`, `build_default_fused_schedule()` with greedy simulation and Kahn's deadlock detection |
| `inference_stage.py` | Wraps PipelineStage at reversed rank (`pp_size-1-train_rank`), eval mode, no_grad, `compute_log_probs()` |
| `loss.py` | `compute_grpo_loss()`, `log_probs_from_logits()`, `gather_response_log_probs()`, `entropy_from_logits()` |

### Worker and config (4 files)
| File | Purpose |
|---|---|
| `verl/workers/torch_pp_workers.py` | `ActorRolloutRefWorker(Worker)` — main worker following exact megatron patterns (`simple_timer`, `reduce_timing`, `aggressive_empty_cache`, `GPUMemoryLogger`, `log_gpu_memory_usage`, `set_numa_affinity`, `set_expandable_segments`, `topk_reduce_ratio_min_max`) |
| `verl/trainer/config/actor/torch_pp_actor.yaml` | Hydra actor config for torch_naive_pp |
| `verl/trainer/config/ppo_torch_pp_trainer.yaml` | Full trainer config (GRPO, no critic) |
| `exp_script/run_vllm_torch_pp_dp4.sh` | Experiment launcher |

## Files Modified

### `verl/trainer/main_ppo.py`
Added `torch_naive_pp` strategy branch in `TaskRunner.add_actor_rollout_worker()`:
```python
elif config.actor_rollout_ref.actor.strategy == "torch_naive_pp":
    from verl.workers.torch_pp_workers import ActorRolloutRefWorker
    actor_rollout_cls = ActorRolloutRefWorker
    ray_worker_group_cls = RayWorkerGroup
```

### `verl/trainer/ppo/ray_trainer.py`
Added fused forward conditional in `RayPPOTrainer.fit()`:
- When `fused_forward=True`: skips `compute_log_prob`, calls `fused_update_actor` after advantage computation (returns old_log_probs + metrics in one call)
- When `fused_forward=False`: standard path unchanged

### `verl/workers/config/actor.py`
Added `TorchPPActorConfig(ActorConfig)`:
```python
strategy: str = "torch_naive_pp"
grad_clip: float = 1.0
param_offload: bool = False
optimizer_offload: bool = False
fused_forward: bool = False
num_micro_batches: int = 4
```

## Worker Methods (matches megatron interface)

| Method | Dispatch | Description |
|---|---|---|
| `init_model()` | ONE_TO_ALL | Load tokenizer, build PipelineStage (partial loading), optimizer, vLLM rollout, ref stage, inference stage |
| `generate_sequences()` | DYNAMIC_INDEX_DISPATCH | rollout_mode → vLLM generate → trainer_mode |
| `compute_log_prob()` | mesh("actor") | Forward-only PP, extract response log probs + entropy |
| `compute_ref_log_prob()` | mesh("actor") | Same PP forward with frozen ref stage |
| `update_actor()` | mesh("actor") | 1F1B schedule, GRPO loss on last stage, optimizer step |
| `fused_update_actor()` | mesh("actor") | Interleaved tF/tB/iF, inline old_log_probs, returns both |

## Dispatch Mechanics

- **"actor" mesh**: all workers `dp_rank=0`, only last PP stage `is_collect=True` → all workers get same data, only last returns results
- **"rollout" mesh**: each worker `dp_rank=rank`, all `is_collect=True` → data split across GPUs for DP rollout

## Bug Fixes

### PP weight sync producing gibberish during SGLang rollout (2026-04-03)

**Symptom:** All rollout outputs were random multilingual gibberish (entropy ~9.4 vs correct ~4.2). Model produced 0% reward on GSM8K. All responses hit max_length. Training was unaffected — only the SGLang rollout was broken.

**Root cause:** In PP mode, each `PipelineStage` wraps the full `Qwen3ForCausalLM` with pruned layers. Non-layer parameters (`model.embed_tokens.weight`, `model.norm.weight`, `lm_head.weight`) exist in **every stage's state dict**, but only the owning stage's copy is kept current during training.

`_collect_full_state_dict()` gathers all stages via `all_gather_object` and yields them sequentially (stage 0 → 3). For duplicate keys, the last stage's tensor is what SGLang's `load_weights()` sees last — overwriting correct weights (e.g., stage 0's embedding) with stale copies from stage 3. The model then runs with a corrupted embedding layer.

**Fix** (`verl/workers/torch_pp_workers.py`, `_collect_full_state_dict`): Deduplicate by tracking seen parameter names. Only yield each name from the first stage that has it:

```python
seen = set()
for stage_sd in gathered:
    for name, tensor in stage_sd.items():
        if name in seen:
            continue
        seen.add(name)
        yield name, tensor.to(self.device) if tensor.device.type == "cpu" else tensor
```

**Impact:** All prior PP>1 + SGLang rollout experiments had invalid rollout data. Training weights were correct (uses `load_state_dict_from_full` which handles this properly).
