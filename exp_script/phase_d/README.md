# Phase D — 6-experiment factorial for paper validation

Two paper contributions tested as a 2×2 factorial (Megatron backbone) + 1×2 generality check (torch_pp backbone).

## Comparison matrix

| # | Script | Rollout topology | Training backbone | fused_forward | Purpose |
|---|--------|------------------|-------------------|---------------|---------|
| **baseline 1** | `1_baseline_megatron_default.sh` | stock SGLang DP=4 | Megatron PP=4 | OFF | Anchor for Megatron timing |
| **baseline 2** | `2_baseline_torchpp_default.sh` | stock SGLang DP=4 | torch_pp PP=4  | OFF | Anchor for torch_pp timing |
| **exp 1** | `3_exp_torchpp_fused.sh`           | stock SGLang DP=4 | torch_pp PP=4  | ON  | Fused forward on torch_pp (generality) |
| **exp 2** | `4_exp_megatron_fused.sh`          | stock SGLang DP=4 | Megatron PP=4  | ON  | Fused forward on Megatron (isolated) |
| **exp 3** | `5_exp_megatron_fanin.sh`          | fan-in fork (DP→TP) | Megatron PP=4 | OFF | Fan-in isolated (no fused) |
| **exp 4** | `6_exp_megatron_fused_fanin.sh`    | fan-in fork (DP→TP) | Megatron PP=4 | ON  | Combined (both contributions) |

### What changes per script (the key flags)

| Script | `rollout.tensor_model_parallel_size` | fan-in env vars | `fused_forward` | `strategy` | tp_groups |
|--------|--------------------------------------|-----------------|-----------------|------------|-----------|
| baseline 1 | `1` | (none) | `false` | `megatron` | (default 4 groups) |
| baseline 2 | `1` | (none) | `false` | `torch_naive_pp` | (default 4 groups) |
| exp 1 | `1` | (none) | `true` | `torch_naive_pp` | (default 4 groups) |
| exp 2 | `1` | (none) | `true` | `megatron` | (default 4 groups) |
| exp 3 | `4` | `VERL_SGLANG_DYNAMIC_TP=1` `VERL_RLPIPE_FANIN=1` `SGLANG_DYNAMIC_TP_INITIAL=dp` | `false` | `megatron` | (default, DP=4 from fork) |
| exp 4 | `4` | same as exp 3 | `true` | `megatron` | (default, DP=4 from fork) |

### Gain decomposition (what each comparison answers)

- **Fused forward on Megatron** (isolated): exp 2 − baseline 1
- **Fused forward on torch_pp** (cross-backbone): exp 1 − baseline 2
- **Fan-in on Megatron** (isolated): exp 3 − baseline 1
- **Combined (both)**: exp 4 − baseline 1
- **Interaction**: (exp 4 − exp 3) − (exp 2 − baseline 1)

## Common config (identical across all 6 scripts)

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-1.7B` |
| Data | `/home/user/data/dapo-math-4k/train.parquet` (4000 rows) |
| Reward | `dapo` (MATH_v2 verifier) |
| Hardware | 4× A100-80GB |
| `max_prompt_length` | 2048 |
| `max_response_length` | 16384 |
| `train_batch_size` | 128 (prompts) |
| `rollout.n` | 16 (samples per prompt → 2048 sequences/step) |
| `ppo_mini_batch_size` | 16 |
| `ppo_micro_batch_size_per_gpu` | 1 |
| `actor.optim.lr` | 1e-6 |
| `ppo_epochs` | 1 |
| `total_training_steps` | 10 |
| `total_epochs` | 5 (dataset cycling for safety) |
| `nccl_timeout` | 600 |
| `enable_gradient_checkpointing` | True |

For Megatron scripts only:
- `recompute_granularity=full`, `recompute_method=uniform`, `recompute_num_layers=1`
- `param_offload=True`, `grad_offload=True`, `optimizer_offload=True`

## Measurement protocol

**Per step**, verl logs (to both stdout and Perfetto):
- `timing_s/step`, `timing_s/gen`, `timing_s/ref`, `timing_s/adv`
- `timing_s/fused_update_actor` (or `timing_s/update_actor` + `timing_s/old_log_prob` for non-fused)
- `timing_s/generation_timing/max`, `/min` (rollout tail gap)
- `perf/throughput`, `perf/max_memory_allocated_gb`
- `actor/grad_norm`, `actor/pg_loss`, `actor/kl_loss`, `critic/score/mean`
- `response_length/mean`, `/max`, `/min`, `/p50`, `/p90`

**Warmup handling**: drop first 2-3 steps. Analysis uses steps 3-10 (8 samples per seed).

**Seeds**: 3 per config → 24 data points per metric per experiment.

## Profiling output structure

All traces saved under `$HOME/profiling_phase_d/`:

```
profiling_phase_d/
├── baseline1_megatron_default/
│   ├── seed_42/
│   │   ├── train.log                    # stdout
│   │   ├── megatron_pp_trace/           # Megatron PP schedule traces (Perfetto JSON)
│   │   └── step_metrics.jsonl           # per-step metrics from log
│   ├── seed_123/
│   └── seed_2024/
├── baseline2_torchpp_default/
│   ├── seed_42/
│   │   ├── train.log
│   │   └── pp_traces/                   # torch_pp Perfetto traces (if enable_pp_trace)
│   └── ...
├── exp1_torchpp_fused/              # same as baseline2 + fused traces
├── exp2_megatron_fused/             # same as baseline1 + fused_update_actor traces
├── exp3_megatron_fanin/             # baseline1 + sglang fan-in traces
│   ├── seed_42/
│   │   ├── train.log
│   │   ├── megatron_pp_trace/
│   │   └── sgfanin_trace/           # SGLang topology-switch traces
│   └── ...
├── exp4_megatron_fused_fanin/       # all traces combined
└── summary/                         # post-run analysis (step_metrics aggregation, plots)
```

## Environment variables used

**Megatron fused forward (exp 2, exp 4)**:
```bash
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1
export RLPIPE_FUSED_REVERSE_PP=1
export RLPIPE_FUSED_FORWARD_SHARDED=1
export RLPIPE_MEGATRON_PP_TRACE=<dir>     # Perfetto traces
```

**Fan-in rollout (exp 3, exp 4)**:
```bash
export VERL_SGLANG_DYNAMIC_TP=1
export VERL_RLPIPE_FANIN=1
export VERL_RLPIPE_FANIN_MIN_IDLE=1
export SGLANG_DYNAMIC_TP_PRE_CAPTURE=1
export SGLANG_DYNAMIC_TP_INITIAL=dp       # start DP, fan-in to TP for tail
export RLPIPE_SGFANIN_TRACE=<dir>         # Perfetto traces
export RLPIPE_SGFANIN_FANIN_DEBUG=<dir>/fanin_debug.log
```

**torch_pp profiling (baseline 2, exp 1)**:
```bash
# In config (via ++): actor.enable_pp_trace=true actor.profiling_save_dir=<dir>
```

## How to run

### Single experiment, single seed
```bash
cd /home/user/rlpipe/verl
SEED=42 bash exp_script/phase_d/1_baseline_megatron_default.sh
```

### All 6 experiments, 3 seeds each (full factorial)
```bash
cd /home/user/rlpipe/verl
bash exp_script/phase_d/run_all.sh
```

### Smoke test first (1 step, 1 seed per config)
```bash
cd /home/user/rlpipe/verl
SMOKE=1 SEED=42 bash exp_script/phase_d/run_all.sh
```

## Risks / known caveats

1. **torch_pp (baseline 2, exp 1)** was last validated on GSM8K not DAPO-Math. May need chat template / prompt_key adjustment. Pilot recommended before running all seeds.
2. **exp 3/4 INITIAL=dp** path is untested — Phase C pilot used `INITIAL=tp` which runs fully in TP mode (no actual DP→TP switching). With `INITIAL=dp` the fan-in trigger logic should fire, but `tp_groups` routing may need adjustment. **Pilot first** before running all seeds.
3. **Total wall time**: at ~10-15 min/step × 10 steps × 6 exp × 3 seeds ≈ **30-45 hours of GPU time**. Consider scaling down for first pass (e.g., 2 seeds × 6 exp × 8 steps).
4. **Memory**: fan-in adds dual KV pools (~3GB HBM overhead). With `gpu_memory_utilization=0.5` there should be headroom but watch for step-2 OOM on longer sequences.

## Post-run analysis

After all runs complete, run:
```bash
python3 exp_script/phase_d/analyze.py $HOME/profiling_phase_d/
```

Outputs in `profiling_phase_d/summary/`:
- `timing_table.csv` — mean±std of key timings per experiment
- `convergence_plot.png` — grad_norm, loss, score curves per experiment × seed
- `perfetto_links.md` — list of trace JSON files for Perfetto UI
