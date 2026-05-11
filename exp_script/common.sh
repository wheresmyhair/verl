#!/bin/bash
# Shared configuration for rlpipe paper experiments.
# Source this from every top-level comparison script.
#
# Acceptance criterion (set by advisor meeting 2026-04-19):
#   OOM-free run at BATCH=16, N_SAMPLES=16, MAX_RESP=16384 on 4×A100-80G.
#
# Override any variable from the environment before sourcing, e.g.
#   TOTAL_STEPS=1 SEED=42 bash exp_script/idea1_baseline.sh

# ---- Model and data ----
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-1.7B}"
export TRAIN_FILES="${TRAIN_FILES:-['$HOME/data/dapo-math-4k/train.parquet']}"
export VAL_FILES="${VAL_FILES:-['$HOME/data/dapo-math-4k/val.parquet']}"

# ---- Batch sizing (paper default) ----
export BATCH="${BATCH:-16}"                              # data.train_batch_size (prompts per step)
export N_SAMPLES="${N_SAMPLES:-16}"                      # rollout.n (responses per prompt)
export MAX_RESP="${MAX_RESP:-16384}"
export MAX_PROMPT="${MAX_PROMPT:-2048}"
export MINI_BATCH="${MINI_BATCH:-16}"                    # actor.ppo_mini_batch_size
export INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-1}" # rollout/ref log_prob micro batch per GPU

# ---- Run control ----
export TOTAL_STEPS="${TOTAL_STEPS:-5}"
export SEED="${SEED:-42}"
export LR="${LR:-1e-6}"
export KL_COEF="${KL_COEF:-0.001}"

# ---- Topology ----
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"

# ---- Output ----
export PROFILING_ROOT="${PROFILING_ROOT:-$HOME/profiling_rlpipe}"

# ---- NCCL stability ----
export GLOO_SOCKET_TIMEOUT="${GLOO_SOCKET_TIMEOUT:-7200}"
export TORCH_NCCL_DEFAULT_TIMEOUT_SECONDS="${TORCH_NCCL_DEFAULT_TIMEOUT_SECONDS:-7200}"
