#!/bin/bash
# M6 Phase B: real GRPO measurement on 14B + DAPO-Math with our dynamic-tp fork.
#
# Adapted from 7_dapo_math_r1distill_14b_confirm.sh (Path B v3). Two changes
# relative to the Path B v3 baseline:
#
#   1. VERL_SGLANG_DYNAMIC_TP=1   — activates our fork's dynamic-tp path
#      (env var + ModelRegistry alias + boot-in-tp + post-boot switch to dp)
#   2. rollout.tensor_model_parallel_size: 1 → 4
#      Needed because our dynamic-tp fork operates on a tp_size=4 scheduler
#      group that can then be SWITCHED between "tp" and "dp" topologies at
#      runtime. Memory footprint is identical (4 full-model replicas, one
#      per GPU) — Path B v3's tp=1 spawned 4 sglang workers via the
#      DataParallelController, ours spawns 4 via mp.spawn and switches to
#      "dp" topology right after boot. Apples-to-apples on layout.
#
# Everything else (data, batch size, response length, n=16, mfs, etc.)
# matches Path B v3 so the step timings are directly comparable.
#
# Expected outcome: our dynamic-tp DP mode should match or slightly beat
# Path B v3's step time because the flashinfer decode wrapper is now
# planned with the correct DP-mode head counts (8 kv heads / 16 q heads
# per rank instead of 2/4 that TP would use). At per-rank decode batch
# ~64 (16 prompts × n=16 / 4 ranks) this is the exact regime the M6
# Phase A bug hid in.
set -x
rm -f log_rank_*.txt

export VERL_SGLANG_DYNAMIC_TP=1
export GLOO_SOCKET_TIMEOUT=7200
export TORCH_NCCL_DEFAULT_TIMEOUT_SECONDS=7200

PROFILING_DIR=/home/user/profiling_p3_m6b_14b_dp
if [ -d "$PROFILING_DIR" ]; then
    BACKUP_DIR="${PROFILING_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$PROFILING_DIR" "$BACKUP_DIR"
fi
mkdir -p "$PROFILING_DIR"

GPUS_PER_NODE=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=8
GPU_MEMORY_UTILIZATION=0.55

YOUR_PROJECT_NAME=verl-rollout-optim
YOUR_RUN_NAME=p3-m6b-14b-dapo

MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

train_files="['$HOME/data/dapo-math-poc2/train.parquet']"
test_files="['$HOME/data/dapo-math-poc2/val.parquet']"

python3 -m verl.trainer.main_ppo \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.prompt_key=prompt \
	data.train_batch_size=16 \
	data.max_prompt_length=2048 \
	data.max_response_length=16384 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.model.use_remove_padding=true \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=16 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=False \
	actor_rollout_ref.actor.kl_loss_coef=0.0 \
	actor_rollout_ref.actor.grad_clip=1.0 \
	actor_rollout_ref.actor.fsdp_config.param_offload=True \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
	actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=16 \
	actor_rollout_ref.rollout.temperature=1.0 \
	actor_rollout_ref.rollout.top_p=1.0 \
	actor_rollout_ref.nccl_timeout=600 \
	actor_rollout_ref.rollout.enable_chunked_prefill=True \
	+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	reward_model.reward_manager=dapo \
	+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
	+reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
	+reward_model.reward_kwargs.max_resp_len=16384 \
	algorithm.use_kl_in_reward=False \
	algorithm.kl_ctrl.kl_coef=0.0 \
	trainer.critic_warmup=0 \
	trainer.logger=['console'] \
	trainer.project_name=$YOUR_PROJECT_NAME \
	trainer.experiment_name=$YOUR_RUN_NAME \
	trainer.resume_mode=disable \
	trainer.n_gpus_per_node=$GPUS_PER_NODE \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=9999 \
	trainer.val_before_train=False \
	trainer.total_training_steps=2 \
	trainer.total_epochs=1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_m6b_14b.txt
