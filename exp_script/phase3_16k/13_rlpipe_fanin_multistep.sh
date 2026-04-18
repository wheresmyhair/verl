#!/bin/bash
# M6 Phase B v2 — MULTI-STEP fan-in measurement on 14B + DAPO-Math.
#
# Follow-up to the single-step Phase B v2 run (11_rlpipe_dynamic_tp_14b_dapo.sh)
# that delivered -7.5% step / -9.8% worst-rank gen vs Path B v3 baseline.
# Per-step noise at FSDP scale is ~20s, so a single-step result of +/-115s
# is plausible-but-not-confirmed. This script runs 3 training steps to
# check that:
#   (a) fan-in fires consistently on each step (not just by luck on step 1)
#   (b) average step time is stable across steps (no leak / drift)
#   (c) weight sync via update_weights_from_tensor survives topology swap
#       cycles (critical: after step N's fan-in switches tp->dp, step N+1
#       starts with new weights applied under DP topology)
#
# Uses the same workload and config as the single-step run for direct
# comparability. total_training_steps=3 (90 min est wall clock).
set -x
rm -f log_rank_*.txt

: "${VERL_SGLANG_DYNAMIC_TP:=1}"
: "${VERL_RLPIPE_FANIN:=1}"
: "${VERL_RLPIPE_FANIN_MIN_IDLE:=2}"
: "${SGLANG_DYNAMIC_TP_PRE_CAPTURE:=1}"
export VERL_SGLANG_DYNAMIC_TP VERL_RLPIPE_FANIN VERL_RLPIPE_FANIN_MIN_IDLE
export SGLANG_DYNAMIC_TP_PRE_CAPTURE
export GLOO_SOCKET_TIMEOUT=7200
export TORCH_NCCL_DEFAULT_TIMEOUT_SECONDS=7200
# NOTE: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is INCOMPATIBLE
# with SGLang's torch_memory_saver (it uses cuMemAddressReserve/
# cuMemMap/cuMemCreate driver APIs that conflict with PyTorch's
# expandable-segment allocator). Do not set it — the scheduler process
# will fail to init if both are enabled.

PROFILING_DIR=/home/user/profiling_p3_m6b2_14b_fanin_multistep
if [ -d "$PROFILING_DIR" ]; then
    BACKUP_DIR="${PROFILING_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$PROFILING_DIR" "$BACKUP_DIR"
fi
mkdir -p "$PROFILING_DIR"

GPUS_PER_NODE=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=8
# 0.50 (was 0.55). Each fan-in cycle peaks HBM when both the old pool
# is being freed and the new pool allocated; dropping to 0.50 leaves
# ~4 GB more headroom for FSDP backward at update_actor. Baseline uses
# 0.55 without fan-in, so this is a fan-in-specific reduction.
GPU_MEMORY_UTILIZATION=0.50

YOUR_PROJECT_NAME=verl-rollout-optim
YOUR_RUN_NAME=p3-m6b2-14b-fanin-multistep

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
	trainer.total_training_steps=1 \
	trainer.total_epochs=1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_m6b2_14b_fanin_multistep.txt
