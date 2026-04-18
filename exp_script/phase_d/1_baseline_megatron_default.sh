#!/bin/bash
# Phase D baseline 1: Megatron PP=4 separate compute_log_prob + update_actor,
# stock SGLang rollout DP=4.
#
# This is the anchor Megatron baseline (no fused forward, no fan-in).
# Gain comparisons:
#   exp 2 - baseline 1 → fused forward gain on Megatron
#   exp 3 - baseline 1 → fan-in gain on Megatron
#   exp 4 - baseline 1 → combined gain
set -o pipefail
set -x
rm -f log_rank_*.txt

# ========== Seed + profiling setup ==========
SEED=${SEED:-42}
TOTAL_STEPS=${TOTAL_STEPS:-10}
# SMOKE overrides (from run_all.sh SMOKE=1). Default: use script's own values.
BATCH=${SMOKE_BATCH:-128}
N_SAMPLES=${SMOKE_N:-16}
MAX_RESP=${SMOKE_MAX_RESP:-16384}
MINI_BATCH=${SMOKE_MINI_BATCH:-16}

EXP_NAME=baseline1_megatron_default
PROFILING_ROOT=$HOME/profiling_phase_d
PROFILING_DIR=$PROFILING_ROOT/$EXP_NAME/seed_${SEED}
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"

# Megatron-specific env (from project_megatron_baseline_fixes.md)
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1
# Perfetto traces for Megatron PP schedule (only emits if PP schedule runs)
export RLPIPE_MEGATRON_PP_TRACE=$PROFILING_DIR/megatron_pp_trace
mkdir -p $RLPIPE_MEGATRON_PP_TRACE

GPUS_PER_NODE=4
TRAIN_TP=1
TRAIN_PP=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.7

MODEL_PATH="Qwen/Qwen3-1.7B"
train_files="['$HOME/data/dapo-math-4k/train.parquet']"
test_files="['$HOME/data/dapo-math-4k/val.parquet']"

YOUR_PROJECT_NAME=phase_d
YOUR_RUN_NAME=${EXP_NAME}_seed${SEED}

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.prompt_key=prompt \
	data.train_batch_size=$BATCH \
	data.max_prompt_length=2048 \
	data.max_response_length=$MAX_RESP \
	data.shuffle=True \
	data.seed=$SEED \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.model.use_remove_padding=true \
	++actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$TRAIN_PP \
	+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
	+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
	+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
	actor_rollout_ref.actor.megatron.param_offload=True \
	actor_rollout_ref.actor.megatron.grad_offload=True \
	actor_rollout_ref.actor.megatron.optimizer_offload=True \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=$N_SAMPLES \
	actor_rollout_ref.rollout.temperature=1.0 \
	actor_rollout_ref.rollout.top_p=1.0 \
	actor_rollout_ref.nccl_timeout=600 \
	actor_rollout_ref.rollout.enable_chunked_prefill=False \
	+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$TRAIN_PP \
	reward_model.reward_manager=dapo \
	+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
	+reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
	+reward_model.reward_kwargs.max_resp_len=$MAX_RESP \
	algorithm.kl_ctrl.kl_coef=0.001 \
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
	trainer.total_training_steps=$TOTAL_STEPS \
	trainer.total_epochs=5 2>&1 | stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' | tee $PROFILING_DIR/train.log
