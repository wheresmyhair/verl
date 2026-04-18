#!/bin/bash
# Phase C combined: Megatron fused forward + fan-in rollout.
# First script that enables both paper contributions simultaneously.
#
# Contributions stacked:
#   1. Fan-in: SGLang dynamic-TP switches topology during rollout tail
#      (VERL_SGLANG_DYNAMIC_TP + VERL_RLPIPE_FANIN)
#   2. Fused forward: Megatron iF/tF/tB interleaved with reverse-PP inference
#      (fused_forward=true + RLPIPE_FUSED_REVERSE_PP)
#
# Training: Megatron PP=4, TP=1
# Rollout:  SGLang TP=4 (initial), fan-in to DP during tail (per fork logic)
set -x
rm -f log_rank_*.txt

# === Fused forward flags ===
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1
export RLPIPE_FUSED_REVERSE_PP=1
export RLPIPE_FUSED_FORWARD_SHARDED=1

# === Fan-in flags ===
export VERL_SGLANG_DYNAMIC_TP=1
export VERL_RLPIPE_FANIN=1
export VERL_RLPIPE_FANIN_MIN_IDLE=1
export SGLANG_DYNAMIC_TP_PRE_CAPTURE=1
export SGLANG_DYNAMIC_TP_INITIAL=tp

# NCCL/Gloo timeouts for long-sequence tolerance
export GLOO_SOCKET_TIMEOUT=7200
export TORCH_NCCL_DEFAULT_TIMEOUT_SECONDS=7200

PROFILING_DIR=/home/user/profiling_p3_megatron_fused_fanin
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"
export RLPIPE_MEGATRON_PP_TRACE=$PROFILING_DIR
export RLPIPE_SGFANIN_TRACE=$PROFILING_DIR
export RLPIPE_SGFANIN_FANIN_DEBUG=$PROFILING_DIR/fanin_debug.log

GPUS_PER_NODE=4
TRAIN_TP=1
TRAIN_PP=4
ROLLOUT_TP=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.5  # lower because dynamic-TP needs dual KV pools

YOUR_PROJECT_NAME=verl-rollout-optim
YOUR_RUN_NAME=p3-16k-megatron-fused-fanin

# Use DAPO-Math (already preprocessed, 4000 rows)
train_files="['$HOME/data/dapo-math-4k/train.parquet']"
test_files="['$HOME/data/dapo-math-4k/val.parquet']"

MODEL_PATH="Qwen/Qwen3-1.7B"

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.prompt_key=prompt \
	data.train_batch_size=16 \
	data.max_prompt_length=2048 \
	data.max_response_length=8192 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.model.use_remove_padding=true \
	++actor_rollout_ref.model.enable_gradient_checkpointing=True \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=16 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$TRAIN_PP \
	+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
	+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
	+actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
	++actor_rollout_ref.actor.fused_forward=true \
	++actor_rollout_ref.actor.use_fused_forward_pp=true \
	++actor_rollout_ref.actor.use_fused_forward_pp_micro_batch_size=4 \
	++actor_rollout_ref.actor.use_fused_forward_pp_model_path=$MODEL_PATH \
	++actor_rollout_ref.ref.use_fused_forward_pp=true \
	++actor_rollout_ref.ref.use_fused_forward_pp_micro_batch_size=4 \
	++actor_rollout_ref.ref.use_fused_forward_pp_model_path=$MODEL_PATH \
	actor_rollout_ref.actor.megatron.param_offload=True \
	actor_rollout_ref.actor.megatron.grad_offload=True \
	actor_rollout_ref.actor.megatron.optimizer_offload=True \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
	++actor_rollout_ref.rollout.tp_groups=[[0,1,2,3]] \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=4 \
	actor_rollout_ref.rollout.temperature=1.0 \
	actor_rollout_ref.rollout.top_p=1.0 \
	actor_rollout_ref.nccl_timeout=600 \
	actor_rollout_ref.rollout.enable_chunked_prefill=False \
	+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
	+actor_rollout_ref.rollout.engine_kwargs.sglang.enable_deterministic_inference=True \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$TRAIN_PP \
	reward_model.reward_manager=dapo \
	+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
	+reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
	+reward_model.reward_kwargs.max_resp_len=16384 \
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
	trainer.total_training_steps=1 \
	trainer.total_epochs=1 2>&1 | stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_16k_megatron_fused_fanin.txt
