#!/bin/bash
# Megatron PP=4 + real fused forward (fused_update_actor) on A100.
# Merges compute_log_prob + update_actor into a single fused_update_actor
# call, eliminating one Megatron load/unload cycle per step.
#
# Comparison:
#   Script 3  — baseline: separate compute_log_prob (Megatron PP) + update_actor
#   Script 4  — Phase 2-Lite: separate compute_log_prob (HF sharded) + update_actor
#   Script 28 — THIS: single fused_update_actor (HF inference + Megatron training)
#
# Expected saving vs script 4: eliminates load_inference + unload_inference
# (~2-4s per step at 1.7B).
set -x
rm -f log_rank_*.txt

export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1

# V2: use PP-sharded reverse-direction inference (Megatron model, each rank
# holds 1/pp_size of inference params). Eliminates the 3.4 GB HF full replica
# and per-iF load/offload overhead.
export RLPIPE_FUSED_REVERSE_PP=1
# Shard HF inference across PP ranks (4×) for maximum speed.
export RLPIPE_FUSED_FORWARD_SHARDED=1

PROFILING_DIR=/home/user/profiling_p3_megatron_fused_update_actor
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"
export RLPIPE_MEGATRON_PP_TRACE=$PROFILING_DIR

GPUS_PER_NODE=4
TRAIN_TP=1
TRAIN_PP=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.7

YOUR_PROJECT_NAME=verl-rollout-optim
YOUR_RUN_NAME=p3-16k-megatron-fused-update-actor

python3 examples/data_preprocess/gsm8k_all.py --local_dir $HOME/data/gsm8k-$YOUR_RUN_NAME

gsm8k_train_path=$HOME/data/gsm8k-$YOUR_RUN_NAME/train.parquet
gsm8k_test_path=$HOME/data/gsm8k-$YOUR_RUN_NAME/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

MODEL_PATH="Qwen/Qwen3-1.7B"

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_megatron_trainer' \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.train_batch_size=128 \
	data.max_prompt_length=1024 \
	data.max_response_length=16384 \
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
	actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=5 \
	actor_rollout_ref.nccl_timeout=120 \
	actor_rollout_ref.rollout.enable_chunked_prefill=False \
	+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.ref.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=$TRAIN_PP \
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
	trainer.total_training_steps=3 \
	trainer.total_epochs=5 2>&1 | stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_16k_megatron_fused_update_actor.txt
