#!/bin/bash
# verl Megatron PP=4 + fused-forward Phase 1 MVP on A100 for 16K response.
# Builds on `3_megatron_baseline_a100.sh` baseline (same config) and flips
# `actor_rollout_ref.actor.use_fused_forward_pp=true` to short-circuit
# `compute_log_prob` through a local HF Qwen3-1.7B replica on each rank.
# Expected saving: ~27 s per step (eliminates Megatron PP-4 compute_log_prob).
set -x
rm -f log_rank_*.txt

# Disable verl's trainer_mode set_expandable_segments(True) because
# expandable_segments tensors participate in CUDA IPC via pidfd_getfd,
# which fails with "Operation not permitted" under restricted Yama
# ptrace_scope (default in our docker). This is a known verl CI
# workaround — see verl/workers/megatron_workers.py line 620.
export MEGATRON_CI_DISABLE_EXPANDABLE_SEGMENTS=1

# Toggle numerical parity check against the Megatron PP=4 path. When set,
# `_compute_log_prob_fused_forward` also runs the baseline Megatron forward
# on the same batch and logs max/mean |HF - MG| across real response tokens.
# Doubles compute_log_prob wall time, so leave unset by default.
# export RLPIPE_FUSED_FORWARD_COMPARE=1

PROFILING_DIR=/home/user/profiling_p3_megatron_fused_forward
if [ -d "$PROFILING_DIR" ]; then
    BACKUP_DIR="${PROFILING_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$PROFILING_DIR" "$BACKUP_DIR"
fi
mkdir -p "$PROFILING_DIR"

GPUS_PER_NODE=4
TRAIN_TP=1
TRAIN_PP=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.7

YOUR_PROJECT_NAME=verl-rollout-optim
YOUR_RUN_NAME=p3-16k-megatron-fused-forward-a100

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
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=16 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.megatron.tensor_model_parallel_size=$TRAIN_TP \
	actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=$TRAIN_PP \
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
	trainer.total_epochs=1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_16k_megatron_fused_forward.txt
