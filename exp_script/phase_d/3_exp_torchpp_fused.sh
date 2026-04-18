#!/bin/bash
# Phase D exp 1: torch_pp PP=4 + fused forward, stock SGLang DP=4.
# Based on proven run_vllm_torch_pp_dp4.sh with DAPO-Math data.
set -o pipefail
set -x
rm -f log_rank_*.txt

SEED=${SEED:-42}
TOTAL_STEPS=${TOTAL_STEPS:-10}
BATCH=${SMOKE_BATCH:-128}
N_SAMPLES=${SMOKE_N:-16}
MAX_RESP=${SMOKE_MAX_RESP:-16384}
MINI_BATCH=${SMOKE_MINI_BATCH:-16}

EXP_NAME=exp1_torchpp_fused
PROFILING_ROOT=$HOME/profiling_phase_d
PROFILING_DIR=$PROFILING_ROOT/$EXP_NAME/seed_${SEED}
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"

GPUS_PER_NODE=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=4
GPU_MEMORY_UTILIZATION=0.7
MODEL_PATH="Qwen/Qwen3-1.7B"
train_files="['$HOME/data/dapo-math-4k/train.parquet']"
test_files="['$HOME/data/dapo-math-4k/val.parquet']"

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_torch_pp_trainer' \
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
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.grad_clip=1.0 \
	actor_rollout_ref.actor.param_offload=True \
	actor_rollout_ref.actor.optimizer_offload=True \
	actor_rollout_ref.actor.fused_forward=True \
	actor_rollout_ref.actor.num_micro_batches=$MINI_BATCH \
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
	actor_rollout_ref.ref.fsdp_config.param_offload=True \
	actor_rollout_ref.enable_pp_trace=true \
	actor_rollout_ref.profiling_save_dir=$PROFILING_DIR/pp_traces \
	reward_model.reward_manager=dapo \
	+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
	+reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
	+reward_model.reward_kwargs.max_resp_len=$MAX_RESP \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console'] \
	trainer.project_name=phase_d \
	trainer.experiment_name=${EXP_NAME}_seed${SEED} \
	trainer.resume_mode=disable \
	trainer.n_gpus_per_node=$GPUS_PER_NODE \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=9999 \
	trainer.val_before_train=False \
	trainer.total_training_steps=$TOTAL_STEPS \
	trainer.total_epochs=5 2>&1 | stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' | tee $PROFILING_DIR/train.log
