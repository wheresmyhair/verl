#!/bin/bash
# Debug script: run 1 epoch of GRPO with rollout data dumping to inspect response quality
set -x
rm -f log_rank_*.txt

GPUS_PER_NODE=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=64
GPU_MEMORY_UTILIZATION=0.7

YOUR_PROJECT_NAME=verl-debug
YOUR_RUN_NAME=debug-rollout-1.7b

# Preprocess data
python3 examples/data_preprocess/gsm8k_all.py --local_dir $HOME/data/gsm8k-$YOUR_RUN_NAME

gsm8k_train_path=$HOME/data/gsm8k-$YOUR_RUN_NAME/train.parquet
gsm8k_test_path=$HOME/data/gsm8k-$YOUR_RUN_NAME/test.parquet

train_files="['$gsm8k_train_path']"
test_files="['$gsm8k_test_path']"

MODEL_PATH="Qwen/Qwen3-1.7B"
ROLLOUT_DUMP_DIR=/home/user/rollout_debug_output

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_torch_pp_trainer' \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.train_batch_size=128 \
	data.max_prompt_length=1024 \
	data.max_response_length=7168 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=16 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=0.001 \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.grad_clip=1.0 \
	actor_rollout_ref.actor.param_offload=True \
	actor_rollout_ref.actor.optimizer_offload=True \
	actor_rollout_ref.actor.fused_forward=False \
	actor_rollout_ref.actor.num_micro_batches=64 \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
	actor_rollout_ref.rollout.name=$ENGINE \
	actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
	actor_rollout_ref.rollout.n=5 \
	actor_rollout_ref.nccl_timeout=600 \
	actor_rollout_ref.rollout.enable_chunked_prefill=False \
	+actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
	actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	algorithm.kl_ctrl.kl_coef=0.001 \
	trainer.critic_warmup=0 \
	trainer.logger=['console'] \
	trainer.project_name=$YOUR_PROJECT_NAME \
	trainer.experiment_name=$YOUR_RUN_NAME \
	trainer.resume_mode=disable \
	trainer.n_gpus_per_node=$GPUS_PER_NODE \
	trainer.nnodes=1 \
	trainer.rollout_data_dir=$ROLLOUT_DUMP_DIR \
	actor_rollout_ref.profiling_save_dir=/home/user/profiling_debug \
	trainer.save_freq=-1 \
	trainer.test_freq=9999 \
	trainer.total_epochs=1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee log_debug_rollout.txt
