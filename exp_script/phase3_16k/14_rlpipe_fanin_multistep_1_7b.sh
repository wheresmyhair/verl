#!/bin/bash
# M6 Phase B v2 fast-iteration variant: 3-step fan-in test on Qwen3-1.7B +
# DAPO-Math so we can debug the step-2 tp->dp switch-back crash with a
# much shorter rollout wall clock (~3-5 min/step vs ~20 min/step on 14B).
#
# The bug is about topology-switch mechanics, not model size, so any
# smaller model that exercises the same set_dynamic_topology path should
# reproduce the crash. We keep the same:
#   - total_training_steps=3 to reach step 2 fan-in
#   - VERL_RLPIPE_FANIN=1 + min_idle_ranks=2 to trigger fan-in
#   - n=16 per prompt so there's enough spread for idle detection
#   - DAPO-Math + overlong penalty to keep straggler tail distribution
# And scale down:
#   - Qwen3-1.7B (10x smaller than 14B)
#   - max_response_length=4096 (4x smaller than 16384)
#   - ppo_mini_batch_size=32 (fits without param offload)
#   - no FSDP offloads (the model is small enough)
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

GPUS_PER_NODE=4
ENGINE=sglang
INFERENCE_BATCH_SIZE=32
GPU_MEMORY_UTILIZATION=0.55

YOUR_PROJECT_NAME=verl-rollout-optim
YOUR_RUN_NAME=p3-m6b2-1_7b-fanin-multistep

MODEL_PATH="Qwen/Qwen3-1.7B"

train_files="['$HOME/data/dapo-math-poc2/train.parquet']"
test_files="['$HOME/data/dapo-math-poc2/val.parquet']"

python3 -m verl.trainer.main_ppo \
	algorithm.adv_estimator=grpo \
	data.train_files=$train_files \
	data.val_files=$test_files \
	data.prompt_key=prompt \
	data.train_batch_size=32 \
	data.max_prompt_length=2048 \
	data.max_response_length=4096 \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.model.use_remove_padding=true \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.actor.ppo_mini_batch_size=32 \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
	actor_rollout_ref.actor.use_kl_loss=False \
	actor_rollout_ref.actor.kl_loss_coef=0.0 \
	actor_rollout_ref.actor.grad_clip=1.0 \
	actor_rollout_ref.actor.fsdp_config.param_offload=False \
	actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
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
	actor_rollout_ref.ref.fsdp_config.param_offload=False \
	reward_model.reward_manager=dapo \
	+reward_model.reward_kwargs.overlong_buffer_cfg.enable=True \
	+reward_model.reward_kwargs.overlong_buffer_cfg.len=512 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
	+reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
	+reward_model.reward_kwargs.max_resp_len=4096 \
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
	trainer.total_training_steps=3 \
	trainer.total_epochs=1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_m6b2_1_7b_fanin_multistep.txt
