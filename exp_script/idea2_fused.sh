#!/bin/bash
# Idea 2: torch_pp PP=4 + fused forward (actor + log_prob in one pass).
# Rollout side is vanilla SGLang DP=4 — identical to idea2_baseline.sh so
# the delta isolates the fused-forward contribution.
set -o pipefail
set -x
rm -f log_rank_*.txt

source "$(dirname "$0")/common.sh"

EXP_NAME=idea2_fused
PROFILING_DIR=$PROFILING_ROOT/$EXP_NAME/seed_${SEED}
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"

ENGINE=sglang
ROLLOUT_TP=1
GPU_MEMORY_UTILIZATION=0.7

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_torch_pp_trainer' \
	algorithm.adv_estimator=grpo \
	data.train_files=$TRAIN_FILES \
	data.val_files=$VAL_FILES \
	data.prompt_key=prompt \
	data.train_batch_size=$BATCH \
	data.max_prompt_length=$MAX_PROMPT \
	data.max_response_length=$MAX_RESP \
	data.shuffle=True \
	data.seed=$SEED \
	actor_rollout_ref.model.path=$MODEL_PATH \
	actor_rollout_ref.model.use_remove_padding=true \
	actor_rollout_ref.actor.optim.lr=$LR \
	actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH \
	actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
	actor_rollout_ref.actor.use_kl_loss=True \
	actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
	actor_rollout_ref.actor.kl_loss_type=low_var_kl \
	actor_rollout_ref.actor.grad_clip=1.0 \
	actor_rollout_ref.actor.param_offload=True \
	actor_rollout_ref.actor.optimizer_offload=True \
	actor_rollout_ref.actor.fused_forward=True \
	actor_rollout_ref.actor.num_micro_batches=${NUM_MICRO_BATCHES:-$MINI_BATCH} \
	actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
	actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
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
	algorithm.kl_ctrl.kl_coef=$KL_COEF \
	trainer.critic_warmup=0 \
	trainer.logger=['console'] \
	trainer.project_name=rlpipe \
	trainer.experiment_name=${EXP_NAME}_seed${SEED} \
	trainer.resume_mode=disable \
	trainer.n_gpus_per_node=$GPUS_PER_NODE \
	trainer.nnodes=1 \
	trainer.save_freq=-1 \
	trainer.test_freq=9999 \
	trainer.val_before_train=False \
	trainer.total_training_steps=$TOTAL_STEPS \
	trainer.total_epochs=5 2>&1 | stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' | tee $PROFILING_DIR/train.log
