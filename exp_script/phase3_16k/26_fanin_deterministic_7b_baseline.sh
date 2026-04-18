#!/bin/bash
# 8B-scale deterministic fan-in test — fork-DP baseline (no fan-in).
# DeepSeek-R1-Distill-Qwen-7B, fork engine tp_size=4 dp mode, no trigger.
set -x
rm -f log_rank_*.txt

export VERL_SGLANG_DYNAMIC_TP=1
unset VERL_RLPIPE_FANIN
export SGLANG_DYNAMIC_TP_PRE_CAPTURE=1
export SGLANG_DYNAMIC_TP_INITIAL=tp

export GLOO_SOCKET_TIMEOUT=7200
export TORCH_NCCL_DEFAULT_TIMEOUT_SECONDS=7200
export RLPIPE_SGFANIN_TRACE=/home/user/profiling_sgfanin_deterministic_7b_baseline
export RLPIPE_FANIN_TOKEN_DUMP=/home/user/profiling_sgfanin_deterministic_7b_baseline

PROFILING_DIR=/home/user/profiling_sgfanin_deterministic_7b_baseline
rm -rf "$PROFILING_DIR"
mkdir -p "$PROFILING_DIR"

GPUS_PER_NODE=4
ENGINE=sglang
MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
YOUR_PROJECT_NAME=sgfanin-deterministic-7b
YOUR_RUN_NAME=baseline-7b

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
    data.shuffle=False \
    data.seed=42 \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.nccl_timeout=600 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.enable_deterministic_inference=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
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
    trainer.total_epochs=1 \
    2>&1 | sed 's/\x1b\[[0-9;]*m//g' | tee log_p3_sgfanin_deterministic_7b_baseline.txt
